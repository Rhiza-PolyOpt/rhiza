"""Direct MOSEK backend for finalized Rhiza relaxations.

Native task construction, streamed coefficient emission, and status
conventions remain confined to this module.
"""

from contextlib import ExitStack
from time import perf_counter

import numpy as np

from rhiza.backends.base import BackendWorker, restore_objective_value
from rhiza.diagnostics import (
    BackendBuildError,
    BackendUnavailableError,
    PipelineError,
    SolutionStatus,
    SolveError,
    to_python_report,
)
from rhiza.tools.polyalg import _NUM_COEFFS_PER_CHUNK


################################################################################
# Backend construction
################################################################################


def build_mosek(
    sdpr,
    *,
    gap_tol,
    feas_tol,
    iter_limit,
    config,
):
    """Build a native MOSEK task from a finalized SDPR recipe.

    Inputs:
        sdpr:
            SDPR. Finalized relaxation recipe.
        gap_tol:
            Float. Requested relative primal-dual gap.
        feas_tol:
            Float. Requested primal and dual feasibility accuracy.
        iter_limit:
            Integer or None. Interior-point iteration limit.
        config:
            Mapping. Additional native MOSEK parameters.

    Outputs:
        worker:
            MosekWorker. Built native backend worker.
    """
    gap_tol = float(gap_tol)
    feas_tol = float(feas_tol)
    iter_limit = None if iter_limit is None else int(iter_limit)
    config = dict(config)

    if not sdpr.finalized:
        raise PipelineError("Finalize the SDPR before building MOSEK.")

    # Import and allocate native resources only after all backend-independent
    # validation has succeeded. ExitStack gives every construction path one
    # deterministic cleanup point.
    mosek = _import_mosek()
    resources = ExitStack()
    start = perf_counter()
    try:
        # A managed task releases MOSEK's native allocations deterministically.
        task = resources.enter_context(mosek.Task())
        _configure_task(
            task,
            mosek,
            gap_tol=gap_tol,
            feas_tol=feas_tol,
            iter_limit=iter_limit,
            config=config,
        )

        # Read dimensions from the finalized recipe before allocating the
        # corresponding free variables, fixed equations, and PSD blocks.
        num_scalar_coeffs = sdpr._num_scalar_coeffs()
        num_rows = sum(
            len(system['row_indices'])
            for system in sdpr._iter_clique_systems()
        )
        gramian_sizes = np.asarray([
            gramian['gramian_size']
            for gramian in sdpr._iter_gramians()
        ], dtype=np.int32)

        # Allocate scalar, equation, and semidefinite blocks.
        task.appendvars(num_scalar_coeffs)
        task.appendcons(num_rows)
        if gramian_sizes.size:
            task.appendbarvars(gramian_sizes)
        if num_scalar_coeffs:
            task.putvarboundsliceconst(
                0,
                num_scalar_coeffs,
                mosek.boundkey.fr,
                0.0,
                0.0,
            )

        # Stream each clique system before setting the bound objective.
        assembly = _append_clique_systems(task, mosek, sdpr)
        objective = sdpr._scalar_objective()
        objective_indices = np.flatnonzero(objective).astype(np.int32)
        if objective_indices.size:
            task.putclist(objective_indices, objective[objective_indices])
        task.putobjsense(mosek.objsense.maximize)

        # MOSEK may canonicalize stored nonzeros, so validate only dimensions
        # with exact semantic counterparts and retain sparsity counts as diagnostics.
        native = _read_native_task_stats(task)
        _validate_native_task_stats(
            native,
            num_rows=num_rows,
            num_scalar_coeffs=num_scalar_coeffs,
            num_gramians=len(gramian_sizes),
        )
        stats = {
            'num_rows': num_rows,
            'num_rows_original': sdpr.stats['num_rows_original'],
            'num_rows_retained': sdpr.stats['num_rows_retained'],
            'num_scalar_coeffs': num_scalar_coeffs,
            'num_gramians': sdpr._num_gramians(),
            'num_scalar_entries_emitted': (
                assembly['num_scalar_entries_emitted']
            ),
            'num_gramian_entries_emitted': (
                assembly['num_gramian_entries_emitted']
            ),
            'native': native,
            'backend_build_time': perf_counter() - start,
        }
        return MosekWorker(
            task,
            mosek,
            stats,
            resources,
            objective_scale=sdpr._objective_scale_value,
            objective_offset=sdpr._objective_offset_value,
        )
    except BackendBuildError:
        resources.close()
        raise
    except Exception as error:
        resources.close()
        if _is_license_error(error, mosek):
            raise BackendUnavailableError(
                "A usable MOSEK license is unavailable."
            ) from error
        raise BackendBuildError(
            "MOSEK task construction failed: " + str(error)
        ) from error


################################################################################
# MOSEK worker
################################################################################


class MosekWorker(BackendWorker):
    """Own one assembled MOSEK task and the copied solution data Rhiza needs.

    The native task is released immediately after optimization because public
    extraction depends only on the copied clique-system dual vector.
    """

    def __init__(
        self,
        task,
        mosek,
        build_stats,
        resources,
        *,
        objective_scale,
        objective_offset,
    ):
        """Adopt one assembled MOSEK task and its managed resources.

        The task remains live until solve statistics and extraction duals have
        been copied, after which only compact Rhiza-owned data remains.

        Inputs:
            task:
                mosek.Task. Native task.
            mosek:
                module. Imported MOSEK module.
            build_stats:
                Mapping. Backend build statistics.
            resources:
                ExitStack. Managed native resources.
            objective_scale:
                Float. Positive SDPR objective scale.
            objective_offset:
                Float. SDPR objective offset.
        """
        self._task = task
        self._mosek = mosek
        self._stats = dict(build_stats)
        self._resources = resources
        self._objective_scale = float(objective_scale)
        self._objective_offset = float(objective_offset)
        self._built = True
        self._closed = False
        self._solved = False
        self._report = None
        self._system_duals = None
        self._solution_type = mosek.soltype.itr

    @property
    def built(self):
        """Return whether the worker remains usable."""
        return self._built and not self._closed

    @property
    def closed(self):
        """Return whether the worker was explicitly closed."""
        return self._closed

    @property
    def stats(self):
        """Return backend build and solve statistics. A copy is returned so
        callers can inspect derived state without mutating the object that
        produced it."""
        return _copy_stats(self._stats)

    def solve(self, *, verbose=False):
        """Solve the assembled MOSEK task once and normalize its result.

        Inputs:
            verbose:
                Boolean. Whether to stream the native MOSEK log.

        Outputs:
            report:
                Report. Normalized solution information.
        """
        verbose = bool(verbose)

        if self._closed:
            raise SolveError("The MOSEK worker is closed.")
        if self._solved:
            report = dict(self._report)
            report['changed'] = False
            return report

        if verbose:
            def stream(text):
                """Write one native log fragment without adding a newline."""
                print(str(text), end='')

            self._task.set_Stream(self._mosek.streamtype.log, stream)

        start = perf_counter()
        try:
            # Optimize the assembled task, then copy every objective, statistic,
            # and extraction dual needed after the native task is released.
            try:
                termination_code = self._task.optimize()
            finally:
                # Optional post-solve information calls can return ordinary
                # "not available" response codes. Once the optimizer log is
                # complete, detach its callback so those handled responses do
                # not appear as alarming native errors in verbose output.
                if verbose:
                    self._task.set_Stream(self._mosek.streamtype.log, None)
            solve_time = perf_counter() - start
            solsta = self._task.getsolsta(self._solution_type)
            prosta = self._task.getprosta(self._solution_type)
            solution_status = _normalize_solution_status(self._mosek, solsta)
            success = solution_status in (
                SolutionStatus.OPTIMAL,
                SolutionStatus.OPTIMAL_INACCURATE,
            )

            primal_objective = None
            dual_objective = None
            relative_gap = None
            if success:
                primal_normalized = float(
                    self._task.getprimalobj(self._solution_type)
                )
                dual_normalized = float(
                    self._task.getdualobj(self._solution_type)
                )
                if not np.isfinite(primal_normalized):
                    raise SolveError(
                        "MOSEK returned a nonfinite primal objective."
                    )
                if not np.isfinite(dual_normalized):
                    raise SolveError(
                        "MOSEK returned a nonfinite dual objective."
                    )
                scale = max(
                    1.0,
                    abs(primal_normalized),
                    abs(dual_normalized),
                )
                relative_gap = abs(
                    primal_normalized - dual_normalized
                )/scale
                primal_objective = restore_objective_value(
                    primal_normalized,
                    scale=self._objective_scale,
                    offset=self._objective_offset,
                    backend='MOSEK',
                )
                dual_objective = restore_objective_value(
                    dual_normalized,
                    scale=self._objective_scale,
                    offset=self._objective_offset,
                    backend='MOSEK',
                )

                # Copy the submitted equation duals while the native task is
                # available. This is the only numerical solution vector used by
                # Solver's public extraction methods.
                system_duals = np.asarray(
                    self._task.gety(self._solution_type), dtype=float
                )
                if system_duals.shape != (self._stats['num_rows'],):
                    raise SolveError(
                        "MOSEK returned an invalid system dual vector."
                    )
                if not np.all(np.isfinite(system_duals)):
                    raise SolveError(
                        "MOSEK returned a nonfinite system dual vector."
                    )
            else:
                system_duals = None

            # Read only documented information items. Missing fields remain
            # None so reports stay portable across supported MOSEK versions.
            iterations = _read_iterations(self._task, self._mosek)
            factorization = _read_factorization_stats(
                self._task, self._mosek
            )
            memory = _read_memory_stats(self._task)
            report = {
                'success': success,
                'changed': True,
                'solution_status': solution_status.value,
                'primal_objective': primal_objective,
                'dual_objective': dual_objective,
                'relative_gap': relative_gap,
                'num_iterations': iterations,
                'solve_time': solve_time,
            }
            self._solved = True
            self._report = report
            self._system_duals = system_duals
            self._stats['solve_time'] = solve_time
            self._stats['num_iterations'] = iterations
            self._stats['factorization'] = factorization
            self._stats['memory'] = memory
            # Native feasibility and solution labels remain backend-specific
            # diagnostics, while the public solve report exposes only Rhiza's
            # normalized solution status.
            native = dict(self._stats.get('native', {}))
            native['termination_status'] = _enum_label(
                termination_code
            ).upper()
            native['problem_status'] = _enum_label(prosta).upper()
            native['solution_status'] = _enum_label(solsta).upper()
            self._stats['native'] = native

            # All public post-solve operations use copied Rhiza data. Releasing
            # the task here avoids retaining the full native model throughout
            # application-specific rounding and reporting.
            self._release_task()
            return dict(report)
        except SolveError:
            raise
        except Exception as error:
            if _is_license_error(error, self._mosek):
                raise BackendUnavailableError(
                    "A usable MOSEK license is unavailable."
                ) from error
            raise SolveError(
                "MOSEK optimization failed: " + str(error)
            ) from error

    def get_system_duals(self):
        """Return the copied submitted clique-system dual vector.

        Outputs:
            system_duals:
                ndarray. Concatenated submitted equation duals.
        """
        self._require_solution()
        return self._system_duals.copy()

    def close(self):
        """Close the worker and discard copied solution data.

        Outputs:
            changed:
                Boolean. True when the worker changed to its closed state.
        """
        if self._closed:
            return False
        self._release_task()
        self._system_duals = None
        self._closed = True
        return True

    def _release_task(self):
        """Release the managed native task while preserving copied solve data."""
        if self._resources is not None:
            self._resources.close()
            self._resources = None
        self._task = None

    def _require_solution(self):
        """Require copied data from a successful MOSEK solution."""
        if self._closed:
            raise SolveError("The MOSEK worker is closed.")
        if not self._solved or not self._report['success']:
            raise SolveError("No successful MOSEK solution is available.")


################################################################################
# Task construction stages
################################################################################


def _append_clique_systems(task, mosek, sdpr):
    """Append fixed clique equations and streamed coefficients. Coefficient chunks
    are streamed directly into the native task so large sampled systems do not
    require one global sparse matrix in memory.

    Inputs:
        task:
            mosek.Task. Native task under construction.
        mosek:
            module. Imported MOSEK module.
        sdpr:
            SDPR. Finalized relaxation recipe.

    Outputs:
        assembly:
            Mapping. Numbers of scalar and Gramian entries emitted.
    """
    row_offset = 0
    num_scalar_entries_emitted = 0
    num_gramian_entries_emitted = 0

    # Each clique occupies one contiguous equation slice. Streaming its
    # coefficients preserves locality and bounds temporary memory by chunk size.
    for clique_id, system in enumerate(sdpr._iter_clique_systems()):
        num_rows = len(system['row_indices'])
        objective_evals = np.asarray(
            sdpr._clique_objective_evals(clique_id), dtype=float
        )
        keys = [mosek.boundkey.fx]*num_rows
        # Fix every equation to its sampled objective residual. The list
        # fallback accommodates older MOSEK Python bindings that reject ndarrays.
        try:
            task.putconboundslice(
                row_offset,
                row_offset + num_rows,
                keys,
                objective_evals,
                objective_evals,
            )
        except TypeError:
            objective_list = objective_evals.tolist()
            task.putconboundslice(
                row_offset,
                row_offset + num_rows,
                keys,
                objective_list,
                objective_list,
            )

        # Free scalar columns include local bounds, multipliers, and separator
        # coefficients. Emit every numerical nonzero without a pruning tolerance.
        for chunk in sdpr._iter_scalar_coeff_chunks(
            clique_id, max_num_coeffs=_NUM_COEFFS_PER_CHUNK
        ):
            row_indices_chunk, column_indices_chunk = np.nonzero(
                chunk.coeff_evals
            )
            if row_indices_chunk.size == 0:
                continue
            rows = (row_offset + row_indices_chunk).astype(np.int32)
            columns = chunk.scalar_indices[
                column_indices_chunk
            ].astype(np.int32)
            values = np.asarray(
                chunk.coeff_evals[
                    row_indices_chunk,column_indices_chunk
                ],
                dtype=float,
            )
            _put_scalar_triplets(task, rows, columns, values)
            num_scalar_entries_emitted += values.size

        # Gramian chunks use Rhiza's lower-triangle ordering. MOSEK's symmetric
        # inner product counts off-diagonal entries twice, so halve those values.
        for chunk in sdpr._iter_gramian_coeff_chunks(
            clique_id, max_num_coeffs=_NUM_COEFFS_PER_CHUNK
        ):
            row_indices_chunk, coefficient_ids = np.nonzero(chunk.coeff_evals)
            if row_indices_chunk.size == 0:
                continue
            rows = (row_offset + row_indices_chunk).astype(np.int32)
            gramian_ids = np.full(
                row_indices_chunk.size, chunk.gramian_id, dtype=np.int32
            )
            gramian_rows = chunk.gramian_rows[coefficient_ids].astype(np.int32)
            gramian_columns = chunk.gramian_columns[
                coefficient_ids
            ].astype(np.int32)
            values = np.asarray(
                chunk.coeff_evals[
                    row_indices_chunk,coefficient_ids
                ],
                dtype=float,
            ).copy()
            values[gramian_rows != gramian_columns] *= 0.5
            _put_gramian_triplets(
                task,
                rows,
                gramian_ids,
                gramian_rows,
                gramian_columns,
                values,
            )
            num_gramian_entries_emitted += values.size

        row_offset += num_rows

    return {
        'num_scalar_entries_emitted': int(num_scalar_entries_emitted),
        'num_gramian_entries_emitted': int(num_gramian_entries_emitted),
    }


def _configure_task(
    task,
    mosek,
    *,
    gap_tol,
    feas_tol,
    iter_limit,
    config,
):
    """Apply native and backend-independent MOSEK parameters. Keeping this
    convention inside the adapter lets the rest of Rhiza use backend-
    independent objectives and extraction duals.

    Inputs:
        task:
            mosek.Task. Native task under construction.
        mosek:
            module. Imported MOSEK module.
        gap_tol:
            Float. Requested relative gap.
        feas_tol:
            Float. Requested feasibility accuracy.
        iter_limit:
            Integer or None. Interior-point iteration limit.
        config:
            Mapping. Additional native parameters.
    """
    # Apply native configuration first; explicit generic tolerances below are
    # the authoritative public Solver arguments and therefore override it.
    for name, value in config.items():
        task.putparam(str(name), str(value))

    task.putdouparam(
        mosek.dparam.intpnt_co_tol_rel_gap, gap_tol
    )
    task.putdouparam(
        mosek.dparam.intpnt_co_tol_pfeas, feas_tol
    )
    task.putdouparam(
        mosek.dparam.intpnt_co_tol_dfeas, feas_tol
    )
    if iter_limit is not None:
        task.putintparam(
            mosek.iparam.intpnt_max_iterations, iter_limit
        )


def _read_native_task_stats(task):
    """Read compact native task dimensions and sparsity counts.

    Inputs:
        task:
            mosek.Task. Built native task.

    Outputs:
        stats:
            Mapping. Native task statistics.
    """
    return {
        'num_rows': int(task.getnumcon()),
        'num_scalar_coeffs': int(task.getnumvar()),
        'num_gramians': int(task.getnumbarvar()),
        'num_scalar_nonzeros_stored': int(task.getnumanz()),
        'num_gramian_block_nonzeros_stored': int(task.getnumbaranz()),
        'num_gramian_entries_stored_upper_bound': int(
            task.getnumbarablocktriplets()
        ),
    }


def _validate_native_task_stats(
    native,
    *,
    num_rows,
    num_scalar_coeffs,
    num_gramians,
):
    """Require MOSEK to receive the expected rows, scalars, and PSD blocks.

    These dimensions are exact assembly invariants. Canonical native nonzero
    counts are intentionally excluded because MOSEK may combine or omit entries
    while storing the task.

    Inputs:
        native:
            Mapping. Native task statistics.
        num_rows:
            Integer. Expected equation count.
        num_scalar_coeffs:
            Integer. Expected scalar-variable count.
        num_gramians:
            Integer. Expected PSD-block count.
    """
    expected = {
        'num_rows': num_rows,
        'num_scalar_coeffs': num_scalar_coeffs,
        'num_gramians': num_gramians,
    }
    for name, value in expected.items():
        if native[name] != value:
            raise BackendBuildError(
                f"Native MOSEK {name} is {native[name]}, expected {value}."
            )


################################################################################
# Solve statistics and statuses
################################################################################


def _read_factorization_stats(task, mosek):
    """Read documented interior-point factorization information.

    Inputs:
        task:
            mosek.Task. Solved native task.
        mosek:
            module. Imported MOSEK module.

    Outputs:
        factorization:
            Mapping. Solve form and factorization statistics.
    """
    solve_dual = _read_int_info(
        task, _info_item(mosek.iinfitem, 'intpnt_solve_dual')
    )
    return {
        'solve_form': (
            None if solve_dual is None
            else ('DUAL' if solve_dual else 'PRIMAL')
        ),
        'num_nonzeros': _read_long_info(
            task, _info_item(mosek.liinfitem, 'intpnt_factor_num_nz')
        ),
        'dense_dimension': _read_int_info(
            task, _info_item(mosek.iinfitem, 'intpnt_factor_dim_dense')
        ),
        'num_flops': _read_double_info(
            task, _info_item(mosek.dinfitem, 'intpnt_factor_num_flops')
        ),
        'num_threads': _read_int_info(
            task, _info_item(mosek.iinfitem, 'intpnt_num_threads')
        ),
    }


def _read_memory_stats(task):
    """Read native task memory usage when available.

    Inputs:
        task:
            mosek.Task. Solved native task.

    Outputs:
        memory:
            Mapping. Current and peak native bytes.
    """
    try:
        num_bytes_current, num_bytes_peak = task.getmemusage()
    except Exception:
        return {
            'num_bytes_current': None,
            'num_bytes_peak': None,
        }
    return {
        'num_bytes_current': int(num_bytes_current),
        'num_bytes_peak': int(num_bytes_peak),
    }


def _normalize_solution_status(mosek, mosek_status):
    """Map one native MOSEK status to a Rhiza solution status.

    Inputs:
        mosek:
            module. Imported MOSEK module.
        mosek_status:
            Object. Native MOSEK solution status.

    Outputs:
        solution_status:
            SolutionStatus. Normalized status.
    """
    if mosek_status == mosek.solsta.optimal:
        return SolutionStatus.OPTIMAL
    if mosek_status == getattr(mosek.solsta, 'near_optimal', None):
        return SolutionStatus.OPTIMAL_INACCURATE
    if mosek_status in (
        getattr(mosek.solsta, 'prim_infeas_cer', None),
        getattr(mosek.solsta, 'near_prim_infeas_cer', None),
    ):
        return SolutionStatus.INFEASIBLE
    if mosek_status in (
        getattr(mosek.solsta, 'dual_infeas_cer', None),
        getattr(mosek.solsta, 'near_dual_infeas_cer', None),
    ):
        return SolutionStatus.UNBOUNDED
    if mosek_status == mosek.solsta.unknown:
        return SolutionStatus.UNKNOWN
    return SolutionStatus.SOLVER_ERROR


def _read_iterations(task, mosek):
    """Read the interior-point iteration count when available.

    Inputs:
        task:
            mosek.Task. Solved native task.
        mosek:
            module. Imported MOSEK module.

    Outputs:
        num_iterations:
            Integer or None. Interior-point iteration count.
    """
    return _read_int_info(
        task, _info_item(mosek.iinfitem, 'intpnt_iter')
    )


def _info_item(group, name):
    """Return one native information token when the installed version defines it.

    Inputs:
        group:
            Object. Native information-token group.
        name:
            String. Information-token name.

    Outputs:
        item:
            Object or None. Native information token.
    """
    return getattr(group, name, None)


def _read_int_info(task, item):
    """Read one optional integer information item.

    Inputs:
        task:
            mosek.Task. Solved native task.
        item:
            Object. Native integer information token.

    Outputs:
        value:
            Integer or None. Native information value.
    """
    if item is None:
        return None
    try:
        return int(task.getintinf(item))
    except Exception:
        return None


def _read_long_info(task, item):
    """Read one optional long-integer information item.

    Inputs:
        task:
            mosek.Task. Solved native task.
        item:
            Object. Native long information token.

    Outputs:
        value:
            Integer or None. Native information value.
    """
    if item is None:
        return None
    try:
        return int(task.getlintinf(item))
    except Exception:
        return None


def _read_double_info(task, item):
    """Read one optional floating-point information item.

    Inputs:
        task:
            mosek.Task. Solved native task.
        item:
            Object. Native floating-point information token.

    Outputs:
        value:
            Float or None. Native information value.
    """
    if item is None:
        return None
    try:
        return float(task.getdouinf(item))
    except Exception:
        return None


################################################################################
# Low-level MOSEK helpers
################################################################################


def _put_scalar_triplets(task, rows, columns, values):
    """Insert scalar affine triplets with a list fallback for older bindings.

    Inputs:
        task:
            mosek.Task. Native task under construction.
        rows:
            ndarray. Constraint row indexes.
        columns:
            ndarray. Scalar variable indexes.
        values:
            ndarray. Affine coefficient values.
    """
    try:
        task.putaijlist(rows, columns, values)
    except TypeError:
        task.putaijlist(rows.tolist(), columns.tolist(), values.tolist())


def _put_gramian_triplets(
    task,
    rows,
    gramian_ids,
    gramian_rows,
    gramian_columns,
    values,
):
    """Insert semidefinite affine triplets with an older-binding fallback.

    Current MOSEK bindings accept NumPy arrays directly. Older supported
    bindings may require ordinary Python sequences for the same public call.

    Inputs:
        task:
            mosek.Task. Native task under construction.
        rows:
            ndarray. Constraint row indexes.
        gramian_ids:
            ndarray. Gramian block indexes.
        gramian_rows:
            ndarray. Symmetric matrix row indexes.
        gramian_columns:
            ndarray. Symmetric matrix column indexes.
        values:
            ndarray. Semidefinite coefficient values.
    """
    try:
        task.putbarablocktriplet(
            rows,
            gramian_ids,
            gramian_rows,
            gramian_columns,
            values,
        )
    except TypeError:
        task.putbarablocktriplet(
            rows.tolist(),
            gramian_ids.tolist(),
            gramian_rows.tolist(),
            gramian_columns.tolist(),
            values.tolist(),
        )


def _is_license_error(error, mosek=None):
    """Return whether a native error indicates unavailable licensing.

    Documented MOSEK response codes are checked first. Text inspection remains
    as a compatibility fallback for binding versions that expose less metadata.

    Inputs:
        error:
            Exception. Native MOSEK error.
        mosek:
            module or None. Imported MOSEK module.

    Outputs:
        is_license_error:
            Boolean. Whether the error describes a license failure.
    """
    if mosek is not None:
        error_class = getattr(mosek, 'Error', None)
        if error_class is not None and isinstance(error, error_class):
            try:
                code = int(error.errno)
            except Exception:
                code = None
            if code is not None and 1000 <= code <= 1028:
                return True

    parts = [str(error)]
    for name in ('errno', 'code', 'rescode'):
        value = getattr(error, name, None)
        if value is not None:
            parts.append(str(value))
    parts.extend(str(value) for value in getattr(error, 'args', ()))
    text = ' '.join(parts).lower()
    return (
        'license' in text
        or 'licence' in text
        or 'missing_license' in text
        or 'err_license' in text
    )


def _enum_label(value):
    """Return the final textual component of an enum-like value.

    Inputs:
        value:
            Object. Enum-like value.

    Outputs:
        label:
            String. Final textual component.
    """
    return str(value).rsplit('.', 1)[-1]


def _copy_stats(stats):
    """Copy backend statistics and their nested mappings. The copy separates
    public diagnostics from mutable arrays and native records owned by the
    backend worker.

    Inputs:
        stats:
            Mapping. Backend statistics.

    Outputs:
        copied:
            dict. Independent shallow tree of statistics.
    """
    copied = dict(stats)
    for name in ('native', 'factorization', 'memory'):
        if name in copied:
            copied[name] = dict(copied[name])
    return to_python_report(copied)


def _import_mosek():
    """Import the official MOSEK package lazily. The dependency is imported only
    when this backend is selected, so ordinary Rhiza imports remain free of
    native solver side effects.

    Outputs:
        mosek:
            module. Imported MOSEK package.
    """
    try:
        import mosek
    except ImportError as error:
        raise BackendUnavailableError(
            "The MOSEK backend requires the optional dependency. Install "
            "Rhiza with the 'mosek' extra."
        ) from error
    return mosek
