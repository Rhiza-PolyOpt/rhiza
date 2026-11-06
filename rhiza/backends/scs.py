"""Direct SCS backend for finalized Rhiza relaxations.

The adapter assembles native SCS cone data and translates numerical solutions
back to Rhiza's objective and extraction conventions.
"""

from time import perf_counter

import numpy as np
from scipy import sparse

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


def build_scs(
    sdpr,
    *,
    gap_tol,
    feas_tol,
    iter_limit,
    config,
):
    """Build native SCS cone data from a finalized SDPR recipe.

    Inputs:
        sdpr:
            SDPR. Finalized relaxation recipe.
        gap_tol:
            Float. Requested relative accuracy.
        feas_tol:
            Float. Requested absolute feasibility accuracy.
        iter_limit:
            Integer or None. Iteration limit.
        config:
            Mapping. Additional native SCS settings.

    Outputs:
        worker:
            SCSWorker. Built backend worker.
    """
    gap_tol = float(gap_tol)
    feas_tol = float(feas_tol)
    iter_limit = None if iter_limit is None else int(iter_limit)
    config = dict(config)

    if not sdpr.finalized:
        raise PipelineError("Finalize the SDPR before building SCS.")

    # Delay the native import until this backend is selected. This keeps the
    # package import lightweight while still treating SCS as a core dependency.
    scs = _import_scs()
    start = perf_counter()
    try:
        # Assemble the dual certificate orientation once. In that orientation,
        # SCS primal variables are the sampled clique-system weights Rhiza later
        # uses for moment extraction.
        data, cone, assembly = _assemble_scs_data(sdpr)
        # Keep Rhiza-emitted counts distinct from SCS's canonical sparse
        # storage so diagnostics retain an unambiguous meaning across versions.
        stats = {
            'num_rows': assembly['num_system_rows'],
            'num_rows_original': sdpr.stats['num_rows_original'],
            'num_rows_retained': sdpr.stats['num_rows_retained'],
            'num_scalar_coeffs': assembly['num_scalar_coeffs'],
            'num_gramians': len(assembly['gramian_sizes']),
            'num_cone_rows': assembly['num_cone_rows'],
            'num_zero_cone_rows': assembly['num_scalar_coeffs'],
            'num_psd_cone_rows': assembly['num_psd_cone_rows'],
            'num_matrix_nonzeros': int(data['A'].nnz),
            'num_scalar_entries_emitted': (
                assembly['num_scalar_entries_emitted']
            ),
            'num_gramian_entries_emitted': (
                assembly['num_gramian_entries_emitted']
            ),
            'gramian_sizes': tuple(assembly['gramian_sizes']),
            'backend_build_time': perf_counter() - start,
        }
        return SCSWorker(
            scs,
            data,
            cone,
            stats,
            objective_scale=sdpr._objective_scale_value,
            objective_offset=sdpr._objective_offset_value,
            gap_tol=gap_tol,
            feas_tol=feas_tol,
            iter_limit=iter_limit,
            config=config,
        )
    except BackendBuildError:
        raise
    except Exception as error:
        raise BackendBuildError(
            "SCS data construction failed: " + str(error)
        ) from error


################################################################################
# SCS worker
################################################################################


class SCSWorker(BackendWorker):
    """Own one assembled SCS problem and its numerical solution. It stores the
    sparse cone data independently of the native workspace so failed solves can
    be retried transactionally."""

    def __init__(
        self,
        scs,
        data,
        cone,
        build_stats,
        *,
        objective_scale,
        objective_offset,
        gap_tol,
        feas_tol,
        iter_limit,
        config,
    ):
        """Adopt assembled SCS cone data before native workspace creation.

        The workspace is created at solve time because SCS fixes verbosity in
        its constructor; retaining the assembled data also permits a clean retry.

        Inputs:
            scs:
                module. Imported SCS module.
            data:
                Mapping. SCS matrix and vector data.
            cone:
                Mapping. SCS cone dimensions.
            build_stats:
                Mapping. Backend build statistics.
            objective_scale:
                Float. Positive SDPR objective scale.
            objective_offset:
                Float. SDPR objective offset.
            gap_tol:
                Float. Relative solver tolerance.
            feas_tol:
                Float. Absolute solver tolerance.
            iter_limit:
                Integer or None. Iteration limit.
            config:
                Mapping. Additional native settings.
        """
        self._scs = scs
        self._data = data
        self._cone = cone
        self._stats = dict(build_stats)
        self._objective_scale = float(objective_scale)
        self._objective_offset = float(objective_offset)
        self._gap_tol = gap_tol
        self._feas_tol = feas_tol
        self._iter_limit = iter_limit
        self._config = dict(config)
        self._native_solver = None
        self._native_settings = None
        self._system_duals = None
        self._report = None
        self._built = True
        self._closed = False
        self._solved = False

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
        """Construct and solve the native SCS workspace once.

        Inputs:
            verbose:
                Boolean. Whether to stream native SCS progress.

        Outputs:
            report:
                Report. Normalized solution information.
        """
        verbose = bool(verbose)

        if self._closed:
            raise SolveError("The SCS worker is closed.")
        if self._solved:
            report = dict(self._report)
            report['changed'] = False
            return report

        # Resolve settings at solve time because verbosity belongs to solve(),
        # while all other native options were fixed when Solver was constructed.
        requested_settings = _resolve_settings(
            self._config,
            gap_tol=self._gap_tol,
            feas_tol=self._feas_tol,
            iter_limit=self._iter_limit,
            verbose=verbose,
        )
        start = perf_counter()
        try:
            if self._native_solver is None:
                requested_settings = _compatible_scs_settings(
                    self._scs, requested_settings
                )
                self._native_solver = self._scs.SCS(
                    self._data,
                    self._cone,
                    **requested_settings,
                )
                self._native_settings = dict(requested_settings)

                # SCS keeps deep copies after workspace construction. Rhiza's
                # complete CSC data can be released before the iterative solve.
                self._data = None
                self._cone = None
            settings = dict(self._native_settings)
            solution = self._native_solver.solve()
            wall_time = perf_counter() - start

            # Copy version-dependent native metadata before normalizing the
            # stable status and objective fields exposed by Rhiza.
            info = dict(solution.get('info', {}))
            status_value = int(info.get('status_val', 0))
            solution_status = _normalize_solution_status(
                self._scs, status_value
            )
            success = solution_status in (
                SolutionStatus.OPTIMAL,
                SolutionStatus.OPTIMAL_INACCURATE,
            )

            primal_objective = None
            dual_objective = None
            relative_gap = None
            if success:
                # SCS solves the dual orientation of Rhiza's certificate SDP.
                primal_normalized = float(info['dobj'])
                dual_normalized = float(info['pobj'])
                if not np.isfinite(primal_normalized):
                    raise SolveError(
                        "SCS returned a nonfinite primal objective."
                    )
                if not np.isfinite(dual_normalized):
                    raise SolveError(
                        "SCS returned a nonfinite dual objective."
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
                    backend='SCS',
                )
                dual_objective = restore_objective_value(
                    dual_normalized,
                    scale=self._objective_scale,
                    offset=self._objective_offset,
                    backend='SCS',
                )

            num_iterations = _optional_int(info.get('iter'))
            report = {
                'success': success,
                'changed': True,
                'solution_status': solution_status.value,
                'primal_objective': primal_objective,
                'dual_objective': dual_objective,
                'relative_gap': relative_gap,
                'num_iterations': num_iterations,
                'solve_time': wall_time,
            }
            # Validate all native vectors while they are available, but retain
            # only x: in Rhiza's submitted dual orientation it is exactly the
            # clique-system dual vector needed for moment extraction.
            solution_arrays = {
                name: (
                    None if solution.get(name) is None
                    else np.asarray(solution[name], dtype=float)
                )
                for name in ('x', 'y', 's')
            }
            system_duals = None
            if success:
                _validate_successful_solution(
                    solution_arrays,
                    num_system_rows=self._stats['num_rows'],
                    num_cone_rows=self._stats['num_cone_rows'],
                )
                # Retain only the clique-system weights used by public moment
                # extraction. The cone dual and slack vectors are validated in
                # place and released with the native solution mapping.
                system_duals = solution_arrays['x'].copy()
            native_stats = _native_stats(info, settings)
        except SolveError:
            raise
        except Exception as error:
            raise SolveError(
                "SCS optimization failed: " + str(error)
            ) from error

        # The native factorization and cone vectors are no longer needed after
        # the normalized report and submitted equation duals have been copied.
        self._native_solver = None
        self._native_settings = None
        self._system_duals = system_duals
        self._report = report
        self._solved = True
        self._stats['solve_time'] = wall_time
        self._stats['num_iterations'] = num_iterations
        self._stats['native'] = native_stats
        self._stats['settings'] = dict(settings)
        return dict(report)

    def get_system_duals(self):
        """Return the copied submitted clique-system dual vector.

        Outputs:
            system_duals:
                ndarray. Concatenated submitted equation duals.
        """
        self._require_solution()
        return self._system_duals.copy()

    def close(self):
        """Close the worker and discard assembled and copied data.

        Outputs:
            changed:
                Boolean. True when the worker changed to its closed state.
        """
        if self._closed:
            return False
        self._native_solver = None
        self._native_settings = None
        self._data = None
        self._cone = None
        self._system_duals = None
        self._closed = True
        return True

    def _require_solution(self):
        """Require an available usable SCS solution."""
        if self._closed:
            raise SolveError("The SCS worker is closed.")
        if not self._solved or not self._report.get('success', False):
            raise SolveError("No usable SCS solution is available.")


################################################################################
# Cone data assembly
################################################################################


def _assemble_scs_data(sdpr):
    """Assemble the dual certificate orientation accepted by SCS. Rhiza submits
    the dual certificate form so SCS primal variables are exactly the clique-
    system weights later used for extraction.

    Inputs:
        sdpr:
            SDPR. Finalized relaxation recipe.

    Outputs:
        data:
            Mapping. SCS matrix and vectors.
        cone:
            Mapping. Zero and PSD cone dimensions.
        assembly:
            Mapping. Dimensions, offsets, and emitted entry counts.
    """
    # The SCS decision vector contains one entry per retained clique equation.
    # Cone rows encode free scalar equalities first, followed by PSD adjoints.
    num_scalar_coeffs = sdpr._num_scalar_coeffs()
    gramian_sizes = tuple(
        gramian['gramian_size'] for gramian in sdpr._iter_gramians()
    )
    num_psd_cone_rows = sum(
        size*(size + 1)//2 for size in gramian_sizes
    )
    num_cone_rows = num_scalar_coeffs + num_psd_cone_rows
    num_system_rows = sum(
        len(system['row_indices'])
        for system in sdpr._iter_clique_systems()
    )

    row_blocks = []
    column_blocks = []
    value_blocks = []
    objective = np.empty(num_system_rows)
    # Record where each column-major PSD vector begins in the stacked cone so
    # streamed Gramian coefficients enter the correct SCS block.
    gramian_row_offsets = []
    row_cursor = num_scalar_coeffs
    for size in gramian_sizes:
        gramian_row_offsets.append(row_cursor)
        row_cursor += size*(size + 1)//2

    num_scalar_entries_emitted = 0
    num_gramian_entries_emitted = 0
    system_row_offset = 0
    # Emit clique systems in finalized order. Their retained rows become
    # contiguous columns of A and contiguous entries of the SCS objective.
    for clique_id, system in enumerate(sdpr._iter_clique_systems()):
        num_rows = len(system['row_indices'])
        objective[
            system_row_offset:system_row_offset + num_rows
        ] = sdpr._clique_objective_evals(clique_id)

        # Equality rows impose B.T@w=d for the free certificate coefficients.
        for chunk in sdpr._iter_scalar_coeff_chunks(
            clique_id, max_num_coeffs=_NUM_COEFFS_PER_CHUNK
        ):
            local_rows, local_columns = np.nonzero(chunk.coeff_evals)
            if local_rows.size == 0:
                continue
            row_blocks.append(chunk.scalar_indices[local_columns])
            column_blocks.append(system_row_offset + local_rows)
            value_blocks.append(np.asarray(
                chunk.coeff_evals[local_rows,local_columns],
                dtype=float,
            ))
            num_scalar_entries_emitted += local_rows.size

        # PSD slacks impose H_j^*(w) positive semidefinite. SCS scales
        # off-diagonal lower-triangle entries by sqrt(2), whereas Rhiza's
        # Gramian columns already include the symmetric factor of two.
        for chunk in sdpr._iter_gramian_coeff_chunks(
            clique_id, max_num_coeffs=_NUM_COEFFS_PER_CHUNK
        ):
            local_rows, coefficient_ids = np.nonzero(chunk.coeff_evals)
            if local_rows.size == 0:
                continue
            gramian = sdpr._gramians[chunk.gramian_id]
            size = gramian['gramian_size']
            gramian_rows = chunk.gramian_rows[coefficient_ids]
            gramian_columns = chunk.gramian_columns[coefficient_ids]
            # Map Rhiza's cached lower-triangle coefficient ordering into
            # SCS's column-major svec positions for this PSD cone block.
            svec_indices = _svec_indices(
                size, gramian_rows, gramian_columns
            )
            values = -np.asarray(
                chunk.coeff_evals[local_rows,coefficient_ids],
                dtype=float,
            ).copy()
            values[gramian_rows != gramian_columns] /= np.sqrt(2.0)
            row_blocks.append(
                gramian_row_offsets[chunk.gramian_id] + svec_indices
            )
            column_blocks.append(system_row_offset + local_rows)
            value_blocks.append(values)
            num_gramian_entries_emitted += local_rows.size

        system_row_offset += num_rows

    if system_row_offset != num_system_rows:
        raise BackendBuildError("SCS system-row count is inconsistent.")

    # Concatenate once, release the per-chunk arrays, and then convert the
    # single coordinate representation to SCS's required CSC storage.
    matrix_values = _concatenate(value_blocks, float)
    matrix_rows = _concatenate(row_blocks, np.int64)
    matrix_columns = _concatenate(column_blocks, np.int64)
    value_blocks.clear()
    row_blocks.clear()
    column_blocks.clear()
    matrix_coordinate = sparse.coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(num_cone_rows, num_system_rows),
    )
    del matrix_values, matrix_rows, matrix_columns
    matrix = matrix_coordinate.tocsc()
    del matrix_coordinate
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    # In Ax+s=b, zero-cone rows carry the scalar objective equations; PSD
    # rows have a zero right-hand side because their slack is the adjoint matrix.
    cone_rhs = np.zeros(num_cone_rows)
    cone_rhs[:num_scalar_coeffs] = sdpr._scalar_objective()
    data = {
        'A': matrix,
        'b': cone_rhs,
        'c': objective,
    }
    cone = {
        'z': num_scalar_coeffs,
        's': list(gramian_sizes),
    }
    assembly = {
        'num_system_rows': num_system_rows,
        'num_scalar_coeffs': num_scalar_coeffs,
        'num_psd_cone_rows': num_psd_cone_rows,
        'num_cone_rows': num_cone_rows,
        'gramian_sizes': gramian_sizes,
        'num_scalar_entries_emitted': int(num_scalar_entries_emitted),
        'num_gramian_entries_emitted': int(num_gramian_entries_emitted),
    }
    return data, cone, assembly


################################################################################
# Settings, statuses, and numerical conversions
################################################################################


def _resolve_settings(
    config,
    *,
    gap_tol,
    feas_tol,
    iter_limit,
    verbose,
):
    """Resolve native SCS settings with public arguments taking precedence.

    Inputs:
        config:
            Mapping. Additional native settings.
        gap_tol:
            Float. Relative accuracy.
        feas_tol:
            Float. Absolute feasibility accuracy.
        iter_limit:
            Integer or None. Iteration limit.
        verbose:
            Boolean. Native progress setting.

    Outputs:
        settings:
            Mapping. Complete explicit SCS settings.
    """
    # Start with user-native settings, supply stable Rhiza defaults only when
    # absent, then let explicit generic Solver arguments take precedence.
    settings = dict(config)
    settings.setdefault('normalize', True)
    settings.setdefault('adaptive_scale', True)
    settings['eps_abs'] = feas_tol
    settings['eps_rel'] = gap_tol
    settings['verbose'] = verbose
    if iter_limit is not None:
        settings['max_iters'] = iter_limit
    return settings


def _compatible_scs_settings(scs, settings):
    """Translate SCS 3.2 solver selectors for the SCS 3.3 enum-based API.

    Use the native default when no solver selector is supplied. Explicit
    legacy selectors retain their SCS 3.2 meaning. Store enum values as strings
    so reports remain serializable.
    """
    settings = dict(settings)
    if not hasattr(scs, 'LinearSolver'):
        settings.setdefault('use_indirect', False)
        return settings

    if 'linear_solver' in settings:
        if any(key in settings for key in ('use_indirect', 'gpu', 'mkl', 'cudss')):
            raise ValueError(
                "Specify linear_solver or the legacy SCS solver selectors, "
                "not both."
            )
        selection = settings['linear_solver']
    elif not any(
        key in settings for key in ('use_indirect', 'gpu', 'mkl', 'cudss')
    ):
        selection = 'auto'
    else:
        indirect = settings.pop('use_indirect', False)
        gpu = settings.pop('gpu', False)
        mkl = settings.pop('mkl', False)
        if settings.pop('cudss', False):
            raise ValueError("To use cuDSS set gpu=True and use_indirect=False.")
        if gpu and mkl:
            raise ValueError("Specify only one of the GPU and MKL solvers.")
        if gpu:
            selection = 'gpu_indirect' if indirect else 'cudss'
        elif mkl:
            if indirect:
                raise ValueError(
                    "MKL indirect solver is unavailable; use_indirect must be False."
                )
            selection = 'mkl'
        else:
            selection = 'cpu_indirect' if indirect else 'qdldl'
    settings['linear_solver'] = scs.LinearSolver(selection).value
    return settings


def _normalize_solution_status(scs, status_value):
    """Map one SCS integer exit flag to a Rhiza solution status.

    Inputs:
        scs:
            module. Imported SCS module.
        status_value:
            Integer. Native SCS exit flag.

    Outputs:
        solution_status:
            SolutionStatus. Normalized status.
    """
    if status_value == getattr(scs, 'SCS_SOLVED', 1):
        return SolutionStatus.OPTIMAL
    if status_value == getattr(scs, 'SCS_SOLVED_INACCURATE', 2):
        return SolutionStatus.OPTIMAL_INACCURATE
    # SCS receives the dual orientation of Rhiza's certificate problem. Its
    # infeasibility and unboundedness certificates therefore exchange meanings
    # when reported in Rhiza's primal certificate convention.
    if status_value in (
        getattr(scs, 'SCS_UNBOUNDED', -1),
        getattr(scs, 'SCS_UNBOUNDED_INACCURATE', -6),
    ):
        return SolutionStatus.INFEASIBLE
    if status_value in (
        getattr(scs, 'SCS_INFEASIBLE', -2),
        getattr(scs, 'SCS_INFEASIBLE_INACCURATE', -7),
    ):
        return SolutionStatus.UNBOUNDED
    if status_value in (
        getattr(scs, 'SCS_UNFINISHED', 0),
        getattr(scs, 'SCS_INDETERMINATE', -3),
    ):
        return SolutionStatus.UNKNOWN
    return SolutionStatus.SOLVER_ERROR


def _native_stats(info, settings):
    """Collect useful backend-specific statistics from the SCS info mapping.

    Inputs:
        info:
            Mapping. Native SCS information.
        settings:
            Mapping. Resolved native SCS settings.

    Outputs:
        stats:
            Mapping. Compact backend-specific statistics.
    """
    return {
        'termination_status': info.get('status'),
        'status_value': _optional_int(info.get('status_val')),
        'primal_residual': _optional_float(info.get('res_pri')),
        'dual_residual': _optional_float(info.get('res_dual')),
        'absolute_gap': _optional_float(info.get('gap')),
        'setup_time': _milliseconds_to_seconds(info.get('setup_time')),
        'solve_time': _milliseconds_to_seconds(info.get('solve_time')),
        'linear_system_mode': _linear_system_mode(settings),
        'native_linear_system_solver': _optional_string(
            info.get('lin_sys_solver')
        ),
        'scale': _optional_float(info.get('scale')),
        'scale_updates': _optional_int(info.get('scale_updates')),
        'accepted_acceleration_steps': _optional_int(
            info.get('accepted_accel_steps')
        ),
        'rejected_acceleration_steps': _optional_int(
            info.get('rejected_accel_steps')
        ),
    }


def _linear_system_mode(settings):
    """Describe the requested linear-system mode. Keeping this
    convention inside the adapter lets the rest of Rhiza use backend-
    independent objectives and extraction duals.

    Inputs:
        settings:
            Mapping. Resolved native SCS settings.

    Outputs:
        mode:
            String. Requested Rhiza linear-system mode.
    """
    if 'linear_solver' in settings:
        return {
            'auto': 'AUTO',
            'qdldl': 'SPARSE_DIRECT',
            'cpu_indirect': 'SPARSE_INDIRECT',
            'mkl': 'MKL_DIRECT',
            'accelerate': 'ACCELERATE_DIRECT',
            'cpu_dense': 'DENSE_DIRECT',
            'gpu_indirect': 'GPU_INDIRECT',
            'cudss': 'GPU_DIRECT',
        }[settings['linear_solver']]
    if settings.get('gpu', False):
        return (
            'GPU_INDIRECT'
            if settings.get('use_indirect', False)
            else 'GPU_DIRECT'
        )
    if settings.get('mkl', False):
        return 'MKL_DIRECT'
    if settings.get('use_indirect', False):
        return 'SPARSE_INDIRECT'
    return 'SPARSE_DIRECT'


def _validate_successful_solution(
    solution,
    *,
    num_system_rows,
    num_cone_rows,
):
    """Require complete finite native vectors after a successful SCS solve.

    Inputs:
        solution:
            Mapping. Copied SCS solution arrays.
        num_system_rows:
            Integer. Expected primal-vector length.
        num_cone_rows:
            Integer. Expected dual and slack lengths.
    """
    expected = {
        'x': num_system_rows,
        'y': num_cone_rows,
        's': num_cone_rows,
    }
    for name, size in expected.items():
        values = solution[name]
        if values is None or values.shape != (size,):
            raise SolveError(
                f"SCS returned an invalid {name} solution vector."
            )
        if not np.all(np.isfinite(values)):
            raise SolveError(
                f"SCS returned a nonfinite {name} solution vector."
            )



def _svec_indices(size, rows, columns):
    """Map lower-triangle matrix indexes to SCS column-major positions. Keeping
    this convention inside the adapter lets the rest of Rhiza use backend-
    independent objectives and extraction duals.

    Inputs:
        size:
            Integer. Symmetric matrix dimension.
        rows:
            ArrayLike. Lower-triangle row indexes.
        columns:
            ArrayLike. Lower-triangle column indexes.

    Outputs:
        indexes:
            ndarray. SCS PSD vector positions.
    """
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)

    # Count every complete preceding column of the lower triangle, then add the
    # row displacement within the current column.
    return columns*size - columns*(columns - 1)//2 + rows - columns


def _concatenate(blocks, dtype):
    """Concatenate numeric blocks or return an empty array. Keeping this
    convention inside the adapter lets the rest of Rhiza use backend-
    independent objectives and extraction duals.

    Inputs:
        blocks:
            Sequence. Numeric arrays.
        dtype:
            dtype. Output data type.

    Outputs:
        values:
            ndarray. Contiguous concatenated values.
    """
    if not blocks:
        return np.empty(0, dtype=dtype)
    return np.ascontiguousarray(np.concatenate(blocks), dtype=dtype)


def _optional_string(value):
    """Convert one optional native value to a string.

    Inputs:
        value:
            Object or None. Native value.

    Outputs:
        converted:
            String or None. Converted value.
    """
    if value is None:
        return None
    return str(value)


def _optional_float(value):
    """Convert one optional native scalar to float.

    Inputs:
        value:
            Object or None. Native scalar.

    Outputs:
        converted:
            Float or None. Converted value.
    """
    if value is None:
        return None
    return float(value)


def _optional_int(value):
    """Convert one optional native scalar to an integer.

    Inputs:
        value:
            Object or None. Native scalar.

    Outputs:
        converted:
            Integer or None. Converted value.
    """
    if value is None:
        return None
    return int(value)


def _milliseconds_to_seconds(value):
    """Convert optional native milliseconds to seconds.

    Inputs:
        value:
            Object or None. Millisecond value.

    Outputs:
        seconds:
            Float or None. Seconds.
    """
    if value is None:
        return None
    return float(value)/1000.0


def _copy_stats(stats):
    """Copy nested backend statistics for public access. The copy separates public
    diagnostics from mutable arrays and native records owned by the backend
    worker.

    Inputs:
        stats:
            Mapping. Internal backend statistics.

    Outputs:
        copied:
            Mapping. Shallow copy with copied nested mappings.
    """
    copied = dict(stats)
    for name in ('native', 'settings'):
        if name in copied:
            copied[name] = dict(copied[name])
    return to_python_report(copied)


################################################################################
# Dependency loading
################################################################################


def _import_scs():
    """Import the SCS package only when its backend is selected. The dependency is
    imported only when this backend is selected, so ordinary Rhiza imports
    remain free of native solver side effects.

    Outputs:
        scs:
            module. Imported SCS package.
    """
    try:
        import scs
    except ImportError as error:
        raise BackendUnavailableError(
            "The SCS Python package is unavailable."
        ) from error
    return scs
