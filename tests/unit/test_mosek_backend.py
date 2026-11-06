"""Backend-independent tests for direct MOSEK task construction."""

import sys

import numpy as np
import pytest

import rhiza as rz
import rhiza.backends.mosek as mosek_backend
from rhiza.backends.mosek import (
    _import_mosek,
    _is_license_error,
    _normalize_solution_status,
    build_mosek,
)
from rhiza.diagnostics import (
    BackendUnavailableError,
    SolutionStatus,
    SolveError,
)
from tests.fixtures.reference_problem import make_reference_pop
from tests.fixtures.scaling_problem import make_chain_pop


################################################################################
# Optional dependency boundary
################################################################################


def test_missing_mosek_package_reports_the_installation_extra(monkeypatch):
    """Explain how to enable the optional backend when its package is absent."""
    monkeypatch.setitem(sys.modules, 'mosek', None)

    with pytest.raises(BackendUnavailableError, match="'mosek' extra"):
        _import_mosek()


################################################################################
# Native test doubles
################################################################################


class _TokenGroup:
    """Provide named token attributes used by the backend."""

    fr = 'fr'
    fx = 'fx'
    maximize = 'maximize'
    intpnt_co_tol_rel_gap = 'rel_gap'
    intpnt_co_tol_pfeas = 'pfeas'
    intpnt_co_tol_dfeas = 'dfeas'
    intpnt_max_iterations = 'iterations'
    itr = 'itr'
    log = 'log'
    optimal = 'optimal'
    near_optimal = 'near_optimal'
    prim_infeas_cer = 'prim_infeas_cer'
    near_prim_infeas_cer = 'near_prim_infeas_cer'
    dual_infeas_cer = 'dual_infeas_cer'
    near_dual_infeas_cer = 'near_dual_infeas_cer'
    unknown = 'unknown'
    intpnt_iter = 'intpnt_iter'
    intpnt_solve_dual = 'intpnt_solve_dual'
    intpnt_factor_dim_dense = 'intpnt_factor_dim_dense'
    intpnt_num_threads = 'intpnt_num_threads'
    intpnt_factor_num_nz = 'intpnt_factor_num_nz'
    intpnt_factor_num_flops = 'intpnt_factor_num_flops'


class _RecordingTask:
    """Record optimizer calls made during direct task construction."""

    def __init__(self):
        """Create empty task records and deterministic solve data."""
        self.num_variables = 0
        self.num_constraints = 0
        self.gramian_sizes = np.empty(0, dtype=int)
        self.bounds = {}
        self.scalar_triplets = []
        self.gramian_triplets = []
        self.objective = {}
        self.objective_sense = None
        self.parameters = {}
        self.disposed = False
        self.stream = None

    def __enter__(self):
        """Enter the managed native-task context."""
        return self

    def __exit__(self, exception_type, exception, traceback):
        """Release the recording task at context exit."""
        self.disposed = True
        return False

    def appendvars(self, count):
        """Record the number of scalar variables."""
        self.num_variables += count

    def appendcons(self, count):
        """Record the number of affine equations."""
        self.num_constraints += count

    def appendbarvars(self, sizes):
        """Record native PSD block dimensions."""
        self.gramian_sizes = np.asarray(sizes, dtype=int).copy()

    def putvarboundsliceconst(self, *arguments):
        """Accept free scalar bounds."""

    def putconboundslice(self, start, stop, keys, lower, upper):
        """Record fixed equation right sides."""
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        for offset, row in enumerate(range(start, stop)):
            self.bounds[row] = (keys[offset], lower[offset], upper[offset])

    def putaijlist(self, rows, columns, values):
        """Record scalar affine triplets."""
        self.scalar_triplets.extend(zip(
            np.asarray(rows, dtype=int),
            np.asarray(columns, dtype=int),
            np.asarray(values, dtype=float),
        ))

    def putbarablocktriplet(
        self,
        rows,
        gramians,
        gramian_rows,
        gramian_columns,
        values,
    ):
        """Record semidefinite affine triplets."""
        self.gramian_triplets.extend(zip(
            np.asarray(rows, dtype=int),
            np.asarray(gramians, dtype=int),
            np.asarray(gramian_rows, dtype=int),
            np.asarray(gramian_columns, dtype=int),
            np.asarray(values, dtype=float),
        ))

    def putclist(self, indices, values):
        """Record scalar objective coefficients."""
        self.objective.update(zip(
            np.asarray(indices, dtype=int),
            np.asarray(values, dtype=float),
        ))

    def putobjsense(self, sense):
        """Record the objective sense."""
        self.objective_sense = sense

    def putparam(self, name, value):
        """Record a generic native parameter."""
        self.parameters[name] = value

    def putdouparam(self, name, value):
        """Record a floating-point native parameter."""
        self.parameters[name] = value

    def putintparam(self, name, value):
        """Record an integer native parameter."""
        self.parameters[name] = value

    def getnumcon(self):
        """Return the native equation count."""
        return self.num_constraints

    def getnumvar(self):
        """Return the native scalar-variable count."""
        return self.num_variables

    def getnumbarvar(self):
        """Return the native PSD-block count."""
        return len(self.gramian_sizes)

    def getnumanz(self):
        """Return a canonical native scalar nonzero count."""
        return max(0, len(self.scalar_triplets) - 1)

    def getnumbaranz(self):
        """Return the native nonzero semidefinite block count."""
        return len({
            (row, gramian)
            for row, gramian, _, _, _ in self.gramian_triplets
        })

    def getnumbarablocktriplets(self):
        """Return a canonical semidefinite triplet bound."""
        return max(0, len(self.gramian_triplets) - 3)

    def set_Stream(self, stream_type, stream):
        """Store the requested native log callback."""
        self.stream = (stream_type, stream)

    def optimize(self):
        """Return a successful deterministic termination code."""
        if self.stream is not None:
            self.stream[1]("recording solve\n")
        return 'rescode.ok'

    def getsolsta(self, solution_type):
        """Return an optimal solution status."""
        return 'optimal'

    def getprosta(self, solution_type):
        """Return a primal-feasible problem status."""
        return 'prosta.prim_feas'

    def gety(self, solution_type):
        """Return deterministic submitted equation duals."""
        return np.linspace(1.0, 2.0, self.num_constraints)

    def getprimalobj(self, solution_type):
        """Return a deterministic primal objective."""
        return -2.0

    def getdualobj(self, solution_type):
        """Return a deterministic dual objective."""
        return -2.0

    def getintinf(self, item):
        """Return deterministic integer solver information."""
        values = {
            'intpnt_iter': 7,
            'intpnt_solve_dual': 0,
            'intpnt_factor_dim_dense': 3,
            'intpnt_num_threads': 1,
        }
        return values[item]

    def getlintinf(self, item):
        """Return deterministic long solver information."""
        assert item == 'intpnt_factor_num_nz'
        return 456

    def getdouinf(self, item):
        """Return deterministic floating-point solver information."""
        assert item == 'intpnt_factor_num_flops'
        return 890000.0

    def getmemusage(self):
        """Return deterministic current and peak native memory."""
        return 1234, 5678


class _RecordingMosek:
    """Provide the MOSEK names needed for backend tests."""

    boundkey = _TokenGroup
    objsense = _TokenGroup
    dparam = _TokenGroup
    iparam = _TokenGroup
    soltype = _TokenGroup
    streamtype = _TokenGroup
    solsta = _TokenGroup
    iinfitem = _TokenGroup
    liinfitem = _TokenGroup
    dinfitem = _TokenGroup

    class Error(Exception):
        """Represent one native MOSEK exception."""

    def __init__(self):
        """Create a module-like task factory."""
        self.last_task = None

    def Task(self):
        """Create and retain one recording task."""
        self.last_task = _RecordingTask()
        return self.last_task


################################################################################
# Native error normalization
################################################################################


def test_license_errors_are_recognized_from_text_and_codes():
    """Recognize textual and native-code license failures."""
    fake_mosek = _RecordingMosek()
    error = fake_mosek.Error("optimization unavailable")
    error.errno = 1008

    assert _is_license_error(RuntimeError("MOSEK license expired"))
    assert _is_license_error(error, fake_mosek)
    assert not _is_license_error(
        RuntimeError("matrix dimension mismatch"), fake_mosek
    )


def test_native_solution_statuses_are_normalized_directly():
    """Normalize documented MOSEK solution-status tokens."""
    mosek = _RecordingMosek()
    cases = {
        'optimal': SolutionStatus.OPTIMAL,
        'near_optimal': SolutionStatus.OPTIMAL_INACCURATE,
        'prim_infeas_cer': SolutionStatus.INFEASIBLE,
        'near_prim_infeas_cer': SolutionStatus.INFEASIBLE,
        'dual_infeas_cer': SolutionStatus.UNBOUNDED,
        'near_dual_infeas_cer': SolutionStatus.UNBOUNDED,
        'unknown': SolutionStatus.UNKNOWN,
        'unexpected': SolutionStatus.SOLVER_ERROR,
    }
    for native, expected in cases.items():
        assert _normalize_solution_status(mosek, native) is expected


################################################################################
# Deterministic task emission
################################################################################


def _build_recording_worker(monkeypatch, config=None):
    """Build a recording worker for the reference SDPR.

    Inputs:
        monkeypatch:
            pytest fixture. Temporary attribute patcher.
        config:
            Mapping or None. Native parameter settings.

    Outputs:
        worker:
            MosekWorker. Built recording worker.
        task:
            _RecordingTask. Captured native task.
        sdpr:
            SDPR. Finalized reference relaxation.
    """
    config = {} if config is None else dict(config)

    pop = make_reference_pop(num_samples=55)
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=4)
    sdpr.compress(method='PQR')
    sdpr.finalize()

    def fail_materialization(clique_id):
        """Raise when the forbidden test path is reached."""
        raise AssertionError("backend requested a full clique system")

    monkeypatch.setattr(
        sdpr,
        '_materialize_augmented_clique_system',
        fail_materialization,
    )
    fake_mosek = _RecordingMosek()
    monkeypatch.setattr(mosek_backend, '_import_mosek', lambda: fake_mosek)
    worker = build_mosek(
        sdpr,
        gap_tol=1e-8,
        feas_tol=1e-8,
        iter_limit=100,
        config=config,
    )
    return worker, fake_mosek.last_task, sdpr


def test_direct_emission_matches_sdpr_coefficients(monkeypatch):
    """Match rows, indexes, signs, and Gramian symmetry conventions."""
    worker, task, sdpr = _build_recording_worker(monkeypatch)

    assert task.num_variables == sdpr.stats['num_scalar_coeffs']
    assert task.num_constraints == sdpr.stats['num_rows_retained']
    np.testing.assert_array_equal(
        task.gramian_sizes,
        [gramian['gramian_size'] for gramian in sdpr._gramians],
    )
    assert task.objective_sense == 'maximize'
    expected_objective = sdpr._scalar_objective()
    actual_objective = np.zeros_like(expected_objective)
    for index, value in task.objective.items():
        actual_objective[index] = value
    np.testing.assert_allclose(actual_objective, expected_objective)

    num_rows = sdpr.stats['num_rows_retained']
    scalar_actual = np.zeros((num_rows, sdpr.stats['num_scalar_coeffs']))
    for row, column, value in task.scalar_triplets:
        scalar_actual[row,column] += value

    scalar_expected = np.zeros_like(scalar_actual)
    gramian_expected = {}
    row_offset = 0
    for clique_id, system in enumerate(sdpr._clique_systems):
        num_clique_rows = len(system['row_indices'])
        objective = sdpr._clique_objective_evals(clique_id)
        for local_row in range(num_clique_rows):
            key, lower, upper = task.bounds[row_offset + local_row]
            assert key == 'fx'
            np.testing.assert_allclose(lower, objective[local_row])
            np.testing.assert_allclose(upper, objective[local_row])

        for chunk in sdpr._iter_scalar_coeff_chunks(
            clique_id, max_num_coeffs=10_000_000
        ):
            rows = slice(row_offset, row_offset + num_clique_rows)
            scalar_expected[rows,chunk.scalar_indices] = chunk.coeff_evals
        for chunk in sdpr._iter_gramian_coeff_chunks(
            clique_id, max_num_coeffs=10_000_000
        ):
            gramian = sdpr._gramians[chunk.gramian_id]
            gramian_id = chunk.gramian_id
            if gramian_id not in gramian_expected:
                gramian_expected[gramian_id] = np.zeros((
                    num_rows,
                    gramian['gramian_size'],
                    gramian['gramian_size'],
                ))
            rows = slice(row_offset, row_offset + num_clique_rows)
            for coefficient_id, (first, second) in enumerate(zip(
                chunk.gramian_rows,
                chunk.gramian_columns,
            )):
                gramian_expected[gramian_id][rows,first,second] = (
                    chunk.coeff_evals[:,coefficient_id]
                )
        row_offset += num_clique_rows

    np.testing.assert_allclose(scalar_actual, scalar_expected)
    gramian_actual = {
        gramian_id: np.zeros_like(values)
        for gramian_id, values in gramian_expected.items()
    }
    for row, gramian_id, first, second, value in task.gramian_triplets:
        gramian_actual[gramian_id][row,first,second] += (
            value if first == second else 2.0*value
        )
    for gramian_id in gramian_expected:
        np.testing.assert_allclose(
            gramian_actual[gramian_id],
            gramian_expected[gramian_id],
        )

    stats = worker.stats
    assert stats['num_scalar_entries_emitted'] == len(task.scalar_triplets)
    assert stats['num_gramian_entries_emitted'] == len(task.gramian_triplets)
    assert stats['native']['num_rows'] == stats['num_rows']
    assert stats['native']['num_scalar_coeffs'] == stats['num_scalar_coeffs']
    assert stats['native']['num_gramians'] == stats['num_gramians']
    assert (
        stats['native']['num_scalar_nonzeros_stored']
        < stats['num_scalar_entries_emitted']
    )
    assert (
        stats['native']['num_gramian_entries_stored_upper_bound']
        < stats['num_gramian_entries_emitted']
    )
    assert worker.close()
    assert task.disposed



def test_chain_emission_preserves_neighbor_coupling(monkeypatch):
    """Emit only local and neighboring couplings along a clique chain."""
    pop = make_chain_pop(num_cliques=8, num_samples=96)
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2)
    sdpr.compress(method='PQR')
    sdpr.finalize()

    fake_mosek = _RecordingMosek()
    monkeypatch.setattr(mosek_backend, '_import_mosek', lambda: fake_mosek)
    worker = build_mosek(
        sdpr,
        gap_tol=1e-8,
        feas_tol=1e-8,
        iter_limit=100,
        config={},
    )
    task = fake_mosek.last_task

    row_counts = [
        len(system['row_indices']) for system in sdpr._clique_systems
    ]
    assert row_counts == [10]*8
    row_cliques = np.repeat(np.arange(8), row_counts)
    scalar_columns = [set() for _ in range(8)]
    gramian_ids = [set() for _ in range(8)]
    for row, column, _ in task.scalar_triplets:
        scalar_columns[row_cliques[row]].add(int(column))
    for row, gramian_id, _, _, _ in task.gramian_triplets:
        gramian_ids[row_cliques[row]].add(int(gramian_id))

    for first in range(8):
        assert gramian_ids[first] == {first}
        for second in range(first + 1, 8):
            shared = scalar_columns[first] & scalar_columns[second]
            if second == first + 1:
                assert shared
            else:
                assert not shared

    assert worker.close()

def test_native_config_and_generic_tolerances_are_applied(monkeypatch):
    """Apply native parameters before explicit Rhiza tolerances."""
    config = {
        'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_PRIMAL',
        'MSK_IPAR_NUM_THREADS': '1',
    }
    worker, task, _ = _build_recording_worker(monkeypatch, config=config)

    assert task.parameters['MSK_IPAR_INTPNT_SOLVE_FORM'] == 'MSK_SOLVE_PRIMAL'
    assert task.parameters['MSK_IPAR_NUM_THREADS'] == '1'
    assert task.parameters['rel_gap'] == 1e-8
    assert task.parameters['pfeas'] == 1e-8
    assert task.parameters['dfeas'] == 1e-8
    assert task.parameters['iterations'] == 100
    worker.close()


def test_solve_reads_documented_native_information(monkeypatch, capsys):
    """Normalize solve status and documented native information items."""
    worker, task, sdpr = _build_recording_worker(monkeypatch)
    report = worker.solve(verbose=True)
    output = capsys.readouterr().out

    assert output == "recording solve\n"
    assert report['success']
    assert report['solution_status'] == 'OPTIMAL'
    assert 'problem_status' not in report
    assert worker.stats['native']['problem_status'] == 'PRIM_FEAS'
    assert worker.stats['native']['solution_status'] == 'OPTIMAL'
    assert 'termination_status' not in report
    assert worker.stats['native']['termination_status'] == 'OK'
    expected_objective = (
        -2.0*sdpr.stats['objective_scale']
        + sdpr.stats['objective_offset']
    )
    assert report['primal_objective'] == expected_objective
    assert report['dual_objective'] == expected_objective
    assert report['relative_gap'] == 0.0
    assert report['num_iterations'] == 7
    factorization = worker.stats['factorization']
    assert factorization == {
        'solve_form': 'PRIMAL',
        'num_nonzeros': 456,
        'dense_dimension': 3,
        'num_flops': 890000.0,
        'num_threads': 1,
    }
    assert worker.stats['memory'] == {
        'num_bytes_current': 1234,
        'num_bytes_peak': 5678,
    }
    assert task.disposed
    repeated = worker.solve(verbose=False)
    assert not repeated['changed']
    assert worker.close()


def test_unavailable_memory_information_is_silent_after_verbose_solve(
    monkeypatch,
    capsys,
):
    """Do not print a handled MOSEK memory-information response as an error."""
    worker, task, _ = _build_recording_worker(monkeypatch)

    def unavailable_memory():
        """Simulate MOSEK reporting unavailable post-solve memory data."""
        if task.stream is not None and task.stream[1] is not None:
            task.stream[1](
                "MOSEK error 1081 (MSK_RES_ERR_SPACE_NO_INFO): "
                "No available information about the space usage.\n"
            )
        raise RuntimeError("memory information unavailable")

    task.getmemusage = unavailable_memory
    report = worker.solve(verbose=True)
    output = capsys.readouterr().out

    assert report['success']
    assert output == "recording solve\n"
    assert worker.stats['memory'] == {
        'num_bytes_current': None,
        'num_bytes_peak': None,
    }
    assert worker.close()


def test_nonfinite_mosek_objective_is_rejected(monkeypatch):
    """Reject a successful native status carrying a nonfinite objective."""
    worker, task, _ = _build_recording_worker(monkeypatch)
    task.getprimalobj = lambda solution_type: np.nan

    with pytest.raises(SolveError, match='nonfinite primal objective'):
        worker.solve(verbose=False)

    assert not task.disposed
    assert worker.close()
    assert task.disposed
