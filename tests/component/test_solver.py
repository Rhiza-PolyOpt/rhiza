"""Component tests for Solver lifecycle using a fake backend worker."""

import json

import numpy as np
import pytest

import rhiza as rz
import rhiza.solver as solver_module
from rhiza.diagnostics import ExtractionError, PipelineError, SolveError
from tests.component.test_extraction import (
    duals_for_weights,
    make_extraction_sdpr,
)


################################################################################
# Fake backend
################################################################################


class FakeWorker:
    """Provide deterministic solve data without a native SDP solver."""

    def __init__(self, system_duals):
        """Create a fake built worker.

        Inputs:
            system_duals:
                ndarray. Duals returned after solve.
        """
        self._system_duals = np.asarray(system_duals, dtype=float)
        self._closed = False
        self.num_solves = 0
        self.num_dual_reads = 0

    @property
    def built(self):
        """Return whether the fake worker remains open."""
        return not self._closed

    @property
    def closed(self):
        """Return whether the fake worker is closed."""
        return self._closed

    @property
    def stats(self):
        """Return small fake backend statistics."""
        return {'num_rows': len(self._system_duals)}

    def solve(self, *, verbose=False):
        """Return one deterministic optimal report."""
        self.num_solves += 1
        return {
            'success': True,
            'changed': True,
            'solution_status': 'OPTIMAL',
            'primal_objective': -1.0,
            'dual_objective': -1.0,
            'relative_gap': 0.0,
            'num_iterations': 3,
            'solve_time': 0.01,
        }

    def get_system_duals(self):
        """Return deterministic submitted-system duals."""
        self.num_dual_reads += 1
        return self._system_duals.copy()

    def close(self):
        """Close the fake worker idempotently."""
        if self._closed:
            return False
        self._closed = True
        return True


################################################################################
# Solver lifecycle
################################################################################


def make_fake_solver(monkeypatch):
    """Build a Solver whose MOSEK builder returns a FakeWorker."""
    sdpr = make_extraction_sdpr()
    weights = {
        0: np.array([0.0, 1.0, 0.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0, 0.0]),
    }
    system_duals = duals_for_weights(sdpr, weights)
    captured = {}

    def build_fake(
        sdpr_argument,
        *,
        gap_tol,
        feas_tol,
        iter_limit,
        config,
    ):
        """Capture builder arguments and return a fake worker."""
        captured.update({
            'sdpr': sdpr_argument,
            'gap_tol': gap_tol,
            'feas_tol': feas_tol,
            'iter_limit': iter_limit,
            'config': config,
        })
        return FakeWorker(system_duals)

    monkeypatch.setattr(solver_module, 'build_mosek', build_fake)
    solver = rz.Solver(
        sdpr,
        backend='mosek',
        gap_tol=1e-7,
        iter_limit=12,
        config={'native': 'value'},
    )
    return solver, captured


def test_solver_build_and_tolerance_resolution(monkeypatch):
    """Normalize the backend name and default feasibility tolerance."""
    solver, captured = make_fake_solver(monkeypatch)
    assert solver.backend == 'MOSEK'
    assert solver.built
    assert captured['gap_tol'] == 1e-7
    assert captured['feas_tol'] == 1e-7
    assert captured['iter_limit'] == 12
    assert captured['config'] == {'native': 'value'}


def test_solver_configuration_is_read_only(monkeypatch):
    """Keep the public settings synchronized with the built native worker."""
    solver, _ = make_fake_solver(monkeypatch)
    settings = {
        'backend': 'SCS',
        'gap_tol': 1e-3,
        'feas_tol': 1e-3,
        'iter_limit': 2,
        'config': {},
    }
    for name, value in settings.items():
        with pytest.raises(AttributeError):
            setattr(solver, name, value)

    config = solver.config
    config['native'] = 'changed'
    assert solver.config == {'native': 'value'}
    solver.close()


def test_solver_iteration_limit_requires_an_exact_integer(monkeypatch):
    """Accept integer-valued reals but reject fractional iteration limits."""
    sdpr = make_extraction_sdpr()
    with pytest.raises(SolveError, match='integer'):
        rz.Solver(sdpr, iter_limit=2.5)

    captured = {}

    def build_fake(
        sdpr_argument,
        *,
        gap_tol,
        feas_tol,
        iter_limit,
        config,
    ):
        """Capture the exact iteration limit passed to SCS."""
        captured['iter_limit'] = iter_limit
        return FakeWorker(np.zeros(sdpr.stats['num_rows_retained']))

    monkeypatch.setattr(solver_module, 'build_scs', build_fake)
    solver = rz.Solver(sdpr, iter_limit=12.0)
    assert solver.iter_limit == 12
    assert captured['iter_limit'] == 12
    solver.close()


def test_solver_extraction_rejects_lossy_integer_inputs(monkeypatch):
    """Reject fractional moment exponents and clique identifiers."""
    solver, _ = make_fake_solver(monkeypatch)
    solver.solve()
    with pytest.raises(ExtractionError, match='integer'):
        solver.extract_moments({"bad": {"x": 1.5}})
    with pytest.raises(ExtractionError, match='integer'):
        solver.extract_clique_moments(
            {"x": {"x": 1}},
            cliques=(0.5,),
        )
    solver.close()


def test_solve_is_idempotent(monkeypatch):
    """Run the backend once and return the stored report later."""
    solver, _ = make_fake_solver(monkeypatch)
    first = solver.solve()
    second = solver.solve()
    assert first['changed']
    assert not second['changed']
    assert solver._worker.num_solves == 1


def test_solver_extraction(monkeypatch):
    """Reuse one compact functional reconstruction across moment queries."""
    solver, _ = make_fake_solver(monkeypatch)
    solver.solve()
    assert solver.extract_means() == {'x': 1.0, 'y': 20.0, 'z': 6.0}
    assert solver.extract_means(variables="x") == {'x': 1.0}
    assert solver.extract_moments({"x2": {"x": 2}}) == {'x2': 1.0}
    report = solver.extract_moment_diagnostics(variables=("y",))
    assert report['stitching_error_max_by_variable']['y'] == 0.0
    assert set(report['total_variation_by_clique']) == {0, 1}
    assert set(report['mass_condition_by_clique']) == {0, 1}
    assert set(report['mass_error_by_clique']) == {0, 1}
    assert solver._worker.num_dual_reads == 1


def test_extract_before_solve_fails(monkeypatch):
    """Reject extraction before a usable solve report exists."""
    solver, _ = make_fake_solver(monkeypatch)
    with pytest.raises(PipelineError):
        solver.extract_means()


def test_close_and_context_manager(monkeypatch):
    """Release backend resources idempotently and on context exit."""
    solver, _ = make_fake_solver(monkeypatch)
    with solver as context_solver:
        assert context_solver is solver
    assert solver.status['closed']
    assert not solver.close()


def test_requires_finalized_sdpr():
    """Reject Solver construction from an unfinalized SDPR."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    pop.add_objective([0.0, 1.0], ('x',), 1)
    pop.finalize()
    sdpr = rz.SDPR(pop)
    with pytest.raises(PipelineError):
        rz.Solver(sdpr)


def test_close_releases_cached_functionals(monkeypatch):
    """Discard cached compact functionals when extraction becomes unavailable."""
    solver, _ = make_fake_solver(monkeypatch)
    solver.solve()
    solver.extract_means()
    assert solver._clique_functionals_cache is not None
    solver.close()
    assert solver._clique_functionals_cache is None
    with pytest.raises(PipelineError):
        solver.extract_means()


def test_default_backend_is_scs_with_backend_specific_tolerances(monkeypatch):
    """Select SCS by default and resolve its default accuracy."""
    sdpr = make_extraction_sdpr()
    captured = {}

    def build_fake(
        sdpr_argument,
        *,
        gap_tol,
        feas_tol,
        iter_limit,
        config,
    ):
        """Capture default SCS builder arguments."""
        captured.update({
            'sdpr': sdpr_argument,
            'gap_tol': gap_tol,
            'feas_tol': feas_tol,
            'iter_limit': iter_limit,
            'config': config,
        })
        return FakeWorker(np.zeros(sdpr.stats['num_rows_retained']))

    monkeypatch.setattr(solver_module, 'build_scs', build_fake)
    solver = rz.Solver(sdpr)
    assert solver.backend == 'SCS'
    assert solver.gap_tol == 1e-5
    assert solver.feas_tol == 1e-5
    assert captured['gap_tol'] == 1e-5
    assert captured['feas_tol'] == 1e-5


def test_mosek_default_tolerance_remains_tighter(monkeypatch):
    """Resolve MOSEK's default accuracy independently of SCS."""
    sdpr = make_extraction_sdpr()
    captured = {}

    def build_fake(
        sdpr_argument,
        *,
        gap_tol,
        feas_tol,
        iter_limit,
        config,
    ):
        """Capture default MOSEK builder arguments."""
        captured['gap_tol'] = gap_tol
        captured['feas_tol'] = feas_tol
        return FakeWorker(np.zeros(sdpr.stats['num_rows_retained']))

    monkeypatch.setattr(solver_module, 'build_mosek', build_fake)
    solver = rz.Solver(sdpr, backend='MOSEK')
    assert solver.gap_tol == 1e-8
    assert solver.feas_tol == 1e-8
    assert captured == {'gap_tol': 1e-8, 'feas_tol': 1e-8}


def test_solver_public_reports_are_json_serializable(monkeypatch):
    """Expose ordinary Python scalar leaves across the Solver lifecycle."""
    solver, _ = make_fake_solver(monkeypatch)
    json.dumps(solver.status)
    json.dumps(solver.stats)
    json.dumps(solver.solve())
    json.dumps(solver.status)
    json.dumps(solver.stats)
    json.dumps(solver.extract_moment_diagnostics())
    solver.close()
