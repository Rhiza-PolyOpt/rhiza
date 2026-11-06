"""MOSEK integration benchmarks for the five-variable reference problem."""

import numpy as np
import pytest

import rhiza as rz
from rhiza.diagnostics import BackendUnavailableError
from tests.fixtures.reference_problem import (
    REFERENCE_OBJECTIVE,
    REFERENCE_SOLUTION,
    evaluate_reference_constraints,
    evaluate_reference_objective,
    make_reference_pop,
    reference_points,
)


################################################################################
# Test configuration
################################################################################


pytestmark = pytest.mark.mosek


################################################################################
# Numerical benchmark variants
################################################################################


@pytest.mark.parametrize(
    (
        'use_sampled_equalities',
        'use_sparse_scopes',
        'use_compression',
        'num_samples',
        'objective_tol',
    ),
    (
        (True, True, False, 220, 2e-6),
        (True, True, True, 220, 2e-6),
        (False, True, False, 260, 2e-5),
        (False, True, True, 260, 2e-5),
        (True, False, True, 420, 2e-6),
    ),
)
def test_mosek_reference_variants(
    use_sampled_equalities,
    use_sparse_scopes,
    use_compression,
    num_samples,
    objective_tol,
):
    """Solve sparse, dense, sampled, explicit, and compressed variants."""
    pytest.importorskip('mosek')
    pop = make_reference_pop(
        num_samples=num_samples,
        use_sampled_equalities=use_sampled_equalities,
        use_sparse_scopes=use_sparse_scopes,
    )
    distances = np.linalg.norm(
        reference_points(pop) - REFERENCE_SOLUTION,
        axis=1,
    )
    assert np.min(distances) > 1e-3
    pop.finalize()
    sdpr = rz.SDPR(
        pop,
        sos_degree=4,
        rank_tol=1e-10,
    )
    if use_compression:
        sdpr.compress(method='PQR')
    sdpr.finalize()

    try:
        solver = rz.Solver(
            sdpr,
            backend='MOSEK',
            gap_tol=1e-8,
            iter_limit=100,
        )
    except BackendUnavailableError:
        pytest.skip("MOSEK is unavailable in this environment.")

    with solver:
        try:
            report = solver.solve(verbose=False)
        except BackendUnavailableError:
            pytest.skip("A usable MOSEK license is unavailable.")

        assert report['success']
        assert report['dual_objective'] is not None
        assert report['relative_gap'] is not None
        assert report['relative_gap'] <= 1e-5
        assert (
            report['primal_objective']
            <= report['dual_objective'] + objective_tol
        )

        bound_error = REFERENCE_OBJECTIVE - report['primal_objective']
        assert bound_error >= -2e-6
        assert bound_error <= objective_tol

        means = solver.extract_means()
        estimate = np.array([
            means[f"x{index}"] for index in range(5)
        ])
        np.testing.assert_allclose(
            estimate,
            REFERENCE_SOLUTION,
            rtol=0.0,
            atol=3e-4,
        )
        np.testing.assert_allclose(
            evaluate_reference_objective(estimate),
            REFERENCE_OBJECTIVE,
            rtol=0.0,
            atol=2e-4,
        )
        inequalities, equalities = evaluate_reference_constraints(estimate)
        assert np.max(inequalities) <= 5e-4
        assert np.max(np.abs(equalities)) <= 5e-4

        coherence = solver.extract_moment_diagnostics()
        np.testing.assert_allclose(
            tuple(coherence['mass_by_clique'].values()),
            1.0,
            rtol=0.0,
            atol=2e-5,
        )
        assert coherence['stitching_error_max'] < 1e-4
        stats = solver.stats
        assert stats['num_rows'] == sdpr.stats['num_rows_retained']
        assert stats['num_gramians'] == sdpr.stats['num_gramians']
        assert stats['native']['num_rows'] == stats['num_rows']
        assert (
            stats['native']['num_scalar_coeffs']
            == stats['num_scalar_coeffs']
        )
        assert stats['native']['num_gramians'] == stats['num_gramians']
        assert (
            stats['native']['num_scalar_nonzeros_stored']
            <= stats['num_scalar_entries_emitted']
        )
        if use_compression:
            assert sdpr.stats['num_rows_retained'] < sdpr.stats['num_rows_original']


def test_scs_and_mosek_agree_on_the_same_finalized_sdpr():
    """Compare opposite conic orientations on one reference relaxation."""
    pytest.importorskip('mosek')
    pop = make_reference_pop(num_samples=220)
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=4, rank_tol=1e-10)
    sdpr.compress(method='PQR')
    sdpr.finalize()

    with rz.Solver(sdpr, backend='SCS') as scs_solver:
        scs_report = scs_solver.solve()
        assert scs_report['success']
        assert scs_report['solution_status'] == 'OPTIMAL'
        scs_means = scs_solver.extract_means()
        scs_masses = scs_solver.extract_moment_diagnostics()['mass_by_clique']

    try:
        mosek_solver = rz.Solver(sdpr, backend='MOSEK')
    except BackendUnavailableError:
        pytest.skip("MOSEK is unavailable in this environment.")
    with mosek_solver:
        try:
            mosek_report = mosek_solver.solve()
        except BackendUnavailableError:
            pytest.skip("A usable MOSEK license is unavailable.")
        assert mosek_report['success']
        mosek_means = mosek_solver.extract_means()
        mosek_masses = mosek_solver.extract_moment_diagnostics()['mass_by_clique']

    np.testing.assert_allclose(
        scs_report['primal_objective'],
        mosek_report['primal_objective'],
        rtol=0.0,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        tuple(scs_means.values()),
        tuple(mosek_means.values()),
        rtol=0.0,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        tuple(scs_masses.values()),
        tuple(mosek_masses.values()),
        rtol=0.0,
        atol=3e-3,
    )
