"""Integration tests for product and general polynomial lifts."""

import numpy as np
import pytest

import rhiza as rz
from rhiza.diagnostics import BackendUnavailableError
from tests.fixtures.lift_problem import (
    LIFT_OBJECTIVE,
    LIFT_SOLUTION,
    lift_points,
    make_lift_problem,
)


################################################################################
# Backend-independent lift pipeline
################################################################################


def test_lifted_and_unlifted_recipes_represent_the_same_problem():
    """Compare independent lifted and unlifted problem constructions."""
    unlifted_pop = make_lift_problem(use_lifts=False)
    lifted_pop = make_lift_problem(use_lifts=True)
    assert np.min(np.linalg.norm(lift_points(unlifted_pop), axis=1)) > 1e-3
    assert np.min(np.linalg.norm(lift_points(lifted_pop), axis=1)) > 1e-3

    unlifted_objective = sum(
        entry['evals'] for entry in unlifted_pop._pending_entries
        if entry['relation'].value in ('MIN', 'MAX')
    )
    lifted_objective = sum(
        entry['evals'] for entry in lifted_pop._pending_entries
        if entry['relation'].value in ('MIN', 'MAX')
    )
    np.testing.assert_allclose(lifted_objective, unlifted_objective)

    lifted_entries = {
        entry['name']: entry for entry in lifted_pop._pending_entries
        if entry['name'] is not None
    }
    assert lifted_entries['product_cost']['scope'] == ('product',)
    assert lifted_entries['product_cost']['degree'] == 2
    assert lifted_entries['affine_cost']['scope'] == ('affine',)
    assert lifted_entries['affine_cost']['degree'] == 2

    unlifted_pop.finalize()
    lifted_pop.finalize()
    assert lifted_pop.stats['num_sampled_identities'] == 2
    assert set(lifted_pop.csp.cliques) == {
        ('x', 'y', 'product'),
        ('x', 'y', 'affine'),
    }
    assert lifted_pop.csp.structural_separator_scopes == (('x', 'y'),)

    unlifted_sdpr = rz.SDPR(unlifted_pop, sos_degree=4)
    lifted_sdpr = rz.SDPR(lifted_pop, sos_degree=2)
    unlifted_analysis = unlifted_sdpr.analyze()
    lifted_analysis = lifted_sdpr.analyze()
    assert unlifted_analysis['num_cliques'] == 1
    assert lifted_analysis['num_cliques'] == 2
    assert lifted_analysis['sample_margin_check'] == 'POSITIVE_MARGIN'
    assert all(
        report['sample_margin'] > 0
        for report in lifted_analysis['cliques']
    )
    assert unlifted_sdpr.compress(method='PQR')['changed']
    assert lifted_sdpr.compress(method='PQR')['changed']
    unlifted_sdpr.finalize()
    lifted_sdpr.finalize()


################################################################################
# MOSEK lift benchmark
################################################################################


@pytest.mark.mosek
def test_lifted_and_unlifted_problems_solve_to_the_same_optimum():
    """Solve independent formulations without sampling the optimizer."""
    pytest.importorskip('mosek')
    results = []
    for use_lifts, sos_degree in ((False, 4), (True, 2)):
        pop = make_lift_problem(use_lifts=use_lifts)
        assert np.min(np.linalg.norm(lift_points(pop), axis=1)) > 1e-3
        pop.finalize()
        sdpr = rz.SDPR(pop, sos_degree=sos_degree, rank_tol=1e-10)
        sdpr.compress(method='PQR')
        sdpr.finalize()
        try:
            solver = rz.Solver(sdpr, backend='MOSEK', gap_tol=1e-9)
        except BackendUnavailableError:
            pytest.skip("MOSEK is unavailable in this environment.")

        with solver:
            try:
                report = solver.solve()
            except BackendUnavailableError:
                pytest.skip("A usable MOSEK license is unavailable.")
            means = solver.extract_means(variables=('x', 'y'))
            coherence = solver.extract_moment_diagnostics(variables=('x', 'y'))
            results.append((report, means, coherence, solver.stats))

    for report, means, coherence, stats in results:
        assert report['success']
        assert report['primal_objective'] <= report['dual_objective'] + 2e-7
        np.testing.assert_allclose(
            report['primal_objective'], LIFT_OBJECTIVE, atol=2e-6, rtol=0.0
        )
        np.testing.assert_allclose(
            (means['x'], means['y']), LIFT_SOLUTION, atol=2e-4, rtol=0.0
        )
        np.testing.assert_allclose(
            tuple(coherence['mass_by_clique'].values()),
            1.0,
            atol=2e-5,
            rtol=0.0,
        )
        assert stats['native']['num_rows'] == stats['num_rows']
        assert stats['native']['num_gramians'] == stats['num_gramians']
    np.testing.assert_allclose(
        results[0][0]['primal_objective'],
        results[1][0]['primal_objective'],
        atol=2e-6,
        rtol=0.0,
    )


################################################################################
# SCS lift benchmark
################################################################################


@pytest.mark.scs
def test_scs_lifted_and_unlifted_problems_agree():
    """Solve both lift formulations with the default backend."""
    results = []
    for use_lifts, sos_degree in ((False, 4), (True, 2)):
        pop = make_lift_problem(use_lifts=use_lifts)
        assert np.min(np.linalg.norm(lift_points(pop), axis=1)) > 1e-3
        pop.finalize()
        sdpr = rz.SDPR(pop, sos_degree=sos_degree, rank_tol=1e-10)
        sdpr.compress(method='PQR')
        sdpr.finalize()
        with rz.Solver(sdpr) as solver:
            report = solver.solve()
            assert report['success']
            assert report['solution_status'] == 'OPTIMAL'
            means = solver.extract_means(variables=('x', 'y'))
            coherence = solver.extract_moment_diagnostics(variables=('x', 'y'))
            results.append((report, means, coherence))

    for report, means, coherence in results:
        np.testing.assert_allclose(
            report['primal_objective'],
            LIFT_OBJECTIVE,
            atol=2e-3,
            rtol=0.0,
        )
        # The minimum is quartically flat along y = -2*x, so objective
        # accuracy alone does not imply comparable coordinate accuracy.
        np.testing.assert_allclose(
            (means['x'], means['y']),
            LIFT_SOLUTION,
            atol=1e-2,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            tuple(coherence['mass_by_clique'].values()),
            1.0,
            atol=2e-3,
            rtol=0.0,
        )
    np.testing.assert_allclose(
        results[0][0]['primal_objective'],
        results[1][0]['primal_objective'],
        atol=2e-3,
        rtol=0.0,
    )
