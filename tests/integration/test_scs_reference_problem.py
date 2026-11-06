"""SCS integration tests for the five-variable reference problem."""

import numpy as np
import pytest
import scs

import rhiza as rz
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


pytestmark = pytest.mark.scs


################################################################################
# Reference problem solves
################################################################################


@pytest.mark.parametrize(
    (
        'use_sampled_equalities, use_sparse_scopes, '
        'use_compression, num_samples'
    ),
    (
        (True, True, False, 220),
        (True, True, True, 220),
        (False, True, True, 300),
        (True, False, True, 420),
    ),
)
def test_scs_reference_variants(
    use_sampled_equalities,
    use_sparse_scopes,
    use_compression,
    num_samples,
):
    """Solve sampled, explicit, sparse, and dense reference variants."""
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
    sdpr = rz.SDPR(pop, sos_degree=4, rank_tol=1e-10)
    if use_compression:
        compression = sdpr.compress(method='PQR')
        assert (
            compression['num_rows_retained']
            < compression['num_rows_original']
        )
    sdpr.finalize()

    with rz.Solver(sdpr) as solver:
        report = solver.solve(verbose=False)
        assert report['success']
        assert report['solution_status'] == 'OPTIMAL'
        assert report['primal_objective'] is not None
        assert report['dual_objective'] is not None
        assert (
            report['primal_objective']
            <= report['dual_objective'] + 5e-3
        )
        bound_error = REFERENCE_OBJECTIVE - report['primal_objective']
        assert bound_error >= -5e-3
        assert bound_error <= 2e-2

        means = solver.extract_means()
        estimate = np.array([
            means[f"x{index}"] for index in range(5)
        ])
        np.testing.assert_allclose(
            estimate,
            REFERENCE_SOLUTION,
            rtol=0.0,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            evaluate_reference_objective(estimate),
            REFERENCE_OBJECTIVE,
            rtol=0.0,
            atol=3e-2,
        )
        inequalities, equalities = evaluate_reference_constraints(estimate)
        assert np.max(inequalities) <= 3e-2
        assert np.max(np.abs(equalities)) <= 3e-2

        coherence = solver.extract_moment_diagnostics()
        np.testing.assert_allclose(
            tuple(coherence['mass_by_clique'].values()),
            1.0,
            rtol=0.0,
            atol=3e-3,
        )
        assert coherence['stitching_error_max'] < 2e-2
        stats = solver.stats
        assert stats['num_rows'] == sdpr.stats['num_rows_retained']
        assert stats['num_gramians'] == sdpr.stats['num_gramians']
        assert stats['num_zero_cone_rows'] == sdpr.stats['num_scalar_coeffs']
        assert stats['num_matrix_nonzeros'] > 0
        expected_mode = (
            'AUTO' if hasattr(scs, 'LinearSolver') else 'SPARSE_DIRECT'
        )
        assert stats['native']['linear_system_mode'] == expected_mode


def test_scs_objective_and_inequality_sign_equivalence():
    """Preserve the solved bound under equivalent relation signs."""
    objectives = []
    means_all = []
    for objective_form, inequality_form in (
        ('MIN', '<=0'),
        ('MAX', '>=0'),
    ):
        pop = make_reference_pop(
            num_samples=220,
            objective_form=objective_form,
            inequality_form=inequality_form,
        )
        pop.finalize()
        sdpr = rz.SDPR(pop, sos_degree=4)
        sdpr.compress(method='PQR')
        sdpr.finalize()
        with rz.Solver(sdpr) as solver:
            report = solver.solve()
            assert report['success']
            assert report['solution_status'] == 'OPTIMAL'
            objectives.append(report['primal_objective'])
            means_all.append(solver.extract_means())

    np.testing.assert_allclose(
        objectives[0], objectives[1], rtol=0.0, atol=5e-3
    )
    for variable in means_all[0]:
        np.testing.assert_allclose(
            means_all[0][variable],
            means_all[1][variable],
            rtol=0.0,
            atol=5e-3,
        )
