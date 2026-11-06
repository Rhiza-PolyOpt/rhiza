"""Native SCS scaling tests for fixed-width clique chains."""

import pytest
import scs

import rhiza as rz
from tests.fixtures.scaling_problem import make_chain_pop


################################################################################
# Test configuration
################################################################################


pytestmark = [pytest.mark.scs, pytest.mark.scaling]


################################################################################
# Native SCS chain scaling
################################################################################


def test_scs_chain_matrix_and_solve_scale_linearly():
    """Preserve fixed-width chain structure in SCS cone data."""
    results = []
    for num_cliques in (8, 16, 32):
        pop = make_chain_pop(num_cliques=num_cliques, num_samples=96)
        pop.finalize()
        sdpr = rz.SDPR(pop, sos_degree=2, rank_tol=1e-10)
        analysis = sdpr.analyze()
        assert analysis['sample_margin_check'] == 'POSITIVE_MARGIN'
        compression = sdpr.compress(method='PQR')
        assert compression['num_rows_retained'] == 10*num_cliques
        sdpr.finalize()

        with rz.Solver(sdpr) as solver:
            report = solver.solve(verbose=False)
            assert report['success']
            assert report['solution_status'] == 'OPTIMAL'
            assert abs(report['primal_objective']) < 2e-3
            stats = solver.stats

        assert stats['num_rows'] == 10*num_cliques
        assert stats['num_gramians'] == num_cliques
        assert stats['num_zero_cone_rows'] == 3*num_cliques - 2
        assert stats['num_psd_cone_rows'] == 10*num_cliques
        assert stats['num_cone_rows'] == 13*num_cliques - 2
        expected_mode = (
            'AUTO' if hasattr(scs, 'LinearSolver') else 'SPARSE_DIRECT'
        )
        assert stats['native']['linear_system_mode'] == expected_mode
        results.append(stats)

    nonzeros = [stats['num_matrix_nonzeros'] for stats in results]
    for previous, current in zip(nonzeros, nonzeros[1:]):
        assert 1.7*previous < current < 2.3*previous
    per_clique = [
        stats['num_matrix_nonzeros']/num_cliques
        for stats, num_cliques in zip(results, (8, 16, 32))
    ]
    assert max(per_clique) < 1.15*min(per_clique)
