"""SCS scaling tests for reduced non-RIP overlap structure."""

import pytest

import rhiza as rz
from tests.fixtures.overlap_problem import make_shared_core_pop


################################################################################
# Test configuration
################################################################################


pytestmark = [pytest.mark.scs, pytest.mark.scaling]


################################################################################
# Shared-core scaling
################################################################################


def test_scs_shared_core_overlap_scales_linearly():
    """Reduce quadratic candidate overlaps to a linear cone formulation."""
    results = []
    for num_cliques in (8, 16, 32):
        pop = make_shared_core_pop(
            num_cliques=num_cliques,
            num_samples=96,
        )
        pop.finalize(graph_policy='OVERLAP')
        sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=2)
        analysis = sdpr.analyze()
        assert analysis['sample_margin_check'] == 'POSITIVE_MARGIN'
        compression = sdpr.compress(method='PQR')
        assert compression['num_rows_retained'] == 6*num_cliques
        sdpr.finalize()

        with rz.Solver(sdpr) as solver:
            report = solver.solve()
            assert report['success']
            assert report['solution_status'] == 'OPTIMAL'
            assert abs(report['primal_objective']) < 2e-4
            stats = solver.stats

        assert pop.csp.stats['num_candidate_overlap_pairs'] == (
            num_cliques*(num_cliques - 1)//2
        )
        assert sdpr.stats['num_separator_edges'] == num_cliques - 1
        assert sdpr.stats['num_separator_directions_reduced_topological'] == (
            2*(num_cliques - 1)
        )
        results.append({
            'num_cliques': num_cliques,
            'num_matrix_nonzeros': stats['num_matrix_nonzeros'],
            'num_scalar_coeffs': sdpr.stats['num_scalar_coeffs'],
        })

    for previous, current in zip(results, results[1:]):
        assert current['num_matrix_nonzeros'] < (
            2.2*previous['num_matrix_nonzeros']
        )
        assert current['num_scalar_coeffs'] < (
            2.2*previous['num_scalar_coeffs']
        )
