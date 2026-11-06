"""Structural scaling tests for correlatively decomposed relaxations."""

import rhiza as rz
from tests.fixtures.scaling_problem import make_chain_pop


################################################################################
# Clique-chain formulation scaling
################################################################################


def test_chain_formulation_counts_grow_linearly():
    """Scale fixed-width clique systems linearly with chain length."""
    for num_cliques in (4, 8, 16):
        pop = make_chain_pop(num_cliques=num_cliques)
        pop.finalize()
        sdpr = rz.SDPR(pop, sos_degree=2)

        assert sdpr.stats['num_cliques'] == num_cliques
        assert sdpr.stats['num_separator_edges'] == num_cliques - 1
        assert sdpr.stats['num_gramians'] == num_cliques
        assert sdpr.stats['num_scalar_coeffs'] == 3*num_cliques - 2
        assert sdpr.stats['num_psd_coeffs'] == 10*num_cliques
        assert sdpr.stats['num_rows_original'] == 96*num_cliques

        analysis = sdpr.analyze()
        assert analysis['sample_margin_check'] == 'POSITIVE_MARGIN'
        assert all(
            report['rank'] == 10
            for report in analysis['cliques']
        )
        compression = sdpr.compress(method='PQR')
        assert compression['num_rows_retained'] == 10*num_cliques
        assert compression['num_rows_retained'] < compression['num_rows_original']


def test_dense_chain_gramian_grows_quadratically():
    """Distinguish correlatively decomposed and dense cone growth."""
    for num_cliques in (4, 8, 16):
        sparse_pop = make_chain_pop(
            num_cliques=num_cliques,
            use_sparse_scopes=True,
        )
        dense_pop = make_chain_pop(
            num_cliques=num_cliques,
            use_sparse_scopes=False,
        )
        sparse_pop.finalize()
        dense_pop.finalize()
        sparse_sdpr = rz.SDPR(sparse_pop, sos_degree=2)
        dense_sdpr = rz.SDPR(dense_pop, sos_degree=2)

        assert sparse_sdpr.stats['num_psd_coeffs'] == 10*num_cliques
        expected_dense_basis = 2*num_cliques + 2
        expected_dense_coeffs = expected_dense_basis*(
            expected_dense_basis + 1
        )//2
        assert dense_sdpr.stats['num_psd_coeffs'] == expected_dense_coeffs
        assert (
            dense_sdpr.stats['num_psd_coeffs']
            > sparse_sdpr.stats['num_psd_coeffs']
        )
