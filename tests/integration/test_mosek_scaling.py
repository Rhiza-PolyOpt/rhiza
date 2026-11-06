"""Native MOSEK scaling benchmarks for fixed-width clique chains."""

import pytest

import rhiza as rz
from rhiza.diagnostics import BackendUnavailableError
from tests.fixtures.overlap_problem import make_shared_core_pop
from tests.fixtures.scaling_problem import make_chain_pop


################################################################################
# Test configuration
################################################################################


pytestmark = [pytest.mark.mosek, pytest.mark.scaling]

_CHAIN_SIZES = (32, 64, 128)


################################################################################
# Native solve helper
################################################################################


def _solve_chain(*, num_cliques, num_samples, use_sparse_scopes):
    """Build, compress, and solve one clique-chain formulation.

    Inputs:
        num_cliques:
            Integer. Number of fixed-width cliques.
        num_samples:
            Integer. Number of sampled points.
        use_sparse_scopes:
            Boolean. Whether objective entries keep local scopes.

    Outputs:
        result:
            Mapping. SDPR, backend, and factorization statistics.
    """
    pop = make_chain_pop(
        num_cliques=num_cliques,
        num_samples=num_samples,
        use_sparse_scopes=use_sparse_scopes,
    )
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2, rank_tol=1e-10)
    analysis = sdpr.analyze()
    assert analysis['sample_margin_check'] == 'POSITIVE_MARGIN'
    compression = sdpr.compress(method='PQR')
    assert compression['num_rows_retained'] < compression['num_rows_original']
    sdpr.finalize()

    config = {
        'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_PRIMAL',
        'MSK_IPAR_NUM_THREADS': '1',
    }
    try:
        solver = rz.Solver(
            sdpr,
            backend='MOSEK',
            gap_tol=1e-8,
            config=config,
        )
    except BackendUnavailableError:
        pytest.skip("MOSEK is unavailable in this environment.")

    with solver:
        try:
            report = solver.solve(verbose=False)
        except BackendUnavailableError:
            pytest.skip("A usable MOSEK license is unavailable.")
        assert report['success']
        stats = solver.stats

    native = stats['native']
    factorization = stats['factorization']
    assert native['num_rows'] == sdpr.stats['num_rows_retained']
    assert native['num_scalar_coeffs'] == sdpr.stats['num_scalar_coeffs']
    assert native['num_gramians'] == sdpr.stats['num_gramians']
    assert native['num_scalar_nonzeros_stored'] >= 0
    assert native['num_gramian_block_nonzeros_stored'] >= 0
    assert native['num_gramian_entries_stored_upper_bound'] >= 0
    assert factorization['solve_form'] == 'PRIMAL'
    assert factorization['num_threads'] == 1
    assert factorization['num_nonzeros'] is not None
    assert factorization['dense_dimension'] is not None
    assert factorization['num_flops'] is not None
    return {
        'num_cliques': num_cliques,
        'sdpr': sdpr.stats,
        'backend': stats,
        'factorization': factorization,
    }


################################################################################
# Native factorization scaling
################################################################################


def test_mosek_preserves_chain_factorization_scaling():
    """Keep primal factorization sparse along a growing clique chain."""
    pytest.importorskip('mosek')
    results = []
    for num_cliques in _CHAIN_SIZES:
        result = _solve_chain(
            num_cliques=num_cliques,
            num_samples=96,
            use_sparse_scopes=True,
        )
        results.append(result)
        print(
            f"{num_cliques} cliques: ",
            result['factorization'],
        )

    for result in results:
        num_cliques = result['num_cliques']
        num_rows = result['sdpr']['num_rows_retained']
        factorization = result['factorization']
        dense_count = num_rows*(num_rows + 1)//2

        assert num_rows == 10*num_cliques
        assert result['sdpr']['num_gramians'] == num_cliques
        assert (
            result['backend']['native'][
                'num_gramian_block_nonzeros_stored'
            ]
            == 10*num_cliques
        )
        assert factorization['dense_dimension'] <= max(4, num_rows//20)
        assert factorization['num_nonzeros'] < 0.30*dense_count

    nonzeros = [
        result['factorization']['num_nonzeros']
        for result in results
    ]
    flops = [
        result['factorization']['num_flops']
        for result in results
    ]
    for previous, current in zip(nonzeros, nonzeros[1:]):
        assert current < 2.75*previous
    for previous, current in zip(flops, flops[1:]):
        assert current < 4.0*previous

    per_clique = [
        result['factorization']['num_nonzeros']/result['num_cliques']
        for result in results
    ]
    assert max(per_clique) < 2.0*min(per_clique)


def test_mosek_dense_cone_contrast_is_fully_sampled():
    """Compare decomposed and dense cones after identifying both systems."""
    pytest.importorskip('mosek')
    sparse = _solve_chain(
        num_cliques=16,
        num_samples=640,
        use_sparse_scopes=True,
    )
    dense = _solve_chain(
        num_cliques=16,
        num_samples=640,
        use_sparse_scopes=False,
    )

    print("Sparse factorization:", sparse['factorization'])
    print("Dense factorization:", dense['factorization'])

    assert sparse['sdpr']['num_rows_retained'] == 160
    assert dense['sdpr']['num_rows_retained'] == 595
    assert (
        sparse['factorization']['num_nonzeros']
        < dense['factorization']['num_nonzeros']
    )
    assert (
        sparse['factorization']['num_flops']
        < dense['factorization']['num_flops']
    )


################################################################################
# Reduced-overlap factorization scaling
################################################################################


def test_mosek_preserves_shared_core_overlap_scaling():
    """Keep a reduced shared-core overlap formulation sparse in MOSEK."""
    pytest.importorskip('mosek')
    results = []
    for num_cliques in _CHAIN_SIZES:
        pop = make_shared_core_pop(
            num_cliques=num_cliques,
            num_samples=96,
        )
        pop.finalize(graph_policy='OVERLAP')
        sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=2)
        analysis = sdpr.analyze()
        assert analysis['sample_margin_check'] == 'POSITIVE_MARGIN'
        sdpr.compress(method='PQR')
        sdpr.finalize()

        config = {
            'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_PRIMAL',
            'MSK_IPAR_NUM_THREADS': '1',
        }
        try:
            solver = rz.Solver(
                sdpr,
                backend='MOSEK',
                config=config,
            )
        except BackendUnavailableError:
            pytest.skip("MOSEK is unavailable in this environment.")
        with solver:
            try:
                report = solver.solve()
            except BackendUnavailableError:
                pytest.skip("A usable MOSEK license is unavailable.")
            assert report['success']
            stats = solver.stats

        factorization = stats['factorization']
        print(
            f"{num_cliques} shared-core cliques: ",
            factorization,
        )
        assert sdpr.stats['num_separator_edges'] == num_cliques - 1
        assert pop.csp.stats['num_candidate_overlap_pairs'] == (
            num_cliques*(num_cliques - 1)//2
        )
        assert factorization['solve_form'] == 'PRIMAL'
        assert factorization['num_threads'] == 1
        assert factorization['dense_dimension'] <= 4
        assert (
            stats['native']['num_gramian_block_nonzeros_stored']
            == 6*num_cliques
        )
        results.append((num_cliques, factorization))

    for (previous_size, previous), (current_size, current) in zip(
        results, results[1:]
    ):
        assert current_size == 2*previous_size
        assert current['num_nonzeros'] < 2.2*previous['num_nonzeros']
        assert current['num_flops'] < 2.2*previous['num_flops']
