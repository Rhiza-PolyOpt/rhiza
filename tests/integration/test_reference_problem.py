"""Integration tests for the five-variable reference problem."""

import numpy as np
import pytest

import rhiza as rz
from tests.fixtures.reference_problem import (
    REFERENCE_OBJECTIVE,
    REFERENCE_SOLUTION,
    evaluate_reference_objective,
    make_reference_pop,
    reference_points,
)


################################################################################
# Reference POP and SDPR pipeline
################################################################################


def test_reference_solution_and_objective():
    """Confirm the known exact solution value used by integration tests."""
    np.testing.assert_allclose(
        evaluate_reference_objective(REFERENCE_SOLUTION),
        REFERENCE_OBJECTIVE,
        rtol=0.0,
        atol=1e-13,
    )


def test_reference_fixtures_do_not_sample_the_optimizer():
    """Keep ordinary integration samples away from the known optimizer."""
    variants = (
        make_reference_pop(use_sampled_equalities=True),
        make_reference_pop(use_sampled_equalities=False),
        make_reference_pop(use_sparse_scopes=False),
    )
    for pop in variants:
        distances = np.linalg.norm(
            reference_points(pop) - REFERENCE_SOLUTION,
            axis=1,
        )
        assert np.min(distances) > 1e-3


def test_sparse_sampled_equality_pipeline():
    """Run POP, CSP, SDPR, analysis, PQR, and finalization end to end."""
    pop = make_reference_pop(num_samples=180)
    assert pop.finalize(graph_policy='RIP')
    assert pop.csp.cliques == (
        ('x0', 'x1'),
        ('x1', 'x2'),
        ('x2', 'x3', 'x4'),
    )
    assert pop.csp.structural_separator_scopes == (('x1',), ('x2',))
    assert pop.stats['num_sampled_identities'] == 2

    sdpr = rz.SDPR(
        pop,
        sos_degree=4,
        rank_tol=1e-10,
    )
    analysis = sdpr.analyze()
    compression = sdpr.compress(method='PQR')
    assert analysis['num_cliques'] == 3
    assert analysis['sample_margin_check'] == 'POSITIVE_MARGIN'
    assert all(
        report['sample_margin'] > 0
        for report in analysis['cliques']
    )
    assert compression['num_rows_retained'] < compression['num_rows_original']
    assert sdpr.finalize()
    assert sdpr.finalized


@pytest.mark.scs
def test_selective_compression_preserves_reference_objective():
    """Solve equivalent mixed compressed and uncompressed row systems."""
    objectives = []
    for threshold in (None, 2):
        pop = make_reference_pop(num_samples=180)
        pop.finalize(graph_policy='RIP')
        sdpr = rz.SDPR(pop, sos_degree=4, rank_tol=1e-10)
        if threshold is not None:
            report = sdpr.compress(
                method='PQR',
                clique_size_threshold=threshold,
            )
            assert report['num_cliques_selected'] == 2
            assert report['num_cliques_skipped'] == 1
        sdpr.finalize()

        with rz.Solver(sdpr, backend='SCS') as solver:
            solve_report = solver.solve()
            assert solve_report['success']
            objectives.append(solve_report['primal_objective'])

    np.testing.assert_allclose(
        objectives,
        REFERENCE_OBJECTIVE,
        atol=2e-4,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        objectives[0], objectives[1], atol=2e-4, rtol=0.0
    )


def test_sparse_and_dense_objective_partition_agree():
    """Preserve the objective under sparse and dense scopes."""
    sparse_pop = make_reference_pop(num_samples=120, use_sparse_scopes=True)
    dense_pop = make_reference_pop(num_samples=120, use_sparse_scopes=False)
    sparse_pop.finalize()
    dense_pop.finalize()
    sparse_sdpr = rz.SDPR(sparse_pop, sos_degree=4)
    dense_sdpr = rz.SDPR(dense_pop, sos_degree=4)
    # SDPR stores an exactly centered and commonly scaled objective. Undo that
    # affine transform before comparing two different clique partitions.
    sparse_total = (
        sparse_sdpr.stats['objective_scale']*sum(
            system['objective_evals']
            for system in sparse_sdpr._clique_systems
        )
        + sparse_sdpr.stats['objective_offset']
    )
    dense_total = (
        dense_sdpr.stats['objective_scale']
        * dense_sdpr._clique_systems[0]['objective_evals']
        + dense_sdpr.stats['objective_offset']
    )
    np.testing.assert_allclose(sparse_total, dense_total)


def test_explicit_equality_pipeline():
    """Build the reference relaxation with free equality multipliers."""
    pop = make_reference_pop(
        num_samples=160,
        use_sampled_equalities=False,
    )
    pop.finalize()
    assert pop.stats['num_sampled_identities'] == 0
    sdpr = rz.SDPR(pop, sos_degree=4)
    assert sum(
        block['kind'] == 'EQUALITY_MULTIPLIER'
        for block in sdpr._scalar_blocks
    ) == 2
    sdpr.finalize()
    assert sdpr.finalized


def test_file_and_python_reference_pops_agree(tmp_path):
    """Round-trip the reference POP while intentionally losing entry names."""
    original = make_reference_pop(num_samples=90)
    original.finalize()
    path = tmp_path/'reference.pop'
    original.save(path)

    loaded = rz.POP(path)
    assert loaded.entry_names == ()
    loaded.finalize()
    assert loaded.variable_names == original.variable_names
    np.testing.assert_array_equal(
        loaded._variable_evals, original._variable_evals
    )
    np.testing.assert_array_equal(
        loaded._entry_evals, original._entry_evals
    )
    assert loaded.csp.cliques == original.csp.cliques
