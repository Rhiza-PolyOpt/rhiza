"""Numerical integration tests for non-RIP overlap relaxations."""

import numpy as np
import pytest

import rhiza as rz
from rhiza.diagnostics import BackendUnavailableError
from tests.fixtures.overlap_problem import (
    OVERLAP_OBJECTIVE,
    OVERLAP_SOLUTION,
    make_overlap_triangle_pop,
)


################################################################################
# SCS integration
################################################################################


@pytest.mark.scs
def test_scs_overlap_matches_dense_quadratic_problem():
    """Match a dense quadratic certificate without chordal extension."""
    results = {}
    for label, use_dense_scope in (
        ('overlap', False),
        ('dense', True),
    ):
        pop = make_overlap_triangle_pop(
            num_samples=180,
            use_dense_scope=use_dense_scope,
        )
        pop.finalize(
            graph_policy='RIP' if use_dense_scope else 'OVERLAP'
        )
        sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=2)
        analysis = sdpr.analyze()
        assert analysis['sample_margin_check'] == 'POSITIVE_MARGIN'
        sdpr.compress(method='PQR')
        sdpr.finalize()

        with rz.Solver(
            sdpr,
            gap_tol=1e-6,
            feas_tol=1e-6,
        ) as solver:
            report = solver.solve()
            assert report['success']
            assert report['solution_status'] == 'OPTIMAL'
            means = solver.extract_means()
            estimate = np.array([
                means['x'], means['y'], means['z']
            ])
            results[label] = (report, estimate, sdpr.stats)

    overlap_report, overlap_estimate, overlap_stats = results['overlap']
    dense_report, dense_estimate, dense_stats = results['dense']
    np.testing.assert_allclose(
        overlap_estimate,
        OVERLAP_SOLUTION,
        rtol=0.0,
        atol=3e-4,
    )
    np.testing.assert_allclose(
        dense_estimate,
        OVERLAP_SOLUTION,
        rtol=0.0,
        atol=3e-4,
    )
    np.testing.assert_allclose(
        overlap_report['primal_objective'],
        dense_report['primal_objective'],
        rtol=0.0,
        atol=2e-4,
    )
    assert abs(overlap_report['primal_objective'] - OVERLAP_OBJECTIVE) < 2e-4
    assert overlap_stats['num_cliques'] == 3
    assert overlap_stats['num_separator_edges'] == 3
    assert overlap_stats['num_separator_cycles'] == 1
    assert dense_stats['num_cliques'] == 1


@pytest.mark.scs
def test_overlap_and_component_splitting_preserve_one_scs_relaxation():
    """Exercise component-localized Gramians inside a non-RIP OVERLAP model."""
    num_samples = 120
    rng = np.random.default_rng(123)
    samples = {
        name: values
        for name, values in zip(
            ('a', 'b', 'c'),
            rng.normal(size=(3, num_samples)),
        )
    }
    endpoint_pattern = np.tile((0.0, 1.0), num_samples//2)
    scopes = (('a', 'b'), ('b', 'c'), ('a', 'c'))

    pop = rz.POP()
    for name, values in samples.items():
        pop.add_variable(name, values)
    for index, (first, second) in enumerate(scopes):
        branch_name = f"q{index}"
        first_product = f"{branch_name}_{first}"
        second_product = f"{branch_name}_{second}"
        branch = np.roll(endpoint_pattern, index)
        first_values = samples[first]
        second_values = samples[second]

        pop.add_variable(branch_name, branch)
        pop.add_product((branch_name, first), name=first_product)
        pop.add_product((branch_name, second), name=second_product)
        first_branch = branch*first_values
        second_branch = branch*second_values
        objective = (
            (first_values - first_branch)**2
            + first_branch**2
            + (second_values - second_branch)**2
            + second_branch**2
            + 0.05*branch
        )
        pop.add_objective(
            objective,
            (
                first,
                second,
                branch_name,
                first_product,
                second_product,
            ),
            2,
            name=f"cost_{index}",
        )
    pop.finalize(graph_policy='OVERLAP')

    split = rz.SDPR(
        pop,
        sos_degree=2,
        sep_degree=2,
        enable_component_splitting=True,
    )
    unsplit = rz.SDPR(
        pop,
        sos_degree=2,
        sep_degree=2,
        enable_component_splitting=False,
    )

    assert not pop.csp.status['rip']
    assert split.stats['num_separator_cycles'] == 1
    assert split.stats['num_component_split_gramians'] == 3
    assert split.stats['num_psd_coeffs'] < unsplit.stats['num_psd_coeffs']
    assert [item['rank'] for item in split.analyze()['cliques']] == [
        item['rank'] for item in unsplit.analyze()['cliques']
    ]

    split.finalize()
    unsplit.finalize()
    with rz.Solver(split, backend='SCS') as solver:
        split_report = solver.solve()
    with rz.Solver(unsplit, backend='SCS') as solver:
        unsplit_report = solver.solve()

    assert split_report['success']
    assert unsplit_report['success']
    assert abs(split_report['primal_objective']) < 2e-5
    assert abs(unsplit_report['primal_objective']) < 2e-5
    assert abs(
        split_report['primal_objective']
        - unsplit_report['primal_objective']
    ) < 5e-5

################################################################################
# MOSEK integration
################################################################################


@pytest.mark.mosek
def test_mosek_overlap_matches_dense_quadratic_problem():
    """Match the dense certificate with the licensed native backend."""
    pytest.importorskip('mosek')
    results = {}
    for label, use_dense_scope in (
        ('overlap', False),
        ('dense', True),
    ):
        pop = make_overlap_triangle_pop(
            num_samples=180,
            use_dense_scope=use_dense_scope,
        )
        pop.finalize(
            graph_policy='RIP' if use_dense_scope else 'OVERLAP'
        )
        sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=2)
        sdpr.compress(method='PQR')
        sdpr.finalize()
        try:
            solver = rz.Solver(sdpr, backend='MOSEK')
        except BackendUnavailableError:
            pytest.skip("MOSEK is unavailable in this environment.")
        with solver:
            try:
                report = solver.solve()
            except BackendUnavailableError:
                pytest.skip("A usable MOSEK license is unavailable.")
            assert report['success']
            means = solver.extract_means()
            estimate = np.array([
                means['x'], means['y'], means['z']
            ])
            results[label] = (report, estimate)

    overlap_report, overlap_estimate = results['overlap']
    dense_report, dense_estimate = results['dense']
    np.testing.assert_allclose(
        overlap_estimate,
        OVERLAP_SOLUTION,
        rtol=0.0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        dense_estimate,
        OVERLAP_SOLUTION,
        rtol=0.0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        overlap_report['primal_objective'],
        dense_report['primal_objective'],
        rtol=0.0,
        atol=5e-7,
    )
