"""Component tests for exact degree-bounded Gramian splitting."""

import numpy as np
import pytest

import rhiza as rz
from rhiza.diagnostics import SDPRError
import rhiza.sdpr as sdpr_module
from rhiza.sdpr import _count_small_exact_value_set
from tests.fixtures.component_problem import (
    make_binary_component_pop,
    make_pose_product_component_pop,
    make_ternary_component_pop,
)


def test_finite_value_probe_stops_after_the_component_cap():
    """Recognize small exact sets while declining a continuous column."""
    assert _count_small_exact_value_set((0, 1, 0, 1), 8) == 2
    assert _count_small_exact_value_set(range(20), 8) is None


def test_finite_value_probe_is_cached_by_global_variable(monkeypatch):
    """Scan each sampled variable once across repeated clique scopes."""
    num_samples = 120
    rng = np.random.default_rng(97)
    x_evals = rng.normal(size=num_samples)
    binary_evals = np.tile((0.0, 1.0), num_samples//2)

    pop = rz.POP()
    pop.add_variable("x", x_evals)
    for index in range(3):
        b_name = f"b_{index}"
        bx_name = f"bx_{index}"
        local_binary = np.roll(binary_evals, index)
        pop.add_variable(b_name, local_binary)
        pop.add_product((b_name, "x"), name=bx_name)
        pop.add_objective(
            (x_evals - 0.1*index)**2 + 0.01*local_binary,
            ("x", b_name, bx_name),
            2,
            name=f"cost_{index}",
        )
    pop.finalize()

    calls = 0
    original = sdpr_module._count_small_exact_value_set

    def counted(values, maximum):
        """Count finite-value probes before delegating to the implementation."""
        nonlocal calls
        calls += 1
        return original(values, maximum)

    monkeypatch.setattr(
        sdpr_module,
        '_count_small_exact_value_set',
        counted,
    )
    sdpr = rz.SDPR(pop, sos_degree=2)

    assert sdpr.stats['num_component_split_gramians'] == 3
    assert calls == len(pop.variable_names)


################################################################################
# Structural equivalence
################################################################################


def test_binary_product_lift_splits_one_gramian_into_two_blocks():
    """Remove cross-component Gramian coordinates when the lift exposes them."""
    pop = make_binary_component_pop()
    pop.finalize()

    split = rz.SDPR(pop, sos_degree=2)
    unsplit = rz.SDPR(
        pop,
        sos_degree=2,
        enable_component_splitting=False,
    )

    assert split.enable_component_splitting
    assert split.stats['num_gramians'] == 2
    assert split.stats['num_component_split_gramians'] == 1
    assert split.stats['num_psd_coeffs'] == 6
    assert split.stats['num_psd_coeffs_without_component_splitting'] == 10
    assert unsplit.stats['num_gramians'] == 1
    assert unsplit.stats['num_bases'] == 1
    assert unsplit.stats['num_psd_coeffs'] == 10
    assert (
        unsplit.stats['num_psd_coeffs_without_component_splitting']
        == 10
    )

    parent = split._bases[0]
    component_bases = [
        split._bases[basis_id]
        for basis_id in parent['component_basis_ids']
    ]
    assert [basis['basis_rank'] for basis in component_bases] == [2, 2]
    assert [basis['component_label'] for basis in component_bases] == [
        (0.0,),
        (1.0,),
    ]
    assert all(
        np.shares_memory(basis['basis_evals'], parent['basis_evals'])
        for basis in component_bases
    )
    assert np.allclose(
        parent['basis_evals'].T@parent['basis_evals'],
        np.eye(parent['basis_rank']),
        atol=1e-12,
    )
    b_evals = pop._variable_evals[:,pop._variable_ids['b']]
    assert np.max(np.abs(component_bases[0]['basis_evals'][b_evals == 1])) < 1e-12
    assert np.max(np.abs(component_bases[1]['basis_evals'][b_evals == 0])) < 1e-12


def test_split_and_unsplit_coefficient_spaces_are_identical():
    """Preserve the complete sampled clique-system column span."""
    pop = make_binary_component_pop(include_constant_inequality=True)
    pop.finalize()
    split = rz.SDPR(pop, sos_degree=2)
    unsplit = rz.SDPR(
        pop,
        sos_degree=2,
        enable_component_splitting=False,
    )

    split_evals = split._materialize_clique_system_evals(0)
    unsplit_evals = unsplit._materialize_clique_system_evals(0)
    assert _matrix_rank(split_evals) == _matrix_rank(unsplit_evals)
    assert _span_residual(split_evals, unsplit_evals) < 1e-11
    assert _span_residual(unsplit_evals, split_evals) < 1e-11
    # Both the base SOS and the constant inequality multiplier use the same
    # validated two-component quotient basis.
    assert split.stats['num_component_split_gramians'] == 2
    assert split.stats['num_gramians'] == 4


def test_split_and_unsplit_analysis_have_the_same_system_rank():
    """Preserve clique-system rank through analysis and PQR compression."""
    pop = make_binary_component_pop(include_constant_inequality=True)
    pop.finalize()
    split = rz.SDPR(pop, sos_degree=2)
    unsplit = rz.SDPR(
        pop,
        sos_degree=2,
        enable_component_splitting=False,
    )

    split_analysis = split.analyze()
    unsplit_analysis = unsplit.analyze()
    assert split_analysis['cliques'][0]['rank'] == (
        unsplit_analysis['cliques'][0]['rank']
    )

    split_compression = split.compress()
    unsplit_compression = unsplit.compress()
    assert split_compression['cliques'][0]['rank'] == (
        unsplit_compression['cliques'][0]['rank']
    )
    assert split_compression['cliques'][0]['row_span_residual'] < 1e-10
    assert unsplit_compression['cliques'][0]['row_span_residual'] < 1e-10


def test_split_and_unsplit_compression_preserve_the_same_system_rank():
    """Exercise component blocks through analysis and PQR row selection."""
    pop = make_binary_component_pop(num_samples=140)
    pop.finalize()
    split = rz.SDPR(pop, sos_degree=2)
    unsplit = rz.SDPR(
        pop,
        sos_degree=2,
        enable_component_splitting=False,
    )

    split_report = split.compress(method='PQR')
    unsplit_report = unsplit.compress(method='PQR')
    assert split_report['cliques'][0]['rank'] == (
        unsplit_report['cliques'][0]['rank']
    )
    assert split_report['cliques'][0]['num_rows_retained'] == (
        unsplit_report['cliques'][0]['num_rows_retained']
    )
    assert split_report['cliques'][0]['row_span_residual'] < 1e-10
    assert unsplit_report['cliques'][0]['row_span_residual'] < 1e-10
    assert split.finalize()
    assert unsplit.finalize()


def test_unlifted_binary_space_is_not_split_at_degree_one():
    """Decline when localizing x requires the missing product b*x."""
    pop = make_binary_component_pop(include_product=False)
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2)

    assert sdpr.stats['num_gramians'] == 1
    assert sdpr.stats['num_component_split_gramians'] == 0
    assert sdpr.stats['num_psd_coeffs'] == 6
    assert sdpr.stats['num_psd_coeffs_without_component_splitting'] == 6


def test_pose_product_lift_splits_as_thirteen_plus_thirteen():
    """Exercise the component dimensions used by PCR and PnP cliques."""
    pop = make_pose_product_component_pop()
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2)

    assert sdpr.stats['num_gramians'] == 2
    assert sdpr.stats['num_component_split_gramians'] == 1
    assert [
        gramian['gramian_size'] for gramian in sdpr._gramians
    ] == [13, 13]
    assert sdpr.stats['num_psd_coeffs'] == 182
    assert (
        sdpr.stats['num_psd_coeffs_without_component_splitting']
        == 351
    )


def test_ternary_lift_uses_three_component_blocks():
    """Discover the observed off, positive, and negative mode components."""
    pop = make_ternary_component_pop()
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2)

    assert sdpr.stats['num_gramians'] == 3
    assert sdpr.stats['num_component_split_gramians'] == 1
    assert [
        gramian['gramian_size'] for gramian in sdpr._gramians
    ] == [2, 2, 2]
    assert sdpr.stats['num_psd_coeffs'] == 9
    assert sdpr.stats['num_psd_coeffs_without_component_splitting'] == 21


def test_component_without_positive_sample_margin_is_not_split():
    """Require redundant rows on every proposed sample component."""
    pop = make_binary_component_pop(
        num_samples=20,
        positive_count=2,
    )
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2)

    assert sdpr.stats['num_gramians'] == 1
    assert sdpr.stats['num_component_split_gramians'] == 0


def test_rank_limited_basis_keeps_one_unsplit_gramian():
    """Avoid claiming an exact split after deliberate basis truncation."""
    pop = make_binary_component_pop()
    pop.finalize()
    sdpr = rz.SDPR(
        pop,
        sos_degree=2,
        basis_rank_limit=3,
    )

    assert sdpr.status['basis_rank_limited']
    assert sdpr.stats['num_gramians'] == 1
    assert sdpr.stats['num_component_split_gramians'] == 0


def test_component_splitting_setting_is_boolean_and_read_only():
    """Expose one explicit immutable switch for regression and diagnostics."""
    pop = make_binary_component_pop()
    pop.finalize()
    sdpr = rz.SDPR(
        pop,
        sos_degree=2,
        enable_component_splitting=False,
    )
    assert not sdpr.enable_component_splitting
    with pytest.raises(AttributeError):
        sdpr.enable_component_splitting = True
    with pytest.raises(SDPRError, match='Boolean'):
        rz.SDPR(pop, enable_component_splitting='yes')


################################################################################
# Native backends
################################################################################


@pytest.mark.scs
def test_scs_component_split_matches_unsplit_formulation():
    """Solve the exact split and unsplit parameterizations to the same optimum."""
    split_bound, split_mean = _solve_binary_problem(True, 'SCS')
    unsplit_bound, unsplit_mean = _solve_binary_problem(False, 'SCS')

    assert abs(split_bound) < 2e-5
    assert abs(unsplit_bound) < 2e-5
    assert abs(split_bound - unsplit_bound) < 2e-5
    assert abs(split_mean - 0.25) < 3e-4
    assert abs(unsplit_mean - 0.25) < 3e-4


@pytest.mark.scs
def test_scs_pose_product_split_matches_unsplit_formulation():
    """Exercise the 13-by-13 component blocks through native SCS."""
    split = _solve_pose_product_problem(True, 'SCS')
    unsplit = _solve_pose_product_problem(False, 'SCS')

    assert abs(split[0] - unsplit[0]) < 1e-5
    assert abs(split[1]) < 2e-4
    assert abs(unsplit[1]) < 2e-4
    assert split[2] < 2e-4
    assert unsplit[2] < 2e-4


@pytest.mark.mosek
def test_mosek_component_split_matches_unsplit_formulation():
    """Exercise multiple component PSD blocks in the licensed backend."""
    split_bound, split_mean = _solve_binary_problem(True, 'MOSEK')
    unsplit_bound, unsplit_mean = _solve_binary_problem(False, 'MOSEK')

    mean_tol = 1e-5
    assert abs(split_bound) < 1e-7
    assert abs(unsplit_bound) < 1e-7
    assert abs(split_bound - unsplit_bound) < 1e-7
    assert abs(split_mean - 0.25) < mean_tol
    assert abs(unsplit_mean - 0.25) < mean_tol
    assert abs(split_mean - unsplit_mean) < mean_tol


@pytest.mark.mosek
def test_mosek_pose_product_split_matches_unsplit_formulation():
    """Exercise the 13-by-13 component blocks through native MOSEK."""
    split = _solve_pose_product_problem(True, 'MOSEK')
    unsplit = _solve_pose_product_problem(False, 'MOSEK')

    assert abs(split[0] - unsplit[0]) < 1e-6
    assert abs(split[1]) < 2e-4
    assert abs(unsplit[1]) < 2e-4
    assert split[2] < 2e-4
    assert unsplit[2] < 2e-4


################################################################################
# Helpers
################################################################################


def _solve_binary_problem(enable_component_splitting, backend):
    """Solve one lifted binary-line problem and return its bound and x mean."""
    pop = make_binary_component_pop(num_samples=140)
    pop.finalize()
    sdpr = rz.SDPR(
        pop,
        sos_degree=2,
        enable_component_splitting=enable_component_splitting,
    )
    sdpr.finalize()
    tolerance = 1e-6 if backend == 'SCS' else 1e-8
    with rz.Solver(
        sdpr,
        backend=backend,
        gap_tol=tolerance,
        feas_tol=tolerance,
    ) as solver:
        report = solver.solve()
        assert report['success']
        means = solver.extract_means()
    return report['primal_objective'], means['x']


def _solve_pose_product_problem(enable_component_splitting, backend):
    """Solve the robust-pose component fixture and summarize its means."""
    pop = make_pose_product_component_pop()
    pop.finalize()
    sdpr = rz.SDPR(
        pop,
        sos_degree=2,
        enable_component_splitting=enable_component_splitting,
    )
    sdpr.finalize()
    tolerance = 1e-6 if backend == 'SCS' else 1e-8
    with rz.Solver(
        sdpr,
        backend=backend,
        gap_tol=tolerance,
        feas_tol=tolerance,
    ) as solver:
        report = solver.solve()
        assert report['success']
        means = solver.extract_means()
    pose_error = max(
        abs(means[f'pose_{index}'] - 0.1)
        for index in range(12)
    )
    return report['primal_objective'], means['b'], pose_error


def _matrix_rank(matrix):
    """Return a stable relative rank for one test matrix."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0
    return int(np.count_nonzero(
        singular_values > 1e-10*singular_values[0]
    ))


def _span_residual(basis_matrix, target_matrix):
    """Return target residual after projection onto a sampled column span."""
    left, singular_values, _ = np.linalg.svd(
        basis_matrix,
        full_matrices=False,
    )
    rank = int(np.count_nonzero(
        singular_values > 1e-10*singular_values[0]
    ))
    basis = left[:,:rank]
    residual = target_matrix - basis@(basis.T@target_matrix)
    return float(np.linalg.norm(residual)/np.linalg.norm(target_matrix))
