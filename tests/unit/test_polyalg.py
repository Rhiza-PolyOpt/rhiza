"""Tests for polynomial evaluation and quotient-basis construction."""

from math import comb

import numpy as np

from rhiza.tools.polyalg import (
    build_quotient_basis,
    build_sampled_basis,
    build_separator_basis,
    enumerate_exponents,
    evaluate_lift,
    evaluate_monomials,
    evaluate_sparse_monomials,
    gramian_size_to_num_coeffs,
    iter_gramian_eval_chunks,
    lower_triangle_indices,
    materialize_gramian_evals,
)


################################################################################
# Monomials and lifts
################################################################################


def test_exponent_enumeration_counts_and_degrees():
    """Enumerate exponent rows exactly once through the degree."""
    for num_variables in range(1, 5):
        for degree in range(5):
            exponents = enumerate_exponents(num_variables, degree)
            assert exponents.shape == (
                comb(num_variables + degree, degree),
                num_variables,
            )
            assert len({tuple(row) for row in exponents}) == len(exponents)
            assert np.all(np.sum(exponents, axis=1) <= degree)


def test_exponent_order_begins_with_constant_and_linear_terms():
    """Use deterministic degree-grouped monomial ordering."""
    exponents = enumerate_exponents(2, 2)
    np.testing.assert_array_equal(
        exponents,
        np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [2, 0],
            [1, 1],
            [0, 2],
        ]),
    )


def test_monomial_evaluation_matches_naive_reference():
    """Match direct products on randomized signed samples."""
    rng = np.random.default_rng(4)
    variable_evals = rng.normal(size=(9, 3))
    exponents = enumerate_exponents(3, 4)
    actual = evaluate_monomials(variable_evals, exponents)
    expected = np.empty_like(actual)
    for sample_id in range(variable_evals.shape[0]):
        for monomial_id, exponent in enumerate(exponents):
            expected[sample_id,monomial_id] = np.prod(
                variable_evals[sample_id]**exponent
            )
    np.testing.assert_allclose(actual, expected)


def test_sparse_monomial_evaluation_matches_dense_exponents():
    """Evaluate monomials from participant IDs without full exponent rows."""
    variable_evals = np.array([
        [2.0, 3.0, 5.0],
        [-1.0, 4.0, 2.0],
    ])
    monomials = (
        ((0,), (2,)),
        ((0, 2), (1, 2)),
        ((1, 2), (3, 1)),
    )
    actual = evaluate_sparse_monomials(variable_evals, monomials)
    expected = evaluate_monomials(
        variable_evals,
        np.array([
            [2, 0, 0],
            [1, 0, 2],
            [0, 3, 1],
        ]),
    )
    np.testing.assert_allclose(actual, expected)


def test_lift_evaluation():
    """Evaluate a coefficient-weighted sum of monomials."""
    variable_evals = np.array([
        [1.0, 2.0],
        [-1.0, 3.0],
    ])
    actual = evaluate_lift(
        variable_evals,
        np.array([[1, 1], [2, 0]]),
        np.array([2.0, -1.0]),
    )
    np.testing.assert_allclose(actual, [3.0, -7.0])


################################################################################
# Quotient bases
################################################################################


def test_boolean_quotient_basis_rank():
    """Reduce one Boolean variable through the relation x squared equals x."""
    variable_evals = np.array([[0.0], [1.0], [0.0], [1.0]])
    basis = build_quotient_basis(
        variable_evals,
        2,
        rank_tol=1e-12,
    )
    assert basis.num_monomials_raw == 3
    assert basis.basis_rank == 2


def test_circle_quotient_basis_rank():
    """Observe one quadratic relation on dense circle samples."""
    angles = np.linspace(0.0, 2.0*np.pi, 30, endpoint=False)
    variable_evals = np.column_stack((np.cos(angles), np.sin(angles)))
    basis = build_quotient_basis(
        variable_evals,
        2,
        rank_tol=1e-12,
    )
    assert basis.num_monomials_raw == 6
    assert basis.basis_rank == 5


def test_sampled_basis_centers_and_reduces_supplied_functions():
    """Build an owned basis from arbitrary sampled function columns."""
    x = np.linspace(-2.0, 2.0, 20)
    sampled_evals = np.column_stack((x, 2.0*x, x**2 + 3.0))
    sampled_evals -= np.mean(sampled_evals, axis=0, keepdims=True)
    basis = build_sampled_basis(
        sampled_evals,
        rank_tol=1e-12,
        basis_rank_limit=2,
    )
    assert basis.basis_rank == 2
    assert basis.basis_evals.base is None
    np.testing.assert_allclose(
        np.mean(basis.basis_evals, axis=0),
        0.0,
        atol=1e-14,
    )


def test_separator_basis_removes_constant_direction():
    """Retain only nonconstant polynomial functions on a separator."""
    variable_evals = np.array([[-1.0], [0.0], [1.0], [2.0]])
    basis = build_separator_basis(
        variable_evals,
        2,
        rank_tol=1e-12,
    )
    assert basis.basis_rank == 2
    np.testing.assert_allclose(
        np.mean(basis.basis_evals, axis=0),
        0.0,
        atol=1e-14,
    )


################################################################################
# Gramian coefficient evaluations
################################################################################


def test_gramian_evals_reconstruct_quadratic_form():
    """Reconstruct a Gramian including off-diagonal factors."""
    rng = np.random.default_rng(5)
    basis_evals = rng.normal(size=(7, 4))
    gramian = rng.normal(size=(4, 4))
    gramian = 0.5*(gramian + gramian.T)
    rows, columns = np.tril_indices(4)
    coeffs = gramian[rows,columns]
    regressor = materialize_gramian_evals(basis_evals)
    actual = regressor@coeffs
    expected = np.einsum(
        'si,ij,sj->s', basis_evals, gramian, basis_evals
    )
    np.testing.assert_allclose(actual, expected)


def test_gramian_chunks_match_materialization():
    """Concatenate bounded Gramian chunks without changing coefficients."""
    rng = np.random.default_rng(6)
    basis_evals = rng.normal(size=(8, 5))
    multiplier_evals = rng.normal(size=8)
    chunks = tuple(iter_gramian_eval_chunks(
        basis_evals,
        multiplier_evals=multiplier_evals,
        sign=-1.0,
        max_num_coeffs=17,
    ))
    actual = np.hstack([chunk.coeff_evals for chunk in chunks])
    expected = materialize_gramian_evals(
        basis_evals,
        multiplier_evals=multiplier_evals,
        sign=-1.0,
    )
    np.testing.assert_allclose(actual, expected)
    assert expected.shape[1] == gramian_size_to_num_coeffs(5)


def test_gramian_row_selection_and_scaling():
    """Apply selected rows and scales during chunk generation."""
    basis_evals = np.arange(12, dtype=float).reshape(4, 3) + 1.0
    row_indices = np.array([0, 2])
    row_scales = np.array([0.5, 2.0])
    actual = np.hstack([
        chunk.coeff_evals
        for chunk in iter_gramian_eval_chunks(
            basis_evals,
            row_indices=row_indices,
            row_scales=row_scales,
            max_num_coeffs=4,
        )
    ])
    expected = row_scales[:,None]*materialize_gramian_evals(
        basis_evals
    )[row_indices,:]
    np.testing.assert_allclose(actual, expected)


def test_retained_basis_does_not_keep_full_svd_allocation_alive():
    """Store an owned retained basis after a restrictive rank limit."""
    rng = np.random.default_rng(17)
    variable_evals = rng.normal(size=(80, 5))
    basis = build_quotient_basis(
        variable_evals,
        3,
        rank_tol=1e-12,
        basis_rank_limit=2,
    )
    assert basis.basis_evals.shape == (80, 2)
    assert basis.basis_evals.base is None

    separator = build_separator_basis(
        variable_evals[:,0:2],
        3,
        rank_tol=1e-12,
        basis_rank_limit=2,
    )
    assert separator.basis_evals.shape == (80, 2)
    assert separator.basis_evals.base is None


def test_combinatorial_caches_are_bounded():
    """Retain a generous working set without unbounded process growth."""
    enumerate_exponents.cache_clear()
    lower_triangle_indices.cache_clear()
    for value in range(1, 141):
        enumerate_exponents(1, value)
        lower_triangle_indices(value)
    assert enumerate_exponents.cache_info().maxsize == 128
    assert enumerate_exponents.cache_info().currsize == 128
    assert lower_triangle_indices.cache_info().maxsize == 128
    assert lower_triangle_indices.cache_info().currsize == 128
