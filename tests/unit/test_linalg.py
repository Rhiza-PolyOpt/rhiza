"""Tests for scale-aware numerical-rank analysis."""

import numpy as np

import rhiza.tools.linalg as linalg
from rhiza.tools.linalg import (
    analyze_balanced_system_rank,
    analyze_rank,
    compute_row_span_residual,
    equilibrate_columns,
    normalize_rows,
    stable_euclidean_norm,
)


################################################################################
# Rank analysis
################################################################################


def test_exact_full_and_deficient_rank():
    """Identify exact full-rank and rank-deficient matrices."""
    full = analyze_rank(np.eye(3), rank_tol=1e-12)
    deficient = analyze_rank(
        np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        rank_tol=1e-12,
    )
    assert full.rank == 3
    assert full.sample_margin == 0
    assert deficient.rank == 1
    assert deficient.sample_margin == 2


def test_rank_is_column_scale_invariant():
    """Preserve rank under extreme nonsingular column scaling."""
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(20, 4))
    scales = np.array([1e-12, 1e-4, 1e4, 1e12])
    first = analyze_rank(matrix, rank_tol=1e-10)
    second = analyze_rank(matrix*scales[None,:], rank_tol=1e-10)
    assert first.rank == second.rank == 4


def test_rank_is_global_scale_invariant():
    """Preserve rank when the whole matrix changes units."""
    matrix = np.diag([1.0, 1e-8, 0.0])
    first = analyze_rank(matrix, rank_tol=1e-7)
    second = analyze_rank(1e20*matrix, rank_tol=1e-7)
    assert first.rank == second.rank


def test_rank_limit_is_applied_after_detection():
    """Record when a user basis-rank limit truncates a full basis."""
    analysis = analyze_rank(
        np.eye(4), rank_tol=1e-12, rank_limit=2
    )
    assert analysis.rank == 2
    assert analysis.rank_limited
    assert analysis.sample_margin == 0


def test_zero_matrix_rank():
    """Return rank zero safely for an all-zero matrix."""
    analysis = analyze_rank(np.zeros((4, 3)), rank_tol=1e-10)
    assert analysis.rank == 0
    assert np.isinf(analysis.condition)


################################################################################
# Scaling and row spans
################################################################################


def test_stable_euclidean_norm_handles_extreme_finite_scales():
    """Avoid premature square overflow and underflow for finite vectors."""
    large = stable_euclidean_norm(np.array([1e154, 1e154]))
    small = stable_euclidean_norm(np.array([1e-200, 1e-200]))

    assert np.isfinite(large)
    assert small > 0.0
    np.testing.assert_allclose(
        large, np.sqrt(2.0)*1e154, rtol=2e-15, atol=0.0
    )
    np.testing.assert_allclose(
        small, np.sqrt(2.0)*1e-200, rtol=2e-15, atol=0.0
    )


def test_column_equilibration_handles_zero_columns():
    """Avoid division by zero while normalizing nonzero columns."""
    matrix = np.array([[3.0, 0.0], [4.0, 0.0]])
    scaled, scales = equilibrate_columns(matrix)
    np.testing.assert_allclose(np.linalg.norm(scaled[:,0]), 1.0)
    np.testing.assert_allclose(scaled[:,1], 0.0)
    np.testing.assert_allclose(scales, [5.0, 1.0])


def test_row_normalization_handles_zero_rows():
    """Leave zero rows unchanged and normalize nonzero rows."""
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]])
    scaled, scales = normalize_rows(matrix)
    np.testing.assert_allclose(scaled[0], [0.6, 0.8])
    np.testing.assert_allclose(scaled[1], [0.0, 0.0])
    np.testing.assert_allclose(scales, [0.2, 1.0])


def test_extreme_finite_units_preserve_rank_and_row_scaling():
    """Preserve independent directions across representable float64 scales."""
    matrix = np.diag([1e154, 1e-200])
    scaled, scales = equilibrate_columns(matrix)

    np.testing.assert_allclose(scaled, np.eye(2), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        scales, [1e154, 1e-200], rtol=0.0, atol=0.0
    )
    assert analyze_rank(matrix, rank_tol=1e-12).rank == 2

    for magnitude in (1e-200, 1e154):
        normalized, _ = normalize_rows(magnitude*np.eye(2))
        np.testing.assert_allclose(normalized, np.eye(2), atol=0.0)


def test_balanced_rank_computes_only_singular_values(monkeypatch):
    """Avoid unused singular-vector allocations during PQR analysis."""
    original = linalg.svd
    calls = []

    def recording_svd(*args, **kwargs):
        """Record whether SciPy was asked to construct singular vectors."""
        calls.append(kwargs.get('compute_uv', True))
        return original(*args, **kwargs)

    monkeypatch.setattr(linalg, 'svd', recording_svd)
    analyze_balanced_system_rank(np.eye(3), rank_tol=1e-12)
    assert calls == [False]


def test_balanced_system_rank():
    """Analyze a system after both column and row equilibration."""
    matrix = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1e8, 1.0],
        [1.0, 1e8, 2.0],
    ])
    analysis, balanced, row_scales = analyze_balanced_system_rank(
        matrix, rank_tol=1e-10
    )
    assert analysis.rank == 2
    assert balanced.shape == matrix.shape
    assert row_scales.shape == (3,)


def test_row_span_residual():
    """Measure zero residual for a spanning row subset."""
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    residual = compute_row_span_residual(matrix, [0, 1])
    assert residual < 1e-12
