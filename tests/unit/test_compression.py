"""Tests for pivoted-QR clique-system compression."""

import numpy as np
import pytest

import rhiza.tools.compression as compression_module
from rhiza.diagnostics import CompressionError
from rhiza.tools.compression import compress_clique_system, select_rows_pqr


################################################################################
# PQR row selection
################################################################################


def test_duplicate_rows_compress():
    """Remove repeated sampled equations while retaining original rows."""
    system = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 3.0],
        [1.0, 0.0, 2.0],
        [1.0, 1.0, 5.0],
    ])
    compression = compress_clique_system(
        system, method='PQR', rank_tol=1e-12
    )
    assert compression.changed
    assert compression.num_rows_retained == 2
    assert len(set(compression.row_indices.tolist())) == 2
    assert compression.row_span_residual < 1e-10


def test_full_rank_square_system_retains_all_rows():
    """Retain every equation when each sample row is independent."""
    compression = compress_clique_system(
        np.eye(5), method='PQR', rank_tol=1e-12
    )
    assert not compression.changed
    assert compression.num_rows_retained == 5
    assert compression.rank_analysis.rank == 5


def test_pqr_requests_only_the_r_factor(monkeypatch):
    """Avoid constructing an orthogonal factor unused by row selection."""
    calls = []

    def recording_qr(matrix, **kwargs):
        """Record the QR mode and return a deterministic pivot order."""
        calls.append(kwargs)
        return np.eye(matrix.shape[0]), np.array([2, 0, 1])

    monkeypatch.setattr(compression_module, 'qr', recording_qr)
    pivots = select_rows_pqr(np.eye(3))
    np.testing.assert_array_equal(pivots, [2, 0, 1])
    assert calls == [{
        'mode': 'r',
        'pivoting': True,
        'check_finite': False,
    }]


def test_bad_row_scaling_does_not_change_rank():
    """Balance extreme row magnitudes before pivot selection."""
    base = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    scaled = np.array([1e-12, 1e12, 1.0])[:,None]*base
    first = compress_clique_system(base, rank_tol=1e-10)
    second = compress_clique_system(scaled, rank_tol=1e-10)
    assert first.rank_analysis.rank == second.rank_analysis.rank == 2
    assert first.num_rows_retained == second.num_rows_retained


def test_selected_rows_preserve_affine_equations():
    """Preserve the least-squares residual of a consistent affine system."""
    rng = np.random.default_rng(8)
    independent = rng.normal(size=(4, 7))
    combinations = rng.normal(size=(20, 4))@independent
    system = np.vstack((independent, combinations))
    compression = compress_clique_system(system, rank_tol=1e-11)
    selected = system[compression.row_indices]
    coefficients = rng.normal(size=7)
    right = system@coefficients
    recovered = np.linalg.lstsq(
        selected,
        right[compression.row_indices],
        rcond=None,
    )[0]
    np.testing.assert_allclose(system@recovered, right, atol=1e-8)


def test_zero_system_fails_cleanly():
    """Reject an augmented clique system with numerical rank zero."""
    with pytest.raises(CompressionError):
        compress_clique_system(np.zeros((4, 3)), rank_tol=1e-10)


def test_unsupported_method():
    """Reject compression methods outside PQR."""
    with pytest.raises(CompressionError):
        compress_clique_system(
            np.eye(2), method='SVD', rank_tol=1e-10
        )


@pytest.mark.parametrize('seed', range(8))
def test_random_low_rank_systems_preserve_row_spaces(seed):
    """Preserve random affine row spaces under severe column scaling."""
    rng = np.random.default_rng(seed)
    num_rows = 36 + seed
    num_columns = 11
    rank = 3 + seed%5
    left = rng.normal(size=(num_rows, rank))
    right = rng.normal(size=(rank, num_columns - 2))
    system = left@right
    system = np.column_stack((
        system,
        np.zeros(num_rows),
        system[:,0] + 1e-12*system[:,1],
    ))
    column_scales = 10.0**rng.uniform(-10.0, 10.0, size=num_columns)
    system_scaled = system*column_scales[None,:]

    first = compress_clique_system(system, rank_tol=1e-10)
    second = compress_clique_system(system_scaled, rank_tol=1e-10)
    assert first.rank_analysis.rank == second.rank_analysis.rank == rank
    assert first.num_rows_retained >= rank
    assert second.num_rows_retained >= rank
    assert first.row_span_residual <= 1e-8
    assert second.row_span_residual <= 1e-8
    assert np.all(first.row_indices >= 0)
    assert np.all(first.row_indices < num_rows)
    assert len(np.unique(first.row_indices)) == first.num_rows_retained

    coefficients = rng.normal(size=num_columns)
    target_evals = system@coefficients
    selected = system[first.row_indices,:]
    recovered = np.linalg.lstsq(
        selected,
        target_evals[first.row_indices],
        rcond=None,
    )[0]
    np.testing.assert_allclose(
        system@recovered,
        target_evals,
        atol=2e-8,
        rtol=2e-8,
    )
