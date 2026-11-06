"""Tests for explicit rounding and manifold projection helpers."""

import numpy as np
import pytest

from rhiza import rounding


################################################################################
# Matrix projections
################################################################################


def test_project_so3():
    """Return an orthogonal proper rotation matrix."""
    matrix = np.array([
        [0.0, -1.2, 0.1],
        [0.9, 0.0, 0.2],
        [0.0, 0.1, 1.1],
    ])
    rotation = rounding.project_so3(matrix)
    np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_project_se3():
    """Preserve translation while projecting the rotation block."""
    matrix = np.column_stack((1.1*np.eye(3), [1.0, 2.0, 3.0]))
    transform = rounding.project_se3(matrix)
    np.testing.assert_allclose(transform[:3,3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])


################################################################################
# Discrete rounding
################################################################################


def test_binary_and_sign_rounding():
    """Round scalar arrays to standard binary and sign sets."""
    np.testing.assert_array_equal(
        rounding.round_binary([-1.0, 0.5, 2.0]),
        [0, 1, 1],
    )
    np.testing.assert_array_equal(
        rounding.round_sign([-2.0, 0.0, 3.0], zero_value=-1),
        [-1, -1, 1],
    )


def test_invalid_sign_zero_value():
    """Require a valid sign for exact zeros."""
    with pytest.raises(ValueError):
        rounding.round_sign([0.0], zero_value=0)


def test_one_hot_rounding():
    """Place one at each axis-wise argmax."""
    values = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 1.0]])
    actual = rounding.round_one_hot(values, axis=1)
    np.testing.assert_array_equal(
        actual,
        [[0, 1, 0], [1, 0, 0]],
    )


def test_so3_projection_corrects_reflections():
    """Flip the final singular direction for an improper orthogonal input."""
    reflection = np.diag([1.0, 1.0, -1.0])
    rotation = rounding.project_so3(reflection)
    np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    'function, matrix',
    (
        (rounding.project_so3, np.eye(2)),
        (rounding.project_se3, np.eye(3)),
    ),
)
def test_matrix_projection_rejects_invalid_shapes(function, matrix):
    """Reject matrices outside the documented manifold shapes."""
    with pytest.raises(ValueError):
        function(matrix)


def test_rounding_options_and_invalid_one_hot_axes():
    """Exercise custom thresholds, negative axes, and axis validation."""
    np.testing.assert_array_equal(
        rounding.round_binary([0.2, 0.3], threshold=0.25),
        [0, 1],
    )
    values = np.array([[1.0, 4.0], [3.0, 2.0]])
    np.testing.assert_array_equal(
        rounding.round_one_hot(values, axis=-1),
        [[0, 1], [1, 0]],
    )
    with pytest.raises(ValueError):
        rounding.round_one_hot(1.0)
    with pytest.raises(ValueError):
        rounding.round_one_hot(values, axis=3)
    with pytest.raises(ValueError, match='integer'):
        rounding.round_one_hot(values, axis=0.5)
    np.testing.assert_array_equal(
        rounding.round_one_hot(values, axis=0.0),
        [[0, 1], [1, 0]],
    )


def test_scalar_sign_rounding():
    """Round positive, negative, and zero scalar inputs safely."""
    assert rounding.round_sign(2.0) == 1
    assert rounding.round_sign(-2.0) == -1
    assert rounding.round_sign(0.0) == 1
    assert rounding.round_sign(0.0, zero_value=-1) == -1


@pytest.mark.parametrize(
    'function, values',
    (
        (rounding.round_binary, [0.0, np.nan]),
        (rounding.round_sign, [0.0, np.inf]),
        (rounding.round_one_hot, [1.0, np.nan]),
    ),
)
def test_discrete_rounding_rejects_nonfinite_values(function, values):
    """Reject nonfinite scores before they become plausible decisions."""
    with pytest.raises(ValueError, match='finite'):
        function(values)


@pytest.mark.parametrize(
    'function, matrix',
    (
        (rounding.project_so3, np.full((3, 3), np.nan)),
        (rounding.project_se3, np.column_stack((np.eye(3), [0.0, np.inf, 0.0]))),
    ),
)
def test_matrix_projection_rejects_nonfinite_values(function, matrix):
    """Reject nonfinite manifold estimates with a stable public error."""
    with pytest.raises(ValueError, match='finite'):
        function(matrix)


def test_one_hot_rejects_empty_category_axis():
    """Require at least one category in every one-hot score vector."""
    with pytest.raises(ValueError, match='nonempty'):
        rounding.round_one_hot(np.empty((2, 0)), axis=1)


def test_binary_rounding_rejects_nonfinite_threshold():
    """Reject a threshold that cannot define a finite decision boundary."""
    with pytest.raises(ValueError, match='finite'):
        rounding.round_binary([0.0, 1.0], threshold=np.nan)
