"""Explicit postprocessing for common polynomial decision spaces.

These helpers turn extracted numerical estimates into application-specific
objects; they do not alter or certify the relaxation itself.
"""

import numpy as np

from rhiza.tools.parsing import (
    cast_exact_int,
    cast_real_array,
    cast_real_scalar,
)


################################################################################
# Matrix manifolds
################################################################################


def project_so3(matrix):
    """Project a finite 3-by-3 matrix onto SO(3).

    The orthogonal Procrustes projection is useful for interpreting an extracted
    rotation estimate, but it remains separate from the SDP certificate because
    the appropriate rounding operation is application-specific.

    Inputs:
        matrix:
            ArrayLike. Matrix to project.

    Outputs:
        rotation:
            ndarray. Nearest proper rotation in Frobenius norm.
    """
    matrix = cast_real_array(matrix, "SO(3) projection matrix", ValueError)

    if matrix.shape != (3, 3):
        raise ValueError("SO(3) projection requires a 3-by-3 matrix.")
    _require_finite(matrix, "SO(3) projection matrix")

    # The orthogonal Procrustes factor U@V.T is the nearest orthogonal matrix.
    # If it is a reflection, flip the least-constrained singular direction to
    # reach the proper-rotation component with the smallest additional change.
    left, _, right_transpose = np.linalg.svd(matrix)
    rotation = left@right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:,-1] *= -1.0
        rotation = left@right_transpose
    return rotation


def project_se3(matrix):
    """Project a finite 3-by-4 or 4-by-4 matrix onto SE(3).

    Only the rotational block requires projection; the translation is already
    an unconstrained coordinate of SE(3) and therefore passes through unchanged.

    Inputs:
        matrix:
            ArrayLike. Transform to project.

    Outputs:
        transform:
            ndarray. Homogeneous rigid transform.
    """
    matrix = cast_real_array(matrix, "SE(3) projection matrix", ValueError)

    if matrix.shape not in ((3, 4), (4, 4)):
        raise ValueError(
            "SE(3) projection requires a 3-by-4 or 4-by-4 matrix."
        )
    _require_finite(matrix, "SE(3) projection matrix")

    transform = np.eye(4)
    transform[:3,:3] = project_so3(matrix[:3,:3])
    transform[:3,3] = matrix[:3,3]
    return transform


################################################################################
# Discrete spaces
################################################################################


def round_binary(values, *, threshold=0.5):
    """Round finite values to zero or one.

    Inputs:
        values:
            ArrayLike. Values to round.
        threshold:
            Float. Values at least this large become one.

    Outputs:
        rounded:
            ndarray. Binary integer values.
    """
    values = cast_real_array(values, "Binary rounding values", ValueError)
    threshold = cast_real_scalar(threshold, "threshold", ValueError)

    _require_finite(values, "Binary rounding values")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    return (values >= threshold).astype(int)


def round_sign(values, *, zero_value=1.0):
    """Round finite values to minus one or plus one.

    Inputs:
        values:
            ArrayLike. Values to round.
        zero_value:
            Float. Sign assigned to exact zeros.

    Outputs:
        rounded:
            ndarray. Signed integer values.
    """
    values = cast_real_array(values, "Sign rounding values", ValueError)
    zero_value = cast_real_scalar(zero_value, "zero_value", ValueError)

    _require_finite(values, "Sign rounding values")
    if zero_value not in (-1.0, 1.0):
        raise ValueError("zero_value must be -1 or +1.")

    # np.where works for both ordinary arrays and zero-dimensional scalar arrays;
    # indexed assignment would fail for the scalar value returned by np.sign.
    return np.where(values == 0.0, zero_value, np.sign(values)).astype(int)


def round_one_hot(values, *, axis=-1):
    """Round each finite score vector along one axis to a one-hot vector.

    Inputs:
        values:
            ArrayLike. Scores to round.
        axis:
            Integer. Axis containing each categorical vector.

    Outputs:
        rounded:
            ndarray. One-hot integer values.
    """
    values = cast_real_array(values, "One-hot rounding values", ValueError)
    axis = cast_exact_int(axis, "axis", ValueError)

    if values.ndim == 0:
        raise ValueError("One-hot rounding requires an array.")
    _require_finite(values, "One-hot rounding values")
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise ValueError("axis is out of range.")
    if values.shape[axis] == 0:
        raise ValueError("The one-hot category axis must be nonempty.")

    # Collapse the category axis to winner indexes, then insert those indexes
    # back into a full-dimensional coordinate tuple for one vectorized write.
    winners = np.argmax(values, axis=axis)
    rounded = np.zeros(values.shape, dtype=int)
    index_grid = np.indices(winners.shape)
    indexes = list(index_grid)
    indexes.insert(axis, winners)
    rounded[tuple(indexes)] = 1
    return rounded


################################################################################
# Validation
################################################################################


def _require_finite(values, label):
    """Require every numerical entry to be finite before postprocessing.

    Explicit validation prevents NaNs from becoming plausible-looking discrete
    decisions and gives manifold projections a consistent public error.

    Inputs:
        values:
            ndarray. Numerical values to inspect.
        label:
            String. Object name used in the error message.
    """
    if not np.all(np.isfinite(values)):
        raise ValueError(label + " must be finite.")
