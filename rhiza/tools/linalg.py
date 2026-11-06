"""Scale-aware numerical linear algebra for sampled polynomial spaces.

Quotient bases, separator spaces, and row compression share the same
rank threshold and equilibration rules so their numerical decisions agree.
"""

import numpy as np
from scipy.linalg import svd

from rhiza.diagnostics import NumericalRankError


################################################################################
# Rank analysis
################################################################################


def analyze_rank(
    matrix,
    *,
    rank_tol,
    rank_limit=None,
    equilibrate=True,
    return_left_vectors=False,
):
    """Analyze matrix rank after optional column equilibration.

    Column equilibration removes units before singular values are compared. By
    default only singular values are computed; basis construction may also ask
    for the left singular vectors spanning the sampled function space.

    Inputs:
        matrix:
            ndarray. Two-dimensional numerical matrix.
        rank_tol:
            Float. Relative singular-value tolerance.
        rank_limit:
            Integer or None. Maximum retained rank.
        equilibrate:
            Boolean. Whether to normalize nonzero columns.
        return_left_vectors:
            Boolean. Whether to return the economy left singular vectors.

    Outputs:
        analysis:
            RankAnalysis. Rank and conditioning diagnostics.
        left_vectors:
            ndarray. Returned only when return_left_vectors is true.
    """
    matrix = np.asarray(matrix, dtype=float)
    rank_tol = float(rank_tol)
    rank_limit = None if rank_limit is None else int(rank_limit)
    equilibrate = bool(equilibrate)
    return_left_vectors = bool(return_left_vectors)

    if matrix.ndim != 2:
        raise NumericalRankError("Rank analysis requires a matrix.")
    if not np.all(np.isfinite(matrix)):
        raise NumericalRankError("Rank-analysis matrices must be finite.")
    if not np.isfinite(rank_tol) or rank_tol < 0.0:
        raise NumericalRankError("rank_tol must be finite and nonnegative.")
    if rank_limit is not None and rank_limit <= 0:
        raise NumericalRankError("rank_limit must be positive.")

    # Remove column units before comparing singular directions. The scaled
    # matrix is disposable; no downstream calculation needs the column norms.
    matrix_scaled = (
        equilibrate_columns(matrix)[0] if equilibrate else matrix.copy()
    )
    if return_left_vectors:
        left_vectors, singular_values, _ = economy_svd(matrix_scaled)
    else:
        singular_values = economy_svd(matrix_scaled, compute_uv=False)
        left_vectors = None

    analysis = _analyze_singular_values(
        singular_values,
        matrix.shape,
        rank_tol=rank_tol,
        rank_limit=rank_limit,
    )
    if return_left_vectors:
        return analysis, left_vectors
    return analysis


def analyze_balanced_system_rank(system_evals, *, rank_tol):
    """Balance a clique system and analyze its numerical rank.

    PQR needs the balanced equation matrix and rank diagnostics, but no singular
    vectors. Computing singular values only avoids allocating factors that row
    selection never consumes.

    Inputs:
        system_evals:
            ndarray. Complete augmented clique-system evaluations.
        rank_tol:
            Float. Relative singular-value tolerance.

    Outputs:
        analysis:
            RankAnalysis. Balanced-system rank analysis.
        system_balanced:
            ndarray. Column-equilibrated and row-normalized system.
        row_scales:
            ndarray. Row scales applied after column equilibration.
    """
    system_evals = np.asarray(system_evals, dtype=float)
    rank_tol = float(rank_tol)

    if system_evals.ndim != 2:
        raise NumericalRankError("System-rank analysis requires a matrix.")
    if not np.all(np.isfinite(system_evals)):
        raise NumericalRankError("Clique-system evals must be finite.")

    # PQR compares equation directions only after both coefficient units and row
    # magnitudes have been removed. Keep this one balanced copy for pivoting and
    # the independent row-span audit.
    system_balanced = equilibrate_columns(system_evals)[0]
    system_balanced, row_scales = normalize_rows(system_balanced)
    singular_values = economy_svd(system_balanced, compute_uv=False)
    analysis = _analyze_singular_values(
        singular_values,
        system_balanced.shape,
        rank_tol=rank_tol,
        rank_limit=None,
    )
    return analysis, system_balanced, row_scales


def _analyze_singular_values(
    singular_values,
    matrix_shape,
    *,
    rank_tol,
    rank_limit,
):
    """Apply Rhiza's common cutoff to descending singular values.

    Keeping the rank decision separate from factorization lets quotient bases
    request range vectors while compression computes only the spectrum.

    Inputs:
        singular_values:
            ndarray. Descending singular values.
        matrix_shape:
            tuple. Shape of the analyzed matrix.
        rank_tol:
            Float. User relative tolerance.
        rank_limit:
            Integer or None. Maximum retained rank.

    Outputs:
        analysis:
            RankAnalysis. Rank and conditioning diagnostics.
    """
    singular_values = np.asarray(singular_values, dtype=float)
    matrix_shape = tuple(int(value) for value in matrix_shape)
    # Combine the requested relative cutoff with the dimension-dependent
    # machine floor used throughout Rhiza's sampled rank decisions.
    effective_rank_tol = compute_effective_rank_tol(
        matrix_shape, rank_tol
    )

    # Compare normalized singular values so multiplying the entire matrix by a
    # scalar does not change its apparent sampled-function dimension.
    if singular_values.size == 0 or singular_values[0] == 0.0:
        apparent_rank = 0
    else:
        apparent_rank = int(np.count_nonzero(
            singular_values/singular_values[0] > effective_rank_tol
        ))

    # Apply the experimental retained-rank cap only after measuring empirical
    # rank. Sample margin is based on the untruncated rank, so an intentionally
    # reduced basis cannot make the sample set appear better qualified.
    rank = apparent_rank
    rank_limited = False
    if rank_limit is not None and rank > rank_limit:
        rank = rank_limit
        rank_limited = True

    return RankAnalysis(
        rank,
        compute_condition(singular_values, rank),
        compute_singular_gap(singular_values, rank),
        int(matrix_shape[0] - apparent_rank),
        rank_limited,
    )


################################################################################
# Scaling and residuals
################################################################################


def stable_euclidean_norm(values, *, axis=None):
    """Compute Euclidean norms without overflowing intermediate squares.

    Ordinary ``sqrt(sum(x*x))`` can overflow or underflow even when the final
    norm is representable. Dividing by the largest magnitude first keeps every
    squared entry in ``[0, 1]``; multiplying the resulting norm by that magnitude
    then restores the original units. This small helper protects the rank and
    row-scaling paths without adding a separate numerical-scaling framework.

    Inputs:
        values:
            ndarray. Finite numerical values.
        axis:
            Integer or None. Axis reduced by the norm.

    Outputs:
        norms:
            ndarray or float. Euclidean norms along ``axis``.
    """
    values = np.asarray(values, dtype=float)
    axis = None if axis is None else int(axis)

    if not np.all(np.isfinite(values)):
        raise NumericalRankError("Euclidean-norm inputs must be finite.")

    magnitudes = np.abs(values)
    if axis is None:
        scale = float(np.max(magnitudes, initial=0.0))
        if scale == 0.0:
            return 0.0
        magnitudes /= scale
        norm = scale*np.linalg.norm(magnitudes)
    else:
        scale = np.max(
            magnitudes,
            axis=axis,
            keepdims=True,
            initial=0.0,
        )
        np.divide(
            magnitudes,
            scale,
            out=magnitudes,
            where=scale != 0.0,
        )
        norm = np.squeeze(scale, axis=axis)*np.linalg.norm(
            magnitudes, axis=axis
        )

    if not np.all(np.isfinite(norm)):
        raise NumericalRankError(
            "A Euclidean norm is too large to represent in floating point."
        )
    return norm


def equilibrate_columns(matrix):
    """Scale each nonzero matrix column to unit Euclidean norm.

    This removes arbitrary coefficient units before numerical rank is assessed;
    zero columns remain exactly zero and receive a harmless scale of one.

    Inputs:
        matrix:
            ndarray. Two-dimensional numerical matrix.

    Outputs:
        matrix_scaled:
            ndarray. Column-equilibrated matrix.
        column_scales:
            ndarray. Removed column norms.
    """
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise NumericalRankError("Column equilibration requires a matrix.")
    column_scales = stable_euclidean_norm(matrix, axis=0)
    safe_scales = np.where(column_scales > 0.0, column_scales, 1.0)
    return matrix/safe_scales[None,:], safe_scales


def normalize_rows(matrix):
    """Scale each nonzero matrix row to unit Euclidean norm.

    Row normalization prevents large sampled equations from dominating PQR
    pivot selection solely because of magnitude. Zero rows remain unchanged.

    Inputs:
        matrix:
            ndarray. Two-dimensional numerical matrix.

    Outputs:
        matrix_scaled:
            ndarray. Row-normalized matrix.
        row_scales:
            ndarray. Multipliers applied to original rows.
    """
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise NumericalRankError("Row normalization requires a matrix.")
    row_norms = stable_euclidean_norm(matrix, axis=1)
    row_scales = np.ones(row_norms.shape)
    nonzero = row_norms > 0.0
    row_scales[nonzero] = 1.0/row_norms[nonzero]
    if not np.all(np.isfinite(row_scales)):
        raise NumericalRankError(
            "A row is too small to normalize in float64 arithmetic."
        )
    return row_scales[:,None]*matrix, row_scales


def compute_row_span_residual(system_evals, row_indices):
    """Measure the relative residual of a selected row span.

    The selected rows define an orthogonal projector after transposition. The
    residual independently audits that PQR retained every equation direction.

    Inputs:
        system_evals:
            ndarray. Balanced clique-system matrix.
        row_indices:
            Sequence. Selected row indexes.

    Outputs:
        residual:
            Float. Relative Frobenius row-span residual.
    """
    system_evals = np.asarray(system_evals, dtype=float)
    row_indices = np.asarray(row_indices, dtype=int)

    if system_evals.size == 0:
        return 0.0
    if row_indices.size == 0:
        return 1.0

    # Row-span membership in the original system becomes column-space
    # membership after transposition. Build an orthonormal basis for the
    # selected rows without reusing the QR factors that chose them.
    selected_transpose = system_evals[row_indices,:].T
    left_vectors, singular_values, _ = economy_svd(selected_transpose)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 1.0
    # Retain every selected-space direction distinguishable at machine
    # precision, then project all original equations onto that independent
    # audit basis. The normalized residual measures any discarded row space.
    dimension = max(selected_transpose.shape)
    threshold = np.finfo(float).eps*dimension*singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))
    basis = left_vectors[:,:rank]
    projected = basis@(basis.T@system_evals.T)
    denominator = stable_euclidean_norm(system_evals)
    if denominator == 0.0:
        return 0.0
    numerator = stable_euclidean_norm(system_evals.T - projected)
    return float(numerator/denominator)


def compute_condition(singular_values, rank):
    """Compute the largest-to-smallest retained singular-value ratio.

    Inputs:
        singular_values:
            ndarray. Descending singular values.
        rank:
            Integer. Retained rank.

    Outputs:
        condition:
            Float. Retained singular-value condition.
    """
    singular_values = np.asarray(singular_values, dtype=float)
    rank = int(rank)

    if rank <= 0:
        return float('inf')
    smallest = singular_values[rank - 1]
    if smallest <= 0.0:
        return float('inf')
    return float(singular_values[0]/smallest)


def compute_singular_gap(singular_values, rank):
    """Compute the retained-to-discarded singular-value ratio.

    Inputs:
        singular_values:
            ndarray. Descending singular values.
        rank:
            Integer. Retained rank.

    Outputs:
        singular_gap:
            Float or None. Cutoff ratio when a discarded value exists.
    """
    singular_values = np.asarray(singular_values, dtype=float)
    rank = int(rank)

    if rank <= 0 or rank >= singular_values.size:
        return None
    discarded = singular_values[rank]
    if discarded == 0.0:
        return float('inf')
    return float(singular_values[rank - 1]/discarded)


def compute_effective_rank_tol(
    shape,
    rank_tol,
    *,
    machine_factor=32.0,
):
    """Combine a user rank tolerance with a dimensional machine floor.

    Inputs:
        shape:
            tuple. Matrix shape.
        rank_tol:
            Float. User relative tolerance.
        machine_factor:
            Float. Machine-floor safety factor.

    Outputs:
        effective_rank_tol:
            Float. Applied relative tolerance.
    """
    shape = tuple(int(value) for value in shape)
    rank_tol = float(rank_tol)
    machine_factor = float(machine_factor)

    machine_tol = machine_factor*np.finfo(float).eps*max(shape, default=1)
    return max(rank_tol, machine_tol)


################################################################################
# Factorization
################################################################################


def economy_svd(matrix, *, overwrite_a=False, compute_uv=True):
    """Compute an economy SVD with LAPACK GESVD.

    GESVD is used for robust rank decisions on ill-conditioned sampled
    polynomial matrices. Callers needing only rank diagnostics may suppress
    singular vectors to reduce memory and work.

    Inputs:
        matrix:
            ndarray. Two-dimensional numerical matrix.
        overwrite_a:
            Boolean. Whether LAPACK may overwrite the input.
        compute_uv:
            Boolean. Whether to compute singular vectors.

    Outputs:
        factors:
            tuple or ndarray. Economy factors, or singular values alone.
    """
    matrix = np.asarray(matrix, dtype=float)

    return svd(
        matrix,
        full_matrices=False,
        compute_uv=compute_uv,
        lapack_driver='gesvd',
        check_finite=False,
        overwrite_a=overwrite_a,
    )


################################################################################
# Numerical record
################################################################################


class RankAnalysis:
    """Store the rank decision and diagnostics consumed downstream.

    Factorization arrays stay local to the calculation that needs them. The
    record therefore remains small enough to reuse for quotient bases and PQR.
    """

    def __init__(
        self,
        rank,
        condition,
        singular_gap,
        sample_margin,
        rank_limited,
    ):
        """Store one common numerical-rank decision.

        Inputs:
            rank:
                Integer. Retained numerical rank.
            condition:
                Float. Retained singular-value condition.
            singular_gap:
                Float or None. Retained-to-discarded singular ratio.
            sample_margin:
                Integer. Sample rows beyond the measured empirical rank.
            rank_limited:
                Boolean. Whether a user limit reduced the rank.
        """
        self.rank = rank
        self.condition = condition
        self.singular_gap = singular_gap
        self.sample_margin = sample_margin
        self.rank_limited = rank_limited
