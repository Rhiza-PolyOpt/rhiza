"""Sample-row compression for clique systems.

Pivoted QR removes redundant equations while retaining actual sampled rows,
which keeps compressed systems interpretable during extraction.
"""

import numpy as np
from scipy.linalg import qr

from rhiza.diagnostics import CompressionError, CompressionMethod
from rhiza.tools.linalg import (
    analyze_balanced_system_rank,
    compute_effective_rank_tol,
    compute_row_span_residual,
    equilibrate_columns,
    normalize_rows,
)


################################################################################
# Clique-system compression
################################################################################


def compress_clique_system(
    system_evals,
    *,
    method='PQR',
    rank_tol,
    rank_analysis=None,
):
    """Compress one augmented clique system by selecting sampled equations.

    PQR keeps actual sample rows rather than arbitrary linear combinations. A
    separate projection audit then verifies that the retained rows reproduce the
    balanced row span of the complete clique system.

    Inputs:
        system_evals:
            ndarray. Complete augmented system evaluations.
        method:
            String or CompressionMethod. Compression method.
        rank_tol:
            Float. Relative numerical-rank tolerance.
        rank_analysis:
            RankAnalysis or None. Cached rank record for this system.

    Outputs:
        compression:
            CliqueCompression. Selected rows and numerical diagnostics.
    """
    system_evals = np.asarray(system_evals, dtype=float)
    method = CompressionMethod.from_value(method)
    rank_tol = float(rank_tol)

    if system_evals.ndim != 2:
        raise CompressionError("Clique-system evals must be a matrix.")
    if system_evals.shape[0] == 0:
        raise CompressionError("A clique system requires sample rows.")
    if not np.all(np.isfinite(system_evals)):
        raise CompressionError("Clique-system evals must be finite.")

    # Analysis and pivoting use the same balanced equation matrix. A cached
    # rank record avoids another SVD, while the balanced matrix itself is cheap
    # enough to reconstruct and too large to retain between public operations.
    if rank_analysis is None:
        rank_analysis, system_balanced, _ = analyze_balanced_system_rank(
            system_evals,
            rank_tol=rank_tol,
        )
    else:
        system_balanced = equilibrate_columns(system_evals)[0]
        system_balanced, _ = normalize_rows(system_balanced)
    if rank_analysis.rank <= 0:
        raise CompressionError("The augmented clique system has rank zero.")

    # Begin with as many pivot rows as the estimated rank. The independent span
    # audit may retain a few additional guard rows when the numerical cutoff is
    # marginal, and retaining all rows remains the conservative fallback.
    pivot_order = select_rows_pqr(system_balanced)
    row_indices, residual = choose_rows_from_pivots(
        system_balanced,
        pivot_order,
        rank=rank_analysis.rank,
        rank_tol=rank_tol,
    )

    # Backends submit the selected equations in their original coefficient
    # units, scaled only by the ordinary row normalization used throughout SDPR.
    _, row_scales_all = normalize_rows(system_evals)
    row_scales = row_scales_all[row_indices]
    num_rows_original = system_evals.shape[0]
    num_rows_retained = row_indices.size
    return CliqueCompression(
        method,
        bool(num_rows_retained < num_rows_original),
        row_indices,
        row_scales,
        rank_analysis,
        num_rows_original,
        num_rows_retained,
        num_rows_retained - rank_analysis.rank,
        residual,
    )


def select_rows_pqr(system_balanced):
    """Order original sampled equations by pivoted QR.

    Applying column-pivoted QR to the transpose makes its pivot columns exactly
    the rows of the original system. The returned ordering therefore identifies
    actual sampled equations rather than transformed combinations of them.

    Inputs:
        system_balanced:
            ndarray. Balanced augmented clique system.

    Outputs:
        pivot_order:
            ndarray. Sample row indexes in pivot order.
    """
    system_balanced = np.asarray(system_balanced, dtype=float)

    # Row selection needs only the pivot order. ``mode='r'`` avoids constructing
    # the orthogonal factor that an economy QR would compute and then discard.
    _, pivots = qr(
        system_balanced.T,
        mode='r',
        pivoting=True,
        check_finite=False,
    )
    return np.asarray(pivots, dtype=int)


def choose_rows_from_pivots(
    system_balanced,
    pivot_order,
    *,
    rank,
    rank_tol,
):
    """Choose enough PQR pivots to reproduce the balanced row span.

    The rank estimate supplies the initial prefix. An independent projection
    residual guards against a marginal cutoff by adding bounded batches of
    further pivot rows until the complete balanced span is recovered.

    Inputs:
        system_balanced:
            ndarray. Balanced augmented clique system.
        pivot_order:
            ndarray. Sample row indexes in pivot order.
        rank:
            Integer. Estimated system rank.
        rank_tol:
            Float. Relative numerical-rank tolerance.

    Outputs:
        row_indices:
            ndarray. Sorted retained sample indexes.
        row_span_residual:
            Float. Final balanced row-span residual.
    """
    system_balanced = np.asarray(system_balanced, dtype=float)
    pivot_order = np.asarray(pivot_order, dtype=int)
    rank = int(rank)
    rank_tol = float(rank_tol)

    residual_tol = compression_residual_tol(
        system_balanced.shape, rank_tol
    )
    num_rows = system_balanced.shape[0]
    num_retained = min(max(1, rank), num_rows)
    guard_step = max(1, min(8, rank//20 + 1))
    while num_retained <= num_rows:
        row_indices = np.sort(pivot_order[:num_retained])
        residual = compute_row_span_residual(
            system_balanced, row_indices
        )
        if residual <= residual_tol or num_retained == num_rows:
            return row_indices, residual
        num_retained = min(num_rows, num_retained + guard_step)
    raise CompressionError("PQR row selection did not terminate.")


def compression_residual_tol(matrix_shape, rank_tol):
    """Derive the tolerance for the independent row-span audit.

    The audit follows the same relative rank policy as basis construction but
    allows one additional decade for projection and factorization roundoff.

    Inputs:
        matrix_shape:
            tuple. Augmented clique-system shape.
        rank_tol:
            Float. Relative numerical-rank tolerance.

    Outputs:
        residual_tol:
            Float. Relative row-span residual tolerance.
    """
    matrix_shape = tuple(int(value) for value in matrix_shape)
    rank_tol = float(rank_tol)

    return 10.0*compute_effective_rank_tol(matrix_shape, rank_tol)


################################################################################
# Compression record
################################################################################


class CliqueCompression:
    """Store one audited clique-system row selection.

    The rank record can be reused by later analysis, while the retained row map
    and span residual explain the actual compression decision.
    """

    def __init__(
        self,
        method,
        changed,
        row_indices,
        row_scales,
        rank_analysis,
        num_rows_original,
        num_rows_retained,
        num_guard_rows,
        row_span_residual,
    ):
        """Store selected rows and their numerical audit.

        Inputs:
            method:
                CompressionMethod. Applied method.
            changed:
                Boolean. Whether any rows were removed.
            row_indices:
                ndarray. Retained sample indexes.
            row_scales:
                ndarray. Equation scales for retained rows.
            rank_analysis:
                RankAnalysis. Balanced-system rank diagnostics.
            num_rows_original:
                Integer. Original row count.
            num_rows_retained:
                Integer. Retained row count.
            num_guard_rows:
                Integer. Rows retained beyond the estimated rank.
            row_span_residual:
                Float. Balanced row-span residual.
        """
        self.method = method
        self.changed = changed
        self.row_indices = row_indices
        self.row_scales = row_scales
        self.rank_analysis = rank_analysis
        self.num_rows_original = num_rows_original
        self.num_rows_retained = num_rows_retained
        self.num_guard_rows = num_guard_rows
        self.row_span_residual = row_span_residual

    def as_report(self):
        """Return the compression result as a compact public report.

        Outputs:
            report:
                Report. Row reduction, rank, and span-audit fields.
        """
        return {
            'changed': self.changed,
            'num_rows_original': self.num_rows_original,
            'num_rows_retained': self.num_rows_retained,
            'num_guard_rows': self.num_guard_rows,
            'rank': self.rank_analysis.rank,
            'sample_margin': self.rank_analysis.sample_margin,
            'condition': self.rank_analysis.condition,
            'singular_gap': self.rank_analysis.singular_gap,
            'row_span_residual': self.row_span_residual,
        }
