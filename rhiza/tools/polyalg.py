"""Sampled polynomial bases, lifts, and Gramian coefficient maps.

These constructions turn evaluations on the sample set into the finite-
dimensional function spaces and coefficient streams used by SDPR.
"""

from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np

from rhiza.diagnostics import NumericalRankError
from rhiza.tools.linalg import analyze_rank


################################################################################
# Configuration
################################################################################


_COMBINATORIAL_CACHE_SIZE = 128
_NUM_COEFFS_PER_CHUNK = 1_000_000


################################################################################
# Quotient bases
################################################################################


def build_quotient_basis(
    variable_evals,
    degree,
    *,
    rank_tol,
    basis_rank_limit=None,
):
    """Construct a quotient basis from degree-bounded sampled monomials.

    The monomial evaluations describe polynomial functions on one scope. Their
    sampled range becomes the compact basis reused by local Gramians and
    unrestricted multipliers.

    Inputs:
        variable_evals:
            ndarray. Sample-by-variable evaluation matrix.
        degree:
            Integer. Maximum monomial degree.
        rank_tol:
            Float. Relative numerical-rank tolerance.
        basis_rank_limit:
            Integer or None. Maximum retained basis rank.

    Outputs:
        basis:
            QuotientBasisData. Numerical quotient basis.
    """
    variable_evals = np.asarray(variable_evals, dtype=float)
    degree = int(degree)

    if variable_evals.ndim != 2:
        raise NumericalRankError("variable_evals must be a matrix.")
    if degree < 0:
        raise NumericalRankError("Basis degree must be nonnegative.")

    # Enumerate the ambient degree-bounded monomials on this variable scope.
    # Their sampled column space is the corresponding coordinate-ring space.
    exponents = enumerate_exponents(variable_evals.shape[1], degree)
    monomial_evals = evaluate_monomials(variable_evals, exponents)

    # Remove sampled algebraic dependencies and retain an orthonormal range
    # basis for every later Gramian or unrestricted multiplier on this scope.
    basis = build_sampled_basis(
        monomial_evals,
        rank_tol=rank_tol,
        basis_rank_limit=basis_rank_limit,
    )
    if basis.basis_rank <= 0:
        raise NumericalRankError("The quotient basis has numerical rank zero.")
    return basis


def build_separator_basis(
    variable_evals,
    degree,
    *,
    rank_tol,
    basis_rank_limit=None,
):
    """Construct a nonconstant separator basis from sampled monomials.

    Separator coefficients redistribute residuals between clique identities.
    Constants are omitted because local bound scalars already provide that
    communication direction.

    Inputs:
        variable_evals:
            ndarray. Sample-by-variable evaluation matrix.
        degree:
            Integer. Maximum separator degree.
        rank_tol:
            Float. Relative numerical-rank tolerance.
        basis_rank_limit:
            Integer or None. Maximum retained basis rank.

    Outputs:
        basis:
            QuotientBasisData. Nonconstant separator basis.
    """
    variable_evals = np.asarray(variable_evals, dtype=float)
    degree = int(degree)

    if variable_evals.ndim != 2:
        raise NumericalRankError("variable_evals must be a matrix.")
    if degree < 0:
        raise NumericalRankError("Separator degree must be nonnegative.")

    exponents = enumerate_exponents(variable_evals.shape[1], degree)
    monomial_evals = evaluate_monomials(variable_evals, exponents)
    # This array is disposable, so remove each column's constant component in
    # place before the common sampled-basis reduction.
    monomial_evals -= np.mean(monomial_evals, axis=0, keepdims=True)
    return build_sampled_basis(
        monomial_evals,
        rank_tol=rank_tol,
        basis_rank_limit=basis_rank_limit,
    )


def build_sampled_basis(
    sampled_evals,
    *,
    rank_tol,
    basis_rank_limit=None,
):
    """Reduce supplied sampled functions to an orthonormal range basis.

    Monomial, separator, and OVERLAP occurrence spaces all delegate to this one
    numerical operation, ensuring they share the same equilibration and rank
    convention.

    Inputs:
        sampled_evals:
            ndarray. Sample-by-function evaluation matrix.
        rank_tol:
            Float. Relative numerical-rank tolerance.
        basis_rank_limit:
            Integer or None. Maximum retained basis rank.

    Outputs:
        basis:
            QuotientBasisData. Numerical sampled-function basis.
    """
    sampled_evals = np.asarray(sampled_evals, dtype=float)
    rank_tol = float(rank_tol)
    basis_rank_limit = (
        None if basis_rank_limit is None else int(basis_rank_limit)
    )

    if sampled_evals.ndim != 2:
        raise NumericalRankError("sampled_evals must be a matrix.")

    analysis, left_vectors = analyze_rank(
        sampled_evals,
        rank_tol=rank_tol,
        rank_limit=basis_rank_limit,
        equilibrate=True,
        return_left_vectors=True,
    )
    # Copy only the retained range basis so the discarded SVD factors can be
    # released immediately after this function returns.
    basis_evals = left_vectors[:,:analysis.rank].copy()
    return QuotientBasisData(
        analysis.rank,
        basis_evals,
        analysis.condition,
        analysis.singular_gap,
        analysis.sample_margin,
        analysis.rank_limited,
        sampled_evals.shape[1],
    )


################################################################################
# Gramian evaluations
################################################################################


def iter_gramian_eval_chunks(
    basis_evals,
    *,
    row_indices=None,
    row_scales=None,
    multiplier_evals=None,
    sign=1.0,
    max_num_coeffs=_NUM_COEFFS_PER_CHUNK,
):
    """Yield bounded chunks of symmetric Gramian coefficient evaluations.
    Streaming is essential because a basis of rank r contributes r(r+1)/2
    symmetric coefficient columns.

    Inputs:
        basis_evals:
            ndarray. Sample-by-basis evaluation matrix.
        row_indices:
            ArrayLike or None. Selected sample rows.
        row_scales:
            ArrayLike or None. Multipliers for selected equations.
        multiplier_evals:
            ArrayLike or None. Polynomial multiplier evaluations.
        sign:
            Float. Overall coefficient sign.
        max_num_coeffs:
            Integer. Approximate maximum array elements per chunk.

    Outputs:
        chunks:
            Iterator. GramianEvalChunk objects.
    """
    basis_evals = np.asarray(basis_evals, dtype=float)

    # Resolve row selection first so every emitted chunk has exactly the shape
    # and equation scaling expected by the finalized clique system.
    use_all_rows = row_indices is None
    if use_all_rows:
        num_selected_rows = basis_evals.shape[0]
    else:
        row_indices = np.asarray(row_indices, dtype=int)
        num_selected_rows = row_indices.size
    row_scales = (
        np.ones(num_selected_rows)
        if row_scales is None else np.asarray(row_scales, dtype=float)
    )
    multiplier_evals = (
        None if multiplier_evals is None
        else np.asarray(multiplier_evals, dtype=float)
    )
    sign = float(sign)
    max_num_coeffs = max(1, int(max_num_coeffs))

    # One coefficient represents each lower-triangle Gramian entry. The chunk
    # width is chosen from the submitted row count and global element budget.
    gramian_rows, gramian_columns = lower_triangle_indices(
        basis_evals.shape[1]
    )
    num_rows = max(1, num_selected_rows)
    num_columns_chunk = max(1, max_num_coeffs//num_rows)
    if use_all_rows:
        basis_selected = basis_evals
        multiplier_selected = multiplier_evals
    else:
        basis_selected = basis_evals[row_indices,:]
        multiplier_selected = (
            None if multiplier_evals is None else multiplier_evals[row_indices]
        )

    for start in range(0, gramian_rows.size, num_columns_chunk):
        stop = min(gramian_rows.size, start + num_columns_chunk)
        rows = gramian_rows[start:stop]
        columns = gramian_columns[start:stop]
        # A symmetric off-diagonal Gramian entry contributes both q_i q_j and
        # q_j q_i, hence the factor of two in Rhiza's coefficient convention.
        coeff_evals = basis_selected[:,rows]*basis_selected[:,columns]
        off_diagonal = rows != columns
        coeff_evals[:,off_diagonal] *= 2.0
        # Inequality Gramians multiply these products by constraint samples;
        # the canonical sign and equation row scales are applied last.
        if multiplier_selected is not None:
            coeff_evals *= multiplier_selected[:,None]
        coeff_evals *= sign*row_scales[:,None]
        yield GramianEvalChunk(rows, columns, coeff_evals)


def materialize_gramian_evals(
    basis_evals,
    *,
    multiplier_evals=None,
    sign=1.0,
):
    """Materialize the complete symmetric Gramian coefficient map.

    Analysis and PQR need all coefficient columns simultaneously; solver
    backends use the bounded iterator above instead.

    Inputs:
        basis_evals:
            ndarray. Sample-by-basis evaluation matrix.
        multiplier_evals:
            ArrayLike or None. Polynomial multiplier evaluations.
        sign:
            Float. Overall coefficient sign.

    Outputs:
        coeff_evals:
            ndarray. Sample-by-Gramian-coefficient matrix.
    """
    basis_evals = np.asarray(basis_evals, dtype=float)
    multiplier_evals = (
        None if multiplier_evals is None
        else np.asarray(multiplier_evals, dtype=float)
    )
    sign = float(sign)

    # This dense path mirrors the streaming iterator exactly and is used only
    # when analysis or compression needs the complete coefficient matrix.
    gramian_rows, gramian_columns = lower_triangle_indices(
        basis_evals.shape[1]
    )
    coeff_evals = basis_evals[:,gramian_rows]*basis_evals[:,gramian_columns]
    coeff_evals[:,gramian_rows != gramian_columns] *= 2.0
    if multiplier_evals is not None:
        coeff_evals *= multiplier_evals[:,None]
    return sign*coeff_evals


################################################################################
# Polynomial evaluations
################################################################################


def evaluate_lift(variable_evals, exponents, coeffs):
    """Evaluate one declared polynomial lift at every sample.

    POP uses this calculation to create the auxiliary coordinate and to verify
    that user-supplied sampled values satisfy the declared polynomial identity.

    Inputs:
        variable_evals:
            ndarray. Sample-by-variable evaluation matrix.
        exponents:
            ndarray. Monomial-by-variable exponent matrix.
        coeffs:
            ndarray. Monomial coefficients.

    Outputs:
        lift_evals:
            ndarray. Sampled lift evaluations.
    """
    variable_evals = np.asarray(variable_evals, dtype=float)
    exponents = np.asarray(exponents, dtype=int)
    coeffs = np.asarray(coeffs, dtype=float)

    # Evaluate the declared monomial family once, then combine its columns by
    # the user coefficients to obtain the sampled lifted coordinate.
    return evaluate_monomials(variable_evals, exponents)@coeffs


def evaluate_moment(variable_evals, exponents):
    """Evaluate one requested monomial moment at every sample.

    Extraction combines this vector with reconstructed clique weights to obtain
    a normalized moment without expanding a symbolic polynomial.

    Inputs:
        variable_evals:
            ndarray. Sample-by-variable evaluation matrix.
        exponents:
            Sequence. One exponent per variable.

    Outputs:
        moment_evals:
            ndarray. Sampled monomial evaluations.
    """
    variable_evals = np.asarray(variable_evals, dtype=float)
    exponents = np.asarray(exponents, dtype=int)

    return evaluate_monomials(variable_evals, exponents[None,:])[:,0]


def evaluate_sparse_monomials(variable_evals, monomials):
    """Evaluate monomials stored by participant IDs and positive exponents.

    OVERLAP occurrence groups use this sparse representation because each
    monomial usually touches only a few variables from the full POP.

    Inputs:
        variable_evals:
            ndarray. Sample-by-variable evaluation matrix.
        monomials:
            Sequence. Pairs of variable-ID and positive-exponent tuples.

    Outputs:
        monomial_evals:
            ndarray. Sample-by-monomial evaluation matrix.
    """
    variable_evals = np.asarray(variable_evals, dtype=float)
    monomials = tuple(monomials)

    if variable_evals.ndim != 2:
        raise ValueError("variable_evals must be a matrix.")
    # Sparse monomial records avoid allocating a dense exponent row over every
    # POP variable for OVERLAP occurrence-group monomials.
    monomial_evals = np.ones(
        (variable_evals.shape[0], len(monomials)), dtype=float
    )
    for column, (participant_ids, exponents) in enumerate(monomials):
        participant_ids = tuple(participant_ids)
        exponents = tuple(exponents)
        if not participant_ids or len(participant_ids) != len(exponents):
            raise ValueError(
                "Sparse monomials require matching nonempty data."
            )
        if any(exponent <= 0 for exponent in exponents):
            raise ValueError(
                "Sparse monomial exponents must be positive."
            )
        # Build each column multiplicatively from only its participants.
        values = monomial_evals[:,column]
        for variable_id, exponent in zip(participant_ids, exponents):
            values *= variable_evals[:,variable_id]**exponent
    return monomial_evals


def evaluate_monomials(variable_evals, exponents):
    """Evaluate specified monomials without a three-dimensional temporary.

    Distinct powers of each variable are shared across matching columns, which
    keeps basis construction proportional to the output matrix rather than a
    sample-by-monomial-by-variable tensor.

    Inputs:
        variable_evals:
            ndarray. Sample-by-variable evaluation matrix.
        exponents:
            ndarray. Monomial-by-variable exponent matrix.

    Outputs:
        monomial_evals:
            ndarray. Sample-by-monomial evaluation matrix.
    """
    variable_evals = np.asarray(variable_evals, dtype=float)
    exponents = np.asarray(exponents, dtype=int)

    if variable_evals.ndim != 2 or exponents.ndim != 2:
        raise ValueError("Monomial evaluation requires two matrices.")
    if variable_evals.shape[1] != exponents.shape[1]:
        raise ValueError("Exponent width must equal the number of variables.")
    if np.any(exponents < 0):
        raise ValueError("Monomial exponents must be nonnegative.")

    num_samples = variable_evals.shape[0]
    num_monomials_here = exponents.shape[0]
    # Work variable by variable and reuse each distinct power across all
    # matching monomial columns, avoiding a sample-by-monomial-by-variable tensor.
    monomial_evals = np.ones((num_samples, num_monomials_here), dtype=float)
    for variable_id in range(variable_evals.shape[1]):
        powers = exponents[:,variable_id]
        for power in np.unique(powers[powers > 0]):
            columns = np.flatnonzero(powers == power)
            monomial_evals[:,columns] *= (
                variable_evals[:,[variable_id]]**int(power)
            )
    return monomial_evals


################################################################################
# Monomial enumeration
################################################################################


@lru_cache(maxsize=_COMBINATORIAL_CACHE_SIZE)
def enumerate_exponents(num_variables, degree):
    """Enumerate all monomial exponent rows through a total degree.

    The deterministic graded ordering is shared by quotient-basis construction
    and OVERLAP monomial enumeration.

    Inputs:
        num_variables:
            Integer. Number of variables.
        degree:
            Integer. Maximum total degree.

    Outputs:
        exponents:
            ndarray. Monomial exponent rows.
    """
    num_variables = int(num_variables)
    degree = int(degree)

    if num_variables <= 0:
        raise ValueError("num_variables must be positive.")
    if degree < 0:
        raise ValueError("degree must be nonnegative.")
    # Group rows by total degree, with the all-zero constant row first. The
    # combinations encode multisets of variable selections deterministically.
    exponents = []
    for total in range(degree + 1):
        for selection in combinations_with_replacement(
            range(num_variables), total
        ):
            exponent = [0]*num_variables
            for variable_id in selection:
                exponent[variable_id] += 1
            exponents.append(exponent)
    # Cached exponent tables are immutable so every basis request can safely
    # share the same array object.
    array = np.asarray(exponents, dtype=int)
    array.setflags(write=False)
    return array


def gramian_size_to_num_coeffs(gramian_size):
    """Count independent coefficients in a symmetric Gramian.

    Inputs:
        gramian_size:
            Integer. Gramian dimension.

    Outputs:
        count:
            Integer. Number of lower-triangle coefficients.
    """
    gramian_size = int(gramian_size)

    return gramian_size*(gramian_size + 1)//2


@lru_cache(maxsize=_COMBINATORIAL_CACHE_SIZE)
def lower_triangle_indices(gramian_size):
    """Return the shared lower-triangle Gramian ordering.

    Materialization and both native backends use these immutable indexes, so a
    coefficient column always refers to the same symmetric matrix entry.

    Inputs:
        gramian_size:
            Integer. Gramian dimension.

    Outputs:
        rows:
            ndarray. Matrix row indexes.
        columns:
            ndarray. Matrix column indexes.
    """
    gramian_size = int(gramian_size)

    # The immutable cached ordering is shared by materialization, streaming,
    # SCS svec conversion, and MOSEK triplet emission.
    rows, columns = np.tril_indices(gramian_size)
    rows.setflags(write=False)
    columns.setflags(write=False)
    return rows, columns


################################################################################
# Numerical records
################################################################################


class QuotientBasisData:
    """Store one reduced sampled polynomial-function basis.

    The record contains only evaluations and rank diagnostics consumed by SDPR;
    monomial tables and factorization data are discarded after construction.
    """

    def __init__(
        self,
        basis_rank,
        basis_evals,
        basis_condition,
        basis_singular_gap,
        sample_margin,
        basis_rank_limited,
        num_monomials_raw,
    ):
        """Store the retained basis and diagnostics.

        Inputs:
            basis_rank:
                Integer. Retained basis rank.
            basis_evals:
                ndarray. Sample-by-basis evaluation matrix.
            basis_condition:
                Float. Retained singular-value condition.
            basis_singular_gap:
                Float or None. Cutoff singular-value ratio.
            sample_margin:
                Integer. Sample rows beyond the measured empirical rank.
            basis_rank_limited:
                Boolean. Whether a user limit reduced the rank.
            num_monomials_raw:
                Integer. Number of supplied function columns.
        """
        self.basis_rank = basis_rank
        self.basis_evals = basis_evals
        self.basis_condition = basis_condition
        self.basis_singular_gap = basis_singular_gap
        self.sample_margin = sample_margin
        self.basis_rank_limited = basis_rank_limited
        self.num_monomials_raw = num_monomials_raw


class GramianEvalChunk:
    """Store one bounded Gramian coefficient-evaluation block.

    Backends consume the matrix coordinates and sampled coefficient columns
    immediately, avoiding a full sample-by-Gramian map in memory.
    """

    def __init__(
        self,
        gramian_rows,
        gramian_columns,
        coeff_evals,
    ):
        """Bundle symmetric coordinates with their sampled evaluations.

        Inputs:
            gramian_rows:
                ndarray. Symmetric matrix row indexes.
            gramian_columns:
                ndarray. Symmetric matrix column indexes.
            coeff_evals:
                ndarray. Sample-by-coefficient evaluations.
        """
        self.gramian_rows = gramian_rows
        self.gramian_columns = gramian_columns
        self.coeff_evals = coeff_evals

