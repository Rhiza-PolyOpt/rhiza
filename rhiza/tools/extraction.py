"""Moment extraction from signed clique-system sample functionals.

The routines reconstruct one signed functional per clique, normalize it by its
total mass, evaluate requested monomials on the original samples, and reconcile
quantities represented by several cliques.
"""

from collections import defaultdict
from math import fsum

import numpy as np

from rhiza.diagnostics import ExtractionError, to_python_report
from rhiza.tools.parsing import cast_exact_int
from rhiza.tools.polyalg import evaluate_moment


################################################################################
# Public extraction operations
################################################################################


def extract_means(sdpr, clique_functionals, *, variables=None):
    """Extract arithmetic averages of containing-clique first moments.

    Shared variables can have one first moment from several signed clique
    functionals. This operation returns their arithmetic average; use
    ``extract_moment_diagnostics`` to inspect the unstitched disagreement.

    Inputs:
        sdpr:
            SDPR. Finalized relaxation recipe.
        clique_functionals:
            Sequence. Reconstructed compact clique functionals.
        variables:
            Sequence or None. Variable names to extract.
    Outputs:
        means:
            dict. Variable names mapped to stitched means.
    """
    clique_functionals = tuple(clique_functionals)
    variables = (
        tuple(sdpr.pop.variable_names)
        if variables is None else tuple(variables)
    )
    unknown = set(variables) - set(sdpr.pop.variable_names)
    if unknown:
        raise ExtractionError(
            "Unknown variables requested: " + ', '.join(sorted(unknown))
        )

    # Use the already reconstructed functionals, normalizing each requested
    # coordinate by its clique's signed mass.
    requested = set(variables)
    means_by_variable = defaultdict(list)
    for functional in clique_functionals:
        clique_id = functional.clique_id
        system = sdpr._clique_systems[clique_id]
        for variable_id in system['scope_ids']:
            variable = sdpr.pop._variable_names[variable_id]
            if variable not in requested:
                continue
            means_by_variable[variable].append(
                _evaluate_clique_functional(
                    functional,
                    sdpr.pop._variable_evals[:,variable_id],
                )
            )

    # Shared variables may have one estimate from several clique functionals;
    # stitch them by the documented arithmetic mean rather than choosing a host.
    means = {}
    for variable in variables:
        values = means_by_variable[variable]
        if not values:
            raise ExtractionError(
                "No clique contains requested variable: " + repr(variable)
            )
        means[variable] = float(np.mean(values))
    return to_python_report(means)


def extract_moments(sdpr, clique_functionals, moments):
    """Extract stitched normalized moments from clique functionals.

    Inputs:
        sdpr:
            SDPR. Finalized relaxation recipe.
        clique_functionals:
            Sequence. Reconstructed compact clique functionals.
        moments:
            MomentSpec. Named variable-exponent mappings.
    Outputs:
        moments_extracted:
            dict. Requested names mapped to stitched moments.
    """
    clique_functionals = tuple(clique_functionals)

    # Normalize names, scopes, and degrees once, then evaluate every containing
    # clique through the common clique-moment path.
    normalized = normalize_moment_spec(sdpr, moments)
    moments_by_clique = _extract_clique_moments_normalized(
        sdpr,
        clique_functionals,
        normalized,
        tuple(range(len(sdpr._clique_systems))),
    )
    # Transpose the clique-keyed result to collect all evaluations of each
    # requested moment before stitching.
    by_moment = defaultdict(dict)
    for clique_id, values in moments_by_clique.items():
        for name, value in values.items():
            by_moment[name][clique_id] = value
    return to_python_report({
        name: stitch_moment_evals(by_moment[name])
        for name in normalized
    })


def extract_clique_moments(
    sdpr,
    clique_functionals,
    moments,
    *,
    cliques=None,
):
    """Extract normalized moments separately for each containing clique.

    Inputs:
        sdpr:
            SDPR. Finalized relaxation recipe.
        clique_functionals:
            Sequence. Reconstructed compact clique functionals.
        moments:
            MomentSpec. Named variable-exponent mappings.
        cliques:
            Sequence or None. Clique IDs to include.
    Outputs:
        moments_by_clique:
            dict. Clique IDs mapped to requested clique moments.
    """
    clique_functionals = tuple(clique_functionals)
    cliques = (
        tuple(range(len(sdpr._clique_systems)))
        if cliques is None else tuple(cliques)
    )
    # Validate the explicit clique selection separately from moment scopes so
    # out-of-range identifiers produce a direct error.
    normalized = normalize_moment_spec(sdpr, moments)
    for clique_id in cliques:
        if clique_id < 0 or clique_id >= len(sdpr._clique_systems):
            raise ExtractionError("Requested clique ID is out of range.")
    return to_python_report(_extract_clique_moments_normalized(
        sdpr,
        clique_functionals,
        normalized,
        cliques,
    ))


def extract_moment_diagnostics(
    sdpr,
    clique_functionals,
    *,
    variables=None,
):
    """Report clique-functional conditioning and shared-moment agreement.

    Submitted clique-system duals reconstruct signed sample functionals. Their
    masses normalize moments, while total variation records cancellation in the
    particular dual representative returned by the backend.

    Inputs:
        sdpr:
            SDPR. Finalized relaxation recipe.
        clique_functionals:
            Sequence. Reconstructed compact clique functionals.
        variables:
            Sequence or None. Variables to compare.
    Outputs:
        report:
            Report. Per-clique mass, total variation, mass condition,
            unit-mass error, and shared-moment disagreement diagnostics.
    """
    clique_functionals = tuple(clique_functionals)
    variables = (
        tuple(sdpr.pop.variable_names)
        if variables is None else tuple(variables)
    )
    unknown = set(variables) - set(sdpr.pop.variable_names)
    if unknown:
        raise ExtractionError(
            "Unknown variables requested: " + ', '.join(sorted(unknown))
        )

    # Keep conditioning data for each signed functional rather than reducing
    # it to a pass/fail flag. Large total variation can reveal dual nonuniqueness
    # or cancellation even when masses and shared moments remain coherent.
    requested = set(variables)
    mass_by_clique = {}
    total_variation_by_clique = {}
    mass_condition_by_clique = {}
    mass_error_by_clique = {}
    means_by_variable = defaultdict(dict)
    for functional in clique_functionals:
        clique_id = functional.clique_id
        system = sdpr._clique_systems[clique_id]
        mass_by_clique[clique_id] = functional.mass
        total_variation_by_clique[clique_id] = functional.total_variation
        mass_condition_by_clique[clique_id] = functional.mass_condition
        mass_error_by_clique[clique_id] = functional.mass_error
        for variable_id in system['scope_ids']:
            variable = sdpr.pop._variable_names[variable_id]
            if variable not in requested:
                continue
            means_by_variable[variable][clique_id] = (
                _evaluate_clique_functional(
                    functional,
                    sdpr.pop._variable_evals[:,variable_id],
                )
            )

    # Only variables represented by multiple cliques have a meaningful
    # disagreement statistic.
    difference_max = {}
    difference_rms = {}
    for variable, values_by_clique in means_by_variable.items():
        values = tuple(values_by_clique.values())
        if len(values) <= 1:
            continue
        difference_max[variable] = compute_difference_max(values)
        difference_rms[variable] = compute_difference_rms(values)

    report = {
        'mass_by_clique': mass_by_clique,
        'total_variation_by_clique': total_variation_by_clique,
        'mass_condition_by_clique': mass_condition_by_clique,
        'mass_error_by_clique': mass_error_by_clique,
        'means_by_variable': dict(means_by_variable),
        'stitching_error_max_by_variable': difference_max,
        'stitching_error_rms_by_variable': difference_rms,
        'stitching_error_max': (
            max(difference_max.values()) if difference_max else 0.0
        ),
        'stitching_error_rms': (
            float(np.sqrt(np.mean(np.square(tuple(difference_rms.values())))))
            if difference_rms else 0.0
        ),
    }
    return to_python_report(report)


################################################################################
# Moment extraction stages
################################################################################


def _extract_clique_moments_normalized(
    sdpr,
    clique_functionals,
    moments,
    cliques,
):
    """Extract clique moments from an already normalized request.

    Each selected clique contributes a signed functional and its validated
    mass. A requested monomial is evaluated only where its variable scope fits,
    normalized by that mass, and left unstitched for the caller.

    Inputs:
        sdpr:
            SDPR. Finalized relaxation recipe.
        clique_functionals:
            Sequence. Reconstructed compact clique functionals.
        moments:
            Mapping. Normalized moment requests.
        cliques:
            Sequence. Clique IDs to include.
    Outputs:
        moments_by_clique:
            dict. Clique IDs mapped to requested clique moments.
    """
    # Evaluate each requested monomial once globally, then reuse the sample
    # vector for every clique whose scope contains it.
    moment_evals = {
        name: _evaluate_moment_samples(sdpr.pop, moment)
        for name, moment in moments.items()
    }
    result = {}
    for clique_id in cliques:
        functional = clique_functionals[clique_id]
        system = sdpr._clique_systems[clique_id]
        # A clique reports only the requested moments it can represent; missing
        # entries are later supplied by other containing cliques or rejected.
        clique_scope = set(system['scope_ids'])
        values = {}
        for name, moment in moments.items():
            moment_scope = {variable_id for variable_id, _ in moment}
            if moment_scope.issubset(clique_scope):
                values[name] = _evaluate_clique_functional(
                    functional, moment_evals[name]
                )
        result[clique_id] = values
    return result


def _evaluate_clique_functional(functional, sample_evals):
    """Evaluate one normalized moment from retained sample weights.

    The compact functional omits only compressed-away rows, whose dual weights
    are zero. A vectorized dot product therefore evaluates the same moment as a
    dense full-sample functional without rebuilding that dense vector.

    Inputs:
        functional:
            CliqueFunctional. Validated compact signed functional.
        sample_evals:
            ndarray. Moment evaluations on the complete sample set.

    Outputs:
        value:
            Float. Moment normalized by the clique functional's mass.
    """
    sample_evals = np.asarray(sample_evals, dtype=float)
    retained_evals = sample_evals[functional.sample_indices]
    if retained_evals.shape != functional.weights.shape:
        raise ExtractionError(
            "Clique weights and retained moment evaluations disagree."
        )
    if not np.all(np.isfinite(retained_evals)):
        raise ExtractionError("Moment evaluations must be finite.")
    numerator = float(functional.weights@retained_evals)
    if not np.isfinite(numerator):
        raise ExtractionError(
            "A clique moment is not representable in floating point."
        )
    value = numerator/functional.mass
    if not np.isfinite(value):
        raise ExtractionError(
            "A normalized clique moment is not representable in floating "
            "point."
        )
    return float(value)


def _evaluate_moment_samples(pop, moment):
    """Evaluate one requested monomial on the supplied POP samples.

    Inputs:
        pop:
            POP. Source sampled problem.
        moment:
            Sequence. Variable-ID exponent pairs.

    Outputs:
        moment_evals:
            ndarray. Requested monomial at every sample.
    """
    if not moment:
        return np.ones(pop.num_samples)
    variable_ids = tuple(variable_id for variable_id, _ in moment)
    exponents = tuple(exponent for _, exponent in moment)
    return evaluate_moment(
        pop._variable_evals[:,variable_ids],
        exponents,
    )


################################################################################
# Dual reconstruction
################################################################################


class CliqueFunctional:
    """Store one compact signed sample functional for extraction.

    Only retained sample rows are stored. Compressed-away rows have zero dual
    weight, so omitting them preserves every mass and moment while avoiding a
    dense number-of-samples vector for each clique.
    """

    __slots__ = (
        'clique_id',
        'sample_indices',
        'weights',
        'mass',
        'total_variation',
        'mass_condition',
        'mass_error',
    )

    def __init__(
        self,
        clique_id,
        sample_indices,
        weights,
        mass,
        total_variation,
        mass_condition,
        mass_error,
    ):
        """Store retained weights and their validated mass diagnostics.

        Inputs:
            clique_id:
                Integer. Clique ID of the associated clique system.
            sample_indices:
                ndarray. Retained sample indexes.
            weights:
                ndarray. Signed weights on retained samples.
            mass:
                Float. Sum of retained weights.
            total_variation:
                Float. Sum of absolute retained weights.
            mass_condition:
                Float. Total variation divided by absolute mass.
            mass_error:
                Float. Absolute deviation from unit mass.
        """
        self.clique_id = clique_id
        self.sample_indices = sample_indices
        self.weights = weights
        self.mass = mass
        self.total_variation = total_variation
        self.mass_condition = mass_condition
        self.mass_error = mass_error


def reconstruct_clique_functionals(sdpr, system_duals, *, mass_abs_tol):
    """Reconstruct and validate one compact signed functional per clique.

    This is the only stage that reverses submitted row scaling and partitions
    the backend dual vector. Solver caches its result so every extraction method
    uses the same masses, conditioning diagnostics, and retained weights.

    Inputs:
        sdpr:
            SDPR. Finalized relaxation recipe.
        system_duals:
            ndarray. Concatenated submitted equation duals.
        mass_abs_tol:
            Float. Absolute near-zero mass tolerance.

    Outputs:
        functionals:
            tuple. Compact CliqueFunctional records in clique-ID order.
    """
    system_duals = np.asarray(system_duals, dtype=float)
    expected = sum(
        len(system['row_indices']) for system in sdpr._clique_systems
    )
    if system_duals.shape != (expected,):
        raise ExtractionError(
            f"Expected {expected} system duals, received {system_duals.shape}."
        )

    # Backends concatenate clique equations in clique order. Partition that
    # vector once, undo each clique's equation scaling, and retain only submitted
    # sample rows. Compressed-away rows have exactly zero dual weight.
    functionals = []
    start = 0
    for clique_id, system in enumerate(sdpr._clique_systems):
        stop = start + len(system['row_indices'])
        weights = np.asarray(
            system['row_scales']*system_duals[start:stop],
            dtype=float,
        )
        start = stop
        mass = compute_clique_mass(weights)
        diagnostics = validate_clique_mass(
            weights,
            mass,
            mass_abs_tol=mass_abs_tol,
            clique_id=clique_id,
        )
        functionals.append(CliqueFunctional(
            clique_id=clique_id,
            sample_indices=system['row_indices'],
            weights=weights,
            mass=mass,
            total_variation=diagnostics['total_variation'],
            mass_condition=diagnostics['mass_condition'],
            mass_error=diagnostics['mass_error'],
        ))
    return tuple(functionals)


def compute_clique_mass(clique_weights):
    """Return the total mass of one signed clique functional.

    The stable summation preserves a small residual mass when large positive
    and negative dual coefficients cancel.

    Inputs:
        clique_weights:
            ndarray. Retained signed sample weights.

    Outputs:
        mass:
            Float. Sum of sample weights.
    """
    clique_weights = np.asarray(clique_weights, dtype=float)

    if clique_weights.ndim != 1 or not np.all(np.isfinite(clique_weights)):
        raise ExtractionError(
            "Clique sample weights must be a finite vector."
        )
    try:
        return float(fsum(clique_weights))
    except (OverflowError, ValueError) as error:
        raise ExtractionError(
            "Clique sample weights have an unrepresentable total mass."
        ) from error


def validate_clique_mass(
    clique_weights,
    mass,
    *,
    mass_abs_tol,
    clique_id,
):
    """Require a clique functional with numerically meaningful total mass.

    Inputs:
        clique_weights:
            ndarray. Retained signed sample weights.
        mass:
            Float. Sum of sample weights.
        mass_abs_tol:
            Float. Absolute near-zero threshold from backend accuracy.
        clique_id:
            Integer. Clique identifier for errors.

    Outputs:
        diagnostics:
            dict. Total variation, mass condition, and unit-mass error.
    """
    clique_weights = np.asarray(clique_weights, dtype=float)
    mass = float(mass)
    mass_abs_tol = float(mass_abs_tol)
    clique_id = int(clique_id)

    if clique_weights.ndim != 1 or not np.all(np.isfinite(clique_weights)):
        raise ExtractionError(
            f"Clique {clique_id} has nonfinite or malformed sample weights."
        )
    if not np.isfinite(mass):
        raise ExtractionError(f"Clique {clique_id} has a nonfinite mass.")
    if not np.isfinite(mass_abs_tol) or mass_abs_tol < 0.0:
        raise ExtractionError("Mass tolerance must be finite and nonnegative.")

    # Backend feasibility gives an absolute normalization floor. Independently,
    # cancellation can lose floating-point digits in a signed sum, so include a
    # machine-scale term proportional to total variation. Solver feasibility
    # is deliberately not multiplied by total variation: a valid dual
    # representative may have large positive and negative weights but total
    # mass one.
    try:
        total_variation = float(
            fsum(abs(value) for value in clique_weights)
        )
    except (OverflowError, ValueError) as error:
        raise ExtractionError(
            f"Clique {clique_id} has unrepresentable total variation."
        ) from error
    if not np.isfinite(total_variation):
        raise ExtractionError(
            f"Clique {clique_id} has unrepresentable total variation."
        )
    # ``fsum`` stabilizes the final signed reduction. The modest safety
    # factor also covers roundoff introduced while reconstructing the
    # sample weights from compressed and row-scaled clique equations.
    roundoff_tol = (
        64.0*np.finfo(float).eps*max(1.0, total_variation)
    )
    zero_mass_tol = max(mass_abs_tol, roundoff_tol)
    if abs(mass) <= zero_mass_tol:
        raise ExtractionError(
            f"Clique {clique_id} has a numerically zero mass "
            f"(|mass|={abs(mass):.3e}, threshold={zero_mass_tol:.3e})."
        )

    return {
        'total_variation': total_variation,
        'mass_condition': total_variation/abs(mass),
        'mass_error': abs(mass - 1.0),
    }


################################################################################
# Moment requests
################################################################################


def normalize_moment_spec(sdpr, moments):
    """Map a named moment specification to canonical variable-ID tuples.

    Inputs:
        sdpr:
            SDPR. Relaxation defining valid moments.
        moments:
            MomentSpec. Named variable-exponent mappings.

    Outputs:
        moments_normalized:
            dict. Names mapped to sorted variable-ID exponent tuples.
    """
    # Convert user variable names to compact IDs and drop zero exponents.
    # Every equivalent moment request then has one canonical tuple form.
    normalized = {}
    for name, spec in moments.items():
        entries = []
        for variable, exponent in spec.items():
            if variable not in sdpr.pop._variable_ids:
                raise ExtractionError(
                    "Unknown moment variable: " + repr(variable)
                )
            if exponent < 0:
                raise ExtractionError("Moment exponents must be nonnegative.")
            if exponent:
                entries.append((sdpr.pop._variable_ids[variable], exponent))
        entries.sort()
        moment = tuple(entries)
        # Degree sufficiency and clique containment are separate requirements:
        # the relaxation must represent the monomial both algebraically and locally.
        validate_moment_degree(moment, sdpr.sos_degree)
        if not find_moment_cliques(sdpr, moment):
            raise ExtractionError(
                "No clique contains requested moment: " + repr(name)
            )
        normalized[name] = moment
    return normalized


def validate_moment_degree(moment, sos_degree):
    """Require a moment degree represented by the relaxation.

    A degree beyond ``sos_degree`` has no corresponding sampled moment in the
    finalized recipe, so extraction rejects it before consulting clique data.

    Inputs:
        moment:
            Sequence. Variable-ID exponent pairs.
        sos_degree:
            Integer. Relaxation moment degree.
    """
    sos_degree = int(sos_degree)

    degree = sum(exponent for _, exponent in moment)
    if degree > sos_degree:
        raise ExtractionError(
            f"Requested moment degree {degree} exceeds "
            f"sos_degree {sos_degree}."
        )


def find_moment_cliques(sdpr, moment):
    """Return cliques whose variable scopes contain a requested moment.

    Moment degree and moment locality are independent requirements: this helper
    checks only whether the requested moment scope is contained in a clique system.

    Inputs:
        sdpr:
            SDPR. Relaxation recipe.
        moment:
            Sequence. Variable-ID exponent pairs.

    Outputs:
        clique_ids:
            tuple. Containing clique IDs.
    """
    scope = {variable_id for variable_id, _ in moment}

    return tuple(
        system['clique_id']
        for system in sdpr._clique_systems
        if scope.issubset(system['scope_ids'])
    )


def stitch_moment_evals(moment_evals_by_clique):
    """Average compatible clique evaluations of one requested moment.

    Every containing clique represents the same polynomial function in an exact
    coherent solution. Arithmetic averaging treats them symmetrically while the
    unstitched differences remain available through ``extract_moment_diagnostics``.

    Inputs:
        moment_evals_by_clique:
            Mapping. Clique IDs mapped to normalized moments.

    Outputs:
        moment:
            Float. Stitched arithmetic mean.
    """
    values = tuple(dict(moment_evals_by_clique).values())

    if not values:
        raise ExtractionError("No clique moment evaluations were available.")

    return float(np.mean(values))


################################################################################
# Stitching diagnostics
################################################################################


def compute_difference_max(evals):
    """Return the range of clique moment evaluations.

    Inputs:
        evals:
            Sequence. Clique moment evaluations.

    Outputs:
        difference:
            Float. Maximum pairwise absolute difference.
    """
    evals = np.asarray(evals, dtype=float)

    return float(np.ptp(evals)) if evals.size else 0.0


def compute_difference_rms(evals):
    """Return RMS deviation from the clique arithmetic mean.

    Unlike the maximum pairwise range, this statistic summarizes the typical
    spread when a variable or moment is represented by several cliques.

    Inputs:
        evals:
            Sequence. Clique moment evaluations.

    Outputs:
        difference:
            Float. Root-mean-square deviation.
    """
    evals = np.asarray(evals, dtype=float)

    if evals.size == 0:
        return 0.0
    centered = evals - np.mean(evals)
    return float(np.sqrt(np.mean(centered*centered)))


################################################################################
# Input helper
################################################################################


def cast_moment_spec(moments):
    """Cast a named moment specification without lossy name or exponent coercion.
    Moment exponents determine represented polynomial degree, so fractional values
    are rejected rather than truncated.

    Inputs:
        moments:
            MomentSpec. Named variable-exponent mappings.

    Outputs:
        moments:
            dict. Literal names mapped to exact integer exponents.
    """
    try:
        items = dict(moments).items()
    except (TypeError, ValueError) as error:
        raise ExtractionError("moments must be a mapping.") from error

    cast = {}
    for name, spec in items:
        name = _cast_literal_name(name, "Moment name")
        try:
            entries = dict(spec).items()
        except (TypeError, ValueError) as error:
            raise ExtractionError(
                "Each moment specification must be a mapping."
            ) from error
        cast[name] = {
            _cast_literal_name(variable, "Moment variable"): cast_exact_int(
                exponent, "Moment exponent", ExtractionError
            )
            for variable, exponent in entries
        }
    return cast


def cast_variable_names(variables):
    """Cast one variable selection while treating a string as a singleton. This
    convention prevents multi-character variable names from being split into
    character requests by extraction methods.

    Inputs:
        variables:
            Sequence or string. Candidate variable names.

    Outputs:
        variables:
            tuple. Literal variable names.
    """
    if isinstance(variables, str):
        values = (variables,)
    else:
        try:
            values = tuple(variables)
        except TypeError as error:
            raise ExtractionError(
                "variables must be a string or an iterable of strings."
            ) from error
    return tuple(
        _cast_literal_name(value, "Requested variable") for value in values
    )


def cast_clique_ids(cliques):
    """Cast a clique selection to exact integer identifiers. Fractional real values
    are rejected before range validation so a request cannot silently target a
    different clique.

    Inputs:
        cliques:
            Sequence. Candidate clique identifiers.

    Outputs:
        cliques:
            tuple. Exact integer clique identifiers.
    """
    if isinstance(cliques, (str, bytes)):
        raise ExtractionError("cliques must be an iterable of integer IDs.")
    try:
        values = tuple(cliques)
    except TypeError as error:
        raise ExtractionError(
            "cliques must be an iterable of integer IDs."
        ) from error
    return tuple(
        cast_exact_int(value, "Clique ID", ExtractionError)
        for value in values
    )


def _cast_literal_name(value, label):
    """Require a literal extraction name rather than stringifying arbitrary input.
    This preserves the exact namespace created by POP and produces direct errors
    for accidental ``None`` or numeric names.

    Inputs:
        value:
            Object. Candidate name.
        label:
            String. Context used in validation errors.

    Outputs:
        name:
            String. Literal name.
    """
    if not isinstance(value, str):
        raise ExtractionError(label + " must be a string.")
    return value
