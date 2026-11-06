"""Tests for mutable POP construction and finalization."""

import json

import numpy as np
import pytest

import rhiza as rz
from rhiza.diagnostics import (
    PipelineError,
    POPParseError,
    POPValidationError,
    Relation,
)


################################################################################
# Basic construction
################################################################################


def make_small_pop():
    """Return a small mutable POP."""
    pop = rz.POP()
    pop.add_variable("x", [-1.0, 0.0, 1.0])
    pop.add_variable("y", [2.0, 3.0, 4.0])
    pop.add_objective(
        [-2.0, 0.0, 4.0],
        ('x', 'y'),
        2,
        form='MIN',
        name="cost",
    )
    return pop


def test_add_methods_return_true():
    """Return a useful success value after every insertion."""
    pop = rz.POP()
    assert pop.add_variable("x", [0.0, 1.0])
    assert pop.add_objective([0.0, 1.0], ('x',), 1, form='MIN')
    assert pop.add_constraint(
        [1.0, 0.0], ('x',), 1, form='>=0', name="bound"
    )


def test_objective_and_constraint_forms():
    """Normalize objective forms while preserving symbolic constraint forms."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    pop.add_objective([0.0, -1.0], ('x',), 1, form='max')
    pop.add_constraint([0.0, 0.0], ('x',), 1, form='==0')
    assert pop._pending_entries[0]['relation'] is Relation.MAX
    assert pop._pending_entries[1]['relation'] is Relation.EQ


@pytest.mark.parametrize('form', ('minimum', 'GE', ''))
def test_invalid_objective_form(form):
    """Reject objective tokens outside MIN and MAX."""
    pop = rz.POP()
    pop.add_variable("x", [0.0])
    with pytest.raises(POPValidationError):
        pop.add_objective([0.0], ('x',), 1, form=form)


@pytest.mark.parametrize('form', ('GE', '>=', '='))
def test_invalid_constraint_form(form):
    """Reject constraint tokens outside the file-format relations."""
    pop = rz.POP()
    pop.add_variable("x", [0.0])
    with pytest.raises(POPValidationError):
        pop.add_constraint([0.0], ('x',), 1, form=form)


def test_global_name_conflicts():
    """Prevent conflicts among variable and named-entry names."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    with pytest.raises(POPValidationError):
        pop.add_objective([0.0, 1.0], ('x',), 1, name="x")
    pop.add_objective([0.0, 1.0], ('x',), 1, name="cost")
    with pytest.raises(POPValidationError):
        pop.add_variable("cost", [0.0, 1.0])


def test_sample_count_mismatch():
    """Reject sampled rows with inconsistent lengths."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    with pytest.raises(POPValidationError):
        pop.add_variable("y", [0.0])


def test_first_entry_establishes_sample_count_and_builder_stats():
    """Store the common row count without rescanning prior declarations."""
    pop = rz.POP()
    pop.add_objective([1.0, 2.0, 3.0], ('x',), 1, name="cost")

    assert pop.num_samples == 3
    assert pop.stats == {
        'num_samples': 3,
        'num_variables': 0,
        'num_entries': 1,
    }

    pop.add_variable("x", [0.0, 1.0, 2.0])
    assert pop.stats == {
        'num_samples': 3,
        'num_variables': 1,
        'num_entries': 1,
    }


def test_lift_without_targets_does_not_inspect_named_entries():
    """Keep ordinary lift declarations independent of prior entry count."""

    class _NoLookupMapping(dict):
        """Fail if a no-target lift inspects the named-entry index."""

        def __contains__(self, key):
            """Reject membership checks that the fast path should avoid."""
            raise AssertionError("Named entries should not be inspected.")

        def items(self):
            """Reject iteration that the fast path should avoid."""
            raise AssertionError("Named entries should not be traversed.")

    pop = rz.POP()
    pop.add_variable("x", [1.0, 2.0])
    pop.add_variable("y", [3.0, 4.0])
    pop._named_entries = _NoLookupMapping()

    assert pop.add_product(('x', 'y'), name="xy")
    np.testing.assert_allclose(pop._pending_variables['xy'], [3.0, 8.0])


def test_public_names_are_literal_strings_without_outer_whitespace():
    """Reject accidental namespace entries created by lossy name coercion."""
    pop = rz.POP()
    with pytest.raises(POPValidationError, match='string'):
        pop.add_variable(None, [0.0, 1.0])
    with pytest.raises(POPValidationError, match='whitespace'):
        pop.add_variable(" x", [0.0, 1.0])

    pop.add_variable("x", [0.0, 1.0])
    with pytest.raises(POPValidationError, match='string'):
        pop.add_objective([0.0, 1.0], ('x',), 1, name=3)
    with pytest.raises(POPValidationError, match='string'):
        pop.add_lift(('x',), ((1,),), name=None)


def test_string_scope_denotes_one_variable():
    """Treat a bare multi-character name as one scope entry."""
    pop = rz.POP()
    pop.add_variable("theta", [0.0, 1.0])
    pop.add_objective([0.0, 1.0], "theta", 1)
    assert pop._pending_entries[0]['scope'] == ('theta',)


def test_public_degrees_require_exact_integers():
    """Accept integer-valued reals but reject fractional polynomial degrees."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    assert pop.add_objective([0.0, 1.0], ('x',), 2.0)
    with pytest.raises(POPValidationError, match='integer'):
        pop.add_constraint([1.0, 0.0], ('x',), 1.5, form='>=0')


def test_lift_exponents_and_target_degrees_require_exact_integers():
    """Prevent fractional lift metadata from changing polynomial structure."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError, match='integer'):
        pop.add_lift(('x',), ((1.5,),), name="bad")
    with pytest.raises(POPValidationError, match='integer'):
        pop.add_product(
            ('x', 'y'),
            name="xy",
            targets={"cost": 1.5},
        )

    assert pop.add_lift(
        ('x',),
        ((2.0,),),
        name="x2",
        targets={"cost": 2.0},
    )


################################################################################
# Products and lifts
################################################################################


def test_add_product_rewrites_targets():
    """Replace a target source scope with the new product variable."""
    pop = make_small_pop()
    assert pop.add_product(
        ('x', 'y'),
        name="xy",
        targets={"cost": 1},
    )
    np.testing.assert_allclose(
        pop._pending_variables['xy'],
        [-2.0, 0.0, 4.0],
    )
    cost = next(entry for entry in pop._pending_entries if entry['name'] == "cost")
    assert cost['scope'] == ('xy',)
    assert cost['degree'] == 1
    identity = pop._pending_entries[-1]
    assert identity['relation'] is Relation.EQ
    assert identity['scope'] == ('x', 'y', 'xy')
    assert np.array_equal(identity['evals'], np.zeros(3))


def test_product_default_name():
    """Derive a compact product name by concatenating source names."""
    pop = make_small_pop()
    pop.add_product(('x', 'y'))
    assert 'xy' in pop.variable_names


def test_add_lift_degree_uses_python_integer_sum():
    """Compute a large supported defining degree without integer overflow."""
    exponents = (1_500_000_000, 600_000_000)
    pop = rz.POP()
    pop.add_variable("x", [1.0, 1.0])
    pop.add_variable("y", [1.0, 1.0])
    pop.add_lift(
        ('x', 'y'),
        (exponents,),
        name="large_degree",
    )

    identity = pop._pending_entries[-1]
    assert identity['degree'] == sum(exponents)


def test_add_lift_evaluates_polynomial():
    """Evaluate a sum-of-monomials lift and record its defining scope."""
    pop = make_small_pop()
    pop.add_lift(
        ('x', 'y'),
        ((1, 1), (2, 0)),
        name="lift",
        coeffs=(2.0, -1.0),
    )
    expected = 2.0*np.array([-2.0, 0.0, 4.0]) - np.array([1.0, 0.0, 1.0])
    np.testing.assert_allclose(pop._pending_variables['lift'], expected)
    identity = pop._pending_entries[-1]
    assert identity['scope'] == ('x', 'y', 'lift')
    assert identity['degree'] == 2


def test_add_lift_rejects_nonmatrix_exponents():
    """Reject lift exponents that are not a two-dimensional array."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError):
        pop.add_lift(('x', 'y'), (1, 1), name="lift")


def test_lift_target_failure_is_transactional():
    """Leave the POP unchanged when any target rewrite is invalid."""
    pop = make_small_pop()
    variables_before = tuple(pop.variable_names)
    entries_before = len(pop._pending_entries)
    with pytest.raises(POPValidationError):
        pop.add_product(
            ('x', 'y'),
            name="xy",
            targets={"missing": 1},
        )
    assert pop.variable_names == variables_before
    assert len(pop._pending_entries) == entries_before


################################################################################
# Removal and finalization
################################################################################


def test_remove_named_entry():
    """Remove named entries and report missing names without error."""
    pop = make_small_pop()
    assert pop.remove("cost")
    assert not pop.remove("cost")
    with pytest.raises(POPValidationError):
        pop.remove("x")


def test_finalize_and_lock():
    """Compact a valid POP and reject later mutation."""
    pop = make_small_pop()
    assert pop.finalize()
    assert not pop.finalize()
    assert pop.finalized
    assert pop.stats['num_variables'] == 2
    with pytest.raises(PipelineError):
        pop.add_variable("z", [0.0, 0.0, 0.0])


def test_finalize_rejects_invalid_identity_tolerance_contextually():
    """Translate invalid tolerance conversion into a POP validation error."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError, match='real scalar'):
        pop.finalize(identity_tol="not-a-number")


def test_finalize_rejects_changed_identity_tolerance():
    """Reject a repeated finalization with different numerical settings."""
    pop = make_small_pop()
    pop.finalize(identity_tol=1e-12)
    with pytest.raises(PipelineError):
        pop.finalize(identity_tol=1e-8)


def test_finalize_failure_preserves_builder():
    """Keep an invalid POP mutable after transactional finalization fails."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    pop.add_objective([0.0, 1.0], ('missing',), 1, form='MIN')
    with pytest.raises(POPValidationError):
        pop.finalize()
    assert not pop.finalized
    assert pop.add_variable("missing", [0.0, 1.0])


def test_finalize_rejects_samplewise_zero_inequality():
    """Reject samples that place an inequality entirely on its boundary."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    pop.add_objective([0.0, 1.0], ('x',), 1)
    pop.add_constraint(
        [0.0, 0.0],
        ('x',),
        1,
        form='>=0',
        name="boundary",
    )

    with pytest.raises(
        POPValidationError,
        match="Inequality 'boundary' evaluates to zero at every sample",
    ):
        pop.finalize()
    assert not pop.finalized


def test_finalize_accepts_any_nonzero_inequality_sample():
    """Use exact zero detection so harmless rescaling stays admissible."""
    smallest = np.nextafter(0.0, 1.0)
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    pop.add_objective([0.0, 1.0], ('x',), 1)
    pop.add_constraint([0.0, smallest], ('x',), 1, form='>=0')
    assert pop.finalize()


def test_sampled_identity_classification():
    """Classify machine-scale zero equality evals once during finalization."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    pop.add_objective([0.0, 1.0], ('x',), 1)
    pop.add_constraint([0.0, 1e-15], ('x',), 2, form='==0')
    pop.finalize()
    assert pop._sampled_identities.tolist() == [False, True]


def test_save_requires_finalization(tmp_path):
    """Save only exact finalized POP data."""
    pop = make_small_pop()
    with pytest.raises(PipelineError):
        pop.save(tmp_path/'problem.pop')
    pop.finalize()
    assert pop.save(tmp_path/'problem.pop')


def test_save_anonymizes_entries(tmp_path):
    """Preserve variable names while intentionally dropping entry names."""
    pop = make_small_pop()
    pop.finalize()
    path = tmp_path/'problem.pop'
    pop.save(path)
    loaded = rz.POP(path)
    assert loaded.entry_names == ()
    assert loaded.variable_names == ('x', 'y')


def test_file_line_order_is_flexible(tmp_path):
    """Load objective rows before their variable declarations."""
    path = tmp_path/'unordered.pop'
    path.write_text(
        'min,2,x-y,0,1\n'
        'var,1,y,0,1\n'
        'var,1,x,0,1\n'
    )
    pop = rz.POP(path)
    assert pop.finalize()
    assert set(pop.variable_names) == {'x', 'y'}


def test_file_parser_accepts_case_insensitive_relation_tokens(tmp_path):
    """Load supported relation tokens through the complete POP parser."""
    path = tmp_path/'case.pop'
    path.write_text('VAR,1,x,0,1\nMIN,2,x,0,1\n')
    pop = rz.POP(path)
    assert pop.finalize()
    assert pop.variable_names == ('x',)
    assert pop.stats['num_objectives'] == 1


def test_printable_before_and_after_finalize():
    """Provide a compact report summary throughout the POP lifecycle."""
    pop = make_small_pop()
    assert 'finalized: False' in str(pop)
    pop.finalize()
    assert 'graph policy: RIP' in str(pop)


################################################################################
# Comprehensive lift validation
################################################################################


def test_add_lift_uses_unit_coeffs_and_rewrites_targets():
    """Use unit coefficients and rewrite a named target transactionally."""
    x_evals = np.array([-1.0, 0.0, 1.0])
    y_evals = np.array([2.0, 3.0, 4.0])
    pop = rz.POP()
    pop.add_variable("x", x_evals)
    pop.add_variable("y", y_evals)
    pop.add_objective(
        x_evals + y_evals,
        ('x', 'y'),
        1,
        name="cost",
    )
    assert pop.add_lift(
        ('x', 'y'),
        ((1, 0), (0, 1)),
        name="sum_xy",
        targets={"cost": 1},
    )

    np.testing.assert_allclose(
        pop._pending_variables['sum_xy'],
        [1.0, 3.0, 5.0],
    )
    cost = next(entry for entry in pop._pending_entries if entry['name'] == "cost")
    assert cost['scope'] == ('sum_xy',)
    assert cost['degree'] == 1


def test_add_lift_retains_participating_declared_source_scope():
    """Retain each participating source variable in the defining identity."""
    pop = make_small_pop()
    pop.add_lift(
        ('x', 'y'),
        ((2, 0), (0, 1)),
        name="quadratic_sum",
    )

    identity = pop._pending_entries[-1]
    assert identity['scope'] == ('x', 'y', 'quadratic_sum')
    assert identity['degree'] == 2


def test_add_lift_rejects_constant_only_polynomial_with_source_scope():
    """Reject a constant lift that declares inactive source variables."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError, match='nonzero monomial'):
        pop.add_lift(
            ('x', 'y'),
            ((0, 0),),
            name="constant",
            coeffs=(3.5,),
        )


@pytest.mark.parametrize(
    'exponents',
    (
        (),
        ((-1, 0),),
        ((1, 0, 0),),
    ),
)
def test_add_lift_rejects_invalid_exponents(exponents):
    """Reject empty, negative, and incorrectly sized exponent arrays."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError):
        pop.add_lift(('x', 'y'), exponents, name="lift")


@pytest.mark.parametrize(
    'coeffs',
    (
        (1.0,),
        (1.0, np.inf),
        (1.0, np.nan),
    ),
)
def test_add_lift_rejects_invalid_coeffs(coeffs):
    """Reject malformed or nonfinite lift coefficients."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError):
        pop.add_lift(
            ('x', 'y'),
            ((1, 0), (0, 1)),
            name="lift",
            coeffs=coeffs,
        )


def test_lift_rejects_unknown_and_repeated_sources():
    """Require a distinct existing variable for every lift source."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError):
        pop.add_product(('x', 'missing'), name="product")
    with pytest.raises(POPValidationError):
        pop.add_product(('x', 'x'), name="product")
    with pytest.raises(POPValidationError):
        pop.add_lift(('x', 'missing'), ((1, 1),), name="lift")


def test_polynomial_lift_rejects_repeated_sources():
    """Require distinct source variables for a general polynomial lift."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError):
        pop.add_lift(
            ('x', 'x'),
            ((1, 1),),
            name="x_product",
        )


def test_lift_helpers_reject_mutation_after_finalization():
    """Reject product and polynomial lifts after POP finalization."""
    product_pop = make_small_pop()
    product_pop.finalize()
    with pytest.raises(PipelineError):
        product_pop.add_product(('x', 'y'), name="xy")

    lift_pop = make_small_pop()
    lift_pop.finalize()
    with pytest.raises(PipelineError):
        lift_pop.add_lift(('x', 'y'), ((1, 1),), name="xy")


def test_lift_rejects_global_name_collisions():
    """Prevent lifted variables from conflicting with variables or entries."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError):
        pop.add_product(('x', 'y'), name="x")
    with pytest.raises(POPValidationError):
        pop.add_lift(('x', 'y'), ((1, 1),), name="cost")


@pytest.mark.parametrize(
    'targets',
    (
        {"missing": 1},
        {"cost": -1},
    ),
)
def test_lift_rejects_invalid_targets_without_mutation(targets):
    """Leave every builder record unchanged after target validation fails."""
    pop = make_small_pop()
    variables_before = tuple(pop.variable_names)
    entries_before = tuple(
        (entry['scope'], entry['degree'], entry['name'])
        for entry in pop._pending_entries
    )

    with pytest.raises(POPValidationError):
        pop.add_lift(
            ('x', 'y'),
            ((1, 1),),
            name="xy",
            targets=targets,
        )

    assert pop.variable_names == variables_before
    assert tuple(
        (entry['scope'], entry['degree'], entry['name'])
        for entry in pop._pending_entries
    ) == entries_before


def test_lift_target_must_contain_source_scope():
    """Reject a target that does not contain every replaced source variable."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    pop.add_variable("y", [2.0, 3.0])
    pop.add_objective([0.0, 1.0], ('x',), 1, name="cost")

    with pytest.raises(POPValidationError):
        pop.add_product(
            ('x', 'y'),
            name="xy",
            targets={"cost": 1},
        )


def test_lift_target_cannot_already_contain_new_variable():
    """Reject target metadata that already names the lifted variable."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    pop.add_variable("y", [2.0, 3.0])
    pop.add_objective(
        [0.0, 3.0],
        ('x', 'y', 'xy'),
        2,
        name="cost",
    )

    with pytest.raises(POPValidationError):
        pop.add_product(
            ('x', 'y'),
            name="xy",
            targets={"cost": 1},
        )


def test_product_overflow_is_transactional():
    """Reject nonfinite product evaluations before changing builder state."""
    pop = rz.POP()
    pop.add_variable("x", [1e308, 1e308])
    pop.add_variable("y", [1e308, -1e308])
    pop.add_objective([0.0, 0.0], ('x', 'y'), 1, name="cost")

    with pytest.raises(POPValidationError):
        pop.add_product(('x', 'y'), name="xy")
    assert pop.variable_names == ('x', 'y')
    assert len(pop._pending_entries) == 1


def test_polynomial_lift_overflow_is_transactional():
    """Reject nonfinite polynomial-lift evaluations before committing."""
    pop = rz.POP()
    pop.add_variable("x", [1e200, -1e200])
    pop.add_objective([0.0, 0.0], ('x',), 1, name="cost")

    with pytest.raises(POPValidationError):
        pop.add_lift(('x',), ((2,),), name="x_squared")
    assert pop.variable_names == ('x',)
    assert len(pop._pending_entries) == 1


def test_lift_finalizes_as_a_sampled_identity():
    """Carry a lift identity through compaction and CSP construction."""
    pop = make_small_pop()
    pop.add_product(
        ('x', 'y'),
        name="xy",
        targets={"cost": 1},
    )
    pop.finalize()

    assert pop._sampled_identities.tolist() == [False, True]
    assert pop.stats['num_sampled_identities'] == 1
    assert pop.csp.cliques == (('x', 'y', 'xy'),)
    assert pop._entry_scope_names(1) == ('x', 'y', 'xy')


def test_lift_save_and_reload_preserves_mathematical_data(tmp_path):
    """Preserve lifted variables and identities across .pop storage."""
    x_evals = np.array([-1.0, 0.0, 1.0])
    y_evals = np.array([2.0, 3.0, 4.0])
    lift_evals = 2.0*x_evals*y_evals - x_evals**2
    pop = rz.POP()
    pop.add_variable("x", x_evals)
    pop.add_variable("y", y_evals)
    pop.add_objective(
        lift_evals,
        ('x', 'y'),
        2,
        name="cost",
    )
    pop.add_lift(
        ('x', 'y'),
        ((1, 1), (2, 0)),
        name="lift",
        coeffs=(2.0, -1.0),
        targets={"cost": 1},
    )
    pop.finalize()
    path = tmp_path/'lifted.pop'
    pop.save(path)

    loaded = rz.POP(path)
    loaded.finalize()
    assert loaded.variable_names == pop.variable_names
    assert loaded._entry_relations == pop._entry_relations
    np.testing.assert_array_equal(loaded._entry_degrees, pop._entry_degrees)
    np.testing.assert_allclose(loaded._variable_evals, pop._variable_evals)
    np.testing.assert_allclose(loaded._entry_evals, pop._entry_evals)
    assert tuple(
        loaded._entry_scope_names(entry_id)
        for entry_id in range(len(loaded._entry_relations))
    ) == tuple(
        pop._entry_scope_names(entry_id)
        for entry_id in range(len(pop._entry_relations))
    )


def test_product_rewrites_multiple_targets_with_remaining_scope():
    """Rewrite several targets while preserving unrelated variables."""
    x_evals = np.array([-1.0, 0.0, 1.0])
    y_evals = np.array([2.0, 3.0, 4.0])
    z_evals = np.array([5.0, 6.0, 7.0])
    product_evals = x_evals*y_evals

    pop = rz.POP()
    pop.add_variable("x", x_evals)
    pop.add_variable("y", y_evals)
    pop.add_variable("z", z_evals)
    pop.add_objective(
        product_evals,
        ('x', 'y'),
        2,
        name="product_cost",
    )
    pop.add_objective(
        z_evals*product_evals,
        ('x', 'y', 'z'),
        3,
        name="scaled_cost",
    )
    pop.add_product(
        ('x', 'y'),
        name="xy",
        targets={
            "product_cost": 1,
            "scaled_cost": 2,
        },
    )

    targets = {
        entry['name']: entry
        for entry in pop._pending_entries
        if entry['name'] is not None
    }
    assert targets['product_cost']['scope'] == ('xy',)
    assert targets['product_cost']['degree'] == 1
    assert targets['scaled_cost']['scope'] == ('z', 'xy')
    assert targets['scaled_cost']['degree'] == 2


@pytest.mark.parametrize('helper', ('PRODUCT', 'LIFT'))
def test_lift_helpers_reject_empty_source_scopes(helper):
    """Require at least one source variable for every lift helper."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError):
        if helper == 'PRODUCT':
            pop.add_product((), name="empty")
        else:
            pop.add_lift((), np.empty((1, 0), dtype=int), name="empty")


def test_default_product_name_collision_is_transactional():
    """Reject a generated product name already used by another variable."""
    pop = make_small_pop()
    pop.add_variable("xy", [0.0, 0.0, 0.0])
    variables_before = pop.variable_names
    entries_before = len(pop._pending_entries)

    with pytest.raises(POPValidationError):
        pop.add_product(('x', 'y'))

    assert pop.variable_names == variables_before
    assert len(pop._pending_entries) == entries_before


################################################################################
# File context and compact-degree boundaries
################################################################################


def test_lift_requires_every_declared_source_variable_to_appear():
    """Reject inactive lift variables before they enlarge the CSP."""
    pop = make_small_pop()
    variables_before = pop.variable_names
    entries_before = len(pop._pending_entries)

    with pytest.raises(POPValidationError, match='nonzero monomial'):
        pop.add_lift(
            ('x', 'y'),
            ((1, 0), (2, 0)),
            name="x_only",
        )

    assert pop.variable_names == variables_before
    assert len(pop._pending_entries) == entries_before


def test_lift_ignores_zero_coefficient_rows_when_finding_polynomial_support():
    """Require scope activity in the nonzero polynomial entries only."""
    pop = make_small_pop()
    with pytest.raises(POPValidationError, match='nonzero monomial'):
        pop.add_lift(
            ('x', 'y'),
            ((1, 0), (0, 4)),
            name="x_only_after_coefficients",
            coeffs=(1.0, 0.0),
        )


def test_lift_ignores_zero_coefficient_rows_when_resolving_degree():
    """Exclude zero-coefficient monomials from the defining identity degree."""
    pop = make_small_pop()
    pop.add_lift(
        ('x', 'y'),
        ((1, 0), (0, 1), (100, 100)),
        name="linear_sum",
        coeffs=(1.0, 1.0, 0.0),
    )

    identity = pop._pending_entries[-1]
    assert identity['degree'] == 1
    np.testing.assert_allclose(
        pop._pending_variables['linear_sum'],
        pop._pending_variables['x'] + pop._pending_variables['y'],
    )


def test_declared_degrees_must_fit_compact_storage():
    """Reject degrees that cannot be represented by finalized POP arrays."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0])
    too_large = np.iinfo(np.int32).max + 1

    with pytest.raises(POPValidationError, match='compact integer range'):
        pop.add_objective([0.0, 1.0], ('x',), too_large)


def test_lift_total_degree_must_fit_compact_storage():
    """Reject a defining lift degree above the finalized integer range."""
    pop = rz.POP()
    pop.add_variable("x", [1.0, 1.0])
    pop.add_objective([1.0, 1.0], ('x',), 1)

    with pytest.raises(POPValidationError, match='compact integer range'):
        pop.add_lift(
            ('x',),
            ((np.iinfo(np.int32).max + 1,),),
            name="huge_power",
        )


def test_loaded_semantic_errors_include_path_and_line(tmp_path):
    """Qualify row-specific semantic failures with their source location."""
    path = tmp_path/'duplicate.pop'
    path.write_text(
        'var,1,x,0,1\n'
        'var,1,x,0,1\n'
        'min,1,x,0,1\n'
    )

    with pytest.raises(POPParseError) as captured:
        rz.POP(path)
    message = str(captured.value)
    assert str(path) in message
    assert 'line 2' in message


def test_loaded_finalization_errors_include_path_and_line(tmp_path):
    """Retain source context for validation deferred until finalization."""
    path = tmp_path/'unknown.pop'
    path.write_text(
        'var,1,x,0,1\n'
        'min,1,y,0,1\n'
    )
    pop = rz.POP(path)

    with pytest.raises(POPParseError) as captured:
        pop.finalize()
    message = str(captured.value)
    assert str(path) in message
    assert 'line 2' in message



def test_loaded_global_finalization_errors_include_path(tmp_path):
    """Qualify file-wide semantic failures with the source path."""
    path = tmp_path/'no_objective.pop'
    path.write_text(
        'var,1,x,0,1\n'
        '==0,1,x,0,0\n'
    )
    pop = rz.POP(path)

    with pytest.raises(POPParseError) as captured:
        pop.finalize()
    assert str(path) in str(captured.value)


def test_pop_public_reports_are_json_serializable():
    """Expose ordinary Python scalar leaves before and after finalization."""
    pop = make_small_pop()
    json.dumps(pop.status)
    json.dumps(pop.stats)
    pop.finalize()
    json.dumps(pop.status)
    json.dumps(pop.stats)
    json.dumps(pop.csp.status)
    json.dumps(pop.csp.stats)
