"""Sampled polynomial optimization problem construction and persistence.

POP is the public description layer from which correlative structure and the
semidefinite relaxation recipe are derived.
"""

from collections import OrderedDict
import numpy as np

from rhiza.csp import CSP
from rhiza.diagnostics import (
    GraphPolicy,
    PipelineError,
    POPParseError,
    POPValidationError,
    Relation,
    format_summary,
    to_python_report,
)
from rhiza.tools.parsing import (
    ParsedRow,
    cast_exact_int,
    cast_real_array,
    cast_real_scalar,
    iter_pop_rows,
    write_pop_rows,
)
from rhiza.tools.polyalg import evaluate_lift


_MAX_DECLARED_DEGREE = int(np.iinfo(np.int32).max)


################################################################################
# Sampled polynomial optimization problem
################################################################################


class POP:
    """Store and finalize a sampled polynomial optimization problem. It is the
    public builder for variables, sampled objectives, constraints, and optional
    lifted coordinates before graph analysis freezes the model."""

    def __init__(self, path=None):
        """Create an empty mutable builder or load rows from a .pop file. Loaded
        rows pass through the same validation path as programmatic construction.

        Inputs:
            path:
                String or None. File to load.
        """
        path = None if path is None else str(path)

        # Builder state keeps user arrays in simple named containers. Finalize
        # later replaces these with compact matrices and integer indexes.
        self._finalized = False
        self._pending_variables = OrderedDict()
        self._variable_source_lines = {}
        self._pending_entries = []
        self._named_entries = {}
        self._names = {}
        self._sample_count = 0
        self._csp = None
        self._graph_policy = None
        self._identity_tol = None
        self._source_path = path
        self._status = {
            'finalized': False,
        }
        self._stats = {
            'num_samples': 0,
            'num_variables': 0,
            'num_entries': 0,
        }
        if path is not None:
            # File loading deliberately reuses the public builder validation path
            # so in-memory and serialized POPs obey the same invariants.
            self._load(path)

    @property
    def finalized(self):
        """Return whether construction has frozen the sampled problem."""
        return self._finalized

    @property
    def csp(self):
        """Return the correlative sparsity pattern derived at finalization."""
        if not self._finalized:
            raise PipelineError("Finalize the POP before accessing its CSP.")
        return self._csp

    @property
    def status(self):
        """Return a copy of qualitative POP status fields."""
        return to_python_report(self._status)

    @property
    def stats(self):
        """Return a copy of numerical POP statistics."""
        return to_python_report(self._stats)

    @property
    def num_samples(self):
        """Return the common number of sampled points."""
        if self._finalized:
            return self._variable_evals.shape[0]
        return self._sample_count

    @property
    def variable_names(self):
        """Return variable names in their stable storage order."""
        if self._finalized:
            return self._variable_names
        return tuple(self._pending_variables)

    @property
    def entry_names(self):
        """Return names assigned to objective and constraint entries."""
        if self._finalized:
            return tuple(name for name in self._entry_names if name is not None)
        return tuple(
            entry['name']
            for entry in self._pending_entries
            if entry['name'] is not None
        )

    ################################################################################
    # Construction
    ################################################################################

    def add_variable(self, name, evals):
        """Add one sampled scalar variable.

        Inputs:
            name:
                String. Variable name.
            evals:
                ArrayLike. Variable evaluations at every sample.

        Outputs:
            added:
                Boolean. True after insertion.
        """
        name = _cast_name(name, label="Variable name")
        evals = _cast_evals(evals)

        # Complete every check before registering the name or copying data into
        # builder state, preserving transactional mutation semantics.
        self._ensure_building()
        self._validate_name(name)
        self._validate_evals(evals)
        self._validate_sample_count(evals)
        self._register_name(name, 'VARIABLE')
        self._pending_variables[name] = evals.copy()
        self._variable_source_lines[name] = None
        self._establish_sample_count(evals)
        self._refresh_building_stats()
        return True

    def add_objective(
        self,
        evals,
        scope,
        degree,
        *,
        form='MIN',
        name=None,
    ):
        """Add one sampled objective entry.

        Inputs:
            evals:
                ArrayLike. Objective evaluations at every sample.
            scope:
                ScopeLike. Variables used by the objective.
            degree:
                Integer. Declared polynomial degree.
            form:
                String. Either 'MIN' or 'MAX'.
            name:
                String or None. Optional entry name.

        Outputs:
            added:
                Boolean. True after insertion.
        """
        evals = _cast_evals(evals)
        scope = _cast_scope(scope)
        degree = cast_exact_int(
            degree, "Objective degree", POPValidationError
        )
        form = str(form).upper()
        name = _cast_optional_name(name, label="Objective name")

        # Normalize objective direction to an internal Relation token; SDPR
        # later converts MAX entries to the fixed minimization sign convention.
        if form not in ('MIN', 'MAX'):
            raise POPValidationError("Objective form must be 'MIN' or 'MAX'.")
        relation = Relation.MIN if form == 'MIN' else Relation.MAX
        self._add_entry(relation, evals, scope, degree, name=name)
        return True

    def add_constraint(
        self,
        evals,
        scope,
        degree,
        *,
        form,
        name=None,
    ):
        """Add one sampled equality or inequality constraint.

        Inputs:
            evals:
                ArrayLike. Constraint evaluations at every sample.
            scope:
                ScopeLike. Variables used by the constraint.
            degree:
                Integer. Declared polynomial degree.
            form:
                String. One of '>=0', '<=0', or '==0'.
            name:
                String or None. Optional entry name.

        Outputs:
            added:
                Boolean. True after insertion.
        """
        evals = _cast_evals(evals)
        scope = _cast_scope(scope)
        degree = cast_exact_int(
            degree, "Constraint degree", POPValidationError
        )
        form = str(form)
        name = _cast_optional_name(name, label="Constraint name")

        # Public symbolic forms map to controlled internal relations before the
        # common entry-validation path is entered.
        relations = {
            '>=0': Relation.GE,
            '<=0': Relation.LE,
            '==0': Relation.EQ,
        }
        if form not in relations:
            raise POPValidationError(
                "Constraint form must be '>=0', '<=0', or '==0'."
            )
        self._add_entry(
            relations[form], evals, scope, degree, name=name
        )
        return True

    def add_product(self, scope, *, name=None, targets=None):
        """Add a lifted product variable and its defining sampled identity. The
        mutation is validated transactionally so a failed request leaves the
        sampled problem unchanged.

        Inputs:
            scope:
                ScopeLike. Variables multiplied once each.
            name:
                String or None. Lifted variable name.
            targets:
                Mapping or None. Named entries mapped to new degrees.

        Outputs:
            added:
                Boolean. True after insertion.
        """
        scope = _cast_scope(scope)
        name = _cast_optional_name(name, label="Product name")
        targets = _cast_targets(targets)

        # Validate the source, new name, and every requested target rewrite
        # before evaluating or committing the lifted coordinate.
        self._ensure_building()
        self._validate_existing_scope(scope)
        name = self._resolve_product_name(scope, name)
        self._validate_name(name)
        target_updates = self._prepare_target_updates(name, scope, targets)

        # The product is evaluated directly on the shared sample rows. Floating
        # overflow is converted into the ordinary finite-evaluation validation error.
        source_evals = np.column_stack(
            [self._pending_variables[variable] for variable in scope]
        )
        with np.errstate(over='ignore', invalid='ignore'):
            evals = np.prod(source_evals, axis=1)
        self._validate_evals(evals)
        # The literal defining polynomial is product(scope) - lift, hence its
        # declared degree is the larger of the source monomial and the
        # linear lifted-coordinate term.
        degree = max(1, len(scope))
        self._commit_lift(
            name,
            evals,
            tuple(scope) + (name,),
            degree,
            target_updates,
        )
        return True

    def add_lift(
        self,
        scope,
        exponents,
        *,
        name,
        coeffs=None,
        targets=None,
    ):
        """Add a polynomial lift and its defining sampled identity. The mutation
        is validated transactionally so a failed request leaves the sampled
        problem unchanged.

        Inputs:
            scope:
                ScopeLike. Source variables.
            exponents:
                ArrayLike. Monomial exponent rows.
            name:
                String. Lifted variable name.
            coeffs:
                ArrayLike or None. Monomial coefficients.
            targets:
                Mapping or None. Named entries mapped to new degrees.

        Outputs:
            added:
                Boolean. True after insertion.
        """
        scope = _cast_scope(scope)
        exponents = _cast_exponents(exponents)
        name = _cast_name(name, label="Lift name")
        coeffs = None if coeffs is None else _cast_coeffs(coeffs)
        targets = _cast_targets(targets)

        # Shape and coefficient checks precede all builder mutation. This also
        # ensures target rewrites can be prepared against the unchanged problem.
        self._ensure_building()
        self._validate_existing_scope(scope)
        self._validate_name(name)
        if exponents.ndim != 2 or exponents.shape[1] != len(scope):
            raise POPValidationError(
                "Lift exponents must have one column per source variable."
            )
        if exponents.shape[0] == 0:
            raise POPValidationError("A lift requires at least one monomial.")
        if np.any(exponents < 0):
            raise POPValidationError("Lift exponents must be nonnegative.")
        if coeffs is None:
            coeffs = np.ones(exponents.shape[0], dtype=float)
        if coeffs.shape != (exponents.shape[0],):
            raise POPValidationError(
                "Lift coeffs must have one entry per exponent row."
            )
        if not np.all(np.isfinite(coeffs)):
            raise POPValidationError("Lift coeffs must be finite.")

        # Every declared source variable must participate in a nonzero
        # monomial term.
        # Otherwise the defining identity would enlarge the correlative graph
        # even though the polynomial is independent of that variable.
        participating_rows = coeffs != 0.0
        nonparticipants = tuple(
            scope[index]
            for index in range(len(scope))
            if not np.any(exponents[participating_rows, index] > 0)
        )
        if nonparticipants:
            raise POPValidationError(
                "Every lift scope variable must have a positive exponent in "
                "at least one nonzero monomial: "
                + ', '.join(nonparticipants)
            )

        # Zero-coefficient rows do not belong to the polynomial support. Remove
        # them before evaluation and degree calculation so they cannot create
        # artificial coupling or inflate the defining identity degree.
        exponents = exponents[participating_rows, :]
        coeffs = coeffs[participating_rows]

        # Target rewrites are validated as a batch before the new sampled
        # coordinate is evaluated and checked for finite values.
        target_updates = self._prepare_target_updates(name, scope, targets)
        source_evals = np.column_stack(
            [self._pending_variables[variable] for variable in scope]
        )
        with np.errstate(over='ignore', invalid='ignore'):
            evals = evaluate_lift(source_evals, exponents, coeffs)
        self._validate_evals(evals)
        # The defining equality includes the new variable and all source
        # variables. Its degree covers both the highest source monomial and lift.
        identity_scope = tuple(scope) + (name,)

        # Sum with Python integers rather than a fixed-width NumPy accumulator.
        # Very large but valid exponents must not wrap while determining the
        # defining equality's declared total degree.
        source_degree = max(
            sum(int(exponent) for exponent in exponent_row)
            for exponent_row in exponents
        )
        degree = max(1, source_degree)
        self._commit_lift(
            name,
            evals,
            identity_scope,
            degree,
            target_updates,
        )
        return True

    def remove(self, name):
        """Remove one named objective or constraint entry. The mutation is validated
        transactionally so a failed request leaves the sampled problem
        unchanged.

        Inputs:
            name:
                String. Entry name.

        Outputs:
            removed:
                Boolean. True if an entry was removed.
        """
        name = _cast_name(name, label="Entry name")

        self._ensure_building()
        # The global namespace distinguishes removable named entries from
        # variables, whose removal would invalidate arbitrary declared scopes.
        kind = self._names.get(name)
        if kind is None:
            return False
        if kind == 'VARIABLE':
            raise POPValidationError("Variables cannot be removed.")

        for index, entry in enumerate(self._pending_entries):
            if entry['name'] == name:
                del self._pending_entries[index]
                del self._names[name]
                del self._named_entries[name]
                self._refresh_building_stats()
                return True
        return False

    ################################################################################
    # Finalization and storage
    ################################################################################

    def finalize(self, *, graph_policy='RIP', identity_tol=None):
        """Validate, compact, and finalize the POP in one transaction.

        Inputs:
            graph_policy:
                String or GraphPolicy. CSP construction policy.
            identity_tol:
                Float or None. Absolute sampled-identity tolerance.

        Outputs:
            changed:
                Boolean. True on the first successful finalization.
        """
        graph_policy = GraphPolicy.from_value(graph_policy)
        identity_tol = self._resolve_identity_tol(identity_tol)

        if self._finalized:
            # Repeating identical finalization is harmless, but changing graph or
            # identity policy after compaction would invalidate derived data.
            same_settings = (
                graph_policy is self._graph_policy
                and identity_tol == self._identity_tol
            )
            if same_settings:
                return False
            raise PipelineError(
                "A finalized POP cannot change its finalization settings."
            )

        # Prepare every finalized object before changing builder state.
        validation = self._validate_for_finalize(identity_tol)
        entry_scopes = tuple(entry['scope'] for entry in self._pending_entries)
        csp = CSP(
            tuple(self._pending_variables),
            entry_scopes,
            graph_policy=graph_policy,
        )
        compact = self._prepare_compact_data(identity_tol)

        # Commit only after every prepared object succeeds.
        self._commit_finalization(
            compact,
            csp,
            graph_policy,
            identity_tol,
            validation,
        )
        return True

    def save(self, path):
        """Write a finalized POP in the readable .pop exchange format.

        Inputs:
            path:
                String. Output path.

        Outputs:
            saved:
                Boolean. True after writing.
        """
        path = str(path)

        if not self._finalized:
            raise PipelineError("Finalize the POP before saving it.")
        return write_pop_rows(path, self._iter_serializable_rows())

    ################################################################################
    # Inspection
    ################################################################################

    def __str__(self):
        """Return a readable POP summary. The summary is meant for interactive
        inspection and therefore uses the same status and statistics fields
        exposed programmatically."""
        return format_summary('Rhiza POP', self._status, self._stats)

    def __repr__(self):
        """Return a concise developer representation. The representation
        highlights identity and lifecycle information without expanding large
        sampled arrays."""
        return (
            f"POP(num_samples={self.num_samples}, "
            f"num_variables={len(self.variable_names)}, "
            f"num_entries={self._stats['num_entries']}, "
            f"finalized={self._finalized})"
        )

    ################################################################################
    # Construction internals
    ################################################################################

    def _load(self, path):
        """Load .pop rows into mutable builder state.

        Row-specific semantic failures are translated to POPParseError with the
        source path and physical line, matching lexical parser diagnostics.

        Inputs:
            path:
                String. Source .pop path.
        """
        for row in iter_pop_rows(path):
            try:
                # Variable rows establish sampled coordinates; every other
                # relation enters the common objective-and-constraint path.
                if row.relation is Relation.VAR:
                    if row.degree != 1 or len(row.scope) != 1:
                        raise POPValidationError(
                            "Variable rows require degree 1 and a singleton "
                            "scope."
                        )
                    self.add_variable(row.scope[0], row.evals)
                    self._variable_source_lines[row.scope[0]] = row.line_number
                else:
                    self._add_entry(
                        row.relation,
                        row.evals,
                        row.scope,
                        row.degree,
                        name=None,
                        source_line=row.line_number,
                    )
            except POPValidationError as error:
                raise self._parse_error(error, row.line_number) from error

    def _add_entry(
        self,
        relation,
        evals,
        scope,
        degree,
        *,
        name=None,
        source_line=None,
    ):
        """Add one validated objective or constraint entry to builder state.

        Inputs:
            relation:
                Relation. Objective or constraint relation.
            evals:
                ndarray. Sampled polynomial evaluations.
            scope:
                tuple. Declared variable scope.
            degree:
                Integer. Declared polynomial degree.
            name:
                String or None. Optional entry name.
            source_line:
                Integer or None. Source .pop line.
        """
        # Validate the complete normalized entry before reserving its optional
        # public name or appending it to the ordered entry list.
        self._ensure_building()
        self._validate_evals(evals)
        self._validate_sample_count(evals)
        self._validate_scope(scope, require_existing=False)
        self._validate_degree(degree)
        if relation is Relation.VAR:
            raise POPValidationError("Variable rows are not POP entries.")
        if name is not None:
            self._validate_name(name)
            self._register_name(name, 'ENTRY')

        # Preserve declaration order because CSP entry hosts and finalized
        # compact arrays use this stable entry-ID ordering.
        entry = {
            'relation': relation,
            'degree': degree,
            'scope': tuple(scope),
            'evals': evals.copy(),
            'name': name,
            'source_line': source_line,
        }
        self._pending_entries.append(entry)
        if name is not None:
            self._named_entries[name] = entry
        self._establish_sample_count(evals)
        self._refresh_building_stats()

    def _prepare_target_updates(self, lift_name, source_scope, targets):
        """Prepare all lift target rewrites without mutating their entries.

        Inputs:
            lift_name:
                String. New lifted variable name.
            source_scope:
                tuple. Variables replaced in each target.
            targets:
                Mapping. Named targets and new degrees.

        Outputs:
            updates:
                list. Validated target mutation records.
        """
        # Most lifts do not rewrite an existing entry. Returning before any
        # entry traversal keeps each ordinary product or lift declaration
        # independent of the number of entries already present in the builder.
        if not targets:
            return []

        # Named entries are indexed as they are declared, so target validation
        # depends only on the number of requested rewrites rather than on the
        # total number of objectives and constraints in the POP.
        updates = []
        source_set = set(source_scope)
        for target_name, degree in targets.items():
            self._validate_degree(degree)
            if target_name not in self._named_entries:
                raise POPValidationError(
                    "Unknown named lift target: " + repr(target_name)
                )
            entry = self._named_entries[target_name]
            if lift_name in entry['scope']:
                raise POPValidationError(
                    "A lift target already contains the new variable: "
                    + repr(target_name)
                )
            if not source_set.issubset(entry['scope']):
                raise POPValidationError(
                    "Lift source scope is not contained in target "
                    + repr(target_name)
                )
            # Replace the complete source scope by the single lifted variable;
            # unrelated target variables retain their original order.
            new_scope = tuple(
                variable
                for variable in entry['scope']
                if variable not in source_set
            ) + (lift_name,)
            updates.append((entry, new_scope, degree))
        return updates

    def _commit_lift(
        self,
        name,
        evals,
        identity_scope,
        degree,
        target_updates,
    ):
        """Commit a validated lift, its identity, and target rewrites.

        Inputs:
            name:
                String. Lifted variable name.
            evals:
                ndarray. Lifted variable evaluations.
            identity_scope:
                tuple. Defining equality scope.
            degree:
                Integer. Defining equality degree.
            target_updates:
                Sequence. Validated target mutations.
        """
        # Commit the lifted coordinate and its literal zero sampled identity as
        # one builder operation after all upstream validation has succeeded.
        self._validate_degree(degree)
        self._register_name(name, 'VARIABLE')
        self._pending_variables[name] = evals.copy()
        self._variable_source_lines[name] = None
        self._pending_entries.append({
            'relation': Relation.EQ,
            'degree': degree,
            'scope': identity_scope,
            'evals': np.zeros_like(evals, dtype=float),
            'name': None,
            'source_line': None,
        })
        # Apply the already prepared target rewrites only after the new variable
        # and defining equality are present.
        for entry, new_scope, new_degree in target_updates:
            entry['scope'] = new_scope
            entry['degree'] = new_degree
        self._establish_sample_count(evals)
        self._refresh_building_stats()

    ################################################################################
    # Finalization internals
    ################################################################################

    def _validate_for_finalize(self, identity_tol):
        """Validate the complete mutable POP before graph construction.

        Inputs:
            identity_tol:
                Float. Sampled-identity tolerance.

        Outputs:
            report:
                Report. Validation counts used during finalization.
        """
        # Finalized problems need at least one sampled coordinate, one entry, and
        # one objective before graph construction can be meaningful.
        if not self._pending_variables:
            raise self._validation_error(
                "A POP requires at least one variable."
            )
        if not self._pending_entries:
            raise self._validation_error("A POP requires at least one entry.")
        if not any(
            entry['relation'] in (Relation.MIN, Relation.MAX)
            for entry in self._pending_entries
        ):
            raise self._validation_error(
                "A POP requires at least one objective."
            )

        # Audit all scopes and classify equalities that already vanish on the
        # supplied sample set under the explicit absolute identity tolerance.
        variable_names = set(self._pending_variables)
        used_variables = set()
        num_sampled_identities = 0
        for entry in self._pending_entries:
            try:
                self._validate_scope(entry['scope'], require_existing=True)
            except POPValidationError as error:
                if entry['source_line'] is not None:
                    raise self._parse_error(
                        error, entry['source_line']
                    ) from error
                raise
            used_variables.update(entry['scope'])
            if (
                entry['relation'] in (Relation.GE, Relation.LE)
                and not np.any(entry['evals'] != 0.0)
            ):
                name = (
                    " " + repr(entry['name'])
                    if entry['name'] is not None else ""
                )
                raise self._validation_error(
                    "Inequality" + name + " evaluates to zero at every "
                    "sample. Supply samples that include at least one point "
                    "off the constraint boundary.",
                    line_number=entry['source_line'],
                )
            is_sampled_identity = (
                entry['relation'] is Relation.EQ
                and np.max(np.abs(entry['evals'])) <= identity_tol
            )
            if is_sampled_identity:
                num_sampled_identities += 1
        # Scope validation above already rejects unknown names. This final check
        # therefore identifies only declared coordinates that no entry uses.
        unused = sorted(variable_names - used_variables)
        if unused:
            source_line = self._variable_source_lines.get(unused[0])
            raise self._validation_error(
                "Variables are unused by every entry: " + ', '.join(unused),
                line_number=source_line,
            )
        return {
            'num_sampled_identities': num_sampled_identities,
        }

    def _prepare_compact_data(self, identity_tol):
        """Build compact finalized arrays without mutating builder state.

        Inputs:
            identity_tol:
                Float. Sampled-identity tolerance.

        Outputs:
            compact:
                Mapping. Prepared compact arrays and indexes.
        """
        # Preserve builder insertion order while converting named variable
        # vectors into one dense sample-by-variable matrix.
        variable_names = tuple(self._pending_variables)
        variable_ids = {
            name: variable_id
            for variable_id, name in enumerate(variable_names)
        }
        variable_evals = np.column_stack(
            [self._pending_variables[name] for name in variable_names]
        )

        # Parallel compact arrays make entry traversal inexpensive in SDPR while
        # retaining optional names for in-memory reports and entry lookup.
        entry_relations = tuple(
            entry['relation'] for entry in self._pending_entries
        )
        entry_degrees = np.asarray(
            [entry['degree'] for entry in self._pending_entries], dtype=np.int32
        )
        entry_evals = np.column_stack(
            [entry['evals'] for entry in self._pending_entries]
        )
        entry_names = tuple(entry['name'] for entry in self._pending_entries)
        sampled_identities = np.asarray([
            entry['relation'] is Relation.EQ
            and np.max(np.abs(entry['evals'])) <= identity_tol
            for entry in self._pending_entries
        ], dtype=bool)

        # Flatten variable-length scopes into one integer array with offsets;
        # this avoids one Python container per entry after finalization.
        scope_offsets = [0]
        scope_variable_ids = []
        for entry in self._pending_entries:
            scope_variable_ids.extend(
                sorted(variable_ids[name] for name in entry['scope'])
            )
            scope_offsets.append(len(scope_variable_ids))

        return {
            'variable_names': variable_names,
            'variable_ids': variable_ids,
            'variable_evals': np.asarray(variable_evals, dtype=float),
            'entry_relations': entry_relations,
            'entry_degrees': entry_degrees,
            'entry_evals': np.asarray(entry_evals, dtype=float),
            'entry_names': entry_names,
            'sampled_identities': sampled_identities,
            'scope_offsets': np.asarray(scope_offsets, dtype=np.int32),
            'scope_variable_ids': np.asarray(
                scope_variable_ids, dtype=np.int32
            ),
        }

    def _commit_finalization(
        self,
        compact,
        csp,
        graph_policy,
        identity_tol,
        validation,
    ):
        """Install prepared compact data and lock the POP.

        Inputs:
            compact:
                Mapping. Prepared compact arrays and indexes.
            csp:
                CSP. Constructed correlative sparsity pattern.
            graph_policy:
                GraphPolicy. Applied graph policy.
            identity_tol:
                Float. Sampled-identity tolerance.
            validation:
                Report. Finalization validation counts.
        """
        # Install all compact arrays and derived objects before flipping the
        # lifecycle flag visible to callers.
        self._variable_names = compact['variable_names']
        self._variable_ids = compact['variable_ids']
        self._variable_evals = compact['variable_evals']
        self._entry_relations = compact['entry_relations']
        self._entry_degrees = compact['entry_degrees']
        self._entry_evals = compact['entry_evals']
        self._entry_names = compact['entry_names']
        self._sampled_identities = compact['sampled_identities']
        self._scope_offsets = compact['scope_offsets']
        self._scope_variable_ids = compact['scope_variable_ids']
        self._csp = csp
        self._graph_policy = graph_policy
        self._identity_tol = identity_tol
        self._finalized = True

        # Publish summary fields from the committed arrays rather than from
        # mutable builder containers.
        self._status = {
            'finalized': True,
            'graph_policy': graph_policy.value,
        }
        self._stats = {
            'num_samples': self._variable_evals.shape[0],
            'num_variables': self._variable_evals.shape[1],
            'num_entries': self._entry_evals.shape[1],
            'num_objectives': sum(
                relation in (Relation.MIN, Relation.MAX)
                for relation in self._entry_relations
            ),
            'num_constraints': sum(
                relation in (Relation.EQ, Relation.GE, Relation.LE)
                for relation in self._entry_relations
            ),
            'num_sampled_identities': validation['num_sampled_identities'],
            'num_cliques': csp.stats['num_cliques'],
        }
        # Finalized access uses the compact arrays, so release the duplicate
        # builder containers after compaction.
        del self._pending_variables
        del self._variable_source_lines
        del self._pending_entries
        del self._named_entries
        del self._names

    def _iter_serializable_rows(self):
        """Yield finalized data in canonical .pop row order. The iterator
        preserves the mathematical ordering while bounding temporary memory
        during downstream assembly.

        Outputs:
            rows:
                Iterator. ParsedRow objects for serialization.
        """
        # The canonical format writes all variable rows first so later entry
        # scopes are interpreted against an established namespace.
        for variable_id, name in enumerate(self._variable_names):
            yield ParsedRow(
                Relation.VAR,
                1,
                (name,),
                self._variable_evals[:,variable_id],
            )
        # Objective and constraint names are intentionally not serialized; the
        # row format preserves only mathematical relation, degree, scope, and data.
        for entry_id, relation in enumerate(self._entry_relations):
            yield ParsedRow(
                relation,
                self._entry_degrees[entry_id],
                self._entry_scope_names(entry_id),
                self._entry_evals[:,entry_id],
            )

    ################################################################################
    # Validation helpers
    ################################################################################

    def _ensure_building(self):
        """Require the POP to remain in its mutable builder state."""
        if self._finalized:
            raise PipelineError("A finalized POP cannot be modified.")

    def _register_name(self, name, kind):
        """Register one globally unique variable or entry name.

        Inputs:
            name:
                String. Name to register.
            kind:
                String. Named object category.
        """
        if name in self._names:
            raise POPValidationError(
                "A POP name is already in use: " + repr(name)
            )
        self._names[name] = kind

    def _validate_name(self, name):
        """Validate one user-facing POP name.

        Inputs:
            name:
                String. Name to validate.
        """
        if not name:
            raise POPValidationError("POP names must be nonempty.")
        if name != name.strip():
            raise POPValidationError(
                "POP names cannot have leading or trailing whitespace."
            )
        if ',' in name or '-' in name or '\n' in name or '\r' in name:
            raise POPValidationError(
                "POP names cannot contain commas, hyphens, or newlines."
            )
        if name in self._names:
            raise POPValidationError(
                "A POP name is already in use: " + repr(name)
            )

    def _validate_scope(self, scope, require_existing):
        """Validate one declared variable scope.

        Inputs:
            scope:
                tuple. Variable names to validate.
            require_existing:
                Boolean. Whether every variable must already exist.
        """
        if not scope:
            raise POPValidationError("A polynomial scope must be nonempty.")
        if len(set(scope)) != len(scope):
            raise POPValidationError("A polynomial scope repeats a variable.")
        for name in scope:
            self._validate_scope_name(name)
            if require_existing and name not in self._pending_variables:
                raise POPValidationError(
                    "A polynomial scope contains an unknown variable: "
                    + repr(name)
                )

    def _validate_existing_scope(self, scope):
        """Validate a nonempty scope of existing source variables.

        Inputs:
            scope:
                tuple. Source variable names.
        """
        self._validate_scope(scope, require_existing=True)

    def _validate_scope_name(self, name):
        """Validate one variable name appearing in a scope.

        Inputs:
            name:
                String. Variable name to validate.
        """
        if not name:
            raise POPValidationError("Variable names must be nonempty.")
        if name != name.strip():
            raise POPValidationError(
                "Variable names cannot have leading or trailing whitespace."
            )
        if ',' in name or '-' in name or '\n' in name or '\r' in name:
            raise POPValidationError(
                "Variable names cannot contain commas, hyphens, or newlines."
            )

    def _validate_evals(self, evals):
        """Validate one sampled evaluation vector.

        Inputs:
            evals:
                ndarray. Sampled evaluations.
        """
        if evals.ndim != 1:
            raise POPValidationError("Sampled evals must be one-dimensional.")
        if evals.size == 0:
            raise POPValidationError("Sampled evals must be nonempty.")
        if not np.all(np.isfinite(evals)):
            raise POPValidationError("Sampled evals must be finite.")

    def _validate_sample_count(self, evals):
        """Require an evaluation vector to match the established sample count.

        Inputs:
            evals:
                ndarray. Sampled evaluations to compare.
        """
        count = self._sample_count
        if count and len(evals) != count:
            raise POPValidationError(
                f"Expected {count} evals, received {len(evals)}."
            )

    def _establish_sample_count(self, evals):
        """Record the common row count after the first successful insertion.

        The count is an invariant of the mutable POP. Storing it directly
        avoids repeatedly scanning every previously declared variable and entry
        while large sampled models are assembled.

        Inputs:
            evals:
                ndarray. Successfully inserted sampled evaluations.
        """
        if self._sample_count == 0:
            self._sample_count = len(evals)

    def _validate_degree(self, degree):
        """Validate one declared polynomial degree.

        Inputs:
            degree:
                Integer. Degree to validate.
        """
        if degree < 0:
            raise POPValidationError("Polynomial degrees must be nonnegative.")
        if degree > _MAX_DECLARED_DEGREE:
            raise POPValidationError(
                "Polynomial degrees exceed the supported compact integer range."
            )

    def _resolve_identity_tol(self, identity_tol):
        """Resolve the tolerance used to recognize sampled identities.

        Inputs:
            identity_tol:
                Float or None. Requested tolerance.

        Outputs:
            identity_tol:
                Float. Finite nonnegative tolerance.
        """
        if identity_tol is None:
            return 64.0*np.finfo(float).eps
        identity_tol = cast_real_scalar(
            identity_tol, "identity_tol", POPValidationError
        )
        if not np.isfinite(identity_tol) or identity_tol < 0.0:
            raise POPValidationError(
                "identity_tol must be finite and nonnegative."
            )
        return identity_tol

    def _resolve_product_name(self, scope, name):
        """Return an explicit product name or the canonical concatenation.

        Inputs:
            scope:
                tuple. Product source variables.
            name:
                String or None. Requested lift name.

        Outputs:
            name:
                String. Explicit or generated lift name.
        """
        if name is not None:
            return name
        return ''.join(scope)

    def _parse_error(self, error, line_number=None):
        """Attach loaded-file context to one semantic validation failure.

        Inputs:
            error:
                Exception. Original semantic validation failure.
            line_number:
                Integer or None. Associated physical source line.

        Outputs:
            contextual:
                POPParseError. Path-qualified loaded-file failure.
        """
        source = self._source_path or "<memory>"
        location = (
            source if line_number is None
            else f"{source}, line {line_number}"
        )
        return POPParseError(f"{location}: {error}")

    def _validation_error(self, message, line_number=None):
        """Create a semantic error appropriate to the POP construction route.

        Programmatic builder failures remain POPValidationError. Failures in a
        loaded file become POPParseError and retain any available source line.

        Inputs:
            message:
                String. Validation failure description.
            line_number:
                Integer or None. Associated loaded-file line.

        Outputs:
            error:
                POPValidationError or POPParseError. Contextual failure.
        """
        error = POPValidationError(message)
        if self._source_path is None:
            return error
        return self._parse_error(error, line_number)

    def _refresh_building_stats(self):
        """Refresh cheap mutable POP statistics. Recomputing the derived fields in
        one place keeps summaries and programmatic diagnostics synchronized
        after state changes."""
        self._stats['num_samples'] = self._sample_count
        self._stats['num_variables'] = len(self._pending_variables)
        self._stats['num_entries'] = len(self._pending_entries)

    ################################################################################
    # Compact-data helpers
    ################################################################################

    def _entry_scope_ids(self, entry_id):
        """Return one finalized entry scope as variable IDs.

        Inputs:
            entry_id:
                Integer. Finalized entry identifier.

        Outputs:
            scope_ids:
                tuple. Variable identifiers in the entry scope.
        """
        start = self._scope_offsets[entry_id]
        stop = self._scope_offsets[entry_id + 1]
        return tuple(self._scope_variable_ids[start:stop])

    def _entry_scope_names(self, entry_id):
        """Return one finalized entry scope as variable names.

        Inputs:
            entry_id:
                Integer. Finalized entry identifier.

        Outputs:
            scope:
                tuple. Variable names in the entry scope.
        """
        return tuple(
            self._variable_names[variable_id]
            for variable_id in self._entry_scope_ids(entry_id)
        )


################################################################################
# Input helpers
################################################################################


def _cast_name(value, *, label):
    """Require one literal string name at the public POP boundary. Names are not
    coerced because values such as ``None`` would otherwise create surprising
    namespace entries that cannot be recovered reliably from a .pop file.

    Inputs:
        value:
            Object. Candidate name.
        label:
            String. Context used in validation errors.

    Outputs:
        name:
            String. Validated literal name.
    """
    if not isinstance(value, str):
        raise POPValidationError(label + " must be a string.")
    return value


def _cast_optional_name(value, *, label):
    """Cast an optional literal POP name without stringifying other objects. This
    keeps ``None`` as the sole sentinel for an unnamed entry or an
    automatically named product.

    Inputs:
        value:
            Object or None. Candidate optional name.
        label:
            String. Context used in validation errors.

    Outputs:
        name:
            String or None. Validated optional name.
    """
    if value is None:
        return None
    return _cast_name(value, label=label)


def _cast_scope(scope):
    """Cast one public variable scope while treating a string as one variable. A
    bare name such as ``"theta"`` must not be interpreted as an iterable of
    character names.

    Inputs:
        scope:
            ScopeLike. Candidate variable scope.

    Outputs:
        scope:
            tuple. Literal variable names in caller order.
    """
    if isinstance(scope, str):
        values = (scope,)
    else:
        try:
            values = tuple(scope)
        except TypeError as error:
            raise POPValidationError(
                "A polynomial scope must be a string or an iterable of strings."
            ) from error
    for value in values:
        if not isinstance(value, str):
            raise POPValidationError(
                "Variable names in a polynomial scope must be strings."
            )
    return values


def _cast_evals(evals):
    """Convert one public sampled-evaluation array to real float64 values.

    Shape, finiteness, and common sample count are checked separately after the
    package-wide public conversion has rejected complex-valued input.

    Inputs:
        evals:
            ArrayLike. Candidate sampled evaluations.

    Outputs:
        evals:
            ndarray. Real float64 sampled evaluations.
    """
    return cast_real_array(
        evals, "Sampled evaluations", POPValidationError
    )


def _cast_coeffs(coeffs):
    """Convert public polynomial lift coefficients to real float64 values.

    The caller subsequently validates vector shape and finiteness against the
    exponent rows.

    Inputs:
        coeffs:
            ArrayLike. Candidate lift coefficients.

    Outputs:
        coeffs:
            ndarray. Real float64 coefficient array.
    """
    return cast_real_array(
        coeffs, "Lift coefficients", POPValidationError
    )


def _cast_exponents(exponents):
    """Cast a lift exponent array without silently truncating fractional entries.
    Lift formulas determine both sampled values and declared polynomial degree, so
    every exponent must remain exactly integral across NumPy dtypes.

    Inputs:
        exponents:
            ArrayLike. Candidate exponent rows.

    Outputs:
        exponents:
            ndarray. Exact integer exponent array.
    """
    try:
        raw = np.asarray(exponents, dtype=object)
    except (TypeError, ValueError) as error:
        raise POPValidationError(
            "Lift exponents must form a rectangular numerical array."
        ) from error
    try:
        values = [
            cast_exact_int(value, "Lift exponents", POPValidationError)
            for value in raw.flat
        ]
        return np.asarray(values, dtype=np.int64).reshape(raw.shape)
    except (OverflowError, ValueError) as error:
        raise POPValidationError(
            "Lift exponents are outside the supported integer range."
        ) from error


def _cast_targets(targets):
    """Cast lift targets to literal names and exact replacement degrees. Target
    rewriting changes declared scope and degree, so lossy name or integer
    coercion would silently change the resulting POP.

    Inputs:
        targets:
            Mapping or None. Named targets and new degrees.

    Outputs:
        targets:
            dict. String names mapped to integer degrees.
    """
    if targets is None:
        return {}
    try:
        items = dict(targets).items()
    except (TypeError, ValueError) as error:
        raise POPValidationError("Lift targets must be a mapping.") from error
    cast = {}
    for name, degree in items:
        name = _cast_name(name, label="Lift target name")
        cast[name] = cast_exact_int(
            degree, "Lift target degree", POPValidationError
        )
    return cast
