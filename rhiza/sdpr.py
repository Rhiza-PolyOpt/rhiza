"""Correlatively decomposed semidefinite relaxations for finalized POPs.

The recipe is backend-independent and can be inspected, compressed, and then
emitted to either native solver without rebuilding the polynomial spaces.
"""

from math import fsum

import numpy as np

from rhiza.diagnostics import (
    CompressionMethod,
    DegreeError,
    GraphPolicy,
    PipelineError,
    Relation,
    SDPRError,
    format_summary,
    print_report,
    to_python_report,
)
from rhiza.tools.compression import compress_clique_system
from rhiza.tools.linalg import (
    analyze_balanced_system_rank,
    stable_euclidean_norm,
)
from rhiza.tools.parsing import cast_exact_int, cast_real_scalar
from rhiza.tools.polyalg import (
    build_quotient_basis,
    build_sampled_basis,
    build_separator_basis,
    enumerate_exponents,
    evaluate_sparse_monomials,
    iter_gramian_eval_chunks,
    gramian_size_to_num_coeffs,
    materialize_gramian_evals,
    _NUM_COEFFS_PER_CHUNK,
)


# Automatic proposals target small discrete modes. Wider finite-valued columns
# and joint partitions are left unsplit rather than turning component discovery
# into a combinatorial preprocessing problem. Every retained proposal is still
# validated against the complete degree-bounded quotient basis.
_MAX_COMPONENT_VALUES_PER_VARIABLE = 8
_MAX_COMPONENT_GROUPS = 16


################################################################################
# Semidefinite programming relaxation
################################################################################


class SDPR:
    """Store a correlatively decomposed semidefinite programming relaxation. It
    records quotient bases, scalar multipliers, PSD Gramians, and clique-system
    equations without committing to a numerical backend."""

    def __init__(
        self,
        pop,
        sos_degree=2,
        mul_degree=None,
        sep_degree=None,
        rank_tol=1e-10,
        basis_rank_limit=None,
        enable_component_splitting=True,
    ):
        """Construct the complete backend-neutral recipe in an unlocked state.
        Analysis and compression may inspect or change row maps before finalization.

        Inputs:
            pop:
                POP. Finalized sampled polynomial optimization problem.
            sos_degree:
                Integer. Even base sum-of-squares degree.
            mul_degree:
                Integer or None. Multiplier-degree policy.
            sep_degree:
                Integer or None. Separator polynomial degree.
            rank_tol:
                Float. Relative numerical-rank tolerance.
            basis_rank_limit:
                Integer or None. Experimental cap on each retained final
                sampled-basis rank.
            enable_component_splitting:
                Boolean. Whether validated disjoint sample components may use
                separate PSD Gramian blocks.
        """
        # Resolve every public degree and tolerance before validation so the
        # construction below can assume one concrete numerical policy.
        sos_degree = cast_exact_int(
            sos_degree, "sos_degree", DegreeError
        )
        mul_degree = (
            None if mul_degree is None
            else cast_exact_int(mul_degree, "mul_degree", DegreeError)
        )
        sep_degree = (
            None if sep_degree is None
            else cast_exact_int(sep_degree, "sep_degree", DegreeError)
        )
        rank_tol = cast_real_scalar(rank_tol, "rank_tol", SDPRError)
        basis_rank_limit = (
            None if basis_rank_limit is None
            else cast_exact_int(
                basis_rank_limit, "basis_rank_limit", SDPRError
            )
        )
        if not isinstance(enable_component_splitting, (bool, np.bool_)):
            raise SDPRError(
                "enable_component_splitting must be Boolean."
            )
        enable_component_splitting = bool(enable_component_splitting)

        if not getattr(pop, 'finalized', False):
            raise PipelineError(
                "Finalize the POP before constructing an SDPR."
            )
        if sos_degree < 0 or sos_degree%2:
            raise DegreeError("sos_degree must be nonnegative and even.")
        if mul_degree is not None and mul_degree < 0:
            raise DegreeError("mul_degree must be nonnegative.")
        if sep_degree is not None and sep_degree < 0:
            raise DegreeError("sep_degree must be nonnegative.")
        if not np.isfinite(rank_tol) or rank_tol < 0.0:
            raise SDPRError("rank_tol must be finite and nonnegative.")
        if basis_rank_limit is not None and basis_rank_limit <= 0:
            raise SDPRError("basis_rank_limit must be positive.")

        # The registries below form a backend-neutral recipe. They describe
        # coefficient blocks and clique equations, but allocate no native task.
        self._pop = pop
        self._sos_degree = sos_degree
        self._mul_degree = mul_degree
        self._sep_degree = sos_degree if sep_degree is None else sep_degree
        self._rank_tol = rank_tol
        self._basis_rank_limit = basis_rank_limit
        self._enable_component_splitting = enable_component_splitting
        self._finalized = False
        self._compressed = False
        self._compression_report = None
        self._basis_cache = {}
        self._component_basis_cache = {}
        # Exact finite-value cardinalities depend only on one sampled variable.
        # Cache them across clique scopes so shared continuous coordinates are
        # rejected as component labels only once.
        self._finite_value_count_cache = {}
        self._bases = []
        self._scalar_blocks = []
        self._gramians = []
        self._clique_systems = []
        self._separator_edges = []
        self._separator_stats = {}
        self._required_sep_degree_value = None
        self._num_scalar_coeffs_value = 0
        self._constraint_scales = []
        self._objective_offset_value = 0.0
        self._objective_scale_value = 1.0
        self._num_component_split_gramians = 0
        self._num_psd_coeffs_without_component_splitting = 0

        # Construct the full symbolic-and-sampled recipe eagerly so analysis can
        # inspect it before compression or backend finalization.
        self._construct_recipe()
        self._refresh_status_stats()

    @property
    def pop(self):
        """Return the source POP used to construct this relaxation. The POP is
        already finalized, so the polynomial data behind the recipe cannot change
        after SDPR construction."""
        return self._pop

    @property
    def sos_degree(self):
        """Return the fixed base SOS degree. The recipe is built eagerly from this
        value, so exposing it read-only keeps reported configuration synchronized
        with the stored quotient bases and Gramians."""
        return self._sos_degree

    @property
    def mul_degree(self):
        """Return the fixed multiplier-degree policy. ``None`` denotes automatic
        degree fitting; an integer denotes the explicit policy used when the recipe
        was constructed."""
        return self._mul_degree

    @property
    def sep_degree(self):
        """Return the fixed separator degree. Separator topology and sampled
        communication spaces are constructed eagerly from this value."""
        return self._sep_degree

    @property
    def rank_tol(self):
        """Return the fixed relative numerical-rank tolerance. Every quotient
        basis, separator basis, and compression audit in the recipe uses this same
        tolerance."""
        return self._rank_tol

    @property
    def basis_rank_limit(self):
        """Return the fixed experimental retained-basis rank cap.

        The cap changes the relaxation and applies to final registered sampled
        bases. It is not a bound on every intermediate OVERLAP construction.
        """
        return self._basis_rank_limit

    @property
    def enable_component_splitting(self):
        """Return whether validated sample-component Gramian splitting is enabled.

        The option changes only the conic parameterization. A split is accepted
        only when the represented degree-bounded quotient space is exactly
        invariant, to numerical precision, under every proposed component mask.
        """
        return self._enable_component_splitting

    @property
    def finalized(self):
        """Return whether the SDPR has been finalized."""
        return self._finalized

    @property
    def compressed(self):
        """Return whether PQR compression has been applied."""
        return self._compressed

    @property
    def separator_edges(self):
        """Return the final degree-dependent separator clique pairs. RIP exposes
        its verified forest, while OVERLAP may expose a cyclic union of trees."""
        return tuple(
            (edge['first_clique_id'], edge['second_clique_id'])
            for edge in self._separator_edges
        )

    @property
    def cliques(self):
        """Return maximal declared clique scopes in clique-ID order."""
        return self._pop.csp.cliques

    @property
    def status(self):
        """Return a copy of qualitative SDPR status fields."""
        return to_python_report(self._status)

    @property
    def stats(self):
        """Return a copy of numerical SDPR statistics."""
        return to_python_report(self._stats)

    ################################################################################
    # Analysis, compression, and finalization
    ################################################################################

    def analyze(
        self,
        *,
        clique_ids=None,
        num_cliques=None,
        seed=0,
        show=False,
    ):
        """Probe numerical clique-system ranks without selecting sample rows.

        Analysis is intentionally cheaper than compression. It balances each
        requested augmented clique system, computes singular values, stores a
        small rank record, and discards the dense matrix. Repeated calls inspect
        only previously unchecked cliques unless explicit IDs are requested.

        Inputs:
            clique_ids:
                Sequence or None. Exact clique IDs to inspect.
            num_cliques:
                Integer or None. Number of unchecked cliques to sample.
            seed:
                Integer. Deterministic random-selection seed.
            show:
                Boolean. Whether to print the report.

        Outputs:
            report:
                Report. Sample-margin evidence for the selected cliques.
        """
        show = bool(show)
        if clique_ids is not None and num_cliques is not None:
            raise SDPRError(
                "Specify clique_ids or num_cliques, not both."
            )

        num_systems = len(self._clique_systems)
        if clique_ids is not None:
            if isinstance(clique_ids, (str, bytes)):
                raise SDPRError("clique_ids must be an iterable of integers.")
            try:
                selected = tuple(
                    cast_exact_int(value, "clique_id", SDPRError)
                    for value in clique_ids
                )
            except TypeError as error:
                raise SDPRError("clique_ids must be an iterable.") from error
            if len(set(selected)) != len(selected):
                raise SDPRError("clique_ids must not contain duplicates.")
            if any(value < 0 or value >= num_systems for value in selected):
                raise SDPRError("A requested clique ID is out of range.")
        else:
            unchecked = tuple(
                system['clique_id']
                for system in self._clique_systems
                if system['rank_analysis'] is None
            )
            if num_cliques is None:
                # Once every clique has been checked, a complete cached report is
                # more useful than an empty result and requires no factorization.
                selected = unchecked if unchecked else tuple(range(num_systems))
            else:
                num_cliques = cast_exact_int(
                    num_cliques, "num_cliques", SDPRError
                )
                seed = cast_exact_int(seed, "seed", SDPRError)
                if num_cliques <= 0:
                    raise SDPRError("num_cliques must be positive.")
                if seed < 0:
                    raise SDPRError("seed must be nonnegative.")
                how_many = min(num_cliques, len(unchecked))
                if how_many:
                    rng = np.random.default_rng(seed)
                    selected = tuple(sorted(
                        int(value) for value in rng.choice(
                            unchecked, size=how_many, replace=False
                        )
                    ))
                else:
                    selected = ()

        checked_this_call = 0
        selected_reports = []
        for clique_id in selected:
            system = self._clique_systems[clique_id]
            if system['rank_analysis'] is None:
                augmented = self._materialize_augmented_clique_system(
                    clique_id
                )
                analysis, _, _ = analyze_balanced_system_rank(
                    augmented,
                    rank_tol=self.rank_tol,
                )
                system['rank_analysis'] = analysis
                checked_this_call += 1
                del augmented
            selected_reports.append(
                self._clique_rank_report(clique_id)
            )

        self._refresh_status_stats()
        checked_count = sum(
            system['rank_analysis'] is not None
            for system in self._clique_systems
        )
        margins = [basis['sample_margin'] for basis in self._bases]
        margins.extend(
            system['rank_analysis'].sample_margin
            for system in self._clique_systems
            if system['rank_analysis'] is not None
        )
        report = {
            'sample_margin_check': self._status['sample_margin_check'],
            'num_samples': self._pop.num_samples,
            'minimum_sample_margin': min(margins) if margins else None,
            'num_cliques': num_systems,
            'num_cliques_checked': checked_count,
            'num_cliques_checked_this_call': checked_this_call,
            'cliques': tuple(selected_reports),
            'basis_issues': tuple(self._basis_issue_reports()),
        }
        report = to_python_report(report)
        if show:
            print_report(report)
        return report

    def compress(self, *, method='PQR', clique_size_threshold=None):
        """Select audited subsets of sampled equations in eligible cliques.

        Inputs:
            method:
                String or CompressionMethod. Compression method.
            clique_size_threshold:
                Integer or None. Compress only cliques whose declared scope has
                at most this many variables. None compresses every clique.

        Outputs:
            report:
                Report. Before-and-after compression metrics.
        """
        method = CompressionMethod.from_value(method)
        clique_size_threshold = (
            None if clique_size_threshold is None
            else cast_exact_int(
                clique_size_threshold,
                "clique_size_threshold",
                SDPRError,
            )
        )
        if (
            clique_size_threshold is not None
            and clique_size_threshold <= 0
        ):
            raise SDPRError(
                "clique_size_threshold must be positive."
            )

        if self._finalized:
            raise PipelineError("A finalized SDPR cannot be compressed.")
        if self._compressed:
            if (
                method.value != self._compression_report['method']
                or clique_size_threshold
                != self._compression_report['clique_size_threshold']
            ):
                raise PipelineError(
                    "Compression settings cannot change after compression."
                )
            report = dict(self._compression_report)
            report['cliques'] = tuple(
                dict(item) for item in report['cliques']
            )
            report['changed'] = False
            return to_python_report(report)

        # Prepare every selected clique before committing any row map. Cached
        # rank-only analyses avoid repeating the singular-value factorization.
        # Skipped cliques remain untouched and receive ordinary uncompressed row
        # maps when the SDPR is finalized.
        compressions = []
        for clique_id, system in enumerate(self._clique_systems):
            clique_size = len(self.cliques[clique_id])
            if (
                clique_size_threshold is not None
                and clique_size > clique_size_threshold
            ):
                compressions.append(None)
                continue

            augmented = self._materialize_augmented_clique_system(clique_id)
            compression = compress_clique_system(
                augmented,
                method=method,
                rank_tol=self.rank_tol,
                rank_analysis=system['rank_analysis'],
            )
            compressions.append(compression)
            del augmented

        clique_reports = []
        for system, compression in zip(self._clique_systems, compressions):
            clique_id = system['clique_id']
            clique_size = len(self.cliques[clique_id])
            if compression is None:
                analysis = system['rank_analysis']
                item = {
                    'compression_applied': False,
                    'changed': False,
                    'num_rows_original': self._pop.num_samples,
                    'num_rows_retained': self._pop.num_samples,
                    'num_guard_rows': 0,
                    'rank': None if analysis is None else analysis.rank,
                    'sample_margin': (
                        None if analysis is None
                        else analysis.sample_margin
                    ),
                    'condition': (
                        None if analysis is None else analysis.condition
                    ),
                    'singular_gap': (
                        None if analysis is None else analysis.singular_gap
                    ),
                    'row_span_residual': None,
                }
            else:
                system['row_indices'] = compression.row_indices
                system['row_scales'] = compression.row_scales
                system['uses_all_rows'] = (
                    compression.num_rows_retained == self._pop.num_samples
                )
                system['rank_analysis'] = compression.rank_analysis
                system['row_span_residual'] = compression.row_span_residual
                item = compression.as_report()
                item['compression_applied'] = True

            item['clique_id'] = clique_id
            item['scope'] = self.cliques[clique_id]
            item['clique_size'] = clique_size
            clique_reports.append(item)

        num_cliques_selected = sum(
            item['compression_applied'] for item in clique_reports
        )
        self._compressed = True
        self._compression_report = {
            'method': method.value,
            'clique_size_threshold': clique_size_threshold,
            'changed': any(item['changed'] for item in clique_reports),
            'num_cliques_selected': num_cliques_selected,
            'num_cliques_skipped': (
                len(clique_reports) - num_cliques_selected
            ),
            'num_rows_original': sum(
                item['num_rows_original'] for item in clique_reports
            ),
            'num_rows_retained': sum(
                item['num_rows_retained'] for item in clique_reports
            ),
            'cliques': tuple(clique_reports),
        }
        self._refresh_status_stats()
        return to_python_report(self._compression_report)

    def finalize(self):
        """Finalize row maps and lock the backend-neutral recipe. Uncompressed
        systems receive explicit rows and scales so every backend sees one form.

        Outputs:
            changed:
                Boolean. True on the first finalization.
        """
        if self._finalized:
            return False

        # Prepare every row map before changing the stored clique systems.
        row_data = []
        for clique_id, system in enumerate(self._clique_systems):
            if system['row_indices'] is None:
                # Uncompressed systems still need explicit row maps and unit-norm
                # equation scaling for a uniform backend interface.
                row_indices = np.arange(self._pop.num_samples, dtype=np.int32)
                row_scales = self._compute_uncompressed_row_scales(clique_id)
                row_data.append((row_indices, row_scales, True))
            else:
                row_data.append((
                    system['row_indices'],
                    system['row_scales'],
                    system['uses_all_rows'],
                ))

        # Commit row maps, then lock the recipe for backend construction.
        for system, (row_indices, row_scales, uses_all_rows) in zip(
            self._clique_systems, row_data
        ):
            system['row_indices'] = np.asarray(row_indices, dtype=np.int32)
            system['row_scales'] = np.asarray(row_scales, dtype=float)
            system['uses_all_rows'] = uses_all_rows

        # Discard construction-only caches after the immutable row maps have
        # been installed; backends consume only finalized records and iterators.
        self._finalized = True
        self._basis_cache = None
        self._component_basis_cache = None
        self._finite_value_count_cache = None
        self._refresh_status_stats()
        return True

    def __str__(self):
        """Return a readable SDPR summary. The summary is meant for interactive
        inspection and therefore uses the same status and statistics fields
        exposed programmatically."""
        return format_summary('Rhiza SDPR', self._status, self._stats)

    def __repr__(self):
        """Return a concise developer representation. The representation
        highlights identity and lifecycle information without expanding large
        sampled arrays."""
        return (
            f"SDPR(sos_degree={self.sos_degree}, "
            f"num_cliques={len(self._clique_systems)}, "
            f"compressed={self._compressed}, "
            f"finalized={self._finalized})"
        )

    ################################################################################
    # Recipe construction
    ################################################################################

    def _construct_recipe(self):
        """Construct every clique-system block and separator attachment."""
        # First create the local SOS cones, then attach declared POP entries, and
        # finally add the communication spaces that couple the local identities.
        self._construct_clique_systems()
        self._construct_objectives_and_constraints()
        self._normalize_objectives()
        # The local residual degree may exceed sos_degree when the user supplies
        # explicit multiplier degrees or higher-degree objective entries. Record the
        # degree needed to communicate every declared residual direction before
        # constructing the chosen separator space.
        self._required_sep_degree_value = (
            self._compute_required_separator_degree()
        )
        self._construct_separators()

    def _construct_clique_systems(self):
        """Create base SOS Gramians and one bound share per maximal declared clique."""
        for clique_id, clique_names in enumerate(self._pop.csp.cliques):
            # Every maximal declared clique receives one logical SOS polynomial. A
            # validated component split can represent that polynomial by several
            # smaller PSD blocks without changing its sampled function image.
            scope_ids = tuple(
                self._pop._variable_ids[name] for name in clique_names
            )
            basis_id = self._get_quotient_basis(
                scope_ids, self.sos_degree//2
            )
            scalar_block_id = self._add_scalar_block(
                'BOUND', 1, f"bound_C{clique_id}", mul_degree=0
            )
            gramian_ids = self._add_gramian_blocks(
                clique_id,
                'BASE',
                basis_id,
                entry_id=None,
                sign=1.0,
                mul_degree=self.sos_degree,
            )
            self._clique_systems.append({
                'clique_id': clique_id,
                'scope_ids': scope_ids,
                'num_samples': self._pop.num_samples,
                'objective_evals': np.zeros(self._pop.num_samples),
                'scalar_contributions': [{
                    'block_id': scalar_block_id,
                    'kind': 'BOUND',
                    'basis_id': None,
                    'entry_id': None,
                    'sign': 1.0,
                }],
                'gramian_ids': list(gramian_ids),
                'row_indices': None,
                'row_scales': None,
                'uses_all_rows': None,
                'rank_analysis': None,
                'row_span_residual': None,
            })

    def _construct_objectives_and_constraints(self):
        """Attach every declared objective and constraint to its host clique."""
        for entry_id, relation in enumerate(self._pop._entry_relations):
            # CSP assigns every declared entry to one host clique,
            # avoiding duplicate polynomial contributions across local systems.
            clique_id = self._pop.csp.entry_hosts[entry_id]
            system = self._clique_systems[clique_id]
            entry_evals = self._pop._entry_evals[:,entry_id]

            # Internally every problem is a minimization. MAX entries enter with
            # the opposite sign so the backend objective convention stays fixed.
            if relation is Relation.MIN:
                system['objective_evals'] += entry_evals
                continue
            if relation is Relation.MAX:
                system['objective_evals'] -= entry_evals
                continue
            if relation in (Relation.GE, Relation.LE):
                # An inequality receives an SOS multiplier. LE rows are negated so
                # both forms become a nonnegative polynomial contribution.
                constraint_scale = float(np.max(np.abs(entry_evals)))
                if constraint_scale == 0.0:
                    raise SDPRError(
                        "An inequality evaluates to zero at every sample."
                    )
                self._constraint_scales.append(constraint_scale)
                mul_degree = self._resolve_inequality_mul_degree(
                    self._pop._entry_degrees[entry_id]
                )
                basis_id = self._get_quotient_basis(
                    self._pop._entry_scope_ids(entry_id),
                    mul_degree//2,
                )
                sign = 1.0 if relation is Relation.GE else -1.0
                gramian_ids = self._add_gramian_blocks(
                    clique_id,
                    'INEQUALITY',
                    basis_id,
                    entry_id=entry_id,
                    sign=sign,
                    mul_degree=mul_degree,
                    constraint_scale=constraint_scale,
                )
                system['gramian_ids'].extend(gramian_ids)
                continue
            if relation is Relation.EQ:
                # A sampled identity already vanishes on every equation row and
                # therefore belongs to the sampled coordinate ring automatically.
                if self._pop._sampled_identities[entry_id]:
                    continue

                # Other equalities receive unrestricted polynomial multipliers,
                # represented by a free scalar coefficient block rather than PSD.
                constraint_scale = float(np.max(np.abs(entry_evals)))
                if constraint_scale == 0.0:
                    raise SDPRError(
                        "A nonidentity equality evaluates to zero at every sample."
                    )
                self._constraint_scales.append(constraint_scale)
                mul_degree = self._resolve_equality_mul_degree(
                    self._pop._entry_degrees[entry_id]
                )
                basis_id = self._get_quotient_basis(
                    self._pop._entry_scope_ids(entry_id),
                    mul_degree,
                )
                basis = self._bases[basis_id]
                block_id = self._add_scalar_block(
                    'EQUALITY_MULTIPLIER',
                    basis['basis_rank'],
                    f"mul_EQ_E{entry_id}",
                    mul_degree=mul_degree,
                )
                system['scalar_contributions'].append({
                    'block_id': block_id,
                    'kind': 'EQUALITY_MULTIPLIER',
                    'basis_id': basis_id,
                    'entry_id': entry_id,
                    'sign': 1.0,
                    'constraint_scale': constraint_scale,
                })
                continue
            raise SDPRError("Unsupported POP relation in SDPR construction.")

    def _normalize_objectives(self):
        """Center local objectives and apply one common positive scale.

        Each local midpoint is a constant polynomial, so subtracting it changes
        only that clique's bound share. A single global scale preserves the
        relative weighting of all objective entries. Solver reports later undo the
        exact affine transformation

            original_bound = objective_scale*normalized_bound + objective_offset.
        """
        centered_objectives = []
        offsets = []
        objective_scale = 0.0
        for system in self._clique_systems:
            objective_evals = system['objective_evals']
            if not np.all(np.isfinite(objective_evals)):
                raise SDPRError(
                    "Combined objective evaluations must be finite."
                )

            minimum = float(np.min(objective_evals))
            maximum = float(np.max(objective_evals))
            # Halving before addition avoids overflowing when two large finite
            # endpoints have the same sign.
            offset = 0.5*minimum + 0.5*maximum
            centered = objective_evals - offset
            if not np.all(np.isfinite(centered)):
                raise SDPRError(
                    "Centered objective evaluations must be finite."
                )
            centered_objectives.append(centered)
            offsets.append(offset)
            objective_scale = max(
                objective_scale,
                float(np.max(np.abs(centered))),
            )

        if objective_scale == 0.0:
            objective_scale = 1.0
        if not np.isfinite(objective_scale):
            raise SDPRError("The objective scale must be finite.")
        try:
            objective_offset = float(fsum(offsets))
        except (OverflowError, ValueError) as error:
            raise SDPRError(
                "The combined objective offset is not representable."
            ) from error
        if not np.isfinite(objective_offset):
            raise SDPRError(
                "The combined objective offset is not representable."
            )

        for system, centered in zip(
            self._clique_systems, centered_objectives
        ):
            system['objective_evals'] = centered/objective_scale
        self._objective_offset_value = objective_offset
        self._objective_scale_value = objective_scale

    def _construct_separators(self):
        """Construct degree-bounded communication spaces for the graph policy."""
        # The policy changes only how shared residual directions are routed;
        # downstream clique-system assembly consumes the same separator records.
        if self._pop.csp.graph_policy is GraphPolicy.RIP:
            self._construct_rip_separators()
            return
        if self._pop.csp.graph_policy is GraphPolicy.OVERLAP:
            self._construct_overlap_separators()
            return
        raise SDPRError("Unsupported graph policy in SDPR construction.")

    def _construct_rip_separators(self):
        """Attach each shared quotient space along the verified RIP forest."""
        # RIP guarantees that one common forest can carry every shared
        # polynomial direction through the variable intersection on each edge.
        separators = self._pop.csp.structural_separator_scopes
        for structural_edge_id, (first, second) in enumerate(
            self._pop.csp.structural_clique_edges
        ):
            separator_names = separators[structural_edge_id]
            scope_ids = tuple(
                self._pop._variable_ids[name] for name in separator_names
            )
            basis_id = self._get_quotient_basis(
                scope_ids,
                self.sep_degree,
                is_separator=True,
            )
            self._attach_separator(first, second, basis_id)

        # RIP uses one verified structural forest directly. Full pairwise
        # overlap is neither constructed nor needed, so overlap-reduction
        # comparison fields are intentionally absent for this policy.
        self._separator_stats = {}
        self._finish_separator_stats()

    def _construct_overlap_separators(self):
        """Reduce full pairwise overlap transfers to occurrence trees.

        The initial greedy routing reuses existing clique pairs. A monotone
        fixed-point pass then reroutes every occurrence group using only edges
        already in the completed union and removes edges unused by all rebuilt
        trees. Every group remains connected, so the transfer span is preserved.
        """
        groups = self._build_overlap_occurrence_groups()
        num_cliques = len(self._clique_systems)
        available_adjacency = _build_edge_adjacency(
            num_cliques, self._pop.csp.structural_clique_edges
        )
        edge_group_ids = {}
        num_groups = len(groups)
        largest_occurrence = max(
            (len(group['occurrence']) for group in groups),
            default=0,
        )
        num_full = 0
        num_reduced = 0

        # Broad spaces establish reusable topology first. The pruning pass below
        # removes any edge that later, narrower spaces make unnecessary.
        for group_id, group in enumerate(groups):
            occurrence = group['occurrence']
            tree_edges = _select_overlap_tree(
                occurrence, available_adjacency
            )
            rank = group['basis_data'].basis_rank
            num_full += len(occurrence)*(len(occurrence) - 1)//2*rank
            num_reduced += (len(occurrence) - 1)*rank
            for edge in tree_edges:
                _add_adjacency_edge(available_adjacency, edge)
                edge_group_ids.setdefault(edge, []).append(group_id)

        edge_group_ids = _prune_overlap_edge_assignments(
            groups, edge_group_ids, num_cliques
        )

        # Identical group collections use one registered sampled basis. A single
        # occurrence group is already rank-reduced and can be reused directly
        # unless the user requested a final retained-rank cap.
        basis_ids_by_groups = {}
        for first, second in sorted(edge_group_ids):
            group_ids = tuple(sorted(edge_group_ids[first,second]))
            basis_id = basis_ids_by_groups.get(group_ids)
            if basis_id is None:
                num_monomials_raw = sum(
                    groups[group_id]['num_monomials_raw']
                    for group_id in group_ids
                )
                singleton_data = (
                    groups[group_ids[0]]['basis_data']
                    if len(group_ids) == 1 else None
                )
                if (
                    singleton_data is not None
                    and (
                        self.basis_rank_limit is None
                        or singleton_data.basis_rank <= self.basis_rank_limit
                    )
                ):
                    data = singleton_data
                else:
                    basis_blocks = [
                        groups[group_id]['basis_data'].basis_evals
                        for group_id in group_ids
                    ]
                    sampled_evals = (
                        basis_blocks[0]
                        if len(basis_blocks) == 1
                        else np.hstack(basis_blocks)
                    )
                    data = build_sampled_basis(
                        sampled_evals,
                        rank_tol=self.rank_tol,
                        basis_rank_limit=self.basis_rank_limit,
                    )
                    del sampled_evals
                scope_ids = tuple(sorted({
                    variable_id
                    for group_id in group_ids
                    for variable_id in groups[group_id]['scope_ids']
                }))
                basis_id = self._register_basis(
                    scope_ids,
                    self.sep_degree,
                    True,
                    data,
                    num_monomials_raw=num_monomials_raw,
                )
                basis_ids_by_groups[group_ids] = basis_id
            self._attach_separator(first, second, basis_id)

        self._separator_stats = {
            'num_overlap_occurrence_groups': num_groups,
            'largest_overlap_occurrence': largest_occurrence,
            'num_separator_directions_full_topological': num_full,
            'num_separator_directions_reduced_topological': num_reduced,
        }
        self._finish_separator_stats()

    def _build_overlap_occurrence_groups(self):
        """Build sampled function spaces sharing one clique occurrence set.

        Clique occurrence depends only on a monomial's participating variable
        set. Nonzero exponent patterns are reused across equal-width cliques,
        while clique occurrences are cached across all powers sharing the
        same participating variable set.

        Outputs:
            groups:
                list. Reduced sampled bases and their containing cliques.
        """
        if self.sep_degree <= 0:
            return []

        cliques = tuple(system['scope_ids'] for system in self._clique_systems)
        variable_cliques = self._pop.csp._variable_cliques
        monomials_by_occurrence = {}
        exponent_patterns_by_width = {}
        occurrence_by_participants = {}
        seen = set()

        for clique in cliques:
            patterns = exponent_patterns_by_width.get(len(clique))
            if patterns is None:
                patterns = _build_exponent_patterns(
                    len(clique), self.sep_degree
                )
                exponent_patterns_by_width[len(clique)] = patterns
            for participant_positions, positive_exponents in patterns:
                participant_ids = tuple(
                    clique[index] for index in participant_positions
                )
                monomial = (participant_ids, positive_exponents)
                if monomial in seen:
                    continue
                seen.add(monomial)

                occurrence = _get_participant_occurrence(
                    participant_ids,
                    variable_cliques,
                    occurrence_by_participants,
                )
                if len(occurrence) < 2:
                    continue
                monomials_by_occurrence.setdefault(occurrence, []).append(
                    monomial
                )

        groups = []
        for occurrence in sorted(
            monomials_by_occurrence,
            key=lambda item: (-len(item), item),
        ):
            monomials = monomials_by_occurrence[occurrence]
            monomial_evals = evaluate_sparse_monomials(
                self._pop._variable_evals, monomials
            )
            monomial_evals -= np.mean(
                monomial_evals, axis=0, keepdims=True
            )
            data = build_sampled_basis(
                monomial_evals,
                rank_tol=self.rank_tol,
            )
            del monomial_evals
            if data.basis_rank == 0:
                continue
            groups.append({
                'occurrence': occurrence,
                'scope_ids': tuple(sorted({
                    variable_id
                    for participant_ids, _ in monomials
                    for variable_id in participant_ids
                })),
                'basis_data': data,
                'num_monomials_raw': len(monomials),
            })
        return groups

    def _attach_separator(self, first, second, basis_id):
        """Attach one shared separator block with opposite endpoint signs.

        Inputs:
            first:
                Integer. First endpoint clique.
            second:
                Integer. Second endpoint clique.
            basis_id:
                Integer. Shared sampled-function basis.
        """
        basis = self._bases[basis_id]
        # A numerically empty shared space creates neither coefficients nor graph
        # edges; this can occur after centering on degenerate samples.
        if basis['basis_rank'] == 0:
            return
        edge_id = len(self._separator_edges)
        block_id = self._add_scalar_block(
            'SEPARATOR',
            basis['basis_rank'],
            f"sep_E{edge_id}",
            mul_degree=self.sep_degree,
        )
        # The same free coefficient vector enters its two endpoint identities
        # with opposite signs, guaranteeing exact cancellation when they are summed.
        self._clique_systems[first]['scalar_contributions'].append({
            'block_id': block_id,
            'kind': 'SEPARATOR',
            'basis_id': basis_id,
            'entry_id': None,
            'sign': 1.0,
        })
        self._clique_systems[second]['scalar_contributions'].append({
            'block_id': block_id,
            'kind': 'SEPARATOR',
            'basis_id': basis_id,
            'entry_id': None,
            'sign': -1.0,
        })
        self._separator_edges.append({
            'edge_id': edge_id,
            'first_clique_id': first,
            'second_clique_id': second,
            'basis_id': basis_id,
            'block_id': block_id,
        })

    def _finish_separator_stats(self):
        """Complete separator graph and reduction statistics. Graph counts
        describe the final topology, while topological direction counts describe
        conceptual communication copies before edge-basis merging.
        """
        edges = tuple(
            (edge['first_clique_id'], edge['second_clique_id'])
            for edge in self._separator_edges
        )
        num_structural_components = _count_graph_components(
            len(self._clique_systems), edges
        )
        # This is the graph-theoretic cycle rank of the union topology. A cycle
        # need not be algebraically redundant because its edges may carry different
        # function spaces.
        num_cycles = max(
            0, len(edges) - len(self._clique_systems) + num_structural_components
        )
        self._separator_stats.update({
            'num_separator_edges': len(edges),
            'num_separator_components': num_structural_components,
            'num_separator_cycles': num_cycles,
        })

        # OVERLAP reports conceptual transfer copies before final edge-basis
        # merging. These topological counts are distinct from the actual free
        # separator coefficient count in the finished SDPR.
        full = self._separator_stats.get(
            'num_separator_directions_full_topological'
        )
        reduced = self._separator_stats.get(
            'num_separator_directions_reduced_topological'
        )
        if full is not None and reduced is not None:
            self._separator_stats['separator_topological_reduction_ratio'] = (
                1.0 if full == 0 else reduced/full
            )

    ################################################################################
    # Basis and block registries
    ################################################################################

    def _get_quotient_basis(self, scope_ids, degree, is_separator=False):
        """Return the quotient basis for one scope, degree, and constant policy.
        Caching lets local Gramians and RIP separators share identical spaces.

        Inputs:
            scope_ids:
                Sequence. Variable identifiers in the basis scope.
            degree:
                Integer. Quotient-basis degree.
            is_separator:
                Boolean. Whether to remove the constant direction.

        Outputs:
            basis_id:
                Integer. Cached or newly constructed basis identifier.
        """
        # Scope, degree, and constant-handling fully determine the sampled
        # function space, so identical requests can safely share one basis record.
        scope_ids = tuple(sorted(scope_ids))
        key = (scope_ids, degree, is_separator)
        if key in self._basis_cache:
            return self._basis_cache[key]

        variable_evals = self._pop._variable_evals[:,scope_ids]
        if is_separator:
            # Separator spaces omit the constant direction because clique-local
            # bound scalars already span constant redistribution.
            data = build_separator_basis(
                variable_evals,
                degree,
                rank_tol=self.rank_tol,
                basis_rank_limit=self.basis_rank_limit,
            )
        else:
            data = build_quotient_basis(
                variable_evals,
                degree,
                rank_tol=self.rank_tol,
                basis_rank_limit=self.basis_rank_limit,
            )
        basis_id = self._register_basis(
            scope_ids, degree, is_separator, data
        )
        self._basis_cache[key] = basis_id
        return basis_id

    def _register_basis(
        self,
        scope_ids,
        degree,
        is_separator,
        data,
        *,
        num_monomials_raw=None,
    ):
        """Register one sampled function-space basis and return its compact ID.
        The record keeps evaluations and numerical-rank diagnostics together.

        Inputs:
            scope_ids:
                Sequence. Variable identifiers in the represented scope.
            degree:
                Integer. Maximum represented polynomial degree.
            is_separator:
                Boolean. Whether the constant direction is absent.
            data:
                QuotientBasisData. Numerical basis data.
            num_monomials_raw:
                Integer or None. Raw represented monomial count.

        Outputs:
            basis_id:
                Integer. Registered basis identifier.
        """
        num_monomials_raw = (
            data.num_monomials_raw
            if num_monomials_raw is None else num_monomials_raw
        )

        # Store sampled evaluations and diagnostics together. Downstream code
        # refers to the immutable record only by its compact basis ID.
        basis_id = len(self._bases)
        self._bases.append({
            'basis_id': basis_id,
            'scope_ids': tuple(scope_ids),
            'degree': degree,
            'is_separator': bool(is_separator),
            'basis_evals': data.basis_evals,
            'basis_rank': data.basis_rank,
            'basis_condition': data.basis_condition,
            'basis_singular_gap': data.basis_singular_gap,
            'sample_margin': data.sample_margin,
            'basis_rank_limited': data.basis_rank_limited,
            'num_monomials_raw': num_monomials_raw,
            'parent_basis_id': None,
            'component_label': None,
            'component_sample_count': None,
            'component_basis_ids': None,
            'component_variable_ids': None,
        })
        return basis_id

    def _get_gramian_basis_ids(self, unsplit_basis_id):
        """Return one unsplit basis or an exact component-localized partition.

        Rhiza proposes row partitions only from exact finite-valued coordinates
        in the basis scope. A proposal is accepted only when every row-mask
        restriction preserves the represented quotient space. This degree-aware
        test rejects cases where a reducible variety cannot yet be separated by
        the current polynomial basis.
        """
        if not self.enable_component_splitting:
            return (unsplit_basis_id,)
        cached = self._component_basis_cache.get(unsplit_basis_id)
        if cached is not None:
            return cached

        unsplit_basis = self._bases[unsplit_basis_id]
        if unsplit_basis['basis_rank_limited']:
            # Deliberate truncation may destroy an otherwise exact direct sum.
            basis_ids = (unsplit_basis_id,)
            self._component_basis_cache[unsplit_basis_id] = basis_ids
            return basis_ids
        scope_ids = unsplit_basis['scope_ids']
        variable_evals = self._pop._variable_evals[:,scope_ids]
        finite_value_counts = {
            variable_id: self._get_finite_value_count(variable_id)
            for variable_id in scope_ids
        }
        candidates = _candidate_component_partitions(
            variable_evals,
            scope_ids,
            finite_value_counts=finite_value_counts,
            max_values_per_variable=(
                _MAX_COMPONENT_VALUES_PER_VARIABLE
            ),
            max_groups=_MAX_COMPONENT_GROUPS,
        )
        best = None
        for partition in candidates:
            split = _split_basis_by_components(
                unsplit_basis['basis_evals'],
                partition['row_labels'],
                rank_tol=self.rank_tol,
            )
            if split is None:
                continue
            num_coeffs = sum(
                rank*(rank + 1)//2
                for rank in split['component_ranks']
            )
            score = (
                num_coeffs,
                -len(split['component_ranks']),
                partition['variable_ids'],
            )
            if best is None or score < best[0]:
                best = (score, partition, split)

        if best is None:
            basis_ids = (unsplit_basis_id,)
        else:
            _, partition, split = best
            # The adapted parent basis spans exactly the original sampled space.
            # Component records are zero-copy column views, so the optimization
            # avoids cross-component Gramian coordinates without duplicating the
            # parent basis matrix in memory.
            unsplit_basis['basis_evals'] = split['adapted_basis_evals']
            component_basis_ids = []
            start = 0
            for component_index, component_rank in enumerate(
                split['component_ranks']
            ):
                stop = start + component_rank
                basis_id = len(self._bases)
                component_count = split['component_counts'][component_index]
                self._bases.append({
                    'basis_id': basis_id,
                    'scope_ids': unsplit_basis['scope_ids'],
                    'degree': unsplit_basis['degree'],
                    'is_separator': False,
                    'basis_evals': unsplit_basis['basis_evals'][:,start:stop],
                    'basis_rank': component_rank,
                    'basis_condition': unsplit_basis['basis_condition'],
                    'basis_singular_gap': unsplit_basis['basis_singular_gap'],
                    'sample_margin': component_count - component_rank,
                    'basis_rank_limited': False,
                    'num_monomials_raw': unsplit_basis['num_monomials_raw'],
                    'parent_basis_id': unsplit_basis_id,
                    'component_label': partition['labels'][component_index],
                    'component_sample_count': component_count,
                    'component_basis_ids': None,
                    'component_variable_ids': partition['variable_ids'],
                })
                component_basis_ids.append(basis_id)
                start = stop
            basis_ids = tuple(component_basis_ids)
            unsplit_basis['component_basis_ids'] = basis_ids
            unsplit_basis['component_variable_ids'] = partition['variable_ids']

        self._component_basis_cache[unsplit_basis_id] = basis_ids
        return basis_ids

    def _get_finite_value_count(self, variable_id):
        """Return one sampled variable's exact small-cardinality value count.

        Continuous coordinates are rejected as soon as their observed exact
        values exceed the automatic component proposal limit. The result is
        cached by global variable ID because the same coordinate may appear in
        many quotient-basis scopes.

        Inputs:
            variable_id:
                Integer. Global POP variable identifier.

        Outputs:
            value_count:
                Integer or None. Exact observed cardinality when it is small.
        """
        if variable_id in self._finite_value_count_cache:
            return self._finite_value_count_cache[variable_id]
        value_count = _count_small_exact_value_set(
            self._pop._variable_evals[:,variable_id],
            _MAX_COMPONENT_VALUES_PER_VARIABLE,
        )
        self._finite_value_count_cache[variable_id] = value_count
        return value_count

    def _add_gramian_blocks(
        self,
        clique_id,
        kind,
        unsplit_basis_id,
        *,
        entry_id,
        sign,
        mul_degree,
        constraint_scale=1.0,
    ):
        """Register one logical SOS Gramian, possibly as component blocks.

        Diagnostics retain the coefficient count of the unsplit Gramian. If a
        validated direct sum exists, each localized component space receives
        one PSD block and no cross-component symmetric coordinates are submitted.
        """
        unsplit_rank = self._bases[unsplit_basis_id]['basis_rank']
        self._num_psd_coeffs_without_component_splitting += (
            unsplit_rank*(unsplit_rank + 1)//2
        )
        basis_ids = self._get_gramian_basis_ids(unsplit_basis_id)
        if len(basis_ids) > 1:
            self._num_component_split_gramians += 1
        return tuple(
            self._add_gramian(
                clique_id,
                kind,
                basis_id,
                entry_id=entry_id,
                sign=sign,
                mul_degree=mul_degree,
                constraint_scale=constraint_scale,
                unsplit_basis_id=unsplit_basis_id,
                component_index=(
                    component_index if len(basis_ids) > 1 else None
                ),
            )
            for component_index, basis_id in enumerate(basis_ids)
        )

    def _add_scalar_block(self, kind, size, name, mul_degree):
        """Allocate one contiguous block in the global free-scalar index space.
        Bounds, equality multipliers, and separators all use this common ordering.

        Inputs:
            kind:
                String. Scalar block category.
            size:
                Integer. Number of scalar coefficients.
            name:
                String. Diagnostic block name.
            mul_degree:
                Integer. Polynomial multiplier degree.

        Outputs:
            block_id:
                Integer. Allocated scalar block identifier.
        """
        # Scalar blocks share one global contiguous index space so every
        # backend can emit columns incrementally in the same order.
        start = self._num_scalar_coeffs_value
        block_id = len(self._scalar_blocks)
        self._scalar_blocks.append({
            'block_id': block_id,
            'kind': kind,
            'name': name,
            'size': size,
            'start': start,
            'stop': start + size,
            'mul_degree': mul_degree,
        })
        self._num_scalar_coeffs_value += size
        return block_id

    def _add_gramian(
        self,
        clique_id,
        kind,
        basis_id,
        *,
        entry_id,
        sign,
        mul_degree,
        constraint_scale=1.0,
        unsplit_basis_id=None,
        component_index=None,
    ):
        """Register one PSD Gramian and the data needed to evaluate its map lazily.
        The associated quotient-basis rank becomes the native cone dimension.

        Inputs:
            clique_id:
                Integer. Clique ID of the associated clique system.
            kind:
                String. Base or inequality category.
            basis_id:
                Integer. Quotient-basis identifier.
            entry_id:
                Integer or None. Multiplied constraint entry.
            sign:
                Float. Canonical constraint sign.
            mul_degree:
                Integer. SOS multiplier degree.
            constraint_scale:
                Float. Positive divisor applied to multiplier evaluations.
            unsplit_basis_id:
                Integer or None. Identifier of the unsplit quotient basis.
            component_index:
                Integer or None. Position in a validated component family.

        Outputs:
            gramian_id:
                Integer. Registered Gramian identifier.
        """
        # The quotient-basis rank is exactly the PSD block dimension; the
        # multiplied entry and sign are retained for lazy coefficient evaluation.
        gramian_id = len(self._gramians)
        basis = self._bases[basis_id]
        self._gramians.append({
            'gramian_id': gramian_id,
            'clique_id': clique_id,
            'kind': kind,
            'basis_id': basis_id,
            'unsplit_basis_id': (
                basis_id if unsplit_basis_id is None else unsplit_basis_id
            ),
            'component_index': component_index,
            'entry_id': entry_id,
            'sign': float(sign),
            'mul_degree': mul_degree,
            'constraint_scale': float(constraint_scale),
            'gramian_size': basis['basis_rank'],
            'name': f"gramian_{kind}_G{gramian_id}",
        })
        return gramian_id

    ################################################################################
    # Analysis and coefficient materialization
    ################################################################################

    def _materialize_clique_system_evals(self, clique_id):
        """Materialize one dense clique coefficient system for reference use.

        Analysis, PQR compression, and algebraic tests need the complete local
        matrix. Native backend construction instead streams bounded coefficient
        chunks and must not call this helper on large uncompressed problems.

        Inputs:
            clique_id:
                Integer. Clique-system identifier.

        Outputs:
            system_evals:
                ndarray. Sample-by-coefficient evaluation matrix.
        """
        num_coeffs = self._clique_system_num_coeffs(clique_id)
        system_evals = np.empty((self._pop.num_samples, num_coeffs))
        self._fill_clique_system_evals(clique_id, system_evals)
        return system_evals

    def _materialize_augmented_clique_system(self, clique_id):
        """Materialize a dense reference system including the objective column.

        PQR uses this augmented matrix to preserve both certificate columns and
        the local affine right-hand side while selecting original sample rows.

        Inputs:
            clique_id:
                Integer. Clique-system identifier.

        Outputs:
            system_evals:
                ndarray. Complete augmented clique-system matrix.
        """
        num_coeffs = self._clique_system_num_coeffs(clique_id)
        system_evals = np.empty((self._pop.num_samples, num_coeffs + 1))
        self._fill_clique_system_evals(clique_id, system_evals[:,:num_coeffs])
        system_evals[:,num_coeffs] = self._clique_systems[
            clique_id
        ]['objective_evals']
        return system_evals

    def _fill_clique_system_evals(self, clique_id, system_evals):
        """Fill a preallocated clique coefficient matrix in backend column order.
        Scalar blocks lead, followed by streamed symmetric Gramian coefficients.

        Inputs:
            clique_id:
                Integer. Clique-system identifier.
            system_evals:
                ndarray. Preallocated sample-by-coefficient matrix.
        """
        system = self._clique_systems[clique_id]
        column_start = 0

        # Scalar blocks occupy the leading columns in registry order. Fill the
        # destination from bounded basis slices so no second full scalar block is
        # materialized beside the clique matrix.
        for scalar_contribution in system['scalar_contributions']:
            block = self._scalar_blocks[scalar_contribution['block_id']]
            for start, stop, coeff_evals in self._iter_scalar_contribution_eval_chunks(
                scalar_contribution,
                max_num_coeffs=_NUM_COEFFS_PER_CHUNK,
            ):
                system_evals[
                    :,column_start + start:column_start + stop
                ] = coeff_evals
            column_start += block['size']

        # Symmetric Gramian coefficients follow scalar blocks. Materialization
        # uses the same lower-triangle ordering later streamed to each backend.
        for gramian_id in system['gramian_ids']:
            gramian = self._gramians[gramian_id]
            basis = self._bases[gramian['basis_id']]
            multiplier_evals = self._gramian_multiplier_evals(gramian)
            coeff_evals = materialize_gramian_evals(
                basis['basis_evals'],
                multiplier_evals=multiplier_evals,
                sign=gramian['sign'],
            )
            column_stop = column_start + coeff_evals.shape[1]
            system_evals[:,column_start:column_stop] = coeff_evals
            column_start = column_stop

        if column_start != system_evals.shape[1]:
            raise SDPRError("Clique-system coefficient count is inconsistent.")

    def _iter_scalar_contribution_eval_chunks(
        self,
        scalar_contribution,
        *,
        row_indices=None,
        row_scales=None,
        max_num_coeffs,
    ):
        """Yield bounded evaluation slices for one free-scalar block.

        Equality multipliers are formed only for the requested basis columns and
        sample rows. The same generator therefore serves dense analysis, row-norm
        accumulation, and native backend emission without changing coefficient
        ordering.

        Inputs:
            scalar_contribution:
                Mapping. Scalar contribution attachment record.
            row_indices:
                ndarray or None. Selected sample rows.
            row_scales:
                ndarray or None. Multipliers for selected equations.
            max_num_coeffs:
                Integer. Approximate maximum elements per yielded matrix.

        Outputs:
            chunks:
                Iterator. Local column bounds and coefficient evaluations.
        """
        block = self._scalar_blocks[scalar_contribution['block_id']]
        num_rows = (
            self._pop.num_samples
            if row_indices is None else len(row_indices)
        )
        num_columns_chunk = max(
            1, int(max_num_coeffs)//max(1, num_rows)
        )

        # Prepare row selections and normalized equality evaluations once per
        # block. Only the basis-column slice changes between yielded chunks.
        basis_evals = None
        multiplier_evals = None
        if scalar_contribution['kind'] != 'BOUND':
            basis_evals = self._bases[
                scalar_contribution['basis_id']
            ]['basis_evals']
        if scalar_contribution['kind'] == 'EQUALITY_MULTIPLIER':
            multiplier_evals = self._pop._entry_evals[
                :,scalar_contribution['entry_id']
            ]/scalar_contribution['constraint_scale']
            if row_indices is not None:
                multiplier_evals = multiplier_evals[row_indices]

        for start in range(0, block['size'], num_columns_chunk):
            stop = min(block['size'], start + num_columns_chunk)
            width = stop - start
            if scalar_contribution['kind'] == 'BOUND':
                coeff_evals = np.ones((num_rows, width))
            else:
                # Slice columns before applying advanced row indexing. This
                # keeps compressed-system temporaries bounded by the requested
                # chunk size instead of copying the full selected basis block.
                basis_slice = basis_evals[:,start:stop]
                if row_indices is not None:
                    basis_slice = basis_slice[row_indices,:]
                coeff_evals = scalar_contribution['sign']*basis_slice
                if multiplier_evals is not None:
                    coeff_evals = multiplier_evals[:,None]*coeff_evals
            if row_scales is not None:
                coeff_evals = row_scales[:,None]*coeff_evals
            yield start, stop, coeff_evals

    def _materialize_scalar_contribution(self, scalar_contribution):
        """Materialize one complete scalar block for reference calculations.

        Production assembly consumes the bounded iterator directly. This helper
        remains useful for focused algebraic tests and small dense comparisons.

        Inputs:
            scalar_contribution:
                Mapping. Scalar contribution attachment record.

        Outputs:
            coeff_evals:
                ndarray. Sample-by-coefficient evaluation matrix.
        """
        block = self._scalar_blocks[scalar_contribution['block_id']]
        coeff_evals = np.empty((self._pop.num_samples, block['size']))
        for start, stop, chunk in self._iter_scalar_contribution_eval_chunks(
            scalar_contribution,
            max_num_coeffs=_NUM_COEFFS_PER_CHUNK,
        ):
            coeff_evals[:,start:stop] = chunk
        return coeff_evals

    def _gramian_multiplier_evals(self, gramian):
        """Return normalized inequality evaluations for one Gramian.

        Base Gramians have no multiplier. Inequality Gramians divide their
        sampled constraint vector by its positive maximum magnitude before any
        products are formed, matching the equality path and avoiding a separate
        potentially huge reciprocal in the coefficient sign.

        Inputs:
            gramian:
                Mapping. Registered Gramian record.

        Outputs:
            multiplier_evals:
                ndarray or None. Normalized inequality evaluations.
        """
        if gramian['entry_id'] is None:
            return None
        multiplier_evals = (
            self._pop._entry_evals[:,gramian['entry_id']]
            / gramian['constraint_scale']
        )
        if not np.all(np.isfinite(multiplier_evals)):
            raise SDPRError(
                "Normalized inequality evaluations must be finite."
            )
        return multiplier_evals

    def _compute_uncompressed_row_scales(self, clique_id):
        """Compute equation row scales without forming the full clique system.

        Inputs:
            clique_id:
                Integer. Clique-system identifier.

        Outputs:
            row_scales:
                ndarray. Unit-norm equation multipliers.
        """
        system = self._clique_systems[clique_id]
        # The augmented equation row includes both coefficient columns and the
        # objective right-hand side, so its running norm starts with the absolute
        # normalized objective value.
        row_norms = np.abs(system['objective_evals']).copy()

        # Combine bounded chunk norms with hypot rather than accumulating
        # explicit squares. This keeps memory linear in the sample count and
        # avoids intermediate overflow or underflow for finite coefficients.
        for scalar_contribution in system['scalar_contributions']:
            for _, _, coeff_evals in self._iter_scalar_contribution_eval_chunks(
                scalar_contribution,
                max_num_coeffs=_NUM_COEFFS_PER_CHUNK,
            ):
                row_norms = np.hypot(
                    row_norms,
                    stable_euclidean_norm(coeff_evals, axis=1),
                )

        for gramian_id in system['gramian_ids']:
            gramian = self._gramians[gramian_id]
            basis = self._bases[gramian['basis_id']]
            multiplier_evals = self._gramian_multiplier_evals(gramian)
            for chunk in iter_gramian_eval_chunks(
                basis['basis_evals'],
                multiplier_evals=multiplier_evals,
                sign=gramian['sign'],
                max_num_coeffs=_NUM_COEFFS_PER_CHUNK,
            ):
                row_norms = np.hypot(
                    row_norms,
                    stable_euclidean_norm(chunk.coeff_evals, axis=1),
                )

        # Zero rows remain unchanged; every nonzero equation is normalized to
        # unit Euclidean norm before native submission.
        if not np.all(np.isfinite(row_norms)):
            raise SDPRError(
                "An augmented clique row norm is not representable."
            )
        row_scales = np.ones(row_norms.shape)
        nonzero = row_norms > 0.0
        row_scales[nonzero] = 1.0/row_norms[nonzero]
        if not np.all(np.isfinite(row_scales)):
            raise SDPRError(
                "An augmented clique row is too small to normalize in "
                "float64 arithmetic."
            )
        return row_scales

    def _clique_system_num_coeffs(self, clique_id):
        """Count the free scalar and Gramian coefficients in one clique system.

        Inputs:
            clique_id:
                Integer. Clique-system identifier.

        Outputs:
            num_coeffs:
                Integer. Clique-system coefficient count.
        """
        system = self._clique_systems[clique_id]
        num_scalar = sum(
            self._scalar_blocks[contribution['block_id']]['size']
            for contribution in system['scalar_contributions']
        )
        num_gramian = sum(
            gramian_size_to_num_coeffs(
                self._gramians[gramian_id]['gramian_size']
            )
            for gramian_id in system['gramian_ids']
        )
        return num_scalar + num_gramian

    ################################################################################
    # Backend interface
    ################################################################################

    def _require_finalized(self):
        """Require the SDPR recipe to be finalized."""
        if not self._finalized:
            raise PipelineError("Finalize the SDPR before building a Solver.")

    def _iter_clique_systems(self):
        """Yield finalized clique-system records. The iterator preserves the
        mathematical ordering while bounding temporary memory during downstream
        assembly.

        Outputs:
            systems:
                Iterator. Finalized clique-system records.
        """
        self._require_finalized()
        return iter(self._clique_systems)

    def _iter_gramians(self):
        """Yield registered Gramian records. The iterator preserves the
        mathematical ordering while bounding temporary memory during downstream
        assembly.

        Outputs:
            gramians:
                Iterator. Registered Gramian records.
        """
        self._require_finalized()
        return iter(self._gramians)

    def _iter_scalar_coeff_chunks(self, clique_id, *, max_num_coeffs):
        """Yield submitted scalar coefficient chunks for one clique.

        Basis columns are sliced before equality multiplication or row selection,
        so the temporary matrix is bounded by ``max_num_coeffs`` even when the
        complete multiplier block is large.

        Inputs:
            clique_id:
                Integer. Clique-system identifier.
            max_num_coeffs:
                Integer. Approximate maximum chunk elements.

        Outputs:
            chunks:
                Iterator. Scalar coefficient chunks.
        """
        self._require_finalized()
        system = self._clique_systems[clique_id]
        row_indices = None if system['uses_all_rows'] else system['row_indices']
        row_scales = system['row_scales']

        for scalar_contribution in system['scalar_contributions']:
            block = self._scalar_blocks[scalar_contribution['block_id']]
            for start, stop, coeff_evals in self._iter_scalar_contribution_eval_chunks(
                scalar_contribution,
                row_indices=row_indices,
                row_scales=row_scales,
                max_num_coeffs=max_num_coeffs,
            ):
                scalar_indices = np.arange(
                    block['start'] + start,
                    block['start'] + stop,
                    dtype=np.int32,
                )
                yield _ScalarCoeffChunk(scalar_indices, coeff_evals)

    def _iter_gramian_coeff_chunks(self, clique_id, *, max_num_coeffs):
        """Yield submitted Gramian coefficient chunks for one clique. The iterator
        preserves the mathematical ordering while bounding temporary memory
        during downstream assembly.

        Inputs:
            clique_id:
                Integer. Clique-system identifier.
            max_num_coeffs:
                Integer. Approximate maximum chunk elements.

        Outputs:
            chunks:
                Iterator. Gramian coefficient chunks.
        """
        self._require_finalized()
        system = self._clique_systems[clique_id]
        # Delegate symmetric-product chunking to polyalg, supplying only the
        # selected rows and scales required by the finalized clique system.
        for gramian_id in system['gramian_ids']:
            gramian = self._gramians[gramian_id]
            basis = self._bases[gramian['basis_id']]
            multiplier_evals = self._gramian_multiplier_evals(gramian)
            for chunk in iter_gramian_eval_chunks(
                basis['basis_evals'],
                row_indices=(
                    None if system['uses_all_rows']
                    else system['row_indices']
                ),
                row_scales=system['row_scales'],
                multiplier_evals=multiplier_evals,
                sign=gramian['sign'],
                max_num_coeffs=max_num_coeffs,
            ):
                yield _GramianCoeffChunk(
                    gramian_id,
                    chunk.gramian_rows,
                    chunk.gramian_columns,
                    chunk.coeff_evals,
                )

    def _clique_objective_evals(self, clique_id):
        """Return the selected and scaled objective evaluations for one clique.

        Inputs:
            clique_id:
                Integer. Clique-system identifier.

        Outputs:
            objective_evals:
                ndarray. Scaled objective evaluations on submitted rows.
        """
        self._require_finalized()
        system = self._clique_systems[clique_id]
        objective_evals = system['objective_evals']
        if not system['uses_all_rows']:
            objective_evals = objective_evals[system['row_indices']]
        return system['row_scales']*objective_evals

    def _scalar_objective(self):
        """Return the canonical objective coefficients of all free scalars. Only
        clique bound shares enter the certificate objective; other blocks are free.

        Outputs:
            objective_coeffs:
                ndarray. Free-scalar objective coefficients.
        """
        objective = np.zeros(self._num_scalar_coeffs_value)
        for block in self._scalar_blocks:
            if block['kind'] == 'BOUND':
                objective[block['start']] = 1.0
        return objective

    def _num_scalar_coeffs(self):
        """Return the total number of globally indexed free scalar coefficients.
        Both backends use this dimension when allocating their scalar variables.

        Outputs:
            count:
                Integer. Number of free scalar coefficients.
        """
        return self._num_scalar_coeffs_value

    def _num_gramians(self):
        """Return the number of PSD Gramian blocks in the relaxation recipe. Each
        block becomes one native semidefinite variable or cone.

        Outputs:
            count:
                Integer. Number of PSD Gramian blocks.
        """
        return len(self._gramians)

    ################################################################################
    # Degree and summary helpers
    ################################################################################

    def _resolve_inequality_mul_degree(self, constraint_degree):
        """Resolve one inequality SOS multiplier degree. Automatic fitting uses
        the largest even degree whose product stays within the base SOS degree.

        Inputs:
            constraint_degree:
                Integer. Declared inequality degree.

        Outputs:
            mul_degree:
                Integer. Even SOS multiplier degree.
        """
        if self.mul_degree is None:
            available = self.sos_degree - constraint_degree
            if available < 0:
                raise DegreeError(
                    "An inequality degree exceeds sos_degree under the "
                    "automatic multiplier policy."
                )
            return available - available%2
        return self.mul_degree - self.mul_degree%2

    def _resolve_equality_mul_degree(self, constraint_degree):
        """Resolve one unrestricted equality multiplier degree. Automatic fitting
        uses exactly the degree remaining below the base SOS degree.

        Inputs:
            constraint_degree:
                Integer. Declared equality degree.

        Outputs:
            mul_degree:
                Integer. Free multiplier degree.
        """
        if self.mul_degree is None:
            available = self.sos_degree - constraint_degree
            if available < 0:
                raise DegreeError(
                    "An equality degree exceeds sos_degree under the "
                    "automatic multiplier policy."
                )
            return available
        return self.mul_degree

    def _compute_required_separator_degree(self):
        """Return the largest declared degree appearing in a local residual. A
        separator degree at least this large can represent every residual direction
        created by objectives, SOS contributions, and actual multiplier products.

        Outputs:
            degree:
                Integer. Conservative communication degree for this recipe.
        """
        required = self.sos_degree
        for entry_id, relation in enumerate(self._pop._entry_relations):
            degree = self._pop._entry_degrees[entry_id]
            if relation in (Relation.MIN, Relation.MAX):
                required = max(required, degree)
                continue
            if relation in (Relation.GE, Relation.LE):
                mul_degree = self._resolve_inequality_mul_degree(degree)
                required = max(required, degree + mul_degree)
                continue
            if relation is Relation.EQ:
                # Sampled identities vanish in the sampled coordinate ring and do
                # not contribute a multiplier contribution to any clique residual.
                if self._pop._sampled_identities[entry_id]:
                    continue
                mul_degree = self._resolve_equality_mul_degree(degree)
                required = max(required, degree + mul_degree)
        return required

    def _clique_rank_report(self, clique_id):
        """Return one cached clique-system rank record.

        Inputs:
            clique_id:
                Integer. Clique-system identifier.

        Outputs:
            report:
                Report. Scope, dimensions, rank, and conditioning.
        """
        system = self._clique_systems[clique_id]
        analysis = system['rank_analysis']
        if analysis is None:
            raise SDPRError("The requested clique has not been analyzed.")
        return {
            'clique_id': clique_id,
            'scope': self.cliques[clique_id],
            'num_coeffs': self._clique_system_num_coeffs(clique_id),
            'rank': analysis.rank,
            'sample_margin': analysis.sample_margin,
            'condition': analysis.condition,
            'singular_gap': analysis.singular_gap,
        }

    def _basis_issue_reports(self):
        """Yield only basis records requiring attention.

        Outputs:
            reports:
                Iterator. Zero-margin or deliberately rank-limited bases.
        """
        for basis in self._bases:
            if basis['sample_margin'] > 0 and not basis['basis_rank_limited']:
                continue
            yield {
                'basis_id': basis['basis_id'],
                'scope': tuple(
                    self._pop._variable_names[index]
                    for index in basis['scope_ids']
                ),
                'degree': basis['degree'],
                'is_separator': basis['is_separator'],
                'rank': basis['basis_rank'],
                'sample_margin': basis['sample_margin'],
                'condition': basis['basis_condition'],
                'singular_gap': basis['basis_singular_gap'],
                'rank_limited': basis['basis_rank_limited'],
            }

    def _sample_margin_check(self):
        """Summarize the completeness and outcome of empirical rank checks.

        Outputs:
            result:
                String. One of UNCHECKED, PARTIAL, NO_MARGIN, or
                POSITIVE_MARGIN.
        """
        if any(basis['sample_margin'] == 0 for basis in self._bases):
            return 'NO_MARGIN'

        analyses = tuple(
            system['rank_analysis']
            for system in self._clique_systems
            if system['rank_analysis'] is not None
        )
        if any(analysis.sample_margin == 0 for analysis in analyses):
            return 'NO_MARGIN'
        if not analyses:
            return 'UNCHECKED'
        if len(analyses) < len(self._clique_systems):
            return 'PARTIAL'
        return 'POSITIVE_MARGIN'

    def _refresh_status_stats(self):
        """Refresh actionable SDPR status and stable formulation statistics."""
        num_rows_original = self._pop.num_samples*len(self._clique_systems)
        num_rows_retained = sum(
            self._pop.num_samples
            if system['row_indices'] is None
            else len(system['row_indices'])
            for system in self._clique_systems
        )

        self._status = {
            'sample_margin_check': self._sample_margin_check(),
            'basis_rank_limited': any(
                basis['basis_rank_limited'] for basis in self._bases
            ),
            'sep_degree_insufficient': (
                self.sep_degree < self._required_sep_degree_value
            ),
        }

        constraint_scale_range = None
        if self._constraint_scales:
            constraint_scale_range = (
                min(self._constraint_scales),
                max(self._constraint_scales),
            )

        self._stats = {
            'sos_degree': self.sos_degree,
            'sep_degree': self.sep_degree,
            'required_sep_degree': self._required_sep_degree_value,
            'num_samples': self._pop.num_samples,
            'num_bases': len(self._bases),
            'num_cliques': len(self._clique_systems),
            'num_gramians': len(self._gramians),
            'enable_component_splitting': self.enable_component_splitting,
            'num_component_split_gramians': (
                self._num_component_split_gramians
            ),
            'num_scalar_coeffs': self._num_scalar_coeffs_value,
            'num_separator_coeffs': sum(
                block['size']
                for block in self._scalar_blocks
                if block['kind'] == 'SEPARATOR'
            ),
            'num_psd_coeffs': sum(
                gramian['gramian_size']*(gramian['gramian_size'] + 1)//2
                for gramian in self._gramians
            ),
            'num_psd_coeffs_without_component_splitting': (
                self._num_psd_coeffs_without_component_splitting
            ),
            'num_rows_original': num_rows_original,
            'num_rows_retained': num_rows_retained,
            'objective_offset': self._objective_offset_value,
            'objective_scale': self._objective_scale_value,
            'constraint_scale_range': constraint_scale_range,
            **self._separator_stats,
        }


################################################################################
# Component splitting helpers
################################################################################


def _candidate_component_partitions(
    variable_evals,
    scope_ids,
    *,
    finite_value_counts=None,
    max_values_per_variable,
    max_groups,
):
    """Propose small exact row partitions from finite-valued coordinates.

    Repeated values only propose a partition. Rhiza does not infer algebraic
    components from approximate geometry, and every proposal must pass the
    degree-bounded invariance test in ``_split_basis_by_components``. Optional
    cached cardinalities avoid rescanning shared variable columns.
    """
    variable_evals = np.asarray(variable_evals, dtype=float)
    scope_ids = tuple(scope_ids)
    finite_columns = []
    for local_id, variable_id in enumerate(scope_ids):
        if finite_value_counts is None:
            value_count = _count_small_exact_value_set(
                variable_evals[:,local_id],
                max_values_per_variable,
            )
        else:
            value_count = finite_value_counts[variable_id]
        if value_count is not None and value_count >= 2:
            finite_columns.append((value_count, local_id, variable_id))
    finite_columns.sort()
    if not finite_columns:
        return ()

    candidates = []
    seen = set()

    def add_partition(local_ids, variable_ids):
        """Record one unique exact row partition with admissible size."""
        values = variable_evals[:,local_ids]
        labels, row_labels = np.unique(
            values,
            axis=0,
            return_inverse=True,
        )
        if labels.shape[0] < 2 or labels.shape[0] > max_groups:
            return
        key = row_labels.astype(np.int32, copy=False).tobytes()
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            'variable_ids': tuple(variable_ids),
            'labels': tuple(
                tuple(float(value) for value in row)
                for row in labels
            ),
            'row_labels': row_labels.astype(np.int32, copy=False),
        })

    # Individual coordinates catch the common one-binary case and remain a
    # fallback when a wider joint partition would exceed the component cap.
    for _, local_id, variable_id in finite_columns:
        add_partition((local_id,), (variable_id,))

    # Cumulative joint partitions catch one-hot and mutually exclusive modes.
    # Observed feasible patterns, rather than the full Cartesian product, set the
    # component count.
    local_ids = []
    variable_ids = []
    for _, local_id, variable_id in finite_columns:
        trial = tuple(local_ids + [local_id])
        if np.unique(variable_evals[:,trial], axis=0).shape[0] > max_groups:
            continue
        local_ids.append(local_id)
        variable_ids.append(variable_id)
        add_partition(tuple(local_ids), tuple(variable_ids))
    return tuple(candidates)


def _count_small_exact_value_set(values, maximum):
    """Count a small exact value set without sorting a continuous column.

    Most sampled variables are continuous. Scanning stops as soon as more than
    ``maximum`` values are seen, so automatic component proposals add only a
    small constant amount of work for those columns. Finite mode coordinates
    are scanned completely to obtain their exact observed cardinality.
    """
    observed = set()
    for value in np.asarray(values, dtype=float):
        observed.add(float(value))
        if len(observed) > maximum:
            return None
    return len(observed)


def _component_invariance_tolerance(num_samples, basis_rank, rank_tol):
    """Return a strict residual tolerance for an exact sampled direct sum."""
    machine_floor = (
        100.0*np.finfo(float).eps*max(1, num_samples, basis_rank)
    )
    rank_floor = min(10.0*float(rank_tol), 1e-8)
    return max(machine_floor, rank_floor)


def _split_basis_by_components(basis_evals, component_ids, *, rank_tol):
    """Split an orthonormal sampled basis when row restrictions preserve it.

    For quotient space ``E`` with orthonormal basis ``U`` and component mask
    ``D``, an exact split at the represented degree requires ``D E`` to remain
    in ``E``. Then ``U.T D U`` is an orthogonal projector. Its one-eigenspace
    gives functions whose sampled evaluations vanish outside that component.
    The accepted component ranks must sum to the parent rank, and each
    component must retain positive sample margin before a split is used.
    """
    basis_evals = np.asarray(basis_evals, dtype=float)
    component_ids = np.asarray(component_ids, dtype=np.int32)
    num_samples, basis_rank = basis_evals.shape
    if component_ids.shape != (num_samples,) or basis_rank == 0:
        return None

    tolerance = _component_invariance_tolerance(
        num_samples, basis_rank, rank_tol
    )
    components = []
    component_ranks = []
    component_counts = []
    for component_id in range(int(np.max(component_ids)) + 1):
        mask = component_ids == component_id
        count = int(np.count_nonzero(mask))
        masked_basis = basis_evals*mask[:,None]
        projector = basis_evals.T@masked_basis
        projector = 0.5*(projector + projector.T)
        residual_scale = max(float(np.linalg.norm(masked_basis)), 1.0)
        invariance_residual = np.linalg.norm(
            masked_basis - basis_evals@projector
        )/residual_scale
        if (
            not np.isfinite(invariance_residual)
            or invariance_residual > tolerance
        ):
            return None

        eigenvalues, eigenvectors = np.linalg.eigh(projector)
        near_zero = eigenvalues <= tolerance
        near_one = eigenvalues >= 1.0 - tolerance
        if not np.all(near_zero | near_one):
            return None
        selected = eigenvectors[:,near_one]
        component_rank = selected.shape[1]
        if component_rank == 0 or count <= component_rank:
            return None

        localized_basis = basis_evals@selected
        local_q, _ = np.linalg.qr(
            localized_basis[mask,:], mode='reduced'
        )
        component_basis = np.zeros((num_samples, component_rank))
        component_basis[mask,:] = local_q[:,:component_rank]
        components.append(component_basis)
        component_ranks.append(component_rank)
        component_counts.append(count)

    if sum(component_ranks) != basis_rank:
        return None
    adapted = np.hstack(components)
    span_residual = np.linalg.norm(
        basis_evals - adapted@(adapted.T@basis_evals)
    )/max(float(np.linalg.norm(basis_evals)), 1.0)
    if not np.isfinite(span_residual) or span_residual > tolerance:
        return None

    unsplit_coeffs = basis_rank*(basis_rank + 1)//2
    split_coeffs = sum(
        rank*(rank + 1)//2 for rank in component_ranks
    )
    if split_coeffs >= unsplit_coeffs:
        return None
    return {
        'adapted_basis_evals': adapted,
        'component_ranks': tuple(component_ranks),
        'component_counts': tuple(component_counts),
    }


################################################################################
# Overlap topology helpers
################################################################################


def _build_edge_adjacency(num_nodes, edges):
    """Build a mutable clique adjacency index for overlap-tree reuse. The index
    lets a function-space occurrence inspect only retained edges touching its own
    cliques, avoiding repeated scans of the complete separator graph.

    Inputs:
        num_nodes:
            Integer. Number of clique nodes.
        edges:
            Sequence. Existing clique pairs.

    Outputs:
        adjacency:
            list. Neighbor sets indexed by clique ID.
    """
    adjacency = [set() for _ in range(num_nodes)]
    for edge in edges:
        _add_adjacency_edge(adjacency, edge)
    return adjacency


def _add_adjacency_edge(adjacency, edge):
    """Insert one undirected clique pair into an overlap adjacency index. Edge
    endpoints are canonicalized so every downstream separator record uses the same
    deterministic orientation.

    Inputs:
        adjacency:
            list. Mutable neighbor sets indexed by clique ID.
        edge:
            Sequence. Two clique identifiers.
    """
    first, second = sorted(edge)
    adjacency[first].add(second)
    adjacency[second].add(first)


def _select_overlap_tree(occurrence, available_adjacency):
    """Select one occurrence tree while reusing retained clique pairs. Existing
    induced edges are reused first, and only missing component connections are
    added, so each function space uses exactly k-1 transfer copies.

    Inputs:
        occurrence:
            Sequence. Clique IDs containing one function subspace.
        available_adjacency:
            Sequence. Retained clique-neighbor sets.

    Outputs:
        tree_edges:
            tuple. Clique pairs carrying the function subspace.
    """
    # Canonical ordering makes tree selection reproducible regardless of the set
    # or dictionary order that produced the occurrence group.
    occurrence = tuple(sorted(occurrence))
    occurrence_set = set(occurrence)
    components = _DisjointSet(occurrence)
    tree_edges = []

    # Intersect each local neighbor set with the occurrence set. This considers
    # exactly the retained edges induced by the current function space rather than
    # rescanning every edge accumulated for earlier groups.
    for first in occurrence:
        neighbors = occurrence_set.intersection(available_adjacency[first])
        for second in sorted(neighbors):
            if second <= first:
                continue
            if components.union(first, second):
                tree_edges.append((first, second))

    if len(tree_edges) == len(occurrence) - 1:
        return tuple(sorted(tree_edges))

    # Every occurrence clique contains the represented function's participating
    # variables, so any pair of remaining components is a legal transfer edge.
    # Connect canonical component representatives in a chain without
    # enumerating full pairwise overlap.
    component_nodes = {}
    for clique_id in occurrence:
        root = components.find(clique_id)
        component_nodes.setdefault(root, []).append(clique_id)
    representatives = sorted(
        min(nodes) for nodes in component_nodes.values()
    )
    for first, second in zip(representatives, representatives[1:]):
        tree_edges.append((first, second))

    # A valid spanning tree on k occurrence cliques has exactly k-1 edges; this
    # invariant is the structural reason reduced transfers span full overlap.
    if len(tree_edges) != len(occurrence) - 1:
        raise SDPRError(
            "Could not connect an overlap occurrence group."
        )
    return tuple(sorted(tree_edges))


def _prune_overlap_edge_assignments(groups, edge_group_ids, num_cliques):
    """Remove edges unused after all occurrence trees can see the final union.

    Every pass rebuilds each group tree using only the current edge set. Because
    the prior assignment already connected every group, no new edge is needed.
    The union decreases monotonically until its edge set reaches a fixed point.
    """
    current = {
        tuple(edge): list(group_ids)
        for edge, group_ids in edge_group_ids.items()
    }
    while current:
        current_edges = set(current)
        adjacency = _build_edge_adjacency(num_cliques, current_edges)
        rebuilt = {}
        for group_id, group in enumerate(groups):
            tree_edges = _select_overlap_tree(
                group['occurrence'], adjacency
            )
            if not set(tree_edges).issubset(current_edges):
                raise SDPRError(
                    "Overlap pruning attempted to introduce a new edge."
                )
            for edge in tree_edges:
                rebuilt.setdefault(edge, []).append(group_id)
        if set(rebuilt) == current_edges:
            return rebuilt
        current = rebuilt
    return current


def _build_exponent_patterns(num_variables, degree):
    """Return nonzero exponent positions and powers for nonconstant monomials."""
    patterns = []
    for exponent_row in enumerate_exponents(num_variables, degree)[1:]:
        positions = tuple(
            int(index) for index in np.flatnonzero(exponent_row)
        )
        powers = tuple(int(exponent_row[index]) for index in positions)
        patterns.append((positions, powers))
    return tuple(patterns)


def _get_participant_occurrence(participant_ids, variable_cliques, cache):
    """Return cliques containing one monomial's participating variables."""
    participant_ids = tuple(participant_ids)
    occurrence = cache.get(participant_ids)
    if occurrence is not None:
        return occurrence
    containing = set(variable_cliques[participant_ids[0]])
    for variable_id in participant_ids[1:]:
        containing.intersection_update(variable_cliques[variable_id])
    occurrence = tuple(sorted(containing))
    cache[participant_ids] = occurrence
    return occurrence


def _count_graph_components(num_nodes, edges):
    """Count connected components in the final separator graph. Together with
    edge and node counts, this determines the reported graph cycle rank.

    Inputs:
        num_nodes:
            Integer. Number of clique nodes.
        edges:
            Sequence. Clique-ID pairs.

    Outputs:
        count:
            Integer. Number of connected components.
    """
    components = _DisjointSet(range(num_nodes))
    for first, second in edges:
        components.union(first, second)
    return len({components.find(node) for node in range(num_nodes)})


################################################################################
# Internal records
################################################################################


class _DisjointSet:
    """Maintain deterministic components for overlap-tree selection. Path
    compression and union by size make repeated connectivity tests inexpensive
    while preserving deterministic edge choices."""

    def __init__(self, nodes):
        """Create singleton occurrence components. Overlap-tree selection merges
        them while reusing retained clique pairs and adding only missing links.

        Inputs:
            nodes:
                Sequence. Hashable component nodes.
        """
        nodes = tuple(nodes)

        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}

    def find(self, node):
        """Return one occurrence-component representative with path compression.
        The compact lookup supports repeated induced-edge connectivity tests.

        Inputs:
            node:
                Hashable. Component node.

        Outputs:
            representative:
                Hashable. Component representative.
        """
        parent = self.parent[node]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while node != parent:
            next_node = self.parent[node]
            self.parent[node] = parent
            node = next_node
        return parent

    def union(self, first, second):
        """Merge two occurrence components and report whether they were distinct.
        Tree selection keeps an edge exactly when this operation returns true.

        Inputs:
            first:
                Hashable. First component node.
            second:
                Hashable. Second component node.

        Outputs:
            changed:
                Boolean. True when two components were merged.
        """
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]
        return True


class _ScalarCoeffChunk:
    """Store one scalar coefficient-evaluation chunk. Backends consume these
    bounded records to emit free scalar columns without materializing an entire
    clique system."""

    def __init__(self, scalar_indices, coeff_evals):
        """Bundle one consecutive set of free scalar columns with its evaluations.
        A backend consumes the record immediately and can then release the chunk.

        Inputs:
            scalar_indices:
                ndarray. Global scalar-variable indexes.
            coeff_evals:
                ndarray. Submitted-row coefficient evaluations.
        """
        self.scalar_indices = scalar_indices
        self.coeff_evals = coeff_evals


class _GramianCoeffChunk:
    """Store one Gramian coefficient-evaluation chunk. Backends consume these
    bounded records to emit one PSD block contribution while preserving the
    symmetric coefficient ordering."""

    def __init__(
        self,
        gramian_id,
        gramian_rows,
        gramian_columns,
        coeff_evals,
    ):
        """Bundle one PSD block's symmetric coordinates with their evaluations.
        The record preserves Gramian IDs and matrix positions during streaming.

        Inputs:
            gramian_id:
                Integer. Gramian block ID.
            gramian_rows:
                ndarray. Matrix row indexes.
            gramian_columns:
                ndarray. Matrix column indexes.
            coeff_evals:
                ndarray. Submitted-row coefficient evaluations.
        """
        self.gramian_id = gramian_id
        self.gramian_rows = gramian_rows
        self.gramian_columns = gramian_columns
        self.coeff_evals = coeff_evals
