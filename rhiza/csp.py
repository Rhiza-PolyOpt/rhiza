"""Correlative sparsity patterns and maximal declared clique structure.

This module turns declared polynomial scopes into the graph information and
entry hosts consumed by SDPR.
"""

from collections import defaultdict
import heapq
from rhiza.diagnostics import (
    CSPError,
    GraphPolicy,
    RIPError,
    format_summary,
    to_python_report,
)


_OVERLAP_PAIR_COUNT_WORK_LIMIT = 1_000_000


################################################################################
# Correlative sparsity pattern
################################################################################


class CSP:
    """Store a standalone correlative sparsity pattern.

    The object turns declared entry scopes into maximal declared cliques,
    structural edges, and deterministic entry hosts before SDPR construction.
    """

    def __init__(
        self,
        variable_names,
        entry_scopes,
        *,
        graph_policy='RIP',
    ):
        """Construct cliques, structural edges, and entry hosts.

        The selected policy determines whether running intersection is required.

        Inputs:
            variable_names:
                Sequence. Unique variable names.
            entry_scopes:
                Sequence. Variable scopes for POP entries.
            graph_policy:
                String or GraphPolicy. Sparsity construction policy.
        """
        # Normalize public names before building graph indexes. The rest of
        # CSP construction works only with immutable tuples and integer IDs.
        variable_names = _cast_name_sequence(
            variable_names, label="CSP variable names"
        )
        if isinstance(entry_scopes, str):
            entry_scopes = (entry_scopes,)
        else:
            try:
                entry_scopes = tuple(entry_scopes)
            except TypeError as error:
                raise CSPError(
                    "entry_scopes must be an iterable of variable scopes."
                ) from error
        entry_scopes = tuple(
            _cast_name_scope(scope) for scope in entry_scopes
        )
        graph_policy = GraphPolicy.from_value(graph_policy)

        if not variable_names:
            raise CSPError("A CSP requires at least one variable.")
        if len(set(variable_names)) != len(variable_names):
            raise CSPError("CSP variable names must be unique.")
        if not entry_scopes:
            raise CSPError("A CSP requires at least one entry scope.")

        # Freeze the name-to-ID map once. Cliques, separator scopes, and
        # entry hosts can then share compact identifiers without repeating
        # string work.
        self._variable_names = variable_names
        self._variable_ids = {
            name: variable_id
            for variable_id, name in enumerate(variable_names)
        }
        self._entry_scopes = self._cast_entry_scopes(entry_scopes)
        self._graph_policy = graph_policy
        self._status = {}
        self._stats = {}

        # RIP builds one verified clique forest. OVERLAP keeps the original
        # maximal declared cliques and defers degree-dependent communication
        # to SDPR.
        if graph_policy is GraphPolicy.RIP:
            self._construct_rip()
        elif graph_policy is GraphPolicy.OVERLAP:
            self._construct_overlap()
        elif graph_policy is GraphPolicy.EXTEND:
            raise NotImplementedError(
                "The 'EXTEND' graph policy is not implemented."
            )

    @property
    def graph_policy(self):
        """Return the selected clique-graph policy."""
        return self._graph_policy

    @property
    def cliques(self):
        """Return inclusion-maximal declared scopes as name tuples."""
        return tuple(self._names_from_ids(clique) for clique in self._cliques)

    @property
    def subsumed_cliques(self):
        """Return declared scopes contained in larger maximal declared cliques."""
        return tuple(
            self._names_from_ids(clique)
            for clique in self._subsumed_cliques
        )

    @property
    def structural_clique_edges(self):
        """Return CSP's structural forest edges as clique-ID pairs."""
        return self._clique_edges

    @property
    def structural_separator_scopes(self):
        """Return separator scopes for the structural forest edges."""
        return tuple(
            self._names_from_ids(separator_scope)
            for separator_scope in self._structural_separator_scopes
        )

    @property
    def entry_hosts(self):
        """Return the deterministic host clique for every POP entry."""
        return self._entry_hosts

    @property
    def status(self):
        """Return a copy of qualitative CSP status fields."""
        return to_python_report(self._status)

    @property
    def stats(self):
        """Return a copy of numerical CSP statistics."""
        return to_python_report(self._stats)

    def summary(self):
        """Return CSP status and structural statistics.

        Outputs:
            report:
                Report. CSP status and statistics.
        """
        return {
            'status': self.status,
            'stats': self.stats,
        }

    def __str__(self):
        """Return a readable CSP summary. The summary is meant for interactive
        inspection and therefore uses the same status and statistics fields
        exposed programmatically."""
        return format_summary('Rhiza CSP', self._status, self._stats)

    def __repr__(self):
        """Return a concise developer representation. The representation
        highlights identity and lifecycle information without expanding large
        sampled arrays."""
        return (
            f"CSP(num_variables={len(self._variable_names)}, "
            f"num_cliques={len(self._cliques)}, "
            f"graph_policy='{self._graph_policy.value}')"
        )

    ################################################################################
    # Construction stages
    ################################################################################

    def _construct_rip(self):
        """Construct and verify a linear-memory RIP clique forest.

        Maximum-cardinality search recognizes running intersection from the
        variable-to-clique incidence index, avoiding the quadratic list of all
        pairwise clique intersections. The resulting forest is independently
        verified before it is committed.
        """
        structure = self._prepare_common_structure()
        clique_edges, rip = build_rip_forest(
            len(self._variable_names),
            structure['cliques'],
            structure['variable_cliques'],
        )
        if not rip:
            raise RIPError(
                "Maximal declared cliques do not satisfy running intersection."
            )
        clique_sets = tuple(set(clique) for clique in structure['cliques'])
        structure['clique_edges'] = clique_edges
        structure['separator_scopes'] = tuple(
            tuple(sorted(clique_sets[first] & clique_sets[second]))
            for first, second in clique_edges
        )
        self._commit_clique_structure(structure, rip=True)

    def _construct_overlap(self):
        """Keep the maximal declared cliques without requiring RIP.

        CSP supplies only a reusable structural backbone. SDPR later assigns
        degree-dependent function spaces to this backbone and adds cyclic edges
        where the declared overlaps require them.
        """
        structure = self._prepare_common_structure()
        maximal = structure['cliques']
        variable_cliques = structure['variable_cliques']

        # Reuse a genuine RIP forest when one exists, so OVERLAP reduces to the
        # ordinary tree formulation on running-intersection scope families.
        clique_edges, rip = build_rip_forest(
            len(self._variable_names), maximal, variable_cliques
        )
        if not rip:
            # Otherwise retain only a cheap acyclic backbone. SDPR adds the
            # function-space edges that this structural forest cannot carry.
            clique_edges = build_occurrence_forest(
                len(maximal), variable_cliques
            )
        clique_sets = tuple(set(clique) for clique in maximal)
        structure['clique_edges'] = clique_edges
        structure['separator_scopes'] = tuple(
            tuple(sorted(clique_sets[first] & clique_sets[second]))
            for first, second in clique_edges
        )
        candidate_pairs = count_overlap_pairs(
            len(maximal),
            variable_cliques,
            max_pair_work=_OVERLAP_PAIR_COUNT_WORK_LIMIT,
        )
        if candidate_pairs is not None:
            structure['num_candidate_overlap_pairs'] = candidate_pairs
        self._commit_clique_structure(structure, rip=rip)

    def _prepare_common_structure(self):
        """Prepare clique data shared by RIP and OVERLAP policies.

        Maximal declared clique classification, entry hosting, and incidence
        indexes do not depend on separator topology. Computing them once makes
        the policy split explicit and prevents the two construction paths from
        drifting.

        Outputs:
            structure:
                Mapping. Common clique data ready for topology construction.
        """
        maximal, subsumed = classify_cliques(self._entry_scopes)
        return {
            'cliques': maximal,
            'subsumed_cliques': subsumed,
            'entry_hosts': assign_entry_hosts(
                len(self._variable_names), self._entry_scopes, maximal
            ),
            'variable_cliques': build_variable_clique_occurrences(
                len(self._variable_names), maximal
            ),
        }

    def _commit_clique_structure(self, structure, *, rip):
        """Install one complete structural graph after policy construction.

        Inputs:
            structure:
                Mapping. Prepared clique data.
            rip:
                Boolean. Whether the structural forest satisfies RIP.
        """
        # Commit the prepared graph atomically so a failed construction cannot
        # leave a partially initialized sparsity pattern.
        self._cliques = structure['cliques']
        self._subsumed_cliques = structure['subsumed_cliques']
        self._clique_edges = structure['clique_edges']
        self._structural_separator_scopes = structure['separator_scopes']
        self._entry_hosts = structure['entry_hosts']
        self._variable_cliques = structure['variable_cliques']

        # Status records qualitative guarantees; stats expose inexpensive,
        # stable structural sizes for analysis and reporting.
        clique_sizes = [len(clique) for clique in self._cliques]
        self._status = {
            'graph_policy': self._graph_policy.value,
            'rip': bool(rip),
        }
        self._stats = {
            'num_variables': len(self._variable_names),
            'num_declared_cliques': len(set(self._entry_scopes)),
            'num_cliques': len(self._cliques),
            'num_subsumed_cliques': len(self._subsumed_cliques),
            'num_structural_clique_edges': len(self._clique_edges),
            'num_structural_components': _count_components(
                len(self._cliques), self._clique_edges
            ),
            'clique_size_min': min(clique_sizes),
            'clique_size_max': max(clique_sizes),
        }
        candidate_pairs = structure.get('num_candidate_overlap_pairs')
        if candidate_pairs is not None:
            self._stats['num_candidate_overlap_pairs'] = candidate_pairs

    def _cast_entry_scopes(self, entry_scopes):
        """Convert named entry scopes to sorted variable-ID tuples.

        Inputs:
            entry_scopes:
                Sequence. Named variable scopes.

        Outputs:
            scopes:
                tuple. Sorted variable-ID scopes.
        """
        scopes = []
        for scope in entry_scopes:
            if not scope:
                raise CSPError("Entry scopes must be nonempty.")
            if len(set(scope)) != len(scope):
                raise CSPError("An entry scope contains a repeated variable.")
            try:
                scope_ids = tuple(sorted(
                    self._variable_ids[name] for name in scope
                ))
            except KeyError as error:
                raise CSPError(
                    "An entry scope contains an unknown variable: "
                    + repr(error.args[0])
                ) from error
            scopes.append(scope_ids)
        return tuple(scopes)

    def _names_from_ids(self, variable_ids):
        """Convert a compact variable-ID scope back to public names.

        Inputs:
            variable_ids:
                Sequence. Variable identifiers.

        Outputs:
            names:
                tuple. Corresponding variable names.
        """
        return tuple(self._variable_names[index] for index in variable_ids)


################################################################################
# Graph construction
################################################################################


def classify_cliques(scopes):
    """Separate maximal and strictly subsumed declared cliques.

    Inputs:
        scopes:
            Sequence. Sorted variable-ID scopes.

    Outputs:
        maximal_cliques:
            tuple. Maximal declared cliques.
        subsumed_cliques:
            tuple. Strictly subsumed declared cliques.
    """
    # Visit larger scopes first. A later scope is subsumed exactly when all
    # of its variables already share one maximal declared clique.
    unique_scopes = sorted(
        set(tuple(scope) for scope in scopes),
        key=lambda scope: (-len(scope), scope),
    )
    maximal = []
    variable_maximal = defaultdict(set)
    subsumed = []

    for scope in unique_scopes:
        # Intersect the accepted-clique incidence sets of its variables rather
        # than scanning every maximal scope for containment. Starting from the
        # rarest variable is decisive for lifted sparse models: a private
        # coordinate usually identifies its containing clique immediately,
        # whereas shared pose coordinates may occur in thousands of cliques.
        candidate_ids = _intersect_occurrence_sets(
            scope,
            variable_maximal,
        )

        if candidate_ids:
            subsumed.append(scope)
            continue

        # No accepted larger scope contains this one. Register a new maximal
        # clique and update the inverted incidence index.
        clique_id = len(maximal)
        maximal.append(scope)
        for variable_id in scope:
            variable_maximal[variable_id].add(clique_id)

    maximal.sort(key=lambda scope: (len(scope), scope))
    subsumed.sort(key=lambda scope: (len(scope), scope))
    return tuple(maximal), tuple(subsumed)


def build_variable_clique_occurrences(num_variables, cliques):
    """Index the maximal declared cliques containing each variable.

    Inputs:
        num_variables:
            Integer. Number of variables.
        cliques:
            Sequence. Maximal declared variable-ID cliques.

    Outputs:
        occurrences:
            tuple. Clique-ID tuples in variable order.
    """
    num_variables = int(num_variables)

    occurrences = [[] for _ in range(num_variables)]
    for clique_id, clique in enumerate(cliques):
        for variable_id in clique:
            occurrences[variable_id].append(clique_id)
    return tuple(tuple(clique_ids) for clique_ids in occurrences)


def build_rip_forest(num_variables, cliques, variable_cliques):
    """Recognize RIP and construct a perfect clique forest.

    The maximum-cardinality traversal works from variable incidence lists and
    never materializes all pairwise clique intersections. A final induced-
    subtree audit independently verifies every parent decision.

    Inputs:
        num_variables:
            Integer. Number of variables.
        cliques:
            Sequence. Maximal declared variable-ID cliques.
        variable_cliques:
            Sequence. Clique occurrences for each variable.

    Outputs:
        clique_edges:
            tuple. Running-intersection forest edges when one exists.
        rip:
            Boolean. Whether the clique family satisfies RIP.
    """
    num_variables = int(num_variables)

    # Maximum-cardinality search scores each unselected clique by how many of
    # its variables have already appeared in the traversal.
    num_cliques = len(cliques)
    scores = [0]*num_cliques
    selected = [False]*num_cliques
    seen_variables = [False]*num_variables
    selected_variable_cliques = [[] for _ in range(num_variables)]
    selected_variable_sets = [set() for _ in range(num_variables)]
    heap = []

    # Heap order selects the highest cardinality score, then the largest clique,
    # lexicographic scope, and stable clique ID. The latter keys only make ties
    # deterministic; they never change the maximum-cardinality criterion.
    for clique_id, clique in enumerate(cliques):
        heapq.heappush(
            heap,
            (0, -len(clique), clique, clique_id),
        )

    clique_edges = []
    rip = True
    for _ in range(num_cliques):
        # Heap updates are lazy: discard stale scores until the current best
        # unselected clique reaches the top.
        while heap:
            score_negative, _, _, clique_id = heapq.heappop(heap)
            if selected[clique_id]:
                continue
            if -score_negative != scores[clique_id]:
                continue
            break
        else:
            raise CSPError("RIP forest construction exhausted its heap.")

        # Already seen variables form the separator scope to the selected prefix.
        # RIP requires one earlier clique to contain that separator scope in full.
        clique = cliques[clique_id]
        separator_scope = tuple(
            variable_id
            for variable_id in clique
            if seen_variables[variable_id]
        )
        if separator_scope:
            # Search the shortest occurrence history backward. The first clique
            # present in every separator-scope variable set is the most recent legal
            # parent. Shared-core families therefore form a path rather than a
            # high-degree star while retaining linear incidence storage.
            anchor = min(
                separator_scope,
                key=lambda variable_id: len(
                    selected_variable_cliques[variable_id]
                ),
            )
            parent = None
            for candidate in reversed(selected_variable_cliques[anchor]):
                if all(
                    candidate in selected_variable_sets[variable_id]
                    for variable_id in separator_scope
                ):
                    parent = candidate
                    break
            if parent is None:
                rip = False
            else:
                clique_edges.append(tuple(sorted((parent, clique_id))))

        # A variable raises the score of its remaining occurrence cliques only
        # when it is first introduced into the selected prefix.
        selected[clique_id] = True
        for variable_id in clique:
            selected_variable_cliques[variable_id].append(clique_id)
            selected_variable_sets[variable_id].add(clique_id)
            if seen_variables[variable_id]:
                continue
            seen_variables[variable_id] = True
            for other in variable_cliques[variable_id]:
                if selected[other]:
                    continue
                scores[other] += 1
                heapq.heappush(
                    heap,
                    (
                        -scores[other],
                        -len(cliques[other]),
                        cliques[other],
                        other,
                    ),
                )

    clique_edges = tuple(sorted(set(clique_edges)))
    if rip:
        try:
            verify_rip(num_variables, cliques, clique_edges)
        except RIPError:
            rip = False
    return clique_edges if rip else (), rip

def build_occurrence_forest(num_cliques, variable_cliques):
    """Build a linear-memory forest from variable occurrence chains. OVERLAP uses
    this inexpensive forest only as reusable topology; SDPR later adds any
    function-space edges required beyond it.

    Inputs:
        num_cliques:
            Integer. Number of maximal declared cliques.
        variable_cliques:
            Sequence. Clique occurrences for each variable.

    Outputs:
        clique_edges:
            tuple. Deterministic structural forest edges.
    """
    num_cliques = int(num_cliques)

    components = _DisjointSet(num_cliques)
    clique_edges = []
    # Broad occurrence sets go first so their chains form a useful backbone
    # for variables shared by fewer cliques.
    occurrences = sorted(
        set(tuple(clique_ids) for clique_ids in variable_cliques),
        key=lambda clique_ids: (-len(clique_ids), clique_ids),
    )
    for clique_ids in occurrences:
        # Consecutive IDs give a legal linear-size chain; union-find suppresses
        # any edge that would only create a redundant structural cycle.
        for first, second in zip(clique_ids, clique_ids[1:]):
            if components.union(first, second):
                clique_edges.append((first, second))
    return tuple(sorted(clique_edges))


def count_overlap_pairs(
    num_cliques, variable_cliques, *, max_pair_work=None
):
    """Count distinct overlapping clique pairs with linear auxiliary memory.

    The work is proportional to clique-variable incidences and the overlap
    visits they induce. An optional budget lets construction omit this purely
    diagnostic count before a dense shared-core family causes quadratic work.

    Inputs:
        num_cliques:
            Integer. Number of maximal declared cliques.
        variable_cliques:
            Sequence. Clique occurrences for each variable.
        max_pair_work:
            Integer or None. Maximum induced pair visits before skipping the
            optional exact diagnostic.

    Outputs:
        count:
            Integer or None. Number of distinct overlapping clique pairs.
    """
    num_cliques = int(num_cliques)
    if max_pair_work is not None:
        max_pair_work = int(max_pair_work)
        estimated_work = sum(
            len(clique_ids)*(len(clique_ids) - 1)//2
            for clique_ids in variable_cliques
        )
        if estimated_work > max_pair_work:
            return None

    # Invert the incidence lists once so each clique visits only variable
    # occurrence sets capable of producing an overlap neighbor.
    clique_variables = [[] for _ in range(num_cliques)]
    for variable_id, clique_ids in enumerate(variable_cliques):
        for clique_id in clique_ids:
            clique_variables[clique_id].append(variable_id)

    # A reusable integer stamp array deduplicates neighbors for one clique at a
    # time. Counting only larger IDs records each unordered pair exactly once.
    marks = [-1]*num_cliques
    count = 0
    for first, variable_ids in enumerate(clique_variables):
        for variable_id in variable_ids:
            for second in variable_cliques[variable_id]:
                if second <= first or marks[second] == first:
                    continue
                marks[second] = first
                count += 1
    return count


def verify_rip(num_variables, cliques, clique_edges):
    """Require every variable occurrence set to induce a connected subtree.

    Inputs:
        num_variables:
            Integer. Number of variables.
        cliques:
            Sequence. Maximal declared variable-ID cliques.
        clique_edges:
            Sequence. Clique-forest edges.
    """
    num_variables = int(num_variables)

    # RIP is equivalent to every variable's occurrence set inducing a
    # connected subtree of the proposed clique forest.
    variable_cliques = [[] for _ in range(num_variables)]
    for clique_id, clique in enumerate(cliques):
        for variable_id in clique:
            variable_cliques[variable_id].append(clique_id)

    adjacency = [[] for _ in cliques]
    for first, second in clique_edges:
        adjacency[first].append(second)
        adjacency[second].append(first)

    for variable_id, containing in enumerate(variable_cliques):
        # Zero or one occurrence is connected vacuously. Otherwise traverse only
        # the forest nodes that contain this variable.
        if len(containing) <= 1:
            continue
        allowed = set(containing)
        visited = {containing[0]}
        stack = [containing[0]]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in allowed and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(visited) != len(allowed):
            raise RIPError(
                f"Variable ID {variable_id} violates running intersection."
            )


def assign_entry_hosts(num_variables, entry_scopes, cliques):
    """Assign each entry scope to one deterministic host clique.

    Inputs:
        num_variables:
            Integer. Number of variables.
        entry_scopes:
            Sequence. Variable-ID entry scopes.
        cliques:
            Sequence. Maximal declared variable-ID cliques.

    Outputs:
        entry_hosts:
            tuple. Host clique IDs.
    """
    num_variables = int(num_variables)

    # One inverted incidence index supports every entry-host query.
    variable_cliques = [set() for _ in range(num_variables)]
    for clique_id, clique in enumerate(cliques):
        for variable_id in clique:
            variable_cliques[variable_id].add(clique_id)

    hosts = []
    for scope in entry_scopes:
        # A legal host lies in the intersection of all variable occurrence sets;
        # the smallest ID makes a nonunique host choice deterministic. The
        # intersection begins with the least frequent variable so private
        # coordinates avoid work proportional to the number of global cliques.
        candidates = _intersect_occurrence_sets(scope, variable_cliques)
        if not candidates:
            raise CSPError(
                "No maximal declared clique contains a declared entry scope."
            )
        hosts.append(min(candidates))
    return tuple(hosts)


def _intersect_occurrence_sets(scope, occurrences):
    """Intersect variable occurrence sets from smallest to largest.

    Sparse lifted formulations often list shared variables before one private
    coordinate. Intersecting in declaration order then creates and repeatedly
    filters a large shared-clique set. Cardinality order produces the same
    mathematical intersection while keeping intermediate sets as small as the
    most selective variable permits.

    Inputs:
        scope:
            Sequence. Variable identifiers.
        occurrences:
            Sequence or mapping. Clique-ID sets by variable identifier.

    Outputs:
        common:
            set. Clique IDs containing the complete scope.
    """
    variable_sets = []
    for variable_id in scope:
        if hasattr(occurrences, 'get'):
            ids = occurrences.get(variable_id, ())
        else:
            ids = occurrences[variable_id]
        variable_sets.append(ids)
    variable_sets.sort(key=len)
    if not variable_sets:
        return set()

    common = set(variable_sets[0])
    for ids in variable_sets[1:]:
        if not common:
            break
        common.intersection_update(ids)
    return common


################################################################################
# Input helpers
################################################################################


def _cast_name_sequence(values, *, label):
    """Cast a public sequence of literal CSP names. A bare string denotes one
    variable name, and other objects are not stringified because CSP must preserve
    the same namespace semantics as POP.

    Inputs:
        values:
            Sequence or string. Candidate names.
        label:
            String. Context used in validation errors.

    Outputs:
        names:
            tuple. Literal names in caller order.
    """
    if isinstance(values, str):
        names = (values,)
    else:
        try:
            names = tuple(values)
        except TypeError as error:
            raise CSPError(label + " must be an iterable of strings.") from error
    for name in names:
        if not isinstance(name, str):
            raise CSPError(label + " must contain only strings.")
        if not name or name != name.strip():
            raise CSPError(
                label + " must contain nonempty names without outer whitespace."
            )
    return names


def _cast_name_scope(scope):
    """Cast one CSP entry scope while treating a string as a singleton scope. This
    mirrors POP's scope semantics and prevents multi-character variable names from
    being split into characters.

    Inputs:
        scope:
            ScopeLike. Candidate variable scope.

    Outputs:
        scope:
            tuple. Literal variable names.
    """
    return _cast_name_sequence(scope, label="A CSP entry scope")


################################################################################
# Low-level graph helpers
################################################################################


def _count_components(num_nodes, edges):
    """Count connected components of a clique forest.

    Inputs:
        num_nodes:
            Integer. Number of forest nodes.
        edges:
            Sequence. Forest edges.

    Outputs:
        count:
            Integer. Number of connected components.
    """
    components = _DisjointSet(num_nodes)
    for first, second in edges:
        components.union(first, second)
    return len({components.find(node) for node in range(num_nodes)})


################################################################################
# Internal record
################################################################################


class _DisjointSet:
    """Maintain disjoint clique components for Kruskal selection. Path compression
    and union by size make repeated connectivity tests inexpensive while
    preserving deterministic edge choices."""

    def __init__(self, size):
        """Create one singleton component per clique ID. Kruskal selection and
        component counting reuse this compact parent-and-size representation.

        Inputs:
            size:
                Integer. Number of elements.
        """
        size = int(size)

        self.parent = list(range(size))
        self.size = [1]*size

    def find(self, item):
        """Return one component representative with path compression. Repeated
        queries become nearly constant-time during spanning-forest construction.

        Inputs:
            item:
                Integer. Disjoint-set item.

        Outputs:
            representative:
                Integer. Component representative.
        """
        parent = self.parent[item]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while item != parent:
            next_item = self.parent[item]
            self.parent[item] = parent
            item = next_item
        return parent

    def union(self, first, second):
        """Merge two clique components and report whether an edge connected them.
        Returning the change flag lets Kruskal accept exactly the acyclic edges.

        Inputs:
            first:
                Integer. First disjoint-set item.
            second:
                Integer. Second disjoint-set item.

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
