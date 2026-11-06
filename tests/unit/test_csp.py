"""Tests for correlative sparsity and RIP clique forests."""

from itertools import combinations
import json

import numpy as np
import pytest

from rhiza.csp import (
    CSP,
    _intersect_occurrence_sets,
    assign_entry_hosts,
    build_rip_forest,
    build_variable_clique_occurrences,
    count_overlap_pairs,
    classify_cliques,
    verify_rip,
)
from rhiza.diagnostics import CSPError, RIPError


def _reference_rip(num_variables, cliques):
    """Recognize RIP through a small independent all-pairs construction."""
    weighted_edges = []
    for first, second in combinations(range(len(cliques)), 2):
        weight = len(set(cliques[first]) & set(cliques[second]))
        if weight:
            weighted_edges.append((-weight, first, second))
    weighted_edges.sort()

    parent = list(range(len(cliques)))

    def find(item):
        """Return one disjoint-set root with path compression."""
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    forest = []
    for _, first, second in weighted_edges:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            continue
        parent[second_root] = first_root
        forest.append((first, second))

    try:
        verify_rip(num_variables, cliques, tuple(forest))
    except RIPError:
        return False
    return True


def _maximum_degree(num_nodes, edges):
    """Return the maximum degree of one compact edge tuple."""
    degrees = [0]*num_nodes
    for first, second in edges:
        degrees[first] += 1
        degrees[second] += 1
    return max(degrees, default=0)


################################################################################
# Hand-designed sparsity patterns
################################################################################


def test_chain_csp():
    """Construct the natural clique chain and separators."""
    csp = CSP(
        ('a', 'b', 'c', 'd'),
        (('a', 'b'), ('b', 'c'), ('c', 'd')),
    )
    assert csp.cliques == (
        ('a', 'b'),
        ('b', 'c'),
        ('c', 'd'),
    )
    assert csp.structural_clique_edges == ((0, 1), (1, 2))
    assert csp.structural_separator_scopes == (('b',), ('c',))
    assert csp.entry_hosts == (0, 1, 2)


def test_subsumed_cliques():
    """Classify strict declared subsets separately from maximal declared cliques."""
    csp = CSP(
        ('a', 'b', 'c'),
        (('a',), ('a', 'b'), ('a', 'b', 'c')),
    )
    assert csp.cliques == (('a', 'b', 'c'),)
    assert csp.subsumed_cliques == (('a',), ('a', 'b'))
    assert csp.entry_hosts == (0, 0, 0)


def test_disconnected_clique_forest():
    """Retain disconnected correlative components as a forest."""
    csp = CSP(
        ('a', 'b', 'c', 'd'),
        (('a', 'b'), ('c', 'd')),
    )
    assert csp.structural_clique_edges == ()
    assert csp.stats['num_structural_components'] == 2


def test_rip_failure():
    """Reject a cyclic three-clique family without running intersection."""
    with pytest.raises(RIPError):
        CSP(
            ('a', 'b', 'c'),
            (('a', 'b'), ('b', 'c'), ('a', 'c')),
        )


def test_extend_policy_is_recognized():
    """Raise a clear error for the unavailable extension policy."""
    with pytest.raises(NotImplementedError, match='EXTEND'):
        CSP(('a',), (('a',),), graph_policy='EXTEND')


def test_overlap_accepts_non_rip_cliques():
    """Keep maximal declared cliques without requiring RIP."""
    csp = CSP(
        ('a', 'b', 'c'),
        (('a', 'b'), ('b', 'c'), ('a', 'c')),
        graph_policy='OVERLAP',
    )
    assert csp.cliques == (
        ('a', 'b'),
        ('a', 'c'),
        ('b', 'c'),
    )
    assert not csp.status['rip']
    assert csp.stats['num_candidate_overlap_pairs'] == 3
    assert len(csp.structural_clique_edges) == 2
    assert csp.entry_hosts == (0, 2, 1)


def test_overlap_recognizes_rip_without_requiring_it():
    """Report RIP when an overlap-policy family already satisfies it."""
    csp = CSP(
        ('a', 'b', 'c'),
        (('a', 'b'), ('b', 'c')),
        graph_policy='OVERLAP',
    )
    assert csp.status['rip']
    assert csp.structural_clique_edges == ((0, 1),)


def test_overlap_avoids_pairwise_intersection_materialization():
    """Construct a shared-core family through linear structural topology."""
    variable_names = ('r',) + tuple(f'u{index}' for index in range(20))
    scopes = tuple(('r', f'u{index}') for index in range(20))
    csp = CSP(variable_names, scopes, graph_policy='OVERLAP')
    assert csp.status['rip']
    assert csp.stats['num_candidate_overlap_pairs'] == 190
    assert len(csp.structural_clique_edges) == 19


def test_csp_preserves_literal_name_and_singleton_scope_semantics():
    """Treat bare strings as single names and reject lossy name coercion."""
    csp = CSP('theta', ('theta',))
    assert csp.cliques == (('theta',),)
    assert csp.entry_hosts == (0,)

    with pytest.raises(CSPError, match='strings'):
        CSP(('x', 2), (('x',),))
    with pytest.raises(CSPError, match='whitespace'):
        CSP((' x',), ((' x',),))


def test_overlap_pair_count_scales_with_sparse_incidence():
    """Count a long clique chain without materializing global neighbor sets."""
    num_cliques = 5000
    variable_cliques = [(0,)]
    variable_cliques.extend(
        (index - 1, index) for index in range(1, num_cliques)
    )
    variable_cliques.append((num_cliques - 1,))
    assert count_overlap_pairs(num_cliques, variable_cliques) == num_cliques - 1


def test_csp_exposes_compact_structural_graph_data():
    """Expose structural edges and intersections as immutable tuples."""
    csp = CSP(('a', 'b', 'c'), (('a', 'b'), ('b', 'c')))
    assert csp.structural_clique_edges == ((0, 1),)
    assert csp.structural_separator_scopes == (('b',),)


################################################################################
# Algorithmic reference comparisons
################################################################################


def test_clique_classification_deduplicates_scopes():
    """Ignore repeated declarations while preserving strict subsumption."""
    maximal, subsumed = classify_cliques(
        ((0, 1), (0, 1), (0,), (1, 2))
    )
    assert maximal == ((0, 1), (1, 2))
    assert subsumed == ((0,),)


def test_rip_forest_matches_all_pairs_reference():
    """Match independent all-pairs RIP recognition on a small family."""
    cliques = (
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (0, 4),
    )
    occurrences = build_variable_clique_occurrences(5, cliques)
    edges, rip = build_rip_forest(5, cliques, occurrences)
    assert rip == _reference_rip(5, cliques)
    if rip:
        verify_rip(5, cliques, edges)


def test_csp_result_is_deterministic():
    """Return the same clique forest across repeated constructions."""
    arguments = (
        ('a', 'b', 'c', 'd'),
        (('a', 'b'), ('b', 'c'), ('b', 'd')),
    )
    first = CSP(*arguments)
    second = CSP(*arguments)
    assert first.structural_clique_edges == second.structural_clique_edges
    assert first.structural_separator_scopes == second.structural_separator_scopes


@pytest.mark.parametrize('seed', range(12))
def test_random_scope_helpers_match_direct_references(seed):
    """Compare custom scope algorithms with simple independent references."""
    rng = np.random.default_rng(seed)
    num_variables = 4 + seed%6
    scopes = []
    for _ in range(8 + seed):
        size = int(rng.integers(1, min(5, num_variables) + 1))
        scope = tuple(sorted(
            rng.choice(num_variables, size=size, replace=False).tolist()
        ))
        scopes.append(scope)

    unique = sorted(set(scopes), key=lambda scope: (len(scope), scope))
    maximal_reference = tuple(
        scope for scope in unique
        if not any(set(scope) < set(other) for other in unique)
    )
    subsumed_reference = tuple(
        scope for scope in unique
        if any(set(scope) < set(other) for other in unique)
    )
    maximal, subsumed = classify_cliques(scopes)
    assert maximal == maximal_reference
    assert subsumed == subsumed_reference

    occurrences = build_variable_clique_occurrences(
        num_variables, maximal
    )
    direct_pairs = {
        (first, second)
        for first, second in combinations(range(len(maximal)), 2)
        if set(maximal[first]) & set(maximal[second])
    }
    assert count_overlap_pairs(len(maximal), occurrences) == len(direct_pairs)

    rip_edges, rip = build_rip_forest(
        num_variables, maximal, occurrences
    )
    assert rip == _reference_rip(num_variables, maximal)
    if rip:
        verify_rip(num_variables, maximal, rip_edges)


@pytest.mark.parametrize('seed', range(10))
def test_random_tree_clique_families_satisfy_rip(seed):
    """Recover and verify random clique trees with unique separators."""
    rng = np.random.default_rng(seed)
    num_cliques = 3 + seed%7
    tree_edges = tuple(
        (node, int(rng.integers(0, node)))
        for node in range(1, num_cliques)
    )
    clique_sets = [set((f'u{index}',)) for index in range(num_cliques)]
    for edge_id, (first, second) in enumerate(tree_edges):
        separator = f's{edge_id}'
        clique_sets[first].add(separator)
        clique_sets[second].add(separator)

    maximal_scopes = [tuple(sorted(scope)) for scope in clique_sets]
    declared_scopes = list(maximal_scopes)
    for scope in maximal_scopes:
        if len(scope) > 1 and rng.random() < 0.8:
            declared_scopes.append((scope[0],))
    variable_names = tuple(sorted(set().union(*clique_sets)))
    csp = CSP(variable_names, declared_scopes)

    assert set(csp.cliques) == set(maximal_scopes)
    clique_sets_named = [set(clique) for clique in csp.cliques]
    for entry_scope, host in zip(declared_scopes, csp.entry_hosts):
        assert set(entry_scope).issubset(clique_sets_named[host])

    edges = tuple(csp.structural_clique_edges)
    cliques_ids = tuple(
        tuple(variable_names.index(name) for name in clique)
        for clique in csp.cliques
    )
    verify_rip(len(variable_names), cliques_ids, edges)
    entry_ids = tuple(
        tuple(variable_names.index(name) for name in scope)
        for scope in declared_scopes
    )
    assert assign_entry_hosts(
        len(variable_names), entry_ids, cliques_ids
    ) == csp.entry_hosts

    occurrences = build_variable_clique_occurrences(
        len(variable_names), cliques_ids
    )
    rip_edges, rip = build_rip_forest(
        len(variable_names), cliques_ids, occurrences
    )
    assert rip
    verify_rip(len(variable_names), cliques_ids, rip_edges)

    overlap_csp = CSP(
        variable_names, declared_scopes, graph_policy='OVERLAP'
    )
    assert overlap_csp.status['rip']
    verify_rip(
        len(variable_names),
        cliques_ids,
        overlap_csp.structural_clique_edges,
    )


def test_rip_avoids_pairwise_intersections_and_keeps_shared_core_path():
    """Build a large shared-core RIP forest with linear structural storage."""
    num_cliques = 200
    variable_names = ('r',) + tuple(
        f'u{index}' for index in range(num_cliques)
    )
    scopes = tuple(
        ('r', f'u{index}') for index in range(num_cliques)
    )
    csp = CSP(variable_names, scopes, graph_policy='RIP')

    assert csp.status['rip']
    assert 'num_candidate_overlap_pairs' not in csp.stats
    assert csp.stats['num_structural_clique_edges'] == num_cliques - 1
    assert _maximum_degree(num_cliques, csp.structural_clique_edges) <= 2


def test_overlap_skips_expensive_exact_pair_count():
    """Omit an optional all-pairs diagnostic above its work budget."""
    num_cliques = 1500
    variable_names = ('r',) + tuple(
        f'u{index}' for index in range(num_cliques)
    )
    scopes = tuple(
        ('r', f'u{index}') for index in range(num_cliques)
    )
    csp = CSP(variable_names, scopes, graph_policy='OVERLAP')

    assert 'num_candidate_overlap_pairs' not in csp.stats
    assert csp.stats['num_structural_clique_edges'] == num_cliques - 1


def test_csp_public_reports_are_json_serializable():
    """Expose ordinary Python scalar leaves in CSP reports."""
    csp = CSP(('a', 'b', 'c'), (('a', 'b'), ('b', 'c')))
    json.dumps(csp.status)
    json.dumps(csp.stats)
    json.dumps(csp.summary())


def test_occurrence_intersection_starts_with_rarest_variable():
    """Keep sparse host lookup anchored by the most selective variable."""
    visitation = []

    class _TrackedOccurrences:
        """Record when one synthetic occurrence set is iterated."""

        def __init__(self, name, values):
            """Store a label and immutable synthetic clique IDs."""
            self.name = name
            self.values = tuple(values)

        def __len__(self):
            """Return the synthetic occurrence count."""
            return len(self.values)

        def __iter__(self):
            """Record visitation before yielding synthetic clique IDs."""
            visitation.append(self.name)
            return iter(self.values)

    occurrences = {
        0: _TrackedOccurrences('shared', range(1000)),
        1: _TrackedOccurrences('private', (317,)),
    }

    assert _intersect_occurrence_sets((0, 1), occurrences) == {317}
    assert visitation[0] == 'private'
