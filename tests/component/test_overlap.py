"""Component tests for reduced overlap separator construction."""

from itertools import combinations

import numpy as np
import pytest

import rhiza as rz
import rhiza.sdpr as sdpr_module
from rhiza.sdpr import (
    _build_exponent_patterns,
    _build_edge_adjacency,
    _get_participant_occurrence,
    _prune_overlap_edge_assignments,
    _select_overlap_tree,
)
from rhiza.tools.polyalg import build_separator_basis
from tests.fixtures.overlap_problem import (
    make_overlap_triangle_pop,
    make_shared_core_pop,
)


################################################################################
# Topology and coefficient structure
################################################################################


def test_overlap_triangle_uses_three_function_specific_edges():
    """Retain the three transfers required by a non-RIP triangle."""
    pop = make_overlap_triangle_pop(num_samples=80)
    pop.finalize(graph_policy='OVERLAP')
    sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=1)

    edges = {
        (edge['first_clique_id'], edge['second_clique_id'])
        for edge in sdpr._separator_edges
    }
    assert edges == {(0, 1), (0, 2), (1, 2)}
    assert set(sdpr.separator_edges) == edges
    assert pop.csp.graph_policy.value == 'OVERLAP'
    assert sdpr.stats['num_overlap_occurrence_groups'] == 3
    assert sdpr.stats['num_separator_edges'] == 3
    assert sdpr.stats['num_separator_cycles'] == 1
    assert sdpr.stats['num_separator_directions_full_topological'] == 3
    assert sdpr.stats['num_separator_directions_reduced_topological'] == 3
    assert [
        gramian['gramian_size'] for gramian in sdpr._gramians
    ] == [3, 3, 3]


def test_overlap_degree_two_carries_linear_and_quadratic_directions():
    """Increase each single-variable overlap basis at degree two."""
    pop = make_overlap_triangle_pop(num_samples=80)
    pop.finalize(graph_policy='OVERLAP')
    sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=2)

    separator_ranks = [
        sdpr._bases[edge['basis_id']]['basis_rank']
        for edge in sdpr._separator_edges
    ]
    assert separator_ranks == [2, 2, 2]
    assert sdpr.stats['num_separator_directions_full_topological'] == 6
    assert sdpr.stats['num_separator_directions_reduced_topological'] == 6


def test_shared_core_reduces_quadratic_overlap_copies():
    """Route one shared core space over a linear-size tree."""
    pop = make_shared_core_pop(num_cliques=8, num_samples=80)
    pop.finalize(graph_policy='OVERLAP')
    sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=2)

    assert sdpr.stats['num_overlap_occurrence_groups'] == 1
    assert sdpr.stats['largest_overlap_occurrence'] == 8
    assert pop.csp.stats['num_candidate_overlap_pairs'] == 28
    assert sdpr.stats['num_separator_edges'] == 7
    assert sdpr.stats['num_separator_directions_full_topological'] == 56
    assert sdpr.stats['num_separator_directions_reduced_topological'] == 14
    assert sdpr.stats['num_separator_coeffs'] == 14
    assert sdpr.stats['separator_topological_reduction_ratio'] == 0.25
    # Every tree edge carries the same shared-core function space, so the sampled
    # basis is stored once even though each edge has independent coefficients.
    assert len({
        edge['basis_id'] for edge in sdpr._separator_edges
    }) == 1


def test_overlap_reuses_rip_forest_when_available():
    """Use the existing clique forest for an overlap-policy RIP family."""
    rng = np.random.default_rng(3)
    points = rng.normal(size=(60, 4))
    scopes = (("a", "b"), ("b", "c"), ("c", "d"))
    pop = rz.POP()
    for variable_id, name in enumerate(("a", "b", "c", "d")):
        pop.add_variable(name, points[:,variable_id])
    for entry_id, scope in enumerate(scopes):
        ids = tuple(("a", "b", "c", "d").index(name) for name in scope)
        pop.add_objective(
            np.sum(points[:,ids]**2, axis=1),
            scope,
            2,
            name=f"cost_{entry_id}",
        )
    pop.finalize(graph_policy='OVERLAP')
    sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=2)

    edges = tuple(
        (edge['first_clique_id'], edge['second_clique_id'])
        for edge in sdpr._separator_edges
    )
    assert pop.csp.status['rip']
    assert edges == pop.csp.structural_clique_edges
    assert sdpr.stats['num_separator_cycles'] == 0


def test_separator_topology_can_grow_with_separator_degree():
    """Add an edge required by a joint monomial but not linear directions."""
    scopes = (
        (0, 1),
        (0, 2, 3),
        (0, 2, 4),
        (0, 3, 4),
        (2, 3, 4),
    )
    pop = _make_scope_pop(scopes, num_samples=90, seed=31)
    pop.finalize(graph_policy='OVERLAP')
    degree_one = rz.SDPR(pop, sos_degree=2, sep_degree=1)
    degree_two = rz.SDPR(pop, sos_degree=2, sep_degree=2)

    edges_one = {
        (edge['first_clique_id'], edge['second_clique_id'])
        for edge in degree_one._separator_edges
    }
    edges_two = {
        (edge['first_clique_id'], edge['second_clique_id'])
        for edge in degree_two._separator_edges
    }
    assert edges_two - edges_one == {(1, 4)}


def test_overlap_reports_disconnected_separator_components():
    """Keep independent overlap components disconnected."""
    scopes = (
        (0, 1),
        (1, 2),
        (0, 2),
        (3, 4),
        (4, 5),
    )
    pop = _make_scope_pop(scopes, num_samples=80, seed=37)
    pop.finalize(graph_policy='OVERLAP')
    sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=1)

    assert sdpr.stats['num_separator_components'] == 2
    assert sdpr.stats['num_separator_cycles'] == 1


def test_overlap_merges_multiple_function_spaces_on_one_edge():
    """Combine occurrence-group bases routed across the same clique pair."""
    scopes = (
        (0, 1, 2),
        (0, 1, 3),
        (0, 4),
    )
    pop = _make_scope_pop(scopes, num_samples=80, seed=43)
    pop.finalize(graph_policy='OVERLAP')
    sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=1)

    edge = next(
        edge for edge in sdpr._separator_edges
        if (edge['first_clique_id'], edge['second_clique_id']) == (1, 2)
    )
    assert sdpr._bases[edge['basis_id']]['basis_rank'] == 2


def test_overlap_separator_blocks_cancel_at_two_endpoints():
    """Attach every overlap coefficient block with opposite endpoint signs."""
    pop = make_overlap_triangle_pop(num_samples=80)
    pop.finalize(graph_policy='OVERLAP')
    sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=2)

    for edge in sdpr._separator_edges:
        attachments = []
        for system in sdpr._clique_systems:
            for contribution in system['scalar_contributions']:
                if contribution['block_id'] == edge['block_id']:
                    attachments.append((system['clique_id'], contribution['sign']))
        assert attachments == [
            (edge['first_clique_id'], 1.0),
            (edge['second_clique_id'], -1.0),
        ]


def test_overlap_pruning_removes_an_edge_made_redundant_later():
    """Reroute a broad group after a mandatory pair closes its triangle."""
    groups = (
        {'occurrence': (0, 1, 2)},
        {'occurrence': (0, 2)},
    )
    initial = {
        (0, 1): [0],
        (1, 2): [0],
        (0, 2): [1],
    }

    pruned = _prune_overlap_edge_assignments(groups, initial, 3)
    assert set(pruned) == {(0, 1), (0, 2)}
    assert pruned[(0, 1)] == [0]
    assert pruned[(0, 2)] == [0, 1]
    assert _prune_overlap_edge_assignments(groups, pruned, 3) == pruned


def test_overlap_singleton_group_reuses_its_reduced_basis(monkeypatch):
    """Avoid a second factorization when one group alone uses an edge."""
    calls = []
    original = sdpr_module.build_sampled_basis

    def tracking_build(*args, **kwargs):
        """Record each numerical basis reduction before delegating."""
        calls.append(np.asarray(args[0]).shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(sdpr_module, 'build_sampled_basis', tracking_build)
    pop = make_shared_core_pop(num_cliques=8, num_samples=80)
    pop.finalize(graph_policy='OVERLAP')
    rz.SDPR(pop, sos_degree=2, sep_degree=2)

    # One call reduces the occurrence monomials. Final tree edges reference that
    # basis directly rather than factorizing its orthonormal columns again.
    assert len(calls) == 1


def test_overlap_singleton_group_reuses_basis_for_a_nonbinding_limit(
    monkeypatch,
):
    """Avoid refactorization when the final cap exceeds the group rank."""
    calls = []
    original = sdpr_module.build_sampled_basis

    def tracking_build(*args, **kwargs):
        """Record each numerical basis reduction before delegating."""
        calls.append(np.asarray(args[0]).shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(sdpr_module, 'build_sampled_basis', tracking_build)
    pop = make_shared_core_pop(num_cliques=8, num_samples=80)
    pop.finalize(graph_policy='OVERLAP')
    rz.SDPR(
        pop,
        sos_degree=2,
        sep_degree=2,
        basis_rank_limit=100,
    )

    assert len(calls) == 1


def test_overlap_singleton_group_refactors_for_a_final_rank_limit(monkeypatch):
    """Apply an explicit final basis cap even to one occurrence group."""
    calls = []
    original = sdpr_module.build_sampled_basis

    def tracking_build(*args, **kwargs):
        """Record each numerical basis reduction before delegating."""
        calls.append(np.asarray(args[0]).shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(sdpr_module, 'build_sampled_basis', tracking_build)
    pop = make_shared_core_pop(num_cliques=8, num_samples=80)
    pop.finalize(graph_policy='OVERLAP')
    rz.SDPR(
        pop,
        sos_degree=2,
        sep_degree=2,
        basis_rank_limit=1,
    )

    assert len(calls) == 2


def test_overlap_participant_helpers_reuse_exponents_and_occurrences():
    """Cache exponent patterns by width and participant occurrences."""
    patterns = _build_exponent_patterns(2, 3)
    assert ((0, 1), (1, 1)) in patterns
    assert ((0, 1), (2, 1)) in patterns

    variable_cliques = ({0, 1, 2}, {1, 2}, {2})
    cache = {}
    first = _get_participant_occurrence((0, 1), variable_cliques, cache)
    second = _get_participant_occurrence((0, 1), variable_cliques, cache)
    assert first == (1, 2)
    assert second is first
    assert cache == {(0, 1): (1, 2)}


def test_overlap_tree_reuses_large_existing_chain():
    """Reuse a broad structural chain without enumerating clique pairs."""
    occurrence = tuple(range(5000))
    available_edges = {
        (clique_id, clique_id + 1)
        for clique_id in range(len(occurrence) - 1)
    }

    adjacency = _build_edge_adjacency(len(occurrence), available_edges)
    tree_edges = _select_overlap_tree(occurrence, adjacency)

    assert len(tree_edges) == len(occurrence) - 1
    assert set(tree_edges) == available_edges


def test_overlap_basis_rank_limit_applies_to_final_edge_bases():
    """Apply the quotient-basis limit after merging edge function spaces."""
    pop = make_overlap_triangle_pop(num_samples=80)
    pop.finalize(graph_policy='OVERLAP')
    sdpr = rz.SDPR(
        pop,
        sos_degree=2,
        sep_degree=2,
        basis_rank_limit=1,
    )

    separator_bases = [
        sdpr._bases[edge['basis_id']]
        for edge in sdpr._separator_edges
    ]
    assert all(basis['basis_rank'] == 1 for basis in separator_bases)
    assert all(basis['basis_rank_limited'] for basis in separator_bases)


################################################################################
# Full-overlap transfer-span equivalence
################################################################################


@pytest.mark.parametrize(
    'scopes',
    (
        ((0, 1), (1, 2), (0, 2)),
        ((0, 1), (1, 2), (2, 3), (0, 3)),
        ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
        ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    ),
)
def test_reduced_overlap_spans_full_pairwise_transfers(scopes):
    """Match the conceptual full-overlap transfer space."""
    pop = _make_scope_pop(scopes, num_samples=90, seed=19)
    pop.finalize(graph_policy='OVERLAP')
    sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=2, rank_tol=1e-11)

    full = _full_overlap_transfer_matrix(pop, degree=2, rank_tol=1e-11)
    reduced = _reduced_transfer_matrix(sdpr)
    assert _matrix_rank(full, 1e-10) == _matrix_rank(reduced, 1e-10)
    assert _span_residual(reduced, full) < 1e-10


def test_reduced_overlap_span_survives_sampled_quotient_relations():
    """Preserve declared-overlap transfers under cross-monomial dependence."""
    scopes = ((0, 1), (1, 2), (0, 2))
    pop = _make_scope_pop(scopes, num_samples=90, seed=29)
    pop._pending_variables["x1"] = pop._pending_variables["x0"].copy()
    pop.finalize(graph_policy='OVERLAP')
    sdpr = rz.SDPR(pop, sos_degree=2, sep_degree=2, rank_tol=1e-11)

    full = _full_overlap_transfer_matrix(pop, degree=2, rank_tol=1e-11)
    reduced = _reduced_transfer_matrix(sdpr)
    assert _matrix_rank(full, 1e-10) == _matrix_rank(reduced, 1e-10)
    assert _span_residual(reduced, full) < 1e-10


################################################################################
# Test helpers
################################################################################


def _make_scope_pop(scopes, *, num_samples, seed):
    """Construct one generic POP from integer variable scopes."""
    num_variables = 1 + max(max(scope) for scope in scopes)
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(num_samples, num_variables))
    names = tuple(f"x{index}" for index in range(num_variables))

    pop = rz.POP()
    for variable_id, name in enumerate(names):
        pop.add_variable(name, points[:,variable_id])
    for entry_id, scope in enumerate(scopes):
        scope_names = tuple(names[index] for index in scope)
        pop.add_objective(
            np.sum(points[:,scope]**2, axis=1),
            scope_names,
            2,
            name=f"cost_{entry_id}",
        )
    return pop


def _full_overlap_transfer_matrix(pop, *, degree, rank_tol):
    """Materialize every pairwise declared-overlap transfer direction."""
    cliques = tuple(
        tuple(pop._variable_ids[name] for name in clique)
        for clique in pop.csp.cliques
    )
    num_samples = pop.num_samples
    columns = []
    for first, second in combinations(range(len(cliques)), 2):
        scope_ids = tuple(sorted(set(cliques[first]) & set(cliques[second])))
        if not scope_ids:
            continue
        basis = build_separator_basis(
            pop._variable_evals[:,scope_ids],
            degree,
            rank_tol=rank_tol,
        )
        for direction in range(basis.basis_rank):
            column = np.zeros(len(cliques)*num_samples)
            values = basis.basis_evals[:,direction]
            column[first*num_samples:(first + 1)*num_samples] = values
            column[second*num_samples:(second + 1)*num_samples] = -values
            columns.append(column)
    if not columns:
        return np.zeros((len(cliques)*num_samples, 0))
    return np.column_stack(columns)


def _reduced_transfer_matrix(sdpr):
    """Materialize the retained overlap transfer directions."""
    num_cliques = len(sdpr._clique_systems)
    num_samples = sdpr.pop.num_samples
    columns = []
    for edge in sdpr._separator_edges:
        basis = sdpr._bases[edge['basis_id']]['basis_evals']
        first = edge['first_clique_id']
        second = edge['second_clique_id']
        for direction in range(basis.shape[1]):
            column = np.zeros(num_cliques*num_samples)
            values = basis[:,direction]
            column[first*num_samples:(first + 1)*num_samples] = values
            column[second*num_samples:(second + 1)*num_samples] = -values
            columns.append(column)
    if not columns:
        return np.zeros((num_cliques*num_samples, 0))
    return np.column_stack(columns)


def _matrix_rank(matrix, tol):
    """Return a relative numerical rank for a test matrix."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0
    return int(np.count_nonzero(singular_values > tol*singular_values[0]))


def _span_residual(basis_matrix, target_matrix):
    """Return the relative residual after projection onto a column span."""
    if target_matrix.size == 0:
        return 0.0
    left_vectors, singular_values, _ = np.linalg.svd(
        basis_matrix, full_matrices=False
    )
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 1.0
    rank = int(np.count_nonzero(
        singular_values > 1e-11*singular_values[0]
    ))
    basis = left_vectors[:,:rank]
    residual = target_matrix - basis@(basis.T@target_matrix)
    return float(np.linalg.norm(residual)/np.linalg.norm(target_matrix))
