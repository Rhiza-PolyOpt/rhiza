"""Non-RIP overlap problems for topology and solver integration tests."""

import numpy as np

import rhiza as rz


################################################################################
# Three-clique overlap problem
################################################################################

OVERLAP_SOLUTION = np.array([1.0, -2.0, 0.5])
OVERLAP_OBJECTIVE = 0.0


def make_overlap_triangle_pop(
    *,
    num_samples=180,
    use_dense_scope=False,
    seed=41,
):
    """Construct a quadratic problem on a non-RIP clique triangle.

    Inputs:
        num_samples:
            Integer. Number of sampled points.
        use_dense_scope:
            Boolean. Whether to combine the objective in one dense clique.
        seed:
            Integer. Random seed.

    Outputs:
        pop:
            POP. Mutable overlap problem.
    """
    num_samples = int(num_samples)
    use_dense_scope = bool(use_dense_scope)
    seed = int(seed)

    rng = np.random.default_rng(seed)
    points = rng.normal(size=(num_samples, 3))
    if np.min(np.linalg.norm(
        points - OVERLAP_SOLUTION[None,:], axis=1
    )) <= 1e-3:
        raise RuntimeError("Overlap fixture sampled too near the optimizer.")

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    local_evals = (
        (x - 1.0)**2 + (y + 2.0)**2 + 3.0*x + 4.0*y,
        (z - 0.5)**2 - 4.0*y + 5.0*z,
        -3.0*x - 5.0*z,
    )
    scopes = (
        ("x", "y"),
        ("y", "z"),
        ("x", "z"),
    )

    pop = rz.POP()
    for variable_id, name in enumerate(("x", "y", "z")):
        pop.add_variable(name, points[:,variable_id])
    if use_dense_scope:
        pop.add_objective(
            sum(local_evals),
            ("x", "y", "z"),
            2,
            name="cost",
        )
    else:
        for entry_id, (evals, scope) in enumerate(zip(local_evals, scopes)):
            pop.add_objective(
                evals,
                scope,
                2,
                name=f"cost_{entry_id}",
            )
    return pop


################################################################################
# Shared-core overlap family
################################################################################


def make_shared_core_pop(
    *,
    num_cliques,
    num_samples=96,
    seed=53,
):
    """Construct many two-variable cliques sharing one core variable.

    Inputs:
        num_cliques:
            Integer. Number of maximal declared cliques.
        num_samples:
            Integer. Number of sampled points.
        seed:
            Integer. Random seed.

    Outputs:
        pop:
            POP. Mutable shared-core problem.
    """
    num_cliques = int(num_cliques)
    num_samples = int(num_samples)
    seed = int(seed)

    if num_cliques <= 1:
        raise ValueError("num_cliques must exceed one.")
    rng = np.random.default_rng(seed)
    variable_names = ("r",) + tuple(
        f"u{index}" for index in range(num_cliques)
    )
    points = rng.normal(size=(num_samples, len(variable_names)))

    pop = rz.POP()
    for variable_id, name in enumerate(variable_names):
        pop.add_variable(name, points[:,variable_id])
    for clique_id in range(num_cliques):
        objective_evals = (
            points[:,0]**2/num_cliques
            + points[:,clique_id + 1]**2
        )
        pop.add_objective(
            objective_evals,
            ("r", f"u{clique_id}"),
            2,
            name=f"cost_{clique_id}",
        )
    return pop
