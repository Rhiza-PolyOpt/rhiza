"""Clique-chain problems for structural and native scaling tests."""

import numpy as np

import rhiza as rz


################################################################################
# Clique-chain construction
################################################################################


def make_chain_pop(
    *,
    num_cliques,
    num_samples=96,
    use_sparse_scopes=True,
    seed=23,
):
    """Construct a fixed-width chain of overlapping polynomial scopes.

    Inputs:
        num_cliques:
            Integer. Number of three-variable cliques.
        num_samples:
            Integer. Number of sampled points.
        use_sparse_scopes:
            Boolean. Whether each objective keeps its local scope.
        seed:
            Integer. Random seed.

    Outputs:
        pop:
            POP. Mutable clique-chain problem.
    """
    num_cliques = int(num_cliques)
    num_samples = int(num_samples)
    use_sparse_scopes = bool(use_sparse_scopes)
    seed = int(seed)

    if num_cliques <= 0:
        raise ValueError("num_cliques must be positive.")
    rng = np.random.default_rng(seed)
    num_variables = 2*num_cliques + 1
    variable_names = tuple(f"x{index}" for index in range(num_variables))
    points = rng.normal(size=(num_samples, num_variables))
    if np.min(np.linalg.norm(points, axis=1)) <= 1e-8:
        raise RuntimeError(
            "Scaling fixture sampled too near the exact optimizer."
        )
    cliques = tuple(
        variable_names[2*index:2*index + 3]
        for index in range(num_cliques)
    )
    dense_scope = variable_names

    pop = rz.POP()
    for variable_id, name in enumerate(variable_names):
        pop.add_variable(name, points[:,variable_id])
    for clique_id, clique in enumerate(cliques):
        ids = tuple(variable_names.index(name) for name in clique)
        objective_evals = np.sum(points[:,ids]**2, axis=1)
        pop.add_objective(
            objective_evals,
            clique if use_sparse_scopes else dense_scope,
            2,
            name=f"cost_{clique_id}",
        )
    return pop
