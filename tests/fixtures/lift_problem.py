"""Lifted and unlifted forms of a two-variable reference problem."""

import numpy as np

import rhiza as rz


################################################################################
# Lift reference problem
################################################################################


LIFT_SOLUTION = np.zeros(2)
LIFT_OBJECTIVE = 0.0


def make_lift_problem(*, num_samples=180, use_lifts, seed=11):
    """Construct equivalent lifted or unlifted polynomial problems.

    Inputs:
        num_samples:
            Integer. Number of random box samples.
        use_lifts:
            Boolean. Whether to introduce product and affine lifts.
        seed:
            Integer. Random seed.

    Outputs:
        pop:
            POP. Mutable sampled problem.
    """
    num_samples = int(num_samples)
    use_lifts = bool(use_lifts)
    seed = int(seed)

    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(num_samples, 2))
    if np.min(np.linalg.norm(points, axis=1)) <= 1e-3:
        raise RuntimeError("Lift fixture sampled too near the exact optimizer.")
    x_evals = points[:,0]
    y_evals = points[:,1]

    pop = rz.POP()
    pop.add_variable("x", x_evals)
    pop.add_variable("y", y_evals)
    pop.add_objective(
        (x_evals*y_evals)**2,
        ("x", "y"),
        4,
        name="product_cost",
    )
    pop.add_objective(
        (2.0*x_evals + y_evals)**2,
        ("x", "y"),
        2,
        name="affine_cost",
    )
    pop.add_constraint(
        1.0 - x_evals**2,
        ("x",),
        2,
        form='>=0',
        name="x_bound",
    )
    pop.add_constraint(
        1.0 - y_evals**2,
        ("y",),
        2,
        form='>=0',
        name="y_bound",
    )

    if use_lifts:
        pop.add_product(
            ("x", "y"),
            name="product",
            targets={"product_cost": 2},
        )
        pop.add_lift(
            ("x", "y"),
            ((1, 0), (0, 1)),
            name="affine",
            coeffs=(2.0, 1.0),
            targets={"affine_cost": 2},
        )
    return pop


def lift_points(pop):
    """Return the source-variable samples of a lift problem.

    Inputs:
        pop:
            POP. Mutable or finalized lift problem.

    Outputs:
        points:
            ndarray. Sample-by-source-variable points.
    """
    if pop.finalized:
        ids = [pop._variable_ids['x'], pop._variable_ids['y']]
        return pop._variable_evals[:,ids].copy()
    return np.column_stack((
        pop._pending_variables['x'],
        pop._pending_variables['y'],
    ))
