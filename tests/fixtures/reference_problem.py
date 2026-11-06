"""Five-variable reference problem used by Rhiza integration tests."""

import numpy as np

import rhiza as rz


################################################################################
# Reference problem
################################################################################

REFERENCE_SOLUTION = np.array([
    0.0,
    -1.0,
    -np.sqrt(2.0),
    0.0,
    np.sqrt(3.0),
])
REFERENCE_OBJECTIVE = -14.0 - np.sqrt(2.0) - np.sqrt(3.0)


def make_reference_pop(
    *,
    num_samples=180,
    use_sampled_equalities=True,
    use_sparse_scopes=True,
    objective_form='MIN',
    inequality_form='<=0',
    seed=0,
):
    """Construct the five-variable reference POP.

    Inputs:
        num_samples:
            Integer. Number of sampled points.
        use_sampled_equalities:
            Boolean. Whether samples lie on the equality variety.
        use_sparse_scopes:
            Boolean. Whether to use three overlapping cliques.
        objective_form:
            String. Either 'MIN' or 'MAX'.
        inequality_form:
            String. Either '<=0' or '>=0'.
        seed:
            Integer. Random seed.

    Outputs:
        pop:
            POP. Mutable reference problem.
    """
    num_samples = int(num_samples)
    use_sampled_equalities = bool(use_sampled_equalities)
    use_sparse_scopes = bool(use_sparse_scopes)
    objective_form = str(objective_form).upper()
    inequality_form = str(inequality_form)
    seed = int(seed)

    rng = np.random.default_rng(seed)
    points = np.empty((num_samples, 5))
    if use_sampled_equalities:
        for sample_id in range(num_samples):
            x0 = rng.uniform(-1.0, 1.0)
            x1 = rng.choice((-1.0, 1.0))*np.sqrt(1.0 - x0**2)
            x2 = rng.choice((-1.0, 1.0))*np.sqrt(1.0 - x1)
            points[sample_id] = (
                x0,
                x1,
                x2,
                rng.normal(),
                rng.normal(),
            )
    else:
        points[:] = rng.normal(size=(num_samples, 5))

    variables = tuple(f"x{index}" for index in range(5))
    sparse_scopes = (
        (variables[0], variables[1]),
        (variables[1], variables[2]),
        (variables[2], variables[3], variables[4]),
    )
    dense_scope = (variables,)*3
    scopes = sparse_scopes if use_sparse_scopes else dense_scope

    objective_evals = (
        points[:,0]**2*points[:,1]**2,
        points[:,1]**2*points[:,2]**2
        + points[:,2]**2
        + points[:,2],
        -points[:,2]**2*points[:,3]**2
        - 3.0*points[:,2]**2*points[:,4]**2
        - points[:,4],
    )
    inequality_evals = (
        0.5 - points[:,1]**2,
        -points[:,2]**2 + points[:,3]**2 + points[:,4]**2 - 1.0,
    )
    equality_evals = (
        points[:,0]**2 + points[:,1]**2 - 1.0,
        points[:,1] + points[:,2]**2 - 1.0,
    )
    if use_sampled_equalities:
        equality_evals = (
            np.zeros(num_samples),
            np.zeros(num_samples),
        )

    if objective_form == 'MAX':
        objective_evals = tuple(-values for values in objective_evals)
    if inequality_form == '>=0':
        inequality_evals = tuple(-values for values in inequality_evals)

    pop = rz.POP()
    for variable_id, variable in enumerate(variables):
        pop.add_variable(variable, points[:,variable_id])
    for entry_id, values in enumerate(objective_evals):
        pop.add_objective(
            values,
            scopes[entry_id],
            4,
            form=objective_form,
            name=f"objective_{entry_id}",
        )
    pop.add_constraint(
        inequality_evals[0],
        scopes[0],
        2,
        form=inequality_form,
        name="inequality_0",
    )
    pop.add_constraint(
        inequality_evals[1],
        scopes[2],
        2,
        form=inequality_form,
        name="inequality_1",
    )
    pop.add_constraint(
        equality_evals[0],
        scopes[0],
        2,
        form='==0',
        name="equality_0",
    )
    pop.add_constraint(
        equality_evals[1],
        scopes[1],
        2,
        form='==0',
        name="equality_1",
    )
    return pop


def reference_points(pop):
    """Return the reference POP samples in variable order.

    Inputs:
        pop:
            POP. Mutable or finalized reference problem.

    Outputs:
        points:
            ndarray. Sample-by-variable reference points.
    """
    if pop.finalized:
        return pop._variable_evals.copy()
    return np.column_stack([
        pop._pending_variables[name] for name in pop.variable_names
    ])


def evaluate_reference_objective(point):
    """Evaluate the exact five-variable reference objective.

    Inputs:
        point:
            ArrayLike. Five-variable point.

    Outputs:
        objective:
            Float. Reference objective value.
    """
    point = np.asarray(point, dtype=float)

    return float(
        point[0]**2*point[1]**2
        + point[1]**2*point[2]**2
        + point[2]**2
        + point[2]
        - point[2]**2*point[3]**2
        - 3.0*point[2]**2*point[4]**2
        - point[4]
    )


def evaluate_reference_constraints(point):
    """Evaluate reference inequalities and equalities at one point.

    Inputs:
        point:
            ArrayLike. Five-variable point.

    Outputs:
        inequalities:
            ndarray. Two left-hand sides required to be nonpositive.
        equalities:
            ndarray. Two equality residuals.
    """
    point = np.asarray(point, dtype=float)

    inequalities = np.array((
        0.5 - point[1]**2,
        -point[2]**2 + point[3]**2 + point[4]**2 - 1.0,
    ))
    equalities = np.array((
        point[0]**2 + point[1]**2 - 1.0,
        point[1] + point[2]**2 - 1.0,
    ))
    return inequalities, equalities
