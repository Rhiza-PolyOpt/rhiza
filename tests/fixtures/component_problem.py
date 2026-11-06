"""Small sampled varieties with degree-visible component structure."""

import numpy as np

import rhiza as rz


################################################################################
# Binary lifted line
################################################################################


def make_binary_component_pop(
    *,
    num_samples=120,
    include_product=True,
    include_constant_inequality=False,
    positive_count=None,
    seed=71,
):
    """Construct two sampled lines distinguished by one binary coordinate.

    With the product coordinate ``bx=b*x``, the degree-one quotient space
    contains component-localized functions on both binary components. Without that
    lift, restricting ``x`` to one branch requires the missing degree-two
    function ``b*x``, so automatic Gramian splitting must decline.
    """
    num_samples = int(num_samples)
    if num_samples <= 8 or num_samples%2:
        raise ValueError("num_samples must be even and exceed eight.")

    rng = np.random.default_rng(seed)
    x_evals = rng.uniform(-1.0, 1.0, size=num_samples)
    if positive_count is None:
        positive_count = num_samples//2
    positive_count = int(positive_count)
    if not 0 < positive_count < num_samples:
        raise ValueError(
            "positive_count must leave samples on both binary components."
        )
    b_evals = np.concatenate((
        np.zeros(num_samples - positive_count),
        np.ones(positive_count),
    ))
    rng.shuffle(b_evals)

    pop = rz.POP()
    pop.add_variable("x", x_evals)
    pop.add_variable("b", b_evals)
    if include_product:
        pop.add_product(("b", "x"), name="bx")
        scope = ("x", "b", "bx")
    else:
        scope = ("x", "b")
    pop.add_objective(
        (x_evals - 0.25)**2 + 0.1*b_evals,
        scope,
        2,
        name="cost",
    )
    if include_constant_inequality:
        pop.add_constraint(
            np.ones(num_samples),
            scope,
            0,
            form='>=0',
            name="positive_constant",
        )
    return pop


################################################################################
# Binary pose-product lift
################################################################################


def make_pose_product_component_pop(*, num_samples=220, seed=79):
    """Construct the 12-coordinate product lift used by robust pose models.

    The binary endpoint components each expose one constant direction and twelve
    affine pose directions. The degree-one quotient basis therefore splits as
    ``26 = 13 + 13``, matching the local structure of the PCR and PnP lifts.
    """
    num_samples = int(num_samples)
    if num_samples <= 200 or num_samples%2:
        raise ValueError(
            "num_samples must be even and exceed 200."
        )

    rng = np.random.default_rng(seed)
    pose_evals = rng.normal(size=(num_samples, 12))
    b_evals = np.repeat((0.0, 1.0), num_samples//2)
    rng.shuffle(b_evals)

    pop = rz.POP()
    pose_names = tuple(f"pose_{index}" for index in range(12))
    product_names = []
    for index, name in enumerate(pose_names):
        pop.add_variable(name, pose_evals[:,index])
    pop.add_variable("b", b_evals)
    for name in pose_names:
        product_name = f"b_{name}"
        pop.add_product(("b", name), name=product_name)
        product_names.append(product_name)

    scope = pose_names + ("b",) + tuple(product_names)
    pop.add_objective(
        np.sum((pose_evals - 0.1)**2, axis=1) + 0.05*b_evals,
        scope,
        2,
        name="pose_cost",
    )
    return pop


################################################################################
# Ternary one-axis mode
################################################################################


def make_ternary_component_pop(*, num_samples=180, seed=73):
    """Construct off, positive, and negative lifted line components."""
    num_samples = int(num_samples)
    if num_samples <= 18 or num_samples%3:
        raise ValueError("num_samples must be divisible by three and exceed 18.")

    rng = np.random.default_rng(seed)
    x_evals = rng.uniform(-1.0, 1.0, size=num_samples)
    patterns = np.tile(
        np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (num_samples//3, 1),
    )
    rng.shuffle(patterns, axis=0)
    positive = patterns[:,0]
    negative = patterns[:,1]

    pop = rz.POP()
    pop.add_variable("x", x_evals)
    pop.add_variable("positive", positive)
    pop.add_variable("negative", negative)
    pop.add_product(("positive", "x"), name="positive_x")
    pop.add_product(("negative", "x"), name="negative_x")
    scope = (
        "x",
        "positive",
        "negative",
        "positive_x",
        "negative_x",
    )
    pop.add_objective(
        (x_evals - 0.1)**2 + 0.05*(positive + negative),
        scope,
        2,
        name="cost",
    )
    return pop
