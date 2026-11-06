# Rhiza

Polynomial optimization problems (POPs) arise frequently across engineering, encompassing models with discrete variables, matrix manifold constraints, complementarity conditions, and more.
Rhiza builds sparse sum-of-squares (SOS) relaxations for POPs directly from *black-box sample evaluations* of their objective and constraint functions.
Each relaxation is a semidefinite program (SDP) that is solved by a standard convex optimizer such as SCS or MOSEK.

The solution yields a certified lower bound on the minimum value of the POP objective, as well as a truncated moment sequence from which the POP's *global* minimizers can often be recovered.
If the optimality gap is inadequate, the relaxation can be *systematically tightened* by increasing the SOS degree or by selecting lift variables, which Rhiza folds in without any overhead.

Rhiza achieves scalability by decomposing a correlatively sparse SOS certificate system into local certificate systems coupled only through low-dimensional *separator* polynomials.
This decomposition exposes simple, repeated quotient-ring structure that Rhiza identifies and exploits *automatically* from the samples.
For typical problem families with bounded local structure, this enables time and memory usage to *scale linearly* with overall problem size!
Please see our publication for the mathematical foundation:

> [Practical Semialgebraic Optimization via Sparse Sample-Based Sum-of-Squares Programming](https://arxiv.org/abs/2611.07108)

## Introduction

The general Rhiza workflow is `POP` → `SDPR` → `Solver`.
At each stage, diagnostic stats can be inspected to assess formulation health and pinpoint potential issues.
As a small guiding example, consider the following nonconvex optimization problem:

```math
\begin{aligned}
    \min_{x \in \mathbb{R}^5} \quad
        & \big( x_1^2x_2^2                     \big)
        + \big( x_2^2x_3^2 + x_3^2 + x_3       \big)
        - \big( x_3^2x_4^2 + 3x_3^2x_5^2 + x_5 \big)
        \\[4pt]
    \text{s.t.}\quad &
        \begin{array}{rcl}
            -2x_2^2 + 1                & \le & 0 \\[3pt]
            -x_3^2 + x_4^2 + x_5^2 - 1 & \le & 0 \\[4pt]
            x_1^2 + x_2^2 - 1          & =   & 0 \\[3pt]
            x_2 + x_3^2 - 1            & =   & 0
        \end{array}
\end{aligned}
```

Rhiza can efficiently compute its global solution by leveraging three key properties:

**(1) Semialgebraic structure.** Simply put, the objective and feasible set can be represented exactly by polynomials.
In this case, that is immediately true.

**(2) Correlative sparsity.** Consider a graph where two variables are connected if they appear together in any constraint, or are multiplied together in the objective.
The graph for this problem is,

```text
x1 ── x2 ── x3
            / \
          x4 ─ x5
```

Rhiza does not require this graph to have any particular structure, but the smaller its treewidth, the more efficient Rhiza can be.
Chain and star structures that arise in optimal control and estimation problems are particularly favorable.

**(3) Poised sampling.** Essentially that it is easy to generate random points satisfying the equality constraints.
For this example all we need is,

```python
import numpy as np
rng = np.random.default_rng(0)
S = 40  # this choice will be explained later

x1 = rng.uniform(-1, 1, S)  # choose x1 samples arbitrarily
x2 = rng.choice((-1,1), S) * np.sqrt(1 - x1**2)  # choose x2 samples satisfying x1^2 + x2^2 = 1
x3 = rng.choice((-1,1), S) * np.sqrt(1 - x2)  # choose x3 samples satisfying x2 + x3^2 = 1
x4, x5 = rng.normal(0, 1, (2,S))  # x4 and x5 have no equalities to satisfy, so sample arbitrarily
```

Rhiza does not require the samples to satisfy any equalities, but each equality satisfied enables Rhiza to discover intrinsic algebraic structure that makes the resulting SDP relaxation both tighter and faster to solve.
Perhaps counterintuitively, there is a sense in which POPs with *more* equality constraints are actually easier for Rhiza to solve!
The catch is that the samples have to be *poised*, informally meaning that they sufficiently cover the intended equalities without accidentally satisfying any other low-degree equalities.
This is the core reason for selecting the samples randomly, and also why here we cover both the plus and minus branches of the square-roots.
See our [publication](https://arxiv.org/abs/2611.07108) and [this related work](https://arxiv.org/abs/1511.06751) for a rigorous explanation of poisedness.

Ultimately, Rhiza receives no symbolic expressions.
The POP is specified entirely through objective and constraint evaluations on the samples, together with their variable scopes and total degrees:

```python
import rhiza as rz
pop = rz.POP()

for i, samples in enumerate((x1, x2, x3, x4, x5)):
    pop.add_variable(f"x{i+1}", samples)

#                          evaluations                     scope        degree
pop.add_objective(x1**2*x2**2,                       ('x1','x2',    ),    4)
pop.add_objective(x2**2*x3**2 + x3**2 + x3,          ('x2','x3',    ),    4)
pop.add_objective(-x3**2*x4**2 - 3*x3**2*x5**2 - x5, ('x3','x4','x5'),    4)

pop.add_constraint(-2*x2**2 + 1,                     ('x2',         ),    2, form='<=0')
pop.add_constraint(-x3**2 + x4**2 + x5**2 - 1,       ('x3','x4','x5'),    2, form='<=0')

pop.add_constraint(x1**2 + x2**2 - 1,                ('x1','x2',    ),    2, form='==0')  # evals are 0
pop.add_constraint(x2 + x3**2 - 1,                   ('x2','x3',    ),    2, form='==0')  # evals are 0

pop.finalize(graph_policy='RIP')  # running intersection property (RIP) is helpful but not required
```

The three objective entries sum to the complete objective, while their declared scopes expose the correlative sparsity mentioned earlier.
Since our samples satisfy the equality constraints, those evaluations are all zero.
In fact, here we could omit those two `add_constraint` calls entirely!
However, it is good practice to keep them anyway and let Rhiza decide what to do, since their scope information can be important to the correlative sparsity pattern (CSP).

Now, with a finalized POP object, we can build the SDP relaxation:

```python
sdpr = rz.SDPR(pop, sos_degree=4)  # sos_degree corresponds to 2*d from our paper
sdpr.analyze(show=True)  # optional numerical check of sample quality margin
sdpr.finalize()
```

The number of `POP` samples needed cannot be known exactly until the `SDPR` is constructed, since it depends on the resulting clique structure and sample-based quotient spaces.
In practice, either start with a reasonable guess (hundreds) and use `SDPR.analyze()` to check the margin, or deliberately oversample (thousands) and let `SDPR.compress()` remove the resulting redundancy.
For this problem, `SDPR.analyze()` reveals that we could use as few as $35$ samples, so we chose $40$ for a slight margin.

The relaxation can be systematically tightened by increasing `sos_degree`, though it will take significantly longer to solve and require many more samples.
The more practical option is to add *lift* variables to the POP itself (see `POP.add_lift`), or to enlarge an objective / constraint entry’s declared scope with variables it does not explicitly depend on.
These actions selectively enrich the local SOS certificates that actually need help, rather than uniformly increasing degree across the entire relaxation.

Once we are satisfied with our `SDPR`, we solve it.
Rhiza will by default use the open-source convex optimizer SCS, but MOSEK is also an option.
The solve produces a report and allows us to extract a candidate solution:

```python
with rz.Solver(sdpr, gap_tol=1e-6) as solver:
    report = solver.solve()
    if report['success']:
        bound = report['primal_objective']  # lower bound on POP optimal value (up to numerics)
        means = solver.extract_means()  # means of a distribution that concentrates on minimizers
    else:
        raise RuntimeError(report)

print("Lower Bound:", bound)
print("Solution Candidate:")
for name, mean in means.items():
    print(" ", name, "=", round(mean, 5))

# OUTPUT
# Lower Bound: -17.14630553
# Solution Candidate:
#   x1 =  0.0
#   x2 = -1.0
#   x3 = -1.41421
#   x4 =  0.0
#   x5 =  1.73206
```

This numerical solution clearly identifies the feasible point $`[0, -1, -\sqrt{2}, 0, \sqrt{3}]^\top`$.
Plugging it into the original objective gives $`-(14+\sqrt{2}+\sqrt{3}) \approx -17.14626437`$.
Thus our lower bound certifies global relative suboptimality of at most $`0.00025\%`$.
With no initialization or heuristics, Rhiza attained this numerically certified global solution in about 10 milliseconds on a consumer-grade laptop (i7-12800H).

Please see the [experiments repository](https://github.com/Rhiza-PolyOpt/rhiza_experiments) for real-world applications, including outlier-robust pose estimation and hybrid trajectory optimization with thousands of continuous and discrete variables solved in seconds.
For more details on using Rhiza, please see [the manual](manual.md).
Lastly, it is worth noting for context that if the POP is a dense QCQP (every polynomial is degree 2 with full scope) and the chosen samples don't satisfy any equalities (2-poised for $`\mathbb{R}^n`$), then the Rhiza degree-2 SOS relaxation becomes equivalent to the well-known Shor relaxation for QCQPs.

## Installation

By default, Rhiza uses the open-source package [SCS](https://www.cvxgrp.org/scs/) as its SDP solver backend.
However, we recommend using the commercial solver [MOSEK](https://www.mosek.com/) if you can obtain a license.
The corresponding Rhiza installations are:

```bash
pip install "rhiza-polyopt @ git+https://github.com/Rhiza-PolyOpt/rhiza.git"
# or
pip install "rhiza-polyopt[mosek] @ git+https://github.com/Rhiza-PolyOpt/rhiza.git"
```

Alternatively, for development, `git clone` this repository and run:

```bash
pip install -e ".[test]"
# or
pip install -e ".[test,mosek]"
```

Note that solver backend packages are imported lazily, so a MOSEK license is not required until you call `Solver(..., backend='MOSEK')`.
SCS can always be used instead.

## Testing

The default suite runs every test that does not require a MOSEK license.
The remaining tests can be run with the flag `-m mosek`.

```bash
pytest
pytest -m mosek
```

These should not take more than a few seconds to run.
If the first MOSEK test hangs, it is likely an issue with the license server rather than Rhiza.

## Disclosure

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of War for Research and Engineering under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of War for Research and Engineering.

© 2026 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
