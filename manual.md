# Rhiza Manual

1. [Purpose and workflow](#1-purpose-and-workflow)
2. [Installation and imports](#2-installation-and-imports)
3. [First solve and object lifecycle](#3-first-solve-and-object-lifecycle)
4. [What Rhiza solves](#4-what-rhiza-solves)
5. [Mathematical foundation](#5-mathematical-foundation)
6. [Input requirements and naming rules](#6-input-requirements-and-naming-rules)
7. [POP construction API](#7-pop-construction-api)
8. [Products, polynomial lifts, and target rewrites](#8-products-polynomial-lifts-and-target-rewrites)
9. [The pop file format](#9-the-pop-file-format)
10. [Correlative sparsity and CSP](#10-correlative-sparsity-and-csp)
11. [SDPR construction API and degree policies](#11-sdpr-construction-api-and-degree-policies)
12. [Sampled quotient bases and numerical rank](#12-sampled-quotient-bases-and-numerical-rank)
13. [Local certificate systems](#13-local-certificate-systems)
14. [Objective and constraint normalization](#14-objective-and-constraint-normalization)
15. [Component-localized Gramian splitting](#15-component-localized-gramian-splitting)
16. [RIP and OVERLAP separators](#16-rip-and-overlap-separators)
17. [Analysis, PQR compression, and SDPR finalization](#17-analysis-pqr-compression-and-sdpr-finalization)
18. [Solver API and backend behavior](#18-solver-api-and-backend-behavior)
19. [Moment extraction and diagnostics](#19-moment-extraction-and-diagnostics)
20. [Rounding helpers](#20-rounding-helpers)
21. [Status, statistics, reports, and exceptions](#21-status-statistics-reports-and-exceptions)
22. [Sparse binary example](#22-sparse-binary-example)
23. [Certificate validity and interpretation](#23-certificate-validity-and-interpretation)
24. [Common failure modes and troubleshooting](#24-common-failure-modes-and-troubleshooting)
25. [Repository and module map](#25-repository-and-module-map)
26. [Internal data flow and backend boundary](#26-internal-data-flow-and-backend-boundary)
27. [Development and testing workflow](#27-development-and-testing-workflow)
28. [Scope and limitations](#28-scope-and-limitations)
29. [Glossary](#29-glossary)
30. [Modeling and development checklists](#30-modeling-and-development-checklists)

---

## 1. Purpose and workflow

Rhiza builds sparse sum-of-squares (SOS) relaxations for polynomial optimization from numerical evaluations of the model.

A **sample point** $`z_s`$ assigns a value to every decision variable. To describe a problem, provide the variable values at sample points $`z_1,\ldots,z_S`$, together with the values of each objective component and constraint at those same points. Every supplied array must use the same sample order. The samples are used to construct the relaxation; they need not be feasible solutions.

Rhiza calls each objective component or constraint an **entry**. For each entry, provide its total polynomial degree and a **scope**, a list containing all the variables it depends on. A scope may include additional variables, which can affect the relaxation's sparsity and multiplier spaces. A constraint declaration also specifies whether its expression must be nonnegative, nonpositive, or zero.

The workflow has three stages. **POP** stores these model declarations, **SDPR** constructs the sparse semidefinite programming relaxation, and **Solver** builds and solves the SCS or MOSEK model. After solving, moment extraction provides estimates of the decision variables for application-specific rounding and feasibility checks.

Rhiza uses the declared dependencies to organize the problem into local variable groups called **cliques**. Algebraic relations visible in the samples can further reduce each clique's polynomial basis. These two reductions determine much of the SDP size; large remaining positive semidefinite (PSD) blocks can still make solving expensive.

Start with [Installation and imports](#2-installation-and-imports) and the [first solve](#31-first-solve). The following problem conventions and mathematical reminders explain how the inputs become a relaxation. The API sections give exact argument rules, while the later numerical and developer sections cover diagnostics and implementation details. Please see our publication for the theory:

> [Practical Semialgebraic Optimization via Sparse Sample-Based Sum-of-Squares Programming](https://arxiv.org/abs/2611.07108)

## 2. Installation and imports

The distribution name is `rhiza-polyopt`, while the imported package is `rhiza`.

Install the default SCS backend with:

```bash
pip install "rhiza-polyopt @ git+https://github.com/Rhiza-PolyOpt/rhiza.git"
```

Install the optional MOSEK interface with:

```bash
pip install "rhiza-polyopt[mosek] @ git+https://github.com/Rhiza-PolyOpt/rhiza.git"
```

SCS is the default backend and is installed with the base package. MOSEK additionally requires its package and a license. Either backend can use the same finalized SDPR without rebuilding its mathematical spaces.

For editable installations and test dependencies, see [Development installation](#271-development-installation).

The package metadata declares Python 3.8.10 or newer and these minimum dependencies:

- NumPy 1.21;
- SciPy 1.7;
- SCS 3.2.7;
- optional MOSEK 10.2.6;
- test extras pytest 7 and pytest-cov 4.

Native `scs` and `mosek` imports occur lazily inside the selected backend build path. Importing `rhiza` does not require a MOSEK installation or license.

Import Rhiza as:

```python
import rhiza as rz
```

The primary top-level public objects are:

```python
rz.POP
rz.SDPR
rz.Solver
rz.rounding
```

Rhiza also exports its public exception hierarchy and `rhiza.__version__` at package level.

Enums for relations, graph policies, compression methods, and solution statuses live in `rhiza.diagnostics`. The principal APIs also accept the documented strings.

The correlative sparsity pattern class, `CSP`, can also be imported directly:

```python
from rhiza.csp import CSP
```

It is normally constructed internally by `POP.finalize()` and is not part of the root import surface.

## 3. First solve and object lifecycle

Build and finalize a `POP`, construct and finalize an `SDPR`, then use a `Solver` for optimization and moment extraction. Application-specific rounding or recovery follows extraction.

### 3.1 First solve

The following program minimizes $`(x-0.25)^2+(y+0.5)^2`$ over the unit disk $`x^2+y^2\le1`$. Its unique minimizer is $`(0.25,-0.5)`$ and its optimum is zero.

```python
import numpy as np
import rhiza as rz

# x[s] and y[s] describe the same sample point.
rng = np.random.default_rng(0)
x, y = rng.uniform(-1.0, 1.0, size=(2, 30))

pop = rz.POP()
pop.add_variable("x", x)
pop.add_variable("y", y)
pop.add_objective(
    (x - 0.25)**2 + (y + 0.5)**2,
    ("x", "y"), 2, name="cost",
)
pop.add_constraint(
    1.0 - x**2 - y**2,
    ("x", "y"), 2, form='>=0', name="disk",
)
pop.finalize()

sdpr = rz.SDPR(pop, sos_degree=2)
sdpr.finalize()

with rz.Solver(sdpr, backend='SCS') as solver:
    report = solver.solve()
    if not report["success"]:
        raise RuntimeError(report)
    means = solver.extract_means()
    diagnostics = solver.extract_moment_diagnostics()

print(report["primal_objective"], means)
print(diagnostics)
```

The arrays `x` and `y` contain 30 sampled assignments, and each expression is evaluated at those same assignments. Some samples lie outside the disk; the inequality is enforced by the relaxation. The final positional argument `2` declares each expression's polynomial degree. Separately, `sos_degree=2` chooses the SOS relaxation degree $`2d`$.

The reported bound should be close to zero and the means close to `{"x": 0.25, "y": -0.5}` within numerical accuracy. In a general problem, extracted means are only candidate variable estimates. Check the solve report and moment diagnostics, then round if needed and evaluate the original constraints and objective. The [sparse binary example](#22-sparse-binary-example) adds multiple cliques, separators, component splitting, and compression.

### 3.2 POP lifecycle

A `POP` is mutable until `POP.finalize()`. Before finalization, variables and entries can be added, named entries can be removed, and lift helpers can rewrite named target scopes. Finalization validates the complete model, builds its correlative sparsity pattern (CSP), compacts its data into arrays and integer indexes, and permanently locks the object.

Repeated finalization with the same `graph_policy` and `identity_tol` is idempotent and returns `False`. Attempting to change either setting after finalization raises `PipelineError`.

### 3.3 SDPR lifecycle

Constructing `SDPR(pop, ...)` requires a finalized POP. Construction immediately builds the polynomial bases and SDP coefficient data, including Gramian and scalar blocks, objective normalization, and separators. It does not yet create an SCS or MOSEK solver instance.

Before `SDPR.finalize()`, an SDPR may be:

- inspected through properties, `status`, and `stats`;
- checked for numerical rank through `analyze()`;
- compressed through `compress(method='PQR')`, optionally only for cliques at or below a declared size threshold.

`analyze()` is diagnostic and remains available after finalization. Compression changes which sample equations are submitted and is therefore forbidden after finalization. Finalization fixes their indices and scales for every clique, discards construction-only caches, and makes the SDPR ready for a `Solver`.

Repeated SDPR finalization is idempotent and returns `False`.

### 3.4 Solver lifecycle

Constructing `Solver(sdpr, ...)` requires a finalized SDPR and immediately assembles the selected solver model. Backend choice, generic tolerances, iteration limit, and solver-specific configuration are fixed at construction.

A Solver is one-use:

- the first `solve()` performs optimization;
- repeated `solve()` calls return the cached report with `changed=False`;
- extraction requires a successful solve;
- `close()` releases solver resources and copied solution data; repeated closing is harmless;
- after `close()`, neither solving nor extraction is allowed.

Context-manager use is preferred:

```python
with rz.Solver(sdpr) as solver:
    report = solver.solve()
    means = solver.extract_means()
```

## 4. What Rhiza solves

Rhiza represents a polynomial optimization problem using the convention

```math
\mathcal F
\coloneqq
\left\{
 x\in\mathbb R^n
 \;\middle|\;
 g_{1:m}(x)\ge0,\;
 h_{1:\ell}(x)=0
\right\},
```

and

```math
f^*
\coloneqq
\min_{x\in\mathcal F} f(x).
```

### 4.1 Notation conventions

| Symbol | Meaning |
|---|---|
| $`x=(x_1,\ldots,x_n)`$ | Scalar decision coordinates. |
| $`f(x)`$ and $`f^*`$ | Canonical minimization objective and its optimum. |
| $`\mathcal F`$ | Feasible semialgebraic set. |
| $`g_{1:m}(x)\ge0`$ | Canonical inequality notation in the exposition. |
| $`h_{1:\ell}(x)=0`$ | Equality constraints. |
| $`\mathcal V=\{x\in\mathbb R^n\mid h_{1:\ell}(x)=0\}`$ | Equality variety when all equalities are regarded as defining relations. |
| $`\mathcal Z`$ | The common ordered sample points $`z_1,\ldots,z_S`$. |
| $`K`$ and $`C_k`$ | Number of maximal declared cliques and the variables in clique $`k`$. |

We use $`\mathbb N=\{0,1,\ldots\}`$, $`a_{i:j}`$ for an ordered tuple, and $`x_C`$ for a coordinate subvector. Specialized basis, multiplier, and separator notation is introduced where those objects are constructed.

The API parameter `sos_degree` equals $`2d`$. Mathematical clique indices run from $`1`$ to $`K`$; public clique IDs are zero-based.

### 4.2 Sampled representation

Rhiza does not receive symbolic formulas for $`f`$, $`g_i`$, or $`h_j`$. It receives their values at one common ordered collection of sample points

```math
\mathcal Z
\coloneqq
\left(z_1,\ldots,z_S\right),\qquad z_s\in\mathbb R^n.
```

Repeated points are allowed and count separately. For clique $`C_k`$, the projected samples $`\mathcal Z_{C_k}=(z_{1,C_k},\ldots,z_{S,C_k})`$ retain the common order.

The samples often lie on the intended equality variety $`\mathcal V`$ or on an easier ambient equality variety, but they need not satisfy the inequalities. For every scalar coordinate function $`x_a`$, Rhiza stores

```math
x_a(\mathcal Z)
\coloneqq
\left(
 x_a(z_1),\ldots,x_a(z_S)
\right)^\top
\in\mathbb R^S.
```

Each objective component or constraint similarly contributes one vector in $`\mathbb R^S`$, together with its scope, total polynomial degree, relation, and optional name. Stacking evaluation vectors as columns gives an evaluation matrix whose row $`s`$ corresponds to sample point $`z_s`$. All declarations must preserve this correspondence; Rhiza does not align independently sampled arrays.

### 4.3 Canonical objective

Rhiza always solves one canonical minimization problem. If the POP contains `MIN` entries $`f_a^{\mathrm{min}}`$ and `MAX` entries $`f_b^{\mathrm{max}}`$, then

```math
f(x)
=
\sum_a f_a^{\mathrm{min}}(x)
-
\sum_b f_b^{\mathrm{max}}(x).
```

A `MAX` entry contributes with a negative sign; it does not produce a separate maximization report. All public primal and dual objectives use the original units of this single signed minimization objective.

### 4.4 Constraint conventions

The public constraint forms are:

- `'>=0'` for an inequality in the exposition's convention $`g_i(x)\ge0`$;
- `'<=0'` for an inequality supplied in the opposite orientation;
- `'==0'` for an equality $`h_j(x)=0`$.

The exposition's $`g_i`$ denotes the sign-normalized inequality polynomial. For a supplied polynomial $`c_i`$, Rhiza uses $`g_i=c_i`$ for a public `'>=0'` entry and $`g_i=-c_i`$ for a public `'<=0'` entry. Thus $`g_i\ge0`$ describes the feasible side, although its sampled values may have either sign.

An equality is classified as a **sampled identity** when

```math
\max_{s=1,\ldots,S}
\left|h_j(z_s)\right|
\le
\texttt{identity\_tol}.
```

Sampled identities receive no explicit equality multiplier; other equalities receive free polynomial multipliers. Basis construction infers quotient relations from the sampled variable values, so an equality declaration alone does not establish a quotient relation.

An inequality that is exactly zero at every supplied sample is rejected at POP finalization. Boundary-only samples cannot distinguish its feasible side.

### 4.5 What Rhiza returns

A successful solve reports a numerical SOS lower-bound estimate as `primal_objective`, the corresponding dual SDP objective as `dual_objective`, and a normalized relative gap. Solver-specific diagnostics describe numerical accuracy.

Moment extraction then provides first moments as estimates of variable values, together with requested higher moments. When a variable belongs to several cliques, Rhiza averages their local estimates and reports their disagreement. These estimates may require rounding and need not satisfy the original constraints. They are derived from signed linear functionals, described in [Moment extraction and diagnostics](#19-moment-extraction-and-diagnostics); the sample weights are not probabilities.

Rhiza does not reconstruct symbolic certificates or prove that the samples characterize the intended variety. Interpreting the numerical bound requires the checks in [Certificate validity and interpretation](#23-certificate-validity-and-interpretation).

## 5. Mathematical foundation

### 5.1 Sampled polynomial-function spaces

Fix a maximal clique scope $`C_k`$ and degree $`d`$. Let

```math
\nu_d(x_{C_k})
\in
\mathbb R^{\varpi(|C_k|,d)}
```

be the column vector of all ambient monomials in the variables $`x_{C_k}`$ with total degree at most $`d`$, where

```math
\varpi(|C_k|,d)
=
\binom{|C_k|+d}{d}.
```

Evaluating these monomials at the common sample points gives the local monomial evaluation matrix

```math
E_d(\mathcal Z_{C_k})
=
\begin{bmatrix}
 \nu_d(z_{1,C_k})^\top\\
 \vdots\\
 \nu_d(z_{S,C_k})^\top
\end{bmatrix}
\in
\mathbb R^{S\times \varpi(|C_k|,d)}.
```

Monomial evaluation columns can be dependent because of equality relations or insufficient sampling. Rhiza equilibrates the columns of $`E_d(\mathcal Z_{C_k})`$, computes an SVD, and retains numerical singular directions. The resulting basis can be represented mathematically by a coefficient matrix

```math
B_k^{(d)}
\in
\mathbb R^{\varpi(|C_k|,d)\times r_k(d)}
```

such that the corresponding reduced polynomial basis is

```math
\varphi_k^{(d)}(x_{C_k})
=
\big(B_k^{(d)}\big)^\top\nu_d(x_{C_k}),
```

and its sampled evaluation matrix is

```math
U_k^{(d)}
=
E_d(\mathcal Z_{C_k})B_k^{(d)}
\in
\mathbb R^{S\times r_k(d)}.
```

The columns of $`U_k^{(d)}`$ are orthonormal to numerical precision. The retained basis size $`r_k(d)`$ equals the empirical numerical rank unless `basis_rank_limit` deliberately removes directions. The implementation stores $`U_k^{(d)}`$ and its numerical diagnostics, not $`B_k^{(d)}`$ or symbolic formulas for the components of $`\varphi_k^{(d)}`$.

For a basis $`\varphi_k^{(d)}`$ and a PSD Gramian $`G\succeq0`$, the corresponding SOS polynomial is evaluated at the sample points by

```math
\sigma(z_s)
=
\varphi_k^{(d)}(z_{s,C_k})^\top
G
\varphi_k^{(d)}(z_{s,C_k}),
\qquad s=1,\ldots,S.
```

At each sample point, this quadratic form is linear in the entries of $`G`$. Its coefficients provide the data needed to assemble the SDP without expanding the model polynomials symbolically.

### 5.2 Equality ideals and ambient-variety sampling

Let

```math
\mathcal I(\mathcal V)
\coloneqq
\left\{
 p\in\mathbb R[x_{1:n}]
 \;\middle|\;
 p(x)=0\ \text{for every }x\in\mathcal V
\right\}
```

be the vanishing ideal of the equality variety. Poisedness on the represented polynomial spaces means that a polynomial vanishing on the samples also vanishes on $`\mathcal V`$. Under this condition, equal evaluation vectors represent the same polynomial function on $`\mathcal V`$.

Rhiza can instead sample an easier containing variety $`\mathcal W\supseteq\mathcal V`$. Relations that hold on $`\mathcal W`$ enter the quotient representation, subject to poisedness. Equalities omitted from $`\mathcal W`$ remain explicit through free multipliers, using the sampled-identity classification in [Constraint conventions](#44-constraint-conventions). This trades easier sampling for additional multiplier variables.

### 5.3 Local sparse certificate identity

Let $`C_{1:K}`$ be the maximal declared clique scopes, and decompose the canonical objective as

```math
f(x)
=
\sum_{k=1}^{K} f_k(x_{C_k}),
```

where each objective entry is assigned to one containing clique. This assigned clique is its **host**; the deterministic assignment rule is given in [Entry hosting](#102-entry-hosting).

Let $`\mathcal G_k`$ and $`\mathcal H_k`$ index all inequality and equality entries hosted by clique $`C_k`$, and let $`J`$ index the equalities retained through explicit multipliers. Thus $`\mathcal H_k\cap J`$ indexes the hosted equalities that are not sampled identities.

Let $`\mathcal E`$ be the separator edge set and $`C_e=C_k\cap C_l`$ for $`e=\{k,l\}`$. A nonconstant separator basis $`\psi_e`$ and free coefficient vector $`\gamma_e`$ parameterize the polynomial $`\gamma_e^\top\psi_e`$ on that overlap. Rhiza's local sampled certificate has the form

```math
f_k-\beta_k
=
\sigma_{k,0}
+
\sum_{i\in\mathcal G_k}g_i\,\sigma_{k,i}
+
\sum_{j\in\mathcal H_k\cap J}h_j\,\lambda_{k,j}
+
\sum_{e\in\mathcal E:\,k\in e}\varepsilon_{k,e}\,\gamma_e^\top\psi_e.
```

Here:

- $`\beta_k`$ is clique $`k`$'s scalar share of the global lower bound;
- $`\sigma_{k,0}`$ is the base SOS polynomial;
- $`\sigma_{k,i}`$ is an SOS multiplier for a hosted inequality;
- $`\lambda_{k,j}`$ is a free polynomial multiplier for a hosted equality that is not a sampled identity;
- $`\psi_e`$ is a quotient basis on $`C_e`$, with free coefficients $`\gamma_e`$ for edge $`e`$;
- $`\varepsilon_{k,e}\in\{-1,+1\}`$ gives opposite signs at the two endpoints.

At endpoint $`k`$ of $`e=\{k,l\}`$, define the directed separator $`\tau_{kl}=\varepsilon_{k,e}\gamma_e^\top\psi_e`$, so $`\tau_{lk}=-\tau_{kl}`$. Set $`\tau_{kl}=0`$ when $`\{k,l\}\notin\mathcal E`$, including $`k=l`$. Writing

```math
\zeta_k=f_k-\beta_k-\sigma_{k,0}-\sum_{i\in\mathcal G_k}g_i\sigma_{k,i}-\sum_{j\in\mathcal H_k\cap J}h_j\lambda_{k,j},
```

the same local equation is $`\zeta_k=\sum_{l:\,\{k,l\}\in\mathcal E}\tau_{kl}`$. These identities are imposed at the supplied samples; extending them to the intended variety requires poisedness.

The global bound is

```math
\beta
=
\sum_{k=1}^{K}\beta_k.
```

Each entry contributes at one host clique. Summing the local identities cancels every separator and recovers the global sampled certificate.

### 5.4 Variables used by constraint multipliers

A multiplier basis is built on the **declared scope of its constraint entry**, not automatically on the full host clique. If a constraint depends only on $`(x_a,x_b)`$ but is hosted in a larger clique $`(x_a,x_b,x_c)`$, its multiplier remains a polynomial in $`(x_a,x_b)`$.

Enlarging a declared scope can strengthen the multiplier family while increasing quotient ranks, Gramian dimensions, separator coefficient counts, and backend cost. Restricted scopes may give weaker fixed-degree bounds than full-clique multiplier families. Convergence claims must account for the scopes actually used.

### 5.5 Bound and moment duality

The certificate SDP maximizes $`\beta=\sum_k\beta_k`$. Its dual has one linear functional per clique. After reversing backend row scaling, clique $`k`$'s signed sample weights $`y_{k,1:S}`$ define

```math
\mathcal L_k[p]
=
\sum_{s=1}^{S}
y_{k,s}\,p(z_{s,C_k}).
```

Here $`p`$ is a polynomial on $`C_k`$. Weights on rows removed by compression are zero; Rhiza stores only retained indices and weights. Dividing by $`\mathcal L_k[1]`$ gives normalized clique-local moments. Shared moments agree in an exact solution when their polynomial directions are carried by the separator spaces; a restricted separator degree or rank cap can omit directions. Extraction arithmetically averages the local values and also exposes the individual clique values and disagreement diagnostics.

## 6. Input requirements and naming rules

The following requirements apply throughout the public API.

### 6.1 Real float64 data only

All public sample arrays and real scalar settings are inspected **before** conversion to `float64`. Complex dtypes and object arrays containing complex values are rejected rather than silently discarding imaginary parts.

Every sample vector must be:

- one-dimensional;
- nonempty;
- finite;
- convertible to real `float64`;
- the same length as every other variable and entry vector in the POP.

The first successfully added variable or entry establishes the shared sample count. Entry scopes may reference variables not yet added, but all names must exist by finalization.

### 6.2 Exact integer metadata

Degrees, exponents, clique IDs, iteration limits, axes, and similar integral metadata use exact conversion. Integer-valued real scalars such as `2.0` are accepted. Fractions, booleans, NaNs, infinities, and values that would require truncation are rejected.

Declared POP degrees must be nonnegative and fit the compact signed 32-bit integer range used after finalization.

### 6.3 Literal string names

Variable names, entry names, moment names, and requested variable names must be literal strings. Rhiza does not stringify arbitrary objects such as `None` or integers.

POP variable and named-entry names share one global namespace. A name must be:

- nonempty;
- free of leading or trailing whitespace;
- free of commas, hyphens, carriage returns, and newlines;
- globally unique among variables and named entries.

The comma and hyphen restrictions ensure that every valid in-memory name can be represented unambiguously in the `.pop` exchange format.

### 6.4 Scope conventions

A scope may be a string or an iterable of strings. A bare string is treated as a singleton scope:

```python
scope="theta"        # means ("theta",), not characters t, h, e, t, a
```

Scopes must be nonempty and cannot repeat a variable. The caller's order is retained during mutable construction, although finalized compact scopes use stable variable-ID order internally.

### 6.5 Transactional mutation

Builder operations validate their full request before committing. Failed variable insertion, lift construction, target rewriting, or POP finalization leaves the previous builder state intact.

## 7. POP construction API

`POP` stores the sampled problem. Finalization locks its data for CSP and SDPR construction.

### 7.1 Constructor and inspection

```python
pop = rz.POP(path=None)
```

| Argument | Accepted values and default |
|---|---|
| `path` | `None` creates an empty mutable builder. Any non-`None` value is converted with `str(path)` and loaded as a `.pop` file. Loading uses the same validation path as programmatic construction and does **not** finalize the POP. |

Public properties are:

| Property | Meaning |
|---|---|
| `finalized` | Whether the POP has been finalized and locked. |
| `csp` | The finalized `CSP`; accessing it before finalization raises `PipelineError`. |
| `status` | A copy of qualitative lifecycle fields. |
| `stats` | A copy of size and count statistics. |
| `num_samples` | Shared sample count, including during mutable construction. |
| `variable_names` | Variable names in stable insertion or finalized storage order. |
| `entry_names` | Names of named objectives and constraints in entry order; anonymous entries are omitted. |

`str(pop)` gives a readable status and statistics summary. `repr(pop)` gives a compact lifecycle-focused representation without expanding sample arrays.

### 7.2 Adding variables

```python
added = pop.add_variable(name, evals)
```

| Argument | Meaning and validation |
|---|---|
| `name` | Required literal string. It must satisfy the global POP naming rules and share no name with any variable or named entry. |
| `evals` | Required real, finite, nonempty one-dimensional array-like object. It is converted to `float64` after complex input is rejected. Its length must equal the established sample count. |

The first successful variable or entry insertion establishes the common sample count. The method copies the evaluation vector and returns `True`. Variables cannot later be removed because arbitrary entries may depend on them.

### 7.3 Adding objective entries

```text
added = pop.add_objective(
    evals,
    scope,
    degree,
    *,
    form='MIN',
    name=None,
)
```

| Argument | Meaning and default |
|---|---|
| `evals` | Sampled objective values on the common rows. The same shape, reality, finiteness, and sample-count rules as variable evaluations apply. |
| `scope` | A nonempty string or iterable of literal variable names. A string is a singleton scope. Repeats are prohibited. Objective and constraint scopes may forward-reference variables that will be added before finalization. |
| `degree` | Exact nonnegative total polynomial degree, within the compact signed 32-bit range used after finalization. Integer-valued real scalars are accepted; booleans and fractional values are not. |
| `form` | Defaults to `'MIN'`. The value is stringified and uppercased, then must be `'MIN'` or `'MAX'`. `MIN` contributes the sampled vector to $`f`$; `MAX` contributes its negative. |
| `name` | Defaults to `None`, meaning anonymous. A supplied value must be a valid literal POP name. Named entries can be removed or targeted by lift rewrites before finalization. |

The declared degree and scope are trusted modeling metadata. Rhiza does not infer them from numerical variation in `evals`. The method returns `True` after insertion.

### 7.4 Adding constraints

```text
added = pop.add_constraint(
    evals,
    scope,
    degree,
    *,
    form,
    name=None,
)
```

| Argument | Meaning and default |
|---|---|
| `evals` | Sampled constraint values on the common rows. |
| `scope` | Nonempty string or iterable of variable names. It may forward-reference variables until finalization. The declared scope determines the multiplier scope. |
| `degree` | Exact nonnegative total polynomial degree within the supported compact range. |
| `form` | Required keyword-only argument. It is converted with `str(form)` and must then equal `'>=0'`, `'<=0'`, or `'==0'`. No case normalization is applied to these symbolic tokens. |
| `name` | Defaults to `None`. A supplied name must be a valid unused literal POP name. |

Choose the scope deliberately. Including additional variables enlarges the multiplier space and can strengthen the relaxation, but also increases correlative coupling. The method returns `True` after insertion.

### 7.5 Removing named entries

```python
removed = pop.remove(name)
```

| Argument | Meaning |
|---|---|
| `name` | Required literal string looked up in the shared POP namespace. |

The method returns `True` when it removes a named objective or constraint and `False` when the name is unknown. It raises `POPValidationError` when the name belongs to a variable. Anonymous entries cannot be removed by name. Any call after finalization raises `PipelineError`.

### 7.6 POP finalization

```python
changed = pop.finalize(
    graph_policy='RIP',
    identity_tol=None,
)
```

| Argument | Meaning and default |
|---|---|
| `graph_policy` | Defaults to `'RIP'`. A string or `GraphPolicy` is accepted; strings are case-insensitive. Implemented values are `'RIP'` and `'OVERLAP'`. `'EXTEND'` is recognized but raises `NotImplementedError`. |
| `identity_tol` | Defaults to `None`, which resolves to $`64\,\varepsilon_{\mathrm{float64}}`$. An explicit value must be real, finite, and nonnegative. It is an absolute threshold. |

A successful first finalization returns `True`. It requires:

- at least one variable, one entry, and one objective;
- every scope variable to exist;
- every declared variable to be used by at least one entry;
- every inequality to be exactly nonzero on at least one sample row;
- a valid CSP under the selected graph policy.

Finalization applies the [sampled-identity test](#44-constraint-conventions), compacts arrays, constructs the CSP, and locks the POP. Failure leaves the POP mutable.

Repeating `finalize()` with the same resolved settings returns `False`; changing either setting raises `PipelineError`.

After finalization, `pop.status` contains `finalized` and `graph_policy`; `pop.stats` contains sample, variable, entry, objective, constraint, sampled-identity, and clique counts.

### 7.7 Persistence

```python
saved = pop.save(path)
```

| Argument | Meaning |
|---|---|
| `path` | Required output path; converted with `str(path)`. |

Saving requires a finalized POP and returns `True`. The writer emits variables first, then entries in stable order. Reloading preserves sampled data, relations, scopes, and degrees; entry names are not serialized, so loaded objectives and constraints are anonymous.

## 8. Products, polynomial lifts, and target rewrites

A polynomial lift introduces a new variable $`z=p(x)`$ so that selected expressions can be written at lower degree in the enlarged variable set. Rhiza computes $`p(x)`$ at the existing sample points and adds the defining equality $`p(x)-z=0`$. The equality is anonymous and vanishes on those samples.

The optional `targets` argument updates the scopes and degrees of named entries that you have expressed using the new variable. Their stored values stay unchanged. Both lift methods validate the entire request before modifying the POP.

### 8.1 Product lifts

```text
added = pop.add_product(
    scope,
    *,
    name=None,
    targets=None,
)
```

| Argument | Meaning and default |
|---|---|
| `scope` | Required nonempty string or iterable of **already existing** source variables. Each source appears once in the product and scope names cannot repeat. |
| `name` | Defaults to `None`. When omitted, the new variable name is the literal concatenation of the source names in caller order. An explicit name is often safer because the generated name can collide or be unclear. |
| `targets` | Defaults to `None`, meaning no entry rewrites. Otherwise it must be convertible with `dict(targets)` from a mapping or an iterable of `(entry_name, replacement_degree)` pairs. Entries must already exist and replacement degrees must be exact nonnegative integers. |

For source variables $`(x_1,\ldots,x_q)`$, the method adds

```math
z=x_1x_2\cdots x_q
```

and an anonymous sampled equality for

```math
x_1x_2\cdots x_q-z=0.
```

The defining identity has scope `(source variables..., new variable)` and degree $`\max(1,q)`$. The product is evaluated directly on the current common sample rows and must remain finite. The method returns `True`.

### 8.2 General polynomial lifts

```text
added = pop.add_lift(
    scope,
    exponents,
    *,
    name,
    coeffs=None,
    targets=None,
)
```

The new coordinate is

```math
z
=
p(x)
=
\sum_{a=1}^{A}c_a x^{\alpha_a}.
```

| Argument | Meaning and default |
|---|---|
| `scope` | Required nonempty scope of already existing source variables. |
| `exponents` | Required rectangular two-dimensional array-like object with one column per source variable and at least one row. Every entry must be an exact nonnegative integer. |
| `name` | Required keyword-only literal name for the new variable. |
| `coeffs` | Defaults to `None`, which resolves to an all-ones vector with one coefficient per exponent row. Otherwise it must be a finite real vector of that exact length. |
| `targets` | Defaults to `None`, meaning no entry rewrites. Otherwise it has the same named-entry-to-replacement-degree contract as `add_product()`. |

Zero-coefficient rows are removed before support checks, evaluation, and degree calculation. Every source variable must appear with a positive exponent in at least one remaining row. This is a syntactic support check; Rhiza does not combine duplicate monomials to detect cancellation.

The defining sampled identity is

```math
p(x)-z=0
```

with scope `(source variables..., new variable)` and degree

```math
\max\!\left(1,\max_a\sum_q\alpha_{a,q}\right).
```

Degree accumulation uses Python integers to avoid overflow before checking POP's compact range. Lift evaluations must be finite. The method returns `True`.

Example:

```python
pop.add_lift(
    ("x", "y"),
    exponents=[
        [2, 0],
        [1, 1],
        [0, 2],
    ],
    coeffs=[1.0, 2.0, 1.0],
    name="x_plus_y_squared",
)
```

This creates the sampled coordinate $`z=x^2+2xy+y^2`$.

### 8.3 Target rewrites

A target mapping has the form

```python
targets={
    "cost": 1,
    "constraint": 2,
}
```

For each target, Rhiza requires:

- the name to exist and refer to an objective or constraint;
- the entire lift source scope to be contained in the target's current scope;
- the new lift variable not already to be present;
- the replacement degree to be an exact nonnegative supported integer.

The target scope is rewritten by removing all source variables, retaining unrelated variables in their previous order, and appending the new lift variable. For example,

```text
("w", "x", "y", "q")
```

becomes

```text
("w", "q", "z")
```

when a lift named `z` replaces source scope `("x", "y")`.

Rhiza does not symbolically prove that the target polynomial factors through the lift. A target rewrite is an explicit user assertion. Its sampled values are not changed, so an incorrect rewrite can silently change the declared polynomial space and therefore the intended relaxation.

### 8.4 Structural effect of a lift

A well-designed lift can:

- lower the declared degree of important objectives or constraints;
- convert term structure into explicit correlative structure;
- expose repeated nonlinear expressions as degree-one coordinates;
- make finite sample components visible to the component-splitting test at a lower represented degree;
- reduce local quotient rank or Gramian dimension at the desired SOS degree.

A lift adds a variable and a defining identity, often enlarging or adding a clique. Evaluate its benefit through local basis ranks, clique sizes, PSD coefficients, separator spaces, sample margins, and bound quality. See [Designing lifts for tightness at lower SOS degrees](#302-designing-lifts-for-tightness-at-lower-sos-degrees) for a modeling checklist.

## 9. The pop file format

The `.pop` format is a line-oriented, CSV-like exchange format:

```text
relation,degree,scope,eval_1,eval_2,...
```

Blank physical lines are ignored. There is no header or comment syntax.

### 9.1 Relation tokens

Canonical file tokens are:

```text
var
min
max
>=0
<=0
==0
```

Alphabetic relation tokens are parsed case-insensitively, so `VAR`, `Min`, and `MAX` are accepted. Internal enum spellings such as `EQ`, `GE`, and `LE` are not file-format tokens.

### 9.2 Variable rows

A variable row must have:

- relation `var`;
- degree `1`;
- a singleton scope.

Example:

```text
var,1,x,0,0.5,1
```

### 9.3 Entry rows

Every non-`var` row is an objective or constraint entry. Example:

```text
min,2,x-y,1,0.25,0
>=0,2,x-y,0.5,0.25,0
==0,2,x,0,0,0
```

Objective and constraint names are not representable in this format.

### 9.4 Scope grammar

Scopes use hyphens as delimiters:

```text
x-y-z
```

Names in a scope must be nonempty and unique. The parser deliberately does not strip scope whitespace that the writer would reject. For example, a scope field with leading whitespace, trailing whitespace, or spaces adjacent to an internal name is invalid.

### 9.5 Numerical grammar

The numerical tail accepts ordinary finite decimal and scientific notation, including signs and decimal points. Examples include:

```text
0
-3.5
.25
2.
1e-6
-4.2E+3
```

It rejects:

- NaN and infinities;
- malformed suffixes that NumPy might otherwise parse partially;
- empty evaluation fields.

Use a period as the decimal separator and no digit grouping. Commas always separate sample values: `1,000` denotes two values, `1` and `0`, while `1,25` denotes `1` and `25`.

Every row must have the same number of evaluations.

### 9.6 Canonical writer

`POP.save()` writes finite float64 values with 17 significant decimal digits using `.17g`. This is sufficient to round-trip finite IEEE-754 binary64 values. It emits:

- exactly one relation per line;
- `\n` line endings;
- no wrapping or array staggering;
- no locale-dependent formatting.

### 9.7 Loading and errors

```python
pop = rz.POP("problem.pop")
```

The loader applies lexical parsing first and then the same semantic builder validation used by the Python API. Lexical and POP semantic validation errors include the source path and, when available, the physical line number. CSP failures such as `RIPError` are reported separately.

Input declarations may appear in any physical order. Entry scopes can forward-reference variable rows that occur later in the file, provided every referenced variable exists by finalization and all rows share one sample count. The canonical writer nevertheless emits all variables first for readability and interoperability.

The loaded POP remains mutable. A typical file workflow is:

```python
pop = rz.POP("problem.pop")
pop.finalize(graph_policy='RIP')
```

## 10. Correlative sparsity and CSP

Every entry's declared scope is treated as internally fully connected. Distinct declared scopes induce the clique family from which Rhiza builds its correlative sparsity pattern.

### 10.1 Maximal declared cliques

Rhiza first deduplicates declared scopes. A **maximal declared clique** is a distinct declared scope that is not strictly contained in another distinct declared scope. A **subsumed clique** is a distinct declared scope strictly contained in a maximal declared clique.

These are maximal members of the declared scope family. Taking maximal cliques of the union of pairwise variable edges can produce different scopes.

For example, declared scopes

```text
(a, b)
(b, c)
(a, c)
```

remain three pair cliques. Rhiza does not replace them by `(a, b, c)` merely because their union graph is a triangle. This distinction is particularly important under `graph_policy='OVERLAP'`.

### 10.2 Entry hosting

Each objective or constraint is hosted by the containing maximal declared clique with the smallest ID. It contributes once to the certificate.

Clique IDs are zero-based positions in `pop.csp.cliques` and `sdpr.cliques`.

### 10.3 Main CSP properties

After POP finalization:

```python
csp = pop.csp
```

Public CSP properties are:

| Property | Meaning |
|---|---|
| `graph_policy` | Controlled graph-policy enum. |
| `cliques` | Maximal declared scopes as name tuples. |
| `subsumed_cliques` | Strictly subsumed distinct scopes. |
| `structural_clique_edges` | RIP forest edges or OVERLAP structural-backbone edges. |
| `structural_separator_scopes` | Variable intersections on structural edges. |
| `entry_hosts` | Host clique ID for each POP entry in entry order. |
| `status` | Qualitative graph guarantees. |
| `stats` | Structural counts and clique sizes. |

`csp.summary()` returns:

```python
{
    'status': csp.status,
    'stats': csp.stats,
}
```

`csp.status` contains:

```text
graph_policy
rip
```

`csp.stats` contains:

```text
num_variables
num_declared_cliques
num_cliques
num_subsumed_cliques
num_structural_clique_edges
num_structural_components
clique_size_min
clique_size_max
```

Under OVERLAP, it may also contain `num_candidate_overlap_pairs`. This diagnostic is omitted when estimating all pairwise overlaps would exceed the built-in work budget.

### 10.4 RIP policy

```python
pop.finalize(graph_policy='RIP')
```

RIP requires the maximal declared clique family to satisfy the running intersection property. Equivalently, for each variable, the cliques containing that variable must form a connected subtree of the clique forest.

Rhiza constructs a deterministic clique forest by maximum-cardinality search on variable-to-clique incidence, then independently verifies RIP before committing it. A disconnected clique family is valid and produces a forest with multiple connected components.

If RIP fails, finalization raises `RIPError`. Rhiza does not silently extend or merge cliques under this policy.

Under RIP, SDPR constructs separators along the structural forest, omitting edges whose separator space has numerical rank zero. Each retained edge carries the nonconstant sampled space on its clique intersection, subject to the configured degree and rank settings.

### 10.5 OVERLAP policy

```python
pop.finalize(graph_policy='OVERLAP')
```

OVERLAP retains the maximal declared cliques and does not require RIP or perform chordal extension.

CSP first asks whether the declared clique family already satisfies RIP. If it does, OVERLAP reuses the verified RIP forest. Otherwise, CSP creates a deterministic acyclic structural backbone from variable occurrence sets. This backbone is only a starting point. SDPR later constructs degree-dependent separator spaces and may add edges or retain a cyclic union of function-specific trees.

The two edge collections serve different roles:

- `CSP.structural_clique_edges` describes the degree-independent forest or backbone;
- `SDPR.separator_edges` describes the final degree-dependent separator topology.

Before any deliberate basis rank truncation, OVERLAP routing preserves the sampled transfer span for the constructed shared function spaces. Validity on the intended POP still depends on the sample-qualification assumptions.

### 10.6 EXTEND policy

`'EXTEND'` is recognized as a graph-policy token but is not implemented. Attempting to use it raises `NotImplementedError`. Rhiza currently performs no automatic chordal extension.

### 10.7 Direct CSP construction

Advanced users can construct a CSP directly:

```python
from rhiza.csp import CSP

csp = CSP(
    variable_names=("x", "y", "z"),
    entry_scopes=(("x", "y"), ("y", "z")),
    graph_policy='RIP',
)
```

| Argument | Meaning and default |
|---|---|
| `variable_names` | Required bare string or iterable of literal strings. A bare string denotes one variable. The sequence must be nonempty, unique, and free of empty names or outer whitespace. Direct CSP construction does not apply POP's comma, hyphen, or newline restrictions because it has no `.pop` serialization contract. |
| `entry_scopes` | Required collection of at least one scope. A bare outer string denotes one singleton scope; each individual scope may likewise be a string or iterable. Every scope must be nonempty, contain no repeated variable, and refer only to `variable_names`. Be careful that `entry_scopes=("x", "y")` means two singleton scopes; one two-variable scope is `entry_scopes=(("x", "y"),)`. |
| `graph_policy` | Defaults to `'RIP'`. A string or `GraphPolicy` is accepted case-insensitively. `'RIP'` and `'OVERLAP'` are implemented; `'EXTEND'` raises `NotImplementedError`. |

Construction immediately classifies maximal and subsumed scopes, chooses structural edges, and assigns entry hosts. `summary()` takes no arguments and returns `{'status': csp.status, 'stats': csp.stats}`.

`POP.finalize()` also enforces namespace, entry, sample, and serialization consistency, making it the usual modeling entry point. For structural development, standalone functions in `rhiza.csp` provide clique classification, occurrence indexing, RIP construction and verification, overlap counting, and host assignment.

## 11. SDPR construction API and degree policies

`SDPR` compiles a finalized POP into a sparse semidefinite relaxation for either backend.

### 11.1 Constructor

```python
sdpr = rz.SDPR(
    pop,
    sos_degree=2,
    mul_degree=None,
    sep_degree=None,
    rank_tol=1e-10,
    basis_rank_limit=None,
    enable_component_splitting=True,
)
```

| Argument | Meaning and default |
|---|---|
| `pop` | Required finalized `POP`. Constructing from an unfinalized POP raises `PipelineError`. |
| `sos_degree` | Defaults to `2`. It must be an exact nonnegative even integer and bounds the polynomial degree of each logical base SOS term. |
| `mul_degree` | Defaults to `None`, selecting automatic degree fitting separately for inequality SOS multipliers and free equality multipliers. An explicit value must be an exact nonnegative integer. |
| `sep_degree` | Defaults to `None`, which uses `sos_degree`. An explicit value must be an exact nonnegative integer. |
| `rank_tol` | Defaults to `1e-10`. It must be a real finite nonnegative scalar. The value is a relative numerical-rank tolerance; zero is accepted and the implementation still applies a dimension-dependent floating-point floor. |
| `basis_rank_limit` | Defaults to `None`, meaning no deliberate rank cap. Otherwise it must be an exact positive integer. It changes the represented spaces when the cap removes directions and is experimental. |
| `enable_component_splitting` | Defaults to `True`. It must be a genuine Boolean or NumPy Boolean, not a merely truthy integer or object. When enabled, candidate sample-space decompositions are split only after numerical invariance, rank, and margin checks pass. |

Construction immediately builds the bases and coefficient records without creating an SCS or MOSEK model. Before `finalize()` fixes the retained sample equations and their scaling, `analyze()` can inspect ranks and `compress()` can select equations.

### 11.2 Read-only configuration properties

| Property | Meaning |
|---|---|
| `pop` | Finalized source POP. |
| `sos_degree` | Even degree bound for each logical base SOS polynomial. |
| `mul_degree` | Explicit multiplier policy or `None` for automatic fitting. |
| `sep_degree` | Degree of separator polynomial spaces. Defaults to `sos_degree`. |
| `rank_tol` | Common relative numerical-rank tolerance. |
| `basis_rank_limit` | Experimental retained-rank cap or `None`. |
| `enable_component_splitting` | Whether validated logical Gramians may split into component blocks. |
| `finalized` | Whether backend row maps are locked. |
| `compressed` | Whether a compression request has completed, even if it selected no cliques or removed no rows. |
| `separator_edges` | Final separator clique pairs. |
| `cliques` | Maximal declared clique scopes in clique-ID order. |
| `status` | Qualitative formulation diagnostics. |
| `stats` | Structural and numerical formulation statistics. |

### 11.3 Base SOS degree

Write

```math
2d=\texttt{sos\_degree}.
```

Each logical base SOS polynomial uses a reduced quotient basis $`\varphi_k^{(d)}`$ for degree-at-most-$`d`$ functions on clique $`C_k`$, subject to numerical rank selection and any deliberate rank cap. Thus `sos_degree=0`, `2`, and `4` use constant, degree-one, and degree-two Gramian bases, respectively.

### 11.4 Automatic inequality multiplier degree

Suppose an inequality has declared degree $`d_g`$. With `mul_degree=None`, Rhiza computes

```math
a=\texttt{sos\_degree}-d_g.
```

If $`a<0`$, construction raises `DegreeError`. Otherwise the inequality receives the largest even multiplier degree no greater than $`a`$:

```math
d_{\sigma}=a-(a\bmod 2).
```

The SOS multiplier Gramian basis has degree $`d_{\sigma}/2`$. Its product with the inequality stays within `sos_degree` under this automatic policy. The manuscript's common-basis exposition uses degree-$`d`$ Gramian bases for all multipliers; Rhiza's automatic fitting can use smaller bases.

### 11.5 Automatic equality multiplier degree

For an equality not classified as a sampled identity, automatic fitting uses its declared degree $`d_h`$ to set

```math
d_{\lambda}
=
\texttt{sos\_degree}-d_h.
```

A negative result raises `DegreeError`. The equality multiplier need not be SOS, so its degree can be odd. Rhiza stores its basis coefficients directly, rather than using the manuscript's symmetric-matrix parameterization $`\lambda_j=\varphi^\top H_j\varphi`$.

Sampled identities receive no multiplier and do not participate in this degree check.

### 11.6 Explicit multiplier degree

When `mul_degree` is an integer:

- equality multipliers use it directly;
- inequality SOS multipliers round it down to the nearest even integer.

Explicit multipliers can produce residual terms above `sos_degree`; Rhiza does not clip them. Their degrees contribute to `required_sep_degree` and the `sep_degree_insufficient` diagnostic.

### 11.7 Separator degree

`sep_degree=None` means:

```python
sep_degree = sos_degree
```

Rhiza computes a conservative `required_sep_degree` equal to the largest degree that may appear in any local residual after accounting for:

- the base SOS degree;
- every objective's declared degree;
- every inequality degree plus its resolved multiplier degree;
- the degree of each equality represented by an explicit multiplier, plus that multiplier's degree.

The manuscript's $`\kappa`$ bounds local residual degrees. `required_sep_degree` is its conservative counterpart based on declared degrees; `sep_degree` is the requested separator degree. The diagnostic

```python
sdpr.status['sep_degree_insufficient']
```

is `True` exactly when

```math
\texttt{sep\_degree}<\texttt{required\_sep\_degree}.
```

Construction still succeeds when this flag is true. A lower separator degree restricts cross-clique communication and can change the relaxation.

### 11.8 Rank tolerance

`rank_tol` is a finite nonnegative, dimensionless relative singular-value threshold. The same setting governs:

- quotient-basis ranks;
- separator-basis ranks;
- complete clique-system ranks;
- PQR compression rank estimates;
- component-splitting checks, with the tolerance defined in [Acceptance conditions](#153-acceptance-conditions).

### 11.9 Experimental basis rank limit

`basis_rank_limit` caps final registered bases **after** empirical rank is measured. Removing directions changes the represented spaces and may change the relaxation. This is an experimental approximation control:

- `status['basis_rank_limited']` becomes true if any registered basis is truncated;
- sample margin is still computed from the untruncated empirical rank, so truncation cannot make sampling appear better qualified;
- a rank-limited Gramian basis is not component-split because truncation may destroy an exact direct-sum structure;
- the setting does not cap every temporary intermediate basis formed during OVERLAP occurrence-group assembly.

### 11.10 Initial SDPR status and statistics

Immediately after construction, `sdpr.status` contains:

```text
sample_margin_check
basis_rank_limited
sep_degree_insufficient
```

`sdpr.stats` contains:

```text
sos_degree
sep_degree
required_sep_degree
num_samples
num_bases
num_cliques
num_gramians
enable_component_splitting
num_component_split_gramians
num_scalar_coeffs
num_separator_coeffs
num_psd_coeffs
num_psd_coeffs_without_component_splitting
num_rows_original
num_rows_retained
objective_offset
objective_scale
constraint_scale_range
```

plus separator-topology statistics described in [RIP and OVERLAP separators](#16-rip-and-overlap-separators).

Before compression, `num_rows_retained` equals `num_rows_original` even though explicit row arrays are installed only at finalization.

## 12. Sampled quotient bases and numerical rank

### 12.1 Ambient exponent enumeration

For clique $`C_k`$ of width $`n_k=|C_k|`$ and degree $`d`$, Rhiza enumerates all exponent vectors

```math
\alpha\in\mathbb N^{n_k},
\qquad
|\alpha|\le d,
```

in deterministic graded order, starting with the constant and then linear terms when $`d\ge1`$. The ambient monomial count is

```math
\binom{n_k+d}{d}.
```

This count may be much larger than the sampled quotient rank when the samples lie on an algebraic variety.

### 12.2 Evaluation and column equilibration

The ambient monomial vector $`\nu_d`$ is evaluated on all $`S`$ common rows to form $`E_d(\mathcal Z_{C_k})`$. Each nonzero column is divided by its stable Euclidean norm before SVD. A zero column remains zero and receives a harmless unit scale.

This equilibration reduces the effect of monomial units on numerical rank selection.

### 12.3 Stable norms

Rhiza rescales each vector by its largest magnitude before computing its Euclidean norm, avoiding intermediate square overflow or underflow. Nonfinite inputs or a final norm outside the `float64` range raise `NumericalRankError`.

### 12.4 SVD rank rule

Rhiza uses an economy SVD with LAPACK `GESVD`. Let

```math
\varsigma_1\ge\varsigma_2\ge\cdots
```

be the singular values of the equilibrated matrix. The effective relative threshold is

```math
\tau_{\mathrm{eff}}
=
\max\!\left(
\texttt{rank\_tol},
32\,\varepsilon_{\mathrm{float64}}\max(p,q)
\right)
```

for a $`p\times q`$ matrix. The apparent rank is the number of singular values satisfying

```math
\frac{\varsigma_i}{\varsigma_1}>\tau_{\mathrm{eff}}.
```

The cutoff is strict. A zero matrix has rank zero; an optional `basis_rank_limit` is applied after measuring this apparent rank.

### 12.5 Basis diagnostics

Each registered basis records its scope, degree, separator flag, retained rank, condition, singular gap, sample margin, and whether a rank cap removed directions. For retained rank $`r>0`$, the condition number is

```math
\mathop{\mathrm{cond}}\nolimits_{\mathrm{ret}}=\frac{\varsigma_1}{\varsigma_r},
```

and, when $`\varsigma_{r+1}`$ exists, the singular gap is

```math
\mathop{\mathrm{gap}}\nolimits_{\mathrm{sv}}=\frac{\varsigma_r}{\varsigma_{r+1}}.
```

The condition is infinite at rank zero. The gap is `None` if there is no retained-to-discarded boundary and infinite when the first discarded singular value is zero. Component-localized records inherit the parent's condition and gap; these fields do not come from a separate component SVD.

### 12.6 Sample margin

Sample margin counts rows beyond the measured numerical rank:

- an ordinary basis has margin $`S-r_{\mathrm{emp}}`$, where $`r_{\mathrm{emp}}`$ is measured before any deliberate rank cap;
- a component-localized basis has margin $`S_a-r_a`$, using that component's sample count and rank;
- a complete clique system has margin $`S-\rho_k`$ before row selection.

A positive margin records numerical row redundancy; zero means that rank consumed every available row. Neither result alone establishes poisedness on the intended variety.

### 12.7 Separator bases remove constants

Rhiza centers candidate separator columns before rank reduction. Constant transfers are redundant because each clique has its own scalar bound share, so separator bases retain only nonconstant directions.

### 12.8 Basis caching

During SDPR construction, ordinary quotient bases are cached by:

```text
(sorted scope IDs, degree, separator flag)
```

Identical requests reuse a sampled basis. Finalization discards construction caches while retaining the basis and coefficient records needed by backends.

## 13. Local certificate systems

### 13.1 One system per maximal declared clique

Every maximal declared clique $`C_k`$ receives one local sampled equation system. It contains:

- one scalar bound share $`\beta_k`$;
- one logical base SOS Gramian;
- all objective entries hosted by the clique;
- all explicit constraint multipliers hosted by the clique;
- every incident separator block.

Component splitting may replace one logical Gramian by several submitted PSD blocks; [Section 15](#15-component-localized-gramian-splitting) describes the corresponding statistics.

### 13.2 Scalar coefficient blocks

Free scalar coefficient blocks have three conceptual kinds:

- `BOUND`: one scalar per clique;
- `EQUALITY_MULTIPLIER`: coefficients of a free polynomial $`\lambda_{k,j}`$;
- `SEPARATOR`: the vector $`\gamma_e`$ parameterizing a free transfer polynomial $`\gamma_e^\top\psi_e`$.

All free scalars share one global index. The scalar objective maximizes the sum of bound shares; other free scalars have zero objective coefficient.

### 13.3 PSD Gramian blocks

The two logical Gramian sources are:

- the base SOS polynomial for every clique;
- an SOS multiplier for every hosted inequality.

For a basis of rank $`r`$, an unsplit symmetric Gramian has

```math
\frac{r(r+1)}2
```

independent lower-triangle coefficients. For a clique basis $`\varphi_k`$, Rhiza represents the sampled quadratic form

```math
\varphi_k(z_{s,C_k})^\top
G
\varphi_k(z_{s,C_k}),
\qquad s=1,\ldots,S,
```

with diagonal coefficient columns $`\varphi_{k,a}^2`$ and off-diagonal columns $`2\varphi_{k,a}\varphi_{k,b}`$ for $`a>b`$.

The manuscript's product matrix $`\Phi_k(\mathcal Z_{C_k})`$ has rows $`\mathop{\mathrm{svec}}\nolimits(\varphi_k\varphi_k^\top)^\top`$, evaluated at each sample. Its inner-product-preserving `svec` convention uses $`\sqrt{2}`$ off the diagonal. Rhiza uses raw lower-triangle Gramian entries, hence the factor $`2`$ above; backend adapters translate to native solver conventions. The sampled basis matrix $`U_k`$ contains basis values, not these products.

### 13.4 Inequality multipliers

For the convention $`g_i(x)\ge0`$, the certificate contains $`+g_i\sigma_i`$. Rhiza converts public inequalities to this orientation:

- a public `'>=0'` entry keeps its normalized evaluations;
- a public `'<=0'` entry negates its normalized evaluations.

This vector multiplies the SOS multiplier's Gramian coefficient columns. Its sample values can have either sign because sampling need not satisfy the inequalities. The positive normalization divisor preserves orientation.

### 13.5 Equality multipliers

An equality that is not classified as a sampled identity receives a free polynomial multiplier on its declared scope. Multiplying the basis for that multiplier by the normalized equality evaluations produces ordinary free scalar coefficient columns.

A sampled identity receives no explicit multiplier because its evaluations are within `identity_tol`. This classification does not modify the variable samples or impose an additional relation during basis construction.

### 13.6 Augmented clique systems

Write the local sampled equations as $`A_k\theta_k=b_k`$, where $`\theta_k`$ collects the local decision coefficients and $`b_k`$ is the local objective evaluation vector in the normalized units used by SDPR. Rank analysis and PQR operate on the **augmented** local matrix $`\overline A_k=[A_k\ b_k]`$, containing:

- every free scalar coefficient column;
- every scalarized Gramian coefficient column;
- the normalized objective evaluation column.

Including $`b_k`$ makes row-span preservation apply to the affine equations. Rhiza estimates $`\rho_k=\mathop{\mathrm{rank}}\nolimits\overline A_k`$ after balancing; the sample margin is $`S-\rho_k`$. The manuscript's $`D_k`$ is the dimension of the adjusted-residual space on the intended variety. Equating $`\rho_k`$ with $`D_k`$ requires poisedness.

### 13.7 Chunked coefficient generation

Backends receive coefficient evaluations in chunks targeting approximately one million array elements. MOSEK streams them into its native task. SCS assembles a complete sparse CSC cone matrix without forming a global dense coefficient array.

## 14. Objective and constraint normalization

Before clique-system rank analysis and backend assembly, Rhiza normalizes constraints and applies a common affine objective transformation. These transformations preserve the sampled relaxation in exact arithmetic and improve numerical scale.

### 14.1 Explicit constraint normalization

For every multiplier-bearing equality or inequality $`c`$ with sampled evaluations $`c(z_1),\ldots,c(z_S)`$, Rhiza computes

```math
\alpha_c
=
\max_{s=1,\ldots,S}
\left|c(z_s)\right|
>0
```

and uses

```math
\widehat c(z_s)
=
\frac{c(z_s)}{\alpha_c},
\qquad s=1,\ldots,S.
```

POP evaluations remain unchanged. Rhiza divides the constraint vector before multiplying by basis values, avoiding a potentially unrepresentable reciprocal. A free equality multiplier absorbs this scaling; for inequalities, the positive divisor preserves the SOS cone and orientation.

`sdpr.stats['constraint_scale_range']` is:

- `None` when there are no explicit multiplier-bearing constraints;
- otherwise `(minimum_scale, maximum_scale)` over their $`\alpha_c`$ values.

Sampled identities are omitted before normalization. An all-zero inequality is rejected by POP rather than given an arbitrary scale.

### 14.2 Per-clique objective centering

Let $`f_k`$ be the canonical signed objective contribution hosted by clique $`C_k`$. Rhiza computes its sampled midpoint safely as

```math
\delta_k
=
\frac12\min_{s=1,\ldots,S} f_k(z_{s,C_k})
+
\frac12\max_{s=1,\ldots,S} f_k(z_{s,C_k}).
```

The sum of halves avoids overflow in `0.5*(min+max)` for large finite endpoints. The centered objective is

```math
f_k(z_{s,C_k})-\delta_k.
```

Each offset is absorbed into the corresponding bound share $`\beta_k`$.

### 14.3 One common objective scale

Rhiza then computes

```math
\alpha_f
=
\max_k\max_{s=1,\ldots,S}
\left|f_k(z_{s,C_k})-\delta_k\right|.
```

If every centered objective is zero, it uses $`\alpha_f=1`$. Every clique is scaled by the **same** positive value:

```math
\widehat f_k(z_{s,C_k})
=
\frac{f_k(z_{s,C_k})-\delta_k}{\alpha_f}.
```

The common scale preserves relative objective weights across cliques. A normalized backend bound $`\widehat\beta`$ is restored as

```math
\beta
=
\alpha_f\widehat\beta
+
\sum_k\delta_k.
```

The inverse-map values are exposed as:

```text
sdpr.stats['objective_scale']
sdpr.stats['objective_offset']
```

Both primal and dual objectives are restored. The public `relative_gap` uses normalized objectives; each backend also applies its native stopping criteria.

### 14.4 Limits of affine normalization

Normalization cannot recover variation already rounded away in the supplied `float64` evaluations, for example in `large_constant + tiny_variation`.

## 15. Component-localized Gramian splitting

Enabled by default, component splitting replaces a logical Gramian by smaller PSD blocks when numerical tests support a decomposition across disjoint sample-row groups.

### 15.1 Mathematical criterion

Let

```math
U_k\in\mathbb R^{S\times r_k}
```

be an orthonormal sampled quotient-basis matrix for one logical Gramian on clique $`C_k`$. Suppose the sample rows are partitioned into components. For component $`a`$, let $`D_a`$ be the diagonal mask that keeps rows in that component and zeros all others.

The quotient space is restriction-invariant for component $`a`$ when

```math
D_a\mathop{\mathrm{col}}\nolimits(U_k)
\subseteq
\mathop{\mathrm{col}}\nolimits(U_k).
```

Define

```math
P_a=U_k^\top D_aU_k.
```

In exact arithmetic, restriction invariance gives

```math
D_aU_k=U_kP_a,
\qquad
P_a^2=P_a,
\qquad
P_a=P_a^\top,
\qquad
\sum_a P_a=I.
```

The one-eigenspace of each $`P_a`$ gives a localized basis $`U_{k,a}`$ that vanishes outside component $`a`$, with

```math
\mathop{\mathrm{col}}\nolimits(U_k)
=
\mathop{\mathrm{col}}\nolimits(U_{k,1})
\oplus\cdots\oplus
\mathop{\mathrm{col}}\nolimits(U_{k,q}).
```

Only one component basis is nonzero at each sample, so cross-component products vanish. One PSD matrix on $`U_k`$ and independent PSD matrices on the $`U_{k,a}`$ blocks therefore generate the same sampled SOS evaluations.

### 15.2 Candidate partitions

Rhiza proposes partitions from coordinates in the Gramian scope with two to eight exactly distinct sampled values. It tests individual coordinates, then cumulative joint partitions with at most sixteen observed groups. This is a bounded search, not an enumeration of all coordinate subsets. Joint labels can expose one-hot or mutually exclusive modes.

Scanning stops once a coordinate exceeds the eight-value cap. Counts are cached by global variable ID and reused across scopes.

### 15.3 Acceptance conditions

A candidate is accepted only when:

- every component mask passes the restriction-invariance residual check;
- symmetric projector candidates have spectra near zero and one;
- localized component ranges are disjoint;
- component ranks sum to the parent rank;
- every component has positive sample margin;
- the split reduces scalarized PSD coefficient count;
- the parent basis was not deliberately rank-limited.

The restriction-invariance residual is

```math
\frac{\left\|D_aU_k-U_k P_a\right\|_F}
{\max\!\left(1,\left\|D_aU_k\right\|_F\right)}.
```

The invariance and span checks use tolerance

```math
\max\!\left(100\,\varepsilon_{\mathrm{float64}}\max(1,S,r_k),\ \min(10\,\texttt{rank\_tol},10^{-8})\right).
```

The cap applies only to the `rank_tol` contribution; the dimension-dependent floating-point floor remains.

### 15.4 Degree visibility matters

Finite-valued variables alone do not imply that the represented degree-bounded space splits. For example,

```math
\mathop{\mathrm{span}}\nolimits\{1,x,b\}
```

may fail to isolate $`x`$ on the branch $`b=1`$ because $`bx`$ is absent. If a product lift exposes $`bx`$ as a degree-one coordinate, the same sampled geometry may become splittable at that degree.

### 15.5 Where splitting applies

Splitting applies to base SOS and inequality-multiplier Gramians. Free equality multipliers and separators have signed coefficients and no PSD blocks to split.

### 15.6 Statistics

Relevant fields are:

```text
enable_component_splitting
num_component_split_gramians
num_gramians
num_psd_coeffs
num_psd_coeffs_without_component_splitting
```

`num_component_split_gramians` counts logical Gramians represented by multiple submitted blocks. `num_gramians` counts actual submitted PSD blocks.

Disable splitting for regression or backend comparison with:

```python
sdpr = rz.SDPR(
    pop,
    sos_degree=2,
    enable_component_splitting=False,
)
```

### 15.7 Interpreting the splitting test

Acceptance checks the exact criterion numerically, including agreement between the original and localized spans. It does not establish coverage of every component of the intended variety $`\mathcal W`$. Rhiza performs neither approximate clustering nor a general irreducible decomposition. If no proposal passes, the Gramian remains unsplit.

## 16. RIP and OVERLAP separators

Separator polynomials depend on variables shared by two cliques. They enter the two local certificate identities with opposite signs, allowing those identities to balance while canceling exactly in their global sum. This section describes how Rhiza chooses the shared polynomial spaces and the clique pairs that carry them.

### 16.1 RIP separators

Under `graph_policy='RIP'`, SDPR uses the verified CSP forest directly. For each forest edge $`e=\{k,l\}`$, it constructs the nonconstant quotient space on

```math
C_e=C_k\cap C_l
```

through `sep_degree`.

If this centered sampled separator space has rank zero, no free block is attached and the edge is omitted from `sdpr.separator_edges`. Otherwise, one separator basis is shared by the two endpoints with opposite signs.

RIP supplies a forest for routing the intersection spaces through `sep_degree`; numerical rank selection and any deliberate cap determine the retained directions.

### 16.2 OVERLAP occurrence groups

Under `graph_policy='OVERLAP'`, Rhiza enumerates nonconstant monomials through `sep_degree` on the maximal cliques. A monomial's **participating variable set** contains the variables with positive exponent: for example, $`x^2y`$ uses `{x, y}` and $`x^3`$ uses `{x}`.

For $`x^\alpha`$, its clique occurrence set is $`\Omega_\alpha=\{k\mid\mathop{\mathrm{supp}}\nolimits(\alpha)\subseteq C_k\}`$. Powers with the same participating set reuse the occurrence lookup.

Monomials with the same clique occurrence set are grouped and reduced to one centered sampled function-space basis. An occurrence represented by fewer than two cliques requires no transfer and is skipped.

### 16.3 Tree routing per function space

Suppose one sampled function space has occurrence set $`\Omega_\alpha`$. A full pairwise transfer construction would use

```math
\binom{|\Omega_\alpha|}{2}
```

copies. Only $`|\Omega_\alpha|-1`$ independent transfers are needed. Rhiza therefore routes that function space over a spanning tree of its occurrence cliques.

The initial routing pass prefers to reuse already selected clique pairs and the CSP backbone when possible. Broader occurrence groups are processed before narrower groups to encourage edge reuse.

### 16.4 Fixed-point topology pruning

After all occurrence groups have been routed, Rhiza repeatedly rebuilds each group's tree using only edges already present in the current union. Any edge unused by every rebuilt tree is removed.

The union can only shrink, and every occurrence group remains connected. This preserves the transfer span before final numerical rank reduction or deliberate truncation. A runtime assertion rejects any new edge during pruning.

The final union may contain cycles because different function spaces can require different trees. A graph cycle carrying distinct sampled spaces is not automatically redundant.

### 16.5 Edge-basis merging

Several occurrence-group spaces may be assigned to the same retained clique pair. Rhiza concatenates their sampled bases and performs a final rank reduction to create one free separator coefficient block on that edge.

When one already reduced occurrence basis is the only space on an edge, Rhiza reuses the basis record directly unless a final `basis_rank_limit` requires another reduction.

### 16.6 Degree dependence

Changing `sep_degree` can change the monomials, occurrence groups, separator edges, graph cycles and components, and free coefficient count. Inspect `sdpr.separator_edges` and its statistics for the final OVERLAP graph; the CSP backbone alone does not describe it.

### 16.7 Separator statistics

All graph policies report:

```text
num_separator_edges
num_separator_components
num_separator_cycles
num_separator_coeffs
```

The cycle count is the graph cycle rank

```math
|\mathcal E|-K+\text{number of connected components}.
```

OVERLAP additionally reports:

```text
num_overlap_occurrence_groups
largest_overlap_occurrence
num_separator_directions_full_topological
num_separator_directions_reduced_topological
separator_topological_reduction_ratio
```

For each occurrence group of rank $`r`$ on $`m`$ cliques, the full and reduced direction counts include $`r\binom m2`$ and $`r(m-1)`$, respectively. They precede edge-basis merging. `num_separator_coeffs` counts actual free separator coefficients after merging and final rank reduction.

The all-pairs comparison fields are not emitted for RIP.

## 17. Analysis, PQR compression, and SDPR finalization

### 17.1 Rank-only analysis

```python
report = sdpr.analyze(
    clique_ids=None,
    num_cliques=None,
    seed=0,
    show=False,
)
```

| Argument | Meaning and default |
|---|---|
| `clique_ids` | Defaults to `None`. Otherwise it must be an iterable of unique exact in-range clique IDs. Strings and bytes are rejected. An empty iterable is valid and selects no cliques. Explicitly selected cliques reuse cached rank analyses when available. |
| `num_cliques` | Defaults to `None`. Otherwise it must be an exact positive integer and cannot be combined with `clique_ids`. Rhiza selects `min(num_cliques, number of unchecked cliques)` without replacement. |
| `seed` | Defaults to `0`. It is used and validated only when `num_cliques` is supplied, and must then be an exact nonnegative integer. Selection uses `numpy.random.default_rng(seed)` and the returned IDs are sorted. |
| `show` | Defaults to `False` and is converted with `bool(show)`. When true, Rhiza prints the formatted report and still returns it. |

With both selection arguments omitted, Rhiza analyzes every currently unchecked clique. Once all cliques are checked, the same unselected call returns a complete cached report rather than an empty report. Analysis is permitted before or after SDPR finalization.

For each unchecked selected clique, `analyze()` balances its augmented matrix, computes singular values and rank, caches a compact record, and discards the matrix. Row maps remain unchanged.

### 17.2 Analysis report

The top-level report contains:

```text
sample_margin_check
num_samples
minimum_sample_margin
num_cliques
num_cliques_checked
num_cliques_checked_this_call
cliques
basis_issues
```

Each selected clique record contains:

```text
clique_id
scope
num_coeffs
rank
sample_margin
condition
singular_gap
```

`num_coeffs` excludes the objective column, while `rank` is measured on the augmented matrix that includes it.

`basis_issues` includes only basis records with zero sample margin or an applied rank limit. Each issue reports:

```text
basis_id
scope
degree
is_separator
rank
sample_margin
condition
singular_gap
rank_limited
```

### 17.3 Sample-margin status logic

`sdpr.status['sample_margin_check']` is determined in this order:

- `NO_MARGIN` if any registered basis or checked complete clique system has zero margin;
- `UNCHECKED` if no complete clique system has been checked;
- `PARTIAL` if only some complete systems have been checked and all observed margins are positive;
- `POSITIVE_MARGIN` if every complete system has been checked and every basis and system margin is positive.

### 17.4 PQR compression

```python
compression = sdpr.compress(
    method='PQR',
    clique_size_threshold=None,
)
```

| Argument | Meaning and default |
|---|---|
| `method` | Defaults to `'PQR'`. A string or `CompressionMethod` is accepted case-insensitively. PQR is the only implemented method. |
| `clique_size_threshold` | Defaults to `None`, selecting every clique. Otherwise it must be an exact positive integer, and PQR is applied only to cliques whose maximal declared scope contains at most this many variables. Larger cliques retain every sample row. |

Compression is legal only before finalization. All selected systems are prepared before row maps are committed, so failure leaves the SDPR uncompressed. For skipped cliques, compression neither assembles the full coefficient matrix nor computes its rank; all sample equations are retained and scaled at finalization.

Repeating the same resolved settings returns the cached report with top-level `changed=False`. Changing either setting after compression, or calling `compress()` after finalization, raises `PipelineError`. A completed request sets `compressed=True` even if it selects no cliques or removes no rows.

For one augmented clique matrix $`\overline A_k`$:

1. columns are equilibrated;
2. rows are normalized;
3. numerical rank is measured or a cached analysis is reused;
4. column-pivoted QR is applied to the transpose of the balanced matrix;
5. its pivot columns identify original sample rows of $`\overline A_k`$;
6. the initial retained prefix has length equal to estimated rank;
7. an independent SVD row-span projection audit is performed;
8. bounded batches of guard rows are added until the audit passes or every row is retained.

The audit compares its relative Frobenius residual with $`10\tau_{\mathrm{eff}}`$ for the balanced matrix. Retaining all rows is the fallback even if this numerical audit exceeds its threshold. Selected indices are sorted into sample order, and backend duals remain weights on actual retained sample equations.

### 17.5 Compression report

Top-level fields are:

```text
method
clique_size_threshold
changed
num_cliques_selected
num_cliques_skipped
num_rows_original
num_rows_retained
cliques
```

Each clique record contains:

```text
compression_applied
changed
clique_size
num_rows_original
num_rows_retained
num_guard_rows
rank
sample_margin
condition
singular_gap
row_span_residual
clique_id
scope
```

For skipped cliques, `compression_applied=False`, all rows are retained, and rank-dependent fields remain `None` unless previously analyzed. `clique_size` is the declared scope size used for selection. On the first compression call, top-level `changed` is true exactly when at least one clique discards a row; repeated calls return false.

Full compression leaves sample-margin status `NO_MARGIN` or `POSITIVE_MARGIN`. Selective compression may leave it `PARTIAL`, or `UNCHECKED` if no clique was selected or previously analyzed and all basis margins are positive. A later `analyze()` can check skipped cliques without changing row maps.

### 17.6 Row scaling

Each retained equation is scaled by the reciprocal of its Euclidean norm in the original, unbalanced augmented system; a zero row receives scale one. PQR stores these scales for selected rows. Finalization computes them in streamed chunks for uncompressed or skipped cliques. Extraction multiplies backend equation duals by the stored scales to recover weights on the original equations.

### 17.7 Finalization

```python
changed = sdpr.finalize()
```

`finalize()` has no arguments. It:

- installs explicit `row_indices` and `row_scales` for every clique;
- uses every sample row for an uncompressed clique;
- locks compression-sensitive state;
- discards ordinary basis, component-basis, and finite-value-count construction caches;
- leaves finalized basis and coefficient records available to backends;
- returns `True` on the first call and `False` on later calls.

Finalization does not force rank analysis: sample-margin status may remain `UNCHECKED` or `PARTIAL`. `analyze()` remains available; `compress()` raises `PipelineError`.

### 17.8 Selective compression for mixed clique families

A common sample table sized for large state cliques may leave many redundant rows in small measurement, actuator, or contact cliques. Compress those small cliques without assembling the complete matrices for the large cliques:

```python
compression = sdpr.compress(
    method='PQR',
    clique_size_threshold=8,
)
```

The inclusive threshold selects declared scopes with at most eight variables, regardless of rank or PSD block size. Width alone does not predict the benefit: small cliques can have nearly full row rank, and large equality-rich cliques can be highly compressible. Use `analyze()` for rank evidence. Compare compressed and uncompressed solves when extracted moments are sensitive to the row representation.

## 18. Solver API and backend behavior

`Solver` assembles an SCS or MOSEK model from a finalized SDPR, solves it once, and retains the data needed for moment extraction.

### 18.1 Constructor

```python
solver = rz.Solver(
    sdpr,
    backend='SCS',
    gap_tol=None,
    feas_tol=None,
    iter_limit=None,
    config=None,
)
```

| Argument | Meaning and default |
|---|---|
| `sdpr` | Required finalized `SDPR`. Constructing from an unfinalized SDPR raises `PipelineError`. The solver model is assembled during `Solver` construction. |
| `backend` | Defaults to `'SCS'`, the free backend. The value is stringified and uppercased, then must be `'SCS'` or `'MOSEK'`. MOSEK requires its optional package and a usable license. |
| `gap_tol` | Defaults to `None`. This resolves to `1e-5` for SCS and `1e-8` for MOSEK. An explicit value must be real, finite, and nonnegative; zero is accepted. |
| `feas_tol` | Defaults to `None`, which resolves to the already resolved `gap_tol`. An explicit value must be real, finite, and nonnegative. |
| `iter_limit` | Defaults to `None`, leaving the native solver's iteration policy in effect. Otherwise it must be an exact positive integer. |
| `config` | Defaults to `None`, which resolves to an empty dictionary. Otherwise `dict(config)` is used, so any dictionary-compatible mapping or iterable of pairs is accepted and copied. |

Public read-only properties are:

| Property | Meaning |
|---|---|
| `backend` | Normalized backend name. |
| `gap_tol` | Fixed resolved relative-gap tolerance. |
| `feas_tol` | Fixed resolved feasibility tolerance. |
| `iter_limit` | Fixed optional iteration limit. |
| `config` | Copy of the supplied native configuration mapping; see [setting precedence](#182-native-setting-precedence). |
| `built` | Whether the worker remains usable and has not been closed. |
| `solved` | Whether a completed solve report is cached. |
| `status` | Lifecycle and normalized solution status. |
| `stats` | Backend build and solve statistics. |

### 18.2 Native-setting precedence

Resolved Rhiza arguments override overlapping native `config` settings, including when Rhiza supplies the default tolerances.

For SCS, the adapter begins with a copy of `config` and supplies these defaults only when the corresponding key is absent:

```python
normalize = True
adaptive_scale = True
```

When no solver selector is supplied, Rhiza uses the native SCS default. With SCS 3.2 this is QDLDL; with SCS 3.3 Rhiza requests `'auto'`, allowing SCS to select a solver for the platform and installed build.

SCS 3.2 receives the `use_indirect`, `gpu`, and `mkl` selectors directly. With SCS 3.3, Rhiza translates these selectors to `linear_solver`; for example, `config={'use_indirect': True}` selects the CPU indirect solver with either API. GPU and MKL choices require the corresponding SCS build support. An explicit `use_indirect=False` selects QDLDL unless a GPU or MKL solver is requested.

With SCS 3.3, `config` can instead specify `linear_solver` as a lowercase string such as `'qdldl'`, or as an enum such as `scs.LinearSolver.QDLDL`. An explicit `linear_solver` cannot be combined with `use_indirect`, `gpu`, `mkl`, or `cudss`.

The adapter then overrides:

```python
eps_abs = feas_tol
eps_rel = gap_tol
verbose = solve(verbose=...)
```

It overrides `max_iters` only when `iter_limit` is not `None`; otherwise a `max_iters` value in `config`, or the native default when absent, remains active.

For MOSEK, all `config` pairs are first applied through the generic parameter interface. Rhiza then overrides the interior-point relative-gap, primal-feasibility, dual-feasibility, and optional maximum-iteration parameters from the resolved generic arguments.

`solver.config` returns the supplied mapping. SCS reports the resolved native settings in `solver.stats['settings']` after a completed solve. For SCS 3.3, this report contains the normalized `linear_solver` string in place of the older selectors, including when an enum was supplied. MOSEK does not expose a corresponding settings report.

### 18.3 Solve operation

```python
report = solver.solve(verbose=False)
```

| Argument | Meaning and default |
|---|---|
| `verbose` | Defaults to `False` and is converted with `bool(verbose)`. A true value streams the native solver log; see [Verbose logs](#187-verbose-logs). |

The first completed call caches and returns these fields:

```text
success
changed
solution_status
primal_objective
dual_objective
relative_gap
num_iterations
solve_time
```

`changed=True` on the first completed call. Until the Solver is closed, later calls return a copy with `changed=False` and perform no optimization, even if the cached result was unsuccessful. A native exception leaves the Solver unsolved and retryable.

`success=True` means the normalized status is `OPTIMAL` or `OPTIMAL_INACCURATE`. An unsuccessful completed call still sets `solved=True`, but its objectives and gap are `None`, and extraction raises `SolveError`.

Rhiza's objective terminology is fixed across backend orientations:

- `primal_objective` is the numerical SOS certificate bound;
- `dual_objective` is the sampled functional value.

For primal and dual objectives $`p_{\mathrm{norm}}`$ and $`d_{\mathrm{norm}}`$ in normalized solver units, the reported relative gap is

```math
\frac{|p_{\mathrm{norm}}-d_{\mathrm{norm}}|}{\max(1,|p_{\mathrm{norm}}|,|d_{\mathrm{norm}}|)}.
```

The returned objective values are restored through SDPR's affine inverse map; the gap remains in normalized units. This common diagnostic need not equal a backend's native stopping criterion.

### 18.4 Normalized solution statuses

Public statuses are:

```text
OPTIMAL
OPTIMAL_INACCURATE
UNKNOWN
INFEASIBLE
UNBOUNDED
SOLVER_ERROR
```

Native labels remain under `solver.stats['native']` rather than being duplicated at the top level.

### 18.5 SCS backend

Omitting `backend` selects SCS, as in the [sparse binary example](#22-sparse-binary-example):

```python
solver = rz.Solver(sdpr)
```

#### Orientation

Rhiza submits the **dual orientation** of the certificate SDP to SCS. In this orientation:

- SCS primal variables are the retained clique-system sample weights used later for extraction;
- zero-cone rows enforce free scalar certificate equations;
- each Rhiza Gramian becomes one SCS PSD cone;
- no separate scalarized Gramian decision variables and identity equations are needed.

Because of this orientation, native infeasibility meanings reverse when translated to Rhiza's certificate-primal convention:

```text
native SCS UNBOUNDED              -> Rhiza INFEASIBLE
native SCS UNBOUNDED_INACCURATE   -> Rhiza INFEASIBLE
native SCS INFEASIBLE             -> Rhiza UNBOUNDED
native SCS INFEASIBLE_INACCURATE  -> Rhiza UNBOUNDED
```

#### PSD vectorization

SCS uses lower-triangle, column-major symmetric vectorization. Off-diagonal entries are scaled by $`\sqrt{2}`$. Rhiza's internal Gramian columns already contain the symmetric factor of two, so the adapter performs the corresponding sign and $`\sqrt{2}`$ conversion when assembling cone rows.

#### Settings

The [default settings](#182-native-setting-precedence) use the native SCS solver selection. The `linear_system_mode` statistic describes the requested selection:

| SCS 3.3 `linear_solver` | Reported mode |
|---|---|
| `'qdldl'` | `SPARSE_DIRECT` |
| `'cpu_indirect'` | `SPARSE_INDIRECT` |
| `'mkl'` | `MKL_DIRECT` |
| `'accelerate'` | `ACCELERATE_DIRECT` |
| `'cpu_dense'` | `DENSE_DIRECT` |
| `'gpu_indirect'` | `GPU_INDIRECT` |
| `'cudss'` | `GPU_DIRECT` |
| `'auto'` | `AUTO` |

With SCS 3.2, the Boolean selectors produce the corresponding CPU, MKL, or GPU mode labels. In particular, `gpu=True, use_indirect=False` requests GPU direct solution, while `gpu=True, use_indirect=True` requests GPU indirect solution. Availability depends on the installed SCS version and build. The separate `native_linear_system_solver` statistic records SCS's own solver description when returned; it can identify the solver chosen by `AUTO`.

#### Workspace and memory

Solver construction assembles the cone data; `solve()` creates the native SCS workspace, whose constructor fixes verbosity. SCS then owns deep copies, allowing Rhiza to release its assembled CSC data before iteration. A successful solve validates the native vectors and retains only the system weights needed for extraction. Every completed solve releases the workspace and cone vectors.

#### SCS statistics

Top-level build statistics include:

```text
num_rows
num_rows_original
num_rows_retained
num_scalar_coeffs
num_gramians
num_cone_rows
num_zero_cone_rows
num_psd_cone_rows
num_matrix_nonzeros
num_scalar_entries_emitted
num_gramian_entries_emitted
gramian_sizes
backend_build_time
```

After solving, top-level statistics add `solve_time` and `num_iterations`. SCS `solve_time` includes native workspace construction and optimization. Resolved settings are reported as described in [§18.2](#182-native-setting-precedence).

`solver.stats['native']` contains:

```text
termination_status
status_value
primal_residual
dual_residual
absolute_gap
setup_time
solve_time
linear_system_mode
native_linear_system_solver
scale
scale_updates
accepted_acceleration_steps
rejected_acceleration_steps
```

Timing fields from the SCS information mapping are converted from milliseconds to seconds.

### 18.6 MOSEK backend

Select MOSEK explicitly:

```python
solver = rz.Solver(sdpr, backend='MOSEK')
```

A missing package or unusable license raises `BackendUnavailableError`, either during construction or when optimization requests the license.

#### Orientation

MOSEK receives the certificate SDP directly:

- bound, equality-multiplier, and separator coefficients are native free scalar variables;
- each Gramian is a native PSD matrix variable;
- clique equations are fixed equality constraints;
- the native objective maximizes the sum of bound shares.

MOSEK's symmetric matrix inner product counts off-diagonal lower-triangle entries twice. Rhiza's internal coefficient columns also use a symmetric factor of two, so the adapter halves off-diagonal coefficient values when emitting native bar-matrix triplets.

#### Streaming assembly

Scalar and Gramian coefficient chunks are streamed directly into the native task. Rhiza does not build one global sparse matrix for MOSEK. Native task dimensions are validated against the finalized SDPR after assembly.

#### Configuration

Native parameters follow the shared [setting-precedence rules](#182-native-setting-precedence).

#### Resource lifecycle

The task remains live through optimization and statistics collection. A successful solve copies the equation duals needed for moments. Every completed solve then releases the task; reports and extraction use the retained Rhiza data.

#### MOSEK statistics

Top-level build statistics include:

```text
num_rows
num_rows_original
num_rows_retained
num_scalar_coeffs
num_gramians
num_scalar_entries_emitted
num_gramian_entries_emitted
backend_build_time
```

`solver.stats['native']` includes native task dimensions and canonical stored sparsity counts:

```text
num_rows
num_scalar_coeffs
num_gramians
num_scalar_nonzeros_stored
num_gramian_block_nonzeros_stored
num_gramian_entries_stored_upper_bound
termination_status
problem_status
solution_status
```

Native stored counts may differ from Rhiza's emitted-entry counts because MOSEK can combine, omit, or canonicalize entries.

MOSEK's public `solve_time` measures the `optimize()` call, excluding native task construction. Its timing scope therefore differs from SCS's.

`solver.stats['factorization']` contains documented interior-point information when available:

```text
solve_form
num_nonzeros
dense_dimension
num_flops
num_threads
```

`solver.stats['memory']` contains:

```text
num_bytes_current
num_bytes_peak
```

with `None` values when the native interface does not provide memory data.

### 18.7 Verbose logs

`solve(verbose=True)` streams the native solver log. Rhiza does not parse or retain the raw log. Stable diagnostics should be read from the normalized report and `solver.stats`.

### 18.8 Closing

```python
changed = solver.close()
```

`close()` returns `True` on the first call and `False` thereafter. It releases native resources, copied system weights, and cached clique functionals. Status and statistics remain available, and `solved` remains true if a completed report was cached. Further solving or extraction raises `PipelineError`.

Using `with rz.Solver(...) as solver:` guarantees closing on exit without suppressing exceptions.

## 19. Moment extraction and diagnostics

After a successful solve, use `extract_means()` for variable estimates and `extract_moments()` for specified monomials. Estimates shared by multiple cliques are averaged; Rhiza calls this **stitching**. Use the per-clique values and diagnostics to check agreement before rounding or otherwise recovering a feasible candidate.

The estimates come from normalized signed functionals. They need not be moments of a probability measure, and the extracted means need not satisfy the original constraints.

### 19.1 When extraction is available

Extraction requires an open Solver with a completed, successful solve. Before solving or after closing it raises `PipelineError`; an unsuccessful completed solve raises `SolveError`.

The first extraction call reconstructs and validates the clique functionals. Subsequent queries reuse them. The formulas and normalization checks are given below.

### 19.2 Extracting means

```python
means = solver.extract_means(variables=None)
```

| Argument | Meaning and default |
|---|---|
| `variables` | Defaults to `None`, selecting every POP variable in `pop.variable_names` order. A string selects one variable. Any other value must be an iterable of literal strings, and caller order is retained. Unknown names raise `ExtractionError`. |

For each requested variable, Rhiza evaluates its normalized first moment in every containing clique and returns the arithmetic mean of those values. This is the stitching operation. The result is an ordinary dictionary keyed in requested order.

`extract_means()` and `extract_moment_diagnostics()` evaluate coordinate samples directly and do not apply the named-moment degree check. They therefore accept `sos_degree=0`, although the constant base SOS space alone does not constrain those first moments.

### 19.3 Extracting named moments

```python
moments = solver.extract_moments({
    "x2": {"x": 2},
    "xy": {"x": 1, "y": 1},
    "constant": {},
})
```

`moments` is required and must be convertible to a mapping:

```text
moment name -> {variable name: exponent}
```

Moment and variable names must be literal strings, and each inner specification must be mapping-compatible. Exponents must be exact nonnegative integers; zero exponents are dropped after validating the variable names. An empty map denotes the constant moment. The total degree must not exceed `sdpr.sos_degree`, and all variables with nonzero exponent must fit in one clique.

For a requested monomial

```math
p(x)=x_{i_1}^{a_1}\cdots x_{i_r}^{a_r},
```

Rhiza evaluates $`p(z_s)`$ once on the original sample table for $`s=1,\ldots,S`$, applies every containing clique functional, normalizes by each mass, and returns the arithmetic mean of those clique values.

The degree limit still applies when sample identities make a monomial dependent on a lower-degree one. A cross-clique moment whose full scope fits no clique is rejected.

### 19.4 Extracting moments by clique

```python
by_clique = solver.extract_clique_moments(
    moments,
    cliques=None,
)
```

| Argument | Meaning and default |
|---|---|
| `moments` | Required moment mapping with the same normalization and validation as `extract_moments()`. |
| `cliques` | Defaults to `None`, selecting every clique in ID order. Otherwise it must be an iterable of exact integer IDs. Strings and bytes are rejected, and each ID is range-checked. |

The result is

```text
{
    clique_id: {
        moment_name: normalized_value,
        ...
    },
    ...
}
```

Every selected clique appears. A clique includes only requested moments whose scope it contains, so its nested mapping may be empty. The request is nevertheless rejected if a moment has no containing clique anywhere in the SDPR.

### 19.5 Moment diagnostics

```python
diagnostics = solver.extract_moment_diagnostics(
    variables=None,
)
```

| Argument | Meaning and default |
|---|---|
| `variables` | Uses the same selection forms as `extract_means()`: `None` for all variables, one string, or an iterable of literal strings. Unknown names raise `ExtractionError`. |

The report contains:

```text
mass_by_clique
total_variation_by_clique
mass_condition_by_clique
mass_error_by_clique
means_by_variable
stitching_error_max_by_variable
stitching_error_rms_by_variable
stitching_error_max
stitching_error_rms
```

`means_by_variable` maps each requested variable to its individual local first moments by clique. Only variables represented by more than one clique receive per-variable stitching errors.

For local values $`v_1,\ldots,v_q`$ of one shared variable,

```math
\mathop{\mathrm{error}}\nolimits_{\max}
=
\max_i v_i-\min_i v_i,
```

and

```math
\mathop{\mathrm{error}}\nolimits_{\mathrm{rms}}
=
\sqrt{\frac1q\sum_i(v_i-\bar v)^2}.
```

Let $`\mathcal J_{\mathrm{shared}}`$ be the set of requested variables represented in more than one clique. The aggregate maximum is the largest per-variable range, and

```math
\mathop{\mathrm{stitching\_error\_rms}}\nolimits
=
\sqrt{
\frac{1}{|\mathcal J_{\mathrm{shared}}|}
\sum_{a\in\mathcal J_{\mathrm{shared}}}
\mathop{\mathrm{error}}\nolimits_{\mathrm{rms}}(a)^2
}.
```

Thus every shared requested variable receives equal weight rather than every clique occurrence. Both aggregate errors are zero when no requested variable is represented by more than one clique.

### 19.6 Interpreting extraction quality

Read mass and stitching diagnostics together. A large mass condition indicates cancellation in the returned weights; it does not by itself invalidate the moments. Mass error measures departure from unit mass, while stitching errors measure disagreement between shared coordinate moments. Their significance depends on backend accuracy and the separator functions represented by the relaxation.

The broader checks for sample qualification, construction quality, and numerical solution quality are collected in [Certificate validity and interpretation](#23-certificate-validity-and-interpretation).

### 19.7 Signed functionals and normalization

For clique $`C_k`$, let the retained signed sample weights be $`y_{k,s}`$. The reconstructed functional is

```math
\mathcal L_k[p]
=
\sum_s y_{k,s}\,p(z_{s,C_k}),
```

where the sum runs over the retained sample indices for that clique.

Weights can have either sign. Redundant sampled equations can make their representation nonunique, allowing large coefficients to cancel while represented moments remain accurate.

The functional mass is computed with stable summation,

```math
m_k
=
\mathcal L_k[1]
=
\sum_s y_{k,s},
```

and each clique moment is normalized as

```math
\widehat{\mathcal L}_k[p]
=
\frac{\mathcal L_k[p]}{m_k}.
```

Normalization gives unit mass; the weights can still be negative.

### 19.8 Mass validation

Solver derives the backend-accuracy floor

```math
\tau_{\mathrm{abs}}
=
\max\!\left(
100\,\varepsilon_{\mathrm{float64}},
10\,\texttt{feas\_tol}
\right).
```

For clique $`C_k`$, define total variation

```math
T_k
=
\sum_s|y_{k,s}|
```

and the independent cancellation-roundoff floor

```math
\tau_{\mathrm{round}}
=
64\,\varepsilon_{\mathrm{float64}}\max(1,T_k).
```

After checking that the weights, mass, and total variation are finite and representable, Rhiza rejects normalization when

```math
|m_k|
\le
\max(\tau_{\mathrm{abs}},\tau_{\mathrm{round}}).
```

The feasibility tolerance is not multiplied by total variation: cancellation in a nonunique dual representative can coexist with meaningful unit mass.

Reported diagnostics are

```math
\text{mass condition}
=
\frac{T_k}{|m_k|},
\qquad
\text{mass error}
=
|m_k-1|.
```

## 20. Rounding helpers

`rz.rounding` provides explicit projections of extracted estimates into common decision spaces. Rounding leaves the SDP unchanged; candidate feasibility and objective value require separate checks.

### 20.1 SO(3) projection

```python
R = rz.rounding.project_so3(matrix)
```

| Argument | Meaning |
|---|---|
| `matrix` | Required finite real array-like object with exact shape `(3, 3)`. Batched matrices are not accepted. |

The function computes the nearest proper rotation in Frobenius norm using an orthogonal Procrustes SVD. If the nearest orthogonal factor is a reflection, the final left singular-vector direction is flipped to enforce determinant `+1`. The result is a floating-point NumPy array of shape `(3, 3)`.

### 20.2 SE(3) projection

```python
T = rz.rounding.project_se3(matrix)
```

| Argument | Meaning |
|---|---|
| `matrix` | Required finite real array-like object with exact shape `(3, 4)` or `(4, 4)`. Batched inputs are not accepted. |

The rotational block is projected with `project_so3()`, translation is copied unchanged, and the last row is set to `[0, 0, 0, 1]`. The result always has shape `(4, 4)`.

### 20.3 Binary rounding

```python
binary = rz.rounding.round_binary(
    values,
    threshold=0.5,
)
```

| Argument | Meaning and default |
|---|---|
| `values` | Required finite real array-like object of any shape, including a scalar. Output shape matches input shape. |
| `threshold` | Defaults to `0.5`. It must be a finite real scalar. Values greater than or equal to the threshold map to one; lower values map to zero. |

The result is an integer NumPy array, or a NumPy integer scalar for scalar input.

### 20.4 Sign rounding

```python
signs = rz.rounding.round_sign(
    values,
    zero_value=1.0,
)
```

| Argument | Meaning and default |
|---|---|
| `values` | Required finite real array-like object of any shape. |
| `zero_value` | Defaults to `1.0`. It must be exactly `-1` or `+1`; equivalent real representations such as `-1.0` are accepted. |

Negative values map to `-1`, positive values to `+1`, and exact zeros to `zero_value`. The result is an integer NumPy array with the input shape, including a zero-dimensional array for scalar input.

### 20.5 One-hot rounding

```python
one_hot = rz.rounding.round_one_hot(
    values,
    axis=-1,
)
```

| Argument | Meaning and default |
|---|---|
| `values` | Required finite real array-like object with at least one dimension. Scalar input is invalid. |
| `axis` | Defaults to `-1`. It must be an exact integer. Negative axes resolve in the ordinary NumPy way. The selected category axis must be in range and nonempty. |

The first maximum along the category axis receives one and all other entries receive zero. Ties therefore follow NumPy `argmax`. The integer output has the same shape as the input.

### 20.6 Rounding input and error policy

All helpers reject complex and nonfinite input with ordinary `ValueError`. Check each rounded candidate against the original equalities, inequalities, and application-specific combinatorial conditions, and evaluate its original objective.

## 21. Status, statistics, reports, and exceptions

### 21.1 Copy semantics

Public `status` and `stats` properties return copies, including nested backend mappings. Editing a returned mapping does not reconfigure the object. NumPy scalar values become ordinary Python scalars; compact report containers are retained without expanding numerical arrays into lists.

### 21.2 Human-readable summaries

`str(pop)`, `str(csp)`, `str(sdpr)`, and `str(solver)` present status fields followed by statistics. Keys are printed with underscores replaced by spaces and floats with compact precision.

`repr(...)` remains concise and lifecycle-focused.

`sdpr.analyze(show=True)` uses the nested report formatter. The same formatter supports mappings and sequences of mappings with indentation.

### 21.3 POP status and stats

Before finalization:

```python
pop.status == {'finalized': False}
```

and building statistics contain:

```text
num_samples
num_variables
num_entries
```

After finalization, status adds `graph_policy`, and statistics add objective, constraint, sampled-identity, and clique counts.

### 21.4 CSP status and stats

Status:

```text
graph_policy
rip
```

Statistics:

```text
num_variables
num_declared_cliques
num_cliques
num_subsumed_cliques
num_structural_clique_edges
num_structural_components
clique_size_min
clique_size_max
num_candidate_overlap_pairs  # optional under OVERLAP
```

### 21.5 SDPR status and stats

Status:

```text
sample_margin_check
basis_rank_limited
sep_degree_insufficient
```

Statistics are listed in [SDPR construction API and degree policies](#11-sdpr-construction-api-and-degree-policies) and separator-specific fields in [RIP and OVERLAP separators](#16-rip-and-overlap-separators).

### 21.6 Solver status

Before solve:

```text
backend
built
solved
closed
```

After a completed solve, status also contains:

```text
solution_status
success
```

`built` remains true until explicit closing even though both backends release their native task or workspace after a completed solve. [Closing](#188-closing) describes the retained reports and discarded extraction data.

### 21.7 Exception hierarchy

Rhiza-specific exceptions use the following hierarchy:

```text
RhizaError
├── POPError
│   ├── POPParseError
│   └── POPValidationError
├── PipelineError
├── CSPError
│   ├── GraphPolicyError
│   └── RIPError
├── SDPRError
│   ├── DegreeError
│   ├── NumericalRankError
│   └── CompressionError
├── BackendError
│   ├── BackendUnavailableError
│   └── BackendBuildError
├── SolveError
└── ExtractionError
```

Use the narrowest useful catch:

- `POPParseError` for malformed or semantically invalid `.pop` input;
- `POPValidationError` for invalid builder data;
- `PipelineError` for an operation in the wrong lifecycle state;
- `RIPError` for a declared clique family that fails RIP;
- `DegreeError` for incompatible SOS or multiplier-degree choices;
- `NumericalRankError` for nonfinite or unrepresentable rank/scaling operations;
- `CompressionError` for unsupported compression or failed row-span preservation;
- `BackendUnavailableError` for a missing dependency or MOSEK license;
- `BackendBuildError` for native model-construction failure;
- `SolveError` for optimization or unusable solve-state failures;
- `ExtractionError` for invalid moment requests or unsafe functional normalization.

`NotImplementedError` is used for the recognized but unimplemented `EXTEND` graph policy.

## 22. Sparse binary example

This example shows how sparsity, sample relations, and numerical reductions interact. It includes:

- two RIP cliques;
- a shared separator variable;
- sampled algebraic identities;
- validated finite-component Gramian splitting;
- basis and equation-rank diagnostics;
- PQR row compression;
- SCS solving;
- averaged and per-clique moment extraction;
- binary rounding.

### 22.1 Problem

Let $`x,y,z\in\{0,1\}`$. Minimize

```math
f(x,y,z)
=
(x-y)^2-x
+
(y-z)^2-z.
```

The unique minimizer is

```math
(x,y,z)=(1,1,1),
```

with optimum

```math
f^*=-2.
```

The two objective terms have scopes $`(x,y)`$ and $`(y,z)`$, so the maximal declared cliques form a two-node RIP chain with separator scope $`(y)`$.

### 22.2 Complete code

```python
import numpy as np
import rhiza as rz

# Enumerate the complete binary variety and repeat each point once.
# Repetition is deliberate: it gives positive empirical sample margin, while
# PQR later removes redundant rows from each clique system.
states = np.array(
    [
        (x, y, z)
        for x in (0.0, 1.0)
        for y in (0.0, 1.0)
        for z in (0.0, 1.0)
    ],
    dtype=float,
)
states = np.repeat(states, repeats=2, axis=0)
x, y, z = states.T

# ---------------------------------------------------------------------------
# 1. Build and finalize the sampled POP.
# ---------------------------------------------------------------------------
pop = rz.POP()
pop.add_variable("x", x)
pop.add_variable("y", y)
pop.add_variable("z", z)

pop.add_objective(
    (x - y)**2 - x,
    ("x", "y"),
    2,
    form='MIN',
    name="left_cost",
)
pop.add_objective(
    (y - z)**2 - z,
    ("y", "z"),
    2,
    form='MIN',
    name="right_cost",
)

# These zero-valued equality entries state the sampled binary relations.
# At POP finalization they are classified as sampled identities and therefore
# receive no explicit equality multipliers in the SDPR.
pop.add_constraint(
    x*(x - 1.0),
    "x",
    2,
    form='==0',
    name="x_binary",
)
pop.add_constraint(
    y*(y - 1.0),
    "y",
    2,
    form='==0',
    name="y_binary",
)
pop.add_constraint(
    z*(z - 1.0),
    "z",
    2,
    form='==0',
    name="z_binary",
)

pop.finalize(graph_policy='RIP')

assert pop.csp.cliques == (("x", "y"), ("y", "z"))
assert pop.csp.structural_clique_edges == ((0, 1),)
assert pop.csp.structural_separator_scopes == (("y",),)

# ---------------------------------------------------------------------------
# 2. Compile, inspect, and compress the sparse SOS relaxation.
# ---------------------------------------------------------------------------
sdpr = rz.SDPR(
    pop,
    sos_degree=4,
    rank_tol=1e-10,
    enable_component_splitting=True,
)

analysis = sdpr.analyze(show=True)
compression = sdpr.compress(method='PQR')
sdpr.finalize()

assert analysis["sample_margin_check"] == "POSITIVE_MARGIN"
assert sdpr.stats["num_rows_original"] == 32
assert sdpr.stats["num_rows_retained"] == 8

# Each two-variable binary clique has four exact sampled components. The two
# logical base Gramians are therefore represented by eight 1-by-1 PSD blocks.
assert sdpr.stats["num_component_split_gramians"] == 2
assert sdpr.stats["num_gramians"] == 8
assert sdpr.stats["num_psd_coeffs"] == 8
assert sdpr.stats["num_psd_coeffs_without_component_splitting"] == 20

# ---------------------------------------------------------------------------
# 3. Solve, extract moments, and round.
# ---------------------------------------------------------------------------
with rz.Solver(sdpr, backend='SCS') as solver:
    solve_report = solver.solve()
    if not solve_report["success"]:
        raise RuntimeError(solve_report)

    means = solver.extract_means()

    moments = solver.extract_moments({
        "constant": {},
        "x_squared": {"x": 2},
        "xy": {"x": 1, "y": 1},
        "yz": {"y": 1, "z": 1},
    })

    y_by_clique = solver.extract_clique_moments(
        {"y": {"y": 1}},
        cliques=(0, 1),
    )

    diagnostics = solver.extract_moment_diagnostics()

rounded = rz.rounding.round_binary([
    means["x"],
    means["y"],
    means["z"],
])

print("solve report:", solve_report)
print("means:", means)
print("moments:", moments)
print("y by clique:", y_by_clique)
print("diagnostics:", diagnostics)
print("rounded:", rounded)
```

### 22.3 What happens internally

1. **POP construction.** All evaluation arrays contain 16 values, one per sample point. The two objective scopes become maximal declared cliques. The three singleton equality scopes are subsumed.

2. **Sampled identities.** The binary equality vectors are exactly zero, so POP records three sampled identities. They receive no explicit equality multiplier columns. The binary variable samples themselves supply the relations used by quotient-basis reduction.

3. **RIP structure.** The cliques `('x', 'y')` and `('y', 'z')` form a tree. Variable `y` is their intersection and produces one separator edge.

4. **Quadratic quotient bases.** With `sos_degree=4`, each base Gramian uses monomials through degree two. On a binary pair, these span all four functions on the four pair states.

5. **Component splitting.** The exact pair values `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` partition each clique's samples into four components. The degree-two quotient basis is restriction-invariant, each component has positive sample margin, and four scalar PSD blocks use fewer coefficients than one rank-four Gramian.

6. **Separator space.** The centered quotient space on binary `y` has one nonconstant direction, so there is one free separator coefficient appearing with opposite signs in the two clique systems.

7. **Rank analysis.** Each clique's 16-row augmented system has rank four and sample margin 12. The smallest basis-level margin is positive, so the complete result is `POSITIVE_MARGIN`.

8. **PQR.** Four actual sample rows per clique reproduce the complete balanced row span. The total submitted equation count drops from 32 to 8.

9. **Solve.** The degree-four sampled relaxation is exact for this finite example. A numerical backend should report primal and dual objectives close to $`-2`$ within its tolerance.

10. **Extraction.** The unique optimizer makes the averaged means close to one. The individual `y` moments from the two cliques should agree to solver accuracy. Binary rounding returns `[1, 1, 1]`.

### 22.4 Variations worth trying

To inspect the effect of one reduction at a time:

```python
# Keep the same sampled SOS cone but use unsplit logical Gramians.
sdpr_unsplit = rz.SDPR(
    pop,
    sos_degree=4,
    enable_component_splitting=False,
)

# Keep every sampled equation rather than applying PQR.
sdpr_uncompressed = rz.SDPR(pop, sos_degree=4)
sdpr_uncompressed.analyze()
sdpr_uncompressed.finalize()
```

The split and unsplit formulations should agree on the represented sampled cone and optimum. Compressed and uncompressed formulations should agree on the represented affine equation system and optimum. Their native dimensions and numerical conditioning may differ.

A moment such as

```python
{"xyz": {"x": 1, "y": 1, "z": 1}}
```

is deliberately rejected because no single clique contains all three variables, even though the global sample table contains all three columns.

## 23. Certificate validity and interpretation

Interpret a result at three levels: algebraic sample qualification, numerical construction quality, and backend solution quality. Each addresses a different requirement for a useful certificate.

### 23.1 Algebraic sample qualification and poisedness

Let $`\mathcal W\supseteq\mathcal V`$ be the intended sampled variety, with $`\mathcal W=\mathcal V`$ when all equality constraints are absorbed by sampling. Let $`\mathcal Z=(z_1,\ldots,z_S)\subseteq\mathcal W`$, and write $`\mathcal W_{C_k}=\{x_{C_k}\mid x\in\mathcal W\}`$. Let $`\mathcal R_k`$ be the span of the adjusted local residuals $`\zeta_k-\sum_l\tau_{kl}`$, including the objective column and incident separator spaces. The sampled local equations represent polynomial identities throughout $`\mathcal W_{C_k}`$ only when

```math
p\in\mathcal R_k,
\qquad
p(z_{s,C_k})=0\ \text{for }s=1,\ldots,S
\quad\Longrightarrow\quad
p(x_{C_k})=0\ \text{for every }x_{C_k}\in\mathcal W_{C_k}.
```

The required poisedness condition applies separately to each represented clique residual space. Once every local identity holds on the intended variety, summing them cancels the separators and yields a valid global certificate. The condition depends jointly on:

- the intended sampled variety $`\mathcal W`$;
- the sampled points;
- the represented local residual spaces;
- the clique decomposition.

Rhiza does not receive a symbolic description of $`\mathcal W`$ and cannot prove this implication in general.

### 23.2 Numerical construction quality

Construction diagnostics test whether the sampled spaces are numerically resolved at the chosen tolerance. Inspect quotient-basis ranks, conditions and singular gaps, basis and complete-system sample margins, PQR row-span residuals, rank-limit and separator-degree flags, and component-splitting invariance tests.

These checks assess the assembled sampled relaxation. Faithfulness to the intended variety still requires the poisedness condition above.

### 23.3 Backend solution quality

The [solve report](#183-solve-operation) and backend statistics describe numerical accuracy through solution status, objectives, gap, native residuals, and iteration count. [Moment diagnostics](#195-moment-diagnostics) check the masses and consistency of the extracted functionals.

`success=True` means the backend returned a numerical solution usable for extraction. It does not establish sample qualification or exact certificate feasibility.

### 23.4 Why positive sample margin is not a proof

A positive sample margin means that the sample count exceeds the measured rank. Many redundant points on one component of a reducible variety can yield a large margin while missing polynomial behavior on an unsampled component.

Assess coverage on the variety actually sampled. An [ambient-variety formulation](#236-ambient-variety-sampling) changes that variety and the residual spaces that must be resolved.

### 23.5 Validity and fixed-degree equivalence

Local poisedness makes each sampled clique identity valid on $`\mathcal W`$. Summing these identities cancels the separators. On $`\mathcal F`$, the explicit equality terms vanish and the inequality-weighted SOS terms are nonnegative, so the sum certifies $`\beta\le f^*`$.

Matching the bound of a single global correlatively sparse relaxation at a fixed degree is a separate question. The manuscript's membership-localization result gives an additional condition: every vanishing relation in the sum of the local residual spaces must decompose into vanishing relations within individual cliques and the same represented degree bounds. The selected multiplier and separator spaces must also provide the required directions. A valid local formulation can otherwise be more restrictive.

OVERLAP routing preserves transfers within the represented sampled spaces and construction tolerance, before any deliberate rank truncation. Poisedness and fixed-degree equivalence remain separate conditions. The manuscript's convergence theorem additionally requires its stated hypotheses, including RIP, suitable boundedness certificates, degree growth, and multiplier spaces.

### 23.6 Ambient-variety sampling

Sampling a convenient containing variety $`\mathcal W`$ can avoid enumerating components of the feasible equality variety $`\mathcal V`$. Omitted equalities must remain explicit multiplier-bearing entries, with degrees sufficient for the intended residual certificate. Sample qualification is then assessed on $`\mathcal W`$ for those residual spaces.

An equality vector already classified as a sampled identity receives no multiplier. The declaration and classification rules are given in [Equality ideals and ambient-variety sampling](#52-equality-ideals-and-ambient-variety-sampling) and [POP finalization](#76-pop-finalization).

### 23.7 Inequality coverage

An inequality evaluation vector must contain at least one exactly nonzero value. Positive rescaling preserves this check provided the evaluations remain finite and do not all round to zero; accepted vectors are subsequently normalized.

If an inequality vanishes identically on the intended sampled variety $`\mathcal W`$, omit it. Its evaluations cannot be distinguished from accidental boundary-only sampling.

### 23.8 Component splitting validity

The restriction-invariance and span tests in [Component-localized Gramian splitting](#15-component-localized-gramian-splitting) check sampled-cone preservation to their stated tolerance. Numerical acceptance establishes neither an exact algebraic proof nor coverage of every intended variety component.

### 23.9 Symbolic validation

Rhiza compiles and solves a sampled relaxation. It does not:

- reconstruct symbolic SOS polynomials;
- reconstruct a symbolic global certificate identity;
- independently test a certificate on arbitrary unsampled points;
- prove exact rational feasibility;
- certify an application-specific rounded decision.

A downstream certificate-reconstruction workflow can test identities on fresh samples. Rigorous certification also requires algebraic or verified numerical validation.

### 23.10 Interpreting the result

First verify the model declarations and local poisedness assumptions in [§23.1](#231-algebraic-sample-qualification-and-poisedness), including component coverage. If a matching global sparse bound is required, check the additional [localization conditions](#235-validity-and-fixed-degree-equivalence).

Then assess [construction diagnostics](#232-numerical-construction-quality), [backend and extraction accuracy](#233-backend-solution-quality), and finally the independently checked rounded candidate. A later numerical check cannot replace an earlier modeling or algebraic assumption.

## 24. Common failure modes and troubleshooting

| Symptom or exception | Likely cause | Correct response |
|---|---|---|
| `POPValidationError` for sample count mismatch | A variable or entry vector has a different length from the established sample count. | Rebuild every evaluation on the same ordered sample set. |
| Complex-input error | A public array or scalar has complex dtype or contains a complex object. | Supply genuinely real data; Rhiza never discards imaginary parts. |
| Nonfinite-evaluation error | NaN, infinity, or overflow occurred while generating samples or a lift. | Fix sample generation or rescale before adding the object. Lift mutations are transactional. |
| Global name collision | A variable or named entry reuses an existing name. | Rename it. Variables and named entries share one namespace. |
| Repeated or empty scope | The scope is malformed. | Use a nonempty sequence of unique literal variable names. |
| Unknown variable at finalization | An entry referenced a name that was never added. | Add the variable or correct the scope. Forward references are allowed only until finalization. |
| “Variables are unused by every entry” | A declared variable appears in no objective or constraint scope. | Remove it from construction or add a mathematically relevant entry using it. |
| POP requires an objective | Only constraints were added. | Add at least one `MIN` or `MAX` entry. |
| All-zero inequality rejected | Every supplied sample lies on that inequality's boundary, or the inequality is redundant on the sampled variety. | Add off-boundary samples or omit the redundant inequality. Do not weaken the check with `identity_tol`; it is exact for inequalities. |
| Sampled equality unexpectedly omitted from multipliers | Its max absolute sampled value is at most `identity_tol`. | Lower `identity_tol` only when the vector is intended to be explicit and its nonzero values are meaningful. |
| Equality unexpectedly receives a multiplier | Its sampled residual exceeds `identity_tol`. | Correct sample evaluation or increase `identity_tol` only when the residual is numerical noise and the identity is truly intended. |
| `RIPError` | Maximal declared cliques do not satisfy running intersection. | Use `OVERLAP`, or redesign scopes/lifts so the clique family satisfies RIP. |
| `NotImplementedError` for `EXTEND` | Automatic chordal extension is not implemented. | Use RIP with a valid declared family, OVERLAP, or construct an explicit lifted/extended formulation yourself. |
| Unexpectedly large clique | Entry scopes were overdeclared, or a lift identity couples many variables. | Tighten scopes to true dependencies and inspect `pop.csp.cliques` before building high-degree bases. |
| `DegreeError` for a constraint | Under automatic fitting, its declared degree exceeds `sos_degree`. | Raise `sos_degree`, lower the true declared degree, use a justified lift, or select an explicit multiplier policy intentionally. |
| `sep_degree_insufficient=True` | The selected separator degree is below a conservative local residual degree. | Raise `sep_degree` when full communication of those directions is intended, or document the deliberate restriction. |
| `basis_rank_limited=True` | `basis_rank_limit` truncated at least one final basis. | Treat the formulation as an intentional approximation and compare against an uncapped construction. |
| `NO_MARGIN` | A basis or checked complete clique system used every sample direction. | Add informative sample rows, improve component coverage, or reduce the represented space only with mathematical justification. |
| Very large basis condition or weak singular gap | Sampling geometry or scaling makes rank ambiguous. | Improve sample placement, inspect degree/scope, and test sensitivity to `rank_tol`; do not tune tolerance solely to obtain a desired rank. |
| Compression rank-zero error | The augmented clique system is numerically zero. | Check that objectives and coefficient spaces are not all zero or rank-lost by invalid sampling. |
| Large PQR row-span residual or guard-row growth | Rank cutoff is marginal or rows are ill-conditioned. | Inspect rank diagnostics, sample placement, and scaling. Retaining more rows is conservative. |
| `PipelineError` when modifying POP | POP is finalized. | Rebuild a new POP; finalized problem data are immutable. |
| `PipelineError` when compressing SDPR | SDPR is finalized. | Construct a new SDPR and compress before finalization. |
| `PipelineError` constructing Solver | SDPR is not finalized. | Call `sdpr.finalize()` first. |
| `BackendUnavailableError` for MOSEK | Optional package is missing or no usable license is available. | Install the `mosek` extra and configure a license, or use SCS. |
| `BackendUnavailableError` for SCS | Required native SCS package is unavailable in the environment. | Install the package dependencies. |
| `BackendBuildError` | Native dimensions, coefficient emission, or task construction failed. | Inspect the underlying message and `sdpr.stats`; reproduce with a small fixture and backend assembly tests. |
| `success=False` | Backend normalized status is not optimal or optimal-inaccurate. | Inspect `solution_status` and native diagnostics before changing tolerances. Infeasible/unbounded meanings are already mapped to Rhiza convention. |
| Large primal-dual gap | Termination accuracy is insufficient or conditioning is poor. | Inspect native residuals and scaling; compare tighter tolerances or the other backend. The public gap uses normalized objectives, as described in [§18.3](#183-solve-operation). |
| Extraction before solve or after close | Solver has no available successful extraction state. | Solve and check `report['success']` before extracting, while the Solver is open. A closed Solver must be replaced. |
| Numerically zero clique mass | The returned signed functional cannot be normalized safely. | Inspect backend feasibility, compression, dual cancellation, and formulation degeneracy; do not divide manually. |
| Very large mass condition | Large positive and negative weights cancel. | Compare moments and stitching errors, try tighter solver tolerances or another backend, and avoid interpreting weights as probabilities. |
| Large stitching error | Shared clique moments disagree. | Inspect separator degree/topology, backend residuals, sample qualification, and whether the requested shared function is represented by the separator space. |
| Moment degree error | Requested total exponent exceeds `sos_degree`. | Build a higher-degree SDPR or request a represented lower-degree moment. |
| No clique contains requested moment | Its full variable scope spans several cliques. | Add a justified larger clique/lift, or recover it through application-specific logic outside the documented moment API. |
| Incorrect result after a target rewrite | The declared entry was not truly a polynomial of the lift and remaining variables at the replacement degree. | Revert the rewrite. Rhiza treats target rewrites as explicit user assertions. |
| Poor scaling despite many variables | One or more cliques or quotient bases are large. | Inspect clique widths, basis ranks, Gramian sizes, and separator coefficients. Global variable count alone is not predictive. |
| SCS memory dominates | SCS must assemble one complete sparse CSC cone matrix. | Apply PQR and component splitting, reduce clique sizes, or use MOSEK's streamed task assembly when licensed. |

## 25. Repository and module map

`README.md` provides the introduction, `manual.md` documents the API and numerical model, and `pyproject.toml` defines packaging, dependencies, and pytest configuration. The implementation is organized as follows.

### 25.1 Public-facing modules

| Module | Responsibility |
|---|---|
| `rhiza.__init__` | Explicit exports of `POP`, `SDPR`, `Solver`, `rounding`, public exceptions, and package metadata. |
| `rhiza.pop` | Sampled problem declarations, lifts and target rewrites, `.pop` persistence, input validation, sampled-identity classification, and finalization. |
| `rhiza.csp` | Maximal declared scopes, containment and entry hosting, deterministic clique IDs, and RIP or OVERLAP structural topology. |
| `rhiza.sdpr` | Degree policies, quotient bases, certificate coefficients, normalization, component splitting, separators, rank analysis, compression, and finalized backend data. |
| `rhiza.solver` | One-solve lifecycle, backend selection, reports, copied solution data, and cached moment extraction. |
| `rhiza.rounding` | Binary, sign, one-hot, rotation, and rigid-pose projections described in [Rounding helpers](#20-rounding-helpers). |
| `rhiza.diagnostics` | Exceptions, controlled enums, report formatting, and shared diagnostic vocabulary. |

CSP owns structural sparsity; SDPR chooses polynomial spaces. Native solver conventions belong in the backend adapters.

### 25.2 Backend modules

| Module | Responsibility |
|---|---|
| `rhiza.backends.base` | `BackendWorker` interface and restoration of normalized objective values. |
| `rhiza.backends.scs` | Sparse CSC assembly of the dual cone formulation, native status and objective mapping, and copied sample weights. |
| `rhiza.backends.mosek` | Streaming assembly of the certificate SDP, native diagnostics, and copied equality duals. |

A worker exposes `solve()`, `get_system_duals()`, `stats`, lifecycle properties, and idempotent `close()`. Both adapters consume the same finalized SDPR. Basis construction, separators, component splitting, compression, and normalization are complete before native assembly.

### 25.3 Numerical tool modules

| Module | Responsibility |
|---|---|
| `rhiza.tools.parsing` | `.pop` grammar, scalar formatting, exact integer conversion, and rejection of complex public inputs before real conversion. |
| `rhiza.tools.polyalg` | Exponent enumeration, monomial and lift evaluation, quotient and separator bases, and symmetric Gramian coefficients. |
| `rhiza.tools.linalg` | Stable norms, equilibration, relative singular-value rank decisions, and numerical diagnostics. |
| `rhiza.tools.compression` | Pivoted-QR selection of original rows and an independent row-span audit. |
| `rhiza.tools.extraction` | Row-scale reversal, signed functional reconstruction, mass validation, moment evaluation, and inter-clique comparisons. |

### 25.4 Test organization

| Directory | Coverage |
|---|---|
| `tests/unit/` | Individual contracts, numerical primitives, parsing, public inputs, exports, and backend assembly. |
| `tests/component/` | Cross-module structure, including separators, component splitting, extraction, and lifecycle behavior. |
| `tests/integration/` | Complete pipelines using available native optimizers. |
| `tests/fixtures/` | Reusable sampled POP families for reference, lifting, sparsity, component, and scaling tests. |

## 26. Internal data flow and backend boundary

The public lifecycle is described in [Architecture and lifecycle](#3-first-solve-and-object-lifecycle). The following details matter when inspecting memory use or extending the implementation.

### 26.1 POP compaction

Finalization stores variable evaluations in a matrix of shape `(num_samples, num_variables)` and entry evaluations in a matrix of shape `(num_samples, num_entries)`. Names, relation codes, degrees, sampled-identity flags, and flattened scopes retain declaration order. Variable and entry IDs are stable, and clique IDs are deterministic for the finalized CSP.

The finalized POP retains the original evaluations. SDPR's normalization uses separate data.

### 26.2 Basis and coefficient records

SDPR registers bases by scope and degree. Each local system orders free scalar coefficients before lower-triangle Gramian coefficients. Free coefficients represent bound shares, equality multipliers, and separators. PSD blocks represent base SOS polynomials and inequality SOS multipliers; their off-diagonal coefficient columns include the symmetric factor of two.

Component splitting replaces a logical Gramian with ordinary PSD block records. Backends consume those records without needing the component-discovery history.

### 26.3 Local equation systems

Each maximal declared clique owns a sampled system $`A_k\theta_k=b_k`$. The coefficient matrix contains bound, multiplier, Gramian, and incident separator contributions. Rank analysis and PQR use the augmented matrix $`\overline A_k=[A_k\ b_k]`$, including the objective column so that row selection preserves the affine equations.

Native assembly applies the retained original-row indices and final row scales. Analysis and PQR materialize one complete clique matrix at a time; uncompressed finalization can accumulate row norms from coefficient chunks.

### 26.4 Coefficient chunks and memory

Coefficient generators target chunks of roughly one million array elements. SCS ultimately assembles one global sparse CSC matrix, while MOSEK streams chunks into its native task. Chunking limits temporary assembly memory but does not reduce cone dimensions or solver cost.

Every computed nonzero coefficient is emitted. Introducing a magnitude-pruning threshold could change row spans, separator cancellation, or scale invariance.

### 26.5 Objective orientation and dual conventions

| Backend | Submitted formulation | Data used for extraction |
|---|---|---|
| MOSEK | Certificate SDP, maximizing the bound | Native equality duals |
| SCS | Dual cone formulation | Native primal sample weights |

The adapters map objective fields and infeasibility labels to the same public certificate-primal convention. Both reported objective values are restored through the affine map in [Objective and constraint normalization](#14-objective-and-constraint-normalization). The public relative gap uses normalized objectives; native stopping criteria also depend on backend tolerances and residuals.

### 26.6 Why extraction state belongs to Solver

A clique functional depends on the finalized row maps and scales, a successful solution, and the backend's dual orientation. Solver retrieves the copied system-dual vector once, reverses row scaling, validates masses, and caches retained indices and signed weights. Subsequent moment queries use that cache after native resources have been released. `close()` clears the cached solution state.

### 26.7 Construction caches and immutability

Finalization discards construction-only caches for bases, components, and finite-valued coordinates while preserving the records needed for assembly. Finalized dimensions, coefficient ordering, and row maps are immutable; a different formulation requires a new SDPR.

## 27. Development and testing workflow

### 27.1 Development installation

From the repository root, install the test dependencies with:

```bash
python -m pip install -e ".[test]"
```

For the optional MOSEK interface and its licensed test partition, use:

```bash
python -m pip install -e ".[test,mosek]"
```

Native solver imports remain lazy. Importing Rhiza must not require MOSEK, and SCS must remain usable when MOSEK is unavailable.

### 27.2 Test partitions

The default pytest configuration excludes licensed MOSEK optimization tests:

```bash
python -m pytest
```

This runs the core tests, SCS tests, and license-free MOSEK assembly checks. To select the licensed partition:

```bash
python -m pytest -m mosek
```

Useful focused commands are:

```bash
python -m pytest -m scs
python -m pytest -m "not scs and not mosek"
python -m pytest tests/component/test_sdpr.py
python -m pytest --cov=rhiza
```

The `scs` and `mosek` markers identify backend tests; `scaling` selects native performance tests. For example, `python -m pytest -m "scs and scaling" -s` selects SCS scaling cases. Assertions should use structural quantities or solution invariance rather than portable wall-clock thresholds.

### 27.3 How to test a change

Choose tests according to the affected contract:

| Change | Relevant checks |
|---|---|
| Parsing, casting, numerical primitives, or report fields | Focused unit tests, including invalid inputs and boundary cases. |
| POP-to-SDPR compilation or shared state | Component tests across the affected lifecycle boundary. |
| PQR or component splitting | Compare compressed and uncompressed row spans, or split and unsplit coefficient images and optima. |
| OVERLAP routing | Compare reduced and full pairwise transfer spans; verify connected occurrence groups. |
| Backend assembly, objective orientation, or extraction | Small native solves and cross-backend comparisons when both solvers are available. |

A structural optimization should be checked against the sampled space or mathematical quantity it claims to preserve. Successful execution alone is insufficient.

### 27.4 Coding conventions

Core modules use reading order, 80-character hash section delimiters, brief function docstrings, and no type annotations. Comments explain numerical policy, mathematical purpose, and invariants.

Public arguments are cast in signature order, subject to lifecycle preconditions. Real arrays and scalar settings pass through `cast_real_array()` or `cast_real_scalar()` before shape and finiteness checks. Integral metadata uses exact conversion. Public mutations remain transactional and finalized state remains immutable.

### 27.5 Invariants that must remain explicit

| Mechanism | Invariant |
|---|---|
| OVERLAP routing | Cache occurrence lookups by participating variable set; group monomials by the resulting clique occurrence set. Every group's routing remains connected, and fixed-point pruning only removes existing edges. |
| Component splitting | Finite-valued coordinates only propose partitions. Acceptance checks restriction invariance, projector behavior, disjoint localized ranges, rank recovery, positive margins, and reduced scalarized PSD dimension. |
| Extraction | Reverse row scaling and backend orientation, validate signed mass, and retain compact row-index and weight records. Repeated queries reuse the solution cache. |

Component discovery caps candidate scanning and caches finite cardinalities by global variable ID. Accepted localized basis columns are views into an adapted parent basis, which remains registered because other coefficients may use it. The unsplit PSD coefficient count is accumulated from logical Gramians; the actual count uses submitted blocks. Keep these distinctions when changing memory layout or statistics.

### 27.6 Release checks

Before publishing a distribution:

1. Run the ordinary suite and the licensed MOSEK partition in suitable environments.
2. Build and inspect the wheel, checking dependencies, extras, package metadata, and public exports.
3. Check that the source archive excludes test caches, bytecode, coverage output, and local repository metadata.

## 28. Scope and limitations

The API supports explicit sampled POP construction, RIP and OVERLAP clique policies, and the documented SCS and MOSEK backends. The following capabilities are not implemented:

- **Automatic graph extension.** `EXTEND` is recognized but raises `NotImplementedError`. Supply a RIP-compatible declared family, use OVERLAP, or reformulate scopes explicitly.
- **Automatic symbolic term-sparsity lifting.** Rhiza does not infer polynomial support or construct lifts from symbolic expressions. Use `add_product()`, `add_lift()`, and verified target rewrites.
- **Structured modeling variables.** Each `add_variable()` call supplies one scalar coordinate. Represent vector and matrix variables through scalar coordinates; there is no public variable-shape or custom structured-Gram interface.
- **Independent per-clique sample tables.** The API requires one shared ordered table, even though the mathematical framework permits more general sampling arrangements.

The numerical and certificate-validation limits are described in [Certificate validity and interpretation](#23-certificate-validity-and-interpretation).

## 29. Glossary

| Term | Meaning |
|---|---|
| **Ambient-variety sampling** | Sampling a convenient containing equality variety while representing omitted equality relations explicitly with multiplier-bearing equality entries. |
| **Basis condition** | A diagnostic ratio derived from retained singular values after equilibration. Large values indicate a numerically delicate basis even when its rank is unchanged. |
| **Basis rank** | The retained dimension of a sampled degree-bounded polynomial-function space. An explicit rank limit can reduce it below the empirical numerical rank. |
| **Canonical objective** | The single minimization objective formed by summing `MIN` entries and subtracting `MAX` entries; also called the canonical signed objective. |
| **Clique functional** | The signed linear functional reconstructed from one clique's equation duals and represented by retained sample indices and weights. |
| **Clique ID** | The zero-based integer index of a maximal declared clique in a finalized CSP. |
| **Clique system** | One local sampled coefficient-matching system containing objective, scalar, Gramian, multiplier, and separator contributions hosted on a maximal clique. |
| **Component splitting** | Replacement of one logical Gramian by smaller PSD blocks after numerical restriction-invariance and span checks. The exact criterion preserves the SOS evaluations representable on the samples; accepted splits reduce scalarized PSD coefficients. |
| **Component-localized basis** | A validated restriction-invariant quotient subspace whose sampled evaluations vanish outside one accepted sample component. |
| **Containing clique** | A maximal declared clique whose variable set contains a requested scope. |
| **Correlative sparsity pattern, or CSP** | The structure derived from declared scopes, with maximal declared cliques, containment, entry hosts, and a graph-policy backbone. |
| **Declared scope** | A supplied variable list containing all dependencies of an entry. It may include additional variables. Rhiza trusts this declaration rather than inferring dependencies from sampled numerical variation. |
| **Entry** | One sampled objective or constraint function together with its relation, declared scope, degree, optional name, and sampled evaluation vector. |
| **Functional mass** | The signed weight sum $`m_k=\sum_s y_{k,s}`$ of a reconstructed clique functional. Represented moments are normalized by this mass. |
| **Gramian** | A PSD matrix variable whose quadratic form in a polynomial basis represents a base SOS polynomial or an SOS multiplier. |
| **Guard row** | An additional original sample row retained by PQR when needed to pass the independent row-span audit. |
| **Host clique** | The deterministic containing clique assigned responsibility for a local item such as an entry or scalar contribution. |
| **Logical Gramian** | The mathematical Gramian before optional component splitting. It may correspond to one or several actual PSD blocks in the assembled SDP. |
| **Mass condition** | The ratio $`\lVert y_k\rVert_1/\lvert\sum_s y_{k,s}\rvert`$, measuring cancellation in a signed clique functional. |
| **Maximal declared clique** | A declared scope not strictly contained in another declared scope after deduplication. These scopes define the local certificate systems. |
| **Objective transform** | SDPR's per-clique objective centering and common positive scaling, together with the exact inverse affine map used for public objective reports. |
| **Occurrence group** | In OVERLAP routing, a sampled function subspace formed from monomial directions that have the same set of representing maximal cliques. |
| **Participating variable set** | The variables with nonzero exponents in a monomial. It ignores the exact positive powers and is used to cache OVERLAP occurrences. |
| **Poisedness** | The property that vanishing at the samples implies vanishing on the intended variety for every polynomial in the represented space. Numerical margins alone do not establish it. |
| **Polynomial support** | The set of monomials, equivalently exponent vectors, whose coefficients in a polynomial are nonzero. Rhiza does not infer this support from sampled values. |
| **POP** | A sampled polynomial optimization problem containing scalar variable evaluations, objective and constraint entry evaluations, scopes, degrees, and relations. |
| **PQR** | Pivoted-QR row selection applied independently to each balanced augmented clique system. It retains original sample rows and audits their span. |
| **Quotient basis** | A numerically rank-reduced basis for degree-bounded polynomial functions evaluated on a clique's sampled coordinates, modulo relations visible on the sample set. |
| **RIP** | A clique family satisfies the running intersection property if it admits an ordering in which each clique's overlap with its predecessors lies in one predecessor. An empty overlap imposes no condition. Public clique IDs need not follow such an ordering. |
| **Sample component** | A row group proposed by exactly finite-valued coordinates and accepted after numerical checks that the represented quotient space is invariant under restriction to that group. It is not a general irreducible-component computation. |
| **Sample margin** | The number of sample rows beyond the empirical rank before any deliberate cap. It is $`S-r_{\mathrm{emp}}`$ for an ordinary basis, $`S_a-r_a`$ for a component-localized basis, and $`S-\rho_k`$ for a complete clique system before row selection. |
| **Sample weights** | The signed coefficients representing a clique functional on retained sample rows. They are not assumed to be probabilities or nonnegative measures. |
| **Sample-margin check** | The aggregate coverage and outcome of empirical rank checks for registered bases and complete clique systems. |
| **Sampled identity** | An equality entry whose supplied evaluations have maximum absolute value at most `identity_tol`. It is treated as satisfied and receives no explicit multiplier; quotient relations are inferred separately from the variable samples. |
| **SDPR** | The backend-neutral semidefinite programming relaxation compiled from a finalized POP at chosen SOS, multiplier, and separator degrees. |
| **Separator edge** | A final SDPR clique pair carrying one separator coefficient block. Endpoint signs are opposite, so every transfer cancels in the sum of local sampled identities. |
| **Separator scope** | The variable intersection attached to a CSP structural edge. |
| **Separator space** | The nonconstant degree-bounded sampled quotient-function space transmitted with opposite signs between clique systems. |
| **Stitched moment** | The arithmetic average of normalized containing-clique moments. Their unaveraged values and disagreement remain available. |
| **Structural edge** | An edge in CSP's RIP forest or OVERLAP backbone. A zero-rank separator can omit a structural edge; OVERLAP routing can also add edges when constructing the final polynomial spaces. |
| **Subsumed clique** | A declared scope strictly contained in a maximal declared clique. It does not receive its own certificate system. |
| **System rank** | The numerical rank of one complete balanced augmented clique system. |
| **Term** | An additive summand of a polynomial expression. Rhiza uses **entry**, not term, for one objective or constraint declaration. |
| **Total variation** | The signed-functional magnitude $`\lVert y_k\rVert_1=\sum_s\lvert y_{k,s}\rvert`$. Large total variation can amplify floating-point cancellation. |
| **Unsplit Gramian** | One PSD block spanning the complete quotient basis of a logical Gramian, used when component splitting is disabled or no candidate passes every invariance, margin, and size check. |

## 30. Modeling and development checklists

### 30.1 Constructing and solving a problem

- [ ] Evaluate every variable and entry at the same ordered sample points $`z_1,\ldots,z_S`$. Supply real, finite arrays and keep instance and sampling randomness independently reproducible.
- [ ] Choose the intended sampled variety, cover its relevant local components, and retain omitted equalities explicitly. Check that `identity_tol` does not suppress an intended multiplier.
- [ ] Declare accurate degrees and inequality orientations. Start with small scopes, broadening constraint scopes when the desired multiplier family requires additional variables. Ensure each inequality has a nonzero sample value.
- [ ] Verify lift-target rewrites, then finalize POP and inspect clique scopes, entry hosts, and graph policy.
- [ ] Choose relaxation degrees and inspect `required_sep_degree`, `sep_degree_insufficient`, and `basis_rank_limited`. Check sample margins and conditioning; compare independent sample tables when qualifying a new family.
- [ ] Apply optional PQR before SDPR finalization and inspect its residuals and guard rows. Use `clique_size_threshold` when only smaller cliques should be compressed.
- [ ] Finalize SDPR, solve through a context-managed Solver, and inspect status, the objective gap, and native residuals. Check functional mass and shared-moment disagreement before rounding.
- [ ] Recover physical coordinates and validate the candidate against the original objective and constraints. For numerical discrepancies, compare compressed and uncompressed systems, split and unsplit Gramians, or available backends.

### 30.2 Designing lifts for tightness at lower SOS degrees

A lift adds an exact polynomial expression as a sampled coordinate. It can lower the represented degree of an entry or enrich a low-degree basis, but also adds coordinates and may enlarge cliques. Compare the resulting local algebra and cone sizes, rather than variable count alone.

- [ ] Identify the expressions that force higher objective, multiplier, or separator degrees. Choose a target SOS degree and reuse meaningful products or residuals where possible.
- [ ] Keep private measurement or actuator lifts local. Avoid a wide clique created solely to lower formal degree.
- [ ] Evaluate lifts exactly from the common sample rows. Use `targets` only after verifying that the rewritten entry is a polynomial of the lift and remaining variables at the declared degree.
- [ ] Check whether the added coordinate exposes useful branch-local functions. For example, lifting $`bx`$ can enable component splitting unavailable to $`\mathop{\mathrm{span}}\nolimits\{1,x,b\}`$.
- [ ] Compare original and lifted formulations at the same degree, then at a lower degree. Record bounds, basis ranks, conditioning, sample requirements, PSD and separator sizes, moment agreement, and candidate feasibility.

`add_product()` and `add_lift()` construct exact coordinates. If a coordinate is supplied independently, add it with `add_variable()` and encode the intended polynomial relation with `add_constraint(..., form='==0')`; its evaluations must still use the common sample rows. An approximate relation requires an explicit modeling decision, not an identity implied by a lift call.

**Robust point-cloud registration.** For pair acceptance $`w_{ij}\in\{0,1\}`$ and shared rotation $`R`$, introduce

```math
Z_{ij}=w_{ij}R.
```

For centered points $`\widetilde a_i=a_i-\bar a`$ and $`\widetilde b_j=b_j-\bar b`$, use $`d=t+R\bar a-\bar b`$ with physical translation $`t`$, and introduce $`u_{ij}=w_{ij}d`$. The accepted residual is

```math
g_{ij}=w_{ij}\widetilde b_j-Z_{ij}\widetilde a_i-u_{ij}.
```

Making its components coordinates gives the degree-two TLS contribution

```math
\lVert g_{ij}\rVert_2^2+c^2(1-w_{ij}),
```

where $`c`$ is the truncation threshold. The pose is shared; $`w_{ij}`$, $`Z_{ij}`$, $`u_{ij}`$, and $`g_{ij}`$ remain private to one correspondence clique.

**Perspective-n-point outlier rejection.** Let $`v_i\in\{0,1\}`$ be a veto and $`T=[R\mid t]`$. For a denominator-cleared residual $`r_i(T)`$ linear in $`T`$, introduce $`z_i=v_iT`$. Then

```math
a_i=r_i(T)-r_i(z_i)=(1-v_i)r_i(T),
```

so $`\lVert a_i\rVert_2^2+c^2v_i`$ is degree two in the pose and product coordinates, with truncation threshold $`c`$. Each measurement keeps its veto and product coordinates private while sharing $`T`$.

**Rigid-body trajectory dynamics.** In physical coordinates, lift the gyroscopic products

```math
\kappa_k=\omega_k\times I\omega_k.
```

The angular update then becomes affine in current angular rate, gyro coordinates, and local actuator commands. For the Cayley attitude step, introduce

```math
G_k=\frac12(R_k+R_{k+1})[\omega_{k+1}]_\times,
```

which satisfies

```math
R_{k+1}-R_k=\Delta t\,G_k.
```

Here $`\kappa_k`$ and $`G_k`$ are dynamics coordinates, distinct from the residual-degree bound and SOS Gramians elsewhere in this manual. The implementation scales these coordinates for conditioning. Complete rollout samples satisfy the dynamics relations, so local sampled coordinate rings capture them while preserving a state chain with local actuator scopes.

### 30.3 Modifying the codebase

- [ ] Assign the change to its lifecycle owner and preserve public contracts, transactional mutations, and finalized-state immutability.
- [ ] Identify the sampled-space invariant affected by the change and select checks from [Development and testing workflow](#27-development-and-testing-workflow).
- [ ] Preserve row maps, objective transforms, and backend orientation when changing assembly or extraction.
- [ ] Update documentation for changed behavior, signatures, reports, or mathematical conventions.
