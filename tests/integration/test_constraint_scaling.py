"""Native regression tests for constraint-scale invariance."""

import json

import numpy as np
import pytest

import rhiza as rz


_EQUALITY_SCALES = (-1e12, -1e-12, 1e-12, 1.0, 1e12)
_INEQUALITY_SCALES = (1e-12, 1e-8, 1.0, 1e8, 1e12)
_COMPRESSED_CASES = (
    ('==0', -1e-12),
    ('==0', 1e12),
    ('>=0', 1e-12),
    ('>=0', 1e12),
)


def _make_problem(scale, *, form, use_compression=False):
    """Construct one analytically solved one-variable constrained problem."""
    x_evals = np.linspace(-2.0, 2.0, 64)
    pop = rz.POP()
    pop.add_variable('x', x_evals)
    if form == '==0':
        pop.add_objective((x_evals - 1.0)**2, ('x',), 2)
    else:
        pop.add_objective((x_evals + 1.0)**2, ('x',), 2)
    pop.add_constraint(scale*x_evals, ('x',), 1, form=form)
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2)
    if use_compression:
        sdpr.compress(method='PQR')
    sdpr.finalize()
    return sdpr


def _solve_scaled_problem(
    scale,
    *,
    form,
    backend,
    use_compression=False,
):
    """Solve one normalized representation and return its bound and moment."""
    sdpr = _make_problem(
        scale,
        form=form,
        use_compression=use_compression,
    )
    tolerance = 1e-7 if backend == 'SCS' else 1e-9
    with rz.Solver(
        sdpr,
        backend=backend,
        gap_tol=tolerance,
        feas_tol=tolerance,
        iter_limit=100000,
    ) as solver:
        report = solver.solve(verbose=False)
        assert report['success']
        json.dumps(report)
        json.dumps(solver.stats)
        return report['primal_objective'], solver.extract_means()['x']


@pytest.mark.scs
@pytest.mark.parametrize('scale', _EQUALITY_SCALES)
def test_scs_equality_scale_invariance(scale):
    """Preserve an equality-constrained optimum across legal rescaling."""
    bound, mean = _solve_scaled_problem(scale, form='==0', backend='SCS')
    np.testing.assert_allclose(bound, 1.0, atol=2e-5, rtol=0.0)
    np.testing.assert_allclose(mean, 0.0, atol=2e-5, rtol=0.0)


@pytest.mark.scs
@pytest.mark.parametrize('scale', _INEQUALITY_SCALES)
def test_scs_inequality_scale_invariance(scale):
    """Preserve an inequality-constrained optimum across positive rescaling."""
    bound, mean = _solve_scaled_problem(scale, form='>=0', backend='SCS')
    np.testing.assert_allclose(bound, 1.0, atol=2e-5, rtol=0.0)
    np.testing.assert_allclose(mean, 0.0, atol=2e-5, rtol=0.0)


@pytest.mark.mosek
@pytest.mark.parametrize('scale', _EQUALITY_SCALES)
def test_mosek_equality_scale_invariance(scale):
    """Preserve an equality-constrained optimum across legal rescaling."""
    pytest.importorskip('mosek')
    bound, mean = _solve_scaled_problem(scale, form='==0', backend='MOSEK')
    np.testing.assert_allclose(bound, 1.0, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(mean, 0.0, atol=2e-6, rtol=0.0)


@pytest.mark.mosek
@pytest.mark.parametrize('scale', _INEQUALITY_SCALES)
def test_mosek_inequality_scale_invariance(scale):
    """Preserve an inequality-constrained optimum across positive rescaling."""
    pytest.importorskip('mosek')
    bound, mean = _solve_scaled_problem(scale, form='>=0', backend='MOSEK')
    np.testing.assert_allclose(bound, 1.0, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(mean, 0.0, atol=2e-6, rtol=0.0)

@pytest.mark.scs
@pytest.mark.parametrize('form, scale', _COMPRESSED_CASES)
def test_scs_compressed_constraint_scale_invariance(form, scale):
    """Preserve normalized constraints through PQR row selection."""
    bound, mean = _solve_scaled_problem(
        scale,
        form=form,
        backend='SCS',
        use_compression=True,
    )
    np.testing.assert_allclose(bound, 1.0, atol=2e-5, rtol=0.0)
    np.testing.assert_allclose(mean, 0.0, atol=2e-5, rtol=0.0)


@pytest.mark.mosek
@pytest.mark.parametrize('form, scale', _COMPRESSED_CASES)
def test_mosek_compressed_constraint_scale_invariance(form, scale):
    """Preserve normalized constraints through native compressed solves."""
    pytest.importorskip('mosek')
    bound, mean = _solve_scaled_problem(
        scale,
        form=form,
        backend='MOSEK',
        use_compression=True,
    )
    np.testing.assert_allclose(bound, 1.0, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(mean, 0.0, atol=2e-6, rtol=0.0)

