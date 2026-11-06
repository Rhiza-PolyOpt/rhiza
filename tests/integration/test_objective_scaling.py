"""Native regression tests for exact affine objective normalization."""

import json

import numpy as np
import pytest

import rhiza as rz


_OBJECTIVE_TRANSFORMS = (
    (1e-12, 0.0),
    (1.0, 1e6),
    (1e12, -2e5),
    (1e-4, 3.0),
)


def _solve_affine_objective(
    scale,
    offset,
    *,
    backend,
    use_compression,
):
    """Solve one positive affine transform of a quadratic objective."""
    x_evals = np.linspace(-2.0, 2.0, 80)
    objective_evals = scale*(x_evals - 0.7)**2 + offset
    pop = rz.POP()
    pop.add_variable('x', x_evals)
    pop.add_objective(objective_evals, ('x',), 2)
    pop.finalize()

    sdpr = rz.SDPR(pop, sos_degree=2)
    if use_compression:
        sdpr.compress(method='PQR')
    sdpr.finalize()

    tolerance = 1e-7 if backend == 'SCS' else 1e-8
    with rz.Solver(
        sdpr,
        backend=backend,
        gap_tol=tolerance,
        feas_tol=tolerance,
        iter_limit=100000,
    ) as solver:
        report = solver.solve()
        assert report['success']
        assert report['solution_status'] == 'OPTIMAL'
        assert 'termination_status' not in report
        json.dumps(report)
        json.dumps(solver.stats)
        mean = solver.extract_means()['x']

    return {
        'primal_normalized': (
            report['primal_objective'] - offset
        )/scale,
        'dual_normalized': (
            report['dual_objective'] - offset
        )/scale,
        'mean': mean,
        'objective_scale': sdpr.stats['objective_scale'],
        'objective_offset': sdpr.stats['objective_offset'],
    }


@pytest.mark.scs
@pytest.mark.parametrize('scale, offset', _OBJECTIVE_TRANSFORMS)
@pytest.mark.parametrize('use_compression', (False, True))
def test_scs_objective_affine_invariance(scale, offset, use_compression):
    """Preserve the SCS bound and first moment across objective units and offsets."""
    result = _solve_affine_objective(
        scale,
        offset,
        backend='SCS',
        use_compression=use_compression,
    )
    np.testing.assert_allclose(
        result['primal_normalized'], 0.0, atol=2e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        result['dual_normalized'], 0.0, atol=2e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        result['mean'], 0.7, atol=2e-4, rtol=0.0
    )
    assert result['objective_scale'] > 0.0
    assert np.isfinite(result['objective_offset'])


@pytest.mark.mosek
@pytest.mark.parametrize('scale, offset', _OBJECTIVE_TRANSFORMS)
@pytest.mark.parametrize('use_compression', (False, True))
def test_mosek_objective_affine_invariance(scale, offset, use_compression):
    """Preserve the MOSEK bound and first moment across objective transforms."""
    pytest.importorskip('mosek')
    result = _solve_affine_objective(
        scale,
        offset,
        backend='MOSEK',
        use_compression=use_compression,
    )
    np.testing.assert_allclose(
        result['primal_normalized'], 0.0, atol=2e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        result['dual_normalized'], 0.0, atol=2e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        result['mean'], 0.7, atol=2e-4, rtol=0.0
    )
    assert result['objective_scale'] > 0.0
    assert np.isfinite(result['objective_offset'])
