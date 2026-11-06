"""MOSEK integration test for direct sample-weight extraction."""

import numpy as np
import pytest

import rhiza as rz
from rhiza.diagnostics import BackendUnavailableError


################################################################################
# Test configuration
################################################################################


pytestmark = pytest.mark.mosek


################################################################################
# Optimizer sample concentration
################################################################################


def test_optimizer_sample_weight_extraction():
    """Concentrate the extracted sample functional on a unique optimizer."""
    pytest.importorskip('mosek')
    x_evals = np.array([-1.0, 0.0, 1.0])
    pop = rz.POP()
    pop.add_variable("x", x_evals)
    pop.add_objective(x_evals**2, ("x",), 2, name="cost")
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2, rank_tol=1e-12)
    sdpr.finalize()

    try:
        solver = rz.Solver(sdpr, backend='MOSEK', gap_tol=1e-10)
    except BackendUnavailableError:
        pytest.skip("MOSEK is unavailable in this environment.")

    with solver:
        try:
            report = solver.solve()
        except BackendUnavailableError:
            pytest.skip("A usable MOSEK license is unavailable.")
        assert report['success']
        functional = solver._clique_functionals()[0]
        normalized = np.zeros(pop.num_samples)
        normalized[functional.sample_indices] = (
            functional.weights/functional.mass
        )
        np.testing.assert_allclose(normalized, (0.0, 1.0, 0.0), atol=2e-7)
        means = solver.extract_means()
        np.testing.assert_allclose(means['x'], 0.0, atol=2e-8, rtol=0.0)
        assert solver.stats['native']['num_rows'] == sdpr.stats['num_rows_retained']
