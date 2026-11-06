"""SCS integration test for direct sample-weight extraction."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import rhiza as rz


################################################################################
# Test configuration
################################################################################


pytestmark = pytest.mark.scs


################################################################################
# Optimizer sample concentration
################################################################################


def test_optimizer_sample_weight_extraction():
    """Concentrate the extracted sample functional on a unique optimizer."""
    x_evals = np.array([-1.0, 0.0, 1.0])
    pop = rz.POP()
    pop.add_variable("x", x_evals)
    pop.add_objective(x_evals**2, ("x",), 2, name="cost")
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2, rank_tol=1e-12)
    sdpr.finalize()

    with rz.Solver(
        sdpr,
        gap_tol=1e-7,
        feas_tol=1e-7,
    ) as solver:
        report = solver.solve()
        assert report['success']
        assert report['solution_status'] == 'OPTIMAL'
        functional = solver._clique_functionals()[0]
        normalized = np.zeros(pop.num_samples)
        normalized[functional.sample_indices] = (
            functional.weights/functional.mass
        )
        np.testing.assert_allclose(
            normalized, (0.0, 1.0, 0.0), atol=2e-4
        )
        means = solver.extract_means()
        np.testing.assert_allclose(
            means['x'], 0.0, atol=2e-5, rtol=0.0
        )
        assert solver.stats['num_rows'] == sdpr.stats['num_rows_retained']


def test_infeasible_certificate_status_reverses_scs_dual_orientation():
    """Report certificate infeasibility when native SCS is dual-unbounded."""
    x_evals = np.linspace(-2.0, 2.0, 64)
    pop = rz.POP()
    pop.add_variable('x', x_evals)
    pop.add_objective(x_evals**3, ('x',), 3)
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2)
    sdpr.finalize()

    with rz.Solver(sdpr) as solver:
        report = solver.solve()
        assert not report['success']
        assert report['solution_status'] == 'INFEASIBLE'
        assert 'termination_status' not in report
        assert solver.stats['native']['termination_status'] == 'unbounded'


def test_scs_solve_does_not_require_the_optional_mosek_package():
    """Import and solve with SCS while every native MOSEK import is blocked."""
    script = r"""
import importlib.abc
import sys

import numpy as np


class BlockMosek(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mosek' or fullname.startswith('mosek.'):
            raise ImportError('MOSEK intentionally unavailable')
        return None


sys.meta_path.insert(0, BlockMosek())
import rhiza as rz

x_evals = np.linspace(-1.0, 1.0, 16)
pop = rz.POP()
pop.add_variable('x', x_evals)
pop.add_objective(x_evals**2, ('x',), 2)
pop.finalize()
sdpr = rz.SDPR(pop, sos_degree=2)
sdpr.finalize()
with rz.Solver(sdpr) as solver:
    report = solver.solve()
assert report['success']
assert report['solution_status'] == 'OPTIMAL'
"""
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, '-c', script],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
