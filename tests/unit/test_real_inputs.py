"""Tests for Rhiza's real-valued public numerical contract."""

import warnings

import numpy as np
import pytest

import rhiza as rz
from rhiza.diagnostics import POPValidationError, SDPRError, SolveError
from rhiza.tools.parsing import format_evals_field


def _real_pop():
    """Build one mutable POP with an ordinary real sample array."""
    pop = rz.POP()
    pop.add_variable("x", np.array([-1.0, 0.0, 1.0]))
    return pop


def _assert_rejected_without_warning(error_type, function, *args, **kwargs):
    """Require complex input to fail before NumPy emits a casting warning."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(error_type, match='real'):
            function(*args, **kwargs)


@pytest.mark.parametrize(
    'values',
    (
        np.array([1.0 + 2.0j, 3.0 + 4.0j]),
        np.array([1.0 + 0.0j, 3.0 + 0.0j]),
        np.array([1.0 + 2.0j], dtype=object),
        np.array([1.0 + 0.0j], dtype=object),
    ),
)
def test_pop_rejects_complex_sample_evaluations(values):
    """Reject complex arrays for variables, objectives, and constraints."""
    _assert_rejected_without_warning(
        POPValidationError,
        rz.POP().add_variable,
        "z",
        values,
    )

    pop = _real_pop()
    _assert_rejected_without_warning(
        POPValidationError,
        pop.add_objective,
        values,
        ('x',),
        1,
    )
    _assert_rejected_without_warning(
        POPValidationError,
        pop.add_constraint,
        values,
        ('x',),
        1,
        form='==0',
    )


def test_lift_and_file_format_reject_complex_arrays():
    """Keep complex coefficients and serialized evaluations out of real POPs."""
    pop = _real_pop()
    _assert_rejected_without_warning(
        POPValidationError,
        pop.add_lift,
        ('x',),
        ((1,),),
        name="scaled_x",
        coeffs=np.array([1.0 + 0.0j]),
    )
    _assert_rejected_without_warning(
        POPValidationError,
        format_evals_field,
        np.array([1.0 + 2.0j]),
    )


@pytest.mark.parametrize(
    'function, value',
    (
        (rz.rounding.project_so3, np.eye(3, dtype=complex)),
        (
            rz.rounding.project_se3,
            np.column_stack((np.eye(3), [0.0, 0.0, 0.0])).astype(complex),
        ),
        (rz.rounding.round_binary, np.array([0.0 + 0.0j, 1.0 + 0.0j])),
        (rz.rounding.round_sign, np.array([-1.0 + 1.0j, 1.0 + 0.0j])),
        (rz.rounding.round_one_hot, np.array([1.0 + 0.0j, 2.0 + 0.0j])),
    ),
)
def test_rounding_rejects_complex_arrays(function, value):
    """Reject complex manifold and discrete-rounding input consistently."""
    _assert_rejected_without_warning(ValueError, function, value)


def test_public_scalar_settings_reject_complex_values():
    """Reject complex tolerances and rounding thresholds before scalar casts."""
    pop = _real_pop()
    pop.add_objective([1.0, 0.0, 1.0], ('x',), 2)
    _assert_rejected_without_warning(
        POPValidationError,
        pop.finalize,
        identity_tol=np.complex128(1e-12 + 0.0j),
    )

    pop.finalize()
    _assert_rejected_without_warning(
        SDPRError,
        rz.SDPR,
        pop,
        rank_tol=np.complex128(1e-10 + 0.0j),
    )

    sdpr = rz.SDPR(pop)
    sdpr.finalize()
    _assert_rejected_without_warning(
        SolveError,
        rz.Solver,
        sdpr,
        gap_tol=np.complex128(1e-5 + 0.0j),
    )
    _assert_rejected_without_warning(
        ValueError,
        rz.rounding.round_binary,
        [0.0, 1.0],
        threshold=np.complex128(0.5 + 0.0j),
    )
    _assert_rejected_without_warning(
        ValueError,
        rz.rounding.round_sign,
        [0.0],
        zero_value=np.complex128(1.0 + 0.0j),
    )


def test_real_object_arrays_remain_accepted():
    """Preserve ordinary numerical conversion for real object arrays."""
    pop = rz.POP()
    pop.add_variable("x", np.array([-1, 0, 1], dtype=object))
    pop.add_objective(
        np.array([1.0, 0.0, 1.0], dtype=object),
        ('x',),
        2,
    )
    assert pop.finalize()
