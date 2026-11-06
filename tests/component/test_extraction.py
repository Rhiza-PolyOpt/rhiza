"""Component tests for moment extraction from synthetic clique duals."""

import json

import numpy as np
import pytest

import rhiza as rz
from rhiza.diagnostics import ExtractionError
from rhiza.tools.extraction import (
    cast_clique_ids,
    cast_moment_spec,
    compute_clique_mass,
    extract_clique_moments,
    extract_moment_diagnostics,
    extract_means,
    extract_moments,
    reconstruct_clique_functionals,
    validate_clique_mass,
)


################################################################################
# Extraction fixture
################################################################################


def make_extraction_sdpr():
    """Construct a two-clique SDPR with simple sample coordinates."""
    pop = rz.POP()
    pop.add_variable("x", [0.0, 1.0, 2.0, 3.0])
    pop.add_variable("y", [10.0, 20.0, 30.0, 40.0])
    pop.add_variable("z", [5.0, 6.0, 7.0, 8.0])
    pop.add_objective(
        [0.0, 0.0, 0.0, 0.0],
        ('x', 'y'),
        1,
        form='MIN',
    )
    pop.add_objective(
        [0.0, 0.0, 0.0, 0.0],
        ('y', 'z'),
        1,
        form='MIN',
    )
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=2)
    sdpr.finalize()
    return sdpr


def duals_for_weights(sdpr, weights_by_clique):
    """Convert sample weights into submitted equation duals."""
    pieces = []
    for clique_id, system in enumerate(sdpr._clique_systems):
        weights = np.asarray(weights_by_clique[clique_id], dtype=float)
        pieces.append(
            weights[system['row_indices']]/system['row_scales']
        )
    return np.concatenate(pieces)


def functionals_for_weights(sdpr, weights_by_clique, *, mass_abs_tol=1e-10):
    """Reconstruct compact clique functionals from desired sample weights."""
    return reconstruct_clique_functionals(
        sdpr,
        duals_for_weights(sdpr, weights_by_clique),
        mass_abs_tol=mass_abs_tol,
    )


################################################################################
# Means and moments
################################################################################


def test_extract_means_and_moments():
    """Recover moments from matching clique masses."""
    sdpr = make_extraction_sdpr()
    weights = {
        0: np.array([0.0, 1.0, 0.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0, 0.0]),
    }
    functionals = functionals_for_weights(sdpr, weights)
    means = extract_means(sdpr, functionals)
    moments = extract_moments(
        sdpr,
        functionals,
        {
            "x_squared": {"x": 2},
            "yz": {"y": 1, "z": 1},
        },
    )
    assert means == {'x': 1.0, 'y': 20.0, 'z': 6.0}
    assert extract_means(
        sdpr, functionals, variables=("x",)
    ) == {'x': 1.0}
    assert moments == {'x_squared': 1.0, 'yz': 120.0}


def test_extract_clique_moments():
    """Return only moments whose scopes fit each requested clique."""
    sdpr = make_extraction_sdpr()
    weights = {
        0: np.array([1.0, 0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0, 0.0]),
    }
    functionals = functionals_for_weights(sdpr, weights)
    clique_moments = extract_clique_moments(
        sdpr,
        functionals,
        {
            "x": {"x": 1},
            "z": {"z": 1},
        },
        cliques=(0, 1),
    )
    assert clique_moments[0] == {'x': 0.0}
    assert clique_moments[1] == {'z': 5.0}


def test_extract_moment_diagnostics_reports_disagreement():
    """Measure shared-variable disagreement without thresholding it."""
    sdpr = make_extraction_sdpr()
    weights = {
        0: np.array([0.0, 1.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 1.0, 0.0]),
    }
    functionals = functionals_for_weights(sdpr, weights)
    report = extract_moment_diagnostics(
        sdpr,
        functionals,
        variables=("y",),
    )
    np.testing.assert_allclose(
        tuple(report['mass_by_clique'].values()), (1.0, 1.0)
    )
    assert set(report['means_by_variable']['y']) == {0, 1}
    np.testing.assert_allclose(
        tuple(report['total_variation_by_clique'].values()),
        (1.0, 1.0),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        tuple(report['mass_condition_by_clique'].values()),
        (1.0, 1.0),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        tuple(report['mass_error_by_clique'].values()),
        (0.0, 0.0),
        rtol=0.0,
        atol=1e-15,
    )
    assert report['stitching_error_max_by_variable']['y'] == 10.0
    assert report['stitching_error_rms_by_variable']['y'] == 5.0


def test_high_cancellation_mass_remains_extractable():
    """Normalize a valid signed functional despite large cancellation."""
    sdpr = make_extraction_sdpr()

    # The fourth finite-difference vector annihilates every univariate
    # polynomial of degree at most two on samples 0, 1, 2, 3. Adding a large
    # multiple changes the dual representative without changing any moment
    # represented by this degree-two relaxation.
    null_weights = 5e7*np.array([-1.0, 3.0, -3.0, 1.0])
    optimizer_atom = np.array([0.0, 1.0, 0.0, 0.0])
    weights = optimizer_atom + null_weights
    functionals = functionals_for_weights(
        sdpr, {0: weights, 1: weights}
    )

    means = extract_means(sdpr, functionals)
    np.testing.assert_allclose(
        tuple(means.values()),
        (1.0, 20.0, 6.0),
        rtol=0.0,
        atol=1e-7,
    )

    report = extract_moment_diagnostics(sdpr, functionals)
    np.testing.assert_allclose(
        tuple(report['mass_by_clique'].values()),
        (1.0, 1.0),
        rtol=0.0,
        atol=2e-8,
    )
    assert min(report['total_variation_by_clique'].values()) > 1e8
    assert min(report['mass_condition_by_clique'].values()) > 1e8
    assert max(report['mass_error_by_clique'].values()) <= 2e-8


################################################################################
# Validation and compact dual reconstruction
################################################################################


def test_reconstructs_compressed_duals_compactly():
    """Reverse row scaling without allocating full-sample weight vectors."""
    class _SyntheticSDPR:
        """Supply the finalized row layout needed by reconstruction."""

        _clique_systems = ({
            'row_indices': np.array([1, 3]),
            'row_scales': np.array([0.5, 2.0]),
        },)

    functional, = reconstruct_clique_functionals(
        _SyntheticSDPR(),
        np.array([4.0, 3.0]),
        mass_abs_tol=1e-10,
    )
    np.testing.assert_array_equal(functional.sample_indices, [1, 3])
    np.testing.assert_allclose(functional.weights, [2.0, 6.0])
    assert functional.mass == 8.0


def test_extraction_input_casts_preserve_integral_metadata():
    """Reject lossy moment and clique coercion while accepting exact reals."""
    assert cast_moment_spec(
        {"x_squared": {"x": 2.0}}
    ) == {"x_squared": {"x": 2}}
    assert cast_clique_ids((0.0,)) == (0,)

    with pytest.raises(ExtractionError, match='integer'):
        cast_moment_spec({"bad": {"x": 1.5}})
    with pytest.raises(ExtractionError, match='string'):
        cast_moment_spec({"bad": {None: 1}})
    with pytest.raises(ExtractionError, match='integer'):
        cast_clique_ids((0.5,))


def test_invalid_high_degree_moment():
    """Reject moments beyond the represented SOS degree."""
    sdpr = make_extraction_sdpr()
    functionals = functionals_for_weights(sdpr, {
        0: np.ones(4),
        1: np.ones(4),
    })
    with pytest.raises(ExtractionError, match='exceeds'):
        extract_moments(
            sdpr,
            functionals,
            {"x_cubed": {"x": 3}},
        )


def test_invalid_cross_clique_moment():
    """Reject a moment scope absent from every clique."""
    sdpr = make_extraction_sdpr()
    functionals = functionals_for_weights(sdpr, {
        0: np.ones(4),
        1: np.ones(4),
    })
    with pytest.raises(ExtractionError, match='No clique'):
        extract_moments(
            sdpr,
            functionals,
            {"xz": {"x": 1, "z": 1}},
        )


def test_near_zero_mass_fails():
    """Reject normalized extraction from a zero-mass functional."""
    sdpr = make_extraction_sdpr()
    duals = duals_for_weights(sdpr, {
        0: np.array([1.0, -1.0, 0.0, 0.0]),
        1: np.ones(4),
    })
    with pytest.raises(ExtractionError, match='zero mass'):
        reconstruct_clique_functionals(
            sdpr, duals, mass_abs_tol=1e-10
        )


@pytest.mark.parametrize('nonfinite', (np.nan, np.inf, -np.inf))
def test_nonfinite_functional_fails_contextually(nonfinite):
    """Reject nonfinite dual weights with a Rhiza extraction error."""
    sdpr = make_extraction_sdpr()
    duals = duals_for_weights(sdpr, {
        0: np.array([nonfinite, 0.0, 0.0, 0.0]),
        1: np.ones(4),
    })
    with pytest.raises(ExtractionError, match='finite vector'):
        reconstruct_clique_functionals(
            sdpr, duals, mass_abs_tol=1e-10
        )


def test_unrepresentable_functional_magnitudes_fail_cleanly():
    """Reject overflow in signed mass and total-variation reductions."""
    maximum = np.finfo(float).max
    with pytest.raises(ExtractionError, match='total mass'):
        compute_clique_mass(np.array([maximum, maximum]))

    weights = np.array([maximum, -maximum, 1.0])
    mass = compute_clique_mass(weights)
    assert mass == 1.0
    with pytest.raises(ExtractionError, match='total variation'):
        validate_clique_mass(
            weights,
            mass,
            mass_abs_tol=1e-10,
            clique_id=0,
        )


def test_moment_diagnostics_are_json_serializable():
    """Expose signed-functional and stitching diagnostics as plain data."""
    sdpr = make_extraction_sdpr()
    functionals = functionals_for_weights(sdpr, {
        0: np.array([0.0, 1.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 1.0, 0.0]),
    })
    report = extract_moment_diagnostics(sdpr, functionals)
    assert set(report) == {
        'mass_by_clique',
        'total_variation_by_clique',
        'mass_condition_by_clique',
        'mass_error_by_clique',
        'means_by_variable',
        'stitching_error_max_by_variable',
        'stitching_error_rms_by_variable',
        'stitching_error_max',
        'stitching_error_rms',
    }
    json.dumps(report)
