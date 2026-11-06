"""Tests for the intentionally exported Rhiza package surface."""

import rhiza


def test_public_export_list_is_complete_and_resolvable():
    """Expose the principal classes, helpers, errors, and package version."""
    expected = {
        'POP',
        'SDPR',
        'Solver',
        'rounding',
        'RhizaError',
        'POPError',
        'POPParseError',
        'POPValidationError',
        'PipelineError',
        'CSPError',
        'GraphPolicyError',
        'RIPError',
        'SDPRError',
        'DegreeError',
        'NumericalRankError',
        'CompressionError',
        'BackendError',
        'BackendUnavailableError',
        'BackendBuildError',
        'SolveError',
        'ExtractionError',
        '__version__',
    }

    assert set(rhiza.__all__) == expected
    assert all(hasattr(rhiza, name) for name in rhiza.__all__)
    assert rhiza.POP.__module__ == 'rhiza.pop'
    assert rhiza.SDPR.__module__ == 'rhiza.sdpr'
    assert rhiza.Solver.__module__ == 'rhiza.solver'
