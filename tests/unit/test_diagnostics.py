"""Tests for controlled tokens and readable reports."""

import json

import numpy as np
import pytest

from rhiza.diagnostics import (
    CompressionMethod,
    GraphPolicy,
    POPValidationError,
    Relation,
    format_report,
    format_summary,
    to_python_report,
)


################################################################################
# Tokens and formatting
################################################################################


def test_controlled_token_normalization():
    """Normalize supported uppercase and symbolic relation tokens."""
    assert Relation.from_value('MIN') is Relation.MIN
    assert Relation.from_value('min') is Relation.MIN
    assert Relation.from_value('<=0') is Relation.LE
    assert GraphPolicy.from_value('rip') is GraphPolicy.RIP
    assert CompressionMethod.from_value('pqr') is CompressionMethod.PQR


def test_invalid_relation_token():
    """Reject unsupported relation strings."""
    with pytest.raises(POPValidationError):
        Relation.from_value('minimize')


def test_summary_and_report_formatting():
    """Format status, statistics, and nested reports deterministically."""
    summary = format_summary(
        'Object',
        {'finalized': True},
        {'num_rows': 3},
    )
    report = format_report({'outer': {'value': 2.0}})
    assert 'finalized: True' in summary
    assert 'num rows: 3' in summary
    assert 'outer:' in report
    assert 'value: 2' in report


def test_public_report_conversion_uses_python_scalars():
    """Convert nested NumPy scalar leaves without expanding arrays."""
    array = np.array([1.0, 2.0])
    report = to_python_report({
        'integer': np.int32(3),
        'boolean': np.bool_(True),
        'nested': (np.float64(2.5), {'array': array}),
    })
    assert type(report['integer']) is int
    assert type(report['boolean']) is bool
    assert type(report['nested'][0]) is float
    assert report['nested'][1]['array'] is array

    serializable = dict(report)
    del serializable['nested']
    json.dumps(serializable)
