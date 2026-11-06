"""Tests for the cross-language .pop parser and writer."""

import numpy as np
import pytest

from rhiza.diagnostics import POPParseError, POPValidationError, Relation
from rhiza.tools.parsing import (
    ParsedRow,
    format_pop_row,
    iter_pop_rows,
    parse_pop_line,
    write_pop_rows,
)


################################################################################
# Relation and row parsing
################################################################################


@pytest.mark.parametrize(
    'token, relation',
    (
        ('min', Relation.MIN),
        ('MIN', Relation.MIN),
        ('max', Relation.MAX),
        ('MAX', Relation.MAX),
        ('var', Relation.VAR),
        ('VAR', Relation.VAR),
        ('VaR', Relation.VAR),
        ('mIn', Relation.MIN),
        ('MaX', Relation.MAX),
        ('>=0', Relation.GE),
        ('<=0', Relation.LE),
        ('==0', Relation.EQ),
    ),
)
def test_relation_tokens(token, relation):
    """Normalize supported variable and objective relation tokens."""
    row = parse_pop_line(token + ',2,x,1,2')
    assert row.relation is relation


def test_parse_valid_row():
    """Parse one ordinary sampled objective row."""
    row = parse_pop_line('min,4,x-y,1,-2.5,3e-4', 7)
    assert row.degree == 4
    assert row.scope == ('x', 'y')
    assert row.line_number == 7
    np.testing.assert_allclose(row.evals, [1.0, -2.5, 3e-4])


@pytest.mark.parametrize(
    'text',
    (
        'min,2,x',
        'min,nope,x,1',
        'min,2,,1',
        'min,2,x-x,1',
        'min,2,x,1,nope',
        'min,2,x,1,2oops',
        'min,2,x,1,2,',
        'min,2,x,nan',
        'other,2,x,1',
    ),
)
def test_parse_invalid_rows(text):
    """Reject malformed metadata and numerical payloads."""
    with pytest.raises(POPParseError):
        parse_pop_line(text)


################################################################################
# Formatting and file round trips
################################################################################


def test_format_uses_seventeen_significant_digits():
    """Round-trip arbitrary finite float64 values through text."""
    rng = np.random.default_rng(2)
    values = rng.normal(size=1000)
    row = ParsedRow(Relation.MIN, 2, ('x',), values)
    parsed = parse_pop_line(format_pop_row(row))
    assert np.array_equal(parsed.evals, values)


def test_formatting_integral_options_are_not_truncated():
    """Accept exact real integers and reject fractional row metadata."""
    row = ParsedRow(Relation.MIN, 2.0, "theta", np.array([0.0, 1.0]))
    text = format_pop_row(row, float_precision=17.0)
    assert text.startswith('min,2,theta,')

    with pytest.raises(POPValidationError, match='integer'):
        format_pop_row(row, float_precision=16.5)
    fractional = ParsedRow(
        Relation.MIN, 2.5, ('theta',), np.array([0.0, 1.0])
    )
    with pytest.raises(POPValidationError, match='integer'):
        format_pop_row(fractional)
    with pytest.raises(POPParseError, match='integer'):
        parse_pop_line('min,2,theta,0,1', line_number=1.5)


def test_format_scope_rejects_ambiguous_names():
    """Prevent the writer from emitting scopes that cannot round-trip exactly."""
    bad_scopes = (
        (' x',),
        ('x-y',),
        ('x', 'x'),
    )
    for scope in bad_scopes:
        row = ParsedRow(Relation.MIN, 2, scope, np.array([0.0, 1.0]))
        with pytest.raises(POPValidationError):
            format_pop_row(row)


def test_file_round_trip(tmp_path):
    """Write and reload the supported row forms."""
    path = tmp_path/'problem.pop'
    rows = (
        ParsedRow(Relation.VAR, 1, ('x',), np.array([0.0, 1.0])),
        ParsedRow(Relation.MIN, 2, ('x',), np.array([0.0, 1.0])),
        ParsedRow(Relation.GE, 1, ('x',), np.array([1.0, 2.0])),
    )
    assert write_pop_rows(path, rows)
    loaded = tuple(iter_pop_rows(path))
    assert [row.relation for row in loaded] == [
        Relation.VAR,
        Relation.MIN,
        Relation.GE,
    ]
    lines = path.read_text().splitlines()
    assert lines[0].startswith('var,1,x,')
    assert lines[1].startswith('min,2,x,')


def test_file_error_has_line_context(tmp_path):
    """Include the source path and line number in parse failures."""
    path = tmp_path/'bad.pop'
    path.write_text('var,1,x,0\nmin,2,x,nope\n')
    with pytest.raises(POPParseError, match='line 2'):
        tuple(iter_pop_rows(path))


@pytest.mark.parametrize('token', ('EQ', 'GE', 'LE'))
def test_file_parser_rejects_internal_relation_tokens(token):
    """Keep internal constraint labels outside the file grammar."""
    with pytest.raises(POPParseError):
        parse_pop_line(token + ',2,x,1,2')


def test_float64_extremes_round_trip_exactly():
    """Preserve normal, subnormal, adjacent, and signed-zero values."""
    values = np.array((
        np.finfo(float).max,
        -np.finfo(float).max,
        np.finfo(float).tiny,
        -np.finfo(float).tiny,
        np.nextafter(0.0, 1.0),
        -np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 2.0),
        np.nextafter(1.0, 0.0),
        0.0,
        -0.0,
        1e-300,
        -1e300,
    ))
    row = ParsedRow(Relation.MIN, 2, ('x',), values)
    parsed = parse_pop_line(format_pop_row(row))
    assert np.array_equal(parsed.evals, values)
    assert np.array_equal(np.signbit(parsed.evals), np.signbit(values))


@pytest.mark.parametrize(
    'line',
    (
        'var,1, x,0,1',
        'var,1,x ,0,1',
        'min,2,x- y,0,1',
        'min,2,x -y,0,1',
    ),
)
def test_file_parser_rejects_scope_whitespace(line):
    """Treat variable names as literal tokens rather than stripped text."""
    with pytest.raises(POPParseError, match='whitespace'):
        parse_pop_line(line)
