"""Parsing and formatting for the row-oriented .pop file format.

The module preserves the exact exchange grammar independently of the richer
validation and naming semantics implemented by POP.
"""

import numbers
import re
import warnings

import numpy as np

from rhiza.diagnostics import POPParseError, POPValidationError, Relation


_NUMBER_PATTERN = (
    r'[+-]?'
    r'(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))'
    r'(?:[eE][+-]?[0-9]+)?'
)
_EVALS_PATTERN = re.compile(
    r'\s*' + _NUMBER_PATTERN
    + r'(?:\s*,\s*' + _NUMBER_PATTERN + r')*\s*'
)


################################################################################
# File operations
################################################################################


def iter_pop_rows(path):
    """Yield parsed rows from a .pop file. The iterator preserves the mathematical
    ordering while bounding temporary memory during downstream assembly.

    Inputs:
        path:
            String. Path to the .pop file.

    Outputs:
        rows:
            Iterator. ParsedRow objects in file order.
    """
    path = str(path)

    # Open explicitly so filesystem errors can be translated before iteration
    # begins, while the context manager still closes the stream on parse errors.
    try:
        handle = open(path, 'r', encoding='utf-8')
    except OSError as error:
        raise POPParseError("Could not open .pop file: " + path) from error

    with handle:
        # Preserve physical line numbers for contextual errors; blank lines are
        # presentation only and do not create ParsedRow records.
        for line_number, text in enumerate(handle, start=1):
            text = text.strip()
            if not text:
                continue
            try:
                yield parse_pop_line(text, line_number)
            except POPParseError as error:
                raise POPParseError(
                    f"{path}, line {line_number}: {error}"
                ) from error


def write_pop_rows(path, rows, *, float_precision=17):
    """Write parsed rows in the canonical .pop exchange format.

    Inputs:
        path:
            String. Output path.
        rows:
            Iterable. ParsedRow objects.
        float_precision:
            Integer. Significant decimal digits.

    Outputs:
        written:
            Boolean. True after writing.
    """
    path = str(path)
    float_precision = _cast_positive_int(
        float_precision, "float_precision", POPValidationError
    )

    # The canonical writer emits exactly one serialized relation per line and
    # uses a fixed newline independent of the host platform.
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        for row in rows:
            handle.write(format_pop_row(row, float_precision=float_precision))
            handle.write('\n')
    return True


################################################################################
# Row conversion
################################################################################


def parse_pop_line(text, line_number=None):
    """Parse one nonempty .pop line.

    Inputs:
        text:
            String. CSV-like .pop row.
        line_number:
            Integer or None. Source line number.

    Outputs:
        row:
            ParsedRow. Parsed row.
    """
    text = str(text)
    line_number = (
        None if line_number is None
        else _cast_positive_int(
            line_number, "line_number", POPParseError
        )
    )

    # Split only the three structural delimiters; the fourth field is the
    # complete comma-separated numerical payload.
    fields = text.split(',', 3)
    if len(fields) != 4:
        raise POPParseError("Expected relation, degree, scope, and evals.")

    # Field helpers own the exact grammar for one token type, keeping this
    # row-level function focused on ordering and source location.
    relation = parse_relation_field(fields[0])
    degree = parse_degree_field(fields[1])
    scope = parse_scope_field(fields[2])
    evals = parse_evals_field(fields[3])
    return ParsedRow(relation, degree, scope, evals, line_number)


def format_pop_row(row, *, float_precision=17):
    """Format one parsed row without a trailing newline. Keeping presentation
    logic here lets computational objects expose plain data rather than
    embedding output policy in their algorithms.

    Inputs:
        row:
            ParsedRow. Row to format.
        float_precision:
            Integer. Significant decimal digits.

    Outputs:
        text:
            String. CSV-like .pop row.
    """
    float_precision = _cast_positive_int(
        float_precision, "float_precision", POPValidationError
    )
    degree = _cast_nonnegative_int(
        row.degree, "row degree", POPValidationError
    )

    # No builder names or internal IDs enter the exchange format; each row is
    # reconstructed solely from its mathematical relation and sampled values.
    return ','.join((
        row.relation.to_file_token(),
        str(degree),
        format_scope_field(row.scope),
        format_evals_field(row.evals, float_precision=float_precision),
    ))


################################################################################
# Field conversion
################################################################################


def parse_relation_field(field):
    """Parse one .pop relation token.

    Inputs:
        field:
            String. Relation token.

    Outputs:
        relation:
            Relation. Normalized relation.
    """
    field = str(field).strip()

    # Objective and variable tokens are case-insensitive words, while
    # constraints retain their symbolic spellings from the file grammar.
    token = field.lower() if field.isalpha() else field
    relations = {
        'var': Relation.VAR,
        'min': Relation.MIN,
        'max': Relation.MAX,
        '==0': Relation.EQ,
        '>=0': Relation.GE,
        '<=0': Relation.LE,
    }
    if token not in relations:
        raise POPParseError("Unsupported relation: " + repr(field))
    return relations[token]


def parse_degree_field(field):
    """Parse one .pop degree token.

    Inputs:
        field:
            String. Degree token.

    Outputs:
        degree:
            Integer. Parsed degree.
    """
    field = str(field).strip()

    try:
        return int(field)
    except ValueError as error:
        raise POPParseError("Invalid degree: " + repr(field)) from error


def parse_scope_field(field):
    """Parse one hyphen-delimited variable scope.

    Inputs:
        field:
            String. Scope token.

    Outputs:
        scope:
            tuple. Variable names.
    """
    field = str(field)

    # Variable names are literal exchange tokens. Silently stripping whitespace
    # would make the reader accept names that the canonical writer refuses to
    # emit and could merge distinct names across language boundaries.
    if field != field.strip():
        raise POPParseError("Scope cannot have leading or trailing whitespace.")
    scope = tuple(field.split('-'))
    if not scope or any(not name for name in scope):
        raise POPParseError("Scope must contain nonempty variable names.")
    if any(name != name.strip() for name in scope):
        raise POPParseError("Variable names cannot have outer whitespace.")
    if len(set(scope)) != len(scope):
        raise POPParseError("Scope contains a repeated variable name.")
    return scope


def parse_evals_field(field):
    """Parse the comma-separated numerical tail of one .pop row.

    Inputs:
        field:
            String. Comma-separated numerical evaluations.

    Outputs:
        evals:
            ndarray. One-dimensional float evaluations.
    """
    field = str(field).strip()

    if not field:
        raise POPParseError("A .pop row must contain at least one evaluation.")

    # Validate the complete payload before np.fromstring so malformed suffixes
    # cannot be silently ignored by NumPy's permissive parser. The grammar is
    # compiled once because large .pop files may contain many relation rows.
    if _EVALS_PATTERN.fullmatch(field) is None:
        raise POPParseError("Malformed numerical evaluation payload.")

    # Delimiter counting supplies an independent full-consumption check after
    # lexical validation and protects against parser-version differences.
    num_expected = field.count(',') + 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            evals = np.fromstring(field, sep=',', dtype=float)
    except ValueError as error:
        raise POPParseError(
            "Malformed numerical evaluation payload."
        ) from error
    if evals.size != num_expected:
        raise POPParseError("Malformed numerical evaluation payload.")
    if not np.all(np.isfinite(evals)):
        raise POPParseError("Evaluations must be finite.")
    return evals


def format_scope_field(scope):
    """Format a variable scope for one .pop row.

    Inputs:
        scope:
            ScopeLike. Variable names.

    Outputs:
        field:
            String. Hyphen-delimited scope.
    """
    if isinstance(scope, str):
        scope = (scope,)
    else:
        try:
            scope = tuple(scope)
        except TypeError as error:
            raise POPValidationError(
                "A formatted scope must be an iterable of strings."
            ) from error
    if not scope or any(not isinstance(name, str) or not name for name in scope):
        raise POPValidationError(
            "A formatted scope must contain nonempty string names."
        )
    if len(set(scope)) != len(scope):
        raise POPValidationError(
            "A formatted scope cannot contain repeated variable names."
        )
    for name in scope:
        if name != name.strip():
            raise POPValidationError(
                "Formatted variable names cannot have outer whitespace."
            )
        if ',' in name or '-' in name or '\n' in name or '\r' in name:
            raise POPValidationError(
                "Formatted variable names cannot contain delimiters or newlines."
            )
    return '-'.join(scope)


def format_evals_field(evals, *, float_precision=17):
    """Format sampled evaluations for one .pop row.

    Inputs:
        evals:
            ArrayLike. Sampled evaluations.
        float_precision:
            Integer. Significant decimal digits.

    Outputs:
        field:
            String. Comma-separated numerical payload.
    """
    evals = cast_real_array(evals, "Evaluations", POPValidationError)
    float_precision = _cast_positive_int(
        float_precision, "float_precision", POPValidationError
    )

    if evals.ndim != 1:
        raise ValueError("Evaluations must be one-dimensional.")
    if not np.all(np.isfinite(evals)):
        raise ValueError("Evaluations must be finite.")
    # Seventeen significant digits, the canonical default, round-trip every
    # finite IEEE-754 binary64 value used by Rhiza sample arrays.
    spec = '.' + str(float_precision) + 'g'
    return ','.join(format(float(value), spec) for value in evals)


################################################################################
# Public numerical conversion
################################################################################


def cast_real_array(values, label, error_type):
    """Return a float64 array without accepting complex-valued input.

    Public Rhiza arrays are real. NumPy's ordinary real cast can discard an
    imaginary part after emitting only a warning, so public boundaries inspect
    the original dtype and object contents before requesting ``float64``.
    Shape, nonemptiness, and finiteness remain caller-specific validations.

    Inputs:
        values:
            ArrayLike. Candidate real-valued array.
        label:
            String. Context used in validation errors.
        error_type:
            Exception class. Public error to raise.

    Outputs:
        array:
            ndarray. Converted float64 values.
    """
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise error_type(label + " must be convertible to real values.") from error

    if np.iscomplexobj(array) or _object_array_contains_complex(array):
        raise error_type(label + " must be real-valued; complex input is unsupported.")
    try:
        return np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise error_type(label + " must be convertible to real values.") from error


def cast_real_scalar(value, label, error_type):
    """Return one real float without accepting complex scalar input.

    Zero-dimensional NumPy arrays are accepted as scalars. Non-scalar arrays
    and every complex dtype are rejected before Python's ``float`` conversion
    can alter the caller's numerical value.

    Inputs:
        value:
            Object. Candidate real scalar.
        label:
            String. Context used in validation errors.
        error_type:
            Exception class. Public error to raise.

    Outputs:
        scalar:
            Float. Converted real value.
    """
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise error_type(label + " must be a real scalar.") from error
    if array.ndim != 0:
        raise error_type(label + " must be a real scalar.")
    if np.iscomplexobj(array) or _object_array_contains_complex(array):
        raise error_type(label + " must be real-valued; complex input is unsupported.")
    try:
        return float(array.item())
    except (TypeError, ValueError, OverflowError) as error:
        raise error_type(label + " must be a real scalar.") from error


def _object_array_contains_complex(array):
    """Return whether an object array contains a complex representation."""
    if array.dtype.kind != 'O':
        return False
    return any(np.iscomplexobj(value) for value in array.flat)


################################################################################
# Integral conversion
################################################################################


def cast_exact_int(value, label, error_type):
    """Cast an integer-valued scalar without silent truncation.

    Public constructors use this common boundary conversion for degrees,
    indexes, and iteration limits. Booleans, fractions, and nonfinite values
    are rejected before Python's ``int`` conversion can change their meaning.

    Inputs:
        value:
            Object. Candidate integral scalar.
        label:
            String. Context used in validation errors.
        error_type:
            Exception class. Public Rhiza error to raise.

    Outputs:
        integer:
            Integer. Exact converted value.
    """
    if isinstance(value, (bool, np.bool_)):
        raise error_type(label + " must be an integer.")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    raise error_type(label + " must be an integer.")


def _cast_nonnegative_int(value, label, error_type):
    """Cast an exact nonnegative integer used by row formatting. Integer-valued
    real scalars are accepted, while booleans, fractions, and nonfinite values are
    rejected instead of being truncated.

    Inputs:
        value:
            Object. Candidate integral scalar.
        label:
            String. Context used in validation errors.
        error_type:
            Exception class. Public Rhiza error to raise.

    Outputs:
        integer:
            Integer. Exact nonnegative value.
    """
    integer = cast_exact_int(value, label, error_type)
    if integer < 0:
        raise error_type(label + " must be nonnegative.")
    return integer


def _cast_positive_int(value, label, error_type):
    """Cast an exact positive integer used by parser and writer diagnostics. This
    builds on the nonnegative conversion and adds the strict positivity required by
    line numbers and significant-digit counts.

    Inputs:
        value:
            Object. Candidate integral scalar.
        label:
            String. Context used in validation errors.
        error_type:
            Exception class. Public Rhiza error to raise.

    Outputs:
        integer:
            Integer. Exact positive value.
    """
    integer = _cast_nonnegative_int(value, label, error_type)
    if integer == 0:
        raise error_type(label + " must be positive.")
    return integer


################################################################################
# Parsed rows
################################################################################


class ParsedRow:
    """Store one parsed .pop row. The parser produces this neutral record before
    POP validates names, sample counts, and lifecycle constraints."""

    def __init__(self, relation, degree, scope, evals, line_number=None):
        """Store fields already separated by the parser or supplied to the writer.
        POP performs semantic validation when it consumes this neutral record.

        Inputs:
            relation:
                Relation. Row relation.
            degree:
                Integer. Declared polynomial degree.
            scope:
                tuple. Variable scope.
            evals:
                ndarray. Sampled evaluations.
            line_number:
                Integer or None. Source line number.
        """
        self.relation = relation
        self.degree = degree
        self.scope = scope
        self.evals = evals
        self.line_number = line_number
