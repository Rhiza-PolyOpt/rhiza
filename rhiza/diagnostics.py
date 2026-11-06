"""Shared Rhiza tokens, exceptions, and report formatting."""

import enum

import numpy as np


################################################################################
# Controlled tokens
################################################################################


class Relation(enum.Enum):
    """Identify one variable, objective-entry, or constraint-entry relation.

    The enum separates public and file-format spellings from the internal
    relation semantics used throughout POP construction.
    """

    VAR = 'VAR'
    MIN = 'MIN'
    MAX = 'MAX'
    EQ = 'EQ'
    GE = 'GE'
    LE = 'LE'

    @classmethod
    def from_value(cls, value):
        """Normalize a public or internal relation token.

        Inputs:
            value:
                String or Relation. Relation token to normalize.

        Outputs:
            relation:
                Relation. Normalized relation.
        """
        if isinstance(value, cls):
            return value
        # Public builders accept symbolic constraint forms, while internal
        # records use compact enum names. One table makes both spellings explicit.
        value = str(value)
        normalized = value.upper()
        aliases = {
            'VAR': cls.VAR,
            'MIN': cls.MIN,
            'MAX': cls.MAX,
            '==0': cls.EQ,
            '>=0': cls.GE,
            '<=0': cls.LE,
            'EQ': cls.EQ,
            'GE': cls.GE,
            'LE': cls.LE,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            raise POPValidationError(
                "Unsupported relation token: " + repr(value)
            ) from error

    def to_file_token(self):
        """Return the canonical relation token written to a .pop file.

        Outputs:
            token:
                String. Canonical file-format relation token.
        """
        # Serialization uses the canonical lowercase objective tokens and
        # symbolic constraint relations required by the .pop grammar.
        tokens = {
            Relation.VAR: 'var',
            Relation.MIN: 'min',
            Relation.MAX: 'max',
            Relation.EQ: '==0',
            Relation.GE: '>=0',
            Relation.LE: '<=0',
        }
        return tokens[self]


class GraphPolicy(enum.Enum):
    """Identify a correlative sparsity construction policy. The selected policy
    determines whether CSP requires RIP or leaves separator communication to
    SDPR overlap construction."""

    RIP = 'RIP'
    EXTEND = 'EXTEND'
    OVERLAP = 'OVERLAP'

    @classmethod
    def from_value(cls, value):
        """Normalize a graph policy token.

        Inputs:
            value:
                String or GraphPolicy. Policy token to normalize.

        Outputs:
            graph_policy:
                GraphPolicy. Normalized policy.
        """
        if isinstance(value, cls):
            return value
        value = str(value).upper()
        try:
            return cls[value]
        except KeyError as error:
            raise GraphPolicyError(
                "Unsupported graph policy: " + repr(value)
            ) from error


class CompressionMethod(enum.Enum):
    """Identify a clique-system compression method. Using a controlled token keeps
    public configuration and stored reports unambiguous as compression methods
    evolve."""

    PQR = 'PQR'

    @classmethod
    def from_value(cls, value):
        """Normalize a compression method token.

        Inputs:
            value:
                String or CompressionMethod. Method token to normalize.

        Outputs:
            method:
                CompressionMethod. Normalized method.
        """
        if isinstance(value, cls):
            return value
        value = str(value).upper()
        try:
            return cls[value]
        except KeyError as error:
            raise CompressionError(
                "Unsupported compression method: " + repr(value)
            ) from error


class SolutionStatus(enum.Enum):
    """Identify a normalized backend solution status. Backends map their native
    status codes here so Solver can apply one success and extraction policy."""

    OPTIMAL = 'OPTIMAL'
    OPTIMAL_INACCURATE = 'OPTIMAL_INACCURATE'
    UNKNOWN = 'UNKNOWN'
    INFEASIBLE = 'INFEASIBLE'
    UNBOUNDED = 'UNBOUNDED'
    SOLVER_ERROR = 'SOLVER_ERROR'


################################################################################
# Exceptions
################################################################################


class RhizaError(Exception):
    """Base exception for Rhiza errors. It provides one catch point for every
    user-facing Rhiza failure."""


class POPError(RhizaError):
    """Base exception for POP errors. It lets callers distinguish problem-
    description failures from later relaxation or solve failures."""


class POPParseError(POPError):
    """Report invalid data loaded from a .pop file.

    The message includes source context for lexical failures and row-specific
    semantic violations discovered while loading or finalizing the file.
    """


class POPValidationError(POPError):
    """Report invalid POP content or arguments. It identifies semantically invalid
    builder input after ordinary Python casts have completed."""


class PipelineError(RhizaError):
    """Report an invalid operation for the current object lifecycle state."""


class CSPError(RhizaError):
    """Base exception for CSP errors. It groups failures that occur while
    translating declared scopes into clique structure."""


class GraphPolicyError(CSPError):
    """Report an unsupported graph policy. It keeps unsupported policy tokens
    distinct from a valid policy whose construction later fails."""


class RIPError(CSPError):
    """Report a running-intersection failure. It reports that some variable
    occurrence set is disconnected in the proposed clique forest."""


class SDPRError(RhizaError):
    """Base exception for SDPR errors. It groups failures in basis construction,
    degree resolution, compression, and coefficient assembly."""


class DegreeError(SDPRError):
    """Report an invalid relaxation degree. It identifies incompatible SOS,
    multiplier, or separator degree choices before assembly proceeds."""


class NumericalRankError(SDPRError):
    """Report a failed numerical-rank construction. It signals that sampled linear
    algebra could not produce a trustworthy finite-dimensional basis."""


class CompressionError(SDPRError):
    """Report a failed clique-system compression. It reports that selected sample
    rows failed to reproduce the original clique-system span."""


class BackendError(RhizaError):
    """Base exception for backend errors. It groups failures at the boundary
    between a finalized SDPR and a native numerical solver."""


class BackendUnavailableError(BackendError):
    """Report an unavailable backend dependency or license. It covers missing
    packages and runtime requirements such as a usable MOSEK license."""


class BackendBuildError(BackendError):
    """Report failure while constructing a backend task. It is raised before
    optimization when the emitted native model is incomplete or inconsistent."""


class SolveError(RhizaError):
    """Report failure while solving an SDPR. It distinguishes unsuccessful
    optimization from errors in POP or SDPR construction."""


class ExtractionError(RhizaError):
    """Report failure while extracting moments. It protects normalization and
    moment queries from unusable masses, scopes, or solution states."""


################################################################################
# Report conversion and formatting
################################################################################


def to_python_report(value):
    """Convert NumPy scalar values in a public report to Python scalars.

    Rhiza reports use ordinary mappings and compact containers. Converting only
    scalar leaves keeps them JSON-serializable without expanding numerical arrays
    into large Python lists.

    Inputs:
        value:
            Object. Report value or nested report container.

    Outputs:
        converted:
            Object. Equivalent report with built-in scalar leaves.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            to_python_report(key): to_python_report(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [to_python_report(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_python_report(item) for item in value)
    return value


def format_summary(title, status, stats):
    """Format one object's status and statistics as a readable summary.

    Inputs:
        title:
            String. Summary heading.
        status:
            Mapping. Qualitative state fields.
        stats:
            Mapping. Numerical statistics.

    Outputs:
        summary:
            String. Readable multiline summary.
    """
    title = str(title)
    status = dict(status)
    stats = dict(stats)

    # Present lifecycle fields first, then numerical diagnostics, matching the
    # order in which a reader usually diagnoses a Rhiza object.
    lines = [title, '-'*len(title)]
    for key, value in status.items():
        lines.append(_format_entry(key, value))
    if status and stats:
        lines.append("")
    for key, value in stats.items():
        lines.append(_format_entry(key, value))
    return "\n".join(lines)


def format_report(report, *, indent=0):
    """Format a nested operation report.

    Inputs:
        report:
            Mapping. Report to format.
        indent:
            Integer. Initial indentation width.

    Outputs:
        text:
            String. Readable multiline report.
    """
    report = dict(report)
    indent = int(indent)

    return _format_mapping(report, indent)


def print_report(report, *, indent=0):
    """Print a nested operation report.

    Inputs:
        report:
            Mapping. Report to print.
        indent:
            Integer. Initial indentation width.

    Outputs:
        printed:
            Boolean. True after printing.
    """
    report = dict(report)
    indent = int(indent)

    print(format_report(report, indent=indent))
    return True


def _format_mapping(mapping, indent):
    """Format one level of a nested report mapping.

    Inputs:
        mapping:
            Mapping. Report level to format.
        indent:
            Integer. Indentation width.

    Outputs:
        text:
            String. Formatted mapping level.
    """
    lines = []
    prefix = " "*indent
    for key, value in mapping.items():
        label = str(key).replace('_', ' ')
        if isinstance(value, dict):
            lines.append(prefix + label + ':')
            lines.append(_format_mapping(value, indent + 4))
        elif (
            isinstance(value, (list, tuple))
            and value
            and isinstance(value[0], dict)
        ):
            lines.append(prefix + label + ':')
            for index, item in enumerate(value):
                lines.append(prefix + ' '*4 + '[' + str(index) + ']')
                lines.append(_format_mapping(item, indent + 8))
        else:
            lines.append(prefix + label + ': ' + _format_value(value))
    return "\n".join(lines)


def _format_entry(key, value):
    """Format one summary key-value entry.

    Inputs:
        key:
            StringLike. Entry key.
        value:
            Object. Entry value.

    Outputs:
        text:
            String. Formatted entry.
    """
    label = str(key).replace('_', ' ')
    return f"{label}: {_format_value(value)}"


def _format_value(value):
    """Format one scalar or compact container value.

    Inputs:
        value:
            Object. Value to format.

    Outputs:
        text:
            String. Compact formatted value.
    """
    if isinstance(value, float):
        return format(value, '.8g')
    return str(value)
