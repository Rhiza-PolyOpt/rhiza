"""Polynomial optimization via sparse sample-based quotient-ring sum-of-squares programming."""

from rhiza.pop import POP
from rhiza.sdpr import SDPR
from rhiza.solver import Solver
from rhiza import rounding
from rhiza.diagnostics import (
    BackendBuildError,
    BackendError,
    BackendUnavailableError,
    CompressionError,
    CSPError,
    DegreeError,
    ExtractionError,
    GraphPolicyError,
    NumericalRankError,
    PipelineError,
    POPError,
    POPParseError,
    POPValidationError,
    RIPError,
    RhizaError,
    SDPRError,
    SolveError,
)


################################################################################
# Package metadata
################################################################################

__version__ = '0.9.0'


################################################################################
# Public exports
################################################################################

__all__ = [
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
]
