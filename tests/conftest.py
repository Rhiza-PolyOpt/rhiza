"""Shared pytest fixtures for Rhiza."""

import pytest

from tests.fixtures.reference_problem import make_reference_pop


################################################################################
# Shared fixtures
################################################################################


@pytest.fixture
def reference_pop():
    """Return a finalized sparse sampled-equality reference POP."""
    pop = make_reference_pop()
    pop.finalize()
    return pop
