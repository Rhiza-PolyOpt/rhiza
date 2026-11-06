"""Tests for distribution metadata and install-time dependencies."""

from pathlib import Path

import rhiza


def _array_section(text, header):
    """Return the textual contents of one simple TOML array assignment."""
    start = text.index(header) + len(header)
    stop = text.index(']', start)
    return text[start:stop]


def test_mosek_is_an_optional_install_extra():
    """Require SCS in the core install while keeping MOSEK optional."""
    project_root = Path(__file__).resolve().parents[2]
    text = (project_root/'pyproject.toml').read_text(encoding='utf-8')
    dependencies = _array_section(text, 'dependencies = [')
    mosek_extra = _array_section(text, 'mosek = [')

    assert 'scs>=' in dependencies
    assert 'mosek>=' not in dependencies
    assert 'mosek>=' in mosek_extra


def test_distribution_name_and_version_match_the_imported_package():
    """Keep wheel metadata synchronized with the imported package version."""
    project_root = Path(__file__).resolve().parents[2]
    text = (project_root/'pyproject.toml').read_text(encoding='utf-8')

    assert 'name = "rhiza-polyopt"' in text
    assert f'version = "{rhiza.__version__}"' in text
    assert rhiza.__version__ == '0.9.0'
