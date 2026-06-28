"""Sphinx configuration for the STARK-ODE documentation."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "stark-ode"
author = "Jonathan M. Fellows"
copyright = "2026, Jonathan M. Fellows"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

root_doc = "index"
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "pydata_sphinx_theme"
html_title = "STARK-ODE"
html_static_path: list[str] = []

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

myst_heading_anchors = 3
