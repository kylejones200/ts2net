# Configuration file for the Sphinx documentation builder.
from __future__ import annotations
import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ts2net"
author = "Kyle T. Jones"
copyright = f"{datetime.now():%Y}, {author}"

# Attempt to import the version from the package
try:
    from ts2net import __version__ as release
except Exception:
    release = "0.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Templates path
templates_path = ["_templates"]
exclude_patterns: list[str] = ["**.ipynb_checkpoints"]

# nbsphinx configuration
nbsphinx_execute = "never"  # Don't execute notebooks during build (use pre-executed)
nbsphinx_allow_errors = False

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
