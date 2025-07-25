"""
Configuration file for the Sphinx documentation builder.

This file contains configuration for building PyRISE documentation
with comprehensive API documentation, tutorials, and examples.
"""

import os
import sys
from pathlib import Path

# Add source code to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------

project = "PyRISE"
copyright = "2024, PyRISE Development Team"
author = "PyRISE Development Team"

# The version info for the project you're documenting
try:
    import pyrise
    version = pyrise.__version__
    release = version
except ImportError:
    version = "unknown"
    release = "unknown"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "nbsphinx",  # For Jupyter notebook tutorials
]

# Add any paths that contain templates here
templates_path = ["_templates"]

# List of patterns to exclude when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The suffix(es) of source filenames
source_suffix = {
    ".rst": None,
    ".md": None,
    ".ipynb": None,
}

# The master toctree document
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = "sphinx_book_theme"

# Theme options for sphinx-book-theme
html_theme_options = {
    "repository_url": "https://github.com/pyrise-project/pyrise",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "path_to_docs": "docs/",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "notebook_interface": "jupyterlab",
    },
    "navigation_with_keys": False,
}

# Custom CSS files
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# The name of the Pygments (syntax highlighting) style to use
pygments_style = "sphinx"

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__"
}

# Generate autosummary automatically
autosummary_generate = True

# Type hints configuration
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# MathJax configuration for LaTeX rendering
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "macros": {
            "E": ["\\mathbb{E}"],
            "Var": ["\\text{Var}"],
            "Cov": ["\\text{Cov}"],
            "p": ["\\mathbb{P}"],
            "R": ["\\mathbb{R}"],
            "N": ["\\mathcal{N}"],
        }
    }
}

# nbsphinx configuration for Jupyter notebooks
nbsphinx_allow_errors = True
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_kernel_name = "python3"

# -- Custom configuration ---------------------------------------------------

def setup(app):
    """Custom setup function for Sphinx."""

    def add_unverified_disclaimer(app, what, name, obj, options, lines):
        """Add disclaimer for unverified content."""
        if lines and any("[Unverified]" in line for line in lines):
            disclaimer = [
                "",
                ".. note::",
                "   This documentation contains [Unverified] claims that have not been",
                "   independently validated. Users should verify results independently",
                "   for production applications.",
                ""
            ]
            lines[:0] = disclaimer

    # Add the disclaimer processor
    app.connect("autodoc-process-docstring", add_unverified_disclaimer)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

# Suppress specific warnings
suppress_warnings = [
    "autodoc.import_object",
    "toc.circular",
    "toc.secnum",
]

# Additional options
html_show_sourcelink = True
html_copy_source = True
html_show_sphinx = True
html_last_updated_fmt = "%b %d, %Y"

# LaTeX output options (for PDF generation)
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": r"""
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
""",
}

latex_documents = [
    (master_doc, "pyrise.tex", "PyRISE Documentation", "PyRISE Development Team", "manual"),
]
