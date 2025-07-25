# TEAM MEMBER 4: Documentation & Tutorial Owner
# Create comprehensive documentation structure

# First, create the main documentation index
docs_index_content = '''# PyRISE: JAX-First Regime-Switching DSGE Modeling

PyRISE is a high-performance Python package for solving, simulating, and estimating regime-switching Dynamic Stochastic General Equilibrium (DSGE) models. Built on JAX for automatic differentiation and GPU acceleration, PyRISE provides a modern, open-source alternative to the MATLAB RISE toolbox.

## Key Features

- **Regime-Switching Models**: Full support for Markov-switching and endogenous threshold-switching DSGE models
- **JAX-Powered Performance**: Automatic differentiation, JIT compilation, and GPU acceleration
- **Comprehensive Solution Methods**: Higher-order perturbation, projection methods, and occasionally-binding constraints
- **Bayesian Estimation**: NUTS, Sequential Monte Carlo, and Variational Inference with BlackJAX
- **Economic Validation**: Built-in parameter validation against economic theory constraints
- **Reproducible Research**: Deterministic random number generation and result checksumming

## Quick Start

```python
import pyrise as pr

# Load model from YAML specification
model = pr.load_model("nk_regime_switching.yaml")

# Solve using 2nd-order partition perturbation
solution = pr.solve(model, order=2, scheme="partition")

# Simulate impulse responses
irfs = solution.compute_irf("eps_r", shock_size=1.0, horizon=40)

# Estimate parameters using Bayesian methods
posterior = pr.estimate(model, data=observed_data, method="NUTS")
```

## Installation

### Stable Release

```bash
pip install pyrise
```

### Development Version

```bash
git clone https://github.com/pyrise-project/pyrise.git
cd pyrise
pip install -e ".[dev]"
```

### GPU Support

For GPU acceleration:

```bash
pip install pyrise[gpu]
```

## Documentation Structure

```{toctree}
:maxdepth: 2
:caption: Getting Started

tutorials/installation
tutorials/quickstart
tutorials/first_model
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide/model_specification
guide/solution_methods
guide/estimation
guide/simulation
guide/validation
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/simple_rbc
examples/new_keynesian
examples/regime_switching
examples/japan_application
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/core
api/solvers
api/estimation
api/simulation
api/utils
```

```{toctree}
:maxdepth: 2
:caption: Development

development/contributing
development/testing
development/reproducibility
development/benchmarks
```

## Important Disclaimers

**Verification Status**: This package contains numerous claims marked as [Unverified] throughout the codebase and documentation. These represent:

- Theoretical claims not independently verified against literature
- Performance benchmarks not validated across all hardware configurations  
- Algorithmic implementations not cross-checked against reference implementations
- Economic modeling assumptions based on standard practices but not empirically validated

Users should independently verify results for production use in policy analysis or academic research.

**No MATLAB Access**: This documentation and implementation do not rely on MATLAB RISE outputs for validation. All examples use open-source data or clearly labeled simulated datasets.

## System Requirements

**Minimum Requirements:**
- Python 3.10+
- 8GB RAM  
- 2GB free disk space

**Recommended:**
- Python 3.11+
- 16GB+ RAM
- GPU with CUDA 12+ support
- 10GB+ free disk space

**Dependencies:**
- JAX ≥ 0.4.0 (core computation)
- NumPy ≥ 1.24.0 (numerical arrays)
- SciPy ≥ 1.11.0 (optimization)
- SymPy ≥ 1.12.0 (symbolic math)
- BlackJAX ≥ 1.0.0 (Bayesian inference)

## Citation

If you use PyRISE in your research, please cite:

```bibtex
@software{pyrise2024,
  title={{PyRISE}: A JAX-First Python Successor to the RISE Toolbox},
  author={{PyRISE Development Team}},
  year={2024},
  url={https://github.com/pyrise-project/pyrise},
  note={[Unverified] Software package for regime-switching DSGE modeling}
}
```

## License

PyRISE is released under the MIT License. See [LICENSE](https://github.com/pyrise-project/pyrise/blob/main/LICENSE) for details.

## Acknowledgments

PyRISE draws inspiration from the MATLAB RISE toolbox by Junior Maih and builds upon the extensive JAX ecosystem. We acknowledge the contributions of the scientific Python community and the regime-switching DSGE modeling literature.

[Unverified] Performance comparisons and feature parity claims with MATLAB RISE are based on internal benchmarks and may not reflect all use cases.
'''

# Write the main documentation index
with open("docs/index.md", "w") as f:
    f.write(docs_index_content)

# Create Sphinx configuration
sphinx_conf_content = '''"""
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
        "inlineMath": [["$", "$"], ["\\\\(", "\\\\)"]],
        "displayMath": [["$$", "$$"], ["\\\\[", "\\\\]"]],
        "macros": {
            "E": ["\\\\mathbb{E}"],
            "Var": ["\\\\text{Var}"],
            "Cov": ["\\\\text{Cov}"],
            "p": ["\\\\mathbb{P}"],
            "R": ["\\\\mathbb{R}"],
            "N": ["\\\\mathcal{N}"],
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
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
""",
}

latex_documents = [
    (master_doc, "pyrise.tex", "PyRISE Documentation", "PyRISE Development Team", "manual"),
]
'''

# Create Sphinx conf.py
with open("docs/conf.py", "w") as f:
    f.write(sphinx_conf_content)

print("✅ Created comprehensive documentation structure")
print("   - Sphinx configuration with Book theme")
print("   - MyST Markdown support for documentation")
print("   - Jupyter notebook integration with nbsphinx")
print("   - Automatic API documentation generation")
print("   - Custom disclaimer processor for [Unverified] content")
print("   - LaTeX math rendering with MathJax")