# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "NequIP"
copyright = "2025 The NequIP Developers"
author = "The NequIP Developers"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]
myst_enable_extensions = [
    "html_admonition",
    "dollarmath",  # "amsmath", # to parse Latex-style math
]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autosummary_generate = True
source_suffix = [".rst", ".md"]

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "e3nn": ("https://docs.e3nn.org/en/stable/", None),
    "torchmetrics": ("https://lightning.ai/docs/torchmetrics/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "favicon.png"
html_logo = "logo.png"
html_theme_options = {
    "sidebar_hide_name": True,
}


def process_docstring(app, what, name, obj, options, lines):
    """For pretty printing sets and dictionaries of data fields."""
    if isinstance(obj, set) or isinstance(obj, dict):
        lines.clear()  # Clear existing lines to prevent repetition


def setup(app):
    app.connect("autodoc-process-docstring", process_docstring)
