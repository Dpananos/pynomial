# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# ReadTheDocs build configuration
# Check if we're building on ReadTheDocs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    # Configure for ReadTheDocs environment
    html_context = {
        'display_github': True,
        'github_user': 'demetripananos',  # Replace with your GitHub username
        'github_repo': 'pynomial',        # Replace with your repo name
        'github_version': 'main',
        'conf_py_path': '/',
    }

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pynomial'
copyright = '2025, Demetri Pananos'
author = 'Demetri Pananos'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__, ConfidenceIntervalType, ConfidenceInterval'
}

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autosummary settings
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# ReadTheDocs theme options
html_theme_options = {
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
