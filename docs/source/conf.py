# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'macro-eeg-model'
copyright = '2025, Mariia Steeghs-Turchina'
author = 'Mariia Steeghs-Turchina'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

# -- Path setup --------------------------------------------------------------

# If extensions (or api to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

autoapi_type = 'python'
autoapi_dirs = ['../../src']

# autoapi_python_class_content = 'both'
autoapi_python_class_content = 'class'

extensions = [
    'sphinx.ext.autodoc',    # Automatically document your code from docstrings
    'sphinx.ext.napoleon',   # Support for NumPy and Google style docstrings
    'autoapi.extension',     # Automatically generate API docs
    'sphinx.ext.viewcode',   # Add links to highlighted source code
    'sphinx_copybutton',     # Add copy button to code blocks
    'myst_parser',           # Markdown parser
    'sphinx.ext.coverage'   # Add coverage report
]

coverage_show_missing_items = True

def skip_attributes(app, what, name, obj, skip, options):
    # print(app, name, obj, skip, options)
    print(name)
    print(what)
    print()

    if name.endswith(".__init__"):
        skip = False

    if what == "attribute":
        skip = True

    return skip

def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_attributes)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
templates_path = ['_templates']

html_theme_options = {
    "repository_url": "https://github.com/thatmariia/macro-eeg-model",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
    "path_to_docs": "docs",
}