# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'macro-eeg-model'
copyright = '2024, Mariia Steeghs-Turchina'
author = 'Mariia Steeghs-Turchina'
release = '1.0'

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

html_theme = "furo"
html_static_path = ['_static']
templates_path = ['_templates']

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/thatmariia/macro-eeg-model",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/thatmariia/macro-eeg-model",
    "source_branch": "main"
}
