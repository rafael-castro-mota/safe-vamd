# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import warnings
from sphinx.ext.napoleon.docstring import NumpyDocstring

warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

sys.path.insert(0, os.path.abspath('../../src'))
sys.path.append(os.path.abspath('sphinxext'))


project = 'SAFE-VAMD'
copyright = ('2026, Rafael Castro-Mota, Paul Williams, Stefan Jacob. '
             'Infrasound research group at the German National Metrology Institute')
author = ('Rafael Castro-Mota, Paul Williams, Stefan Jacob. '
          'Infrasound research group at the German National Metrology Institute')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#extensions = ['matplotlib.sphinxext.plot_directive', 'sphinx.ext.napoleon',
 #             'numpydoc', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary']

extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

# Napoleon configuration (Handles NumPy docstrings natively)
napoleon_numpy_docstring = True
napoleon_use_ivar = False       # Crucial: Forces Attributes to style like Parameters
napoleon_use_param = False
#napoleon_use_rtype = True
#napoleon_preprocess_types = True
#napoleon_type_aliases = None

# 4. Autodoc settings
autoclass_content = 'both'
autodoc_typehints = "none"
autosummary_generate = True
autodoc_member_order = 'bysource'

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": False,
}

# --- MONKEY-PATCH NAPOLEON FOR ATTRIBUTES ---
"""
def _parse_attributes_section(self, section):
    fields = self._consume_fields()

    # Manually run Napoleon's type processor over each attribute's type string
    processed_fields = []
    for name, type_str, desc in fields:
        if type_str:
            # Converts type strings (e.g. 'Source') into active Sphinx roles
            type_str = ' '.join(self._re_lex_type(type_str))
        processed_fields.append((name, type_str, desc))

    return self._format_fields('Attributes', processed_fields)

NumpyDocstring._parse_attributes_section = _parse_attributes_section
"""
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_static_path = ['_static']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinxdoc'

rawfiles = ['image']
