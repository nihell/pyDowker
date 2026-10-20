import sys
import os

# Make the package importable without installing
sys.path.insert(0, os.path.abspath("../src"))

project = "pyDowker"
author = "Niklas Hellmer"
copyright = "2024, Niklas Hellmer"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "pyDowker"

# Napoleon (numpy docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "gudhi": ("https://gudhi.inria.fr/python/latest", None),
}

# MyST
myst_enable_extensions = ["dollarmath", "amsmath"]

# nbsphinx: execute notebooks during the doc build so outputs appear
nbsphinx_execute = "always"
nbsphinx_timeout = 120
nbsphinx_kernel_name = "python3"
# The notebook paths in the toctree are relative to conf.py
nbsphinx_prolog = ""
