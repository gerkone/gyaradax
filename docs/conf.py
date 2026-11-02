import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(".."))

project = "gyaradax"
author = "Gianluca Galletti, Eric Volkmann"
copyright = "2026, Gianluca Galletti, Eric Volkmann"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

default_role = "literal"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "PROMPT.md", "cuda_experiments"]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": False}
autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = "gyaradax"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/gerkone/gyaradax",
    "source_branch": "master",
    "source_directory": "docs/",
    "light_css_variables": {"color-brand-primary": "#16857E", "color-brand-content": "#16857E"},
    "dark_css_variables": {"color-brand-primary": "#24B6AD", "color-brand-content": "#24B6AD"},
}
