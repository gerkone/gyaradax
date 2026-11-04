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
    "sphinxarg.ext",
    "sphinx_llms_txt",
]

default_role = "literal"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "PROMPT.md", "cuda_experiments"]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
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

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
llms_txt_summary = (
    "Local flux-tube gyrokinetics in JAX, following GKW. The API pages are generated from "
    "the docstrings in the gyaradax package at https://github.com/gerkone/gyaradax."
)

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
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/gerkone/gyaradax",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
            '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
            "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 "
            "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15"
            "-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82"
            ".44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 "
            '2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}
