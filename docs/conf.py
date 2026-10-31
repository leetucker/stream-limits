"""Sphinx configuration for the stream-limits documentation.

Build locally with

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -b html docs docs/_build/html
"""

import stream_limits

project = "stream-limits"
author = "Lee Tucker"
copyright = "2026, Lee Tucker"
release = stream_limits.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# fenced blocks without a language are output and tables, not code
highlight_language = "text"

myst_enable_extensions = ["colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 4

autodoc_typehints = "none"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
default_role = "py:obj"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
    "hdfe_stream": ("https://leetucker.github.io/hdfe-stream", None),
}
# the inventories are fetched at build time; a build without network access
# still succeeds, with those links left as plain text
suppress_warnings = ["intersphinx.external", "config.cache"]

html_theme = "furo"
html_title = f"stream-limits {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/leetucker/stream-limits/",
    "source_branch": "main",
    "source_directory": "docs/",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
