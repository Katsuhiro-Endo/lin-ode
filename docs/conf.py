"""Sphinx configuration for the lin_ode documentation site.

The source directory is the repository root, so every relative link in the notebooks
resolves the same way on the site as it does on disk.  Build with::

    uv run --group docs sphinx-build -c docs -b html . docs/_build/html
    uv run --group docs sphinx-build -t internal -c docs -b html . docs/_build/internal
"""

import pathlib

project = "lin-ode"
author = ""
release = "0.1.0"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # the docstrings are Google-style
    "sphinx.ext.viewcode",
]

root_doc = "docs/index"

# ``-t internal`` adds the pages below.  Without the tag they are excluded outright, so a
# public build cannot render them even if something links to one.
# ``tags`` is injected by Sphinx into this namespace, so neither linter can see it bound.
internal = tags.has("internal")  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
internal_only = [
    "docs/internal",
    "docs/reports",
    "notebooks/internal",
    "docs/_generated",
]

# ``docs/_generated/examples`` holds the @example pages a test run writes, one per spec
# anchor and one per suite.  ``make internal`` regenerates them, and only when the library
# or the tests changed:
#
#     uv run pytest --examples-pages docs/_generated/examples
#
# They are ordinary source pages, so unlike a copied-in file they reach the sidebar, the
# search index and the theme.
_examples = pathlib.Path(__file__).parent / "_generated" / "examples"

# The ``only`` block in ``docs/index.md`` still *reads* its toctree in a public build, so
# the excluded entries warn once each.  Suppressed for that build only -- the internal
# build reads every page and so still reports a genuinely broken toctree entry.  A build
# with no test run behind it has no examples pages either, and is suppressed the same way.
suppress_warnings = [] if internal and _examples.is_dir() else ["toc.not_readable"]

# Everything that is not documentation.  ``.claude`` and the ``CLAUDE*.md`` files hold
# internal working notes and must never be published.
exclude_patterns = [
    ".claude",
    "CLAUDE.md",
    "CLAUDE.local.md",
    "README.md",
    ".git",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "docs/_build",
    "graphify-out",
    "lin_ode",
    "scripts",
    "tests",
    # Include-only: a configuration is read on its generated page, which pulls the tables
    # in with ``{include}`` and appends the tests that check them. Building these as
    # documents too would put every heading on the site twice. Only ``configs/`` is
    # include-only; every other page under a report is an ordinary document.
    "docs/reports/*/configs/*.md",
] + ([] if internal else internal_only)

# Notebooks are committed with their outputs, and the examples take up to ten minutes, so
# the site renders what is stored rather than re-running anything.
nb_execution_mode = "off"
nb_merge_streams = True

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
# 4, not 3: a report's toggles land as h4, and @example(report=...) links at that depth.
myst_heading_anchors = 4

html_theme = "furo"
html_title = "lin-ode (internal)" if internal else "lin-ode"
html_short_title = "lin-ode"
html_theme_options = {
    "source_repository": "https://github.com/Katsuhiro-Endo/lin-ode",
    "source_branch": "main",
    "source_directory": "",
}
html_static_path: list[str] = ["_static"]
html_css_files = ["custom.css"]

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# API reference. The docstrings use Google sections; every module defines __all__, which
# autodoc uses to decide what is public.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False


# The report index, and the generated pages that ``{include}`` a configuration from it.
_REPORT_PAGES = ("docs/reports/", "docs/_generated/examples/config-")


def _mark_report_pages(app, pagename, templatename, context, doctree):
    """Wrap a report page in a class ``custom.css`` can target.

    Its derivations are far denser than the tutorials', so they get looser leading and
    smaller maths -- and nothing else on the site should change. Furo puts no class on
    ``<body>``, so the wrapper goes around the rendered body instead of on the tag.
    """
    if pagename.startswith(_REPORT_PAGES) and (body := context.get("body")):
        context["body"] = f'<div class="report-page">{body}</div>'


def setup(app):
    app.connect("html-page-context", _mark_report_pages)
