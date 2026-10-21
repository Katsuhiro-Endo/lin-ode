# Docs build.
#
#   make            public site, no internal pages
#   make internal   internal site, examples report refreshed first
#   make docs       both
#
# The examples report is written by a pytest run, so it is a build artifact with real
# prerequisites. Declaring them is what keeps `make internal` from running the suite on
# every build: the report is remade only when the library or the tests have changed.

SPHINX := uv run --group docs sphinx-build -c docs -b html .
PAGES := docs/_generated/examples
STAMP := $(PAGES)/conformance.md
# The report markdown too: a generated page embeds line ranges read off it, so an edited
# report leaves a stale page including nothing.
SOURCES := $(shell find lin_ode tests -name '*.py') $(shell find docs/reports -name '*.md')

.PHONY: public internal docs clean

public:
	$(SPHINX) docs/_build/html

internal: $(STAMP)
	$(SPHINX) -t internal docs/_build/internal

docs: public internal

# ponytail: `-` so a failing test still leaves its pages. Outcome is a row on each entry
# and pytest_sessionfinish writes them either way, so the failure shows up on the page.
# Read the pages, not make's exit code. Drop the `-` if docs should fail with the suite.
# ``tests`` is passed explicitly: pytest reads --examples-pages from tests/conftest.py,
# and only treats that as an initial conftest if a path under it is an argument. Without
# it, a rebuild where $(PAGES) already exists has pytest take it for a collection path
# instead, and the option comes back unrecognized.
$(STAMP): $(SOURCES)
	-uv run pytest -q tests --examples-pages $(PAGES)

clean:
	rm -rf docs/_build $(PAGES)
