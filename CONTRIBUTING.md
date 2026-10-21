# Contributing

All commands below use [uv](https://docs.astral.sh/uv/). Install dependencies with:

```bash
uv sync --locked --all-groups
```

## Linting and formatting

Lint with [Ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff check .
```

Format with:

```bash
uv run ruff format .
```

## Docstring style

[docformatter](https://github.com/PyCQA/docformatter) enforces docstring style across `lin_ode/`
and `tests/`. Check without changing any file:

```bash
uv run docformatter --check --diff -r lin_ode/ tests/
```

Rewrite them in place with:

```bash
uv run docformatter -r -i lin_ode/ tests/
```

## Type checking

Type-check with [Pyright](https://github.com/microsoft/pyright):

```bash
uv run pyright
```

## Tests

Run the tests:

```bash
uv run pytest
```

## Re-executing the notebooks

The site renders stored outputs, so refresh them before building if a change affects results:

```bash
uv run --group docs jupyter nbconvert --execute --inplace $(find notebooks -name "*.ipynb")
```

## Figures

Two scripts draw the figures the notebooks embed. Both write into `notebooks/figures/`, which the
notebooks reference by relative path, and the PNGs are committed.

### Algorithm figures

The KPP–Fisher and pivot-switching examples carry eight figures of the assembly algorithm.

| figure | shows |
|---|---|
| `reach-on-lattice` | what `find_reachable_bras` produces, on the grid |
| `shift-symmetry` | what `canonicalise_positions` does to the site labels |
| `build-trace` | `build` traced over its own loop |
| `zone-map` | where each coupling reads its coefficients |
| `zone-classes` | what the zone tag changes |
| `psc-fold` | the weights the fold leaves on each retained basis state |
| `psc-elements` | which elements the fold turns on, and which repeat across sites |
| `psc-column` | where the fold applies, and how one call covers a whole shape |

```bash
uv run python docs/make_figures.py
```

The script executes each notebook's own helpers rather than a copy, so a rename fails loudly instead
of drifting. It never edits a notebook, so captions stay put.

Run it after changing `iter_neighbours`, `find_reachable_bras`, `classify_by_shift`,
`classify_by_zone` or `build`.

### The overview figure

`01_overview` carries one more figure, the map of the five stages. A separate script draws it:

```bash
uv run python notebooks/tutorials/_figures/five_stages.py
```

Run it after adding a class to the pipeline or renaming one. The boxes and their labels are literals
in that script, so a rename does not reach them.

Keep every label short. The figure size cancels out of both the apparent text size and the fit of a
label in its box, so what decides readability is the ratio of box width to label length. Enlarging
the font does not help. The script's docstring gives the arithmetic.
