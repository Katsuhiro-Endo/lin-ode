# lin-ode

Linearize a system of nonlinear ODEs and read any matrix element of the result in closed form,
without the grid size entering the cost.

$$\frac{d}{dt}\mathbf{x}(t)=\mathbf{F}\bigl(\mathbf{x}(t)\bigr),
\qquad \mathbf{x}(t)\in\mathbb{R}^{N},$$

with $\mathbf{F}$ polynomial in the state. A PDE becomes such a system once space is discretised, so
$N$ is the number of grid points.

## Install

Clone the repository, then install it from that checkout. To work in it:

```bash
uv sync --locked
```

To use it from another project, give the path to the checkout:

```bash
uv add /path/to/lin-ode          # or: pip install /path/to/lin-ode
```

## Where to start

Read [**Quick start**](docs/index.md) first. It runs a four-site example end to end.
[Vocabulary and symbols](docs/concepts.md) defines every term and symbol.

## Examples

Worked notebooks: [`notebooks/`](notebooks/README.md).

## Documentation site

[Sphinx](https://www.sphinx-doc.org/) with the [Furo](https://pradyunsg.me/furo/) theme and
[MyST-NB](https://myst-nb.readthedocs.io/), which renders the notebooks directly:

```bash
uv run --group docs sphinx-build -c docs -b html . docs/_build/html
open docs/_build/html/docs/index.html
```

Add `-t internal` for the internal build, which additionally renders `docs/internal/spec.md` and
`notebooks/internal/`:

```bash
uv run --group docs sphinx-build -t internal -c docs -b html . docs/_build/internal
```

Without the tag Sphinx excludes those from the source tree, so a public build cannot render them.

Sphinx builds from the repository root, so notebook links resolve identically on disk and on the
site.

Notebooks render from their **stored outputs** (`nb_execution_mode = "off"`), so the build takes
seconds and never re-runs the ten-minute examples.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development commands: linting, type checking and tests.
