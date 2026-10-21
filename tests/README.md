# Writing usage examples for the documentation

Add `@example(...)` (from `tests.testing`) above a test to include it in the pages
`pytest --examples-pages=DIR` writes. Undecorated tests are unaffected.

```python
from tests.testing import example

@example(spec_ref="periodic-boundary-1d")  # or @example() with no spec_ref
def test_something():
    """Title shown in the report.

    Description shown in the report -- the rest of the docstring.
    """
    ode = ...  # usage example

    # --- verification ---
    assert ode.foo == bar
```

- Docstring's first line -> report title; the rest -> description.
- `spec_ref`: anchor (string or list) of an `<a id="...">` entry in `docs/internal/spec.md`.
  Omit it (`@example()`) for examples that don't back a written requirement.
- `# --- verification ---`: everything above this marker is shown as the usage
  example; everything below (usually the asserts) is left out of the report. Omit
  the marker to show the whole test body.
