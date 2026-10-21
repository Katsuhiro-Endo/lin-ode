"""Marks a test as a usage example, and reads back what the report needs, by section.

the marks         ``@example`` and ``@helper`` what to show      a test's source, its
helpers, the values it ran with finding a report  a heading in docs/reports/ and the
file that holds it titles and order  the heading a test file lands under, from its
``_toc.py``
"""

import ast
import importlib.util
import inspect
import sys
import types
import pprint
import re
import textwrap
import unicodedata
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar
import warnings

from lin_ode.basis import (
    CarlemanBasis,
    GaussianHermiteBasis,
    PSCarlemanBasis,
    TruncatedBasisSet,
)
from lin_ode.indexed_expression import IndexedExpression
from lin_ode.matrix import Matrix, OperatorMatrix
from lin_ode.ordinary_differential_system import (
    Box,
    Condition,
    DerivativeCondition,
    DirichletCondition,
    Facet,
    OrdinaryDifferentialSystem,
    PinnedRegion,
    StateCondition,
)
from lin_ode.operators import LinearOperator, OperatorSum, SimpleOperator
from lin_ode.vector import Vector

SPEC_MARKDOWN_PATH = "docs/internal/spec.md"

REPORTS_DIR = "docs/reports"
"""Verification reports, one ``<stem>.md`` index each.

`@example(report=...)` names a heading in one of them; conftest checks it exists. These
markdown files are the source of truth, edited here. Each splits its per-system sections
into ``<stem>/<slug>.md``, which is what a generated page includes.
"""

SPEC_DOC_PATH = "docs/internal/spec.html"
"""What Sphinx renders SPEC_MARKDOWN_PATH to; an entry's spec links point here."""

TESTS_DIR = Path(__file__).parent

_VERIFICATION_MARKER = "# --- verification ---"

LEVELS = ("basic", "advanced")
"""Reading order for the examples index: basic first, advanced after."""

_F = TypeVar("_F", bound=Callable)


def assert_equal(x: Any, y: Any) -> None:
    match x, y:
        case CarlemanBasis(), CarlemanBasis():
            _assert_equal_carleman_basis(x, y)
        case PSCarlemanBasis(), PSCarlemanBasis():
            _assert_equal_psc_basis(x, y)
        case GaussianHermiteBasis(), GaussianHermiteBasis():
            _assert_equal_gauss_hermite_basis(x, y)
        # We should not compare two TruncatedBasisSets since instances of different subclasses can be regarded as equal
        # case TruncatedBasisSet(), TruncatedBasisSet():
        #     _assert_equal_truncated_basis_set(x, y)
        case Vector(), Vector():
            _assert_equal_vector(x, y)
        case IndexedExpression(), IndexedExpression():
            _assert_equal_indexed_expression(x, y)
        case Box(), Box():
            _assert_equal_box(x, y)
        case Facet(), Facet():
            _assert_equal_facet(x, y)
        case PinnedRegion(), PinnedRegion():
            _assert_equal_pinned_region(x, y)
        case DirichletCondition(), DirichletCondition():
            _assert_equal_dirichlet_condition(x, y)
        case DerivativeCondition(), DerivativeCondition():
            _assert_equal_derivative_condition(x, y)
        case OrdinaryDifferentialSystem(), OrdinaryDifferentialSystem():
            _assert_equal_ode(x, y)
        case OperatorSum(), OperatorSum():
            _assert_equal_operator_sum(x, y)
        case SimpleOperator(), SimpleOperator():
            _assert_equal_simple_operator(x, y)
        # We should not compare two LinearOperator since instances of different subclasses can be regarded as equal
        # case LinearOperator(), LinearOperator():
        #     _assert_equal_linear_operator(x, y)
        case OperatorMatrix(), OperatorMatrix():
            _assert_equal_operator_matrix(x, y)
        case Matrix(), Matrix():
            assert x == y
        case list(), list():
            _assert_equal_float_list(x, y)
        case _:
            assert False


def _assert_equal_truncated_basis_set(
    x: TruncatedBasisSet, y: TruncatedBasisSet
) -> None:
    assert x.state_type == y.state_type
    assert x.is_l2 == y.is_l2
    assert x.n_vars == y.n_vars
    assert x.degree == y.degree
    warnings.warn("TruncatedBasisSet.projectors cannot be compared yet")


def _assert_equal_carleman_basis(x: CarlemanBasis, y: CarlemanBasis) -> None:
    _assert_equal_truncated_basis_set(x, y)


def _assert_equal_psc_basis(x: PSCarlemanBasis, y: PSCarlemanBasis) -> None:
    _assert_equal_truncated_basis_set(x, y)
    assert x.pivot == y.pivot


def _assert_equal_gauss_hermite_basis(
    x: GaussianHermiteBasis, y: GaussianHermiteBasis
) -> None:
    _assert_equal_truncated_basis_set(x, y)


def _assert_equal_vector(x: Vector, y: Vector) -> None:
    assert x._terms == y._terms
    assert_equal(x._basis, y._basis)
    assert x._u == y._u
    assert x._shape == y._shape


def _assert_equal_indexed_expression(
    x: IndexedExpression, y: IndexedExpression
) -> None:
    assert x._expr == y._expr
    assert x._state_variable == y._state_variable
    assert x._indices == y._indices


def _assert_equal_pinned_region(x: PinnedRegion, y: PinnedRegion) -> None:
    assert x._fixed == y._fixed


def _assert_equal_facet(x: Facet, y: Facet) -> None:
    assert x == y


def _assert_equal_box(x: Box, y: Box) -> None:
    assert x._ranges == y._ranges


def _assert_equal_condition(x: Condition, y: Condition) -> None:
    assert_equal(x._region, y._region)


def _assert_equal_state_condition(x: StateCondition, y: StateCondition) -> None:
    _assert_equal_condition(x, y)
    assert x._value == y._value


def _assert_equal_dirichlet_condition(
    x: DirichletCondition, y: DirichletCondition
) -> None:
    _assert_equal_state_condition(x, y)


def _assert_equal_derivative_condition(
    x: DerivativeCondition, y: DerivativeCondition
) -> None:
    _assert_equal_condition(x, y)
    assert x._rhs == y._rhs


def _assert_equal_ode(
    x: OrdinaryDifferentialSystem, y: OrdinaryDifferentialSystem
) -> None:
    assert_equal(x._equation, y._equation)
    assert len(x._state_conditions) == len(y._state_conditions)
    for sc_x, sc_y in zip(x._state_conditions, y._state_conditions):
        _assert_equal_state_condition(sc_x, sc_y)
    assert len(x._derivative_conditions) == len(y._derivative_conditions)
    for dc_x, dc_y in zip(x._derivative_conditions, y._derivative_conditions):
        _assert_equal_derivative_condition(dc_x, dc_y)


def _assert_equal_linear_operator(x: LinearOperator, y: LinearOperator) -> None:
    assert x.requires_l2_basis == y.requires_l2_basis
    assert x.method_name == y.method_name
    assert_equal(x._operator, y._operator)
    assert x._diff_base == y._diff_base


def _assert_equal_operator_sum(x: OperatorSum, y: OperatorSum) -> None:
    _assert_equal_linear_operator(x, y)
    assert x._indices == y._indices


def _assert_equal_simple_operator(x: SimpleOperator, y: SimpleOperator) -> None:
    _assert_equal_linear_operator(x, y)


def _assert_equal_operator_matrix(x: OperatorMatrix, y: OperatorMatrix) -> None:
    assert_equal(x._basis, y._basis)
    assert_equal(x._operator, y._operator)


def _assert_equal_float_list(x: list[float], y: list[float]) -> None:
    assert x == y


def example(
    spec_ref: str | list[str] | None = None,
    report: str | None = None,
    report_section: str | None = None,
    problem: str | None = None,
    checks: str | None = None,
    expects: str | None = None,
    level: str = "basic",
) -> Callable[[_F], _F]:
    """Mark a test as a report example; its docstring supplies the title and
    description.

    ``spec_ref`` anchors into the spec. ``report``/``report_section`` name the
    derivation page and heading. ``problem``/``checks``/``expects`` each render as one
    table row. ``level`` is "basic" or "advanced".
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {sorted(LEVELS)}, not {level!r}")
    anchors = [spec_ref] if isinstance(spec_ref, str) else (spec_ref or [])

    def decorator(func: _F) -> _F:
        if not inspect.getdoc(func):
            raise ValueError(
                f"@example on {func.__qualname__} needs a docstring: its summary line "
                "becomes the report title, the rest becomes the description."
            )
        func.__example__ = [  # type: ignore[attr-defined]
            f"{SPEC_DOC_PATH}#{anchor}" for anchor in anchors
        ]
        func.__report__ = report  # type: ignore[attr-defined]
        func.__report_section__ = report_section  # type: ignore[attr-defined]
        func.__problem__ = problem  # type: ignore[attr-defined]
        func.__checks__ = checks  # type: ignore[attr-defined]
        func.__expects__ = expects  # type: ignore[attr-defined]
        func.__level__ = level  # type: ignore[attr-defined]
        return func

    return decorator


def helper(func: _F) -> _F:
    """Mark a module-level function as supporting code for the tests that call it.

    The report renders it in its own block beside the test. A test body can then stay
    short without leaving the reader to guess what it does.
    """
    func.__helper__ = True  # type: ignore[attr-defined]
    return func


def _names_in_body(target: Callable) -> set[str]:
    """Names the test itself uses, in the part the report shows.

    A name appearing only in ``@example(report=...)`` routes the entry; it is not
    something the test ran with. One appearing only after the verification marker is not
    on the page, so listing its value would point at code nobody can see.
    """
    try:
        source = textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError):
        return set()
    lines = source.splitlines()
    heads = (
        i for i, text in enumerate(lines) if text.startswith(("def ", "async def "))
    )
    start = next(heads, 0)
    body = "\n".join(lines[start:]).split(_VERIFICATION_MARKER)[0]
    try:
        tree = ast.parse(body)
    except SyntaxError:  # the truncation can cut a bracket open
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _names_in(target: Callable) -> set[str]:
    """Names in the part of the test the report shows, so nothing is listed unseen."""
    try:
        source = textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError):
        return set()
    source = source.split(_VERIFICATION_MARKER)[0]
    try:
        tree = ast.parse(source)
    except SyntaxError:  # the truncation can cut a bracket open
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _source_without_decorator(target: Callable) -> str:
    """The helper's own source; the ``@helper`` line is report plumbing, not content."""
    lines = textwrap.dedent(inspect.getsource(target)).splitlines()
    while lines and lines[0].startswith("@helper"):
        lines.pop(0)
    return "\n".join(lines).strip("\n")


def helpers_used_by(func: Callable) -> list[tuple[str, str]]:
    """The ``@helper`` functions this test names, in file order.

    Only what the test itself calls. Decorating is deliberate, so the block stays the
    two or three worth reading, not every function the call graph reaches.
    """
    module = inspect.getmodule(func)
    if module is None:
        return []
    marked = {
        name: value
        for name, value in vars(module).items()
        if callable(value)
        and getattr(value, "__helper__", False)
        and name in _names_in(func)
    }

    def line_of(name: str) -> int:
        try:
            return inspect.getsourcelines(marked[name])[1]
        except (OSError, TypeError):
            return 0

    return [
        (name, _source_without_decorator(marked[name]))
        for name in sorted(marked, key=line_of)
    ]


def _assignments_in(module: object, wanted: set[str]) -> list[tuple[int, str, str]]:
    """Every ``wanted`` name this module assigns, as ``(line, name, source)``."""
    try:
        source = inspect.getsource(module)  # type: ignore[arg-type]
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return []
    lines = source.splitlines()
    found = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [x.id for x in targets if isinstance(x, ast.Name)]
        if not names or names[0] not in wanted:
            continue
        value = getattr(module, names[0], None)
        if isinstance(value, (type, types.GenericAlias)) or repr(value).startswith(
            "typing."
        ):
            continue  # a type alias describes the shape, not what the test ran with
        text = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        found.append((node.lineno, names[0], _plainest_form(names[0], value, text)))
    return found


def constants_used_by(func: Callable) -> list[tuple[str, str]]:
    """Module-level constants the test names, as ``(name, source)``.

    A reader with no source access still needs to know what ``METHODS`` holds. Imported
    constants are followed to the module that assigns them.
    """
    module = inspect.getmodule(func)
    if module is None:
        return []
    referenced = _names_in_body(func)
    found = _assignments_in(module, referenced)
    named_here = {name for _, name, _ in found}
    for source_module in _modules_imported_by(module):
        for line, name, text in _assignments_in(source_module, referenced - named_here):
            found.append((line, name, text))
            named_here.add(name)
    return [(name, text) for _, name, text in sorted(found)]


def _imports_of(module: object) -> list[object]:
    """The suite's own modules this one imports names from, in import order.

    A report's inputs are test inputs, so library imports are skipped.
    """
    try:
        tree = ast.parse(inspect.getsource(module))  # type: ignore[arg-type]
    except (OSError, TypeError, SyntaxError):
        return []
    package = getattr(module, "__package__", "") or ""
    named = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        # a relative import names no package, so resolve it against this module's own
        name = node.module or ""
        if node.level:
            name = f"{package}.{name}" if name else package
        if name.startswith("tests"):
            named.append(name)
    return [sys.modules[name] for name in named if name in sys.modules]


def _modules_imported_by(module: object) -> list[object]:
    """Where a test's names are assigned, following a package re-export one hop.

    ``_shared/__init__.py`` only re-exports, so the assignment sits one module further.
    """
    direct = _imports_of(module)
    seen = {id(module), *(id(m) for m in direct)}
    out = list(direct)
    for imported in direct:
        for onward in _imports_of(imported):
            if id(onward) not in seen:
                seen.add(id(onward))
                out.append(onward)
    return out


def _plainest_form(name: str, value: object, source: str) -> str:
    """The constant's value when that is plain data, else the assignment as written.

    A source line reading ``CONFIGS = {RING_UNIFORM: ...}`` only moves the reader's
    question along by one name; the value spells the heading out. Objects have no useful
    ``repr``, so those keep the source.
    """
    if not _is_plain_data(value):
        return source
    body = pprint.pformat(value, width=88, sort_dicts=False)
    return f"{name} = {body}"


def _is_plain_data(value: object) -> bool:
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_is_plain_data(item) for item in value)
    if isinstance(value, dict):
        return all(
            _is_plain_data(key) and _is_plain_data(item) for key, item in value.items()
        )
    return False


def render_inputs(params: dict[str, list], constants: list[tuple[str, str]]) -> str:
    """The concrete values a test ran with, as Python a reader can just read.

    A parametrised test binds its argument to each value in turn. The first is shown as
    an assignment, the rest listed under it.
    """
    blocks: list[str] = []
    for name, values in params.items():
        blocks.append(f"{name} = {values[0]!r}")
        rest = values[1:]
        if rest:
            tail = (
                "the other one" if len(rest) == 1 else f"each of the other {len(rest)}"
            )
            blocks.append(f"# then repeats for {tail}:")
            blocks += [f"#     {value!r}" for value in rest]
    blocks += [source for _, source in constants]
    return "\n".join(blocks)


def example_title_and_description(func: Callable) -> tuple[str, str]:
    """Split a docstring into (summary line, rest), PEP 257-style.

    The summary's closing period is docformatter's, not a heading's, so it is dropped.
    """
    doc = inspect.getdoc(func) or ""
    title, _, rest = doc.partition("\n")
    return title.strip().removesuffix("."), inspect.cleandoc(rest)


def extract_usage_source(func: Callable) -> str:
    """A test's source with decorators and docstring stripped, both shown elsewhere.

    Truncated at ``_VERIFICATION_MARKER`` if present.
    """
    source = inspect.getsource(func)

    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "async def ")):
            source = textwrap.dedent("\n".join(lines[index:]))
            break

    if _VERIFICATION_MARKER in source:
        source = source.split(_VERIFICATION_MARKER)[0]

    return _without_docstring(source).strip("\n")


def _without_docstring(source: str) -> str:
    (func_node,) = ast.parse(source).body
    body = func_node.body  # type: ignore[attr-defined]
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        doc_node = body[0]
        lines = source.splitlines(keepends=True)
        del lines[doc_node.lineno - 1 : doc_node.end_lineno]
        source = "".join(lines)
    return source


def report_slug(heading: str) -> str:
    """A report heading -> the anchor Sphinx gives it.

    Mirrors ``docutils.nodes.make_id``, which drops non-ASCII instead of hyphenating it,
    so ``KPP-Fisher`` becomes ``kppfisher``. Copied rather than imported, since docutils
    ships with the docs group and the suite never installs it.
    """
    folded = (
        unicodedata.normalize("NFKD", heading.lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(
        r"^[-0-9]+|-+$", "", re.sub(r"[^a-z0-9]+", "-", " ".join(folded.split()))
    )


def report_section_path(heading: str) -> Path | None:
    """The include-only source split out for this heading, in whichever report owns it.

    Which headings get one is the report's own call. It is read off the files on disk,
    not guessed from the text. ``configs/`` holds exactly those: every other page under a
    report is built as a document of its own.
    """
    for index in report_indexes():
        section = index.with_suffix("") / "configs" / f"{report_slug(heading)}.md"
        if section.exists():
            return section
    return None


def report_indexes() -> list[Path]:
    """Each report's index page: ``docs/reports/<stem>.md``."""
    return sorted(Path(REPORTS_DIR).glob("*.md"))


def report_href(heading: str) -> str:
    """Where to link a report heading, relative to ``docs/``.

    A configuration's tables and the tests verifying them share one generated page, so
    that page is the link; every other heading is an anchor on the page that holds it.
    """
    slug = report_slug(heading)
    if report_section_path(heading):
        return f"_generated/examples/config-{slug}.html"
    for source in report_sources():
        if heading in report_headings(source.read_text(encoding="utf-8")):
            return (
                f"{source.with_suffix('').as_posix().removeprefix('docs/')}.html#{slug}"
            )
    return f"{REPORTS_DIR.removeprefix('docs/')}.html#{slug}"


def report_sources() -> list[Path]:
    """Every report markdown file: each index, then every page under it."""
    out = []
    for index in report_indexes():
        out.append(index)
        out += sorted(index.with_suffix("").rglob("*.md"))
    return out


def report_headings(source: str) -> set[str]:
    """Every heading in the report markdown, verbatim.

    Fenced blocks are skipped: a Python comment is `# like a heading` and would
    otherwise let a mistyped `report=` validate against it.
    """
    headings, fenced = set(), False
    for line in source.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and (heading := re.match(r"#+ (.+)$", line)):
            headings.add(heading.group(1).strip())
    return headings


def category_for(test_path: Path) -> str:
    """Report category for a test file: the name of its immediate parent directory
    under tests/, or "General" for a test file directly under tests/. Fallback for
    `section_title_and_order`, used when there's no `_toc.py` entry."""
    relative = test_path.relative_to(TESTS_DIR)
    if len(relative.parts) < 2:
        return "General"
    return relative.parts[-2].replace("_", " ").title()


TOC_FILENAME = "_toc.py"
"""A directory opts every section directly inside it into explicit report titles/order
by placing a file with this name inside it, holding a `TOC` dict:

`{section_dir_name: (h2_title, {file_name: h3_title})}` (see tests/integration/_toc.py).
Not collected by pytest (no `test_` prefix).
"""


def section_title_and_order(test_path: Path) -> tuple[str, int | None]:
    """``<h2>`` title and sort order for *test_path*'s section, its parent directory.

    Taken from the parent's ``TOC_FILENAME`` entry if any, else ``category_for`` with no
    order.
    """
    section = _toc_section(test_path)
    if section is None:
        return category_for(test_path), None
    toc, key = section
    title, _files = toc[key]
    return title, list(toc).index(key)


def file_title_and_order(test_path: Path) -> tuple[str, int | None]:
    """``<h3>`` title and sort order for *test_path*.

    From the same TOC entry ``section_title_and_order`` reads, else an auto-generated
    title with no order.
    """
    auto_title = test_path.stem.removeprefix("test_").replace("_", " ").title()
    section = _toc_section(test_path)
    if section is None:
        return auto_title, None
    toc, key = section
    _title, files = toc[key]
    if test_path.name not in files:
        return auto_title, None
    return files[test_path.name], list(files).index(test_path.name)


def _toc_section(test_path: Path) -> tuple[dict, str] | None:
    """(TOC dict, section key) from *test_path*'s grandparent's `TOC_FILENAME`."""
    toc_path = test_path.parent.parent / TOC_FILENAME
    if not toc_path.is_file():
        return None
    toc = _load_by_path(toc_path).TOC
    key = test_path.parent.name
    return (toc, key) if key in toc else None


def _load_by_path(path: Path) -> ModuleType:
    """Import a file by path.

    ``tests/`` has no ``__init__.py``, so a dotted name cannot reach it.
    """
    spec = importlib.util.spec_from_file_location(f"_toc_{path.parent.name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
