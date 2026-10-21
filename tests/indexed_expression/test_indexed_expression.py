"""Construction and property contracts of ``IndexedExpression``."""

from pytest import fixture, mark, raises
from sympy import Eq, Indexed, Piecewise, Symbol, true
from sympy.core.cache import clear_cache

from lin_ode.indexed_expression import IndexedExpression
from tests.util.comparison import idx as _idx
from tests.util.comparison import indexed_base as _indexed_base


@fixture(autouse=True)
def _clear_sympy_cache() -> None:
    """Idx and IndexedBase are cached by sympy, so identical names leak between
    tests."""
    clear_cache()


class TestAccepts:
    """What ``IndexedExpression`` keeps and derives from a well-formed equation."""

    def test_keeps_expr_state_variable_and_indices(self) -> None:
        """The three constructor arguments come back unchanged."""
        u, h = _indexed_base("u"), Symbol("h")
        i, j = _idx("i", (0, 20)), _idx("j", (0, 16))
        expr = (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) / h

        indexed_expr = IndexedExpression(expr, u, (i, j))

        assert indexed_expr.expr == expr
        assert indexed_expr.state_variable == u
        assert indexed_expr.indices == (i, j)

    @mark.parametrize(
        "index_ranges, expected_shape, expected_size",
        [
            (((0, 20),), (21,), 21),
            (((0, 20), (0, 16)), (21, 17), 21 * 17),
            (((-5, 5),), (11,), 11),
            ((None,), None, None),
            (((0, 20), None), None, None),
        ],
        ids=["1d", "2d", "shifted-range", "unbounded", "one-axis-unbounded"],
    )
    def test_shape_and_size_are_derived_from_index_bounds(
        self,
        index_ranges: tuple[tuple[int, int] | None, ...],
        expected_shape: tuple[int, ...] | None,
        expected_size: int | None,
    ) -> None:
        """``shape[k] = upper - lower + 1`` so a shifted range like (-5, 5) gives 11, and a
        single unbounded index makes the whole shape unknowable."""
        u = _indexed_base("u")
        indices = tuple(
            _idx(name, index_range) for name, index_range in zip("ij", index_ranges)
        )

        expr: Indexed = u[indices]  # type: ignore[assignment]

        indexed_expr = IndexedExpression(expr, u, indices)

        assert indexed_expr.shape == expected_shape
        assert indexed_expr.size == expected_size

    def test_cases_wraps_plain_expr_as_one_unconditional_arm(self) -> None:
        """``cases`` reads ``[(expr, true)]`` when no ``Piecewise`` is present."""
        u, h = _indexed_base("u"), Symbol("h")
        i = _idx("i", (0, 20))
        expr = (u[i - 1] + u[i + 1]) / h

        cases = IndexedExpression(expr, u, (i,)).cases

        assert cases == [(expr, true)]

    def test_cases_returns_piecewise_arms_in_evaluation_order(self) -> None:
        """The boundary arm comes first, exactly as ``Piecewise`` evaluates it."""
        u, h = _indexed_base("u"), Symbol("h")
        i = _idx("i", (0, 20))
        boundary, interior = u[i + 1] / h, (u[i - 1] + u[i + 1]) / h
        expr = Piecewise((boundary, Eq(i, 0)), (interior, true))

        cases = IndexedExpression(expr, u, (i,)).cases

        assert cases == [(boundary, Eq(i, 0)), (interior, true)]

    def test_accepts_expr_using_other_indexed_bases(self) -> None:
        """Only ``u`` is the state variable; ``v`` is a coefficient array and is allowed
        to appear alongside it."""
        u, v, h = _indexed_base("u"), _indexed_base("v"), Symbol("h")
        i = _idx("i", (0, 20))

        indexed_expr = IndexedExpression((u[i] + v[i]) / h, u, (i,))

        assert indexed_expr.state_variable == u
        assert {atom.base for atom in indexed_expr.atoms(Indexed)} == {u, v}

    def test_free_symbols_exposes_scalar_parameters_of_expr(self) -> None:
        """``h`` and ``alpha`` both appear in ``free_symbols``."""
        u, h, alpha = _indexed_base("u"), Symbol("h"), Symbol("alpha")
        i = _idx("i", (0, 20))

        indexed_expr = IndexedExpression((u[i - 1] + u[i + 1]) / h + alpha, u, (i,))

        assert {h, alpha} <= indexed_expr.free_symbols


class TestRejects:
    """Every malformed equation construction refuses, one `raises` each."""

    @mark.parametrize("not_an_expr", [42, "test", None], ids=["int", "str", "none"])
    def test_expr_that_is_not_sympy_expr(self, not_an_expr: object) -> None:
        """An ``int``, a ``str`` or ``None`` raises ``TypeError``."""
        u = _indexed_base("u")
        i = _idx("i", (0, 20))

        with raises(TypeError, match="expr must be a SymPy Expr"):
            _ = IndexedExpression(not_an_expr, u, (i,))  # type: ignore[arg-type]

    @mark.parametrize(
        "not_a_state_variable", [Symbol("u"), None, 123], ids=["symbol", "none", "int"]
    )
    def test_state_variable_not_indexedbase(
        self,
        not_a_state_variable: object,
    ) -> None:
        """``Symbol("u")`` is the near miss: right name, wrong type, so it cannot be
        subscripted."""
        i = _idx("i", (0, 20))

        with raises(TypeError, match="state_variable must be an IndexedBase"):
            _ = IndexedExpression(Symbol("h"), not_a_state_variable, (i,))  # type: ignore[arg-type]

    @mark.parametrize(
        "indices, message",
        [
            ((Symbol("i"),), r"indices\[0\] must be an Idx"),
            ((_idx("i", (0, 20)), Symbol("j")), r"indices\[1\] must be an Idx"),
            ((_idx("i", (0, 20)), 5), r"indices\[1\] must be an Idx"),
            ([_idx("i", (0, 20))], "indices must be a tuple"),
        ],
        ids=["symbol-at-0", "symbol-at-1", "int-at-1", "a-list-not-a-tuple"],
    )
    def test_indices_not_tuple_of_idx(self, indices: object, message: str) -> None:
        """The error names the failing position, and a list is rejected too."""
        u = _indexed_base("u")

        with raises(TypeError, match=message):
            _ = IndexedExpression(u[_idx("i", (0, 20))], u, indices)  # type: ignore[arg-type]

    def test_expr_without_state_variable(self) -> None:
        """The expression is built from ``v`` while ``u`` is declared as the state.

        There is nothing for the linearization to act on.
        """
        u, v, h = _indexed_base("u"), _indexed_base("v"), Symbol("h")
        i = _idx("i", (0, 20))

        with raises(ValueError, match="state variable u not found in expression"):
            _ = IndexedExpression(v[i] / h, u, (i,))
