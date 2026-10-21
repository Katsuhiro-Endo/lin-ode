# pyright: reportArgumentType=false
# sympy stubs type Piecewise as possibly NaN, so passing one back in reads as an error.
"""Input validation of ``OrdinaryDifferentialSystem.from_sympy`` and
``from_strings``."""

from math import prod
from typing import NamedTuple

from pytest import mark, raises
from sympy import Eq, Expr, Indexed, Integer, Mod, Piecewise, Symbol, true
from sympy.core.cache import clear_cache

from lin_ode.ordinary_differential_system import (
    DirichletCondition,
    Facet,
    OrdinaryDifferentialSystem,
)
from tests.util.comparison import idx as _idx
from tests.util.comparison import indexed_base as _indexed_base

N = 21  # sites of the 1-D grid
NX, NY = 21, 17  # sites per axis of the 2-D grid


def setup_function() -> None:
    """Idx and IndexedBase are cached by sympy, so identical names leak between
    tests."""
    clear_cache()


class TestFromSympy:
    """Every input contract ``from_sympy`` enforces, one case each."""

    def test_derives_shape_and_dof_from_index_ranges(self) -> None:
        """No shape is passed anywhere: (21, 17) comes from i's and j's bounds alone."""
        u, h = _indexed_base("u"), Symbol("h")
        i, j = _idx("i", (0, NX - 1)), _idx("j", (0, NY - 1))
        expr = (
            u[i, j]
            - (u[i + 1, j] + u[i - 1, j]) ** 2 / h  # type: ignore[operator]
            - (u[i, j + 1] + u[i, j - 1]) ** 2 / h  # type: ignore[operator]
        )

        ode = OrdinaryDifferentialSystem.from_sympy(expr, u, (i, j))

        assert ode.state_variable == u
        assert ode.index_symbols == (i, j)
        assert ode.shape == (NX, NY)
        assert ode.total_dof == prod((NX, NY)) == 357

    def test_returns_same_idx_object_that_was_passed_in(self) -> None:
        """index_symbols holds the same Idx object, not a copy with equal bounds."""
        u, h = _indexed_base("u"), Symbol("h")
        i = _idx("i", (0, N - 1))
        expr = Piecewise((0, Eq(i, 0)), ((u[i - 1] + u[i + 1]) / h, true))

        ode = OrdinaryDifferentialSystem.from_sympy(expr, u, (i,))

        assert ode.index_symbols[0] is i

    @mark.parametrize(
        "indices, error, message",
        [
            ((), ValueError, "indices must not be empty"),
            (
                (_idx("i", (0, N - 1)), Symbol("j")),
                TypeError,
                r"indices\[1\] must be an Idx",
            ),
        ],
        ids=["empty", "symbol-at-1"],
    )
    def test_rejects_indices_empty_or_holding_non_idx(
        self, indices: tuple, error: type[Exception], message: str
    ) -> None:
        """Empty indices raise ValueError, and a plain Symbol raises TypeError."""
        u = _indexed_base("u")
        expr: Indexed = u[_idx("i", (0, N - 1))]  # type: ignore[assignment]

        with raises(error, match=message):
            _ = OrdinaryDifferentialSystem.from_sympy(expr, u, indices)

    @mark.parametrize(
        "i_range, first_unbounded",
        [(None, "i"), ((0, NX - 1), "j")],
        ids=["both-unbounded", "only-j-unbounded"],
    )
    def test_rejects_index_without_explicit_range(
        self, i_range: tuple[int, int] | None, first_unbounded: str
    ) -> None:
        """J never carries a range, so the system is rejected whether or not i carries
        one.

        The first unbounded index is the one named.
        """
        u, h = _indexed_base("u"), Symbol("h")
        i, j = _idx("i", i_range), _idx("j")
        expr = (
            u[i, j]
            - (u[i + 1, j] + u[i - 1, j]) ** 2 / h  # type: ignore[operator]
            - (u[i, j + 1] + u[i, j - 1]) ** 2 / h  # type: ignore[operator]
        )

        with raises(ValueError, match=f"Index {first_unbounded} has no explicit range"):
            _ = OrdinaryDifferentialSystem.from_sympy(expr, u, (i, j))

    def test_rejects_state_variable_with_explicit_shape(self) -> None:
        """A shaped IndexedBase could disagree with the index bounds, so it is an
        error."""
        u, h = _indexed_base("u", shape=(NX, NY)), Symbol("h")
        i, j = _idx("i", (0, NX - 1)), _idx("j", (0, NY - 1))
        expr = (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) / h  # type: ignore[operator]

        with raises(ValueError, match="was created with an explicit shape"):
            _ = OrdinaryDifferentialSystem.from_sympy(expr, u, (i, j))

    def test_accepts_declared_index_absent_from_expr(self) -> None:
        """``u[0]`` is a complete equation on its own, one arm of a boundary Piecewise.

        The declared index i therefore need not appear in it.
        """
        u = _indexed_base("u")
        i = _idx("i", (0, N - 1))
        expr: Indexed = u[0]  # type: ignore[assignment]

        ode = OrdinaryDifferentialSystem.from_sympy(expr, u, (i,))

        assert ode.index_symbols == (i,)
        assert ode.total_dof == N

    def test_rejects_u_subscripted_out_of_order(self) -> None:
        """``u[j-1, i]`` puts j where i is expected, which would silently transpose a
        stencil."""
        u, h = _indexed_base("u"), Symbol("h")
        i, j = _idx("i", (0, NX - 1)), _idx("j", (0, NY - 1))
        expr = (
            u[i, j]
            - (u[i + 1, j] + u[j - 1, i]) ** 2 / h  # type: ignore[operator]
            - (u[i, j + 1] + u[i, j - 1]) ** 2 / h  # type: ignore[operator]
        )

        with raises(ValueError, match="Expression indexes u inconsistently"):
            _ = OrdinaryDifferentialSystem.from_sympy(expr, u, (i, j))

    @mark.parametrize(
        "literal, resolved",
        [(-1, N - 1), (N, 0)],
        ids=["below-the-range", "above-the-range"],
    )
    def test_out_of_range_literal_index_wraps_on_the_torus(
        self,
        literal: int,
        resolved: int,
    ) -> None:
        """Literals wrap modulo the index size: u[-1] is the last site, u[N] is u[0]."""
        u, h = _indexed_base("u"), Symbol("h")
        i = _idx("i", (0, N - 1))
        expr = Piecewise(
            ((u[literal] + u[i + 1]) / h, Eq(i, 0)),
            ((u[i - 1] + u[i + 1]) / h, true),
        )

        ode = OrdinaryDifferentialSystem.from_sympy(expr, u, (i,))

        assert u[resolved] in ode.equation.expr.atoms(Indexed)

    @mark.parametrize(
        "position, literal, resolved",
        [(0, NX, 0), (1, -1, NY - 1), (1, NY, 0)],
        ids=["axis-0-above", "axis-1-below", "axis-1-above"],
    )
    def test_out_of_range_literal_wraps_on_its_own_axis(
        self, position: int, literal: int, resolved: int
    ) -> None:
        """Each subscript wraps modulo the size of the axis it sits on."""
        u, h = _indexed_base("u"), Symbol("h")
        i, j = _idx("i", (0, NX - 1)), _idx("j", (0, NY - 1))
        wrapped = u[literal, j] if position == 0 else u[i, literal]
        expected = u[resolved, j] if position == 0 else u[i, resolved]
        expr = Piecewise(
            ((wrapped + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) / h, Eq(i, 0)),
            ((u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) / h, true),
        )

        ode = OrdinaryDifferentialSystem.from_sympy(expr, u, (i, j))

        assert expected in ode.equation.expr.atoms(Indexed)

    def test_accepts_literal_indices_on_both_2d_axes(self) -> None:
        """U[0,j], u[NX-1,j], u[i,0] and u[i,NY-1] all sit on their axis' bounds, so the
        5-arm system is accepted."""
        u, h = _indexed_base("u"), Symbol("h")
        i, j = _idx("i", (0, NX - 1)), _idx("j", (0, NY - 1))
        expr = Piecewise(
            ((u[0, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) / h, Eq(i, 0)),
            (
                (u[i - 1, j] + u[NX - 1, j] + u[i, j - 1] + u[i, j + 1]) / h,
                Eq(i, NX - 1),
            ),
            ((u[i - 1, j] + u[i + 1, j] + u[i, 0] + u[i, j + 1]) / h, Eq(j, 0)),
            (
                (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, NY - 1]) / h,
                Eq(j, NY - 1),
            ),
            ((u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) / h, true),
        )

        ode = OrdinaryDifferentialSystem.from_sympy(expr, u, (i, j))

        assert ode.total_dof == NX * NY
        assert len(ode.equation.cases) == 5


# N == NX == 21, so one i serves the 1-D and the 2-D rows.
u, h = _indexed_base("u"), Symbol("h")
i, j = _idx("i", (0, NX - 1)), _idx("j", (0, NY - 1))

INTERIOR_1D = (u[i - 1] + u[i + 1]) / h
INTERIOR_2D = (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) / h


class Stencil(NamedTuple):
    """A hand-written piecewise right-hand side, with what sympy keeps of it."""

    expr: Piecewise
    indices: tuple
    cases: int
    first_arm: Expr


STENCILS = [
    Stencil(  # both boundaries evaluate to 0, so sympy merges them: 3 arms in, 2 out
        expr=Piecewise((0, Eq(i, 0)), (0, Eq(i, N - 1)), (INTERIOR_1D, true)),
        indices=(i,),
        cases=2,
        first_arm=Integer(0),
    ),
    Stencil(  # all four edges evaluate to 0 and merge the same way: 5 in, 2 out
        expr=Piecewise(
            (0, Eq(i, 0)),
            (0, Eq(i, NX - 1)),
            (0, Eq(j, 0)),
            (0, Eq(j, NY - 1)),
            (INTERIOR_2D, true),
        ),
        indices=(i, j),
        cases=2,
        first_arm=Integer(0),
    ),
    Stencil(  # periodic: the arms differ, so all 3 survive with their far-side neighbours
        expr=Piecewise(
            ((u[N - 1] + u[i + 1]) / h, Eq(i, 0)),
            ((u[i - 1] + u[0]) / h, Eq(i, N - 1)),
            (INTERIOR_1D, true),
        ),
        indices=(i,),
        cases=3,
        # symbolic u[i + 1] wraps to u[Mod(i + 1, N)] on the torus
        first_arm=(u[N - 1] + u[Mod(i + 1, N)]) / h,
    ),
    Stencil(  # Dirichlet at i=0, aliasing at i=NX-1, periodic in j: 5 distinct arms
        expr=Piecewise(
            (0, Eq(i, 0)),
            (u[i - 1, j] / h, Eq(i, NX - 1)),
            ((u[i - 1, j] + u[i + 1, j] + u[i, NY - 1] + u[i, j + 1]) / h, Eq(j, 0)),
            ((u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, 0]) / h, Eq(j, NY - 1)),
            (INTERIOR_2D, true),
        ),
        indices=(i, j),
        cases=5,
        first_arm=Integer(0),
    ),
]

STENCIL_IDS = ["1d-dirichlet", "2d-dirichlet", "1d-periodic", "2d-mixed-bc"]


class TestDirectPiecewise:
    """A right-hand side written directly as a sympy Piecewise, without the conditions
    API."""

    @mark.parametrize("stencil", STENCILS, ids=STENCIL_IDS)
    def test_keeps_only_the_arms_sympy_cannot_merge(
        self,
        stencil: Stencil,
    ) -> None:
        """Equal arms collapse into one Or arm; differing arms all survive, in order."""
        cases = OrdinaryDifferentialSystem.from_sympy(
            stencil.expr, u, stencil.indices
        ).equation.cases

        assert len(cases) == stencil.cases
        assert cases[0][0] == stencil.first_arm
        assert cases[-1][1] == true


class TestConditionsMatchDirectPiecewise:
    """A boundary condition supplied through the conditions API assembles the same
    equation as the piecewise a user would write for it by hand."""

    def test_dirichlet_edges_1d(self) -> None:
        """u = 0 on both ends: Dirichlet conditions match the hand-written arms."""
        direct = OrdinaryDifferentialSystem.from_sympy(
            Piecewise((0, Eq(i, 0)), (0, Eq(i, N - 1)), (INTERIOR_1D, true)),
            u,
            (i,),
        )
        via_conditions = OrdinaryDifferentialSystem.from_sympy(
            INTERIOR_1D,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), 0),
                DirichletCondition(Facet.high(i), 0),
            ],
        )

        assert via_conditions.equation.expr == direct.equation.expr

    def test_dirichlet_edges_2d(self) -> None:
        """u = 0 on all four edges of the 2-D grid."""
        direct = OrdinaryDifferentialSystem.from_sympy(
            Piecewise(
                (0, Eq(i, 0)),
                (0, Eq(i, NX - 1)),
                (0, Eq(j, 0)),
                (0, Eq(j, NY - 1)),
                (INTERIOR_2D, true),
            ),
            u,
            (i, j),
        )
        via_conditions = OrdinaryDifferentialSystem.from_sympy(
            INTERIOR_2D,
            u,
            (i, j),
            conditions=[
                DirichletCondition(Facet.low(i), 0),
                DirichletCondition(Facet.high(i), 0),
                DirichletCondition(Facet.low(j), 0),
                DirichletCondition(Facet.high(j), 0),
            ],
        )

        assert via_conditions.equation.expr == direct.equation.expr


class TestFromStrings:
    """What the string parser derives, and what it refuses."""

    @mark.parametrize(
        "equation, indices, expected_dof, expected_names",
        [
            ("u[i] - (u[i+1] + u[i-1])**2/h", (_idx("i", (0, N - 1)),), N, ("i",)),
            (
                "u[i,j] - (u[i+1,j] + u[i-1,j])**2/h - (u[i,j+1] + u[i,j-1])**2/h",
                (_idx("i", (0, NX - 1)), _idx("j", (0, NY - 1))),
                NX * NY,
                ("i", "j"),
            ),
        ],
        ids=["one-index", "two-indices"],
    )
    def test_derives_dof_and_index_order(
        self,
        equation: str,
        indices: tuple,
        expected_dof: int,
        expected_names: tuple[str, ...],
    ) -> None:
        """The index ranges alone fix the degrees of freedom and the index order."""
        ode = OrdinaryDifferentialSystem.from_strings(equation, indices=indices)

        assert ode.total_dof == expected_dof
        assert tuple(idx.name for idx in ode.index_symbols) == expected_names

    def test_rejects_what_from_sympy_rejects(self) -> None:
        """from_strings shares from_sympy's validation: an inconsistently-indexed
        expression is rejected the same way."""
        i, j = _idx("i", (0, NX - 1)), _idx("j", (0, NY - 1))
        with raises(ValueError, match="Expression indexes u inconsistently"):
            _ = OrdinaryDifferentialSystem.from_strings(
                "u[i,j] - (u[i+1,j] + u[j-1,i])**2/h", indices=(i, j)
            )
