# pyright: reportArgumentType=false, reportIndexIssue=false, reportOperatorIssue=false
# sympy stubs mistype Indexed subscripting and Idx arithmetic.
"""The right-hand sides the builder tests drive, one function each.

Every one is an ``OrdinaryDifferentialSystem``, so a test states only which shape of
equation it needs. ``eps`` and ``nu`` are the Burgers constants.
"""

import sympy as sp

from lin_ode.indexed_expression import IndexedExpression
from lin_ode.ordinary_differential_system import OrdinaryDifferentialSystem
from tests.testing import helper

eps, nu = sp.symbols("epsilon nu")


def _burgers_stencil(
    u: sp.IndexedBase, i: sp.Idx, left: sp.Expr, right: sp.Expr
) -> sp.Expr:
    """Centred-difference Burgers stencil at site ``i``, between the given
    neighbours."""
    alpha = -1 / (2 * eps)
    beta = nu / eps**2
    return alpha * (u[i] * right - u[i] * left) + beta * (right - 2 * u[i] + left)


def burgers_equation(n_site: int) -> OrdinaryDifferentialSystem:
    """Notebook Burgers RHS, periodic (``u[i + n] = u[i]``): the wrap written compactly as
    ``Mod`` rather than unrolled endpoint arms."""
    u = sp.IndexedBase("u")
    i = sp.Idx("i", (0, n_site - 1))
    left, right = u[sp.Mod(i - 1, n_site)], u[sp.Mod(i + 1, n_site)]
    return OrdinaryDifferentialSystem.from_sympy(
        _burgers_stencil(u, i, left, right), u, (i,)
    )


def _unrolled_periodic_burgers(n_site: int) -> OrdinaryDifferentialSystem:
    """Periodicity as explicit arms: the independent spelling, so it must not come from
    the ``Mod`` form it checks."""
    u = sp.IndexedBase("u")
    i = sp.Idx("i", (0, n_site - 1))
    rhs = sp.Piecewise(
        (_burgers_stencil(u, i, u[n_site - 1], u[i + 1]), sp.Eq(i, 0)),
        (_burgers_stencil(u, i, u[i - 1], u[0]), sp.Eq(i, n_site - 1)),
        (_burgers_stencil(u, i, u[i - 1], u[i + 1]), sp.true),
    )
    return OrdinaryDifferentialSystem(IndexedExpression(rhs, u, (i,)))


def stride_equation_mod(n_site: int) -> OrdinaryDifferentialSystem:
    """Stride-2 stencil, periodic: ``Mod(2*i - 1, n)`` has no unique inverse (3 fires at both
    i=2 and i=5), so the deltas pin the index instead."""
    u = sp.IndexedBase("u")
    i = sp.Idx("i", (0, n_site - 1))
    return OrdinaryDifferentialSystem.from_sympy(
        u[i] * u[sp.Mod(2 * i - 1, n_site)], u, (i,)
    )


@helper
def quadratic_equation(n_site: int) -> OrdinaryDifferentialSystem:
    """du_i/dt = u_i**2: a single-case (no boundary) nonlinear RHS."""
    u = sp.IndexedBase("u")
    i = sp.Idx("i", (0, n_site - 1))
    return OrdinaryDifferentialSystem.from_sympy(u[i] ** 2, u, (i,))


@helper
def stencil_2d_equation_mod(n: int) -> OrdinaryDifferentialSystem:
    """2-D stencil, periodic on the stepped axis only, so one axis is ``Mod`` and the
    other a bare index."""
    u = sp.IndexedBase("u")
    i, j = sp.Idx("i", (0, n - 1)), sp.Idx("j", (0, n - 1))
    return OrdinaryDifferentialSystem.from_sympy(
        u[i, j] * u[sp.Mod(i - 1, n), j], u, (i, j)
    )


@helper
def quadratic_equation_2d(n: int) -> OrdinaryDifferentialSystem:
    """du_{i,j}/dt = u_{i,j}**2 on an n x n grid: boundary-free, 2 index symbols."""
    u = sp.IndexedBase("u")
    i, j = sp.Idx("i", (0, n - 1)), sp.Idx("j", (0, n - 1))
    return OrdinaryDifferentialSystem.from_sympy(u[i, j] ** 2, u, (i, j))


def corner_equation_2d(n: int) -> OrdinaryDifferentialSystem:
    """Corner ``(0, 0)`` quadratic, linear elsewhere: the interior arm's condition
    couples both axes, and the linear interior is index-free under Liouville
    ordering."""
    u = sp.IndexedBase("u")
    i, j = sp.Idx("i", (0, n - 1)), sp.Idx("j", (0, n - 1))
    rhs = sp.Piecewise(
        (u[i, j] ** 2, sp.And(sp.Eq(i, 0), sp.Eq(j, 0))),
        (u[i, j], sp.true),
    )
    return OrdinaryDifferentialSystem.from_sympy(rhs, u, (i, j))


def corner_and_edges_equation_2d(n: int) -> OrdinaryDifferentialSystem:
    """The usual shape of boundary conditions: corner, then each edge, then the
    interior.

    Every arm past the first carries the earlier corner negated: ``Eq(i, 0) &
    ~(Eq(i, 0) & Eq(j, 0))`` is the i=0 edge minus the corner, which a per-axis count
    cannot handle.
    """
    u = sp.IndexedBase("u")
    i, j = sp.Idx("i", (0, n - 1)), sp.Idx("j", (0, n - 1))
    rhs = sp.Piecewise(
        (2 * u[i, j], sp.And(sp.Eq(i, 0), sp.Eq(j, 0))),
        (3 * u[i, j], sp.Eq(i, 0)),
        (4 * u[i, j], sp.Eq(j, 0)),
        (u[i, j], sp.true),
    )
    return OrdinaryDifferentialSystem.from_sympy(rhs, u, (i, j))


def stencil_2d_equation(n: int) -> OrdinaryDifferentialSystem:
    """2-D stencil behind a Dirichlet ``i = 0`` face: that face is held fixed, so its arm is
    0, and the interior wraps periodically on the stepped axis."""
    u = sp.IndexedBase("u")
    i, j = sp.Idx("i", (0, n - 1)), sp.Idx("j", (0, n - 1))
    rhs = sp.Piecewise(
        (sp.S.Zero, sp.Eq(i, 0)),
        (u[i, j] * u[sp.Mod(i - 1, n), j], sp.true),
    )
    return OrdinaryDifferentialSystem.from_sympy(rhs, u, (i, j))
