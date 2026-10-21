# pyright: reportArgumentType=false, reportIndexIssue=false
# sympy stubs type IndexedBase subscripting and Idx tuples loosely; the noise is
# irreducible here.
"""Regions taken through coefficient extraction, one box at a time and as unions.

``_box_count`` counts a conjunction of per-axis factors, so a union of boxes is not a
product of per-axis sizes.  Every case in ``TestUnionOfBoxes`` raised
``NotImplementedError`` before ``_region_count``; every expected value comes from the
grid scan in ``oracle``, which shares no code with ``lin_ode``.
"""

import itertools

import pytest
import sympy as sp

from lin_ode.basis import GaussianHermiteBasis
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import Liouville
from lin_ode.ordinary_differential_system import (
    Box,
    DerivativeCondition,
    DirichletCondition,
    Facet,
    OrdinaryDifferentialSystem,
)


def _idx(name: str, index_range: tuple[int, int]) -> sp.Idx:
    """Typed wrapper for ``Idx(...)``; sympy's own constructor return type is too wide
    for pyright to narrow."""
    return sp.Idx(name, index_range)  # type: ignore[assignment]


N = 3
u = sp.IndexedBase("u")
a, c = sp.symbols("a c")
i, j, k = (_idx(name, (0, N - 1)) for name in "ijk")
PLANE, SHAPE = (i, j), (N, N)
BOX = Box({i: (1, 2), j: (1, 2)})


def system(indices, conditions):
    state = u[indices] if len(indices) > 1 else u[indices[0]]
    return OrdinaryDifferentialSystem.from_sympy(
        a * state, u, indices, conditions=conditions
    )


def vac_element(indices, shape, conditions):
    """``-sum_grid div f``: a pure free-axis count on every axis.

    Liouville needs an L2 basis, so the probe is Gauss-Hermite, whose vacuum diagonal is
    ``-sum_grid div f / 2``; doubling it gives the count the oracle enumerates.
    """
    ode = system(indices, conditions)
    basis = GaussianHermiteBasis(shape, 2)
    matrix = OperatorMatrix(basis, Liouville().create_operator(ode))
    return sp.expand(2 * matrix[[()], [()]][0, 0])


def oracle(indices, conditions):
    """The same quantity by enumerating the grid, sharing no code with the builder."""
    piecewise = system(indices, conditions).equation.expr
    total = 0
    for cell in itertools.product(range(N), repeat=len(indices)):
        site = cell if len(indices) > 1 else cell[0]
        total += sp.diff(sp.simplify(piecewise.subs(dict(zip(indices, cell)))), u[site])
    return sp.expand(-total)


def assert_counts(conditions, expected, indices=PLANE, shape=SHAPE):
    assert oracle(indices, conditions) == expected
    assert vac_element(indices, shape, conditions) == expected


class TestOneBox:
    """Regions whose condition is already a conjunction: the pre-existing path."""

    @pytest.mark.parametrize(
        "conditions, expected",
        [
            pytest.param([], -9 * a, id="no-conditions"),
            pytest.param(
                [DerivativeCondition(BOX, c * u[i, j])], -5 * a - 4 * c, id="square"
            ),
            pytest.param(
                [DerivativeCondition(Box({i: (1, 2), j: (0, 2)}), c * u[i, j])],
                -3 * a - 6 * c,
                id="not-square",
            ),
            pytest.param(
                [DerivativeCondition(BOX.boundary.walls[0], c * u[i, j])],
                -7 * a - 2 * c,
                id="one-wall",
            ),
            pytest.param(
                [DirichletCondition(Facet.low(i), 0)], -6 * a, id="dirichlet-face"
            ),
            pytest.param(
                [DirichletCondition(Facet.low(i) & Facet.low(j), 0)],
                -8 * a,
                id="dirichlet-corner",
            ),
            pytest.param(
                [
                    DerivativeCondition(Box({i: (0, 0), j: (0, 0)}), c * u[i, j]),
                    DerivativeCondition(
                        Box({i: (2, 2), j: (2, 2)}), sp.Symbol("d") * u[i, j]
                    ),
                ],
                -7 * a - c - sp.Symbol("d"),
                id="two-boxes-two-stencils-stay-separate-arms",
            ),
        ],
    )
    def test_it_counts(self, conditions, expected):
        assert_counts(conditions, expected)


class TestUnionOfBoxes:
    """Every case here raised ``NotImplementedError`` before ``_region_count``.

    Two sources of a union: ``Box.boundary`` is one by construction, and conditions
    sharing a value or a stencil merge into a single arm whose condition is theirs.
    """

    @pytest.mark.parametrize(
        "conditions, expected",
        [
            pytest.param(
                [DirichletCondition(BOX.boundary, 0)], -5 * a, id="shell-dirichlet"
            ),
            pytest.param(
                [DerivativeCondition(BOX.boundary, c * u[i, j])],
                -5 * a - 4 * c,
                id="shell-derivative",
            ),
            pytest.param(
                [
                    DerivativeCondition(
                        Box({i: (0, 2), j: (1, 2)}).boundary, c * u[i, j]
                    )
                ],
                -3 * a - 6 * c,
                id="shell-of-a-non-square-box",
            ),
            pytest.param(
                [
                    DirichletCondition(Facet.low(i), 0),
                    DirichletCondition(Facet.low(j), 0),
                ],
                -4 * a,
                id="two-faces-one-value",
            ),
            pytest.param(
                [
                    DirichletCondition(Facet.low(i) & Facet.low(j), 0),
                    DirichletCondition(Facet.high(i) & Facet.high(j), 0),
                ],
                -7 * a,
                id="two-corners-one-value",
            ),
            pytest.param(
                [
                    DerivativeCondition(BOX.boundary.walls[0], c * u[i, j]),
                    DerivativeCondition(BOX.boundary.walls[1], c * u[i, j]),
                ],
                -5 * a - 4 * c,
                id="two-walls-one-stencil",
            ),
            pytest.param(
                [
                    DerivativeCondition(Box({i: (0, 0), j: (0, 0)}), c * u[i, j]),
                    DerivativeCondition(Box({i: (2, 2), j: (2, 2)}), c * u[i, j]),
                ],
                -7 * a - 2 * c,
                id="two-boxes-one-stencil",
            ),
            pytest.param(
                [
                    DerivativeCondition(Box({i: (0, 0), j: (0, 0)}), c * u[i, j]),
                    DerivativeCondition(Box({i: (2, 2), j: (2, 2)}), c * u[i, j]),
                    DerivativeCondition(Box({i: (0, 0), j: (2, 2)}), c * u[i, j]),
                ],
                -6 * a - 3 * c,
                id="three-boxes-one-stencil",
            ),
            pytest.param(
                [
                    DirichletCondition(Box({i: (0, 2), j: (0, 2)}).boundary, 0),
                    DerivativeCondition(BOX, c * u[i, j]),
                ],
                -c,
                id="shell-pins-all-but-the-centre-which-a-box-owns",
            ),
        ],
    )
    def test_it_counts(self, conditions, expected):
        assert_counts(conditions, expected)

    def test_a_three_dimensional_shell_counts(self):
        """8 cells in the 2x2x2 box, all of them shell; 27 - 8 = 19 ambient."""
        shell = Box({i: (0, 1), j: (0, 1), k: (0, 1)}).boundary
        assert_counts(
            [DerivativeCondition(shell, c * u[i, j, k])],
            -19 * a - 8 * c,
            indices=(i, j, k),
            shape=(N, N, N),
        )

    def test_it_stays_n_independent(self):
        """A shell's count is closed-form in N, so a billion sites cost the same."""
        big = sp.Idx("p", (0, 10**9 - 1)), sp.Idx("q", (0, 10**9 - 1))
        p, q = big
        shell = Box({p: (1, 10**9 - 2), q: (1, 10**9 - 2)}).boundary
        ode = OrdinaryDifferentialSystem.from_sympy(
            a * u[p, q], u, big, conditions=[DerivativeCondition(shell, c * u[p, q])]
        )
        matrix = OperatorMatrix(
            GaussianHermiteBasis((10**9, 10**9), 2), Liouville().create_operator(ode)
        )
        shell_cells = (10**9 - 2) ** 2 - (10**9 - 4) ** 2
        expected = -(10**18 - shell_cells) * a - shell_cells * c
        assert sp.expand(2 * matrix[[()], [()]][0, 0]) == expected


def test_an_eliminated_arm_counts_across_the_hole_elimination_punches():
    """Eliminating both Dirichlet faces leaves the interior arm a disconnected per-axis
    support (``Ne(i, 0) & Ne(i, 2)``), two Ranges whose sizes add.

    The count is the region's, not the representation's, so the eliminated form must give
    the same vacuum element as the uneliminated one.
    """

    def vac_element_of(eliminate):
        ode = OrdinaryDifferentialSystem.from_sympy(
            a * (u[i - 1] - 2 * u[i] + u[i + 1]),  # type: ignore[operator]
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), 0),
                DirichletCondition(Facet.high(i), 0),
            ],
            eliminate=eliminate,
        )
        basis = GaussianHermiteBasis((N,), 2)
        matrix = OperatorMatrix(basis, Liouville().create_operator(ode))
        return sp.expand(2 * matrix[[()], [()]][0, 0])

    assert vac_element_of(eliminate=True) == vac_element_of(eliminate=False) == 2 * a


def test_a_region_coupling_two_free_axes_still_refuses():
    """A union of N boxes, not of a fixed number, so inclusion-exclusion would track the
    grid: still out of scope, and still a refusal rather than a wrong count."""
    ode = OrdinaryDifferentialSystem.from_sympy(
        sp.Piecewise((c * u[i, j], sp.Ne(i, j)), (a * u[i, j], True)), u, PLANE
    )
    basis = GaussianHermiteBasis(SHAPE, 2)
    matrix = OperatorMatrix(basis, Liouville().create_operator(ode))
    with pytest.raises(NotImplementedError, match="not an axis-aligned region"):
        matrix[[()], [()]]
