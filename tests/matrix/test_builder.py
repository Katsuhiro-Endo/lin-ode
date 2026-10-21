# pyright: reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false
# sympy stubs mistype .subs/.replace and the block as having no __getitem__.
"""Unit tests for ``lin_ode/matrix/builder.py``.

The builder turns an operator and a basis into matrix entries. It solves for the
contributing index rather than scanning the range, which is what keeps an entry
independent of the grid size, and it is where an arm of a ``Piecewise`` is selected.

``brute_force`` writes the right-hand side out site by site instead, so agreement pins
the solving against an enumeration. Each test below names the part it reaches.
"""

from collections.abc import Callable


import sympy as sp

from lin_ode.basis import (
    CarlemanBasis,
    GaussianHermiteBasis,
    GaussianHermiteState,
    PSCarlemanBasis,
)
from lin_ode.matrix import OperatorMatrix

from tests.util import brute_force
from tests.util.comparison import list_basis_labels, s_pivot
from tests.util import METHODS
from tests.util.rhs_at_sites import write_out_per_site
from tests.util.equations import (
    _unrolled_periodic_burgers,
    burgers_equation,
    corner_and_edges_equation_2d,
    corner_equation_2d,
    quadratic_equation,
    quadratic_equation_2d,
    stencil_2d_equation,
    stencil_2d_equation_mod,
    stride_equation_mod,
)
from tests.testing import example
from lin_ode.basis import TruncatedBasisSet
from lin_ode.ordinary_differential_system import OrdinaryDifferentialSystem


def _read_brute_force_column(
    method: str,
    ket: tuple,
    bras: list,
    basis: TruncatedBasisSet,
    f: dict,
    sym: dict,
    pivot: Callable | None,
) -> list:
    """The brute force's answer for one column, one entry per bra."""
    # Gauss-Hermite has no monomial to read a coefficient off
    if basis.state_type is GaussianHermiteState:
        return [
            brute_force.read_hermite_coefficient(method, bra, ket, f, sym)
            for bra in bras
        ]
    poly = brute_force.build_column(
        method, ket, f, sym, degree=basis.degree, pivot=pivot
    )
    return [brute_force.read_coefficient(poly, bra, sym) for bra in bras]


def assert_every_entry_equals_the_brute_force(
    ode: OrdinaryDifferentialSystem,
    basis: TruncatedBasisSet,
    method: str,
    pivot: Callable | None = None,
) -> None:
    """Compare the whole matrix, entry by entry, to the same one written out per site.

    Bases stay small on purpose: each call is ``len(basis)**2`` entries.
    """
    labels = list_basis_labels(basis)
    op = METHODS[method].create_operator(ode)
    block = OperatorMatrix(basis, op)[labels, labels]
    f, sym = write_out_per_site(ode)

    assert block.shape == (len(basis), len(basis))
    for col, ket in enumerate(labels):
        expected = _read_brute_force_column(method, ket, labels, basis, f, sym, pivot)
        for row, (bra, want) in enumerate(zip(labels, expected)):
            assert sp.simplify(block[row, col] - want) == 0, f"<{bra}| L |{ket}>"


class TestOneDimensionalChains:
    """Right-hand sides on a chain, each reaching a part of the solving the others
    miss."""

    def test_periodic_stencil(self) -> None:
        """Burgers on a ring: the wrap and the interior stencil in one right-hand
        side."""
        assert_every_entry_equals_the_brute_force(
            burgers_equation(4), CarlemanBasis(4, 2), "Koopman"
        )

    def test_same_site_double_action(self) -> None:
        """``du/dt = u**2`` acts twice at one site, where ``Sum.doit()`` recurses unpinned."""
        assert_every_entry_equals_the_brute_force(
            quadratic_equation(3), CarlemanBasis(3, 2), "Koopman"
        )

    def test_same_site_double_action_at_degree_3(self) -> None:
        """The same action deeper: out-of-range dropping, and a collapse with no
        stencil."""
        assert_every_entry_equals_the_brute_force(
            quadratic_equation(3), CarlemanBasis(3, 3), "Koopman"
        )

    def test_orthonormal_basis_collapses_through_the_bra(self) -> None:
        """Gauss-Hermite has ``K|0> != 0``, so no term is annihilated into place."""
        assert_every_entry_equals_the_brute_force(
            quadratic_equation(3), GaussianHermiteBasis(3, 2), "Koopman"
        )

    def test_liouville_raises_an_unoccupied_site_then_lowers_it(self) -> None:
        """``-K_i f_i(U)`` applies ``f`` first, which a padded builder gets wrong.

        Gauss-Hermite, since Liouville needs an L2 basis.
        """
        assert_every_entry_equals_the_brute_force(
            quadratic_equation(3), GaussianHermiteBasis(3, 2), "Liouville"
        )

    def test_kvn_sums_both_orderings_at_one_site(self) -> None:
        """``-{f, K}/2`` fires the same-site pin in both directions at once."""
        assert_every_entry_equals_the_brute_force(
            quadratic_equation(3), GaussianHermiteBasis(3, 2), "KvN"
        )

    def test_non_unit_stride_wrap(self) -> None:
        """``Mod(2*i - 1, n)`` cannot be pinned by the site map, so the ``K_i`` delta must."""
        assert_every_entry_equals_the_brute_force(
            stride_equation_mod(4), CarlemanBasis(4, 2), "Koopman"
        )


class TestGrids:
    """The same solving on a 2-D grid, where 2-D stands in for any D."""

    def test_smallest_grid_with_no_boundary(self) -> None:
        """``u[i,j]**2`` on a 2x2 grid: every axis solved, no arm to select."""
        assert_every_entry_equals_the_brute_force(
            quadratic_equation_2d(2), CarlemanBasis((2, 2), 2), "Koopman"
        )

    def test_psc_folds_states_back_into_the_basis(self) -> None:
        """A basis that folds rather than drops; without ``truncation_rule`` every
        folded entry reads 0."""
        assert_every_entry_equals_the_brute_force(
            quadratic_equation_2d(2),
            PSCarlemanBasis((2, 2), 2, s_pivot),
            "Koopman",
            lambda _: s_pivot,
        )

    def test_boundary_face_against_the_interior(self) -> None:
        """Site (0,0) against (1,0): the per-arm filter, and its interior
        fallthrough."""
        assert_every_entry_equals_the_brute_force(
            stencil_2d_equation(2), CarlemanBasis((2, 2), 2), "Koopman"
        )

    def test_corner_arm_couples_both_axes(self) -> None:
        """``~(Eq(i,0) & Eq(j,0))`` is the one condition over both axes, taken by
        complement."""
        assert_every_entry_equals_the_brute_force(
            corner_equation_2d(2), CarlemanBasis((2, 2), 2), "Koopman"
        )

    def test_four_arms_mix_positive_and_negated_regions(self) -> None:
        """The shape ordinary boundary conditions produce, which a per-axis count
        misses."""
        assert_every_entry_equals_the_brute_force(
            corner_and_edges_equation_2d(2), CarlemanBasis((2, 2), 2), "Koopman"
        )

    def test_wrap_on_one_axis_only(self) -> None:
        """The ``Mod`` axis is inverted, the bare axis solved affinely, in one
        equation."""
        assert_every_entry_equals_the_brute_force(
            stencil_2d_equation_mod(2), CarlemanBasis((2, 2), 2), "Koopman"
        )


class TestPeriodicWrap:
    """Inverting a ``Mod``-wrapped index.

    sp.solve has no Mod algorithm, and a scan
    over the range would not be N-independent.
    """

    def test_mod_wrap_equals_periodicity_arm_by_arm(self) -> None:
        """The wrapped and unrolled Burgers matrices agree in all 15 by 15 entries."""
        # The Mod wrap must equal periodicity spelled out arm by arm: pins the
        # inversion against an independent enumeration.
        n_site, degree = 4, 2
        basis = CarlemanBasis(n_site, degree)
        labels = list_basis_labels(basis)
        wrapped = METHODS["Koopman"].create_operator(burgers_equation(n_site))
        unrolled = METHODS["Koopman"].create_operator(
            _unrolled_periodic_burgers(n_site)
        )
        lin_ode_wrapped = OperatorMatrix(basis, wrapped)[labels, labels]
        lin_ode_unrolled = OperatorMatrix(basis, unrolled)[labels, labels]

        assert len(labels) == len(basis) == 15  # else the sweep below is vacuous
        for a in range(len(labels)):
            for b in range(len(labels)):
                assert (
                    sp.simplify(lin_ode_wrapped[a, b] - lin_ode_unrolled[a, b]) == 0
                ), (
                    labels[a],
                    labels[b],
                )

    @example(spec_ref=["A-2", "C-2", "D-1"], level="advanced")
    def test_on_grid_axis_is_n_independent(self) -> None:
        """A wrap on one grid axis

        Cost independent of N with a periodic wrap: inverting ``Mod`` is O(1).
        ``<u_(0,5) u_(1,5)|L|u_(1,5)> = 1`` on a 10**6 x 10**6 torus resolves at once, since
        ``i=1`` gives ``Mod(0, N) = 0`` by solving, not scanning. Sites stay small because a
        far-corner site costs O(flat index) in ``k_to_state``, unrelated to ``Mod``.
        """
        big = 1_000_000
        op = METHODS["Koopman"].create_operator(stencil_2d_equation_mod(big))
        lin_ode_matrix = OperatorMatrix(CarlemanBasis((big, big), 2), op)
        bra = ((sp.Tuple(0, 5), 1), (sp.Tuple(1, 5), 1))
        ket = ((sp.Tuple(1, 5), 1),)
        assert lin_ode_matrix[bra, ket] == 1
