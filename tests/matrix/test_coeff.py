# pyright: reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false, reportOperatorIssue=false
# sympy stubs mistype Idx arithmetic and read Function subscripting as errors.
"""Unit tests for ``lin_ode/matrix/_coeff.py``.

``get_coeff`` reads a bra's coefficient out of an operator expansion, summing over the
index range. Reaching it takes two steps this module also owns: lowering a multi-axis
delta to scalar ones, and inverting a ``Mod``-wrapped index in closed form.

The last argument to ``get_coeff`` is the arm's ``owns`` condition, ``sp.true`` for a
lone case.
"""

import pytest
import sympy as sp

from lin_ode.basis import (
    AntiKroneckerDelta,
    CarlemanBasis,
    CarlemanState,
    GaussianHermiteBasis,
    GaussianHermiteState,
)
from lin_ode.basis.multi_delta import MultiAntiKroneckerDelta, MultiKroneckerDelta
from lin_ode.basis.op import K, U, apply_op, evaluate
from lin_ode.matrix._coeff import (
    _delta_trigger_points,
    _lower_multi_deltas,
    _solve_index_equation,
    _solve_mod,
    get_coeff,
)

from tests.matrix._util import (
    DECLINED,
    PIVOTS,
    PROBED,
    REGION_SHAPES,
    RIGHT_HAND_SIDES,
    resolve_case,
    ask_lin_ode,
    read_brute_force_entry,
    expand_over_sites,
)
from tests.testing import example

S = GaussianHermiteState
C = CarlemanState

IDX_3 = sp.Idx("i", (0, 2))
"""Index of a 3-site chain: the smallest grid with both an end site and an interior
one."""

IDX_8 = sp.Idx("i", (0, 7))
"""Index of an 8-site chain, where a Mod wrap has room to land on the wrong site."""

SITE, TARGET = sp.Tuple(IDX_8, 2), sp.Tuple(4, 2)
"""A 2-D site whose second component already matches, so only i == 4 is left."""


class TestConcreteExpansion:
    """No index to solve for: the read-off degenerates to matching terms against the
    bra."""

    def test_picks_each_component_of_linear_combination(self) -> None:
        """Each bra reads its own factor, and an absent bra reads 0."""
        expr = 2 * S((0, 1)) + 3 * S((1, 2)) + 5 * S()
        assert get_coeff(expr, S((0, 1)), (), (), sp.true) == 2
        assert get_coeff(expr, S((1, 2)), (), (), sp.true) == 3
        assert get_coeff(expr, S(), (), (), sp.true) == 5  # the vacuum bra
        assert get_coeff(expr, S((2, 1)), (), (), sp.true) == 0  # absent from the sum

    def test_splits_operator_expansion_by_bra(self) -> None:
        """``U_0 |psi_1>`` gives 1 on ``<psi_2|`` and ``sqrt(1/2)`` on the vacuum."""
        # U_0|psi_1> = |psi_2> + sqrt(1/2)|psi_0>; each component is picked out by its bra.
        res = evaluate(apply_op(U[0], S((0, 1))), GaussianHermiteBasis(3, 2).registry)
        assert get_coeff(res, S((0, 2)), (), (), sp.true) == 1
        ground = get_coeff(res, S(), (), (), sp.true)
        assert sp.simplify(ground - sp.sqrt(sp.Rational(1, 2))) == 0

    def test_needs_basis_but_not_orthonormality(self) -> None:
        """``K_0 U_0 |u_0> = 2 |u_0>`` reads 2 on a non-orthonormal basis."""
        # Carleman monomials are not orthonormal: K_0 U_0 |u_0> = 2|u_0>, so O_{u0,u0} = 2.
        res = evaluate(apply_op(K[0] * U[0], C((0, 1))), CarlemanBasis(3, 2).registry)
        assert get_coeff(res, C((0, 1)), (), (), sp.true) == 2
        assert get_coeff(res, C((0, 2)), (), (), sp.true) == 0


class TestOverTheIndexBox:
    """A symbolic index: the contributing values are solved for, then summed."""

    def test_sums_sites_that_reach_bra(self) -> None:
        """One site reaches each bra, so the sum keeps one term."""
        # sum_i U_i |psi_1>: only i=0 reaches |psi_2>, only i=1 reaches |psi_0 psi_1>.
        res = evaluate(
            apply_op(U[IDX_3], S((0, 1))), GaussianHermiteBasis(3, 2).registry
        )
        assert get_coeff(res, S((0, 2)), (IDX_3,), ((0, 2),), sp.true) == 1
        off_site = get_coeff(res, S((0, 1), (1, 1)), (IDX_3,), ((0, 2),), sp.true)
        assert sp.simplify(off_site - sp.sqrt(sp.Rational(1, 2))) == 0

    def test_solves_grid_sites_componentwise(self) -> None:
        """Site ``(0, 1)`` reads 1 and site ``(1, 1)`` reads 0 on the 2x2 grid."""
        # D-dim sites are sp.Tuple labels; each axis is solved separately.
        j, k = sp.Idx("j", (0, 1)), sp.Idx("k", (0, 1))
        res = evaluate(
            apply_op(U[j, k] * U[j, k] * K[j, k], C((sp.Tuple(0, 1), 1))),
            CarlemanBasis((2, 2), 2).registry,
        )
        bounds = ((0, 1), (0, 1))
        assert get_coeff(res, C((sp.Tuple(0, 1), 2)), (j, k), bounds, sp.true) == 1
        assert get_coeff(res, C((sp.Tuple(1, 1), 2)), (j, k), bounds, sp.true) == 0

    def test_solving_contributing_index_is_n_independent(self) -> None:
        """A range of ``10**9`` sites returns 1 without enumerating the range."""
        # A billion-site range resolves at once: the contributing index is solved, not scanned.
        big = 10**9
        i = sp.Idx("i", (0, big - 1))
        res = evaluate(
            apply_op(U[i] * U[i] * K[i], C((3, 1))), CarlemanBasis(big, 2).registry
        )
        assert get_coeff(res, C((3, 2)), (i,), ((0, big - 1),), sp.true) == 1


class TestIndexFreeTerm:
    """A term without the index is constant along that axis, so it contributes at every
    value the arm owns. It is counted, never enumerated.

    Lowering ``AntiKroneckerDelta(i, 0)`` to ``1 - KroneckerDelta(i, 0)`` leaves such a
    term: the ``1`` carries no index at all.
    """

    def _expansion(self, i: sp.Idx, n_site: int) -> sp.Expr:
        # K_i U_i |u_0> = |u_0> for i != 0 and 2|u_0> at i = 0: the linear RHS du/dt = u
        # under Liouville (KU) ordering. Over the range: (N - 1) + 2.
        return evaluate(
            apply_op(K[i] * U[i], C((0, 1))), CarlemanBasis(n_site, 2).registry
        )

    @pytest.mark.parametrize("n_site", [3, 10**9], ids=["small", "n-independent"])
    def test_constant_term_is_counted_over_whole_box(self, n_site: int) -> None:
        """``(n_site - 1)`` sites contribute 1 and site 0 contributes 2.

        The total is ``n_site + 1`` however large the range, since it is counted, never scanned.
        """
        i = sp.Idx("i", (0, n_site - 1))
        res = self._expansion(i, n_site)
        bounds = ((0, n_site - 1),)
        assert get_coeff(res, C((0, 1)), (i,), bounds, sp.true) == n_site + 1

    def test_arm_condition_restricts_count(self) -> None:
        """The arm ``i != 0`` counts sites 1 and 2, giving 2 instead of 4."""
        # An interior arm owns every site but the pinned one. The constant term is
        # counted over the range minus that site: sites 1 and 2 give 1 each.
        res = self._expansion(IDX_3, 3)
        interior = sp.Not(sp.Eq(IDX_3, 0))
        assert get_coeff(res, C((0, 1)), (IDX_3,), ((0, 2),), interior) == 2


class TestArmCondition:
    """Piecewise first-match: an arm owns the tuples where its condition holds and no
    earlier arm's does (the builder passes ``And(cond, *[Not(earlier)])``)."""

    def _expansion(self) -> sp.Expr:
        return evaluate(
            apply_op(U[IDX_3], S((0, 1))), GaussianHermiteBasis(3, 2).registry
        )

    def test_arm_keeps_only_indices_it_governs(self) -> None:
        """The arm ``i = 1`` keeps the contributor, and the arm ``i = 2`` reads 0."""
        res, target = self._expansion(), S((0, 1), (1, 1))
        kept = get_coeff(
            res, target, (IDX_3,), ((0, 2),), sp.Eq(IDX_3, 1)
        )  # i=1 is the contributor
        assert sp.simplify(kept - sp.sqrt(sp.Rational(1, 2))) == 0
        assert get_coeff(res, target, (IDX_3,), ((0, 2),), sp.Eq(IDX_3, 2)) == 0

    def test_fallthrough_excludes_earlier_arms(self) -> None:
        """An earlier arm owns ``i = 1``, so the fallthrough arm reads 0."""
        res, target = self._expansion(), S((0, 1), (1, 1))
        interior = sp.And(
            sp.true, sp.Not(sp.Eq(IDX_3, 1))
        )  # i=1 taken by an earlier arm
        assert get_coeff(res, target, (IDX_3,), ((0, 2),), interior) == 0


class TestDeltaLowering:
    """Every multi-site delta rewritten as plain ``KroneckerDelta`` factors."""

    @pytest.mark.parametrize(
        "delta, expected",
        [
            (AntiKroneckerDelta(IDX_8, 3), 1 - sp.KroneckerDelta(IDX_8, 3)),
            (MultiKroneckerDelta(SITE, TARGET), sp.KroneckerDelta(IDX_8, 4)),
            (MultiAntiKroneckerDelta(SITE, TARGET), 1 - sp.KroneckerDelta(IDX_8, 4)),
        ],
        ids=["anti", "multi", "multi-anti"],
    )
    def test_lowers_to_plain_delta(self, delta: sp.Expr, expected: sp.Expr) -> None:
        """An anti delta lowers to ``1 - delta`` and a multi delta to one factor."""
        assert _lower_multi_deltas(delta) == expected

    def test_trigger_point_is_where_lowered_term_survives(self) -> None:
        """Index 3 is the only trigger point, and the term there equals -1."""
        terms = sp.Add.make_args(_lower_multi_deltas(AntiKroneckerDelta(IDX_8, 3)))
        delta_term = next(t for t in terms if t.has(sp.KroneckerDelta))
        assert _delta_trigger_points(delta_term, IDX_8) == frozenset({3})
        assert (
            delta_term.subs(IDX_8, 3) == -1
        )  # nonzero there, so the candidate is a real one


class TestModInversion:
    """The one index a ``Mod`` wrap fires at, or ``None`` when there is not one."""

    @pytest.mark.parametrize(
        "lhs, rhs, expected",
        [
            (sp.Mod(IDX_8 - 1, 8), 0, 1),  # left neighbour, interior:  Mod(1-1, 8) = 0
            (sp.Mod(IDX_8 - 1, 8), 7, 0),  # left neighbour, THE WRAP:  Mod(0-1, 8) = 7
            (sp.Mod(IDX_8 + 1, 8), 5, 4),  # right neighbour, interior: Mod(4+1, 8) = 5
            (sp.Mod(IDX_8 + 1, 8), 0, 7),  # right neighbour, THE WRAP: Mod(7+1, 8) = 0
        ],
        ids=["left-interior", "left-wrap", "right-interior", "right-wrap"],
    )
    def test_solves_single_firing_index(
        self, lhs: sp.Expr, rhs: sp.Expr, expected: sp.Expr
    ) -> None:
        """The returned index reproduces the right-hand side under substitution."""
        resolved = _solve_mod(lhs, sp.Integer(rhs), IDX_8)
        assert resolved == expected
        assert lhs.subs(IDX_8, resolved) == rhs  # the wrap really does land there

    def test_solves_with_nonzero_lower_bound(self) -> None:
        """With a lower bound of 1, ``rhs = 8`` solves to ``j = 1``."""
        # lower = 1, size = 8: u[j-1] wraps as 1 + Mod(j-2, 8)
        j = sp.Idx("j", (1, 8))
        expr = 1 + sp.Mod(j - 2, 8)
        assert _solve_mod(expr, sp.Integer(2), j) == 3  # interior: 1 + Mod(3-2, 8) = 2
        assert (
            _solve_mod(expr, sp.Integer(8), j) == 1
        )  # wrap: j=1 reads 1 + Mod(-1, 8) = 8

    UNBOUNDED = sp.Idx("m")
    """No range at all, so ``idx.lower is None``."""

    @pytest.mark.parametrize(
        "lhs, rhs, idx",
        [
            (sp.Mod(IDX_8 - 1, sp.Symbol("N", integer=True, positive=True)), 3, IDX_8),
            (sp.Mod(2 * IDX_8 - 1, 8), 3, IDX_8),
            (IDX_8 - 1, 3, IDX_8),
            (sp.Mod(IDX_8 - 1, 8), sp.Idx("k", (0, 7)), IDX_8),
            (sp.Mod(IDX_8 - 1, 8) + sp.Mod(IDX_8 + 1, 8), 3, IDX_8),
            (sp.Mod(IDX_8 - 1, 8) * sp.Mod(IDX_8 + 1, 8), 3, IDX_8),
            (2 + sp.Mod(IDX_8 - 1, 8), sp.Integer(0), IDX_8),
            (sp.Mod(UNBOUNDED - 1, 8), sp.Integer(0), UNBOUNDED),
        ],
        ids=[
            "symbolic-modulus",
            "non-unit-stride",
            "no-mod",
            "symbolic-rhs",
            "two-mods-added",
            "two-mods-multiplied",
            "wrap-never-fires",
            "missing-lower-bound",
        ],
    )
    def test_declines_rather_than_guessing(
        self, lhs: sp.Expr, rhs: sp.Expr, idx: sp.Idx
    ) -> None:
        """Every condition under which there is no single contributing index to return.

        The first six are the wrong *form*; ``wrap-never-fires`` has the right form but
        ranges over [2, 9] so rhs = 0 is unreachable; the last has no bounds to wrap within.
        """
        assert _solve_mod(lhs, rhs, idx) is None

    def test_inversion_is_n_independent(self) -> None:
        """A modulus of ``10**9`` inverts at once, with wrap at ``10**9 - 1``."""
        # A billion-site modulus resolves instantly: the range is never enumerated.
        big = sp.Idx("i", (0, 10**9 - 1))
        assert _solve_mod(sp.Mod(big - 1, 10**9), sp.Integer(0), big) == 1
        assert (
            _solve_mod(sp.Mod(big + 1, 10**9), sp.Integer(0), big) == 10**9 - 1
        )  # wrap

    @pytest.mark.parametrize(
        "lhs, rhs",
        [
            (
                sp.Mod(IDX_8 - 1, 8),
                sp.Idx("k", (0, 7)),
            ),  # symbolic rhs (cross-index delta)
            (
                sp.Mod(IDX_8 - 1, sp.Symbol("N", integer=True, positive=True)),
                sp.Integer(3),
            ),
            (IDX_8 - 1, sp.Idx("k", (0, 7))),  # the affine path needs the same guard
        ],
        ids=["symbolic-rhs", "symbolic-modulus", "affine-with-symbolic-rhs"],
    )
    def test_dispatcher_declines_instead_of_reaching_int(
        self, lhs: sp.Expr, rhs: sp.Expr
    ) -> None:
        """``_solve_index_equation`` routes to the Mod or the affine arm; either way an
        operand that is symbolic-but-integer must return None rather than raise on
        int()."""
        assert _solve_index_equation(lhs, rhs, IDX_8) is None

    def test_dispatcher_routes_concrete_mod_to_solver(self) -> None:
        """``i = 4`` is the only index in ``[0, 7]`` with ``Mod(i - 1, 8) = 3``."""
        # Mod(4 - 1, 8) = 3, and i = 4 is the only index in [0, 7] that lands there.
        assert _solve_index_equation(sp.Mod(IDX_8 - 1, 8), sp.Integer(3), IDX_8) == 4


class TestRegionCoefficient:
    """One region, one right-hand side, every pivot: exact or declined."""

    @example(spec_ref=["C-5", "B-5-2", "C-2", "D-1"], level="advanced")
    @pytest.mark.parametrize(
        "region", sorted(REGION_SHAPES), ids=lambda name: name.replace(" ", "-")
    )
    @pytest.mark.parametrize(
        "rhs", sorted(RIGHT_HAND_SIDES), ids=lambda name: name.split()[0]
    )
    def test_region_coefficient_is_exact_or_declines(
        self, region: str, rhs: str
    ) -> None:
        """Exact, or declined.

        For every pivot: the brute force's answer, or ``NotImplementedError``. Silence
        is what this excludes. An arm that cannot be counted has to say so. It must not
        hand back an expression that reads plausibly and is short a term. Koopman is the
        only method here, since every pivot in ``PIVOTS`` is a non-L2 Carleman basis.
        """
        compared = 0
        for pivot in PIVOTS:
            case = resolve_case(region, rhs, pivot, "Koopman")
            lin_ode_entries = ask_lin_ode(case, PROBED)
            if lin_ode_entries is DECLINED:
                continue
            for (bra, ket), lin_ode_entry in lin_ode_entries.items():
                expected = read_brute_force_entry(case, bra, ket)
                assert expand_over_sites(lin_ode_entry) == expand_over_sites(
                    expected
                ), (
                    f"{case.name} <{bra}|L|{ket}>: "
                    f"{expand_over_sites(lin_ode_entry)} != {expand_over_sites(expected)}"
                )
                compared += 1
        # declining is a supported answer, but declining *everything* would leave nothing
        # checked and still read as a pass
        assert compared, f"{region}/{rhs}: all declined"
