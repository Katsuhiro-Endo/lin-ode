# pyright: reportIndexIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false, reportAttributeAccessIssue=false, reportReturnType=false
# sympy stubs mistype indexing, Idx, .args and plain-tuple site labels here; all noise.
# OperatorMatrix's overloads do not narrow a plain bra tuple to the single element
# ``_element`` asks for either.
"""PSC Carleman basis (arXiv:2411.16233): same states and ladder rules as Carleman,
pivot enters only through ``truncation_rule``."""

from typing import Callable

import pytest
import sympy as sp

from lin_ode.basis import BasisFunction, CarlemanState, PSCarlemanBasis
from lin_ode.basis.op import K, U, apply_op, evaluate
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import Koopman
from lin_ode.ordinary_differential_system import OrdinaryDifferentialSystem
from tests.util.comparison import list_basis_labels, s_pivot, s_site

PIVOT = sp.Rational(3, 2)


@pytest.mark.parametrize("n", range(5))
def test_u_raises_power_at_site_by_one(n: int) -> None:
    """PSC registers the plain Carleman ladder rules; truncation is a separate step."""
    registry = PSCarlemanBasis(1, 5, PIVOT).registry
    lin_ode_rule = evaluate(apply_op(U[0], CarlemanState((0, n))), registry)
    assert lin_ode_rule == CarlemanState((0, n + 1))


@pytest.mark.parametrize("n", range(1, 5))
def test_k_lowers_power_and_scales_by_old_power(n: int) -> None:
    """PSC reuses the plain Carleman lowering rule, K|n> = n|n-1>."""
    registry = PSCarlemanBasis(1, 5, PIVOT).registry
    lin_ode_rule = evaluate(apply_op(K[0], CarlemanState((0, n))), registry)
    assert lin_ode_rule == n * CarlemanState((0, n - 1))


def _ket_to_poly(expr: sp.Expr, u: sp.Symbol, site: int = 0) -> sp.Expr:
    """Single-site combination of CarlemanStates as a polynomial in ``u``."""
    poly = sp.S.Zero
    for term in sp.expand(expr).as_ordered_terms():
        coeff, ket = term.as_independent(CarlemanState)
        poly += coeff * u ** sum(n for s_pivot, n in ket.args if s_pivot == site)
    return poly


def _taylor_polynomial(m: int, degree: int, s_pivot: sp.Expr, site: int = 0) -> sp.Expr:
    """Degree-``degree`` Taylor polynomial of ``u^m`` about ``s_pivot``, on basis."""
    u = sp.Symbol("_u")
    poly = sp.Poly(
        sum(
            sp.diff(u**m, u, j).subs(u, s_pivot) / sp.factorial(j) * (u - s_pivot) ** j
            for j in range(degree + 1)
        ),
        u,
    )
    return sp.Add(
        *(
            coeff * CarlemanState(*([(site, b)] if b else []))
            for (b,), coeff in poly.as_dict().items()
        )
    )


def _max_degree_of_state(expr: sp.Expr) -> int:
    """Highest total power degree in a Carleman state sum."""
    return max(
        sum(n for _, n in term.as_independent(CarlemanState)[1].args)
        for term in sp.expand(expr).as_ordered_terms()
    )


@pytest.mark.parametrize("n", range(4))
def test_state_within_degree_is_returned_unchanged(n: int) -> None:
    """Truncation acts only above the degree, so the pivot never enters here."""
    basis = PSCarlemanBasis(1, 3, PIVOT)
    assert basis.truncation_rule((0, n)) == CarlemanState((0, n))


@pytest.mark.parametrize(
    "m,degree",
    [(3, 2), (4, 2), (5, 3), (6, 3), (7, 3)],
    ids=["u3-in-K2", "u4-in-K2", "u5-in-K3", "u6-in-K3", "u7-in-K3"],
)
def test_out_of_basis_state_becomes_taylor_polynomial(m: int, degree: int) -> None:
    """Covers ``m`` past ``degree + 1``, where the single ``(u - s_pivot)^m = 0`` relation
    stops agreeing with the Taylor projection."""
    basis = PSCarlemanBasis(1, degree, PIVOT)
    lin_ode_rule = basis.truncation_rule((0, m))
    assert sp.expand(lin_ode_rule - _taylor_polynomial(m, degree, PIVOT)) == 0
    assert _max_degree_of_state(lin_ode_rule) <= degree


def test_residual_is_pivot_relation() -> None:
    """At ``m = degree + 1`` the Taylor remainder is the single relation ``(u - s_pivot)^m``.

    That is why this one case alone cannot tell the two rules apart.
    """
    u, degree = sp.Symbol("u"), 3
    basis = PSCarlemanBasis(1, degree, PIVOT)
    lin_ode_fold = _ket_to_poly(basis.truncation_rule((0, degree + 1)), u)
    assert sp.expand(u ** (degree + 1) - lin_ode_fold) == sp.expand(
        (u - PIVOT) ** (degree + 1)
    )


def test_multi_site_residual_is_order_degree_plus_one_in_shift() -> None:
    """The discarded part of a multi-site fold vanishes to order ``degree + 1`` at the
    pivot.

    The whole approximation rests on that property.
    """
    u0, u1, s_pivot, eps = sp.symbols("u0 u1 s_pivot epsilon")
    basis = PSCarlemanBasis(2, 2, s_pivot)
    lin_ode_fold = sp.expand(basis.truncation_rule((0, 2), (1, 2))).replace(
        CarlemanState,
        lambda *f: sp.Mul(*[(u0, u1)[site] ** n for site, n in f]),
    )
    residual = (u0**2 * u1**2 - lin_ode_fold).subs(
        {u0: s_pivot + eps, u1: s_pivot + eps}
    )
    assert sp.expand(residual) != 0
    assert sp.series(residual, eps, 0, 3).removeO() == 0


@pytest.mark.parametrize(
    "pivot,s0,s1",
    [(s_pivot, s_pivot, s_pivot), (s_site, s_site[0], s_site[1])],
    ids=["shared", "per-site"],
)
def test_multi_site_folds_product(pivot: sp.Expr, s0: sp.Expr, s1: sp.Expr) -> None:
    """``u_0**2 u_1`` at degree 2 folds site-wise, keeping ``|beta| <= 2``.

    Distinct per-site pivots tell the two sites' weights apart: a rule that read one
    site's pivot for the other would still match the shared case.
    """
    basis = PSCarlemanBasis(2, 2, pivot)
    lin_ode_rule = basis.truncation_rule((0, 2), (1, 1))
    expected = (
        s0**2 * s1 * CarlemanState()
        - 2 * s0 * s1 * CarlemanState((0, 1))
        - s0**2 * CarlemanState((1, 1))
        + s1 * CarlemanState((0, 2))
        + 2 * s0 * CarlemanState((0, 1), (1, 1))
    )
    assert sp.expand(lin_ode_rule - expected) == 0
    assert _max_degree_of_state(lin_ode_rule) <= 2


def test_symbolic_pivot_reduces_to_plain_truncation_at_zero() -> None:
    """``s_pivot = 0`` collapses the Taylor projection to dropping the state, i.e. plain
    Carleman."""
    lin_ode_rule = PSCarlemanBasis(1, 2, s_pivot).truncation_rule((0, 4))
    assert sp.expand(lin_ode_rule.subs(s_pivot, 0)) == 0


def test_symbolic_power_is_out_of_contract_and_fails_loudly() -> None:
    """``truncation_rule`` branches on ``sum(powers) <= degree`` with a plain ``if``, so
    a power must be a concrete integer; what goes symbolic is the *site* label."""
    m = sp.Symbol("m", integer=True, positive=True)
    with pytest.raises(TypeError, match="truth value of Relational"):
        PSCarlemanBasis(1, 2, PIVOT).truncation_rule((0, m))


def test_ladder_only_ever_produces_concrete_powers() -> None:
    """Acting at a symbolic site on a concrete ket: the new site label carries ``i``.

    Its power does not, which is why the branch above is always a real bool.
    """
    i = sp.Idx("i", (0, 5))
    registry = PSCarlemanBasis(6, 2, PIVOT).registry
    word = K[i] * U[i] * U[i]
    states = evaluate(apply_op(word, CarlemanState((0, 1), (1, 1))), registry).atoms(
        CarlemanState
    )
    factors = [f for state in states for f in state.args]
    assert any(site.free_symbols for site, _ in factors)  # else vacuously true
    assert all(power.is_Integer for _, power in factors)


def test_logistic_generator_at_degree_3_matches_paper_eq_16() -> None:
    """The one PSC generator arXiv:2411.16233 writes out in full: ``dx/dt = x(1 - x)``
    at ``P = 3``, symbolic pivot. The paper writes ``dy/dt = A y``, acting on the state
    vector, so ``A`` is the transpose of ``<bra| L |ket>``."""
    # A quadratic right-hand side never reaches past m = degree + 1. There the two
    # rules agree, so this case cannot tell them apart. The truncation_rule tests
    # above cover the rest.
    x, i = sp.IndexedBase("x"), sp.Idx("i", (0, 0))
    ode = OrdinaryDifferentialSystem.from_sympy(x[i] - x[i] ** 2, x, (i,))
    basis = PSCarlemanBasis(1, 3, s_pivot)
    args = list_basis_labels(basis)

    lin_ode_block = OperatorMatrix(basis, Koopman().create_operator(ode))

    expected = sp.Matrix(
        [
            [0, 0, 0, 0],
            [0, 1, -1, 0],
            [0, 0, 2, -2],
            [3 * s_pivot**4, -12 * s_pivot**3, 18 * s_pivot**2, 3 - 12 * s_pivot],
        ]
    )
    assert sp.simplify(lin_ode_block[args, args].T - expected) == sp.zeros(4, 4)


def _grid_quadratic_ode() -> OrdinaryDifferentialSystem:
    """``du_{i,j}/dt = u_{i,j}**2`` on a 2x2 grid: boundary-free, two index symbols."""
    u = sp.IndexedBase("u")
    i, j = sp.Idx("i", (0, 1)), sp.Idx("j", (0, 1))
    return OrdinaryDifferentialSystem.from_sympy(u[i, j] ** 2, u, (i, j))


class TestPSCMatrixElements:
    """PSC matrix entries of ``du_{i,j}/dt = u_{i,j}**2`` on a 2x2 grid."""

    GRID, DEGREE = (2, 2), 2
    SITE = [sp.Tuple(a, b) for a in range(2) for b in range(2)]  # 2x2 grid, row-major
    BASIS = PSCarlemanBasis(GRID, DEGREE, s_pivot)
    BRAS = list_basis_labels(BASIS)

    OPERATOR = Koopman().create_operator(_grid_quadratic_ode())
    MATRICES = {
        "shared": (OperatorMatrix(BASIS, OPERATOR), lambda _: s_pivot),
        "per-site": (
            OperatorMatrix(PSCarlemanBasis(GRID, DEGREE, s_site), OPERATOR),
            lambda site: s_site[tuple(site)],
        ),
    }

    def _powers(self, state: BasisFunction) -> dict[sp.Tuple, int]:
        """``{site: power}`` of a basis function; absent sites are power 0."""
        return {site: int(n) for site, n in state.args}

    def _fold_weight(
        self,
        m: dict[sp.Tuple, int],
        beta: dict[sp.Tuple, int],
        pivot: Callable[[sp.Tuple], sp.Expr],
    ) -> sp.Expr:
        """Coefficient of ``y_beta`` in the PSC image of ``y_m``, for the only case this
        operator reaches: ``|m| = DEGREE + 1``, where the Taylor projection equals
        ``-prod_q C(m_q, beta_q) (-s_q)^(m_q - beta_q)``. In basis, it is the identity.
        """
        if sum(m.values()) <= self.DEGREE:
            return sp.S.One if beta == m else sp.S.Zero
        if sum(beta.values()) > self.DEGREE or not beta.keys() <= m.keys():
            return sp.S.Zero
        weight = -sp.S.One
        for site, m_q in m.items():
            beta_q = beta.get(site, 0)
            weight *= sp.binomial(m_q, beta_q) * (-pivot(site)) ** (m_q - beta_q)
        return weight

    def _theory_element(
        self,
        alpha: dict[sp.Tuple, int],
        beta: dict[sp.Tuple, int],
        pivot: Callable[[sp.Tuple], sp.Expr],
    ) -> sp.Expr:
        """``<y_beta| L |y_alpha>`` for the Koopman generator of ``du_p/dt = u_p**2``.

        ``L = sum_p u_p**2 d/du_p`` sends ``y_alpha`` to ``sum_p alpha_p y_{alpha + e_p}``.
        Each raised monomial is then folded by :meth:`_fold_weight`.
        """
        return sp.expand(
            sp.Add(
                *(
                    a_p * self._fold_weight({**alpha, p: a_p + 1}, beta, pivot)
                    for p, a_p in alpha.items()
                )
            )
        )

    @pytest.mark.parametrize("pivot_kind", list(MATRICES), ids=list(MATRICES))
    @pytest.mark.parametrize(
        "ket_args",
        [
            ((SITE[0], 1),),
            ((SITE[0], 2),),
            ((SITE[0], 1), (SITE[1], 1)),
        ],
        ids=["stays-in-basis", "one-site-fold", "two-site-fold"],
    )
    def test_every_bra_matches_closed_form_coefficient(
        self, ket_args: tuple, pivot_kind: str
    ) -> None:
        """The builder folds each out-of-basis state, then solves both axes to reach the
        bra.

        The theory side does none of that. Before ``truncation_rule`` every folded entry read 0.
        """
        lin_ode_matrix, pivot = self.MATRICES[pivot_kind]
        alpha = self._powers(CarlemanState(*ket_args))
        column = lin_ode_matrix[self.BRAS, [ket_args]]
        nonzero = 0
        for k, bra_args in enumerate(self.BRAS):
            beta = self._powers(self.BASIS[bra_args])
            expected = self._theory_element(alpha, beta, pivot)
            assert sp.expand(column[k, 0] - expected) == 0, f"row {bra_args}"
            nonzero += expected != 0
        assert nonzero, "no bra reached"


class TestPerSitePivotLimit:
    """A per-site pivot keeps each pivot on its own site, because Koopman always pins
    the folded one."""

    N = 4
    IDX = sp.Idx("i", (0, N - 1))

    def _element(self, pivot: sp.Expr, bra: tuple) -> sp.Expr:
        # <bra| L |u_0> for du_i/dt = u_i**2 on a 4-ring at degree 1, under Koopman.
        u = sp.IndexedBase("u")
        ode = OrdinaryDifferentialSystem.from_sympy(u[self.IDX] ** 2, u, (self.IDX,))
        basis = PSCarlemanBasis(self.N, 1, pivot)
        return OperatorMatrix(basis, Koopman().create_operator(ode))[bra, ((0, 1),)]

    def test_koopman_pins_the_folded_site(self) -> None:
        """Koopman (``f K``) lowers first, so a delta pins the index to a site the ket
        holds, leaving no free axis to sum over."""
        s0 = s_site[0]
        # u_0**2 about s_0 to degree 1: -s_0**2 + 2*s_0*u_0, both rows pinned to site 0
        assert sp.expand(self._element(s_site, ())) == sp.expand(-(s0**2))
        assert sp.expand(self._element(s_site, ((0, 1),))) == 2 * s0

    def test_per_site_pivot_resolves_where_bra_pins_site(self) -> None:
        """The other half: a pinned fold keeps each pivot on its own site.

        Sites here are plain ints, not the grid tuples of :class:`TestPSCMatrixElements`.
        """
        n = 4
        u = sp.IndexedBase("u")
        i = sp.Idx("i", (0, n - 1))
        ode = OrdinaryDifferentialSystem.from_sympy(u[i] ** 2, u, (i,))
        basis = PSCarlemanBasis(n, 2, s_site)
        op = Koopman().create_operator(ode)
        rows = [((0, 2),), ((1, 2),)]
        lin_ode_column = OperatorMatrix(basis, op)[rows, [((0, 1), (1, 1))]]
        # raising one site folds the *other* away, so a site mixup swaps these two
        assert [sp.expand(lin_ode_column[k, 0]) for k in (0, 1)] == [
            s_site[1],
            s_site[0],
        ]
