# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false
# sympy stubs type .subs as Basic and sum() as Expr | Literal[0], so arithmetic on either
# does not check (same suppression as tests/basis/test_psc.py).
"""Reading coefficients off a ``Vector``: one basis at a time, one property per test.

Every lookup uses the structured ``(site, power)`` label. A vector entry and a matrix
row are therefore addressed the same way.
"""

from typing import Sequence, cast

import pytest
import sympy as sp

from lin_ode.basis import (
    CarlemanBasis,
    CarlemanProjector,
    GaussianHermiteBasis,
    GHPolyProjector,
    PSCarlemanBasis,
    ProjectionError,
)
from lin_ode.vector import Vector

from tests.util.comparison import s_pivot
from lin_ode.basis import TruncatedBasisSet
from lin_ode.basis import Projector

u: sp.IndexedBase = sp.IndexedBase("u")  # type: ignore[assignment]


def x(j: int) -> sp.Expr:
    """Coordinate u[j] as an Expr; sympy's IndexedBase stubs return a wide union."""
    return cast(sp.Expr, u[j])


def xy(a: int, b: int) -> sp.Expr:
    """Grid coordinate u[a, b] as an Expr."""
    return cast(sp.Expr, u[a, b])


class TestCarleman:
    def test_subscript_extracts_coefficients(self) -> None:
        """``rho[0, n]`` reads the coefficient of ``u_0**n`` in the polynomial."""
        rho = Vector(x(0) ** 2 + 3 * x(0) + 5, CarlemanBasis(1, 3), u)
        assert rho[0, 0] == 5
        assert rho[0, 1] == 3
        assert rho[0, 2] == 1

    def test_vector_answers_to_the_matrixs_own_label(self) -> None:
        """Flat index 4 of the 2-site degree-2 basis is ``u_0 u_1``, whose coefficient
        in ``u_0 u_1`` is 1; ``u_0`` alone has none."""
        basis = CarlemanBasis(2, 2)
        g = Vector(x(0) * x(1), basis, u)

        label = basis.structured_idx_of_flat_index(4)

        assert label == ((0, 1), (1, 1))
        assert g[label] == 1
        assert g[(0, 1), (1, 1)] == 1  # spelled out, the same entry
        assert g[0, 1] == 0

    def test_grid_dofs_are_addressed_by_coordinate(self) -> None:
        """Squaring ``u[0,0] - u[0,1]`` gives 1, 1 and -2 on tuple-labelled sites."""
        # a grid site is u[a, b]; substituting a flat u[k] matches nothing and hands
        # every index the un-projected expression straight back
        basis = CarlemanBasis((2, 2), 2)
        g = Vector((xy(0, 0) - xy(0, 1)) ** 2, basis, u)
        s00, s01, s10 = sp.Tuple(0, 0), sp.Tuple(0, 1), sp.Tuple(1, 0)
        assert g[(s00, 2),] == 1
        assert g[(s01, 2),] == 1
        assert g[(s00, 1), (s01, 1)] == -2
        assert g[(s10, 2),] == 0

    def test_repeated_lookup_is_cached(self) -> None:
        """A second lookup reads the cache instead of the projectors."""
        basis = CarlemanBasis(1, 2)
        rho = Vector(x(0) ** 2, basis, u)
        assert rho[0, 2] == 1
        basis.projectors = ()  # a second projection would now raise
        assert rho[0, 2] == 1


def gh_by_integration(expr: sp.Expr, powers: Sequence[int]) -> sp.Expr:
    """Brute-force ``<psi_{powers} | expr>`` by integrating one site at a time."""
    z = sp.Symbol("z")
    for j, n in enumerate(powers):
        norm = 1 / sp.sqrt(2**n * sp.factorial(n) * sp.sqrt(sp.pi))
        psi_n = norm * sp.exp(-(z**2) / 2) * sp.hermite(n, z)
        expr = sp.integrate(psi_n * expr.subs(u[j], z), (z, -sp.oo, sp.oo))
    return sp.simplify(expr)


class TestGaussianHermite:
    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_polynomial_matches_direct_integration(self, n: int) -> None:
        """The coefficient equals ``<psi_n | expr>`` from direct integration."""
        expr = x(0) ** 2 + 3 * x(0) + 5
        rho = Vector(expr, GaussianHermiteBasis(1, 3), u)
        assert sp.simplify(rho[0, n] - gh_by_integration(expr, [n])) == 0

    def test_two_dofs_match_direct_integration(self) -> None:
        """A two-site label reads the overlap ``<psi_1 psi_2 | expr>``."""
        expr = x(0) * x(1) ** 2
        g = Vector(expr, GaussianHermiteBasis(2, 3), u)
        assert sp.simplify(g[(0, 1), (1, 2)] - gh_by_integration(expr, [1, 2])) == 0


class TestSymbolicIndexRejected:
    """A symbolic subscript is refused, not projected to zero.

    Projection reads terms off as written, so ``u[k]`` would contribute nothing rather
    than stand for a sum over the sites ``k`` ranges over.
    """

    k = sp.Idx("k", (0, 2))

    @pytest.mark.parametrize(
        "expr",
        [cast(sp.Expr, u[k]) ** 2, cast(sp.Expr, u[k + 1])],
        ids=["bare-index", "offset-index"],
    )
    def test_symbolic_subscript_raises(self, expr: sp.Expr) -> None:
        with pytest.raises(ValueError, match="literal integer indices"):
            _ = Vector(expr, CarlemanBasis(3, 2), u)

    def test_the_written_out_sum_is_accepted(self) -> None:
        """The same observable over literal sites projects as usual."""
        g = Vector(x(0) ** 2 + x(1) ** 2 + x(2) ** 2, CarlemanBasis(3, 2), u)
        assert [g[((j, 2),)] for j in range(3)] == [1, 1, 1]

    def test_a_parameter_may_stay_symbolic(self) -> None:
        """Only the state's subscripts are checked, so a coefficient may be symbolic."""
        h = sp.Symbol("h")
        assert Vector(h * x(0) ** 2, CarlemanBasis(3, 2), u)[((0, 2),)] == h


@pytest.mark.parametrize(
    "projector",
    [CarlemanProjector(), GHPolyProjector()],
    ids=["carleman", "gaussian-hermite"],
)
class TestProjectorGuards:
    """``project`` enforces its own contract, since Vector is not the only caller."""

    v = sp.Symbol("v")

    @pytest.mark.parametrize(
        "not_a_polynomial",
        [sp.sin(v), sp.exp(v), 1 / v],
        ids=["sin", "exp", "reciprocal"],
    )
    def test_can_project_is_false_for_what_it_cannot_expand(
        self, projector: Projector, not_a_polynomial: sp.Expr
    ) -> None:
        """Both projectors decline ``sin(v)``, ``exp(v)`` and ``1/v``."""
        assert projector.can_project(not_a_polynomial, self.v) is False

    @pytest.mark.parametrize(
        "factor, mode",
        [(sp.sin(v), 1), (v**2, -1)],
        ids=["non-polynomial-factor", "negative-mode"],
    )
    def test_project_refuses_input_outside_contract(
        self, projector: Projector, factor: sp.Expr, mode: int
    ) -> None:
        """``can_project`` reporting False is advisory; ``project`` itself raises, so a
        caller that skips the check cannot get a silently wrong coefficient."""
        with pytest.raises(ProjectionError):
            projector.project(factor, mode, self.v)


class TestPSC:
    """An above-degree monomial lands on the retained states as a Taylor polynomial.

    The fold is that polynomial about the pivot; arXiv:2411.16233 Eqs. 9 and 11.
    """

    def test_out_of_basis_monomial_becomes_taylor_polynomial(self) -> None:
        """``u_0**3`` folds to ``s**3 - 3 s**2 u_0 + 3 s u_0**2`` at pivot ``s``."""
        # u0^3 ~ s_pivot^3 + 3s^2 (u0 - s_pivot) + 3s (u0 - s_pivot)^2 = s_pivot^3 - 3s^2 u0 + 3s u0^2
        rho = Vector(x(0) ** 3, PSCarlemanBasis(1, 2, s_pivot), u)
        assert sp.expand(rho[0, 0] - s_pivot**3) == 0
        assert sp.expand(rho[0, 1] + 3 * s_pivot**2) == 0
        assert sp.expand(rho[0, 2] - 3 * s_pivot) == 0

    def test_folded_vector_agrees_with_expr_at_pivot(self) -> None:
        """Every residual derivative up to order ``degree`` is zero at the pivot."""
        # the defining property of the fold: the residual is O((u - s_pivot)^{degree+1})
        v = sp.Symbol("v")
        basis = PSCarlemanBasis(1, 3, s_pivot)
        expr = x(0) ** 5 - 2 * x(0) ** 4 + 7
        rho = Vector(expr, basis, u)
        folded = sum(rho[0, n] * v**n for n in range(basis.degree + 1))
        residual = sp.expand(folded - expr.subs(u[0], v))
        for order in range(basis.degree + 1):
            assert sp.expand(sp.diff(residual, v, order).subs(v, s_pivot)) == 0, order

    def test_zero_pivot_reduces_to_plain_truncation(self) -> None:
        """At pivot 0 the fold of ``u_0**3`` is zero, leaving only the constant 5."""
        rho = Vector(x(0) ** 3 + 5, PSCarlemanBasis(1, 2, 0), u)
        assert rho[0, 0] == 5
        assert rho[0, 1] == 0
        assert rho[0, 2] == 0

    def test_in_basis_terms_are_untouched(self) -> None:
        """``3 u_0 + 5`` keeps its coefficients 5 and 3 under the fold."""
        rho = Vector(3 * x(0) + 5, PSCarlemanBasis(1, 2, s_pivot), u)
        assert rho[0, 0] == 5
        assert rho[0, 1] == 3
        assert rho[0, 2] == 0

    def test_fold_is_joint_in_total_degree_over_two_dofs(self) -> None:
        """``u_0**2 u_1`` has total degree 3, so the fold expands it about the pivot."""
        # u0^2 u1 folds to s_pivot^3 - 2s^2 u0 - s_pivot^2 u1 + s_pivot u0^2 + 2s u0 u1
        rho = Vector(x(0) ** 2 * x(1), PSCarlemanBasis(2, 2, s_pivot), u)
        assert sp.expand(rho[(0, 0),] - s_pivot**3) == 0
        assert sp.expand(rho[0, 1] + 2 * s_pivot**2) == 0
        assert sp.expand(rho[1, 1] + s_pivot**2) == 0
        assert sp.expand(rho[0, 2] - s_pivot) == 0
        assert sp.expand(rho[(0, 1), (1, 1)] - 2 * s_pivot) == 0
        assert rho[1, 2] == 0

    def test_per_site_pivot_folds_at_that_sites_pivot(self) -> None:
        """An ``IndexedBase`` pivot gives coefficient ``3 p[0]`` for ``u_0**3``."""
        pivot: sp.IndexedBase = sp.IndexedBase("p")  # type: ignore[assignment]
        rho = Vector(x(0) ** 3, PSCarlemanBasis(2, 2, pivot), u)
        assert sp.expand(rho[0, 2] - 3 * pivot[0]) == 0

    def test_grid_monomial_folds_at_tuple_site(self) -> None:
        """``u[0,1]**3`` gives coefficient ``3 s`` at label ``((0, 1), 2)``."""
        basis = PSCarlemanBasis((2, 2), 2, s_pivot)
        g = Vector(xy(0, 1) ** 3, basis, u)
        assert sp.expand(g[(sp.Tuple(0, 1), 2),] - 3 * s_pivot) == 0

    def test_non_monomial_term_is_left_to_projectors(self) -> None:
        """The fold skips ``sin(u_0)``, so projection raises ``ProjectionError``."""
        # nothing to fold a Taylor series onto, so the term reaches projection unchanged
        rho = Vector(sp.sin(x(0)) + x(0) ** 3, PSCarlemanBasis(1, 2, s_pivot), u)
        with pytest.raises(ProjectionError):
            rho[0, 1]


@pytest.mark.parametrize(
    "basis",
    [CarlemanBasis(1, 2), GaussianHermiteBasis(1, 2)],
    ids=["carleman", "gaussian-hermite"],
)
def test_dropping_basis_leaves_expr_alone(basis: TruncatedBasisSet) -> None:
    """Only a *folding* rule rewrites the expression; a dropping one leaves the above-
    degree term for projection to discard, so u0**5 comes back verbatim."""
    assert basis.truncate_expression(x(0) ** 5, u) == x(0) ** 5


@pytest.mark.parametrize(
    "basis",
    [CarlemanBasis(1, 2), GaussianHermiteBasis(1, 2), PSCarlemanBasis(1, 2, 0)],
    ids=["carleman", "gaussian-hermite", "psc-at-zero-pivot"],
)
def test_index_above_degree_reads_zero(basis: TruncatedBasisSet) -> None:
    """``u0**3`` has a genuine component at index 3 on every one of these bases.

    But 3 > 2, so the truncated vector has no entry to hold it.
    """
    assert Vector(x(0) ** 3, basis, u)[0, 3] == 0


class TestProjectorDispatch:
    def test_prepended_projector_shadows_shipped_one(self) -> None:
        """The first projector to claim the factor wins, so the entry reads 7."""

        class ConstantProjector:
            """Claims every factor and reports a coefficient of 7."""

            def can_project(self, factor: sp.Expr, v: sp.IndexedBase) -> bool:
                return True

            def project(self, factor: sp.Expr, n: int, v: sp.IndexedBase) -> sp.Expr:
                return sp.Integer(7)

        basis = CarlemanBasis(1, 2)
        basis.projectors = (ConstantProjector(), *basis.projectors)
        assert Vector(x(0) ** 2, basis, u)[0, 2] == 7

    def test_unhandled_factor_raises(self) -> None:
        """An empty projector tuple leaves ``u_0`` unhandled, so lookup raises."""
        basis = CarlemanBasis(
            1, 2, projectors=[]
        )  # wholesale replacement, nothing left
        rho = Vector(x(0), basis, u)
        with pytest.raises(ProjectionError):
            rho[0, 1]
