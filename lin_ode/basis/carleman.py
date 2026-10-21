"""Carleman basis: state-variable monomials, non-orthogonal.

Basis functions are monomials ``u_0^{n_0} u_1^{n_1} ...``; Carleman linearization embeds
a polynomial ODE into a linear system in them, and CarlemanBasis is the finite degree-
truncation.
"""

from typing import Sequence

import sympy as sp

from lin_ode.basis.basis_set import TruncatedBasisSet
from lin_ode.basis.projection import (
    Projector,
    check_polynomial_factor,
    is_polynomial_factor,
)
from lin_ode.basis.state import BasisFunction
from lin_ode.basis.op import KFamily, UFamily


class CarlemanState(BasisFunction):
    """Monomial ``u_0^{n_0} u_1^{n_1} ...`` as sparse ``(site, power)`` factors."""


class CarlemanURule:
    """Multiplication by ``u_i``: raises the power at ``site`` by one."""

    def __call__(self, site: sp.Expr, n: sp.Expr) -> sp.Expr:
        return CarlemanState((site, n + 1))


class CarlemanKRule:
    """Derivative ``d/du_i``: lowers the power by one with factor ``n``."""

    def __call__(self, site: sp.Expr, n: sp.Expr) -> sp.Expr:
        return n * CarlemanState((site, n - 1))


class CarlemanProjector:
    """Polynomial-factor projection onto the Carleman (monomial) basis.

    ``project(factor, n, v)`` extracts the coefficient of v^n.
    """

    def can_project(self, factor: sp.Expr, v: sp.Symbol) -> bool:
        return is_polynomial_factor(factor, v)

    def project(self, factor: sp.Expr, n: int, v: sp.Symbol) -> sp.Expr:
        """Coefficient of v^n in `factor`."""
        check_polynomial_factor(factor, n, v)
        if not factor.has(v):
            return factor if n == 0 else sp.S.Zero
        poly = sp.Poly(factor, v)
        return poly.as_dict().get((n,), sp.S.Zero)


class CarlemanBasis(TruncatedBasisSet):
    state_type = CarlemanState

    def __init__(
        self,
        n_vars: int | tuple[int, ...],
        degree: int,
        projectors: Sequence[Projector] | None = None,
    ):
        """``projectors=None`` uses the set shipped for this basis; pass a sequence to
        replace it wholesale."""
        super().__init__(
            n_vars, degree, (CarlemanProjector(),) if projectors is None else projectors
        )
        self.registry.register(UFamily, CarlemanURule())
        self.registry.register(KFamily, CarlemanKRule())
