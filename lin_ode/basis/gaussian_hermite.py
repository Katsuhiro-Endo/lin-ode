"""Gauss-Hermite basis: products of Hermite functions, orthogonal.

Each factor is a Hermite function ``H_{n_i}(u_i) exp(-u_i^2/2)`` (physicists' ``H_n``);
a basis function is their product.

Used for the Gauss-Hermite linearization, which expands the state on this orthogonal
basis.
"""

import functools
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

_X = sp.Symbol("__x_gh")  # integration variable of the cached moments


class GaussianHermiteState(BasisFunction):
    """Product of Hermite functions ``prod_i psi_{n_i}(u_i)``."""


class GHURule:
    """Multiplication by ``x``: three-term recurrence on Hermite functions."""

    def __call__(self, site: sp.Expr, n: sp.Expr) -> sp.Expr:
        n = sp.sympify(n)  # keep coefficients exact when n arrives as a plain int
        return sp.sqrt((n + 1) / 2) * GaussianHermiteState((site, n + 1)) + sp.sqrt(
            n / 2
        ) * GaussianHermiteState((site, n - 1))


class GHKRule:
    """Derivative ``d psi_n``: three-term recurrence with opposite raising sign."""

    def __call__(self, site: sp.Expr, n: sp.Expr) -> sp.Expr:
        n = sp.sympify(n)  # keep coefficients exact when n arrives as a plain int
        return -sp.sqrt((n + 1) / 2) * GaussianHermiteState((site, n + 1)) + sp.sqrt(
            n / 2
        ) * GaussianHermiteState((site, n - 1))


@functools.cache
def _hermite_moment(n: int, m: int) -> sp.Expr:
    """``<psi_n | x^m>`` over the real line; zero by parity when ``n + m`` is odd."""
    if (n + m) % 2 == 1:
        return sp.S.Zero
    norm = 1 / sp.sqrt(2**n * sp.factorial(n) * sp.sqrt(sp.pi))
    psi_n = norm * sp.exp(-(_X**2) / 2) * sp.hermite(n, _X)
    return sp.integrate(psi_n * _X**m, (_X, -sp.oo, sp.oo))


class GHPolyProjector:
    """Polynomial-factor projection onto the Gauss-Hermite basis.

    ``project(factor, n, v)`` integrates ``psi_n(v) factor(v)`` monomial by monomial.
    """

    def can_project(self, factor: sp.Expr, v: sp.Symbol) -> bool:
        return is_polynomial_factor(factor, v)

    def project(self, factor: sp.Expr, n: int, v: sp.Symbol) -> sp.Expr:
        """``int psi_n(v) factor(v) dv`` for a polynomial factor."""
        check_polynomial_factor(factor, n, v)
        if not factor.has(v):
            return sp.expand(factor * _hermite_moment(n, 0))
        poly = sp.Poly(factor, v)
        return sp.expand(
            sum(c * _hermite_moment(n, m) for (m,), c in poly.as_dict().items())
        )


class GaussianHermiteBasis(TruncatedBasisSet):
    state_type = GaussianHermiteState
    is_l2 = True

    def __init__(
        self,
        n_vars: int | tuple[int, ...],
        degree: int,
        projectors: Sequence[Projector] | None = None,
    ):
        """``projectors=None`` uses the set shipped for this basis; pass a sequence to
        replace it wholesale."""
        super().__init__(
            n_vars, degree, (GHPolyProjector(),) if projectors is None else projectors
        )
        self.registry.register(UFamily, GHURule())
        self.registry.register(KFamily, GHKRule())
