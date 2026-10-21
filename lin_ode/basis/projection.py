"""Projection strategies a basis dispatches over to expand a factor."""

from __future__ import annotations

from typing import Protocol

import sympy as sp


class ProjectionError(Exception):
    """Raised when no strategy on a basis can handle a factor."""


class IncompatibleBasisError(ValueError):
    """Raised when an operator needing an L2 basis is projected onto a non-L2 one."""


def is_polynomial_factor(factor: sp.Expr, v: sp.Symbol) -> bool:
    """Shared ``can_project`` for the polynomial projectors."""
    return sp.expand(factor).is_polynomial(v) is True


def check_polynomial_factor(factor: sp.Expr, n: int, v: sp.Symbol) -> None:
    """Guard the public ``project`` contract: Vector is not the only caller."""
    if n < 0:
        raise ProjectionError(f"mode index must be non-negative, got n={n}")
    if not is_polynomial_factor(factor, v):
        raise ProjectionError(f"{factor} is not polynomial in {v}")


class Projector(Protocol):
    """One projection strategy for a single-DOF factor (polynomial, trig, exp, ...).

    A basis owns a tuple of these and dispatches to the first whose
    ``can_project`` is True. Each handles a factor in a single plain symbol, as
    produced by ``Vector`` after factorization by DOF.
    """

    def can_project(self, factor: sp.Expr, v: sp.Symbol) -> bool:
        """Return True if this strategy can handle the given single-DOF factor."""
        ...

    def project(self, factor: sp.Expr, n: int, v: sp.Symbol) -> sp.Expr:
        """Compute the 1D inner product for mode n.

        Carleman: coefficient of v^n in factor. GH: ∫ ψ_n(v) · factor(v) dv.
        """
        ...
