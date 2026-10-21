# pyright: reportGeneralTypeIssues=false, reportIncompatibleMethodOverride=false
# sympy stubs: .args is Basic (not iterable) and overriding Function.eval trips the
# override check; both noise for this Function subclass.
from typing import Self, TypeAlias, cast

import sympy as sp

IdxSpec: TypeAlias = tuple[int | sp.Expr | sp.Tuple, sp.Expr | int]


def _site_sort_key(site: int | sp.Expr | sp.Tuple) -> tuple[int | str, ...]:
    """Canonical sort key for a site label, scalar (1-D) or sp.Tuple (D-dim).

    The leading 0/1 separates concrete from symbolic sites so mixed sites never compare
    an int against a str.
    """
    coords = tuple(site) if isinstance(site, sp.Tuple) else (site,)
    if all(c.is_Integer for c in coords):  # type: ignore[union-attr]
        return (0, *(int(c) for c in coords))  # type: ignore[arg-type]
    return (1, *(str(c) for c in coords))


def _sites_equal(dim: sp.Expr | sp.Tuple, target: sp.Expr | sp.Tuple) -> bool:
    """Whether two labels denote the same site, compared coordinate-wise.

    A grid site may be given as an sp.Tuple or a plain python tuple, and in 1-D the
    scalar ``i``, ``(i,)`` and ``sp.Tuple(i)`` all denote the same site.
    """
    dim_coords = tuple(dim) if isinstance(dim, (sp.Tuple, tuple)) else (dim,)
    target_coords = (
        tuple(target) if isinstance(target, (sp.Tuple, tuple)) else (target,)
    )
    return len(dim_coords) == len(target_coords) and all(
        a == b or sp.simplify(cast(sp.Expr, a) - cast(sp.Expr, b)) == sp.S.Zero
        for a, b in zip(dim_coords, target_coords)
    )


class BasisFunction(sp.Function):
    """Separable basis function with one index per degree of freedom.

    - A product of univariate factors, one per variable, stored sparsely as
      ``(site, power)`` pairs with zero powers dropped.
    - ``site`` labels the dof (scalar, or sp.Tuple for a grid); ``power`` is its
      single order.
    - The factor itself (monomial for Carleman, Hermite for GH) is set by the
      subclass; ``eval`` canonicalises the ordering.
    - Non-separable bases, or ones needing more than one index per dof, are out
      of scope.
    """

    nargs = None

    def __new__(cls, *args: IdxSpec) -> Self:
        # sympy Function.__new__ is typed as returning a class, not an instance
        return super().__new__(cls, *args)  # type: ignore[return-value]

    @classmethod
    def eval(cls, *factors: IdxSpec) -> Self | None:  # type: ignore[override]
        # Drop zero-power factors and sort by site so equal states share one form
        filt_factors = filter(lambda fact: len(fact) > 0 and fact[1] != 0, factors)
        nonzero = [(site, power) for site, power in filt_factors]
        canonical = tuple(sorted(nonzero, key=lambda f: _site_sort_key(f[0])))
        if factors != canonical:
            return cls(*canonical)
        return None

    def get_n(self, target: sp.Expr | sp.Tuple) -> sp.Expr:
        """Order (power) at ``target`` site, or 0 if absent."""
        for dim, n in self.args:
            if _sites_equal(dim, target):
                return n
        return sp.S.Zero

    def update_dim(self, target: sp.Expr | sp.Tuple, new_n) -> sp.Expr:
        """Return a new state with the order at ``target`` set to ``new_n`` (dropping it
        if 0)."""
        new_args = []
        found = False
        for dim, n in self.args:
            if _sites_equal(dim, target):
                if new_n != sp.S.Zero:
                    new_args.append((dim, new_n))
                found = True
            else:
                new_args.append((dim, n))
        if not found and new_n != sp.S.Zero:
            new_args.append((target, new_n))
        if not new_args:
            return type(self)()
        return type(self)(*new_args)


class AntiKroneckerDelta(sp.Function):
    """Represents 1 - delta_{ij}. Evaluates to 0 if equal, 1 if different."""

    @classmethod
    def eval(cls, i: sp.Expr, j: sp.Expr):
        if i == j:
            return 0
        if i.is_number and j.is_number:
            return 1

    def _latex(self, printer):
        arg1, arg2 = [printer._print(arg) for arg in self.args]
        return rf"\bar{{\delta}}_{{{arg1}{arg2}}}"
