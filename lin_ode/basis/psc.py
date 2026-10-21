# pyright: reportReturnType=false, reportGeneralTypeIssues=false
# sympy stubs type IndexedBase subscripting and .args as Basic, so pivot[site] does not
# read as an Expr and state.args does not read as iterable (same suppressions as
# lin_ode/basis/op.py).
"""Pivot-switching Carleman (PSC) basis (arXiv:2411.16233).

Same ``u^k`` states as :class:`CarlemanBasis`; the pivot enters only through truncation.
Where standard truncation drops a monomial of total degree > ``degree``, PSC replaces it
with its degree-``degree`` Taylor polynomial about a pivot ``s`` (Eqs. 9 and 11), whose error
is ``O((u - s)^{degree+1})`` -- so the retained states carry the discarded ones' behaviour
near the pivot instead of losing it.

Note the *whole* Taylor sum is needed, not just the leading relation ``(u - s)^m = 0``:
those coincide only at ``m = degree + 1``. A right-hand side of order M raises a state by
``M - 1``, so anything past quadratic (the paper's own logistic example) reaches
``m > degree + 1``, where imposing the single relation drops terms that are not small near
the pivot.

The pivot is one value shared by every site, or an ``sp.IndexedBase`` for a per-site one
-- see :meth:`PSCarlemanBasis._pivot_at` for where the latter leaves an ``sp.Sum``.
"""

import itertools
from math import comb
from typing import Sequence, TypeAlias, cast

import sympy as sp

from lin_ode.basis.carleman import CarlemanBasis
from lin_ode.basis.projection import Projector
from lin_ode.basis.state import BasisFunction, IdxSpec

Site: TypeAlias = int | sp.Expr | sp.Tuple


def _site_of(indexed: sp.Indexed) -> Site:
    """Site label of ``u[...]``: bare subscript in 1-D, an ``sp.Tuple`` on a grid."""
    idx = indexed.indices
    return idx[0] if len(idx) == 1 else sp.Tuple(*idx)


def _monomial_factors(
    term: sp.Expr, u: sp.IndexedBase
) -> tuple[sp.Expr, tuple[IdxSpec, ...]] | None:
    """``term`` split into scalar and ``(site, power)`` factors, None if not a monomial.

    A site passes through as written -- symbolic ones are fine, the builder resolves
    them later; only the power must be a literal positive integer (symbolic -> None).
    """
    scalar: sp.Expr = sp.S.One
    factors: list[IdxSpec] = []
    for factor in sp.Mul.make_args(term):
        base, exp = factor.as_base_exp()
        if not (isinstance(base, sp.Indexed) and base.base == u):
            if factor.has(u):
                return None
            scalar *= factor
            continue
        if not (exp.is_Integer and exp > 0):
            return None
        factors.append((_site_of(cast(sp.Indexed, base)), int(exp)))
    return scalar, tuple(factors)


def _monomial_of(state: BasisFunction, u: sp.IndexedBase) -> sp.Expr:
    """The ``u`` monomial a Carleman state stands for."""
    return sp.Mul(
        *[
            cast(sp.Expr, u[tuple(site)] if isinstance(site, sp.Tuple) else u[site])
            ** n
            for site, n in state.args
        ]
    )


class PSCarlemanBasis(CarlemanBasis):
    """Carleman basis whose truncation projects out-of-basis states instead of dropping.

    ``pivot`` is the reference state ``s``: one (possibly symbolic) value shared by
    every site, or an ``sp.IndexedBase`` read as ``pivot[site]`` for a per-site pivot.
    """

    def __init__(
        self,
        n_vars: int | tuple[int, ...],
        degree: int,
        pivot: sp.Expr | sp.IndexedBase | float,
        projectors: Sequence[Projector] | None = None,
    ):
        super().__init__(n_vars, degree, projectors)
        self.pivot = sp.sympify(pivot)

    def _extra_json_args(self) -> dict:
        from lin_ode import serialization as _ser

        return {"pivot": _ser.sympy_to_str(self.pivot)}

    @classmethod
    def _decode_extra_json_args(cls, extra: dict) -> dict:
        from lin_ode import serialization as _ser

        return {"pivot": _ser.sympy_from_str(extra["pivot"])}

    def _pivot_at(self, site: Site) -> sp.Expr:
        """Pivot at ``site``: the shared value, or ``pivot[site]`` for a per-site one.

        Folding keeps the dropped site's pivot in the coefficient, so a per-site pivot
        needs something to solve that site. Koopman (``f K``) lowers first and always
        pins it; Liouville (``-K f``) raises first and can invent a site nothing pins,
        so its vacuum row and diagonal come back as an unevaluated ``sp.Sum`` over that
        site instead of a closed form.
        """
        if not isinstance(self.pivot, sp.IndexedBase):
            return self.pivot
        return (
            self.pivot[tuple(site)] if isinstance(site, sp.Tuple) else self.pivot[site]
        )

    def _shifted_monomial(self, order: Sequence[tuple[Site, int]]) -> sp.Expr:
        """``prod_i (u_i - s_i)^{order_i}`` on the basis (its degree is ``sum(order)``)."""
        result = sp.S.Zero
        for combo in itertools.product(*[range(j + 1) for _, j in order]):
            coeff = sp.S.One
            for (site, j), b in zip(order, combo):
                coeff *= comb(j, b) * (-self._pivot_at(site)) ** (j - b)
            beta = [(site, b) for (site, _), b in zip(order, combo) if b > 0]
            result += coeff * self.state_type(*beta)
        return result

    def truncation_rule(self, *args: IdxSpec) -> sp.Expr:
        """PSC truncation (arXiv:2411.16233 Eqs. 9 and 11): keep in-basis states, replace the
        rest by their degree-``degree`` Taylor polynomial about the pivot.

        ``prod_i u_i^{m_i} = prod_i (s_i + (u_i - s_i))^{m_i}`` expanded in the shifted
        variables, dropping total order > ``degree`` -- the dropped terms are exactly the
        ``O((u - s)^{degree+1})`` remainder.
        """
        if sum(n for _, n in args) <= self.degree:
            return self.state_type(*args)
        result = sp.S.Zero
        for combo in itertools.product(*[range(int(m) + 1) for _, m in args]):
            if sum(combo) > self.degree:
                continue
            coeff = sp.S.One
            for (site, m), j in zip(args, combo):
                coeff *= comb(int(m), j) * self._pivot_at(site) ** (int(m) - j)
            order = [(site, j) for (site, _), j in zip(args, combo)]
            result += coeff * self._shifted_monomial(order)
        return result

    def truncate_expression(self, expr: sp.Expr, u: sp.IndexedBase) -> sp.Expr:
        """:meth:`truncation_rule` at expression level -- a Carleman state *is* a ``u``
        monomial, so it reads off directly; a non-monomial term names no state."""
        return sp.expand(
            sp.Add(
                *[self._truncate_term(t, u) for t in sp.Add.make_args(sp.expand(expr))]
            )
        )

    def _truncate_term(self, term: sp.Expr, u: sp.IndexedBase) -> sp.Expr:
        split = _monomial_factors(term, u)
        if split is None:
            return term
        scalar, factors = split
        if sum(n for _, n in factors) <= self.degree:
            return term
        folded = self.truncation_rule(*factors)
        return scalar * folded.xreplace(
            {s: _monomial_of(s, u) for s in folded.atoms(self.state_type)}
        )
