"""Project a symbolic expression onto a basis as a coefficient vector."""

from __future__ import annotations

from typing import TypeAlias, cast

import sympy as sp

from .. import serialization as _ser
from ..basis import BasisFunction, ProjectionError, TruncatedBasisSet
from ..basis.state import IdxSpec
from ..indexing import BasisFunctionArgs, _flat_to_site, get_shape_and_count
from ..utils import require_literal_indices

StructuredIdx: TypeAlias = "IdxSpec | BasisFunctionArgs"

_V = sp.Symbol("__v")  # placeholder for the dof being integrated out


class Vector:
    r"""Lazy coefficient-vector view of an expression projected onto a basis.

    The vector counterpart of ``OperatorMatrix``: holds the expression, the basis
    and the coordinate base, and computes coefficients on demand in ``__getitem__``,
    addressed by the same structured ``(site, power)`` index the matrix is sliced with.

    Note:
        For each dof, substitutes ``u[site] -> __v`` and dispatches to the first of
        ``basis.projectors`` whose ``can_project`` returns True, treating the remaining
        dofs as symbolic parameters. By Fubini's theorem this iterated projection
        equals the full multi-dof inner product, so non-separable expressions (e.g.
        ``sin(u[0]*u[1])``) are handled provided a projector supports them.

    Note:
        Goes through ``basis.truncate_expression`` first, so PSC keeps the above-degree
        part (folded onto the retained states) where a dropping basis loses it.

    Raises:
        ValueError: If a subscript of *u* is not a literal integer. Write the sites out:
            ``u[k]`` projects to nothing, it is not a sum over ``k``.
    """

    def __init__(
        self, vector: sp.Expr, basis: TruncatedBasisSet, u: sp.IndexedBase
    ) -> None:
        require_literal_indices(vector, "a Vector", u)
        self._terms = sp.Add.make_args(basis.truncate_expression(vector, u))
        self._basis = basis
        self._u = u
        self._shape, _ = get_shape_and_count(basis.n_vars)
        # keyed by basis function, so the equivalent spellings of an index share a hit
        self._cache: dict[BasisFunction, sp.Expr] = {}

    def dump_json(self, indent: int | None = None) -> str:
        """Return this vector as a JSON string.

        Implementation note (for contributors): the stored expression is already
        truncated. Truncating it again on reconstruction changes nothing, so the value is
        preserved.
        """
        return _ser.json_dumps(
            {
                "vector": _ser.sympy_to_str(sp.Add(*self._terms)),
                "basis": self._basis.to_json_data(),
                "u": _ser.sympy_to_str(self._u),
            },
            indent,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Vector":
        """Rebuild from a string produced by :meth:`dump_json`."""
        json_data = _ser.json_loads(json_str)
        return cls(
            _ser.sympy_from_str(json_data["vector"]),
            TruncatedBasisSet.from_json_data(json_data["basis"]),
            _ser.sympy_from_str(json_data["u"]),
        )

    @property
    def basis(self) -> TruncatedBasisSet:
        """The basis whose truncation and projectors define the coefficients."""
        return self._basis

    def _project_1d(self, expr: sp.Expr, n_j: int) -> sp.Expr:
        # read basis.projectors per call, so prepending one post-construction takes effect
        for p in self._basis.projectors:
            if p.can_project(expr, _V):
                return p.project(expr, n_j, _V)
        raise ProjectionError(
            f"No projector on {type(self._basis).__name__} can handle: {expr!r}"
        )

    def _project_term(self, term: sp.Expr, state: BasisFunction) -> sp.Expr:
        expr = term
        for j in range(self._basis.n_dof):
            site = cast(sp.Expr, _flat_to_site(j, self._shape))
            substituted = cast(sp.Expr, expr.subs(self._u[site], _V))
            expr = self._project_1d(substituted, int(state.get_n(site)))
        return expr

    def __getitem__(self, idx: StructuredIdx) -> sp.Expr:
        """Coefficient at structured index ``idx``; 0 for an index outside the basis.

        Raises:
            ProjectionError: if no projector on the basis can handle an intermediate
                factor.
        """
        if self._basis.flat_index_of_structured_index(idx) is None:
            return sp.S.Zero
        state = self._basis[idx]
        if state not in self._cache:
            self._cache[state] = sp.expand(
                sum(self._project_term(t, state) for t in self._terms)
            )
        return self._cache[state]
