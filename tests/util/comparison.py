"""Symbolic helpers shared across test modules.

Each was defined once per module before. They live here so a report cell, a matrix entry
and a brute-force row all follow one rule. Sympy's wide return types are narrowed here
too.
"""

import sympy as sp
from sympy import Idx, IndexedBase

from tests.testing import helper
from lin_ode.basis import TruncatedBasisSet


def indexed_base(name: str, shape: tuple[int, ...] | None = None) -> IndexedBase:
    """Typed wrapper for ``IndexedBase(...)``; sympy's own constructor return type is
    too wide (``MatrixBase | NDimArray | Basic | IndexedBase``) for pyright to
    narrow."""
    if shape is not None:
        return IndexedBase(name, shape=shape)  # type: ignore[return-value]
    return IndexedBase(name)  # type: ignore[return-value]


def idx(name: str, index_range: tuple[int, int] | None = None) -> Idx:
    """Typed wrapper for ``Idx(...)``; sympy's own constructor return type is too wide
    (``Basic | Idx``) for pyright to narrow."""
    if index_range is not None:
        return Idx(name, index_range)  # type: ignore[return-value]
    return Idx(name)  # type: ignore[return-value]


s_pivot = sp.Symbol("s", real=True)
"""The PSC pivot when every site shares one."""

s_site = sp.IndexedBase("s")
"""The PSC pivot when each site carries its own, as ``s[p]``.

A shared pivot collapses the divergence's site sum into a count; a per-site one keeps it
as a sum, one term per site. That sum is the only place a region's *outside* sites reach
a folded entry, so it is where a miscount hides.
"""


@helper
def list_basis_labels(basis: TruncatedBasisSet) -> list:
    """Every basis function's ``(site, power)`` label, in flat-index order: the labels
    ``matrix[...]`` is indexed by."""
    return [
        label
        for k in range(len(basis))
        if (label := basis.structured_idx_of_flat_index(k)) is not None
    ]


def evaluate_sums(entry: sp.Expr) -> sp.Expr:
    """A per-site pivot leaves an unevaluated ``Sum``, which has to run first."""
    return entry.doit() if hasattr(entry, "doit") else entry


@helper
def agrees(lin_ode_entry: sp.Expr, expected: sp.Expr) -> bool:
    """Whether the library's entry and the expected one are the same value.

    ``expand`` settles almost every entry and ``simplify`` is ~100x dearer, so it only
    runs on what expand leaves behind.
    """
    difference = lin_ode_entry - expected
    return sp.expand(difference) == 0 or sp.simplify(difference) == 0
