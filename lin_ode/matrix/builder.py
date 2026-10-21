# pyright: reportReturnType=false, reportIndexIssue=false, reportArgumentType=false
# pyright: reportOperatorIssue=false
# sympy's stubs type .replace/.subs as Basic, loosely type Indexed subscripting, and type
# Expr + Expr as unsupported, so these rules are pure noise here (same suppression as
# lin_ode/basis/op.py).
"""Matrix blocks of a :class:`~lin_ode.operators.LinearOperator` on a basis.

:func:`build_matrix` materialises a dense block, dispatching on the operator's concrete
type: an :class:`~lin_ode.operators.OperatorSum` is summed over its symbolic index box
(each element solved for the contributing indices, so cost is independent of the grid
size N), while a :class:`~lin_ode.operators.SimpleOperator` acts at its literal site
index(es) and the coefficient is read off directly.

The operator objects are basis-independent and live in :mod:`lin_ode.operators`; this
module holds the basis-dependent extraction.
"""

from __future__ import annotations

import itertools
from typing import Sequence, TYPE_CHECKING, cast

import sympy as sp

from lin_ode.basis.basis_set import TruncatedBasisSet
from lin_ode.basis.op import apply_op, evaluate
from lin_ode.basis.state import BasisFunction
from lin_ode.matrix._coeff import get_coeff
from lin_ode.matrix.dense import Matrix
from lin_ode.operators.linear_operator import (
    LinearOperator,
    OperatorSum,
    SimpleOperator,
)

if TYPE_CHECKING:
    from lin_ode.matrix.interface import MatrixElement, MatrixLike
    from lin_ode.operators.interface import Basis


def _state_at(basis: TruncatedBasisSet, flat: int) -> BasisFunction:
    """Basis function at flat index ``flat`` (must be in-basis)."""
    factors = basis.structured_idx_of_flat_index(flat)
    assert factors is not None, f"flat index {flat} is out of basis"
    return basis.state_type(*factors)


def _truncate(expr: sp.Expr, basis: TruncatedBasisSet) -> sp.Expr:
    """Rewrite every state in an expansion through the basis's truncation rule.

    Without this an out-of-basis state simply fails to match the bra, which is right
    only for a basis that *drops* it; one that folds it back onto the retained states
    (PSC) contributes here and would otherwise be silently ignored. One pass suffices --
    the rule always lands within the degree.
    """
    states = expr.atoms(basis.state_type)
    return expr.xreplace({s: basis.truncation_rule(*s.args) for s in states})


def _sum_expansions(
    operator: OperatorSum, ket: BasisFunction, basis: TruncatedBasisSet
) -> list[sp.Expr]:
    """Each arm's operator expansion on ``ket``, truncated into the basis.

    This is the expensive half of an element and depends only on the ket, so a block
    shares it down each column instead of redoing it per row.
    """
    return [
        _truncate(evaluate(apply_op(word, ket), basis.registry), basis)
        for word, _ in operator.cases
    ]


def _sum_element(
    operator: OperatorSum, expansions: list[sp.Expr], bra: BasisFunction
) -> sp.Expr:
    """``<bra| O |ket>`` for an :class:`OperatorSum`: the coefficient of ``bra`` in each
    arm's expansion of the ket, summed over the index box, arm by arm (each arm's
    condition restricts it)."""
    bounds = [(int(idx.lower), int(idx.upper)) for idx in operator.indices]
    total: sp.Expr = sp.S.Zero
    earlier: list[sp.Basic] = []
    for (_, cond), res in zip(operator.cases, expansions):
        active = sp.And(cond, *(sp.Not(c) for c in earlier))  # Piecewise first-match
        total += get_coeff(res, bra, operator.indices, bounds, active)
        earlier.append(cond)
    return sp.expand(total)  # leaves an unpinned-axis Sum unevaluated


def _sum_entries(
    operator: OperatorSum,
    row_ids: list[int],
    col_ids: list[int],
    basis: TruncatedBasisSet,
) -> list[sp.Expr]:
    state = {k: _state_at(basis, k) for k in set(row_ids) | set(col_ids)}
    expansion = {k: _sum_expansions(operator, state[k], basis) for k in set(col_ids)}
    return [
        _sum_element(operator, expansion[c], state[r])
        for r, c in itertools.product(row_ids, col_ids)
    ]


def _simple_element(
    operator: SimpleOperator,
    ket: BasisFunction,
    bra: BasisFunction,
    basis: TruncatedBasisSet,
) -> sp.Expr:
    """``<bra| O |ket>`` for a :class:`SimpleOperator`: it acts on ``ket`` at its
    literal site index(es); the coefficient of ``bra`` is read off directly."""
    res = evaluate(apply_op(operator.word, ket), basis.registry)
    coeff = cast(sp.Expr, res.coeff(bra))
    return sp.simplify(coeff)


def _simple_entries(
    operator: SimpleOperator,
    row_ids: list[int],
    col_ids: list[int],
    basis: TruncatedBasisSet,
) -> list[sp.Expr]:
    state = {k: _state_at(basis, k) for k in set(row_ids) | set(col_ids)}
    return [
        _simple_element(operator, state[c], state[r], basis)
        for r, c in itertools.product(row_ids, col_ids)
    ]


def build_matrix(
    operator: LinearOperator,
    rows: Sequence[int] | int,
    cols: Sequence[int] | int,
    basis: Basis,
) -> MatrixLike | MatrixElement:
    """The dense ``rows`` x ``cols`` block of *operator* projected onto *basis* (flat
    indices), or a single element when both ``rows`` and ``cols`` are bare ``int``."""
    if not isinstance(basis, TruncatedBasisSet):
        raise TypeError(f"{type(operator).__name__} needs a TruncatedBasisSet")
    row_ids = [rows] if isinstance(rows, int) else list(rows)
    col_ids = [cols] if isinstance(cols, int) else list(cols)

    if isinstance(operator, OperatorSum):
        entries = _sum_entries(operator, row_ids, col_ids, basis)
    elif isinstance(operator, SimpleOperator):
        entries = _simple_entries(operator, row_ids, col_ids, basis)
    else:
        raise TypeError(f"cannot build a matrix for {type(operator).__name__}")

    if isinstance(rows, int) and isinstance(cols, int):
        return entries[0]
    return Matrix(len(row_ids), len(col_ids), entries)
