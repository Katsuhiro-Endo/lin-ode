# pyright: reportReturnType=false, reportIndexIssue=false, reportArgumentType=false
# sympy's stubs type .replace/.subs as Basic and loosely type Indexed subscripting, so
# these three rules are pure noise here (same suppression as lin_ode/basis/op.py).
"""Symbolic linear operators, lifted to the raising/lowering algebra.

A :class:`LinearOperator` is a basis-independent linear operator on the function space,
built from an :class:`~lin_ode.indexed_expression.IndexedExpression` (an ODE right-hand
side, or an observable) by rewriting its state/derivative bases as the raising/lowering
operator families. The lift happens at construction, so the operator object holds the
symbolic word(s), independent of any basis.

Projecting an operator onto a basis (:class:`~lin_ode.matrix.OperatorMatrix`) hands it to
the matrix builders in :mod:`lin_ode.matrix`, which read the operator's word(s) and
extract coefficients; the operator itself carries no basis-dependent state.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import sympy as sp

from lin_ode import serialization as _ser
from lin_ode.basis.op import K, U
from lin_ode.indexed_expression import IndexedExpression
from lin_ode.utils import require_literal_indices

if TYPE_CHECKING:
    from lin_ode.operators.interface import Basis, IndexSpec


def _to_op_algebra(
    operator: IndexedExpression, diff_base: sp.IndexedBase
) -> list[tuple[sp.Expr, sp.Basic]]:
    """Rewrite the operator's state/derivative IndexedBases as the raising/lowering
    operator families, per Piecewise branch (all subscripts, so multi-dimensional
    indices survive)."""
    u = operator.state_variable

    def rewrite(expr: sp.Expr) -> sp.Expr:
        in_u = expr.replace(
            lambda e: isinstance(e, sp.Indexed) and e.base == u,
            lambda e: U[e.indices],
        )
        return in_u.replace(
            lambda e: isinstance(e, sp.Indexed) and e.base == diff_base,
            lambda e: K[e.indices],
        )

    return [(rewrite(case), cond) for case, cond in operator.cases]


class LinearOperator(ABC):
    r"""A symbolic linear operator on the function space, lifted to the raising/lowering
    algebra at construction and ready for projection onto a basis.

    Lifts the source :class:`~lin_ode.indexed_expression.IndexedExpression` at
    construction and holds the resulting operator word(s) -- the ``(word, condition)``
    pairs, one per Piecewise arm. Concrete subclasses (:class:`OperatorSum`,
    :class:`SimpleOperator`) tag which internal structure the operator has and expose
    the lift the way their builder reads it -- :attr:`OperatorSum.cases` (all arms) or
    :attr:`SimpleOperator.word` (the single arm) -- which the matrix builders in
    :mod:`lin_ode.matrix` dispatch on. This class is basis-independent; an
    :class:`~lin_ode.matrix.OperatorMatrix` binds it to a basis.

    Args:
        operator: The source :class:`~lin_ode.indexed_expression.IndexedExpression` to
            lift.
        diff_base: The :class:`sympy.IndexedBase` marking the derivative operator to
            rewrite into the word.
    """

    #: Set from :attr:`~.interface.LinearizationMethod.requires_l2_basis` by
    #: ``create_operator``, read by :class:`~lin_ode.matrix.OperatorMatrix`.
    requires_l2_basis: bool = False

    #: Class name of the :class:`~.interface.LinearizationMethod` that built this
    #: operator, so errors name the method rather than this wrapper.
    method_name: str = ""

    def __init__(
        self,
        operator: IndexedExpression,
        diff_base: sp.IndexedBase,
    ) -> None:
        self._operator = operator
        self._cases = _to_op_algebra(operator, diff_base)
        self._diff_base = diff_base

    @property
    def diff_base(self) -> sp.IndexedBase:
        """The :class:`sympy.IndexedBase` marking the derivative operator in the
        word."""
        return self._diff_base

    def compatible_with_basis(self, basis: Basis[IndexSpec]) -> bool:
        """Whether *basis* can carry this operator; membership of L2(R^N) is the only
        condition so far."""
        return not self.requires_l2_basis or basis.is_l2

    def to_json_data(self) -> dict:
        """Return the source inputs as a JSON-ready dict.

        The dict is tagged with the concrete class name. An embedding object
        (``OperatorMatrix``) reads that tag to rebuild the right subclass. Both
        subclasses inherit this method. ``method_name`` and ``requires_l2_basis`` are set
        by ``create_operator`` after construction, so they are stored and restored too.
        """
        return {
            "type": type(self).__name__,
            "operator": self._operator.to_json_data(),
            "diff_base": _ser.sympy_to_str(self._diff_base),
            "method_name": self.method_name,
            "requires_l2_basis": self.requires_l2_basis,
        }

    @staticmethod
    def from_json_data(json_data: dict) -> LinearOperator:
        """Rebuild the concrete operator named in ``json_data['type']``."""
        # Look up the class name on the lin_ode.operators package. This finds only the
        # operators that package exports. A LinearOperator subclass defined outside
        # lin_ode.operators does not resolve here. Support external subclasses by replacing
        # this getattr with an explicit name-to-class registry that each subclass joins.
        from lin_ode import operators as _ops

        cls = getattr(_ops, json_data["type"])
        operator = cls(
            IndexedExpression.from_json_data(json_data["operator"]),
            _ser.sympy_from_str(json_data["diff_base"]),
        )
        operator.method_name = json_data["method_name"]
        operator.requires_l2_basis = json_data["requires_l2_basis"]
        return operator

    def dump_json(self, indent: int | None = None) -> str:
        """Return this operator as a JSON string.

        Both subclasses inherit this method.
        """
        return _ser.json_dumps(self.to_json_data(), indent)

    @classmethod
    def from_json(cls, json_str: str) -> LinearOperator:
        """Rebuild the concrete subclass recorded in the string."""
        return LinearOperator.from_json_data(_ser.json_loads(json_str))


class OperatorSum(LinearOperator):
    r"""A linearized operator as an unevaluated sum over symbolic indices,
    :math:`O = \sum_{i,\dots} O[i,\dots]`.

    Each matrix element ``<bra| O |ket>`` is extracted by solving for the contributing
    indices over the index box, so cost is independent of the grid size N.

    Args:
        operator: The :class:`~lin_ode.indexed_expression.IndexedExpression` to lift.
        diff_base: The :class:`sympy.IndexedBase` marking the derivative operator in it.
    """

    def __init__(self, operator: IndexedExpression, diff_base: sp.IndexedBase) -> None:
        super().__init__(operator, diff_base)
        self._indices = operator.indices

    @property
    def cases(self) -> list[tuple[sp.Expr, sp.Basic]]:
        """The lifted ``(word, condition)`` pairs, one per Piecewise arm, in first-match
        order."""
        return self._cases

    @property
    def indices(self) -> tuple[sp.Idx, ...]:
        """The :class:`sympy.Idx` symbols the operator is summed over."""
        return self._indices


class SimpleOperator(LinearOperator):
    r"""An operator with no symbolic indices and a single arm (an observable in the KvN
    context): it acts at its literal site index(es) and the coefficient of the bra is
    read off directly, with no summation over an index box.

    Args:
        operator: The :class:`~lin_ode.indexed_expression.IndexedExpression` to lift; its
            subscripts must be literal integers.
        diff_base: The :class:`sympy.IndexedBase` marking the derivative operator in it.
    """

    def __init__(self, operator: IndexedExpression, diff_base: sp.IndexedBase) -> None:
        super().__init__(operator, diff_base)
        assert len(self._cases) == 1
        self._validate_literal_indices()

    @property
    def word(self) -> sp.Expr:
        """The single lifted operator word."""
        return self._cases[0][0]

    def _validate_literal_indices(self) -> None:
        """Reject a word whose subscripts aren't literal ints (e.g. ``u[0, 1]`` is fine,
        ``u[0, j]`` is not): a :class:`SimpleOperator` reads off concrete-index terms
        directly, it doesn't solve over a symbolic index box like :class:`OperatorSum`
        does."""
        require_literal_indices(self.word, "a SimpleOperator")
