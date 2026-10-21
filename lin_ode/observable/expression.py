"""An observable of the raw state, ready for direct matrix projection."""

from __future__ import annotations

from typing import Any, Sequence, cast

import sympy as sp

from ..indexed_expression import IndexedExpression
from ..matrix import OperatorMatrix
from ..operators import Basis, SimpleOperator
from ..utils import parse_indexed_string, require_ranges


class ObservableExpression(IndexedExpression):
    r"""An observable of the state :math:`u`, in the same shape an ODE's
    right-hand side is: a single per-site expression, summed over one
    symbolic index.

    Written directly in terms of the state :math:`u` -- an energy, a moment,
    or any other classical observable -- so it can be projected onto a basis
    as a :class:`~lin_ode.matrix.SimpleOperator`, exactly like a
    Koopman/KvN generator.
    """

    def __init__(
        self,
        expr: sp.Expr,
        state_variable: sp.IndexedBase,
        indices: tuple[sp.Idx, ...],
    ) -> None:
        require_ranges(indices)
        super().__init__(expr, state_variable, indices)

    @classmethod
    def from_sympy(
        cls,
        expr: sp.Expr,
        state_variable: sp.IndexedBase,
        indices: tuple[sp.Idx, ...],
    ) -> ObservableExpression:
        r"""Build an observable directly from a SymPy expression.

        Args:
            expr: The per-site observable expression, in ``state_variable``
                and ``indices``.
            state_variable: The state array the observable is a function of
                (e.g. ``u``).
            indices: The index symbols ``expr`` is summed over, in subscript
                order.  Every index must carry an explicit range, e.g.
                ``Idx('k', (0, N - 1))``.
        """
        return cls(expr, state_variable, indices)

    @classmethod
    def from_strings(
        cls,
        expr: str,
        indices: Sequence[sp.Idx],
        state_variable: str = "u",
    ) -> ObservableExpression:
        r"""Build an observable by parsing a single expression string.

        Args:
            expr: The per-site observable expression, e.g.
                ``'m * omega**2 * u[k]**2 / 2'``.
            indices: The index symbols ``expr`` is summed over, in subscript
                order.  Every index must carry an explicit range, e.g.
                ``Idx('k', (0, N - 1))``.
            state_variable: Name of the indexed state variable in ``expr``.
                Defaults to ``'u'``.
        """
        parsed, u_base = parse_indexed_string(expr, indices, state_variable)
        return cls(parsed, u_base, tuple(indices))

    def get_matrix(self, basis: Basis[Any]) -> OperatorMatrix:
        r"""Project this observable onto *basis*, returning an
        :class:`~lin_ode.matrix.OperatorMatrix`.

        Lifts this observable to a :class:`~lin_ode.operators.SimpleOperator` (the KvN
        ``_to_op_algebra`` step) and projects it onto *basis*, returning an
        :class:`~lin_ode.matrix.OperatorMatrix` that builds blocks lazily on demand when
        they are extracted via ``__getitem__``. The observable has no derivative operator
        in it, so the ``diff_base`` passed to :class:`~lin_ode.operators.SimpleOperator`
        is a placeholder that is never matched.

        Args:
            basis: The function basis to project onto.
        """
        diff_base = cast(sp.IndexedBase, sp.IndexedBase("K", commutative=False))
        return OperatorMatrix(basis, SimpleOperator(self, diff_base))
