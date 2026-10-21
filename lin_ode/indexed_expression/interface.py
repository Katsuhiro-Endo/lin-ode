"""Multi-index symbolic indexed expression."""

from __future__ import annotations

from math import prod
from typing import Any, TypeVar, overload

import sympy as sp

from lin_ode import serialization as _ser

_TBasic = TypeVar("_TBasic", bound=sp.Basic)


def _validate_indexed_expression_inputs(
    expr: object,
    state_variable: object,
    indices: object,
) -> None:
    """Validate inputs for IndexedExpression construction.

    Args:
        expr: The symbolic expression
        state_variable: The IndexedBase representing the state array
        indices: Tuple of Idx symbols

    Raises:
        TypeError: If any input has an invalid type
        ValueError: If state_variable is not found in expr
    """
    # Validate expr type
    if not isinstance(expr, sp.Expr):
        raise TypeError(f"expr must be a SymPy Expr, got {type(expr).__name__}")

    # Validate state_variable type
    if not isinstance(state_variable, sp.IndexedBase):
        raise TypeError(
            f"state_variable must be an IndexedBase, got {type(state_variable).__name__}"
        )

    # Validate indices type
    if not isinstance(indices, tuple):
        raise TypeError(f"indices must be a tuple, got {type(indices).__name__}")

    # Validate each index is an Idx
    for i, idx in enumerate(indices):
        if not isinstance(idx, sp.Idx):
            raise TypeError(f"indices[{i}] must be an Idx, got {type(idx).__name__}")

    # Validate that state_variable appears in the expression
    state_atoms = [
        atom for atom in expr.atoms(sp.Indexed) if atom.base == state_variable
    ]
    if not state_atoms:
        raise ValueError(f"state variable {state_variable} not found in expression")


class IndexedExpression:
    r"""A symbolic expression parameterized by multi-indices and state variables.

    Wraps a SymPy expression (including :class:`sympy.Piecewise`) that represents
    dynamics or operators parameterized by one or more symbolic indices. The
    expression depends on a state array (typically :math:`u`) indexed by symbolic
    :class:`sympy.Idx` objects.

    This class provides a unified representation for indexed equations, pairing
    the expression with its state variable and index structure.

    Args:
        expr: The symbolic expression in ``state_variable`` and ``indices``.
        state_variable: The :class:`sympy.IndexedBase` representing the state array :math:`u`.
        indices: Tuple of :class:`sympy.Idx` symbols that parameterize the expression.
    """

    def __init__(
        self,
        expr: sp.Expr,
        state_variable: sp.IndexedBase,
        indices: tuple[sp.Idx, ...],
    ) -> None:
        _validate_indexed_expression_inputs(expr, state_variable, indices)
        self._expr = expr
        self._state_variable = state_variable
        self._indices = indices

    def to_json_data(self) -> dict[str, Any]:
        """Return the source inputs as a JSON-ready dict of srepr strings."""
        return {
            "expr": _ser.sympy_to_str(self._expr),
            "state_variable": _ser.sympy_to_str(self._state_variable),
            "indices": _ser.sympy_to_str(self._indices),
        }

    @classmethod
    def from_json_data(cls, json_data: dict[str, Any]) -> IndexedExpression:
        """Rebuild from the dict produced by :meth:`to_json_data`."""
        return cls(
            _ser.sympy_from_str(json_data["expr"]),
            _ser.sympy_from_str(json_data["state_variable"]),
            tuple(_ser.sympy_from_str(json_data["indices"])),
        )

    def dump_json(self, indent: int | None = None) -> str:
        """Return this expression as a JSON string."""
        return _ser.json_dumps(self.to_json_data(), indent)

    @classmethod
    def from_json(cls, json_str: str) -> IndexedExpression:
        """Rebuild from a string produced by :meth:`dump_json`."""
        return cls.from_json_data(_ser.json_loads(json_str))

    @property
    def expr(self) -> sp.Expr:
        """The underlying symbolic expression."""
        return self._expr

    @property
    def cases(self) -> list[tuple[sp.Expr, sp.Basic]]:
        """(expression, condition) pairs.

        For a :class:`sympy.Piecewise` expression returns all branches in evaluation
        order.  For any other expression returns ``[(expr, sp.true)]``.
        """
        if isinstance(self._expr, sp.Piecewise):
            return [(expr, cond) for expr, cond in self._expr.args]  # type: ignore[misc]
        return [(self._expr, sp.true)]

    @property
    def state_variable(self) -> sp.IndexedBase:
        r"""The state array :math:`u`."""
        return self._state_variable

    @property
    def indices(self) -> tuple[sp.Idx, ...]:
        """Tuple of symbolic index symbols parameterising this expression."""
        return self._indices

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Size of each index dimension (``upper - lower + 1``), or ``None`` if any index is unbounded."""
        if any(idx.lower is None or idx.upper is None for idx in self._indices):
            return None
        try:
            return tuple(
                int(idx.upper) - int(idx.lower) + 1  # type: ignore[arg-type]
                for idx in self._indices
            )
        except (TypeError, OverflowError):
            return None

    @property
    def size(self) -> int | None:
        """Total number of valid index combinations, or ``None`` if the shape is
        unknown.

        Defaults to ``prod(shape)``.
        """
        s = self.shape
        return prod(s) if s is not None else None

    @property
    def free_symbols(self) -> set[sp.Basic]:
        """Free symbols in the underlying expression."""
        return self._expr.free_symbols

    @overload
    def atoms(self) -> set[sp.Basic]: ...
    @overload
    def atoms(self, *types: type[_TBasic]) -> set[_TBasic]: ...
    def atoms(self, *types: type) -> set[sp.Basic] | set[_TBasic]:  # type: ignore[reportInvalidTypeVarUse]
        """Return atoms of the given types."""
        return self._expr.atoms(*types)
