"""Small shared helpers with no home of their own."""

from __future__ import annotations

from typing import Sequence

import sympy as sp


def parse_indexed_string(
    expr: str,
    indices: Sequence[sp.Idx],
    state_variable: str = "u",
) -> tuple[sp.Expr, sp.IndexedBase]:
    """Parse *expr* with *state_variable* and *indices* in scope.

    The shared front half of every ``from_strings`` constructor: makes the state
    base and index names available to the parser and returns the parsed
    expression alongside the constructed :class:`sympy.IndexedBase`, ready to
    hand to a ``from_sympy``.

    Args:
        expr: The expression string, e.g. ``'(u[i - 1] + u[i + 1]) / h'``.
        indices: The :class:`sympy.Idx` symbols in scope while parsing.
        state_variable: Name of the indexed state variable in *expr*.
    """
    u_base: sp.IndexedBase = sp.IndexedBase(state_variable)  # type: ignore[assignment]
    local_dict: dict[str, sp.Basic] = {
        state_variable: u_base,
        **{idx.name: idx for idx in indices},
    }
    parsed: sp.Expr = sp.sympify(expr, locals=local_dict)  # type: ignore[call-overload]
    return parsed, u_base


def require_ranges(indices: Sequence[sp.Idx]) -> None:
    """Validate that every index carries an explicit ``(lower, upper)`` range.

    Shape is derived from the index bounds, so an unranged index is ambiguous.

    Raises:
        ValueError: If any index lacks explicit bounds.
    """
    for idx in indices:
        if idx.lower is None or idx.upper is None:
            raise ValueError(
                f"Index {idx} has no explicit range; every index must be "
                "created with bounds, e.g. Idx('i', (0, N - 1))."
            )


def require_literal_indices(
    expr: sp.Expr, what: str, base: sp.IndexedBase | None = None
) -> None:
    """Validate that every subscript in *expr* is a literal integer.

    Callers read terms off as written, so ``u[k]`` would contribute nothing instead of
    standing for a sum over the sites it ranges over.

    Args:
        expr: The expression to check.
        what: How to name the caller in the message, e.g. ``"a Vector"``.
        base: Check only this base's subscripts; ``None`` checks every indexed atom.

    Raises:
        ValueError: If a checked subscript is not a literal integer.
    """
    for atom in expr.atoms(sp.Indexed):
        if base is not None and atom.base != base:
            continue
        if not all(idx.is_Integer for idx in atom.indices):
            raise ValueError(f"{what} requires literal integer indices, got {atom}")
