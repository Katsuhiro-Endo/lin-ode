"""JSON serialization primitives shared by every A-6 serializable object.

The contract: each object exposes ``dump_json(indent=None) -> str`` and a classmethod
``from_json(str) -> instance``, and ``from_json(dump_json(x))`` reconstructs ``x``.
Objects record their *source constructor inputs*, not derived state, and rebuild through
the normal constructor.

A SymPy object is stored as its ``sympy.srepr`` string and rebuilt with
``sympy.sympify``. This is lossless for expressions, for an ``Idx`` with its bounds, for
an ``IndexedBase``, and for tuples of these.
"""

from __future__ import annotations

import json
from typing import Any

import sympy as sp


def json_dumps(json_data: dict[str, Any], indent: int | None = None) -> str:
    """Serialize a plain-data dict to a JSON string."""
    return json.dumps(json_data, indent=indent)


def json_loads(json_str: str) -> dict[str, Any]:
    """Parse a JSON string back into its dict form."""
    return json.loads(json_str)


def sympy_to_str(obj: sp.Basic | tuple[sp.Basic, ...]) -> str:
    """Encode a SymPy object, or a tuple of them, as a lossless ``srepr`` string."""
    return sp.srepr(obj)


def sympy_from_str(s: str) -> Any:
    """Rebuild a SymPy object from a ``srepr`` string that :func:`sympy_to_str` made."""
    return sp.sympify(s)
