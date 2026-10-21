"""A dense sympy matrix with JSON serialization.

:meth:`OperatorMatrix.__getitem__` returns a block of this type. It behaves as a
:class:`sympy.MutableDenseMatrix` and adds ``dump_json`` and ``from_json``.

Implementation note (for contributors): each entry is a sympy expression, so it is
stored as its ``srepr`` string and rebuilt with ``sympify``. This backs D-1 in the A-6
JSON contract.
"""

from __future__ import annotations

import json

import sympy as sp

from lin_ode import serialization as _ser


class Matrix(sp.MutableDenseMatrix):
    """A ``sympy`` dense matrix with ``dump_json`` and ``from_json``."""

    def dump_json(self, indent: int | None = None) -> str:
        """Return this matrix as a JSON string.

        The string holds a list of rows. Each entry is a sympy ``srepr`` string.
        """
        rows = [[_ser.sympy_to_str(entry) for entry in row] for row in self.tolist()]
        return json.dumps(rows, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Matrix":
        """Rebuild the matrix from a string that :meth:`dump_json` produced."""
        rows = json.loads(json_str)
        return cls([[_ser.sympy_from_str(entry) for entry in row] for row in rows])
