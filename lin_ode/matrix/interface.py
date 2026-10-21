"""Matrix types for a linearized operator projected onto a basis."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Generic,
    cast,
    overload,
    Protocol,
    Sequence,
    TypeVar,
)

import sympy as sp

from lin_ode import serialization as _ser
from lin_ode.basis.basis_set import TruncatedBasisSet
from lin_ode.basis.projection import IncompatibleBasisError
from lin_ode.matrix.builder import build_matrix
from lin_ode.operators.linear_operator import LinearOperator

if TYPE_CHECKING:
    from lin_ode.operators.interface import Basis

#: Type variable for a basis-specific specification of a single basis function;
#: ranges of basis functions are described as ``Sequence[IndexSpec]``
IndexSpec = TypeVar("IndexSpec", contravariant=True)


class MatrixLike(Protocol):
    r"""A materialised matrix block, e.g. a :class:`numpy.ndarray` or
    :class:`sympy.Matrix`."""

    @property
    def shape(self) -> tuple[int, int]: ...


#: A single matrix element, as produced by :func:`~lin_ode.matrix.builder.build_matrix`
#: for (int, int) input
MatrixElement = complex | sp.Expr


class OperatorMatrix(Generic[IndexSpec]):
    r"""Lazy matrix view of a :class:`~lin_ode.operators.LinearOperator` projected onto a
    basis.

    Matrix blocks are computed on demand in :meth:`__getitem__` by delegating index
    conversion to the :class:`~lin_ode.operators.Basis` and then calling
    :func:`~lin_ode.matrix.builder.build_matrix` on the stored operator.

    The ``IndexSpec`` type parameter is inherited from the
    :class:`~lin_ode.operators.Basis` and determines how single basis functions are
    named in :meth:`__getitem__`; ranges are ``Sequence[IndexSpec]``.

    Args:
        basis: The function basis that defines the multi-index space and
            index conversion logic.
        operator: The :class:`~lin_ode.operators.LinearOperator` whose matrix blocks
            :func:`~lin_ode.matrix.builder.build_matrix` materialises.
    """

    def __init__(
        self,
        basis: Basis[IndexSpec],
        operator: LinearOperator,
    ) -> None:
        if not operator.compatible_with_basis(basis):
            method = operator.method_name or type(operator).__name__
            raise IncompatibleBasisError(
                f"{method} requires an L2(R^N) basis, but "
                f"{type(basis).__name__} is not in L2, so the expansion is undefined"
            )
        self._basis = basis
        self._operator = operator

    def dump_json(self, indent: int | None = None) -> str:
        """Return this matrix as a JSON string.

        Serializes the source inputs ``(basis, operator)``. This matrix is a lazy view.
        It recomputes its blocks on demand after reconstruction.

        Raises:
            NotImplementedError: If the basis is not a :class:`TruncatedBasisSet`. That is
                the only basis family with serialization. The ``Basis`` protocol has none.
        """
        if not isinstance(self._basis, TruncatedBasisSet):
            raise NotImplementedError(
                f"{type(self._basis).__name__} is not JSON-serializable"
            )
        return _ser.json_dumps(
            {
                "basis": self._basis.to_json_data(),
                "operator": self._operator.to_json_data(),
            },
            indent,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "OperatorMatrix":
        """Rebuild from a string produced by :meth:`dump_json`."""
        json_data = _ser.json_loads(json_str)
        basis = cast("Basis[IndexSpec]", TruncatedBasisSet.from_json_data(json_data["basis"]))
        return cls(basis, LinearOperator.from_json_data(json_data["operator"]))

    @property
    def basis(self) -> Basis[IndexSpec]:
        """The function basis whose multi-index space defines the row/column
        structure."""
        return self._basis

    @overload
    def __getitem__(
        self,
        indices: tuple[IndexSpec, IndexSpec],
    ) -> MatrixElement: ...

    @overload
    def __getitem__(
        self,
        indices: tuple[Sequence[IndexSpec], Sequence[IndexSpec] | IndexSpec],
    ) -> MatrixLike: ...

    @overload
    def __getitem__(
        self,
        indices: tuple[Sequence[IndexSpec] | IndexSpec, Sequence[IndexSpec]],
    ) -> MatrixLike: ...

    def __getitem__(
        self,
        indices: tuple[
            IndexSpec | Sequence[IndexSpec], IndexSpec | Sequence[IndexSpec]
        ],
    ) -> MatrixLike | MatrixElement:
        """Extract a matrix block or a single element from the linear operator.

        Args:
            indices: Tuple of (row_indices, col_indices) where each names a
                single basis function (``IndexSpec``) or a range of them
                (``Sequence[IndexSpec]``).

        Returns:
            :class:`MatrixLike` block — or, when both axes name a single basis
            function, the single :data:`MatrixElement` at that position.
        """
        row_ids, col_ids = indices
        row_spec = self._basis.flat_indices_of_structured_indices(row_ids)
        col_spec = self._basis.flat_indices_of_structured_indices(col_ids)
        return build_matrix(self._operator, row_spec, col_spec, self._basis)
