"""Matrix of an operator projected onto a basis (see :mod:`lin_ode.matrix.builder`)."""

from lin_ode.matrix.dense import Matrix
from lin_ode.matrix.interface import (
    MatrixElement,
    MatrixLike,
    OperatorMatrix,
)
from lin_ode.matrix.builder import build_matrix

__all__ = [
    "Matrix",
    "MatrixElement",
    "MatrixLike",
    "OperatorMatrix",
    "build_matrix",
]
