# pyright: reportArgumentType=false, reportIndexIssue=false
# pyright: reportOperatorIssue=false, reportCallIssue=false
# sympy stubs type IndexedBase subscripting too loosely, and do not match simplify
# on a matrix block (same suppressions as tests/matrix/test_builder.py).
r"""$g(x)$ can also be represented as a matrix $\langle a|g|b \rangle$.

That is the form used with KvN's Hilbert-space state, where the observable acts as an
operator rather than a vector.
"""

import sympy as sp

from lin_ode import ObservableExpression
from lin_ode.basis import GaussianHermiteBasis

from tests.testing import example


@example()
def test_objective_matrix_on_gauss_hermite_basis() -> None:
    r"""Objective matrix in the Gauss-Hermite basis.

    Multiplying by $u_0^2$ changes the mode by $0$ or $\pm 2$, so in this block only
    modes 0 and 2 couple. The entries are inner products, so the block is symmetric.
    """
    u = sp.IndexedBase("u")
    k = sp.Idx("k", (0, 3))
    g = ObservableExpression.from_sympy(u[0] ** 2, u, (k,))
    matrix = g.get_matrix(GaussianHermiteBasis(4, 2))
    labels = [(), ((0, 1),), ((0, 2),)]
    block = sp.simplify(matrix[labels, labels])
    print(block)

    # --- verification ---
    expected = sp.Matrix(
        [
            [sp.Rational(1, 2), 0, sp.sqrt(2) / 2],
            [0, sp.Rational(3, 2), 0],
            [sp.sqrt(2) / 2, 0, sp.Rational(5, 2)],
        ]
    )
    assert block == expected
    assert block == block.T


@example()
def test_objective_matrix_with_non_orthogonal_basis_is_illegal() -> None:
    r"""The objective matrix does not check L2.

    ``get_matrix`` builds a matrix on any basis and checks nothing itself.
    ``OperatorMatrix`` rejects a basis outside L2 when it builds the KvN generator
    matrix.
    """
