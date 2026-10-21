# pyright: reportAttributeAccessIssue=false
# MatrixBuilder's return type includes non-Matrix branches, so `.T` on a block
# that's actually a Matrix looks unsafe to pyright.
"""KvN is designed to be anti-Hermitian: the generator of a norm-preserving (unitary)
evolution on a genuine Hilbert space.

KvN transports an amplitude, so it needs an L2 basis, and ``OperatorMatrix`` enforces
that. The one L2 basis here is Gaussian-Hermite (GH), whose orthonormal states make
the matrix anti-symmetric. Carleman and PSC are rejected.
"""

import pytest
import sympy as sp

from lin_ode.basis import (
    CarlemanBasis,
    GaussianHermiteBasis,
    IncompatibleBasisError,
    PSCarlemanBasis,
)
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import KoopmanVonNeumann

from tests.util.equations import quadratic_equation
from tests.util.comparison import s_pivot
from tests.testing import example

LABELS = [(), ((3, 1),), ((3, 2),)]  # vacuum, then modes 1 and 2 at site 3


class TestKvnAcrossBases:
    @example(spec_ref=["B-4-2", "B-5-3"])
    def test_kvn_on_hermite_functions_is_exactly_antisymmetric(self) -> None:
        """KvN operator matrix in the Gaussian-weighted Hermite basis.

        KvN is the symmetrized Koopman generator
        $-\\frac{1}{2} \\sum_i (f_i \\partial_{u_i} + \\partial_{u_i} f_i)$, with
        $f_i = u_i^2$ here. The block equals minus its own transpose.
        """
        op = KoopmanVonNeumann().create_operator(quadratic_equation(6))
        matrix = OperatorMatrix(GaussianHermiteBasis(6, 2), op)
        block = matrix[LABELS, LABELS]
        print(f"sub-matrix on {LABELS} =\n{block}")

        # --- verification ---
        assert block == sp.Matrix(
            [[0, -sp.sqrt(2) / 4, 0], [sp.sqrt(2) / 4, 0, -1], [0, 1, 0]]
        )
        assert block == -block.T

    @example(spec_ref=["B-4-2", "B-5-1", "B-5-2"])
    @pytest.mark.parametrize("basis_name", ["Carleman", "PSC"])
    def test_kvn_with_non_l2_basis_is_illegal(self, basis_name: str) -> None:
        """KvN operator matrix in a basis outside L2 is illegal.

        KvN transports an amplitude, so it needs an L2 basis. Carleman and PSC are not
        in L2, so ``OperatorMatrix`` raises instead of returning a matrix.
        """
        op = KoopmanVonNeumann().create_operator(quadratic_equation(6))
        non_l2 = (
            CarlemanBasis(6, 2)
            if basis_name == "Carleman"
            else PSCarlemanBasis(6, 2, s_pivot)
        )

        # --- verification ---
        with pytest.raises(IncompatibleBasisError, match="not in L2"):
            OperatorMatrix(non_l2, op)
