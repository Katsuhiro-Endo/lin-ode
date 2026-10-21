# pyright: reportAttributeAccessIssue=false
# MatrixBuilder's return type includes non-Matrix branches, so `.T` on a block
# that's actually a Matrix looks unsafe to pyright.
"""Liouville is meant to be Koopman's adjoint operator for density transport: its matrix
should equal Koopman's transpose.

Liouville acts on a probability distribution, so it needs an L2 basis, and
``OperatorMatrix`` enforces that. The one L2 basis here is Gaussian-Hermite (GH), whose
orthonormal states make transpose adjoint. Carleman and PSC are rejected.
"""

import pytest

from lin_ode.basis import (
    CarlemanBasis,
    GaussianHermiteBasis,
    IncompatibleBasisError,
    PSCarlemanBasis,
)
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import Koopman, Liouville

from tests.util.equations import quadratic_equation
from tests.util.comparison import s_pivot
from tests.testing import example

LABELS = [(), ((3, 1),), ((3, 2),)]  # vacuum, then modes 1 and 2 at site 3


class TestLiouvilleAcrossBases:
    @example(spec_ref=["B-4-3", "B-5-3"])
    def test_liouville_on_hermite_functions_is_koopmans_transpose(self) -> None:
        """Liouville operator matrix in the Gaussian-weighted Hermite basis.

        Liouville is $-\\sum_i \\partial_{u_i} f_i$ and Koopman is
        $\\sum_i f_i \\partial_{u_i}$, with $f_i = u_i^2$ here. GH is orthonormal, so
        transpose is adjoint: the Liouville block equals the transpose of the Koopman
        block.
        """
        ode = quadratic_equation(6)
        koopman = OperatorMatrix(
            GaussianHermiteBasis(6, 2), Koopman().create_operator(ode)
        )
        liouville = OperatorMatrix(
            GaussianHermiteBasis(6, 2), Liouville().create_operator(ode)
        )
        koopman_block = koopman[LABELS, LABELS]
        liouville_block = liouville[LABELS, LABELS]
        print(f"Liouville sub-matrix on {LABELS} =\n{liouville_block}")

        # --- verification ---
        assert liouville_block == koopman_block.T

    @example(spec_ref=["B-4-3", "B-5-1", "B-5-2"])
    @pytest.mark.parametrize("basis_name", ["Carleman", "PSC"])
    def test_liouville_with_non_l2_basis_is_illegal(self, basis_name: str) -> None:
        """Liouville operator matrix in a basis outside L2 is illegal.

        Liouville acts on a probability distribution, so it needs an L2 basis. Carleman
        and PSC are not in L2, so ``OperatorMatrix`` raises instead of returning a
        matrix.
        """
        op = Liouville().create_operator(quadratic_equation(6))
        non_l2 = (
            CarlemanBasis(6, 2)
            if basis_name == "Carleman"
            else PSCarlemanBasis(6, 2, s_pivot)
        )

        # --- verification ---
        with pytest.raises(IncompatibleBasisError, match="not in L2"):
            OperatorMatrix(non_l2, op)
