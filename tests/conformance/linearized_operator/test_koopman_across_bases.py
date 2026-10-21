"""Koopman acts algebraically on the basis functions themselves, so it does not need an
L2 basis: legal on Carleman, Pivot Switching Carleman (PSC), and Gaussian-Hermite (GH)
alike."""

import sympy as sp

from lin_ode.basis import CarlemanBasis, GaussianHermiteBasis, PSCarlemanBasis
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import Koopman

from tests.util.equations import quadratic_equation
from tests.testing import example

LABELS = [(), ((3, 1),), ((3, 2),)]  # vacuum, u_3, u_3**2


@example(spec_ref=["B-4-1", "B-5-1"])
def test_koopman_on_carleman_raises_power_at_acting_site() -> None:
    """Koopman operator matrix in the Carleman basis.

    Carleman states are the monomials themselves. With $f_i = u_i^2$ here,
    $L = \\sum_i f_i \\partial_{u_i}$ raises the power at each site: $L u_3 = u_3^2$,
    i.e. $\\langle u_3^2| L |u_3\\rangle = 1$. The other columns are 0, because $L$ maps
    the constant to 0 and $L u_3^2 = 2 u_3^3$ is truncated away.
    """
    op = Koopman().create_operator(quadratic_equation(6))
    matrix = OperatorMatrix(CarlemanBasis(6, 2), op)
    block = matrix[LABELS, LABELS]
    print(f"sub-matrix on {LABELS} =\n{block}")

    # --- verification ---
    assert block == sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 1, 0]])


@example(spec_ref=["B-4-1", "B-5-2"])
def test_koopman_on_psc_folds_into_pivot_polynomials() -> None:
    """Koopman operator matrix in the Pivot-switching Carleman basis.

    Same states as Carleman, but truncation folds rather than drops.
    $L = \\sum_i f_i \\partial_{u_i}$, with $f_i = u_i^2$ here, maps $u_3^2$ to the
    out-of-basis $2 u_3^3$, so PSC replaces $u_3^3$ by its Taylor expansion about the
    pivot $s$, $s^3 - 3 s^2 u_3 + 3 s u_3^2$. The other two columns match Carleman,
    because their images $0$ and $u_3^2$ are in the basis.
    """
    s = sp.Symbol("s")
    op = Koopman().create_operator(quadratic_equation(6))
    matrix = OperatorMatrix(PSCarlemanBasis(6, 2, s), op)
    block = matrix[LABELS, LABELS]
    print(f"sub-matrix on {LABELS} =\n{block}")

    # --- verification ---
    assert block == sp.Matrix([[0, 0, 2 * s**3], [0, 0, -6 * s**2], [0, 1, 6 * s]])


@example(spec_ref=["B-4-1", "B-5-3"])
def test_koopman_on_hermite_carries_recursion_coefficients() -> None:
    """Koopman operator matrix in the Gaussian-weighted Hermite basis.

    States are Hermite functions rather than monomials, so $L = \\sum_i f_i
    \\partial_{u_i}$, with $f_i = u_i^2$ here, both raises and lowers the mode, and its
    entries combine recursion coefficients where Carleman has a bare $1$. The vacuum
    is a Gaussian, so $L$ does not map it to 0. Truncation does not empty the mode-2
    column: $L$ changes parity, which forbids the even rows, and the two terms
    reaching mode 1 cancel.
    """
    op = Koopman().create_operator(quadratic_equation(6))
    matrix = OperatorMatrix(GaussianHermiteBasis(6, 2), op)
    block = matrix[LABELS, LABELS]
    print(f"sub-matrix on {LABELS} =\n{block}")

    # --- verification ---
    assert block == sp.Matrix(
        [
            [0, -sp.sqrt(2) / 4, 0],
            [-3 * sp.sqrt(2) / 4, 0, 0],
            [0, -2, 0],
        ]
    )
