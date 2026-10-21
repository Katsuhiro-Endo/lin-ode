"""End-to-end Koopman-von Neumann: ``-{f, K}/2``, the two orderings averaged.

KvN moves a probability density, so it needs an L2 basis. The Carleman basis
``test_koopman.py`` uses is not one, and the pair is refused rather than computed.

``du_i/dt = u_i**2`` on 10 sites, truncated at degree 2.
"""

import pytest

from lin_ode.basis import CarlemanBasis
from lin_ode.basis.projection import IncompatibleBasisError
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import KoopmanVonNeumann

from tests.util.equations import quadratic_equation
from tests.testing import example

METHOD = KoopmanVonNeumann()
"""``L = -(f.K + K.f)/2``, the mean of the Koopman and Liouville orderings."""

SITES, DEGREE = 10, 2


@example(spec_ref=["C-1", "C-2", "D-1"])
def test_carleman_basis_is_refused() -> None:
    """KvN on a Carleman basis.

    Averaging the two orderings does not change the space they act on, so the refusal
    Liouville meets on a non-L2 basis applies unchanged.
    """
    ode = quadratic_equation(SITES)

    # --- verification ---
    with pytest.raises(IncompatibleBasisError, match="not in L2"):
        OperatorMatrix(CarlemanBasis(SITES, DEGREE), METHOD.create_operator(ode))
