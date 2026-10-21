"""End-to-end Liouville: ``-K_i f_i(U)``, the derivative applied before f.

Liouville moves a probability density, so it needs an L2 basis. The Carleman basis
``test_koopman.py`` uses is not one, and the pair is refused rather than computed.

``du_i/dt = u_i**2`` on 10 sites, truncated at degree 2.
"""

import pytest

from lin_ode.basis import CarlemanBasis
from lin_ode.basis.projection import IncompatibleBasisError
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import Liouville

from tests.util.equations import quadratic_equation
from tests.testing import example

METHOD = Liouville()
"""``L = -K_i f_i(u)``, which expands to ``-div f - K``."""

SITES, DEGREE = 10, 2


@example(spec_ref=["C-1", "C-2", "D-1"])
def test_carleman_basis_is_refused() -> None:
    """Liouville on a Carleman basis.

    Carleman monomials are not square-integrable, so the density Liouville propagates
    has no expansion in them and no entry is defined to read.
    """
    ode = quadratic_equation(SITES)

    # --- verification ---
    with pytest.raises(IncompatibleBasisError, match="not in L2"):
        OperatorMatrix(CarlemanBasis(SITES, DEGREE), METHOD.create_operator(ode))
