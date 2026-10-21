# pyright: reportArgumentType=false
# sympy stubs type ``sp.Idx(...)`` as ``Basic | Idx``, not ``Idx``.
"""Liouville/KvN act on a probability distribution, so only an L2 basis carries them."""

import pytest
import sympy as sp

from lin_ode.basis import (
    CarlemanBasis,
    GaussianHermiteBasis,
    IncompatibleBasisError,
    PSCarlemanBasis,
)
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import Koopman, KoopmanVonNeumann, Liouville
from lin_ode.operators.interface import Basis, LinearizationMethod
from lin_ode.ordinary_differential_system import OrdinaryDifferentialSystem


def _matrix(method: LinearizationMethod, basis: Basis) -> OperatorMatrix:
    ode = OrdinaryDifferentialSystem.from_strings("-u[i]", indices=(sp.Idx("i", (0, 2)),))
    return OperatorMatrix(basis, method.create_operator(ode))


@pytest.mark.parametrize(
    "method, basis",
    [
        (Liouville(), CarlemanBasis(n_vars=2, degree=2)),
        (KoopmanVonNeumann(), PSCarlemanBasis(n_vars=2, degree=2, pivot=1.0)),
    ],
)
def test_non_l2_basis_rejects_distribution_operator(method, basis):
    """Carleman with Liouville and PSC with KvN raise IncompatibleBasisError."""
    with pytest.raises(IncompatibleBasisError, match="not in L2"):
        _matrix(method, basis)


@pytest.mark.parametrize(
    "method, basis",
    [
        (Liouville(), GaussianHermiteBasis(n_vars=2, degree=2)),
        (Koopman(), CarlemanBasis(n_vars=2, degree=2)),
    ],
)
def test_compatible_pair_builds(method, basis):
    """Liouville accepts Gauss-Hermite and Koopman accepts Carleman."""
    assert _matrix(method, basis).basis is basis
