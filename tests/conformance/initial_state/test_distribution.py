# pyright: reportArgumentType=false, reportIndexIssue=false, reportOperatorIssue=false
# sympy stubs type IndexedBase subscripting too loosely (same suppression as
# tests/matrix/test_builder.py).
r"""A distribution $\rho_0(x)$ is projected onto the basis as an inner product, $\langle
\psi_a, \rho_0 \rangle$.

That inner product is the expansion coefficient of $\rho_0$ only when the states are
orthonormal. Gauss-Hermite is the one such basis here, and the one in L2 that Liouville
and KvN need to transport a density.
"""

import sympy as sp

from lin_ode.basis import GaussianHermiteBasis
from lin_ode.vector import Vector

from tests.testing import example


@example()
def test_distribution_projects_onto_gauss_hermite_basis() -> None:
    r"""Initial distribution in the Gauss-Hermite basis.

    Each entry is the integral $\langle \psi_a, \rho_0 \rangle$ over both sites. For
    $\rho_0 = u_0^2 + u_1^2$ the odd state $\psi_1$ gives $0$, and the two sites enter
    alike, so their degree-2 coefficients match.
    """
    u = sp.IndexedBase("u")
    rho0 = Vector(u[0] ** 2 + u[1] ** 2, GaussianHermiteBasis(2, 2), u)
    labels = [(), ((0, 1),), ((0, 2),), ((1, 2),)]
    y0_entries = [sp.simplify(rho0[label]) for label in labels]
    print(y0_entries)

    # --- verification ---
    assert y0_entries == [
        4 * sp.sqrt(sp.pi),
        0,
        6 * sp.sqrt(2) * sp.sqrt(sp.pi),
        6 * sp.sqrt(2) * sp.sqrt(sp.pi),
    ]


@example()
def test_distribution_with_non_orthogonal_basis_is_illegal() -> None:
    r"""Projecting a distribution does not check L2.

    ``Vector`` projects onto any basis, so on Carleman or PSC a distribution lands as
    plain polynomial coefficients. It needs no check of its own: ``OperatorMatrix``
    rejects a basis outside L2 when it builds the Liouville or KvN generator matrix.
    """
