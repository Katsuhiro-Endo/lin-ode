# pyright: reportArgumentType=false, reportIndexIssue=false, reportOperatorIssue=false
# sympy stubs type IndexedBase subscripting too loosely (same suppression as
# tests/matrix/test_builder.py).
r"""The objective function $g(x)$ is represented as a coefficient vector $G$, defined
by $g = \sum_\beta G_\beta \phi_\beta$, so that $g(x(t)) = G^\mathsf{T} \Phi(t)$.

The expansion asks nothing of the basis, so it is legal on Carleman, PSC and GH alike.
Only the way a coefficient is found changes.
"""

import sympy as sp

from lin_ode.basis import CarlemanBasis, GaussianHermiteBasis, PSCarlemanBasis
from lin_ode.vector import Vector

from tests.util.comparison import s_pivot
from tests.testing import example

u = sp.IndexedBase("u")

OBJECTIVE = 2 * u[0] ** 2 + 3 * u[0] * u[1] + u[0] ** 3
"""Two terms inside the degree-2 truncation, one above it."""

LABELS = [(), ((0, 1),), ((0, 2),), ((0, 1), (1, 1))]
"""The states G is read on: the vacuum, two powers of u_0, and the cross term."""

@example()
def test_objective_vector_projects_onto_carleman_basis() -> None:
    r"""Objective vector in the Carleman basis.

    Carleman states are the monomials themselves, so $G$ is the polynomial
    coefficients of $g$: $2$ on $u_0^2$ and $3$ on $u_0 u_1$. $u_0^3$ is above degree
    2, and Carleman discards it without leaving anything behind.
    """
    g_vector = Vector(OBJECTIVE, CarlemanBasis(2, 2), u)
    g_entries = [g_vector[label] for label in LABELS]
    print(g_entries)

    # --- verification ---
    assert g_entries == [0, 0, 2, 3]


@example()
def test_objective_vector_projects_onto_psc_basis() -> None:
    r"""Objective vector in the Pivot-switching Carleman basis.

    PSC folds an objective with the same rule the matrix uses, so the discarded
    $u_0^3$ returns as its Taylor expansion about the pivot $s$,
    $s^3 - 3 s^2 u_0 + 3 s u_0^2$. That lands on top of Carleman's coefficients: the
    $u_0^2$ entry becomes $2 + 3 s$, and the vacuum and $u_0$ entries turn non-zero.
    The same $g$ therefore leaves a different $G$ on the two bases.
    """
    g_vector = Vector(OBJECTIVE, PSCarlemanBasis(2, 2, s_pivot), u)
    g_entries = [sp.simplify(g_vector[label]) for label in LABELS]
    print(g_entries)

    # --- verification ---
    assert g_entries == [s_pivot**3, -3 * s_pivot**2, 3 * s_pivot + 2, 3]


@example()
def test_objective_vector_projects_onto_gauss_hermite_basis() -> None:
    r"""Objective vector in the Gauss-Hermite basis.

    Gauss-Hermite states are not monomials, so a coefficient is the integral $\langle
    \psi_a, g \rangle$ over both sites. Each site contributes a factor $\pi^{1/4}$, so
    $\sqrt{\pi}$ runs through the whole vector and no entry is a coefficient of $g$.
    """
    g_vector = Vector(OBJECTIVE, GaussianHermiteBasis(2, 2), u)
    g_entries = [sp.simplify(g_vector[label]) for label in LABELS]
    print(g_entries)

    # --- verification ---
    assert g_entries == [
        4 * sp.sqrt(sp.pi),
        6 * sp.sqrt(2) * sp.sqrt(sp.pi),
        10 * sp.sqrt(2) * sp.sqrt(sp.pi),
        12 * sp.sqrt(sp.pi),
    ]
