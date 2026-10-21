# pyright: reportGeneralTypeIssues=false, reportArgumentType=false
# sympy stubs do not read a state's args as iterable, and type get_n's site as an
# Expr rather than the plain int used here.
"""A known point $x_0$ needs no projection: the initial vector $y_0$ is each basis
function evaluated at $x_0$.

That asks nothing of the basis, so it is legal on Carleman, PSC and GH alike.
"""

import math

import pytest
import sympy as sp

from lin_ode.basis import CarlemanBasis, GaussianHermiteBasis, PSCarlemanBasis

from tests.testing import example, helper

X0 = (0.3, 0.7, 1.1)
"""The known point, one coordinate per site."""

PIVOT = 0.5
"""The PSC pivot, the same at every site."""

LABELS = [(), ((0, 1),), ((1, 1),), ((0, 2),), ((0, 1), (1, 1))]
"""The states y_0 is read on: the vacuum, two single sites, a square, and a product."""

@helper
def psi(n: int, x: float) -> float:
    """Normalised Hermite function $\\psi_n(x)$, the GH factor at one site."""
    norm = sp.sqrt(2**n * sp.factorial(n) * sp.sqrt(sp.pi))
    return float(sp.hermite(n, x) * sp.exp(-(x**2) / 2) / norm)


@example()
def test_carleman_state_evaluates_to_monomial_at_x0() -> None:
    """Fixed initial value in the Carleman basis.

    A Carleman state is a monomial, so an entry is a product of powers of $x_0$. The
    five labels give $1$, $0.3$, $0.7$, $0.3^2$ and $0.3 \\times 0.7$.
    """
    basis = CarlemanBasis(3, 2)
    y0_entries = [
        math.prod(X0[int(site)] ** int(power) for site, power in basis[label].args)
        for label in LABELS
    ]
    print(y0_entries)

    # --- verification ---
    assert y0_entries == pytest.approx([1.0, 0.3, 0.7, 0.09, 0.21])


@example()
def test_psc_state_evaluates_like_carleman_at_x0() -> None:
    """Fixed initial value in the PSC basis.

    PSC holds the same monomial states as Carleman, so $y_0$ is the same vector. The
    pivot enters only through truncation, which sits in the generator, not in $y_0$.
    That is what lets a run switch the pivot and keep the state vector it already has.
    """
    basis = PSCarlemanBasis(3, 2, sp.Float(PIVOT))
    y0_entries = [
        math.prod(X0[int(site)] ** int(power) for site, power in basis[label].args)
        for label in LABELS
    ]
    print(y0_entries)

    # --- verification ---
    assert y0_entries == pytest.approx([1.0, 0.3, 0.7, 0.09, 0.21])


@example()
def test_gauss_hermite_state_evaluates_to_hermite_function_at_x0() -> None:
    """Fixed initial value in the Gauss-Hermite basis.

    A GH state is a product of normalised Hermite functions $\\psi_n$, one per site.
    Every site contributes, including those the label leaves at mode 0, because
    $\\psi_0$ is a Gaussian rather than $1$. So no entry is $1$, not even the vacuum.
    """
    basis = GaussianHermiteBasis(3, 2)
    y0_entries = [
        math.prod(psi(int(basis[label].get_n(site)), x) for site, x in enumerate(X0))
        for label in LABELS
    ]
    print(y0_entries)

    # --- verification ---
    assert y0_entries == pytest.approx(
        [0.173159, 0.073465, 0.171418, -0.100402, 0.072727], abs=1e-6
    )
