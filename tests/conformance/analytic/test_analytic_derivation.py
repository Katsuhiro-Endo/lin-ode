# pyright: reportArgumentType=false, reportIndexIssue=false, reportOperatorIssue=false
# sympy stubs type Indexed subscripting loosely and IndexedBase as Basic, so u[i] and
# every call taking it read as errors; the noise is irreducible here.
"""The generator against a closed-form derivation, plus the cases solvable end to end.

For ``L = sum_i f_i d/du_i`` on a Carleman monomial ``phi_alpha = prod_s u_s**alpha_s``,

    L phi_alpha = sum_i alpha_i f_i phi_alpha / u_i.

With ``f_i = D/h**2 (u_{i-1} - 2 u_i + u_{i+1}) + Q u_i + R u_i**2`` every entry is one
of four cases; see ``closed_form_entry``. That formula shares no code with the library.
"""

import numpy as np
import pytest
import sympy as sp
from scipy.linalg import expm

from lin_ode.basis import CarlemanBasis
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import Koopman
from lin_ode.ordinary_differential_system import OrdinaryDifferentialSystem
from tests.util.comparison import list_basis_labels
from tests.testing import example, helper
from lin_ode.basis import TruncatedBasisSet

D, Q, R, h = sp.symbols("D Q R h")


@helper
def periodic_fisher(
    n: int, coefficients: tuple | None = None
) -> OrdinaryDifferentialSystem:
    """KPP-Fisher on an n-site ring, ``u[-1] = u[n-1]`` written with ``Mod``."""
    u, i = sp.IndexedBase("u"), sp.Idx("i", (0, n - 1))
    left, right = u[sp.Mod(i - 1, n)], u[sp.Mod(i + 1, n)]
    rhs = D * (left - 2 * u[i] + right) / h**2 + Q * u[i] + R * u[i] ** 2
    return OrdinaryDifferentialSystem.from_sympy(rhs.subs(coefficients or {}), u, (i,))


def occupation(state: tuple, n: int) -> tuple[int, ...]:
    """``((0, 2), (3, 1))`` -> the length-n vector of powers per site."""
    counts = [0] * n
    for site, power in state:
        counts[int(site)] += int(power)
    return tuple(counts)


def closed_form_entry(bra: tuple, ket: tuple, n: int, degree: int) -> sp.Expr:
    """``<bra|L|ket>`` from the derivation in the module docstring."""
    alpha, beta = occupation(ket, n), occupation(bra, n)
    total_degree = sum(alpha)

    if beta == alpha:  # growth, and the -2 of the Laplacian
        return total_degree * (Q - 2 * D / h**2)

    entry = sp.S.Zero
    for i, power in enumerate(alpha):
        if power == 0:
            continue
        for neighbour in ((i - 1) % n, (i + 1) % n):  # diffusion bond i -> neighbour
            hopped = list(alpha)
            hopped[i] -= 1
            hopped[neighbour] += 1
            if beta == tuple(hopped):
                entry += power * D / h**2
        if total_degree < degree:  # reaction, unless truncation drops it
            raised = list(alpha)
            raised[i] += 1
            if beta == tuple(raised):
                entry += power * R
    return entry


@helper
def generator(
    ode: OrdinaryDifferentialSystem, basis: TruncatedBasisSet, states: list
) -> np.ndarray:
    """``A`` of ``dPhi/dt = A Phi``, which is the transpose of the entry matrix."""
    lin_ode_matrix = OperatorMatrix(basis, Koopman().create_operator(ode))
    return np.array(lin_ode_matrix[states, states], dtype=float).T


@helper
def scalar_riccati() -> OrdinaryDifferentialSystem:
    """``dv/dt = v - v**2``: the one-site equation a uniform ring collapses to."""
    v, j = sp.IndexedBase("v"), sp.Idx("j", (0, 0))
    return OrdinaryDifferentialSystem.from_sympy(v[j] - v[j] ** 2, v, (j,))


@helper
def powers_up_to(degree: int) -> list:
    """The single-site Carleman labels of ``1, v, v**2, ..., v**degree``."""
    return [()] + [((0, k),) for k in range(1, degree + 1)]


@helper
def scalar_carleman_at(u_0: float, t: float, degree: int) -> float:
    """``u(t)`` of ``dv/dt = v - v**2`` through the one-site Carleman system.

    Truncated at ``degree``, evolved with ``expm``, read off the degree-1 row.
    """
    matrix = generator(scalar_riccati(), CarlemanBasis(1, degree), powers_up_to(degree))
    initial = np.array([u_0**k for k in range(degree + 1)])
    return float((expm(matrix * t) @ initial)[1])


@helper
def logistic(u_0: float, t: float) -> float:
    """Exact solution of ``du/dt = u - u**2``."""
    return u_0 * np.exp(t) / (1 - u_0 + u_0 * np.exp(t))


def test_every_entry_matches_closed_form() -> None:
    """Every entry of the 4-site degree-2 Koopman matrix matches closed_form_entry."""
    n, degree = 4, 2
    basis = CarlemanBasis(n, degree)
    states = list_basis_labels(basis)
    lin_ode_matrix = OperatorMatrix(
        basis, Koopman().create_operator(periodic_fisher(n))
    )

    # one block pull, not 225 single-entry lookups: the block measures ~30x cheaper
    block = lin_ode_matrix[states, states]
    mismatches, non_zero = [], 0
    for row, bra in enumerate(states):
        for col, ket in enumerate(states):
            lin_ode_entry = block[row, col]
            if sp.simplify(lin_ode_entry - closed_form_entry(bra, ket, n, degree)) != 0:
                mismatches.append((bra, ket))
            non_zero += lin_ode_entry != 0

    assert not mismatches, f"{len(mismatches)} of {len(states) ** 2} entries disagree"
    assert non_zero == 58  # else an all-zero matrix would agree with itself vacuously


def test_linear_diffusion_reproduces_analytic_spectrum() -> None:
    """With R = 0 the linearization is exact, so the spectrum must match to round-off."""
    n = 8
    coefficients = {D: 1, Q: 1, R: 0, h: sp.Rational(1, n)}
    basis = CarlemanBasis(n, 1)
    ode = periodic_fisher(n, coefficients)
    eigenvalues = np.linalg.eigvals(
        generator(ode, basis, [((s, 1),) for s in range(n)])
    )

    analytic = [1 - 4 * n**2 * np.sin(np.pi * k / n) ** 2 for k in range(n)]
    assert np.allclose(np.sort(np.real(eigenvalues)), np.sort(analytic), atol=1e-9)


@pytest.mark.parametrize("degree", [3, 5, 8])
def test_scalar_riccati_generator_is_bidiagonal(degree: int) -> None:
    """``dPhi_n/dt = n Q Phi_n + n R Phi_{n+1}``, so A is triangular with diagonal nQ."""
    lin_ode_matrix = generator(
        scalar_riccati(), CarlemanBasis(1, degree), powers_up_to(degree)
    )

    expected = np.diag(np.arange(degree + 1, dtype=float))
    expected += np.diag(-np.arange(0, degree, dtype=float), 1)
    assert np.allclose(lin_ode_matrix, expected)
    spectrum = np.sort(np.real(np.linalg.eigvals(lin_ode_matrix)))
    assert np.allclose(spectrum, np.arange(degree + 1))


def test_truncation_error_falls_with_degree() -> None:
    """Degrees 2, 3, 5 and 8 give errors that fall in order, ending below 3e-3."""
    u_0, t = 0.3, 1.0

    errors = [
        abs(scalar_carleman_at(u_0, t, degree) - logistic(u_0, t))
        for degree in (2, 3, 5, 8)
    ]

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 3e-3


@example(spec_ref=["C-2", "D-1"], level="advanced")
def test_uniform_initial_condition_reduces_to_scalar_riccati() -> None:
    """Uniform ring, scalar Riccati

    A ring at a uniform initial value solves the scalar equation exactly.
    The stencil cancels on a constant profile, so the N-site system collapses to
    ``du/dt = Q u + R u**2``, which has a closed form. The truncated linearization must
    collapse the same way, one truncation error from the exact logistic.
    """
    n, degree, u_0, t = 3, 3, 0.3, 1.0
    coefficients = {D: 1, Q: 1, R: -1, h: sp.Rational(1, n)}

    basis = CarlemanBasis(n, degree)
    states = list_basis_labels(basis)
    lin_ode_grid = generator(periodic_fisher(n, coefficients), basis, states)
    initial = np.array(
        [float(np.prod([u_0 ** int(p) for _, p in state])) for state in states]
    )
    evolved = expm(lin_ode_grid * t) @ initial
    grid_answer = evolved[[states.index(((s, 1),)) for s in range(n)]]

    scalar_answer = scalar_carleman_at(u_0, t, degree)

    print(f"exact logistic  = {logistic(u_0, t):.12f}")
    print(f"scalar Carleman = {scalar_answer:.12f}")
    print(f"grid Carleman   = {grid_answer[0]:.12f}")

    assert np.ptp(grid_answer) < 1e-12  # stays uniform across the ring
    assert abs(grid_answer[0] - scalar_answer) < 1e-12
    assert abs(scalar_answer - logistic(u_0, t)) < 1e-1  # truncation, not error
