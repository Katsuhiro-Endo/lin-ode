# pyright: reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false
# sympy stubs mistype the label subscripts and the block as having no __getitem__.
"""End-to-end Koopman: ``f_i(U) K_i``, the derivative applied after f.

Koopman is the one of the three methods a Carleman basis carries; ``test_liouville.py``
and ``test_koopman_von_neumann.py`` show the same ODE and basis refused instead.

``du_i/dt = u_i**2`` on 10 sites, truncated at degree 2.
"""

import sympy as sp

from lin_ode.basis import CarlemanBasis
from lin_ode.matrix import OperatorMatrix
from lin_ode.operators import Koopman

from tests.util.comparison import list_basis_labels
from tests.util.equations import quadratic_equation, quadratic_equation_2d
from tests.testing import example

METHOD = Koopman()
"""``L = sum_i f_i(u) d/du_i``, so only the site being lowered contributes."""

SITES, DEGREE = 10, 2


@example(spec_ref=["C-1", "C-2", "D-1"])
def test_raising_the_acted_site() -> None:
    """Raising the acted site

    ``L u_3 = u_3**2``, so the raise carries the coefficient 1.
    """
    ode = quadratic_equation(SITES)
    lin_ode_matrix = OperatorMatrix(
        CarlemanBasis(SITES, DEGREE), METHOD.create_operator(ode)
    )

    # --- verification ---
    assert lin_ode_matrix[((3, 2),), ((3, 1),)] == 1


def test_diagonal_is_zero_on_a_quadratic_right_hand_side() -> None:
    """``L`` changes the degree by one here, so no state maps to itself."""
    ode = quadratic_equation(SITES)
    lin_ode_matrix = OperatorMatrix(
        CarlemanBasis(SITES, DEGREE), METHOD.create_operator(ode)
    )

    # --- verification ---
    assert lin_ode_matrix[((3, 1),), ((3, 1),)] == 0
    assert lin_ode_matrix[(), ()] == 0


def test_raising_a_site_the_ket_never_occupied() -> None:
    """Koopman lowers first, and ``K|vac> = 0``, so nothing reaches an empty site."""
    ode = quadratic_equation(SITES)
    lin_ode_matrix = OperatorMatrix(
        CarlemanBasis(SITES, DEGREE), METHOD.create_operator(ode)
    )

    # --- verification ---
    assert lin_ode_matrix[((5, 1),), ()] == 0


def test_one_site_block_holds_the_states_on_that_site() -> None:
    """A block is sliced with the same ``(site, power)`` labels an entry is read
    with."""
    basis = CarlemanBasis(SITES, DEGREE)
    lin_ode_matrix = OperatorMatrix(
        basis, METHOD.create_operator(quadratic_equation(SITES))
    )
    on_site_3 = [
        label for label in list_basis_labels(basis) if {s for s, _ in label} <= {3}
    ]

    block = lin_ode_matrix[on_site_3, on_site_3]

    # --- verification ---
    assert block.shape == (3, 3)  # |vac>, |u_3>, |u_3**2>


@example(spec_ref=["A-2", "C-2", "D-1"], level="advanced")
def test_entry_costs_the_same_at_any_grid_size() -> None:
    """One entry at any grid size.

    One entry off a 10**6 x 10**6 grid, so 10**12 sites resolve at once. The
    contributing index is solved for, never enumerated.
    """
    big = 1_000_000
    site = sp.Tuple(5, 5)
    ode = quadratic_equation_2d(big)
    lin_ode_matrix = OperatorMatrix(
        CarlemanBasis((big, big), DEGREE), METHOD.create_operator(ode)
    )

    # --- verification ---
    assert lin_ode_matrix[((site, 2),), ((site, 1),)] == 1
