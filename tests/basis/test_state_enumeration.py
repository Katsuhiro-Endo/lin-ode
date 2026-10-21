"""Enumeration and truncation of the concrete bases.

Carleman and Gauss-Hermite differ only in their state type; the flat-index order and the
degree truncation are shared, so every case below runs against both.
"""

import sympy as sp
from pytest import mark

from lin_ode.basis import (
    CarlemanBasis,
    CarlemanState,
    GaussianHermiteBasis,
    GaussianHermiteState,
)

site = sp.Tuple

BASES = mark.parametrize(
    "basis_type, state_type",
    [(CarlemanBasis, CarlemanState), (GaussianHermiteBasis, GaussianHermiteState)],
    ids=["carleman", "gaussian-hermite"],
)

CHAIN_ORDER = [
    (),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
    ((0, 2),),
    ((0, 1), (1, 1)),
    ((0, 1), (2, 1)),
    ((1, 2),),
    ((1, 1), (2, 1)),
    ((2, 2),),
]
"""Flat order of the 3-site, degree-2 basis: degree 1 first, then degree 2."""

GRID_ORDER = [
    (),
    ((site(0, 0), 1),),
    ((site(0, 1), 1),),
    ((site(1, 0), 1),),
    ((site(1, 1), 1),),
    ((site(0, 0), 2),),
    ((site(0, 0), 1), (site(0, 1), 1)),
    ((site(0, 0), 1), (site(1, 0), 1)),
    ((site(0, 0), 1), (site(1, 1), 1)),
    ((site(0, 1), 2),),
    ((site(0, 1), 1), (site(1, 0), 1)),
    ((site(0, 1), 1), (site(1, 1), 1)),
    ((site(1, 0), 2),),
    ((site(1, 0), 1), (site(1, 1), 1)),
    ((site(1, 1), 2),),
]
"""Same order on a 2x2 grid, whose four sites themselves run row-major."""

@BASES
@mark.parametrize(
    "shape, order", [(3, CHAIN_ORDER), ((2, 2), GRID_ORDER)], ids=["chain", "grid"]
)
def test_flat_indices_enumerate_in_degree_then_site_order(
    basis_type: type, state_type: type, shape: tuple[int, ...], order: list[tuple]
) -> None:
    """The basis enumerates CHAIN_ORDER and GRID_ORDER, and every index round-trips."""
    basis = basis_type(shape, 2)
    expected = [state_type(*factors) for factors in order]

    assert len(basis) == len(expected)
    for k, state in enumerate(expected):
        structured = basis.structured_idx_of_flat_index(k)
        assert structured is not None  # k < len(basis), so every index resolves
        assert basis[structured] == state, k
        assert basis.flat_index_of_basis_function(state) == k, state
        assert basis.flat_index_of_structured_index(structured) == k, structured


@BASES
@mark.parametrize(
    "over_degree, within_degree",
    [
        (((0, 1), (2, 3)), (((30, 1), (2, 1)), ((6, 1),))),
        (
            ((site(0, 0), 1), (site(2, 2), 3)),
            (((site(300, 123), 1), (site(2, 1), 1)), ((site(6, 12), 1),)),
        ),
    ],
    ids=["chain-sites", "grid-sites"],
)
def test_truncation_drops_only_states_over_degree(
    basis_type: type, state_type: type, over_degree: tuple, within_degree: tuple
) -> None:
    """``over_degree`` totals 1 + 3 = 4 > 2 so its term goes; the other two total 2 and
    stay, however far out of range their site labels are."""
    basis = basis_type(n_vars=3, degree=2)
    kept = (
        state_type()
        + 2 * state_type(*within_degree[0])
        + 3 * state_type(*within_degree[1])
    )

    folded = (kept + state_type(*over_degree)).replace(
        basis.state_type, basis.truncation_rule
    )

    assert folded == kept


@BASES
def test_basis_declares_its_state_type(basis_type: type, state_type: type) -> None:
    """Each basis names the basis function type its states are built from."""
    assert basis_type.state_type is state_type
