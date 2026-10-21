"""The flat-index bijection underneath every basis: ``state_to_k`` and ``k_to_state``.

Degree-graded and site-ascending, so index 0 is the vacuum and 1..n_vars are the single
sites; anything past the truncation has no index at all.
"""

from math import comb

from pytest import mark

from lin_ode.indexing import k_to_state, state_to_k


def test_flat_index_zero_and_vacuum_state_map_to_each_other() -> None:
    """The degree grading puts the empty state first, so its flat index is 0."""
    assert state_to_k((), 3, 2) == 0
    assert k_to_state(0, 3, 2) == ()


@mark.parametrize(
    "site, expected_k", [(0, 1), (1, 2), (2, 3)], ids=["site-0", "site-1", "site-2"]
)
def test_degree_one_monomial_takes_its_sites_flat_position(
    site: int, expected_k: int
) -> None:
    """The degree-1 block runs 1..n_vars in ascending site order, straight after the
    vacuum at 0."""
    assert state_to_k(((site, 1),), 3, 2) == expected_k
    assert k_to_state(expected_k, 3, 2) == ((site, 1),)


def test_out_of_basis_index_is_none_both_ways() -> None:
    """A degree-3 state and a flat index at C(5, 3) both sit one step past the basis.

    Absence is reported rather than raised.
    """
    assert state_to_k(((0, 3),), 3, 2) is None
    assert k_to_state(comb(3 + 2, 3), 3, 2) is None


def test_every_flat_index_round_trips_at_12_sites() -> None:
    """12 sites on purpose: past 9, a string sort would put site 10 before site 2.

    The round trip would then break.
    """
    n_vars, degree = 12, 3

    for k in range(comb(n_vars + degree, n_vars)):
        state = k_to_state(k, n_vars, degree)
        assert state is not None
        assert state_to_k(state, n_vars, degree) == k, state
