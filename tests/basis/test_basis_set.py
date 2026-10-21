"""Flat-index API of ``BasisSet``, on Carleman as the concrete carrier."""

from math import comb

from pytest import mark, raises

from lin_ode.basis import CarlemanBasis


@mark.parametrize("shape, n_sites", [(4, 4), ((4, 4), 16)], ids=["chain", "grid"])
def test_flat_index_round_trips_through_both_lookups(
    shape: tuple[int, ...], n_sites: int
) -> None:
    """The basis holds every multiset of size <= 2 over its sites, C(n + 2, n) of them;
    asserting that count first is what stops the loop below passing vacuously."""
    basis = CarlemanBasis(shape, 2)

    assert len(basis) == comb(n_sites + 2, n_sites)
    for k in range(len(basis)):
        structured = basis.structured_idx_of_flat_index(k)
        assert structured is not None  # k < len(basis), so every index resolves
        assert basis.flat_index_of_structured_index(structured) == k, structured
        assert basis.flat_index_of_basis_function(basis[structured]) == k, structured


def test_out_of_basis_lookup_is_none_but_raises_in_batch() -> None:
    """Single lookups return None for out-of-basis indices; batch API raises
    IndexError."""
    basis = CarlemanBasis(3, 2)

    # Single lookups: (0, 3) has degree 3 > 2 so it's out of basis
    assert basis.flat_index_of_structured_index((0, 3)) is None
    assert basis.structured_idx_of_flat_index(len(basis)) is None

    # Batch mode raises for out-of-basis indices
    with raises(IndexError):
        basis.flat_indices_of_structured_indices([(0, 3)])
    with raises(IndexError):
        basis.flat_indices_of_structured_indices((0, 3))


@mark.parametrize(
    "spec, expected",
    [
        ((0, 1), 1),
        (((0, 1), (2, 1)), 6),
        ([(0, 1), ((0, 1), (2, 1))], [1, 6]),
    ],
    ids=["one-bare-factor", "one-multi-factor-state", "a-batch-of-both"],
)
def test_flat_index_lookup_returns_spec_shape(
    spec: tuple, expected: int | list[int]
) -> None:
    """A tuple is always ONE state: ``((0, 1), (2, 1))`` is ``u_0 u_2`` at flat index 6.

    It is never a batch of ``u_0`` and ``u_2**1``; only a list means a batch.
    """
    assert CarlemanBasis(3, 2).flat_indices_of_structured_indices(spec) == expected
