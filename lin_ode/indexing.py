"""Flat-index mapping for a truncated basis (graded reverse-lexicographic).

state_to_k(state, n_vars_or_shape, degree)          -> int | None k_to_state(k,
state_type, n_vars_or_shape, degree)  -> BasisFunction | None

n_vars_or_shape is an int for 1-D sites, or a shape tuple for D-dimensional sites
(sp.Tuple labels), which are flattened row-major before ranking.
"""

from math import comb, prod
from typing import Sequence, TypeAlias

import sympy as sp

IdxSpec: TypeAlias = tuple[int | sp.Expr | sp.Tuple, sp.Expr | int]
BasisFunctionArgs: TypeAlias = tuple[IdxSpec, ...]


def get_shape_and_count(
    n_vars_or_shape: int | tuple[int, ...],
) -> tuple[tuple[int, ...], int]:
    """Normalise an int n_vars or a shape tuple to (shape, total variable count)."""
    shape = (n_vars_or_shape,) if isinstance(n_vars_or_shape, int) else n_vars_or_shape
    return shape, prod(shape)


def _site_to_flat(
    site: "int | sp.Expr | sp.Tuple | tuple[int, ...]", shape: tuple[int, ...]
) -> int:
    """Flatten a site label (scalar, sp.Tuple or plain tuple) to a row-major integer."""
    # sympy Integer is int-convertible at runtime
    axes = site if isinstance(site, (sp.Tuple, tuple)) else (site,)
    coords = [int(c) for c in axes]  # type: ignore[arg-type]
    flat = 0
    for coord, size in zip(coords, shape):
        flat = flat * size + coord
    return flat


def _flat_to_site(flat: int, shape: tuple[int, ...]) -> "sp.Tuple | int":
    """Inverse of _site_to_flat: a scalar for 1-D shapes, else an sp.Tuple."""
    if len(shape) == 1:
        return flat
    coords = []
    for size in reversed(shape):
        coords.append(flat % size)
        flat //= size
    return sp.Tuple(*reversed(coords))


def _graded_rank(sparse_alpha: Sequence[tuple[int, int]], n_vars: int) -> int:
    """Degree-grouped flat index matching the map_to_lexicographical_1D convention.

    Ordering: k=0 is the constant; then all degree-1 monomials in dim-ascending
    order (u_0=1, u_1=2, ...); then degree-2 in combinations-with-replacement
    lex order (u_0²=N+1, u_0u_1=N+2, ...), and so on.

    sparse_alpha: (dim, exponent) pairs for non-zero dimensions only (any order).
    n_vars:       N — total number of variables.

    Time: O(K²) where K = total degree, bounded by P.
    """
    K = sum(exp for _, exp in sparse_alpha)
    if K == 0:
        return 0
    offset = 1 + sum(comb(n_vars + k - 1, k) for k in range(1, K))
    indices = sorted(dim for dim, exp in sparse_alpha for _ in range(exp))
    M = n_vars + K - 1
    total = comb(M, K)
    strict_c = [indices[m] + m for m in range(K)]
    sum_d = sum(comb(M - 1 - strict_c[K - 1 - m], m + 1) for m in range(K))
    return offset + (total - 1 - sum_d)


def k_to_state(
    k: int, n_vars_or_shape: int | tuple[int, ...], degree: int
) -> BasisFunctionArgs | None:
    """Inverse of state_to_k: flat index -> BasisFunction.

    Returns None if k is out of the truncated basis (k >= C(n_vars+degree, n_vars)).
    Round-trip: state_to_k(k_to_state(k, T, N, P), N, P) == k for all valid k.
    A shape tuple yields D-dimensional sp.Tuple site labels.

    Time: O(degree x n_vars).
    """
    shape, n_vars = get_shape_and_count(n_vars_or_shape)

    if k == 0:
        return ()

    offset = 1
    for D in range(1, degree + 1):
        block = comb(n_vars + D - 1, D)
        if k < offset + block:
            local_rank = k - offset
            break
        offset += block
    else:
        return None

    # Unrank local_rank as combinations-with-replacement(D, n_vars) in lex order.
    combo: list[int] = []
    remaining = local_rank
    start = 0
    for i in range(D):
        for v in range(start, n_vars):
            r = D - i - 1
            count = comb(n_vars - v + r - 1, r)
            if remaining < count:
                combo.append(v)
                start = v
                break
            remaining -= count

    sparse: dict[sp.Tuple | int, int] = {}
    for flat_dim in combo:
        site = _flat_to_site(flat_dim, shape)
        sparse[site] = sparse.get(site, 0) + 1
    return tuple(sparse.items())  # eval canonicalises the ordering


def state_to_k(
    base_func_args: BasisFunctionArgs,
    n_vars_or_shape: int | tuple[int, ...],
    degree: int,
) -> int | None:
    """Degree-grouped flat index of a BasisFunction.

    Returns None if the total exponent exceeds degree (truncated).
    A shape tuple flattens D-dimensional sp.Tuple site labels row-major.

    Time: O(P²).
    """
    shape, n_vars = get_shape_and_count(n_vars_or_shape)
    sparse = [(_site_to_flat(dim, shape), int(n)) for dim, n in base_func_args]
    if sum(exp for _, exp in sparse) > degree:
        return None
    return _graded_rank(sparse, n_vars)
