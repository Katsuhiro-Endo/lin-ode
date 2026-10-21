# pyright: reportReturnType=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
# sympy stubs type .subs/.xreplace/.args/.solve as Basic (non-iterable, no .subs overload)
# and omit .as_set/.size from the relational and set classes; the resulting mismatches are
# noise here (same suppression as lin_ode/basis/op.py).
"""Coefficient of a ``target`` bra in ``res = evaluate(apply_op(word, ket))``, summed over
the index box. Solved for the contributing indices, never scanned -- so cost is independent
of the grid size N.
"""

from __future__ import annotations

import functools
import itertools
from typing import Sequence

import sympy as sp

from lin_ode.basis import AntiKroneckerDelta, BasisFunction
from lin_ode.basis.multi_delta import MultiAntiKroneckerDelta, MultiKroneckerDelta

_Bounds = Sequence[tuple[int, int]]


@functools.lru_cache(maxsize=512)
def _lower_multi_deltas(expr: sp.Expr) -> sp.Expr:
    """Rewrite every delta flavour as plain ``sp.KroneckerDelta``, so the only deltas
    left fire *at* the point they are solved for (solving an anti-delta gives the one
    point where it is zero instead)."""

    def kron(a, b):
        return sp.Mul(*[sp.KroneckerDelta(x, y) for x, y in zip(a, b)])

    lowered = (
        expr.replace(MultiKroneckerDelta, kron)
        .replace(MultiAntiKroneckerDelta, lambda a, b: 1 - kron(a, b))
        .replace(AntiKroneckerDelta, lambda a, b: 1 - sp.KroneckerDelta(a, b))
    )
    return sp.expand(lowered)


def _solve_mod(lhs: sp.Expr, rhs: sp.Expr, idx: sp.Expr) -> int | None:
    """``idx`` solving ``lhs == rhs`` for ``lhs = c + Mod(idx + d, m)``, else None.

    Assumes one ``Mod`` only. In each example ``i`` runs over ``[0, 7]``; what differs is
    the range of ``lhs`` itself::

        Mod(i - 1, 8) == 7             ->  i = 0   (lhs runs [0, 7])
        1 + Mod(i - 2, 8) == 2         ->  i = 3   (lhs runs [1, 8])
        2 + Mod(i - 1, 8) == 0         ->  None    (lhs runs [2, 9], so never 0)
        Mod(i - 1, 8) + Mod(i + 1, 8)  ->  None    (two Mods, likewise if multiplied)
    """
    mods = lhs.atoms(sp.Mod)
    lower = getattr(idx, "lower", None)
    if len(mods) != 1 or lower is None:
        return None
    mod = mods.pop()
    inner, modulus = mod.args
    shift, outer = inner - idx, lhs - mod  # inner == idx + shift, lhs == outer + mod
    residue = rhs - outer - shift  # idx must be congruent to this, mod m
    if not modulus.is_Integer or idx in shift.free_symbols or not residue.is_number:
        return None
    lo, m = int(lower), int(modulus)
    resolved = lo + int(residue - lo) % m
    return resolved if lhs.subs(idx, resolved) == rhs else None


@functools.lru_cache(maxsize=None)
def _solve_index_equation(lhs: sp.Expr, rhs: sp.Expr, idx: sp.Expr) -> int | None:
    """Unique integer ``idx`` with ``lhs == rhs`` (fast path for ``idx + const`` and for
    a periodic ``Mod`` label), or None.

    One solution assumed: shipped index maps are linear. A nonlinear index, e.g.
    ``U[i**2]``, is not supported -- only the first solution would be kept.
    """
    diff = lhs - idx
    if diff.is_Number:
        sol = rhs - diff
        return (
            int(sol) if sol.is_Integer else None
        )  # is_Integer: concrete, not just integer-valued
    if lhs.has(sp.Mod):  # never hand a Mod to sp.solve -- it has no algorithm for it
        return _solve_mod(lhs, rhs, idx)
    sols = sp.solve(sp.Eq(lhs, rhs), idx)
    if not sols:
        return None
    v = sols[0] if isinstance(sols, list) else sols.get(idx)
    return int(v) if v is not None and v.is_Integer else None


def _solve_site_pair(ss: sp.Expr, ts: sp.Expr, idx: sp.Expr) -> frozenset[int]:
    """Candidate ``idx`` values making site label ``ss`` equal ``ts``, solving each
    component that depends on ``idx`` (componentwise for grid-tuple sites)."""
    both_tuples = isinstance(ss, sp.Tuple) and isinstance(ts, sp.Tuple)
    components = zip(ss, ts) if both_tuples else [(ss, ts)]
    solutions = set()
    for s, t in components:
        if not s.has(idx):
            continue
        val = _solve_index_equation(s, t, idx)
        if val is not None:
            solutions.add(val)
    return frozenset(solutions)


@functools.lru_cache(maxsize=None)
def _candidate_index_values(
    state: BasisFunction, target: BasisFunction, idx: sp.Expr
) -> frozenset[int]:
    """Integer ``idx`` values for which ``state`` could equal ``target``, from pairing
    each ``idx``-bearing state site against each target site.

    Only sites are solved: in a ``(site, power)`` pair the index sits in ``site``, e.g.
    ``F((i + 1, 2))`` -- never in ``power``.
    """
    state_sites = [s for s, _ in state.args if s.has(idx)]
    target_sites = [s for s, _ in target.args]
    candidates: set[int] = set()
    for ss in state_sites:
        for ts in target_sites:
            candidates |= _solve_site_pair(ss, ts, idx)
    return frozenset(candidates)


def _delta_trigger_point(factor: sp.Expr, idx: sp.Expr) -> int | None:
    """The ``idx`` value at which one KroneckerDelta factor fires, or None."""
    a, b = factor.args
    for lhs, rhs in ((a, b), (b, a)):
        if lhs.has(idx):
            return _solve_index_equation(lhs, rhs, idx)
    return None


def _delta_trigger_points(scalar: sp.Expr, idx: sp.Expr) -> frozenset[int]:
    """Values of ``idx`` where any KroneckerDelta factor of ``scalar`` fires (deltas are
    plain by then -- see ``_lower_multi_deltas``)."""
    triggers: set[int] = set()
    for factor in sp.Mul.make_args(scalar):
        if not isinstance(factor, sp.KroneckerDelta):
            continue
        pt = _delta_trigger_point(factor, idx)
        if pt is not None:
            triggers.add(pt)
    return frozenset(triggers)


@functools.lru_cache(maxsize=None)
def _extract_state_scalar(term: sp.Expr) -> tuple[BasisFunction, sp.Expr] | None:
    """Split a term into its single BasisFunction factor and the scalar coefficient."""
    state = None
    scalar: sp.Expr = sp.Integer(1)
    for factor in sp.Mul.make_args(term):
        if isinstance(factor, BasisFunction):
            state = factor
        else:
            scalar *= factor
    return (state, scalar) if state is not None else None


def _candidates_for_index(
    state: BasisFunction, scalar: sp.Expr, target: BasisFunction, idx: sp.Idx
) -> frozenset[int]:
    """Sites solved against the target, plus the scalar's delta trigger points -- a site
    label the solver declines is still pinned by the deltas (extras dropped by
    ``_term_coeff``)."""
    return _candidate_index_values(state, target, idx) | _delta_trigger_points(
        scalar, idx
    )


def _unpinned_in_coefficient(
    state: BasisFunction, scalar: sp.Expr, idx: sp.Idx
) -> bool:
    """Whether ``idx`` survives in the scalar with nothing to solve it against.

    The index normally sits in a state site or a delta, either of which fixes it. A
    coefficient carrying it on its own -- a per-site pivot folded onto a state that no
    longer holds that site -- takes a different value at every index in the box, so its
    contribution is a genuine sum over the axis (emitted as ``sp.Sum``).
    """
    if state.has(idx) or _delta_trigger_points(scalar, idx):
        return False
    plain = [
        f for f in sp.Mul.make_args(scalar) if not isinstance(f, sp.KroneckerDelta)
    ]
    return sp.Mul(*plain).has(idx)


def _per_dim_candidates(
    state: BasisFunction,
    scalar: sp.Expr,
    target: BasisFunction,
    indices: Sequence[sp.Idx],
    bounds: _Bounds,
) -> list[list[int]]:
    """Sorted, in-range candidate values for each index dimension of one term."""
    per_dim = []
    for idx, (lo, hi) in zip(indices, bounds):
        raw = _candidates_for_index(state, scalar, target, idx)
        in_range = [v for v in raw if lo <= v <= hi]
        per_dim.append(sorted(in_range))
    return per_dim


def _couples_other_axes(
    factors: Sequence[sp.Basic],
    indices: Sequence[sp.Idx],
    free: Sequence[int],
    axis: int,
) -> bool:
    """Whether a factor constraining free ``axis`` also constrains another free axis.

    Such a factor, e.g., ``Eq(i, j)``, ties the axes together, so the region it cuts is
    not a rectangle and its size is not the product of per-axis sizes.
    """
    other_axes = [indices[other] for other in free if other != axis]
    return any(factor.has(*other_axes) for factor in factors)


def _axis_values(
    factors: Sequence[sp.Basic], idx: sp.Idx, lower: int, upper: int
) -> sp.Set:
    """Values in ``[lower, upper]`` satisfying every factor, from the solution set of
    the factors rather than by testing each value, so a billion-site axis costs O(1)."""
    unbounded = sp.Dummy(str(idx), integer=True)  # an Idx's own bounds block as_set
    satisfying = sp.And(*factors).subs(idx, unbounded).as_set()
    return sp.Range(lower, upper + 1).intersect(satisfying)


def _axis_size(values: sp.Set) -> int:
    """How many values the set holds, a disconnected one (the two Ranges a hole such as
    ``Ne(i, 1)`` leaves behind) as the sum of its pieces."""
    if isinstance(values, sp.Union):
        return sum(len(part) for part in values.args)
    return len(values)


def _box_count(
    region: sp.Basic, indices: Sequence[sp.Idx], free: Sequence[int], bounds: _Bounds
) -> int:
    """How many values of the ``free`` index dimensions satisfy ``region``, which must
    be axis-aligned (every factor constraining at most one dimension) so that the count
    is a product of per-axis sizes.

    Raises:
        NotImplementedError: If a factor couples two free dimensions, e.g. ``Eq(i, j)``.
    """
    if region == sp.false:
        return 0
    count = 1
    for axis in free:
        lower, upper = bounds[axis]
        idx = indices[axis]
        constraining = [f for f in sp.And.make_args(region) if f.has(idx)]
        if not constraining:
            count *= upper - lower + 1
            continue
        if _couples_other_axes(constraining, indices, free, axis):
            raise NotImplementedError(f"{region} is not an axis-aligned region")
        count *= _axis_size(_axis_values(constraining, idx, lower, upper))
    return count


def _split_required_and_excluded(
    active: sp.Basic,
) -> tuple[list[sp.Basic], list[sp.Basic]]:
    """Split an arm's condition into the regions it requires and the ones it excludes.

    An arm holds where its own condition does and no earlier arm's does, so the condition
    reaches here as a conjunction with the earlier ones negated:
    ``Eq(i, 0) & ~(Eq(i, 0) & Eq(j, 0))`` requires the i=0 edge and excludes the corner an
    earlier arm already took.
    """
    factors = sp.And.make_args(active)
    required = [factor for factor in factors if not isinstance(factor, sp.Not)]
    excluded = [factor.args[0] for factor in factors if isinstance(factor, sp.Not)]
    return required, excluded


def _active_pin(active: sp.Basic, idx: sp.Idx, bound: tuple[int, int]) -> int | None:
    """The one ``idx`` value the arm's own condition allows, else None.

    An arm condition pins as firmly as a delta: under ``Eq(i, 0)`` a bare ``s[i]`` is
    ``s[0]``. Only a singleton counts -- a sublattice or a wider arm is still a sum, and
    a condition ``as_set`` cannot solve reads as no pin rather than raising through.
    """
    required, _ = _split_required_and_excluded(active)
    constraining = [factor for factor in required if factor.has(idx)]
    # a condition tying idx to another symbol, e.g. Eq(i, j), pins the pair and not the
    # axis; as_set refuses it anyway, this just declines without raising through
    if not constraining or sp.And(*constraining).free_symbols != {idx}:
        return None
    try:
        values = _axis_values(constraining, idx, *bound)
    except NotImplementedError:
        return None
    return int(next(iter(values))) if _axis_size(values) == 1 else None


def _active_index_count(
    active: sp.Basic, indices: Sequence[sp.Idx], free: Sequence[int], bounds: _Bounds
) -> int:
    """How many values of the ``free`` index dimensions satisfy ``active`` (already fixed at
    the other dimensions).

    Sympy does not reduce "required minus excluded" to a single region, so the exclusions
    come off by inclusion-exclusion and every remaining term is one box::

        |R \\ (N1 u N2)| = |R| - |R n N1| - |R n N2| + |R n N1 n N2|

    A union of boxes, required or excluded, comes off the same way
    (:func:`_region_count`).
    """
    if active == sp.false:
        return 0
    required, excluded = _split_required_and_excluded(active)
    total = 0
    for taken_count in range(len(excluded) + 1):
        sign = -1 if taken_count % 2 else 1
        for taken in itertools.combinations(excluded, taken_count):
            total += sign * _region_count(
                sp.And(*required, *taken), indices, free, bounds
            )
    return total


def _region_count(
    region: sp.Basic, indices: Sequence[sp.Idx], free: Sequence[int], bounds: _Bounds
) -> int:
    """:func:`_box_count`, but a union of boxes too, by inclusion-exclusion over the DNF
    disjuncts -- each intersection of those is a conjunction again, hence one box.

    ``simplify`` holds the term count to the number of boxes rather than the raw DNF's,
    which is exponentially many mostly-contradictory terms.
    """
    if not any(isinstance(factor, sp.Or) for factor in sp.And.make_args(region)):
        return _box_count(region, indices, free, bounds)
    dnf = sp.to_dnf(region, simplify=True, force=True)
    boxes = sp.Or.make_args(dnf) if isinstance(dnf, sp.Or) else (dnf,)
    total = 0
    for taken_count in range(1, len(boxes) + 1):
        sign = 1 if taken_count % 2 else -1
        for taken in itertools.combinations(boxes, taken_count):
            total += sign * _box_count(sp.And(*taken), indices, free, bounds)
    return total


def _countable_factors(
    active: sp.Basic,
    indices: Sequence[sp.Idx],
    summed: Sequence[int],
    free: Sequence[int],
) -> sp.Basic:
    """``active`` without the factors that constrain a summed axis.

    Those ride on the ``Piecewise`` guard inside the ``Sum`` instead (see
    :func:`_sum_over_unpinned`). Counting them here too cancels the term: with no free
    axis left, :func:`_box_count` cannot tell an excluded region from the trivial one,
    so inclusion-exclusion subtracts a whole copy and returns 0.

    Raises:
        NotImplementedError: If one factor constrains a summed *and* a free axis, where
            the guard and the count would each need a part of it.
    """
    if not summed:
        return active
    summed_idxs = [indices[axis] for axis in summed]
    free_idxs = [indices[axis] for axis in free]
    kept = []
    for factor in sp.And.make_args(active):
        if not factor.has(*summed_idxs):
            kept.append(factor)
            continue
        if free_idxs and factor.has(*free_idxs):
            raise NotImplementedError(
                f"{factor} couples a summed axis to a free one, so its guard and its "
                "count cannot be separated"
            )
    return sp.And(*kept)


def _sum_over_unpinned(
    expr: sp.Expr,
    active: sp.Basic,
    indices: Sequence[sp.Idx],
    summed: Sequence[int],
    bounds: _Bounds,
) -> sp.Expr:
    """``expr`` wrapped in an unevaluated ``sp.Sum`` over each axis nothing pins.

    Left unevaluated so the element stays N-independent. The arm condition cannot ride
    on a ``Sum`` limit, so the factors of ``active`` naming a summed axis become a
    ``Piecewise`` guard on the summand instead.
    """
    if not summed:
        return expr
    idxs = [indices[axis] for axis in summed]
    constraining = [f for f in sp.And.make_args(active) if f.has(*idxs)]
    if constraining:
        expr = sp.Piecewise((expr, sp.And(*constraining)), (sp.S.Zero, True))
    return sp.Sum(expr, *[(indices[axis], *bounds[axis]) for axis in summed])


def _term_coeff(
    term: sp.Expr,
    target: BasisFunction,
    indices: Sequence[sp.Idx],
    bounds: _Bounds,
    active: sp.Basic,
) -> sp.Expr:
    """Contribution of one ``scalar * state`` term to ``<target|.|ket>`` over the index
    box.

    A dimension the term never mentions is *free*: the term takes the same value at every
    one of its index values, so those are counted rather than enumerated (lowering an
    anti-delta leaves exactly such a term, and the box can be astronomically large).

    A dimension the term *does* vary over but nothing pins is *summed*: an ``sp.Sum`` over
    it, left unevaluated (see :func:`_sum_over_unpinned`).

    ``BasisFunction`` sorts its ``(site, power)`` pairs by site, so pair order never
    affects the match below::

        F((1, 2), (0, 1)) == F((0, 1), (1, 2))
    """
    split = _extract_state_scalar(term)
    if split is None:
        return sp.S.Zero
    state, scalar = split
    if not any(state.has(idx) for idx in indices) and state != target:
        return sp.S.Zero
    free = [axis for axis, idx in enumerate(indices) if not term.has(idx)]
    pinned = [axis for axis in range(len(indices)) if axis not in free]
    per_axis = _per_dim_candidates(state, scalar, target, indices, bounds)
    summed = []
    for axis in pinned:
        if per_axis[axis] or not _unpinned_in_coefficient(state, scalar, indices[axis]):
            continue
        pin = _active_pin(active, indices[axis], bounds[axis])
        if pin is None:  # nothing fixes it: the contribution is a sum, so emit one
            summed.append(axis)
        else:
            per_axis[axis] = [pin]
    pinned = [axis for axis in pinned if axis not in summed]
    total: sp.Expr = sp.S.Zero
    for values in itertools.product(*(per_axis[axis] for axis in pinned)):
        fixed = {
            indices[axis]: sp.Integer(value) for axis, value in zip(pinned, values)
        }
        if state.xreplace(fixed) != target:
            continue
        rest = active.subs(fixed)
        free_values = _active_index_count(
            _countable_factors(rest, indices, summed, free), indices, free, bounds
        )
        total += _sum_over_unpinned(
            scalar.xreplace(fixed) * free_values, rest, indices, summed, bounds
        )
    return total


def get_coeff(
    res: sp.Expr,
    target: BasisFunction,
    indices: Sequence[sp.Idx],
    bounds: _Bounds,
    active: sp.Basic,
) -> sp.Expr:
    """Coefficient of ``target`` in ``sum_v res(v)`` over the index box, restricted to
    the ``v`` where ``active`` holds; each term is solved for its contributing ``v`` (no
    range scan)."""
    res = _lower_multi_deltas(sp.expand(res))
    return sum(
        (
            _term_coeff(term, target, indices, bounds, active)
            for term in sp.Add.make_args(res)
        ),
        sp.S.Zero,
    )
