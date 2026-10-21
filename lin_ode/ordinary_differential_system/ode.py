"""Concrete centred finite-difference scheme and discretised ODE value object."""

from __future__ import annotations

from math import prod
from typing import Sequence

import sympy as sp

from .. import serialization as _ser
from ..indexed_expression import IndexedExpression
from ..utils import parse_indexed_string
from .constraints import (
    Condition,
    DerivativeCondition,
    RegionStateCondition,
    StateCondition,
    _substitute_once,
    normalize_constraints,
)


def _partition_conditions(
    conditions: Sequence[Condition],
) -> tuple[tuple[StateCondition, ...], tuple[DerivativeCondition, ...]]:
    """Split a mixed ``conditions`` list into (state conditions, derivative conditions).

    Grouping is by family, not list position: derivative conditions build the
    effective field and state conditions substitute into it, whatever order the
    user listed them in.

    Raises:
        TypeError: If a condition is neither a :class:`StateCondition` nor a
            :class:`DerivativeCondition`.
    """
    state: list[StateCondition] = []
    derivative: list[DerivativeCondition] = []
    for cond in conditions:
        if isinstance(cond, StateCondition):
            state.append(cond)
        elif isinstance(cond, DerivativeCondition):
            derivative.append(cond)
        else:
            raise TypeError(
                f"{type(cond).__name__} is not a Condition (StateCondition or "
                "DerivativeCondition)."
            )
    return tuple(state), tuple(derivative)


def _index_name_or_none(idx_expr: sp.Basic) -> str | None:
    """Name of the free symbol driving *idx_expr*, or ``None`` for a
    literal/constant."""
    if isinstance(idx_expr, sp.Idx):
        return idx_expr.name
    if isinstance(idx_expr, (sp.Integer, int)) or not idx_expr.free_symbols:
        return None
    return next(iter(idx_expr.free_symbols)).name  # type: ignore[union-attr]


def _validate_consistent_indexing(
    state_atoms: list[sp.Indexed],
    ordered_indices: Sequence[sp.Idx],
    state_variable: sp.IndexedBase,
) -> None:
    """Validate that all occurrences of state_variable use indices in the same order.

    Args:
        state_atoms: All Indexed expressions using the state_variable
        ordered_indices: The canonical index ordering
        state_variable: The IndexedBase being validated

    Raises:
        ValueError: If any occurrence uses indices in a different order

    Note:
        This validation allows literal integer indices (e.g., u[0], u[N-1]) which are
        needed for boundary conditions like periodic BCs. It only validates that symbolic
        indices appear in the correct order when they are present.
    """
    expected_names = [getattr(idx, "name", str(idx)) for idx in ordered_indices]

    for indexed in state_atoms:
        actual_names = [_index_name_or_none(idx_expr) for idx_expr in indexed.indices]

        if len(actual_names) != len(expected_names) or any(
            actual is not None and actual != expected
            for expected, actual in zip(expected_names, actual_names)
        ):
            raise ValueError(f"Expression indexes {state_variable} inconsistently")


def _reject_state_variable_with_shape(state_variable: sp.IndexedBase) -> None:
    """Reject a state variable created with an explicit shape.

    Raises:
        ValueError: If *state_variable* carries a shape. Shape is always
            derived from index bounds, so a shaped state variable is
            ambiguous rather than merely redundant.
    """
    if getattr(state_variable, "shape", None):
        raise ValueError(
            f"'{state_variable}' was created with an explicit shape, but "
            "OrdinaryDifferentialSystem derives shape from the index bounds "
            "(e.g. Idx('i', (0, N - 1))); pass an unshaped IndexedBase instead."
        )


def _validate_indices_have_ranges(ordered_indices: Sequence[sp.Idx]) -> None:
    """Validate that every index carries an explicit ``(lower, upper)`` range.

    Raises:
        ValueError: If any index lacks explicit bounds.
    """
    for idx in ordered_indices:
        if idx.lower is None or idx.upper is None:
            raise ValueError(
                f"Index {idx} has no explicit range; every index must be "
                "created with bounds, e.g. Idx('i', (0, N - 1))."
            )


def _validate_equation(equation: IndexedExpression) -> None:
    """Validate that *equation* uses ranged indices, consistently.

    Literal indices are not range-checked: out-of-range literals wrap on the
    torus (``u[N]`` is ``u[0]``, ``u[-1]`` is the last site), resolved by
    :func:`_wrap_indices_mod`.

    Args:
        equation: The equation to validate.

    Raises:
        ValueError: If any index lacks an explicit range, or indices are used
            inconsistently.
    """
    _validate_indices_have_ranges(equation.indices)
    state_atoms = [
        atom
        for atom in equation.expr.atoms(sp.Indexed)
        if atom.base == equation.state_variable
    ]
    _validate_consistent_indexing(
        state_atoms, equation.indices, equation.state_variable
    )


def _wrap_indices_mod(
    expr: sp.Expr,
    state_variable: sp.IndexedBase,
    indices: Sequence[sp.Idx],
) -> sp.Expr:
    """Make state-variable indexing periodic by wrapping index arithmetic in ``Mod``.

    The index space is periodic by default: symbolic index arithmetic such as
    ``u[i - 1]`` becomes ``u[Mod(i - 1, N)]`` (shifted for a nonzero lower
    bound), so wrap-around resolves by itself when concrete integers are
    substituted downstream.  A bare index like ``u[i]`` is left alone.
    Literal indices resolve immediately, wrapped modulo the index's size the
    same way (values already in bounds are left as-is).

    Args:
        expr: The expression whose state references to wrap.
        state_variable: The state :class:`sympy.IndexedBase`.
        indices: The system's indices, in subscript order, each carrying
            concrete ``(lower, upper)`` bounds.
    """
    replacements: dict[sp.Indexed, sp.Expr] = {}
    for atom in expr.atoms(sp.Indexed):
        if atom.base != state_variable:
            continue
        if len(atom.indices) != len(indices):
            continue  # reported by _validate_consistent_indexing
        new_indices: list[sp.Expr] = []
        changed = False
        for idx_expr, idx in zip(atom.indices, indices):
            lower = int(idx.lower)  # type: ignore[arg-type]
            upper = int(idx.upper)  # type: ignore[arg-type]
            size = upper - lower + 1
            try:
                literal = int(idx_expr)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                literal = None

            if literal is not None:
                if lower <= literal <= upper:
                    new_indices.append(idx_expr)  # type: ignore[arg-type]
                else:
                    new_indices.append(sp.Integer(lower + (literal - lower) % size))
                    changed = True
            elif isinstance(idx_expr, sp.Idx):
                new_indices.append(idx_expr)
            else:
                new_indices.append(lower + sp.Mod(idx_expr - lower, size))  # type: ignore[operator]
                changed = True
        if changed:
            replacements[atom] = state_variable[tuple(new_indices)]  # type: ignore[assignment]
    return expr.xreplace(replacements) if replacements else expr


def _pinned_maps(condition: sp.Basic) -> list[dict[sp.Idx, int]]:
    """The ``{index: position}`` pin-maps a region's condition ranges over.

    A region's :attr:`~.region.Region.condition` is a disjunction of conjunctions whose
    ``Eq(index, position)`` conjuncts pin axes (any inequality conjuncts merely clip and
    are ignored here).  One map per disjunct: a single pinned region (facet, corner)
    yields one; a shell (``Or`` of walls) yields one per wall.
    """
    disjuncts = condition.args if isinstance(condition, sp.Or) else (condition,)
    maps: list[dict[sp.Idx, int]] = []
    for disjunct in disjuncts:
        conjuncts = disjunct.args if isinstance(disjunct, sp.And) else (disjunct,)
        pinned: dict[sp.Idx, int] = {}
        for conj in conjuncts:
            if isinstance(conj, sp.Eq) and isinstance(conj.lhs, sp.Idx):
                pinned[conj.lhs] = int(conj.rhs)  # type: ignore[arg-type]
        maps.append(pinned)
    return maps


def _specificity(condition: sp.Basic) -> int:
    """How specific a region is: the number of axes pinned in *every* disjunct.

    More pinned axes means a smaller region (a corner beats an edge), so this orders
    overlap precedence.  A shell whose walls pin different axes shares none, so it is
    least specific.
    """
    maps = _pinned_maps(condition)
    if not maps:
        return 0
    common = set(maps[0])
    for pinned in maps[1:]:
        common &= set(pinned)
    return len(common)


def _substitute_constrained(
    expr: sp.Expr,
    state_variable: sp.IndexedBase,
    indices: Sequence[sp.Idx],
    constraints: Sequence[RegionStateCondition],
) -> sp.Expr:
    """Replace state references lying in a constrained region by the constraint's value,
    repeating until no reference remains.

    More specific constraints (corners) win over less specific ones (edges)
    when both cover an atom.  Values referencing other constrained regions
    resolve on subsequent passes.

    Raises:
        ValueError: If substitution does not converge (constraint values
            alias each other cyclically).
    """
    ordered = sorted(constraints, key=lambda c: -_specificity(c.region.condition))

    def wrap(value: sp.Expr) -> sp.Expr:
        return _wrap_indices_mod(value, state_variable, indices)

    # len(constraints) + 1 passes: the longest acyclic dependency chain among n
    # constraints is n - 1 hops deep, plus one pass to confirm no further change (this
    # loop, unlike normalize_constraints, only detects convergence via an empty pass
    # after the last productive one, so it needs one more iteration than that bound).
    for _ in range(len(constraints) + 1):
        substituted = _substitute_once(
            expr, ordered, state_variable, indices, transform=wrap
        )
        if substituted == expr:
            return expr
        expr = substituted
    raise ValueError("Constraint values alias each other cyclically; cannot eliminate.")


def _solve_axis_entry(
    entry: sp.Expr,
    idx: sp.Idx,
    position: int,
) -> int | str | None:
    """Grid values of *idx* for which *entry* references *position*.

    Returns ``"always"`` for a matching literal, ``None`` for a
    non-matching literal, or the unique solution for a bare index or a
    wrapped literal offset (``lower + Mod(idx + k - lower, size)``, the
    form :func:`_wrap_indices_mod` produces).

    Raises:
        ValueError: For any other index expression — inverting a general
            index map may have many (even O(size)) solutions, so only
            plain stencil offsets are supported.
    """
    lower = int(idx.lower)  # type: ignore[arg-type]
    upper = int(idx.upper)  # type: ignore[arg-type]
    size = upper - lower + 1
    try:
        literal = int(entry)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        literal = None
    if literal is not None:
        return "always" if literal == position else None
    if entry == idx:
        return position
    mods = entry.atoms(sp.Mod)
    if len(mods) == 1:
        mod = next(iter(mods))
        arg, modulus = mod.args
        offset = sp.expand(arg - (idx - lower))  # type: ignore[operator]
        if entry - mod == lower and modulus == size and offset.is_Integer:
            return lower + (position - lower - int(offset)) % size
    raise ValueError(
        f"Cannot eliminate through index expression {entry}; only literal "
        "offsets like i + k are supported with eliminate=True."
    )


def _elimination_branches(
    interior: sp.Expr,
    state_variable: sp.IndexedBase,
    indices: Sequence[sp.Idx],
    constraints: Sequence[RegionStateCondition],
) -> list[tuple[sp.Expr, sp.Basic]]:
    """Special-case branches for interior points that reference a constrained region.

    For each state atom and constraint, solves which index values make the atom land in
    the region; each solution pins some indices and yields a branch of the interior with
    those values substituted and the region references eliminated.  Assignments are
    closed under compatible union so points adjacent to several regions get a combined
    branch, and assignments lying wholly inside a region are dropped (the constraint's
    own branch shadows them).
    """
    hits: set[frozenset[tuple[sp.Idx, int]]] = set()
    for atom in interior.atoms(sp.Indexed):
        if atom.base != state_variable or len(atom.indices) != len(indices):
            continue
        for constraint in constraints:
            for pinned in _pinned_maps(constraint.region.condition):
                assignment: dict[sp.Idx, int] = {}
                feasible = True
                for idx, position in pinned.items():
                    entry = atom.indices[list(indices).index(idx)]
                    solution = _solve_axis_entry(entry, idx, position)  # type: ignore[arg-type]
                    if solution is None:
                        feasible = False
                        break
                    if solution != "always":
                        assignment[idx] = solution  # type: ignore[assignment]
                if feasible and assignment:
                    hits.add(frozenset(assignment.items()))

    # close under compatible union so overlapping regions combine
    changed = True
    while changed:
        changed = False
        for a in list(hits):
            for b in list(hits):
                da, db = dict(a), dict(b)
                if all(da.get(k, v) == v for k, v in db.items()):
                    union = frozenset({**da, **db}.items())
                    if union not in hits:
                        hits.add(union)
                        changed = True

    def _shadowed(assignment: dict[sp.Idx, int]) -> bool:
        return any(
            all(assignment.get(idx) == pos for idx, pos in pinned.items())
            for c in constraints
            for pinned in _pinned_maps(c.region.condition)
        )

    kept = [dict(h) for h in hits if not _shadowed(dict(h))]

    branches: list[tuple[sp.Expr, sp.Basic]] = []
    for assignment in sorted(
        kept, key=lambda d: (-len(d), str(sorted(map(str, d.items()))))
    ):
        expr = interior.subs(assignment, simultaneous=True)  # type: ignore[arg-type]
        expr = _substitute_constrained(expr, state_variable, indices, constraints)
        condition = sp.And(*(sp.Eq(k, v) for k, v in assignment.items()))  # type: ignore[arg-type]
        branches.append((expr, condition))
    return branches


def _validate_constraints(
    constraints: Sequence[RegionStateCondition],
    indices: Sequence[sp.Idx],
) -> None:
    """Validate *constraints* against the system's indices.

    Raises:
        ValueError: If a constraint fixes an index that is not the system's.
    """
    for constraint in constraints:
        for idx in constraint.region.condition.atoms(sp.Idx):
            if idx not in indices:
                raise ValueError(
                    f"Constraint fixes index {idx}, which is not one of the "
                    f"system's indices {tuple(indices)}."
                )


def _validate_subdomains(
    subdomains: Sequence[DerivativeCondition],
    indices: Sequence[sp.Idx],
) -> None:
    """Validate *subdomains* against the system's indices.

    Overlapping boxes are *not* an error: derivative conditions fold in user list
    order and the Piecewise is first-match, so the first-listed one wins on a shared
    cell.

    Raises:
        ValueError: If a subdomain restricts an index that is not the system's.
    """
    for subdomain in subdomains:
        for idx in subdomain.region.condition.atoms(sp.Idx):
            if idx not in indices:
                raise ValueError(
                    f"DerivativeCondition restricts index {idx}, which is not one of "
                    f"the system's indices {tuple(indices)}."
                )


def _eliminate_constrained_references(
    interior: sp.Expr,
    effective: sp.Expr,
    branches: list[tuple[sp.Expr, sp.Basic]],
    state_variable: sp.IndexedBase,
    indices: Sequence[sp.Idx],
    constraints: Sequence[RegionStateCondition],
) -> tuple[sp.Expr, list[tuple[sp.Expr, sp.Basic]]]:
    """Substitute constrained state references out of every branch and the interior.

    Constraints are normalized first so that no constraint's value references another
    constrained region (e.g. ``u[0] = u[1]``, ``u[1] = 5`` becomes ``u[0] = 5``,
    ``u[1] = 5``); substitution against the normalized constraints then resolves in a
    single step per reference.  Also adds specialised branches for interior points that
    reference a constrained region; constraint branches always precede elimination
    branches so region dynamics win where they overlap.

    Elimination branches are derived against *effective* -- the subdomain-aware interior
    -- so a point inside a subdomain uses that subdomain's stencil.  The fallthrough
    *interior* returned is the ambient one (the subdomain branches are already explicit),
    with constrained references substituted out.

    Returns:
        The updated ``(interior, branches)`` pair.
    """
    branches = [
        (_substitute_constrained(e, state_variable, indices, constraints), cond)
        for e, cond in branches
    ]
    branches += _elimination_branches(effective, state_variable, indices, constraints)
    interior = _substitute_constrained(interior, state_variable, indices, constraints)
    return interior, branches


def _assemble_equation(
    interior: sp.Expr,
    state_variable: sp.IndexedBase,
    indices: tuple[sp.Idx, ...],
    constraints: Sequence[StateCondition],
    eliminate: bool = False,
    subdomains: Sequence[DerivativeCondition] = (),
) -> IndexedExpression:
    """Assemble the full equation from an interior expression, constraints and
    subdomains.

    Wraps the interior's index arithmetic in ``Mod`` (periodic by default), derives each
    state condition's RHS branch, and folds them into a Piecewise with the interior as
    fallthrough.  Branches follow the user's list order: the Piecewise is first-match,
    so where two regions overlap the first-listed condition wins (list a corner before
    an edge it sits on for the corner to win).

    Conditions are normalized before their branches are derived, so a value that
    references another constrained region sees that region's fixed value.

    Each *subdomain* contributes a branch whose expression is its own (``Mod``-wrapped)
    stencil and whose condition is its box.  Subdomain branches follow the state-
    condition branches, so a boundary condition on a face lying on a subdomain edge wins
    over the subdomain there.  Overlapping subdomains are resolved the same way -- the
    first-listed wins on a shared cell.

    With *eliminate*, constrained state references are substituted out of every branch
    expression, and interior points that reference a constrained region get their own
    specialised branches; state-condition branches always precede elimination branches
    so region dynamics win where they overlap.

    A subdomain replaces the ambient dynamics on its box, so a constraint or elimination
    branch that references ``du/dt`` at a point *inside* a subdomain must use the
    subdomain's stencil there, not the ambient one.  This is handled by deriving those
    branches against the *effective interior* -- a ``Piecewise`` of the subdomain
    stencils over the ambient one -- so substituting a concrete point picks the
    governing stencil automatically.
    """
    _validate_indices_have_ranges(indices)
    # the assembler resolves state conditions through their region's condition, whose
    # disjuncts pin the axes it eliminates through (a facet/corner, or a box shell).
    point_constraints: list[RegionStateCondition] = []
    for constraint in constraints:
        if not isinstance(constraint, RegionStateCondition):
            raise TypeError(
                f"{type(constraint).__name__} is not a region state condition."
            )
        point_constraints.append(constraint)
    constraints = point_constraints
    _validate_constraints(point_constraints, indices)
    _validate_subdomains(subdomains, indices)

    interior = _wrap_indices_mod(interior, state_variable, indices)

    # the dynamics of a point are the subdomain stencil if it is in a box, else ambient;
    # constraint / elimination branches are derived against this so a reference into a
    # subdomain picks up the subdomain stencil (see rhs_branch's per-point evaluation)
    subdomain_branches = [
        (_wrap_indices_mod(s.rhs, state_variable, indices), s.region.condition)
        for s in subdomains
    ]
    effective = (
        sp.Piecewise(*subdomain_branches, (interior, sp.true))
        if subdomain_branches
        else interior
    )

    # state conditions are folded in USER LIST ORDER -- the Piecewise is first-match,
    # so the first-listed condition wins where two regions overlap (list a corner
    # before an edge it sits on for the corner to win).  No specificity sort.
    # normalized on both paths: the chain rule would otherwise differentiate the
    # unconstrained dynamics at the referenced site and break the aliasing condition
    ordered: list[RegionStateCondition] = list(
        normalize_constraints(point_constraints, state_variable, indices)
    )
    branches = [c.rhs_branch(effective, state_variable, indices) for c in ordered]

    if eliminate and ordered:
        interior, branches = _eliminate_constrained_references(
            interior, effective, branches, state_variable, indices, ordered
        )

    # subdomain branches follow the state-condition / elimination branches: a point-fix
    # on a subdomain edge is an earlier (higher-precedence) branch, so it wins there
    branches += subdomain_branches

    if branches:
        expr = sp.Piecewise(*branches, (interior, sp.true))
    else:
        expr = interior
    return IndexedExpression(expr, state_variable, indices)


class OrdinaryDifferentialSystem:
    r"""A symbolic ODE system with multi-index structure.

    The full dynamics are encoded in a single
    :class:`~.indexed_expression.IndexedExpression` that carries its own
    state variable :math:`u` and a tuple of symbolic indices
    :math:`(i, j, \ldots)`. Each index represents a spatial dimension.

    Args:
        equation: The stencil encoded as a single
            :class:`~.indexed_expression.IndexedExpression` in the flat index.
        conditions: The :class:`~.constraints.Condition` objects the equation was
            assembled from, if any (state and/or derivative conditions).  Stored
            for later use (e.g. constructing or checking condition-consistent
            initial conditions) and split into :attr:`state_conditions` /
            :attr:`derivative_conditions`; they do not modify *equation*, which is
            taken as-is — use :meth:`from_sympy` to have the piecewise derived from
            conditions.
    """

    def __init__(
        self,
        equation: IndexedExpression,
        conditions: Sequence[Condition] = (),
    ) -> None:
        _validate_equation(equation)
        self._equation = equation
        self._state_conditions, self._derivative_conditions = _partition_conditions(
            conditions
        )

    def dump_json(self, indent: int | None = None) -> str:
        """Return this system as a JSON string.

        Serializes the ``equation`` and any ``conditions``.
        """
        conditions = (*self._state_conditions, *self._derivative_conditions)
        return _ser.json_dumps(
            {
                "equation": self._equation.to_json_data(),
                "conditions": [c.to_json_data() for c in conditions],
            },
            indent,
        )

    @classmethod
    def from_json(cls, json_str: str) -> OrdinaryDifferentialSystem:
        """Rebuild from a string produced by :meth:`dump_json`."""
        json_data = _ser.json_loads(json_str)
        return cls(
            IndexedExpression.from_json_data(json_data["equation"]),
            [Condition.from_json_data(c) for c in json_data.get("conditions", [])],
        )

    @property
    def equation(self) -> IndexedExpression:
        r"""Dynamics as an :class:`~.indexed_expression.IndexedExpression`."""
        return self._equation

    @property
    def state_conditions(self) -> tuple[StateCondition, ...]:
        r"""The state conditions (Dirichlet), as provided at construction.

        The equation already encodes their dynamical consequences; these are kept so
        initial conditions can be built or checked against them.
        """
        return self._state_conditions

    @property
    def derivative_conditions(self) -> tuple[DerivativeCondition, ...]:
        r"""The derivative conditions (:class:`~.constraints.DerivativeCondition`), as
        provided at construction.

        The equation already encodes their dynamics (each is a branch of the Piecewise);
        these are kept for introspection.
        """
        return self._derivative_conditions

    @property
    def index_symbols(self) -> Sequence[sp.Idx]:
        r"""The symbolic indices parameterizing the dynamics.

        Returns a tuple of :class:`sympy.Idx` symbols, one for each dimension
        of the multi-index space. For a single-index system this returns a
        1-tuple; for multi-index systems it returns multiple indices.

        Returns:
            Sequence of symbolic index objects.
        """
        return self.equation.indices

    @property
    def state_variable(self) -> sp.IndexedBase:
        r"""The state variable :math:`u` as a multi-indexable array.

        Returns the single :class:`sympy.IndexedBase` that carries the system state.
        This variable can be indexed by one or more symbolic indices from
        :attr:`index_symbols`, e.g. ``u[i, j, ...]``.

        Returns:
            The state :class:`sympy.IndexedBase`, derived from :attr:`equation`.
        """
        return self.equation.state_variable

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Concrete size of each index dimension, derived from index bounds."""
        return tuple(
            int(idx.upper) - int(idx.lower) + 1  # type: ignore[arg-type]
            for idx in self.index_symbols
        )

    @property
    def total_dof(self) -> int:
        r"""Total degrees of freedom."""
        return prod(self.shape)

    @classmethod
    def from_strings(
        cls,
        expr: str,
        indices: Sequence[sp.Idx],
        state_variable: str = "u",
        conditions: Sequence[Condition] = (),
        eliminate: bool = False,
    ) -> OrdinaryDifferentialSystem:
        """Create an ODE from an interior expression string.

        Convenience wrapper around :meth:`from_sympy`: parses *expr* with the
        given indices in scope and hands off to the same assembly.

        Args:
            expr: The interior right-hand side in terms of the indexed state
                variable, e.g. ``'(u[i - 1] + u[i + 1]) / h'`` — a 1-D
                stencil with state variable ``u[i]``, index ``i``, and free
                parameter ``h``.  Boundary conditions are given via
                *conditions*, not encoded in the string.
            indices: The index symbols parameterising the system, in
                subscript order.  Every index must carry an explicit range,
                e.g. ``Idx('i', (0, N - 1))``; shape is derived from these
                bounds.  Their names are made available when parsing *expr*.
            state_variable: Name of the indexed state variable in *expr*.
                Defaults to ``'u'``.
            conditions: Boundary/interior conditions -- state conditions
                (Dirichlet) and derivative conditions
                (:class:`~.constraints.DerivativeCondition`) -- referencing the same
                :class:`sympy.Idx` objects as *indices*; see :meth:`from_sympy`.
            eliminate: Substitute constrained state references out of the
                assembled equation; see :meth:`from_sympy`.
        """
        parsed, u_base = parse_indexed_string(expr, indices, state_variable)
        return cls.from_sympy(parsed, u_base, tuple(indices), conditions, eliminate)

    @classmethod
    def from_sympy(
        cls,
        expr: sp.Expr,
        state_variable: sp.IndexedBase,
        indices: tuple[sp.Idx, ...],
        conditions: Sequence[Condition] = (),
        eliminate: bool = False,
    ) -> OrdinaryDifferentialSystem:
        """Create an ODE from an interior SymPy expression and conditions.

        Indexing of the state variable is periodic by default: symbolic
        index arithmetic wraps around the ends of each axis (``u[i - 1]``
        at ``i = 0`` reads ``u[N - 1]``), and negative literal indices count
        from the end, Python-style (``u[-3]`` is ``u[N - 3]``).  Constraints
        carve pinned regions out of that torus: each contributes a Piecewise
        branch derived from its ``u[region] = value`` equation, with the
        interior expression as fallthrough and more-specific regions
        (corners) taking precedence over less-specific ones (edges).

        Args:
            expr: The interior right-hand side of the ODE.  Must contain
                *state_variable* somewhere.
            state_variable: The :class:`~sympy.IndexedBase` that represents the
                unknown state.  Must not itself carry a ``shape`` (create it
                as ``IndexedBase('u')``, not ``IndexedBase('u', shape=...)``).
            indices: The index symbols parameterising the system, in
                subscript order (e.g. ``(i, j)`` for ``u[i, j]``).  Every
                index must carry an explicit range, e.g. ``Idx('i', (0, N - 1))``;
                shape is derived from these bounds.
            conditions: Boundary/interior conditions, referencing the same
                :class:`sympy.Idx` objects as *indices*.  Two families, sorted
                internally by kind:

                - *state conditions* (Dirichlet) pin ``u`` on a
                  region; unconditioned boundaries remain periodic.
                - *derivative conditions*
                  (:class:`~.constraints.DerivativeCondition`) run their own stencil on
                  a box that replaces the ambient interior there; each becomes a
                  Piecewise branch below the state-condition branches (so a
                  boundary condition on a subdomain edge wins), ``Mod``-wrapped
                  like the interior, and is unaffected by *eliminate*.
            eliminate: Substitute constrained state references out of the
                assembled equation.  Every branch expression has references
                into constrained regions replaced by the state-condition values,
                and interior points whose stencil touches a constrained
                region get their own specialised branches (e.g. with
                ``u[0] = u[1]``, the ``i = 1`` row becomes explicit with
                ``u[0]`` replaced by ``u[1]``).  The pinned points keep
                their rows, so the system dimension is unchanged — only the
                references to them disappear.  Requires state references to
                use plain literal index offsets (``i + k``).  Affects only state
                conditions.

        Returns:
            A new :class:`OrdinaryDifferentialSystem` instance.

        Raises:
            TypeError: If *indices* contains a non-``Idx`` element.
            ValueError: If *state_variable* carries an explicit shape, does
                not appear in *expr*, *indices* is empty, any index lacks an
                explicit range, a literal index lies beyond one period below
                the axis, or a constraint does not fit the system's indices.
                Also if state conditions alias each other cyclically with no
                unconstrained anchor, or a condition aliases its own region.

        Examples:
            >>> u = sp.IndexedBase('u')
            >>> i = sp.Idx('i', (0, 9))
            >>> ode = OrdinaryDifferentialSystem.from_sympy(u[i - 1] + u[i + 1], u, (i,))

            >>> # Pinned ends instead of the default periodic wrap
            >>> from lin_ode.ordinary_differential_system import Facet, DirichletCondition
            >>> ode = OrdinaryDifferentialSystem.from_sympy(
            ...     u[i - 1] + u[i + 1], u, (i,),
            ...     conditions=[
            ...         DirichletCondition(Facet.low(i), 0),
            ...         DirichletCondition(Facet.high(i), 0),
            ...     ],
            ... )
        """
        _reject_state_variable_with_shape(state_variable)

        if not indices:
            raise ValueError("indices must not be empty; provide at least one Idx.")
        for pos, idx in enumerate(indices):
            if not isinstance(idx, sp.Idx):
                raise TypeError(
                    f"indices[{pos}] must be an Idx, got {type(idx).__name__}: {idx}."
                )

        state_atoms = [
            atom for atom in expr.atoms(sp.Indexed) if atom.base == state_variable
        ]
        if not state_atoms:
            raise ValueError(
                f"State variable '{state_variable}' does not appear in the expression."
            )

        state_conditions, derivative_conditions = _partition_conditions(conditions)
        equation = _assemble_equation(
            expr,
            state_variable,
            indices,
            state_conditions,
            eliminate,
            derivative_conditions,
        )

        return cls(equation, conditions)
