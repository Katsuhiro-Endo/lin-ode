"""Boundary conditions and interior dynamics, imposed on regions of the grid.

A *condition* is a rule imposed on a :class:`~.region.Region`.  There are two
families:

- :class:`StateCondition` pins the state ``u`` on a region; the ODE assembly
  differentiates ``u[region] = value`` in time to derive the Piecewise branch.
  An abstract category -- :class:`DirichletCondition` is the concrete member the user
  instantiates.
- :class:`DerivativeCondition` sets ``du/dt`` directly on a region -- a box (or
  face) running its own stencil.  It is concrete.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Self, Sequence, TypeVar, Generic

import sympy as sp

from .. import serialization as _ser
from .region import (
    PinnedRegion,
    Region,
    _resolve_position,
)

_RegionT = TypeVar("_RegionT", bound=Region)

class Condition(ABC, Generic[_RegionT]):
    """A rule attached to a region of the grid.

    The common base of both families: every condition keeps a
    :class:`~.region.Region` (its :attr:`region`), whose
    :attr:`~.region.Region.condition` is where the rule applies.  Subclasses add what
    the rule *is* -- a pinned value (:class:`StateCondition`) or a prescribed
    right-hand side (:class:`DerivativeCondition`).
    """

    def __init__(self, region: _RegionT) -> None:
        self._region = region

    @property
    def region(self) -> _RegionT:
        """The region this condition acts on."""
        return self._region

    @abstractmethod
    def to_json_data(self) -> dict[str, Any]:
        """Return the source inputs as a JSON-ready dict, tagged with the concrete
        class."""

    @staticmethod
    def from_json_data(json_data: dict[str, Any]) -> Condition:
        """Rebuild the concrete condition named in ``json_data['type']``."""
        return _CONDITION_TYPES[json_data["type"]]._from_json_data(json_data)

    @classmethod
    @abstractmethod
    def _from_json_data(cls, json_data: dict[str, Any]) -> Condition:
        """Rebuild from the dict that :meth:`to_json_data` made.

        Each class implements it.
        """


def _substitute_once(
    expr: sp.Expr,
    constraints: Sequence[StateCondition],
    state_variable: sp.IndexedBase,
    indices: Sequence[sp.Idx],
    transform: Callable[[sp.Expr], sp.Expr] = lambda value: value,
) -> sp.Expr:
    """One substitution pass: replace each state atom of *expr* that lies in a
    constraint's region by that constraint's value.

    For each ``u[...]`` atom, the first *constraints* entry whose region contains it
    (via :meth:`RegionStateCondition._binding_for`) supplies the replacement: its value
    with free indices bound to the atom's entries, passed through *transform* (the ODE
    assembler wraps the result to keep indexing periodic; the identity otherwise).  An
    atom that maps to itself is left untouched.

    A single pass -- callers loop it to a fixpoint when values may reference other
    constrained regions.
    """
    replacements: dict[sp.Indexed, sp.Basic] = {}
    for atom in expr.atoms(sp.Indexed):
        if atom.base != state_variable or len(atom.indices) != len(indices):
            continue
        for constraint in constraints:
            binding = constraint._binding_for(atom.indices, indices)
            if binding is not None:
                value = transform(
                    constraint.value.subs(binding, simultaneous=True)  # type: ignore[arg-type]
                )
                if value != atom:
                    replacements[atom] = value
                break
    return expr.xreplace(replacements) if replacements else expr


class StateCondition(Condition):
    r"""A rule pinning the state on a region: :math:`u[\text{region}] = \text{value}`.

    Abstract category -- the user instantiates :class:`DirichletCondition`.  The shared
    branch-emission machinery lives here.

    The region's unfixed indices are free, so the value may vary along the region
    (e.g. ``sin(j)`` along an edge) and may reference the state variable itself
    (e.g. the mirror ``u[0] = u[1]``).  The implied right-hand-side branch is the
    time derivative of the value: zero for a state-independent value, and the
    chain-rule combination of the interior dynamics at the referenced points
    otherwise.

    Args:
        region: The :class:`~.region.Region` the value is pinned on.
        value: What the state variable equals on the region.
    """

    def __init__(self, region: Region, value: sp.Basic | float) -> None:
        super().__init__(region)
        self._value: sp.Expr = sp.sympify(value)  # type: ignore[call-overload]

    @abstractmethod
    def _with_value(self, value: sp.Basic | float) -> Self:
        """This condition's region with a rewritten value.

        Internal: :meth:`substitute_into` and normalization only change the value,
        never the region, so they reuse ``self._region``.  A subclass returns an
        instance of its own type.
        """

    @abstractmethod
    def _binding_for(
        self, atom_indices: Sequence[sp.Basic], indices: Sequence[sp.Idx]
    ) -> dict[sp.Idx, sp.Expr] | None:
        """The free-index binding if a state atom with *atom_indices* lies in this
        condition's region, else ``None``.

        This is the one region-shape-specific hook the substitution machinery needs. The
        base does not know how a region decides membership; a region state condition
        answers it via its pinned positions (see :class:`RegionStateCondition`).  A
        region with no free indices returns an empty dict when the atom is inside.
        """

    @property
    def value(self) -> sp.Expr:
        """What the state variable equals on the region."""
        return self._value

    def rhs_branch(
        self,
        interior: sp.Expr,
        state_variable: sp.IndexedBase,
        indices: Sequence[sp.Idx],
    ) -> tuple[sp.Expr, sp.Basic]:
        r"""The Piecewise branch this condition implies for the RHS.

        Differentiates ``u[region] = value`` in time:

        .. math::
            \frac{d}{dt} u[\text{region}]
            = \sum_k \frac{\partial\, \text{value}}{\partial u[k]}
              \cdot \text{RHS}(k)

        so a state-independent value yields ``(0, condition)`` and an aliasing value
        like ``u[1]`` yields the interior expression evaluated at the referenced
        point.

        Args:
            interior: The interior RHS expression (already index-wrapped).
            state_variable: The system's state :class:`sympy.IndexedBase`.
            indices: The system's indices, in subscript order.

        Returns:
            The ``(expression, condition)`` pair for the Piecewise.
        """
        refs = [
            atom
            for atom in self._value.atoms(sp.Indexed)
            if atom.base == state_variable
        ]
        branch: sp.Expr = sp.S.Zero
        for ref in refs:
            if len(ref.indices) != len(indices):
                raise ValueError(
                    f"Condition value references {ref} with {len(ref.indices)} "
                    f"indices, but the system has {len(indices)}."
                )
            subs_map: dict[sp.Idx, sp.Expr] = {}
            for idx, entry in zip(indices, ref.indices):
                try:
                    literal = int(entry)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    subs_map[idx] = entry  # type: ignore[assignment]
                    continue
                # negative literals count from the end, Python-style
                subs_map[idx] = sp.Integer(_resolve_position(literal, idx))
            rhs_at_ref = interior.subs(subs_map, simultaneous=True)  # type: ignore[arg-type]
            branch = branch + self._value.diff(ref) * rhs_at_ref  # type: ignore[operator]
        return branch, self._region.condition

    def substitute_into(
        self,
        condition: RegionStateCondition,
        state_variable: sp.IndexedBase,
        indices: Sequence[sp.Idx],
    ) -> RegionStateCondition:
        r"""Replace references to *self*'s region in *condition*'s value by *self*'s
        value.

        Wherever *condition*'s value references a point in *self*'s region (all of
        *self*'s fixed axes matching literally), that reference is replaced by
        *self*'s value, with *self*'s free indices bound to the reference's entries
        at those axes.

        Returns:
            A new :class:`StateCondition` with *condition*'s fixed region and the
            substituted value.
        """
        new_value = _substitute_once(condition.value, [self], state_variable, indices)
        return condition._with_value(new_value)


def normalize_constraints(
    constraints: Sequence[RegionStateCondition],
    state_variable: sp.IndexedBase,
    indices: Sequence[sp.Idx],
) -> Sequence[RegionStateCondition]:
    r"""Resolve condition-to-condition aliasing so no value references another
    constrained region.

    Conditions may alias each other, e.g. ``u[0] = u[1]`` and ``u[1] = 5``: the
    first is only unambiguous once the second is known.  This substitutes each
    value against the others' fixed regions until every value is expressed without
    reference to another constrained point.

    A single pass may not fully resolve a chain or fan-out of aliases, so passes
    repeat until one changes nothing.

    Raises:
        ValueError: If a value references its own region (e.g. ``u[0] = u[0]``), or
            if conditions alias each other cyclically with no non-constrained anchor.
    """

    if not constraints:
        return []

    def _references_own_region(constraint: RegionStateCondition) -> bool:
        for atom in constraint.value.atoms(sp.Indexed):
            if atom.base != state_variable or len(atom.indices) != len(indices):
                continue
            if constraint._binding_for(atom.indices, indices) is not None:
                return True
        return False

    for constraint in constraints:
        if _references_own_region(constraint):
            raise ValueError(
                f"Condition {constraint.region.condition} aliases its own region; "
                "a condition aliasing self cannot be normalized."
            )

    current = list(constraints)
    # len(constraints) passes: the longest acyclic dependency chain among n conditions
    # is n - 1 hops deep, plus one pass to confirm no further change.
    for _ in range(len(constraints)):
        updated: list[RegionStateCondition] = []
        for i, original_target in enumerate(current):
            target = original_target
            for j, source in enumerate(current):
                if j == i:
                    continue
                target = source.substitute_into(target, state_variable, indices)
            updated.append(target)
        if all(new.value == old.value for new, old in zip(updated, current)):
            if any(_references_own_region(constraint) for constraint in updated):
                raise ValueError(
                    "Condition values alias each other cyclically; cannot normalize."
                )
            return updated
        current = updated
    raise ValueError("Condition values alias each other cyclically; cannot normalize.")


class RegionStateCondition(StateCondition):
    """A :class:`StateCondition` that resolves membership against its region.

    For a :class:`~.region.PinnedRegion` a state atom lies in the region when its
    entries at the pinned axes match, and its entries at the free axes give the binding
    for the value's free indices.  For any other region (e.g. a
    :class:`~.region.BoxBoundary`) membership is decided by evaluating the region's
    :attr:`~.region.Region.condition` at the atom's literal indices.  The named physics
    (Dirichlet) extend this.
    """

    def __init__(self, region: Region, value: sp.Basic | float) -> None:
        super().__init__(region, value)

    def _with_value(self, value: sp.Basic | float) -> RegionStateCondition:
        return RegionStateCondition(self._region, value)

    def to_json_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "region": self._region.to_json_data(),
            "value": _ser.sympy_to_str(self._value),
        }

    @classmethod
    def _from_json_data(cls, json_data: dict[str, Any]) -> RegionStateCondition:
        return cls(
            Region.from_json_data(json_data["region"]),
            _ser.sympy_from_str(json_data["value"]),
        )

    def _binding_for(
        self, atom_indices: Sequence[sp.Basic], indices: Sequence[sp.Idx]
    ) -> dict[sp.Idx, sp.Expr] | None:
        """Whether an atom lies on the region, and how its free indices bind.

        A :class:`~.region.PinnedRegion` resolves membership through its pinned axes:
        the atom must match at every fixed axis, and its free-axis entries give the
        binding for the value's free indices.  Any other region (e.g. a
        :class:`~.region.BoxBoundary`) has no single pinned point, so membership is
        decided by evaluating the region's :attr:`~.region.Region.condition` at the
        atom's literal indices; such a region binds no free indices.
        """
        if not isinstance(self._region, PinnedRegion):
            return {} if self._condition_holds_at(atom_indices, indices) else None
        fixed = self._region.fixed
        binding: dict[sp.Idx, sp.Expr] = {}
        for idx, entry in zip(indices, atom_indices):
            if idx in fixed:
                if entry != fixed[idx]:
                    return None
            else:
                binding[idx] = entry  # type: ignore[assignment]
        return binding

    def _condition_holds_at(
        self, atom_indices: Sequence[sp.Basic], indices: Sequence[sp.Idx]
    ) -> bool:
        """Whether the region's condition is satisfied at literal *atom_indices*.

        Non-literal entries (a free symbolic index) can't decide membership, so the atom
        is treated as off-region.
        """
        subs_map: dict[sp.Idx, sp.Basic] = {}
        for idx, entry in zip(indices, atom_indices):
            if not getattr(entry, "is_Integer", False):
                return False
            subs_map[idx] = entry
        return bool(self._region.condition.xreplace(subs_map) == sp.true)


class DirichletCondition(RegionStateCondition):
    """Dirichlet boundary: the state is pinned to a value on a region.

    ``u[region] = value``.  For a state-independent *value* the implied RHS branch is
    zero (the boundary never moves); a *value* that references the state (e.g. the
    mirror ``u[0] = u[1]``) yields the time derivative of that reference by the chain
    rule.

    Args:
        region: The :class:`~.region.Region` the value is pinned on -- a single
            :class:`~.region.Facet`, or a corner built explicitly with ``Facet & Facet``.
        value: The value the state is held at (defaults to ``0``).  It may vary along
            the free indices (e.g. ``sin(j)``) and may reference the state variable
            (e.g. ``u[1]``).
    """

    def __init__(self, region: Region, value: sp.Basic | float = 0) -> None:
        super().__init__(region, value)


class DerivativeCondition(Condition):
    r"""A region of the grid that runs its own dynamics: it sets ``du/dt`` directly.

    The derivative-side counterpart of :class:`StateCondition`: rather than pinning
    the state, it prescribes the dynamics, so it contributes its expression to the
    assembled :class:`sympy.Piecewise` verbatim (no chain rule).  On the region
    ``du/dt`` is *this* rule's stencil, replacing the ambient interior there;
    everywhere else the ambient interior applies.

    The stencil is written in the system's own indices (e.g. ``u[i - 1, j]``), and
    at the region edge it reads the neighbouring ambient ``u`` like any other point
    -- it has no boundary conditions of its own.  A region edge may sit on the grid
    boundary, but the region may not cross it (checked by the ODE).

    Args:
        region: The :class:`~.region.Region` the dynamics run on -- e.g. a
            :class:`~.region.Box`.
        rhs: The region's own right-hand side, in the system's state variable and
            indices.
    """

    def __init__(self, region: Region, rhs: sp.Basic | float) -> None:
        super().__init__(region)
        self._rhs: sp.Expr = sp.sympify(rhs)  # type: ignore[call-overload]

    @property
    def rhs(self) -> sp.Expr:
        """The region's own right-hand side, as given."""
        return self._rhs

    def to_json_data(self) -> dict[str, Any]:
        return {
            "type": "DerivativeCondition",
            "region": self._region.to_json_data(),
            "rhs": _ser.sympy_to_str(self._rhs),
        }

    @classmethod
    def _from_json_data(cls, json_data: dict[str, Any]) -> DerivativeCondition:
        return cls(
            Region.from_json_data(json_data["region"]),
            _ser.sympy_from_str(json_data["rhs"]),
        )


#: Concrete conditions by class name, for :meth:`Condition.from_json_data` dispatch.
_CONDITION_TYPES: dict[str, type[Condition]] = {
    c.__name__: c
    for c in (RegionStateCondition, DirichletCondition, DerivativeCondition)
}
