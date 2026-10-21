"""Regions of the grid, and the algebra that composes them.

A :class:`Region` is a subset of the grid, described by a symbolic
:attr:`~Region.condition` on the indices -- and nothing more at the root.  Two
capabilities layer on top, each on the regions that genuinely have it:

- :class:`PinnedRegion` -- a region defined by pinning indices to positions.  It
  exposes :attr:`~PinnedRegion.fixed` (the ``{index: position}`` map) and composes
  with ``&`` into the region pinned by both.
- :class:`Box` -- a box-shaped volume, each axis restricted to a range.  Its
  :attr:`~Box.boundary` is the :class:`BoxBoundary` shell around it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import sympy as sp

from .. import serialization as _ser


def _dump_idx_map(m: Mapping[sp.Idx, Any]) -> list[tuple[str, Any]]:
    """Encode an ``{Idx: value}`` map as ``[(srepr(idx), value), ...]``.

    JSON has no non-string keys. The ``Idx`` key is stored as its ``srepr`` string so it
    reconstructs without loss.
    """
    return [(_ser.sympy_to_str(idx), value) for idx, value in m.items()]


def _load_idx_map(pairs: list) -> dict[sp.Idx, Any]:
    """Invert :func:`_dump_idx_map`.

    Tuple values arrive from JSON as lists.
    """
    return {
        _ser.sympy_from_str(k): (tuple(v) if isinstance(v, list) else v)
        for k, v in pairs
    }


def _resolve_position(position: int, index: sp.Idx) -> int:
    """Resolve *position* against *index*'s bounds, Python-style.

    A position within ``[lower, upper]`` is taken literally; a position in
    ``[lower - size, lower)`` counts from the end (so ``-1`` is the last grid point
    when ``lower == 0``).

    Raises:
        ValueError: If *position* falls outside both ranges.
    """
    lower, upper = int(index.lower), int(index.upper)  # type: ignore[arg-type]
    size = upper - lower + 1
    resolved = position + size if position < lower else position
    if not lower <= resolved <= upper:
        raise ValueError(
            f"Position {position} is out of bounds for index {index}; "
            f"valid range is [{lower - size}, {upper}]."
        )
    return resolved


def _require_bounds(index: sp.Idx) -> None:
    if not isinstance(index, sp.Idx):
        raise TypeError(f"index must be an Idx, got {type(index).__name__}: {index}.")
    if index.lower is None or index.upper is None:
        raise ValueError(
            f"Index {index} has no explicit range; every index must be "
            "created with bounds, e.g. Idx('i', (0, N - 1))."
        )


class Region(ABC):
    """A subset of the grid, described by a symbolic condition on the indices.

    The root of the geometry hierarchy.  A region is *only* its :attr:`condition`;
    richer structure (pinned points) lives on the subtypes that actually have it.
    """

    @property
    @abstractmethod
    def condition(self) -> sp.Basic:
        """The region as a Boolean condition on the indices."""

    @abstractmethod
    def to_json_data(self) -> dict[str, Any]:
        """Return the source inputs as a JSON-ready dict, tagged with the concrete
        class."""

    @staticmethod
    def from_json_data(json_data: dict[str, Any]) -> Region:
        """Rebuild the concrete region named in ``json_data['type']``."""
        return _REGION_TYPES[json_data["type"]]._from_json_data(json_data)

    @classmethod
    @abstractmethod
    def _from_json_data(cls, json_data: dict[str, Any]) -> Region:
        """Rebuild from the dict that :meth:`to_json_data` made.

        Each class implements it.
        """


class PinnedRegion(Region):
    """A region defined by pinning indices to positions.

    Exposes :attr:`fixed` (the ``{index: position}`` map); its :attr:`condition` is the
    conjunction of those pins.  Concrete on its own (constructed from an explicit map);
    subtypes such as :class:`Facet` derive the map instead.  Two pinned regions compose
    with ``&`` into the region pinned by both.
    """

    def __init__(self, fixed: Mapping[sp.Idx, int]) -> None:
        self._fixed = dict(fixed)

    @property
    def fixed(self) -> dict[sp.Idx, int]:
        """Mapping from each pinned index to its resolved (non-negative) position."""
        return dict(self._fixed)

    @property
    def condition(self) -> sp.Basic:
        return sp.And(*(sp.Eq(idx, pos) for idx, pos in self.fixed.items()))  # type: ignore[arg-type]

    def to_json_data(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "fixed": _dump_idx_map(self._fixed)}

    @classmethod
    def _from_json_data(cls, json_data: dict[str, Any]) -> PinnedRegion:
        return cls(_load_idx_map(json_data["fixed"]))

    def __and__(self, other: PinnedRegion) -> PinnedRegion:
        """Intersect two pinned regions: narrow to the region pinned by both.

        Raises:
            ValueError: If the two regions pin the same axis at different positions.
        """
        merged = dict(self.fixed)
        for idx, pos in other.fixed.items():
            if idx in merged and merged[idx] != pos:
                raise ValueError(
                    f"Cannot intersect: axis {idx} is pinned at {merged[idx]} and "
                    f"{pos}."
                )
            merged[idx] = pos
        return PinnedRegion(merged)


@runtime_checkable
class SupportsBoundary(Protocol):
    """A region that can name its own :class:`Region` boundary (the shell around it)."""

    @property
    def boundary(self) -> Region: ...


@dataclass(frozen=True)
class Facet(PinnedRegion):
    """One axis pinned to one grid position: an endpoint in 1D, an edge in 2D, a face in
    3D (rigorously, an (N-1)-dimensional facet of the grid).

    Build one with :meth:`low` / :meth:`high`, which read the end positions off the
    index's own bounds; the bare constructor accepts an interior position but does not
    resolve or validate it (that happens on :attr:`fixed` access).

    Attributes:
        index: The :class:`sympy.Idx` this facet pins.
        position: The grid position along ``index`` (resolved on access).
    """

    index: sp.Idx
    position: int

    @classmethod
    def low(cls, index: sp.Idx) -> Facet:
        """The facet at the low end of ``index`` (``index.lower``)."""
        _require_bounds(index)
        return cls(index, int(index.lower))  # type: ignore[arg-type]

    @classmethod
    def high(cls, index: sp.Idx) -> Facet:
        """The facet at the high end of ``index`` (``index.upper``)."""
        _require_bounds(index)
        return cls(index, int(index.upper))  # type: ignore[arg-type]

    @property
    def fixed(self) -> dict[sp.Idx, int]:
        _require_bounds(self.index)
        return {self.index: _resolve_position(self.position, self.index)}

    def to_json_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "index": _ser.sympy_to_str(self.index),
            "position": self.position,
        }

    @classmethod
    def _from_json_data(cls, json_data: dict[str, Any]) -> Facet:
        return cls(_ser.sympy_from_str(json_data["index"]), json_data["position"])


@dataclass(frozen=True)
class Wall(Facet):
    """A :class:`Facet` clipped to a sub-range of its other axes.

    A bare facet at position ``pos`` on ``index`` is the whole (infinite) plane
    ``index = pos``; a wall is that plane confined by :attr:`clip`, which it folds into
    its :attr:`condition`.

    Attributes:
        clip: The Boolean condition confining the wall on its *other* axes
            (``sympy.true`` leaves the plane unclipped, i.e. a bare point in 1-D).
    """

    clip: sp.Basic = sp.true  # type: ignore[assignment]

    @property
    def condition(self) -> sp.Basic:
        """The wall: the facet's pinned condition clipped to the box's other axes."""
        return sp.And(super().condition, self.clip)  # type: ignore[arg-type]

    def to_json_data(self) -> dict[str, Any]:
        return {**super().to_json_data(), "clip": _ser.sympy_to_str(self.clip)}

    @classmethod
    def _from_json_data(cls, json_data: dict[str, Any]) -> Wall:
        return cls(
            _ser.sympy_from_str(json_data["index"]),
            json_data["position"],
            _ser.sympy_from_str(json_data["clip"]),
        )


class Box(Region):
    """A box-shaped volume region: each axis restricted to an inclusive range.

    An axis omitted from the range map spans its full grid extent (a slab).  Its
    :attr:`condition` is the conjunction of ``lo <= idx <= hi`` over the restricted
    axes.  A box is a plain :class:`Region` -- it has no single pinned point, so it is
    not a :class:`PinnedRegion`.

    Args:
        ranges: Mapping from an :class:`sympy.Idx` to the inclusive ``(lo, hi)`` range
            it is restricted to.  Negative positions count from the end, Python-style.

    Raises:
        ValueError: If *ranges* is empty, an axis lacks bounds, or a resolved range
            is empty (``lo > hi``).
        TypeError: If a key is not an ``Idx``.
    """

    def __init__(self, ranges: Mapping[sp.Idx, tuple[int, int]]) -> None:
        if not ranges:
            raise ValueError("box must not be empty; give at least one axis a range.")
        resolved: dict[sp.Idx, tuple[int, int]] = {}
        for index, (lo, hi) in ranges.items():
            _require_bounds(index)
            rlo, rhi = _resolve_position(lo, index), _resolve_position(hi, index)
            if rlo > rhi:
                raise ValueError(
                    f"box range for {index} is empty: resolved ({lo}, {hi}) -> "
                    f"({rlo}, {rhi}) with lo > hi."
                )
            resolved[index] = (rlo, rhi)
        self._ranges = resolved

    @property
    def ranges(self) -> dict[sp.Idx, tuple[int, int]]:
        """Mapping from each restricted index to its resolved ``(lo, hi)`` range."""
        return dict(self._ranges)

    @property
    def condition(self) -> sp.Basic:
        return sp.And(
            *(
                sp.And(lo <= idx, idx <= hi)  # type: ignore[operator]
                for idx, (lo, hi) in self._ranges.items()
            )
        )

    def to_json_data(self) -> dict[str, Any]:
        return {"type": "Box", "ranges": _dump_idx_map(self._ranges)}

    @classmethod
    def _from_json_data(cls, json_data: dict[str, Any]) -> Box:
        return cls(_load_idx_map(json_data["ranges"]))

    @property
    def boundary(self) -> BoxBoundary:
        """The shell of this box: the union of its walls."""
        return BoxBoundary(self)


class BoxBoundary(Region):
    """The boundary of a :class:`Box`: the :class:`Wall`\\ s at the two ends of each
    restricted axis, clipped to the box's other axes.

    Each wall owns its (clipped) condition, so the shell's :attr:`condition` is their
    union.
    """

    def __init__(self, box: Box) -> None:
        self._box = box

    def to_json_data(self) -> dict[str, Any]:
        return {"type": "BoxBoundary", "box": self._box.to_json_data()}

    @classmethod
    def _from_json_data(cls, json_data: dict[str, Any]) -> BoxBoundary:
        return cls(Box._from_json_data(json_data["box"]))

    @property
    def walls(self) -> list[Wall]:
        """Each :class:`Wall` of the box (both ends of every restricted axis)."""
        ranges = self._box.ranges
        walls: list[Wall] = []
        for axis, (lo, hi) in ranges.items():
            clip = sp.And(
                *(
                    sp.And(o_lo <= other, other <= o_hi)  # type: ignore[operator]
                    for other, (o_lo, o_hi) in ranges.items()
                    if other != axis
                )
            )
            for pos in (lo, hi):
                walls.append(Wall(axis, pos, clip))
        return walls

    @property
    def condition(self) -> sp.Basic:
        """The whole shell: the union of the walls' conditions."""
        return sp.Or(*(wall.condition for wall in self.walls))  # type: ignore[arg-type]


#: Concrete regions by class name, for :meth:`Region.from_json_data` dispatch.
_REGION_TYPES: dict[str, type[Region]] = {
    r.__name__: r for r in (PinnedRegion, Facet, Wall, Box, BoxBoundary)
}
