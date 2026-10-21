"""The geometry layer: Region, Facet, Wall, the & algebra, and the box boundary
shell."""

from unittest import TestCase

from pytest import raises
from sympy import And, Eq, Idx, IndexedBase, Or
from sympy.core.cache import clear_cache

from lin_ode.ordinary_differential_system import (
    Box,
    BoxBoundary,
    Facet,
    PinnedRegion,
    Region,
    SupportsBoundary,
    Wall,
)


def _idx(name: str, index_range: tuple[int, int]) -> Idx:
    return Idx(name, index_range)  # type: ignore[assignment]


def _base(name: str) -> IndexedBase:
    return IndexedBase(name)  # type: ignore[assignment]


class FaceGeometry(TestCase):
    def setUp(self) -> None:
        clear_cache()

    def test_low_high_read_the_bounds(self) -> None:
        i = _idx("i", (0, 9))
        assert Facet.low(i).fixed == {i: 0}
        assert Facet.high(i).fixed == {i: 9}

    def test_condition_is_an_equality(self) -> None:
        i = _idx("i", (0, 9))
        assert Facet.low(i).condition == Eq(i, 0)

    def test_face_is_a_point_region(self) -> None:
        i = _idx("i", (0, 9))
        f = Facet.low(i)
        assert isinstance(f, PinnedRegion)
        assert isinstance(f, Region)

    def test_condition_is_the_pinned_point(self) -> None:
        i = _idx("i", (0, 9))
        assert Facet(i, 4).condition == Eq(i, 4)

    def test_negative_position_counts_from_the_end(self) -> None:
        i = _idx("i", (0, 9))
        assert Facet(i, -1).fixed == {i: 9}


class IntersectionAlgebra(TestCase):
    def setUp(self) -> None:
        clear_cache()

    def test_facet_and_facet_is_a_corner(self) -> None:
        i, j = _idx("i", (0, 9)), _idx("j", (0, 9))
        corner = Facet.low(i) & Facet.high(j)
        assert isinstance(corner, PinnedRegion)
        assert corner.fixed == {i: 0, j: 9}

    def test_corner_condition_is_a_conjunction(self) -> None:
        i, j = _idx("i", (0, 9)), _idx("j", (0, 9))
        corner = Facet.low(i) & Facet.low(j)
        assert corner.condition == And(Eq(i, 0), Eq(j, 0))  # type: ignore[arg-type]

    def test_chained_intersection(self) -> None:
        i, j, k = _idx("i", (0, 4)), _idx("j", (0, 4)), _idx("k", (0, 4))
        corner = Facet.low(i) & Facet.low(j) & Facet.high(k)
        assert corner.fixed == {i: 0, j: 0, k: 4}

    def test_same_axis_intersection_rejected(self) -> None:
        i = _idx("i", (0, 9))
        with raises(ValueError, match="pinned at"):
            _ = Facet.low(i) & Facet.high(i)


class BoxGeometry(TestCase):
    def setUp(self) -> None:
        clear_cache()

    def test_condition_is_the_range_conjunction(self) -> None:
        i = _idx("i", (0, 9))
        assert Box({i: (2, 5)}).condition == And(2 <= i, i <= 5)  # type: ignore[operator]

    def test_negative_range_resolves_from_the_end(self) -> None:
        i = _idx("i", (0, 9))
        assert Box({i: (-3, -1)}).ranges == {i: (7, 9)}

    def test_empty_box_rejected(self) -> None:
        with raises(ValueError, match="must not be empty"):
            _ = Box({})

    def test_empty_range_rejected(self) -> None:
        i = _idx("i", (0, 9))
        with raises(ValueError, match="empty"):
            _ = Box({i: (5, 2)})


class BoxBoundaryGeometry(TestCase):
    def setUp(self) -> None:
        clear_cache()

    def test_box_supports_boundary(self) -> None:
        i = _idx("i", (0, 9))
        box = Box({i: (2, 5)})
        assert isinstance(box, SupportsBoundary)
        assert isinstance(box.boundary, BoxBoundary)

    def test_boundary_is_a_region(self) -> None:
        i = _idx("i", (0, 9))
        assert isinstance(Box({i: (2, 5)}).boundary, Region)

    def test_condition_is_the_union_of_the_walls(self) -> None:
        """A 1D box has two walls (its two ends); the shell is their union."""
        i = _idx("i", (0, 9))
        bd = Box({i: (2, 5)}).boundary
        assert bd.condition == Or(Eq(i, 2), Eq(i, 5))  # type: ignore[arg-type]

    def test_walls_are_faces_that_own_their_clipped_condition(self) -> None:
        """A Wall is a Facet; in 2D it carries the clip on the other axis, so its
        condition (and the shell's) is the clipped wall, not the infinite plane."""
        i, j = _idx("i", (0, 9)), _idx("j", (0, 9))
        bd = Box({i: (2, 5), j: (1, 4)}).boundary
        assert all(isinstance(wall, Facet) for wall in bd.walls)
        low_i = next(w for w in bd.walls if w.condition.has(Eq(i, 2)))
        assert isinstance(low_i, Wall)
        assert low_i.condition == And(Eq(i, 2), And(1 <= j, j <= 4))  # type: ignore[arg-type]
        # the clipped wall condition broadcasts into the shell
        assert low_i.condition in bd.condition.args
