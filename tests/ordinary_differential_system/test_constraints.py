from unittest import TestCase
from pytest import raises

from lin_ode.ordinary_differential_system import (
    Box,
    DerivativeCondition,
    DirichletCondition,
    Facet,
    OrdinaryDifferentialSystem,
    normalize_constraints,
)
from sympy import And, Eq, Idx, Indexed, IndexedBase, Mod, Symbol, sin, true
from sympy.core.cache import clear_cache


def _indexed_base(name: str) -> IndexedBase:
    """Typed wrapper for ``IndexedBase(...)``; sympy's own constructor return type is
    too wide for pyright to narrow."""
    return IndexedBase(name)  # type: ignore[assignment]


def _idx(name: str, index_range: tuple[int, int]) -> Idx:
    """Typed wrapper for ``Idx(...)``; sympy's own constructor return type is too wide
    for pyright to narrow."""
    return Idx(name, index_range)  # type: ignore[assignment]


class RegionConstraintConstruction(TestCase):
    def setUp(self) -> None:
        clear_cache()

    def test_condition_single_index(self) -> None:
        i = _idx("i", (0, 20))
        constraint = DirichletCondition(Facet.low(i), 0)
        assert constraint.region.condition == Eq(i, 0)

    def test_condition_multiple_indices(self) -> None:
        i, j = _idx("i", (0, 20)), _idx("j", (0, 16))
        constraint = DirichletCondition(Facet.low(i) & Facet.high(j), 1)
        assert constraint.region.condition == And(Eq(i, 0), Eq(j, 16))  # type: ignore[arg-type]

    def test_negative_position_counts_from_end(self) -> None:
        i = _idx("i", (0, 20))
        constraint = DirichletCondition(Facet.high(i), 0)
        assert constraint.region.fixed == {i: 20}
        assert constraint.region.condition == Eq(i, 20)

    def test_negative_position_with_negative_lower_bound(self) -> None:
        # Range (-5, 5): -5 is a literal in-range position, -6 counts from the end
        i = _idx("i", (-5, 5))
        assert DirichletCondition(Facet(i, -5), 0).region.fixed == {i: -5}
        assert DirichletCondition(Facet(i, -6), 0).region.fixed == {i: 5}

    def test_position_out_of_bounds_rejected(self) -> None:
        """Only one Python-style wraparound band below lower is valid; a position past
        upper, or more than one period below lower, is rejected (no positive-side
        wraparound exists in Python indexing, and negative indexing only ever reaches
        back one period)."""
        i = _idx("i", (0, 20))
        with raises(ValueError, match="out of bounds"):
            _ = DirichletCondition(Facet(i, 21), 0).region.fixed
        with raises(ValueError, match="out of bounds"):
            _ = DirichletCondition(Facet(i, -22), 0).region.fixed

    def test_non_idx_key_rejected(self) -> None:
        with raises(TypeError, match="must be an Idx"):
            _ = DirichletCondition(Facet(Symbol("i"), 0), 0).region.fixed  # type: ignore[arg-type]

    def test_unbounded_index_rejected(self) -> None:
        i: Idx = Idx("i")  # type: ignore[assignment]
        with raises(ValueError, match="no explicit range"):
            _ = DirichletCondition(Facet.low(i), 0)


class ConstraintAssembly(TestCase):
    """Constraint -> Piecewise assembly through from_sympy."""

    def setUp(self) -> None:
        clear_cache()
        self.N = 21
        self.u = _indexed_base("u")
        self.h = Symbol("h")
        self.i = _idx("i", (0, self.N - 1))
        self.interior = (self.u[self.i - 1] + self.u[self.i + 1]) / self.h

    def test_dirichlet_branch_is_zero(self) -> None:
        """A state-independent value pins the point: du/dt = 0 there."""
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.interior,
            self.u,
            (self.i,),
            conditions=[DirichletCondition(Facet.low(self.i), 3)],
        )
        cases = ode.equation.cases
        assert cases[0] == (0, Eq(self.i, 0))
        assert cases[-1][1] == true

    def test_index_dependent_value_branch_is_zero(self) -> None:
        """A value varying along an edge is still state-independent: branch 0."""
        Nx, Ny = 5, 4
        u = _indexed_base("u")
        i, j = _idx("i", (0, Nx - 1)), _idx("j", (0, Ny - 1))
        interior = u[i - 1, j] + u[i, j + 1]

        ode = OrdinaryDifferentialSystem.from_sympy(
            interior, u, (i, j), conditions=[DirichletCondition(Facet.low(i), j**2)]
        )
        cases = ode.equation.cases
        assert cases[0] == (0, Eq(i, 0))

    def test_mirror_value_branch_is_interior_at_reference(self) -> None:
        """u[0] = u[1] derives du[0]/dt = RHS(1) by the chain rule."""
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.interior,
            self.u,
            (self.i,),
            conditions=[DirichletCondition(Facet.low(self.i), self.u[1])],
        )
        branch_expr, branch_cond = ode.equation.cases[0]
        assert branch_cond == Eq(self.i, 0)
        assert branch_expr == (self.u[0] + self.u[2]) / self.h

    def test_scaled_alias_value_uses_coefficient(self) -> None:
        """u[0] = 2*u[1] derives du[0]/dt = 2 * RHS(1)."""
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.interior,
            self.u,
            (self.i,),
            conditions=[DirichletCondition(Facet.low(self.i), 2 * self.u[1])],
        )
        branch_expr, _ = ode.equation.cases[0]
        assert (branch_expr - 2 * (self.u[0] + self.u[2]) / self.h).expand() == 0

    def test_user_order_decides_corner_vs_edge(self) -> None:
        """Precedence follows the user's list order (Piecewise is first-match): the
        first-listed condition wins where a corner and an edge overlap."""
        Nx, Ny = 5, 4
        u = _indexed_base("u")
        i, j = _idx("i", (0, Nx - 1)), _idx("j", (0, Ny - 1))
        interior = u[i - 1, j] + u[i, j + 1]

        # corner listed FIRST -> its branch precedes the edge it sits on
        corner_first = OrdinaryDifferentialSystem.from_sympy(
            interior,
            u,
            (i, j),
            conditions=[
                DirichletCondition(Facet.low(i) & Facet.low(j), 7),
                DirichletCondition(Facet.low(j), u[i, 1]),
            ],
        )
        assert corner_first.equation.cases[0][1] == And(Eq(i, 0), Eq(j, 0))  # type: ignore[arg-type]
        assert corner_first.equation.cases[1][1] == Eq(j, 0)

        # edge listed FIRST -> the edge branch precedes; the corner no longer wins
        edge_first = OrdinaryDifferentialSystem.from_sympy(
            interior,
            u,
            (i, j),
            conditions=[
                DirichletCondition(Facet.low(j), u[i, 1]),
                DirichletCondition(Facet.low(i) & Facet.low(j), 7),
            ],
        )
        assert edge_first.equation.cases[0][1] == Eq(j, 0)

    def test_constraints_stored(self) -> None:
        constraints = [DirichletCondition(Facet.low(self.i), 0)]
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.interior, self.u, (self.i,), conditions=constraints
        )
        assert ode.state_conditions == tuple(constraints)

    def test_foreign_index_rejected(self) -> None:
        k = _idx("k", (0, 5))
        with raises(ValueError, match="not one of the system's indices"):
            _ = OrdinaryDifferentialSystem.from_sympy(
                self.interior,
                self.u,
                (self.i,),
                conditions=[DirichletCondition(Facet.low(k), 0)],
            )

    def test_value_with_wrong_reference_arity_rejected(self) -> None:
        with raises(ValueError, match="indices, but the system has"):
            _ = OrdinaryDifferentialSystem.from_sympy(
                self.interior,
                self.u,
                (self.i,),
                conditions=[
                    DirichletCondition(Facet.low(self.i), Indexed(self.u, 0, 0))
                ],
            )


class PeriodicByDefault(TestCase):
    """Index arithmetic wraps around the axis unless a constraint pins it."""

    def setUp(self) -> None:
        clear_cache()
        self.N = 21
        self.u = _indexed_base("u")
        self.i = _idx("i", (0, self.N - 1))

    def test_symbolic_offsets_wrapped_in_mod(self) -> None:
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.u[self.i - 1] + self.u[self.i + 1], self.u, (self.i,)
        )
        atoms = ode.equation.atoms(Indexed)
        assert self.u[Mod(self.i - 1, self.N)] in atoms
        assert self.u[Mod(self.i + 1, self.N)] in atoms

    def test_wrap_resolves_at_boundaries(self) -> None:
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.u[self.i - 1] + self.u[self.i + 1], self.u, (self.i,)
        )
        expr = ode.equation.expr
        assert expr.subs(self.i, 0) == self.u[self.N - 1] + self.u[1]
        assert expr.subs(self.i, self.N - 1) == self.u[self.N - 2] + self.u[0]

    def test_bare_index_not_wrapped(self) -> None:
        ode = OrdinaryDifferentialSystem.from_sympy(
            -self.u[self.i],  # type: ignore[operator]
            self.u,
            (self.i,),
        )
        assert ode.equation.expr == -self.u[self.i]  # type: ignore[operator]

    def test_nonzero_lower_bound_wrap(self) -> None:
        u = _indexed_base("u")
        i = _idx("i", (1, 5))  # size 5, lower bound 1
        ode = OrdinaryDifferentialSystem.from_sympy(u[i + 1] + u[i], u, (i,))
        expr = ode.equation.expr
        assert expr.subs(i, 5) == u[1] + u[5]


class Elimination(TestCase):
    """Eliminate=True substitutes constrained references out of the piecewise."""

    def setUp(self) -> None:
        clear_cache()
        self.N = 8
        self.u = _indexed_base("u")
        self.h = Symbol("h")
        self.i = _idx("i", (0, self.N - 1))
        self.diffusion = (
            self.u[self.i - 1] - 2 * self.u[self.i] + self.u[self.i + 1]  # type: ignore[operator]
        ) / self.h**2

    def test_mirror_eliminates_aliased_point(self) -> None:
        """u[0] = u[1]: rows 0 and 1 both become (-u[1] + u[2])/h**2 and no
        expression references u[0] anymore."""
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            self.u,
            (self.i,),
            conditions=[DirichletCondition(Facet.low(self.i), self.u[1])],
            eliminate=True,
        )
        u, h, i = self.u, self.h, self.i
        expected = (-u[1] + u[2]) / h**2  # type: ignore[operator]
        assert ode.equation.expr.subs(i, 0) == expected
        assert ode.equation.expr.subs(i, 1) == expected
        # the periodic wrap reference from the far end resolves to u[1] too
        assert ode.equation.expr.subs(i, 7) == (u[1] + u[6] - 2 * u[7]) / h**2  # type: ignore[operator]
        assert u[0] not in ode.equation.atoms(Indexed)

    def test_dirichlet_values_substituted_into_adjacent_rows(self) -> None:
        g = Symbol("g")
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            self.u,
            (self.i,),
            conditions=[
                DirichletCondition(Facet.low(self.i), 0),
                DirichletCondition(Facet.high(self.i), g),
            ],
            eliminate=True,
        )
        u, h, i = self.u, self.h, self.i
        assert ode.equation.expr.subs(i, 1) == (-2 * u[1] + u[2]) / h**2
        assert ode.equation.expr.subs(i, 6) == (g + u[5] - 2 * u[6]) / h**2
        assert u[0] not in ode.equation.atoms(Indexed)
        assert u[7] not in ode.equation.atoms(Indexed)

    def test_pinned_rows_survive_elimination(self) -> None:
        """Elimination rewrites RHS expressions; the constrained rows keep their own
        branches (system dimension is unchanged)."""
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            self.u,
            (self.i,),
            conditions=[DirichletCondition(Facet.low(self.i), 3)],
            eliminate=True,
        )
        assert ode.equation.expr.subs(self.i, 0) == 0

    def test_literal_reference_in_interior_eliminated(self) -> None:
        """A literal reference to a constrained point directly in the interior
        expression (not via a stencil offset) is also substituted."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            u[0] + u[i],
            u,
            (i,),
            conditions=[DirichletCondition(Facet.low(i), 5)],
            eliminate=True,
        )
        assert ode.equation.expr.subs(i, 2) == u[2] + 5
        assert u[0] not in ode.equation.expr.atoms(Indexed)

    def test_negative_index_resolves_to_pinned_hard_wall(self) -> None:
        """U[-1] (Python-style, i.e. the axis's last point) is the pinned Dirichlet
        value when that end is constrained, not a periodic wraparound artifact."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            u[i - 1] + u[-1],  # type: ignore[operator]
            u,
            (i,),
            conditions=[DirichletCondition(Facet.high(i), 9)],
            eliminate=True,
        )
        assert ode.equation.expr.subs(i, 3) == u[2] + 9
        assert u[self.N - 1] not in ode.equation.atoms(Indexed)

    def test_wide_stencil_past_hard_wall_eliminated_on_every_touching_row(self) -> None:
        """A stencil wider than the pin's immediate reach (u[i-2]/u[i+2]) still gets the
        pinned Dirichlet value substituted on every row it touches, not just the row
        adjacent to the wall."""
        u, h, i = self.u, self.h, self.i
        wide_stencil = (u[i - 2] + u[i + 2]) / h**2  # type: ignore[operator]
        ode = OrdinaryDifferentialSystem.from_sympy(
            wide_stencil,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), 0),
                DirichletCondition(Facet.high(i), 0),
            ],
            eliminate=True,
        )
        # i=1: u[i-2] == u[-1] wraps Python-style to u[7], the pinned right wall -> 0
        assert ode.equation.expr.subs(i, 1) == u[3] / h**2
        # i=6: u[i+2] == u[8] wraps Python-style to u[0], the pinned left wall -> 0
        assert ode.equation.expr.subs(i, 6) == u[4] / h**2
        assert u[0] not in ode.equation.atoms(Indexed)
        assert u[self.N - 1] not in ode.equation.atoms(Indexed)

    def test_2d_corner_intersection_branch(self) -> None:
        """A point diagonally coupled to a corner gets a combined-assignment branch with
        the corner value substituted."""
        u, h = self.u, self.h
        i, j = _idx("i", (0, 4)), _idx("j", (0, 3))
        lap = (
            u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
        ) / h**2  # type: ignore[operator]

        ode = OrdinaryDifferentialSystem.from_sympy(
            lap,
            u,
            (i, j),
            conditions=[DirichletCondition(Facet.low(i) & Facet.low(j), 7)],
            eliminate=True,
        )
        row = ode.equation.expr.subs({i: 1, j: 0})
        assert (
            row - (7 + u[2, 0] + u[1, 3] + u[1, 1] - 4 * u[1, 0]) / h**2  # type: ignore[operator]
        ).expand() == 0
        assert u[0, 0] not in ode.equation.atoms(Indexed)

    def test_unsupported_index_map_rejected(self) -> None:
        with raises(ValueError, match="only literal offsets"):
            _ = OrdinaryDifferentialSystem.from_sympy(
                self.u[2 * self.i] + self.u[self.i],
                self.u,
                (self.i,),
                conditions=[DirichletCondition(Facet.low(self.i), 0)],
                eliminate=True,
            )

    def test_unsupported_index_map_fine_without_eliminate(self) -> None:
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.u[2 * self.i] + self.u[self.i],
            self.u,
            (self.i,),
            conditions=[DirichletCondition(Facet.low(self.i), 0)],
        )
        assert ode.total_dof == self.N

    def test_cyclic_aliasing_rejected(self) -> None:
        """u[0] = u[7] and u[7] = u[0] cannot be eliminated."""
        with raises(ValueError, match="cyclically"):
            _ = OrdinaryDifferentialSystem.from_sympy(
                self.diffusion,
                self.u,
                (self.i,),
                conditions=[
                    DirichletCondition(Facet.low(self.i), self.u[7]),
                    DirichletCondition(Facet.high(self.i), self.u[0]),
                ],
                eliminate=True,
            )

    def test_eliminate_defaults_off(self) -> None:
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            self.u,
            (self.i,),
            conditions=[DirichletCondition(Facet.low(self.i), self.u[1])],
        )
        # without elimination the chain-rule branch still references u[0]
        assert self.u[0] in ode.equation.atoms(Indexed)

    def test_negative_literal_in_constraint_value(self) -> None:
        """A value like u[-2] counts from the end, also inside the derived branch."""
        u, h, i = self.u, self.h, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[DirichletCondition(Facet.high(i), u[-2])],
        )
        # d/dt u[7] = RHS(6)
        assert ode.equation.expr.subs(i, 7) == (u[5] - 2 * u[6] + u[7]) / h**2  # type: ignore[operator]


class NestedConstraints(TestCase):
    """Constraints that reference or overlap with other constraints under
    eliminate=True."""

    def setUp(self) -> None:
        clear_cache()
        self.N = 8
        self.u = _indexed_base("u")
        self.h = Symbol("h")
        self.i = _idx("i", (0, self.N - 1))
        self.diffusion = (
            self.u[self.i - 1] - 2 * self.u[self.i] + self.u[self.i + 1]  # type: ignore[operator]
        ) / self.h**2

    def test_chain_mirror_then_constant(self) -> None:
        """u[0] = u[1], u[1] = 5: both rows 0 and 1 pin to the constant, chained
        through one hop of aliasing."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), u[1]),
                DirichletCondition(Facet(i, 1), 5),
            ],
            eliminate=True,
        )
        assert ode.equation.expr.subs(i, 0) == 0
        assert ode.equation.expr.subs(i, 1) == 0
        assert u[0] not in ode.equation.atoms(Indexed)
        assert u[1] not in ode.equation.atoms(Indexed)

    def test_chain_reverse_direction(self) -> None:
        """u[1] = u[0], u[0] = 5: same chain, opposite aliasing direction and
        opposite constraint list order; same result as test_chain_mirror_then_constant."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet(i, 1), u[0]),
                DirichletCondition(Facet.low(i), 5),
            ],
            eliminate=True,
        )
        assert ode.equation.expr.subs(i, 0) == 0
        assert ode.equation.expr.subs(i, 1) == 0
        assert u[0] not in ode.equation.atoms(Indexed)
        assert u[1] not in ode.equation.atoms(Indexed)

    def test_three_deep_chain(self) -> None:
        """u[0] = u[1], u[1] = u[2], u[2] = 5: substitution recurses two hops."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), u[1]),
                DirichletCondition(Facet(i, 1), u[2]),
                DirichletCondition(Facet(i, 2), 5),
            ],
            eliminate=True,
        )
        assert ode.equation.expr.subs(i, 0) == 0
        assert ode.equation.expr.subs(i, 1) == 0
        assert ode.equation.expr.subs(i, 2) == 0
        for k in (0, 1, 2):
            assert u[k] not in ode.equation.atoms(Indexed)

    def test_five_deep_chain(self) -> None:
        """U[0]=u[1], u[1]=u[2], u[2]=u[3], u[3]=u[4], u[4]=9: a longer chain still
        converges (stresses the substitution loop's iteration bound)."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), u[1]),
                DirichletCondition(Facet(i, 1), u[2]),
                DirichletCondition(Facet(i, 2), u[3]),
                DirichletCondition(Facet(i, 3), u[4]),
                DirichletCondition(Facet(i, 4), 9),
            ],
            eliminate=True,
        )
        for k in range(5):
            assert ode.equation.expr.subs(i, k) == 0
            assert u[k] not in ode.equation.atoms(Indexed)

    def test_three_cycle_rejected(self) -> None:
        """U[0]=u[1], u[1]=u[2], u[2]=u[0]: a longer cycle with no constant anchor still
        cannot be eliminated."""
        u, i = self.u, self.i
        with raises(ValueError, match="cyclically"):
            _ = OrdinaryDifferentialSystem.from_sympy(
                self.diffusion,
                u,
                (i,),
                conditions=[
                    DirichletCondition(Facet.low(i), u[1]),
                    DirichletCondition(Facet(i, 1), u[2]),
                    DirichletCondition(Facet(i, 2), u[0]),
                ],
                eliminate=True,
            )

    def test_scaled_cycle_rejected(self) -> None:
        """u[0] = 2*u[1], u[1] = u[0]: a coefficient on one leg doesn't make the
        cycle resolvable."""
        u, i = self.u, self.i
        with raises(ValueError, match="cyclically"):
            _ = OrdinaryDifferentialSystem.from_sympy(
                self.diffusion,
                u,
                (i,),
                conditions=[
                    DirichletCondition(Facet.low(i), 2 * u[1]),
                    DirichletCondition(Facet(i, 1), u[0]),
                ],
                eliminate=True,
            )

    def test_fan_in_shared_reference(self) -> None:
        """u[0] = u[2] and u[7] = u[2]: both ends alias to the same interior point;
        both resolve to that point's dynamics and neither end remains referenced."""
        u, h, i = self.u, self.h, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), u[2]),
                DirichletCondition(Facet.high(i), u[2]),
            ],
            eliminate=True,
        )
        expected_at_2 = (u[1] - 2 * u[2] + u[3]) / h**2  # type: ignore[operator]
        assert ode.equation.expr.subs(i, 0) == expected_at_2
        assert ode.equation.expr.subs(i, 7) == expected_at_2
        assert u[0] not in ode.equation.atoms(Indexed)
        assert u[7] not in ode.equation.atoms(Indexed)

    def test_corner_aliases_edge_value_on_same_edge(self) -> None:
        """Edge i=0 is pinned to the constant 7; the corner (0,0) aliases to a
        neighboring point on that same edge (u[0,1]), which the edge's own constant
        resolves.

        The corner (higher precedence) must substitute through the edge's constant
        rather than leave u[0,1] unresolved.
        """
        u, h = self.u, self.h
        i, j = _idx("i", (0, 4)), _idx("j", (0, 3))
        lap = (
            u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
        ) / h**2  # type: ignore[operator]

        ode = OrdinaryDifferentialSystem.from_sympy(
            lap,
            u,
            (i, j),
            conditions=[
                DirichletCondition(Facet.low(i), 7),
                DirichletCondition(Facet.low(i) & Facet.low(j), u[0, 1]),
            ],
            eliminate=True,
        )
        assert ode.equation.expr.subs({i: 0, j: 0}) == 0
        assert u[0, 0] not in ode.equation.atoms(Indexed)
        assert u[0, 1] not in ode.equation.atoms(Indexed)

    def test_corner_aliases_point_on_other_constrained_edge(self) -> None:
        """Corner (0,0) aliases to u[1,0], which lies on the OTHER constrained edge
        (j=0, pinned to the constant 3), not the edge (i=0) the corner itself sits on.

        Precedence by number of fixed indices must not cause this to be resolved against
        the wrong edge.
        """
        u, h = self.u, self.h
        i, j = _idx("i", (0, 4)), _idx("j", (0, 3))
        lap = (
            u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
        ) / h**2  # type: ignore[operator]

        ode = OrdinaryDifferentialSystem.from_sympy(
            lap,
            u,
            (i, j),
            conditions=[
                DirichletCondition(Facet.low(j), 3),
                DirichletCondition(Facet.low(i) & Facet.low(j), u[1, 0]),
            ],
            eliminate=True,
        )
        assert ode.equation.expr.subs({i: 0, j: 0}) == 0
        assert u[0, 0] not in ode.equation.atoms(Indexed)
        assert u[1, 0] not in ode.equation.atoms(Indexed)

    def test_edge_aliases_far_end_pinned_to_constant(self) -> None:
        """u[0] = u[7] (mirrors to the far end), and u[7] = 3 independently (the far
        end is itself pinned). Interior rows that wrap around to reference u[0]
        must resolve all the way to the constant 3, not stop at the intermediate
        u[7]."""
        u, h, i = self.u, self.h, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), u[7]),
                DirichletCondition(Facet.high(i), 3),
            ],
            eliminate=True,
        )
        # i=1: u[i-1] wraps to u[0], which aliases u[7], which is pinned to 3
        assert ode.equation.expr.subs(i, 1) == (3 - 2 * u[1] + u[2]) / h**2  # type: ignore[operator]
        assert ode.equation.expr.subs(i, 0) == 0
        assert ode.equation.expr.subs(i, 7) == 0
        assert u[0] not in ode.equation.atoms(Indexed)
        assert u[7] not in ode.equation.atoms(Indexed)

    def test_self_referential_constraint_rejected(self) -> None:
        """u[0] = u[0] is a degenerate self-alias, not a resolvable constraint;
        it must be rejected with a message identifying it as self-referential,
        not fall through to the generic cyclic-aliasing error."""
        u, i = self.u, self.i
        with raises(ValueError, match="aliasing self"):
            _ = OrdinaryDifferentialSystem.from_sympy(
                self.diffusion,
                u,
                (i,),
                conditions=[DirichletCondition(Facet.low(i), u[0])],
                eliminate=True,
            )


class AliasingWithoutElimination(TestCase):
    """Aliasing between conditions is resolved whether or not eliminate is set.

    A value that references another constrained site must see that site's fixed value:
    deriving the chain rule against the unconstrained dynamics there would break the
    aliasing condition itself.
    """

    def setUp(self) -> None:
        clear_cache()
        self.N = 8
        self.u = _indexed_base("u")
        self.h = Symbol("h")
        self.i = _idx("i", (0, self.N - 1))
        self.diffusion = (
            self.u[self.i - 1] - 2 * self.u[self.i] + self.u[self.i + 1]  # type: ignore[operator]
        ) / self.h**2

    def test_chain_resolves_without_eliminate(self) -> None:
        """u[0] = u[7], u[7] = 5: row 0 follows row 7, which is pinned, so both rows
        are 0.  Deriving row 0 against the unconstrained dynamics at site 7 would give
        it a non-zero derivative and let u[0] drift away from u[7]."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), u[7]),
                DirichletCondition(Facet.high(i), 5),
            ],
        )
        assert ode.equation.expr.subs(i, 0) == 0
        assert ode.equation.expr.subs(i, 7) == 0

    def test_cyclic_aliasing_rejected_without_eliminate(self) -> None:
        """u[0] = u[7] and u[7] = u[0] has no anchor, on either path."""
        u, i = self.u, self.i
        with raises(ValueError, match="cyclically"):
            _ = OrdinaryDifferentialSystem.from_sympy(
                self.diffusion,
                u,
                (i,),
                conditions=[
                    DirichletCondition(Facet.low(i), u[7]),
                    DirichletCondition(Facet.high(i), u[0]),
                ],
            )

    def test_no_conditions_is_accepted(self) -> None:
        """Normalizing an empty condition list returns it unchanged, on either path."""
        u, i = self.u, self.i
        for eliminate in (False, True):
            ode = OrdinaryDifferentialSystem.from_sympy(
                self.diffusion, u, (i,), eliminate=eliminate
            )
            assert ode.equation.expr == self.diffusion.subs(
                {u[i - 1]: u[Mod(i + 7, 8)], u[i + 1]: u[Mod(i + 1, 8)]}
            )


class SubstituteInto(TestCase):
    """StateCondition.substitute_into replaces references to one constraint's region in
    another constraint's value."""

    def setUp(self) -> None:
        clear_cache()
        self.u = _indexed_base("u")
        self.i = _idx("i", (0, 7))

    def test_replaces_matching_reference(self) -> None:
        """u[1] = 5 substituted into u[0] = u[1] gives u[0] = 5."""
        u, i = self.u, self.i
        source = DirichletCondition(Facet(i, 1), 5)
        target = DirichletCondition(Facet.low(i), u[1])
        result = source.substitute_into(target, u, (i,))
        assert result.region.fixed == {i: 0}
        assert result.value == 5

    def test_no_match_is_a_no_op(self) -> None:
        """u[1] = 5 substituted into u[0] = u[2] (no reference to u[1]) changes
        nothing."""
        u, i = self.u, self.i
        source = DirichletCondition(Facet(i, 1), 5)
        target = DirichletCondition(Facet.low(i), u[2])
        result = source.substitute_into(target, u, (i,))
        assert result.region.fixed == {i: 0}
        assert result.value == u[2]

    def test_2d_edge_free_index_binding(self) -> None:
        """Edge i=0 pinned to sin(j); a corner aliasing u[0, 2] picks up sin(2) (the
        source's free index j bound to the reference's entry there)."""
        u = self.u
        i, j = _idx("i", (0, 4)), _idx("j", (0, 3))
        source = DirichletCondition(Facet.low(i), sin(j))  # type: ignore[arg-type]
        target = DirichletCondition(Facet.high(i) & Facet.high(j), u[0, 2])
        result = source.substitute_into(target, u, (i, j))
        assert result.region.fixed == {i: 4, j: 3}
        assert result.value == sin(2)

    def test_self_returns_own_value_for_matching_region(self) -> None:
        """A constraint substituted into itself (self-alias) replaces its own
        reference with its own value: u[0] = u[0] normalizes its literal
        self-reference away."""
        u, i = self.u, self.i
        constraint = DirichletCondition(Facet.low(i), u[0])
        result = constraint.substitute_into(constraint, u, (i,))
        assert result.value == u[0]


class NormalizeConstraints(TestCase):
    """normalize_constraints resolves constraint-to-constraint aliasing so no
    constraint's value references another constrained region."""

    def setUp(self) -> None:
        clear_cache()
        self.u = _indexed_base("u")
        self.i = _idx("i", (0, 7))

    def test_alias_to_edge_pinned_constant(self) -> None:
        """u[0] = u[1], u[1] = 5 -> u[0] = 5, u[1] = 5."""
        u, i = self.u, self.i
        result = normalize_constraints(
            [
                DirichletCondition(Facet.low(i), u[1]),
                DirichletCondition(Facet(i, 1), 5),
            ],
            u,
            (i,),
        )
        assert [c.region.fixed for c in result] == [{i: 0}, {i: 1}]
        assert [c.value for c in result] == [5, 5]

    def test_alias_reverse_direction(self) -> None:
        """u[0] = 5, u[1] = u[0] -> u[0] = 5, u[1] = 5 (order of the constraint
        list, and which side is the alias, doesn't matter)."""
        u, i = self.u, self.i
        result = normalize_constraints(
            [
                DirichletCondition(Facet.low(i), 5),
                DirichletCondition(Facet(i, 1), u[0]),
            ],
            u,
            (i,),
        )
        assert [c.region.fixed for c in result] == [{i: 0}, {i: 1}]
        assert [c.value for c in result] == [5, 5]

    def test_chain_of_aliases_to_third_point(self) -> None:
        """u[0] = u[1], u[1] = u[2] -> u[0] = u[2], u[1] = u[2] (u[2] itself is
        not constrained, so it remains a state reference)."""
        u, i = self.u, self.i
        result = normalize_constraints(
            [
                DirichletCondition(Facet.low(i), u[1]),
                DirichletCondition(Facet(i, 1), u[2]),
            ],
            u,
            (i,),
        )
        assert [c.region.fixed for c in result] == [{i: 0}, {i: 1}]
        assert [c.value for c in result] == [u[2], u[2]]

    def test_mutual_alias_to_common_third_point(self) -> None:
        """u[0] = u[2], u[1] = u[0] -> u[0] = u[2], u[1] = u[2]."""
        u, i = self.u, self.i
        result = normalize_constraints(
            [
                DirichletCondition(Facet.low(i), u[2]),
                DirichletCondition(Facet(i, 1), u[0]),
            ],
            u,
            (i,),
        )
        assert [c.region.fixed for c in result] == [{i: 0}, {i: 1}]
        assert [c.value for c in result] == [u[2], u[2]]

    def test_independent_references_unchanged(self) -> None:
        """u[0] = u[2], u[1] = u[3]: neither references a constrained region, so
        normalization is a no-op."""
        u, i = self.u, self.i
        result = normalize_constraints(
            [
                DirichletCondition(Facet.low(i), u[2]),
                DirichletCondition(Facet(i, 1), u[3]),
            ],
            u,
            (i,),
        )
        assert [c.region.fixed for c in result] == [{i: 0}, {i: 1}]
        assert [c.value for c in result] == [u[2], u[3]]

    def test_self_referential_constraint_rejected(self) -> None:
        """u[0] = u[0] is rejected directly, not via the generic cyclic error."""
        u, i = self.u, self.i
        with raises(ValueError, match="aliasing self"):
            _ = normalize_constraints([DirichletCondition(Facet.low(i), u[0])], u, (i,))

    def test_two_cycle_rejected(self) -> None:
        """u[0] = u[1], u[1] = u[0]: no constant anchor, cannot be normalized."""
        u, i = self.u, self.i
        with raises(ValueError, match="cyclically"):
            _ = normalize_constraints(
                [
                    DirichletCondition(Facet.low(i), u[1]),
                    DirichletCondition(Facet(i, 1), u[0]),
                ],
                u,
                (i,),
            )

    def test_three_cycle_rejected(self) -> None:
        """u[0] = u[1], u[1] = u[2], u[2] = u[0]: a longer cycle with no anchor."""
        u, i = self.u, self.i
        with raises(ValueError, match="cyclically"):
            _ = normalize_constraints(
                [
                    DirichletCondition(Facet.low(i), u[1]),
                    DirichletCondition(Facet(i, 1), u[2]),
                    DirichletCondition(Facet(i, 2), u[0]),
                ],
                u,
                (i,),
            )

    def test_2d_corner_aliases_edge_pinned_constant(self) -> None:
        """Corner (i=0, j=0) = u[1,0], and edge j=0 is pinned to the constant 5
        (the corner's alias target u[1,0] lies on that edge) -> corner's value
        normalizes to 5."""
        u = self.u
        i, j = _idx("i", (0, 4)), _idx("j", (0, 3))
        result = normalize_constraints(
            [
                DirichletCondition(Facet.low(i) & Facet.low(j), u[1, 0]),
                DirichletCondition(Facet.low(j), 5),
            ],
            u,
            (i, j),
        )
        assert [c.region.fixed for c in result] == [{i: 0, j: 0}, {j: 0}]
        assert [c.value for c in result] == [5, 5]

    def test_2d_corner_aliases_other_corner(self) -> None:
        """Corner (i=0, j=0) = u[4,3], corner (i=4, j=3) = 5 -> first
        corner's value normalizes to 5."""
        u = self.u
        i, j = _idx("i", (0, 4)), _idx("j", (0, 3))
        result = normalize_constraints(
            [
                DirichletCondition(Facet.low(i) & Facet.low(j), u[4, 3]),
                DirichletCondition(Facet.high(i) & Facet.high(j), 5),
            ],
            u,
            (i, j),
        )
        assert [c.region.fixed for c in result] == [{i: 0, j: 0}, {i: 4, j: 3}]
        assert [c.value for c in result] == [5, 5]

    def test_2d_corner_cycle_rejected(self) -> None:
        """Two corners aliasing each other with no anchor cannot be normalized."""
        u = self.u
        i, j = _idx("i", (0, 4)), _idx("j", (0, 3))
        with raises(ValueError, match="cyclically"):
            _ = normalize_constraints(
                [
                    DirichletCondition(Facet.low(i) & Facet.low(j), u[4, 3]),
                    DirichletCondition(Facet.high(i) & Facet.high(j), u[0, 0]),
                ],
                u,
                (i, j),
            )

    def test_2d_unrelated_constraints_unchanged(self) -> None:
        """An edge pinned to a constant and a corner whose value depends only on a free
        index (not on any constrained region): normalization is a no-op."""
        u = self.u
        i, j = _idx("i", (0, 4)), _idx("j", (0, 3))
        result = normalize_constraints(
            [
                DirichletCondition(Facet.low(i), 5),
                DirichletCondition(Facet.high(i) & Facet.high(j), sin(j)),  # type: ignore[arg-type]
            ],
            u,
            (i, j),
        )
        assert [c.region.fixed for c in result] == [{i: 0}, {i: 4, j: 3}]
        assert [c.value for c in result] == [5, sin(j)]

    def test_chain_with_fan_out_converges_across_multiple_passes(self) -> None:
        """U[0]=u[1], u[1]=u[2], u[4]=u[1], u[2]=5: a chain (u[0]->u[1]->u[2]) with a
        fan-out (u[4] also aliasing u[1]) that a single pass cannot fully resolve
        regardless of list order, but repeated passes do."""
        u, i = self.u, self.i
        result = normalize_constraints(
            [
                DirichletCondition(Facet.low(i), u[1]),
                DirichletCondition(Facet(i, 1), u[2]),
                DirichletCondition(Facet(i, 4), u[1]),
                DirichletCondition(Facet(i, 2), 5),
            ],
            u,
            (i,),
        )
        assert [c.region.fixed for c in result] == [{i: 0}, {i: 1}, {i: 4}, {i: 2}]
        assert [c.value for c in result] == [5, 5, 5, 5]


class NamedBoundaryConditions(TestCase):
    def setUp(self) -> None:
        clear_cache()
        self.u = _indexed_base("u")
        self.i = _idx("i", (0, 5))
        self.j = _idx("j", (0, 5))

    def test_dirichlet_pins_a_constant_value(self) -> None:
        """Dirichlet is a constant-value face; its RHS branch is zero."""
        d = DirichletCondition(Facet.low(self.i), 3)
        assert d.value == 3
        expr, cond = d.rhs_branch(
            self.u[self.i - 1] + self.u[self.i + 1], self.u, (self.i,)
        )
        assert expr == 0
        assert cond == Eq(self.i, 0)

    def test_dirichlet_defaults_to_zero(self) -> None:
        assert DirichletCondition(Facet.low(self.i)).value == 0


class Subdomains(TestCase):
    """A DerivativeCondition runs its own stencil on a box, replacing the ambient
    interior there; its edges read the ambient field, and boundary conditions win over
    it."""

    def setUp(self) -> None:
        clear_cache()
        self.N = 8
        self.u = _indexed_base("u")
        self.h = Symbol("h")
        self.i = _idx("i", (0, self.N - 1))
        self.diffusion = (
            self.u[self.i - 1] - 2 * self.u[self.i] + self.u[self.i + 1]  # type: ignore[operator]
        ) / self.h**2

    def test_box_replaces_ambient_inside_only(self) -> None:
        """On the box du/dt is the subdomain stencil; off it, the ambient interior."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[DerivativeCondition(Box({i: (3, 5)}), -u[i])],  # type: ignore[operator]
        )
        expr = ode.equation.expr
        assert expr.subs(i, 4) == -u[4]  # type: ignore[operator]  # inside the box: reaction
        assert expr.subs(i, 1) == (u[0] - 2 * u[1] + u[2]) / self.h**2  # type: ignore[operator]  # outside

    def test_edge_reads_ambient_field(self) -> None:
        """At a box edge the subdomain stencil reads the ambient u just outside."""
        u, i = self.u, self.i
        stencil = (u[i - 1] + u[i + 1]) / 2  # type: ignore[operator]
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[DerivativeCondition(Box({i: (3, 5)}), stencil)],
        )
        # i=3 is the low edge; u[i-1] = u[2] is ambient (outside the box)
        assert ode.equation.expr.subs(i, 3) == u[2] / 2 + u[4] / 2  # type: ignore[operator]

    def test_boundary_condition_wins_on_shared_edge(self) -> None:
        """A Dirichlet face on a subdomain edge beats the subdomain dynamics there."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), 0),  # pins i=0
                DerivativeCondition(Box({i: (0, 2)}), -u[i]),  # type: ignore[operator]  # box touches i=0
            ],
        )
        expr = ode.equation.expr
        assert expr.subs(i, 0) == 0  # Dirichlet wins
        assert expr.subs(i, 1) == -u[1]  # type: ignore[operator]  # subdomain interior

    def test_state_beats_derivative_regardless_of_list_order(self) -> None:
        """Layering is by family, not list position: a state condition wins over a
        derivative one on a shared cell even when the derivative is listed first."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DerivativeCondition(Box({i: (0, 2)}), -u[i]),  # type: ignore[operator]  # listed FIRST
                DirichletCondition(Facet.low(i), 0),  # pins i=0
            ],
        )
        # the state condition still wins at i=0 (derivative builds the field, state
        # substitutes into it -- order between families is irrelevant)
        assert ode.equation.expr.subs(i, 0) == 0
        assert ode.equation.expr.subs(i, 1) == -u[1]  # type: ignore[operator]

    def test_slab_leaves_unrestricted_axis_full(self) -> None:
        """An axis omitted from the box spans its full range."""
        u = self.u
        i, j = _idx("i", (0, 5)), _idx("j", (0, 4))
        box = Box({i: (2, 3)})
        sub = DerivativeCondition(box, -u[i, j])  # type: ignore[operator]
        assert box.ranges == {i: (2, 3)}
        assert j not in box.ranges
        assert sub.region is box

    def test_negative_positions_resolved(self) -> None:
        """Box positions count from the end, Python-style."""
        u, i = self.u, self.i
        box = Box({i: (-3, -1)})
        sub = DerivativeCondition(box, -u[i])  # type: ignore[operator]
        assert box.ranges == {i: (5, 7)}
        assert sub.region is box

    def test_eliminate_passes_subdomains_through(self) -> None:
        """Eliminate substitutes value-constraints but leaves subdomain dynamics."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), 5),
                DerivativeCondition(Box({i: (4, 6)}), -u[i]),  # type: ignore[operator]
            ],
            eliminate=True,
        )
        assert ode.equation.expr.subs(i, 5) == -u[5]  # type: ignore[operator]

    def test_eliminate_row_inside_subdomain_uses_subdomain_stencil(self) -> None:
        """An elimination row inside a subdomain uses the subdomain's stencil (with the
        coefficient D), not the ambient one.

        Box [0, 3] with a scaled-diffusion stencil overlaps a Dirichlet at i=0 (u[0]=0).
        The i=1 row (in the box, adjacent to the pinned point) is an elimination row: it
        must be D*(u[0] - 2u[1] + u[2])/h**2 with u[0]=0, i.e. keep D.
        """
        u, h = self.u, self.h
        i = _idx("i", (0, 7))
        D = Symbol("D")
        ambient = (u[i - 1] - 2 * u[i] + u[i + 1]) / h**2  # type: ignore[operator]
        scaled = D * (u[i - 1] - 2 * u[i] + u[i + 1]) / h**2  # type: ignore[operator]
        ode = OrdinaryDifferentialSystem.from_sympy(
            ambient,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), 0),
                DerivativeCondition(Box({i: (0, 3)}), scaled),
            ],
            eliminate=True,
        )
        row = ode.equation.expr.subs(i, 1)
        assert row.has(D)  # the subdomain coefficient survives elimination
        assert row == D * (-2 * u[1] + u[2]) / h**2  # type: ignore[operator]

    def test_constraint_referencing_into_subdomain_chain_rules_through_it(self) -> None:
        """A state condition whose referenced neighbour lies in a subdomain chain-rules
        through the subdomain's stencil.

        The mirror u[0] = -u[1] on the low face; the box [0, 3] runs scaled diffusion, so
        u[1] (referenced by the value) evolves with D. Hence du[0]/dt = -du[1]/dt must
        carry D too -- the i=0 branch is not the ambient stencil.
        """
        u, h = self.u, self.h
        i = _idx("i", (0, 7))
        D = Symbol("D")
        ambient = (u[i - 1] - 2 * u[i] + u[i + 1]) / h**2  # type: ignore[operator]
        scaled = D * (u[i - 1] - 2 * u[i] + u[i + 1]) / h**2  # type: ignore[operator]
        ode = OrdinaryDifferentialSystem.from_sympy(
            ambient,
            u,
            (i,),
            conditions=[
                DirichletCondition(Facet.low(i), -u[1]),  # type: ignore[operator]
                DerivativeCondition(Box({i: (0, 3)}), scaled),
            ],
            eliminate=True,
        )
        at0 = ode.equation.expr.subs(i, 0)
        at1 = ode.equation.expr.subs(i, 1)
        assert at0.has(
            D
        )  # chain-rule went through the subdomain, not the ambient stencil
        # u[0] = -u[1] means du[0]/dt = -du[1]/dt
        assert (at0 + at1).simplify() == 0  # type: ignore[operator]

    def test_stored_on_the_ode(self) -> None:
        u, i = self.u, self.i
        subs = [DerivativeCondition(Box({i: (3, 5)}), -u[i])]  # type: ignore[operator]
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion, u, (i,), conditions=subs
        )
        assert ode.derivative_conditions == tuple(subs)

    def test_condition_is_the_box_predicate(self) -> None:
        u, i = self.u, self.i
        sub = DerivativeCondition(Box({i: (3, 5)}), -u[i])  # type: ignore[operator]
        assert sub.region.condition == And(3 <= i, i <= 5)  # type: ignore[operator]

    def test_foreign_index_rejected(self) -> None:
        u, i = self.u, self.i
        k = _idx("k", (0, 3))
        with raises(ValueError, match="not one of the system's indices"):
            OrdinaryDifferentialSystem.from_sympy(
                self.diffusion,
                u,
                (i,),
                conditions=[DerivativeCondition(Box({k: (0, 1)}), -u[i])],  # type: ignore[operator]
            )

    def test_overlapping_subdomains_first_listed_wins(self) -> None:
        """Overlapping boxes are allowed; the first-listed derivative condition wins on
        a shared cell (the Piecewise is first-match)."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DerivativeCondition(Box({i: (0, 3)}), -u[i]),  # type: ignore[operator]
                DerivativeCondition(Box({i: (2, 5)}), 2 * u[i]),  # overlaps 2..3
            ],
        )
        expr = ode.equation.expr
        assert expr.subs(i, 1) == -u[1]  # type: ignore[operator]  # only in the first box
        assert expr.subs(i, 2) == -u[2]  # type: ignore[operator]  # in both: first wins
        assert expr.subs(i, 3) == -u[3]  # type: ignore[operator]  # in both: first wins
        assert expr.subs(i, 4) == 2 * u[4]  # only in the second box

    def test_adjacent_subdomains_do_not_overlap(self) -> None:
        """Touching-but-disjoint boxes (0..2 and 3..5) are allowed."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[
                DerivativeCondition(Box({i: (0, 2)}), -u[i]),  # type: ignore[operator]
                DerivativeCondition(Box({i: (3, 5)}), 2 * u[i]),
            ],  # type: ignore[operator]
        )
        assert ode.equation.expr.subs(i, 1) == -u[1]  # type: ignore[operator]
        assert ode.equation.expr.subs(i, 4) == 2 * u[4]

    def test_empty_range_rejected(self) -> None:
        u, i = self.u, self.i
        with raises(ValueError, match="empty"):
            DerivativeCondition(Box({i: (5, 2)}), -u[i])  # type: ignore[operator]

    def test_out_of_grid_range_rejected(self) -> None:
        u, i = self.u, self.i
        with raises(ValueError, match="out of bounds"):
            DerivativeCondition(Box({i: (0, 20)}), -u[i])  # type: ignore[operator]

    def test_empty_box_rejected(self) -> None:
        u, i = self.u, self.i
        with raises(ValueError, match="must not be empty"):
            DerivativeCondition(Box({}), -u[i])  # type: ignore[operator]


class BoxBoundaryConditions(TestCase):
    """A state condition may act on the whole shell of a box via ``box.boundary``, which
    is a :class:`Region` whose condition is the union of the box's walls."""

    def setUp(self) -> None:
        clear_cache()
        self.N = 8
        self.u = _indexed_base("u")
        self.h = Symbol("h")
        self.i = _idx("i", (0, self.N - 1))
        self.diffusion = (
            self.u[self.i - 1] - 2 * self.u[self.i] + self.u[self.i + 1]  # type: ignore[operator]
        ) / self.h**2

    def test_dirichlet_pins_every_wall(self) -> None:
        """Dirichlet on a box boundary holds both walls; the box interior and the
        outside run the ambient stencil."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[DirichletCondition(Box({i: (2, 5)}).boundary, 0)],
        )
        assert ode.equation.expr.subs(i, 2) == 0  # low wall held
        assert ode.equation.expr.subs(i, 5) == 0  # high wall held
        assert ode.equation.expr.subs(i, 3) == (u[2] - 2 * u[3] + u[4]) / self.h**2  # type: ignore[operator]

    def test_dirichlet_eliminates_wall_references(self) -> None:
        """With eliminate=True the shell's pinned walls are substituted out of the rows
        that reference them -- the same box.boundary condition works as it does without
        elimination (no raise)."""
        u, i = self.u, self.i
        ode = OrdinaryDifferentialSystem.from_sympy(
            self.diffusion,
            u,
            (i,),
            conditions=[DirichletCondition(Box({i: (2, 5)}).boundary, 0)],
            eliminate=True,
        )
        # walls still held
        assert ode.equation.expr.subs(i, 2) == 0
        assert ode.equation.expr.subs(i, 5) == 0
        # a neighbour of the low wall (i=3 references u[2]) has that reference dropped
        assert not ode.equation.expr.subs(i, 3).has(u[2])

    def test_2d_dirichlet_holds_the_whole_shell(self) -> None:
        u = self.u
        i, j = _idx("i", (0, 5)), _idx("j", (0, 4))
        lap = (
            u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
        ) / self.h**2  # type: ignore[operator]
        ode = OrdinaryDifferentialSystem.from_sympy(
            lap,
            u,
            (i, j),
            conditions=[DirichletCondition(Box({i: (1, 3), j: (1, 3)}).boundary, 0)],
        )
        # every wall / corner of the [1,3]x[1,3] shell is held
        for p, q in [(1, 2), (3, 2), (2, 1), (2, 3), (1, 1), (3, 3)]:
            assert ode.equation.expr.subs({i: p, j: q}) == 0
        # the interior of the box runs the ambient stencil
        assert ode.equation.expr.subs({i: 2, j: 2}) != 0
