# pyright: reportArgumentType=false, reportIndexIssue=false, reportOperatorIssue=false
# sympy stubs mistype Idx arithmetic and IndexedBase subscripting.
"""The grid of cases: a region, a right-hand side, a pivot, and the entries probed.

Each region and right-hand side is stated twice, once over ``u[i]`` and once over the
brute force's per-site variables. The duplication is the point, since agreement between
the two is what the test checks.
"""

from collections.abc import Callable
from typing import NamedTuple

import sympy as sp

from lin_ode.basis import CarlemanBasis, PSCarlemanBasis, TruncatedBasisSet
from lin_ode.ordinary_differential_system import OrdinaryDifferentialSystem

from tests.util.comparison import s_pivot, s_site

from .grid import DEGREE, GRID, D, IDX_4, N_SITE, Q_0, Q_1, R_0, R_1, SYM, h, u

PROBED = [
    # (bra, ket): applying L to |u_0**2> leaves a degree-2 basis and a sum over every
    # site, so that column plus the degree-1 one is where an omitted site is visible
    ((), ((0, DEGREE),)),
    (((0, 1),), ((0, DEGREE),)),
    (((0, DEGREE),), ((0, DEGREE),)),
    (((0, 1),), ((0, 1),)),
    (((1, 1),), ((0, 1),)),
    (((0, 1), (1, 1)), ((0, 1),)),
    ((), ((0, 1),)),
]

PIVOTS = ("carleman", "shared pivot", "per-site pivot")

REGION_SHAPES = {
    # name: (condition on the index, the same region as a predicate on a site)
    "interior rectangle": (sp.And(IDX_4 >= 1, IDX_4 <= 2), lambda site: 1 <= site <= 2),
    "single site": (sp.Eq(IDX_4, 0), lambda site: site == 0),
    "at the edge": (sp.And(IDX_4 >= 0, IDX_4 <= 0), lambda site: site == 0),
    "the whole grid": (sp.And(IDX_4 >= 0, IDX_4 <= N_SITE - 1), lambda _: True),
    "empty": (sp.And(IDX_4 >= 2, IDX_4 <= 1), lambda _: False),
    "every other site": (sp.Eq(sp.Mod(IDX_4, 2), 0), lambda site: site % 2 == 0),
}
"""The last two are the awkward ones: an empty region has nothing to count, and a
periodic one is not a rectangle at all."""


def _laplacian(site: int) -> sp.Expr:
    """Nearest-neighbour stencil on the ring, in the brute force's own variables."""
    return SYM[(site - 1) % N_SITE] - 2 * SYM[site] + SYM[(site + 1) % N_SITE]


RIGHT_HAND_SIDES = {
    # name: (one arm over u[i], the same arm at one site over the brute force's
    # variables); `in_region` picks the subscript-1 coefficients over the -0 ones
    "R u**2": (
        lambda in_region: (R_1 if in_region else R_0) * u[IDX_4] ** 2,
        lambda in_region, site: (R_1 if in_region else R_0) * SYM[site] ** 2,
    ),
    "Q u": (
        lambda in_region: (Q_1 if in_region else Q_0) * u[IDX_4],
        lambda in_region, site: (Q_1 if in_region else Q_0) * SYM[site],
    ),
    "Q u + R u**2": (
        lambda in_region: (
            (Q_1 if in_region else Q_0) * u[IDX_4]
            + (R_1 if in_region else R_0) * u[IDX_4] ** 2
        ),
        lambda in_region, site: (
            (Q_1 if in_region else Q_0) * SYM[site]
            + (R_1 if in_region else R_0) * SYM[site] ** 2
        ),
    ),
    "laplacian + R u**2": (
        lambda in_region: (
            D
            * (
                u[sp.Mod(IDX_4 - 1, N_SITE)]
                - 2 * u[IDX_4]
                + u[sp.Mod(IDX_4 + 1, N_SITE)]
            )
            / h**2
            + (R_1 if in_region else R_0) * u[IDX_4] ** 2
        ),
        lambda in_region, site: (
            D * _laplacian(site) / h**2 + (R_1 if in_region else R_0) * SYM[site] ** 2
        ),
    ),
}
"""A degree-1 term is what puts a *constant* into ``div f``.

A constant carries no index, so it leaves a free axis, and a free axis is the thing that
has to be counted, which is where an uncountable region declines.
"""


class Case(NamedTuple):
    """One (region, right-hand side, pivot, method), resolved into both forms.

    Everything below this point takes a ``Case`` and reads nothing else, so the tables
    above are looked up exactly once, here.
    """

    name: str
    method: str
    ode: OrdinaryDifferentialSystem
    """The two-armed ``Piecewise`` right-hand side, as the library takes it."""

    per_site: tuple
    """The region resolved: one ``(site, polynomial)`` pair each, so a ``Case``
    hashes."""

    basis: TruncatedBasisSet
    fold: Callable[[sp.Symbol], sp.Expr] | None
    """What a folded variable is measured from, or ``None`` where they are dropped."""


def resolve_case(region: str, rhs: str, pivot: str, method: str) -> Case:
    """Look the four names up once, and hand back everything the checks need."""
    condition, inside = REGION_SHAPES[region]
    arm, arm_at = RIGHT_HAND_SIDES[rhs]
    folded = sp.piecewise_fold(
        sp.Piecewise((arm(in_region=True), condition), (arm(in_region=False), sp.true))
    )
    return Case(
        name=f"{region}/{rhs}/{pivot}/{method}",
        method=method,
        ode=OrdinaryDifferentialSystem.from_sympy(folded, u, (IDX_4,)),
        per_site=tuple((site, arm_at(inside(site), site)) for site in GRID),
        basis=_build_basis(pivot),
        fold=_pick_fold(pivot),
    )


def _build_basis(pivot: str) -> TruncatedBasisSet:
    """The lin_ode basis that pivot names."""
    if pivot == "carleman":
        return CarlemanBasis(N_SITE, DEGREE)
    return PSCarlemanBasis(
        N_SITE, DEGREE, s_pivot if pivot == "shared pivot" else s_site
    )


def _pick_fold(pivot: str) -> Callable | None:
    """The brute force's pivot callable, matching that basis."""
    if pivot == "carleman":
        return None  # beyond-degree monomials are dropped, not folded
    if pivot == "shared pivot":
        return lambda _: s_pivot
    return {SYM[site]: s_site[site] for site in GRID}.get
