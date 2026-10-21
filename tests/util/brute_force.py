"""The same matrix entry, computed the obvious slow way, importing nothing from lin_ode.

Write each ``f_i`` as an explicit polynomial, apply the generator with ``sympy.diff``,
expand, then read the coefficient off the result.

site symbols      one `y_<site>` per grid site, the same keys `f` uses the grid
sites and their in-grid neighbours monomial bases    an entry is a coefficient of an
expanded polynomial Gauss-Hermite     an entry is one integral against the Hermite bra
state
"""

from collections.abc import Callable
from functools import lru_cache

import sympy as sp

from tests.testing import helper

Site = int | tuple[int, ...]
"""A grid site: an index in 1-D, a coordinate tuple in 2-D."""

State = tuple[tuple[Site, int], ...]
"""A basis monomial as `(site, power)` pairs, the same labels the matrix is indexed
by."""

Coefficients = Callable[[Site], tuple[sp.Expr, sp.Expr, sp.Expr]]
"""`site -> (D, Q, R)`, so a region is an arm rather than a `Piecewise`."""


def make_site_symbols(grid: list[Site]) -> dict[Site, sp.Symbol]:
    """One plain sympy symbol per site: the state `u` written out variable by
    variable."""
    return {site: sp.Symbol(f"y_{site}") for site in grid}


def list_grid_sites(shape: tuple[int, ...]) -> list[Site]:
    """Every site of the grid, as the labels `sym` and `f` are keyed by."""
    if len(shape) == 1:
        return list(range(shape[0]))
    return [(i, j) for i in range(shape[0]) for j in range(shape[1])]


def find_neighbours(site: Site, shape: tuple[int, ...], periodic: bool) -> list[Site]:
    """In-grid +-1 neighbours; periodic wraps, open omits the out-of-grid ones."""
    coords = (site,) if isinstance(site, int) else site
    out: list[Site] = []
    for axis, size in enumerate(shape):
        for step in (-1, 1):
            moved = list(coords)
            moved[axis] = (
                (coords[axis] + step) % size if periodic else coords[axis] + step
            )
            if 0 <= moved[axis] < size:
                out.append(moved[0] if len(shape) == 1 else tuple(moved))
    return out


def write_rhs_per_site(
    grid: list[Site],
    shape: tuple[int, ...],
    periodic: bool,
    sym: dict[Site, sp.Symbol],
    coeff: Coefficients,
    h: sp.Symbol,
) -> dict[Site, sp.Expr]:
    """`{site: f_s}`, the explicit KPP-Fisher right-hand side at every site."""
    f = {}
    for s in grid:
        d_s, q_s, r_s = coeff(s)
        lap = -2 * len(shape) * sym[s]
        for nb in find_neighbours(s, shape, periodic):
            lap += sym[nb]
        f[s] = d_s * lap / h**2 + q_s * sym[s] + r_s * sym[s] ** 2
    return f


def build_monomial(label: State, sym: dict[Site, sp.Symbol]) -> sp.Expr:
    """The basis function `prod u_site**power`, in the site symbols."""
    return sp.Mul(*[sym[site] ** power for site, power in label])


def combine(method: str, koopman: sp.Expr, divergence: sp.Expr) -> sp.Expr:
    """The three methods from the two pieces every one of them is built from.

    Koopman is `f.grad`, Liouville is `-div f - Koopman`, KvN is their half-difference.
    """
    if method == "Koopman":
        return sp.expand(koopman)
    liouville = sp.expand(-divergence - koopman)
    if method == "Liouville":
        return liouville
    return sp.expand((liouville - koopman) / 2)


def apply_generator(
    method: str,
    label: State,
    f: dict[Site, sp.Expr],
    sym: dict[Site, sp.Symbol],
) -> sp.Expr:
    """`L` applied to one basis monomial, as an explicit expanded polynomial."""
    phi = build_monomial(label, sym)
    koopman = sum((f[s] * sp.diff(phi, sym[s]) for s in f), sp.S.Zero)
    divergence = sum((sp.diff(f[s], sym[s]) for s in f), sp.S.Zero)
    return combine(method, koopman, divergence * phi)


def _about_pivots(
    term: sp.Expr, variables: list, powers: tuple[int, ...], pivot: Callable
) -> sp.Expr:
    """The same term with every occupied variable measured from its own pivot."""
    for var, power in zip(variables, powers, strict=True):
        if power:
            term = term.subs(var, var - pivot(var), simultaneous=True)
    return term


def fold_to_degree(
    poly: sp.Expr,
    degree: int,
    sym: dict[Site, sp.Symbol],
    pivot: Callable[[sp.Symbol], sp.Expr] | None,
) -> sp.Expr:
    """Dropped (pivot None) or Taylor-folded.

    Subtracting ``prod (u_t - s_t)**p_t`` removes exactly the one top-degree term. That is
    what makes PSC a fold rather than a second truncation rule.
    """
    variables = list(sym.values())
    kept = sp.S.Zero
    for term in sp.Add.make_args(sp.expand(poly)):
        powers = sp.Poly(term, *variables).monoms()[0]
        if sum(powers) <= degree:
            kept += term
        elif pivot is not None:
            kept += sp.expand(term - _about_pivots(term, variables, powers, pivot))
    return sp.expand(kept)


def build_column(
    method: str,
    ket: tuple,
    f: dict,
    sym: dict,
    *,
    degree: int,
    pivot: Callable[[sp.Symbol], sp.Expr] | None,
) -> sp.Poly | None:
    """``L|ket>``, truncated, as a polynomial in the site symbols.

    ``None`` where the whole column vanished.
    """
    truncated = fold_to_degree(apply_generator(method, ket, f, sym), degree, sym, pivot)
    return sp.Poly(truncated, *sym.values()) if truncated != 0 else None


def read_coefficient(poly: sp.Poly | None, bra: tuple, sym: dict) -> sp.Expr:
    """The bra's coefficient in that column, or zero where the column vanished."""
    if poly is None:
        return sp.S.Zero
    return poly.coeff_monomial(build_monomial(bra, sym))


@helper
def compute_entry(
    bra: State,
    ket: State,
    *,
    method: str,
    shape: tuple[int, ...],
    periodic: bool,
    coeff: Coefficients,
    h: sp.Symbol,
    degree: int = 2,
    pivot: Callable[[sp.Symbol], sp.Expr] | None = None,
) -> sp.Expr:
    """`<bra|L|ket>`: apply the generator to the ket, truncate, read off the bra's
    coefficient."""
    grid = list_grid_sites(shape)
    sym = make_site_symbols(grid)
    f = write_rhs_per_site(grid, shape, periodic, sym, coeff, h)
    poly = build_column(method, ket, f, sym, degree=degree, pivot=pivot)
    return read_coefficient(poly, bra, sym)


_Z = sp.Symbol("z", real=True)


def _hermite_state(mode: int, y: sp.Symbol) -> sp.Expr:
    """Polynomial part of the normalised Hermite function at `mode`."""
    return sp.hermite(mode, y) / sp.sqrt(2**mode * sp.factorial(mode) * sp.sqrt(sp.pi))


def _diff_state(poly: sp.Expr, y: sp.Symbol) -> sp.Expr:
    """`d/dy` of `poly * exp(-y**2/2)`, with that Gaussian left off both sides."""
    return sp.diff(poly, y) - y * poly


@lru_cache(maxsize=None)
def _overlap(bra_mode: int, power: int) -> sp.Expr:
    """`<bra_mode| y**power |Gaussian>`, the one integral this file does."""
    bra = _hermite_state(bra_mode, _Z) * sp.exp(-(_Z**2) / 2)
    return sp.integrate(bra * _Z**power * sp.exp(-(_Z**2) / 2), (_Z, -sp.oo, sp.oo))


def read_hermite_coefficient(
    method: str,
    bra: State,
    ket: State,
    f: dict[Site, sp.Expr],
    sym: dict[Site, sp.Symbol],
) -> sp.Expr:
    """`<bra|L|ket>` on Gauss-Hermite: apply the generator, integrate against the bra.

    A label here is a mode number, not a power: sites absent from `bra`/`ket` sit at
    mode 0. The Gaussian is carried by `_diff_state` and absorbed by `_overlap`.
    """
    grid = list(sym)
    bra_mode = dict.fromkeys(grid, 0) | dict(bra)
    ket_mode = dict.fromkeys(grid, 0) | dict(ket)
    phi = sp.Mul(*[_hermite_state(ket_mode[s], sym[s]) for s in grid])
    koopman = sum((f[s] * _diff_state(phi, sym[s]) for s in f), sp.S.Zero)
    divergence = sum((sp.diff(f[s], sym[s]) for s in f), sp.S.Zero) * phi
    acted = sp.Poly(combine(method, koopman, divergence), *[sym[s] for s in grid])
    total = sp.S.Zero
    for powers, factor in zip(acted.monoms(), acted.coeffs(), strict=True):
        for site, power in zip(grid, powers, strict=True):
            factor *= _overlap(bra_mode[site], power)
        total += factor
    return sp.expand(total)


@helper
def compute_hermite_entry(
    bra: State,
    ket: State,
    *,
    method: str,
    shape: tuple[int, ...],
    periodic: bool,
    coeff: Coefficients,
    h: sp.Symbol,
    grid: list[Site],
) -> sp.Expr:
    """`<bra|L|ket>` on Gauss-Hermite, for KPP-Fisher on the given grid."""
    sym = make_site_symbols(grid)
    f = write_rhs_per_site(grid, shape, periodic, sym, coeff, h)
    return read_hermite_coefficient(method, bra, ket, f, sym)
