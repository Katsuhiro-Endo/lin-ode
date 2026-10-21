# pyright: reportArgumentType=false, reportCallIssue=false
# sympy stubs mistype .subs and .replace as taking Basic only.
"""Any ODE's right-hand side, written out one grid site at a time.

Substituting a concrete site leaves an explicit polynomial in the site symbols, which is
what ``brute_force`` takes. Arms are selected here, so agreement checks the builder's
own arm selection.
"""

import sympy as sp

from lin_ode.ordinary_differential_system import OrdinaryDifferentialSystem
from tests.testing import helper
from tests.util.brute_force import Site, list_grid_sites, make_site_symbols

Symbols = dict[Site, sp.Symbol]


def _site_of(atom: sp.Indexed) -> Site:
    """The grid site a `u[...]` atom points at, keyed as `grid_sites` keys them."""
    indices = [int(index) for index in atom.indices]
    return indices[0] if len(indices) == 1 else tuple(indices)


def _arm_at(ode: OrdinaryDifferentialSystem, site: Site, sym: Symbols) -> sp.Expr:
    """The active arm at `site`, with every `u[...]` replaced by its site symbol."""
    coords = (site,) if isinstance(site, int) else site
    at_site = dict(zip(ode.index_symbols, coords, strict=True))
    state = ode.state_variable
    for expr, condition in ode.equation.cases:
        if condition.subs(at_site) == True:  # noqa: E712 - sympy truthiness
            return sp.expand(
                expr.subs(at_site).replace(
                    lambda e: isinstance(e, sp.Indexed) and e.base == state,
                    lambda e: sym[_site_of(e)],
                )
            )
    raise AssertionError(f"no arm of the equation is active at site {site}")


@helper
def write_out_per_site(
    ode: OrdinaryDifferentialSystem,
) -> tuple[dict[Site, sp.Expr], Symbols]:
    """`(f, sym)` for the whole grid: `f[site]` is `du[site]/dt` as a polynomial."""
    grid = list_grid_sites(ode.shape)
    sym = make_site_symbols(grid)
    return {site: _arm_at(ode, site, sym) for site in grid}, sym
