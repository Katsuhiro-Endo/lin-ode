"""The same entry as the brute force gets it, by enumerating the sites."""

import functools

import sympy as sp

from tests.util import brute_force
from tests.testing import helper

from .cases import Case
from .grid import DEGREE, SYM


@functools.lru_cache(maxsize=None)
def _build_column(case: Case, ket: tuple) -> sp.Poly | None:
    """One column, cached: every probed bra reads its coefficient off the same one."""
    return brute_force.build_column(
        case.method, ket, dict(case.per_site), SYM, degree=DEGREE, pivot=case.fold
    )


@helper
def read_brute_force_entry(case: Case, bra: tuple, ket: tuple) -> sp.Expr:
    """The entry the brute force gives, from the per-site polynomials of this case."""
    return brute_force.read_coefficient(_build_column(case, ket), bra, SYM)
