# pyright: reportIndexIssue=false
# sympy stubs read the block extraction as having no __getitem__.
"""What the library answers for one case, or its refusal to answer at all."""

import sympy as sp

from lin_ode.matrix import OperatorMatrix

from tests.util import METHODS
from tests.testing import helper

from .cases import Case

DECLINED = object()
"""`lin_ode`'s other supported answer: this region cannot be counted."""

@helper
def ask_lin_ode(case: Case, probed: list) -> dict:
    """Every probed entry as the library computes it, or `DECLINED` if it will not."""
    bras = sorted({bra for bra, _ in probed}, key=str)
    kets = sorted({ket for _, ket in probed}, key=str)
    try:
        operator = METHODS[case.method].create_operator(case.ode)
        # one block over the distinct bras and kets, rather than an entry at a time
        block = OperatorMatrix(case.basis, operator)[bras, kets]
    except NotImplementedError:
        return DECLINED  # type: ignore[return-value]  # the sentinel, checked by identity
    return {(bra, ket): block[bras.index(bra), kets.index(ket)] for bra, ket in probed}


@helper
def expand_over_sites(entry: sp.Expr) -> sp.Expr:
    """A `Sum` over sites carries the region's guard; expand it to compare values."""
    return sp.expand(entry.doit() if hasattr(entry, "doit") else entry)
