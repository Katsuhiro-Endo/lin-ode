# pyright: reportGeneralTypeIssues=false, reportIncompatibleMethodOverride=false, reportArgumentType=false, reportCallIssue=false
# sympy stubs: .args is Basic (not iterable), and overriding Function.eval/doit trips
# override and call checks; all noise for these Function subclasses.
"""Multi-index Kronecker delta functions for D-dimensional site labels.

MultiKroneckerDelta((i0, i1, ...), (j0, j1, ...))
    = KroneckerDelta(i0, j0) * KroneckerDelta(i1, j1) * ...

MultiAntiKroneckerDelta((i0, i1, ...), (j0, j1, ...))
    = 1 - MultiKroneckerDelta((i0, i1, ...), (j0, j1, ...))

Both accept sp.Tuple args.  For D=1, MultiKroneckerDelta((i,), (j,)) reduces
to KroneckerDelta(i, j) and MultiAntiKroneckerDelta to AntiKroneckerDelta.
"""

import sympy as sp


def _check_same_length(i_tup, j_tup) -> None:
    """Both index tuples must label the same number of dimensions."""
    if len(i_tup) != len(j_tup):
        raise ValueError(
            f"index tuples must have equal length, got {len(i_tup)} and {len(j_tup)}"
        )


class MultiKroneckerDelta(sp.Function):
    """Ordered product-form Kronecker delta for D-dimensional site labels.

    eval() fires for concrete equal/unequal tuples. doit() expands to a product of
    scalar KroneckerDelta factors.
    """

    nargs = 2

    @classmethod
    def eval(cls, i_tup, j_tup):
        _check_same_length(i_tup, j_tup)
        if i_tup == j_tup:
            return sp.S.One
        for i, j in zip(i_tup, j_tup):
            if i.is_number and j.is_number and i != j:
                return sp.S.Zero

    def doit(self, **hints):
        i_tup, j_tup = self.args
        return sp.Mul(*[sp.KroneckerDelta(i, j) for i, j in zip(i_tup, j_tup)])

    def _latex(self, printer):
        i_tup, j_tup = self.args
        i_str = ",".join(printer._print(x) for x in i_tup)
        j_str = ",".join(printer._print(x) for x in j_tup)
        return rf"\delta_{{({i_str}),({j_str})}}"


class MultiAntiKroneckerDelta(sp.Function):
    """1 - MultiKroneckerDelta(i_tup, j_tup)."""

    nargs = 2

    @classmethod
    def eval(cls, i_tup, j_tup):
        _check_same_length(i_tup, j_tup)
        if i_tup == j_tup:
            return sp.S.Zero
        for i, j in zip(i_tup, j_tup):
            if i.is_number and j.is_number and i != j:
                return sp.S.One

    def doit(self, **hints):
        return sp.S.One - MultiKroneckerDelta(*self.args).doit()

    def _latex(self, printer):
        i_tup, j_tup = self.args
        i_str = ",".join(printer._print(x) for x in i_tup)
        j_str = ",".join(printer._print(x) for x in j_tup)
        return rf"\bar{{\delta}}_{{({i_str}),({j_str})}}"
