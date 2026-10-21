# pyright: reportCallIssue=false
# sympy stubs type Function subclasses as AppliedUndef, so instantiating them here reads
# as a call error; noise.
"""Multi-axis Kronecker deltas.

A concrete site pair evaluates on construction. A symbolic one waits for ``doit``, then
lowers to one scalar delta per undecided axis.
"""

import sympy as sp
from pytest import mark

from lin_ode.basis.multi_delta import MultiAntiKroneckerDelta, MultiKroneckerDelta
from lin_ode.basis.state import AntiKroneckerDelta

i, j, k, m = sp.symbols("i j k m")


@mark.parametrize(
    "delta, left, right, expected",
    [
        (MultiKroneckerDelta, sp.Tuple(0, 1), sp.Tuple(0, 1), 1),
        (MultiKroneckerDelta, sp.Tuple(0, 2), sp.Tuple(0, 1), 0),
        (MultiAntiKroneckerDelta, sp.Tuple(0, 0), sp.Tuple(0, 0), 0),
        (MultiAntiKroneckerDelta, sp.Tuple(0, 0), sp.Tuple(1, 0), 1),
        (AntiKroneckerDelta, sp.Integer(0), sp.Integer(0), 0),
        (AntiKroneckerDelta, sp.Integer(0), sp.Integer(1), 1),
    ],
    ids=[
        "multi-kd-equal",
        "multi-kd-unequal",
        "multi-anti-equal",
        "multi-anti-unequal",
        "scalar-anti-equal",
        "scalar-anti-unequal",
    ],
)
def test_fully_concrete_index_pair_evaluates_on_construction(
    delta: type[sp.Function], left: sp.Basic, right: sp.Basic, expected: int
) -> None:
    """Equal indices make the Kronecker delta 1 and its anti-delta complement 0; a
    mismatch on any single axis flips both."""
    assert delta(left, right) == expected


@mark.parametrize(
    "left, right, expected",
    [
        (sp.Tuple(0, i), sp.Tuple(0, j), sp.KroneckerDelta(i, j)),
        (
            sp.Tuple(i, j),
            sp.Tuple(k, m),
            sp.KroneckerDelta(i, k) * sp.KroneckerDelta(j, m),
        ),
    ],
    ids=["one-axis-already-agrees", "both-axes-symbolic"],
)
def test_doit_lowers_to_one_scalar_delta_per_undecided_axis(
    left: sp.Basic, right: sp.Basic, expected: sp.Expr
) -> None:
    """Axis 0 of the first case already agrees concretely.

    It therefore contributes the factor 1 and drops out of the product.
    """
    assert MultiKroneckerDelta(left, right).doit() == expected
