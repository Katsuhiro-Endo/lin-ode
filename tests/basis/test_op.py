# pyright: reportIndexIssue=false, reportArgumentType=false
# sympy stubs: U[sym] indexing and passing sympy exprs where ints/Idx are expected
# read as errors; all noise in these symbolic-algebra tests.
"""Supported cases for the operator algebra (``lin_ode.basis.op``), as a table.

Each row is a ``Case(op, state, expected)``: an operator, an input state, and the
expected output written out in full. Rows cover concrete, symbolic, partial and
multi-dimensional indices.

In the ids, ``d`` is KroneckerDelta and ``d_`` its complement. Carleman rows compare
with ``==``; Gauss-Hermite rows use ``_agree``, which allows a different sqrt form.
"""

from typing import NamedTuple

import sympy as sp
from pytest import mark, raises

from lin_ode.basis import AntiKroneckerDelta as AKD
from lin_ode.basis import (
    CarlemanBasis,
    CarlemanState,
    GaussianHermiteBasis,
    GaussianHermiteState,
    OpRuleRepo,
)
from lin_ode.basis.multi_delta import MultiAntiKroneckerDelta as MAKD
from lin_ode.basis.multi_delta import MultiKroneckerDelta as MKD
from lin_ode.basis.op import K, U, apply_op, evaluate


REG = CarlemanBasis(3, 2).registry
REG_GH = GaussianHermiteBasis(3, 2).registry
Site = sp.Tuple
KD = sp.KroneckerDelta
i = sp.Idx("i")
i0, i1, i2 = sp.Idx("i0"), sp.Idx("i1"), sp.Idx("i2")
j, j0, j1, n = sp.symbols("j j0 j1 n")


class Case(NamedTuple):
    """One golden row: op |state> should evaluate to expected."""

    op: sp.Expr
    state: sp.Expr
    expected: sp.Expr
    id: str


def _agree(a: sp.Expr, b: sp.Expr) -> bool:
    """Equal as basis-function expressions (robust to KD / sqrt normal forms)."""
    return sp.simplify(sp.expand(a - b)) == 0


CARLEMAN = [
    Case(
        op=U[0],
        state=CarlemanState((0, 1)),
        expected=CarlemanState((0, 2)),
        id="U_0|u0>=|u0^2>",
    ),
    Case(
        op=K[0],
        state=CarlemanState((0, 2)),
        expected=2 * CarlemanState((0, 1)),
        id="K_0|u0^2>=2|u0>",
    ),
    Case(
        op=K[0] * U[0],
        state=CarlemanState((0, 1)),
        expected=2 * CarlemanState((0, 1)),
        id="K_0 U_0|u0>=2|u0>",  # composed
    ),
    Case(
        op=U[0] * K[1],
        state=CarlemanState((0, 1), (1, 1)),
        expected=CarlemanState((0, 2)),
        id="U_0 K_1|u0 u1>=|u0^2>",  # product across sites
    ),
    Case(
        op=K[0],
        state=CarlemanState(),
        expected=sp.S.Zero,
        id="K_0|vac>=0",
    ),
    Case(
        op=U[1],
        state=CarlemanState((0, 1)),
        expected=CarlemanState((0, 1), (1, 1)),
        id="U_1|u0>=|u0 u1>",
    ),
    Case(
        op=U[0] ** 2,
        state=CarlemanState((0, 1)),
        expected=CarlemanState((0, 3)),
        id="U_0^2|u0>=|u0^3>",
    ),
    Case(
        op=(U[0] + K[0]) ** 2,
        state=CarlemanState((0, 2)),
        expected=(
            CarlemanState((0, 4)) + 5 * CarlemanState((0, 2)) + 2 * CarlemanState()
        ),
        id="(U_0+K_0)^2|u0^2>",
    ),
    Case(
        op=U[Site(1, 0)],
        state=CarlemanState((Site(0, 0), 1)),
        expected=CarlemanState((Site(0, 0), 1), (Site(1, 0), 1)),
        id="U_(1,0)|u_(0,0)>",  # tuple site labels, concrete on both sides
    ),
    Case(
        op=U[i],
        state=CarlemanState((i, 1)),
        expected=CarlemanState((i, 2)),
        id="U_i|u_i>=|u_i^2>",  # op symbol == state symbol
    ),
    Case(
        op=U[i],
        state=CarlemanState((0, 1)),
        expected=(
            KD(0, i) * CarlemanState((0, 2)) + AKD(i, 0) * CarlemanState((0, 1), (i, 1))
        ),
        id="U_i|u0>=d_0i|u0^2>+d_i0|u0 u_i>",
    ),
    Case(
        op=U[i],
        state=CarlemanState((j, n)),
        expected=(
            KD(j, i) * CarlemanState((j, n + 1))
            + AKD(i, j) * CarlemanState((i, 1), (j, n))
        ),
        id="U_i|u_j^n>",  # distinct scalar symbols, symbolic power
    ),
    Case(
        op=U[i] ** 2,
        state=CarlemanState((0, 1)),
        expected=(
            KD(0, i) * CarlemanState((0, 3))
            + AKD(i, 0) * CarlemanState((0, 1), (i, 2))
            + AKD(i, 0) * KD(0, i) * CarlemanState((0, 2), (i, 1))
        ),
        id="U_i^2|u0>",  # polynomial + symbolic
    ),
    Case(
        op=U[i0, i1],
        state=CarlemanState((Site(i0, i1), 1)),
        expected=CarlemanState((Site(i0, i1), 2)),
        id="U_{i0,i1}|u_{i0,i1}>",  # multi op symbol == state symbol
    ),
    Case(
        op=U[i0, i1],
        state=CarlemanState((Site(0, 0), 1)),
        expected=(
            MKD(Site(0, 0), Site(i0, i1)) * CarlemanState((Site(0, 0), 2))
            + MAKD(Site(i0, i1), Site(0, 0))
            * CarlemanState((Site(0, 0), 1), (Site(i0, i1), 1))
        ),
        id="U_{i0,i1}|u_{0,0}>",  # symbolic op, concrete grid state
    ),
    Case(
        op=U[i0, i1],
        state=CarlemanState((Site(j0, j1), 1)),
        expected=(
            MKD(Site(j0, j1), Site(i0, i1)) * CarlemanState((Site(j0, j1), 2))
            + MAKD(Site(i0, i1), Site(j0, j1))
            * CarlemanState((Site(i0, i1), 1), (Site(j0, j1), 1))
        ),
        id="U_{i0,i1}|u_{j0,j1}>",  # both 2-index symbolic
    ),
    Case(
        op=U[i0, i1, i2],
        state=CarlemanState((Site(i0, 5, j), 1)),
        expected=(
            MKD(Site(i0, 5, j), Site(i0, i1, i2)) * CarlemanState((Site(i0, 5, j), 2))
            + MAKD(Site(i0, i1, i2), Site(i0, 5, j))
            * CarlemanState((Site(i0, 5, j), 1), (Site(i0, i1, i2), 1))
        ),
        id="U_{i0,i1,i2}|u_{i0,5,j}>",  # op shares coord-0 symbol; mixed coords
    ),
    Case(
        op=U[i0, 2, i2],
        state=CarlemanState((Site(i0, 5, j), 1)),
        expected=CarlemanState((Site(i0, 2, i2), 1), (Site(i0, 5, j), 1)),
        id="U_{i0,2,i2}|u_{i0,5,j}>",  # numeric 2!=5 -> plain product
    ),
    Case(
        op=U[i0, 5, i2],
        state=CarlemanState((Site(i0, 5, j), 1)),
        expected=(
            MKD(Site(i0, 5, j), Site(i0, 5, i2)) * CarlemanState((Site(i0, 5, j), 2))
            + MAKD(Site(i0, 5, i2), Site(i0, 5, j))
            * CarlemanState((Site(i0, 5, i2), 1), (Site(i0, 5, j), 1))
        ),
        id="U_{i0,5,i2}|u_{i0,5,j}>",  # shared i0, numeric 5==5, distinct i2 vs j
    ),
    Case(
        op=K[i0, 3, i2],
        state=CarlemanState((Site(i0, 3, j), n)),
        expected=(
            n
            * MKD(Site(i0, 3, j), Site(i0, 3, i2))
            * CarlemanState((Site(i0, 3, j), n - 1))
        ),
        id="K_{i0,3,i2}|u_{i0,3,j}^n>",  # lowering: fresh term dies; symbolic power
    ),
]


GH = [
    Case(
        op=U[0],
        state=GaussianHermiteState((0, 1)),
        expected=(
            GaussianHermiteState((0, 2))
            + sp.sqrt(sp.Rational(1, 2)) * GaussianHermiteState((0, 0))
        ),
        id="U_0|psi_1>=|psi_2>+sqrt(1/2)|psi_0>",
    ),
    Case(
        op=K[0],
        state=GaussianHermiteState((0, 1)),
        expected=(
            -GaussianHermiteState((0, 2))
            + sp.sqrt(sp.Rational(1, 2)) * GaussianHermiteState((0, 0))
        ),
        id="K_0|psi_1>=-|psi_2>+sqrt(1/2)|psi_0>",
    ),
    Case(
        op=U[0],
        state=GaussianHermiteState((0, 0)),
        expected=sp.sqrt(sp.Rational(1, 2)) * GaussianHermiteState((0, 1)),
        id="U_0|psi_0>=sqrt(1/2)|psi_1>",  # ground: lower term vanishes
    ),
    Case(
        op=K[0],
        state=GaussianHermiteState((0, 0)),
        expected=-sp.sqrt(sp.Rational(1, 2)) * GaussianHermiteState((0, 1)),
        id="K_0|psi_0>=-sqrt(1/2)|psi_1>",  # K does not annihilate the ground
    ),
    Case(
        op=K[0],
        state=GaussianHermiteState((0, 2)),
        expected=(
            GaussianHermiteState((0, 1))
            - sp.sqrt(sp.Rational(3, 2)) * GaussianHermiteState((0, 3))
        ),
        id="K_0|psi_2>=|psi_1>-sqrt(3/2)|psi_3>",
    ),
    Case(
        op=U[1],
        state=GaussianHermiteState((0, 1)),
        expected=(sp.sqrt(sp.Rational(1, 2)) * GaussianHermiteState((0, 1), (1, 1))),
        id="U_1|psi_(0,1)>",  # new site, order 0
    ),
    Case(
        op=U[i],
        state=GaussianHermiteState((i, 1)),
        expected=(
            GaussianHermiteState((i, 2))
            + sp.sqrt(sp.Rational(1, 2)) * GaussianHermiteState()
        ),
        id="U_i|psi_(i,1)>",  # op symbol == state symbol
    ),
    Case(
        op=U[i],
        state=GaussianHermiteState((0, 1)),
        expected=(
            KD(0, i)
            * (
                GaussianHermiteState((0, 2))
                + sp.sqrt(sp.Rational(1, 2)) * GaussianHermiteState()
            )
            + AKD(i, 0)
            * sp.sqrt(sp.Rational(1, 2))
            * GaussianHermiteState((0, 1), (i, 1))
        ),
        id="U_i|psi_(0,1)>",  # symbolic op, concrete state
    ),
    Case(
        op=U[i0, i1],
        state=GaussianHermiteState((Site(j0, j1), 1)),
        expected=(
            MKD(Site(j0, j1), Site(i0, i1))
            * (
                GaussianHermiteState((Site(j0, j1), 2))
                + sp.sqrt(sp.Rational(1, 2)) * GaussianHermiteState()
            )
            + MAKD(Site(i0, i1), Site(j0, j1))
            * sp.sqrt(sp.Rational(1, 2))
            * GaussianHermiteState((Site(i0, i1), 1), (Site(j0, j1), 1))
        ),
        id="U_{i0,i1}|psi_{j0,j1}>",  # multi 2D symbolic
    ),
    Case(
        op=U[i0, 2, i2],
        state=GaussianHermiteState((Site(i0, 5, j), 1)),
        expected=(
            sp.sqrt(sp.Rational(1, 2))
            * GaussianHermiteState((Site(i0, 2, i2), 1), (Site(i0, 5, j), 1))
        ),
        id="U_{i0,2,i2}|psi_{i0,5,j}>",  # numeric 2!=5 -> single fresh term
    ),
    Case(
        op=K[i0, 3, i2],
        state=GaussianHermiteState((Site(i0, 3, j), n)),
        expected=(
            MKD(Site(i0, 3, j), Site(i0, 3, i2))
            * (
                sp.sqrt(n / 2) * GaussianHermiteState((Site(i0, 3, j), n - 1))
                - sp.sqrt((n + 1) / 2) * GaussianHermiteState((Site(i0, 3, j), n + 1))
            )
            - MAKD(Site(i0, 3, i2), Site(i0, 3, j))
            * sp.sqrt(sp.Rational(1, 2))
            * GaussianHermiteState((Site(i0, 3, i2), 1), (Site(i0, 3, j), n))
        ),
        id="K_{i0,3,i2}|psi_{i0,3,j}^n>",  # lowering, multi partial, symbolic power
    ),
]


@mark.parametrize("case", CARLEMAN, ids=lambda c: c.id)
def test_carleman_ket_evaluates_to_tabulated_result(case: Case) -> None:
    """Each CARLEMAN row evaluates to its expected column under exact equality."""
    assert evaluate(apply_op(case.op, case.state), REG) == case.expected


@mark.parametrize("case", GH, ids=lambda c: c.id)
def test_gaussian_hermite_ket_evaluates_to_tabulated_result(
    case: Case,
) -> None:
    """Each GH row evaluates to its expected column up to the sqrt normal form."""
    assert _agree(evaluate(apply_op(case.op, case.state), REG_GH), case.expected)


def test_multi_deltas_stay_unexpanded_until_doit() -> None:
    """Calling doit replaces the multi deltas with scalar KroneckerDelta factors."""
    # the op algebra keeps Multi* deltas unexpanded; downstream extraction can
    # .doit() to lower them to scalar KroneckerDelta factors when it needs to.
    res = evaluate(apply_op(U[i0, i1], CarlemanState((Site(0, 0), 1))), REG)
    assert res.has(MKD) and res.has(MAKD)
    lowered = res.doit()
    assert not lowered.has(MKD)
    assert not lowered.has(MAKD)
    assert lowered.has(KD)


class _UnregisteredFamily:
    """A state family no registry has a rule for."""


def test_unregistered_family_raises_keyerror() -> None:
    """``OpRuleRepo`` is the registry every ``evaluate`` above reads; registration and
    lookup are exercised by consequence there, and a missing family is the one outcome
    none of those reach."""
    with raises(KeyError):
        OpRuleRepo().get_rule(_UnregisteredFamily)
