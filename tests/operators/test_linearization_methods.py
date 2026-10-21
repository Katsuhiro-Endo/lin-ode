"""Tests for the concrete linearization methods (Koopman, Liouville, KvN).

Each method produces a non-commutative operator expression in the state ``U`` and the
derivative marker ``K``; the tests pin down the ordering and sign that distinguish them
(Koopman ``f·K``, Liouville ``-K·f``, KvN ``-(f·K + K·f)/2``).
"""

from collections.abc import Callable

from typing import cast

import sympy as sp
from pytest import mark

from lin_ode.operators import Koopman, KoopmanVonNeumann, Liouville
from lin_ode.operators.interface import LinearizationMethod
from lin_ode.ordinary_differential_system import (
    DirichletCondition,
    Facet,
    OrdinaryDifferentialSystem,
)
from tests.util.comparison import idx as _idx


def _placeholders() -> tuple[sp.Symbol, sp.Symbol]:
    return sp.Symbol("f", commutative=False), sp.Symbol("K", commutative=False)


def _indexed(base: sp.IndexedBase, *indices: sp.Idx) -> sp.Expr:
    """Typed wrapper for ``base[indices]``; sympy's stubs type ``__getitem__`` too wide
    for pyright."""
    return cast(sp.Expr, base[indices])


def _operator_symbols(
    method: LinearizationMethod, ode: OrdinaryDifferentialSystem
) -> tuple[sp.IndexedBase, sp.Expr]:
    """The (U, K[indices]) atoms expected in ``method.operator(ode)``'s expr."""
    state = cast(sp.IndexedBase, sp.IndexedBase("U", commutative=False))
    return state, _indexed(method.diff_base, *ode.equation.indices)


@mark.parametrize(
    "method, expected_ordering",
    [
        (Koopman(), lambda f, K: f * K),
        (Liouville(), lambda f, K: -K * f),
        (KoopmanVonNeumann(), lambda f, K: -(f * K + K * f) / 2),
    ],
    ids=["koopman", "liouville", "kvn"],
)
def test_template_is_methods_own_ordering_of_f_and_k(
    method: LinearizationMethod, expected_ordering: Callable
) -> None:
    """Koopman gives f*K, Liouville gives -K*f, KvN gives -(f*K + K*f)/2."""
    f, K = _placeholders()

    assert method._template(f, K) == expected_ordering(f, K)


def test_templates_do_not_commute() -> None:
    """The Koopman template is not the negative of the Liouville template."""
    # The ordering is the whole point: f·K and K·f must stay distinct.
    f, K = _placeholders()
    assert Koopman()._template(f, K) != -Liouville()._template(f, K)


@mark.parametrize(
    "method, expected_expr",
    [
        (Koopman(), lambda U_i, K_i: U_i**2 * K_i),
        (Liouville(), lambda U_i, K_i: -K_i * U_i**2),
        (KoopmanVonNeumann(), lambda U_i, K_i: -(U_i**2 * K_i + K_i * U_i**2) / 2),
    ],
    ids=["koopman", "liouville", "kvn"],
)
def test_create_operator_substitutes_ode_and_diff_marker(
    method: LinearizationMethod, expected_expr: Callable
) -> None:
    """``f = u[i]**2`` becomes ``U[i]**2`` and the derivative becomes ``K[i]``.

    Each method's operator is its own template with those two substituted in.
    """
    ode = OrdinaryDifferentialSystem.from_strings("u[i]**2", indices=(_idx("i", (0, 2)),))
    result = method.create_operator(ode)._operator
    U, K_i = _operator_symbols(method, ode)
    (i,) = result.indices

    assert result.expr == expected_expr(_indexed(U, i), K_i)
    assert result.state_variable == U
    assert result.indices == ode.equation.indices


def test_substituted_operators_remain_noncommutative() -> None:
    """Substitution keeps Koopman and Liouville from being negatives of each other."""
    # For f = -u[i], Koopman gives -U·K and Liouville gives K·U; if the
    # substituted U/K atoms commuted, the two would collapse to negatives
    # of each other.
    ode = OrdinaryDifferentialSystem.from_strings("-u[i]", indices=(_idx("i", (0, 4)),))
    koopman = Koopman().create_operator(ode)._operator.expr
    liouville = Liouville().create_operator(ode)._operator.expr

    assert liouville != -koopman
    assert sp.expand(liouville + koopman) != 0


def test_operator_2d_indices_index_diff_base() -> None:
    """A two-index equation indexes the derivative marker by both i and j."""
    ode = OrdinaryDifferentialSystem.from_strings(
        "u[i,j]**2", indices=(_idx("i", (0, 2)), _idx("j", (0, 3)))
    )
    method = Koopman()
    result = method.create_operator(ode)._operator

    U, K_ij = _operator_symbols(method, ode)
    i, j = result.indices
    assert result.expr == _indexed(U, i, j) ** 2 * K_ij


def test_operator_on_piecewise_ode_folds_diff_into_arms() -> None:
    i = _idx("i", (0, 4))
    """Only the interior arm carries the derivative marker, and the first arm is 0."""
    ode = OrdinaryDifferentialSystem.from_strings(
        "u[i-1] + u[i+1]",
        indices=(i,),
        conditions=[DirichletCondition(Facet.low(i), 0)],
    )
    result = Koopman().create_operator(ode)._operator

    cases = result.cases
    assert len(cases) == 2
    # The interior arm carries the derivative operator; the Dirichlet one is 0.
    K = cast(sp.IndexedBase, sp.IndexedBase("K", commutative=False))
    assert cases[0][0] == 0
    assert _indexed(K, result.indices[0]) in cases[1][0].atoms(sp.Indexed)
