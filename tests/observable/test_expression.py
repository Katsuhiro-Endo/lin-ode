# pyright: reportArgumentType=false, reportIndexIssue=false, reportOperatorIssue=false
# sympy stubs type IndexedBase/Idx construction and Indexed subscripting too loosely
# (same suppression as tests/matrix/test_builder.py); the noise is irreducible here.
"""Tests for ObservableExpression: a thin IndexedExpression subclass for observables of
the raw state, with convenience constructors and its own basis projection."""

import sympy as sp
from pytest import raises

from lin_ode.basis import CarlemanBasis
from lin_ode.indexed_expression import IndexedExpression
from lin_ode.matrix import build_matrix
from lin_ode.observable import ObservableExpression
from lin_ode.operators import SimpleOperator


def test_from_sympy_accepts_expr_with_only_literal_indices() -> None:
    """``u[0]**2 + u[1]**2 + u[2]**2`` already sums over the declared index k.

    So k never appears in the expr, yet it still sets the observable's index range.
    """
    u = sp.IndexedBase("u")
    k = sp.Idx("k", (0, 2))
    expr = u[0] ** 2 + u[1] ** 2 + u[2] ** 2

    observable = ObservableExpression.from_sympy(expr, u, (k,))

    assert isinstance(observable, IndexedExpression)
    assert observable.expr == expr
    assert observable.state_variable == u
    assert observable.indices == (k,)


def test_from_sympy_rejects_expr_without_state_variable() -> None:
    """Inherited from IndexedExpression: from_sympy must not bypass that validation."""
    u, v = sp.IndexedBase("u"), sp.IndexedBase("v")
    k = sp.Idx("k", (0, 2))

    with raises(ValueError, match="state variable u not found in expression"):
        ObservableExpression.from_sympy(v[0] ** 2, u, (k,))


def test_rejects_index_without_explicit_range() -> None:
    """Shape is derived from index bounds, so an unranged index is rejected."""
    u = sp.IndexedBase("u")
    k = sp.Idx("k")  # no explicit range

    with raises(ValueError, match="Index k has no explicit range"):
        ObservableExpression.from_sympy(u[0] ** 2, u, (k,))


def test_from_strings_parses_expr_with_default_state_variable() -> None:
    """The state variable defaults to u, so m and omega stay plain symbols."""
    m, omega = sp.symbols("m omega")
    u = sp.IndexedBase("u")
    k = sp.Idx("k", (0, 2))

    observable = ObservableExpression.from_strings(
        "m * omega**2 * u[0]**2 / 2", indices=(k,)
    )

    assert observable.expr == m * omega**2 * u[0] ** 2 / 2
    assert observable.state_variable == u
    assert observable.indices == (k,)


def test_from_strings_honours_state_variable_override() -> None:
    k = sp.Idx("k", (0, 2))
    """Passing state_variable makes phi the indexed base instead of u."""
    observable = ObservableExpression.from_strings(
        "phi[k]**2", indices=(k,), state_variable="phi"
    )

    assert observable.state_variable.name == "phi"


def test_get_matrix_equals_matrix_built_from_lifted_operator() -> None:
    """get_matrix is a thin wrapper: it must return exactly what the lifted
    SimpleOperator builds directly, with an inert diff_base placeholder."""
    u = sp.IndexedBase("u")
    k = sp.Idx("k", (0, 2))
    observable = ObservableExpression.from_sympy(u[0] ** 2, u, (k,))
    basis = CarlemanBasis(3, degree=2)
    diff_base = sp.IndexedBase("K", commutative=False)

    matrix = observable.get_matrix(basis)

    vacuum = basis.flat_index_of_structured_index(())
    u0_squared = basis.flat_index_of_structured_index(((0, 2),))
    expected = build_matrix(
        SimpleOperator(observable, diff_base), u0_squared, vacuum, basis
    )
    assert matrix.basis is basis
    assert matrix[((0, 2),), ()] == expected == 1
