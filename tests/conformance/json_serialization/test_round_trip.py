# pyright: reportArgumentType=false, reportIndexIssue=false, reportOperatorIssue=false
# pyright: reportGeneralTypeIssues=false
# sympy stubs type IndexedBase subscripting and Basic.args too loosely (same
# suppression as tests/conformance/objective_function/test_matrix_representation.py).
# pyright: reportAttributeAccessIssue=false, reportUndefinedVariable=false
# ruff: noqa: F821
"""A-6: every input and output object must be JSON-serializable.

One test per object A-6 requires: B-1 to B-5 on the inputs, D-1 to D-3 on the outputs.
The library has no ``dump_json`` or ``from_json`` yet, so every test that calls them
fails. Only the fixed initial value round-trips today, through plain ``json``.
"""

import json
import math

import sympy as sp

from lin_ode.basis import CarlemanBasis, GaussianHermiteBasis, PSCarlemanBasis
from lin_ode.matrix import Matrix, OperatorMatrix
from lin_ode.observable import ObservableExpression
from lin_ode.operators import Koopman, OperatorSum
from lin_ode.ordinary_differential_system import (
    Box,
    DerivativeCondition,
    DirichletCondition,
    Facet,
    OrdinaryDifferentialSystem,
)
from lin_ode.vector import Vector

from tests.testing import assert_equal, example
from tests.util.comparison import s_pivot
from tests.util.equations import quadratic_equation


class TestJsonSerialization:
    @example(spec_ref=["A-6", "B-1"])
    def test_ode_round_trips_through_json(self):
        """B-1: the ODE round-trips through JSON."""
        u = sp.IndexedBase("u")
        i, j = sp.Idx("i", (0, 2)), sp.Idx("j", (0, 3))
        ode = OrdinaryDifferentialSystem.from_sympy(
            u[i, j] * u[i - 1, j],
            u,
            (i, j),
            conditions=[
                DirichletCondition(Box({i: (0, 1), j: (1, 2)}), sp.Rational(1, 2)),
                DerivativeCondition(Facet.high(i), sp.Rational(1, 3)),
                DerivativeCondition(Facet.low(i) & Facet.high(j), sp.Rational(1, 4)),
            ],
        )
        json_str = ode.dump_json(indent=2)
        print(json_str)
        parsed = OrdinaryDifferentialSystem.from_json(json_str)

        # --- verification ---
        assert_equal(parsed, ode)

    @example(spec_ref=["A-6", "B-2-1", "D-2"])
    def test_fixed_initial_value_round_trips_through_json(self):
        """B-2-1 / D-2: the fixed initial value x_0 round-trips through JSON."""
        LABELS = [(), ((0, 1),), ((1, 1),), ((0, 2),), ((0, 1), (1, 1))]
        x0 = [0.3, 0.7, 1.1]
        basis = CarlemanBasis(3, 2)
        y0_entries = [
            math.prod(x0[int(site)] ** int(power) for site, power in basis[label].args)
            for label in LABELS
        ]
        json_str = json.dumps(y0_entries, indent=2)
        print(json_str)
        parsed = json.loads(json_str)

        # --- verification ---
        assert_equal(parsed, y0_entries)

    @example(spec_ref=["A-6", "B-2-2", "D-2"])
    def test_initial_distribution_round_trips_through_json(self):
        """B-2-2 / D-2: the initial distribution rho_0(x) round-trips through JSON."""
        u = sp.IndexedBase("u")
        rho0 = Vector(u[0] ** 2 + u[1] ** 2, GaussianHermiteBasis(2, 2), u)
        json_str = rho0.dump_json(indent=2)
        print(json_str)
        parsed = Vector.from_json(json_str)

        # --- verification ---
        assert_equal(parsed, rho0)

    @example(spec_ref=["A-6", "B-3", "D-3"])
    def test_objective_function_vector_round_trips_through_json(self):
        """B-3 / D-3: the objective function g(x) vector round-trips through JSON."""
        u = sp.IndexedBase("u")
        vector = Vector(u[0] ** 2, GaussianHermiteBasis(2, 2), u)
        json_str = vector.dump_json(indent=2)
        print(json_str)
        parsed = Vector.from_json(json_str)

        # --- verification ---
        assert_equal(parsed, vector)

    @example(spec_ref=["A-6", "B-3", "D-3"])
    def test_objective_function_matrix_round_trips_through_json(self):
        """B-3 / D-3: the objective function g(x) matrix (KvN) round-trips through
        JSON."""
        u = sp.IndexedBase("u")
        k = sp.Idx("k", (0, 3))
        g = ObservableExpression.from_sympy(u[0] ** 2, u, (k,))
        matrix = g.get_matrix(GaussianHermiteBasis(4, 2))
        json_str = matrix.dump_json(indent=2)
        print(json_str)
        parsed = OperatorMatrix.from_json(json_str)

        # --- verification ---
        assert_equal(parsed, matrix)

    @example(spec_ref=["A-6", "B-4"])
    def test_linearized_operator_round_trips_through_json(self):
        """B-4: the chosen linearization method's operator round-trips through JSON."""
        op = Koopman().create_operator(quadratic_equation(6))
        json_str = op.dump_json(indent=2)
        print(json_str)
        parsed = OperatorSum.from_json(json_str)

        # --- verification ---
        assert_equal(parsed, op)

    @example(spec_ref=["A-6", "B-5-1"])
    def test_carleman_basis_round_trips_through_json(self):
        """B-5-1: a Carleman basis round-trips through JSON."""
        basis = CarlemanBasis(6, 2)
        json_str = basis.dump_json(indent=2)
        print(json_str)
        parsed = CarlemanBasis.from_json(json_str)

        # --- verification ---
        assert_equal(parsed, basis)

    @example(spec_ref=["A-6", "B-5-2"])
    def test_psc_basis_round_trips_through_json(self):
        """B-5-2: a Pivot Switching Carleman basis round-trips through JSON."""
        basis = PSCarlemanBasis(6, 2, s_pivot)
        json_str = basis.dump_json(indent=2)
        print(json_str)
        parsed = PSCarlemanBasis.from_json(json_str)

        # --- verification ---
        assert_equal(parsed, basis)

    @example(spec_ref=["A-6", "B-5-3"])
    def test_gaussian_hermite_basis_round_trips_through_json(self):
        """B-5-3: a Gaussian-weighted Hermite basis round-trips through JSON."""
        basis = GaussianHermiteBasis(4, 2)
        json_str = basis.dump_json(indent=2)
        print(json_str)
        parsed = GaussianHermiteBasis.from_json(json_str)

        # --- verification ---
        assert_equal(parsed, basis)

    @example(spec_ref=["A-6", "D-1"])
    def test_linearized_matrix_round_trips_through_json(self):
        """D-1: a linearized-matrix sub-block round-trips through JSON."""
        LABELS = [(), ((3, 1),), ((3, 2),)]  # vacuum, u_3, u_3**2
        op = Koopman().create_operator(quadratic_equation(6))
        matrix = OperatorMatrix(CarlemanBasis(6, 2), op)
        block = matrix[LABELS, LABELS]
        json_str = block.dump_json(indent=2)
        print(json_str)
        parsed = Matrix.from_json(json_str)

        # --- verification ---
        assert_equal(parsed, block)
