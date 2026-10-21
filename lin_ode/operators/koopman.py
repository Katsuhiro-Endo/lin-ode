"""Koopman linearization method.

The Koopman operator linearizes dynamics by lifting to an infinite-dimensional
function space. For ODE du/dt = f(u), the Koopman operator acts on observables φ
as: dφ/dt = Kφ = f(u) · ∇φ

In operator notation: K = f * ∂/∂u (right action)
"""

import sympy as sp
from .interface import LinearizationMethod


class Koopman(LinearizationMethod):
    r"""Koopman linearization: :math:`\mathcal{K} = f(u) \cdot \nabla`

    The Koopman operator acts on observables via the chain rule:

    .. math::

        \frac{d\phi}{dt} = \sum_i f_i(u) \frac{\partial \phi}{\partial u_i}

    In operator algebra notation:

    .. math::

        \mathcal{K} = f \cdot K

    where :math:`K` represents the derivative operator and :math:`f` is the
    ODE right-hand side. When applied to a basis, this becomes a sum over
    the ODE's indices.

    Examples:
        >>> from lin_ode.operators.koopman import Koopman
        >>> from lin_ode.ordinary_differential_system import OrdinaryDifferentialSystem
        >>> from lin_ode.basis import CarlemanBasis
        >>> from lin_ode.matrix import OperatorMatrix
        >>> from sympy import Idx
        >>>
        >>> # Define ODE: du[i]/dt = u[i]²
        >>> ode = OrdinaryDifferentialSystem.from_strings("u[i]**2", indices=(Idx("i", (0, 2)),))
        >>>
        >>> # Create Koopman operator
        >>> method = Koopman()
        >>> operator = method.create_operator(ode)
        >>>
        >>> # Project onto Carleman basis
        >>> basis = CarlemanBasis(n_vars=3, max_degree=5)
        >>> matrix = OperatorMatrix(basis, operator)
        >>>
        >>> # Extract a block
        >>> block = matrix[(2, 0), (1, 0)]  # degree 2 to degree 1
    """

    def _template(self, ode: sp.Symbol, diff: sp.Symbol) -> sp.Expr:
        r"""The Koopman operator template: :math:`f \cdot K`

        Args:
            ode: Placeholder for the ODE right-hand side :math:`f`.
            diff: Placeholder for the derivative operator :math:`K`.
        """
        return ode * diff
