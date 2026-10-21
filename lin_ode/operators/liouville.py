"""Liouville (adjoint) linearization method.

The Liouville operator is the adjoint of the Koopman operator. For ODE
du/dt = f(u), the Liouville operator acts on observables as:
dφ/dt = L†φ = -∇ · (f(u) φ)

In operator notation: L = -∂/∂u · f (left action)
"""

import sympy as sp
from .interface import LinearizationMethod


class Liouville(LinearizationMethod):
    r"""Liouville (adjoint) linearization: :math:`\mathcal{L} = -\nabla \cdot f(u)`

    The Liouville operator is the adjoint of the Koopman operator and acts
    on observables as:

    .. math::

        \frac{d\phi}{dt} = -\sum_i \frac{\partial}{\partial u_i}(f_i(u) \phi)

    In operator algebra notation:

    .. math::

        \mathcal{L} = -K \cdot f

    where :math:`K` represents the derivative operator and :math:`f` is the
    ODE right-hand side. The minus sign and ordering (K on left) distinguish
    this from the Koopman operator.

    The Liouville operator preserves normalization for probability distributions
    and is commonly used in density evolution problems.

    Examples:
        >>> from lin_ode.operators.liouville import Liouville
        >>> from lin_ode.ordinary_differential_system import OrdinaryDifferentialSystem
        >>> from lin_ode.basis import GaussianHermiteBasis
        >>> from lin_ode.matrix import OperatorMatrix
        >>> from sympy import Idx
        >>>
        >>> # Define ODE: du[i]/dt = -u[i]
        >>> ode = OrdinaryDifferentialSystem.from_strings("-u[i]", indices=(Idx("i", (0, 2)),))
        >>>
        >>> # Create Liouville operator
        >>> method = Liouville()
        >>> operator = method.create_operator(ode)
        >>>
        >>> # Project onto an L2 basis; Carleman and PSC are rejected here
        >>> basis = GaussianHermiteBasis(n_vars=3, degree=5)
        >>> matrix = OperatorMatrix(basis, operator)
    """

    requires_l2_basis = True

    def _template(self, ode: sp.Symbol, diff: sp.Symbol) -> sp.Expr:
        r"""The Liouville operator template: :math:`-K \cdot f`

        Note the ordering: :math:`K` (derivative) on the left encodes the
        adjoint/left action, with a sign flip relative to Koopman.

        Args:
            ode: Placeholder for the ODE right-hand side :math:`f`.
            diff: Placeholder for the derivative operator :math:`K`.
        """
        return -diff * ode
