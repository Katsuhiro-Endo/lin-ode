"""Koopman von Neumann linearization method.

The Koopman von Neumann (KvN) operator linearizes dynamics by lifting to an
infinite-dimensional function space, like the Koopman operator, but
symmetrized so its operator ordering matches the KvN Hamiltonian. For ODE
du/dt = f(u), the KvN operator acts on observables φ as:
dφ/dt = Lφ = -{f(u), ∂/∂u} φ / 2

In operator notation: L = -{f, K}/2 (minus the anticommutator of f and the
derivative operator K)
"""

import sympy as sp
from .interface import LinearizationMethod


class KoopmanVonNeumann(LinearizationMethod):
    r"""Koopman-von Neumann linearization: :math:`\mathcal{L} = -\{f(u), \nabla\}/2`

    The KvN operator is the symmetrized Koopman generator, acting on
    observables as:

    .. math::

        \frac{d\phi}{dt} = -\frac{1}{2}\sum_i \left(
            f_i(u) \frac{\partial \phi}{\partial u_i}
            + \frac{\partial}{\partial u_i} \bigl(f_i(u) \phi\bigr)
        \right)

    In operator algebra notation:

    .. math::

        \mathcal{L} = -\frac{f \cdot K + K \cdot f}{2}

    where :math:`K` represents the derivative operator and :math:`f` is the
    ODE right-hand side. Unlike the plain Koopman operator, this ordering
    matches the KvN Hamiltonian, :math:`H_\textup{KvN} = i\hbar \mathcal{L}`.
    When applied to a basis, this becomes a sum over the ODE's indices.

    Examples:
        >>> from lin_ode.operators import KoopmanVonNeumann
        >>> from lin_ode.ordinary_differential_system import OrdinaryDifferentialSystem
        >>> from lin_ode.basis import GaussianHermiteBasis
        >>> from lin_ode.matrix import OperatorMatrix
        >>> from sympy import Idx
        >>>
        >>> # Define ODE: du[i]/dt = u[i]**2
        >>> ode = OrdinaryDifferentialSystem.from_strings("u[i]**2", indices=(Idx("i", (0, 2)),))
        >>>
        >>> # Create the KvN operator
        >>> method = KoopmanVonNeumann()
        >>> operator = method.create_operator(ode)
        >>>
        >>> # Project onto a degree-2 L2 basis; Carleman and PSC are rejected here
        >>> basis = GaussianHermiteBasis(n_vars=3, degree=2)
        >>> matrix = OperatorMatrix(basis, operator)
        >>>
        >>> # A single matrix element: <phi_(0,2)| L |phi_(0,1)>
        >>> matrix[((0, 2),), ((0, 1),)]
        1
    """

    requires_l2_basis = True

    def _template(self, ode: sp.Symbol, diff: sp.Symbol) -> sp.Expr:
        r"""The Koopman von Neumann operator template: :math:`-\{f, K\}/2`

        The expression calculated is related to the KvN Hamiltonian as

        .. math::

            H_\textup{KvN} = i\hbar \mathcal{L} = -i\hbar \{f, K\}/2

        Args:
            ode: Placeholder for the ODE right-hand side :math:`f`.
            diff: Placeholder for the derivative operator :math:`K`.
        """
        return -(ode * diff + diff * ode) / sp.Integer(2)
