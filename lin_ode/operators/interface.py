"""Abstract base class and concrete types for linearization of nonlinear ODEs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast, overload, Sequence, Protocol, TypeVar

import sympy as sp

from lin_ode.ordinary_differential_system.ode import (
    OrdinaryDifferentialSystem,
)
from lin_ode.indexed_expression import IndexedExpression
from lin_ode.operators.linear_operator import OperatorSum

#: Type variable for a basis-specific specification of a single basis function;
#: ranges of basis functions are described as ``Sequence[IndexSpec]``
IndexSpec = TypeVar("IndexSpec", contravariant=True)


class Basis(Protocol[IndexSpec]):
    r"""What this module requires of a basis defined on a state space.

    Represents

    .. math::

        \phi[\mathbf{k}](u), \qquad \mathbf{k} \in \mathcal{I}

    where :math:`\mathbf{k}` is a generic index for the basis, (e.g.
    ``(site, power)`` factors for a Carleman basis) and each
    :math:`\phi[\mathbf{k}]` is a SymPy expression in the state entries
    :math:`u[i,\ldots]`

    This is a :class:`typing.Protocol`: any object with these members
    satisfies it structurally (e.g. :class:`~.basis.TruncatedBasisSet`).
    The ``IndexSpec`` type parameter specifies how the basis expects a
    single basis function to be named in
    :meth:`flat_indices_of_structured_indices` and in
    :class:`OperatorMatrix` indexing operations; a range of basis
    functions is a ``Sequence[IndexSpec]``.
    """

    #: Whether the basis spans L2(R^N), which a method transporting a density needs.
    is_l2: bool

    @overload
    def flat_indices_of_structured_indices(self, indices: IndexSpec) -> int: ...

    @overload
    def flat_indices_of_structured_indices(
        self, indices: Sequence[IndexSpec]
    ) -> Sequence[int]: ...

    def flat_indices_of_structured_indices(
        self, indices: IndexSpec | Sequence[IndexSpec]
    ) -> Sequence[int] | int:
        r"""Convert structured basis-function indices to flat basis indices.

        Args:
            indices: A single basis-function index, yielding its bare ``int``
                flat index — or a sequence of them, yielding a sequence.

        Raises:
            IndexError: If an index has no flat position in this basis.
        """
        ...


class LinearizationMethod(ABC):
    r"""Stateful recipe for lifting a nonlinear ODE to a linear operator.

    Carries method-specific parameters (e.g. truncation order for Carleman,
    quadrature settings for KvN).  Concrete subclasses implement :meth:`_template`
    to combine the ODE and derivative placeholders into an operator formula;
    :meth:`create_operator` wraps it with placeholder creation and substitution
    of a concrete ODE.

    Typical usage::

        method = Carleman()
        operator = method.create_operator(ode)
        matrix = OperatorMatrix(basis, operator)
        block = matrix[(0, 1), (2, 3)]
    """

    #: Liouville/KvN act on a probability distribution, so they need an L2(R^N)
    #: basis; Carleman and PSC are not in L2, so the expansion is undefined.
    requires_l2_basis: bool = False

    @property
    def diff_base(self) -> sp.IndexedBase:
        r"""The :class:`sympy.IndexedBase` marking the derivative operator in
        :meth:`create_operator`'s output.

        Exposed so :func:`~lin_ode.matrix.lift` can recognise the derivative operator in
        the substituted expression without depending on any particular operator-algebra
        implementation.
        """
        return cast(sp.IndexedBase, sp.IndexedBase("K", commutative=False))

    @abstractmethod
    def _template(self, ode: sp.Symbol, diff: sp.Symbol) -> sp.Expr:
        r"""Combine the ODE and derivative placeholders into the operator formula.

        Args:
            ode: Non-commutative placeholder for the ODE right-hand side.
            diff: Non-commutative placeholder for the derivative operator.

        Returns:
            The symbolic expression defining this method's operator template.
        """

    def create_operator(
        self,
        ode: OrdinaryDifferentialSystem,
    ) -> OperatorSum:
        r"""Lift *ode* with this method, returning an :class:`OperatorSum`.

        Builds the ``ode``/``diff`` placeholders, asks :meth:`_template` to
        combine them into a formula, then substitutes *ode*'s right-hand side
        for the ODE placeholder and (if the formula uses it and *ode* is
        indexed) an indexed derivative operator (:attr:`diff_base`) for the
        diff placeholder. The resulting :class:`~.indexed_expression.IndexedExpression`
        is lifted to the raising/lowering algebra by :class:`OperatorSum`.

        Args:
            ode: The nonlinear ODE to lift.
        """
        ode_ph = sp.Symbol("_f_ode", commutative=False)
        diff_ph = sp.Symbol(self.diff_base.name, commutative=False)
        template = self._template(ode_ph, diff_ph)

        ode_expr = ode.equation
        state: sp.IndexedBase = sp.IndexedBase("U", commutative=False)  # type: ignore[assignment]
        subs_dict = {ode_ph: ode_expr.expr.subs(ode_expr.state_variable, state)}

        if template.has(diff_ph) and ode_expr.indices:
            subs_dict[diff_ph] = self.diff_base[*ode_expr.indices]  # type: ignore[dict-item]

        substituted = template.subs(list(subs_dict.items()))
        folded = sp.piecewise_fold(substituted)
        if not isinstance(folded, sp.Expr):
            folded = sp.sympify(folded)

        assert isinstance(folded, sp.Expr)  # Type narrowing for pyright

        operator = IndexedExpression(folded, state, ode_expr.indices)
        operator_sum = OperatorSum(operator, self.diff_base)
        operator_sum.requires_l2_basis = self.requires_l2_basis
        operator_sum.method_name = type(self).__name__
        return operator_sum
