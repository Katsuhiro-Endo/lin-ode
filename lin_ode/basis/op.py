# pyright: reportReturnType=false, reportGeneralTypeIssues=false, reportIncompatibleMethodOverride=false, reportAssignmentType=false
# sympy's stubs type .args/.replace as Basic and Function subclasses as AppliedUndef,
# so the above rules are pure noise here; reportArgumentType is kept on and the two
# irreducible .args spots use scoped `# pyright: ignore[reportArgumentType]`.
"""Indexed-operator algebra acting on basis functions.

Defines the operator families ``U`` (raising) and ``K`` (lowering) as indexed
symbols (``U[i]``, ``K[i]``), their symbolic action on a :class:`BasisFunction`
via a basis's rule registry, and the machinery to build and evaluate operator
words applied to a ket:

- :func:`apply_op` builds an unevaluated :class:`OpActedState` (``op |psi>``).
- :func:`evaluate` reduces it to a linear combination of basis functions.

Site indices may be concrete or symbolic, and scalar or D-dimensional tuples
(e.g. ``U[0, i1]`` with a partially-concrete tuple); symbolic indices produce
N-independent results carrying (Multi)KroneckerDelta bookkeeping.
"""

from __future__ import annotations

import functools
from abc import abstractmethod
from typing import Sequence, Union

import sympy as sp

from lin_ode.basis.rule_repo import OpRule, OpRuleRepo
from lin_ode.basis.state import AntiKroneckerDelta, BasisFunction
from lin_ode.basis.multi_delta import MultiAntiKroneckerDelta, MultiKroneckerDelta


def _canonicalize_site_label(indices):
    """OpBase indices -> site label: 1-tuple -> scalar (1-D), D-tuple -> sp.Tuple."""
    if isinstance(indices, sp.Tuple):
        return indices
    if isinstance(indices, tuple):
        return indices[0] if len(indices) == 1 else sp.Tuple(*indices)
    return indices


class OpBase(sp.Indexed):
    """An indexed operator instance, e.g. U[1] or K[3].

    Produced by indexing an OpFamily (U[i] returns OpBase(U, i)). Delegates its action
    on a basis function to its parent OpFamily.
    """

    is_commutative = False

    @property
    def idx(self) -> sp.Expr:
        """Site label: a scalar for 1-D operators, an sp.Tuple for D-dim ones."""
        return _canonicalize_site_label(self.indices)

    @property
    def base(self) -> "OpFamily":
        """The parent operator family.

        Narrows sp.Indexed.base (typed as IndexedBase). It is always an OpFamily here
        because OpBase instances are only ever constructed by OpFamily.__getitem__,
        which passes the family as the base.
        """
        return super().base  # type: ignore[return-value]

    def act_on(self, basis_state: BasisFunction, registry: OpRuleRepo) -> sp.Expr:
        """Apply this operator to a basis function.

        Args:
            basis_state: The ket to act on.
            registry: OpRuleRepo mapping operator types to action rules.

        Returns:
            A SymPy expression of basis functions.
        """
        return self.base.act_on(basis_state, self.idx, registry)

    def _sympystr(self, printer) -> str:
        return f"{self.base.name}[{printer.doprint(self.idx)}]"

    def _latex(self, printer) -> str:
        return rf"{self.base.name}_{{{printer.doprint(self.idx)}}}"


class OpFamily(sp.IndexedBase):
    """Abstract family of indexed operators (e.g. U or K).

    Subclass and implement act_on to define a new operator type.
    Indexing an OpFamily produces an OpBase: U[i] returns OpBase(U, i).
    """

    @abstractmethod
    def act_on(
        self, basis_state: BasisFunction, idx: sp.Expr, registry: OpRuleRepo
    ) -> sp.Expr:
        """Define how this operator family acts on a basis function at index idx.

        Args:
            basis_state: The ket to act on.
            idx: The operator index (e.g. the dimension being targeted).
            registry: OpRuleRepo mapping operator types to action rules.

        Returns:
            A SymPy expression of basis functions.
        """
        ...

    def __getitem__(self, indices: int | sp.Idx | tuple) -> OpBase:
        if isinstance(indices, tuple):
            return OpBase(self, *indices)
        return OpBase(self, indices)


def _kron(site: sp.Expr, target_idx: sp.Expr) -> sp.Expr:
    """Site == target_idx delta.

    Dispatches to MultiKroneckerDelta for D-dimensional (sp.Tuple) site labels. Kept
    unexpanded (no .doit()) so the downstream coefficient-extraction algorithm decides
    when to lower it to scalar KroneckerDelta factors; MultiKroneckerDelta.eval still
    collapses fully-concrete tuples.
    """
    if isinstance(target_idx, sp.Tuple):
        site_tup = site if isinstance(site, sp.Tuple) else sp.Tuple(site)
        return MultiKroneckerDelta(site_tup, target_idx)
    return sp.KroneckerDelta(site, target_idx)


def _anti_kron(target_idx: sp.Expr, site: sp.Expr) -> sp.Expr:
    """target_idx != site delta (1 - KroneckerDelta), kept unexpanded.

    Dispatches to MultiAntiKroneckerDelta for D-dim (sp.Tuple) labels; see
    _kron on why lowering is deferred to downstream extraction.
    """
    if isinstance(target_idx, sp.Tuple):
        site_tup = site if isinstance(site, sp.Tuple) else sp.Tuple(site)
        return MultiAntiKroneckerDelta(target_idx, site_tup)
    return AntiKroneckerDelta(target_idx, site)


def distribute_rule(
    basis_state: BasisFunction, target_idx: sp.Expr, rule: OpRule
) -> sp.Expr:
    """Apply rule to basis_state at target_idx.

    target_idx may be a scalar site label (1-D) or an sp.Tuple (D-dim). When it is one
    of the ket's sites the rule is applied directly; otherwise the result carries
    Kronecker-delta bookkeeping (scalar KroneckerDelta / AntiKroneckerDelta for 1-D,
    MultiKroneckerDelta / MultiAntiKroneckerDelta for D-dim) over the sites it might
    equal. Concrete indices collapse the deltas automatically.
    """
    state_type = type(basis_state)
    sites = [site for site, _ in basis_state.args]

    def sew(acted_site: sp.Expr, acted_expr: sp.Expr) -> sp.Expr:
        def _sew(*args):
            if not args:  # BasisFunction() — site lowered to order 0, eval stripped it
                return basis_state.update_dim(acted_site, sp.S.Zero)
            ((s, new_n),) = args
            return basis_state.update_dim(s, new_n)

        return acted_expr.replace(state_type, _sew)

    if target_idx in sites:
        n = next(occ_n for site, occ_n in basis_state.args if site == target_idx)
        return sew(target_idx, rule(target_idx, n))

    existing = sum(
        (
            _kron(site, target_idx) * sew(site, rule(site, n))
            for site, n in basis_state.args
        ),
        sp.S.Zero,
    )
    anti_delta_prefix = functools.reduce(
        sp.Mul, [_anti_kron(target_idx, s) for s in sites], sp.Integer(1)
    )
    return existing + anti_delta_prefix * sew(target_idx, rule(target_idx, sp.S.Zero))


class UFamily(OpFamily):
    """Raising operator family.

    Encodes multiplication by u_i: increments the order at dimension i by 1 in
    the Carleman basis; encodes x*psi_n via the three-term recurrence in the
    Gaussian-Hermite basis.
    """

    def act_on(
        self, basis_state: BasisFunction, idx: sp.Expr, registry: OpRuleRepo
    ) -> sp.Expr:
        return distribute_rule(basis_state, idx, registry.get_rule(type(self)))


class KFamily(OpFamily):
    """Lowering (differentiation) operator family.

    Encodes d/du_i: decrements the order at dimension i with a combinatorial factor in
    the Carleman basis; encodes d psi_n / dx in the Gaussian-Hermite basis.
    """

    def act_on(
        self, basis_state: BasisFunction, idx: sp.Expr, registry: OpRuleRepo
    ) -> sp.Expr:
        return distribute_rule(basis_state, idx, registry.get_rule(type(self)))


U = UFamily("U")
K = KFamily("K")


class OpActedState(sp.Expr):
    """Unevaluated symbolic expression of operators acting on a ket.

    Represents e.g. U[0] K[1] |psi> without computing the result. Operators are stored
    left-to-right (outermost first); the rightmost operator acts first when flattened.
    Call evaluate() to reduce to basis functions.
    """

    is_commutative = False

    def __new__(cls, ket: BasisFunction, *ops: OpBase) -> "OpActedState":
        for op in ops:
            assert isinstance(op, OpBase)
        return super().__new__(cls, ket, sp.Tuple(*ops))

    @property
    def ket(self) -> BasisFunction:
        return self.args[0]

    @property
    def ops(self) -> tuple[OpBase, ...]:
        """Operators in left-to-right (outermost-first) order."""
        return tuple(self.args[1])  # pyright: ignore[reportArgumentType]

    def apply(self, *new_ops: OpBase) -> "OpActedState":
        """Prepend new_ops to the operator stack, acting from the left.

        Args:
            *new_ops: Operators to prepend, in left-to-right order.

        Returns:
            A new OpActedState with new_ops acting before the existing ops.

        Example:
            state.apply(U[2]) ->  U[2] [existing ops] |ket>
        """
        return OpActedState(self.ket, *new_ops, *self.ops)

    def flatten_once(
        self, registry: OpRuleRepo
    ) -> Union["OpActedState", BasisFunction, sp.Expr]:
        """Apply the rightmost (innermost) operator once.

        Args:
            registry: OpRuleRepo mapping operator types to action rules.

        Returns:
            - OpActedState with one fewer operator if operators remain.
            - A BasisFunction expression if this was the last operator.
            - sp.S.Zero if the action annihilates the state.
        """
        *rest, last = self.ops
        last_acted_state = last.act_on(self.ket, registry)
        if len(rest) == 0:
            return last_acted_state

        if not last_acted_state.has(BasisFunction):
            return sp.S.Zero
        cls = type(self.ket)

        def _rule(*args):
            state = cls(*args)
            return OpActedState(state, *rest)

        return last_acted_state.replace(BasisFunction, _rule)

    def _sympystr(self, printer) -> str:
        ops_str = " ".join(printer.doprint(op) for op in self.ops)
        return f"{ops_str} |{printer.doprint(self.ket)}>"

    def _latex(self, printer) -> str:
        ops_str = " ".join(printer._print(op) for op in self.ops)
        return rf"{ops_str} \left| {printer._print(self.ket)} \right\rangle"


def evaluate(expr: sp.Expr, registry: OpRuleRepo) -> sp.Expr:
    """Fully evaluate all OpActedState objects in an expression.

    Repeatedly flattens OpActedState nodes until none remain, returning
    a pure linear combination of BasisFunction atoms.

    Args:
        expr: A SymPy expression possibly containing OpActedState nodes.
        registry: OpRuleRepo mapping operator types to action rules.

    Returns:
        An expression containing only BasisFunction atoms and scalar coefficients.
    """

    def rule(*args) -> sp.Expr:
        ket, ops = args
        new_state = OpActedState(ket, *ops)
        return new_state.flatten_once(registry).expand()

    while expr.has(OpActedState):
        expr = expr.replace(OpActedState, rule)
        expr = expr.expand()

    return expr


def _separate_coeff_from_op(op: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Split a Mul expression into its scalar coefficient and operator part.

    Args:
        op: A SymPy Mul containing operator and scalar factors.

    Returns:
        (pure_op, coeff) where pure_op * coeff == op.
    """
    coeffs = [e for e in op.args if not e.has(OpBase)]
    ops = [e for e in op.args if e.has(OpBase)]
    coeff = functools.reduce(lambda a, b: a * b, coeffs, sp.S.One)
    pure_op = functools.reduce(lambda a, b: a * b, ops, sp.S.One)
    return pure_op, coeff


def _parse_op(op_args: Sequence[sp.Expr]) -> Sequence[OpBase]:
    """Expand a sequence of OpBase and Pow into a flat list of OpBase.

    Powers are expanded by repetition: K[0]**2 becomes [K[0], K[0]].

    Args:
        op_args: Ordered sequence of OpBase or sp.Pow(OpBase, int).

    Returns:
        Flat list of OpBase in the same left-to-right order.

    Raises:
        TypeError: If an element is neither OpBase nor sp.Pow.
    """
    ops = []
    for op in op_args:
        if isinstance(op, OpBase):
            ops.append(op)
        else:
            if not isinstance(op, sp.Pow):
                raise TypeError(f"unexpected operator type: {type(op)}")
            _op, pow = op.args
            ops.extend([_op for _ in range(int(pow))])
    return ops


def _apply_op_to_basis_state(
    coeff: sp.Expr, op: OpBase | sp.Mul | sp.Pow, state: BasisFunction
) -> sp.Expr:
    """Wrap a BasisFunction in an OpActedState for the given operator and coefficient.

    Args:
        coeff: Scalar prefactor.
        op: Operator expression (OpBase, Mul of OpBase, or Pow of OpBase).
        state: The ket to act on.

    Returns:
        coeff * OpActedState(state, *parsed_ops).

    Raises:
        TypeError: If op is an unsupported type.
    """
    if isinstance(op, OpBase):
        return coeff * OpActedState(state, op)
    elif isinstance(op, sp.Mul):
        return coeff * OpActedState(state, *_parse_op(op.args))
    elif isinstance(op, sp.Pow):
        return coeff * OpActedState(state, *_parse_op([op]))
    else:
        raise TypeError(f"unexpected operator type: {type(op)}")


def _apply_op_to_op_acted_state(
    coeff: sp.Expr, op: OpBase | sp.Mul | sp.Pow, state: OpActedState
) -> sp.Expr:
    """Prepend an operator to an existing OpActedState.

    Args:
        coeff: Scalar prefactor.
        op: Operator expression (OpBase, Mul of OpBase, or Pow of OpBase).
        state: The existing OpActedState to prepend to.

    Returns:
        coeff * state.apply(*parsed_ops).

    Raises:
        TypeError: If op is an unsupported type.
    """
    if isinstance(op, OpBase):
        return coeff * state.apply(op)
    elif isinstance(op, sp.Mul):
        return coeff * state.apply(*_parse_op(op.args))
    elif isinstance(op, sp.Pow):
        return coeff * state.apply(*_parse_op([op]))
    else:
        raise TypeError(f"unexpected operator type: {type(op)}")


def _apply_op(
    coeff: sp.Expr,
    op: OpBase | sp.Mul | sp.Pow,
    state: BasisFunction | OpActedState,
) -> sp.Expr:
    """Dispatch to the BasisFunction or OpActedState handler based on state type."""
    if isinstance(state, BasisFunction):
        return _apply_op_to_basis_state(coeff, op, state)
    return _apply_op_to_op_acted_state(coeff, op, state)


def apply_op(op: sp.Expr, state: BasisFunction | OpActedState) -> sp.Expr:
    """Apply a symbolic operator expression to a single state (unevaluated).

    Constructs an OpActedState representing op|state> without reducing it.
    Call evaluate() on the result to obtain the explicit basis-function expansion.

    Handles:
    - Add: distributes linearly over operator terms.
    - Mul: separates scalar coefficient from operator factors.
    - OpBase: single indexed operator.
    - Pow: repeated application via _parse_op.
    - Scalar: returns op * state unchanged.

    Args:
        op: Operator expression built from U[i], K[i], and SymPy scalars.
        state: The ket to act on.

    Returns:
        An unevaluated OpActedState expression (or a scalar times state).
    """
    op = sp.expand(op)  # distribute sums/powers: (U+K)**2, U*(U+K) -> Add of op words
    if op.is_Add:
        return sp.Add(*[apply_op(arg, state) for arg in op.args])  # pyright: ignore[reportArgumentType]
    if op.is_Mul:
        pure_op, coeff = _separate_coeff_from_op(op)
        if not isinstance(pure_op, (OpBase, sp.Mul, sp.Pow)):
            return coeff * apply_op(pure_op, state)
        return _apply_op(coeff, pure_op, state)
    if isinstance(op, OpBase):
        return _apply_op(sp.S.One, op, state)
    if isinstance(op, sp.Pow):
        return _apply_op(sp.S.One, op, state)

    return op * state
