from math import comb
from typing import Any, Sequence, cast, overload

import sympy as sp

from lin_ode import serialization as _ser
from lin_ode.basis.projection import Projector
from lin_ode.basis.rule_repo import OpRuleRepo
from lin_ode.basis.state import BasisFunction, IdxSpec
from lin_ode.indexing import (
    BasisFunctionArgs,
    get_shape_and_count,
    k_to_state,
    state_to_k,
)


def _as_factors(idx_spec: "IdxSpec | BasisFunctionArgs") -> BasisFunctionArgs:
    """Normalise a subscript into a tuple of ``(site, power)`` factors."""
    factors = idx_spec if all(isinstance(f, tuple) for f in idx_spec) else (idx_spec,)
    return cast(BasisFunctionArgs, factors)


class TruncatedBasisSet:
    """Degree-truncated separable basis labelled by sparse ``(site, power)`` orders.

    Holds every label of total degree (sum of powers) ``<= degree``, ordered so each
    maps to a flat integer index (grlex, see :mod:`lin_ode.indexing`). Subclasses set
    ``state_type`` (monomial for Carleman, Hermite for GH); override ``truncation_rule``
    for a different projection. It also owns the projectors used to expand an expression
    onto it, so projection and truncation cannot be configured inconsistently.

    Structurally satisfies the :class:`~.operators.Basis` protocol, with
    ``IndexSpec = IdxSpec | BasisFunctionArgs``.

    Args:
        n_vars: Variable count — an int, or a shape tuple for D-dim grid sites
            (``sp.Tuple`` labels, flattened row-major).
        degree: Truncation order (max total degree).
        projectors: Projection strategies, tried in order, so a specific one must
            precede a greedier fallback; prepend to ``self.projectors`` after
            construction to shadow a shipped projector with your own.
    """

    state_type: type[BasisFunction]  # set by subclass
    is_l2: bool = False  # in L2(R^N)? only an L2 basis can carry Liouville/KvN

    def __init__(
        self,
        n_vars: int | tuple[int, ...],
        degree: int,
        projectors: Sequence[Projector] = (),
    ):
        self.n_vars = n_vars
        self.degree = degree
        self.projectors = tuple(projectors)
        _, self._count = get_shape_and_count(n_vars)
        self.registry = OpRuleRepo()

    def _extra_json_args(self) -> dict[str, Any]:
        """Return the constructor args beyond ``(n_vars, degree)`` as a JSON-ready dict.

        The base has none. A subclass with extra state, such as PSC's ``pivot``,
        overrides this method and :meth:`_decode_extra_json_args`.
        """
        return {}

    @classmethod
    def _decode_extra_json_args(cls, extra: dict[str, Any]) -> dict[str, Any]:
        """Turn the dict from :meth:`_extra_json_args` back into constructor kwargs."""
        return {}

    def to_json_data(self) -> dict[str, Any]:
        """Return the source inputs as a JSON-ready dict.

        The dict is tagged with the concrete class name. An embedding object
        (``Vector``, ``OperatorMatrix``) reads that tag to rebuild the right subclass.
        ``projectors`` are not serialized. :meth:`from_json_data` recreates the shipped
        defaults instead.
        """
        # ponytail: projectors aren't serialized. A basis with caller-replaced projectors
        # does not reconstruct; add projector serialization when one needs to.
        # n_vars may be a shape tuple. JSON has no tuple, so mark it to restore on load.
        n_vars = list(self.n_vars) if isinstance(self.n_vars, tuple) else self.n_vars
        return {
            "type": type(self).__name__,
            "n_vars": n_vars,
            "n_vars_is_tuple": isinstance(self.n_vars, tuple),
            "degree": self.degree,
            **self._extra_json_args(),
        }

    @staticmethod
    def from_json_data(json_data: dict[str, Any]) -> "TruncatedBasisSet":
        """Rebuild the concrete basis named in ``json_data['type']``."""
        # Look up the class name on the lin_ode.basis package. This finds only the bases
        # that package exports. A basis subclass defined outside lin_ode.basis does not
        # resolve here. Support external subclasses by replacing this getattr with an
        # explicit name-to-class registry that each subclass joins.
        # The import is local: the subclasses live in sibling modules that import this one.
        from lin_ode import basis as _basis

        cls = getattr(_basis, json_data["type"])
        n_vars = (
            tuple(json_data["n_vars"]) if json_data["n_vars_is_tuple"] else json_data["n_vars"]
        )
        return cls(n_vars, json_data["degree"], **cls._decode_extra_json_args(json_data))

    def dump_json(self, indent: int | None = None) -> str:
        """Return this basis as a JSON string.

        All three concrete bases inherit this.
        """
        return _ser.json_dumps(self.to_json_data(), indent)

    @classmethod
    def from_json(cls, json_str: str) -> "TruncatedBasisSet":
        """Rebuild the concrete subclass recorded in the string."""
        return cls.from_json_data(_ser.json_loads(json_str))

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom (flattened site count)."""
        return self._count

    def __len__(self) -> int:
        return comb(self._count + self.degree, self._count)

    @overload
    def flat_indices_of_structured_indices(
        self, indices: "IdxSpec | BasisFunctionArgs"
    ) -> int: ...

    @overload
    def flat_indices_of_structured_indices(
        self, indices: Sequence["IdxSpec | BasisFunctionArgs"]
    ) -> Sequence[int]: ...

    def flat_indices_of_structured_indices(
        self,
        indices: "IdxSpec | BasisFunctionArgs | Sequence[IdxSpec | BasisFunctionArgs]",
    ) -> Sequence[int] | int:
        """Flat indices of one structured ``(site, power)`` index or a batch.

        A ``tuple`` is a single spec, in the shape
        ``flat_index_of_structured_index`` accepts, and yields a bare ``int``;
        any other sequence (a list, typically) is a batch of specs and yields a
        list. The distinction matters because a bare multi-factor spec like
        ``((0, 1), (2, 1))`` would otherwise also parse as a batch of two
        single-factor specs — as a tuple it always means one basis state.

        Raises:
            IndexError: If an index is truncated out of the basis.
        """
        if isinstance(indices, tuple):
            return self._flat_index_or_raise(
                cast("IdxSpec | BasisFunctionArgs", indices)
            )
        return [self._flat_index_or_raise(spec) for spec in indices]

    def _flat_index_or_raise(self, spec: "IdxSpec | BasisFunctionArgs") -> int:
        k = self.flat_index_of_structured_index(spec)
        if k is None:
            raise IndexError(f"{spec!r} exceeds truncation degree {self.degree}")
        return k

    def flat_index_of_structured_index(
        self, idx_spec: "IdxSpec | BasisFunctionArgs"
    ) -> int | None:
        """Flat index of a structured ``(site, power)`` index (None if truncated out).

        Takes the index as a single object — one bare factor ``(site, power)`` or a
        tuple of them — exactly the shape ``__getitem__`` accepts and
        ``structured_idx_of_flat_index`` returns, so the round-trip
        ``flat_index_of_structured_index(structured_idx_of_flat_index(k)) == k`` holds.
        """
        return state_to_k(_as_factors(idx_spec), self.n_vars, self.degree)

    def __getitem__(self, idx_spec: IdxSpec | BasisFunctionArgs) -> BasisFunction:
        """Basis function for one ``(site, power)`` factor or several — as
        ``state_type(*factors)``.

        ``basis[(0, 1)]`` is a single factor; ``basis[(0, 1), (2, 1)]`` is two.
        """
        return self.state_type(*_as_factors(idx_spec))

    def structured_idx_of_flat_index(self, k: int) -> BasisFunctionArgs | None:
        """Structured ``(site, power)`` factors at flat index ``k`` (None if out of
        basis)."""
        return k_to_state(k, self.n_vars, self.degree)

    def flat_index_of_basis_function(self, state: BasisFunction) -> int | None:
        """Flat index of basis function ``state`` (None if total degree exceeds
        ``degree``)."""
        return state_to_k(cast(BasisFunctionArgs, state.args), self.n_vars, self.degree)

    def truncation_rule(self, *args: IdxSpec) -> sp.Expr:
        """In-basis representation of one basis state — default drops it if total degree
        > ``degree``.

        Override (e.g. PSC) to approximate out-of-basis states instead of dropping.
        """
        if sum(n for _, n in args) > self.degree:
            return sp.S.Zero
        return self.state_type(*args)

    def truncate_expression(self, expr: sp.Expr, u: sp.IndexedBase) -> sp.Expr:
        """:meth:`truncation_rule` across a whole expression in ``u``.

        Identity up to expansion unless the rule folds, since projection already
        discards what a dropping one drops; expanded because callers read off terms.
        """
        return sp.expand(expr)
