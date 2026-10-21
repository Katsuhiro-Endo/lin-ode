"""Operator-action rule protocol and per-basis rule registry."""

from __future__ import annotations

from typing import Protocol

import sympy as sp


class OpRule(Protocol):
    def __call__(self, site: sp.Expr, n: sp.Expr) -> sp.Expr:
        """How an operator family acts on a single (site, order) factor."""
        ...


class OpRuleRepo:
    """Maps operator-family types to their action rules for a given basis."""

    def __init__(self) -> None:
        self._rules: dict[type, OpRule] = {}

    def register(self, op_type: type, rule: OpRule) -> None:
        self._rules[op_type] = rule

    def get_rule(self, op_type: type) -> OpRule:
        rule = self._rules.get(op_type)
        if rule is None:
            raise KeyError(f"No rule registered for {op_type!r}")
        return rule
