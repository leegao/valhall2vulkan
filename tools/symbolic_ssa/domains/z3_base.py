#!/usr/bin/env python3
"""
Composable Z3 Symbolic Domain Interface (`Z3SymbolicDomain`).

Defines the uniform boundary contract between any Z3-backed symbolic value
domain (`SymbolicZ3BitvecDomain`, `SymbolicZ3UDomain`, or custom product/union
Z3 domains) and `SymbolicSsaInterpreter` / `BlockEffectState` / `SharedShaderEnv`.

By putting `lift_z3_*`, `lower_z3_*`, `merge_z3`, and `to_z3_real_f32` on this
interface, `SymbolicSsaInterpreter` and downstream verifiers never inspect
domain-specific value types (`UVal`, etc.) or branch on domain implementations.
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Union

from tools.symbolic_ssa.domains.base import SsaDomain

try:
    import z3
except ImportError:
    z3 = None


class Z3SymbolicDomain(SsaDomain, abc.ABC):
    """Abstract interface for Z3-backed symbolic domains consumed by `SymbolicSsaInterpreter`."""

    @abc.abstractmethod
    def lift_z3_bv32(self, raw: Any) -> Any:
        """Lift a raw 32-bit Z3 BitVec from `SharedShaderEnv`/`BlockEffectState` into this domain."""

    @abc.abstractmethod
    def lift_z3_bv64(self, raw: Any) -> Any:
        """Lift a raw 64-bit Z3 BitVec into this domain."""

    @abc.abstractmethod
    def lift_z3_bool(self, raw: Any) -> Any:
        """Lift a raw Z3 Bool into this domain."""

    @abc.abstractmethod
    def lower_z3_bv32(self, val: Any) -> Any:
        """Lower a domain value to a raw 32-bit Z3 BitVec for memory/output/env boundaries."""

    @abc.abstractmethod
    def lower_z3_bv64(self, val: Any) -> Any:
        """Lower a domain value to a raw 64-bit Z3 BitVec for address/env boundaries."""

    @abc.abstractmethod
    def lower_z3_bool(self, val: Any) -> Any:
        """Lower a domain value to a raw Z3 Bool for branch/path/kill conditions."""

    def lower_z3_val(self, val: Any) -> Any:
        """Recursively lower a scalar or tuple of domain values to raw Z3 values."""
        if isinstance(val, tuple):
            return tuple(self.lower_z3_val(v) for v in val)
        return self.lower_z3_bv32(val)

    @abc.abstractmethod
    def merge_z3(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        """Merge two domain values under a raw Z3 Bool path condition (e.g. Phi nodes)."""

    @abc.abstractmethod
    def to_z3_real_f32(self, val: Any) -> Any:
        """Project a scalar f32 domain value to a Z3 Real expression."""


def make_z3_symbolic_domain(
    domain: Union[str, Z3SymbolicDomain, None] = None,
    fp_mode: str = "exact",
) -> Z3SymbolicDomain:
    """Factory resolving a symbolic domain spec (`'bitvec'`, `'udomain'`, or instance)."""
    if isinstance(domain, Z3SymbolicDomain):
        return domain
    if domain is None or domain in ("udomain", "default", ""):
        from tools.symbolic_ssa.domains.udomain import SymbolicZ3UDomain
        return SymbolicZ3UDomain(exact_bitcast=False)
    if domain == "bitvec":
        from tools.symbolic_ssa.domains.symbolic_z3_bitvec import SymbolicZ3BitvecDomain
        return SymbolicZ3BitvecDomain(fp_mode=fp_mode)
        from tools.symbolic_ssa.domains.udomain import SymbolicZ3UDomain
        return SymbolicZ3UDomain(exact_bitcast=False)
    if domain == "udomain_exact":
        from tools.symbolic_ssa.domains.udomain import SymbolicZ3UDomain
        return SymbolicZ3UDomain(exact_bitcast=True)
    raise ValueError(f"Unknown symbolic SSA domain mode: {domain!r}")
