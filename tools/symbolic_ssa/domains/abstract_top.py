#!/usr/bin/env python3
"""Lattice-Top abstract domain (`AbstractTopDomain`)."""

from __future__ import annotations

from typing import Any

from tools.symbolic_ssa.domains.base import SsaDomain


class AbstractTopDomain(SsaDomain):
    """Lattice-Top domain returning `None` for all value operations."""

    def const_u32(self, imm: int) -> Any: return None
    def const_u64(self, imm: int) -> Any: return None
    def const_bool(self, val: bool) -> Any: return None
    def default_value(self) -> Any: return None
    def bottom(self) -> Any: return None
    def top(self) -> Any: return None
    def cast_value(self, op: str, val: Any, val_type: str) -> Any: return None
    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any: return None
    def idot4(self, op: str, a: Any, b: Any) -> Any: return None
    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> Any: return None
    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> Any: return None
    def fp_unary(self, op: str, x: Any) -> Any: return None
    def fp_binary(self, op: str, a: Any, b: Any) -> Any: return None
    def fp_fma(self, a: Any, b: Any, c: Any) -> Any: return None
