#!/usr/bin/env python3
"""
Domain-Aware Symbolic Shader Environment (`SymbolicShaderEnv`).

Initialized with the requested `Z3SymbolicDomain` (e.g., `SymbolicZ3UDomain`,
`SymbolicZ3BitvecDomain`, or any custom domain) so symbolic inputs, FAU loads,
and buffer/image arrays stay aligned with the interpreter's analysis domain
instead of constructing ad-hoc raw Z3 expressions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tools.symbolic_ssa import z3utils as z3
from tools.symbolic_ssa.domains.shader_env.base import MASK32, MASK64, ShaderEnvBase
from tools.symbolic_ssa.domains.z3_base import Z3SymbolicDomain, make_z3_symbolic_domain
from tools.symbolic_ssa.z3utils import dsl
from tools.symbolic_ssa.z3utils.helpers import (
    BV32,
    BV64,
    Z3Val,
    _dsl_bv32,
    _try_const_u64,
)


class SymbolicShaderEnv(ShaderEnvBase):
    """Shared symbolic input & disjoint buffer address environment initialized with a `Z3SymbolicDomain`."""

    def __init__(
        self,
        bundle: Dict[str, Any],
        domain: Union[str, Z3SymbolicDomain, None] = None,
        fp_mode: str = "exact",
    ):
        self._init_abi_layout(bundle)
        self.domain: Z3SymbolicDomain = make_z3_symbolic_domain(domain, fp_mode=fp_mode)
        self.inputs: Dict[str, Any] = {}
        self.init_buf_mem: Dict[str, Any] = {}
        self.init_img_mem: Dict[str, Tuple[Any, ...]] = {}
        self.fallback_mem: Any = z3.Array("mem_fallback", BV64, BV32)
        self.fau_sym: Any = z3.Array("fau_sym", BV32, BV32)
        self.lane_id: Any = z3.BitVec("lane_id", 5)
        self.inputs["lane_id"] = self.lane_id

    def read_untyped_slot(self, domain: Optional[Z3SymbolicDomain], raw_bv32: Any) -> Any:
        """Lift an untyped 32-bit shader input/memory word into `domain` (or `self.domain`)."""
        dom = domain if domain is not None else self.domain
        if hasattr(dom, "lift_z3_bv32"):
            return dom.lift_z3_bv32(raw_bv32)
        return raw_bv32

    def is_shared_sym_bkey(self, bkey: str) -> bool:
        return bkey in getattr(self, "shared_sym_bkeys", set())

    def get_sym_buf_slot(self, sym_tag: str, off32: Any) -> Any:
        if not hasattr(self, "sym_buf_ufs"):
            self.sym_buf_ufs: Dict[str, Any] = {}
        if sym_tag not in self.sym_buf_ufs:
            safe = "".join(c if c.isalnum() or c == "_" else "_" for c in sym_tag)
            self.sym_buf_ufs[sym_tag] = z3.Function(f"sym_slot_{safe}", BV32, BV32)
        return self.sym_buf_ufs[sym_tag](off32)

    def get_buf_mem(self, key: str) -> Any:
        if key == "wls_shared" and self.unique_workgroup_key is not None:
            key = self.unique_workgroup_key
        if key not in self.init_buf_mem:
            safe = key.replace(":", "_")
            self.init_buf_mem[key] = z3.Array(f"mem_{safe}", BV32, BV32)
        return self.init_buf_mem[key]

    def build_preconditions(self) -> List[Any]:
        """Build semantic execution-environment preconditions (e.g. Compute LocalSize bounds)."""
        conds: List[Any] = []
        if self.stage == "Compute" and self.local_size is not None:
            lx, ly, lz = self.local_size
            for c, lim in enumerate((lx, ly, lz)):
                lid_k = f"bi_27_{c}"
                if lid_k in self.inputs:
                    conds.append(z3.ULT(self.inputs[lid_k], _dsl_bv32(lim)))
                gid_k = f"bi_28_{c}"
                wg_k = f"bi_26_{c}"
                if gid_k in self.inputs and lid_k in self.inputs and wg_k in self.inputs:
                    conds.append(self.inputs[gid_k] == self.inputs[wg_k] * _dsl_bv32(lim) + self.inputs[lid_k])
            if "bi_29" in self.inputs:
                if all(f"bi_27_{c}" in self.inputs for c in range(3)):
                    lidx_expr = (
                        self.inputs["bi_27_2"] * _dsl_bv32(lx * ly)
                        + self.inputs["bi_27_1"] * _dsl_bv32(lx)
                        + self.inputs["bi_27_0"]
                    )
                    conds.append(self.inputs["bi_29"] == lidx_expr)
                else:
                    conds.append(z3.ULT(self.inputs["bi_29"], _dsl_bv32(lx * ly * lz)))
        return conds

    def get_img_mem(self, key: str) -> Tuple[Any, ...]:
        if key not in self.init_img_mem:
            safe = key.replace(":", "_")
            self.init_img_mem[key] = tuple(z3.Array(f"img_{safe}_c{c}", BV64, BV32) for c in range(4))
        return self.init_img_mem[key]

    def _find_base_region_in_add(self, expr: Any) -> Optional[str]:
        found: Set[str] = set()
        stack: List[Any] = [expr]
        visited = 0
        while stack and visited < 64:
            cur = stack.pop()
            visited += 1
            if z3.is_bv_value(cur) and cur.size() == 64:
                region = (dsl.as_long(cur) >> 32) << 32
                if region in self.base_to_buf:
                    found.add(self.base_to_buf[region])
                    if len(found) > 1:
                        return None
            elif isinstance(cur, dsl.Z3OpBvAdd) or (
                not isinstance(cur, dsl.Z3Expr) and z3.is_app(cur) and cur.decl().kind() == z3.Z3_OP_BADD
            ):
                for ch in cur.children():
                    stack.append(ch)
        return next(iter(found)) if len(found) == 1 else None

    def resolve_addr(
        self,
        addr_expr: Any,
        hint_key: str = "",
    ) -> Tuple[Optional[str], Optional[int], Any]:
        """Decompose a 64-bit address into (buf_key, const_offset_or_None, offset_bv32)."""
        c = _try_const_u64(addr_expr)
        if c is not None:
            region = (c >> 32) << 32
            off = c & MASK32
            if region in self.base_to_buf:
                return self.base_to_buf[region], off, _dsl_bv32(off)
            if hint_key:
                return hint_key, off, _dsl_bv32(off)
            return None, c, _dsl_bv32(off)

        hi = _try_const_u64(z3.Extract(63, 32, addr_expr))
        if hi is not None:
            region = hi << 32
            if region in self.base_to_buf:
                lo = z3.simplify(z3.Extract(31, 0, addr_expr), relift=True)
                lo_c = _try_const_u64(lo)
                return self.base_to_buf[region], lo_c, lo

        bkey = self._find_base_region_in_add(addr_expr)
        if bkey is not None:
            lo = z3.simplify(z3.Extract(31, 0, addr_expr), relift=True)
            lo_c = _try_const_u64(lo)
            return bkey, lo_c, lo

        if hint_key:
            lo = z3.simplify(z3.Extract(31, 0, addr_expr), relift=True)
            lo_c = _try_const_u64(lo)
            return hint_key, lo_c, lo

        return None, None, z3.Extract(31, 0, addr_expr)

    def get_builtin(self, builtin_id: int, val_type: str) -> Z3Val:
        name = f"bi_{builtin_id}"
        if val_type == "V4F32":
            lanes = tuple(z3.BitVec(f"{name}_{c}", 32) for c in range(4))
            for c, lv in enumerate(lanes):
                self.inputs[f"{name}_{c}"] = lv
            return lanes
        if val_type == "V3U32":
            lanes = tuple(z3.BitVec(f"{name}_{c}", 32) for c in range(3))
            for c, lv in enumerate(lanes):
                self.inputs[f"{name}_{c}"] = lv
            return lanes
        if val_type == "Bool":
            v = z3.Bool(name)
            self.inputs[name] = v
            return v
        v = z3.BitVec(name, 32)
        self.inputs[name] = v
        return v

    def get_input_vec4(self, kind: str, location: int) -> Tuple[Any, ...]:
        lanes = tuple(z3.BitVec(f"{kind}_loc{location}_{c}", 32) for c in range(4))
        for c, lv in enumerate(lanes):
            self.inputs[f"{kind}_loc{location}_{c}"] = lv
        return lanes

    def load_fau_word(self, word_idx: int) -> Any:
        if word_idx in self.fcst_by_word:
            return _dsl_bv32(self.fcst_by_word[word_idx])
        if word_idx in self.viewport_fau_words:
            return _dsl_bv32(self.viewport_fau_words[word_idx])
        if word_idx in self.pilot_by_word:
            pf, is_hi = self.pilot_by_word[word_idx]
            vkind = pf.get("value_kind", "")
            if vkind == "const_int" and pf.get("has_const_value", False):
                cv = int(pf["const_value"])
                return _dsl_bv32((cv >> 32) & MASK32) if is_hi else _dsl_bv32(cv & MASK32)
            if vkind in ("buffer_size", "descriptor_addr", "abi_preload", "unknown"):
                return z3.Select(self.fau_sym, _dsl_bv32(word_idx))
            if vkind == "symbol_value" or pf.get("is_loaded_value", False):
                if pf.get("has_byte_offset", False) and not pf.get("is_derived", False):
                    return z3.Select(self.get_buf_mem(pf["key"]), _dsl_bv32(int(pf["byte_offset"])))
                return z3.Select(self.fau_sym, _dsl_bv32(word_idx))
            if pf.get("is_derived", False):
                return z3.Select(self.fau_sym, _dsl_bv32(word_idx))
            base = self.get_buf_base(pf["key"])
            if pf.get("has_pointer_offset", False):
                base = (base + int(pf.get("pointer_offset", 0))) & MASK64
            if not is_hi:
                return _dsl_bv32(base & MASK32)
            return _dsl_bv32((base >> 32) & MASK32)
        if word_idx in self.rloc_by_word:
            bkey, uoff = self.rloc_by_word[word_idx]
            return z3.Select(self.get_buf_mem(bkey), _dsl_bv32(uoff))
        return z3.Select(self.fau_sym, _dsl_bv32(word_idx))
