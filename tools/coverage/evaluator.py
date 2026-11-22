#!/usr/bin/env python3
"""
Unified Concrete & Symbolic (Z3) SSA Evaluator for ValhallSsaModule.

Houses:
  - ShaderEnvBase: single source of truth for ABI / MBS2 / FAU / pilot / rloc /
    ViewportTransform / workgroup / LocalSize / disjoint 4 GB VA slot mapping.
  - SharedShaderEnv(ShaderEnvBase) + BlockEffectState + DagSsaZ3Evaluator (symbolic Z3).
  - ConcreteShaderEnv(ShaderEnvBase) + ConcreteSsaInterpreter (fast concrete Python).
  - evaluate_ssa(...): unified entrypoint supporting mode="symbolic" or mode="concrete".
"""

from __future__ import annotations

import dataclasses
import fractions
import hashlib
import math
import random
import struct
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import z3

from tools.verify.panfrost_z3_model import (
    BV8,
    BV32,
    BV64,
    FP32_SORT,
    _bitrev_bv32,
    _clz_bv,
    _fp_to_int_sat,
    _fp_total_le_bv,
    _hw_uf,
    _popcount_bv32,
    bv32,
    bv64,
    bv32_to_fp32,
    f16_bv_to_f32_bv,
    f32_bv_to_f16_bv,
    fp32_to_bv32,
)


class ShaderEnvBase:
    """Shared ABI / MBS2 / FAU / viewport / VA-region layout for concrete and symbolic envs."""

    def _init_abi_layout(self, bundle: Dict[str, Any]) -> None:
        self.bundle = bundle
        self.stage: str = bundle.get("stage", "Compute")
        self.mbs2: Dict[str, Any] = bundle.get("mbs2_context", {})
        self.direct_bindings: List[Dict[str, Any]] = bundle.get("direct_bindings", [])
        self.buf_base_map: Dict[str, int] = {}
        self.base_to_buf: Dict[int, str] = {}
        self._next_region = 1

        ls_raw = self.mbs2.get("local_size")
        if isinstance(ls_raw, list) and len(ls_raw) == 3 and all(int(x) > 0 for x in ls_raw):
            self.local_size: Optional[Tuple[int, int, int]] = (int(ls_raw[0]), int(ls_raw[1]), int(ls_raw[2]))
        else:
            self.local_size = None

        for b in self.direct_bindings:
            key = b.get("key", "")
            if b.get("kind") in ("UBO", "SSBO", "PushConstant") and key:
                self.get_buf_base(key)

        wg_keys: List[str] = []
        for n in bundle.get("direct_ssa", {}).get("nodes", []):
            astr = n.get("aux_str", "")
            if n.get("op") == "LoadSidebandU64" and astr.startswith("workgroup:"):
                if astr not in wg_keys:
                    wg_keys.append(astr)
        self.unique_workgroup_key: Optional[str] = wg_keys[0] if len(wg_keys) == 1 else None
        if self.unique_workgroup_key is not None:
            self.buf_base_map["wls_shared"] = self.get_buf_base(self.unique_workgroup_key)

        for pf in self.mbs2.get("pilot_fau", []):
            key = pf.get("key", "")
            if pf.get("resolved") and key:
                self.get_buf_base(key)
        for rl in self.mbs2.get("rlocs", []):
            self.get_buf_base(f"ubo:0:{rl.get('binding', 0)}")

        self.fcst_by_word: Dict[int, int] = {
            int(fc["word_idx"]): int(fc["value"]) & 0xFFFFFFFF for fc in self.mbs2.get("fcsts", [])
        }
        self.viewport_fau_words: Dict[int, int] = {}
        self.pilot_by_word: Dict[int, Tuple[Dict[str, Any], bool]] = {}
        for pf in self.mbs2.get("pilot_fau", []):
            vkind = pf.get("value_kind", "")
            is_const = (vkind == "const_int" and pf.get("has_const_value", False))
            if is_const or (pf.get("resolved") and pf.get("key")):
                w = int(pf["word_idx"])
                is_hi = (int(pf["fau_offset"]) % 8) != 0
                self.pilot_by_word[w] = (pf, is_hi)
                is_buf_addr = (vkind == "buffer_addr") or (not vkind and not pf.get("is_loaded_value", False))
                if is_buf_addr and not is_hi and int(pf.get("byte_size", 4)) >= 8:
                    if (w + 1) not in self.pilot_by_word:
                        self.pilot_by_word[w + 1] = (pf, True)

        self.rloc_by_word: Dict[int, Tuple[str, int]] = {}
        for rl in self.mbs2.get("rlocs", []):
            sym = rl.get("symbol", "")
            w0 = int(rl["word_idx"])
            nwords = max(1, (int(rl.get("size", 4)) + 3) // 4)
            uoff = int(rl.get("uni_offset", 0))
            if sym == "gl_mali_ViewportTransform":
                for k in range(nwords):
                    off = uoff + k * 4
                    self.viewport_fau_words[w0 + k] = 0x3F800000 if off < 16 else 0x00000000
                continue
            bkey = f"ubo:0:{rl.get('binding', 0)}"
            for k in range(nwords):
                self.rloc_by_word[w0 + k] = (bkey, uoff + k * 4)

        self.sinp_addr_to_loc: Dict[int, int] = {
            int(s["addr"]): int(s["location"]) for s in self.mbs2.get("sinp", []) if int(s.get("location", -1)) >= 0
        }
        self.sout_addr_to_loc: Dict[int, int] = {
            int(s["addr"]): int(s["location"]) for s in self.mbs2.get("sout", []) if int(s.get("location", -1)) >= 0
        }

    def get_buf_base(self, key: str) -> int:
        if key not in self.buf_base_map:
            base = self._next_region << 32
            self._next_region += 1
            self.buf_base_map[key] = base
            self.base_to_buf[base] = key
        return self.buf_base_map[key]

Z3Val = Union[z3.ExprRef, Tuple[Any, ...]]

REAL_SORT = z3.RealSort()
_REAL_ZERO = z3.RealVal(0)
_REAL_ONE = z3.RealVal(1)
_REAL_TWO = z3.RealVal(2)
_REAL_HALF = z3.Q(1, 2)
_REAL_NEG_HALF = z3.Q(-1, 2)
_REAL_NEG_ONE = z3.RealVal(-1)


def _f32_bits_to_real_const(u32: int) -> Optional[z3.ArithRef]:
    f = struct.unpack("<f", struct.pack("<I", u32 & 0xFFFFFFFF))[0]
    if not math.isfinite(f):
        return None
    if f == 0.0:
        return _REAL_ZERO
    frac = fractions.Fraction(f)
    return z3.Q(frac.numerator, frac.denominator)


_LOG2_E_REAL = _f32_bits_to_real_const(0x3FB8AA3B)  # log2(e) in binary32
_LN_2_REAL = _f32_bits_to_real_const(0x3F317218)    # ln(2) in binary32


def _is_real2bv(x: z3.ExprRef) -> bool:
    return z3.is_app(x) and x.decl() == _hw_uf("real2bv", REAL_SORT, BV32) and len(x.children()) == 1


def _bv32_to_real(x: z3.BitVecRef) -> z3.ArithRef:
    """Decode BV32 to exact Z3 Real, cancelling real2bv wrappers and pushing through ITEs."""
    if _is_real2bv(x):
        return x.children()[0]
    if z3.is_app(x) and x.decl().kind() == z3.Z3_OP_ITE and len(x.children()) == 3:
        c, t, e = x.children()
        return z3.If(c, _bv32_to_real(t), _bv32_to_real(e))
    if isinstance(x, z3.BitVecNumRef):
        rc = _f32_bits_to_real_const(x.as_long())
        if rc is not None:
            return rc
    return _hw_uf("bv2real", BV32, REAL_SORT)(x)


def _real_to_bv32(r: z3.ArithRef) -> z3.BitVecRef:
    """Wrap a Z3 Real expression back into BV32 via the canonical real2bv bridge."""
    return _hw_uf("real2bv", REAL_SORT, BV32)(r)


def _real_sqrt(rx: z3.ArithRef) -> z3.ArithRef:
    if isinstance(rx, z3.RatNumRef):
        num, den = rx.numerator_as_long(), rx.denominator_as_long()
        if num == 0:
            return _REAL_ZERO
        if num == den:
            return _REAL_ONE
    return _hw_uf("real_sqrt", REAL_SORT, REAL_SORT)(rx)


def _real_exp2(rx: z3.ArithRef) -> z3.ArithRef:
    if isinstance(rx, z3.RatNumRef):
        num, den = rx.numerator_as_long(), rx.denominator_as_long()
        if num == 0:
            return _REAL_ONE
        if num == den:
            return _REAL_TWO
    return _hw_uf("real_exp2", REAL_SORT, REAL_SORT)(rx)


def _real_log2(rx: z3.ArithRef) -> z3.ArithRef:
    if isinstance(rx, z3.RatNumRef):
        num, den = rx.numerator_as_long(), rx.denominator_as_long()
        if num == den:
            return _REAL_ZERO
        if num == 2 * den:
            return _REAL_ONE
    return _hw_uf("real_log2", REAL_SORT, REAL_SORT)(rx)


def _real_sin(rx: z3.ArithRef) -> z3.ArithRef:
    if isinstance(rx, z3.RatNumRef) and rx.numerator_as_long() == 0:
        return _REAL_ZERO
    return _hw_uf("real_sin", REAL_SORT, REAL_SORT)(rx)


def _real_cos(rx: z3.ArithRef) -> z3.ArithRef:
    if isinstance(rx, z3.RatNumRef) and rx.numerator_as_long() == 0:
        return _REAL_ONE
    return _hw_uf("real_cos", REAL_SORT, REAL_SORT)(rx)


def _as_bv32(v: Z3Val) -> z3.BitVecRef:
    if isinstance(v, tuple):
        return _as_bv32(v[0]) if v else bv32(0)
    if isinstance(v, z3.BoolRef):
        return z3.If(v, bv32(1), bv32(0))
    if isinstance(v, z3.FPRef):
        return fp32_to_bv32(v)
    assert isinstance(v, z3.BitVecRef), f"Expected BitVecRef, got {type(v)}"
    if v.size() == 32:
        return v
    if v.size() < 32:
        return z3.ZeroExt(32 - v.size(), v)
    return z3.Extract(31, 0, v)


def _as_bv64(v: Z3Val) -> z3.BitVecRef:
    if isinstance(v, tuple):
        return _as_bv64(v[0]) if v else bv64(0)
    if isinstance(v, z3.BitVecRef):
        if v.size() == 64:
            return v
        if v.size() < 64:
            return z3.ZeroExt(64 - v.size(), v)
        return z3.Extract(63, 0, v)
    return z3.ZeroExt(32, _as_bv32(v))


def _as_bool(v: Z3Val) -> z3.BoolRef:
    if isinstance(v, tuple):
        return _as_bool(v[0]) if v else z3.BoolVal(False)
    if isinstance(v, z3.BoolRef):
        return v
    if isinstance(v, z3.BitVecRef):
        if isinstance(v, z3.BitVecNumRef):
            return z3.BoolVal(v.as_long() != 0)
        return v != z3.BitVecVal(0, v.size())
    return z3.BoolVal(False)


def _z3_and(a: z3.BoolRef, b: z3.BoolRef) -> z3.BoolRef:
    if z3.is_false(a) or z3.is_false(b):
        return z3.BoolVal(False)
    if z3.is_true(a):
        return b
    if z3.is_true(b):
        return a
    if a.eq(b):
        return a
    return z3.And(a, b)


def _z3_or(a: z3.BoolRef, b: z3.BoolRef) -> z3.BoolRef:
    if z3.is_true(a) or z3.is_true(b):
        return z3.BoolVal(True)
    if z3.is_false(a):
        return b
    if z3.is_false(b):
        return a
    if a.eq(b):
        return a
    return z3.Or(a, b)


def _z3_not(a: z3.BoolRef) -> z3.BoolRef:
    if z3.is_true(a):
        return z3.BoolVal(False)
    if z3.is_false(a):
        return z3.BoolVal(True)
    return z3.Not(a)


def _z3_if(cond: z3.BoolRef, t_val: Z3Val, f_val: Z3Val) -> Z3Val:
    if z3.is_true(cond):
        return t_val
    if z3.is_false(cond):
        return f_val
    if isinstance(t_val, tuple) and isinstance(f_val, tuple) and len(t_val) == len(f_val):
        return tuple(_z3_if(cond, a, b) for a, b in zip(t_val, f_val))
    if isinstance(t_val, z3.ExprRef) and isinstance(f_val, z3.ExprRef):
        if t_val.eq(f_val):
            return t_val
        if isinstance(t_val, z3.BoolRef) and isinstance(f_val, z3.BoolRef):
            return z3.If(cond, t_val, f_val)
        if isinstance(t_val, z3.BitVecRef) and isinstance(f_val, z3.BitVecRef) and t_val.size() == f_val.size():
            return z3.If(cond, t_val, f_val)
        return z3.If(cond, _as_bv32(t_val), _as_bv32(f_val))
    return t_val


def _try_const_u64(e: z3.ExprRef) -> Optional[int]:
    if isinstance(e, z3.BitVecNumRef):
        return e.as_long()
    s = z3.simplify(e)
    if isinstance(s, z3.BitVecNumRef):
        return s.as_long()
    return None



def _vec_len(vt: str) -> int:
    if vt.startswith("V2"):
        return 2
    if vt.startswith("V3"):
        return 3
    if vt.startswith("V4"):
        return 4
    return 1


def _eval_elementwise(vt: str, args: List[Z3Val], fn: Any) -> Z3Val:
    vlen = _vec_len(vt)
    is_vec = vlen > 1 or any(isinstance(a, tuple) for a in args)
    if not is_vec:
        return fn(*args)
    if vlen == 1:
        vlen = max(len(a) for a in args if isinstance(a, tuple))
    lanes = []
    for i in range(vlen):
        lane_args = [a[i] if isinstance(a, tuple) and i < len(a) else a for a in args]
        lanes.append(fn(*lane_args))
    return tuple(lanes)


class SharedShaderEnv(ShaderEnvBase):
    """Shared symbolic input & disjoint buffer address environment for both SSA modules."""

    def __init__(self, bundle: Dict[str, Any]):
        self.stage: str = bundle.get("stage", "Compute")
        self.mbs2: Dict[str, Any] = bundle.get("mbs2_context", {})
        self.direct_bindings: List[Dict[str, Any]] = bundle.get("direct_bindings", [])
        self.inputs: Dict[str, z3.ExprRef] = {}

        # Disjoint 4GB VA slots per canonical buffer key (e.g. "ssbo:0:1" -> 0x0001_0000_0000)
        self.buf_base_map: Dict[str, int] = {}
        self.base_to_buf: Dict[int, str] = {}
        self._next_region = 1

        # Compute workgroup LocalSize (from MBS2 KERN or SPIR-V OpExecutionMode)
        ls_raw = self.mbs2.get("local_size")
        if isinstance(ls_raw, list) and len(ls_raw) == 3 and all(int(x) > 0 for x in ls_raw):
            self.local_size: Optional[Tuple[int, int, int]] = (int(ls_raw[0]), int(ls_raw[1]), int(ls_raw[2]))
        else:
            self.local_size = None

        # Pre-allocate regions for all direct_bindings and pilot_fau keys
        for b in self.direct_bindings:
            key = b.get("key", "")
            if b.get("kind") in ("UBO", "SSBO", "PushConstant") and key:
                self.get_buf_base(key)

        # Unify Valhall wls_shared with a unique SPIR-V workgroup:<id> shared variable
        wg_keys: List[str] = []
        for n in bundle.get("direct_ssa", {}).get("nodes", []):
            astr = n.get("aux_str", "")
            if n.get("op") == "LoadSidebandU64" and astr.startswith("workgroup:"):
                if astr not in wg_keys:
                    wg_keys.append(astr)
        self.unique_workgroup_key: Optional[str] = wg_keys[0] if len(wg_keys) == 1 else None
        if self.unique_workgroup_key is not None:
            wg_base = self.get_buf_base(self.unique_workgroup_key)
            self.buf_base_map["wls_shared"] = wg_base
        for pf in self.mbs2.get("pilot_fau", []):
            key = pf.get("key", "")
            if pf.get("resolved") and key:
                self.get_buf_base(key)
        for rl in self.mbs2.get("rlocs", []):
            key = f"ubo:0:{rl.get('binding', 0)}"
            self.get_buf_base(key)

        # Initial symbolic memory arrays per buffer (word-aligned byte-indexed: BV32 -> BV32)
        self.init_buf_mem: Dict[str, z3.ArrayRef] = {}
        self.fallback_mem: z3.ArrayRef = z3.Array("mem_fallback", BV64, BV32)

        # Build fast lookup maps from MBS2 context
        self.fcst_by_word: Dict[int, int] = {}
        for fc in self.mbs2.get("fcsts", []):
            self.fcst_by_word[int(fc["word_idx"])] = int(fc["value"]) & 0xFFFFFFFF

        self.init_img_mem: Dict[str, Tuple[z3.ArrayRef, ...]] = {}
        self.viewport_fau_words: Dict[int, int] = {}

        # Pilot FAU bindings: (pf_dict, is_hi_word)
        self.pilot_by_word: Dict[int, Tuple[Dict[str, Any], bool]] = {}
        for pf in self.mbs2.get("pilot_fau", []):
            vkind = pf.get("value_kind", "")
            is_const = (vkind == "const_int" and pf.get("has_const_value", False))
            if is_const or (pf.get("resolved") and pf.get("key")):
                w = int(pf["word_idx"])
                is_hi_word = (int(pf["fau_offset"]) % 8) != 0
                self.pilot_by_word[w] = (pf, is_hi_word)
                is_buf_addr = (vkind == "buffer_addr") or (not vkind and not pf.get("is_loaded_value", False))
                if is_buf_addr and not is_hi_word and int(pf.get("byte_size", 4)) >= 8:
                    if (w + 1) not in self.pilot_by_word:
                        self.pilot_by_word[w + 1] = (pf, True)

        self.rloc_by_word: Dict[int, Tuple[str, int]] = {}
        for rl in self.mbs2.get("rlocs", []):
            sym = rl.get("symbol", "")
            w0 = int(rl["word_idx"])
            nwords = max(1, (int(rl.get("size", 4)) + 3) // 4)
            uoff = int(rl.get("uni_offset", 0))
            if sym == "gl_mali_ViewportTransform":
                # TODO(leegao@): Lift symbolic gl_mali_ViewportTransform epilogue onto direct_ssa
                # (ndc.xyz = clip.xyz / clip.w; win.xyz = ndc.xyz * vp_scale.xyz + vp_bias.xyz)
                # so Vertex gl_Position equivalence holds for arbitrary non-identity viewports and clip.w != 1.0.
                # Identity viewport precondition: scale = (1,1,1,1) at uni+0..0xc, bias = (0,0,0,0) at uni+0x10..0x1c.
                for k in range(nwords):
                    off = uoff + k * 4
                    self.viewport_fau_words[w0 + k] = 0x3F800000 if off < 16 else 0x00000000
                continue
            bkey = f"ubo:0:{rl.get('binding', 0)}"
            for k in range(nwords):
                self.rloc_by_word[w0 + k] = (bkey, uoff + k * 4)

        # Valhall slot (addr) -> SPIR-V location maps
        self.sinp_addr_to_loc: Dict[int, int] = {}
        for s in self.mbs2.get("sinp", []):
            loc = int(s.get("location", -1))
            if loc >= 0:
                self.sinp_addr_to_loc[int(s["addr"])] = loc

        self.sout_addr_to_loc: Dict[int, int] = {}
        for s in self.mbs2.get("sout", []):
            loc = int(s.get("location", -1))
            if loc >= 0:
                self.sout_addr_to_loc[int(s["addr"])] = loc

        # Shared symbolic FAU array for unbound FAU words
        self.fau_sym = z3.Array("fau_sym", BV32, BV32)
        self.lane_id = z3.BitVec("lane_id", 5)
        self.inputs["lane_id"] = self.lane_id

    def get_buf_base(self, key: str) -> int:
        if key not in self.buf_base_map:
            base = self._next_region << 32
            self._next_region += 1
            self.buf_base_map[key] = base
            self.base_to_buf[base] = key
        return self.buf_base_map[key]

    def is_shared_sym_bkey(self, bkey: str) -> bool:
        return bkey in getattr(self, 'shared_sym_bkeys', set())

    def get_sym_buf_slot(self, sym_tag: str, off32: z3.BitVecRef) -> z3.BitVecRef:
        if not hasattr(self, "sym_buf_ufs"):
            self.sym_buf_ufs: Dict[str, z3.FuncDeclRef] = {}
        if sym_tag not in self.sym_buf_ufs:
            safe = "".join(c if c.isalnum() or c == "_" else "_" for c in sym_tag)
            self.sym_buf_ufs[sym_tag] = z3.Function(f"sym_slot_{safe}", BV32, BV32)
        return self.sym_buf_ufs[sym_tag](off32)

    def get_buf_mem(self, key: str) -> z3.ArrayRef:
        if key == "wls_shared" and self.unique_workgroup_key is not None:
            key = self.unique_workgroup_key
        if key not in self.init_buf_mem:
            safe = key.replace(":", "_")
            self.init_buf_mem[key] = z3.Array(f"mem_{safe}", BV32, BV32)
        return self.init_buf_mem[key]

    def build_preconditions(self) -> List[z3.BoolRef]:
        """Build semantic execution-environment preconditions (e.g. Compute LocalSize bounds)."""
        conds: List[z3.BoolRef] = []
        if self.stage == "Compute" and self.local_size is not None:
            lx, ly, lz = self.local_size
            for c, lim in enumerate((lx, ly, lz)):
                lid_k = f"bi_27_{c}"
                if lid_k in self.inputs:
                    conds.append(z3.ULT(self.inputs[lid_k], bv32(lim)))
                gid_k = f"bi_28_{c}"
                wg_k = f"bi_26_{c}"
                if gid_k in self.inputs and lid_k in self.inputs and wg_k in self.inputs:
                    conds.append(self.inputs[gid_k] == self.inputs[wg_k] * bv32(lim) + self.inputs[lid_k])
            if "bi_29" in self.inputs:
                if all(f"bi_27_{c}" in self.inputs for c in range(3)):
                    lidx_expr = (
                        self.inputs["bi_27_2"] * bv32(lx * ly)
                        + self.inputs["bi_27_1"] * bv32(lx)
                        + self.inputs["bi_27_0"]
                    )
                    conds.append(self.inputs["bi_29"] == lidx_expr)
                else:
                    conds.append(z3.ULT(self.inputs["bi_29"], bv32(lx * ly * lz)))
        return conds

    def get_img_mem(self, key: str) -> Tuple[z3.ArrayRef, ...]:
        if key not in self.init_img_mem:
            safe = key.replace(":", "_")
            self.init_img_mem[key] = tuple(z3.Array(f"img_{safe}_c{c}", BV64, BV32) for c in range(4))
        return self.init_img_mem[key]

    def _find_base_region_in_add(self, expr: z3.ExprRef) -> Optional[str]:
        """Walk an additive 64-bit address AST to recover a unique registered buffer region base."""
        found: Set[str] = set()
        stack: List[z3.ExprRef] = [expr]
        visited = 0
        while stack and visited < 64:
            cur = stack.pop()
            visited += 1
            if isinstance(cur, z3.BitVecNumRef) and cur.size() == 64:
                region = (cur.as_long() >> 32) << 32
                if region in self.base_to_buf:
                    found.add(self.base_to_buf[region])
                    if len(found) > 1:
                        return None
            elif z3.is_app(cur) and cur.decl().kind() == z3.Z3_OP_BADD:
                for ch in cur.children():
                    stack.append(ch)
        return next(iter(found)) if len(found) == 1 else None

    def resolve_addr(
        self,
        addr_expr: z3.BitVecRef,
        hint_key: str = "",
    ) -> Tuple[Optional[str], Optional[int], z3.BitVecRef]:
        """Decompose a 64-bit address into (buf_key, const_offset_or_None, offset_bv32)."""
        c = _try_const_u64(addr_expr)
        if c is not None:
            region = (c >> 32) << 32
            off = c & 0xFFFFFFFF
            if region in self.base_to_buf:
                return self.base_to_buf[region], off, bv32(off)
            if hint_key:
                return hint_key, off, bv32(off)
            return None, c, bv32(off)

        # Try extracting high 32 bits to identify buffer region for dynamic offsets
        hi = _try_const_u64(z3.Extract(63, 32, addr_expr))
        if hi is not None:
            region = hi << 32
            if region in self.base_to_buf:
                lo = z3.simplify(z3.Extract(31, 0, addr_expr))
                lo_c = _try_const_u64(lo)
                return self.base_to_buf[region], lo_c, lo

        # Walk additive chains (e.g. base + SignExt(32, idx * stride) + field_off)
        bkey = self._find_base_region_in_add(addr_expr)
        if bkey is not None:
            lo = z3.simplify(z3.Extract(31, 0, addr_expr))
            lo_c = _try_const_u64(lo)
            return bkey, lo_c, lo

        if hint_key:
            lo = z3.simplify(z3.Extract(31, 0, addr_expr))
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

    def get_input_vec4(self, kind: str, location: int) -> Tuple[z3.BitVecRef, ...]:
        lanes = tuple(z3.BitVec(f"{kind}_loc{location}_{c}", 32) for c in range(4))
        for c, lv in enumerate(lanes):
            self.inputs[f"{kind}_loc{location}_{c}"] = lv
        return lanes

    def load_fau_word(self, word_idx: int) -> z3.BitVecRef:
        if word_idx in self.fcst_by_word:
            return bv32(self.fcst_by_word[word_idx])
        if word_idx in self.viewport_fau_words:
            return bv32(self.viewport_fau_words[word_idx])
        if word_idx in self.pilot_by_word:
            pf, is_hi = self.pilot_by_word[word_idx]
            vkind = pf.get("value_kind", "")
            if vkind == "const_int" and pf.get("has_const_value", False):
                cv = int(pf["const_value"])
                return bv32((cv >> 32) & 0xFFFFFFFF) if is_hi else bv32(cv & 0xFFFFFFFF)
            if vkind in ("buffer_size", "descriptor_addr", "abi_preload", "unknown"):
                return z3.Select(self.fau_sym, bv32(word_idx))
            if vkind == "symbol_value" or pf.get("is_loaded_value", False):
                if pf.get("has_byte_offset", False) and not pf.get("is_derived", False):
                    return z3.Select(self.get_buf_mem(pf["key"]), bv32(int(pf["byte_offset"])))
                return z3.Select(self.fau_sym, bv32(word_idx))
            if pf.get("is_derived", False):
                return z3.Select(self.fau_sym, bv32(word_idx))
            base = self.get_buf_base(pf["key"])
            if pf.get("has_pointer_offset", False):
                base = (base + int(pf.get("pointer_offset", 0))) & 0xFFFFFFFFFFFFFFFF
            if not is_hi:
                return bv32(base & 0xFFFFFFFF)
            return bv32((base >> 32) & 0xFFFFFFFF)
        if word_idx in self.rloc_by_word:
            bkey, uoff = self.rloc_by_word[word_idx]
            return z3.Select(self.get_buf_mem(bkey), bv32(uoff))
        return z3.Select(self.fau_sym, bv32(word_idx))


@dataclasses.dataclass
class BlockEffectState:
    """Symbolic observable state at the end of a basic block."""

    discarded: z3.BoolRef
    terminated: z3.BoolRef
    demoted: z3.BoolRef
    frag_colors: Dict[int, Tuple[z3.BitVecRef, ...]]
    builtin_outputs: Dict[int, Z3Val]
    vert_varyings: Dict[int, Tuple[z3.BitVecRef, ...]]
    # Constant-offset buffer writes: (buf_key, offset) -> (written_cond, val_bv32)
    const_stores: Dict[Tuple[str, int], Tuple[z3.BoolRef, z3.BitVecRef]]
    # Symbolized fixed-stride SSBO/UBO stores: sym_tag -> (written_cond, val_bv32, off_bv32)
    sym_buf_stores: Dict[str, Tuple[z3.BoolRef, z3.BitVecRef, z3.BitVecRef]]
    # Dynamic buffer memory arrays: buf_key -> Array(BV32, BV32)
    buf_mems: Dict[str, z3.ArrayRef]
    # Image write arrays: img_key -> Tuple[Array(BV64, BV32) x 4]
    image_mems: Dict[str, Tuple[z3.ArrayRef, ...]]
    fallback_mem: z3.ArrayRef
    # Ordered per-buffer atomic RMW event log: buf_key -> [(active_cond, op_tag, off_or_addr, arg, cmp)]
    atomic_events: Dict[str, List[Tuple[z3.BoolRef, str, z3.BitVecRef, z3.BitVecRef, z3.BitVecRef]]]
    # Track originating PCs per output for counterexample localization
    output_pcs: Dict[str, Set[int]]


def _clone_state(st: BlockEffectState) -> BlockEffectState:
    return BlockEffectState(
        discarded=st.discarded,
        terminated=st.terminated,
        demoted=st.demoted,
        frag_colors=dict(st.frag_colors),
        builtin_outputs=dict(st.builtin_outputs),
        vert_varyings=dict(st.vert_varyings),
        const_stores=dict(st.const_stores),
        sym_buf_stores=dict(st.sym_buf_stores),
        buf_mems=dict(st.buf_mems),
        image_mems=dict(st.image_mems),
        fallback_mem=st.fallback_mem,
        atomic_events={k: list(v) for k, v in st.atomic_events.items()},
        output_pcs={k: set(v) for k, v in st.output_pcs.items()},
    )


def _merge_states(
    incoming: List[Tuple[z3.BoolRef, BlockEffectState]],
    env: SharedShaderEnv,
) -> BlockEffectState:
    if len(incoming) == 1:
        return _clone_state(incoming[0][1])

    # Build cascading z3.If merge across incoming (edge_cond, state) pairs.
    base_st = _clone_state(incoming[0][1])
    for cond, st in incoming[1:]:
        if z3.is_false(cond):
            continue
        base_st.terminated = _z3_if(cond, st.terminated, base_st.terminated)
        base_st.demoted = _z3_if(cond, st.demoted, base_st.demoted)
        base_st.discarded = _z3_or(base_st.terminated, base_st.demoted)
        all_rts = set(base_st.frag_colors.keys()) | set(st.frag_colors.keys())
        for rt in all_rts:
            a = base_st.frag_colors.get(rt, (bv32(0),) * 4)
            b = st.frag_colors.get(rt, (bv32(0),) * 4)
            base_st.frag_colors[rt] = tuple(_z3_if(cond, bi, ai) for ai, bi in zip(a, b))
        all_bis = set(base_st.builtin_outputs.keys()) | set(st.builtin_outputs.keys())
        for bi in all_bis:
            if bi in st.builtin_outputs and bi in base_st.builtin_outputs:
                base_st.builtin_outputs[bi] = _z3_if(cond, st.builtin_outputs[bi], base_st.builtin_outputs[bi])
            elif bi in st.builtin_outputs:
                base_st.builtin_outputs[bi] = st.builtin_outputs[bi]
        all_vv = set(base_st.vert_varyings.keys()) | set(st.vert_varyings.keys())
        for loc in all_vv:
            a = base_st.vert_varyings.get(loc, (bv32(0),) * 4)
            b = st.vert_varyings.get(loc, (bv32(0),) * 4)
            base_st.vert_varyings[loc] = tuple(_z3_if(cond, bi, ai) for ai, bi in zip(a, b))
        all_sym = set(base_st.sym_buf_stores.keys()) | set(st.sym_buf_stores.keys())
        for stag in all_sym:
            wa, va, oa = base_st.sym_buf_stores.get(stag, (z3.BoolVal(False), bv32(0), bv32(0)))
            wb, vb, ob = st.sym_buf_stores.get(stag, (z3.BoolVal(False), bv32(0), bv32(0)))
            base_st.sym_buf_stores[stag] = (_z3_if(cond, wb, wa), _z3_if(cond, vb, va), _z3_if(cond, ob, oa))
        all_cs = set(base_st.const_stores.keys()) | set(st.const_stores.keys())
        for key in all_cs:
            w_a, v_a = base_st.const_stores.get(key, (z3.BoolVal(False), z3.Select(env.get_buf_mem(key[0]), bv32(key[1]))))
            w_b, v_b = st.const_stores.get(key, (z3.BoolVal(False), z3.Select(env.get_buf_mem(key[0]), bv32(key[1]))))
            base_st.const_stores[key] = (_z3_if(cond, w_b, w_a), _z3_if(cond, v_b, v_a))
        all_bm = set(base_st.buf_mems.keys()) | set(st.buf_mems.keys())
        for bkey in all_bm:
            m_a = base_st.buf_mems.get(bkey, env.get_buf_mem(bkey))
            m_b = st.buf_mems.get(bkey, env.get_buf_mem(bkey))
            if not m_a.eq(m_b):
                base_st.buf_mems[bkey] = z3.If(cond, m_b, m_a)
        all_im = set(base_st.image_mems.keys()) | set(st.image_mems.keys())
        for ikey in all_im:
            im_a = base_st.image_mems.get(ikey, env.get_img_mem(ikey))
            im_b = st.image_mems.get(ikey, env.get_img_mem(ikey))
            base_st.image_mems[ikey] = tuple(
                z3.If(cond, b, a) if not a.eq(b) else a for a, b in zip(im_a, im_b)
            )
        if not base_st.fallback_mem.eq(st.fallback_mem):
            base_st.fallback_mem = z3.If(cond, st.fallback_mem, base_st.fallback_mem)
        all_ae = set(base_st.atomic_events.keys()) | set(st.atomic_events.keys())
        for akey in all_ae:
            ev_a = base_st.atomic_events.get(akey, [])
            ev_b = st.atomic_events.get(akey, [])
            if len(ev_a) == len(ev_b):
                merged_ev = []
                for ea, eb in zip(ev_a, ev_b):
                    merged_ev.append((
                        _z3_if(cond, eb[0], ea[0]),
                        eb[1] if ea[1] == eb[1] else f"{ea[1]}|{eb[1]}",
                        _z3_if(cond, eb[2], ea[2]),
                        _z3_if(cond, eb[3], ea[3]),
                        _z3_if(cond, eb[4], ea[4]),
                    ))
                base_st.atomic_events[akey] = merged_ev
            else:
                # Preserve shared prefix once without duplicating across CFG diamonds
                pref = 0
                min_len = min(len(ev_a), len(ev_b))
                while pref < min_len:
                    ea, eb = ev_a[pref], ev_b[pref]
                    if ea[1] == eb[1] and ea[0].eq(eb[0]) and ea[2].eq(eb[2]) and ea[3].eq(eb[3]) and ea[4].eq(eb[4]):
                        pref += 1
                    else:
                        break
                ncond = _z3_not(cond)
                combined = list(ev_a[:pref])
                combined.extend((_z3_and(ncond, e[0]), e[1], e[2], e[3], e[4]) for e in ev_a[pref:])
                combined.extend((_z3_and(cond, e[0]), e[1], e[2], e[3], e[4]) for e in ev_b[pref:])
                base_st.atomic_events[akey] = combined
        for k, pcs in st.output_pcs.items():
            base_st.output_pcs.setdefault(k, set()).update(pcs)
    return base_st


class DagSsaZ3Evaluator:
    """Whole-CFG topological DAG evaluator for a serialized ValhallSsaModule."""

    def __init__(
        self,
        ssa_json: Dict[str, Any],
        env: SharedShaderEnv,
        is_valhall_side: bool,
        fp_mode: str = "exact",
        max_unroll: int = 1,
    ):
        self.ssa = ssa_json
        self.env = env
        self.is_valhall = is_valhall_side
        self.fp_mode = fp_mode
        self.max_unroll = max(1, int(max_unroll))
        self.blocks: List[Dict[str, Any]] = ssa_json.get("blocks", [])
        self.nodes_by_id: Dict[int, Dict[str, Any]] = {int(n["id"]): n for n in ssa_json.get("nodes", [])}
        self.values: Dict[int, Z3Val] = {0: bv32(0)}
        self.has_loops: bool = False
        self.max_unroll_depth: int = 0

    def collect_pcs_for_nodes(self, root_nids: Set[int], max_visit: int = 2048) -> List[int]:
        """Lazily collect originating Valhall PCs reachable from sink node IDs (bounded BFS)."""
        pcs: Set[int] = set()
        seen: Set[int] = set()
        queue: List[int] = list(root_nids)
        while queue and len(seen) < max_visit:
            nid = queue.pop()
            if nid in seen:
                continue
            seen.add(nid)
            node = self.nodes_by_id.get(nid)
            if not node:
                continue
            if "orig_pc" in node:
                pcs.add(int(node["orig_pc"]))
            for aid in node.get("args", []):
                ia = int(aid)
                if ia not in seen:
                    queue.append(ia)
            for _, vid in node.get("phi", []):
                iv = int(vid)
                if iv not in seen:
                    queue.append(iv)
        return sorted(pcs)

    def _fp_unary(self, op: str, x: z3.BitVecRef) -> z3.BitVecRef:
        if self.fp_mode == "real":
            rx = _bv32_to_real(x)
            if op == "FAbs":
                return _real_to_bv32(z3.If(rx < _REAL_ZERO, -rx, rx))
            if op == "FNeg":
                return _real_to_bv32(-rx)
            if op == "Sqrt":
                return _real_to_bv32(_real_sqrt(rx))
            if op == "InverseSqrt":
                return _real_to_bv32(_REAL_ONE / _real_sqrt(rx))
            if op == "Exp2":
                return _real_to_bv32(_real_exp2(rx))
            if op == "Exp":
                return _real_to_bv32(_real_exp2(rx * _LOG2_E_REAL))
            if op == "Log2":
                return _real_to_bv32(_real_log2(rx))
            if op == "Log":
                return _real_to_bv32(_real_log2(rx) * _LN_2_REAL)
            if op == "Sin":
                return _real_to_bv32(_real_sin(rx))
            if op == "Cos":
                return _real_to_bv32(_real_cos(rx))
            return _real_to_bv32(_hw_uf(f"real_{op}", REAL_SORT, REAL_SORT)(rx))
        if op == "FAbs":
            return x & bv32(0x7FFFFFFF)
        if op == "FNeg":
            return x ^ bv32(0x80000000)
        if self.fp_mode == "uf" or op in ("InverseSqrt", "Sin", "Cos", "Exp", "Exp2", "Log", "Log2", "Atan", "Asin", "Acos", "Sinh", "Cosh", "Tanh", "DPdx", "DPdy", "Fwidth", "FrexpExp", "FrexpSignificand"):
            return _hw_uf(f"fp_{op}", BV32, BV32)(x)
        f = bv32_to_fp32(x)
        if op == "Floor":
            return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTN(), f))
        if op == "Ceil":
            return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTP(), f))
        if op == "Trunc":
            return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTZ(), f))
        if op == "RoundEven":
            return fp32_to_bv32(z3.fpRoundToIntegral(z3.RNE(), f))
        if op == "Sqrt":
            return fp32_to_bv32(z3.fpSqrt(z3.RNE(), f))
        return _hw_uf(f"fp_{op}", BV32, BV32)(x)

    def _fp_binary(self, op: str, a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
        if self.fp_mode == "real":
            ra, rb = _bv32_to_real(a), _bv32_to_real(b)
            if op == "FAdd":
                return _real_to_bv32(ra + rb)
            if op == "FSub":
                return _real_to_bv32(ra - rb)
            if op == "FMul":
                return _real_to_bv32(ra * rb)
            if op == "FDiv":
                return _real_to_bv32(ra / rb)
            if op in ("FMin", "NMin"):
                return _real_to_bv32(z3.If(ra <= rb, ra, rb))
            if op in ("FMax", "NMax"):
                return _real_to_bv32(z3.If(ra >= rb, ra, rb))
            if op == "Pow":
                if isinstance(rb, z3.RatNumRef):
                    num, den = rb.numerator_as_long(), rb.denominator_as_long()
                    if num == 0:
                        return _real_to_bv32(_REAL_ONE)
                    if num == den:
                        return _real_to_bv32(ra)
                    if num == 2 * den:
                        return _real_to_bv32(ra * ra)
                    if 2 * num == den:
                        return _real_to_bv32(_real_sqrt(ra))
                    if 2 * num == -den:
                        return _real_to_bv32(_REAL_ONE / _real_sqrt(ra))
                    if num == -den:
                        return _real_to_bv32(_REAL_ONE / ra)
                return _real_to_bv32(_real_exp2(rb * _real_log2(ra)))
            return _real_to_bv32(_hw_uf(f"real_{op}", REAL_SORT, REAL_SORT, REAL_SORT)(ra, rb))
        if self.fp_mode == "uf" or op in ("Pow", "Ldexp", "Atan2"):
            return _hw_uf(f"fp_{op}", BV32, BV32, BV32)(a, b)
        fa, fb = bv32_to_fp32(a), bv32_to_fp32(b)
        if op == "FAdd":
            return fp32_to_bv32(z3.fpAdd(z3.RNE(), fa, fb))
        if op == "FSub":
            return fp32_to_bv32(z3.fpSub(z3.RNE(), fa, fb))
        if op == "FMul":
            return fp32_to_bv32(z3.fpMul(z3.RNE(), fa, fb))
        if op == "FDiv":
            return fp32_to_bv32(z3.fpDiv(z3.RNE(), fa, fb))
        if op in ("FMin", "NMin"):
            return fp32_to_bv32(z3.fpMin(fa, fb))
        if op in ("FMax", "NMax"):
            return fp32_to_bv32(z3.fpMax(fa, fb))
        return _hw_uf(f"fp_{op}", BV32, BV32, BV32)(a, b)

    def _fp_cmp(self, op: str, a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
        if self.fp_mode == "real":
            ra, rb = _bv32_to_real(a), _bv32_to_real(b)
            if op == "FCmpOrdEq":
                return ra == rb
            if op == "FCmpOrdGt":
                return ra > rb
            if op == "FCmpOrdGe":
                return ra >= rb
            if op == "FCmpOrdLt":
                return ra < rb
            if op == "FCmpOrdLe":
                return ra <= rb
            if op in ("FCmpOrdNe", "FCmpUnordNe"):
                return ra != rb
            if op == "FCmpTotal":
                return ra <= rb
            return _hw_uf(f"realcmp_{op}", REAL_SORT, REAL_SORT, z3.BoolSort())(ra, rb)
        if self.fp_mode == "uf":
            return _hw_uf(f"fpcmp_{op}", BV32, BV32, z3.BoolSort())(a, b)
        fa, fb = bv32_to_fp32(a), bv32_to_fp32(b)
        if op == "FCmpOrdEq":
            return z3.fpEQ(fa, fb)
        if op == "FCmpOrdGt":
            return z3.fpGT(fa, fb)
        if op == "FCmpOrdGe":
            return z3.fpGEQ(fa, fb)
        if op == "FCmpOrdLt":
            return z3.fpLT(fa, fb)
        if op == "FCmpOrdLe":
            return z3.fpLEQ(fa, fb)
        if op == "FCmpOrdNe":
            return z3.Or(z3.fpLT(fa, fb), z3.fpGT(fa, fb))
        if op == "FCmpUnordNe":
            return z3.Not(z3.fpEQ(fa, fb))
        if op == "FCmpTotal":
            return _fp_total_le_bv(32, a, b)
        return _hw_uf(f"fpcmp_{op}", BV32, BV32, z3.BoolSort())(a, b)

    def _eval_node(self, node: Dict[str, Any], st: BlockEffectState, reach: z3.BoolRef) -> Z3Val:
        op = node["op"]
        vt = node.get("type", "U32")
        args = [self.values.get(int(aid), bv32(0)) for aid in node.get("args", [])]
        imm_u32 = int(node.get("imm_u32", 0))
        imm_u32_2 = int(node.get("imm_u32_2", 0))
        imm_u64 = int(node.get("imm_u64", 0))
        aux_bool = bool(node.get("aux_bool", False))
        aux_str = node.get("aux_str", "")

        if op in ("ConstU32", "ConstS32", "ConstF32"):
            return bv32(imm_u32)
        if op == "ConstU64":
            return bv64(imm_u64)
        if op == "ConstBool":
            return z3.BoolVal(imm_u32 != 0 or aux_bool)
        if op in ("Nop", "Copy", "Bitcast"):
            if not args:
                return bv32(0)
            v = args[0]
            if vt in ("U64", "S64"):
                return _as_bv64(v)
            if vt == "Bool":
                return _as_bool(v)
            if vt in ("U32", "S32", "F32") and not isinstance(v, tuple):
                return _as_bv32(v)
            return v

        # Vector / Tuple / 64-bit Pack & Split
        if op in ("VecConstruct2", "VecConstruct3", "VecConstruct4"):
            return tuple(args)
        if op == "VecExtract":
            v = args[0]
            return v[imm_u32] if isinstance(v, tuple) and imm_u32 < len(v) else _as_bv32(v)
        if op == "VecShuffle":
            a, b = args[0], args[1]
            comb = (list(a) if isinstance(a, tuple) else [a]) + (list(b) if isinstance(b, tuple) else [b])
            return (comb[imm_u32], comb[imm_u32_2])
        if op in ("Pack64", "ImageTexelAddr"):
            lo, hi = _as_bv32(args[0]), _as_bv32(args[1])
            if isinstance(lo, z3.BitVecNumRef) and isinstance(hi, z3.BitVecNumRef):
                return bv64((hi.as_long() << 32) | lo.as_long())
            return z3.Concat(hi, lo)
        if op == "Split64":
            v64 = _as_bv64(args[0])
            if isinstance(v64, z3.BitVecNumRef):
                n = v64.as_long()
                return (bv32(n & 0xFFFFFFFF), bv32((n >> 32) & 0xFFFFFFFF))
            return (z3.Extract(31, 0, v64), z3.Extract(63, 32, v64))
        if op == "TupleExtract":
            tup = args[0]
            return tup[imm_u32] if isinstance(tup, tuple) and imm_u32 < len(tup) else tup
        if op == "UnpackHalf2x16":
            u = _as_bv32(args[0])
            return (f16_bv_to_f32_bv(z3.Extract(15, 0, u)), f16_bv_to_f32_bv(z3.Extract(31, 16, u)))
        if op == "PackHalf2x16":
            v2 = args[0]
            if isinstance(v2, tuple) and len(v2) >= 2:
                f0 = f32_bv_to_f16_bv(_as_bv32(v2[0]))
                f1 = f32_bv_to_f16_bv(_as_bv32(v2[1]))
                return z3.Concat(f1, f0)
            s = f32_bv_to_f16_bv(_as_bv32(v2))
            return z3.Concat(s, s)

        # 32-bit Integer & Bitwise
        if op == "BitNot":
            return ~_as_bv32(args[0])
        if op == "SNeg":
            return -_as_bv32(args[0])
        if op == "SAbs":
            x = _as_bv32(args[0])
            return z3.If(x < bv32(0), -x, x)
        if op == "Clz":
            return _clz_bv(32, _as_bv32(args[0]), mask_zero=False)
        if op == "BitCount":
            return _popcount_bv32(_as_bv32(args[0]))
        if op == "BitReverse":
            return _bitrev_bv32(_as_bv32(args[0]))
        if op in ("IAdd", "ISub", "IMul", "BitAnd", "BitOr", "BitXor", "Shl", "ShrU", "ShrS", "RotL", "RotR", "UDiv", "SDiv", "UMod", "SRem", "SMod"):
            a, b = _as_bv32(args[0]), _as_bv32(args[1])
            if op == "IAdd": return a + b
            if op == "ISub": return a - b
            if op == "IMul": return a * b
            if op == "BitAnd": return a & b
            if op == "BitOr": return a | b
            if op == "BitXor": return a ^ b
            if op == "Shl": return a << (b & bv32(31))
            if op == "ShrU": return z3.LShR(a, b & bv32(31))
            if op == "ShrS": return a >> (b & bv32(31))
            if op == "RotL": return z3.RotateLeft(a, b & bv32(31))
            if op == "RotR": return z3.RotateRight(a, b & bv32(31))
            if op == "UDiv": return z3.UDiv(a, b)
            if op == "SDiv": return a / b
            if op == "UMod": return z3.URem(a, b)
            if op == "SRem": return z3.SRem(a, b)
            if op == "SMod":
                rem = z3.SRem(a, b)
                return z3.If(z3.And(rem != bv32(0), (rem ^ b) < bv32(0)), rem + b, rem)
        if op in ("BfeU", "BfeS"):
            x, off, w = _as_bv32(args[0]), _as_bv32(args[1]), _as_bv32(args[2])
            if isinstance(off, z3.BitVecNumRef) and isinstance(w, z3.BitVecNumRef):
                o_i, w_i = off.as_long(), w.as_long()
                if w_i == 0:
                    return bv32(0)
                hi_i = min(31, o_i + w_i - 1)
                sl = z3.Extract(hi_i, o_i, x)
                ext = 32 - sl.size()
                if ext == 0:
                    return sl
                return z3.SignExt(ext, sl) if op == "BfeS" else z3.ZeroExt(ext, sl)
            if op == "BfeS":
                sh = bv32(32) - w
                extracted = (x << (sh - off)) >> sh
                return z3.If(w == bv32(0), bv32(0), z3.If(z3.UGE(w, bv32(32)), x, extracted))
            mask = (bv32(1) << (w & bv32(31))) - bv32(1)
            extracted = z3.LShR(x, off & bv32(31)) & mask
            return z3.If(w == bv32(0), bv32(0), z3.If(z3.UGE(w, bv32(32)), x, extracted))
        if op == "Bfi":
            base, ins_v, off, w = _as_bv32(args[0]), _as_bv32(args[1]), _as_bv32(args[2]), _as_bv32(args[3])
            if isinstance(off, z3.BitVecNumRef) and isinstance(w, z3.BitVecNumRef):
                o_i, w_i = off.as_long(), w.as_long()
                if w_i >= 32: return ins_v
                if w_i == 0: return base
                mask_v = ((1 << w_i) - 1) << o_i
                return (base & bv32(~mask_v)) | ((ins_v << o_i) & bv32(mask_v))
            mask = ((bv32(1) << (w & bv32(31))) - bv32(1)) << (off & bv32(31))
            res_bfi = (base & ~mask) | ((ins_v << (off & bv32(31))) & mask)
            return z3.If(w == bv32(0), base, z3.If(z3.UGE(w, bv32(32)), ins_v, res_bfi))
        if op in ("IDot4S", "IDot4SU", "IDot4U"):
            a32, b32 = _as_bv32(args[0]), _as_bv32(args[1])
            acc = bv32(0)
            for i in range(4):
                ba = z3.Extract(8 * i + 7, 8 * i, a32)
                bb = z3.Extract(8 * i + 7, 8 * i, b32)
                ea = z3.SignExt(24, ba) if op in ("IDot4S", "IDot4SU") else z3.ZeroExt(24, ba)
                eb = z3.SignExt(24, bb) if op == "IDot4S" else z3.ZeroExt(24, bb)
                acc = acc + (ea * eb)
            return acc

        # 64-bit Integer & Bitwise
        if op == "BitNot64":
            return ~_as_bv64(args[0])
        if op in ("IAdd64", "ISub64", "IMul64", "BitAnd64", "BitOr64", "BitXor64", "Shl64", "ShrU64", "ShrS64"):
            a64, b64 = _as_bv64(args[0]), _as_bv64(args[1])
            if isinstance(a64, z3.BitVecNumRef) and isinstance(b64, z3.BitVecNumRef):
                av, bv_ = a64.as_long(), b64.as_long()
                if op == "IAdd64": return bv64((av + bv_) & 0xFFFFFFFFFFFFFFFF)
                if op == "ISub64": return bv64((av - bv_) & 0xFFFFFFFFFFFFFFFF)
                if op == "IMul64": return bv64((av * bv_) & 0xFFFFFFFFFFFFFFFF)
            if op == "IAdd64": return a64 + b64
            if op == "ISub64": return a64 - b64
            if op == "IMul64": return a64 * b64
            if op == "BitAnd64": return a64 & b64
            if op == "BitOr64": return a64 | b64
            if op == "BitXor64": return a64 ^ b64
            if op == "Shl64": return a64 << (b64 & bv64(63))
            if op == "ShrU64": return z3.LShR(a64, b64 & bv64(63))
            if op == "ShrS64": return a64 >> (b64 & bv64(63))

        # Integer Comparisons, Logical, Select
        if op.startswith("ICmp"):
            a, b = _as_bv32(args[0]), _as_bv32(args[1])
            if op == "ICmpEq": return a == b
            if op == "ICmpNe": return a != b
            if op == "ICmpSLt": return a < b
            if op == "ICmpSLe": return a <= b
            if op == "ICmpSGt": return a > b
            if op == "ICmpSGe": return a >= b
            if op == "ICmpULt": return z3.ULT(a, b)
            if op == "ICmpULe": return z3.ULE(a, b)
            if op == "ICmpUGt": return z3.UGT(a, b)
            if op == "ICmpUGe": return z3.UGE(a, b)
        if op == "LogicalAnd":
            return _z3_and(_as_bool(args[0]), _as_bool(args[1]))
        if op == "LogicalOr":
            return _z3_or(_as_bool(args[0]), _as_bool(args[1]))
        if op == "LogicalNot":
            return _z3_not(_as_bool(args[0]))
        if op == "Select":
            return _z3_if(_as_bool(args[0]), args[1], args[2])

        # Floating-point operations (lifted elementwise for V2F32 / v2f16 vectors)
        if op in ("FAbs", "FNeg", "Floor", "Ceil", "Trunc", "RoundEven", "Sqrt", "InverseSqrt",
                  "Sin", "Cos", "Exp", "Exp2", "Log", "Log2", "Atan", "Asin", "Acos", "Sinh", "Cosh", "Tanh", "DPdx", "DPdy", "Fwidth", "FrexpExp", "FrexpSignificand"):
            return _eval_elementwise(vt, args, lambda x: self._fp_unary(op, _as_bv32(x)))
        if op in ("FAdd", "FSub", "FMul", "FDiv", "FMin", "FMax", "NMin", "NMax", "Pow", "Ldexp", "Atan2"):
            return _eval_elementwise(vt, args, lambda x, y: self._fp_binary(op, _as_bv32(x), _as_bv32(y)))
        if op == "FClamp":
            def _fclamp(x: Z3Val, lo: Z3Val, hi: Z3Val) -> z3.BitVecRef:
                bx, blo, bhi = _as_bv32(x), _as_bv32(lo), _as_bv32(hi)
                if self.fp_mode == "real":
                    rx, rlo, rhi = _bv32_to_real(bx), _bv32_to_real(blo), _bv32_to_real(bhi)
                    return _real_to_bv32(z3.If(rx < rlo, rlo, z3.If(rx > rhi, rhi, rx)))
                if self.fp_mode == "uf":
                    return _hw_uf("fp_FClamp", BV32, BV32, BV32, BV32)(bx, blo, bhi)
                return fp32_to_bv32(z3.fpMin(z3.fpMax(bv32_to_fp32(bx), bv32_to_fp32(blo)), bv32_to_fp32(bhi)))
            return _eval_elementwise(vt, args, _fclamp)
        if op == "Fma":
            def _ffma(a: Z3Val, b: Z3Val, c: Z3Val) -> z3.BitVecRef:
                ba, bb, bc = _as_bv32(a), _as_bv32(b), _as_bv32(c)
                if self.fp_mode == "real":
                    return _real_to_bv32(_bv32_to_real(ba) * _bv32_to_real(bb) + _bv32_to_real(bc))
                if self.fp_mode == "uf":
                    return _hw_uf("fp_Fma", BV32, BV32, BV32, BV32)(ba, bb, bc)
                # Exact IEEE-754 RNE identity: FMA(x, 1.0, +0.0) == (+0.0 if x == -0.0; qNaN if isNaN(x); else x).
                bc_is_pos_zero = isinstance(bc, z3.BitVecNumRef) and bc.as_long() == 0
                if bc_is_pos_zero:
                    if isinstance(bb, z3.BitVecNumRef) and bb.as_long() == 0x3F800000:
                        return z3.If(z3.fpIsNaN(bv32_to_fp32(ba)), fp32_to_bv32(z3.fpNaN(FP32_SORT)),
                                     z3.If(ba == bv32(0x80000000), bv32(0), ba))
                    if isinstance(ba, z3.BitVecNumRef) and ba.as_long() == 0x3F800000:
                        return z3.If(z3.fpIsNaN(bv32_to_fp32(bb)), fp32_to_bv32(z3.fpNaN(FP32_SORT)),
                                     z3.If(bb == bv32(0x80000000), bv32(0), bb))
                return fp32_to_bv32(z3.fpFMA(z3.RNE(), bv32_to_fp32(ba), bv32_to_fp32(bb), bv32_to_fp32(bc)))
            return _eval_elementwise(vt, args, _ffma)
        if op == "IsNan":
            x = _as_bv32(args[0])
            return z3.fpIsNaN(bv32_to_fp32(x)) if self.fp_mode == "exact" else _hw_uf("fp_IsNan", BV32, z3.BoolSort())(x)
        if op == "IsInf":
            x = _as_bv32(args[0])
            return z3.fpIsInf(bv32_to_fp32(x)) if self.fp_mode == "exact" else _hw_uf("fp_IsInf", BV32, z3.BoolSort())(x)
        if op.startswith("FCmp"):
            return self._fp_cmp(op, _as_bv32(args[0]), _as_bv32(args[1]))

        # Conversions
        if op == "ConvertFToS":
            x = _as_bv32(args[0])
            if self.fp_mode == "real":
                return _hw_uf("real_FToS", REAL_SORT, BV32)(_bv32_to_real(x))
            return _fp_to_int_sat(bv32_to_fp32(x), 32, True, z3.RTZ()) if self.fp_mode == "exact" else _hw_uf("fp_FToS", BV32, BV32)(x)
        if op == "ConvertFToU":
            x = _as_bv32(args[0])
            if self.fp_mode == "real":
                return _hw_uf("real_FToU", REAL_SORT, BV32)(_bv32_to_real(x))
            return _fp_to_int_sat(bv32_to_fp32(x), 32, False, z3.RTZ()) if self.fp_mode == "exact" else _hw_uf("fp_FToU", BV32, BV32)(x)
        if op == "ConvertSToF":
            x = _as_bv32(args[0])
            if self.fp_mode == "real":
                if isinstance(x, z3.BitVecNumRef):
                    return _real_to_bv32(z3.RealVal(x.as_signed_long()))
                return _real_to_bv32(_hw_uf("real_SToF", BV32, REAL_SORT)(x))
            return fp32_to_bv32(z3.fpSignedToFP(z3.RNE(), x, FP32_SORT)) if self.fp_mode == "exact" else _hw_uf("fp_SToF", BV32, BV32)(x)
        if op == "ConvertUToF":
            x = _as_bv32(args[0])
            if self.fp_mode == "real":
                if isinstance(x, z3.BitVecNumRef):
                    return _real_to_bv32(z3.RealVal(x.as_long()))
                return _real_to_bv32(_hw_uf("real_UToF", BV32, REAL_SORT)(x))
            return fp32_to_bv32(z3.fpUnsignedToFP(z3.RNE(), x, FP32_SORT)) if self.fp_mode == "exact" else _hw_uf("fp_UToF", BV32, BV32)(x)
        if op == "SConvert32To64":
            return z3.SignExt(32, _as_bv32(args[0]))
        if op == "UConvert32To64":
            return z3.ZeroExt(32, _as_bv32(args[0]))
        if op == "UConvert64To32":
            return z3.Extract(31, 0, _as_bv64(args[0]))

        # Resources, Sideband, FAU, Builtins, Memory, Outputs
        if op == "LoadFau":
            return self.env.load_fau_word(imm_u32 & 63)
        if op == "LoadSidebandU64":
            if aux_str and (aux_str.startswith("ssbo:") or aux_str.startswith("ubo:") or aux_str == "push_const" or aux_str.startswith("workgroup:")):
                return bv64(self.env.get_buf_base(aux_str))
            if imm_u32 == 6:
                return bv64(self.env.get_buf_base("tls_private"))
            if imm_u32 == 7:
                return bv64(self.env.get_buf_base("wls_shared"))
            return _hw_uf(f"sideband_u64_{imm_u32}", BV32, BV64)(bv32(imm_u32))
        if op == "LoadSidebandU32":
            return _hw_uf(f"sideband_u32_{imm_u32}", BV32, BV32)(bv32(imm_u32_2))
        if op == "LoadBuiltinInput":
            return self.env.get_builtin(imm_u32, vt)
        if op == "LoadVertAttr":
            loc = self.env.sinp_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
            return self.env.get_input_vec4("vert_attr", loc)
        if op == "LoadFragVarying":
            loc = self.env.sinp_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
            return self.env.get_input_vec4("frag_var", loc)
        if op == "ImageQuerySize":
            lod = _as_bv32(args[0]) if args else bv32(0)
            qkey = (aux_str or "img").replace(":", "_")
            return _hw_uf(f"imgq_{qkey}_c{imm_u32}", BV32, BV32)(lod)

        if op in ("TexSample", "TexSampleLod", "TexSampleDref", "TexGather", "TexFetch", "ImageQueryLod"):
            uv = args[0] if args else (bv32(0), bv32(0))
            u0 = _as_bv32(uv[0]) if isinstance(uv, tuple) else _as_bv32(uv)
            u1 = _as_bv32(uv[1]) if isinstance(uv, tuple) and len(uv) > 1 else bv32(0)
            tkey = aux_str or f"tex_{imm_u32}"
            lanes = tuple(_hw_uf(f"tex_{tkey}_c{c}", BV32, BV32, BV32)(u0, u1) for c in range(4))
            return (lanes, args[-1] if args else bv32(0))

        if op == "PsbLoad":
            addr = _as_bv64(args[0])
            eff_hint = aux_str
            if aux_str.startswith("sym:"):
                bkey_hint = aux_str[4:].split("[", 1)[0]
                if self.env.is_shared_sym_bkey(bkey_hint):
                    _, _, off32 = self.env.resolve_addr(addr, hint_key=bkey_hint)
                    if aux_str in st.sym_buf_stores:
                        w_cond, w_val, w_off = st.sym_buf_stores[aux_str]
                        val = _z3_if(_z3_and(w_cond, off32 == w_off), w_val, self.env.get_sym_buf_slot(aux_str, off32))
                    else:
                        val = self.env.get_sym_buf_slot(aux_str, off32)
                    return (val, args[1] if len(args) > 1 else bv32(0))
                eff_hint = bkey_hint
            bkey, const_off, off32 = self.env.resolve_addr(addr, hint_key=eff_hint)
            if bkey == "wls_shared" and self.env.unique_workgroup_key is not None:
                bkey = self.env.unique_workgroup_key
            if bkey is not None:
                if const_off is not None and (bkey, const_off) in st.const_stores:
                    w_cond, w_val = st.const_stores[(bkey, const_off)]
                    mem_v = z3.Select(st.buf_mems.get(bkey, self.env.get_buf_mem(bkey)), off32)
                    val = _z3_if(w_cond, w_val, mem_v)
                else:
                    val = z3.Select(st.buf_mems.get(bkey, self.env.get_buf_mem(bkey)), off32)
            else:
                val = z3.Select(st.fallback_mem, addr)
            return (val, args[1] if len(args) > 1 else bv32(0))

        if op == "PsbStore":
            addr = _as_bv64(args[0])
            val = _as_bv32(args[1])
            store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
            nid = int(node["id"])
            eff_hint = aux_str
            if aux_str.startswith("sym:"):
                bkey_hint = aux_str[4:].split("[", 1)[0]
                if self.env.is_shared_sym_bkey(bkey_hint):
                    _, _, off32 = self.env.resolve_addr(addr, hint_key=bkey_hint)
                    prev_w, prev_v, prev_o = st.sym_buf_stores.get(aux_str, (z3.BoolVal(False), self.env.get_sym_buf_slot(aux_str, off32), off32))
                    st.sym_buf_stores[aux_str] = (_z3_or(prev_w, store_mask), _z3_if(store_mask, val, prev_v), _z3_if(store_mask, off32, prev_o))
                    st.output_pcs.setdefault(aux_str, set()).add(nid)
                    return args[2] if len(args) > 2 else bv32(0)
                eff_hint = bkey_hint
            # Normalize OpImageWrite (emitted as PsbStore on img:* with packed xy + c*4) into image_mems
            if aux_str.startswith("img:"):
                ikey = aux_str
                ch_idx = 0
                xy_addr = addr
                if z3.is_app(addr) and addr.decl().kind() == z3.Z3_OP_BADD and len(addr.children()) == 2:
                    c0, c1 = addr.children()
                    if isinstance(c1, z3.BitVecNumRef) and c1.as_long() in (4, 8, 12):
                        ch_idx = c1.as_long() // 4
                        xy_addr = c0
                cur_im = list(st.image_mems.get(ikey, self.env.get_img_mem(ikey)))
                cur_im[ch_idx] = (
                    z3.If(store_mask, z3.Store(cur_im[ch_idx], xy_addr, val), cur_im[ch_idx])
                    if not z3.is_true(store_mask)
                    else z3.Store(cur_im[ch_idx], xy_addr, val)
                )
                st.image_mems[ikey] = tuple(cur_im)
                st.output_pcs.setdefault(f"img:{ikey}", set()).add(nid)
                return args[2] if len(args) > 2 else bv32(0)

            bkey, const_off, off32 = self.env.resolve_addr(addr, hint_key=eff_hint)
            if bkey == "wls_shared" and self.env.unique_workgroup_key is not None:
                bkey = self.env.unique_workgroup_key
            if bkey is not None:
                cur_mem = st.buf_mems.get(bkey, self.env.get_buf_mem(bkey))
                st.buf_mems[bkey] = (
                    z3.If(store_mask, z3.Store(cur_mem, off32, val), cur_mem)
                    if not z3.is_true(store_mask)
                    else z3.Store(cur_mem, off32, val)
                )
                if const_off is not None:
                    prev_w, prev_v = st.const_stores.get((bkey, const_off), (z3.BoolVal(False), z3.Select(cur_mem, off32)))
                    st.const_stores[(bkey, const_off)] = (_z3_or(prev_w, store_mask), _z3_if(store_mask, val, prev_v))
                    st.output_pcs.setdefault(f"{bkey}+0x{const_off:x}", set()).add(nid)
                else:
                    st.output_pcs.setdefault(f"{bkey}:dyn", set()).add(nid)
            else:
                st.fallback_mem = z3.If(store_mask, z3.Store(st.fallback_mem, addr, val), st.fallback_mem)
                st.output_pcs.setdefault("fallback_mem", set()).add(nid)
            return args[2] if len(args) > 2 else bv32(0)

        if op == "Kill":
            cond = _as_bool(args[0]) if (aux_bool and len(args) >= 2) else z3.BoolVal(True)
            is_demote = bool(node.get("aux_bool2", False))
            eff = _z3_and(reach, cond)
            if is_demote:
                st.demoted = _z3_or(st.demoted, eff)
            else:
                st.terminated = _z3_or(st.terminated, eff)
            st.discarded = _z3_or(st.terminated, st.demoted)
            st.output_pcs.setdefault("discarded", set()).add(int(node["id"]))
            return bv32(0)

        if op == "StoreFragColor":
            col = args[0]
            store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
            vec4 = tuple(_as_bv32(col[c]) if isinstance(col, tuple) and c < len(col) else bv32(0) for c in range(4))
            prev = st.frag_colors.get(imm_u32, (bv32(0),) * 4)
            st.frag_colors[imm_u32] = tuple(_z3_if(store_mask, n, p) for n, p in zip(vec4, prev))
            st.output_pcs.setdefault(f"frag_color[{imm_u32}]", set()).add(int(node["id"]))
            return args[-1] if args else bv32(0)

        if op == "StoreBuiltinOutput":
            payload = args[0]
            store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
            if imm_u32 in st.builtin_outputs:
                st.builtin_outputs[imm_u32] = _z3_if(store_mask, payload, st.builtin_outputs[imm_u32])
            else:
                st.builtin_outputs[imm_u32] = payload
            st.output_pcs.setdefault(f"builtin_out[{imm_u32}]", set()).add(int(node["id"]))
            return args[-1] if args else bv32(0)

        if op == "StoreVertVarying":
            loc = self.env.sout_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
            store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
            vec = args[0]
            vec4 = tuple(_as_bv32(vec[c]) if isinstance(vec, tuple) and c < len(vec) else bv32(0) for c in range(4))
            prev = st.vert_varyings.get(loc, (bv32(0),) * 4)
            st.vert_varyings[loc] = tuple(_z3_if(store_mask, n, p) for n, p in zip(vec4, prev))
            st.output_pcs.setdefault(f"vert_var[{loc}]", set()).add(int(node["id"]))
            return args[-1] if args else bv32(0)

        if op == "StoreTexel":
            coord = args[0]
            store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
            texel = args[1] if len(args) > 1 else (bv32(0),) * 4
            cx = _as_bv32(coord[0]) if isinstance(coord, tuple) and len(coord) > 0 else _as_bv32(coord)
            cy = _as_bv32(coord[1]) if isinstance(coord, tuple) and len(coord) > 1 else bv32(0)
            xy = z3.Concat(cy, cx)
            vec4 = tuple(_as_bv32(texel[c]) if isinstance(texel, tuple) and c < len(texel) else bv32(0) for c in range(4))
            ikey = aux_str or f"img_{imm_u32}"
            cur_im = st.image_mems.get(ikey, self.env.get_img_mem(ikey))
            updated = tuple(
                z3.If(store_mask, z3.Store(cur_im[c], xy, vec4[c]), cur_im[c]) if not z3.is_true(store_mask)
                else z3.Store(cur_im[c], xy, vec4[c])
                for c in range(4)
            )
            st.image_mems[ikey] = updated
            st.output_pcs.setdefault(f"img:{ikey}", set()).add(int(node["id"]))
            return args[-1] if args else bv32(0)

        if op in ("AtomicRmw", "AtomicExchange", "AtomicCmpXchg"):
            addr = _as_bv64(args[0])
            arg = _as_bv32(args[1]) if len(args) > 1 else bv32(0)
            cmp_val = _as_bv32(args[2]) if (op == "AtomicCmpXchg" and len(args) > 2) else bv32(0)
            store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
            op_tag = aux_str.lower() if aux_str else op.lower()
            bkey, const_off, off32 = self.env.resolve_addr(addr, hint_key="")
            if bkey == "wls_shared" and self.env.unique_workgroup_key is not None:
                bkey = self.env.unique_workgroup_key
            nid = int(node["id"])
            if bkey is not None:
                cur_mem = st.buf_mems.get(bkey, self.env.get_buf_mem(bkey))
                old_val = z3.Select(cur_mem, off32)
            else:
                old_val = z3.Select(st.fallback_mem, addr)

            # Compute exact single-invocation RMW post-state value
            if op == "AtomicExchange":
                new_val = arg
            elif op == "AtomicCmpXchg":
                new_val = z3.If(old_val == cmp_val, arg, old_val)
            elif op_tag in ("aadd", "add"):
                new_val = old_val + arg
            elif op_tag in ("asub", "sub"):
                new_val = old_val - arg
            elif op_tag in ("aand", "and"):
                new_val = old_val & arg
            elif op_tag in ("aor", "or"):
                new_val = old_val | arg
            elif op_tag in ("axor", "xor"):
                new_val = old_val ^ arg
            elif op_tag in ("asmin", "smin"):
                new_val = z3.If(old_val < arg, old_val, arg)
            elif op_tag in ("aumin", "umin"):
                new_val = z3.If(z3.ULT(old_val, arg), old_val, arg)
            elif op_tag in ("asmax", "smax"):
                new_val = z3.If(old_val > arg, old_val, arg)
            elif op_tag in ("aumax", "umax"):
                new_val = z3.If(z3.UGT(old_val, arg), old_val, arg)
            else:
                new_val = _hw_uf(f"atomic_post_{op_tag}", BV32, BV32, BV32)(old_val, arg)

            if bkey is not None:
                cur_mem = st.buf_mems.get(bkey, self.env.get_buf_mem(bkey))
                st.buf_mems[bkey] = (
                    z3.If(store_mask, z3.Store(cur_mem, off32, new_val), cur_mem)
                    if not z3.is_true(store_mask)
                    else z3.Store(cur_mem, off32, new_val)
                )
                st.atomic_events.setdefault(bkey, []).append((store_mask, op_tag, off32, arg, cmp_val))
                st.output_pcs.setdefault(f"atomic:{bkey}", set()).add(nid)
                st.output_pcs.setdefault(f"{bkey}:dyn", set()).add(nid)
            else:
                st.fallback_mem = z3.If(store_mask, z3.Store(st.fallback_mem, addr, new_val), st.fallback_mem)
                st.atomic_events.setdefault("fallback_mem", []).append((store_mask, op_tag, addr, arg, cmp_val))
                st.output_pcs.setdefault("atomic:fallback_mem", set()).add(nid)
                st.output_pcs.setdefault("fallback_mem", set()).add(nid)
            return (old_val, args[-1] if args else bv32(0))

        if op in ("ControlBarrier", "Branch", "BranchCond", "Return"):
            return args[-1] if args else bv32(0)

        if op == "SubgroupBallot":
            return _hw_uf("SubgroupBallot", BV32, z3.BitVecSort(5), BV32)(_as_bv32(args[0]), self.env.lane_id)
        if op == "SubgroupShuffle":
            return _hw_uf("SubgroupShuffle", BV32, BV32, z3.BitVecSort(5), BV32)(_as_bv32(args[0]), _as_bv32(args[1]), self.env.lane_id)

        return bv32(0)

    def evaluate(self) -> BlockEffectState:
        """Evaluate all blocks in topological order using DAG path-merging (with configurable bounded loop unrolling)."""
        init_state = BlockEffectState(
            discarded=z3.BoolVal(False),
            terminated=z3.BoolVal(False),
            demoted=z3.BoolVal(False),
            frag_colors={},
            builtin_outputs={},
            vert_varyings={},
            const_stores={},
            sym_buf_stores={},
            buf_mems={},
            image_mems={},
            fallback_mem=self.env.fallback_mem,
            atomic_events={},
            output_pcs={},
        )
        if not self.blocks:
            return init_state

        blk_by_id: Dict[int, Dict[str, Any]] = {int(b["id"]): b for b in self.blocks}
        visited: Set[int] = set()
        postorder: List[int] = []
        dfs_stack: List[Tuple[int, int]] = [(0, 0)] if 0 in blk_by_id else []
        while dfs_stack:
            u, step = dfs_stack.pop()
            if step == 0:
                if u in visited:
                    continue
                visited.add(u)
                dfs_stack.append((u, 1))
                bu = blk_by_id[u]
                succs: List[int] = []
                bt_u = int(bu.get("branch_target", -1))
                ft_u = int(bu.get("fallthrough", -1))
                if bool(bu.get("is_cond", False)):
                    if bt_u >= 0: succs.append(bt_u)
                    if ft_u >= 0 and ft_u != bt_u: succs.append(ft_u)
                elif bt_u >= 0:
                    succs.append(bt_u)
                elif ft_u >= 0:
                    succs.append(ft_u)
                for s in reversed(succs):
                    if s in blk_by_id and s not in visited:
                        dfs_stack.append((s, 0))
            else:
                postorder.append(u)
        rpo_ids = list(reversed(postorder))
        rpo_index: Dict[int, int] = {bid: idx for idx, bid in enumerate(rpo_ids)}
        rpo_blocks = [blk_by_id[bid] for bid in rpo_ids]

        blk_reach: Dict[int, z3.BoolRef] = {}
        blk_out_state: Dict[int, BlockEffectState] = {}
        edge_conds: Dict[int, List[Tuple[int, z3.BoolRef]]] = {b["id"]: [] for b in self.blocks}
        exit_states: List[Tuple[z3.BoolRef, BlockEffectState]] = []

        def _eval_block_slice(block_list: List[Dict[str, Any]], allow_backedges: bool) -> List[Tuple[int, int, z3.BoolRef]]:
            backedges: List[Tuple[int, int, z3.BoolRef]] = []
            for blk in block_list:
                bid = int(blk["id"])
                b_rpo = rpo_index.get(bid, bid)
                if bid == 0 and not allow_backedges:
                    reach = z3.BoolVal(True)
                    cur_st = _clone_state(init_state)
                else:
                    inc_edges = [(p, c) for (p, c) in edge_conds.get(bid, []) if p in blk_out_state]
                    if not inc_edges:
                        blk_reach[bid] = z3.BoolVal(False)
                        blk_out_state[bid] = _clone_state(init_state)
                        continue
                    reach = z3.BoolVal(False)
                    for _, c in inc_edges:
                        reach = _z3_or(reach, c)
                    cur_st = _merge_states([(c, blk_out_state[p]) for (p, c) in inc_edges], self.env)

                blk_reach[bid] = reach

                for p in blk.get("preds", []):
                    ip = int(p)
                    if ip in rpo_index and rpo_index[ip] >= b_rpo:
                        self.has_loops = True

                inc_cond_map = {p: c for (p, c) in edge_conds.get(bid, [])}
                for pid in blk.get("phis", []):
                    pnode = self.nodes_by_id.get(int(pid))
                    if not pnode:
                        continue
                    incomings = pnode.get("phi", [])
                    fwd = [(int(p), int(v)) for p, v in incomings if int(p) in blk_out_state and int(v) in self.values]
                    if not fwd:
                        vt = pnode.get("type", "U32")
                        self.values[int(pid)] = z3.BoolVal(False) if vt == "Bool" else bv32(0)
                        continue
                    if len(fwd) == 1:
                        self.values[int(pid)] = self.values[fwd[0][1]]
                    else:
                        acc = self.values[fwd[0][1]]
                        for pred_b, val_id in fwd[1:]:
                            ec = inc_cond_map.get(pred_b, z3.BoolVal(False))
                            acc = _z3_if(ec, self.values[val_id], acc)
                        self.values[int(pid)] = acc

                for nid in blk.get("nodes", []):
                    node = self.nodes_by_id.get(int(nid))
                    if not node:
                        continue
                    self.values[int(nid)] = self._eval_node(node, cur_st, reach)

                blk_out_state[bid] = cur_st

                # Hard OpKill / OpTerminateInvocation suppresses outgoing control-flow reachability;
                # OpDemoteToHelperInvocation continues executing with stores masked.
                cf_reach = _z3_and(reach, _z3_not(cur_st.terminated))

                bt = int(blk.get("branch_target", -1))
                ft = int(blk.get("fallthrough", -1))
                is_cond = bool(blk.get("is_cond", False))
                is_end = bool(blk.get("is_end", False))
                is_discard = bool(blk.get("is_discard", False))

                bt_rpo = rpo_index.get(bt, -1)
                ft_rpo = rpo_index.get(ft, -1)
                if is_cond and bt >= 0 and ft >= 0:
                    bcond_id = int(blk.get("branch_cond", 0))
                    bcond = _as_bool(self.values.get(bcond_id, z3.BoolVal(False)))
                    t_cond = _z3_and(cf_reach, bcond)
                    f_cond = _z3_and(cf_reach, _z3_not(bcond))
                    if bt_rpo > b_rpo:
                        edge_conds.setdefault(bt, []).append((bid, t_cond))
                    else:
                        self.has_loops = True
                        if not z3.is_false(t_cond):
                            backedges.append((bid, bt, t_cond))
                    if ft_rpo > b_rpo:
                        edge_conds.setdefault(ft, []).append((bid, f_cond))
                    else:
                        self.has_loops = True
                        if not z3.is_false(f_cond):
                            backedges.append((bid, ft, f_cond))
                elif bt >= 0 and bt_rpo > b_rpo:
                    edge_conds.setdefault(bt, []).append((bid, cf_reach))
                elif ft >= 0 and ft_rpo > b_rpo:
                    edge_conds.setdefault(ft, []).append((bid, cf_reach))
                elif bt >= 0 and bt_rpo <= b_rpo:
                    self.has_loops = True
                    if not z3.is_false(cf_reach):
                        backedges.append((bid, bt, cf_reach))

                if is_end or is_discard or (bt < 0 and ft < 0):
                    exit_states.append((reach, cur_st))
            return backedges

        backedges = _eval_block_slice(rpo_blocks, allow_backedges=False)
        self.max_unroll_depth = 1 if self.has_loops else 0

        # Configurable bounded loop unrolling when max_unroll > 1
        unroll_iter = 1
        while backedges and unroll_iter < self.max_unroll:
            unroll_iter += 1
            self.max_unroll_depth = unroll_iter
            min_hdr_rpo = min(rpo_index.get(hdr, 0) for _, hdr, _ in backedges)
            for src_b, hdr, bcond in backedges:
                edge_conds.setdefault(hdr, []).append((src_b, bcond))
            loop_slice = [b for b in rpo_blocks if rpo_index[int(b["id"])] >= min_hdr_rpo]
            exit_states.clear()
            backedges = _eval_block_slice(loop_slice, allow_backedges=True)

        self.block_reach = blk_reach
        if not exit_states:
            return blk_out_state.get(len(self.blocks) - 1, init_state)
        return _merge_states(exit_states, self.env)



MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def _u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def _f32_to_u32(f: float) -> int:
    if math.isnan(f):
        return 0x7FC00000
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return 0x7F800000 if f > 0 else 0xFF800000


def _s32(u: int) -> int:
    u &= MASK32
    return u - 0x100000000 if u & 0x80000000 else u


def _s64(u: int) -> int:
    u &= MASK64
    return u - 0x10000000000000000 if u & 0x8000000000000000 else u


def _f16_to_f32_u32(h: int) -> int:
    try:
        val = struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]
        return _f32_to_u32(val)
    except Exception:
        return 0x7FC00000


def _f32_to_f16_u16(u: int) -> int:
    f = _u32_to_f32(u)
    if math.isnan(f):
        return 0x7E00
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return 0x7C00 if f > 0 else 0xFC00


def _f32_ulp_diff(a_u32: int, b_u32: int) -> Optional[int]:
    """Return ULP distance between two finite float32 bit patterns, or None if NaN/Inf."""
    a_u32 &= MASK32
    b_u32 &= MASK32
    if a_u32 == b_u32:
        return 0
    fa, fb = _u32_to_f32(a_u32), _u32_to_f32(b_u32)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return None
    if fa == 0.0 and fb == 0.0:
        return 0
    def _ordered(u: int) -> int:
        return (0x80000000 - (u & 0x7FFFFFFF)) if (u & 0x80000000) else (0x80000000 + u)
    return abs(_ordered(a_u32) - _ordered(b_u32))


def _det_hash_u32(tag: str, *ints: int) -> int:
    h = hashlib.blake2b(digest_size=4)
    h.update(tag.encode("utf-8"))
    for v in ints:
        h.update((int(v) & MASK64).to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")


# 16 structured IEEE-754 / bitvector corner-case scalars
_CORNER_U32S: List[int] = [
    0x00000000,  # +0.0 / 0
    0x80000000,  # -0.0
    0x3F800000,  # +1.0
    0xBF800000,  # -1.0
    0x3F000000,  # +0.5
    0x40000000,  # +2.0
    0x3E800000,  # +0.25
    0x40800000,  # +4.0
    0x3C23D70A,  # 0.01
    0x42C80000,  # 100.0
    0xC2C80000,  # -100.0
    0x00000001,  # smallest +subnormal / 1
    0x00000007,  # small int 7
    0x7FFFFFFF,  # INT_MAX
    0x80000001,  # negative int / -subnormal
    0xFFFFFFFF,  # -1 / UINT_MAX
]


class ConcreteShaderEnv(ShaderEnvBase):
    """Shared concrete input & memory oracle aligned with SharedShaderEnv."""

    def __init__(
        self,
        bundle: Dict[str, Any],
        vec_idx: int,
        seed: int = 0x5A17,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self.bundle = bundle
        self.vec_idx = vec_idx
        self.seed = seed
        self.overrides: Dict[str, Any] = overrides if overrides is not None else {}
        self.stage: str = bundle.get("stage", "Compute")
        self.mbs2: Dict[str, Any] = bundle.get("mbs2_context", {})
        self.direct_bindings: List[Dict[str, Any]] = bundle.get("direct_bindings", [])

        # Disjoint 4GB VA slots matching SharedShaderEnv
        self.buf_base_map: Dict[str, int] = {}
        self.base_to_buf: Dict[int, str] = {}
        self._next_region = 1

        ls_raw = self.mbs2.get("local_size")
        if isinstance(ls_raw, list) and len(ls_raw) == 3 and all(int(x) > 0 for x in ls_raw):
            self.local_size: Optional[Tuple[int, int, int]] = (int(ls_raw[0]), int(ls_raw[1]), int(ls_raw[2]))
        else:
            self.local_size = None

        for b in self.direct_bindings:
            key = b.get("key", "")
            if b.get("kind") in ("UBO", "SSBO", "PushConstant") and key:
                self.get_buf_base(key)

        wg_keys: List[str] = []
        for n in bundle.get("direct_ssa", {}).get("nodes", []):
            astr = n.get("aux_str", "")
            if n.get("op") == "LoadSidebandU64" and astr.startswith("workgroup:"):
                if astr not in wg_keys:
                    wg_keys.append(astr)
        self.unique_workgroup_key: Optional[str] = wg_keys[0] if len(wg_keys) == 1 else None
        if self.unique_workgroup_key is not None:
            self.buf_base_map["wls_shared"] = self.get_buf_base(self.unique_workgroup_key)

        for pf in self.mbs2.get("pilot_fau", []):
            key = pf.get("key", "")
            if pf.get("resolved") and key:
                self.get_buf_base(key)
        for rl in self.mbs2.get("rlocs", []):
            self.get_buf_base(f"ubo:0:{rl.get('binding', 0)}")

        self.fcst_by_word: Dict[int, int] = {
            int(fc["word_idx"]): int(fc["value"]) & MASK32 for fc in self.mbs2.get("fcsts", [])
        }
        self.viewport_fau_words: Dict[int, int] = {}
        self.pilot_by_word: Dict[int, Tuple[Dict[str, Any], bool]] = {}
        for pf in self.mbs2.get("pilot_fau", []):
            vkind = pf.get("value_kind", "")
            is_const = (vkind == "const_int" and pf.get("has_const_value", False))
            if is_const or (pf.get("resolved") and pf.get("key")):
                w = int(pf["word_idx"])
                is_hi = (int(pf["fau_offset"]) % 8) != 0
                self.pilot_by_word[w] = (pf, is_hi)
                is_buf_addr = (vkind == "buffer_addr") or (not vkind and not pf.get("is_loaded_value", False))
                if is_buf_addr and not is_hi and int(pf.get("byte_size", 4)) >= 8:
                    if (w + 1) not in self.pilot_by_word:
                        self.pilot_by_word[w + 1] = (pf, True)

        self.rloc_by_word: Dict[int, Tuple[str, int]] = {}
        for rl in self.mbs2.get("rlocs", []):
            sym = rl.get("symbol", "")
            w0 = int(rl["word_idx"])
            nwords = max(1, (int(rl.get("size", 4)) + 3) // 4)
            uoff = int(rl.get("uni_offset", 0))
            if sym == "gl_mali_ViewportTransform":
                for k in range(nwords):
                    off = uoff + k * 4
                    self.viewport_fau_words[w0 + k] = 0x3F800000 if off < 16 else 0x00000000
                continue
            bkey = f"ubo:0:{rl.get('binding', 0)}"
            for k in range(nwords):
                self.rloc_by_word[w0 + k] = (bkey, uoff + k * 4)

        self.sinp_addr_to_loc: Dict[int, int] = {
            int(s["addr"]): int(s["location"]) for s in self.mbs2.get("sinp", []) if int(s.get("location", -1)) >= 0
        }
        self.sout_addr_to_loc: Dict[int, int] = {
            int(s["addr"]): int(s["location"]) for s in self.mbs2.get("sout", []) if int(s.get("location", -1)) >= 0
        }
        self.lane_id = int(self.overrides.get('lane_id', vec_idx & 31)) & 31

    def get_buf_base(self, key: str) -> int:
        if key not in self.buf_base_map:
            base = self._next_region << 32
            self._next_region += 1
            self.buf_base_map[key] = base
            self.base_to_buf[base] = key
        return self.buf_base_map[key]

    def sample_u32(self, name: str) -> int:
        if name in self.overrides:
            v = self.overrides[name]
            return (1 if v else 0) if isinstance(v, bool) else (int(v) & MASK32)
        if self.vec_idx < len(_CORNER_U32S):
            # Vary slightly across distinct input names while exercising corner values
            h = _det_hash_u32(name, self.seed, self.vec_idx)
            if (h & 3) == 0:
                return _CORNER_U32S[self.vec_idx]
            return _CORNER_U32S[(self.vec_idx + h) % len(_CORNER_U32S)]
        rng = random.Random(_det_hash_u32(name, self.seed, self.vec_idx))
        # Mix well-behaved finite floats in [-16, 16] with arbitrary 32-bit integers
        if (self.vec_idx & 3) != 3:
            val = rng.uniform(-16.0, 16.0)
            return _f32_to_u32(val)
        return rng.getrandbits(32)

    def read_init_buf(self, bkey: str, off32: int) -> int:
        if bkey == "wls_shared" and self.unique_workgroup_key is not None:
            bkey = self.unique_workgroup_key
        key = f"mem:{bkey}:{off32 & MASK32}"
        if key in self.overrides:
            return int(self.overrides[key]) & MASK32
        def_key = f"mem_default:{bkey}"
        if def_key in self.overrides:
            return int(self.overrides[def_key]) & MASK32
        return self.sample_u32(key)

    def read_sym_buf_slot(self, sym_tag: str, off32: int) -> int:
        o = int(off32) & MASK32
        key = f"sym_slot:{sym_tag}:{o}"
        if key in self.overrides:
            return int(self.overrides[key]) & MASK32
        dkey = f"sym_slot_default:{sym_tag}"
        if dkey in self.overrides:
            return int(self.overrides[dkey]) & MASK32
        return self.sample_u32(key)

    def read_init_img(self, ikey: str, xy64: int) -> Tuple[int, int, int, int]:
        xy = int(xy64) & MASK64
        out = []
        for c in range(4):
            key = f"img:{ikey}:c{c}:{xy}"
            dkey = f"img_default:{ikey}:c{c}"
            if key in self.overrides:
                out.append(int(self.overrides[key]) & MASK32)
            elif dkey in self.overrides:
                out.append(int(self.overrides[dkey]) & MASK32)
            else:
                out.append(self.sample_u32(key))
        return (out[0], out[1], out[2], out[3])

    def read_fallback_mem(self, addr64: int) -> int:
        a = int(addr64) & MASK64
        key = f"mem_fallback:{a}"
        if key in self.overrides:
            return int(self.overrides[key]) & MASK32
        if "mem_fallback_default" in self.overrides:
            return int(self.overrides["mem_fallback_default"]) & MASK32
        return self.sample_u32(key)

    def eval_hw_uf(self, uf_name: str, *args: int, fallback_tag: str = "") -> int:
        spec = self.overrides.get(f"uf:{uf_name}")
        if isinstance(spec, dict):
            key = ",".join(str(int(a) & MASK32) for a in args)
            tbl = spec.get("table", {})
            if key in tbl:
                return int(tbl[key]) & MASK32
            if spec.get("default") is not None:
                return int(spec["default"]) & MASK32
        tag = fallback_tag or uf_name
        return _det_hash_u32(tag, self.seed, self.vec_idx, *(int(a) & MASK32 for a in args))

    def resolve_addr(self, addr64: int, hint_key: str = "") -> Tuple[Optional[str], int]:
        region = (addr64 >> 32) << 32
        off = addr64 & MASK32
        if region in self.base_to_buf:
            return self.base_to_buf[region], off
        if hint_key:
            return hint_key, off
        return None, off

    def get_builtin(self, bi: int, vt: str) -> Any:
        if self.stage == "Compute" and self.local_size is not None:
            lx, ly, lz = self.local_size
            lid = (
                self.sample_u32("bi_27_0") % lx,
                self.sample_u32("bi_27_1") % ly,
                self.sample_u32("bi_27_2") % lz,
            )
            wg = (
                self.sample_u32("bi_26_0") & 0xFFFF,
                self.sample_u32("bi_26_1") & 0xFFFF,
                self.sample_u32("bi_26_2") & 0xFFFF,
            )
            if bi == 27:
                return lid
            if bi == 26:
                return wg
            if bi == 28:
                return (
                    (wg[0] * lx + lid[0]) & MASK32,
                    (wg[1] * ly + lid[1]) & MASK32,
                    (wg[2] * lz + lid[2]) & MASK32,
                )
            if bi == 29:
                return (lid[2] * (lx * ly) + lid[1] * lx + lid[0]) & MASK32
        if vt == "V4F32":
            return tuple(self.sample_u32(f"bi_{bi}_{c}") for c in range(4))
        if vt == "V3U32":
            return tuple(self.sample_u32(f"bi_{bi}_{c}") for c in range(3))
        if vt == "Bool":
            return (self.sample_u32(f"bi_{bi}") & 1) != 0
        return self.sample_u32(f"bi_{bi}")

    def get_input_vec4(self, kind: str, loc: int) -> Tuple[int, ...]:
        return tuple(self.sample_u32(f"{kind}_loc{loc}_{c}") for c in range(4))

    def _sample_fau_word(self, word_idx: int) -> int:
        fkey = f"fau_sym_{word_idx}"
        if fkey in self.overrides:
            return int(self.overrides[fkey]) & MASK32
        if "fau_sym_default" in self.overrides:
            return int(self.overrides["fau_sym_default"]) & MASK32
        return self.sample_u32(fkey)

    def load_fau_word(self, word_idx: int) -> int:
        if word_idx in self.fcst_by_word:
            return self.fcst_by_word[word_idx]
        if word_idx in self.viewport_fau_words:
            return self.viewport_fau_words[word_idx]
        if word_idx in self.pilot_by_word:
            pf, is_hi = self.pilot_by_word[word_idx]
            vkind = pf.get("value_kind", "")
            if vkind == "const_int" and pf.get("has_const_value", False):
                cv = int(pf["const_value"])
                return ((cv >> 32) & MASK32) if is_hi else (cv & MASK32)
            if vkind in ("buffer_size", "descriptor_addr", "abi_preload", "unknown") or pf.get("is_derived", False):
                return self._sample_fau_word(word_idx)
            if vkind == "symbol_value" or pf.get("is_loaded_value", False):
                if pf.get("has_byte_offset", False):
                    return self.read_init_buf(pf["key"], int(pf["byte_offset"]))
                return self._sample_fau_word(word_idx)
            base = self.get_buf_base(pf["key"])
            if pf.get("has_pointer_offset", False):
                base = (base + int(pf.get("pointer_offset", 0))) & MASK64
            return ((base >> 32) & MASK32) if is_hi else (base & MASK32)
        if word_idx in self.rloc_by_word:
            bkey, uoff = self.rloc_by_word[word_idx]
            return self.read_init_buf(bkey, uoff)
        return self._sample_fau_word(word_idx)


def _c_u32(v: Any) -> int:
    if isinstance(v, tuple):
        return _c_u32(v[0]) if v else 0
    if isinstance(v, bool):
        return 1 if v else 0
    return int(v) & MASK32


def _c_u64(v: Any) -> int:
    if isinstance(v, tuple):
        return _c_u64(v[0]) if v else 0
    if isinstance(v, bool):
        return 1 if v else 0
    return int(v) & MASK64


def _c_bool(v: Any) -> bool:
    if isinstance(v, tuple):
        return _c_bool(v[0]) if v else False
    if isinstance(v, bool):
        return v
    return (int(v) & MASK64) != 0


def _c_vec_len(vt: str) -> int:
    if vt.startswith("V2"):
        return 2
    if vt.startswith("V3"):
        return 3
    if vt.startswith("V4"):
        return 4
    return 1


def _c_elemwise(vt: str, args: List[Any], fn: Any) -> Any:
    vlen = _c_vec_len(vt)
    is_vec = vlen > 1 or any(isinstance(a, tuple) for a in args)
    if not is_vec:
        return fn(*args)
    if vlen == 1:
        vlen = max(len(a) for a in args if isinstance(a, tuple))
    out = []
    for i in range(vlen):
        largs = [a[i] if isinstance(a, tuple) and i < len(a) else a for a in args]
        out.append(fn(*largs))
    return tuple(out)


class ConcreteSsaInterpreter:
    """Fast concrete bit-exact interpreter for ValhallSsaModule JSON."""

    def __init__(self, ssa_json: Dict[str, Any], is_valhall_side: bool, max_block_steps: int = 256):
        self.blocks_by_id: Dict[int, Dict[str, Any]] = {int(b["id"]): b for b in ssa_json.get("blocks", [])}
        self.nodes_by_id: Dict[int, Dict[str, Any]] = {int(n["id"]): n for n in ssa_json.get("nodes", [])}
        self.is_valhall = is_valhall_side
        self.max_block_steps = max_block_steps

    def _eval_fp_unary(self, op: str, u: int) -> int:
        u &= MASK32
        if op == "FAbs":
            return u & 0x7FFFFFFF
        if op == "FNeg":
            return u ^ 0x80000000
        f = _u32_to_f32(u)
        if math.isnan(f):
            return 0x7FC00000
        if op == "Floor":
            return _f32_to_u32(math.floor(f)) if math.isfinite(f) else u
        if op == "Ceil":
            return _f32_to_u32(math.ceil(f)) if math.isfinite(f) else u
        if op == "Trunc":
            return _f32_to_u32(math.trunc(f)) if math.isfinite(f) else u
        if op == "RoundEven":
            return _f32_to_u32(float(round(f))) if math.isfinite(f) else u
        if op == "Sqrt":
            return 0x7FC00000 if f < 0.0 else _f32_to_u32(math.sqrt(f))
        if op == "InverseSqrt":
            if f < 0.0:
                return 0x7FC00000
            if f == 0.0:
                return 0x7F800000
            return _f32_to_u32(1.0 / math.sqrt(f))
        if op == "Sin":
            return _f32_to_u32(math.sin(f)) if math.isfinite(f) else 0x7FC00000
        if op == "Cos":
            return _f32_to_u32(math.cos(f)) if math.isfinite(f) else 0x7FC00000
        if op == "Exp":
            try:
                return _f32_to_u32(math.exp(f))
            except OverflowError:
                return 0x7F800000
        if op == "Exp2":
            try:
                return _f32_to_u32(math.pow(2.0, f))
            except OverflowError:
                return 0x7F800000
        if op == "Log":
            if f < 0.0: return 0x7FC00000
            if f == 0.0: return 0xFF800000
            return _f32_to_u32(math.log(f))
        if op == "Log2":
            if f < 0.0: return 0x7FC00000
            if f == 0.0: return 0xFF800000
            return _f32_to_u32(math.log2(f))
        return _det_hash_u32(f"fp_{op}", u)

    def _eval_fp_binary(self, op: str, ua: int, ub: int) -> int:
        fa, fb = _u32_to_f32(ua), _u32_to_f32(ub)
        if op == "FAdd":
            return _f32_to_u32(fa + fb)
        if op == "FSub":
            return _f32_to_u32(fa - fb)
        if op == "FMul":
            return _f32_to_u32(fa * fb)
        if op == "FDiv":
            if math.isnan(fa) or math.isnan(fb):
                return 0x7FC00000
            if fb == 0.0:
                if fa == 0.0:
                    return 0x7FC00000
                sgn = (ua ^ ub) & 0x80000000
                return sgn | 0x7F800000
            return _f32_to_u32(fa / fb)
        if op in ("FMin", "NMin"):
            if math.isnan(fa): return ub & MASK32
            if math.isnan(fb): return ua & MASK32
            return ua if fa <= fb else ub
        if op in ("FMax", "NMax"):
            if math.isnan(fa): return ub & MASK32
            if math.isnan(fb): return ua & MASK32
            return ua if fa >= fb else ub
        if op == "Pow":
            try:
                return _f32_to_u32(math.pow(fa, fb))
            except Exception:
                return 0x7FC00000
        return _det_hash_u32(f"fp_{op}", ua, ub)

    def _eval_fp_cmp(self, op: str, ua: int, ub: int) -> bool:
        fa, fb = _u32_to_f32(ua), _u32_to_f32(ub)
        unord = math.isnan(fa) or math.isnan(fb)
        if op == "FCmpOrdEq": return (not unord) and (fa == fb)
        if op == "FCmpOrdGt": return (not unord) and (fa > fb)
        if op == "FCmpOrdGe": return (not unord) and (fa >= fb)
        if op == "FCmpOrdLt": return (not unord) and (fa < fb)
        if op == "FCmpOrdLe": return (not unord) and (fa <= fb)
        if op == "FCmpOrdNe": return (not unord) and (fa != fb)
        if op == "FCmpUnordNe": return unord or (fa != fb)
        if op == "FCmpTotal":
            def _key(u: int) -> int:
                return (0x80000000 - (u & 0x7FFFFFFF)) if (u & 0x80000000) else (0x80000000 + u)
            return _key(ua & MASK32) <= _key(ub & MASK32)
        return False

    def execute_one(self, env: ConcreteShaderEnv) -> Tuple[Dict[int, Any], Dict[str, Any]]:
        vals: Dict[int, Any] = {0: 0}
        terminated = False
        demoted = False
        frag_colors: Dict[int, Tuple[int, ...]] = {}
        builtin_outputs: Dict[int, Any] = {}
        vert_varyings: Dict[int, Tuple[int, ...]] = {}
        buf_writes: Dict[Tuple[str, int], int] = {}
        img_writes: Dict[Tuple[str, int], Tuple[int, ...]] = {}
        fallback_writes: Dict[int, int] = {}

        if 0 not in self.blocks_by_id:
            return vals, {}

        cur_bid = 0
        prev_bid = -1
        steps = 0
        while cur_bid in self.blocks_by_id and steps < self.max_block_steps:
            steps += 1
            blk = self.blocks_by_id[cur_bid]
            # Evaluate Phis from prev_bid
            for pid in blk.get("phis", []):
                pn = self.nodes_by_id.get(pid)
                if not pn:
                    continue
                chosen = None
                for pb, vid in pn.get("phi", []):
                    if int(pb) == prev_bid and int(vid) in vals:
                        chosen = vals[int(vid)]
                        break
                if chosen is None:
                    for _, vid in pn.get("phi", []):
                        if int(vid) in vals:
                            chosen = vals[int(vid)]
                            break
                vals[pid] = chosen if chosen is not None else (False if pn.get("type") == "Bool" else 0)

            for nid in blk.get("nodes", []):
                node = self.nodes_by_id.get(nid)
                if not node:
                    continue
                op = node["op"]
                vt = node.get("type", "U32")
                args = [vals.get(int(a), 0) for a in node.get("args", [])]
                imm32 = int(node.get("imm_u32", 0))
                imm32_2 = int(node.get("imm_u32_2", 0))
                imm64 = int(node.get("imm_u64", 0))
                aux_bool = bool(node.get("aux_bool", False))
                aux_bool2 = bool(node.get("aux_bool2", False))
                aux_str = node.get("aux_str", "")

                if op in ("ConstU32", "ConstS32", "ConstF32"):
                    res: Any = imm32 & MASK32
                elif op == "ConstU64":
                    res = imm64 & MASK64
                elif op == "ConstBool":
                    res = (imm32 != 0) or aux_bool
                elif op in ("Nop", "Copy", "Bitcast"):
                    if not args:
                        res = 0
                    else:
                        v = args[0]
                        if vt in ("U64", "S64"): res = _c_u64(v)
                        elif vt == "Bool": res = _c_bool(v)
                        elif vt in ("U32", "S32", "F32") and not isinstance(v, tuple): res = _c_u32(v)
                        else: res = v
                elif op in ("VecConstruct2", "VecConstruct3", "VecConstruct4"):
                    res = tuple(args)
                elif op == "VecExtract":
                    v = args[0]
                    res = v[imm32] if isinstance(v, tuple) and imm32 < len(v) else _c_u32(v)
                elif op == "VecShuffle":
                    a, b = args[0], args[1]
                    comb = (list(a) if isinstance(a, tuple) else [a]) + (list(b) if isinstance(b, tuple) else [b])
                    res = (comb[imm32], comb[imm32_2])
                elif op == "Pack64":
                    res = ((_c_u32(args[1]) << 32) | _c_u32(args[0])) & MASK64
                elif op == "Split64":
                    v64 = _c_u64(args[0])
                    res = (v64 & MASK32, (v64 >> 32) & MASK32)
                elif op == "TupleExtract":
                    tup = args[0]
                    res = tup[imm32] if isinstance(tup, tuple) and imm32 < len(tup) else tup
                elif op == "UnpackHalf2x16":
                    u = _c_u32(args[0])
                    res = (_f16_to_f32_u32(u & 0xFFFF), _f16_to_f32_u32((u >> 16) & 0xFFFF))
                elif op == "PackHalf2x16":
                    v2 = args[0]
                    if isinstance(v2, tuple) and len(v2) >= 2:
                        res = ((_f32_to_f16_u16(_c_u32(v2[1])) << 16) | _f32_to_f16_u16(_c_u32(v2[0]))) & MASK32
                    else:
                        h = _f32_to_f16_u16(_c_u32(v2))
                        res = ((h << 16) | h) & MASK32
                elif op == "BitNot": res = (~_c_u32(args[0])) & MASK32
                elif op == "SNeg": res = (-_c_u32(args[0])) & MASK32
                elif op == "SAbs":
                    x = _s32(_c_u32(args[0]))
                    res = (-x if x < 0 else x) & MASK32
                elif op == "Clz":
                    x = _c_u32(args[0])
                    res = 32 if x == 0 else (32 - x.bit_length())
                elif op == "BitCount": res = _c_u32(args[0]).bit_count()
                elif op == "BitReverse":
                    x = _c_u32(args[0])
                    res = int(f"{x:032b}"[::-1], 2)
                elif op == "IAdd": res = (_c_u32(args[0]) + _c_u32(args[1])) & MASK32
                elif op == "ISub": res = (_c_u32(args[0]) - _c_u32(args[1])) & MASK32
                elif op == "IMul": res = (_c_u32(args[0]) * _c_u32(args[1])) & MASK32
                elif op == "BitAnd": res = (_c_u32(args[0]) & _c_u32(args[1])) & MASK32
                elif op == "BitOr": res = (_c_u32(args[0]) | _c_u32(args[1])) & MASK32
                elif op == "BitXor": res = (_c_u32(args[0]) ^ _c_u32(args[1])) & MASK32
                elif op == "Shl": res = (_c_u32(args[0]) << (_c_u32(args[1]) & 31)) & MASK32
                elif op == "ShrU": res = (_c_u32(args[0]) >> (_c_u32(args[1]) & 31)) & MASK32
                elif op == "ShrS": res = (_s32(_c_u32(args[0])) >> (_c_u32(args[1]) & 31)) & MASK32
                elif op == "RotL":
                    a, sh = _c_u32(args[0]), _c_u32(args[1]) & 31
                    res = ((a << sh) | (a >> ((32 - sh) & 31))) & MASK32
                elif op == "RotR":
                    a, sh = _c_u32(args[0]), _c_u32(args[1]) & 31
                    res = ((a >> sh) | (a << ((32 - sh) & 31))) & MASK32
                elif op == "UDiv":
                    a, b = _c_u32(args[0]), _c_u32(args[1])
                    res = MASK32 if b == 0 else (a // b) & MASK32
                elif op == "SDiv":
                    a, b = _s32(_c_u32(args[0])), _s32(_c_u32(args[1]))
                    if b == 0: res = MASK32
                    elif a == -0x80000000 and b == -1: res = 0x80000000
                    else: res = int(a / b) & MASK32
                elif op == "UMod":
                    a, b = _c_u32(args[0]), _c_u32(args[1])
                    res = a if b == 0 else (a % b) & MASK32
                elif op == "SRem":
                    a, b = _s32(_c_u32(args[0])), _s32(_c_u32(args[1]))
                    if b == 0: res = a & MASK32
                    elif a == -0x80000000 and b == -1: res = 0
                    else: res = (a - int(a / b) * b) & MASK32
                elif op == "SMod":
                    a, b = _s32(_c_u32(args[0])), _s32(_c_u32(args[1]))
                    if b == 0: res = a & MASK32
                    elif a == -0x80000000 and b == -1: res = 0
                    else:
                        r = a - int(a / b) * b
                        if r != 0 and (r < 0) != (b < 0):
                            r += b
                        res = r & MASK32
                elif op in ("BfeU", "BfeS"):
                    x, off, w = _c_u32(args[0]), _c_u32(args[1]) & 31, _c_u32(args[2])
                    if w == 0: res = 0
                    elif w >= 32: res = x
                    else:
                        mask = (1 << w) - 1
                        ex = (x >> off) & mask
                        if op == "BfeS" and (ex & (1 << (w - 1))):
                            ex |= (~mask) & MASK32
                        res = ex & MASK32
                elif op == "Bfi":
                    base, ins_v, off, w = _c_u32(args[0]), _c_u32(args[1]), _c_u32(args[2]) & 31, _c_u32(args[3])
                    if w == 0: res = base
                    elif w >= 32: res = ins_v
                    else:
                        mask = (((1 << w) - 1) << off) & MASK32
                        res = ((base & ~mask) | ((ins_v << off) & mask)) & MASK32
                elif op == "BitNot64": res = (~_c_u64(args[0])) & MASK64
                elif op == "IAdd64": res = (_c_u64(args[0]) + _c_u64(args[1])) & MASK64
                elif op == "ISub64": res = (_c_u64(args[0]) - _c_u64(args[1])) & MASK64
                elif op == "IMul64": res = (_c_u64(args[0]) * _c_u64(args[1])) & MASK64
                elif op == "BitAnd64": res = (_c_u64(args[0]) & _c_u64(args[1])) & MASK64
                elif op == "BitOr64": res = (_c_u64(args[0]) | _c_u64(args[1])) & MASK64
                elif op == "BitXor64": res = (_c_u64(args[0]) ^ _c_u64(args[1])) & MASK64
                elif op == "Shl64": res = (_c_u64(args[0]) << (_c_u64(args[1]) & 63)) & MASK64
                elif op == "ShrU64": res = (_c_u64(args[0]) >> (_c_u64(args[1]) & 63)) & MASK64
                elif op == "ShrS64": res = (_s64(_c_u64(args[0])) >> (_c_u64(args[1]) & 63)) & MASK64
                elif op == "ICmpEq": res = _c_u32(args[0]) == _c_u32(args[1])
                elif op == "ICmpNe": res = _c_u32(args[0]) != _c_u32(args[1])
                elif op == "ICmpSLt": res = _s32(_c_u32(args[0])) < _s32(_c_u32(args[1]))
                elif op == "ICmpSLe": res = _s32(_c_u32(args[0])) <= _s32(_c_u32(args[1]))
                elif op == "ICmpSGt": res = _s32(_c_u32(args[0])) > _s32(_c_u32(args[1]))
                elif op == "ICmpSGe": res = _s32(_c_u32(args[0])) >= _s32(_c_u32(args[1]))
                elif op == "ICmpULt": res = _c_u32(args[0]) < _c_u32(args[1])
                elif op == "ICmpULe": res = _c_u32(args[0]) <= _c_u32(args[1])
                elif op == "ICmpUGt": res = _c_u32(args[0]) > _c_u32(args[1])
                elif op == "ICmpUGe": res = _c_u32(args[0]) >= _c_u32(args[1])
                elif op == "LogicalAnd": res = _c_bool(args[0]) and _c_bool(args[1])
                elif op == "LogicalOr": res = _c_bool(args[0]) or _c_bool(args[1])
                elif op == "LogicalNot": res = not _c_bool(args[0])
                elif op == "Select": res = args[1] if _c_bool(args[0]) else args[2]
                elif op in ("FAbs", "FNeg", "Floor", "Ceil", "Trunc", "RoundEven", "Sqrt", "InverseSqrt",
                            "Sin", "Cos", "Exp", "Exp2", "Log", "Log2", "Atan", "Asin", "Acos", "Sinh",
                            "Cosh", "Tanh", "DPdx", "DPdy", "Fwidth", "FrexpExp", "FrexpSignificand"):
                    res = _c_elemwise(vt, args, lambda x: self._eval_fp_unary(op, _c_u32(x)))
                elif op in ("FAdd", "FSub", "FMul", "FDiv", "FMin", "FMax", "NMin", "NMax", "Pow", "Ldexp", "Atan2"):
                    res = _c_elemwise(vt, args, lambda x, y: self._eval_fp_binary(op, _c_u32(x), _c_u32(y)))
                elif op == "FClamp":
                    res = _c_elemwise(vt, args, lambda x, lo, hi: self._eval_fp_binary("FMin", self._eval_fp_binary("FMax", _c_u32(x), _c_u32(lo)), _c_u32(hi)))
                elif op == "Fma":
                    def _fma(a: Any, b: Any, c: Any) -> int:
                        fa, fb, fc = _u32_to_f32(_c_u32(a)), _u32_to_f32(_c_u32(b)), _u32_to_f32(_c_u32(c))
                        # float64 product + add gives exact single-rounding FMA for binary32 inputs
                        return _f32_to_u32(fa * fb + fc)
                    res = _c_elemwise(vt, args, _fma)
                elif op == "IsNan": res = math.isnan(_u32_to_f32(_c_u32(args[0])))
                elif op == "IsInf": res = math.isinf(_u32_to_f32(_c_u32(args[0])))
                elif op.startswith("FCmp"): res = self._eval_fp_cmp(op, _c_u32(args[0]), _c_u32(args[1]))
                elif op == "ConvertFToS":
                    f = _u32_to_f32(_c_u32(args[0]))
                    if math.isnan(f): res = 0
                    elif f >= 2147483647.0: res = 0x7FFFFFFF
                    elif f <= -2147483648.0: res = 0x80000000
                    else: res = int(math.trunc(f)) & MASK32
                elif op == "ConvertFToU":
                    f = _u32_to_f32(_c_u32(args[0]))
                    if math.isnan(f) or f <= 0.0: res = 0
                    elif f >= 4294967295.0: res = MASK32
                    else: res = int(math.trunc(f)) & MASK32
                elif op == "ConvertSToF": res = _f32_to_u32(float(_s32(_c_u32(args[0]))))
                elif op == "ConvertUToF": res = _f32_to_u32(float(_c_u32(args[0])))
                elif op == "SConvert32To64": res = _s32(_c_u32(args[0])) & MASK64
                elif op == "UConvert32To64": res = _c_u32(args[0]) & MASK64
                elif op == "UConvert64To32": res = _c_u64(args[0]) & MASK32
                elif op == "LoadFau": res = env.load_fau_word(imm32 & 63)
                elif op == "LoadSidebandU64":
                    if aux_str and (aux_str.startswith(("ssbo:", "ubo:", "workgroup:")) or aux_str == "push_const"):
                        res = env.get_buf_base(aux_str)
                    elif imm32 == 6: res = env.get_buf_base("tls_private")
                    elif imm32 == 7: res = env.get_buf_base("wls_shared")
                    else: res = _det_hash_u32(f"sb64_{imm32}", env.seed, env.vec_idx)
                elif op == "LoadSidebandU32":
                    res = env.eval_hw_uf(f"hw_sideband_u32_{imm32}", imm32_2, fallback_tag=f"sb32_{imm32}_{imm32_2}")
                elif op == "LoadBuiltinInput": res = env.get_builtin(imm32, vt)
                elif op == "LoadVertAttr":
                    loc = env.sinp_addr_to_loc.get(imm32, imm32) if self.is_valhall else imm32
                    res = env.get_input_vec4("vert_attr", loc)
                elif op == "LoadFragVarying":
                    loc = env.sinp_addr_to_loc.get(imm32, imm32) if self.is_valhall else imm32
                    res = env.get_input_vec4("frag_var", loc)
                elif op == "ImageQuerySize":
                    lod = _c_u32(args[0]) if args else 0
                    qkey = aux_str or "img"
                    res = env.eval_hw_uf(f"hw_imgq_{qkey}_c{imm32}", lod, fallback_tag=f"imgq_{qkey}_{imm32}")
                elif op in ("TexSample", "TexSampleLod", "TexSampleDref", "TexGather", "TexFetch", "ImageQueryLod"):
                    uv = args[0] if args else (0, 0)
                    u0 = _c_u32(uv[0]) if isinstance(uv, tuple) else _c_u32(uv)
                    u1 = _c_u32(uv[1]) if isinstance(uv, tuple) and len(uv) > 1 else 0
                    tkey = aux_str or f"tex_{imm32}"
                    lanes = tuple(env.eval_hw_uf(f"hw_tex_{tkey}_c{c}", u0, u1, fallback_tag=f"tex_{tkey}_c{c}") for c in range(4))
                    res = (lanes, args[-1] if args else 0)
                elif op == "PsbLoad":
                    addr = _c_u64(args[0])
                    hint = aux_str[4:].split("[", 1)[0] if aux_str.startswith("sym:") else aux_str
                    bkey, off = env.resolve_addr(addr, hint_key=hint)
                    if bkey is not None:
                        val = buf_writes.get((bkey, off), env.read_init_buf(bkey, off))
                    else:
                        val = fallback_writes.get(addr, env.read_fallback_mem(addr))
                    res = (val, args[1] if len(args) > 1 else 0)
                elif op == "PsbStore":
                    if not (terminated or demoted):
                        addr = _c_u64(args[0])
                        val = _c_u32(args[1])
                        hint = aux_str[4:].split("[", 1)[0] if aux_str.startswith("sym:") else aux_str
                        bkey, off = env.resolve_addr(addr, hint_key=hint)
                        if bkey is not None:
                            buf_writes[(bkey, off)] = val
                        else:
                            fallback_writes[addr] = val
                    res = args[2] if len(args) > 2 else 0
                elif op == "Kill":
                    kcond = _c_bool(args[0]) if (aux_bool and len(args) >= 2) else True
                    if kcond:
                        if aux_bool2: demoted = True
                        else: terminated = True
                    res = 0
                elif op == "StoreFragColor":
                    if not (terminated or demoted):
                        col = args[0]
                        frag_colors[imm32] = tuple(_c_u32(col[c]) if isinstance(col, tuple) and c < len(col) else 0 for c in range(4))
                    res = args[-1] if args else 0
                elif op == "StoreBuiltinOutput":
                    if not (terminated or demoted):
                        builtin_outputs[imm32] = args[0]
                    res = args[-1] if args else 0
                elif op == "StoreVertVarying":
                    if not (terminated or demoted):
                        loc = env.sout_addr_to_loc.get(imm32, imm32) if self.is_valhall else imm32
                        vec = args[0]
                        vert_varyings[loc] = tuple(_c_u32(vec[c]) if isinstance(vec, tuple) and c < len(vec) else 0 for c in range(4))
                    res = args[-1] if args else 0
                elif op == "StoreTexel":
                    if not (terminated or demoted):
                        coord = args[0]
                        texel = args[1] if len(args) > 1 else (0, 0, 0, 0)
                        cx = _c_u32(coord[0]) if isinstance(coord, tuple) and len(coord) > 0 else _c_u32(coord)
                        cy = _c_u32(coord[1]) if isinstance(coord, tuple) and len(coord) > 1 else 0
                        xy = ((cy << 32) | cx) & MASK64
                        ikey = aux_str or f"img_{imm32}"
                        img_writes[(ikey, xy)] = tuple(_c_u32(texel[c]) if isinstance(texel, tuple) and c < len(texel) else 0 for c in range(4))
                    res = args[-1] if args else 0
                elif op == "SubgroupBallot":
                    res = _det_hash_u32("SubgroupBallot", env.seed, env.vec_idx, _c_u32(args[0]), env.lane_id)
                elif op == "SubgroupShuffle":
                    res = _det_hash_u32("SubgroupShuffle", env.seed, env.vec_idx, _c_u32(args[0]), _c_u32(args[1]), env.lane_id)
                else:
                    res = args[-1] if args else 0
                vals[nid] = res

            if terminated or blk.get("is_end") or blk.get("is_discard"):
                break
            prev_bid = cur_bid
            bt = int(blk.get("branch_target", -1))
            ft = int(blk.get("fallthrough", -1))
            if blk.get("is_cond") and bt >= 0 and ft >= 0:
                cond_v = _c_bool(vals.get(int(blk.get("branch_cond", 0)), False))
                cur_bid = bt if cond_v else ft
            elif bt >= 0:
                cur_bid = bt
            elif ft >= 0:
                cur_bid = ft
            else:
                break

        outs: Dict[str, Any] = {}
        if env.stage == "Fragment" or terminated or demoted:
            outs["discarded"] = bool(terminated or demoted)
        if not (terminated or demoted):
            for rt, col in frag_colors.items():
                for c, cname in enumerate(("r", "g", "b", "a")):
                    outs[f"frag_color[{rt}].{cname}"] = col[c]
            for bi, payload in builtin_outputs.items():
                if isinstance(payload, tuple):
                    for c, v in enumerate(payload):
                        outs[f"builtin_out[{bi}][{c}]"] = _c_u32(v)
                else:
                    outs[f"builtin_out[{bi}]"] = _c_u32(payload)
            for loc, vec in vert_varyings.items():
                for c in range(4):
                    outs[f"vert_var[{loc}][{c}]"] = vec[c]
            for (bkey, off), v in buf_writes.items():
                if bkey in ("tls_private", "wls_shared") or bkey.startswith("workgroup:"):
                    continue
                outs[f"{bkey}+0x{off:x}"] = v
            for (ikey, xy), vec in img_writes.items():
                for c, cname in enumerate(("r", "g", "b", "a")):
                    outs[f"img:{ikey}[0x{xy:x}].{cname}"] = vec[c]
        return vals, outs




def evaluate_ssa(
    bundle: Dict[str, Any],
    ssa_json: Dict[str, Any],
    is_valhall_side: bool = False,
    mode: str = "symbolic",
    fp_mode: str = "exact",
    max_unroll: int = 1,
    vec_idx: int = 0,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Unified entrypoint for evaluating a ValhallSsaModule symbolically or concretely."""
    if mode == "concrete":
        cenv = ConcreteShaderEnv(bundle, vec_idx=vec_idx, overrides=overrides)
        return ConcreteSsaInterpreter(ssa_json, is_valhall_side=is_valhall_side).execute_one(cenv)
    env = SharedShaderEnv(bundle)
    ev = DagSsaZ3Evaluator(ssa_json, env, is_valhall_side=is_valhall_side, fp_mode=fp_mode, max_unroll=max_unroll)
    st = ev.evaluate()
    return ev, st, env
