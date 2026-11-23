#!/usr/bin/env python3
"""
Shared ABI/MBS2/FAU/memory-region environment for concrete and symbolic ValhallSSA
interpreters, plus bit-level float32/float16/int32/int64 conversion helpers.
"""

from __future__ import annotations

import fractions
import hashlib
import math
import random
import struct
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import z3

from tools.verify.panfrost_z3_model import (
    BV8,
    BV32,
    BV64,
    FP32_SORT,
    _hw_uf,
    bv32,
    bv64,
    fp32_to_bv32,
)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

Z3Val = Union[z3.ExprRef, Tuple[Any, ...]]

REAL_SORT = z3.RealSort()
_REAL_ZERO = z3.RealVal(0)
_REAL_ONE = z3.RealVal(1)
_REAL_TWO = z3.RealVal(2)
_REAL_HALF = z3.Q(1, 2)
_REAL_NEG_HALF = z3.Q(-1, 2)
_REAL_NEG_ONE = z3.RealVal(-1)


def _f32_bits_to_real_const(u32: int) -> Optional[z3.ArithRef]:
    f = struct.unpack("<f", struct.pack("<I", u32 & MASK32))[0]
    if not math.isfinite(f):
        return None
    if f == 0.0:
        return _REAL_ZERO
    frac = fractions.Fraction(f)
    return z3.Q(frac.numerator, frac.denominator)


_LOG2_E_REAL = _f32_bits_to_real_const(0x3FB8AA3B)
_LN_2_REAL = _f32_bits_to_real_const(0x3F317218)


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


def _eval_elementwise(vt: str, args: List[Any], fn: Any) -> Any:
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


_c_vec_len = _vec_len
_c_elemwise = _eval_elementwise


# Structured IEEE-754 / bitvector corner-case scalars including NaN and +/-Inf
_CORNER_U32S: List[int] = [
    0x00000000,  # +0.0 / 0
    0x80000000,  # -0.0
    0x3F800000,  # +1.0
    0xBF800000,  # -1.0
    0x3F000000,  # +0.5
    0x40000000,  # +2.0
    0x3E800000,  # +0.25
    0x40800000,  # +4.0
    0x7FC00000,  # qNaN
    0x7F800000,  # +Inf
    0xFF800000,  # -Inf
    0x3C23D70A,  # 0.01
    0x42C80000,  # 100.0
    0xC2C80000,  # -100.0
    0x00000001,  # smallest +subnormal / 1
    0x00000007,  # small int 7
    0x7FFFFFFF,  # INT_MAX
    0x80000001,  # negative int / -subnormal
    0xFFFFFFFF,  # -1 / UINT_MAX
]


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

    def get_buf_base(self, key: str) -> int:
        if key not in self.buf_base_map:
            base = self._next_region << 32
            self._next_region += 1
            self.buf_base_map[key] = base
            self.base_to_buf[base] = key
        return self.buf_base_map[key]


class SharedShaderEnv(ShaderEnvBase):
    """Shared symbolic input & disjoint buffer address environment for both SSA modules."""

    def __init__(self, bundle: Dict[str, Any]):
        self._init_abi_layout(bundle)
        self.inputs: Dict[str, z3.ExprRef] = {}
        self.init_buf_mem: Dict[str, z3.ArrayRef] = {}
        self.init_img_mem: Dict[str, Tuple[z3.ArrayRef, ...]] = {}
        self.fallback_mem: z3.ArrayRef = z3.Array("mem_fallback", BV64, BV32)
        self.fau_sym = z3.Array("fau_sym", BV32, BV32)
        self.lane_id = z3.BitVec("lane_id", 5)
        self.inputs["lane_id"] = self.lane_id

    def is_shared_sym_bkey(self, bkey: str) -> bool:
        return bkey in getattr(self, "shared_sym_bkeys", set())

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
            off = c & MASK32
            if region in self.base_to_buf:
                return self.base_to_buf[region], off, bv32(off)
            if hint_key:
                return hint_key, off, bv32(off)
            return None, c, bv32(off)

        hi = _try_const_u64(z3.Extract(63, 32, addr_expr))
        if hi is not None:
            region = hi << 32
            if region in self.base_to_buf:
                lo = z3.simplify(z3.Extract(31, 0, addr_expr))
                lo_c = _try_const_u64(lo)
                return self.base_to_buf[region], lo_c, lo

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
                return bv32((cv >> 32) & MASK32) if is_hi else bv32(cv & MASK32)
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
                base = (base + int(pf.get("pointer_offset", 0))) & MASK64
            if not is_hi:
                return bv32(base & MASK32)
            return bv32((base >> 32) & MASK32)
        if word_idx in self.rloc_by_word:
            bkey, uoff = self.rloc_by_word[word_idx]
            return z3.Select(self.get_buf_mem(bkey), bv32(uoff))
        return z3.Select(self.fau_sym, bv32(word_idx))


class ConcreteShaderEnv(ShaderEnvBase):
    """Shared concrete input & memory oracle aligned with SharedShaderEnv."""

    def __init__(
        self,
        bundle: Dict[str, Any],
        vec_idx: int,
        seed: int = 0x5A17,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self._init_abi_layout(bundle)
        self.vec_idx = vec_idx
        self.seed = seed
        self.overrides: Dict[str, Any] = overrides if overrides is not None else {}
        self.lane_id = int(self.overrides.get("lane_id", vec_idx & 31)) & 31

    def sample_u32(self, name: str) -> int:
        if name in self.overrides:
            v = self.overrides[name]
            return (1 if v else 0) if isinstance(v, bool) else (int(v) & MASK32)
        if self.vec_idx < len(_CORNER_U32S):
            h = _det_hash_u32(name, self.seed, self.vec_idx)
            if (h & 3) == 0:
                return _CORNER_U32S[self.vec_idx]
            return _CORNER_U32S[(self.vec_idx + h) % len(_CORNER_U32S)]
        rng = random.Random(_det_hash_u32(name, self.seed, self.vec_idx))
        if (self.vec_idx & 3) != 3:
            return _f32_to_u32(rng.uniform(-16.0, 16.0))
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
