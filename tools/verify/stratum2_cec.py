#!/usr/bin/env python3
"""
Stratum 2: Data-Plane Decomposition via Simulation-Guided Cutpoint Sweeping (CEC).

Three composable stages operating on Stratum-1 canonicalized ValhallSsaModule pairs:
  1. ConcreteSsaInterpreter: Fast deterministic bit-exact IEEE-754 + BV32/BV64
     differential simulator over K edge-case + seeded random input vectors.
     Immediately catches real lowering bugs in <5ms and records per-node K-vector
     signatures.
  2. discover_cutpoint_candidates: Hashes node K-vector signatures across
     direct_ssa and valhall_ssa to identify candidate internal equivalence
     cutpoints (exact bit-match, negated Bool, and FNeg-flipped F32).
  3. CecCutpointSweeper: Bottom-up incremental Z3 (QF_BV + QF_FP) prover that
     discharges local cones between cutpoint frontiers, soundly abstracts proved
     pairs to fresh shared symbolic variables (by congruence), and runs CEGAR
     cone refinement if an abstracted output check yields a spurious SAT.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import random
import struct
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import z3

from tools.verify.stratum1_canon import SIDE_EFFECT_OPS, canonicalize_ssa_module
from tools.verify.shader_equiv_verifier import (
    BV32,
    BV64,
    BlockEffectState,
    DagSsaZ3Evaluator,
    SharedShaderEnv,
    _as_bool,
    _as_bv32,
    _as_bv64,
    _solve_miter_eq,
    _z3_and,
    _z3_not,
    _z3_or,
    bv32,
    bv32_to_fp32,
    bv64,
)

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


class ConcreteShaderEnv:
    """Shared concrete input & memory oracle aligned with SharedShaderEnv."""

    def __init__(self, bundle: Dict[str, Any], vec_idx: int, seed: int = 0x5A17):
        self.bundle = bundle
        self.vec_idx = vec_idx
        self.seed = seed
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
        self.lane_id = vec_idx & 31

    def get_buf_base(self, key: str) -> int:
        if key not in self.buf_base_map:
            base = self._next_region << 32
            self._next_region += 1
            self.buf_base_map[key] = base
            self.base_to_buf[base] = key
        return self.buf_base_map[key]

    def sample_u32(self, name: str) -> int:
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
        return self.sample_u32(f"mem:{bkey}:{off32 & MASK32}")

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
                return self.sample_u32(f"fau_sym_{word_idx}")
            if vkind == "symbol_value" or pf.get("is_loaded_value", False):
                if pf.get("has_byte_offset", False):
                    return self.read_init_buf(pf["key"], int(pf["byte_offset"]))
                return self.sample_u32(f"fau_sym_{word_idx}")
            base = self.get_buf_base(pf["key"])
            if pf.get("has_pointer_offset", False):
                base = (base + int(pf.get("pointer_offset", 0))) & MASK64
            return ((base >> 32) & MASK32) if is_hi else (base & MASK32)
        if word_idx in self.rloc_by_word:
            bkey, uoff = self.rloc_by_word[word_idx]
            return self.read_init_buf(bkey, uoff)
        return self.sample_u32(f"fau_sym_{word_idx}")


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
                    res = _det_hash_u32(f"sb32_{imm32}_{imm32_2}", env.seed, env.vec_idx)
                elif op == "LoadBuiltinInput": res = env.get_builtin(imm32, vt)
                elif op == "LoadVertAttr":
                    loc = env.sinp_addr_to_loc.get(imm32, imm32) if self.is_valhall else imm32
                    res = env.get_input_vec4("vert_attr", loc)
                elif op == "LoadFragVarying":
                    loc = env.sinp_addr_to_loc.get(imm32, imm32) if self.is_valhall else imm32
                    res = env.get_input_vec4("frag_var", loc)
                elif op == "ImageQuerySize":
                    lod = _c_u32(args[0]) if args else 0
                    res = _det_hash_u32(f"imgq_{aux_str}_{imm32}", env.seed, env.vec_idx, lod)
                elif op == "TexSample":
                    uv = args[0] if args else (0, 0)
                    u0 = _c_u32(uv[0]) if isinstance(uv, tuple) else _c_u32(uv)
                    u1 = _c_u32(uv[1]) if isinstance(uv, tuple) and len(uv) > 1 else 0
                    tkey = aux_str or f"tex_{imm32}"
                    lanes = tuple(_det_hash_u32(f"tex_{tkey}_c{c}", env.seed, env.vec_idx, u0, u1) for c in range(4))
                    res = (lanes, args[-1] if args else 0)
                elif op == "PsbLoad":
                    addr = _c_u64(args[0])
                    hint = aux_str[4:].split("[", 1)[0] if aux_str.startswith("sym:") else aux_str
                    bkey, off = env.resolve_addr(addr, hint_key=hint)
                    if bkey is not None:
                        val = buf_writes.get((bkey, off), env.read_init_buf(bkey, off))
                    else:
                        val = fallback_writes.get(addr, _det_hash_u32("fb_mem", env.seed, env.vec_idx, addr))
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


@dataclasses.dataclass
class ConcreteDiffResult:
    status: str  # EXACT_MATCH, APPROX_ULP1, MISMATCH, SKIP
    k_vectors: int
    checked_outputs: int
    exact_outputs: int
    approx_ulp1_outputs: int
    mismatched_outputs: int
    first_mismatch: Optional[str]
    elapsed_ms: float
    d_node_traces: Dict[int, List[Any]]
    v_node_traces: Dict[int, List[Any]]


def run_concrete_diff(
    bundle: Dict[str, Any],
    direct_ssa: Dict[str, Any],
    valhall_ssa: Dict[str, Any],
    k: int = 32,
    seed: int = 0x5A17,
) -> ConcreteDiffResult:
    """Run K-vector concrete differential simulation and collect per-node value traces."""
    t0 = time.perf_counter()
    d_interp = ConcreteSsaInterpreter(direct_ssa, is_valhall_side=False)
    v_interp = ConcreteSsaInterpreter(valhall_ssa, is_valhall_side=True)

    d_node_traces: Dict[int, List[Any]] = {}
    v_node_traces: Dict[int, List[Any]] = {}
    d_out_traces: Dict[str, List[Any]] = {}
    v_out_traces: Dict[str, List[Any]] = {}

    for i in range(k):
        cenv = ConcreteShaderEnv(bundle, vec_idx=i, seed=seed)
        d_vals, d_outs = d_interp.execute_one(cenv)
        v_vals, v_outs = v_interp.execute_one(cenv)
        for nid, val in d_vals.items():
            d_node_traces.setdefault(nid, [None] * k)[i] = val
        for nid, val in v_vals.items():
            v_node_traces.setdefault(nid, [None] * k)[i] = val
        all_okeys = set(d_outs.keys()) | set(v_outs.keys())
        for ok in all_okeys:
            if ok not in d_out_traces:
                d_out_traces[ok] = [None] * k
                v_out_traces[ok] = [None] * k
            d_out_traces[ok][i] = d_outs.get(ok)
            v_out_traces[ok][i] = v_outs.get(ok)

    all_keys = sorted(d_out_traces.keys())
    if not all_keys:
        return ConcreteDiffResult("SKIP", k, 0, 0, 0, 0, None, (time.perf_counter() - t0) * 1000.0, d_node_traces, v_node_traces)

    exact_cnt = 0
    approx_cnt = 0
    mismatch_cnt = 0
    first_mm: Optional[str] = None
    for ok in all_keys:
        dl, vl = d_out_traces[ok], v_out_traces[ok]
        if dl == vl:
            exact_cnt += 1
            continue
        # Check if all diffs are finite float32 within <= 1 ULP (e.g. FMA contraction)
        max_ulp = 0
        is_approx = True
        first_idx = -1
        for idx, (dv, vv) in enumerate(zip(dl, vl)):
            if dv == vv:
                continue
            if first_idx < 0:
                first_idx = idx
            if isinstance(dv, int) and isinstance(vv, int) and not isinstance(dv, bool) and not isinstance(vv, bool):
                ud = _f32_ulp_diff(dv, vv)
                if ud is None or ud > 1:
                    is_approx = False
                else:
                    max_ulp = max(max_ulp, ud)
            else:
                is_approx = False
        if is_approx and max_ulp <= 1:
            approx_cnt += 1
        else:
            mismatch_cnt += 1
            if first_mm is None:
                dv0, vv0 = dl[first_idx], vl[first_idx]
                d_str = f"0x{dv0:08x}" if isinstance(dv0, int) and not isinstance(dv0, bool) else str(dv0)
                v_str = f"0x{vv0:08x}" if isinstance(vv0, int) and not isinstance(vv0, bool) else str(vv0)
                first_mm = f"{ok}@vec[{first_idx}]: direct={d_str} valhall={v_str}"

    if mismatch_cnt > 0:
        status = "MISMATCH"
    elif approx_cnt > 0:
        status = "APPROX_ULP1"
    else:
        status = "EXACT_MATCH"
    return ConcreteDiffResult(
        status=status,
        k_vectors=k,
        checked_outputs=len(all_keys),
        exact_outputs=exact_cnt,
        approx_ulp1_outputs=approx_cnt,
        mismatched_outputs=mismatch_cnt,
        first_mismatch=first_mm,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        d_node_traces=d_node_traces,
        v_node_traces=v_node_traces,
    )


_SKIP_CUTPOINT_OPS: Set[str] = SIDE_EFFECT_OPS | {
    "Nop",
    "Copy",
    "Bitcast",
    "VecExtract",
    "Phi",
    "ConstU32",
    "ConstS32",
    "ConstF32",
    "ConstU64",
    "ConstBool",
    "LoadBuiltinInput",
    "LoadVertAttr",
    "LoadFragVarying",
    "LoadSidebandU64",
    "LoadSidebandU32",
    "LoadFau",
    "Branch",
    "BranchCond",
    "Return",
}


def _compute_node_depths(ssa: Dict[str, Any]) -> Dict[int, int]:
    nodes = {int(n["id"]): n for n in ssa.get("nodes", [])}
    depth: Dict[int, int] = {}

    def get_d(nid: int, visiting: Set[int]) -> int:
        if nid in depth:
            return depth[nid]
        if nid not in nodes or nid in visiting:
            return 0
        visiting.add(nid)
        n = nodes[nid]
        arg_ds = [get_d(int(a), visiting) for a in n.get("args", [])]
        d = (1 + max(arg_ds)) if arg_ds else 0
        visiting.remove(nid)
        depth[nid] = d
        return d

    for nid in nodes:
        get_d(nid, set())
    return depth


def discover_cutpoint_candidates(
    direct_ssa: Dict[str, Any],
    valhall_ssa: Dict[str, Any],
    d_traces: Dict[int, List[Any]],
    v_traces: Dict[int, List[Any]],
) -> List[Tuple[int, int, str]]:
    """
    Match internal scalar/bool SSA nodes across direct_ssa and valhall_ssa by
    K-vector concrete signature (EXACT, NEGATED_BOOL, FNEG).
    Returns candidates sorted topologically by direct node depth.
    """
    d_nodes = {int(n["id"]): n for n in direct_ssa.get("nodes", [])}
    v_nodes = {int(n["id"]): n for n in valhall_ssa.get("nodes", [])}
    d_depths = _compute_node_depths(direct_ssa)

    # Index valhall nodes by (type_class, signature_tuple)
    v_by_sig: Dict[Tuple[str, Tuple[Any, ...]], List[int]] = {}
    for vid, vn in v_nodes.items():
        if vn["op"] in _SKIP_CUTPOINT_OPS:
            continue
        vt = vn.get("type", "U32")
        if vt not in ("U32", "S32", "F32", "Bool", "U64", "S64"):
            continue
        tr = v_traces.get(vid)
        if not tr or any(x is None for x in tr):
            continue
        # Skip constant-valued traces (every element identical) to avoid spurious constant collisions
        if len(set(tr)) <= 1:
            continue
        tcls = "Bool" if vt == "Bool" else ("64" if vt in ("U64", "S64") else "32")
        v_by_sig.setdefault((tcls, tuple(tr)), []).append(vid)

    candidates: List[Tuple[int, int, int, str]] = []  # (depth, d_nid, v_nid, kind)
    used_v: Set[int] = set()
    for did in sorted(d_nodes.keys(), key=lambda x: (d_depths.get(x, 0), x)):
        dn = d_nodes[did]
        if dn["op"] in _SKIP_CUTPOINT_OPS:
            continue
        vt = dn.get("type", "U32")
        if vt not in ("U32", "S32", "F32", "Bool", "U64", "S64"):
            continue
        tr = d_traces.get(did)
        if not tr or any(x is None for x in tr) or len(set(tr)) <= 1:
            continue
        tcls = "Bool" if vt == "Bool" else ("64" if vt in ("U64", "S64") else "32")
        sig = tuple(tr)

        matched: Optional[Tuple[int, str]] = None
        for vid in v_by_sig.get((tcls, sig), []):
            if vid not in used_v:
                matched = (vid, "EXACT")
                break
        if matched is None and tcls == "Bool":
            neg_sig = tuple(not x for x in tr)
            for vid in v_by_sig.get(("Bool", neg_sig), []):
                if vid not in used_v:
                    matched = (vid, "NEGATED_BOOL")
                    break
        if matched is None and vt == "F32":
            fneg_sig = tuple(
                0x7FC00000 if ((x & 0x7FFFFFFF) > 0x7F800000) else ((x ^ 0x80000000) & MASK32)
                for x in tr
            )
            for vid in v_by_sig.get(("32", fneg_sig), []):
                if vid not in used_v:
                    matched = (vid, "FNEG")
                    break
        if matched is not None:
            vid, kind = matched
            used_v.add(vid)
            candidates.append((d_depths.get(did, 0), did, vid, kind))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return [(did, vid, kind) for _, did, vid, kind in candidates]


class _CutpointDagEvaluator(DagSsaZ3Evaluator):
    """DagSsaZ3Evaluator with sound cutpoint substitution map."""

    def __init__(
        self,
        ssa_json: Dict[str, Any],
        env: SharedShaderEnv,
        is_valhall_side: bool,
        fp_mode: str = "exact",
        max_unroll: int = 1,
        cutpoints: Optional[Dict[int, Any]] = None,
    ):
        super().__init__(ssa_json, env, is_valhall_side, fp_mode=fp_mode, max_unroll=max_unroll)
        self.cutpoints: Dict[int, Any] = cutpoints if cutpoints is not None else {}

    def _eval_node(self, node: Dict[str, Any], st: BlockEffectState, reach: z3.BoolRef) -> Any:
        nid = int(node["id"])
        if nid in self.cutpoints:
            return self.cutpoints[nid]
        return super()._eval_node(node, st, reach)


def _extract_cutpoint_vars(expr: z3.ExprRef) -> Set[str]:
    """Collect names of cut_* symbolic variables appearing in a Z3 expression."""
    found: Set[str] = set()
    visited: Set[int] = set()
    stack: List[z3.ExprRef] = [expr]
    while stack:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        if z3.is_const(cur) and cur.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            nm = cur.decl().name()
            if nm.startswith("cut_"):
                found.add(nm)
        elif z3.is_app(cur):
            for ch in cur.children():
                stack.append(ch)
    return found


@dataclasses.dataclass
class Stratum2Result:
    status: str  # UNSAT, SAT_CONCRETE, SAT_SMT, UNKNOWN, SKIP
    elapsed_ms: float
    concrete_ms: float
    sweep_ms: float
    final_ms: float
    concrete_status: str
    candidates_found: int
    cutpoints_proved: int
    cutpoints_failed: int
    cegar_refinements: int
    checked_outputs: int
    unsat_outputs: int
    unknown_outputs: int
    sat_outputs: int
    first_mismatch: Optional[str] = None


def verify_with_stratum2_cec(
    bundle: Dict[str, Any],
    use_stratum1: bool = True,
    k_sim: int = 32,
    cone_timeout_ms: int = 250,
    output_timeout_ms: int = 1000,
    max_cutpoints: int = 128,
    max_cegar_rounds: int = 3,
    skip_smt_on_concrete_mismatch: bool = True,
) -> Stratum2Result:
    """
    Run Stratum 1 canonicalization (optional) + Stratum 2 concrete differential
    simulation + bottom-up CEC cutpoint sweeping + CEGAR output verification.
    """
    t0 = time.perf_counter()
    direct_ssa = bundle["direct_ssa"]
    valhall_ssa = bundle["valhall_ssa"]
    if use_stratum1:
        direct_ssa, _ = canonicalize_ssa_module(direct_ssa)
        valhall_ssa, _ = canonicalize_ssa_module(valhall_ssa)

    # Stage 1: Fast concrete differential simulation
    cdiff = run_concrete_diff(bundle, direct_ssa, valhall_ssa, k=k_sim)
    cands = discover_cutpoint_candidates(direct_ssa, valhall_ssa, cdiff.d_node_traces, cdiff.v_node_traces)

    if cdiff.status == "MISMATCH" and skip_smt_on_concrete_mismatch:
        return Stratum2Result(
            status="SAT_CONCRETE",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            concrete_ms=cdiff.elapsed_ms,
            sweep_ms=0.0,
            final_ms=0.0,
            concrete_status=cdiff.status,
            candidates_found=len(cands),
            cutpoints_proved=0,
            cutpoints_failed=0,
            cegar_refinements=0,
            checked_outputs=cdiff.checked_outputs,
            unsat_outputs=0,
            unknown_outputs=0,
            sat_outputs=cdiff.mismatched_outputs,
            first_mismatch=cdiff.first_mismatch,
        )

    # Stage 2: Bottom-up incremental CEC cutpoint sweeping (exact QF_BV + QF_FP)
    t_sw = time.perf_counter()
    env = SharedShaderEnv(bundle)
    env.shared_sym_bkeys = set()
    preconditions = env.build_preconditions()

    d_cut_map: Dict[int, z3.ExprRef] = {}
    v_cut_map: Dict[int, z3.ExprRef] = {}
    cut_name_to_pair: Dict[str, Tuple[int, int]] = {}
    proved = 0
    failed = 0

    d_nodes_by_id = {int(n["id"]): n for n in direct_ssa.get("nodes", [])}

    # Only sweep internal cutpoints on acyclic 1-block shaders (multi-block sweeping
    # requires Stratum 3 SESE region alignment so reachability conditions match).
    can_sweep = (len(direct_ssa.get("blocks", [])) == 1 and len(valhall_ssa.get("blocks", [])) == 1)
    if can_sweep and cands:
        for d_nid, v_nid, kind in cands[:max_cutpoints]:
            # Evaluate both modules up to current frontier (fast on single-block DAG with cuts)
            de = _CutpointDagEvaluator(direct_ssa, env, False, "exact", 1, cutpoints=d_cut_map)
            ve = _CutpointDagEvaluator(valhall_ssa, env, True, "exact", 1, cutpoints=v_cut_map)
            de.evaluate()
            ve.evaluate()
            if d_nid not in de.values or v_nid not in ve.values:
                continue
            d_val = de.values[d_nid]
            v_val = ve.values[v_nid]
            vt = d_nodes_by_id[d_nid].get("type", "U32")
            if vt == "Bool":
                lhs = _as_bool(d_val)
                rhs = _z3_not(_as_bool(v_val)) if kind == "NEGATED_BOOL" else _as_bool(v_val)
            elif vt in ("U64", "S64"):
                lhs = _as_bv64(d_val)
                rhs = _as_bv64(v_val)
            else:
                lhs = _as_bv32(d_val)
                rhs = (_as_bv32(v_val) ^ bv32(0x80000000)) if kind == "FNEG" else _as_bv32(v_val)

            rc, _ = _solve_miter_eq(lhs, rhs, None, cone_timeout_ms, env, preconditions=preconditions)
            if rc == "UNSAT":
                cname = f"cut_{proved}"
                if vt == "Bool":
                    cvar: z3.ExprRef = z3.Bool(cname)
                    v_cvar = _z3_not(cvar) if kind == "NEGATED_BOOL" else cvar
                elif vt in ("U64", "S64"):
                    cvar = z3.BitVec(cname, 64)
                    v_cvar = cvar
                else:
                    cvar = z3.BitVec(cname, 32)
                    v_cvar = (cvar ^ bv32(0x80000000)) if kind == "FNEG" else cvar
                d_cut_map[d_nid] = cvar
                v_cut_map[v_nid] = v_cvar
                cut_name_to_pair[cname] = (d_nid, v_nid)
                proved += 1
            else:
                failed += 1
    sweep_ms = (time.perf_counter() - t_sw) * 1000.0

    # Stage 3: Final output verification with CEGAR refinement
    t_fin = time.perf_counter()
    stage = bundle.get("stage", "Unknown")

    def _build_output_checks(cur_d_cuts: Dict[int, z3.ExprRef], cur_v_cuts: Dict[int, z3.ExprRef]):
        de = _CutpointDagEvaluator(direct_ssa, env, False, "exact", 1, cutpoints=cur_d_cuts)
        ve = _CutpointDagEvaluator(valhall_ssa, env, True, "exact", 1, cutpoints=cur_v_cuts)
        d_st = de.evaluate()
        v_st = ve.evaluate()
        checks: List[Tuple[str, z3.ExprRef, z3.ExprRef, Optional[z3.BoolRef]]] = []
        if stage == "Fragment" or not z3.is_false(d_st.discarded) or not z3.is_false(v_st.discarded):
            checks.append(("discarded", d_st.discarded, v_st.discarded, None))
        not_disc = _z3_not(d_st.discarded)
        for rt in sorted(set(d_st.frag_colors) | set(v_st.frag_colors)):
            dc = d_st.frag_colors.get(rt, (bv32(0),) * 4)
            vc = v_st.frag_colors.get(rt, (bv32(0),) * 4)
            for ch, cname in enumerate(("r", "g", "b", "a")):
                checks.append((f"frag_color[{rt}].{cname}", dc[ch], vc[ch], not_disc))
        for bi in sorted(set(d_st.builtin_outputs) | set(v_st.builtin_outputs)):
            do = d_st.builtin_outputs.get(bi)
            vo = v_st.builtin_outputs.get(bi)
            if do is None or vo is None:
                checks.append((f"builtin_out[{bi}].present", z3.BoolVal(do is not None), z3.BoolVal(vo is not None), not_disc))
                continue
            bi_guard = not_disc
            if stage == "Vertex" and bi == 0 and env.viewport_fau_words and isinstance(do, tuple) and len(do) >= 4:
                vp_conds = [_as_bv32(do[3]) == bv32(0x3F800000)]
                for c in range(3):
                    dbv = _as_bv32(do[c])
                    vp_conds.append(z3.Not(z3.fpIsNaN(bv32_to_fp32(dbv))))
                    vp_conds.append(dbv != bv32(0x80000000))
                bi_guard = _z3_and(bi_guard, z3.And(*vp_conds))
            if isinstance(do, tuple) and isinstance(vo, tuple):
                for c in range(min(len(do), len(vo))):
                    checks.append((f"builtin_out[{bi}][{c}]", _as_bv32(do[c]), _as_bv32(vo[c]), bi_guard))
            else:
                checks.append((f"builtin_out[{bi}]", _as_bv32(do), _as_bv32(vo), bi_guard))
        for loc in sorted(set(d_st.vert_varyings) | set(v_st.vert_varyings)):
            dv = d_st.vert_varyings.get(loc, (bv32(0),) * 4)
            vv = v_st.vert_varyings.get(loc, (bv32(0),) * 4)
            for c in range(4):
                checks.append((f"vert_var[{loc}][{c}]", dv[c], vv[c], None))
        for bkey, off in sorted(set(d_st.const_stores) | set(v_st.const_stores)):
            if bkey in ("tls_private", "wls_shared") or bkey.startswith("workgroup:"):
                continue
            dw, dv = d_st.const_stores.get((bkey, off), (z3.BoolVal(False), z3.Select(env.get_buf_mem(bkey), bv32(off))))
            vw, vv = v_st.const_stores.get((bkey, off), (z3.BoolVal(False), z3.Select(env.get_buf_mem(bkey), bv32(off))))
            tag = f"{bkey}+0x{off:x}"
            checks.append((f"{tag}.written", dw, vw, None))
            checks.append((f"{tag}.value", dv, vv, _z3_or(dw, vw)))
        return checks

    checks = _build_output_checks(d_cut_map, v_cut_map)
    if not checks:
        return Stratum2Result(
            "SKIP", (time.perf_counter() - t0) * 1000.0, cdiff.elapsed_ms, sweep_ms,
            (time.perf_counter() - t_fin) * 1000.0, cdiff.status, len(cands), proved, failed,
            0, 0, 0, 0, 0,
        )

    unsat_o = 0
    unk_o = 0
    sat_o = 0
    cegar_refs = 0
    first_mm: Optional[str] = None

    for name, lhs, rhs, guard in checks:
        rc, cex = _solve_miter_eq(lhs, rhs, guard, output_timeout_ms, env, preconditions=preconditions)
        if rc == "SAT" and d_cut_map:
            # CEGAR: inline cutpoints referenced in this failing output cone
            local_d = dict(d_cut_map)
            local_v = dict(v_cut_map)
            cur_l, cur_r, cur_g = lhs, rhs, guard
            for _ in range(max_cegar_rounds):
                used_cuts = _extract_cutpoint_vars(cur_l) | _extract_cutpoint_vars(cur_r)
                if cur_g is not None:
                    used_cuts |= _extract_cutpoint_vars(cur_g)
                if not used_cuts:
                    break
                cegar_refs += 1
                for cn in used_cuts:
                    pair = cut_name_to_pair.get(cn)
                    if pair:
                        local_d.pop(pair[0], None)
                        local_v.pop(pair[1], None)
                ref_checks = dict((n, (l, r, g)) for n, l, r, g in _build_output_checks(local_d, local_v))
                if name not in ref_checks:
                    break
                cur_l, cur_r, cur_g = ref_checks[name]
                rc, cex = _solve_miter_eq(cur_l, cur_r, cur_g, output_timeout_ms, env, preconditions=preconditions)
                if rc != "SAT":
                    break
        if rc == "UNSAT":
            unsat_o += 1
        elif rc == "SAT":
            sat_o += 1
            if first_mm is None and cex is not None:
                first_mm = f"{name}: direct={cex[0]} valhall={cex[1]}"
        else:
            unk_o += 1

    final_ms = (time.perf_counter() - t_fin) * 1000.0
    if sat_o > 0:
        status = "SAT_SMT"
    elif unk_o > 0:
        status = "UNKNOWN"
    else:
        status = "UNSAT"

    return Stratum2Result(
        status=status,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        concrete_ms=cdiff.elapsed_ms,
        sweep_ms=sweep_ms,
        final_ms=final_ms,
        concrete_status=cdiff.status,
        candidates_found=len(cands),
        cutpoints_proved=proved,
        cutpoints_failed=failed,
        cegar_refinements=cegar_refs,
        checked_outputs=len(checks),
        unsat_outputs=unsat_o,
        unknown_outputs=unk_o,
        sat_outputs=sat_o,
        first_mismatch=first_mm,
    )
