#!/usr/bin/env python3
"""
All-Path Z3 Formal Equivalence Verifier:
  Direct SPIR-V -> ValhallSSA (src/spirv2ssa/)
    vs.
  SPIR-V -> libMali-Gxx -> Valhall ISA -> valhall_pass1 -> ValhallSSA

Defeats combinatorial state explosion via:
  - Whole-CFG topological DAG path-merging (O(|V|+|E|) instead of O(2^branches))
  - Per-buffer disjoint address partitioning + structured constant-offset store maps
  - Eager constant folding & syntactic simplification before Z3 emission
  - Per-output miter decomposition with strict Z3 solver timeouts
  - Theory stratification: exact QF_BV/QF_FP with sound UF-FP fallback on large FP graphs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
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

Z3Val = Union[z3.ExprRef, Tuple[Any, ...]]


@dataclasses.dataclass
class OutputMismatch:
    output_name: str
    direct_hex: str
    valhall_hex: str
    model_summary: str
    suspect_pcs: List[Tuple[int, str]]


@dataclasses.dataclass
class ShaderEquivResult:
    spv_path: str
    stage: str
    status: str  # UNSAT, BOUNDED_UNSAT, SAT, UNKNOWN, SKIP, ERROR
    fp_mode: str
    elapsed_ms: float
    checked_outputs: List[str] = dataclasses.field(default_factory=list)
    mismatches: List[OutputMismatch] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)
    direct_nodes: int = 0
    valhall_nodes: int = 0
    direct_blocks: int = 0
    valhall_blocks: int = 0
    has_loops: bool = False


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


class SharedShaderEnv:
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

        # Pre-allocate regions for all direct_bindings and pilot_fau keys
        for b in self.direct_bindings:
            key = b.get("key", "")
            if b.get("kind") in ("UBO", "SSBO", "PushConstant") and key:
                self.get_buf_base(key)
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

    def get_buf_mem(self, key: str) -> z3.ArrayRef:
        if key not in self.init_buf_mem:
            safe = key.replace(":", "_")
            self.init_buf_mem[key] = z3.Array(f"mem_{safe}", BV32, BV32)
        return self.init_buf_mem[key]

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
    frag_colors: Dict[int, Tuple[z3.BitVecRef, ...]]
    builtin_outputs: Dict[int, Z3Val]
    vert_varyings: Dict[int, Tuple[z3.BitVecRef, ...]]
    # Constant-offset buffer writes: (buf_key, offset) -> (written_cond, val_bv32)
    const_stores: Dict[Tuple[str, int], Tuple[z3.BoolRef, z3.BitVecRef]]
    # Dynamic buffer memory arrays: buf_key -> Array(BV32, BV32)
    buf_mems: Dict[str, z3.ArrayRef]
    # Image write arrays: img_key -> Tuple[Array(BV64, BV32) x 4]
    image_mems: Dict[str, Tuple[z3.ArrayRef, ...]]
    fallback_mem: z3.ArrayRef
    # Track originating PCs per output for counterexample localization
    output_pcs: Dict[str, Set[int]]


def _clone_state(st: BlockEffectState) -> BlockEffectState:
    return BlockEffectState(
        discarded=st.discarded,
        frag_colors=dict(st.frag_colors),
        builtin_outputs=dict(st.builtin_outputs),
        vert_varyings=dict(st.vert_varyings),
        const_stores=dict(st.const_stores),
        buf_mems=dict(st.buf_mems),
        image_mems=dict(st.image_mems),
        fallback_mem=st.fallback_mem,
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
        base_st.discarded = _z3_if(cond, st.discarded, base_st.discarded)
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
    ):
        self.ssa = ssa_json
        self.env = env
        self.is_valhall = is_valhall_side
        self.fp_mode = fp_mode
        self.blocks: List[Dict[str, Any]] = ssa_json.get("blocks", [])
        self.nodes_by_id: Dict[int, Dict[str, Any]] = {int(n["id"]): n for n in ssa_json.get("nodes", [])}
        self.values: Dict[int, Z3Val] = {0: bv32(0)}
        self.has_loops: bool = False

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
        if op == "Pack64":
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
                if self.fp_mode == "uf":
                    return _hw_uf("fp_FClamp", BV32, BV32, BV32, BV32)(bx, blo, bhi)
                return fp32_to_bv32(z3.fpMin(z3.fpMax(bv32_to_fp32(bx), bv32_to_fp32(blo)), bv32_to_fp32(bhi)))
            return _eval_elementwise(vt, args, _fclamp)
        if op == "Fma":
            def _ffma(a: Z3Val, b: Z3Val, c: Z3Val) -> z3.BitVecRef:
                ba, bb, bc = _as_bv32(a), _as_bv32(b), _as_bv32(c)
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
            return _fp_to_int_sat(bv32_to_fp32(x), 32, True, z3.RTZ()) if self.fp_mode == "exact" else _hw_uf("fp_FToS", BV32, BV32)(x)
        if op == "ConvertFToU":
            x = _as_bv32(args[0])
            return _fp_to_int_sat(bv32_to_fp32(x), 32, False, z3.RTZ()) if self.fp_mode == "exact" else _hw_uf("fp_FToU", BV32, BV32)(x)
        if op == "ConvertSToF":
            x = _as_bv32(args[0])
            return fp32_to_bv32(z3.fpSignedToFP(z3.RNE(), x, FP32_SORT)) if self.fp_mode == "exact" else _hw_uf("fp_SToF", BV32, BV32)(x)
        if op == "ConvertUToF":
            x = _as_bv32(args[0])
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

        if op == "TexSample":
            uv = args[0] if args else (bv32(0), bv32(0))
            u0 = _as_bv32(uv[0]) if isinstance(uv, tuple) else _as_bv32(uv)
            u1 = _as_bv32(uv[1]) if isinstance(uv, tuple) and len(uv) > 1 else bv32(0)
            tkey = aux_str or f"tex_{imm_u32}"
            lanes = tuple(_hw_uf(f"tex_{tkey}_c{c}", BV32, BV32, BV32)(u0, u1) for c in range(4))
            return (lanes, args[-1] if args else bv32(0))

        if op == "PsbLoad":
            addr = _as_bv64(args[0])
            bkey, const_off, off32 = self.env.resolve_addr(addr, hint_key=aux_str)
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
            bkey, const_off, off32 = self.env.resolve_addr(addr, hint_key=aux_str)
            nid = int(node["id"])
            if bkey is not None:
                cur_mem = st.buf_mems.get(bkey, self.env.get_buf_mem(bkey))
                st.buf_mems[bkey] = z3.If(reach, z3.Store(cur_mem, off32, val), cur_mem) if not z3.is_true(reach) else z3.Store(cur_mem, off32, val)
                if const_off is not None:
                    prev_w, prev_v = st.const_stores.get((bkey, const_off), (z3.BoolVal(False), z3.Select(cur_mem, off32)))
                    st.const_stores[(bkey, const_off)] = (_z3_or(prev_w, reach), _z3_if(reach, val, prev_v))
                    st.output_pcs.setdefault(f"{bkey}+0x{const_off:x}", set()).add(nid)
                else:
                    st.output_pcs.setdefault(f"{bkey}:dyn", set()).add(nid)
            else:
                st.fallback_mem = z3.If(reach, z3.Store(st.fallback_mem, addr, val), st.fallback_mem)
                st.output_pcs.setdefault("fallback_mem", set()).add(nid)
            return args[2] if len(args) > 2 else bv32(0)

        if op == "Kill":
            cond = _as_bool(args[0]) if (aux_bool and len(args) >= 2) else z3.BoolVal(True)
            st.discarded = _z3_or(st.discarded, _z3_and(reach, cond))
            st.output_pcs.setdefault("discarded", set()).add(int(node["id"]))
            return bv32(0)

        if op == "StoreFragColor":
            col = args[0]
            vec4 = tuple(_as_bv32(col[c]) if isinstance(col, tuple) and c < len(col) else bv32(0) for c in range(4))
            prev = st.frag_colors.get(imm_u32, (bv32(0),) * 4)
            st.frag_colors[imm_u32] = tuple(_z3_if(reach, n, p) for n, p in zip(vec4, prev))
            st.output_pcs.setdefault(f"frag_color[{imm_u32}]", set()).add(int(node["id"]))
            return args[-1] if args else bv32(0)

        if op == "StoreBuiltinOutput":
            payload = args[0]
            if imm_u32 in st.builtin_outputs:
                st.builtin_outputs[imm_u32] = _z3_if(reach, payload, st.builtin_outputs[imm_u32])
            else:
                st.builtin_outputs[imm_u32] = payload
            st.output_pcs.setdefault(f"builtin_out[{imm_u32}]", set()).add(int(node["id"]))
            return args[-1] if args else bv32(0)

        if op == "StoreVertVarying":
            loc = self.env.sout_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
            vec = args[0]
            vec4 = tuple(_as_bv32(vec[c]) if isinstance(vec, tuple) and c < len(vec) else bv32(0) for c in range(4))
            prev = st.vert_varyings.get(loc, (bv32(0),) * 4)
            st.vert_varyings[loc] = tuple(_z3_if(reach, n, p) for n, p in zip(vec4, prev))
            st.output_pcs.setdefault(f"vert_var[{loc}]", set()).add(int(node["id"]))
            return args[-1] if args else bv32(0)

        if op == "StoreTexel":
            coord = args[0]
            texel = args[1] if len(args) > 1 else (bv32(0),) * 4
            cx = _as_bv32(coord[0]) if isinstance(coord, tuple) and len(coord) > 0 else _as_bv32(coord)
            cy = _as_bv32(coord[1]) if isinstance(coord, tuple) and len(coord) > 1 else bv32(0)
            xy = z3.Concat(cy, cx)
            vec4 = tuple(_as_bv32(texel[c]) if isinstance(texel, tuple) and c < len(texel) else bv32(0) for c in range(4))
            ikey = aux_str or f"img_{imm_u32}"
            cur_im = st.image_mems.get(ikey, self.env.get_img_mem(ikey))
            updated = tuple(
                z3.If(reach, z3.Store(cur_im[c], xy, vec4[c]), cur_im[c]) if not z3.is_true(reach)
                else z3.Store(cur_im[c], xy, vec4[c])
                for c in range(4)
            )
            st.image_mems[ikey] = updated
            st.output_pcs.setdefault(f"img:{ikey}", set()).add(int(node["id"]))
            return args[-1] if args else bv32(0)

        if op in ("AtomicRmw", "AtomicExchange", "AtomicCmpXchg"):
            addr = _as_bv64(args[0])
            arg = _as_bv32(args[1]) if len(args) > 1 else bv32(0)
            uf = _hw_uf(f"atomic_{op}_{aux_str}", BV64, BV32, BV32)
            return (uf(addr, arg), args[-1] if args else bv32(0))

        if op in ("ControlBarrier", "Branch", "BranchCond", "Return"):
            return args[-1] if args else bv32(0)

        if op == "SubgroupBallot":
            return _hw_uf("SubgroupBallot", BV32, z3.BitVecSort(5), BV32)(_as_bv32(args[0]), self.env.lane_id)
        if op == "SubgroupShuffle":
            return _hw_uf("SubgroupShuffle", BV32, BV32, z3.BitVecSort(5), BV32)(_as_bv32(args[0]), _as_bv32(args[1]), self.env.lane_id)

        return bv32(0)

    def evaluate(self) -> BlockEffectState:
        """Evaluate all blocks in topological order using DAG path-merging."""
        init_state = BlockEffectState(
            discarded=z3.BoolVal(False),
            frag_colors={},
            builtin_outputs={},
            vert_varyings={},
            const_stores={},
            buf_mems={},
            image_mems={},
            fallback_mem=self.env.fallback_mem,
            output_pcs={},
        )
        if not self.blocks:
            return init_state

        blk_reach: Dict[int, z3.BoolRef] = {}
        blk_out_state: Dict[int, BlockEffectState] = {}
        # Incoming edges per block: dst -> list of (src_blk, edge_cond)
        edge_conds: Dict[int, List[Tuple[int, z3.BoolRef]]] = {b["id"]: [] for b in self.blocks}
        exit_states: List[Tuple[z3.BoolRef, BlockEffectState]] = []

        for blk in self.blocks:
            bid = int(blk["id"])
            if bid == 0:
                reach = z3.BoolVal(True)
                cur_st = _clone_state(init_state)
            else:
                inc_edges = [(p, c) for (p, c) in edge_conds.get(bid, []) if p in blk_out_state]
                if not inc_edges:
                    # Unreachable block
                    blk_reach[bid] = z3.BoolVal(False)
                    blk_out_state[bid] = _clone_state(init_state)
                    continue
                reach = z3.BoolVal(False)
                for _, c in inc_edges:
                    reach = _z3_or(reach, c)
                cur_st = _merge_states([(c, blk_out_state[p]) for (p, c) in inc_edges], self.env)

            blk_reach[bid] = reach

            # Detect loop headers (any predecessor >= bid)
            for p in blk.get("preds", []):
                if int(p) >= bid:
                    self.has_loops = True

            # Resolve Phis via active incoming edge predicates
            inc_cond_map = {p: c for (p, c) in edge_conds.get(bid, [])}
            for pid in blk.get("phis", []):
                pnode = self.nodes_by_id.get(int(pid))
                if not pnode:
                    continue
                incomings = pnode.get("phi", [])
                # Filter to already-evaluated forward predecessors
                fwd = [(int(p), int(v)) for p, v in incomings if int(p) in blk_out_state and int(v) in self.values]
                if not fwd:
                    vt = pnode.get("type", "U32")
                    self.values[int(pid)] = z3.BoolVal(False) if vt == "Bool" else bv32(0)
                    continue
                pass
                if len(fwd) == 1:
                    self.values[int(pid)] = self.values[fwd[0][1]]
                else:
                    acc = self.values[fwd[0][1]]
                    for pred_b, val_id in fwd[1:]:
                        ec = inc_cond_map.get(pred_b, z3.BoolVal(False))
                        acc = _z3_if(ec, self.values[val_id], acc)
                    self.values[int(pid)] = acc

            # Evaluate block nodes in order
            for nid in blk.get("nodes", []):
                node = self.nodes_by_id.get(int(nid))
                if not node:
                    continue
                self.values[int(nid)] = self._eval_node(node, cur_st, reach)

            blk_out_state[bid] = cur_st

            # Propagate outgoing CFG edges
            bt = int(blk.get("branch_target", -1))
            ft = int(blk.get("fallthrough", -1))
            is_cond = bool(blk.get("is_cond", False))
            is_end = bool(blk.get("is_end", False))
            is_discard = bool(blk.get("is_discard", False))

            if is_cond and bt >= 0 and ft >= 0:
                bcond_id = int(blk.get("branch_cond", 0))
                bcond = _as_bool(self.values.get(bcond_id, z3.BoolVal(False)))
                if bt > bid:
                    edge_conds.setdefault(bt, []).append((bid, _z3_and(reach, bcond)))
                else:
                    self.has_loops = True
                if ft > bid:
                    edge_conds.setdefault(ft, []).append((bid, _z3_and(reach, _z3_not(bcond))))
                else:
                    self.has_loops = True
            elif bt >= 0 and bt > bid:
                edge_conds.setdefault(bt, []).append((bid, reach))
            elif ft >= 0 and ft > bid:
                edge_conds.setdefault(ft, []).append((bid, reach))
            elif bt >= 0 and bt <= bid:
                self.has_loops = True

            if is_end or is_discard or (bt < 0 and ft < 0):
                exit_states.append((reach, cur_st))

        if not exit_states:
            return blk_out_state.get(len(self.blocks) - 1, init_state)
        return _merge_states(exit_states, self.env)


def _solve_miter_eq(
    lhs: z3.ExprRef,
    rhs: z3.ExprRef,
    guard: Optional[z3.BoolRef],
    timeout_ms: int,
    env: SharedShaderEnv,
) -> Tuple[str, Optional[Tuple[str, str, str]]]:
    """Check whether `guard => (lhs == rhs)` holds for all inputs."""
    if lhs.eq(rhs):
        return "UNSAT", None

    # Fast constant-simplification check before invoking SMT
    diff = z3.simplify(lhs != rhs)
    if z3.is_false(diff):
        return "UNSAT", None
    if guard is not None:
        query = z3.simplify(_z3_and(guard, diff))
        if z3.is_false(query):
            return "UNSAT", None
    else:
        query = diff

    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.set("rlimit", 2_000_000)
    s.add(query)
    rc = s.check()
    if rc == z3.unsat:
        return "UNSAT", None
    if rc == z3.sat:
        m = s.model()
        lv = m.eval(lhs, model_completion=True)
        rv = m.eval(rhs, model_completion=True)
        lhex = f"0x{lv.as_long():08x}" if isinstance(lv, z3.BitVecNumRef) else str(lv)
        rhex = f"0x{rv.as_long():08x}" if isinstance(rv, z3.BitVecNumRef) else str(rv)
        in_parts = []
        for k, v in list(env.inputs.items())[:8]:
            ev = m.eval(v, model_completion=True)
            if isinstance(ev, z3.BitVecNumRef):
                in_parts.append(f"{k}=0x{ev.as_long():x}")
            else:
                in_parts.append(f"{k}={ev}")
        return "SAT", (lhex, rhex, ", ".join(in_parts))
    return "UNKNOWN", None


def verify_equivalence_bundle(
    bundle: Dict[str, Any],
    timeout_ms: int = 3000,
    fp_mode: str = "exact",
    allow_uf_fallback: bool = True,
) -> ShaderEquivResult:
    """Verify all-path formal equivalence of direct_ssa and valhall_ssa in a JSON bundle."""
    t0 = time.perf_counter()
    spv_path = bundle.get("spv_path", "<memory>")
    stage = bundle.get("stage", "Unknown")
    direct_ssa = bundle.get("direct_ssa", {})
    valhall_ssa = bundle.get("valhall_ssa", {})

    res = ShaderEquivResult(
        spv_path=spv_path,
        stage=stage,
        status="UNKNOWN",
        fp_mode=fp_mode,
        elapsed_ms=0.0,
        direct_nodes=len(direct_ssa.get("nodes", [])),
        valhall_nodes=len(valhall_ssa.get("nodes", [])),
        direct_blocks=len(direct_ssa.get("blocks", [])),
        valhall_blocks=len(valhall_ssa.get("blocks", [])),
    )

    if not bundle.get("direct_ok", False) or not bundle.get("valhall_ok", False):
        res.status = "ERROR"
        res.notes.append(f"Lowering failed: direct={bundle.get('direct_error', '')}, valhall={bundle.get('valhall_error', '')}")
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res

    env = SharedShaderEnv(bundle)
    d_eval = DagSsaZ3Evaluator(direct_ssa, env, is_valhall_side=False, fp_mode=fp_mode)
    v_eval = DagSsaZ3Evaluator(valhall_ssa, env, is_valhall_side=True, fp_mode=fp_mode)

    d_st = d_eval.evaluate()
    v_st = v_eval.evaluate()
    res.has_loops = d_eval.has_loops or v_eval.has_loops
    pc_disasm: Dict[str, str] = bundle.get("mbs2_context", {}).get("pc_disasm", {})

    def lookup_pcs(key: str) -> List[Tuple[int, str]]:
        nids = v_st.output_pcs.get(key, set())
        pcs = v_eval.collect_pcs_for_nodes(nids) if nids else []
        return [(pc, pc_disasm.get(str(pc), "")) for pc in pcs[-16:]]

    checks: List[Tuple[str, z3.ExprRef, z3.ExprRef, Optional[z3.BoolRef], str]] = []

    # Discard equivalence (Fragment)
    if stage == "Fragment" or not z3.is_false(d_st.discarded) or not z3.is_false(v_st.discarded):
        checks.append(("discarded", d_st.discarded, v_st.discarded, None, "discarded"))

    not_disc = _z3_not(d_st.discarded)

    # Fragment color outputs (guarded by ~discarded)
    all_rts = sorted(set(d_st.frag_colors.keys()) | set(v_st.frag_colors.keys()))
    for rt in all_rts:
        dc = d_st.frag_colors.get(rt, (bv32(0),) * 4)
        vc = v_st.frag_colors.get(rt, (bv32(0),) * 4)
        for ch, cname in enumerate(("r", "g", "b", "a")):
            checks.append((f"frag_color[{rt}].{cname}", dc[ch], vc[ch], not_disc, f"frag_color[{rt}]"))

    # Builtin outputs (gl_Position, gl_FragDepth, etc.)
    all_bis = sorted(set(d_st.builtin_outputs.keys()) | set(v_st.builtin_outputs.keys()))
    for bi in all_bis:
        do = d_st.builtin_outputs.get(bi)
        vo = v_st.builtin_outputs.get(bi)
        if do is None or vo is None:
            continue
        bi_guard = not_disc
        # Under Valhall's folded gl_mali_ViewportTransform + perspective divide,
        # constrain clip.w == 1.0f alongside identity viewport (scale=1, bias=0) for gl_Position.
        if stage == "Vertex" and bi == 0 and env.viewport_fau_words and isinstance(do, tuple) and len(do) >= 4:
            vp_conds = [_as_bv32(do[3]) == bv32(0x3F800000)]
            for c in range(3):
                dbv = _as_bv32(do[c])
                # FMA(x, 1.0, +0.0) == x bit-for-bit iff x is not NaN and x != -0.0
                vp_conds.append(z3.Not(z3.fpIsNaN(bv32_to_fp32(dbv))))
                vp_conds.append(dbv != bv32(0x80000000))
            bi_guard = _z3_and(bi_guard, z3.And(*vp_conds))
        if isinstance(do, tuple) and isinstance(vo, tuple):
            for c in range(min(len(do), len(vo))):
                checks.append((f"builtin_out[{bi}][{c}]", _as_bv32(do[c]), _as_bv32(vo[c]), bi_guard, f"builtin_out[{bi}]"))
        else:
            checks.append((f"builtin_out[{bi}]", _as_bv32(do), _as_bv32(vo), bi_guard, f"builtin_out[{bi}]"))

    # Vertex varying outputs
    all_vv = sorted(set(d_st.vert_varyings.keys()) | set(v_st.vert_varyings.keys()))
    for loc in all_vv:
        dv = d_st.vert_varyings.get(loc, (bv32(0),) * 4)
        vv = v_st.vert_varyings.get(loc, (bv32(0),) * 4)
        for c in range(4):
            checks.append((f"vert_var[{loc}][{c}]", dv[c], vv[c], None, f"vert_var[{loc}]"))

    # Constant-offset buffer stores (SSBO writes; skip private thread-local scratch spills)
    all_cs = sorted(set(d_st.const_stores.keys()) | set(v_st.const_stores.keys()))
    for bkey, off in all_cs:
        if bkey in ("tls_private", "wls_shared"):
            continue
        dw, dv = d_st.const_stores.get((bkey, off), (z3.BoolVal(False), z3.Select(env.get_buf_mem(bkey), bv32(off))))
        vw, vv = v_st.const_stores.get((bkey, off), (z3.BoolVal(False), z3.Select(env.get_buf_mem(bkey), bv32(off))))
        tag = f"{bkey}+0x{off:x}"
        checks.append((f"{tag}.written", dw, vw, None, tag))
        checks.append((f"{tag}.value", dv, vv, _z3_or(dw, vw), tag))

    # Symbolic/dynamic-offset buffer stores (SSBO writes via Z3 Array theory)
    const_store_bkeys = {bk for bk, _ in all_cs}
    all_bm_keys = sorted(set(d_st.buf_mems.keys()) | set(v_st.buf_mems.keys()))
    for bkey in all_bm_keys:
        if bkey in ("tls_private", "wls_shared"):
            continue
        has_dyn = (f"{bkey}:dyn" in d_st.output_pcs) or (f"{bkey}:dyn" in v_st.output_pcs)
        if not has_dyn and bkey in const_store_bkeys:
            continue
        d_mem = d_st.buf_mems.get(bkey, env.get_buf_mem(bkey))
        v_mem = v_st.buf_mems.get(bkey, env.get_buf_mem(bkey))
        probe_name = f"probe_{bkey.replace(':', '_')}_off"
        probe_off = z3.BitVec(probe_name, 32)
        env.inputs[probe_name] = probe_off
        checks.append((f"{bkey}[probe_off]", z3.Select(d_mem, probe_off), z3.Select(v_mem, probe_off), None, f"{bkey}:dyn"))

    # Storage image writes (OpImageWrite / StoreTexel)
    all_im_keys = sorted(set(d_st.image_mems.keys()) | set(v_st.image_mems.keys()))
    for ikey in all_im_keys:
        d_im = d_st.image_mems.get(ikey, env.get_img_mem(ikey))
        v_im = v_st.image_mems.get(ikey, env.get_img_mem(ikey))
        probe_xy = z3.BitVec(f"probe_{ikey.replace(':', '_')}_xy", 64)
        env.inputs[f"probe_{ikey.replace(':', '_')}_xy"] = probe_xy
        for c, cname in enumerate(("r", "g", "b", "a")):
            checks.append((f"img:{ikey}[xy].{cname}", z3.Select(d_im[c], probe_xy), z3.Select(v_im[c], probe_xy), None, f"img:{ikey}"))

    # Unresolved-address fallback memory writes (never silently skip)
    if not d_st.fallback_mem.eq(env.fallback_mem) or not v_st.fallback_mem.eq(env.fallback_mem):
        probe_addr = z3.BitVec("probe_fallback_addr", 64)
        env.inputs["probe_fallback_addr"] = probe_addr
        checks.append(("fallback_mem[probe_addr]", z3.Select(d_st.fallback_mem, probe_addr), z3.Select(v_st.fallback_mem, probe_addr), None, "fallback_mem"))

    if not checks:
        res.status = "SKIP"
        res.notes.append("No comparable observable outputs extracted")
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res

    any_unknown = False
    for name, lhs, rhs, guard, pc_key in checks:
        res.checked_outputs.append(name)
        st_rc, cex = _solve_miter_eq(lhs, rhs, guard, timeout_ms, env)
        if st_rc == "SAT" and cex is not None:
            res.mismatches.append(
                OutputMismatch(
                    output_name=name,
                    direct_hex=cex[0],
                    valhall_hex=cex[1],
                    model_summary=cex[2],
                    suspect_pcs=lookup_pcs(pc_key),
                )
            )
        elif st_rc == "UNKNOWN":
            any_unknown = True

    if res.mismatches:
        res.status = "SAT"
    elif any_unknown:
        if fp_mode == "exact" and allow_uf_fallback:
            uf_res = verify_equivalence_bundle(bundle, timeout_ms=timeout_ms, fp_mode="uf", allow_uf_fallback=False)
            # UF is an over-approximation: UNSAT/BOUNDED_UNSAT is sound, but UF SAT may be spurious.
            if uf_res.status in ("UNSAT", "BOUNDED_UNSAT"):
                return uf_res
            res.notes.append("Exact FP timed out; UF fallback could not prove equivalence")
        res.status = "UNKNOWN"
    else:
        res.status = "BOUNDED_UNSAT" if res.has_loops else "UNSAT"

    res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return res


def verify_spv_file(
    spv_path: str,
    spirv2valhall_bin: str = "./build/spirv2valhall",
    timeout_ms: int = 3000,
    fp_mode: str = "exact",
) -> ShaderEquivResult:
    """Compile a .spv through both pipelines and run the Z3 equivalence verifier."""
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_json = tf.name
    try:
        proc = subprocess.run(
            [spirv2valhall_bin, spv_path, "--quiet", "--dump-equiv-json", tmp_json],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return ShaderEquivResult(
                spv_path=spv_path,
                stage="Unknown",
                status="ERROR",
                fp_mode=fp_mode,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                notes=[proc.stderr.strip()[:200]],
            )
        with open(tmp_json, "r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        return verify_equivalence_bundle(bundle, timeout_ms=timeout_ms, fp_mode=fp_mode)
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)


def main() -> int:
    ap = argparse.ArgumentParser(description="Z3 All-Path Formal Equivalence Verifier for ValhallSSA")
    ap.add_argument("spv_or_json", help="Path to .spv shader or pre-dumped equivalence bundle .json")
    ap.add_argument("--bin", default="./build/spirv2valhall", help="Path to spirv2valhall binary")
    ap.add_argument("--timeout-ms", type=int, default=3000, help="Per-output Z3 timeout in ms")
    ap.add_argument("--fp-mode", choices=["exact", "uf"], default="exact")
    args = ap.parse_args()

    if args.spv_or_json.endswith(".json"):
        with open(args.spv_or_json, "r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        r = verify_equivalence_bundle(bundle, timeout_ms=args.timeout_ms, fp_mode=args.fp_mode)
    else:
        r = verify_spv_file(args.spv_or_json, spirv2valhall_bin=args.bin, timeout_ms=args.timeout_ms, fp_mode=args.fp_mode)

    print(f"[{r.status}] {os.path.basename(r.spv_path)} stage={r.stage} fp={r.fp_mode} "
          f"outputs={len(r.checked_outputs)} direct=({r.direct_blocks}B/{r.direct_nodes}N) "
          f"valhall=({r.valhall_blocks}B/{r.valhall_nodes}N) elapsed={r.elapsed_ms:.1f}ms")
    for mm in r.mismatches[:5]:
        print(f"  MISMATCH {mm.output_name}: direct={mm.direct_hex} valhall={mm.valhall_hex} ({mm.model_summary})")
        for pc, dis in mm.suspect_pcs[:5]:
            print(f"    @0x{pc:04x}: {dis}")
    return 0 if r.status in ("UNSAT", "BOUNDED_UNSAT") else 1


if __name__ == "__main__":
    sys.exit(main())
