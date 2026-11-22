#!/usr/bin/env python3
"""
Whole-CFG Symbolic Z3 Interpreter (`SymbolicSsaInterpreter` / `DagSsaZ3Evaluator`).

Subclasses `SsaInterpreter` with `SymbolicDomain(fp_mode=...)` to perform
reverse-postorder symbolic evaluation with bounded loop unrolling, per-iteration
branch decision tracking, cutpoint substitution, and incremental single-block
cone evaluation.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

import z3

from tools.symbolic_ssa.domains import SymbolicDomain
from tools.symbolic_ssa.env import (
    BV32,
    SharedShaderEnv,
    Z3Val,
    _z3_and,
    _z3_if,
    _z3_not,
    _z3_or,
)
from tools.symbolic_ssa.interpreter import SsaInterpreter
from tools.verify.panfrost_z3_model import _hw_uf, bv32


@dataclasses.dataclass
class BlockEffectState:
    """Path-conditioned side-effect state carried across CFG blocks during symbolic evaluation."""

    discarded: z3.BoolRef
    terminated: z3.BoolRef
    demoted: z3.BoolRef
    frag_colors: Dict[int, Tuple[z3.BitVecRef, ...]]
    builtin_outputs: Dict[int, Z3Val]
    vert_varyings: Dict[int, Tuple[z3.BitVecRef, ...]]
    const_stores: Dict[Tuple[str, int], Tuple[z3.BoolRef, z3.BitVecRef]]
    sym_buf_stores: Dict[str, Tuple[z3.BoolRef, z3.BitVecRef, z3.BitVecRef]]
    buf_mems: Dict[str, z3.ArrayRef]
    image_mems: Dict[str, Tuple[z3.ArrayRef, ...]]
    fallback_mem: z3.ArrayRef
    atomic_events: Dict[str, List[Tuple[z3.BoolRef, str, z3.BitVecRef, z3.BitVecRef, z3.BitVecRef]]]
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
    cond_states: List[Tuple[z3.BoolRef, BlockEffectState]],
    env: SharedShaderEnv,
) -> BlockEffectState:
    if len(cond_states) == 1:
        return _clone_state(cond_states[0][1])

    base_st = _clone_state(cond_states[0][1])
    for cond, st in cond_states[1:]:
        base_st.discarded = _z3_if(cond, st.discarded, base_st.discarded)
        base_st.terminated = _z3_if(cond, st.terminated, base_st.terminated)
        base_st.demoted = _z3_if(cond, st.demoted, base_st.demoted)

        for rt in set(base_st.frag_colors) | set(st.frag_colors):
            a = st.frag_colors.get(rt, (bv32(0),) * 4)
            b = base_st.frag_colors.get(rt, (bv32(0),) * 4)
            base_st.frag_colors[rt] = tuple(_z3_if(cond, ai, bi) for ai, bi in zip(a, b))

        for bi in set(base_st.builtin_outputs) | set(st.builtin_outputs):
            if bi in st.builtin_outputs and bi in base_st.builtin_outputs:
                base_st.builtin_outputs[bi] = _z3_if(cond, st.builtin_outputs[bi], base_st.builtin_outputs[bi])
            elif bi in st.builtin_outputs:
                base_st.builtin_outputs[bi] = st.builtin_outputs[bi]

        for loc in set(base_st.vert_varyings) | set(st.vert_varyings):
            a = st.vert_varyings.get(loc, (bv32(0),) * 4)
            b = base_st.vert_varyings.get(loc, (bv32(0),) * 4)
            base_st.vert_varyings[loc] = tuple(_z3_if(cond, ai, bi) for ai, bi in zip(a, b))

        for key in set(base_st.const_stores) | set(st.const_stores):
            bkey, off = key
            def_v = z3.Select(env.get_buf_mem(bkey), bv32(off))
            aw, av = st.const_stores.get(key, (z3.BoolVal(False), def_v))
            bw, bv_ = base_st.const_stores.get(key, (z3.BoolVal(False), def_v))
            base_st.const_stores[key] = (_z3_if(cond, aw, bw), _z3_if(cond, av, bv_))

        for skey in set(base_st.sym_buf_stores) | set(st.sym_buf_stores):
            if skey in st.sym_buf_stores and skey in base_st.sym_buf_stores:
                aw, av, ao = st.sym_buf_stores[skey]
                bw, bv_, bo = base_st.sym_buf_stores[skey]
                base_st.sym_buf_stores[skey] = (_z3_if(cond, aw, bw), _z3_if(cond, av, bv_), _z3_if(cond, ao, bo))
            elif skey in st.sym_buf_stores:
                aw, av, ao = st.sym_buf_stores[skey]
                def_v = env.get_sym_buf_slot(skey, ao)
                base_st.sym_buf_stores[skey] = (_z3_and(cond, aw), _z3_if(cond, av, def_v), ao)
            else:
                bw, bv_, bo = base_st.sym_buf_stores[skey]
                def_v = env.get_sym_buf_slot(skey, bo)
                base_st.sym_buf_stores[skey] = (_z3_and(_z3_not(cond), bw), _z3_if(cond, def_v, bv_), bo)

        for bkey in set(base_st.buf_mems) | set(st.buf_mems):
            a_m = st.buf_mems.get(bkey, env.get_buf_mem(bkey))
            b_m = base_st.buf_mems.get(bkey, env.get_buf_mem(bkey))
            if not a_m.eq(b_m):
                base_st.buf_mems[bkey] = z3.If(cond, a_m, b_m)

        for ikey in set(base_st.image_mems) | set(st.image_mems):
            a_im = st.image_mems.get(ikey, env.get_img_mem(ikey))
            b_im = base_st.image_mems.get(ikey, env.get_img_mem(ikey))
            base_st.image_mems[ikey] = tuple(
                z3.If(cond, ac, bc) if not ac.eq(bc) else ac
                for ac, bc in zip(a_im, b_im)
            )

        if not st.fallback_mem.eq(base_st.fallback_mem):
            base_st.fallback_mem = z3.If(cond, st.fallback_mem, base_st.fallback_mem)

        for akey in set(base_st.atomic_events) | set(st.atomic_events):
            b_evs = base_st.atomic_events.get(akey, [])
            s_evs = st.atomic_events.get(akey, [])
            if b_evs == s_evs:
                continue
            merged_evs = [(_z3_and(_z3_not(cond), m), op, off, arg, cmp_v) for (m, op, off, arg, cmp_v) in b_evs]
            merged_evs.extend((_z3_and(cond, m), op, off, arg, cmp_v) for (m, op, off, arg, cmp_v) in s_evs)
            base_st.atomic_events[akey] = merged_evs

        for k, pcs in st.output_pcs.items():
            base_st.output_pcs.setdefault(k, set()).update(pcs)
    return base_st


class SymbolicSsaInterpreter(SsaInterpreter):
    """Whole-CFG RPO symbolic Z3 interpreter for ValhallSsaModule."""

    def __init__(
        self,
        ssa_json: Dict[str, Any],
        env: SharedShaderEnv,
        is_valhall_side: bool,
        fp_mode: str = "exact",
        max_unroll: int = 1,
        cutpoints: Optional[Dict[int, Any]] = None,
        domain: Optional[SymbolicDomain] = None,
    ):
        self.ssa = ssa_json
        self.env = env
        self._env = env
        self.is_valhall = is_valhall_side
        self.fp_mode = fp_mode
        self.domain: SymbolicDomain = domain if domain is not None else SymbolicDomain(fp_mode=fp_mode)
        self.max_unroll = max(1, int(max_unroll))
        self.cutpoints: Dict[int, Any] = cutpoints if cutpoints is not None else {}
        self.blocks: List[Dict[str, Any]] = ssa_json.get("blocks", [])
        self.nodes_by_id: Dict[int, Dict[str, Any]] = {int(n["id"]): n for n in ssa_json.get("nodes", [])}
        self.values: Dict[int, Z3Val] = {0: bv32(0)}
        self.block_reach: Dict[int, z3.BoolRef] = {}
        self.iter_branch_decisions: List[Tuple[int, int, int, z3.BoolRef, z3.BoolRef]] = []
        self.has_loops: bool = False
        self.max_unroll_depth: int = 0

    def image_query_size(self, lod: Any, imm_u32: int, aux_str: str) -> z3.BitVecRef:
        qkey = (aux_str or "img").replace(":", "_")
        return _hw_uf(f"imgq_{qkey}_c{imm_u32}", BV32, BV32)(self.domain.as_u32(lod))

    def tex_sample(self, op: str, args: List[Any], imm_u32: int, aux_str: str) -> Tuple[Tuple[z3.BitVecRef, ...], Z3Val]:
        uv = args[0] if args else (bv32(0), bv32(0))
        u0 = self.domain.as_u32(uv[0]) if isinstance(uv, tuple) else self.domain.as_u32(uv)
        u1 = self.domain.as_u32(uv[1]) if isinstance(uv, tuple) and len(uv) > 1 else bv32(0)
        tkey = aux_str or f"tex_{imm_u32}"
        lanes = tuple(_hw_uf(f"tex_{tkey}_c{c}", BV32, BV32, BV32)(u0, u1) for c in range(4))
        return (lanes, args[-1] if args else bv32(0))

    def psb_load(self, node: Dict[str, Any], args: List[Any], st: BlockEffectState) -> Tuple[z3.BitVecRef, Z3Val]:
        addr = self.domain.as_u64(args[0])
        aux_str = node.get("aux_str", "")
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

    def psb_store(self, node: Dict[str, Any], args: List[Any], st: BlockEffectState, reach: z3.BoolRef) -> Z3Val:
        addr = self.domain.as_u64(args[0])
        val = self.domain.as_u32(args[1])
        store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
        nid = int(node["id"])
        aux_str = node.get("aux_str", "")
        eff_hint = aux_str
        if aux_str.startswith("sym:"):
            bkey_hint = aux_str[4:].split("[", 1)[0]
            if self.env.is_shared_sym_bkey(bkey_hint):
                _, _, off32 = self.env.resolve_addr(addr, hint_key=bkey_hint)
                prev_w, prev_v, prev_o = st.sym_buf_stores.get(
                    aux_str, (z3.BoolVal(False), self.env.get_sym_buf_slot(aux_str, off32), off32)
                )
                st.sym_buf_stores[aux_str] = (
                    _z3_or(prev_w, store_mask), _z3_if(store_mask, val, prev_v), _z3_if(store_mask, off32, prev_o)
                )
                st.output_pcs.setdefault(aux_str, set()).add(nid)
                return args[2] if len(args) > 2 else bv32(0)
            eff_hint = bkey_hint
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

    def kill(self, node: Dict[str, Any], args: List[Any], st: BlockEffectState, reach: z3.BoolRef) -> z3.BitVecRef:
        aux_bool = bool(node.get("aux_bool", False))
        cond = self.domain.as_bool(args[0]) if (aux_bool and len(args) >= 2) else z3.BoolVal(True)
        is_demote = bool(node.get("aux_bool2", False))
        eff = _z3_and(reach, cond)
        if is_demote:
            st.demoted = _z3_or(st.demoted, eff)
        else:
            st.terminated = _z3_or(st.terminated, eff)
        st.discarded = _z3_or(st.terminated, st.demoted)
        st.output_pcs.setdefault("discarded", set()).add(int(node["id"]))
        return bv32(0)

    def store_frag_color(self, node: Dict[str, Any], args: List[Any], st: BlockEffectState, reach: z3.BoolRef) -> Z3Val:
        imm_u32 = int(node.get("imm_u32", 0))
        col = args[0]
        store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
        vec4 = tuple(self.domain.as_u32(col[c]) if isinstance(col, tuple) and c < len(col) else bv32(0) for c in range(4))
        prev = st.frag_colors.get(imm_u32, (bv32(0),) * 4)
        st.frag_colors[imm_u32] = tuple(_z3_if(store_mask, n, p) for n, p in zip(vec4, prev))
        st.output_pcs.setdefault(f"frag_color[{imm_u32}]", set()).add(int(node["id"]))
        return args[-1] if args else bv32(0)

    def store_builtin_output(self, node: Dict[str, Any], args: List[Any], st: BlockEffectState, reach: z3.BoolRef) -> Z3Val:
        imm_u32 = int(node.get("imm_u32", 0))
        payload = args[0]
        store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
        if imm_u32 in st.builtin_outputs:
            st.builtin_outputs[imm_u32] = _z3_if(store_mask, payload, st.builtin_outputs[imm_u32])
        else:
            st.builtin_outputs[imm_u32] = payload
        st.output_pcs.setdefault(f"builtin_out[{imm_u32}]", set()).add(int(node["id"]))
        return args[-1] if args else bv32(0)

    def store_vert_varying(self, node: Dict[str, Any], args: List[Any], st: BlockEffectState, reach: z3.BoolRef) -> Z3Val:
        imm_u32 = int(node.get("imm_u32", 0))
        loc = self.env.sout_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
        store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
        vec = args[0]
        vec4 = tuple(self.domain.as_u32(vec[c]) if isinstance(vec, tuple) and c < len(vec) else bv32(0) for c in range(4))
        prev = st.vert_varyings.get(loc, (bv32(0),) * 4)
        st.vert_varyings[loc] = tuple(_z3_if(store_mask, n, p) for n, p in zip(vec4, prev))
        st.output_pcs.setdefault(f"vert_var[{loc}]", set()).add(int(node["id"]))
        return args[-1] if args else bv32(0)

    def store_texel(self, node: Dict[str, Any], args: List[Any], st: BlockEffectState, reach: z3.BoolRef) -> Z3Val:
        imm_u32 = int(node.get("imm_u32", 0))
        aux_str = node.get("aux_str", "")
        coord = args[0]
        store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
        texel = args[1] if len(args) > 1 else (bv32(0),) * 4
        cx = self.domain.as_u32(coord[0]) if isinstance(coord, tuple) and len(coord) > 0 else self.domain.as_u32(coord)
        cy = self.domain.as_u32(coord[1]) if isinstance(coord, tuple) and len(coord) > 1 else bv32(0)
        xy = z3.Concat(cy, cx)
        vec4 = tuple(self.domain.as_u32(texel[c]) if isinstance(texel, tuple) and c < len(texel) else bv32(0) for c in range(4))
        ikey = aux_str or f"img_{imm_u32}"
        cur_im = st.image_mems.get(ikey, self.env.get_img_mem(ikey))
        st.image_mems[ikey] = tuple(
            z3.If(store_mask, z3.Store(cur_im[c], xy, vec4[c]), cur_im[c]) if not z3.is_true(store_mask)
            else z3.Store(cur_im[c], xy, vec4[c])
            for c in range(4)
        )
        st.output_pcs.setdefault(f"img:{ikey}", set()).add(int(node["id"]))
        return args[-1] if args else bv32(0)

    def atomic_op(self, node: Dict[str, Any], args: List[Any], st: BlockEffectState, reach: z3.BoolRef) -> Tuple[z3.BitVecRef, Z3Val]:
        op = node["op"]
        aux_str = node.get("aux_str", "")
        addr = self.domain.as_u64(args[0])
        arg = self.domain.as_u32(args[1]) if len(args) > 1 else bv32(0)
        cmp_val = self.domain.as_u32(args[2]) if (op == "AtomicCmpXchg" and len(args) > 2) else bv32(0)
        store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
        op_tag = aux_str.lower() if aux_str else op.lower()
        bkey, _, off32 = self.env.resolve_addr(addr, hint_key="")
        if bkey == "wls_shared" and self.env.unique_workgroup_key is not None:
            bkey = self.env.unique_workgroup_key
        nid = int(node["id"])
        if bkey is not None:
            cur_mem = st.buf_mems.get(bkey, self.env.get_buf_mem(bkey))
            old_val = z3.Select(cur_mem, off32)
        else:
            old_val = z3.Select(st.fallback_mem, addr)

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

    def _eval_node(self, node: Dict[str, Any], st: BlockEffectState, reach: z3.BoolRef) -> Z3Val:
        nid = int(node["id"])
        if nid in self.cutpoints:
            return self.cutpoints[nid]
        return self.eval_node(node, self.values, st, reach)

    def collect_pcs_for_nodes(self, root_nids: Set[int], max_visit: int = 2048) -> List[int]:
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

    def evaluate_single_block_cone(self, target_nid: int, st: BlockEffectState) -> Z3Val:
        """
        Incrementally evaluate only the uncached backward cone of `target_nid`
        on a single-block DAG, stopping at already-cached values or cutpoints.
        """
        if target_nid in self.cutpoints:
            return self.cutpoints[target_nid]
        if target_nid in self.values:
            return self.values[target_nid]

        order: List[int] = []
        visited: Set[int] = set()
        stack: List[Tuple[int, int]] = [(target_nid, 0)]
        while stack:
            nid, step = stack.pop()
            if nid in self.values or nid in self.cutpoints:
                continue
            if step == 0:
                if nid in visited:
                    continue
                visited.add(nid)
                node = self.nodes_by_id.get(nid)
                if not node:
                    continue
                stack.append((nid, 1))
                for aid in reversed(node.get("args", [])):
                    ia = int(aid)
                    if ia not in self.values and ia not in self.cutpoints and ia not in visited:
                        stack.append((ia, 0))
            else:
                order.append(nid)

        reach = z3.BoolVal(True)
        for nid in order:
            node = self.nodes_by_id[nid]
            self.values[nid] = self._eval_node(node, st, reach)
        return self.values.get(target_nid, bv32(0))

    def invalidate_cone_from(self, root_nid: int, users_map: Dict[int, List[int]]) -> None:
        """Invalidate cached symbolic values downstream of a newly installed cutpoint."""
        queue = [root_nid]
        seen: Set[int] = set()
        while queue:
            cur = queue.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur != 0:
                self.values.pop(cur, None)
            for u in users_map.get(cur, ()):
                if u not in seen:
                    queue.append(u)

    def evaluate(self) -> BlockEffectState:
        """Evaluate all blocks in RPO topological order with bounded loop unrolling."""
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
        self.iter_branch_decisions = []
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
        edge_conds: Dict[int, List[Tuple[int, z3.BoolRef]]] = {int(b["id"]): [] for b in self.blocks}
        exit_states: List[Tuple[z3.BoolRef, BlockEffectState]] = []

        def _eval_block_slice(
            block_list: List[Dict[str, Any]],
            allow_backedges: bool,
            iter_idx: int,
        ) -> List[Tuple[int, int, z3.BoolRef]]:
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

                blk_reach[bid] = _z3_or(blk_reach[bid], reach) if bid in blk_reach else reach

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
                    bcond = self.domain.as_bool(self.values.get(bcond_id, z3.BoolVal(False)))
                    self.iter_branch_decisions.append((iter_idx, bid, bcond_id, cf_reach, bcond))
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

        backedges = _eval_block_slice(rpo_blocks, allow_backedges=False, iter_idx=0)
        self.max_unroll_depth = 1 if self.has_loops else 0

        unroll_iter = 1
        while backedges and unroll_iter < self.max_unroll:
            unroll_iter += 1
            self.max_unroll_depth = unroll_iter
            min_hdr_rpo = min(rpo_index.get(hdr, 0) for _, hdr, _ in backedges)
            for src_b, hdr, bcond in backedges:
                edge_conds.setdefault(hdr, []).append((src_b, bcond))
            loop_slice = [b for b in rpo_blocks if rpo_index[int(b["id"])] >= min_hdr_rpo]
            exit_states.clear()
            backedges = _eval_block_slice(loop_slice, allow_backedges=True, iter_idx=unroll_iter - 1)

        self.block_reach = blk_reach
        if not exit_states:
            return blk_out_state.get(len(self.blocks) - 1, init_state)
        return _merge_states(exit_states, self.env)


DagSsaZ3Evaluator = SymbolicSsaInterpreter
