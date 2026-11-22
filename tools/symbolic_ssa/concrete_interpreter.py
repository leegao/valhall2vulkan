#!/usr/bin/env python3
"""
Bit-Exact Concrete SSA Interpreter (`ConcreteSsaInterpreter`).

Subclasses `SsaInterpreter` with `ConcreteDomain()` to execute ValhallSsaModule
graphs concretely against a `ConcreteShaderEnv`, achieving full opcode parity
with `SymbolicSsaInterpreter` (including Atomics, IDot4*, ImageTexelAddr, and
`sym:`/`img:` loads/stores).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from tools.symbolic_ssa.domains import ConcreteDomain
from tools.symbolic_ssa.env import (
    MASK32,
    MASK64,
    ConcreteShaderEnv,
    _det_hash_u32,
    _s32,
)
from tools.symbolic_ssa.interpreter import SsaInterpreter


@dataclasses.dataclass
class _ConcreteState:
    terminated: bool = False
    demoted: bool = False
    frag_colors: Dict[int, Tuple[int, ...]] = dataclasses.field(default_factory=dict)
    builtin_outputs: Dict[int, Any] = dataclasses.field(default_factory=dict)
    vert_varyings: Dict[int, Tuple[int, ...]] = dataclasses.field(default_factory=dict)
    buf_writes: Dict[Tuple[str, int], int] = dataclasses.field(default_factory=dict)
    sym_buf_writes: Dict[Tuple[str, int], int] = dataclasses.field(default_factory=dict)
    img_writes: Dict[Tuple[str, int], List[int]] = dataclasses.field(default_factory=dict)
    fallback_writes: Dict[int, int] = dataclasses.field(default_factory=dict)
    atomic_events: Dict[str, List[Tuple[str, int, int, int]]] = dataclasses.field(default_factory=dict)


class ConcreteSsaInterpreter(SsaInterpreter):
    """Fast concrete bit-exact interpreter subclassing SsaInterpreter with ConcreteDomain."""

    def __init__(
        self,
        ssa_json: Dict[str, Any],
        is_valhall_side: bool,
        max_block_steps: int = 256,
        domain: Optional[ConcreteDomain] = None,
    ):
        self.blocks_by_id: Dict[int, Dict[str, Any]] = {int(b["id"]): b for b in ssa_json.get("blocks", [])}
        self.nodes_by_id: Dict[int, Dict[str, Any]] = {int(n["id"]): n for n in ssa_json.get("nodes", [])}
        self.is_valhall = is_valhall_side
        self.max_block_steps = max_block_steps
        self.domain: ConcreteDomain = domain if domain is not None else ConcreteDomain()
        self._env: Optional[ConcreteShaderEnv] = None

    def image_query_size(self, lod: Any, imm_u32: int, aux_str: str) -> int:
        assert self._env is not None
        qkey = (aux_str or "img").replace(":", "_")
        return self._env.eval_hw_uf(f"hw_imgq_{qkey}_c{imm_u32}", self.domain.as_u32(lod), fallback_tag=f"imgq_{qkey}_{imm_u32}")

    def tex_sample(self, op: str, args: List[Any], imm_u32: int, aux_str: str) -> Tuple[Tuple[int, ...], Any]:
        assert self._env is not None
        uv = args[0] if args else (0, 0)
        u0 = self.domain.as_u32(uv[0]) if isinstance(uv, tuple) else self.domain.as_u32(uv)
        u1 = self.domain.as_u32(uv[1]) if isinstance(uv, tuple) and len(uv) > 1 else 0
        tkey = aux_str or f"tex_{imm_u32}"
        lanes = tuple(self._env.eval_hw_uf(f"hw_tex_{tkey}_c{c}", u0, u1, fallback_tag=f"tex_{tkey}_c{c}") for c in range(4))
        return (lanes, args[-1] if args else 0)

    def psb_load(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState) -> Tuple[int, Any]:
        assert self._env is not None
        addr = self.domain.as_u64(args[0])
        aux_str = node.get("aux_str", "")
        if aux_str.startswith("sym:"):
            bkey_hint = aux_str[4:].split("[", 1)[0]
            _, off = self._env.resolve_addr(addr, hint_key=bkey_hint)
            if (aux_str, off) in st.sym_buf_writes:
                return (st.sym_buf_writes[(aux_str, off)], args[1] if len(args) > 1 else 0)
            if f"sym_slot:{aux_str}:{off}" in self._env.overrides or f"sym_slot_default:{aux_str}" in self._env.overrides:
                return (self._env.read_sym_buf_slot(aux_str, off), args[1] if len(args) > 1 else 0)
            hint = bkey_hint
        else:
            hint = aux_str
        bkey, off = self._env.resolve_addr(addr, hint_key=hint)
        if bkey == "wls_shared" and self._env.unique_workgroup_key is not None:
            bkey = self._env.unique_workgroup_key
        if bkey is not None:
            val = st.buf_writes.get((bkey, off), self._env.read_init_buf(bkey, off))
        else:
            val = st.fallback_writes.get(addr, self._env.read_fallback_mem(addr))
        return (val, args[1] if len(args) > 1 else 0)

    def psb_store(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState, reach: Any) -> Any:
        assert self._env is not None
        if not (st.terminated or st.demoted):
            addr = self.domain.as_u64(args[0])
            val = self.domain.as_u32(args[1])
            aux_str = node.get("aux_str", "")
            if aux_str.startswith("img:"):
                ikey = aux_str
                ch_idx = 0
                xy_addr = addr
                if (addr & 0xF) in (4, 8, 12):
                    ch_idx = (addr & 0xF) // 4
                    xy_addr = addr - (addr & 0xF)
                cur = st.img_writes.get((ikey, xy_addr))
                if cur is None:
                    cur = list(self._env.read_init_img(ikey, xy_addr))
                cur[ch_idx] = val
                st.img_writes[(ikey, xy_addr)] = cur
                return args[2] if len(args) > 2 else 0
            if aux_str.startswith("sym:"):
                bkey_hint = aux_str[4:].split("[", 1)[0]
                _, off = self._env.resolve_addr(addr, hint_key=bkey_hint)
                st.sym_buf_writes[(aux_str, off)] = val
                hint = bkey_hint
            else:
                hint = aux_str
            bkey, off = self._env.resolve_addr(addr, hint_key=hint)
            if bkey == "wls_shared" and self._env.unique_workgroup_key is not None:
                bkey = self._env.unique_workgroup_key
            if bkey is not None:
                st.buf_writes[(bkey, off)] = val
            else:
                st.fallback_writes[addr] = val
        return args[2] if len(args) > 2 else 0

    def kill(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState, reach: Any) -> int:
        aux_bool = bool(node.get("aux_bool", False))
        aux_bool2 = bool(node.get("aux_bool2", False))
        kcond = self.domain.as_bool(args[0]) if (aux_bool and len(args) >= 2) else True
        if kcond:
            if aux_bool2: st.demoted = True
            else: st.terminated = True
        return 0

    def store_frag_color(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState, reach: Any) -> Any:
        if not (st.terminated or st.demoted):
            col = args[0]
            imm32 = int(node.get("imm_u32", 0))
            st.frag_colors[imm32] = tuple(self.domain.as_u32(col[c]) if isinstance(col, tuple) and c < len(col) else 0 for c in range(4))
        return args[-1] if args else 0

    def store_builtin_output(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState, reach: Any) -> Any:
        if not (st.terminated or st.demoted):
            st.builtin_outputs[int(node.get("imm_u32", 0))] = args[0]
        return args[-1] if args else 0

    def store_vert_varying(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState, reach: Any) -> Any:
        assert self._env is not None
        if not (st.terminated or st.demoted):
            imm32 = int(node.get("imm_u32", 0))
            loc = self._env.sout_addr_to_loc.get(imm32, imm32) if self.is_valhall else imm32
            vec = args[0]
            st.vert_varyings[loc] = tuple(self.domain.as_u32(vec[c]) if isinstance(vec, tuple) and c < len(vec) else 0 for c in range(4))
        return args[-1] if args else 0

    def store_texel(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState, reach: Any) -> Any:
        if not (st.terminated or st.demoted):
            coord = args[0]
            texel = args[1] if len(args) > 1 else (0, 0, 0, 0)
            cx = self.domain.as_u32(coord[0]) if isinstance(coord, tuple) and len(coord) > 0 else self.domain.as_u32(coord)
            cy = self.domain.as_u32(coord[1]) if isinstance(coord, tuple) and len(coord) > 1 else 0
            xy = ((cy << 32) | cx) & MASK64
            ikey = node.get("aux_str", "") or f"img_{int(node.get('imm_u32', 0))}"
            st.img_writes[(ikey, xy)] = [self.domain.as_u32(texel[c]) if isinstance(texel, tuple) and c < len(texel) else 0 for c in range(4)]
        return args[-1] if args else 0

    def atomic_op(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState, reach: Any) -> Tuple[int, Any]:
        assert self._env is not None
        op = node["op"]
        aux_str = node.get("aux_str", "")
        addr = self.domain.as_u64(args[0])
        arg = self.domain.as_u32(args[1]) if len(args) > 1 else 0
        cmp_val = self.domain.as_u32(args[2]) if (op == "AtomicCmpXchg" and len(args) > 2) else 0
        op_tag = aux_str.lower() if aux_str else op.lower()
        bkey, off = self._env.resolve_addr(addr, hint_key="")
        if bkey == "wls_shared" and self._env.unique_workgroup_key is not None:
            bkey = self._env.unique_workgroup_key
        if bkey is not None:
            old_val = st.buf_writes.get((bkey, off), self._env.read_init_buf(bkey, off))
        else:
            old_val = st.fallback_writes.get(addr, self._env.read_fallback_mem(addr))

        if op == "AtomicExchange": new_val = arg
        elif op == "AtomicCmpXchg": new_val = arg if old_val == cmp_val else old_val
        elif op_tag in ("aadd", "add"): new_val = (old_val + arg) & MASK32
        elif op_tag in ("asub", "sub"): new_val = (old_val - arg) & MASK32
        elif op_tag in ("aand", "and"): new_val = (old_val & arg) & MASK32
        elif op_tag in ("aor", "or"): new_val = (old_val | arg) & MASK32
        elif op_tag in ("axor", "xor"): new_val = (old_val ^ arg) & MASK32
        elif op_tag in ("asmin", "smin"): new_val = old_val if _s32(old_val) < _s32(arg) else arg
        elif op_tag in ("aumin", "umin"): new_val = old_val if old_val < arg else arg
        elif op_tag in ("asmax", "smax"): new_val = old_val if _s32(old_val) > _s32(arg) else arg
        elif op_tag in ("aumax", "umax"): new_val = old_val if old_val > arg else arg
        else: new_val = _det_hash_u32(f"atomic_post_{op_tag}", old_val, arg)

        if not (st.terminated or st.demoted):
            if bkey is not None:
                st.buf_writes[(bkey, off)] = new_val
                st.atomic_events.setdefault(bkey, []).append((op_tag, off, arg, cmp_val))
            else:
                st.fallback_writes[addr] = new_val
                st.atomic_events.setdefault("fallback_mem", []).append((op_tag, addr, arg, cmp_val))
        return (old_val, args[-1] if args else 0)

    def execute_one(self, env: ConcreteShaderEnv) -> Tuple[Dict[int, Any], Dict[str, Any]]:
        self._env = env
        vals: Dict[int, Any] = {0: 0}
        st = _ConcreteState()

        if 0 not in self.blocks_by_id:
            return vals, {}

        cur_bid = 0
        prev_bid = -1
        steps = 0
        while cur_bid in self.blocks_by_id and steps < self.max_block_steps:
            steps += 1
            blk = self.blocks_by_id[cur_bid]
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
                vals[nid] = self.eval_node(node, vals, st, True)

            if st.terminated or blk.get("is_end") or blk.get("is_discard"):
                break
            prev_bid = cur_bid
            bt = int(blk.get("branch_target", -1))
            ft = int(blk.get("fallthrough", -1))
            if blk.get("is_cond") and bt >= 0 and ft >= 0:
                cond_v = self.domain.as_bool(vals.get(int(blk.get("branch_cond", 0)), False))
                cur_bid = bt if cond_v else ft
            elif bt >= 0:
                cur_bid = bt
            elif ft >= 0:
                cur_bid = ft
            else:
                break

        outs: Dict[str, Any] = {}
        if env.stage == "Fragment" or st.terminated or st.demoted:
            outs["discarded"] = bool(st.terminated or st.demoted)
        if not (st.terminated or st.demoted):
            for rt, col in st.frag_colors.items():
                for c, cname in enumerate(("r", "g", "b", "a")):
                    outs[f"frag_color[{rt}].{cname}"] = col[c]
            for bi, payload in st.builtin_outputs.items():
                if isinstance(payload, tuple):
                    for c, v in enumerate(payload):
                        outs[f"builtin_out[{bi}][{c}]"] = self.domain.as_u32(v)
                else:
                    outs[f"builtin_out[{bi}]"] = self.domain.as_u32(payload)
            for loc, vec in st.vert_varyings.items():
                for c in range(4):
                    outs[f"vert_var[{loc}][{c}]"] = vec[c]
            for (bkey, off), v in st.buf_writes.items():
                if bkey in ("tls_private", "wls_shared") or bkey.startswith("workgroup:"):
                    continue
                outs[f"{bkey}+0x{off:x}"] = v
            for (ikey, xy), vec in st.img_writes.items():
                for c, cname in enumerate(("r", "g", "b", "a")):
                    outs[f"img:{ikey}[0x{xy:x}].{cname}"] = vec[c]
            for akey, evs in st.atomic_events.items():
                if akey in ("tls_private", "wls_shared") or akey.startswith("workgroup:"):
                    continue
                outs[f"atomic:{akey}"] = tuple(evs)
        return vals, outs
