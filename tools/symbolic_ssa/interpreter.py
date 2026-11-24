"""
Abstract Denotational SSA Interpreter (`SsaInterpreter`) & Reusable Graph/Dataflow Engines.

Provides:
  - Canonical opcode dispatch (`eval_node`) delegating pure value semantics to
    `self.domain` (`SsaDomain`) and memory/output/kill/atomic semantics to
    minimal state-backend hooks (`_active_mask`, `_read_buf`, `_write_buf`,
    `_write_img_channel`, `_write_img_texel`, `_record_kill`,
    `_write_frag_color`, `_write_builtin_output`, `_write_vert_varying`,
    `_record_atomic`), eliminating DRY duplication across Concrete, Symbolic,
    and instruction-level (`ssa_z3_eval`) interpreters.
  - Standard CFG/SSA traversal & dataflow engines (`compute_rpo`, `compute_idom`,
    `walk_dom_tree`, `run_dfs`, `run_rpo`, `run_fixpoint`,
    `run_backward_worklist`) usable by all interpreters and canonicalization passes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import (
    Any,
    Generic,
    TypeVar,
)

V = TypeVar("V")

from tools.symbolic_ssa.domains import AbstractTopDomain, SsaDomain

INT32_UNARY_OPS = frozenset({"BitNot", "SNeg", "SAbs", "Clz", "BitCount", "BitReverse"})
INT32_BINARY_OPS = frozenset({
    "IAdd", "ISub", "IMul", "BitAnd", "BitOr", "BitXor",
    "Shl", "ShrU", "ShrS", "RotL", "RotR",
    "UDiv", "SDiv", "UMod", "SRem", "SMod",
})
INT64_BINARY_OPS = frozenset({
    "IAdd64", "ISub64", "IMul64", "BitAnd64", "BitOr64", "BitXor64",
    "Shl64", "ShrU64", "ShrS64",
})
QUAD_DERIV_OPS = frozenset({"DPdx", "DPdy", "Fwidth"})
INT_DIV_OPS = frozenset({"UDiv", "SDiv", "UMod", "SRem", "SMod"})
FP_UNARY_OPS = frozenset({
    "FAbs", "FNeg", "Floor", "Ceil", "Trunc", "RoundEven", "Sqrt", "InverseSqrt",
    "Sin", "Cos", "Exp", "Exp2", "Log", "Log2", "Atan", "Asin", "Acos",
    "Sinh", "Cosh", "Tanh", "FrexpExp", "FrexpSignificand",
})
FP_BINARY_OPS = frozenset({
    "FAdd", "FSub", "FMul", "FDiv", "FMin", "FMax", "NMin", "NMax", "Pow", "Ldexp", "Atan2",
})
CONVERT_OPS = frozenset({
    "ConvertFToS", "ConvertFToU", "ConvertSToF", "ConvertUToF",
    "SConvert32To64", "UConvert32To64", "UConvert64To32",
})
TEX_SAMPLE_OPS = frozenset({
    "TexSample", "TexSampleLod", "TexSampleDref", "TexGather", "TexFetch", "ImageQueryLod",
})
ATOMIC_OPS = frozenset({"AtomicRmw", "AtomicExchange", "AtomicCmpXchg"})
CONTROL_PASSTHROUGH_OPS = frozenset({"Branch", "BranchCond", "Return"})


class SsaInterpreter(Generic[V]):
    """
    Unified denotational interpreter and graph-traversal engine for ValhallSsaModule.
    """

    domain: SsaDomain[V] = AbstractTopDomain()  # type: ignore[assignment]
    _env: Any = None
    is_valhall: bool = False

    # ------------------------------------------------------------------
    # Reusable CFG & SSA traversal / dataflow primitives
    # ------------------------------------------------------------------
    @staticmethod
    def block_succs(blk: dict[str, Any]) -> list[int]:
        if blk.get("is_end") or blk.get("is_discard"):
            return []
        bt = int(blk.get("branch_target", -1))
        ft = int(blk.get("fallthrough", -1))
        if blk.get("is_cond"):
            res = []
            if bt >= 0: res.append(bt)
            if ft >= 0 and ft != bt: res.append(ft)
            return res
        if blk.get("is_uncond"):
            return [bt] if bt >= 0 else []
        if bt >= 0: return [bt]
        if ft >= 0: return [ft]
        return []

    @classmethod
    def compute_rpo(cls, blocks_by_id: dict[int, dict[str, Any]], entry: int = 0) -> list[int]:
        if entry not in blocks_by_id:
            return []
        visited: set[int] = set()
        postorder: list[int] = []
        stack: list[tuple[int, int]] = [(entry, 0)]
        while stack:
            u, step = stack.pop()
            if step == 0:
                if u in visited:
                    continue
                visited.add(u)
                stack.append((u, 1))
                for s in reversed(cls.block_succs(blocks_by_id[u])):
                    if s in blocks_by_id and s not in visited:
                        stack.append((s, 0))
            else:
                postorder.append(u)
        return list(reversed(postorder))

    @staticmethod
    def compute_idom(blocks_by_id: dict[int, dict[str, Any]], rpo: list[int]) -> dict[int, int]:
        if not rpo:
            return {}
        entry = rpo[0]
        rpo_idx = {b: i for i, b in enumerate(rpo)}
        idom: dict[int, int] = {entry: entry}

        def intersect(b1: int, b2: int) -> int:
            f1, f2 = b1, b2
            while f1 != f2:
                while rpo_idx[f1] > rpo_idx[f2]:
                    f1 = idom[f1]
                while rpo_idx[f2] > rpo_idx[f1]:
                    f2 = idom[f2]
            return f1

        changed = True
        while changed:
            changed = False
            for b in rpo[1:]:
                preds = [int(p) for p in blocks_by_id[b].get("preds", []) if int(p) in idom]
                if not preds:
                    continue
                new_idom = preds[0]
                for p in preds[1:]:
                    new_idom = intersect(p, new_idom)
                if idom.get(b) != new_idom:
                    idom[b] = new_idom
                    changed = True
        return idom

    @classmethod
    def walk_dom_tree(
        cls,
        blocks_by_id: dict[int, dict[str, Any]],
        rpo: list[int],
        on_enter: Callable[[int], Any],
        on_exit: Callable[[int, Any], None] | None = None,
    ) -> None:
        """Pre-order / post-order walk of the dominator tree rooted at `rpo[0]`."""
        if not rpo:
            return
        idom = cls.compute_idom(blocks_by_id, rpo)
        children: dict[int, list[int]] = {b: [] for b in rpo}
        for b in rpo[1:]:
            p = idom.get(b)
            if p is not None and p != b:
                children[p].append(b)

        def dfs(bid: int) -> None:
            token = on_enter(bid)
            for ch in children.get(bid, ()):
                dfs(ch)
            if on_exit is not None:
                on_exit(bid, token)

        dfs(rpo[0])

    @classmethod
    def run_dfs(
        cls,
        blocks_by_id: dict[int, dict[str, Any]],
        entry: int = 0,
        on_enter: Callable[[int], None] | None = None,
        on_exit: Callable[[int], None] | None = None,
    ) -> list[int]:
        """Iterative DFS returning postorder block IDs."""
        if entry not in blocks_by_id:
            return []
        visited: set[int] = set()
        postorder: list[int] = []
        stack: list[tuple[int, int]] = [(entry, 0)]
        while stack:
            u, step = stack.pop()
            if step == 0:
                if u in visited:
                    continue
                visited.add(u)
                if on_enter is not None:
                    on_enter(u)
                stack.append((u, 1))
                for s in reversed(cls.block_succs(blocks_by_id[u])):
                    if s in blocks_by_id and s not in visited:
                        stack.append((s, 0))
            else:
                if on_exit is not None:
                    on_exit(u)
                postorder.append(u)
        return postorder

    @staticmethod
    def run_backward_worklist(
        nodes_by_id: dict[int, dict[str, Any]],
        roots: Iterable[int],
        on_visit: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> set[int]:
        """Standard backward dataflow/liveness worklist from root SSA node IDs."""
        visited: set[int] = set()
        worklist: list[int] = []

        def mark(nid: int) -> None:
            if nid > 0 and nid not in visited and nid in nodes_by_id:
                visited.add(nid)
                worklist.append(nid)

        for r in roots:
            mark(int(r))

        while worklist:
            cur = worklist.pop()
            n = nodes_by_id[cur]
            if on_visit is not None:
                on_visit(cur, n)
            for a in n.get("args", []):
                mark(int(a))
            for _, v in n.get("phi", []):
                mark(int(v))
        return visited

    def run_rpo_pass(
        self,
        blocks_by_id: dict[int, dict[str, Any]],
        nodes_by_id: dict[int, dict[str, Any]],
        values: dict[int, Any],
        eval_phi: Callable[[dict[str, Any]], Any] | None = None,
        state: Any = None,
    ) -> dict[int, Any]:
        """Single forward RPO evaluation pass over all reachable blocks."""
        for bid in self.compute_rpo(blocks_by_id, 0):
            blk = blocks_by_id[bid]
            for pid in blk.get("phis", []):
                pn = nodes_by_id.get(pid)
                if pn is not None and eval_phi is not None:
                    values[pid] = eval_phi(pn)
            for nid in blk.get("nodes", []):
                n = nodes_by_id.get(nid)
                if n is not None:
                    values[nid] = self.eval_node(n, values, state)
        return values

    def run_fixpoint(
        self,
        blocks_by_id: dict[int, dict[str, Any]],
        nodes_by_id: dict[int, dict[str, Any]],
        values: dict[int, Any],
        eval_phi: Callable[[dict[str, Any]], Any],
        join_fn: Callable[[Any, Any], Any],
        max_iters: int = 32,
        state: Any = None,
    ) -> dict[int, Any]:
        """Chaotic RPO fixpoint iteration until all SSA lattice values stabilize."""
        rpo = self.compute_rpo(blocks_by_id, 0)
        for _ in range(max_iters):
            changed = False
            for bid in rpo:
                blk = blocks_by_id[bid]
                for pid in blk.get("phis", []):
                    pn = nodes_by_id.get(pid)
                    if pn is None:
                        continue
                    old_v = values.get(pid, self.default_value())
                    new_v = join_fn(old_v, eval_phi(pn))
                    if pid not in values or new_v != old_v:
                        values[pid] = new_v
                        changed = True
                for nid in blk.get("nodes", []):
                    n = nodes_by_id.get(nid)
                    if n is None:
                        continue
                    old_v = values.get(nid, self.default_value())
                    new_v = self.eval_node(n, values, state)
                    if nid not in values or new_v != old_v:
                        values[nid] = new_v
                        changed = True
            if not changed:
                break
        return values

    # ------------------------------------------------------------------
    # Domain-delegated value & environment operations
    # ------------------------------------------------------------------
    def default_value(self) -> Any: return self.domain.default_value()
    def lookup_arg(self, values: dict[int, Any], aid: int) -> Any: return values.get(aid, self.default_value())
    def const_u32(self, imm: int) -> Any: return self.domain.const_u32(imm)
    def const_u64(self, imm: int) -> Any: return self.domain.const_u64(imm)
    def const_bool(self, val: bool) -> Any: return self.domain.const_bool(val)
    def cast_value(self, op: str, val: Any, val_type: str) -> Any: return self.domain.cast_value(op, val, val_type)
    def pack64(self, lo: Any, hi: Any) -> Any: return self.domain.pack64(lo, hi)
    def split64(self, val64: Any) -> tuple[Any, Any]: return self.domain.split64(val64)
    def unpack_half2x16(self, u: Any) -> tuple[Any, Any]: return self.domain.unpack_half2x16(u)
    def pack_half2x16(self, v2: Any) -> Any: return self.domain.pack_half2x16(v2)
    def int32_unary(self, op: str, x: Any) -> Any: return self.domain.int32_unary(op, x)
    def int32_binary(self, op: str, a: Any, b: Any) -> Any: return self.domain.int32_binary(op, a, b)
    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> Any: return self.domain.bfe(signed, x, off, width)
    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> Any: return self.domain.bfi(base, insert, off, width)
    def idot4(self, op: str, a: Any, b: Any) -> Any: return self.domain.idot4(op, a, b)
    def int64_unary(self, op: str, x: Any) -> Any: return self.domain.int64_unary(op, x)
    def int64_binary(self, op: str, a: Any, b: Any) -> Any: return self.domain.int64_binary(op, a, b)
    def icmp(self, op: str, a: Any, b: Any) -> Any: return self.domain.icmp(op, a, b)
    def logical_and(self, a: Any, b: Any) -> Any: return self.domain.logical_and(a, b)
    def logical_or(self, a: Any, b: Any) -> Any: return self.domain.logical_or(a, b)
    def logical_not(self, a: Any) -> Any: return self.domain.logical_not(a)
    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any: return self.domain.select(cond, t_val, f_val)
    def fp_unary(self, op: str, x: Any) -> Any: return self.domain.fp_unary(op, x)
    def fp_binary(self, op: str, a: Any, b: Any) -> Any: return self.domain.fp_binary(op, a, b)
    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> Any: return self.domain.fp_clamp(x, lo, hi)
    def fp_fma(self, a: Any, b: Any, c: Any) -> Any: return self.domain.fp_fma(a, b, c)
    def fp_is_nan(self, x: Any) -> Any: return self.domain.fp_is_nan(x)
    def fp_is_inf(self, x: Any) -> Any: return self.domain.fp_is_inf(x)
    def fp_cmp(self, op: str, a: Any, b: Any) -> Any: return self.domain.fp_cmp(op, a, b)
    def convert(self, op: str, x: Any) -> Any: return self.domain.convert(op, x)

    def load_fau(self, word_idx: int) -> Any:
        return self._env.load_fau_word(word_idx) if self._env is not None else self.default_value()

    def load_sideband_u64(self, imm_u32: int, aux_str: str) -> Any:
        if self._env is None:
            return self.default_value()
        if aux_str and (
            aux_str.startswith(("ssbo:", "ubo:", "workgroup:"))
            or aux_str in ("push_const", "gl_SamplePosition")
        ):
            return self.const_u64(self._env.get_buf_base(aux_str))
        if imm_u32 == 2:
            return self.const_u64(self._env.get_buf_base("pilot_srt_0"))
        if imm_u32 == 3:
            return self.const_u64(self._env.get_buf_base("pilot_fau_0"))
        if imm_u32 == 6:
            return self.const_u64(self._env.get_buf_base("tls_private"))
        if imm_u32 == 7:
            return self.const_u64(self._env.get_buf_base("wls_shared"))
        return self.domain.unknown_sideband_u64(self._env, imm_u32)

    def load_sideband_u32(self, imm_u32: int, imm_u32_2: int) -> Any:
        return self.domain.sideband_u32(self._env, imm_u32, imm_u32_2) if self._env is not None else self.default_value()

    def load_builtin(self, builtin_id: int, val_type: str) -> Any:
        return self._env.get_builtin(builtin_id, val_type) if self._env is not None else self.default_value()

    def load_vert_attr(self, imm_u32: int) -> Any:
        if self._env is None:
            return self.default_value()
        loc = self._env.sinp_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
        return self._env.get_input_vec4("vert_attr", loc)

    def load_frag_varying(self, imm_u32: int) -> Any:
        if self._env is None:
            return self.default_value()
        loc = self._env.sinp_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
        return self._env.get_input_vec4("frag_var", loc)

    def image_query_size(self, lod: Any, imm_u32: int, aux_str: str) -> Any:
        if self._env is None:
            return self.default_value()
        qkey = (aux_str or "img").replace(":", "_")
        return self.domain.hw_uf(self._env, f"imgq_{qkey}_c{imm_u32}", lod, fallback_tag=f"imgq_{qkey}_{imm_u32}")

    def tex_sample(self, op: str, args: list[Any], imm_u32: int, aux_str: str, imm_u32_2: int = 0) -> Any:
        if self._env is None:
            return (tuple(self.default_value() for _ in range(4)), args[-1] if args else self.default_value())
        data_args = args[:-1] if len(args) > 1 else list(args)
        uf_args: list[Any] = []
        if data_args:
            uv = data_args[0]
            if isinstance(uv, tuple):
                uf_args.append(self.domain.as_u32(uv[0]) if len(uv) > 0 else self.const_u32(0))
                uf_args.append(self.domain.as_u32(uv[1]) if len(uv) > 1 else self.const_u32(0))
                for extra_uv in uv[2:]:
                    uf_args.append(self.domain.as_u32(extra_uv))
            else:
                uf_args.append(self.domain.as_u32(uv))
                uf_args.append(self.const_u32(0))
            for extra in data_args[1:]:
                if isinstance(extra, tuple):
                    for lane in extra:
                        uf_args.append(self.domain.as_u32(lane))
                else:
                    uf_args.append(self.domain.as_u32(extra))
        else:
            uf_args.extend([self.const_u32(0), self.const_u32(0)])
        if op == "TexGather" and imm_u32_2 != 0 and len(data_args) <= 1:
            uf_args.append(self.const_u32(imm_u32_2))
        tkey = aux_str or f"tex_{imm_u32}"
        op_tag = op.lower()
        arity = len(uf_args)
        lanes = tuple(
            self.domain.hw_uf(
                self._env,
                f"tex_{op_tag}_{tkey}_a{arity}_c{c}",
                *uf_args,
                fallback_tag=f"tex_{op_tag}_{tkey}_c{c}",
            )
            for c in range(4)
        )
        return (lanes, args[-1] if args else self.const_u32(0))

    def subgroup_ballot(self, x: Any) -> Any:
        return self.domain.subgroup_ballot(self._env, x) if self._env is not None else self.default_value()

    def subgroup_shuffle(self, x: Any, lane: Any) -> Any:
        return self.domain.subgroup_shuffle(self._env, x, lane) if self._env is not None else self.default_value()

    # ------------------------------------------------------------------
    # Unified stateful memory / kill / atomic / output semantics
    # ------------------------------------------------------------------
    def _active_mask(self, state: Any, reach: Any) -> Any: return self.const_bool(False)
    def _resolve_buf_addr(self, addr: Any, hint_key: str) -> tuple[str | None, int | None, Any]: return (None, None, self.const_u32(0))
    def _read_sym_slot(self, state: Any, sym_tag: str, off: Any) -> Any | None: return None
    def _write_sym_slot(self, state: Any, mask: Any, sym_tag: str, off: Any, val: Any, nid: int) -> bool: return False
    def _read_buf(self, state: Any, bkey: str | None, const_off: int | None, off: Any, addr: Any) -> Any: return self.default_value()
    def _read_buf_sized(self, state: Any, bkey: str | None, const_off: int | None, off: Any, addr: Any, width: int) -> Any:
        return self._read_buf(state, bkey, const_off, off, addr)
    def _write_buf(self, state: Any, mask: Any, bkey: str | None, const_off: int | None, off: Any, addr: Any, val: Any, nid: int) -> None: pass
    def _write_buf_sized(self, state: Any, mask: Any, bkey: str | None, const_off: int | None, off: Any, addr: Any, val: Any, nid: int, width: int) -> None:
        self._write_buf(state, mask, bkey, const_off, off, addr, val, nid)
    def _extract_img_channel(self, addr: Any) -> tuple[int, Any]: return (0, addr)
    def _write_img_channel(self, state: Any, mask: Any, ikey: str, xy_addr: Any, ch_idx: int, val: Any, nid: int) -> None: pass
    def _write_img_texel(self, state: Any, mask: Any, ikey: str, xy: Any, vec4: tuple[Any, ...], nid: int) -> None: pass
    def _record_kill(self, state: Any, eff: Any, is_demote: bool, nid: int) -> None: pass
    def _write_frag_color(self, state: Any, mask: Any, rt: int, vec4: tuple[Any, ...], nid: int) -> None: pass
    def _write_builtin_output(self, state: Any, mask: Any, bi: int, payload: Any, nid: int) -> None: pass
    def _write_vert_varying(self, state: Any, mask: Any, loc: int, vec4: tuple[Any, ...], nid: int) -> None: pass
    def _record_atomic(self, state: Any, mask: Any, bkey: str | None, off: Any, addr: Any, op_tag: str, arg: Any, cmp_val: Any, new_val: Any, nid: int) -> None: pass
    def _havoc_atomic_old_val(self, state: Any, bkey: str | None, const_off: int | None, off: Any, addr: Any, op_tag: str, nid: int) -> Any:
        return self._read_buf(state, bkey, const_off, off, addr)
    def _on_control_barrier(self, state: Any, mask: Any, nid: int) -> None: pass
    def _record_div_guard(self, state: Any, reach: Any, op: str, denom: Any) -> None: pass

    def quad_deriv(self, op: str, x: Any, state: Any) -> Any:
        return self.fp_unary(op, x)

    def control_barrier(self, node: dict[str, Any], args: list[Any], state: Any, reach: Any) -> Any:
        tok = args[-1] if args else self.const_u32(0)
        if state is not None:
            self._on_control_barrier(state, self._active_mask(state, reach), int(node["id"]))
        return tok

    def _normalize_bkey(self, bkey: str | None) -> str | None:
        if bkey == "wls_shared" and self._env is not None and self._env.unique_workgroup_key is not None:
            return self._env.unique_workgroup_key
        return bkey

    def _as_vec4(self, vec: Any) -> tuple[Any, ...]:
        return tuple(
            self.domain.as_u32(vec[c]) if isinstance(vec, tuple) and c < len(vec) else self.const_u32(0)
            for c in range(4)
        )

    def psb_load(self, node: dict[str, Any], args: list[Any], state: Any) -> Any:
        if state is None or self._env is None:
            return (self.default_value(), args[1] if len(args) > 1 else self.default_value())
        addr = self.domain.as_u64(args[0])
        aux_str = node.get("aux_str", "")
        eff_hint = aux_str
        if aux_str.startswith("sym:"):
            bkey_hint = aux_str[4:].split("[", 1)[0]
            _, _, off = self._resolve_buf_addr(addr, bkey_hint)
            sym_val = self._read_sym_slot(state, aux_str, off)
            if sym_val is not None:
                return (sym_val, args[1] if len(args) > 1 else self.const_u32(0))
            eff_hint = bkey_hint
        bkey, const_off, off = self._resolve_buf_addr(addr, eff_hint)
        bkey = self._normalize_bkey(bkey)
        width = int(node.get("imm_u32", 4))
        if width not in (1, 2, 4):
            width = 4
        val = self._read_buf_sized(state, bkey, const_off, off, addr, width)
        return (val, args[1] if len(args) > 1 else self.const_u32(0))

    def psb_store(self, node: dict[str, Any], args: list[Any], state: Any, reach: Any) -> Any:
        tok = args[2] if len(args) > 2 else self.const_u32(0)
        if state is None or self._env is None:
            return tok
        addr = self.domain.as_u64(args[0])
        val = self.domain.as_u32(args[1])
        mask = self._active_mask(state, reach)
        nid = int(node["id"])
        aux_str = node.get("aux_str", "")
        if aux_str.startswith("sym:"):
            bkey_hint = aux_str[4:].split("[", 1)[0]
            _, _, off = self._resolve_buf_addr(addr, bkey_hint)
            if self._write_sym_slot(state, mask, aux_str, off, val, nid):
                return tok
            eff_hint = bkey_hint
        elif aux_str.startswith("img:"):
            ch_idx, xy_addr = self._extract_img_channel(addr)
            self._write_img_channel(state, mask, aux_str, xy_addr, ch_idx, val, nid)
            return tok
        else:
            eff_hint = aux_str
        bkey, const_off, off = self._resolve_buf_addr(addr, eff_hint)
        bkey = self._normalize_bkey(bkey)
        width = int(node.get("imm_u32", 4))
        if width not in (1, 2, 4):
            width = 4
        self._write_buf_sized(state, mask, bkey, const_off, off, addr, val, nid, width)
        return tok

    def kill(self, node: dict[str, Any], args: list[Any], state: Any, reach: Any) -> Any:
        if state is None:
            return self.default_value()
        aux_bool = bool(node.get("aux_bool", False))
        cond = self.domain.as_bool(args[0]) if (aux_bool and len(args) >= 2) else self.const_bool(True)
        eff = self.domain.logical_and(reach, cond)
        self._record_kill(state, eff, bool(node.get("aux_bool2", False)), int(node["id"]))
        return self.const_u32(0)

    def store_frag_color(self, node: dict[str, Any], args: list[Any], state: Any, reach: Any) -> Any:
        tok = args[-1] if args else self.const_u32(0)
        if state is not None:
            self._write_frag_color(state, self._active_mask(state, reach), int(node.get("imm_u32", 0)), self._as_vec4(args[0]), int(node["id"]))
        return tok

    def store_builtin_output(self, node: dict[str, Any], args: list[Any], state: Any, reach: Any) -> Any:
        tok = args[-1] if args else self.const_u32(0)
        if state is not None:
            self._write_builtin_output(state, self._active_mask(state, reach), int(node.get("imm_u32", 0)), args[0], int(node["id"]))
        return tok

    def store_vert_varying(self, node: dict[str, Any], args: list[Any], state: Any, reach: Any) -> Any:
        tok = args[-1] if args else self.const_u32(0)
        if state is not None and self._env is not None:
            imm32 = int(node.get("imm_u32", 0))
            loc = self._env.sout_addr_to_loc.get(imm32, imm32) if self.is_valhall else imm32
            self._write_vert_varying(state, self._active_mask(state, reach), loc, self._as_vec4(args[0]), int(node["id"]))
        return tok

    def store_texel(self, node: dict[str, Any], args: list[Any], state: Any, reach: Any) -> Any:
        tok = args[-1] if args else self.const_u32(0)
        if state is not None:
            coord = args[0]
            texel = args[1] if len(args) > 1 else (self.const_u32(0),) * 4
            cx = self.domain.as_u32(coord[0]) if isinstance(coord, tuple) and len(coord) > 0 else self.domain.as_u32(coord)
            cy = self.domain.as_u32(coord[1]) if isinstance(coord, tuple) and len(coord) > 1 else self.const_u32(0)
            if isinstance(coord, tuple) and len(coord) > 2:
                cz = self.domain.as_u32(coord[2])
                xy = self.domain.hw_uf(self._env, "img_coord3d", cx, cy, cz, out_bits=64, fallback_tag="img_coord3d")
            else:
                xy = self.pack64(cx, cy)
            ikey = node.get("aux_str", "") or f"img_{int(node.get('imm_u32', 0))}"
            self._write_img_texel(state, self._active_mask(state, reach), ikey, xy, self._as_vec4(texel), int(node["id"]))
        return tok

    def atomic_op(self, node: dict[str, Any], args: list[Any], state: Any, reach: Any) -> Any:
        tok = args[-1] if args else self.const_u32(0)
        if state is None or self._env is None:
            return (self.default_value(), tok)
        op = node["op"]
        aux_str = node.get("aux_str", "")
        addr = self.domain.as_u64(args[0])
        arg = self.domain.as_u32(args[1]) if len(args) > 1 else self.const_u32(0)
        cmp_val = self.domain.as_u32(args[2]) if (op == "AtomicCmpXchg" and len(args) > 2) else self.const_u32(0)
        op_tag = aux_str.lower() if aux_str else op.lower()
        bkey, const_off, off = self._resolve_buf_addr(addr, "")
        bkey = self._normalize_bkey(bkey)
        old_val = self._havoc_atomic_old_val(state, bkey, const_off, off, addr, op_tag, int(node["id"]))
        new_val = self.domain.atomic_compute(self._env, op, op_tag, old_val, arg, cmp_val)
        self._record_atomic(state, self._active_mask(state, reach), bkey, off, addr, op_tag, arg, cmp_val, new_val, int(node["id"]))
        return (old_val, tok)

    # ------------------------------------------------------------------
    # Canonical single-node denotational dispatch
    # ------------------------------------------------------------------
    def eval_node(
        self,
        node: dict[str, Any],
        values: dict[int, V],
        state: Any = None,
        reach: Any = None,
    ) -> V:
        """Canonical denotational dispatch over a single ValhallSSA node."""
        op = node["op"]
        vt = node.get("type", "U32")
        args = [self.lookup_arg(values, int(aid)) for aid in node.get("args", [])]
        imm_u32 = int(node.get("imm_u32", 0))
        imm_u32_2 = int(node.get("imm_u32_2", 0))
        imm_u64 = int(node.get("imm_u64", 0))
        aux_bool = bool(node.get("aux_bool", False))
        aux_str = node.get("aux_str", "")

        if op in ("ConstU32", "ConstS32", "ConstF32"):
            return self.const_u32(imm_u32)
        if op == "ConstU64":
            return self.const_u64(imm_u64)
        if op == "ConstBool":
            return self.const_bool(imm_u32 != 0 or aux_bool)
        if op in ("Nop", "Copy", "Bitcast"):
            if not args:
                return self.default_value()
            return self.cast_value(op, args[0], vt)

        if op in ("VecConstruct2", "VecConstruct3", "VecConstruct4"):
            return self.domain.pack_lanes(args)
        if op == "VecExtract":
            return self.domain.extract_lane(args[0], imm_u32)
        if op == "VecShuffle":
            return self.domain.shuffle_lanes(args[0], args[1], (imm_u32, imm_u32_2))
        if op in ("Pack64", "ImageTexelAddr"):
            return self.pack64(args[0], args[1])
        if op == "Split64":
            return self.split64(args[0])
        if op == "TupleExtract":
            return self.domain.extract_tuple(args[0], imm_u32)
        if op == "UnpackHalf2x16":
            return self.unpack_half2x16(args[0])
        if op == "PackHalf2x16":
            return self.pack_half2x16(args[0])

        if op in INT32_UNARY_OPS:
            return self._elemwise(lambda x: self.int32_unary(op, x), args[0])
        if op in INT32_BINARY_OPS:
            if op in INT_DIV_OPS and state is not None:
                self._record_div_guard(state, reach, op, args[1])
            return self._elemwise(lambda a, b: self.int32_binary(op, a, b), args[0], args[1])
        if op in ("BitFieldUExtract", "BitFieldSExtract", "BfeU", "BfeS"):
            signed = op in ("BitFieldSExtract", "BfeS")
            return self._elemwise(lambda x, o, w: self.bfe(signed, x, o, w), args[0], args[1], args[2])
        if op in ("BitFieldInsert", "Bfi"):
            return self._elemwise(lambda b, i, o, w: self.bfi(b, i, o, w), args[0], args[1], args[2], args[3])
        if op in ("IDot4S", "IDot4SU", "IDot4U"):
            return self.idot4(op, args[0], args[1])
        if op == "BitNot64":
            return self.int64_unary(op, args[0])
        if op in INT64_BINARY_OPS:
            return self.int64_binary(op, args[0], args[1])

        if op.startswith("ICmp"):
            return self._elemwise(lambda a, b: self.icmp(op, a, b), args[0], args[1])
        if op == "LogicalAnd":
            return self._elemwise(lambda a, b: self.logical_and(a, b), args[0], args[1])
        if op == "LogicalOr":
            return self._elemwise(lambda a, b: self.logical_or(a, b), args[0], args[1])
        if op == "LogicalNot":
            return self._elemwise(lambda a: self.logical_not(a), args[0])
        if op == "Select":
            return self._elemwise(lambda c, t, f: self.select(c, t, f), args[0], args[1], args[2])

        if op in QUAD_DERIV_OPS:
            return self._elemwise(lambda x: self.quad_deriv(op, x, state), args[0])
        if op in FP_UNARY_OPS:
            return self._elemwise(lambda x: self.fp_unary(op, x), args[0])
        if op in FP_BINARY_OPS:
            return self._elemwise(lambda a, b: self.fp_binary(op, a, b), args[0], args[1])
        if op == "FClamp":
            return self._elemwise(lambda x, lo, hi: self.fp_clamp(x, lo, hi), args[0], args[1], args[2])
        if op == "Fma":
            return self._elemwise(lambda a, b, c: self.fp_fma(a, b, c), args[0], args[1], args[2])
        if op == "FMix":
            return self._elemwise(
                lambda x, y, a: self.fp_binary("FAdd", x, self.fp_binary("FMul", self.fp_binary("FSub", y, x), a)),
                args[0],
                args[1],
                args[2],
            )
        if op == "IsNan":
            return self._elemwise(lambda x: self.fp_is_nan(x), args[0])
        if op == "IsInf":
            return self._elemwise(lambda x: self.fp_is_inf(x), args[0])
        if op.startswith("FCmp"):
            return self._elemwise(lambda a, b: self.fp_cmp(op, a, b), args[0], args[1])
        if op in CONVERT_OPS:
            return self._elemwise(lambda x: self.convert(op, x), args[0])

        if op == "LoadFau":
            return self.load_fau(imm_u32)
        if op == "LoadSidebandU64":
            return self.load_sideband_u64(imm_u32, aux_str)
        if op == "LoadSidebandU32":
            return self.load_sideband_u32(imm_u32, imm_u32_2)
        if op == "LoadBuiltinInput":
            return self.load_builtin(imm_u32, vt)
        if op == "LoadVertAttr":
            return self.load_vert_attr(imm_u32)
        if op == "LoadFragVarying":
            return self.load_frag_varying(imm_u32)
        if op == "ImageQuerySize":
            return self.image_query_size(args[0] if args else self.const_u32(0), imm_u32, aux_str)
        if op in TEX_SAMPLE_OPS:
            return self.tex_sample(op, args, imm_u32, aux_str, imm_u32_2)

        if op == "PsbLoad":
            return self.psb_load(node, args, state)
        if op == "PsbStore":
            return self.psb_store(node, args, state, reach)
        if op == "Kill":
            return self.kill(node, args, state, reach)
        if op == "StoreFragColor":
            return self.store_frag_color(node, args, state, reach)
        if op == "StoreBuiltinOutput":
            return self.store_builtin_output(node, args, state, reach)
        if op == "StoreVertVarying":
            return self.store_vert_varying(node, args, state, reach)
        if op == "StoreTexel":
            return self.store_texel(node, args, state, reach)
        if op in ATOMIC_OPS:
            return self.atomic_op(node, args, state, reach)
        if op == "SubgroupBallot":
            return self.subgroup_ballot(args[0] if args else self.const_u32(0))
        if op == "SubgroupShuffle":
            return self.subgroup_shuffle(
                args[0] if args else self.const_u32(0),
                args[1] if len(args) > 1 else self.const_u32(0),
            )
        if op == "ControlBarrier":
            return self.control_barrier(node, args, state, reach)
        if op in CONTROL_PASSTHROUGH_OPS:
            return args[-1] if args else self.default_value()
        return self.default_value()

    def _elemwise(self, fn: Callable[..., V], *xs: V) -> V:
        return self.domain.map_lanes(fn, *xs)
