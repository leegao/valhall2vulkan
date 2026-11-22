#!/usr/bin/env python3
"""
Abstract Denotational SSA Interpreter and Concrete / Symbolic (Z3) Implementations.

Provides:
  - SsaInterpreter: abstract base class defining the denotational dispatch over
    all ValhallSsaModule opcodes, delegating value/effect operations to a clean
    domain interface implemented by concrete, symbolic, and abstract-interpretation
    subclasses.
  - ConcreteSsaInterpreter(SsaInterpreter): bit-exact Python concrete interpreter
    with full op parity (including Atomics, IDot4*, ImageTexelAddr, sym:/img:).
  - SymbolicSsaInterpreter(SsaInterpreter) / DagSsaZ3Evaluator: whole-CFG RPO
    symbolic Z3 evaluator with configurable FP mode, bounded loop unrolling,
    per-iteration branch decision tracking, and cutpoint substitution.
  - evaluate_ssa(...): unified entrypoint for concrete or symbolic evaluation.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import z3

from tools.symbolic_ssa.env import (
    BV32,
    BV64,
    FP32_SORT,
    MASK32,
    MASK64,
    REAL_SORT,
    ConcreteShaderEnv,
    SharedShaderEnv,
    Z3Val,
    _LN_2_REAL,
    _LOG2_E_REAL,
    _REAL_ONE,
    _REAL_ZERO,
    _as_bool,
    _as_bv32,
    _as_bv64,
    _bv32_to_real,
    _c_bool,
    _c_u32,
    _c_u64,
    _det_hash_u32,
    _eval_elementwise,
    _f16_to_f32_u32,
    _f32_to_f16_u16,
    _f32_to_u32,
    _real_cos,
    _real_exp2,
    _real_log2,
    _real_sin,
    _real_sqrt,
    _real_to_bv32,
    _s32,
    _s64,
    _u32_to_f32,
    _z3_and,
    _z3_if,
    _z3_not,
    _z3_or,
)
from tools.verify.panfrost_z3_model import (
    _bitrev_bv32,
    _clz_bv,
    _fp_to_int_sat,
    _fp_total_le_bv,
    _hw_uf,
    _popcount_bv32,
    bv32,
    bv32_to_fp32,
    bv64,
    f16_bv_to_f32_bv,
    f32_bv_to_f16_bv,
    fp32_to_bv32,
)

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
FP_UNARY_OPS = frozenset({
    "FAbs", "FNeg", "Floor", "Ceil", "Trunc", "RoundEven", "Sqrt", "InverseSqrt",
    "Sin", "Cos", "Exp", "Exp2", "Log", "Log2", "Atan", "Asin", "Acos",
    "Sinh", "Cosh", "Tanh", "DPdx", "DPdy", "Fwidth", "FrexpExp", "FrexpSignificand",
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
CONTROL_PASSTHROUGH_OPS = frozenset({"ControlBarrier", "Branch", "BranchCond", "Return"})


class SsaInterpreter(abc.ABC):
    """
    Abstract denotational interpreter for ValhallSsaModule.

    Subclasses implement the domain-specific operations (concrete Python ints/bools,
    symbolic Z3 expressions, or abstract domains for canonicalization passes) while
    sharing the single opcode dispatch table in `eval_node`.
    """

    def default_value(self) -> Any:
        return self.const_u32(0)

    def lookup_arg(self, values: Dict[int, Any], aid: int) -> Any:
        return values.get(aid, self.default_value())

    @abc.abstractmethod
    def const_u32(self, imm: int) -> Any: ...

    @abc.abstractmethod
    def const_u64(self, imm: int) -> Any: ...

    @abc.abstractmethod
    def const_bool(self, val: bool) -> Any: ...

    def cast_value(self, op: str, val: Any, val_type: str) -> Any:
        return val

    def pack64(self, lo: Any, hi: Any) -> Any:
        return self.default_value()

    def split64(self, val64: Any) -> Tuple[Any, Any]:
        return (self.default_value(), self.default_value())

    def unpack_half2x16(self, u: Any) -> Tuple[Any, Any]:
        return (self.default_value(), self.default_value())

    def pack_half2x16(self, v2: Any) -> Any:
        return self.default_value()

    def int32_unary(self, op: str, x: Any) -> Any:
        return self.default_value()

    def int32_binary(self, op: str, a: Any, b: Any) -> Any:
        return self.default_value()

    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> Any:
        return self.default_value()

    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> Any:
        return self.default_value()

    def idot4(self, op: str, a: Any, b: Any) -> Any:
        return self.default_value()

    def int64_unary(self, op: str, x: Any) -> Any:
        return self.default_value()

    def int64_binary(self, op: str, a: Any, b: Any) -> Any:
        return self.default_value()

    def icmp(self, op: str, a: Any, b: Any) -> Any:
        return self.const_bool(False)

    def logical_and(self, a: Any, b: Any) -> Any:
        return self.const_bool(False)

    def logical_or(self, a: Any, b: Any) -> Any:
        return self.const_bool(False)

    def logical_not(self, a: Any) -> Any:
        return self.const_bool(False)

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        return t_val

    def fp_unary(self, op: str, x: Any) -> Any:
        return self.default_value()

    def fp_binary(self, op: str, a: Any, b: Any) -> Any:
        return self.default_value()

    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> Any:
        return self.default_value()

    def fp_fma(self, a: Any, b: Any, c: Any) -> Any:
        return self.default_value()

    def fp_is_nan(self, x: Any) -> Any:
        return self.const_bool(False)

    def fp_is_inf(self, x: Any) -> Any:
        return self.const_bool(False)

    def fp_cmp(self, op: str, a: Any, b: Any) -> Any:
        return self.const_bool(False)

    def convert(self, op: str, x: Any) -> Any:
        return self.default_value()

    def load_fau(self, word_idx: int) -> Any:
        return self.default_value()

    def load_sideband_u64(self, imm_u32: int, aux_str: str) -> Any:
        return self.default_value()

    def load_sideband_u32(self, imm_u32: int, imm_u32_2: int) -> Any:
        return self.default_value()

    def load_builtin(self, builtin_id: int, val_type: str) -> Any:
        return self.default_value()

    def load_vert_attr(self, imm_u32: int) -> Any:
        return self.default_value()

    def load_frag_varying(self, imm_u32: int) -> Any:
        return self.default_value()

    def image_query_size(self, lod: Any, imm_u32: int, aux_str: str) -> Any:
        return self.default_value()

    def tex_sample(self, op: str, args: List[Any], imm_u32: int, aux_str: str) -> Any:
        return (tuple(self.default_value() for _ in range(4)), args[-1] if args else self.default_value())

    def psb_load(self, node: Dict[str, Any], args: List[Any], state: Any) -> Any:
        return (self.default_value(), args[1] if len(args) > 1 else self.default_value())

    def psb_store(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Any:
        return args[2] if len(args) > 2 else self.default_value()

    def kill(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Any:
        return self.default_value()

    def store_frag_color(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Any:
        return args[-1] if args else self.default_value()

    def store_builtin_output(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Any:
        return args[-1] if args else self.default_value()

    def store_vert_varying(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Any:
        return args[-1] if args else self.default_value()

    def store_texel(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Any:
        return args[-1] if args else self.default_value()

    def atomic_op(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Any:
        return (self.default_value(), args[-1] if args else self.default_value())

    def subgroup_ballot(self, x: Any) -> Any:
        return self.default_value()

    def subgroup_shuffle(self, x: Any, lane: Any) -> Any:
        return self.default_value()

    def eval_node(
        self,
        node: Dict[str, Any],
        values: Dict[int, Any],
        state: Any = None,
        reach: Any = None,
    ) -> Any:
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
            return tuple(args)
        if op == "VecExtract":
            v = args[0]
            return v[imm_u32] if isinstance(v, tuple) and imm_u32 < len(v) else self.cast_value("Copy", v, "U32")
        if op == "VecShuffle":
            a, b = args[0], args[1]
            comb = (list(a) if isinstance(a, tuple) else [a]) + (list(b) if isinstance(b, tuple) else [b])
            return (comb[imm_u32], comb[imm_u32_2])
        if op in ("Pack64", "ImageTexelAddr"):
            return self.pack64(args[0], args[1])
        if op == "Split64":
            return self.split64(args[0])
        if op == "TupleExtract":
            tup = args[0]
            return tup[imm_u32] if isinstance(tup, tuple) and imm_u32 < len(tup) else tup
        if op == "UnpackHalf2x16":
            return self.unpack_half2x16(args[0])
        if op == "PackHalf2x16":
            return self.pack_half2x16(args[0])

        if op in INT32_UNARY_OPS:
            return self.int32_unary(op, args[0])
        if op in INT32_BINARY_OPS:
            return self.int32_binary(op, args[0], args[1])
        if op in ("BfeU", "BfeS"):
            return self.bfe(op == "BfeS", args[0], args[1], args[2])
        if op == "Bfi":
            return self.bfi(args[0], args[1], args[2], args[3])
        if op in ("IDot4S", "IDot4SU", "IDot4U"):
            return self.idot4(op, args[0], args[1])

        if op == "BitNot64":
            return self.int64_unary(op, args[0])
        if op in INT64_BINARY_OPS:
            return self.int64_binary(op, args[0], args[1])

        if op.startswith("ICmp"):
            return self.icmp(op, args[0], args[1])
        if op == "LogicalAnd":
            return self.logical_and(args[0], args[1])
        if op == "LogicalOr":
            return self.logical_or(args[0], args[1])
        if op == "LogicalNot":
            return self.logical_not(args[0])
        if op == "Select":
            return self.select(args[0], args[1], args[2])

        if op in FP_UNARY_OPS:
            return _eval_elementwise(vt, args, lambda x: self.fp_unary(op, x))
        if op in FP_BINARY_OPS:
            return _eval_elementwise(vt, args, lambda x, y: self.fp_binary(op, x, y))
        if op == "FClamp":
            return _eval_elementwise(vt, args, lambda x, lo, hi: self.fp_clamp(x, lo, hi))
        if op == "Fma":
            return _eval_elementwise(vt, args, lambda a, b, c: self.fp_fma(a, b, c))
        if op == "IsNan":
            return self.fp_is_nan(args[0])
        if op == "IsInf":
            return self.fp_is_inf(args[0])
        if op.startswith("FCmp"):
            return self.fp_cmp(op, args[0], args[1])

        if op in CONVERT_OPS:
            return self.convert(op, args[0])

        if op == "LoadFau":
            return self.load_fau(imm_u32 & 63)
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
            lod = args[0] if args else self.const_u32(0)
            return self.image_query_size(lod, imm_u32, aux_str)
        if op in TEX_SAMPLE_OPS:
            return self.tex_sample(op, args, imm_u32, aux_str)

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
        if op in CONTROL_PASSTHROUGH_OPS:
            return args[-1] if args else self.default_value()
        if op == "SubgroupBallot":
            return self.subgroup_ballot(args[0])
        if op == "SubgroupShuffle":
            return self.subgroup_shuffle(args[0], args[1])

        return self.default_value()


@dataclasses.dataclass
class BlockEffectState:
    """Symbolic observable state at the end of a basic block."""

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
    incoming: List[Tuple[z3.BoolRef, BlockEffectState]],
    env: SharedShaderEnv,
) -> BlockEffectState:
    if len(incoming) == 1:
        return _clone_state(incoming[0][1])

    base_st = _clone_state(incoming[0][1])
    for cond, st in incoming[1:]:
        if z3.is_false(cond):
            continue
        base_st.terminated = _z3_if(cond, st.terminated, base_st.terminated)
        base_st.demoted = _z3_if(cond, st.demoted, base_st.demoted)
        base_st.discarded = _z3_or(base_st.terminated, base_st.demoted)
        for rt in set(base_st.frag_colors.keys()) | set(st.frag_colors.keys()):
            a = base_st.frag_colors.get(rt, (bv32(0),) * 4)
            b = st.frag_colors.get(rt, (bv32(0),) * 4)
            base_st.frag_colors[rt] = tuple(_z3_if(cond, bi, ai) for ai, bi in zip(a, b))
        for bi in set(base_st.builtin_outputs.keys()) | set(st.builtin_outputs.keys()):
            if bi in st.builtin_outputs and bi in base_st.builtin_outputs:
                base_st.builtin_outputs[bi] = _z3_if(cond, st.builtin_outputs[bi], base_st.builtin_outputs[bi])
            elif bi in st.builtin_outputs:
                base_st.builtin_outputs[bi] = st.builtin_outputs[bi]
        for loc in set(base_st.vert_varyings.keys()) | set(st.vert_varyings.keys()):
            a = base_st.vert_varyings.get(loc, (bv32(0),) * 4)
            b = st.vert_varyings.get(loc, (bv32(0),) * 4)
            base_st.vert_varyings[loc] = tuple(_z3_if(cond, bi, ai) for ai, bi in zip(a, b))
        for stag in set(base_st.sym_buf_stores.keys()) | set(st.sym_buf_stores.keys()):
            wa, va, oa = base_st.sym_buf_stores.get(stag, (z3.BoolVal(False), bv32(0), bv32(0)))
            wb, vb, ob = st.sym_buf_stores.get(stag, (z3.BoolVal(False), bv32(0), bv32(0)))
            base_st.sym_buf_stores[stag] = (_z3_if(cond, wb, wa), _z3_if(cond, vb, va), _z3_if(cond, ob, oa))
        for key in set(base_st.const_stores.keys()) | set(st.const_stores.keys()):
            w_a, v_a = base_st.const_stores.get(key, (z3.BoolVal(False), z3.Select(env.get_buf_mem(key[0]), bv32(key[1]))))
            w_b, v_b = st.const_stores.get(key, (z3.BoolVal(False), z3.Select(env.get_buf_mem(key[0]), bv32(key[1]))))
            base_st.const_stores[key] = (_z3_if(cond, w_b, w_a), _z3_if(cond, v_b, v_a))
        for bkey in set(base_st.buf_mems.keys()) | set(st.buf_mems.keys()):
            m_a = base_st.buf_mems.get(bkey, env.get_buf_mem(bkey))
            m_b = st.buf_mems.get(bkey, env.get_buf_mem(bkey))
            if not m_a.eq(m_b):
                base_st.buf_mems[bkey] = z3.If(cond, m_b, m_a)
        for ikey in set(base_st.image_mems.keys()) | set(st.image_mems.keys()):
            im_a = base_st.image_mems.get(ikey, env.get_img_mem(ikey))
            im_b = st.image_mems.get(ikey, env.get_img_mem(ikey))
            base_st.image_mems[ikey] = tuple(
                z3.If(cond, b, a) if not a.eq(b) else a for a, b in zip(im_a, im_b)
            )
        if not base_st.fallback_mem.eq(st.fallback_mem):
            base_st.fallback_mem = z3.If(cond, st.fallback_mem, base_st.fallback_mem)
        for akey in set(base_st.atomic_events.keys()) | set(st.atomic_events.keys()):
            ev_a = base_st.atomic_events.get(akey, [])
            ev_b = st.atomic_events.get(akey, [])
            if len(ev_a) == len(ev_b):
                base_st.atomic_events[akey] = [
                    (
                        _z3_if(cond, eb[0], ea[0]),
                        eb[1] if ea[1] == eb[1] else f"{ea[1]}|{eb[1]}",
                        _z3_if(cond, eb[2], ea[2]),
                        _z3_if(cond, eb[3], ea[3]),
                        _z3_if(cond, eb[4], ea[4]),
                    )
                    for ea, eb in zip(ev_a, ev_b)
                ]
            else:
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
    ):
        self.ssa = ssa_json
        self.env = env
        self.is_valhall = is_valhall_side
        self.fp_mode = fp_mode
        self.max_unroll = max(1, int(max_unroll))
        self.cutpoints: Dict[int, Any] = cutpoints if cutpoints is not None else {}
        self.blocks: List[Dict[str, Any]] = ssa_json.get("blocks", [])
        self.nodes_by_id: Dict[int, Dict[str, Any]] = {int(n["id"]): n for n in ssa_json.get("nodes", [])}
        self.values: Dict[int, Z3Val] = {0: bv32(0)}
        self.block_reach: Dict[int, z3.BoolRef] = {}
        self.iter_branch_decisions: List[Tuple[int, int, int, z3.BoolRef, z3.BoolRef]] = []
        self.has_loops: bool = False
        self.max_unroll_depth: int = 0

    def const_u32(self, imm: int) -> z3.BitVecRef:
        return bv32(imm)

    def const_u64(self, imm: int) -> z3.BitVecRef:
        return bv64(imm)

    def const_bool(self, val: bool) -> z3.BoolRef:
        return z3.BoolVal(val)

    def cast_value(self, op: str, val: Any, val_type: str) -> Z3Val:
        if val_type in ("U64", "S64"):
            return _as_bv64(val)
        if val_type == "Bool":
            return _as_bool(val)
        if val_type in ("U32", "S32", "F32") and not isinstance(val, tuple):
            return _as_bv32(val)
        return val

    def pack64(self, lo: Any, hi: Any) -> z3.BitVecRef:
        blo, bhi = _as_bv32(lo), _as_bv32(hi)
        if isinstance(blo, z3.BitVecNumRef) and isinstance(bhi, z3.BitVecNumRef):
            return bv64((bhi.as_long() << 32) | blo.as_long())
        return z3.Concat(bhi, blo)

    def split64(self, val64: Any) -> Tuple[z3.BitVecRef, z3.BitVecRef]:
        v64 = _as_bv64(val64)
        if isinstance(v64, z3.BitVecNumRef):
            n = v64.as_long()
            return (bv32(n & MASK32), bv32((n >> 32) & MASK32))
        return (z3.Extract(31, 0, v64), z3.Extract(63, 32, v64))

    def unpack_half2x16(self, u: Any) -> Tuple[z3.BitVecRef, z3.BitVecRef]:
        bu = _as_bv32(u)
        return (f16_bv_to_f32_bv(z3.Extract(15, 0, bu)), f16_bv_to_f32_bv(z3.Extract(31, 16, bu)))

    def pack_half2x16(self, v2: Any) -> z3.BitVecRef:
        if isinstance(v2, tuple) and len(v2) >= 2:
            f0 = f32_bv_to_f16_bv(_as_bv32(v2[0]))
            f1 = f32_bv_to_f16_bv(_as_bv32(v2[1]))
            return z3.Concat(f1, f0)
        s = f32_bv_to_f16_bv(_as_bv32(v2))
        return z3.Concat(s, s)

    def int32_unary(self, op: str, x: Any) -> z3.BitVecRef:
        bx = _as_bv32(x)
        if op == "BitNot":
            return ~bx
        if op == "SNeg":
            return -bx
        if op == "SAbs":
            return z3.If(bx < bv32(0), -bx, bx)
        if op == "Clz":
            return _clz_bv(32, bx, mask_zero=False)
        if op == "BitCount":
            return _popcount_bv32(bx)
        if op == "BitReverse":
            return _bitrev_bv32(bx)
        return bv32(0)

    def int32_binary(self, op: str, a: Any, b: Any) -> z3.BitVecRef:
        ba, bb = _as_bv32(a), _as_bv32(b)
        if op == "IAdd": return ba + bb
        if op == "ISub": return ba - bb
        if op == "IMul": return ba * bb
        if op == "BitAnd": return ba & bb
        if op == "BitOr": return ba | bb
        if op == "BitXor": return ba ^ bb
        if op == "Shl": return ba << (bb & bv32(31))
        if op == "ShrU": return z3.LShR(ba, bb & bv32(31))
        if op == "ShrS": return ba >> (bb & bv32(31))
        if op == "RotL": return z3.RotateLeft(ba, bb & bv32(31))
        if op == "RotR": return z3.RotateRight(ba, bb & bv32(31))
        if op == "UDiv": return z3.UDiv(ba, bb)
        if op == "SDiv": return ba / bb
        if op == "UMod": return z3.URem(ba, bb)
        if op == "SRem": return z3.SRem(ba, bb)
        if op == "SMod":
            rem = z3.SRem(ba, bb)
            return z3.If(z3.And(rem != bv32(0), (rem ^ bb) < bv32(0)), rem + bb, rem)
        return bv32(0)

    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> z3.BitVecRef:
        bx, boff, bw = _as_bv32(x), _as_bv32(off), _as_bv32(width)
        if isinstance(boff, z3.BitVecNumRef) and isinstance(bw, z3.BitVecNumRef):
            o_i, w_i = boff.as_long(), bw.as_long()
            if w_i == 0:
                return bv32(0)
            hi_i = min(31, o_i + w_i - 1)
            sl = z3.Extract(hi_i, o_i, bx)
            ext = 32 - sl.size()
            if ext == 0:
                return sl
            return z3.SignExt(ext, sl) if signed else z3.ZeroExt(ext, sl)
        if signed:
            sh = bv32(32) - bw
            extracted = (bx << (sh - boff)) >> sh
            return z3.If(bw == bv32(0), bv32(0), z3.If(z3.UGE(bw, bv32(32)), bx, extracted))
        mask = (bv32(1) << (bw & bv32(31))) - bv32(1)
        extracted = z3.LShR(bx, boff & bv32(31)) & mask
        return z3.If(bw == bv32(0), bv32(0), z3.If(z3.UGE(bw, bv32(32)), bx, extracted))

    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> z3.BitVecRef:
        bb, bi, boff, bw = _as_bv32(base), _as_bv32(insert), _as_bv32(off), _as_bv32(width)
        if isinstance(boff, z3.BitVecNumRef) and isinstance(bw, z3.BitVecNumRef):
            o_i, w_i = boff.as_long(), bw.as_long()
            if w_i >= 32: return bi
            if w_i == 0: return bb
            mask_v = ((1 << w_i) - 1) << o_i
            return (bb & bv32(~mask_v)) | ((bi << o_i) & bv32(mask_v))
        mask = ((bv32(1) << (bw & bv32(31))) - bv32(1)) << (boff & bv32(31))
        res_bfi = (bb & ~mask) | ((bi << (boff & bv32(31))) & mask)
        return z3.If(bw == bv32(0), bb, z3.If(z3.UGE(bw, bv32(32)), bi, res_bfi))

    def idot4(self, op: str, a: Any, b: Any) -> z3.BitVecRef:
        a32, b32 = _as_bv32(a), _as_bv32(b)
        acc = bv32(0)
        for i in range(4):
            ba = z3.Extract(8 * i + 7, 8 * i, a32)
            bb = z3.Extract(8 * i + 7, 8 * i, b32)
            ea = z3.SignExt(24, ba) if op in ("IDot4S", "IDot4SU") else z3.ZeroExt(24, ba)
            eb = z3.SignExt(24, bb) if op == "IDot4S" else z3.ZeroExt(24, bb)
            acc = acc + (ea * eb)
        return acc

    def int64_unary(self, op: str, x: Any) -> z3.BitVecRef:
        return ~_as_bv64(x)

    def int64_binary(self, op: str, a: Any, b: Any) -> z3.BitVecRef:
        a64, b64 = _as_bv64(a), _as_bv64(b)
        if isinstance(a64, z3.BitVecNumRef) and isinstance(b64, z3.BitVecNumRef):
            av, bv_ = a64.as_long(), b64.as_long()
            if op == "IAdd64": return bv64((av + bv_) & MASK64)
            if op == "ISub64": return bv64((av - bv_) & MASK64)
            if op == "IMul64": return bv64((av * bv_) & MASK64)
        if op == "IAdd64": return a64 + b64
        if op == "ISub64": return a64 - b64
        if op == "IMul64": return a64 * b64
        if op == "BitAnd64": return a64 & b64
        if op == "BitOr64": return a64 | b64
        if op == "BitXor64": return a64 ^ b64
        if op == "Shl64": return a64 << (b64 & bv64(63))
        if op == "ShrU64": return z3.LShR(a64, b64 & bv64(63))
        if op == "ShrS64": return a64 >> (b64 & bv64(63))
        return bv64(0)

    def icmp(self, op: str, a: Any, b: Any) -> z3.BoolRef:
        ba, bb = _as_bv32(a), _as_bv32(b)
        if op == "ICmpEq": return ba == bb
        if op == "ICmpNe": return ba != bb
        if op == "ICmpSLt": return ba < bb
        if op == "ICmpSLe": return ba <= bb
        if op == "ICmpSGt": return ba > bb
        if op == "ICmpSGe": return ba >= bb
        if op == "ICmpULt": return z3.ULT(ba, bb)
        if op == "ICmpULe": return z3.ULE(ba, bb)
        if op == "ICmpUGt": return z3.UGT(ba, bb)
        if op == "ICmpUGe": return z3.UGE(ba, bb)
        return z3.BoolVal(False)

    def logical_and(self, a: Any, b: Any) -> z3.BoolRef:
        return _z3_and(_as_bool(a), _as_bool(b))

    def logical_or(self, a: Any, b: Any) -> z3.BoolRef:
        return _z3_or(_as_bool(a), _as_bool(b))

    def logical_not(self, a: Any) -> z3.BoolRef:
        return _z3_not(_as_bool(a))

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Z3Val:
        return _z3_if(_as_bool(cond), t_val, f_val)

    def fp_unary(self, op: str, x: Any) -> z3.BitVecRef:
        bx = _as_bv32(x)
        if self.fp_mode == "real":
            rx = _bv32_to_real(bx)
            if op == "FAbs": return _real_to_bv32(z3.If(rx < _REAL_ZERO, -rx, rx))
            if op == "FNeg": return _real_to_bv32(-rx)
            if op == "Sqrt": return _real_to_bv32(_real_sqrt(rx))
            if op == "InverseSqrt": return _real_to_bv32(_REAL_ONE / _real_sqrt(rx))
            if op == "Exp2": return _real_to_bv32(_real_exp2(rx))
            if op == "Exp": return _real_to_bv32(_real_exp2(rx * _LOG2_E_REAL))
            if op == "Log2": return _real_to_bv32(_real_log2(rx))
            if op == "Log": return _real_to_bv32(_real_log2(rx) * _LN_2_REAL)
            if op == "Sin": return _real_to_bv32(_real_sin(rx))
            if op == "Cos": return _real_to_bv32(_real_cos(rx))
            return _real_to_bv32(_hw_uf(f"real_{op}", REAL_SORT, REAL_SORT)(rx))
        if op == "FAbs":
            return bx & bv32(0x7FFFFFFF)
        if op == "FNeg":
            return bx ^ bv32(0x80000000)
        if self.fp_mode == "uf" or op in (
            "InverseSqrt", "Sin", "Cos", "Exp", "Exp2", "Log", "Log2",
            "Atan", "Asin", "Acos", "Sinh", "Cosh", "Tanh",
            "DPdx", "DPdy", "Fwidth", "FrexpExp", "FrexpSignificand",
        ):
            return _hw_uf(f"fp_{op}", BV32, BV32)(bx)
        f = bv32_to_fp32(bx)
        if op == "Floor": return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTN(), f))
        if op == "Ceil": return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTP(), f))
        if op == "Trunc": return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTZ(), f))
        if op == "RoundEven": return fp32_to_bv32(z3.fpRoundToIntegral(z3.RNE(), f))
        if op == "Sqrt": return fp32_to_bv32(z3.fpSqrt(z3.RNE(), f))
        return _hw_uf(f"fp_{op}", BV32, BV32)(bx)

    def fp_binary(self, op: str, a: Any, b: Any) -> z3.BitVecRef:
        ba, bb = _as_bv32(a), _as_bv32(b)
        if self.fp_mode == "real":
            ra, rb = _bv32_to_real(ba), _bv32_to_real(bb)
            if op == "FAdd": return _real_to_bv32(ra + rb)
            if op == "FSub": return _real_to_bv32(ra - rb)
            if op == "FMul": return _real_to_bv32(ra * rb)
            if op == "FDiv": return _real_to_bv32(ra / rb)
            if op in ("FMin", "NMin"): return _real_to_bv32(z3.If(ra <= rb, ra, rb))
            if op in ("FMax", "NMax"): return _real_to_bv32(z3.If(ra >= rb, ra, rb))
            if op == "Pow":
                if isinstance(rb, z3.RatNumRef):
                    num, den = rb.numerator_as_long(), rb.denominator_as_long()
                    if num == 0: return _real_to_bv32(_REAL_ONE)
                    if num == den: return _real_to_bv32(ra)
                    if num == 2 * den: return _real_to_bv32(ra * ra)
                    if 2 * num == den: return _real_to_bv32(_real_sqrt(ra))
                    if 2 * num == -den: return _real_to_bv32(_REAL_ONE / _real_sqrt(ra))
                    if num == -den: return _real_to_bv32(_REAL_ONE / ra)
                return _real_to_bv32(_real_exp2(rb * _real_log2(ra)))
            return _real_to_bv32(_hw_uf(f"real_{op}", REAL_SORT, REAL_SORT, REAL_SORT)(ra, rb))
        if self.fp_mode == "uf" or op in ("Pow", "Ldexp", "Atan2"):
            return _hw_uf(f"fp_{op}", BV32, BV32, BV32)(ba, bb)
        fa, fb = bv32_to_fp32(ba), bv32_to_fp32(bb)
        if op == "FAdd": return fp32_to_bv32(z3.fpAdd(z3.RNE(), fa, fb))
        if op == "FSub": return fp32_to_bv32(z3.fpSub(z3.RNE(), fa, fb))
        if op == "FMul": return fp32_to_bv32(z3.fpMul(z3.RNE(), fa, fb))
        if op == "FDiv": return fp32_to_bv32(z3.fpDiv(z3.RNE(), fa, fb))
        if op in ("FMin", "NMin"): return fp32_to_bv32(z3.fpMin(fa, fb))
        if op in ("FMax", "NMax"): return fp32_to_bv32(z3.fpMax(fa, fb))
        return _hw_uf(f"fp_{op}", BV32, BV32, BV32)(ba, bb)

    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> z3.BitVecRef:
        bx, blo, bhi = _as_bv32(x), _as_bv32(lo), _as_bv32(hi)
        if self.fp_mode == "real":
            rx, rlo, rhi = _bv32_to_real(bx), _bv32_to_real(blo), _bv32_to_real(bhi)
            return _real_to_bv32(z3.If(rx < rlo, rlo, z3.If(rx > rhi, rhi, rx)))
        if self.fp_mode == "uf":
            return _hw_uf("fp_FClamp", BV32, BV32, BV32, BV32)(bx, blo, bhi)
        return fp32_to_bv32(z3.fpMin(z3.fpMax(bv32_to_fp32(bx), bv32_to_fp32(blo)), bv32_to_fp32(bhi)))

    def fp_fma(self, a: Any, b: Any, c: Any) -> z3.BitVecRef:
        ba, bb, bc = _as_bv32(a), _as_bv32(b), _as_bv32(c)
        if self.fp_mode == "real":
            return _real_to_bv32(_bv32_to_real(ba) * _bv32_to_real(bb) + _bv32_to_real(bc))
        if self.fp_mode == "uf":
            return _hw_uf("fp_Fma", BV32, BV32, BV32, BV32)(ba, bb, bc)
        bc_is_pos_zero = isinstance(bc, z3.BitVecNumRef) and bc.as_long() == 0
        if bc_is_pos_zero:
            if isinstance(bb, z3.BitVecNumRef) and bb.as_long() == 0x3F800000:
                return z3.If(
                    z3.fpIsNaN(bv32_to_fp32(ba)), fp32_to_bv32(z3.fpNaN(FP32_SORT)),
                    z3.If(ba == bv32(0x80000000), bv32(0), ba),
                )
            if isinstance(ba, z3.BitVecNumRef) and ba.as_long() == 0x3F800000:
                return z3.If(
                    z3.fpIsNaN(bv32_to_fp32(bb)), fp32_to_bv32(z3.fpNaN(FP32_SORT)),
                    z3.If(bb == bv32(0x80000000), bv32(0), bb),
                )
        return fp32_to_bv32(z3.fpFMA(z3.RNE(), bv32_to_fp32(ba), bv32_to_fp32(bb), bv32_to_fp32(bc)))

    def fp_is_nan(self, x: Any) -> z3.BoolRef:
        bx = _as_bv32(x)
        return z3.fpIsNaN(bv32_to_fp32(bx)) if self.fp_mode == "exact" else _hw_uf("fp_IsNan", BV32, z3.BoolSort())(bx)

    def fp_is_inf(self, x: Any) -> z3.BoolRef:
        bx = _as_bv32(x)
        return z3.fpIsInf(bv32_to_fp32(bx)) if self.fp_mode == "exact" else _hw_uf("fp_IsInf", BV32, z3.BoolSort())(bx)

    def fp_cmp(self, op: str, a: Any, b: Any) -> z3.BoolRef:
        ba, bb = _as_bv32(a), _as_bv32(b)
        if self.fp_mode == "real":
            ra, rb = _bv32_to_real(ba), _bv32_to_real(bb)
            if op == "FCmpOrdEq": return ra == rb
            if op == "FCmpOrdGt": return ra > rb
            if op == "FCmpOrdGe": return ra >= rb
            if op == "FCmpOrdLt": return ra < rb
            if op == "FCmpOrdLe": return ra <= rb
            if op in ("FCmpOrdNe", "FCmpUnordNe"): return ra != rb
            if op == "FCmpTotal": return ra <= rb
            return _hw_uf(f"realcmp_{op}", REAL_SORT, REAL_SORT, z3.BoolSort())(ra, rb)
        if self.fp_mode == "uf":
            return _hw_uf(f"fpcmp_{op}", BV32, BV32, z3.BoolSort())(ba, bb)
        fa, fb = bv32_to_fp32(ba), bv32_to_fp32(bb)
        if op == "FCmpOrdEq": return z3.fpEQ(fa, fb)
        if op == "FCmpOrdGt": return z3.fpGT(fa, fb)
        if op == "FCmpOrdGe": return z3.fpGEQ(fa, fb)
        if op == "FCmpOrdLt": return z3.fpLT(fa, fb)
        if op == "FCmpOrdLe": return z3.fpLEQ(fa, fb)
        if op == "FCmpOrdNe": return z3.Or(z3.fpLT(fa, fb), z3.fpGT(fa, fb))
        if op == "FCmpUnordNe": return z3.Not(z3.fpEQ(fa, fb))
        if op == "FCmpTotal": return _fp_total_le_bv(32, ba, bb)
        return _hw_uf(f"fpcmp_{op}", BV32, BV32, z3.BoolSort())(ba, bb)

    def convert(self, op: str, x: Any) -> z3.BitVecRef:
        if op == "ConvertFToS":
            bx = _as_bv32(x)
            if self.fp_mode == "real":
                return _hw_uf("real_FToS", REAL_SORT, BV32)(_bv32_to_real(bx))
            return _fp_to_int_sat(bv32_to_fp32(bx), 32, True, z3.RTZ()) if self.fp_mode == "exact" else _hw_uf("fp_FToS", BV32, BV32)(bx)
        if op == "ConvertFToU":
            bx = _as_bv32(x)
            if self.fp_mode == "real":
                return _hw_uf("real_FToU", REAL_SORT, BV32)(_bv32_to_real(bx))
            return _fp_to_int_sat(bv32_to_fp32(bx), 32, False, z3.RTZ()) if self.fp_mode == "exact" else _hw_uf("fp_FToU", BV32, BV32)(bx)
        if op == "ConvertSToF":
            bx = _as_bv32(x)
            if self.fp_mode == "real":
                if isinstance(bx, z3.BitVecNumRef):
                    return _real_to_bv32(z3.RealVal(bx.as_signed_long()))
                return _real_to_bv32(_hw_uf("real_SToF", BV32, REAL_SORT)(bx))
            return fp32_to_bv32(z3.fpSignedToFP(z3.RNE(), bx, FP32_SORT)) if self.fp_mode == "exact" else _hw_uf("fp_SToF", BV32, BV32)(bx)
        if op == "ConvertUToF":
            bx = _as_bv32(x)
            if self.fp_mode == "real":
                if isinstance(bx, z3.BitVecNumRef):
                    return _real_to_bv32(z3.RealVal(bx.as_long()))
                return _real_to_bv32(_hw_uf("real_UToF", BV32, REAL_SORT)(bx))
            return fp32_to_bv32(z3.fpUnsignedToFP(z3.RNE(), bx, FP32_SORT)) if self.fp_mode == "exact" else _hw_uf("fp_UToF", BV32, BV32)(bx)
        if op == "SConvert32To64":
            return z3.SignExt(32, _as_bv32(x))
        if op == "UConvert32To64":
            return z3.ZeroExt(32, _as_bv32(x))
        if op == "UConvert64To32":
            return z3.Extract(31, 0, _as_bv64(x))
        return bv32(0)

    def load_fau(self, word_idx: int) -> z3.BitVecRef:
        return self.env.load_fau_word(word_idx)

    def load_sideband_u64(self, imm_u32: int, aux_str: str) -> z3.BitVecRef:
        if aux_str and (aux_str.startswith(("ssbo:", "ubo:", "workgroup:")) or aux_str == "push_const"):
            return bv64(self.env.get_buf_base(aux_str))
        if imm_u32 == 6:
            return bv64(self.env.get_buf_base("tls_private"))
        if imm_u32 == 7:
            return bv64(self.env.get_buf_base("wls_shared"))
        return _hw_uf(f"sideband_u64_{imm_u32}", BV32, BV64)(bv32(imm_u32))

    def load_sideband_u32(self, imm_u32: int, imm_u32_2: int) -> z3.BitVecRef:
        return _hw_uf(f"sideband_u32_{imm_u32}", BV32, BV32)(bv32(imm_u32_2))

    def load_builtin(self, builtin_id: int, val_type: str) -> Z3Val:
        return self.env.get_builtin(builtin_id, val_type)

    def load_vert_attr(self, imm_u32: int) -> Tuple[z3.BitVecRef, ...]:
        loc = self.env.sinp_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
        return self.env.get_input_vec4("vert_attr", loc)

    def load_frag_varying(self, imm_u32: int) -> Tuple[z3.BitVecRef, ...]:
        loc = self.env.sinp_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
        return self.env.get_input_vec4("frag_var", loc)

    def image_query_size(self, lod: Any, imm_u32: int, aux_str: str) -> z3.BitVecRef:
        qkey = (aux_str or "img").replace(":", "_")
        return _hw_uf(f"imgq_{qkey}_c{imm_u32}", BV32, BV32)(_as_bv32(lod))

    def tex_sample(self, op: str, args: List[Any], imm_u32: int, aux_str: str) -> Tuple[Tuple[z3.BitVecRef, ...], Z3Val]:
        uv = args[0] if args else (bv32(0), bv32(0))
        u0 = _as_bv32(uv[0]) if isinstance(uv, tuple) else _as_bv32(uv)
        u1 = _as_bv32(uv[1]) if isinstance(uv, tuple) and len(uv) > 1 else bv32(0)
        tkey = aux_str or f"tex_{imm_u32}"
        lanes = tuple(_hw_uf(f"tex_{tkey}_c{c}", BV32, BV32, BV32)(u0, u1) for c in range(4))
        return (lanes, args[-1] if args else bv32(0))

    def psb_load(self, node: Dict[str, Any], args: List[Any], st: BlockEffectState) -> Tuple[z3.BitVecRef, Z3Val]:
        addr = _as_bv64(args[0])
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
        addr = _as_bv64(args[0])
        val = _as_bv32(args[1])
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

    def store_frag_color(self, node: Dict[str, Any], args: List[Any], st: BlockEffectState, reach: z3.BoolRef) -> Z3Val:
        imm_u32 = int(node.get("imm_u32", 0))
        col = args[0]
        store_mask = _z3_and(reach, _z3_not(_z3_or(st.terminated, st.demoted)))
        vec4 = tuple(_as_bv32(col[c]) if isinstance(col, tuple) and c < len(col) else bv32(0) for c in range(4))
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
        vec4 = tuple(_as_bv32(vec[c]) if isinstance(vec, tuple) and c < len(vec) else bv32(0) for c in range(4))
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
        cx = _as_bv32(coord[0]) if isinstance(coord, tuple) and len(coord) > 0 else _as_bv32(coord)
        cy = _as_bv32(coord[1]) if isinstance(coord, tuple) and len(coord) > 1 else bv32(0)
        xy = z3.Concat(cy, cx)
        vec4 = tuple(_as_bv32(texel[c]) if isinstance(texel, tuple) and c < len(texel) else bv32(0) for c in range(4))
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
        addr = _as_bv64(args[0])
        arg = _as_bv32(args[1]) if len(args) > 1 else bv32(0)
        cmp_val = _as_bv32(args[2]) if (op == "AtomicCmpXchg" and len(args) > 2) else bv32(0)
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

    def subgroup_ballot(self, x: Any) -> z3.BitVecRef:
        return _hw_uf("SubgroupBallot", BV32, z3.BitVecSort(5), BV32)(_as_bv32(x), self.env.lane_id)

    def subgroup_shuffle(self, x: Any, lane: Any) -> z3.BitVecRef:
        return _hw_uf("SubgroupShuffle", BV32, BV32, z3.BitVecSort(5), BV32)(
            _as_bv32(x), _as_bv32(lane), self.env.lane_id
        )

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
                    bcond = _as_bool(self.values.get(bcond_id, z3.BoolVal(False)))
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
    """Fast concrete bit-exact interpreter subclassing SsaInterpreter with full op parity."""

    def __init__(self, ssa_json: Dict[str, Any], is_valhall_side: bool, max_block_steps: int = 256):
        self.blocks_by_id: Dict[int, Dict[str, Any]] = {int(b["id"]): b for b in ssa_json.get("blocks", [])}
        self.nodes_by_id: Dict[int, Dict[str, Any]] = {int(n["id"]): n for n in ssa_json.get("nodes", [])}
        self.is_valhall = is_valhall_side
        self.max_block_steps = max_block_steps
        self._env: Optional[ConcreteShaderEnv] = None

    def const_u32(self, imm: int) -> int:
        return imm & MASK32

    def const_u64(self, imm: int) -> int:
        return imm & MASK64

    def const_bool(self, val: bool) -> bool:
        return bool(val)

    def cast_value(self, op: str, val: Any, val_type: str) -> Any:
        if val_type in ("U64", "S64"):
            return _c_u64(val)
        if val_type == "Bool":
            return _c_bool(val)
        if val_type in ("U32", "S32", "F32") and not isinstance(val, tuple):
            return _c_u32(val)
        return val

    def pack64(self, lo: Any, hi: Any) -> int:
        return ((_c_u32(hi) << 32) | _c_u32(lo)) & MASK64

    def split64(self, val64: Any) -> Tuple[int, int]:
        v64 = _c_u64(val64)
        return (v64 & MASK32, (v64 >> 32) & MASK32)

    def unpack_half2x16(self, u: Any) -> Tuple[int, int]:
        cu = _c_u32(u)
        return (_f16_to_f32_u32(cu & 0xFFFF), _f16_to_f32_u32((cu >> 16) & 0xFFFF))

    def pack_half2x16(self, v2: Any) -> int:
        if isinstance(v2, tuple) and len(v2) >= 2:
            return ((_f32_to_f16_u16(_c_u32(v2[1])) << 16) | _f32_to_f16_u16(_c_u32(v2[0]))) & MASK32
        h = _f32_to_f16_u16(_c_u32(v2))
        return ((h << 16) | h) & MASK32

    def int32_unary(self, op: str, x: Any) -> int:
        cx = _c_u32(x)
        if op == "BitNot": return (~cx) & MASK32
        if op == "SNeg": return (-cx) & MASK32
        if op == "SAbs":
            sx = _s32(cx)
            return (-sx if sx < 0 else sx) & MASK32
        if op == "Clz": return 32 if cx == 0 else (32 - cx.bit_length())
        if op == "BitCount": return cx.bit_count()
        if op == "BitReverse": return int(f"{cx:032b}"[::-1], 2)
        return 0

    def int32_binary(self, op: str, a: Any, b: Any) -> int:
        ca, cb = _c_u32(a), _c_u32(b)
        if op == "IAdd": return (ca + cb) & MASK32
        if op == "ISub": return (ca - cb) & MASK32
        if op == "IMul": return (ca * cb) & MASK32
        if op == "BitAnd": return (ca & cb) & MASK32
        if op == "BitOr": return (ca | cb) & MASK32
        if op == "BitXor": return (ca ^ cb) & MASK32
        if op == "Shl": return (ca << (cb & 31)) & MASK32
        if op == "ShrU": return (ca >> (cb & 31)) & MASK32
        if op == "ShrS": return (_s32(ca) >> (cb & 31)) & MASK32
        if op == "RotL":
            sh = cb & 31
            return ((ca << sh) | (ca >> ((32 - sh) & 31))) & MASK32
        if op == "RotR":
            sh = cb & 31
            return ((ca >> sh) | (ca << ((32 - sh) & 31))) & MASK32
        if op == "UDiv": return MASK32 if cb == 0 else (ca // cb) & MASK32
        if op == "SDiv":
            sa, sb = _s32(ca), _s32(cb)
            if sb == 0: return MASK32
            if sa == -0x80000000 and sb == -1: return 0x80000000
            return int(sa / sb) & MASK32
        if op == "UMod": return ca if cb == 0 else (ca % cb) & MASK32
        if op == "SRem":
            sa, sb = _s32(ca), _s32(cb)
            if sb == 0: return sa & MASK32
            if sa == -0x80000000 and sb == -1: return 0
            return (sa - int(sa / sb) * sb) & MASK32
        if op == "SMod":
            sa, sb = _s32(ca), _s32(cb)
            if sb == 0: return sa & MASK32
            if sa == -0x80000000 and sb == -1: return 0
            r = sa - int(sa / sb) * sb
            if r != 0 and (r < 0) != (sb < 0):
                r += sb
            return r & MASK32
        return 0

    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> int:
        cx, coff, cw = _c_u32(x), _c_u32(off) & 31, _c_u32(width)
        if cw == 0: return 0
        if cw >= 32: return cx
        mask = (1 << cw) - 1
        ex = (cx >> coff) & mask
        if signed and (ex & (1 << (cw - 1))):
            ex |= (~mask) & MASK32
        return ex & MASK32

    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> int:
        cb, ci, coff, cw = _c_u32(base), _c_u32(insert), _c_u32(off) & 31, _c_u32(width)
        if cw == 0: return cb
        if cw >= 32: return ci
        mask = (((1 << cw) - 1) << coff) & MASK32
        return ((cb & ~mask) | ((ci << coff) & mask)) & MASK32

    def idot4(self, op: str, a: Any, b: Any) -> int:
        a32, b32 = _c_u32(a), _c_u32(b)
        acc = 0
        for i in range(4):
            ba = (a32 >> (8 * i)) & 0xFF
            bb = (b32 >> (8 * i)) & 0xFF
            if op in ("IDot4S", "IDot4SU") and (ba & 0x80):
                ba -= 256
            if op == "IDot4S" and (bb & 0x80):
                bb -= 256
            acc += ba * bb
        return acc & MASK32

    def int64_unary(self, op: str, x: Any) -> int:
        return (~_c_u64(x)) & MASK64

    def int64_binary(self, op: str, a: Any, b: Any) -> int:
        ca, cb = _c_u64(a), _c_u64(b)
        if op == "IAdd64": return (ca + cb) & MASK64
        if op == "ISub64": return (ca - cb) & MASK64
        if op == "IMul64": return (ca * cb) & MASK64
        if op == "BitAnd64": return (ca & cb) & MASK64
        if op == "BitOr64": return (ca | cb) & MASK64
        if op == "BitXor64": return (ca ^ cb) & MASK64
        if op == "Shl64": return (ca << (cb & 63)) & MASK64
        if op == "ShrU64": return (ca >> (cb & 63)) & MASK64
        if op == "ShrS64": return (_s64(ca) >> (cb & 63)) & MASK64
        return 0

    def icmp(self, op: str, a: Any, b: Any) -> bool:
        ca, cb = _c_u32(a), _c_u32(b)
        if op == "ICmpEq": return ca == cb
        if op == "ICmpNe": return ca != cb
        if op == "ICmpSLt": return _s32(ca) < _s32(cb)
        if op == "ICmpSLe": return _s32(ca) <= _s32(cb)
        if op == "ICmpSGt": return _s32(ca) > _s32(cb)
        if op == "ICmpSGe": return _s32(ca) >= _s32(cb)
        if op == "ICmpULt": return ca < cb
        if op == "ICmpULe": return ca <= cb
        if op == "ICmpUGt": return ca > cb
        if op == "ICmpUGe": return ca >= cb
        return False

    def logical_and(self, a: Any, b: Any) -> bool:
        return _c_bool(a) and _c_bool(b)

    def logical_or(self, a: Any, b: Any) -> bool:
        return _c_bool(a) or _c_bool(b)

    def logical_not(self, a: Any) -> bool:
        return not _c_bool(a)

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        return t_val if _c_bool(cond) else f_val

    def fp_unary(self, op: str, x: Any) -> int:
        u = _c_u32(x)
        if op == "FAbs": return u & 0x7FFFFFFF
        if op == "FNeg": return u ^ 0x80000000
        f = _u32_to_f32(u)
        if math.isnan(f): return 0x7FC00000
        if op == "Floor": return _f32_to_u32(math.floor(f)) if math.isfinite(f) else u
        if op == "Ceil": return _f32_to_u32(math.ceil(f)) if math.isfinite(f) else u
        if op == "Trunc": return _f32_to_u32(math.trunc(f)) if math.isfinite(f) else u
        if op == "RoundEven": return _f32_to_u32(float(round(f))) if math.isfinite(f) else u
        if op == "Sqrt": return 0x7FC00000 if f < 0.0 else _f32_to_u32(math.sqrt(f))
        if op == "InverseSqrt":
            if f < 0.0: return 0x7FC00000
            if f == 0.0: return 0x7F800000
            return _f32_to_u32(1.0 / math.sqrt(f))
        if op == "Sin": return _f32_to_u32(math.sin(f)) if math.isfinite(f) else 0x7FC00000
        if op == "Cos": return _f32_to_u32(math.cos(f)) if math.isfinite(f) else 0x7FC00000
        if op == "Exp":
            try: return _f32_to_u32(math.exp(f))
            except OverflowError: return 0x7F800000
        if op == "Exp2":
            try: return _f32_to_u32(math.pow(2.0, f))
            except OverflowError: return 0x7F800000
        if op == "Log":
            if f < 0.0: return 0x7FC00000
            if f == 0.0: return 0xFF800000
            return _f32_to_u32(math.log(f))
        if op == "Log2":
            if f < 0.0: return 0x7FC00000
            if f == 0.0: return 0xFF800000
            return _f32_to_u32(math.log2(f))
        return _det_hash_u32(f"fp_{op}", u)

    def fp_binary(self, op: str, a: Any, b: Any) -> int:
        ua, ub = _c_u32(a), _c_u32(b)
        fa, fb = _u32_to_f32(ua), _u32_to_f32(ub)
        if op == "FAdd": return _f32_to_u32(fa + fb)
        if op == "FSub": return _f32_to_u32(fa - fb)
        if op == "FMul": return _f32_to_u32(fa * fb)
        if op == "FDiv":
            if math.isnan(fa) or math.isnan(fb): return 0x7FC00000
            if fb == 0.0:
                if fa == 0.0: return 0x7FC00000
                return ((ua ^ ub) & 0x80000000) | 0x7F800000
            return _f32_to_u32(fa / fb)
        if op in ("FMin", "NMin"):
            if math.isnan(fa): return ub
            if math.isnan(fb): return ua
            return ua if fa <= fb else ub
        if op in ("FMax", "NMax"):
            if math.isnan(fa): return ub
            if math.isnan(fb): return ua
            return ua if fa >= fb else ub
        if op == "Pow":
            try: return _f32_to_u32(math.pow(fa, fb))
            except Exception: return 0x7FC00000
        return _det_hash_u32(f"fp_{op}", ua, ub)

    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> int:
        return self.fp_binary("FMin", self.fp_binary("FMax", x, lo), hi)

    def fp_fma(self, a: Any, b: Any, c: Any) -> int:
        fa, fb, fc = _u32_to_f32(_c_u32(a)), _u32_to_f32(_c_u32(b)), _u32_to_f32(_c_u32(c))
        return _f32_to_u32(fa * fb + fc)

    def fp_is_nan(self, x: Any) -> bool:
        return math.isnan(_u32_to_f32(_c_u32(x)))

    def fp_is_inf(self, x: Any) -> bool:
        return math.isinf(_u32_to_f32(_c_u32(x)))

    def fp_cmp(self, op: str, a: Any, b: Any) -> bool:
        ua, ub = _c_u32(a), _c_u32(b)
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
            return _key(ua) <= _key(ub)
        return False

    def convert(self, op: str, x: Any) -> int:
        if op == "ConvertFToS":
            f = _u32_to_f32(_c_u32(x))
            if math.isnan(f): return 0
            if f >= 2147483647.0: return 0x7FFFFFFF
            if f <= -2147483648.0: return 0x80000000
            return int(math.trunc(f)) & MASK32
        if op == "ConvertFToU":
            f = _u32_to_f32(_c_u32(x))
            if math.isnan(f) or f <= 0.0: return 0
            if f >= 4294967295.0: return MASK32
            return int(math.trunc(f)) & MASK32
        if op == "ConvertSToF": return _f32_to_u32(float(_s32(_c_u32(x))))
        if op == "ConvertUToF": return _f32_to_u32(float(_c_u32(x)))
        if op == "SConvert32To64": return _s32(_c_u32(x)) & MASK64
        if op == "UConvert32To64": return _c_u32(x) & MASK64
        if op == "UConvert64To32": return _c_u64(x) & MASK32
        return 0

    def load_fau(self, word_idx: int) -> int:
        assert self._env is not None
        return self._env.load_fau_word(word_idx)

    def load_sideband_u64(self, imm_u32: int, aux_str: str) -> int:
        assert self._env is not None
        if aux_str and (aux_str.startswith(("ssbo:", "ubo:", "workgroup:")) or aux_str == "push_const"):
            return self._env.get_buf_base(aux_str)
        if imm_u32 == 6: return self._env.get_buf_base("tls_private")
        if imm_u32 == 7: return self._env.get_buf_base("wls_shared")
        return _det_hash_u32(f"sb64_{imm_u32}", self._env.seed, self._env.vec_idx)

    def load_sideband_u32(self, imm_u32: int, imm_u32_2: int) -> int:
        assert self._env is not None
        return self._env.eval_hw_uf(f"hw_sideband_u32_{imm_u32}", imm_u32_2, fallback_tag=f"sb32_{imm_u32}_{imm_u32_2}")

    def load_builtin(self, builtin_id: int, val_type: str) -> Any:
        assert self._env is not None
        return self._env.get_builtin(builtin_id, val_type)

    def load_vert_attr(self, imm_u32: int) -> Tuple[int, ...]:
        assert self._env is not None
        loc = self._env.sinp_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
        return self._env.get_input_vec4("vert_attr", loc)

    def load_frag_varying(self, imm_u32: int) -> Tuple[int, ...]:
        assert self._env is not None
        loc = self._env.sinp_addr_to_loc.get(imm_u32, imm_u32) if self.is_valhall else imm_u32
        return self._env.get_input_vec4("frag_var", loc)

    def image_query_size(self, lod: Any, imm_u32: int, aux_str: str) -> int:
        assert self._env is not None
        qkey = (aux_str or "img").replace(":", "_")
        return self._env.eval_hw_uf(f"hw_imgq_{qkey}_c{imm_u32}", _c_u32(lod), fallback_tag=f"imgq_{qkey}_{imm_u32}")

    def tex_sample(self, op: str, args: List[Any], imm_u32: int, aux_str: str) -> Tuple[Tuple[int, ...], Any]:
        assert self._env is not None
        uv = args[0] if args else (0, 0)
        u0 = _c_u32(uv[0]) if isinstance(uv, tuple) else _c_u32(uv)
        u1 = _c_u32(uv[1]) if isinstance(uv, tuple) and len(uv) > 1 else 0
        tkey = aux_str or f"tex_{imm_u32}"
        lanes = tuple(self._env.eval_hw_uf(f"hw_tex_{tkey}_c{c}", u0, u1, fallback_tag=f"tex_{tkey}_c{c}") for c in range(4))
        return (lanes, args[-1] if args else 0)

    def psb_load(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState) -> Tuple[int, Any]:
        assert self._env is not None
        addr = _c_u64(args[0])
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
            addr = _c_u64(args[0])
            val = _c_u32(args[1])
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
        kcond = _c_bool(args[0]) if (aux_bool and len(args) >= 2) else True
        if kcond:
            if aux_bool2: st.demoted = True
            else: st.terminated = True
        return 0

    def store_frag_color(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState, reach: Any) -> Any:
        if not (st.terminated or st.demoted):
            col = args[0]
            imm32 = int(node.get("imm_u32", 0))
            st.frag_colors[imm32] = tuple(_c_u32(col[c]) if isinstance(col, tuple) and c < len(col) else 0 for c in range(4))
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
            st.vert_varyings[loc] = tuple(_c_u32(vec[c]) if isinstance(vec, tuple) and c < len(vec) else 0 for c in range(4))
        return args[-1] if args else 0

    def store_texel(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState, reach: Any) -> Any:
        if not (st.terminated or st.demoted):
            coord = args[0]
            texel = args[1] if len(args) > 1 else (0, 0, 0, 0)
            cx = _c_u32(coord[0]) if isinstance(coord, tuple) and len(coord) > 0 else _c_u32(coord)
            cy = _c_u32(coord[1]) if isinstance(coord, tuple) and len(coord) > 1 else 0
            xy = ((cy << 32) | cx) & MASK64
            ikey = node.get("aux_str", "") or f"img_{int(node.get('imm_u32', 0))}"
            st.img_writes[(ikey, xy)] = [_c_u32(texel[c]) if isinstance(texel, tuple) and c < len(texel) else 0 for c in range(4)]
        return args[-1] if args else 0

    def atomic_op(self, node: Dict[str, Any], args: List[Any], st: _ConcreteState, reach: Any) -> Tuple[int, Any]:
        assert self._env is not None
        op = node["op"]
        aux_str = node.get("aux_str", "")
        addr = _c_u64(args[0])
        arg = _c_u32(args[1]) if len(args) > 1 else 0
        cmp_val = _c_u32(args[2]) if (op == "AtomicCmpXchg" and len(args) > 2) else 0
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

    def subgroup_ballot(self, x: Any) -> int:
        assert self._env is not None
        return _det_hash_u32("SubgroupBallot", self._env.seed, self._env.vec_idx, _c_u32(x), self._env.lane_id)

    def subgroup_shuffle(self, x: Any, lane: Any) -> int:
        assert self._env is not None
        return _det_hash_u32("SubgroupShuffle", self._env.seed, self._env.vec_idx, _c_u32(x), _c_u32(lane), self._env.lane_id)

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
                cond_v = _c_bool(vals.get(int(blk.get("branch_cond", 0)), False))
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
                        outs[f"builtin_out[{bi}][{c}]"] = _c_u32(v)
                else:
                    outs[f"builtin_out[{bi}]"] = _c_u32(payload)
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
    ev = SymbolicSsaInterpreter(ssa_json, env, is_valhall_side=is_valhall_side, fp_mode=fp_mode, max_unroll=max_unroll)
    st = ev.evaluate()
    return ev, st, env
