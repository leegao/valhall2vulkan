#!/usr/bin/env python3
"""
Abstract Denotational SSA Interpreter (`SsaInterpreter`).

Provides the single canonical opcode dispatch table (`eval_node`) over all
ValhallSsaModule instructions, delegating value-level operations to a pluggable
`SsaDomain` (`ConcreteDomain`, `SymbolicDomain`, or `AbstractTopDomain`) and
sharing default environment loads (`load_fau`, `load_sideband_u64`,
`load_builtin`, `load_vert_attr`, `load_frag_varying`) across subclasses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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


class SsaInterpreter:
    """
    Denotational interpreter base class for ValhallSsaModule.

    Delegates pure value operations to `self.domain` (an `SsaDomain`) and
    implements common environment reads (`load_fau`, `load_sideband_u64`,
    `load_builtin`, `load_vert_attr`, `load_frag_varying`) once for both
    concrete and symbolic subclasses.
    """

    domain: SsaDomain = AbstractTopDomain()
    _env: Any = None
    is_valhall: bool = False

    def default_value(self) -> Any:
        return self.domain.default_value()

    def lookup_arg(self, values: Dict[int, Any], aid: int) -> Any:
        return values.get(aid, self.default_value())

    def const_u32(self, imm: int) -> Any:
        return self.domain.const_u32(imm)

    def const_u64(self, imm: int) -> Any:
        return self.domain.const_u64(imm)

    def const_bool(self, val: bool) -> Any:
        return self.domain.const_bool(val)

    def cast_value(self, op: str, val: Any, val_type: str) -> Any:
        return self.domain.cast_value(op, val, val_type)

    def pack64(self, lo: Any, hi: Any) -> Any:
        return self.domain.pack64(lo, hi)

    def split64(self, val64: Any) -> Tuple[Any, Any]:
        return self.domain.split64(val64)

    def unpack_half2x16(self, u: Any) -> Tuple[Any, Any]:
        return self.domain.unpack_half2x16(u)

    def pack_half2x16(self, v2: Any) -> Any:
        return self.domain.pack_half2x16(v2)

    def int32_unary(self, op: str, x: Any) -> Any:
        return self.domain.int32_unary(op, x)

    def int32_binary(self, op: str, a: Any, b: Any) -> Any:
        return self.domain.int32_binary(op, a, b)

    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> Any:
        return self.domain.bfe(signed, x, off, width)

    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> Any:
        return self.domain.bfi(base, insert, off, width)

    def idot4(self, op: str, a: Any, b: Any) -> Any:
        return self.domain.idot4(op, a, b)

    def int64_unary(self, op: str, x: Any) -> Any:
        return self.domain.int64_unary(op, x)

    def int64_binary(self, op: str, a: Any, b: Any) -> Any:
        return self.domain.int64_binary(op, a, b)

    def icmp(self, op: str, a: Any, b: Any) -> Any:
        return self.domain.icmp(op, a, b)

    def logical_and(self, a: Any, b: Any) -> Any:
        return self.domain.logical_and(a, b)

    def logical_or(self, a: Any, b: Any) -> Any:
        return self.domain.logical_or(a, b)

    def logical_not(self, a: Any) -> Any:
        return self.domain.logical_not(a)

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        return self.domain.select(cond, t_val, f_val)

    def fp_unary(self, op: str, x: Any) -> Any:
        return self.domain.fp_unary(op, x)

    def fp_binary(self, op: str, a: Any, b: Any) -> Any:
        return self.domain.fp_binary(op, a, b)

    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> Any:
        return self.domain.fp_clamp(x, lo, hi)

    def fp_fma(self, a: Any, b: Any, c: Any) -> Any:
        return self.domain.fp_fma(a, b, c)

    def fp_is_nan(self, x: Any) -> Any:
        return self.domain.fp_is_nan(x)

    def fp_is_inf(self, x: Any) -> Any:
        return self.domain.fp_is_inf(x)

    def fp_cmp(self, op: str, a: Any, b: Any) -> Any:
        return self.domain.fp_cmp(op, a, b)

    def convert(self, op: str, x: Any) -> Any:
        return self.domain.convert(op, x)

    def load_fau(self, word_idx: int) -> Any:
        if self._env is None:
            return self.default_value()
        return self._env.load_fau_word(word_idx)

    def load_sideband_u64(self, imm_u32: int, aux_str: str) -> Any:
        if self._env is None:
            return self.default_value()
        if aux_str and (aux_str.startswith(("ssbo:", "ubo:", "workgroup:")) or aux_str == "push_const"):
            return self.const_u64(self._env.get_buf_base(aux_str))
        if imm_u32 == 6:
            return self.const_u64(self._env.get_buf_base("tls_private"))
        if imm_u32 == 7:
            return self.const_u64(self._env.get_buf_base("wls_shared"))
        return self.domain.unknown_sideband_u64(self._env, imm_u32)

    def load_sideband_u32(self, imm_u32: int, imm_u32_2: int) -> Any:
        if self._env is None:
            return self.default_value()
        return self.domain.sideband_u32(self._env, imm_u32, imm_u32_2)

    def load_builtin(self, builtin_id: int, val_type: str) -> Any:
        if self._env is None:
            return self.default_value()
        return self._env.get_builtin(builtin_id, val_type)

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
        if self._env is None:
            return self.default_value()
        return self.domain.subgroup_ballot(self._env, x)

    def subgroup_shuffle(self, x: Any, lane: Any) -> Any:
        if self._env is None:
            return self.default_value()
        return self.domain.subgroup_shuffle(self._env, x, lane)

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
            return self._elemwise(lambda x: self.int32_unary(op, x), args[0])
        if op in INT32_BINARY_OPS:
            return self._elemwise(lambda a, b: self.int32_binary(op, a, b), args[0], args[1])
        if op in ("BitFieldUExtract", "BitFieldSExtract"):
            return self.bfe(op == "BitFieldSExtract", args[0], args[1], args[2])
        if op == "BitFieldInsert":
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
            return self.fp_is_nan(args[0])
        if op == "IsInf":
            return self.fp_is_inf(args[0])
        if op.startswith("FCmp"):
            return self.fp_cmp(op, args[0], args[1])
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
        if op == "SubgroupBallot":
            return self.subgroup_ballot(args[0] if args else self.const_u32(0))
        if op == "SubgroupShuffle":
            return self.subgroup_shuffle(
                args[0] if args else self.const_u32(0),
                args[1] if len(args) > 1 else self.const_u32(0),
            )
        if op in CONTROL_PASSTHROUGH_OPS:
            return args[-1] if args else self.default_value()
        return self.default_value()

    @staticmethod
    def _elemwise(fn, *xs):
        lens = [len(x) for x in xs if isinstance(x, tuple)]
        if not lens:
            return fn(*xs)
        n = min(lens)
        return tuple(fn(*(x[i] if isinstance(x, tuple) else x for x in xs)) for i in range(n))
