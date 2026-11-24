"""
Pure-Python SPIR-V 1.3 Compute Shader Binary Builder.

Emits valid, self-contained Vulkan GLCompute SPIR-V binaries with:
  - Fixed ABI harness: `set=0, binding=0` read-only input SSBO (`uint in_data[16]`)
    and `set=0, binding=1` output SSBO (`uint out_data[4]`).
  - Anti-DCE sink accumulators: every generated instruction folds into live
    `acc_u32` / `acc_f32` accumulators written to `out_data[0..3]` so `libMali`
    cannot dead-code-eliminate the synthesized body.
  - Configurable `num_blocks` (`N >= 1`): `N = 1` emits a straight-line single
    block; `N > 1` emits structured acyclic `OpSelectionMerge` diamonds driven
    by `in_data` predicates with `OpPhi` accumulator merges.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

SPV_MAGIC = 0x07230203
SPV_VERSION_1_3 = 0x00010300
SPV_GENERATOR = 0x00000000

# Core SPIR-V opcodes
OP_NOP = 0
OP_EXT_INST_IMPORT = 11
OP_EXT_INST = 12
OP_MEMORY_MODEL = 14
OP_ENTRY_POINT = 15
OP_EXECUTION_MODE = 16
OP_CAPABILITY = 17
OP_TYPE_VOID = 19
OP_TYPE_BOOL = 20
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_VECTOR = 23
OP_TYPE_ARRAY = 28
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_TYPE_FUNCTION = 33
OP_CONSTANT = 43
OP_FUNCTION = 54
OP_FUNCTION_END = 56
OP_VARIABLE = 59
OP_LOAD = 61
OP_STORE = 62
OP_ACCESS_CHAIN = 65
OP_DECORATE = 71
OP_MEMBER_DECORATE = 72
OP_COMPOSITE_CONSTRUCT = 80
OP_COMPOSITE_EXTRACT = 81
OP_CONVERT_F_TO_U = 109
OP_CONVERT_F_TO_S = 110
OP_CONVERT_S_TO_F = 111
OP_CONVERT_U_TO_F = 112
OP_BITCAST = 124
OP_SNEGATE = 126
OP_FNEGATE = 127
OP_IADD = 128
OP_FADD = 129
OP_ISUB = 130
OP_FSUB = 131
OP_IMUL = 132
OP_FMUL = 133
OP_UDIV = 134
OP_SDIV = 135
OP_FDIV = 136
OP_UMOD = 137
OP_SREM = 138
OP_SMOD = 139
OP_IS_NAN = 156
OP_IS_INF = 157
OP_LOGICAL_OR = 166
OP_LOGICAL_AND = 167
OP_LOGICAL_NOT = 168
OP_SELECT = 169
OP_IEQUAL = 170
OP_INOT_EQUAL = 171
OP_UGREATER_THAN = 172
OP_SGREATER_THAN = 173
OP_ULESS_THAN = 176
OP_SLESS_THAN = 177
OP_FORD_EQUAL = 180
OP_FORD_LESS_THAN = 184
OP_FORD_GREATER_THAN = 186
OP_SHIFT_RIGHT_LOGICAL = 194
OP_SHIFT_RIGHT_ARITHMETIC = 195
OP_SHIFT_LEFT_LOGICAL = 196
OP_BITWISE_OR = 197
OP_BITWISE_XOR = 198
OP_BITWISE_AND = 199
OP_NOT = 200
OP_BITFIELD_INSERT = 201
OP_BITFIELD_S_EXTRACT = 202
OP_BITFIELD_U_EXTRACT = 203
OP_BIT_REVERSE = 204
OP_BIT_COUNT = 205
OP_PHI = 245
OP_SELECTION_MERGE = 247
OP_LABEL = 248
OP_BRANCH = 249
OP_BRANCH_CONDITIONAL = 250
OP_RETURN = 253

# GLSL.std.450 extended instruction IDs
GLSL_ROUND_EVEN = 2
GLSL_TRUNC = 3
GLSL_FABS = 4
GLSL_SABS = 5
GLSL_FLOOR = 8
GLSL_CEIL = 9
GLSL_SIN = 13
GLSL_COS = 14
GLSL_POW = 26
GLSL_EXP = 27
GLSL_LOG = 28
GLSL_EXP2 = 29
GLSL_LOG2 = 30
GLSL_SQRT = 31
GLSL_INVERSE_SQRT = 32
GLSL_FMIN = 37
GLSL_UMIN = 38
GLSL_SMIN = 39
GLSL_FMAX = 40
GLSL_UMAX = 41
GLSL_SMAX = 42
GLSL_FCLAMP = 43
GLSL_FMIX = 46
GLSL_FMA = 50
GLSL_PACK_HALF2X16 = 58
GLSL_UNPACK_HALF2X16 = 62
GLSL_FIND_UMSB = 75
GLSL_FIND_SMSB = 76


def _encode_str(s: str) -> List[int]:
    b = s.encode("utf-8") + b"\x00"
    while len(b) % 4 != 0:
        b += b"\x00"
    return list(struct.unpack(f"<{len(b)//4}I", b))


def _inst(op: int, *operands: int) -> List[int]:
    return [((len(operands) + 1) << 16) | op, *operands]


class SpirvComputeBuilder:
    """Builds a valid Vulkan Compute SPIR-V 1.3 binary from a sequence of typed OpSpecs."""

    def __init__(self) -> None:
        self._next_id = 1
        self.header_decls: List[int] = []
        self.decorations: List[int] = []
        self.types_consts: List[int] = []
        self.body: List[int] = []

        self.glsl_ext = self.alloc_id()
        self.entry_fn = self.alloc_id()

        # Core types
        self.t_void = self.alloc_id()
        self.t_fn = self.alloc_id()
        self.t_bool = self.alloc_id()
        self.t_u32 = self.alloc_id()
        self.t_s32 = self.alloc_id()
        self.t_f32 = self.alloc_id()
        self.t_v2f32 = self.alloc_id()
        self.t_v3u32 = self.alloc_id()
        self.t_ptr_v3u32_in = self.alloc_id()
        self.t_ptr_u32_in = self.alloc_id()
        self.var_gid = self.alloc_id()

        # Constants
        self.c_u32: Dict[int, int] = {}
        self.c_s32: Dict[int, int] = {}
        self.c_f32: Dict[int, int] = {}

        # Buffer ABI types & variables
        self.t_arr16 = self.alloc_id()
        self.t_arr4 = self.alloc_id()
        self.t_in_struct = self.alloc_id()
        self.t_out_struct = self.alloc_id()
        self.t_ptr_in_sb = self.alloc_id()
        self.t_ptr_out_sb = self.alloc_id()
        self.t_ptr_u32_sb = self.alloc_id()
        self.var_in = self.alloc_id()
        self.var_out = self.alloc_id()

        self._init_module_preamble()

    def alloc_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def get_const_u32(self, val: int) -> int:
        v = val & 0xFFFFFFFF
        if v not in self.c_u32:
            cid = self.alloc_id()
            self.types_consts.extend(_inst(OP_CONSTANT, self.t_u32, cid, v))
            self.c_u32[v] = cid
        return self.c_u32[v]

    def get_const_s32(self, val: int) -> int:
        v = val & 0xFFFFFFFF
        if v not in self.c_s32:
            cid = self.alloc_id()
            self.types_consts.extend(_inst(OP_CONSTANT, self.t_s32, cid, v))
            self.c_s32[v] = cid
        return self.c_s32[v]

    def get_const_f32_bits(self, bits: int) -> int:
        v = bits & 0xFFFFFFFF
        if v not in self.c_f32:
            cid = self.alloc_id()
            self.types_consts.extend(_inst(OP_CONSTANT, self.t_f32, cid, v))
            self.c_f32[v] = cid
        return self.c_f32[v]

    def _init_module_preamble(self) -> None:
        self.types_consts.extend(_inst(OP_TYPE_VOID, self.t_void))
        self.types_consts.extend(_inst(OP_TYPE_FUNCTION, self.t_fn, self.t_void))
        self.types_consts.extend(_inst(OP_TYPE_BOOL, self.t_bool))
        self.types_consts.extend(_inst(OP_TYPE_INT, self.t_u32, 32, 0))
        self.types_consts.extend(_inst(OP_TYPE_INT, self.t_s32, 32, 1))
        self.types_consts.extend(_inst(OP_TYPE_FLOAT, self.t_f32, 32))
        self.types_consts.extend(_inst(OP_TYPE_VECTOR, self.t_v2f32, self.t_f32, 2))
        self.types_consts.extend(_inst(OP_TYPE_VECTOR, self.t_v3u32, self.t_u32, 3))
        self.types_consts.extend(_inst(OP_TYPE_POINTER, self.t_ptr_v3u32_in, 1, self.t_v3u32))
        self.types_consts.extend(_inst(OP_TYPE_POINTER, self.t_ptr_u32_in, 1, self.t_u32))
        self.types_consts.extend(_inst(OP_VARIABLE, self.t_ptr_v3u32_in, self.var_gid, 1))

        c16 = self.get_const_u32(16)
        c4 = self.get_const_u32(4)
        self.types_consts.extend(_inst(OP_TYPE_ARRAY, self.t_arr16, self.t_u32, c16))
        self.types_consts.extend(_inst(OP_TYPE_ARRAY, self.t_arr4, self.t_u32, c4))
        self.types_consts.extend(_inst(OP_TYPE_STRUCT, self.t_in_struct, self.t_arr16))
        self.types_consts.extend(_inst(OP_TYPE_STRUCT, self.t_out_struct, self.t_arr4))
        # StorageClass StorageBuffer = 12
        self.types_consts.extend(_inst(OP_TYPE_POINTER, self.t_ptr_in_sb, 12, self.t_in_struct))
        self.types_consts.extend(_inst(OP_TYPE_POINTER, self.t_ptr_out_sb, 12, self.t_out_struct))
        self.types_consts.extend(_inst(OP_TYPE_POINTER, self.t_ptr_u32_sb, 12, self.t_u32))
        self.types_consts.extend(_inst(OP_VARIABLE, self.t_ptr_in_sb, self.var_in, 12))
        self.types_consts.extend(_inst(OP_VARIABLE, self.t_ptr_out_sb, self.var_out, 12))

        # ArrayStride = 6, Block = 2, Offset = 35, DescriptorSet = 34, Binding = 33, NonWritable = 24
        self.decorations.extend(_inst(OP_DECORATE, self.t_arr16, 6, 4))
        self.decorations.extend(_inst(OP_DECORATE, self.t_arr4, 6, 4))
        self.decorations.extend(_inst(OP_DECORATE, self.t_in_struct, 2))
        self.decorations.extend(_inst(OP_DECORATE, self.t_out_struct, 2))
        self.decorations.extend(_inst(OP_MEMBER_DECORATE, self.t_in_struct, 0, 35, 0))
        self.decorations.extend(_inst(OP_MEMBER_DECORATE, self.t_out_struct, 0, 35, 0))
        self.decorations.extend(_inst(OP_DECORATE, self.var_in, 34, 0))
        self.decorations.extend(_inst(OP_DECORATE, self.var_in, 33, 0))
        self.decorations.extend(_inst(OP_DECORATE, self.var_in, 24))
        self.decorations.extend(_inst(OP_DECORATE, self.var_out, 34, 0))
        self.decorations.extend(_inst(OP_DECORATE, self.var_out, 33, 1))
        # BuiltIn = 11, GlobalInvocationId = 28
        self.decorations.extend(_inst(OP_DECORATE, self.var_gid, 11, 28))

    def emit(self, op: int, *operands: int) -> None:
        self.body.extend(_inst(op, *operands))

    def load_gid_x(self) -> int:
        c0 = self.get_const_u32(0)
        ptr = self.alloc_id()
        val = self.alloc_id()
        self.emit(OP_ACCESS_CHAIN, self.t_ptr_u32_in, ptr, self.var_gid, c0)
        self.emit(OP_LOAD, self.t_u32, val, ptr)
        return val

    def load_in_u32_dyn(self, idx_id: int) -> int:
        c0 = self.get_const_u32(0)
        ptr = self.alloc_id()
        val = self.alloc_id()
        self.emit(OP_ACCESS_CHAIN, self.t_ptr_u32_sb, ptr, self.var_in, c0, idx_id)
        self.emit(OP_LOAD, self.t_u32, val, ptr)
        return val

    def load_in_u32(self, slot: int) -> int:
        c0 = self.get_const_u32(0)
        cs = self.get_const_u32(slot)
        ptr = self.alloc_id()
        val = self.alloc_id()
        self.emit(OP_ACCESS_CHAIN, self.t_ptr_u32_sb, ptr, self.var_in, c0, cs)
        self.emit(OP_LOAD, self.t_u32, val, ptr)
        return val

    def store_out_u32(self, slot: int, val_id: int) -> None:
        c0 = self.get_const_u32(0)
        cs = self.get_const_u32(slot)
        ptr = self.alloc_id()
        self.emit(OP_ACCESS_CHAIN, self.t_ptr_u32_sb, ptr, self.var_out, c0, cs)
        self.emit(OP_STORE, ptr, val_id)

    def build_bytes(self) -> bytes:
        words: List[int] = [SPV_MAGIC, SPV_VERSION_1_3, SPV_GENERATOR, self._next_id, 0]
        # Capability Shader (1)
        words.extend(_inst(OP_CAPABILITY, 1))
        words.extend(_inst(OP_EXT_INST_IMPORT, self.glsl_ext, *_encode_str("GLSL.std.450")))
        # MemoryModel Logical (0) GLSL450 (1)
        words.extend(_inst(OP_MEMORY_MODEL, 0, 1))
        # EntryPoint GLCompute (5)
        words.extend(_inst(OP_ENTRY_POINT, 5, self.entry_fn, *_encode_str("main"), self.var_gid))
        # ExecutionMode LocalSize (17) 1 1 1
        words.extend(_inst(OP_EXECUTION_MODE, self.entry_fn, 17, 1, 1, 1))
        words.extend(self.decorations)
        words.extend(self.types_consts)
        words.extend(_inst(OP_FUNCTION, self.t_void, self.entry_fn, 0, self.t_fn))
        words.extend(self.body)
        words.extend(_inst(OP_FUNCTION_END))
        return struct.pack(f"<{len(words)}I", *words)
