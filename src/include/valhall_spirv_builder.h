#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <tuple>

namespace valhall::spirv {

// ============================================================================
// SPIR-V 1.3 Core Constants & Opcodes
// ============================================================================
constexpr uint32_t kSpvMagicNumber = 0x07230203u;
constexpr uint32_t kSpvVersion13   = 0x00010300u;
constexpr uint32_t kSpvGeneratorId = 0x001B0001u;

enum class ExecutionModel : uint32_t {
    Vertex = 0,
    Fragment = 4,
    GLCompute = 5,
};

enum class AddressingModel : uint32_t {
    Logical = 0,
    PhysicalStorageBuffer64 = 5348,
};

enum class MemoryModel : uint32_t {
    GLSL450 = 1,
};

enum class ExecutionMode : uint32_t {
    OriginUpperLeft = 7,
    DepthReplacing = 12,
    LocalSize = 17,
};

enum class StorageClass : uint32_t {
    UniformConstant = 0,
    Input = 1,
    Uniform = 2,
    Output = 3,
    Workgroup = 4,
    Private = 6,
    Function = 7,
    PushConstant = 9,
    StorageBuffer = 12,
    PhysicalStorageBuffer = 5349,
};

enum class Dim : uint32_t {
    Dim1D = 0,
    Dim2D = 1,
    Dim3D = 2,
    Cube = 3,
    Rect = 4,
    Buffer = 5,
};

enum class Decoration : uint32_t {
    Block = 2,
    BufferBlock = 3,
    ArrayStride = 6,
    BuiltIn = 11,
    NoPerspective = 13,
    Flat = 14,
    NonWritable = 24,
    NonReadable = 25,
    Location = 30,
    Binding = 33,
    DescriptorSet = 34,
    Offset = 35,
    AliasedPointer = 5350,
};

enum class BuiltIn : uint32_t {
    Position = 0,
    PointSize = 1,
    FragCoord = 15,
    PointCoord = 16,
    FrontFacing = 17,
    SampleId = 18,
    SampleMask = 20,
    FragDepth = 22,
    NumWorkgroups = 24,
    WorkgroupSize = 25,
    WorkgroupId = 26,
    LocalInvocationId = 27,
    GlobalInvocationId = 28,
    LocalInvocationIndex = 29,
    SubgroupSize = 36,
    NumSubgroups = 38,
    SubgroupId = 40,
    SubgroupLocalInvocationId = 41,
    VertexIndex = 42,
    InstanceIndex = 43,
};

enum class Capability : uint32_t {
    Matrix = 0,
    Shader = 1,
    Float16 = 9,
    Float64 = 10,
    Int64 = 11,
    Int16 = 22,
    Int8 = 39,
    Sampled1D = 43,
    Image1D = 44,
    SampledBuffer = 46,
    ImageBuffer = 47,
    ImageQuery = 50,
    DerivativeControl = 51,
    GroupNonUniform = 61,
    GroupNonUniformVote = 62,
    GroupNonUniformArithmetic = 63,
    GroupNonUniformBallot = 64,
    GroupNonUniformShuffle = 65,
    GroupNonUniformShuffleRelative = 66,
    StorageBuffer16BitAccess = 4433,
    UniformAndStorageBuffer16BitAccess = 4434,
    StorageBuffer8BitAccess = 4448,
    UniformAndStorageBuffer8BitAccess = 4449,
    PhysicalStorageBufferAddresses = 5347,
};

enum class Op : uint16_t {
    OpNop = 0,
    OpSource = 3,
    OpName = 5,
    OpMemberName = 6,
    OpExtension = 10,
    OpExtInstImport = 11,
    OpExtInst = 12,
    OpMemoryModel = 14,
    OpEntryPoint = 15,
    OpExecutionMode = 16,
    OpCapability = 17,
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeMatrix = 24,
    OpTypeImage = 25,
    OpTypeSampler = 26,
    OpTypeSampledImage = 27,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpTypeFunction = 33,
    OpConstantTrue = 41,
    OpConstantFalse = 42,
    OpConstant = 43,
    OpConstantComposite = 44,
    OpConstantNull = 46,
    OpFunction = 54,
    OpFunctionParameter = 55,
    OpFunctionEnd = 56,
    OpFunctionCall = 57,
    OpVariable = 59,
    OpLoad = 61,
    OpStore = 62,
    OpAccessChain = 65,
    OpDecorate = 71,
    OpMemberDecorate = 72,
    OpVectorExtractDynamic = 77,
    OpVectorInsertDynamic = 78,
    OpVectorShuffle = 79,
    OpCompositeConstruct = 80,
    OpCompositeExtract = 81,
    OpCompositeInsert = 82,
    OpTranspose = 84,
    OpImageSampleImplicitLod = 87,
    OpImageSampleExplicitLod = 88,
    OpImageSampleDrefImplicitLod = 89,
    OpImageSampleDrefExplicitLod = 90,
    OpImageFetch = 95,
    OpImageGather = 96,
    OpImageDrefGather = 97,
    OpImage = 100,
    OpImageQuerySizeLod = 103,
    OpImageQuerySize = 104,
    OpConvertFToU = 109,
    OpConvertFToS = 110,
    OpConvertSToF = 111,
    OpConvertUToF = 112,
    OpUConvert = 113,
    OpSConvert = 114,
    OpFConvert = 115,
    OpConvertPtrToU = 117,
    OpConvertUToPtr = 120,
    OpBitcast = 124,
    OpSNegate = 126,
    OpFNegate = 127,
    OpIAdd = 128,
    OpFAdd = 129,
    OpISub = 130,
    OpFSub = 131,
    OpIMul = 132,
    OpFMul = 133,
    OpUDiv = 134,
    OpSDiv = 135,
    OpFDiv = 136,
    OpUMod = 137,
    OpSRem = 138,
    OpFMod = 141,
    OpVectorTimesScalar = 142,
    OpDot = 148,
    OpIAddCarry = 149,
    OpISubBorrow = 150,
    OpUMulExtended = 151,
    OpSMulExtended = 152,
    OpAny = 154,
    OpAll = 155,
    OpIsNan = 156,
    OpIsInf = 157,
    OpLogicalEqual = 164,
    OpLogicalNotEqual = 165,
    OpLogicalOr = 166,
    OpLogicalAnd = 167,
    OpLogicalNot = 168,
    OpSelect = 169,
    OpIEqual = 170,
    OpINotEqual = 171,
    OpUGreaterThan = 172,
    OpSGreaterThan = 173,
    OpUGreaterThanEqual = 174,
    OpSGreaterThanEqual = 175,
    OpULessThan = 176,
    OpSLessThan = 177,
    OpULessThanEqual = 178,
    OpSLessThanEqual = 179,
    OpFOrdEqual = 180,
    OpFUnordEqual = 181,
    OpFOrdNotEqual = 182,
    OpFUnordNotEqual = 183,
    OpFOrdLessThan = 184,
    OpFUnordLessThan = 185,
    OpFOrdGreaterThan = 186,
    OpFUnordGreaterThan = 187,
    OpFOrdLessThanEqual = 188,
    OpFUnordLessThanEqual = 189,
    OpFOrdGreaterThanEqual = 190,
    OpFUnordGreaterThanEqual = 191,
    OpShiftRightLogical = 194,
    OpShiftRightArithmetic = 195,
    OpShiftLeftLogical = 196,
    OpBitwiseOr = 197,
    OpBitwiseXor = 198,
    OpBitwiseAnd = 199,
    OpNot = 200,
    OpBitFieldInsert = 201,
    OpBitFieldSExtract = 202,
    OpBitFieldUExtract = 203,
    OpBitReverse = 204,
    OpBitCount = 205,
    OpDPdx = 207,
    OpDPdy = 208,
    OpFwidth = 209,
    OpControlBarrier = 224,
    OpMemoryBarrier = 225,
    OpAtomicLoad = 227,
    OpAtomicStore = 228,
    OpAtomicExchange = 229,
    OpAtomicCompareExchange = 230,
    OpAtomicIIncrement = 232,
    OpAtomicIDecrement = 233,
    OpAtomicIAdd = 234,
    OpAtomicISub = 235,
    OpAtomicSMin = 236,
    OpAtomicUMin = 237,
    OpAtomicSMax = 238,
    OpAtomicUMax = 239,
    OpAtomicAnd = 240,
    OpAtomicOr = 241,
    OpAtomicXor = 242,
    OpPhi = 245,
    OpLoopMerge = 246,
    OpSelectionMerge = 247,
    OpLabel = 248,
    OpBranch = 249,
    OpBranchConditional = 250,
    OpSwitch = 251,
    OpKill = 252,
    OpReturn = 253,
    OpUnreachable = 255,
    OpGroupNonUniformElect = 333,
    OpGroupNonUniformAll = 334,
    OpGroupNonUniformAny = 335,
    OpGroupNonUniformAllEqual = 336,
    OpGroupNonUniformBroadcast = 337,
    OpGroupNonUniformBroadcastFirst = 338,
    OpGroupNonUniformBallot = 339,
    OpGroupNonUniformInverseBallot = 340,
    OpGroupNonUniformBallotBitExtract = 341,
    OpGroupNonUniformBallotBitCount = 342,
    OpGroupNonUniformBallotFindLSB = 343,
    OpGroupNonUniformBallotFindMSB = 344,
    OpGroupNonUniformShuffle = 345,
    OpGroupNonUniformShuffleXor = 346,
    OpGroupNonUniformShuffleUp = 347,
    OpGroupNonUniformShuffleDown = 348,
    OpGroupNonUniformIAdd = 349,
    OpGroupNonUniformFAdd = 350,
    OpGroupNonUniformIMul = 351,
    OpGroupNonUniformFMul = 352,
    OpGroupNonUniformSMin = 353,
    OpGroupNonUniformUMin = 354,
    OpGroupNonUniformFMin = 355,
    OpGroupNonUniformSMax = 356,
    OpGroupNonUniformUMax = 357,
    OpGroupNonUniformFMax = 358,
    OpGroupNonUniformBitwiseAnd = 359,
    OpGroupNonUniformBitwiseOr = 360,
    OpGroupNonUniformBitwiseXor = 361,
};

// GLSL.std.450 extended instruction opcodes
enum class GlslStd450 : uint32_t {
    Round = 1,
    RoundEven = 2,
    Trunc = 3,
    FAbs = 4,
    SAbs = 5,
    FSign = 6,
    SSign = 7,
    Floor = 8,
    Ceil = 9,
    Fract = 10,
    Radians = 11,
    Degrees = 12,
    Sin = 13,
    Cos = 14,
    Tan = 15,
    Asin = 16,
    Acos = 17,
    Atan = 18,
    Atan2 = 25,
    Pow = 26,
    Exp = 27,
    Log = 28,
    Exp2 = 29,
    Log2 = 30,
    Sqrt = 31,
    InverseSqrt = 32,
    FMin = 37,
    UMin = 38,
    SMin = 39,
    FMax = 40,
    UMax = 41,
    SMax = 42,
    FClamp = 43,
    UClamp = 44,
    SClamp = 45,
    FMix = 46,
    Fma = 50,
    Frexp = 51,
    Ldexp = 53,
    PackSnorm4x8 = 54,
    PackUnorm4x8 = 55,
    PackSnorm2x16 = 56,
    PackUnorm2x16 = 57,
    PackHalf2x16 = 58,
    UnpackSnorm2x16 = 60,
    UnpackUnorm2x16 = 61,
    UnpackHalf2x16 = 62,
    UnpackSnorm4x8 = 63,
    UnpackUnorm4x8 = 64,
    FindILsb = 73,
    FindSMsb = 74,
    FindUMsb = 75,
    NMin = 79,
    NMax = 80,
    NClamp = 81,
};

// ImageOperands bitmask
namespace ImageOperandsMask {
    constexpr uint32_t None = 0x0;
    constexpr uint32_t Bias = 0x1;
    constexpr uint32_t Lod = 0x2;
    constexpr uint32_t Grad = 0x4;
    constexpr uint32_t ConstOffset = 0x8;
    constexpr uint32_t Offset = 0x10;
    constexpr uint32_t Sample = 0x40;
}

// MemoryAccess bitmask
namespace MemoryAccessMask {
    constexpr uint32_t None = 0x0;
    constexpr uint32_t Volatile = 0x1;
    constexpr uint32_t Aligned = 0x2;
}

class SpirvBuilder {
public:
    explicit SpirvBuilder(ExecutionModel model, bool enable_bda = true);

    uint32_t alloc_id() { return next_id_++; }
    uint32_t glsl_ext_id() const { return glsl_ext_id_; }
    ExecutionModel execution_model() const { return exec_model_; }

    // Capabilities & Extensions
    void add_capability(Capability cap);
    void add_extension(const std::string& ext);
    void add_execution_mode(ExecutionMode mode, const std::vector<uint32_t>& literals = {});

    // Debug names & decorations
    void set_name(uint32_t id, const std::string& name);
    void set_member_name(uint32_t struct_type_id, uint32_t member_idx, const std::string& name);
    void decorate(uint32_t target_id, Decoration dec, const std::vector<uint32_t>& literals = {});
    void member_decorate(uint32_t struct_type_id, uint32_t member_idx, Decoration dec, const std::vector<uint32_t>& literals = {});

    // Deduplicated Types
    uint32_t type_void();
    uint32_t type_bool();
    uint32_t type_int(uint32_t width, bool is_signed);
    uint32_t type_u8()  { return type_int(8, false); }
    uint32_t type_s8()  { return type_int(8, true); }
    uint32_t type_u16() { return type_int(16, false); }
    uint32_t type_s16() { return type_int(16, true); }
    uint32_t type_u32() { return type_int(32, false); }
    uint32_t type_s32() { return type_int(32, true); }
    uint32_t type_u64() { return type_int(64, false); }
    uint32_t type_s64() { return type_int(64, true); }
    uint32_t type_float(uint32_t width);
    uint32_t type_f16() { return type_float(16); }
    uint32_t type_f32() { return type_float(32); }
    uint32_t type_f64() { return type_float(64); }
    uint32_t type_vector(uint32_t elem_type_id, uint32_t count);
    uint32_t type_array(uint32_t elem_type_id, uint32_t length_const_id, uint32_t stride_bytes = 0);
    uint32_t type_runtime_array(uint32_t elem_type_id, uint32_t stride_bytes = 0);
    uint32_t type_struct(const std::vector<uint32_t>& member_types, const std::string& debug_name = "");
    uint32_t type_pointer(StorageClass sc, uint32_t pointee_type_id);
    uint32_t type_function(uint32_t return_type_id, const std::vector<uint32_t>& param_types = {});
    uint32_t type_image(uint32_t sampled_type_id, Dim dim, uint32_t depth, uint32_t arrayed, uint32_t ms, uint32_t sampled, uint32_t format = 0);
    uint32_t type_sampler();
    uint32_t type_sampled_image(uint32_t image_type_id);

    // Deduplicated Constants
    uint32_t const_bool(bool val);
    uint32_t const_u32(uint32_t val);
    uint32_t const_s32(int32_t val);
    uint32_t const_f32(float val);
    uint32_t const_f32_bits(uint32_t bits);
    uint32_t const_u64(uint64_t val);
    uint32_t const_u16(uint16_t val);
    uint32_t const_composite(uint32_t type_id, const std::vector<uint32_t>& constituents);
    uint32_t const_null(uint32_t type_id);

    // Global & Local Variables
    uint32_t add_global_var(uint32_t ptr_type_id, StorageClass sc, const std::string& name = "", bool add_to_interface = false);
    uint32_t add_local_var(uint32_t value_type_id, const std::string& name = "");

    // BuiltIn Input/Output variable helpers
    uint32_t get_or_create_builtin_input(BuiltIn builtin, uint32_t value_type_id, const std::string& name, bool flat_decor = false);
    uint32_t get_or_create_builtin_output(BuiltIn builtin, uint32_t value_type_id, const std::string& name);

    // Function & Block Emission
    void begin_main_function();
    void end_main_function();
    uint32_t emit_label(uint32_t label_id = 0);
    uint32_t current_label() const { return current_label_id_; }

    // Generic instruction emitter into current basic block
    void emit_inst(Op opcode, const std::vector<uint32_t>& words);
    uint32_t emit_typed_inst(Op opcode, uint32_t result_type_id, const std::vector<uint32_t>& operands);
    uint32_t emit_glsl_ext(GlslStd450 ext_op, uint32_t result_type_id, const std::vector<uint32_t>& operands);

    // High-level value helpers
    uint32_t emit_load(uint32_t result_type_id, uint32_t ptr_id, uint32_t align_bytes = 0);
    void emit_store(uint32_t ptr_id, uint32_t val_id, uint32_t align_bytes = 0);
    uint32_t emit_access_chain(uint32_t result_ptr_type_id, uint32_t base_ptr_id, const std::vector<uint32_t>& indices);
    uint32_t emit_bitcast(uint32_t target_type_id, uint32_t val_id);
    uint32_t emit_u_to_ptr(uint32_t ptr_type_id, uint32_t u64_addr_id);
    uint32_t emit_ptr_to_u(uint32_t ptr_id);
    uint32_t emit_select(uint32_t result_type_id, uint32_t cond_id, uint32_t true_id, uint32_t false_id);

    // Control flow helpers
    void emit_selection_merge(uint32_t merge_block_id, uint32_t selection_control = 0);
    void emit_loop_merge(uint32_t merge_block_id, uint32_t continue_block_id, uint32_t loop_control = 0);
    void emit_branch(uint32_t target_block_id);
    void emit_branch_conditional(uint32_t cond_bool_id, uint32_t true_block_id, uint32_t false_block_id);
    void emit_return();
    void emit_kill();

    // Finalize binary SPIR-V words
    std::vector<uint32_t> assemble() const;

private:
    ExecutionModel exec_model_;
    bool enable_bda_;
    uint32_t next_id_ = 1;
    uint32_t glsl_ext_id_ = 0;
    uint32_t main_func_id_ = 0;
    uint32_t entry_label_id_ = 0;
    uint32_t current_label_id_ = 0;
    bool in_entry_block_ = false;

    std::vector<Capability> capabilities_;
    std::vector<std::string> extensions_;
    std::vector<uint32_t> entry_interface_ids_;
    std::vector<uint32_t> execution_modes_words_;
    std::vector<uint32_t> debug_names_words_;
    std::vector<uint32_t> decorations_words_;
    std::vector<uint32_t> types_consts_globals_words_;
    std::vector<uint32_t> local_vars_words_;
    std::vector<uint32_t> body_words_;

    // Deduplication maps
    uint32_t void_type_id_ = 0;
    uint32_t bool_type_id_ = 0;
    uint32_t sampler_type_id_ = 0;
    std::map<std::pair<uint32_t, bool>, uint32_t> int_types_;
    std::map<uint32_t, uint32_t> float_types_;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> vector_types_;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> array_types_;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> runtime_array_types_;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> pointer_types_;
    std::map<std::vector<uint32_t>, uint32_t> func_types_;
    std::map<std::vector<uint32_t>, uint32_t> image_types_;
    std::map<uint32_t, uint32_t> sampled_image_types_;

    uint32_t const_true_id_ = 0;
    uint32_t const_false_id_ = 0;
    std::map<uint32_t, uint32_t> const_u32_map_;
    std::map<int32_t, uint32_t> const_s32_map_;
    std::map<uint32_t, uint32_t> const_f32_map_;
    std::map<uint64_t, uint32_t> const_u64_map_;
    std::map<uint16_t, uint32_t> const_u16_map_;
    std::map<std::vector<uint32_t>, uint32_t> const_composite_map_;
    std::map<uint32_t, uint32_t> const_null_map_;
    std::map<uint32_t, uint32_t> builtin_inputs_;
    std::map<uint32_t, uint32_t> builtin_outputs_;

    static void append_op(std::vector<uint32_t>& buf, Op opcode, const std::vector<uint32_t>& args);
    static std::vector<uint32_t> encode_string(const std::string& s);
};

} // namespace valhall::spirv
