#pragma once

#include "valhall_to_spirv.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace valhall::spirv2ssa {

/// Structured representation of a single raw SPIR-V instruction.
struct SpirvInst {
    uint16_t opcode = 0;
    std::vector<uint32_t> words; // Full instruction words including header word[0]
};

enum class SpirvTypeKind : uint8_t {
    Unknown = 0,
    Void,
    Bool,
    Int,
    Float,
    Vector,
    Matrix,
    Image,
    Sampler,
    SampledImage,
    Array,
    RuntimeArray,
    Struct,
    Pointer,
    Function,
};

struct SpirvType {
    uint32_t id = 0;
    SpirvTypeKind kind = SpirvTypeKind::Unknown;

    // Int / Float
    uint32_t width = 0;
    bool is_signed = false;

    // Vector / Matrix / Array / RuntimeArray
    uint32_t elem_type = 0;
    uint32_t count = 0;        // Vector/Matrix count, or resolved Array length
    uint32_t length_id = 0;    // OpTypeArray length constant ID
    uint32_t array_stride = 0; // From OpDecorate ArrayStride

    // Pointer
    uint32_t storage_class = 0;
    uint32_t pointee_type = 0;

    // Struct
    std::vector<uint32_t> member_types;
    std::vector<uint32_t> member_offsets;      // UINT32_MAX if undecorated
    std::vector<int32_t> member_builtins;      // -1 if not a BuiltIn
    std::vector<uint32_t> member_matrix_stride;
    std::vector<bool> member_row_major;
    bool is_block = false;
    bool is_buffer_block = false;

    // Image
    uint32_t image_sampled_type = 0;
    uint32_t image_dim = 0;
    uint32_t image_depth = 0;
    uint32_t image_arrayed = 0;
    uint32_t image_ms = 0;
};

enum class SpirvConstKind : uint8_t {
    Unknown = 0,
    Bool,
    Scalar32,
    Scalar64,
    Composite,
    Null,
    Undef,
};

struct SpirvConstant {
    uint32_t id = 0;
    uint32_t type_id = 0;
    SpirvConstKind kind = SpirvConstKind::Unknown;
    uint64_t scalar_bits = 0;
    std::vector<uint32_t> constituents;
    bool is_spec_const = false;
    uint32_t spec_id = 0;
};

struct SpirvDecorationInfo {
    int32_t builtin = -1;
    int32_t location = -1;
    uint32_t component = 0;
    int32_t descriptor_set = -1;
    int32_t binding = -1;
    uint32_t array_stride = 0;
    uint32_t offset = 0xFFFFFFFFu;
    int32_t spec_id = -1;
    bool flat = false;
    bool block = false;
    bool buffer_block = false;
    bool non_writable = false;
};

struct SpirvVariable {
    uint32_t id = 0;
    uint32_t ptr_type_id = 0;
    uint32_t pointee_type_id = 0;
    uint32_t storage_class = 0;
    uint32_t initializer_id = 0;
    SpirvDecorationInfo dec;
    std::string name;
};

struct SpirvBasicBlock {
    uint32_t label_id = 0;
    std::vector<SpirvInst> insts;
};

struct SpirvFunction {
    uint32_t id = 0;
    uint32_t return_type_id = 0;
    uint32_t func_type_id = 0;
    std::vector<std::pair<uint32_t, uint32_t>> params; // (param_id, type_id)
    std::vector<SpirvBasicBlock> blocks;
};

struct SpirvEntryPoint {
    uint32_t execution_model = 0xFFFFFFFFu;
    uint32_t func_id = 0;
    std::string name;
    std::vector<uint32_t> interface_ids;
    uint32_t local_size_x = 1;
    uint32_t local_size_y = 1;
    uint32_t local_size_z = 1;
};

/// Parsed SPIR-V binary module ready for direct ValhallSSA lowering.
struct SpirvModule {
    uint32_t version = 0;
    uint32_t bound = 0;

    SpirvEntryPoint entry_point;
    std::unordered_map<uint32_t, std::string> ext_inst_imports;
    std::unordered_map<uint32_t, std::string> names;
    std::unordered_map<uint32_t, SpirvDecorationInfo> decorations;
    std::map<std::pair<uint32_t, uint32_t>, SpirvDecorationInfo> member_decorations;

    std::unordered_map<uint32_t, SpirvType> types;
    std::unordered_map<uint32_t, SpirvConstant> constants;
    std::unordered_map<uint32_t, SpirvVariable> global_vars;
    std::unordered_map<uint32_t, SpirvFunction> functions;

    const SpirvType* get_type(uint32_t id) const;
    const SpirvConstant* get_const(uint32_t id) const;
    const SpirvVariable* get_global_var(uint32_t id) const;
    const SpirvFunction* get_function(uint32_t id) const;

    /// Compute the byte size of a scalar, vector, matrix, array, or struct type.
    uint32_t type_byte_size(uint32_t type_id) const;

    /// Compute the byte offset of struct member `member_idx` (using Offset decoration or natural packing).
    uint32_t struct_member_offset(const SpirvType& st, uint32_t member_idx) const;
};

/// Parse raw SPIR-V binary words into a structured SpirvModule.
bool parse_spirv_module(
    const std::vector<uint32_t>& words,
    SpirvModule* out,
    std::string* err = nullptr);

} // namespace valhall::spirv2ssa
