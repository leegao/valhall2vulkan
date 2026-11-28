#pragma once
#include "valhall_isa_generated.h"
#include "valhall_csf_descriptors_generated.h"
#include "valhall_spirv_builder.h"
#include "valhall_cfg.h"
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace valhall {

enum class ShaderStage : uint8_t {
    Compute = 0,
    Pilot = 1,
    Vertex = 2,
    Fragment = 3,
};

struct VertexAttributeBindingSpec {
    uint32_t location = 0;
    uint32_t components = 4; // 1..4
    bool is_integer = false;
    bool is_signed = false;
};

struct VaryingSlotSpec {
    uint32_t location = 0;
    uint32_t components = 4; // 1..4
    bool flat = false;
    bool is_integer = false;
};

struct CombinedSamplerBindingSpec {
    uint32_t descriptor_set = 0;
    uint32_t binding = 1;
    uint32_t tex_index = 0;
    uint32_t sampler_index = 0;
    spirv::Dim dim = spirv::Dim::Dim2D;
    bool arrayed = false;
    bool shadow = false;
};

// Sideband Uniform Block (bound at DescriptorSet = 0, Binding = 0 in every translated shader)
// Matches the GPU memory layout of FAU + CSF/SPD sideband parameters.
struct alignas(16) ValhallSidebandData {
    uint32_t fau_words[64];         // Offset   0 (256B): u0.w0 .. u31.w1 (64 x 32-bit FAU words)
    uint64_t shader_pc_base;        // Offset 256 (  8B): 64-bit GPU VA of SPD.Shader Program (for ADR / pc.w0 / pc.w1)
    uint64_t srt_base_va;           // Offset 264 (  8B): 64-bit GPU VA of SRT (or FAU dest for Pilot)
    uint64_t pilot_fau_dst_va;      // Offset 272 (  8B): 64-bit GPU VA of FAU output buffer for Pilot shader [r0:r1]
    uint64_t blend_descs[4];        // Offset 280 ( 32B): blend_descriptor_0 .. 3
    uint32_t local_size_xyz[4];     // Offset 312 ( 16B): local_size_x, y, z, lane_width
    uint32_t fb_info[4];            // Offset 328 ( 16B): width, height, sample_cnt, rt_count
};

struct ValhallSymbolBindings {
    std::string entry_point_name;                      // Optional OpEntryPoint / OpName override (default: "main")
    std::string sideband_var_name;                     // Optional sideband UBO/SSBO variable name (default: "u_sideband")
    std::map<uint32_t, std::string> vertex_input_names;   // location -> symbol name (e.g. 0 -> "in_pos")
    std::map<uint32_t, std::string> varying_input_names;  // location -> symbol name (e.g. 0 -> "v_col")
    std::map<uint32_t, std::string> color_output_names;   // location -> symbol name (e.g. 0 -> "out_col")
    std::map<uint32_t, std::string> fau_word_names;       // FAU byte offset (0, 4, 8, ...) -> symbol name
};

struct ValhallTranslateOptions {
    ShaderStage stage = ShaderStage::Compute;
    uint8_t arch = 11; // Valhall v9..v14 (default v11 / Mali-G615)
    bool fau32 = false;
    uint32_t local_size_x = 1;
    uint32_t local_size_y = 1;
    uint32_t local_size_z = 1;

    // Optional second EBIN for Vertex shaders (Varying variant 0x81 fused after Position variant 0x180)
    const uint8_t* secondary_vertex_code = nullptr;
    size_t secondary_vertex_code_size = 0;

    // Vertex / Fragment interface metadata
    std::vector<VertexAttributeBindingSpec> vertex_attributes;
    std::vector<VaryingSlotSpec> varyings;
    std::vector<CombinedSamplerBindingSpec> samplers;
    std::map<uint32_t, uint32_t> fau_tex_handle_indices;
    std::map<uint32_t, std::string> fau_pka_buf_names;
    std::map<uint32_t, std::string> srt_pka_buf_names;
    std::map<uint32_t, uint32_t> fau_fcst_words;
    std::map<uint32_t, uint32_t> frag_varying_addr_to_loc;
    std::map<uint32_t, uint32_t> vert_attr_idx_to_loc;
    std::map<uint32_t, std::string> vert_sout_attr_names;
    struct IdvsVaryingOffsetSpec {
        int32_t user_location = -1;
        bool is_point_size = false;
        bool is_position = false;
        uint32_t ncomps = 4;
        std::string name;
    };
    std::map<int64_t, IdvsVaryingOffsetSpec> idvs_varying_offsets;
    uint32_t num_color_attachments = 1;
    bool emits_depth = false;

    // Sideband uniform binding location
    uint32_t sideband_set = 0;
    uint32_t sideband_binding = 0;

    // Optional debug/reflection symbol names for OpName emission (empty = default generated names)
    ValhallSymbolBindings symbol_bindings;
    bool tolerant_ssa = false;

    // Pass-2 tiler & optional SSA optimization pipeline controls (strict by default)
    bool enable_fast_math_tiles = false;
    bool run_ssa_opt = false;
    bool ssa_opt_fp_fast_math = false;
    bool ssa_opt_fp_tile_rewrites = true;
    bool ssa_opt_fp_cpp_rewrites = true;
};

struct ValhallTranslationResult {
    bool ok = false;
    std::string error_message;
    std::vector<uint32_t> spirv_words;
    cfg::ControlFlowGraph primary_cfg;
    cfg::ControlFlowGraph secondary_cfg;
    std::string disassembly_listing;
};

// Translates raw Valhall binary instruction stream(s) into a binary SPIR-V 1.3 module.
ValhallTranslationResult translate_valhall_to_spirv(
    const uint8_t* code_bytes,
    size_t code_size_bytes,
    const ValhallTranslateOptions& opts);

} // namespace valhall
