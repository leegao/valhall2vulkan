#pragma once

#include "panfrost_compiler.h"
#include "valhall_isa_generated.h"
#include "valhall_ssa.h"
#include "valhall_to_spirv.h"

#include <cstdint>
#include <vector>

namespace valhall::spirv2valhall {

/// Result of splitting a concatenated Panfrost IDVS vertex binary into its
/// Position shader (primary) and Varying shader (secondary) instruction streams.
struct PanfrostIdvsSplit {
    bool is_idvs_split = false;
    uint32_t split_pc = 0;
    std::vector<valhall::DecodedValhallInst> primary_insts;
    std::vector<valhall::DecodedValhallInst> secondary_insts;
};

/// Summary of explicit Panfrost <-> canonical DDK/SPIR-V ABI translations applied
/// to a lowered ValhallSSA module.
struct PanfrostAbiBridgeReport {
    bool idvs_split = false;
    uint32_t primary_inst_count = 0;
    uint32_t secondary_inst_count = 0;
    uint32_t viewport_fau_words_devirt = 0;
    uint32_t compute_sysval_words_devirt = 0;
    uint32_t push_const_words_devirt = 0;
    uint32_t immediate_words_devirt = 0;
    uint32_t descriptor_samplers_mapped = 0;
};

/// Decode a Panfrost EBIN byte buffer and, for IDVS Vertex shaders, explicitly split
/// the concatenated Position shader and 128-byte-aligned Varying shader at the `.end`
/// + zero-padding boundary so Pass 1 lowers them as primary + secondary CFGs matching
/// Mali DDK's dual-EBIN Vertex layout.
PanfrostIdvsSplit decode_and_split_panfrost_binary(
    const std::vector<uint8_t>& ebin_bytes,
    uint8_t arch,
    valhall::ShaderStage stage,
    bool idvs_enabled);

/// Populate `ValhallTranslateOptions` with Panfrost's descriptor-handle (`srt_pka_buf_names`,
/// `fau_pka_buf_names`), promoted FAU constant (`fau_fcst_words`), and texture/sampler
/// binding tables so Pass 1 lowering resolves `LEA_PKA`, `LD_PKA`, and `TEX_SINGLE` into
/// canonical resource keys (`ssbo:S:B`, `ubo:S:B`, `tex:S:B`).
void populate_panfrost_translate_options(
    const PanfrostCompileResult& cres,
    const PanfrostKernelBinary& kb,
    valhall::ValhallTranslateOptions* vopts,
    PanfrostAbiBridgeReport* report = nullptr);

/// Apply the explicit Panfrost Vulkan (`panvk`) -> canonical DDK/SPIR-V SSA ABI bridge:
/// - Devirtualizes compacted `panvk_compute_sysvals` (`base_workgroup_id` -> 0,
///   `num_work_groups` -> `gl_NumWorkGroups`, `local_group_size` -> `LocalSize`).
/// - Devirtualizes compacted `panvk_graphics_sysvals` (`viewport.scale` -> (1,1,1),
///   `viewport.offset` -> (0,0,0), `vs.first_vertex`/`base_instance` -> 0) so IDVS
///   Position shaders yield clip-space `gl_Position` directly comparable to Mali DDK.
/// - Devirtualizes compacted user `push_const` FAU slots and promoted FAU immediates.
void apply_panfrost_vk_abi_bridge(
    valhall::compiler::ValhallSsaModule& ssa,
    const PanfrostCompileResult& cres,
    const PanfrostKernelBinary& kb,
    PanfrostAbiBridgeReport* report = nullptr);

/// Apply the legacy OpenCL (`--panfrost-cl`) FAU -> canonical SSA ABI bridge
/// (64-bit buffer pointer arguments at FAU byte offset 24 + 8*i and `gl_NumWorkGroups`
/// at FAU words 0..2).
void apply_panfrost_cl_abi_bridge(
    valhall::compiler::ValhallSsaModule& ssa,
    const std::vector<PanfrostArgBinding>& bindings);

}  // namespace valhall::spirv2valhall
