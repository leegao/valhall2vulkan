#pragma once

#include "valhall_to_spirv.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

/// Compiled output from Mesa Panfrost's native Vulkan SPIR-V -> NIR -> Valhall ISA backend.
struct PanfrostVkCompileOutput {
    bool ok = false;
    std::string error;
    std::string entry_point = "main";
    valhall::ShaderStage stage = valhall::ShaderStage::Compute;
    uint8_t arch = 10;
    uint32_t local_size[3] = {1, 1, 1};
    uint32_t work_reg_count = 32;
    uint32_t tls_size = 0;
    uint32_t wls_size = 0;
    std::vector<uint8_t> ebin_bytes;
    /// Promoted FAU constant words emitted by Panfrost (FAU word index -> 32-bit value).
    std::map<uint32_t, uint32_t> fau_const_words;
    /// Packed Valhall descriptor ID (((4 + set) << 24) | binding) -> canonical buffer key.
    std::map<uint32_t, std::string> srt_pka_buf_names;
};

/// Compile a Vulkan SPIR-V shader (Vertex, Fragment, or Compute) directly to Valhall ISA
/// using Mesa Panfrost's native Vulkan NIR pipeline (`spirv_to_nir(NIR_SPIRV_VULKAN)` ->
/// `pan_preprocess_nir` -> Vulkan descriptor/IO lowering -> `pan_postprocess_nir` ->
/// `pan_shader_compile`), isolated in a forked child process so Mesa internal assertions
/// are reported cleanly as compilation errors without terminating the caller.
PanfrostVkCompileOutput compile_vulkan_spv_via_panfrost(
    const std::string& spv_path,
    uint8_t arch,
    const std::string& panfrost_bin,
    bool verbose = false);

}  // namespace valhall::spirv2valhall
