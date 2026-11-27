#pragma once

#include "valhall_to_spirv.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

/// Backend selection for the standalone `panfrost_vk_compile` driver.
enum class PanfrostVkBackend {
    Auto,     // Use panfrost_vk_compile's default backend (Rust Kraid on v11)
    Kraid,    // Force Rust Kraid backend (--kraid)
    Bifrost,  // Force C Bifrost/Valhall backend (--no-kraid)
};

/// Invocation options for `panfrost_vk_compile`.
struct PanfrostVkCompileOptions {
    uint8_t arch = 11;
    PanfrostVkBackend backend = PanfrostVkBackend::Auto;
    bool robust = false;
    bool no_idvs = false;
    bool no_promote_imm = false;
    bool no_clamp_version = false;
    uint32_t core_max_id = 0;
    std::string panfrost_vk_bin;
    bool verbose = false;
};

/// Descriptor binding slot emitted by `panfrost_vk_compile --info-json`.
struct PanfrostVkDescriptorInfo {
    uint32_t set = 0;
    uint32_t table = 1;
    uint32_t binding = 0;
    std::string kind;
    uint32_t desc_idx = 0;
    uint32_t desc_count = 1;
};

/// Compiled output from Mesa's standalone `panfrost_vk_compile` Vulkan SPIR-V compiler.
struct PanfrostVkCompileOutput {
    bool ok = false;
    std::string error;
    std::string entry_point = "main";
    std::string backend_used;  // "kraid" or "bifrost"
    valhall::ShaderStage stage = valhall::ShaderStage::Compute;
    uint8_t arch = 11;
    bool idvs = true;
    bool robust = false;
    bool merge_workgroups = false;
    uint32_t local_size[3] = {1, 1, 1};
    uint32_t instrs = 0;
    uint32_t work_reg_count = 32;
    uint32_t tls_size = 0;
    uint32_t wls_size = 0;
    uint32_t fau_sysval_slots = 0;
    uint32_t fau_driver_slots = 0;
    uint32_t fau_reserved_words = 0;
    uint32_t fau_total_words = 0;
    std::vector<uint32_t> used_sysval_words;
    std::vector<uint32_t> used_push_const_words;
    std::vector<uint8_t> ebin_bytes;
    /// Promoted FAU constant words emitted by Panfrost (FAU 32-bit word index -> 32-bit value).
    std::map<uint32_t, uint32_t> fau_const_words;
    /// Packed Valhall resource handle ((table << 24) | desc_idx) -> canonical buffer key ("ssbo:S:B" / "ubo:S:B").
    std::map<uint32_t, std::string> srt_pka_buf_names;
    /// Full descriptor table entries reported by `--info-json`.
    std::vector<PanfrostVkDescriptorInfo> descriptors;
};

/// Locate the standalone `panfrost_vk_compile` executable.
/// Search order: explicit_path, PANFROST_VK_COMPILE_BIN env var,
/// third_party/prebuilts/panfrost/panfrost_vk_compile (CWD or parent dirs), then PATH.
std::string find_panfrost_vk_compiler(const std::string& explicit_path = "");

/// Compile a Vulkan SPIR-V shader (Vertex, Fragment, or Compute) directly to Valhall ISA
/// by invoking Mesa's standalone `panfrost_vk_compile` binary with `--info-json`.
PanfrostVkCompileOutput compile_vulkan_spv_via_panfrost(
    const std::string& spv_path,
    const PanfrostVkCompileOptions& opts = {});

}  // namespace valhall::spirv2valhall
