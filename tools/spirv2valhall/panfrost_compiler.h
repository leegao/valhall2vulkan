#pragma once

#include "equiv_bundle.h"
#include "panfrost_vk_compiler.h"
#include "valhall_to_spirv.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

enum class PanfrostMode {
    None,
    Vulkan,  // --panfrost: Native Mesa Panfrost Vulkan SPIR-V -> NIR -> Valhall ISA (VS/FS/CS) via panfrost_vk_compile
    OpenCL,  // --panfrost-cl: OpenCL (or translated Vulkan compute) SPIR-V -> panfrost_compile
};

struct PanfrostKernelBinary {
    std::string name;
    valhall::ShaderStage stage = valhall::ShaderStage::Compute;
    uint8_t arch = 11;
    uint32_t local_size[3] = {1, 1, 1};
    uint32_t work_reg_count = 32;
    uint32_t tls_size = 0;
    uint32_t wls_size = 0;
    uint32_t constant_data_size = 0;
    std::vector<uint8_t> ebin_bytes;
    std::string header_text;
};

struct PanfrostArgBinding {
    uint32_t arg_idx = 0;
    uint32_t fau_offset = 24;
    uint32_t descriptor_set = 0;
    uint32_t binding = 0;
    bool is_ssbo = true;
    std::string key;
};

struct PanfrostCompileResult {
    bool ok = false;
    std::string error;
    std::string backend_used;  // "kraid", "bifrost", or "opencl"
    bool idvs = true;
    bool robust = false;
    bool merge_workgroups = false;
    uint32_t fau_sysval_slots = 0;
    uint32_t fau_driver_slots = 0;
    uint32_t fau_reserved_words = 0;
    uint32_t fau_total_words = 0;
    std::vector<uint32_t> used_sysval_words;
    std::vector<uint32_t> used_push_const_words;
    std::vector<PanfrostKernelBinary> kernels;
    std::vector<PanfrostArgBinding> arg_bindings;
    std::map<uint32_t, uint32_t> fau_const_words;
    std::map<uint32_t, std::string> srt_pka_buf_names;
    std::vector<PanfrostVkDescriptorInfo> descriptors;
    uint32_t constant_data_size = 0;
    bool secondary_enable = false;
    uint32_t secondary_offset = 0;
    std::vector<PanfrostVkVaryingInfo> varyings;
    std::vector<uint32_t> vert_input_locations;
    bool has_frag_noperspective_varyings = false;
};

struct PanfrostRunOptions {
    PanfrostMode mode = PanfrostMode::None;
    uint8_t arch = 11;
    PanfrostVkBackend vk_backend = PanfrostVkBackend::Auto;
    bool vk_robust = false;
    bool vk_no_idvs = false;
    std::string panfrost_bin;
    std::string out_bin_prefix;
    std::string out_spirv_prefix;
    std::string dump_ssa_json_path;
    std::string dump_equiv_json_path;
    EquivBundleOptions equiv_opts;
    bool dump_ssa = false;
    bool dump_spirv = false;
    bool include_mods = false;
    bool quiet = false;
    bool verbose = false;
};

/// Locate the Panfrost OpenCL `panfrost_compile` compiler executable.
/// Search order: explicit_path, PANFROST_COMPILE_BIN env var,
/// third_party/prebuilts/panfrost/panfrost_compile (CWD or parent dirs), then PATH.
std::string find_panfrost_compiler(const std::string& explicit_path = "");

/// Translate a Vulkan GLCompute SPIR-V module into an OpenCL 2.2 Kernel SPIR-V module
/// accepted by Mesa's `panfrost_compile` (converting StorageBuffer/Uniform SSBOs to
/// CrossWorkgroup pointer kernel parameters and GLSL.std.450 ext-insts to OpenCL.std).
bool translate_vulkan_spv_to_opencl_spv(
    const std::vector<uint32_t>& vk_words,
    std::vector<uint32_t>* cl_words,
    std::string* err = nullptr,
    std::vector<PanfrostArgBinding>* out_bindings = nullptr);

/// Parse the generated C source emitted by `panfrost_compile` and extract each kernel's
/// `bifrost_precompiled_kernel_info` metadata and raw Valhall EBIN bytecode.
bool parse_panfrost_c_output(
    const std::string& c_source,
    const std::string& h_source,
    uint8_t arch,
    std::vector<PanfrostKernelBinary>* out_kernels,
    std::string* err = nullptr);

/// Compile an input SPIR-V file via Panfrost (`--panfrost` or `--panfrost-cl`).
PanfrostCompileResult compile_with_panfrost(
    const std::string& spv_path,
    PanfrostMode mode,
    uint8_t arch,
    const std::string& panfrost_bin = "",
    bool verbose = false,
    PanfrostVkBackend vk_backend = PanfrostVkBackend::Auto,
    bool vk_robust = false,
    bool vk_no_idvs = false);

/// End-to-end CLI driver for `--panfrost` and `--panfrost-cl` modes in `spirv2valhall`.
/// Supports Valhall ISA disassembly, `--out-bin`, `--dump-ssa`, `--dump-ssa-json`,
/// `--opt-ssa`, `--dump-spirv`, `--out-spirv`, and `--dump-equiv-json`.
int run_panfrost_pipeline(const std::string& spv_path, const PanfrostRunOptions& opts);

}  // namespace valhall::spirv2valhall
