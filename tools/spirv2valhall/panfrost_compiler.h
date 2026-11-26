#pragma once

#include "equiv_bundle.h"

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

enum class PanfrostMode {
    None,
    Vulkan,  // --panfrost: Vulkan compute SPIR-V -> OpenCL SPIR-V -> Panfrost -> Valhall ISA (-> SSA)
    OpenCL,  // --panfrost-cl: OpenCL SPIR-V -> Panfrost -> Valhall ISA (-> SSA)
};

struct PanfrostKernelBinary {
    std::string name;
    uint8_t arch = 10;
    uint32_t local_size[3] = {1, 1, 1};
    uint32_t work_reg_count = 32;
    std::vector<uint8_t> ebin_bytes;
    std::string header_text;
};

struct PanfrostCompileResult {
    bool ok = false;
    std::string error;
    std::vector<PanfrostKernelBinary> kernels;
};

struct PanfrostRunOptions {
    PanfrostMode mode = PanfrostMode::None;
    uint8_t arch = 10;
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

/// Locate the Panfrost `panfrost_compile` compiler executable.
/// Search order: explicit_path, PANFROST_COMPILE_BIN env var,
/// third_party/prebuilts/panfrost/panfrost_compile (CWD or parent dirs), then PATH.
std::string find_panfrost_compiler(const std::string& explicit_path = "");

/// Translate a Vulkan GLCompute SPIR-V module into an OpenCL 2.2 Kernel SPIR-V module
/// accepted by Mesa's `panfrost_compile` (converting StorageBuffer/Uniform SSBOs to
/// CrossWorkgroup pointer kernel parameters and GLSL.std.450 ext-insts to OpenCL.std).
bool translate_vulkan_spv_to_opencl_spv(
    const std::vector<uint32_t>& vk_words,
    std::vector<uint32_t>* cl_words,
    std::string* err = nullptr);

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
    bool verbose = false);

/// End-to-end CLI driver for `--panfrost` and `--panfrost-cl` modes in `spirv2valhall`.
/// Supports Valhall ISA disassembly, `--out-bin`, `--dump-ssa`, `--dump-ssa-json`,
/// `--opt-ssa`, `--dump-spirv`, `--out-spirv`, and `--dump-equiv-json`.
int run_panfrost_pipeline(const std::string& spv_path, const PanfrostRunOptions& opts);

}  // namespace valhall::spirv2valhall
