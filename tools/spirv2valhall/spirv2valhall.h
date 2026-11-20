#pragma once

#include "valhall_isa_generated.h"

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

enum class ShaderStage {
    Unknown = 0,
    Vertex,
    TessControl,
    TessEval,
    Geometry,
    Fragment,
    Compute,
    RayGen,
    AnyHit,
    ClosestHit,
    Miss,
    Intersection,
    Callable,
};

struct SpirvStageInfo {
    ShaderStage stage = ShaderStage::Unknown;
    uint32_t execution_model = 0xFFFFFFFFu;
    std::string entry_point;
    std::string malioc_flag;  // e.g. "--vertex", "--fragment", "--compute"
};

struct ExtractedEbin {
    uint32_t index = 0;
    uint32_t variant_flags = 0;
    uint32_t code_size = 0;
    std::vector<uint8_t> bytecode;
    std::vector<valhall::DecodedValhallInst> instructions;
    size_t invalid_count = 0;
};

struct ExtractOptions {
    std::string core = "Mali-G710";
    uint8_t arch = 0;  // 0 = infer from core name
    std::string malioc_path = "mali_offline_compiler/malioc";
    std::string moc_hook_path;  // empty = default build/libmoc_hook.so or $SPIRV2VALHALL_MOC_HOOK
    std::string dump_header_path;  // optional path to persist generated MBS2 C header
    bool strip_unused_caps = true;
    bool disassemble = true;
    bool verbose = false;
};

struct ExtractResult {
    SpirvStageInfo stage_info;
    uint8_t arch = 10;
    bool stripped_caps = false;
    std::string dump_header_text;
    std::vector<ExtractedEbin> ebins;
    std::string malioc_output;
};

/// Infer Valhall architecture version (9..14) from a Mali GPU core name (e.g. "Mali-G710" -> 10).
uint8_t infer_arch_from_core(const std::string& core);

/// Parse a SPIR-V word stream and infer the shader stage & malioc CLI flag from OpEntryPoint.
bool infer_spirv_stage(const std::vector<uint32_t>& words, SpirvStageInfo* out, std::string* err = nullptr);

/// Read a .spv file from disk and infer its shader stage & malioc CLI flag.
bool infer_spirv_stage_from_file(const std::string& spv_path, SpirvStageInfo* out, std::string* err = nullptr);

/// Strip unused OpCapability Float64 (10) (when OpTypeFloat 64 is absent) and unused
/// FragmentShaderPixelInterlockEXT (5378) / SPV_EXT_fragment_shader_interlock so malioc accepts the module.
std::vector<uint32_t> strip_unused_malioc_capabilities(const std::vector<uint32_t>& words, bool* stripped_any = nullptr);

/// Parse raw EBIN bytecode blobs (and optionally disassemble) from a moc_hook-dumped MBS2 C header.
bool parse_mbs2_header_ebins(
    const std::string& header_text,
    uint8_t arch,
    bool disassemble,
    std::vector<ExtractedEbin>* out,
    std::string* err = nullptr);

/// Run malioc + libmoc_hook.so on a SPIR-V binary, auto-inferring shader stage, and extract Valhall ISA.
bool extract_valhall_from_spirv(
    const std::string& spv_path,
    const ExtractOptions& opts,
    ExtractResult* out,
    std::string* err = nullptr);

/// Format decoded EBINs as human-readable Valhall disassembly.
std::string format_disassembly(const ExtractResult& res, bool include_modifiers = false);

}  // namespace valhall::spirv2valhall
