#pragma once

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

/// Human-readable stage name (e.g. "Compute", "Fragment", "Vertex").
const char* shader_stage_name(ShaderStage stage);

/// Read a binary .spv file from disk and validate its SPIR-V header magic.
bool read_spirv_words(const std::string& path, std::vector<uint32_t>* out, std::string* err = nullptr);

/// Infer Valhall architecture version (9..14) from a Mali GPU core name (e.g. "Mali-G710" -> 10).
uint8_t infer_arch_from_core(const std::string& core);

/// Parse a SPIR-V word stream and infer the shader stage & malioc CLI flag from OpEntryPoint.
bool infer_spirv_stage(const std::vector<uint32_t>& words, SpirvStageInfo* out, std::string* err = nullptr);

/// Read a .spv file from disk and infer its shader stage & malioc CLI flag.
bool infer_spirv_stage_from_file(const std::string& spv_path, SpirvStageInfo* out, std::string* err = nullptr);

/// Strip unused OpCapability Float64 (10) (when OpTypeFloat 64 is absent) and unused
/// FragmentShaderPixelInterlockEXT (5378) / SPV_EXT_fragment_shader_interlock so malioc accepts the module.
std::vector<uint32_t> strip_unused_malioc_capabilities(const std::vector<uint32_t>& words, bool* stripped_any = nullptr);

}  // namespace valhall::spirv2valhall
