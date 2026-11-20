#pragma once

#include "isa_enricher.h"
#include "mbs2_ast_parser.h"
#include "mbs2_types.h"
#include "spirv_inspect.h"

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

struct LiftedSpirvShader {
    std::string label;        // e.g. "MAIN COMPUTE SHADER", "PILOT SHADER", "VERTEX SHADER", "FRAGMENT SHADER"
    EbinRole role = EbinRole::Unknown;
    bool ok = false;
    std::string error_message;
    std::vector<uint32_t> spirv_words;
    bool reflect_valid = false;
    std::string reflect_entry_point;
    uint32_t reflect_shader_stage = 0;
    std::string disassembly;  // Human-readable SPIR-V assembly text
};

/// Lift extracted Valhall EBIN(s) back into SPIR-V via valhall2spirv (`translate_valhall_to_spirv`),
/// validate each lifted module with SPIRV-Reflect, and disassemble to readable SPIR-V text.
bool lift_ebins_to_spirv(
    const SpirvStageInfo& stage_info,
    uint8_t arch,
    const Mbs2Module& mod,
    const std::vector<ExtractedEbin>& ebins,
    std::vector<LiftedSpirvShader>* out_shaders,
    const PilotAnalysisResult* pilot_info = nullptr,
    std::string* err = nullptr);

/// Disassemble a binary SPIR-V word stream into readable SPIR-V assembly text.
std::string format_spirv_disassembly(const std::vector<uint32_t>& words);

/// Format all lifted SPIR-V modules (header + SPIRV-Reflect validation summary + SPIR-V assembly).
std::string format_lifted_spirv_dump(const std::vector<LiftedSpirvShader>& shaders);

}  // namespace valhall::spirv2valhall
