#pragma once

#include "mbs2_types.h"
#include "spirv_inspect.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

/// Semantic category of a value written into FAU_0 by a Pilot Shader.
enum class PilotValueKind : uint8_t {
    Unknown = 0,
    BufferAddr,     // 64-bit base pointer (plus optional static pointer_offset) into an SSBO/UBO
    BufferSize,     // 32-bit buffer size word from descriptor
    DescriptorAddr, // Address of a resource descriptor entry
    SymbolValue,    // Preloaded UBO/SSBO data word
    ConstInt,       // Compile-time integer constant folded by the pilot shader
    AbiPreload,     // Hardware ABI preload register (e.g. workgroup/dispatch metadata)
};

/// A single FAU slot (byte offset -> semantic description) inferred by chasing the Pilot Shader.
struct PilotFauEntry {
    uint32_t fau_offset = 0;      // Byte offset inside FAU_0 (0x00, 0x08, 0x10, 0x18, ...)
    uint32_t byte_size = 4;       // 4 or 8 bytes
    PilotValueKind value_kind = PilotValueKind::Unknown;
    std::string symbol_name;      // e.g. "_83_input_0" or "_43_pushConstants_0.u_TextureDimensions_0.x"
    std::string expression;       // e.g. "&_83_input_0 (SRT[4]->Buf[0].addr)" or "derived(_43_pushConstants_0.u_TextureDimensions_0.x)"
    int32_t srt_rack = -1;        // SRT rack index if sourced from SRT
    int32_t desc_index = -1;      // Descriptor index within rack
    bool is_loaded_value = false; // True for preloaded UBO/SSBO data words (vs. 64-bit buffer base ptr)
    bool has_byte_offset = false; // True when loaded word byte offset inside buffer is statically known
    int64_t byte_offset = 0;      // Byte offset inside UBO/SSBO for preloaded data word
    bool is_derived = false;      // True if arithmetic/transform was applied after loading
    bool has_pointer_offset = false; // True when a BufferAddr carries a statically known constant byte offset
    int64_t pointer_offset = 0;      // Constant byte offset added to BufferAddr by the pilot shader
    bool has_const_value = false;    // True when value_kind == ConstInt
    int64_t const_value = 0;         // Constant integer value written by the pilot shader
};

struct PilotAnalysisResult {
    bool found_pilot = false;
    uint32_t pilot_ebin_index = 0;
    std::map<uint32_t, PilotFauEntry> fau_map;  // keyed by FAU byte offset
};

/// Analyze any Pilot EBIN present in `ebins` by chasing SRT -> Resource -> Buffer -> FAU stores.
PilotAnalysisResult analyze_pilot_shaders(
    const Mbs2Module& mod,
    const std::vector<ExtractedEbin>& ebins);

/// Build per-instruction inline annotations for a single EBIN using MBS2 symbols,
/// live ABI entry registers (before they are clobbered), Pilot FAU bindings,
/// RLOC/FCST entries, and intra-shader chased pointer offsets.
std::vector<std::string> enrich_ebin_instructions(
    const Mbs2Module& mod,
    const ExtractedEbin& ebin,
    const PilotAnalysisResult& pilot_info);

/// Format decoded EBINs as human-readable Valhall disassembly, with optional MBS2 enrichment
/// and clear section markers (including `[PILOT SHADER]` for compute pilot EBINs).
std::string format_enriched_disassembly(
    const SpirvStageInfo& stage_info,
    uint8_t arch,
    const Mbs2Module& mod,
    const std::vector<ExtractedEbin>& ebins,
    const PilotAnalysisResult& pilot_info,
    bool include_modifiers = false,
    bool enrich = true);

}  // namespace valhall::spirv2valhall
