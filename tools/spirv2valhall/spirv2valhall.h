#pragma once

#include "isa_enricher.h"
#include "mbs2_context_dumper.h"
#include "mbs2_direct_reader.h"
#include "mbs2_types.h"
#include "spirv_inspect.h"
#include "spirv_roundtrip.h"

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

struct ExtractOptions {
    std::string core = "Mali-G710";
    uint8_t arch = 0;                       // 0 = infer from core name
    std::string mali_lib_path = "mali_offline_compiler/graphics/libMali-Gxx_r55p0-00rel0.so";
    std::string dump_header_path;           // optional path to persist decompiled MBS2 C header
    bool strip_unused_caps = true;
    bool disassemble = true;
    bool lift_to_spirv = false;             // true = also lift Valhall ISA back to SPIR-V via valhall2spirv
    bool verbose = false;
};

struct ExtractResult {
    SpirvStageInfo stage_info;
    uint8_t arch = 10;
    bool stripped_caps = false;
    std::string dump_header_text;
    Mbs2Module mbs2;                        // Complete parsed cmpbe_chunk_MBS2 graph
    std::vector<ExtractedEbin> ebins;       // All EBINs (including Pilot shader for Compute)
    PilotAnalysisResult pilot_analysis;     // Chased SRT -> Resource -> Buffer -> FAU bindings from Pilot shader
    std::vector<LiftedSpirvShader> lifted_spirv;  // Populated when opts.lift_to_spirv == true
};

/// Compile a SPIR-V binary via direct in-process libMali-Gxx (`malioc_inproc`),
/// read the live `cmpbe_chunk_MBS2*` directly in memory, analyze pilot shaders via
/// SSA-level `src/dataflow`, disassemble all EBINs, and optionally lift back to SPIR-V.
bool extract_valhall_from_spirv(
    const std::string& spv_path,
    const ExtractOptions& opts,
    ExtractResult* out,
    std::string* err = nullptr);

/// Format the CSF Execution Context & Descriptor Graph (from cmpbe_chunk_MBS2, minus EBINs).
std::string format_context(const ExtractResult& res);

/// Format decoded EBINs as human-readable Valhall disassembly (enriched with MBS2 & Pilot metadata by default).
std::string format_disassembly(
    const ExtractResult& res,
    bool include_modifiers = false,
    bool enrich = true);

}  // namespace valhall::spirv2valhall
