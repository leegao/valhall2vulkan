#pragma once

#include "isa_enricher.h"
#include "mbs2_ast_parser.h"
#include "mbs2_context_dumper.h"
#include "mbs2_types.h"
#include "moc_runner.h"
#include "spirv_inspect.h"
#include "spirv_roundtrip.h"

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

struct ExtractOptions {
    std::string core = "Mali-G710";
    uint8_t arch = 0;                       // 0 = infer from core name
    std::string malioc_path = "mali_offline_compiler/malioc";
    std::string moc_hook_path;              // empty = default build/libmoc_hook.so or $SPIRV2VALHALL_MOC_HOOK
    std::string clang_ast_dumper_path;      // empty = default build/clang_ast_dumper or $SPIRV2VALHALL_CLANG_AST_DUMPER
    std::string dump_header_path;           // optional path to persist generated MBS2 C header
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
    Mbs2Module mbs2;                        // Complete parsed cmpbe_chunk_MBS2 graph (via clang_ast_dumper)
    std::vector<ExtractedEbin> ebins;       // All EBINs (including Pilot shader for Compute)
    PilotAnalysisResult pilot_analysis;     // Chased SRT -> Resource -> Buffer -> FAU bindings from Pilot shader
    std::vector<LiftedSpirvShader> lifted_spirv;  // Populated when opts.lift_to_spirv == true
    std::string malioc_output;
};

/// Backwards-compatible wrapper that parses an MBS2 C header via clang_ast_dumper.
bool parse_mbs2_header_ebins(
    const std::string& header_text,
    uint8_t arch,
    bool disassemble,
    std::vector<ExtractedEbin>* out,
    std::string* err = nullptr);

/// Run malioc + libmoc_hook.so on a SPIR-V binary, parse the MBS2 C dump via clang_ast_dumper,
/// analyze pilot shaders, disassemble all EBINs, and optionally lift back to SPIR-V.
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
