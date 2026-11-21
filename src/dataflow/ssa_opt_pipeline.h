#pragma once

#include "pointer_analysis.h"
#include "ssa_cse.h"
#include "ssa_dce.h"
#include "ssa_fp_opt.h"
#include "ssa_mem_opt.h"
#include "ssa_opt_utils.h"
#include "ssa_sccp.h"
#include "valhall_ssa.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace valhall::dataflow {

/// Configuration for the verification-oriented Valhall SSA optimization pipeline.
struct SsaOptPipelineOptions {
    bool run_sroa = true;
    bool run_spill_mem2reg = true;
    bool run_buffer_symbolization = true;
    bool run_sccp = true;
    bool run_fp_canonicalize = true;
    bool run_cse = true;
    bool run_dce = true;

    FpOptMode fp_mode = FpOptMode::StrictIEEE;
    bool enable_fp_tile_rewrites = true;
    bool enable_fp_cpp_rewrites = true;
    bool fau_is_readonly = true;
    uint32_t tls_sideband_member = 6;
    uint32_t max_fixpoint_rounds = 4;

    /// Optional pointer analysis result passed to buffer symbolization.
    const PointerAnalysisResult* ptr_analysis = nullptr;

    /// Record human-readable per-round diagnostics in `SsaOptPipelineResult::trace`.
    bool record_trace = false;
};

struct SsaOptPipelineResult {
    SsaPassStats total_stats;
    uint32_t rounds_executed = 0;
    std::vector<SymbolizedBufferAccess> symbolized_accesses;
    std::map<std::string, uint32_t> fp_rule_hits;
    std::vector<std::string> trace;
};

/// Executes the full verification-oriented Valhall SSA optimization pipeline:
///   1. Fixed-size struct SROA & TLS spill mem2reg promotion
///   2. Fixed-stride SSBO/UBO symbolization & store-to-load forwarding
///   3. Bounded fixpoint loop over SCCP -> FP canonicalization -> CSE -> DCE
///
/// This pipeline operates purely in `valhall::dataflow` and does not alter
/// `valhall_to_spirv`'s production ISA -> SSA lowering.
SsaOptPipelineResult run_ssa_opt_pipeline(compiler::ValhallSsaModule& mod,
                                          const SsaOptPipelineOptions& opts = {});

}  // namespace valhall::dataflow
