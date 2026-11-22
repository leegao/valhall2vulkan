#pragma once

#include "domains/sym_expr.h"
#include "pointer_analysis.h"
#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace valhall::dataflow {

/// Map from FAU uniform byte offset (0, 4, 8, ...) to resolved symbolic expression
/// derived from static `FCST` constants, `RLOC` relocations, and Pilot shader (`EBIN[1]`) analysis.
using FauSummaryMap = std::map<uint32_t, std::shared_ptr<const SymExpr>>;

/// Diagnostic classification for Valhall constructs that could not be statically de-FAU'd.
enum class FallbackReason : uint8_t {
    UnresolvedFauSlot = 0,
    DynamicFauOffset,
    SymExprDepthExceeded,
    UnsupportedOpInExpr,
};

/// Returns a short diagnostic string for `FallbackReason`.
const char* fallback_reason_name(FallbackReason reason);

struct FallbackRecord {
    uint64_t valhall_pc = 0;
    compiler::SsaId ssa_id = compiler::kInvalidSsa;
    tables::SsaOp op = tables::SsaOp::Nop;
    FallbackReason reason = FallbackReason::UnresolvedFauSlot;
    std::string detail;
};

struct DeFauDevirtOptions {
    bool materialize_fcst_constants = true;
    bool materialize_pilot_loads = true;
    bool materialize_pilot_exprs = true;
    bool materialize_buffer_addrs = true;
    bool record_fallbacks = true;
};

struct DeFauDevirtResult {
    SsaPassStats stats;
    uint32_t fau_consts_materialized = 0;
    uint32_t fau_loads_materialized = 0;
    uint32_t fau_exprs_materialized = 0;
    uint32_t fau_addrs_materialized = 0;
    std::vector<FallbackRecord> fallbacks;
};

/// Summarizes a Valhall Pilot shader (`EBIN[1]`) into a `FauSummaryMap` mapping
/// each FAU byte offset written by the Pilot to its symbolic expression
/// (`ConstU32`, `BufferAddr`, `BufferScalarLoad`, or bounded pure ALU/compare DAG).
FauSummaryMap run_pilot_fau_summary(const compiler::ValhallSsaModule& pilot_ssa,
                                    const PointerAnalysisEnv& pilot_env);

/// Seeds static `FCST` 32-bit constant words into a `FauSummaryMap` (without
/// overwriting existing Pilot-derived entries unless `overwrite == true`).
void seed_fcst_into_fau_summary(const std::map<uint32_t, uint32_t>& fcst_words,
                                FauSummaryMap& summary,
                                bool overwrite = false);

/// Rewrites `LoadFau` (and 64-bit `Pack64(LoadFau, LoadFau)` buffer addresses) in a
/// Main shader (`EBIN[0]`) using `fau_summary`, materializing constants, canonical
/// `LoadSidebandU64` + `PsbLoad` UBO/SSBO reads, and Pilot-hoisted predicate expressions.
DeFauDevirtResult run_ssa_defau_devirt(compiler::ValhallSsaModule& main_ssa,
                                       const FauSummaryMap& fau_summary,
                                       const DeFauDevirtOptions& opts = {});

}  // namespace valhall::dataflow
