#pragma once

#include "domains/known_bits.h"
#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

#include <cstdint>
#include <vector>

namespace valhall::dataflow {

struct FactProverOptions {
    bool eliminate_roundtrip_bitcasts = true;
    bool canonicalize_isnan = true;
    bool unwrap_bool_masks = true;
    bool simplify_bfi_bfe_16x2 = true;
    /// When true, assumes Valhall `LD_TEX_IMM` (`texture_u16`) integer coordinates fit in `[0, 65535]`.
    bool assume_valhall_tex16_range = true;
    bool enable_path_sensitive_refinement = true;
};

struct FactProverResult {
    SsaPassStats stats;
    uint32_t bitcasts_eliminated = 0;
    uint32_t isnan_canonicalized = 0;
    uint32_t bool_masks_unwrapped = 0;
    uint32_t bfi_bfe_projected = 0;
    uint32_t path_consts_folded = 0;
    uint32_t csel_ladders_collapsed = 0;
    std::vector<KnownBitsValue> known_bits;
};

/// Runs forward `KnownBitsDomain` dataflow analysis (via `SsaDomainTransferVisitor`
/// and `dispatch_ssa_transfer`) with path-sensitive `cond_br` edge refinement, then
/// discharges algebraic and extensional-equality identities in `mod`.
FactProverResult run_ssa_fact_prover(compiler::ValhallSsaModule& mod,
                                     const FactProverOptions& opts = {});

}  // namespace valhall::dataflow
