#pragma once

#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

#include <cstdint>
#include <map>
#include <string>

namespace valhall::dataflow {

/// Precision mode governing which floating-point algebraic identities are applied.
enum class FpOptMode : uint8_t {
    /// Bit-exact IEEE-754 identities only (Tier 1: preserves `-0.0`, NaN, Inf,
    /// and rounding; safe for exact Z3 `QF_FP` equivalence proofs).
    StrictIEEE = 0,

    /// Enables Tier-2 (`nsz`/`nnan`/`ninf`) and Tier-3 (`contract`/fast-math)
    /// identities to canonicalize Mali DDK (`libGLES_mali.so`) and Panfrost
    /// Valhall lowering idioms (`FRCP+FMUL -> FDiv`, `FRSQ+FRCP -> Sqrt`,
    /// `FMA` fusion, `*_RSCALE` `Ldexp` normalization, `Exp2/Log2/Pow`, `mix`).
    FastMath,
};

struct FpOptOptions {
    FpOptMode mode = FpOptMode::StrictIEEE;

    /// Canonicalize `FSub(a, b)` into `FAdd(a, FNeg(b))` to match Valhall's
    /// lack of a native `FSub` instruction.
    bool desugar_fsub = true;

    /// Maximum fixpoint rewrite iterations.
    uint32_t max_iterations = 8;
};

struct FpOptResult {
    SsaPassStats stats;
    /// Per-rule rewrite counts (keyed by rule ID such as `"I1_fadd_neg0"`,
    /// `"I6_fma_neg0"`, `"R1_fmul_rcp"`, `"R2_rcp_rsqrt"`, etc.) for diagnostics.
    std::map<std::string, uint32_t> rule_hits;
};

/// Canonicalizes and reduces floating-point algebraic identities emitted by
/// Arm's Mali DDK (`libGLES_mali.so` / `malioc`) and Mesa Panfrost/Bifrost/Valhall
/// over `compiler::ValhallSsaModule`.
FpOptResult run_ssa_fp_canonicalize(compiler::ValhallSsaModule& mod,
                                    const FpOptOptions& opts = {});

}  // namespace valhall::dataflow
