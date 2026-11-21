#pragma once

#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

#include <cstdint>

namespace valhall::dataflow {

/// Options for fixed-size struct SROA (`run_ssa_sroa`).
struct SroaOptions {
    /// Sideband member index for `thread_local_pointer`.
    uint32_t tls_sideband_member = 6;

    /// Maximum struct footprint in bytes eligible for scalar replacement.
    uint32_t max_struct_bytes = 256;

    /// Promote scalarized 4-byte fields to SSA registers via `run_ssa_spill_mem2reg`.
    bool run_mem2reg_after_split = true;
};

/// Scalar Replacement of Aggregates (SROA) for fixed-size structs stored in TLS
/// or local scratch memory. Splits 64-bit `Pack64` / `Split64` struct stores and
/// loads into independent 32-bit scalar field accesses and optionally promotes
/// them to SSA values via `run_ssa_spill_mem2reg`.
SsaPassStats run_ssa_sroa(compiler::ValhallSsaModule& mod,
                          const SroaOptions& opts = {});

}  // namespace valhall::dataflow
