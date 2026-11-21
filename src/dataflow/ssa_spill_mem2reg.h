#pragma once

#include "pointer_analysis.h"
#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

#include <cstdint>

namespace valhall::dataflow {

/// Options for TLS spill promotion (`run_ssa_spill_mem2reg`).
struct SpillMem2RegOptions {
    /// Sideband member index for `thread_local_pointer` (default 6 in Valhall ABI).
    uint32_t tls_sideband_member = 6;

    /// Remove promoted dead TLS stores and loads after rewriting uses.
    bool remove_dead_memory_ops = true;
};

/// Result of classifying a pointer SSA value against the TLS address space.
struct TlsAddrInfo {
    bool is_tls = false;
    bool offset_known = false;
    int64_t byte_offset = 0;
};

/// Classifies whether `addr_id` is a TLS pointer (with optional constant byte
/// offset) directly from `pa` (`PointerVal::Kind::TlsPtr`).
TlsAddrInfo classify_tls_addr(const compiler::ValhallSsaModule& mod,
                              compiler::SsaId addr_id,
                              const PointerAnalysisResult& pa);

/// Returns the byte width accessed by a `PsbLoad` or `PsbStore` node.
uint32_t psb_elem_bytes(const compiler::SsaNode& n);

/// Promotes thread-local spill slots (`PsbStore` / `PsbLoad` to constant TLS
/// offsets) into SSA registers and `Phi` nodes using Braun-style SSA construction.
SsaPassStats run_ssa_spill_mem2reg(compiler::ValhallSsaModule& mod,
                                   const SpillMem2RegOptions& opts = {});

}  // namespace valhall::dataflow
