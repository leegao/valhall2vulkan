#pragma once

#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

#include <cstdint>

namespace valhall::dataflow {

struct CseOptions {
    /// Treat `LoadFau` as a pure expression eligible for CSE.
    bool fau_is_readonly = true;

    /// Deduplicate identical `Phi` nodes within the same basic block.
    bool deduplicate_phis = true;

    /// Maximum dominator-tree passes (1 suffices for acyclic CFGs; 2+ catches
    /// loop-header `Phi` expressions exposed after a back-edge body is simplified).
    uint32_t max_iterations = 2;
};

/// Performs dominator-tree-scoped Global Value Numbering / CSE over pure SSA
/// nodes in `mod`, canonicalizing commutative binary operations and `Fma`
/// multiplicative operands (`args[0..1]`).
SsaPassStats run_ssa_cse(compiler::ValhallSsaModule& mod, const CseOptions& opts = {});

}  // namespace valhall::dataflow
