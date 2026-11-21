#pragma once

#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

namespace valhall::dataflow {

struct SccpOptions {
    /// Fold constant floating-point operations using IEEE-754 binary32 semantics.
    bool fold_fp = true;

    /// Simplify conditional branches with constant conditions and prune dead CFG edges.
    bool fold_branches = true;

    /// Run a trailing DCE sweep to clean up folded nodes and unreachable blocks.
    bool run_trailing_dce = true;

    /// Treat `LoadFau` as read-only during trailing DCE.
    bool fau_is_readonly = true;
};

/// Constant Folding and Wegman-Zadeck Sparse Conditional Constant Propagation (SCCP).
///
/// Propagates typed 32-bit, 64-bit, boolean, and IEEE-754 binary32 constants over
/// executable CFG edges using a 3-level lattice (`Undef < Constant < Overdefined`).
/// Materializes discovered constants, folds constant conditional branches into
/// unconditional branches, prunes dead Phi incomings, and cleans up dead nodes.
SsaPassStats run_ssa_sccp(compiler::ValhallSsaModule& mod, const SccpOptions& opts = {});

}  // namespace valhall::dataflow
