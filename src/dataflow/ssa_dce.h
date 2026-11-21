#pragma once

#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

namespace valhall::dataflow {

struct DceOptions {
    /// Treat `LoadFau` as a pure read-only op (true for Main shaders, false for Pilot shaders).
    bool fau_is_readonly = true;

    /// Collapse `Copy` / identity `Bitcast` nodes and trivial Phis before sweeping.
    bool propagate_copies = true;

    /// Prune unreachable CFG blocks before marking live roots.
    bool prune_unreachable_blocks = true;
};

/// Standard mark-and-sweep Dead Code Elimination over `compiler::ValhallSsaModule`.
///
/// Marks all side-effecting instructions, barriers, control-flow terminators,
/// branch conditions of reachable blocks, and their transitive operand closures
/// as live, then removes all remaining dead nodes and Phis.
SsaPassStats run_ssa_dce(compiler::ValhallSsaModule& mod, const DceOptions& opts = {});

}  // namespace valhall::dataflow
