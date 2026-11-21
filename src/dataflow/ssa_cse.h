#pragma once

#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

namespace valhall::dataflow {

struct CseOptions {
    /// Treat `LoadFau` as pure and eligible for CSE (true in Main shaders, false in Pilot shaders).
    bool fau_is_readonly = true;

    /// Deduplicate identical Phi nodes within the same basic block.
    bool deduplicate_phis = true;
};

/// Dominator-tree scoped Common Subexpression Elimination (Global Value Numbering).
///
/// Performs a pre-order walk of the dominator tree maintaining a scoped expression
/// table. Pure instructions (and optionally identical block-local Phis) whose
/// canonicalized opcode, type, operands, and immediates match a dominating
/// computation are replaced with the dominating `SsaId`.
SsaPassStats run_ssa_cse(compiler::ValhallSsaModule& mod, const CseOptions& opts = {});

}  // namespace valhall::dataflow
