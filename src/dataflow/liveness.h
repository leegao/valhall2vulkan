#pragma once

#include "domains.h"
#include "valhall_ssa.h"

#include <cstdint>
#include <vector>

namespace valhall::dataflow {

/// Result of backwards SSA liveness dataflow analysis over a `ValhallSsaModule`.
///
/// Built directly on `BitsetDomain` and `run_cfg_worklist(DataflowDirection::Backward)`.
///
/// Standard SSA liveness semantics (Brandner et al. / Boissinot et al.):
///   LIVE_OUT[B] = ⋃_{S ∈ succ(B)} ( (LIVE_IN[S] \ PHI_DEFS[S]) ∪ PHI_USES(B → S) )
///   LIVE_IN[B]  = UE_USE[B] ∪ (LIVE_OUT[B] \ DEF[B])
struct SsaLivenessResult {
    bool converged = false;
    uint32_t iterations = 0;
    size_t num_nodes = 0;

    /// Per-block live-in and live-out sets expressed as `BitsetDomain::Value`.
    std::vector<BitsetDomain::Value> block_live_in;
    std::vector<BitsetDomain::Value> block_live_out;

    /// Per-SSA-node list of SsaIds killed (last-used) at that node.
    /// Indexed by `SsaId` (same size as `mod.nodes`).
    std::vector<std::vector<compiler::SsaId>> killed_after_node;

    bool is_live_in(uint32_t block_id, compiler::SsaId val) const {
        if (block_id >= block_live_in.size() || val == compiler::kInvalidSsa || val >= num_nodes) {
            return false;
        }
        return block_live_in[block_id].test(val);
    }

    bool is_live_out(uint32_t block_id, compiler::SsaId val) const {
        if (block_id >= block_live_out.size() || val == compiler::kInvalidSsa || val >= num_nodes) {
            return false;
        }
        return block_live_out[block_id].test(val);
    }

    bool is_killed_at(compiler::SsaId node_id, compiler::SsaId val) const {
        if (node_id >= killed_after_node.size()) return false;
        for (compiler::SsaId k : killed_after_node[node_id]) {
            if (k == val) return true;
        }
        return false;
    }
};

/// Run backwards SSA liveness dataflow analysis over `mod` until fixpoint.
SsaLivenessResult run_ssa_liveness(
    const compiler::ValhallSsaModule& mod,
    uint32_t max_iterations = 64);

}  // namespace valhall::dataflow
