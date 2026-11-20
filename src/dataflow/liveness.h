#pragma once

#include "valhall_ssa.h"

#include <cstdint>
#include <vector>

namespace valhall::dataflow {

/// Result of backwards SSA liveness dataflow analysis over a `ValhallSsaModule`.
///
/// Uses compact bitsets (64 SSA IDs per word) so liveness scales to 100KB+ shaders
/// in O(|B| * |V| / 64) memory and single-digit milliseconds.
///
/// Standard SSA liveness semantics (Brandner et al. / Boissinot et al.):
///   - A Phi node's incoming operand `(pred_block, val)` is a *use* along the
///     CFG edge `pred_block -> block`, NOT at the entry of `block`.
///   - `is_live_in(B, v)` is true iff `v` is live at the start of `B` (including
///     `B`'s own Phi definitions, excluding Phi incoming uses from predecessors).
///   - `is_live_out(B, v)` is true iff `v` is live on exit from `B` (including
///     Phi incoming values required by any successor `S` along edge `B -> S`).
///   - `killed_after_node[n]` lists SSA values whose last use on the current
///     path occurs at SSA node `n` (i.e. used by `n`, not live after `n`).
struct SsaLivenessResult {
    bool converged = false;
    uint32_t iterations = 0;
    size_t num_nodes = 0;
    size_t bit_words = 0;

    /// Per-block live-in and live-out compact bitsets (size = num_blocks * bit_words).
    std::vector<uint64_t> block_live_in_bits;
    std::vector<uint64_t> block_live_out_bits;

    /// Per-SSA-node list of SsaIds killed (last-used) at that node.
    /// Indexed by `SsaId` (same size as `mod.nodes`).
    std::vector<std::vector<compiler::SsaId>> killed_after_node;

    bool is_live_in(uint32_t block_id, compiler::SsaId val) const {
        if (val == compiler::kInvalidSsa || val >= num_nodes || bit_words == 0) return false;
        size_t idx = static_cast<size_t>(block_id) * bit_words + (val >> 6);
        if (idx >= block_live_in_bits.size()) return false;
        return (block_live_in_bits[idx] & (1ULL << (val & 63u))) != 0;
    }

    bool is_live_out(uint32_t block_id, compiler::SsaId val) const {
        if (val == compiler::kInvalidSsa || val >= num_nodes || bit_words == 0) return false;
        size_t idx = static_cast<size_t>(block_id) * bit_words + (val >> 6);
        if (idx >= block_live_out_bits.size()) return false;
        return (block_live_out_bits[idx] & (1ULL << (val & 63u))) != 0;
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
