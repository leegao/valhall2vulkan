#include "liveness.h"
#include "worklist.h"

#include <algorithm>

namespace valhall::dataflow {

namespace {

inline bool is_valid_ssa(compiler::SsaId id, size_t num_nodes) {
    return id != compiler::kInvalidSsa && id < num_nodes;
}

}  // namespace

SsaLivenessResult run_ssa_liveness(
    const compiler::ValhallSsaModule& mod,
    uint32_t max_iterations) {
    SsaLivenessResult res;
    const size_t num_blocks = mod.blocks.size();
    const size_t num_nodes = mod.nodes.size();
    res.num_nodes = num_nodes;
    res.killed_after_node.resize(num_nodes);

    if (num_blocks == 0 || num_nodes == 0) {
        res.converged = true;
        return res;
    }

    res.block_live_in.assign(num_blocks, BitsetValue(num_nodes));
    res.block_live_out.assign(num_blocks, BitsetValue(num_nodes));

    // Precompute per-block domain facts (expressed directly as BitsetDomain values):
    //   phi_defs[B]    = { p | p ∈ B.phis }
    //   body_def[B]    = { n | n ∈ B.nodes }
    //   body_ue_use[B] = upward-exposed uses in B.nodes and B.branch_cond
    std::vector<BitsetValue> phi_defs(num_blocks, BitsetValue(num_nodes));
    std::vector<BitsetValue> body_def(num_blocks, BitsetValue(num_nodes));
    std::vector<BitsetValue> body_ue_use(num_blocks, BitsetValue(num_nodes));

    for (size_t b = 0; b < num_blocks; ++b) {
        const auto& blk = mod.blocks[b];
        for (compiler::SsaId pid : blk.phis) {
            if (is_valid_ssa(pid, num_nodes)) phi_defs[b].set(pid);
        }
        for (compiler::SsaId nid : blk.nodes) {
            if (!is_valid_ssa(nid, num_nodes)) continue;
            const auto& node = mod.get(nid);
            for (compiler::SsaId arg : node.args) {
                if (is_valid_ssa(arg, num_nodes) && !body_def[b].test(arg)) {
                    body_ue_use[b].set(arg);
                }
            }
            body_def[b].set(nid);
        }
        if (is_valid_ssa(blk.branch_cond, num_nodes) && !body_def[b].test(blk.branch_cond)) {
            body_ue_use[b].set(blk.branch_cond);
        }
    }

    // Solve backwards dataflow equations via shared directional CFG worklist:
    //   LIVE_OUT[B] = ⋃_{S ∈ succ(B)} ( (LIVE_IN[S] \ PHI_DEFS[S]) ∪ PHI_USES(B → S) )
    //   LIVE_IN[B]  = UE_USE[B] ∪ (LIVE_OUT[B] \ BODY_DEF[B])
    BitsetValue new_out(num_nodes);
    BitsetValue new_in(num_nodes);

    WorklistResult wr = run_cfg_worklist(
        mod,
        DataflowDirection::Backward,
        [&](uint32_t b) -> bool {
            const auto& blk = mod.blocks[b];
            new_out.clear_all();

            auto merge_succ = [&](int32_t succ_id) {
                if (succ_id < 0 || static_cast<size_t>(succ_id) >= num_blocks) return;
                const size_t s = static_cast<size_t>(succ_id);
                new_out.or_diff_with(res.block_live_in[s], phi_defs[s]);
                for (compiler::SsaId pid : mod.blocks[s].phis) {
                    if (!is_valid_ssa(pid, num_nodes)) continue;
                    for (const auto& inc : mod.get(pid).phi_incomings) {
                        if (inc.pred_block == b && is_valid_ssa(inc.value, num_nodes)) {
                            new_out.set(inc.value);
                        }
                    }
                }
            };
            merge_succ(blk.fallthrough_block);
            if (blk.branch_target_block != blk.fallthrough_block) {
                merge_succ(blk.branch_target_block);
            }

            new_in = body_ue_use[b];
            new_in.or_diff_with(new_out, body_def[b]);

            bool changed = !BitsetDomain::equals(new_out, res.block_live_out[b]) ||
                           !BitsetDomain::equals(new_in, res.block_live_in[b]);
            if (changed) {
                res.block_live_out[b] = new_out;
                res.block_live_in[b] = new_in;
            }
            return changed;
        },
        max_iterations);

    res.converged = wr.converged;
    res.iterations = wr.iterations;

    // Compute per-node killed sets by walking each block backwards from LIVE_OUT[B]
    for (size_t b = 0; b < num_blocks; ++b) {
        const auto& blk = mod.blocks[b];
        BitsetValue live = res.block_live_out[b];
        if (is_valid_ssa(blk.branch_cond, num_nodes)) {
            live.set(blk.branch_cond);
        }
        for (auto it = blk.nodes.rbegin(); it != blk.nodes.rend(); ++it) {
            compiler::SsaId nid = *it;
            if (!is_valid_ssa(nid, num_nodes)) continue;
            const auto& node = mod.get(nid);
            live.clear(nid);
            std::vector<compiler::SsaId> killed;
            for (compiler::SsaId arg : node.args) {
                if (is_valid_ssa(arg, num_nodes) && !live.test(arg)) {
                    if (std::find(killed.begin(), killed.end(), arg) == killed.end()) {
                        killed.push_back(arg);
                    }
                }
            }
            if (!killed.empty()) {
                res.killed_after_node[nid] = std::move(killed);
            }
            for (compiler::SsaId arg : node.args) {
                if (is_valid_ssa(arg, num_nodes)) live.set(arg);
            }
        }
    }

    return res;
}

}  // namespace valhall::dataflow
