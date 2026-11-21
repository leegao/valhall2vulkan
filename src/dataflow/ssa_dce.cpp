#include "ssa_dce.h"

#include <queue>
#include <vector>

namespace valhall::dataflow {

using compiler::kInvalidSsa;
using compiler::SsaId;
using compiler::ValhallSsaModule;
using tables::SsaOp;

SsaPassStats run_ssa_dce(ValhallSsaModule& mod, const DceOptions& opts) {
    SsaPassStats stats;
    if (mod.blocks.empty()) return stats;

    if (opts.prune_unreachable_blocks) {
        stats.blocks_pruned += rebuild_cfg_preds_and_prune_phis(mod);
    }

    if (opts.propagate_copies) {
        stats.nodes_replaced += run_ssa_copy_prop(mod);
        stats.phis_simplified += simplify_trivial_phis(mod);
    }

    const size_t num_nodes = mod.nodes.size();
    std::vector<uint8_t> live(num_nodes, 0);
    std::queue<SsaId> worklist;

    auto mark = [&](SsaId id) {
        if (id > 0 && id < num_nodes && !live[id]) {
            live[id] = 1;
            worklist.push(id);
        }
    };

    // Seed critical roots across reachable blocks.
    DominatorTree dt = build_dominator_tree(mod);
    for (size_t b = 0; b < mod.blocks.size(); ++b) {
        if (!dt.reachable[b]) continue;
        const auto& blk = mod.blocks[b];
        if (blk.branch_cond != kInvalidSsa) {
            mark(blk.branch_cond);
        }
        for (SsaId nid : blk.nodes) {
            const auto& n = mod.get(nid);
            if (n.op == SsaOp::Nop) continue;
            if (!is_pure_ssa_op(n, opts.fau_is_readonly)) {
                mark(nid);
            }
        }
    }

    // Backward mark transitive operand closure.
    while (!worklist.empty()) {
        SsaId cur = worklist.front();
        worklist.pop();
        const auto& n = mod.get(cur);
        for (SsaId arg : n.args) {
            mark(arg);
        }
        for (const auto& inc : n.phi_incomings) {
            if (inc.pred_block < dt.reachable.size() && dt.reachable[inc.pred_block]) {
                mark(inc.value);
            }
        }
    }

    std::vector<uint8_t> dead(num_nodes, 0);
    for (size_t b = 0; b < mod.blocks.size(); ++b) {
        const auto& blk = mod.blocks[b];
        for (SsaId pid : blk.phis) {
            if (pid > 0 && pid < num_nodes && !live[pid]) {
                dead[pid] = 1;
            }
        }
        for (SsaId nid : blk.nodes) {
            if (nid > 0 && nid < num_nodes && !live[nid]) {
                dead[nid] = 1;
            }
        }
    }

    stats.nodes_removed += erase_dead_nodes(mod, dead);
    return stats;
}

}  // namespace valhall::dataflow
