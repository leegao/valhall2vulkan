#include "valhall_cfg.h"
#include <set>
#include <algorithm>
#include <vector>

namespace valhall::cfg {

static bool is_branch_inst(const DecodedValhallInst& ins) {
    return ins.canonical_mnemonic == Mnemonic::BRANCH ||
           ins.canonical_mnemonic == Mnemonic::JUMP ||
           ins.canonical_mnemonic == Mnemonic::JUMP_EX ||
           ins.mnemonic == Mnemonic::BRANCH ||
           ins.mnemonic == Mnemonic::JUMP ||
           ins.mnemonic == Mnemonic::JUMP_EX;
}

/// Non-.end BLEND encodes a fixed-function return PC that skips the immediately
/// following software blend-shader fallback trampoline (ADR r48, ret; JUMP uN.w1).
static bool is_blend_return_jump(const DecodedValhallInst& ins) {
    return (ins.canonical_mnemonic == Mnemonic::BLEND || ins.mnemonic == Mnemonic::BLEND) &&
           ins.flow != FlowControlM::end &&
           ins.has_branch_offset &&
           ins.branch_target_pc > static_cast<int64_t>(ins.pc);
}

ControlFlowGraph build_cfg(const std::vector<DecodedValhallInst>& raw_instructions) {
    ControlFlowGraph cfg;
    if (raw_instructions.empty()) return cfg;

    // 1. Determine the executable instruction boundary:
    // Stop before trailing INVALID instructions or after the last reachable .end / branch target.
    size_t max_valid = 0;
    uint64_t max_branch_target_pc = 0;
    for (size_t i = 0; i < raw_instructions.size(); ++i) {
        const auto& ins = raw_instructions[i];
        if (!ins.valid || ins.mnemonic == Mnemonic::INVALID) {
            break;
        }
        max_valid = i + 1;
        if ((is_branch_inst(ins) || is_blend_return_jump(ins)) && ins.has_branch_offset && ins.branch_target_pc > 0) {
            uint64_t tgt = static_cast<uint64_t>(ins.branch_target_pc);
            if (tgt > max_branch_target_pc) max_branch_target_pc = tgt;
        }
        if (ins.flow == FlowControlM::end && ins.pc >= max_branch_target_pc) {
            break;
        }
    }
    if (max_valid == 0) max_valid = raw_instructions.size();
    cfg.executable_inst_count = max_valid;
    cfg.instructions.assign(raw_instructions.begin(), raw_instructions.begin() + max_valid);

    // 2. Discover basic block leaders (by PC)
    std::set<uint64_t> leader_pcs;
    leader_pcs.insert(cfg.instructions.front().pc);

    uint64_t max_exec_pc = cfg.instructions.back().pc;
    for (size_t i = 0; i < cfg.instructions.size(); ++i) {
        const auto& ins = cfg.instructions[i];
        bool is_br = is_branch_inst(ins) || is_blend_return_jump(ins);
        bool is_end = (ins.flow == FlowControlM::end) || (ins.canonical_mnemonic == Mnemonic::DISCARD && ins.srcs.size() < 2);
        bool is_reconv = (ins.flow == FlowControlM::reconverge);

        if (is_br) {
            if (ins.has_branch_offset) {
                uint64_t tgt = static_cast<uint64_t>(ins.branch_target_pc);
                if (tgt <= max_exec_pc + 8) {
                    leader_pcs.insert(tgt);
                }
            }
            if (i + 1 < cfg.instructions.size()) {
                leader_pcs.insert(cfg.instructions[i + 1].pc);
            }
        } else if (is_end || is_reconv) {
            if (i + 1 < cfg.instructions.size()) {
                leader_pcs.insert(cfg.instructions[i + 1].pc);
            }
        }
    }

    // 3. Partition instructions into BasicBlocks
    BasicBlock cur;
    bool has_cur = false;
    for (size_t i = 0; i < cfg.instructions.size(); ++i) {
        const auto& ins = cfg.instructions[i];
        if (leader_pcs.count(ins.pc) || !has_cur) {
            if (has_cur && !cur.inst_indices.empty()) {
                cfg.blocks.push_back(cur);
            }
            cur = BasicBlock{};
            cur.id = static_cast<uint32_t>(cfg.blocks.size());
            cur.start_pc = ins.pc;
            cur.end_pc = ins.pc;
            has_cur = true;
            cfg.pc_to_block_id[ins.pc] = cur.id;
        }
        cur.inst_indices.push_back(i);
        cur.end_pc = ins.pc;
    }
    if (has_cur && !cur.inst_indices.empty()) {
        cfg.blocks.push_back(cur);
    }

    // 4. Populate block terminators & CFG edges
    for (size_t b = 0; b < cfg.blocks.size(); ++b) {
        auto& blk = cfg.blocks[b];
        const auto& last_ins = cfg.instructions[blk.inst_indices.back()];
        blk.has_reconverge = (last_ins.flow == FlowControlM::reconverge);
        blk.is_terminated_by_end = (last_ins.flow == FlowControlM::end);
        blk.is_terminated_by_discard = (last_ins.canonical_mnemonic == Mnemonic::DISCARD && last_ins.srcs.size() < 2);

        if (is_branch_inst(last_ins) || is_blend_return_jump(last_ins)) {
            if (last_ins.has_branch_offset) {
                uint64_t tgt_pc = static_cast<uint64_t>(last_ins.branch_target_pc);
                blk.branch_target_pc = tgt_pc;
                auto it_tgt = cfg.pc_to_block_id.find(tgt_pc);
                if (it_tgt != cfg.pc_to_block_id.end()) {
                    blk.branch_target_block = static_cast<int32_t>(it_tgt->second);
                }
            }
            // Check if unconditional vs conditional
            bool uncond = is_blend_return_jump(last_ins);
            auto it_cond = last_ins.modifier_strs.find("condition");
            if (it_cond != last_ins.modifier_strs.end() && it_cond->second == "always") {
                uncond = true;
            }
            if (last_ins.mnemonic == Mnemonic::JUMP || last_ins.mnemonic == Mnemonic::JUMP_EX || last_ins.srcs.empty()) {
                uncond = true;
            }
            if (uncond) {
                blk.is_unconditional_branch = true;
            } else {
                blk.is_conditional_branch = true;
                if (b + 1 < cfg.blocks.size()) {
                    blk.fallthrough_pc = cfg.blocks[b + 1].start_pc;
                    blk.fallthrough_block = static_cast<int32_t>(b + 1);
                }
            }
        } else if (!blk.is_terminated_by_end && !blk.is_terminated_by_discard) {
            if (b + 1 < cfg.blocks.size()) {
                blk.fallthrough_pc = cfg.blocks[b + 1].start_pc;
                blk.fallthrough_block = static_cast<int32_t>(b + 1);
            }
        }
    }

    // 5. Compute immediate post-dominators (Cooper-Harvey-Kennedy) for structured merge blocks
    const size_t N = cfg.blocks.size();
    const size_t EXIT_NODE = N;
    std::vector<std::vector<size_t>> succs(N + 1);
    for (size_t b = 0; b < N; ++b) {
        const auto& blk = cfg.blocks[b];
        if (blk.branch_target_block >= 0) {
            succs[b].push_back(static_cast<size_t>(blk.branch_target_block));
        }
        if (blk.fallthrough_block >= 0) {
            succs[b].push_back(static_cast<size_t>(blk.fallthrough_block));
        }
        if (succs[b].empty()) {
            succs[b].push_back(EXIT_NODE);
        }
    }

    // Post-order on reverse CFG (i.e., reverse post-order from EXIT_NODE)
    // Since blocks are ordered roughly in program order, node N (EXIT_NODE) has highest post-order index,
    // and we compute ipdom using iterative dataflow over post-dominator sets or CHK intersect.
    // Using RPO indices from EXIT_NODE on the reverse graph:
    std::vector<int> ipdom(N + 1, -1);
    ipdom[EXIT_NODE] = static_cast<int>(EXIT_NODE);

    // Reverse post-order from EXIT_NODE (which is topological order in reverse CFG):
    // For small CFGs (N <= few hundred), simple bitset/vector or CHK with RPO is exact.
    // Let's build RPO of reverse CFG starting at EXIT_NODE:
    std::vector<std::vector<size_t>> r_succs(N + 1);
    for (size_t u = 0; u < N; ++u) {
        for (size_t v : succs[u]) {
            r_succs[v].push_back(u);
        }
    }
    std::vector<bool> visited(N + 1, false);
    std::vector<size_t> post_order;
    post_order.reserve(N + 1);
    auto dfs_rev = [&](auto& self, size_t u) -> void {
        visited[u] = true;
        for (size_t v : r_succs[u]) {
            if (!visited[v]) self(self, v);
        }
        post_order.push_back(u);
    };
    dfs_rev(dfs_rev, EXIT_NODE);

    std::vector<int> rpo_rank(N + 1, -1);
    std::vector<size_t> rpo_nodes;
    rpo_nodes.reserve(post_order.size());
    for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
        rpo_rank[*it] = static_cast<int>(rpo_nodes.size());
        rpo_nodes.push_back(*it);
    }

    auto intersect = [&](int b1, int b2) -> int {
        while (b1 != b2) {
            while (rpo_rank[b1] > rpo_rank[b2]) b1 = ipdom[b1];
            while (rpo_rank[b2] > rpo_rank[b1]) b2 = ipdom[b2];
        }
        return b1;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t u : rpo_nodes) {
            if (u == EXIT_NODE) continue;
            int new_ipdom = -1;
            for (size_t v : succs[u]) {
                if (ipdom[v] != -1) {
                    if (new_ipdom == -1) new_ipdom = static_cast<int>(v);
                    else new_ipdom = intersect(static_cast<int>(v), new_ipdom);
                }
            }
            if (new_ipdom != -1 && new_ipdom != ipdom[u]) {
                ipdom[u] = new_ipdom;
                changed = true;
            }
        }
    }

    // 6. Identify loops (backward branches target_block <= b) and forward selections
    for (size_t b = 0; b < N; ++b) {
        auto& blk = cfg.blocks[b];
        if ((blk.is_conditional_branch || blk.is_unconditional_branch) && blk.branch_target_block >= 0) {
            uint32_t tgt_b = static_cast<uint32_t>(blk.branch_target_block);
            if (tgt_b <= b) {
                // Backward branch: tgt_b is a loop header!
                cfg.blocks[tgt_b].is_loop_header = true;
                // Keep the furthest backedge if multiple exist
                if (cfg.blocks[tgt_b].loop_backedge_block < static_cast<int32_t>(b)) {
                    cfg.blocks[tgt_b].loop_backedge_block = static_cast<int32_t>(b);
                }
                int loop_ipdom = ipdom[tgt_b];
                if (loop_ipdom >= 0 && static_cast<size_t>(loop_ipdom) < N && static_cast<size_t>(loop_ipdom) > b) {
                    cfg.blocks[tgt_b].loop_merge_block = loop_ipdom;
                } else if (b + 1 < N) {
                    cfg.blocks[tgt_b].loop_merge_block = static_cast<int32_t>(b + 1);
                }
            } else if (blk.is_conditional_branch) {
                // Forward conditional branch: use immediate post-dominator as selection merge block!
                int sel_ipdom = ipdom[b];
                if (sel_ipdom >= 0 && static_cast<size_t>(sel_ipdom) < N && static_cast<size_t>(sel_ipdom) > b) {
                    blk.selection_merge_block = sel_ipdom;
                } else {
                    // One arm terminates (e.g. .end or DISCARD); target block is valid merge target
                    blk.selection_merge_block = static_cast<int32_t>(tgt_b);
                }
            }
        }
    }

    return cfg;
}

} // namespace valhall::cfg
