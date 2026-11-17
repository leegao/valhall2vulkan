#include "valhall_cfg.h"
#include <set>
#include <algorithm>

namespace valhall::cfg {

static bool is_branch_inst(const DecodedValhallInst& ins) {
    return ins.canonical_mnemonic == Mnemonic::BRANCH ||
           ins.canonical_mnemonic == Mnemonic::JUMP ||
           ins.canonical_mnemonic == Mnemonic::JUMP_EX ||
           ins.mnemonic == Mnemonic::BRANCH ||
           ins.mnemonic == Mnemonic::JUMP ||
           ins.mnemonic == Mnemonic::JUMP_EX;
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
        if (is_branch_inst(ins) && ins.has_branch_offset && ins.branch_target_pc > 0) {
            uint64_t tgt = static_cast<uint64_t>(ins.branch_target_pc);
            if (tgt > max_branch_target_pc) max_branch_target_pc = tgt;
        }
        if (ins.flow == FlowControlM::end && ins.pc >= max_branch_target_pc) {
            // Check if any subsequent valid instruction is a branch target
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
        bool is_br = is_branch_inst(ins);
        bool is_end = (ins.flow == FlowControlM::end) || (ins.canonical_mnemonic == Mnemonic::DISCARD);
        bool is_reconv = (ins.flow == FlowControlM::reconverge);

        if (is_br) {
            if (ins.has_branch_offset && ins.branch_target_pc >= 0) {
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
        blk.is_terminated_by_discard = (last_ins.canonical_mnemonic == Mnemonic::DISCARD);

        if (is_branch_inst(last_ins)) {
            uint64_t tgt_pc = static_cast<uint64_t>(last_ins.branch_target_pc);
            blk.branch_target_pc = tgt_pc;
            auto it_tgt = cfg.pc_to_block_id.find(tgt_pc);
            if (it_tgt != cfg.pc_to_block_id.end()) {
                blk.branch_target_block = static_cast<int32_t>(it_tgt->second);
            }
            // Check if unconditional vs conditional
            bool uncond = false;
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
        } else if (!blk.is_terminated_by_end) {
            if (b + 1 < cfg.blocks.size()) {
                blk.fallthrough_pc = cfg.blocks[b + 1].start_pc;
                blk.fallthrough_block = static_cast<int32_t>(b + 1);
            }
        }
    }

    // 5. Identify loops (backward branches target_block <= b) and forward selections
    for (size_t b = 0; b < cfg.blocks.size(); ++b) {
        auto& blk = cfg.blocks[b];
        if ((blk.is_conditional_branch || blk.is_unconditional_branch) && blk.branch_target_block >= 0) {
            uint32_t tgt_b = static_cast<uint32_t>(blk.branch_target_block);
            if (tgt_b <= b) {
                // Backward branch: tgt_b is a loop header!
                cfg.blocks[tgt_b].is_loop_header = true;
                cfg.blocks[tgt_b].loop_backedge_block = static_cast<int32_t>(b);
                if (b + 1 < cfg.blocks.size()) {
                    cfg.blocks[tgt_b].loop_merge_block = static_cast<int32_t>(b + 1);
                }
            } else if (blk.is_conditional_branch) {
                // Forward conditional branch: selection merge at target block (or after if/else)
                blk.selection_merge_block = static_cast<int32_t>(tgt_b);
            }
        }
    }

    return cfg;
}

} // namespace valhall::cfg
