#pragma once
#include "valhall_isa_generated.h"
#include <cstdint>
#include <vector>
#include <map>

namespace valhall::cfg {

struct BasicBlock {
    uint32_t id = 0;
    uint64_t start_pc = 0;
    uint64_t end_pc = 0; // inclusive PC of last instruction in block
    std::vector<size_t> inst_indices;

    // Terminator info
    bool is_terminated_by_end = false;
    bool is_terminated_by_discard = false;
    bool is_unconditional_branch = false;
    bool is_conditional_branch = false;
    bool has_reconverge = false;

    uint64_t branch_target_pc = 0;
    uint64_t fallthrough_pc = 0;

    int32_t branch_target_block = -1;
    int32_t fallthrough_block = -1;

    // Structured control flow metadata
    bool is_loop_header = false;
    int32_t loop_backedge_block = -1;
    int32_t loop_merge_block = -1;
    int32_t selection_merge_block = -1;
};

struct ControlFlowGraph {
    std::vector<DecodedValhallInst> instructions;
    size_t executable_inst_count = 0;
    std::vector<BasicBlock> blocks;
    std::map<uint64_t, uint32_t> pc_to_block_id;
};

// Reconstructs the Control Flow Graph and structured merge points from decoded Valhall instructions.
// Automatically detects trailing non-executable constant pools after the last reachable .end instruction.
ControlFlowGraph build_cfg(const std::vector<DecodedValhallInst>& raw_instructions);

} // namespace valhall::cfg
