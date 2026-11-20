#include "worklist.h"

#include <algorithm>
#include <deque>
#include <vector>

namespace valhall::dataflow {

WorklistResult run_cfg_worklist(
    const compiler::ValhallSsaModule& mod,
    DataflowDirection dir,
    const std::function<bool(uint32_t block_id)>& transfer_block,
    uint32_t max_iterations) {
    WorklistResult res;
    const size_t num_blocks = mod.blocks.size();
    if (num_blocks == 0 || !transfer_block) {
        res.converged = true;
        return res;
    }

    std::deque<uint32_t> worklist;
    std::vector<bool> in_worklist(num_blocks, false);

    if (dir == DataflowDirection::Forward) {
        for (uint32_t b = 0; b < num_blocks; ++b) {
            worklist.push_back(b);
            in_worklist[b] = true;
        }
    } else {
        for (size_t b = num_blocks; b-- > 0; ) {
            worklist.push_back(static_cast<uint32_t>(b));
            in_worklist[b] = true;
        }
    }

    const size_t max_visits = static_cast<size_t>(max_iterations) * std::max<size_t>(1, num_blocks);

    auto enqueue = [&](int32_t blk_id) {
        if (blk_id >= 0 && static_cast<size_t>(blk_id) < num_blocks && !in_worklist[blk_id]) {
            worklist.push_back(static_cast<uint32_t>(blk_id));
            in_worklist[blk_id] = true;
        }
    };

    while (!worklist.empty() && res.visits < max_visits) {
        uint32_t b = worklist.front();
        worklist.pop_front();
        in_worklist[b] = false;
        ++res.visits;

        if (transfer_block(b)) {
            const auto& blk = mod.blocks[b];
            if (dir == DataflowDirection::Forward) {
                enqueue(blk.fallthrough_block);
                if (blk.branch_target_block != blk.fallthrough_block) {
                    enqueue(blk.branch_target_block);
                }
            } else {
                for (uint32_t pred : blk.preds) {
                    enqueue(static_cast<int32_t>(pred));
                }
            }
        }
    }

    res.converged = worklist.empty();
    res.iterations = static_cast<uint32_t>((res.visits + num_blocks - 1) / std::max<size_t>(1, num_blocks));
    return res;
}

}  // namespace valhall::dataflow
