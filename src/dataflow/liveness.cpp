#include "liveness.h"

#include <algorithm>
#include <cstring>
#include <deque>

namespace valhall::dataflow {

namespace {

inline bool is_valid_ssa(compiler::SsaId id, size_t num_nodes) {
    return id != compiler::kInvalidSsa && id < num_nodes;
}

inline void set_bit(uint64_t* bits, compiler::SsaId id) {
    bits[id >> 6] |= (1ULL << (id & 63u));
}

inline void clear_bit(uint64_t* bits, compiler::SsaId id) {
    bits[id >> 6] &= ~(1ULL << (id & 63u));
}

inline bool test_bit(const uint64_t* bits, compiler::SsaId id) {
    return (bits[id >> 6] & (1ULL << (id & 63u))) != 0;
}

}  // namespace

SsaLivenessResult run_ssa_liveness(
    const compiler::ValhallSsaModule& mod,
    uint32_t max_iterations) {
    SsaLivenessResult res;
    const size_t num_blocks = mod.blocks.size();
    const size_t num_nodes = mod.nodes.size();
    const size_t words = (num_nodes + 63u) / 64u;
    res.num_nodes = num_nodes;
    res.bit_words = words;
    res.killed_after_node.resize(num_nodes);

    if (num_blocks == 0 || words == 0) {
        res.converged = true;
        return res;
    }

    res.block_live_in_bits.assign(num_blocks * words, 0ULL);
    res.block_live_out_bits.assign(num_blocks * words, 0ULL);

    // Precompute compact bitsets per block:
    //   phi_defs[B], body_def[B], body_ue_use[B]
    std::vector<uint64_t> phi_defs(num_blocks * words, 0ULL);
    std::vector<uint64_t> body_def(num_blocks * words, 0ULL);
    std::vector<uint64_t> body_ue_use(num_blocks * words, 0ULL);

    for (size_t b = 0; b < num_blocks; ++b) {
        const auto& blk = mod.blocks[b];
        uint64_t* pdef = &phi_defs[b * words];
        uint64_t* bdef = &body_def[b * words];
        uint64_t* bue  = &body_ue_use[b * words];

        for (compiler::SsaId pid : blk.phis) {
            if (is_valid_ssa(pid, num_nodes)) set_bit(pdef, pid);
        }
        for (compiler::SsaId nid : blk.nodes) {
            if (!is_valid_ssa(nid, num_nodes)) continue;
            const auto& node = mod.get(nid);
            for (compiler::SsaId arg : node.args) {
                if (is_valid_ssa(arg, num_nodes) && !test_bit(bdef, arg)) {
                    set_bit(bue, arg);
                }
            }
            set_bit(bdef, nid);
        }
        if (is_valid_ssa(blk.branch_cond, num_nodes) && !test_bit(bdef, blk.branch_cond)) {
            set_bit(bue, blk.branch_cond);
        }
    }

    // Backwards chaotic worklist (seeded in reverse block order)
    std::deque<uint32_t> worklist;
    std::vector<bool> in_worklist(num_blocks, false);
    for (size_t b = num_blocks; b-- > 0; ) {
        worklist.push_back(static_cast<uint32_t>(b));
        in_worklist[b] = true;
    }

    const size_t max_visits = static_cast<size_t>(max_iterations) * std::max<size_t>(1, num_blocks);
    size_t visits = 0;

    std::vector<uint64_t> new_out(words, 0ULL);
    std::vector<uint64_t> new_in(words, 0ULL);

    while (!worklist.empty() && visits < max_visits) {
        uint32_t b = worklist.front();
        worklist.pop_front();
        in_worklist[b] = false;
        ++visits;

        const auto& blk = mod.blocks[b];
        std::fill(new_out.begin(), new_out.end(), 0ULL);

        // LIVE_OUT[B] = U_{S in succ(B)} ( (LIVE_IN[S] \ PHI_DEFS[S]) U PHI_USES(B -> S) )
        auto merge_succ = [&](int32_t succ_id) {
            if (succ_id < 0 || static_cast<size_t>(succ_id) >= num_blocks) return;
            const size_t s = static_cast<size_t>(succ_id);
            const uint64_t* sin = &res.block_live_in_bits[s * words];
            const uint64_t* spdef = &phi_defs[s * words];
            for (size_t w = 0; w < words; ++w) {
                new_out[w] |= (sin[w] & ~spdef[w]);
            }
            for (compiler::SsaId pid : mod.blocks[s].phis) {
                if (!is_valid_ssa(pid, num_nodes)) continue;
                const auto& pnode = mod.get(pid);
                for (const auto& inc : pnode.phi_incomings) {
                    if (inc.pred_block == b && is_valid_ssa(inc.value, num_nodes)) {
                        set_bit(new_out.data(), inc.value);
                    }
                }
            }
        };
        merge_succ(blk.fallthrough_block);
        if (blk.branch_target_block != blk.fallthrough_block) {
            merge_succ(blk.branch_target_block);
        }

        // LIVE_IN[B] = UE_USE[B] U (LIVE_OUT[B] \ BODY_DEF[B])
        const uint64_t* bue  = &body_ue_use[b * words];
        const uint64_t* bdef = &body_def[b * words];
        uint64_t* cur_out = &res.block_live_out_bits[b * words];
        uint64_t* cur_in  = &res.block_live_in_bits[b * words];
        bool changed = false;
        for (size_t w = 0; w < words; ++w) {
            new_in[w] = bue[w] | (new_out[w] & ~bdef[w]);
            if (new_out[w] != cur_out[w] || new_in[w] != cur_in[w]) {
                cur_out[w] = new_out[w];
                cur_in[w] = new_in[w];
                changed = true;
            }
        }

        if (changed) {
            for (uint32_t pred : blk.preds) {
                if (pred < num_blocks && !in_worklist[pred]) {
                    worklist.push_back(pred);
                    in_worklist[pred] = true;
                }
            }
        }
    }

    res.converged = worklist.empty();
    res.iterations = static_cast<uint32_t>((visits + num_blocks - 1) / std::max<size_t>(1, num_blocks));

    // Compute per-node killed sets by walking each block backwards from LIVE_OUT[B]
    std::vector<uint64_t> live(words, 0ULL);
    for (size_t b = 0; b < num_blocks; ++b) {
        const auto& blk = mod.blocks[b];
        std::memcpy(live.data(), &res.block_live_out_bits[b * words], words * sizeof(uint64_t));
        if (is_valid_ssa(blk.branch_cond, num_nodes)) {
            set_bit(live.data(), blk.branch_cond);
        }
        for (auto it = blk.nodes.rbegin(); it != blk.nodes.rend(); ++it) {
            compiler::SsaId nid = *it;
            if (!is_valid_ssa(nid, num_nodes)) continue;
            const auto& node = mod.get(nid);
            clear_bit(live.data(), nid);
            std::vector<compiler::SsaId> killed;
            for (compiler::SsaId arg : node.args) {
                if (is_valid_ssa(arg, num_nodes) && !test_bit(live.data(), arg)) {
                    if (std::find(killed.begin(), killed.end(), arg) == killed.end()) {
                        killed.push_back(arg);
                    }
                }
            }
            if (!killed.empty()) {
                res.killed_after_node[nid] = std::move(killed);
            }
            for (compiler::SsaId arg : node.args) {
                if (is_valid_ssa(arg, num_nodes)) {
                    set_bit(live.data(), arg);
                }
            }
        }
    }

    return res;
}

}  // namespace valhall::dataflow
