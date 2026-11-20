#include "valhall_ssa.h"
#include <algorithm>
#include <array>
#include <unordered_map>

namespace valhall::compiler {

void ValhallSsaModule::recompute_use_counts() {
    std::fill(use_counts.begin(), use_counts.end(), 0u);
    for (size_t i = 1; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        for (SsaId arg : n.args) {
            if (arg > 0 && arg < use_counts.size()) {
                use_counts[arg]++;
            }
        }
        for (const auto& inc : n.phi_incomings) {
            if (inc.value > 0 && inc.value < use_counts.size()) {
                use_counts[inc.value]++;
            }
        }
    }
    for (const auto& blk : blocks) {
        if (blk.branch_cond > 0 && blk.branch_cond < use_counts.size()) {
            use_counts[blk.branch_cond]++;
        }
    }
}

SplitCfgResult split_cfg_subblocks(const cfg::ControlFlowGraph& cfg) {
    SplitCfgResult res;
    if (cfg.blocks.empty()) return res;

    // First pass: split each cfg::BasicBlock at non-terminal instructions with splits_block == true
    // Map original cfg block ID -> (first sub-block ID, last sub-block ID)
    std::vector<uint32_t> orig_first(cfg.blocks.size(), 0);
    std::vector<uint32_t> orig_last(cfg.blocks.size(), 0);

    for (size_t ob = 0; ob < cfg.blocks.size(); ++ob) {
        const auto& ob_blk = cfg.blocks[ob];
        orig_first[ob] = static_cast<uint32_t>(res.blocks.size());

        std::vector<DecodedValhallInst> cur_insts;
        auto push_subblock = [&](bool is_final_subblock) {
            uint32_t sb_id = static_cast<uint32_t>(res.blocks.size());
            SsaBlock sb{};
            sb.id = sb_id;
            if (!is_final_subblock) {
                sb.fallthrough_block = static_cast<int32_t>(sb_id + 1);
            } else {
                sb.is_conditional_branch = ob_blk.is_conditional_branch;
                sb.is_unconditional_branch = ob_blk.is_unconditional_branch;
                sb.is_terminated_by_end = ob_blk.is_terminated_by_end;
                sb.is_terminated_by_discard = ob_blk.is_terminated_by_discard;
                sb.branch_target_block = ob_blk.branch_target_block;
                sb.fallthrough_block = ob_blk.fallthrough_block;
                sb.is_loop_header = ob_blk.is_loop_header;
                sb.loop_backedge_block = ob_blk.loop_backedge_block;
                sb.loop_merge_block = ob_blk.loop_merge_block;
                sb.selection_merge_block = ob_blk.selection_merge_block;
            }
            res.blocks.push_back(std::move(sb));
            res.block_insts.push_back(std::move(cur_insts));
            cur_insts.clear();
        };

        for (size_t k = 0; k < ob_blk.inst_indices.size(); ++k) {
            const auto& ins = cfg.instructions[ob_blk.inst_indices[k]];
            bool is_last_in_orig = (k + 1 == ob_blk.inst_indices.size());
            cur_insts.push_back(ins);

            const auto* rule = tables::lookup_lowering_rule(ins.spec_index);
            bool splits = rule ? rule->splits_block : false;
            // Only split mid-block if not already the last instruction of the original block
            if (splits && !is_last_in_orig) {
                push_subblock(false);
            }
        }
        push_subblock(true);
        orig_last[ob] = static_cast<uint32_t>(res.blocks.size() - 1);
        // Loop header flag belongs on the FIRST subblock of the original block
        if (ob_blk.is_loop_header && orig_first[ob] != orig_last[ob]) {
            res.blocks[orig_last[ob]].is_loop_header = false;
            res.blocks[orig_first[ob]].is_loop_header = true;
            res.blocks[orig_first[ob]].loop_backedge_block = ob_blk.loop_backedge_block;
            res.blocks[orig_first[ob]].loop_merge_block = ob_blk.loop_merge_block;
        }
    }

    // Remap original block IDs in all sub-block edges:
    // - branch_target_block -> orig_first[target]
    // - fallthrough_block (for final sub-blocks) -> orig_first[fall]
    // - loop_merge_block -> orig_first[merge]
    // - selection_merge_block -> orig_first[merge]
    // - loop_backedge_block -> orig_last[backedge]
    for (size_t ob = 0; ob < cfg.blocks.size(); ++ob) {
        for (uint32_t sb_id = orig_first[ob]; sb_id <= orig_last[ob]; ++sb_id) {
            auto& sb = res.blocks[sb_id];
            bool is_final = (sb_id == orig_last[ob]);
            if (is_final) {
                if (sb.branch_target_block >= 0) {
                    sb.branch_target_block = static_cast<int32_t>(orig_first[sb.branch_target_block]);
                }
                if (sb.fallthrough_block >= 0) {
                    sb.fallthrough_block = static_cast<int32_t>(orig_first[sb.fallthrough_block]);
                }
                if (sb.selection_merge_block >= 0) {
                    sb.selection_merge_block = static_cast<int32_t>(orig_first[sb.selection_merge_block]);
                }
            }
            if (sb.loop_merge_block >= 0) {
                sb.loop_merge_block = static_cast<int32_t>(orig_first[sb.loop_merge_block]);
            }
            if (sb.loop_backedge_block >= 0) {
                sb.loop_backedge_block = static_cast<int32_t>(orig_last[sb.loop_backedge_block]);
            }
        }
    }

    // Build predecessor lists
    for (uint32_t i = 0; i < res.blocks.size(); ++i) {
        const auto& sb = res.blocks[i];
        if (sb.branch_target_block >= 0) {
            res.blocks[sb.branch_target_block].preds.push_back(i);
        }
        if (sb.fallthrough_block >= 0 && sb.fallthrough_block != sb.branch_target_block) {
            res.blocks[sb.fallthrough_block].preds.push_back(i);
        }
    }

    return res;
}

SsaId SsaBuilder::current_reg(uint32_t blk_id, uint8_t reg) const {
    return mod_.block_out_regs[blk_id][reg & 63u];
}

void SsaBuilder::set_reg(uint32_t blk_id, uint8_t reg, SsaId val) {
    mod_.block_out_regs[blk_id][reg & 63u] = val;
}

SsaId SsaBuilder::current_token(uint32_t blk_id) const {
    return mod_.block_out_token[blk_id];
}

void SsaBuilder::set_token(uint32_t blk_id, SsaId tok) {
    mod_.block_out_token[blk_id] = tok;
}

SsaId SsaBuilder::emit(uint32_t blk_id, SsaNode n) {
    n.block_id = blk_id;
    if (n.orig_inst_idx < 0 && cur_orig_inst_idx_ >= 0) {
        n.orig_inst_idx = cur_orig_inst_idx_;
        n.orig_pc = cur_orig_pc_;
    }
    SsaId id = mod_.add_node(std::move(n));
    if (mod_.get(id).op == tables::SsaOp::Phi) {
        mod_.blocks[blk_id].phis.push_back(id);
    } else {
        mod_.blocks[blk_id].nodes.push_back(id);
    }
    return id;
}

SsaId SsaBuilder::emit_const_u32(uint32_t blk_id, uint32_t v) {
    SsaNode n{};
    n.op = tables::SsaOp::ConstU32;
    n.type = ValType::U32;
    n.imm_u32 = v;
    return emit(blk_id, std::move(n));
}

SsaId SsaBuilder::emit_const_s32(uint32_t blk_id, int32_t v) {
    SsaNode n{};
    n.op = tables::SsaOp::ConstS32;
    n.type = ValType::S32;
    n.imm_u32 = static_cast<uint32_t>(v);
    return emit(blk_id, std::move(n));
}

SsaId SsaBuilder::emit_const_f32_bits(uint32_t blk_id, uint32_t bits) {
    SsaNode n{};
    n.op = tables::SsaOp::ConstF32;
    n.type = ValType::F32;
    n.imm_u32 = bits;
    return emit(blk_id, std::move(n));
}

SsaId SsaBuilder::emit_const_u64(uint32_t blk_id, uint64_t v) {
    SsaNode n{};
    n.op = tables::SsaOp::ConstU64;
    n.type = ValType::U64;
    n.imm_u64 = v;
    n.effects.cap_bits |= EffectSet::kCapInt64;
    return emit(blk_id, std::move(n));
}

SsaId SsaBuilder::emit_const_bool(uint32_t blk_id, bool v) {
    SsaNode n{};
    n.op = tables::SsaOp::ConstBool;
    n.type = ValType::Bool;
    n.imm_u32 = v ? 1u : 0u;
    return emit(blk_id, std::move(n));
}

SsaId SsaBuilder::emit_bitcast(uint32_t blk_id, ValType target_t, SsaId val) {
    if (val != kInvalidSsa && mod_.get(val).type == target_t) {
        return val;
    }
    SsaNode n{};
    n.op = tables::SsaOp::Bitcast;
    n.type = target_t;
    n.args = { val };
    return emit(blk_id, std::move(n));
}

void SsaBuilder::seal_all_phis() {
    // Populate phi_incomings for all Phi nodes from block_out_regs / block_out_token
    for (size_t b = 0; b < mod_.blocks.size(); ++b) {
        const auto& blk = mod_.blocks[b];
        for (size_t p_idx = 0; p_idx < blk.phis.size(); ++p_idx) {
            SsaId phi_id = blk.phis[p_idx];
            auto& phi = mod_.get_mut(phi_id);
            phi.phi_incomings.clear();
            if (phi.aux_bool) {
                // Token Phi
                for (uint32_t pred : blk.preds) {
                    phi.phi_incomings.push_back({ pred, mod_.block_out_token[pred] });
                }
            } else {
                uint8_t reg = static_cast<uint8_t>(phi.aux_index & 63u);
                for (uint32_t pred : blk.preds) {
                    phi.phi_incomings.push_back({ pred, mod_.block_out_regs[pred][reg] });
                }
            }
        }
    }

    // Iteratively eliminate trivial Phis (Braun et al. 2013):
    // A Phi is trivial if all incoming values other than itself equal a single unique SsaId `same`.
    std::vector<SsaId> remap(mod_.nodes.size(), kInvalidSsa);
    for (size_t i = 0; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);

    auto resolve = [&](SsaId id) -> SsaId {
        while (id > 0 && id < remap.size() && remap[id] != id) {
            id = remap[id];
        }
        return id;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& blk : mod_.blocks) {
            std::vector<SsaId> kept;
            kept.reserve(blk.phis.size());
            for (SsaId phi_id : blk.phis) {
                auto& phi = mod_.get_mut(phi_id);
                SsaId unique = kInvalidSsa;
                bool trivial = true;
                for (auto& inc : phi.phi_incomings) {
                    inc.value = resolve(inc.value);
                    if (inc.value == phi_id || inc.value == unique) continue;
                    if (unique == kInvalidSsa) {
                        unique = inc.value;
                    } else {
                        trivial = false;
                        break;
                    }
                }
                if (trivial && unique != kInvalidSsa) {
                    remap[phi_id] = unique;
                    changed = true;
                } else {
                    kept.push_back(phi_id);
                }
            }
            blk.phis.swap(kept);
        }
    }

    // Apply remap to all node args and branch conditions
    for (size_t i = 1; i < mod_.nodes.size(); ++i) {
        auto& n = mod_.nodes[i];
        for (auto& a : n.args) a = resolve(a);
        for (auto& inc : n.phi_incomings) inc.value = resolve(inc.value);
    }
    for (auto& blk : mod_.blocks) {
        if (blk.branch_cond != kInvalidSsa) {
            blk.branch_cond = resolve(blk.branch_cond);
        }
    }

    mod_.recompute_use_counts();
}

} // namespace valhall::compiler
