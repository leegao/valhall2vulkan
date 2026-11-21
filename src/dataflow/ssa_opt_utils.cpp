#include "ssa_opt_utils.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <unordered_set>

namespace valhall::dataflow {

using compiler::kInvalidSsa;
using compiler::SsaBlock;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValhallSsaModule;
using compiler::ValType;
using tables::SsaOp;

std::vector<uint32_t> get_block_succs(const SsaBlock& blk) {
    std::vector<uint32_t> succs;
    if (blk.is_terminated_by_end || blk.is_terminated_by_discard) {
        return succs;
    }
    if (blk.fallthrough_block >= 0) {
        succs.push_back(static_cast<uint32_t>(blk.fallthrough_block));
    }
    if (blk.branch_target_block >= 0 && blk.branch_target_block != blk.fallthrough_block) {
        succs.push_back(static_cast<uint32_t>(blk.branch_target_block));
    }
    return succs;
}

DominatorTree build_dominator_tree(const ValhallSsaModule& mod) {
    DominatorTree dt;
    const size_t n = mod.blocks.size();
    dt.reachable.assign(n, 0);
    dt.idom.assign(n, DominatorTree::kNoIdom);
    dt.dom_children.resize(n);
    dt.tin.assign(n, 0);
    dt.tout.assign(n, 0);
    if (n == 0) return dt;

    // DFS post-order from block 0 to compute reachable set and RPO.
    std::vector<uint32_t> postorder;
    postorder.reserve(n);
    std::vector<uint8_t> visited(n, 0);
    std::function<void(uint32_t)> dfs = [&](uint32_t b) {
        visited[b] = 1;
        dt.reachable[b] = 1;
        for (uint32_t s : get_block_succs(mod.blocks[b])) {
            if (s < n && !visited[s]) {
                dfs(s);
            }
        }
        postorder.push_back(b);
    };
    dfs(0);

    dt.rpo_order.assign(postorder.rbegin(), postorder.rend());
    std::vector<int32_t> rpo_num(n, -1);
    for (size_t i = 0; i < dt.rpo_order.size(); ++i) {
        rpo_num[dt.rpo_order[i]] = static_cast<int32_t>(i);
    }

    // Build reachable predecessor lists directly from CFG edges so stale
    // `blk.preds` never corrupt dominance.
    std::vector<std::vector<uint32_t>> preds(n);
    for (uint32_t b : dt.rpo_order) {
        for (uint32_t s : get_block_succs(mod.blocks[b])) {
            if (s < n && dt.reachable[s]) {
                preds[s].push_back(b);
            }
        }
    }

    auto intersect = [&](int32_t f1, int32_t f2) -> int32_t {
        while (f1 != f2) {
            while (rpo_num[f1] > rpo_num[f2]) f1 = dt.idom[f1];
            while (rpo_num[f2] > rpo_num[f1]) f2 = dt.idom[f2];
        }
        return f1;
    };

    dt.idom[0] = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (uint32_t b : dt.rpo_order) {
            if (b == 0) continue;
            int32_t new_idom = DominatorTree::kNoIdom;
            for (uint32_t p : preds[b]) {
                if (dt.idom[p] == DominatorTree::kNoIdom) continue;
                if (new_idom == DominatorTree::kNoIdom) {
                    new_idom = static_cast<int32_t>(p);
                } else {
                    new_idom = intersect(static_cast<int32_t>(p), new_idom);
                }
            }
            if (new_idom != DominatorTree::kNoIdom && dt.idom[b] != new_idom) {
                dt.idom[b] = new_idom;
                changed = true;
            }
        }
    }

    for (uint32_t b : dt.rpo_order) {
        if (b != 0 && dt.idom[b] >= 0) {
            dt.dom_children[static_cast<size_t>(dt.idom[b])].push_back(b);
        }
    }

    uint32_t timer = 0;
    std::function<void(uint32_t)> euler = [&](uint32_t b) {
        dt.tin[b] = ++timer;
        for (uint32_t c : dt.dom_children[b]) {
            euler(c);
        }
        dt.tout[b] = ++timer;
    };
    euler(0);

    return dt;
}

bool is_pure_ssa_op(const SsaNode& node, bool fau_is_readonly) {
    if (node.is_side_effect || node.is_barrier_or_cf) {
        return false;
    }
    switch (node.op) {
        case SsaOp::Branch:
        case SsaOp::BranchCond:
        case SsaOp::ControlBarrier:
        case SsaOp::Kill:
        case SsaOp::Return:
        case SsaOp::PsbStore:
        case SsaOp::StoreBuiltinOutput:
        case SsaOp::StoreFragColor:
        case SsaOp::StoreVertVarying:
        case SsaOp::AtomicCmpXchg:
        case SsaOp::AtomicExchange:
        case SsaOp::AtomicRmw:
        case SsaOp::PsbLoad:
        case SsaOp::TexSample:
        case SsaOp::SubgroupBallot:
        case SsaOp::SubgroupShuffle:
        case SsaOp::DPdx:
        case SsaOp::DPdy:
        case SsaOp::Fwidth:
            return false;
        case SsaOp::LoadFau:
            return fau_is_readonly;
        default:
            return true;
    }
}

bool is_commutative_ssa_op(SsaOp op) {
    switch (op) {
        case SsaOp::IAdd:
        case SsaOp::IAdd64:
        case SsaOp::IMul:
        case SsaOp::IMul64:
        case SsaOp::BitAnd:
        case SsaOp::BitAnd64:
        case SsaOp::BitOr:
        case SsaOp::BitOr64:
        case SsaOp::BitXor:
        case SsaOp::BitXor64:
        case SsaOp::ICmpEq:
        case SsaOp::ICmpNe:
        case SsaOp::LogicalAnd:
        case SsaOp::LogicalOr:
        case SsaOp::FAdd:
        case SsaOp::FMul:
        case SsaOp::FMin:
        case SsaOp::FMax:
        case SsaOp::NMin:
        case SsaOp::NMax:
        case SsaOp::FCmpOrdEq:
        case SsaOp::FCmpOrdNe:
        case SsaOp::FCmpUnordNe:
            return true;
        default:
            return false;
    }
}

SsaId resolve_remap(const std::vector<SsaId>& remap, SsaId id) {
    uint32_t guard = 0;
    while (id > 0 && id < remap.size() && remap[id] != id && guard < remap.size()) {
        id = remap[id];
        ++guard;
    }
    return id;
}

uint32_t apply_value_remap(ValhallSsaModule& mod, const std::vector<SsaId>& remap) {
    uint32_t rewritten = 0;
    auto fix = [&](SsaId& slot) {
        SsaId r = resolve_remap(remap, slot);
        if (r != slot) {
            slot = r;
            ++rewritten;
        }
    };

    for (auto& blk : mod.blocks) {
        for (SsaId nid : blk.phis) {
            auto& n = mod.get_mut(nid);
            for (auto& inc : n.phi_incomings) fix(inc.value);
            for (auto& arg : n.args) fix(arg);
        }
        for (SsaId nid : blk.nodes) {
            auto& n = mod.get_mut(nid);
            for (auto& arg : n.args) fix(arg);
            for (auto& inc : n.phi_incomings) fix(inc.value);
        }
        fix(blk.branch_cond);
    }

    for (auto& regs : mod.block_in_regs) {
        for (auto& r : regs) fix(r);
    }
    for (auto& regs : mod.block_out_regs) {
        for (auto& r : regs) fix(r);
    }
    for (auto& t : mod.block_in_token) fix(t);
    for (auto& t : mod.block_out_token) fix(t);

    mod.recompute_use_counts();
    return rewritten;
}

uint32_t erase_dead_nodes(ValhallSsaModule& mod, const std::vector<uint8_t>& dead_mask) {
    uint32_t removed = 0;
    auto filter_vec = [&](std::vector<SsaId>& vec) {
        std::vector<SsaId> kept;
        kept.reserve(vec.size());
        for (SsaId id : vec) {
            if (id < dead_mask.size() && dead_mask[id]) {
                auto& n = mod.get_mut(id);
                n.op = SsaOp::Nop;
                n.args.clear();
                n.phi_incomings.clear();
                n.is_side_effect = false;
                n.is_barrier_or_cf = false;
                ++removed;
            } else {
                kept.push_back(id);
            }
        }
        vec = std::move(kept);
    };

    for (auto& blk : mod.blocks) {
        filter_vec(blk.phis);
        filter_vec(blk.nodes);
    }
    mod.recompute_use_counts();
    return removed;
}

uint32_t rebuild_cfg_preds_and_prune_phis(ValhallSsaModule& mod) {
    const size_t n = mod.blocks.size();
    if (n == 0) return 0;

    DominatorTree dt = build_dominator_tree(mod);
    for (auto& blk : mod.blocks) {
        blk.preds.clear();
    }
    for (uint32_t b : dt.rpo_order) {
        for (uint32_t s : get_block_succs(mod.blocks[b])) {
            if (s < n && dt.reachable[s]) {
                mod.blocks[s].preds.push_back(b);
            }
        }
    }

    uint32_t pruned_blocks = 0;
    for (size_t b = 0; b < n; ++b) {
        auto& blk = mod.blocks[b];
        if (!dt.reachable[b]) {
            if (!blk.phis.empty() || !blk.nodes.empty() || blk.branch_cond != kInvalidSsa) {
                ++pruned_blocks;
            }
            for (SsaId id : blk.phis) {
                auto& node = mod.get_mut(id);
                node.op = SsaOp::Nop;
                node.args.clear();
                node.phi_incomings.clear();
            }
            for (SsaId id : blk.nodes) {
                auto& node = mod.get_mut(id);
                node.op = SsaOp::Nop;
                node.args.clear();
                node.phi_incomings.clear();
            }
            blk.phis.clear();
            blk.nodes.clear();
            blk.branch_cond = kInvalidSsa;
            blk.branch_target_block = -1;
            blk.fallthrough_block = -1;
            blk.is_conditional_branch = false;
            blk.is_unconditional_branch = false;
            continue;
        }

        std::unordered_set<uint32_t> live_preds(blk.preds.begin(), blk.preds.end());
        for (SsaId pid : blk.phis) {
            auto& phi = mod.get_mut(pid);
            std::vector<compiler::PhiIncoming> kept;
            kept.reserve(phi.phi_incomings.size());
            for (const auto& inc : phi.phi_incomings) {
                if (live_preds.count(inc.pred_block)) {
                    kept.push_back(inc);
                }
            }
            phi.phi_incomings = std::move(kept);
        }
    }

    mod.recompute_use_counts();
    return pruned_blocks;
}

uint32_t simplify_trivial_phis(ValhallSsaModule& mod) {
    std::vector<SsaId> remap(mod.nodes.size(), kInvalidSsa);
    for (size_t i = 0; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
    std::vector<uint8_t> dead(mod.nodes.size(), 0);

    uint32_t eliminated = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& blk : mod.blocks) {
            for (SsaId pid : blk.phis) {
                if (dead[pid]) continue;
                const auto& phi = mod.get(pid);
                if (phi.op != SsaOp::Phi || phi.phi_incomings.empty()) continue;

                SsaId unique = kInvalidSsa;
                bool trivial = true;
                for (const auto& inc : phi.phi_incomings) {
                    SsaId v = resolve_remap(remap, inc.value);
                    if (v == pid || v == kInvalidSsa) continue;
                    if (unique == kInvalidSsa) {
                        unique = v;
                    } else if (unique != v) {
                        trivial = false;
                        break;
                    }
                }
                if (trivial && unique != kInvalidSsa) {
                    remap[pid] = unique;
                    dead[pid] = 1;
                    ++eliminated;
                    changed = true;
                }
            }
        }
    }

    if (eliminated > 0) {
        apply_value_remap(mod, remap);
        erase_dead_nodes(mod, dead);
    }
    return eliminated;
}

uint32_t run_ssa_copy_prop(ValhallSsaModule& mod) {
    std::vector<SsaId> remap(mod.nodes.size(), kInvalidSsa);
    for (size_t i = 0; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
    std::vector<uint8_t> dead(mod.nodes.size(), 0);

    uint32_t replaced = 0;
    for (const auto& blk : mod.blocks) {
        for (SsaId nid : blk.nodes) {
            const auto& n = mod.get(nid);
            if (n.op == SsaOp::Copy && n.args.size() == 1 && n.args[0] != kInvalidSsa) {
                SsaId src = resolve_remap(remap, n.args[0]);
                if (src != nid) {
                    remap[nid] = src;
                    dead[nid] = 1;
                    ++replaced;
                }
            } else if (n.op == SsaOp::Bitcast && n.args.size() == 1 && n.args[0] != kInvalidSsa) {
                SsaId src = resolve_remap(remap, n.args[0]);
                if (src != nid && mod.get(src).type == n.type) {
                    remap[nid] = src;
                    dead[nid] = 1;
                    ++replaced;
                }
            }
        }
    }

    if (replaced > 0) {
        apply_value_remap(mod, remap);
        erase_dead_nodes(mod, dead);
    }
    return replaced;
}

bool try_get_const_u32(const ValhallSsaModule& mod, SsaId id, uint32_t* out) {
    uint32_t depth = 0;
    while (id > 0 && id < mod.nodes.size() && depth < 16) {
        const auto& n = mod.get(id);
        if (n.op == SsaOp::Copy && n.args.size() == 1) {
            id = n.args[0];
            ++depth;
            continue;
        }
        if (n.op == SsaOp::Bitcast && n.args.size() == 1 &&
            (n.type == ValType::U32 || n.type == ValType::S32 || n.type == ValType::F32)) {
            id = n.args[0];
            ++depth;
            continue;
        }
        if (n.op == SsaOp::ConstU32 || n.op == SsaOp::ConstS32 ||
            n.op == SsaOp::ConstF32 || n.op == SsaOp::ConstBool) {
            if (out) *out = n.imm_u32;
            return true;
        }
        return false;
    }
    return false;
}

bool try_get_const_f32_bits(const ValhallSsaModule& mod, SsaId id, uint32_t* out_bits) {
    uint32_t depth = 0;
    while (id > 0 && id < mod.nodes.size() && depth < 16) {
        const auto& n = mod.get(id);
        if (n.op == SsaOp::Copy && n.args.size() == 1) {
            id = n.args[0];
            ++depth;
            continue;
        }
        if (n.op == SsaOp::Bitcast && n.args.size() == 1 && n.type == ValType::F32) {
            id = n.args[0];
            ++depth;
            continue;
        }
        if (n.op == SsaOp::ConstF32 || n.op == SsaOp::ConstU32 || n.op == SsaOp::ConstS32) {
            if (out_bits) *out_bits = n.imm_u32;
            return true;
        }
        return false;
    }
    return false;
}

bool try_get_const_u64(const ValhallSsaModule& mod, SsaId id, uint64_t* out) {
    uint32_t depth = 0;
    while (id > 0 && id < mod.nodes.size() && depth < 16) {
        const auto& n = mod.get(id);
        if (n.op == SsaOp::Copy && n.args.size() == 1) {
            id = n.args[0];
            ++depth;
            continue;
        }
        if (n.op == SsaOp::ConstU64) {
            if (out) *out = n.imm_u64;
            return true;
        }
        if (n.op == SsaOp::Pack64 && n.args.size() == 2) {
            uint32_t lo = 0, hi = 0;
            if (try_get_const_u32(mod, n.args[0], &lo) && try_get_const_u32(mod, n.args[1], &hi)) {
                if (out) *out = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
                return true;
            }
        }
        if (n.op == SsaOp::UConvert32To64 && n.args.size() == 1) {
            uint32_t v = 0;
            if (try_get_const_u32(mod, n.args[0], &v)) {
                if (out) *out = static_cast<uint64_t>(v);
                return true;
            }
        }
        if (n.op == SsaOp::SConvert32To64 && n.args.size() == 1) {
            uint32_t v = 0;
            if (try_get_const_u32(mod, n.args[0], &v)) {
                if (out) *out = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v)));
                return true;
            }
        }
        return false;
    }
    return false;
}

bool try_get_const_bool(const ValhallSsaModule& mod, SsaId id, bool* out) {
    uint32_t depth = 0;
    while (id > 0 && id < mod.nodes.size() && depth < 16) {
        const auto& n = mod.get(id);
        if (n.op == SsaOp::Copy && n.args.size() == 1) {
            id = n.args[0];
            ++depth;
            continue;
        }
        if (n.op == SsaOp::ConstBool) {
            if (out) *out = (n.imm_u32 != 0);
            return true;
        }
        return false;
    }
    return false;
}

uint32_t f32_to_bits(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

float bits_to_f32(uint32_t bits) {
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

SsaId insert_node_before(ValhallSsaModule& mod, uint32_t blk_id, SsaId before_node_id, SsaNode node) {
    node.block_id = blk_id;
    SsaId id = mod.add_node(std::move(node));
    auto& blk = mod.blocks[blk_id];
    if (mod.get(id).op == SsaOp::Phi) {
        blk.phis.push_back(id);
        return id;
    }
    if (before_node_id != kInvalidSsa) {
        auto it = std::find(blk.nodes.begin(), blk.nodes.end(), before_node_id);
        if (it != blk.nodes.end()) {
            blk.nodes.insert(it, id);
            return id;
        }
    }
    blk.nodes.push_back(id);
    return id;
}

}  // namespace valhall::dataflow
