#include "ssa_cse.h"

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace valhall::dataflow {

namespace {

using compiler::kInvalidSsa;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValhallSsaModule;
using compiler::ValType;
using tables::SsaOp;

struct ExprKey {
    SsaOp op = SsaOp::Nop;
    ValType type = ValType::Void;
    std::vector<SsaId> args;
    uint64_t imm_u64 = 0;
    uint32_t imm_u32 = 0;
    uint32_t imm_u32_2 = 0;
    uint32_t aux_index = 0;
    bool aux_bool = false;
    bool aux_bool2 = false;
    std::string aux_str;

    bool operator==(const ExprKey& o) const {
        return op == o.op && type == o.type && args == o.args &&
               imm_u64 == o.imm_u64 && imm_u32 == o.imm_u32 && imm_u32_2 == o.imm_u32_2 &&
               aux_index == o.aux_index && aux_bool == o.aux_bool &&
               aux_bool2 == o.aux_bool2 && aux_str == o.aux_str;
    }
};

struct ExprKeyHash {
    size_t operator()(const ExprKey& k) const {
        size_t h = static_cast<size_t>(k.op) * 1315423911ULL;
        auto mix = [&](uint64_t v) {
            h ^= static_cast<size_t>(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(static_cast<uint64_t>(k.type));
        mix(k.imm_u64);
        mix(k.imm_u32);
        mix(k.imm_u32_2);
        mix(k.aux_index);
        mix((k.aux_bool ? 1ULL : 0ULL) | (k.aux_bool2 ? 2ULL : 0ULL));
        for (SsaId a : k.args) mix(a);
        if (!k.aux_str.empty()) {
            mix(std::hash<std::string>{}(k.aux_str));
        }
        return h;
    }
};

struct PhiKey {
    ValType type = ValType::Void;
    std::vector<std::pair<uint32_t, SsaId>> incomings;

    bool operator==(const PhiKey& o) const {
        return type == o.type && incomings == o.incomings;
    }
};

struct PhiKeyHash {
    size_t operator()(const PhiKey& k) const {
        size_t h = static_cast<size_t>(k.type) * 2654435761ULL;
        for (const auto& [pred, val] : k.incomings) {
            h ^= ((static_cast<size_t>(pred) << 16) ^ static_cast<size_t>(val)) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

ExprKey make_expr_key(const SsaNode& n, const std::vector<SsaId>& remap) {
    ExprKey k;
    k.op = n.op;
    k.type = n.type;
    k.imm_u64 = n.imm_u64;
    k.imm_u32 = n.imm_u32;
    k.imm_u32_2 = n.imm_u32_2;
    k.aux_index = n.aux_index;
    k.aux_bool = n.aux_bool;
    k.aux_bool2 = n.aux_bool2;
    k.aux_str = n.aux_str;

    k.args.reserve(n.args.size());
    for (SsaId a : n.args) {
        k.args.push_back(resolve_remap(remap, a));
    }
    if (is_commutative_ssa_op(n.op) && k.args.size() == 2) {
        if (k.args[1] < k.args[0]) {
            std::swap(k.args[0], k.args[1]);
        }
    } else if (n.op == SsaOp::Fma && k.args.size() == 3) {
        if (k.args[1] < k.args[0]) {
            std::swap(k.args[0], k.args[1]);
        }
    }
    return k;
}

}  // namespace

SsaPassStats run_ssa_cse(ValhallSsaModule& mod, const CseOptions& opts) {
    SsaPassStats stats;
    if (mod.blocks.empty()) return stats;

    DominatorTree dt = build_dominator_tree(mod);
    const size_t num_nodes = mod.nodes.size();
    std::vector<SsaId> remap(num_nodes, kInvalidSsa);
    for (size_t i = 0; i < num_nodes; ++i) remap[i] = static_cast<SsaId>(i);
    std::vector<uint8_t> dead(num_nodes, 0);

    std::unordered_map<ExprKey, SsaId, ExprKeyHash> table;
    std::vector<ExprKey> undo_stack;

    std::function<void(uint32_t)> dfs = [&](uint32_t b) {
        const size_t scope_mark = undo_stack.size();
        const auto& blk = mod.blocks[b];

        // Deduplicate identical Phi nodes within the same block.
        if (opts.deduplicate_phis && blk.phis.size() > 1) {
            std::unordered_map<PhiKey, SsaId, PhiKeyHash> phi_table;
            for (SsaId pid : blk.phis) {
                const auto& phi = mod.get(pid);
                if (phi.op != SsaOp::Phi || phi.phi_incomings.empty()) continue;
                PhiKey pk;
                pk.type = phi.type;
                pk.incomings.reserve(phi.phi_incomings.size());
                for (const auto& inc : phi.phi_incomings) {
                    pk.incomings.emplace_back(inc.pred_block, resolve_remap(remap, inc.value));
                }
                std::sort(pk.incomings.begin(), pk.incomings.end());
                auto it = phi_table.find(pk);
                if (it != phi_table.end()) {
                    remap[pid] = it->second;
                    dead[pid] = 1;
                    ++stats.nodes_replaced;
                    ++stats.phis_simplified;
                } else {
                    phi_table.emplace(std::move(pk), pid);
                }
            }
        }

        for (SsaId nid : blk.nodes) {
            const auto& n = mod.get(nid);
            if (n.op == SsaOp::Nop || n.op == SsaOp::Phi) continue;
            if (!is_pure_ssa_op(n, opts.fau_is_readonly)) continue;

            // Resolve Copy chains inline so downstream expressions see canonical IDs.
            if (n.op == SsaOp::Copy && n.args.size() == 1 && n.args[0] != kInvalidSsa) {
                SsaId src = resolve_remap(remap, n.args[0]);
                if (src != nid) {
                    remap[nid] = src;
                    dead[nid] = 1;
                    ++stats.nodes_replaced;
                }
                continue;
            }

            ExprKey key = make_expr_key(n, remap);
            auto it = table.find(key);
            if (it != table.end()) {
                remap[nid] = it->second;
                dead[nid] = 1;
                ++stats.nodes_replaced;
            } else {
                table.emplace(key, nid);
                undo_stack.push_back(std::move(key));
            }
        }

        for (uint32_t child : dt.dom_children[b]) {
            dfs(child);
        }

        while (undo_stack.size() > scope_mark) {
            table.erase(undo_stack.back());
            undo_stack.pop_back();
        }
    };

    if (dt.reachable[0]) {
        dfs(0);
    }

    if (stats.nodes_replaced > 0) {
        apply_value_remap(mod, remap);
        stats.nodes_removed += erase_dead_nodes(mod, dead);
    }
    return stats;
}

}  // namespace valhall::dataflow
