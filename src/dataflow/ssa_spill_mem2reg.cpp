#include "ssa_spill_mem2reg.h"

#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace valhall::dataflow {

using compiler::kInvalidSsa;
using compiler::PhiIncoming;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValhallSsaModule;
using compiler::ValType;
using tables::SsaOp;

TlsAddrInfo classify_tls_addr(const ValhallSsaModule& mod,
                              SsaId addr_id,
                              const PointerAnalysisResult& pa) {
    TlsAddrInfo res;
    if (addr_id == kInvalidSsa || addr_id >= mod.nodes.size()) {
        return res;
    }
    const TrackedVal tv = pa.val(addr_id);
    if (const auto* pv = tv.inner.get_if<PointerDomain>();
        pv && pv->is_valid_pointer() && pv->kind == PointerVal::Kind::TlsPtr) {
        res.is_tls = true;
        res.offset_known = pv->offset_known;
        res.byte_offset = pv->byte_offset;
    }
    return res;
}

uint32_t psb_elem_bytes(const SsaNode& n) {
    if (n.imm_u32 > 0) return n.imm_u32;
    if (n.type == ValType::U64 || n.type == ValType::S64) return 8;
    return 4;
}

namespace {

struct TlsSlotAccess {
    uint32_t block_id = 0;
    SsaId node_id = kInvalidSsa;
    bool is_store = false;
    int64_t offset = 0;
    uint32_t elem_bytes = 4;
    ValType val_type = ValType::U32;
};

struct TlsDiscoveryResult {
    bool has_unknown_tls_access = false;
    std::map<int64_t, uint32_t> slot_widths;
    std::set<int64_t> poisoned_slots;
    std::vector<TlsSlotAccess> accesses;
};

TlsDiscoveryResult discover_tls_accesses(const ValhallSsaModule& mod,
                                         const PointerAnalysisResult& pa) {
    TlsDiscoveryResult disc;
    for (size_t b = 0; b < mod.blocks.size(); ++b) {
        for (SsaId nid : mod.blocks[b].nodes) {
            const auto& n = mod.get(nid);
            if (n.op != SsaOp::PsbLoad && n.op != SsaOp::PsbStore) continue;
            if (n.args.empty()) continue;
            TlsAddrInfo info = classify_tls_addr(mod, n.args[0], pa);
            if (!info.is_tls) continue;
            if (!info.offset_known || info.byte_offset < 0) {
                disc.has_unknown_tls_access = true;
                continue;
            }
            uint32_t eb = psb_elem_bytes(n);
            auto it = disc.slot_widths.find(info.byte_offset);
            if (it == disc.slot_widths.end()) {
                disc.slot_widths[info.byte_offset] = eb;
            } else if (it->second != eb) {
                disc.poisoned_slots.insert(info.byte_offset);
            }
            TlsSlotAccess acc;
            acc.block_id = static_cast<uint32_t>(b);
            acc.node_id = nid;
            acc.is_store = (n.op == SsaOp::PsbStore);
            acc.offset = info.byte_offset;
            acc.elem_bytes = eb;
            if (acc.is_store && n.args.size() >= 2 && n.args[1] != kInvalidSsa) {
                acc.val_type = mod.get(n.args[1]).type;
            } else {
                acc.val_type = (eb == 8) ? ValType::U64 : ValType::U32;
            }
            disc.accesses.push_back(acc);
        }
    }
    return disc;
}

std::set<int64_t> select_promotable_tls_slots(TlsDiscoveryResult& disc) {
    std::set<int64_t> promotable;
    if (disc.has_unknown_tls_access || disc.accesses.empty()) {
        return promotable;
    }
    for (const auto& [off_a, w_a] : disc.slot_widths) {
        for (const auto& [off_b, w_b] : disc.slot_widths) {
            if (off_a == off_b) continue;
            if (off_a < off_b + static_cast<int64_t>(w_b) &&
                off_b < off_a + static_cast<int64_t>(w_a)) {
                disc.poisoned_slots.insert(off_a);
                disc.poisoned_slots.insert(off_b);
            }
        }
    }
    for (const auto& acc : disc.accesses) {
        if (acc.is_store && !disc.poisoned_slots.count(acc.offset)) {
            promotable.insert(acc.offset);
        }
    }
    return promotable;
}

struct BraunPromoteResult {
    std::vector<std::pair<SsaId, SsaId>> replacements;
    std::vector<SsaId> promoted_loads;
    std::vector<SsaId> promoted_stores;
    std::unordered_map<SsaId, std::vector<SsaId>> load_val_extracts;
    std::unordered_map<SsaId, std::vector<SsaId>> load_tok_extracts;
};

BraunPromoteResult promote_tls_slots_braun(ValhallSsaModule& mod,
                                           const DominatorTree& dt,
                                           const TlsDiscoveryResult& disc,
                                           const std::set<int64_t>& promotable) {
    BraunPromoteResult prom;
    for (const auto& blk : mod.blocks) {
        for (SsaId nid : blk.nodes) {
            const auto& n = mod.get(nid);
            if (n.op == SsaOp::TupleExtract && !n.args.empty()) {
                SsaId base = n.args[0];
                if (base != kInvalidSsa && mod.get(base).op == SsaOp::PsbLoad) {
                    if (n.imm_u32 == 0) prom.load_val_extracts[base].push_back(nid);
                    else if (n.imm_u32 == 1) prom.load_tok_extracts[base].push_back(nid);
                }
            }
        }
    }

    const size_t num_blocks = mod.blocks.size();
    std::map<int64_t, std::vector<SsaId>> current_def;
    std::map<int64_t, ValType> slot_type;
    for (int64_t off : promotable) {
        current_def[off].assign(num_blocks, kInvalidSsa);
        slot_type[off] = (disc.slot_widths.at(off) == 8) ? ValType::U64 : ValType::U32;
    }

    std::unordered_map<SsaId, int64_t> store_slot;
    std::unordered_map<SsaId, int64_t> load_slot;
    for (const auto& acc : disc.accesses) {
        if (!promotable.count(acc.offset)) continue;
        if (acc.is_store) {
            store_slot[acc.node_id] = acc.offset;
            slot_type[acc.offset] = acc.val_type;
        } else {
            load_slot[acc.node_id] = acc.offset;
        }
    }

    std::vector<std::vector<uint32_t>> preds(num_blocks);
    for (uint32_t b : dt.rpo_order) {
        for (uint32_t s : get_block_succs(mod.blocks[b])) {
            if (s < num_blocks && dt.reachable[s]) {
                preds[s].push_back(b);
            }
        }
    }

    std::function<SsaId(int64_t, uint32_t)> read_var = [&](int64_t off, uint32_t b) -> SsaId {
        if (current_def[off][b] != kInvalidSsa) {
            return current_def[off][b];
        }
        const auto& ps = preds[b];
        if (ps.empty()) {
            SsaNode z{};
            ValType vt = slot_type[off];
            z.type = vt;
            if (vt == ValType::U64 || vt == ValType::S64) {
                z.op = SsaOp::ConstU64;
                z.imm_u64 = 0;
            } else if (vt == ValType::F32) {
                z.op = SsaOp::ConstF32;
                z.imm_u32 = 0;
            } else if (vt == ValType::S32) {
                z.op = SsaOp::ConstS32;
                z.imm_u32 = 0;
            } else {
                z.op = SsaOp::ConstU32;
                z.imm_u32 = 0;
            }
            SsaId before = mod.blocks[b].nodes.empty() ? kInvalidSsa : mod.blocks[b].nodes.front();
            SsaId zid = insert_node_before(mod, b, before, std::move(z));
            current_def[off][b] = zid;
            return zid;
        }
        if (ps.size() == 1) {
            SsaId v = read_var(off, ps[0]);
            current_def[off][b] = v;
            return v;
        }
        SsaNode phi{};
        phi.op = SsaOp::Phi;
        phi.type = slot_type[off];
        phi.block_id = b;
        SsaId pid = mod.add_node(std::move(phi));
        mod.blocks[b].phis.push_back(pid);
        current_def[off][b] = pid;

        std::vector<PhiIncoming> incs;
        incs.reserve(ps.size());
        for (uint32_t p : ps) {
            incs.push_back(PhiIncoming{p, read_var(off, p)});
        }
        mod.get_mut(pid).phi_incomings = std::move(incs);
        return pid;
    };

    for (uint32_t b : dt.rpo_order) {
        for (SsaId nid : mod.blocks[b].nodes) {
            auto sit = store_slot.find(nid);
            if (sit != store_slot.end()) {
                const auto& sn = mod.get(nid);
                if (sn.args.size() >= 2 && sn.args[1] != kInvalidSsa) {
                    current_def[sit->second][b] = sn.args[1];
                }
            }
        }
    }

    for (uint32_t b : dt.rpo_order) {
        std::map<int64_t, SsaId> local_def;
        for (SsaId nid : mod.blocks[b].nodes) {
            auto sit = store_slot.find(nid);
            if (sit != store_slot.end()) {
                const auto& sn = mod.get(nid);
                if (sn.args.size() >= 2 && sn.args[1] != kInvalidSsa) {
                    local_def[sit->second] = sn.args[1];
                    prom.promoted_stores.push_back(nid);
                    if (sn.args.size() >= 3 && sn.args[2] != kInvalidSsa) {
                        prom.replacements.emplace_back(nid, sn.args[2]);
                    }
                }
                continue;
            }

            auto lit = load_slot.find(nid);
            if (lit != load_slot.end()) {
                int64_t off = lit->second;
                SsaId val = kInvalidSsa;
                auto ldit = local_def.find(off);
                if (ldit != local_def.end()) {
                    val = ldit->second;
                } else {
                    SsaId saved = current_def[off][b];
                    current_def[off][b] = kInvalidSsa;
                    val = read_var(off, b);
                    if (saved != kInvalidSsa) {
                        current_def[off][b] = saved;
                    }
                    local_def[off] = val;
                }

                const auto& ln = mod.get(nid);
                auto vit = prom.load_val_extracts.find(nid);
                if (vit != prom.load_val_extracts.end()) {
                    for (SsaId ex_id : vit->second) {
                        prom.replacements.emplace_back(ex_id, val);
                    }
                } else {
                    prom.replacements.emplace_back(nid, val);
                }
                auto tit = prom.load_tok_extracts.find(nid);
                if (tit != prom.load_tok_extracts.end() && ln.args.size() >= 2 && ln.args[1] != kInvalidSsa) {
                    for (SsaId ex_id : tit->second) {
                        prom.replacements.emplace_back(ex_id, ln.args[1]);
                    }
                }
                prom.promoted_loads.push_back(nid);
            }
        }
    }
    return prom;
}

void rewrite_and_erase_promoted_tls_ops(ValhallSsaModule& mod,
                                        const BraunPromoteResult& prom,
                                        const SpillMem2RegOptions& opts,
                                        SsaPassStats& stats) {
    if (prom.promoted_loads.empty() && prom.promoted_stores.empty()) {
        return;
    }
    std::vector<SsaId> remap(mod.nodes.size(), kInvalidSsa);
    for (size_t i = 0; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
    for (const auto& [old_id, new_id] : prom.replacements) {
        if (old_id > 0 && old_id < remap.size() && new_id > 0 && new_id < remap.size()) {
            remap[old_id] = new_id;
        }
    }
    apply_value_remap(mod, remap);

    stats.loads_promoted = static_cast<uint32_t>(prom.promoted_loads.size());
    if (opts.remove_dead_memory_ops) {
        std::vector<uint8_t> dead(mod.nodes.size(), 0);
        for (SsaId lid : prom.promoted_loads) {
            if (lid < dead.size()) dead[lid] = 1;
            auto vit = prom.load_val_extracts.find(lid);
            if (vit != prom.load_val_extracts.end()) {
                for (SsaId ex : vit->second) {
                    if (ex < dead.size()) dead[ex] = 1;
                }
            }
            auto tit = prom.load_tok_extracts.find(lid);
            if (tit != prom.load_tok_extracts.end()) {
                for (SsaId ex : tit->second) {
                    if (ex < dead.size()) dead[ex] = 1;
                }
            }
        }
        for (SsaId sid : prom.promoted_stores) {
            if (sid < dead.size()) dead[sid] = 1;
        }
        stats.stores_eliminated = static_cast<uint32_t>(prom.promoted_stores.size());
        stats.nodes_removed = erase_dead_nodes(mod, dead);
    }
    stats.phis_simplified = simplify_trivial_phis(mod);
}

}  // namespace

SsaPassStats run_ssa_spill_mem2reg(ValhallSsaModule& mod, const SpillMem2RegOptions& opts) {
    SsaPassStats stats;
    if (mod.blocks.empty()) return stats;

    PointerAnalysisEnv env;
    env.tls_base_sideband_member = opts.tls_sideband_member;
    const PointerAnalysisResult& pa = (opts.tls_sideband_member == 6)
        ? get_or_run_pointer_analysis(mod)
        : get_or_run_pointer_analysis(mod, env);

    TlsDiscoveryResult disc = discover_tls_accesses(mod, pa);
    std::set<int64_t> promotable = select_promotable_tls_slots(disc);
    if (promotable.empty()) return stats;

    DominatorTree dt = get_or_build_dominator_tree(mod);
    BraunPromoteResult prom = promote_tls_slots_braun(mod, dt, disc, promotable);
    rewrite_and_erase_promoted_tls_ops(mod, prom, opts, stats);
    return stats;
}

}  // namespace valhall::dataflow
