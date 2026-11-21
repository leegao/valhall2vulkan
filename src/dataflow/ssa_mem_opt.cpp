#include "ssa_mem_opt.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace valhall::dataflow {

namespace {

using compiler::kInvalidSsa;
using compiler::PhiIncoming;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValhallSsaModule;
using compiler::ValType;
using tables::SsaOp;

struct TlsAddrInfo {
    bool is_tls = false;
    bool offset_known = false;
    int64_t byte_offset = 0;
};

/// Traces `addr_id` backward through Copy, IAdd64, ISub64, UConvert32To64,
/// SConvert32To64, and Pack64 to determine whether it derives from the TLS base
/// pointer (`LoadSidebandU64(imm_u32 == tls_member)` or `aux_str == "tls_private"`).
TlsAddrInfo classify_tls_addr(const ValhallSsaModule& mod,
                              SsaId addr_id,
                              uint32_t tls_member,
                              uint32_t depth = 0) {
    TlsAddrInfo res;
    if (addr_id == kInvalidSsa || addr_id >= mod.nodes.size() || depth > 24) {
        return res;
    }
    const auto& n = mod.get(addr_id);
    if (n.op == SsaOp::Copy && n.args.size() == 1) {
        return classify_tls_addr(mod, n.args[0], tls_member, depth + 1);
    }
    if (n.op == SsaOp::LoadSidebandU64) {
        if (n.imm_u32 == tls_member || n.aux_str == "tls_private") {
            res.is_tls = true;
            res.offset_known = true;
            res.byte_offset = 0;
        }
        return res;
    }
    if ((n.op == SsaOp::IAdd64 || n.op == SsaOp::ISub64) && n.args.size() == 2) {
        TlsAddrInfo lhs = classify_tls_addr(mod, n.args[0], tls_member, depth + 1);
        uint64_t rc = 0;
        if (lhs.is_tls) {
            if (lhs.offset_known && try_get_const_u64(mod, n.args[1], &rc)) {
                res.is_tls = true;
                res.offset_known = true;
                int64_t delta = static_cast<int64_t>(rc);
                res.byte_offset = (n.op == SsaOp::IAdd64) ? (lhs.byte_offset + delta)
                                                          : (lhs.byte_offset - delta);
                return res;
            }
            res.is_tls = true;
            res.offset_known = false;
            return res;
        }
        if (n.op == SsaOp::IAdd64) {
            TlsAddrInfo rhs = classify_tls_addr(mod, n.args[1], tls_member, depth + 1);
            uint64_t lc = 0;
            if (rhs.is_tls) {
                if (rhs.offset_known && try_get_const_u64(mod, n.args[0], &lc)) {
                    res.is_tls = true;
                    res.offset_known = true;
                    res.byte_offset = rhs.byte_offset + static_cast<int64_t>(lc);
                    return res;
                }
                res.is_tls = true;
                res.offset_known = false;
                return res;
            }
        }
    }
    return res;
}

uint32_t psb_elem_bytes(const SsaNode& n) {
    if (n.imm_u32 > 0) return n.imm_u32;
    if (n.type == ValType::U64 || n.type == ValType::S64) return 8;
    return 4;
}

/// Checks if a `PsbLoad` / `PsbStore` node's value type is a 64-bit value.
bool is_64bit_store_val(const ValhallSsaModule& mod, const SsaNode& store_node) {
    if (psb_elem_bytes(store_node) == 8) return true;
    if (store_node.args.size() >= 2 && store_node.args[1] != kInvalidSsa) {
        ValType vt = mod.get(store_node.args[1]).type;
        if (vt == ValType::U64 || vt == ValType::S64) return true;
    }
    return false;
}

struct AffineTerm {
    bool valid = false;
    SsaId index_ssa = kInvalidSsa;
    uint32_t stride_bytes = 0;
    int64_t const_offset = 0;
};

/// Unwraps `Copy`, `UConvert32To64`, `SConvert32To64` to extract a linear affine
/// offset expression `index_ssa * stride_bytes + const_offset`.
AffineTerm match_affine_offset(const ValhallSsaModule& mod, SsaId id, uint32_t depth = 0) {
    AffineTerm res;
    if (id == kInvalidSsa || id >= mod.nodes.size() || depth > 16) return res;

    uint64_t c64 = 0;
    if (try_get_const_u64(mod, id, &c64)) {
        res.valid = true;
        res.index_ssa = kInvalidSsa;
        res.stride_bytes = 0;
        res.const_offset = static_cast<int64_t>(c64);
        return res;
    }
    uint32_t c32 = 0;
    if (try_get_const_u32(mod, id, &c32)) {
        res.valid = true;
        res.index_ssa = kInvalidSsa;
        res.stride_bytes = 0;
        res.const_offset = static_cast<int64_t>(c32);
        return res;
    }

    const auto& n = mod.get(id);
    if ((n.op == SsaOp::Copy || n.op == SsaOp::UConvert32To64 || n.op == SsaOp::SConvert32To64) &&
        n.args.size() == 1) {
        return match_affine_offset(mod, n.args[0], depth + 1);
    }

    if ((n.op == SsaOp::IAdd || n.op == SsaOp::IAdd64) && n.args.size() == 2) {
        AffineTerm a = match_affine_offset(mod, n.args[0], depth + 1);
        AffineTerm b = match_affine_offset(mod, n.args[1], depth + 1);
        if (a.valid && b.valid) {
            if (a.index_ssa == kInvalidSsa) {
                b.const_offset += a.const_offset;
                return b;
            }
            if (b.index_ssa == kInvalidSsa) {
                a.const_offset += b.const_offset;
                return a;
            }
            if (a.index_ssa == b.index_ssa) {
                a.stride_bytes += b.stride_bytes;
                a.const_offset += b.const_offset;
                return a;
            }
        }
    }

    if ((n.op == SsaOp::ISub || n.op == SsaOp::ISub64) && n.args.size() == 2) {
        AffineTerm a = match_affine_offset(mod, n.args[0], depth + 1);
        AffineTerm b = match_affine_offset(mod, n.args[1], depth + 1);
        if (a.valid && b.valid && b.index_ssa == kInvalidSsa) {
            a.const_offset -= b.const_offset;
            return a;
        }
    }

    if ((n.op == SsaOp::IMul || n.op == SsaOp::IMul64) && n.args.size() == 2) {
        AffineTerm a = match_affine_offset(mod, n.args[0], depth + 1);
        AffineTerm b = match_affine_offset(mod, n.args[1], depth + 1);
        if (a.valid && b.valid && b.index_ssa == kInvalidSsa && b.const_offset > 0 && b.const_offset <= 65536) {
            uint32_t mul = static_cast<uint32_t>(b.const_offset);
            a.stride_bytes *= mul;
            a.const_offset *= static_cast<int64_t>(mul);
            return a;
        }
        if (a.valid && b.valid && a.index_ssa == kInvalidSsa && a.const_offset > 0 && a.const_offset <= 65536) {
            uint32_t mul = static_cast<uint32_t>(a.const_offset);
            b.stride_bytes *= mul;
            b.const_offset *= static_cast<int64_t>(mul);
            return b;
        }
    }

    if ((n.op == SsaOp::Shl || n.op == SsaOp::Shl64) && n.args.size() == 2) {
        uint32_t sh = 0;
        uint64_t sh64 = 0;
        bool has_sh = try_get_const_u32(mod, n.args[1], &sh) ||
                      (try_get_const_u64(mod, n.args[1], &sh64) && (sh = static_cast<uint32_t>(sh64), true));
        if (has_sh && sh <= 16) {
            AffineTerm a = match_affine_offset(mod, n.args[0], depth + 1);
            if (a.valid) {
                uint32_t mul = 1u << sh;
                a.stride_bytes *= mul;
                a.const_offset *= static_cast<int64_t>(mul);
                return a;
            }
        }
    }

    // Leaf symbolic index with unit stride 1.
    res.valid = true;
    res.index_ssa = id;
    res.stride_bytes = 1;
    res.const_offset = 0;
    return res;
}

struct BufferAddrMatch {
    bool matched = false;
    std::string buf_key;
    SsaId index_ssa = kInvalidSsa;
    uint32_t stride_bytes = 0;
    int64_t field_offset = 0;
};

/// Resolves a base pointer node to a canonical buffer key (`ssbo:*`, `ubo:*`, `push_const`).
bool try_get_buffer_base_key(const ValhallSsaModule& mod,
                             SsaId id,
                             const PointerAnalysisResult* ptr_res,
                             std::string* out_key) {
    if (id == kInvalidSsa || id >= mod.nodes.size()) return false;
    const auto& n = mod.get(id);
    if (n.op == SsaOp::LoadSidebandU64 && !n.aux_str.empty()) {
        if (n.aux_str.rfind("ssbo:", 0) == 0 || n.aux_str.rfind("ubo:", 0) == 0 ||
            n.aux_str == "push_const") {
            *out_key = n.aux_str;
            return true;
        }
    }
    if (ptr_res) {
        TrackedVal tv = ptr_res->val(id);
        if (const auto* pv = tv.inner.get_if<PointerDomain>();
            pv && pv->is_valid_pointer() && pv->kind == PointerVal::Kind::BufferAddr &&
            pv->offset_known && pv->byte_offset == 0) {
            if (!pv->symbol_name.empty() &&
                (pv->symbol_name.rfind("ssbo:", 0) == 0 || pv->symbol_name.rfind("ubo:", 0) == 0)) {
                *out_key = pv->symbol_name;
            } else {
                *out_key = "buf:" + std::to_string(pv->rack) + ":" + std::to_string(pv->desc_idx);
            }
            return true;
        }
    }
    return false;
}

BufferAddrMatch match_buffer_affine_addr(const ValhallSsaModule& mod,
                                         SsaId addr_id,
                                         const PointerAnalysisResult* ptr_res,
                                         uint32_t depth = 0) {
    BufferAddrMatch res;
    if (addr_id == kInvalidSsa || addr_id >= mod.nodes.size() || depth > 20) return res;

    std::string direct_key;
    if (try_get_buffer_base_key(mod, addr_id, ptr_res, &direct_key)) {
        res.matched = true;
        res.buf_key = std::move(direct_key);
        res.index_ssa = kInvalidSsa;
        res.stride_bytes = 0;
        res.field_offset = 0;
        return res;
    }

    const auto& n = mod.get(addr_id);
    if (n.op == SsaOp::Copy && n.args.size() == 1) {
        return match_buffer_affine_addr(mod, n.args[0], ptr_res, depth + 1);
    }

    if ((n.op == SsaOp::IAdd64 || n.op == SsaOp::ISub64) && n.args.size() == 2) {
        BufferAddrMatch lhs = match_buffer_affine_addr(mod, n.args[0], ptr_res, depth + 1);
        if (lhs.matched) {
            AffineTerm off = match_affine_offset(mod, n.args[1]);
            if (off.valid) {
                if (n.op == SsaOp::IAdd64) {
                    if (lhs.index_ssa == kInvalidSsa) {
                        lhs.index_ssa = off.index_ssa;
                        lhs.stride_bytes = off.stride_bytes;
                        lhs.field_offset += off.const_offset;
                        return lhs;
                    }
                    if (off.index_ssa == kInvalidSsa) {
                        lhs.field_offset += off.const_offset;
                        return lhs;
                    }
                    if (lhs.index_ssa == off.index_ssa) {
                        lhs.stride_bytes += off.stride_bytes;
                        lhs.field_offset += off.const_offset;
                        return lhs;
                    }
                } else if (off.index_ssa == kInvalidSsa) {
                    lhs.field_offset -= off.const_offset;
                    return lhs;
                }
            }
        }
        if (n.op == SsaOp::IAdd64) {
            BufferAddrMatch rhs = match_buffer_affine_addr(mod, n.args[1], ptr_res, depth + 1);
            if (rhs.matched) {
                AffineTerm off = match_affine_offset(mod, n.args[0]);
                if (off.valid) {
                    if (rhs.index_ssa == kInvalidSsa) {
                        rhs.index_ssa = off.index_ssa;
                        rhs.stride_bytes = off.stride_bytes;
                        rhs.field_offset += off.const_offset;
                        return rhs;
                    }
                    if (off.index_ssa == kInvalidSsa) {
                        rhs.field_offset += off.const_offset;
                        return rhs;
                    }
                    if (rhs.index_ssa == off.index_ssa) {
                        rhs.stride_bytes += off.stride_bytes;
                        rhs.field_offset += off.const_offset;
                        return rhs;
                    }
                }
            }
        }
    }

    return res;
}

std::string format_sym_slot_tag(const std::string& buf_key, uint32_t stride, int64_t off) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "sym:%s[stride=%u,off=%lld]",
                  buf_key.c_str(), stride, static_cast<long long>(off));
    return std::string(buf);
}

}  // namespace

SsaPassStats run_ssa_spill_mem2reg(ValhallSsaModule& mod, const SpillMem2RegOptions& opts) {
    SsaPassStats stats;
    if (mod.blocks.empty()) return stats;

    // Discover all TLS accesses and detect whether any dynamic/unknown-offset
    // TLS access exists (which conservatively blocks promotion of aliased slots).
    bool has_unknown_tls_access = false;
    struct SlotAccess {
        uint32_t block_id = 0;
        SsaId node_id = kInvalidSsa;
        bool is_store = false;
        int64_t offset = 0;
        uint32_t elem_bytes = 4;
        ValType val_type = ValType::U32;
    };
    std::map<int64_t, uint32_t> slot_widths;
    std::set<int64_t> poisoned_slots;
    std::vector<SlotAccess> accesses;

    for (size_t b = 0; b < mod.blocks.size(); ++b) {
        for (SsaId nid : mod.blocks[b].nodes) {
            const auto& n = mod.get(nid);
            if (n.op != SsaOp::PsbLoad && n.op != SsaOp::PsbStore) continue;
            if (n.args.empty()) continue;
            TlsAddrInfo info = classify_tls_addr(mod, n.args[0], opts.tls_sideband_member);
            if (!info.is_tls) continue;
            if (!info.offset_known || info.byte_offset < 0) {
                has_unknown_tls_access = true;
                continue;
            }
            uint32_t eb = psb_elem_bytes(n);
            auto it = slot_widths.find(info.byte_offset);
            if (it == slot_widths.end()) {
                slot_widths[info.byte_offset] = eb;
            } else if (it->second != eb) {
                poisoned_slots.insert(info.byte_offset);
            }
            SlotAccess acc;
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
            accesses.push_back(acc);
        }
    }

    if (has_unknown_tls_access || accesses.empty()) {
        return stats;
    }

    // Check for overlapping slots (e.g. an 8-byte access at +0 overlapping a 4-byte access at +4).
    for (const auto& [off_a, w_a] : slot_widths) {
        for (const auto& [off_b, w_b] : slot_widths) {
            if (off_a == off_b) continue;
            if (off_a < off_b + static_cast<int64_t>(w_b) &&
                off_b < off_a + static_cast<int64_t>(w_a)) {
                poisoned_slots.insert(off_a);
                poisoned_slots.insert(off_b);
            }
        }
    }

    // Identify promotable slots (must have at least one store and no poison).
    std::set<int64_t> promotable;
    for (const auto& acc : accesses) {
        if (acc.is_store && !poisoned_slots.count(acc.offset)) {
            promotable.insert(acc.offset);
        }
    }
    if (promotable.empty()) return stats;

    // Map each PsbLoad -> its TupleExtract users (elem 0 = value, elem 1 = token).
    std::unordered_map<SsaId, std::vector<SsaId>> load_val_extracts;
    std::unordered_map<SsaId, std::vector<SsaId>> load_tok_extracts;
    for (const auto& blk : mod.blocks) {
        for (SsaId nid : blk.nodes) {
            const auto& n = mod.get(nid);
            if (n.op == SsaOp::TupleExtract && !n.args.empty()) {
                SsaId base = n.args[0];
                if (base != kInvalidSsa && mod.get(base).op == SsaOp::PsbLoad) {
                    if (n.imm_u32 == 0) load_val_extracts[base].push_back(nid);
                    else if (n.imm_u32 == 1) load_tok_extracts[base].push_back(nid);
                }
            }
        }
    }

    DominatorTree dt = build_dominator_tree(mod);
    const size_t num_blocks = mod.blocks.size();

    // Braun-style SSA construction per promotable TLS slot.
    // current_def[slot][block] = reaching SsaId at end of block.
    std::map<int64_t, std::vector<SsaId>> current_def;
    std::map<int64_t, ValType> slot_type;
    for (int64_t off : promotable) {
        current_def[off].assign(num_blocks, kInvalidSsa);
        slot_type[off] = (slot_widths[off] == 8) ? ValType::U64 : ValType::U32;
    }

    // Record per-block final store for each slot first, and collect loads to resolve.
    std::unordered_map<SsaId, int64_t> store_slot;
    std::unordered_map<SsaId, int64_t> load_slot;
    for (const auto& acc : accesses) {
        if (!promotable.count(acc.offset)) continue;
        if (acc.is_store) {
            store_slot[acc.node_id] = acc.offset;
            slot_type[acc.offset] = acc.val_type;
        } else {
            load_slot[acc.node_id] = acc.offset;
        }
    }

    // Build reachable predecessor table.
    std::vector<std::vector<uint32_t>> preds(num_blocks);
    for (uint32_t b : dt.rpo_order) {
        for (uint32_t s : get_block_succs(mod.blocks[b])) {
            if (s < num_blocks && dt.reachable[s]) {
                preds[s].push_back(b);
            }
        }
    }

    // Recursive Braun readVariableRecursive(slot, block) with cycle-breaking Phis.
    std::map<std::pair<int64_t, uint32_t>, SsaId> created_phis;
    std::function<SsaId(int64_t, uint32_t)> read_var = [&](int64_t off, uint32_t b) -> SsaId {
        if (current_def[off][b] != kInvalidSsa) {
            return current_def[off][b];
        }
        const auto& ps = preds[b];
        if (ps.empty()) {
            // Entry block with no prior store: synthesize a zero constant of the slot type.
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
        // Multiple predecessors: insert a Phi first to break cycles in loops.
        SsaNode phi{};
        phi.op = SsaOp::Phi;
        phi.type = slot_type[off];
        phi.block_id = b;
        SsaId pid = mod.add_node(std::move(phi));
        mod.blocks[b].phis.push_back(pid);
        current_def[off][b] = pid;
        created_phis[{off, b}] = pid;

        std::vector<PhiIncoming> incs;
        incs.reserve(ps.size());
        for (uint32_t p : ps) {
            incs.push_back(PhiIncoming{p, read_var(off, p)});
        }
        mod.get_mut(pid).phi_incomings = std::move(incs);
        return pid;
    };

    // Walk blocks in RPO order, tracking intra-block definitions and resolving loads.
    std::vector<std::pair<SsaId, SsaId>> replacements;
    std::vector<SsaId> promoted_loads;
    std::vector<SsaId> promoted_stores;

    // First pass in RPO: populate local end-of-block definitions for blocks with stores
    // so back-edges in loops see the store value rather than recursing infinitely.
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

    // Second pass in RPO: resolve every load using intra-block last-store or predecessor lookup.
    for (uint32_t b : dt.rpo_order) {
        std::map<int64_t, SsaId> local_def;
        for (SsaId nid : mod.blocks[b].nodes) {
            auto sit = store_slot.find(nid);
            if (sit != store_slot.end()) {
                const auto& sn = mod.get(nid);
                if (sn.args.size() >= 2 && sn.args[1] != kInvalidSsa) {
                    local_def[sit->second] = sn.args[1];
                    promoted_stores.push_back(nid);
                    // Bypass store token chain: store result -> input token (args[2]).
                    if (sn.args.size() >= 3 && sn.args[2] != kInvalidSsa) {
                        replacements.emplace_back(nid, sn.args[2]);
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
                    // Save & temporarily clear current_def[off][b] if this block also
                    // stores to `off` *later* in the same block!
                    SsaId saved = current_def[off][b];
                    current_def[off][b] = kInvalidSsa;
                    val = read_var(off, b);
                    if (saved != kInvalidSsa) {
                        current_def[off][b] = saved;
                    }
                    local_def[off] = val;
                }

                // Rewrite TupleExtract(load, 0) -> val, and TupleExtract(load, 1) -> input token.
                const auto& ln = mod.get(nid);
                auto vit = load_val_extracts.find(nid);
                if (vit != load_val_extracts.end()) {
                    for (SsaId ex_id : vit->second) {
                        replacements.emplace_back(ex_id, val);
                    }
                } else {
                    // Direct use of PsbLoad value (non-tuple form).
                    replacements.emplace_back(nid, val);
                }
                auto tit = load_tok_extracts.find(nid);
                if (tit != load_tok_extracts.end() && ln.args.size() >= 2 && ln.args[1] != kInvalidSsa) {
                    for (SsaId ex_id : tit->second) {
                        replacements.emplace_back(ex_id, ln.args[1]);
                    }
                }
                promoted_loads.push_back(nid);
            }
        }
    }

    if (promoted_loads.empty() && promoted_stores.empty()) {
        return stats;
    }

    std::vector<SsaId> remap(mod.nodes.size(), kInvalidSsa);
    for (size_t i = 0; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
    for (const auto& [old_id, new_id] : replacements) {
        if (old_id > 0 && old_id < remap.size() && new_id > 0 && new_id < remap.size()) {
            remap[old_id] = new_id;
        }
    }
    apply_value_remap(mod, remap);

    stats.loads_promoted = static_cast<uint32_t>(promoted_loads.size());
    if (opts.remove_dead_memory_ops) {
        std::vector<uint8_t> dead(mod.nodes.size(), 0);
        for (SsaId lid : promoted_loads) {
            if (lid < dead.size()) dead[lid] = 1;
            for (SsaId ex : load_val_extracts[lid]) {
                if (ex < dead.size()) dead[ex] = 1;
            }
            for (SsaId ex : load_tok_extracts[lid]) {
                if (ex < dead.size()) dead[ex] = 1;
            }
        }
        for (SsaId sid : promoted_stores) {
            if (sid < dead.size()) dead[sid] = 1;
        }
        stats.stores_eliminated = static_cast<uint32_t>(promoted_stores.size());
        stats.nodes_removed += erase_dead_nodes(mod, dead);
    }

    stats.phis_simplified += simplify_trivial_phis(mod);
    return stats;
}

SsaPassStats run_ssa_sroa(ValhallSsaModule& mod, const SroaOptions& opts) {
    SsaPassStats stats;
    if (mod.blocks.empty()) return stats;

    // Scan TLS accesses for fixed-size structs with mixed 64-bit / 32-bit field
    // accesses (e.g., 64-bit store of Pack64(lo, hi) read back as two 32-bit fields,
    // or two 32-bit stores read back as a 64-bit load).
    bool has_unknown_tls = false;
    std::set<int64_t> wide_offsets;
    std::set<int64_t> narrow_offsets;

    for (const auto& blk : mod.blocks) {
        for (SsaId nid : blk.nodes) {
            const auto& n = mod.get(nid);
            if (n.op != SsaOp::PsbLoad && n.op != SsaOp::PsbStore) continue;
            if (n.args.empty()) continue;
            TlsAddrInfo info = classify_tls_addr(mod, n.args[0], opts.tls_sideband_member);
            if (!info.is_tls) continue;
            if (!info.offset_known || info.byte_offset < 0 ||
                info.byte_offset >= static_cast<int64_t>(opts.max_struct_bytes)) {
                has_unknown_tls = true;
                continue;
            }
            uint32_t eb = psb_elem_bytes(n);
            if (eb == 8 || (n.op == SsaOp::PsbStore && is_64bit_store_val(mod, n))) {
                wide_offsets.insert(info.byte_offset);
            } else if (eb == 4) {
                narrow_offsets.insert(info.byte_offset);
            }
        }
    }

    if (has_unknown_tls) return stats;

    // Split 8-byte TLS stores/loads into two aligned 4-byte scalar field accesses
    // whenever any 4-byte access overlaps [off, off+4] (or for any 8-byte TLS struct slot).
    std::set<int64_t> split_offsets;
    for (int64_t off : wide_offsets) {
        if ((off & 3) == 0 && (narrow_offsets.count(off) || narrow_offsets.count(off + 4))) {
            split_offsets.insert(off);
        }
    }

    if (!split_offsets.empty()) {
        std::vector<SsaId> remap(mod.nodes.size(), kInvalidSsa);
        for (size_t i = 0; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
        std::vector<uint8_t> dead(mod.nodes.size(), 0);

        // Map PsbLoad -> TupleExtract(0) / TupleExtract(1)
        std::unordered_map<SsaId, std::vector<SsaId>> load_val_ex;
        std::unordered_map<SsaId, std::vector<SsaId>> load_tok_ex;
        for (const auto& blk : mod.blocks) {
            for (SsaId nid : blk.nodes) {
                const auto& n = mod.get(nid);
                if (n.op == SsaOp::TupleExtract && !n.args.empty() &&
                    mod.get(n.args[0]).op == SsaOp::PsbLoad) {
                    if (n.imm_u32 == 0) load_val_ex[n.args[0]].push_back(nid);
                    else if (n.imm_u32 == 1) load_tok_ex[n.args[0]].push_back(nid);
                }
            }
        }

        auto ensure_remap_size = [&]() {
            size_t old_sz = remap.size();
            if (mod.nodes.size() > old_sz) {
                remap.resize(mod.nodes.size());
                for (size_t i = old_sz; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
                dead.resize(mod.nodes.size(), 0);
            }
        };

        for (size_t b = 0; b < mod.blocks.size(); ++b) {
            std::vector<SsaId> orig_nodes = mod.blocks[b].nodes;
            for (SsaId nid : orig_nodes) {
                const SsaNode n = mod.get(nid);
                if (n.op != SsaOp::PsbStore && n.op != SsaOp::PsbLoad) continue;
                if (n.args.empty()) continue;
                TlsAddrInfo info = classify_tls_addr(mod, n.args[0], opts.tls_sideband_member);
                if (!info.is_tls || !info.offset_known || !split_offsets.count(info.byte_offset)) {
                    continue;
                }
                uint32_t eb = psb_elem_bytes(n);
                if (eb != 8 && !(n.op == SsaOp::PsbStore && is_64bit_store_val(mod, n))) {
                    continue;
                }

                // Synthesize `addr_hi = addr + 4`.
                SsaNode c4{};
                c4.op = SsaOp::ConstU64;
                c4.type = ValType::U64;
                c4.imm_u64 = 4;
                SsaId c4_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(c4));

                SsaNode add_hi{};
                add_hi.op = SsaOp::IAdd64;
                add_hi.type = ValType::U64;
                add_hi.args = {n.args[0], c4_id};
                SsaId addr_hi = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(add_hi));

                if (n.op == SsaOp::PsbStore && n.args.size() >= 2) {
                    SsaId val64 = n.args[1];
                    SsaId tok_in = (n.args.size() >= 3) ? n.args[2] : kInvalidSsa;
                    SsaId lo_val = kInvalidSsa;
                    SsaId hi_val = kInvalidSsa;
                    const auto& vn = mod.get(val64);
                    if (vn.op == SsaOp::Pack64 && vn.args.size() == 2) {
                        lo_val = vn.args[0];
                        hi_val = vn.args[1];
                    } else {
                        SsaNode sp{};
                        sp.op = SsaOp::Split64;
                        sp.type = ValType::TupleU32x2;
                        sp.args = {val64};
                        SsaId sp_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(sp));

                        SsaNode ex0{};
                        ex0.op = SsaOp::TupleExtract;
                        ex0.type = ValType::U32;
                        ex0.imm_u32 = 0;
                        ex0.args = {sp_id};
                        lo_val = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(ex0));

                        SsaNode ex1{};
                        ex1.op = SsaOp::TupleExtract;
                        ex1.type = ValType::U32;
                        ex1.imm_u32 = 1;
                        ex1.args = {sp_id};
                        hi_val = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(ex1));
                    }

                    SsaNode st_lo{};
                    st_lo.op = SsaOp::PsbStore;
                    st_lo.type = ValType::Token;
                    st_lo.imm_u32 = 4;
                    st_lo.is_side_effect = true;
                    st_lo.args = (tok_in != kInvalidSsa) ? std::vector<SsaId>{n.args[0], lo_val, tok_in}
                                                         : std::vector<SsaId>{n.args[0], lo_val};
                    SsaId st_lo_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(st_lo));

                    SsaNode st_hi{};
                    st_hi.op = SsaOp::PsbStore;
                    st_hi.type = ValType::Token;
                    st_hi.imm_u32 = 4;
                    st_hi.is_side_effect = true;
                    st_hi.args = {addr_hi, hi_val, st_lo_id};
                    SsaId st_hi_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(st_hi));

                    ensure_remap_size();
                    remap[nid] = st_hi_id;
                    dead[nid] = 1;
                    ++stats.structs_scalarized;
                } else if (n.op == SsaOp::PsbLoad) {
                    SsaId tok_in = (n.args.size() >= 2) ? n.args[1] : kInvalidSsa;

                    SsaNode ld_lo{};
                    ld_lo.op = SsaOp::PsbLoad;
                    ld_lo.type = ValType::TupleValToken;
                    ld_lo.imm_u32 = 4;
                    ld_lo.args = (tok_in != kInvalidSsa) ? std::vector<SsaId>{n.args[0], tok_in}
                                                         : std::vector<SsaId>{n.args[0]};
                    SsaId ld_lo_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(ld_lo));

                    SsaNode v_lo{};
                    v_lo.op = SsaOp::TupleExtract;
                    v_lo.type = ValType::U32;
                    v_lo.imm_u32 = 0;
                    v_lo.args = {ld_lo_id};
                    SsaId v_lo_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(v_lo));

                    SsaNode t_lo{};
                    t_lo.op = SsaOp::TupleExtract;
                    t_lo.type = ValType::Token;
                    t_lo.imm_u32 = 1;
                    t_lo.args = {ld_lo_id};
                    SsaId t_lo_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(t_lo));

                    SsaNode ld_hi{};
                    ld_hi.op = SsaOp::PsbLoad;
                    ld_hi.type = ValType::TupleValToken;
                    ld_hi.imm_u32 = 4;
                    ld_hi.args = {addr_hi, t_lo_id};
                    SsaId ld_hi_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(ld_hi));

                    SsaNode v_hi{};
                    v_hi.op = SsaOp::TupleExtract;
                    v_hi.type = ValType::U32;
                    v_hi.imm_u32 = 0;
                    v_hi.args = {ld_hi_id};
                    SsaId v_hi_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(v_hi));

                    SsaNode t_hi{};
                    t_hi.op = SsaOp::TupleExtract;
                    t_hi.type = ValType::Token;
                    t_hi.imm_u32 = 1;
                    t_hi.args = {ld_hi_id};
                    SsaId t_hi_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(t_hi));

                    SsaNode pk{};
                    pk.op = SsaOp::Pack64;
                    pk.type = ValType::U64;
                    pk.args = {v_lo_id, v_hi_id};
                    SsaId pk_id = insert_node_before(mod, static_cast<uint32_t>(b), nid, std::move(pk));

                    ensure_remap_size();
                    for (SsaId ex : load_val_ex[nid]) {
                        remap[ex] = pk_id;
                        dead[ex] = 1;
                    }
                    for (SsaId ex : load_tok_ex[nid]) {
                        remap[ex] = t_hi_id;
                        dead[ex] = 1;
                    }
                    if (load_val_ex[nid].empty()) {
                        remap[nid] = pk_id;
                    }
                    dead[nid] = 1;
                    ++stats.structs_scalarized;
                }
            }
        }

        if (stats.structs_scalarized > 0) {
            apply_value_remap(mod, remap);
            stats.nodes_removed += erase_dead_nodes(mod, dead);
        }
    }

    if (opts.run_mem2reg_after_split) {
        SpillMem2RegOptions mopts;
        mopts.tls_sideband_member = opts.tls_sideband_member;
        mopts.remove_dead_memory_ops = true;
        SsaPassStats mstats = run_ssa_spill_mem2reg(mod, mopts);
        stats.merge(mstats);
    }

    return stats;
}

BufferSymbolizeResult run_ssa_symbolize_buffers(ValhallSsaModule& mod,
                                                const BufferSymbolizeOptions& opts) {
    BufferSymbolizeResult result;
    if (mod.blocks.empty()) return result;

    // Collect PsbLoad TupleExtract value users for store-to-load forwarding.
    std::unordered_map<SsaId, std::vector<SsaId>> load_val_ex;
    std::unordered_map<SsaId, std::vector<SsaId>> load_tok_ex;
    for (const auto& blk : mod.blocks) {
        for (SsaId nid : blk.nodes) {
            const auto& n = mod.get(nid);
            if (n.op == SsaOp::TupleExtract && !n.args.empty() &&
                mod.get(n.args[0]).op == SsaOp::PsbLoad) {
                if (n.imm_u32 == 0) load_val_ex[n.args[0]].push_back(nid);
                else if (n.imm_u32 == 1) load_tok_ex[n.args[0]].push_back(nid);
            }
        }
    }

    struct SlotKey {
        std::string buf_key;
        SsaId index_ssa = kInvalidSsa;
        uint32_t stride_bytes = 0;
        int64_t field_offset = 0;

        bool operator<(const SlotKey& o) const {
            return std::tie(buf_key, index_ssa, stride_bytes, field_offset) <
                   std::tie(o.buf_key, o.index_ssa, o.stride_bytes, o.field_offset);
        }
    };

    std::vector<SsaId> remap(mod.nodes.size(), kInvalidSsa);
    for (size_t i = 0; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
    std::vector<uint8_t> dead(mod.nodes.size(), 0);
    bool any_forwarded = false;

    for (auto& blk : mod.blocks) {
        // Intra-block known slot values (cleared on barrier/atomic/aliasing store).
        std::map<SlotKey, SsaId> live_slots;

        for (SsaId nid : blk.nodes) {
            auto& n = mod.get_mut(nid);

            if (n.op == SsaOp::ControlBarrier || n.op == SsaOp::AtomicCmpXchg ||
                n.op == SsaOp::AtomicExchange || n.op == SsaOp::AtomicRmw) {
                live_slots.clear();
                continue;
            }
            if (n.op != SsaOp::PsbLoad && n.op != SsaOp::PsbStore) continue;
            if (n.args.empty()) continue;

            BufferAddrMatch m = match_buffer_affine_addr(mod, n.args[0], opts.ptr_analysis);
            if (!m.matched) {
                if (n.op == SsaOp::PsbStore) {
                    // Unknown store may alias any buffer slot.
                    live_slots.clear();
                }
                continue;
            }

            // Require non-negative field offset and either constant-offset or fixed-stride access.
            if (m.field_offset < 0) {
                if (n.op == SsaOp::PsbStore) live_slots.clear();
                continue;
            }

            std::string sym_tag = format_sym_slot_tag(m.buf_key, m.stride_bytes, m.field_offset);
            SymbolizedBufferAccess acc;
            acc.node_id = nid;
            acc.is_store = (n.op == SsaOp::PsbStore);
            acc.buf_key = m.buf_key;
            acc.index_ssa = m.index_ssa;
            acc.stride_bytes = m.stride_bytes;
            acc.field_offset = m.field_offset;
            acc.elem_bytes = psb_elem_bytes(n);
            acc.sym_slot_tag = sym_tag;
            result.accesses.push_back(acc);
            ++result.stats.buffer_accesses_symbolized;

            if (opts.annotate_ssa_nodes) {
                n.aux_str = sym_tag;
                n.imm_u32_2 = m.stride_bytes;
                n.aux_index = static_cast<uint32_t>(m.field_offset);
                n.aux_bool = true;
            }

            SlotKey sk{m.buf_key, m.index_ssa, m.stride_bytes, m.field_offset};
            if (n.op == SsaOp::PsbStore) {
                // Invalidate any cached slots in the same buffer whose index differs or whose
                // stride/offset could alias.
                for (auto it = live_slots.begin(); it != live_slots.end();) {
                    if (it->first.buf_key == m.buf_key &&
                        (it->first.index_ssa != m.index_ssa ||
                         it->first.stride_bytes != m.stride_bytes ||
                         it->first.field_offset == m.field_offset)) {
                        it = live_slots.erase(it);
                    } else {
                        ++it;
                    }
                }
                if (opts.forward_store_to_load && n.args.size() >= 2 && n.args[1] != kInvalidSsa) {
                    live_slots[sk] = resolve_remap(remap, n.args[1]);
                }
            } else if (opts.forward_store_to_load) {
                auto it = live_slots.find(sk);
                if (it != live_slots.end()) {
                    SsaId fwd_val = it->second;
                    auto vit = load_val_ex.find(nid);
                    if (vit != load_val_ex.end()) {
                        for (SsaId ex : vit->second) {
                            remap[ex] = fwd_val;
                            dead[ex] = 1;
                        }
                    } else {
                        remap[nid] = fwd_val;
                    }
                    auto tit = load_tok_ex.find(nid);
                    if (tit != load_tok_ex.end() && n.args.size() >= 2 && n.args[1] != kInvalidSsa) {
                        for (SsaId ex : tit->second) {
                            remap[ex] = n.args[1];
                            dead[ex] = 1;
                        }
                    }
                    dead[nid] = 1;
                    any_forwarded = true;
                    ++result.stats.loads_promoted;
                } else {
                    // Cache loaded value for redundant subsequent reloads of the same slot.
                    auto vit = load_val_ex.find(nid);
                    if (vit != load_val_ex.end() && !vit->second.empty()) {
                        live_slots[sk] = vit->second.front();
                    }
                }
            }
        }
    }

    if (any_forwarded) {
        apply_value_remap(mod, remap);
        result.stats.nodes_removed += erase_dead_nodes(mod, dead);
    }

    return result;
}

}  // namespace valhall::dataflow
