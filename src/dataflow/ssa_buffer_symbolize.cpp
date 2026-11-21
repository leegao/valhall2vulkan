#include "ssa_buffer_symbolize.h"

#include "ssa_spill_mem2reg.h"

#include <cstdio>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace valhall::dataflow {

namespace {

using compiler::kInvalidSsa;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValhallSsaModule;
using tables::SsaOp;

struct AffineTerm {
    bool valid = false;
    SsaId index_ssa = kInvalidSsa;
    uint32_t stride_bytes = 0;
    int64_t const_offset = 0;
};

AffineTerm match_affine_offset(const ValhallSsaModule& mod,
                               SsaId id,
                               const PointerAnalysisResult& pa,
                               uint32_t depth = 0) {
    AffineTerm res;
    if (id == kInvalidSsa || id >= mod.nodes.size() || depth > 16) return res;

    uint64_t cv = 0;
    if (TrackedValDomain::try_as_const_u64(pa.val(id), &cv)) {
        res.valid = true;
        res.index_ssa = kInvalidSsa;
        res.stride_bytes = 0;
        res.const_offset = static_cast<int64_t>(cv);
        return res;
    }

    const auto& n = mod.get(id);
    if ((n.op == SsaOp::Copy || n.op == SsaOp::UConvert32To64 || n.op == SsaOp::SConvert32To64) &&
        n.args.size() == 1) {
        return match_affine_offset(mod, n.args[0], pa, depth + 1);
    }

    if ((n.op == SsaOp::IAdd || n.op == SsaOp::IAdd64) && n.args.size() == 2) {
        AffineTerm a = match_affine_offset(mod, n.args[0], pa, depth + 1);
        AffineTerm b = match_affine_offset(mod, n.args[1], pa, depth + 1);
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
        AffineTerm a = match_affine_offset(mod, n.args[0], pa, depth + 1);
        AffineTerm b = match_affine_offset(mod, n.args[1], pa, depth + 1);
        if (a.valid && b.valid && b.index_ssa == kInvalidSsa) {
            a.const_offset -= b.const_offset;
            return a;
        }
    }

    if ((n.op == SsaOp::IMul || n.op == SsaOp::IMul64) && n.args.size() == 2) {
        AffineTerm a = match_affine_offset(mod, n.args[0], pa, depth + 1);
        AffineTerm b = match_affine_offset(mod, n.args[1], pa, depth + 1);
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
        uint64_t sh64 = 0;
        if (TrackedValDomain::try_as_const_u64(pa.val(n.args[1]), &sh64) && sh64 <= 16) {
            AffineTerm a = match_affine_offset(mod, n.args[0], pa, depth + 1);
            if (a.valid) {
                uint32_t mul = 1u << static_cast<uint32_t>(sh64);
                a.stride_bytes *= mul;
                a.const_offset *= static_cast<int64_t>(mul);
                return a;
            }
        }
    }

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
                             const PointerAnalysisResult& pa,
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
    const TrackedVal tv = pa.val(id);
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
    return false;
}

BufferAddrMatch match_buffer_affine_addr(const ValhallSsaModule& mod,
                                         SsaId addr_id,
                                         const PointerAnalysisResult& pa,
                                         uint32_t depth = 0) {
    BufferAddrMatch res;
    if (addr_id == kInvalidSsa || addr_id >= mod.nodes.size() || depth > 20) return res;

    std::string direct_key;
    if (try_get_buffer_base_key(mod, addr_id, pa, &direct_key)) {
        res.matched = true;
        res.buf_key = std::move(direct_key);
        res.index_ssa = kInvalidSsa;
        res.stride_bytes = 0;
        res.field_offset = 0;
        return res;
    }

    const auto& n = mod.get(addr_id);
    if (n.op == SsaOp::Copy && n.args.size() == 1) {
        return match_buffer_affine_addr(mod, n.args[0], pa, depth + 1);
    }

    if ((n.op == SsaOp::IAdd64 || n.op == SsaOp::ISub64) && n.args.size() == 2) {
        BufferAddrMatch lhs = match_buffer_affine_addr(mod, n.args[0], pa, depth + 1);
        if (lhs.matched) {
            AffineTerm off = match_affine_offset(mod, n.args[1], pa);
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
            BufferAddrMatch rhs = match_buffer_affine_addr(mod, n.args[1], pa, depth + 1);
            if (rhs.matched) {
                AffineTerm off = match_affine_offset(mod, n.args[0], pa);
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

BufferSymbolizeResult run_ssa_symbolize_buffers(ValhallSsaModule& mod,
                                                const BufferSymbolizeOptions& opts) {
    BufferSymbolizeResult result;
    if (mod.blocks.empty()) return result;

    const PointerAnalysisResult& pa = opts.ptr_analysis
        ? *opts.ptr_analysis
        : get_or_run_pointer_analysis(mod);

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
        std::map<SlotKey, SsaId> live_slots;

        for (SsaId nid : blk.nodes) {
            auto& n = mod.get_mut(nid);

            if (n.op == SsaOp::LoadFau) {
                uint64_t cv = 0;
                if (TrackedValDomain::try_as_const_u64(pa.val(nid), &cv)) {
                    n.op = SsaOp::ConstU32;
                    n.type = compiler::ValType::U32;
                    n.imm_u32 = static_cast<uint32_t>(cv);
                    n.imm_u32_2 = 0;
                    n.args.clear();
                    n.aux_str.clear();
                    n.effects = {};
                    ++result.stats.nodes_folded;
                }
                continue;
            }
            if (n.op == SsaOp::ControlBarrier || n.op == SsaOp::AtomicCmpXchg ||
                n.op == SsaOp::AtomicExchange || n.op == SsaOp::AtomicRmw) {
                live_slots.clear();
                continue;
            }
            if (n.op != SsaOp::PsbLoad && n.op != SsaOp::PsbStore) continue;
            if (n.args.empty()) continue;

            BufferAddrMatch m = match_buffer_affine_addr(mod, n.args[0], pa);
            if (!m.matched) {
                if (n.op == SsaOp::PsbStore) {
                    live_slots.clear();
                }
                continue;
            }

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
