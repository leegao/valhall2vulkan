#include "ssa_sroa.h"

#include "ssa_spill_mem2reg.h"

#include <set>
#include <unordered_map>
#include <vector>

namespace valhall::dataflow {

namespace {

using compiler::kInvalidSsa;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValhallSsaModule;
using compiler::ValType;
using tables::SsaOp;

bool is_64bit_store_val(const ValhallSsaModule& mod, const SsaNode& store_node) {
    if (psb_elem_bytes(store_node) == 8) return true;
    if (store_node.args.size() >= 2 && store_node.args[1] != kInvalidSsa) {
        ValType vt = mod.get(store_node.args[1]).type;
        if (vt == ValType::U64 || vt == ValType::S64) return true;
    }
    return false;
}

}  // namespace

SsaPassStats run_ssa_sroa(ValhallSsaModule& mod, const SroaOptions& opts) {
    SsaPassStats stats;
    if (mod.blocks.empty()) return stats;

    PointerAnalysisEnv env;
    env.tls_base_sideband_member = opts.tls_sideband_member;
    const PointerAnalysisResult pa = (opts.tls_sideband_member == 6)
        ? get_or_run_pointer_analysis(mod)
        : get_or_run_pointer_analysis(mod, env);

    bool has_unknown_tls = false;
    std::set<int64_t> wide_offsets;
    std::set<int64_t> narrow_offsets;

    for (const auto& blk : mod.blocks) {
        for (SsaId nid : blk.nodes) {
            const auto& n = mod.get(nid);
            if (n.op != SsaOp::PsbLoad && n.op != SsaOp::PsbStore) continue;
            if (n.args.empty()) continue;
            TlsAddrInfo info = classify_tls_addr(mod, n.args[0], pa);
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
                TlsAddrInfo info = classify_tls_addr(mod, n.args[0], pa);
                if (!info.is_tls || !info.offset_known || !split_offsets.count(info.byte_offset)) {
                    continue;
                }
                uint32_t eb = psb_elem_bytes(n);
                if (eb != 8 && !(n.op == SsaOp::PsbStore && is_64bit_store_val(mod, n))) {
                    continue;
                }

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

}  // namespace valhall::dataflow
