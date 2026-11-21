#include "ssa_defau_devirt.h"

#include <cstring>
#include <unordered_map>

namespace valhall::dataflow {

namespace {

using compiler::kInvalidSsa;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValhallSsaModule;
using compiler::ValType;
using tables::SsaOp;

bool is_float_op(SsaOp op) {
    switch (op) {
        case SsaOp::FAdd:
        case SsaOp::FSub:
        case SsaOp::FMul:
        case SsaOp::FDiv:
        case SsaOp::FNeg:
        case SsaOp::FAbs:
        case SsaOp::FMin:
        case SsaOp::FMax:
        case SsaOp::FCmpOrdEq:
        case SsaOp::FCmpOrdNe:
        case SsaOp::FCmpOrdLt:
        case SsaOp::FCmpOrdLe:
        case SsaOp::FCmpOrdGt:
        case SsaOp::FCmpOrdGe:
        case SsaOp::FCmpUnordNe:
            return true;
        default:
            return false;
    }
}

bool is_bool_producing_op(SsaOp op) {
    switch (op) {
        case SsaOp::FCmpOrdEq:
        case SsaOp::FCmpOrdNe:
        case SsaOp::FCmpOrdLt:
        case SsaOp::FCmpOrdLe:
        case SsaOp::FCmpOrdGt:
        case SsaOp::FCmpOrdGe:
        case SsaOp::FCmpUnordNe:
        case SsaOp::ICmpEq:
        case SsaOp::ICmpNe:
        case SsaOp::ICmpSLt:
        case SsaOp::ICmpSLe:
        case SsaOp::ICmpSGt:
        case SsaOp::ICmpSGe:
        case SsaOp::ICmpULt:
        case SsaOp::ICmpULe:
        case SsaOp::ICmpUGt:
        case SsaOp::ICmpUGe:
        case SsaOp::LogicalAnd:
        case SsaOp::LogicalOr:
        case SsaOp::LogicalNot:
            return true;
        default:
            return false;
    }
}

bool is_pure_unary_sym_op(SsaOp op) {
    switch (op) {
        case SsaOp::FNeg:
        case SsaOp::FAbs:
        case SsaOp::BitNot:
        case SsaOp::LogicalNot:
        case SsaOp::SNeg:
            return true;
        default:
            return false;
    }
}

bool is_pure_binary_sym_op(SsaOp op) {
    switch (op) {
        case SsaOp::FAdd:
        case SsaOp::FSub:
        case SsaOp::FMul:
        case SsaOp::FDiv:
        case SsaOp::FMin:
        case SsaOp::FMax:
        case SsaOp::IAdd:
        case SsaOp::ISub:
        case SsaOp::IMul:
        case SsaOp::BitAnd:
        case SsaOp::BitOr:
        case SsaOp::BitXor:
        case SsaOp::Shl:
        case SsaOp::ShrU:
        case SsaOp::ShrS:
        case SsaOp::FCmpOrdEq:
        case SsaOp::FCmpOrdNe:
        case SsaOp::FCmpOrdLt:
        case SsaOp::FCmpOrdLe:
        case SsaOp::FCmpOrdGt:
        case SsaOp::FCmpOrdGe:
        case SsaOp::ICmpEq:
        case SsaOp::ICmpNe:
        case SsaOp::ICmpSLt:
        case SsaOp::ICmpSLe:
        case SsaOp::ICmpSGt:
        case SsaOp::ICmpSGe:
        case SsaOp::ICmpULt:
        case SsaOp::ICmpULe:
        case SsaOp::ICmpUGt:
        case SsaOp::ICmpUGe:
        case SsaOp::LogicalAnd:
        case SsaOp::LogicalOr:
            return true;
        default:
            return false;
    }
}

std::string canonical_buf_key(int32_t rack, int32_t desc_idx, const std::string& sym) {
    if (!sym.empty() && (sym.rfind("ubo:", 0) == 0 || sym.rfind("ssbo:", 0) == 0)) {
        return sym;
    }
    if (!sym.empty()) return sym;
    return "buf:" + std::to_string(rack) + ":" + std::to_string(desc_idx);
}

bool try_extract_buffer_addr(const TrackedVal& tv,
                             int32_t* out_rack,
                             int32_t* out_desc,
                             std::string* out_sym,
                             int64_t* out_off,
                             bool* out_off_known) {
    if (const auto* pv = tv.inner.get_if<PointerDomain>();
        pv && pv->is_valid_pointer() && pv->kind == PointerVal::Kind::BufferAddr) {
        *out_rack = pv->rack;
        *out_desc = pv->desc_idx;
        *out_sym = pv->symbol_name;
        *out_off = pv->byte_offset;
        *out_off_known = pv->offset_known;
        return true;
    }
    return false;
}

bool try_extract_fau_dst_offset(const TrackedVal& tv, uint32_t* out_fau_off) {
    if (const auto* pv = tv.inner.get_if<PointerDomain>();
        pv && pv->is_valid_pointer() && pv->kind == PointerVal::Kind::FauDstBase &&
        pv->offset_known && pv->byte_offset >= 0) {
        *out_fau_off = static_cast<uint32_t>(pv->byte_offset);
        return true;
    }
    return false;
}

}  // namespace

FauSummaryMap run_pilot_fau_summary(const ValhallSsaModule& pilot_ssa,
                                    const PointerAnalysisEnv& pilot_env) {
    FauSummaryMap summary;
    if (pilot_ssa.blocks.empty()) return summary;

    PointerAnalysisResult pa = run_ssa_pointer_analysis(pilot_ssa, pilot_env);
    std::vector<std::shared_ptr<const SymExpr>> exprs(pilot_ssa.nodes.size());
    std::vector<std::vector<std::shared_ptr<const SymExpr>>> vec_exprs(pilot_ssa.nodes.size());

    auto get_expr = [&](SsaId id) -> std::shared_ptr<const SymExpr> {
        if (id == kInvalidSsa || id >= exprs.size()) return nullptr;
        return exprs[id];
    };

    for (const auto& blk : pilot_ssa.blocks) {
        for (SsaId nid : blk.nodes) {
            const auto& n = pilot_ssa.get(nid);
            uint64_t cv = 0;

            if (n.op == SsaOp::ConstU32 || n.op == SsaOp::ConstF32 || n.op == SsaOp::ConstS32) {
                exprs[nid] = SymExpr::make_const_u32(n.imm_u32);
                continue;
            }
            if (n.op == SsaOp::ConstBool) {
                exprs[nid] = SymExpr::make_const_u32(n.imm_u32 ? 1u : 0u);
                continue;
            }
            if (TrackedValDomain::try_as_const_u64(pa.val(nid), &cv)) {
                exprs[nid] = SymExpr::make_const_u32(static_cast<uint32_t>(cv));
                continue;
            }

            int32_t rack = -1, desc = -1;
            std::string sym;
            int64_t off = 0;
            bool off_known = false;
            if (try_extract_buffer_addr(pa.val(nid), &rack, &desc, &sym, &off, &off_known)) {
                exprs[nid] = SymExpr::make_buffer_addr(rack, desc, sym, off, off_known);
                continue;
            }

            if ((n.op == SsaOp::Copy || n.op == SsaOp::Bitcast) && !n.args.empty()) {
                exprs[nid] = get_expr(n.args[0]);
                if (n.args[0] < vec_exprs.size() && !vec_exprs[n.args[0]].empty()) {
                    vec_exprs[nid] = vec_exprs[n.args[0]];
                }
                continue;
            }

            if (n.op == SsaOp::TupleExtract && !n.args.empty() && n.imm_u32 == 0u) {
                exprs[nid] = get_expr(n.args[0]);
                if (n.args[0] < vec_exprs.size() && !vec_exprs[n.args[0]].empty()) {
                    vec_exprs[nid] = vec_exprs[n.args[0]];
                }
                continue;
            }

            if (n.op == SsaOp::PsbLoad && !n.args.empty()) {
                if (try_extract_buffer_addr(pa.val(n.args[0]), &rack, &desc, &sym, &off, &off_known) &&
                    off_known && off >= 0) {
                    uint32_t bytes = (n.imm_u32 > 0) ? n.imm_u32 : 4u;
                    exprs[nid] = SymExpr::make_buffer_load(rack, desc, sym, off, bytes);
                }
                continue;
            }

            if ((n.op == SsaOp::VecConstruct2 || n.op == SsaOp::VecConstruct4) && !n.args.empty()) {
                vec_exprs[nid].reserve(n.args.size());
                for (SsaId a : n.args) {
                    vec_exprs[nid].push_back(get_expr(a));
                }
                continue;
            }

            if (n.op == SsaOp::VecExtract && !n.args.empty()) {
                SsaId base = n.args[0];
                if (base < vec_exprs.size() && n.imm_u32 < vec_exprs[base].size()) {
                    exprs[nid] = vec_exprs[base][n.imm_u32];
                }
                continue;
            }

            if (n.op == SsaOp::Pack64 && n.args.size() >= 2) {
                auto lo = get_expr(n.args[0]);
                auto hi = get_expr(n.args[1]);
                if (lo && lo->kind == SymExpr::Kind::BufferAddr) {
                    exprs[nid] = lo;
                } else if (hi && hi->kind == SymExpr::Kind::BufferAddr) {
                    exprs[nid] = hi;
                }
                continue;
            }

            if (is_pure_unary_sym_op(n.op) && !n.args.empty()) {
                if (auto a = get_expr(n.args[0])) {
                    exprs[nid] = SymExpr::make_unary(n.op, std::move(a));
                }
                continue;
            }

            if (is_pure_binary_sym_op(n.op) && n.args.size() >= 2) {
                auto a = get_expr(n.args[0]);
                auto b = get_expr(n.args[1]);
                if (a && b) {
                    exprs[nid] = SymExpr::make_binary(n.op, std::move(a), std::move(b));
                }
                continue;
            }

            if (n.op == SsaOp::Select && n.args.size() >= 3) {
                auto cond = get_expr(n.args[0]);
                auto tv = get_expr(n.args[1]);
                auto fv = get_expr(n.args[2]);
                if (cond && tv && fv) {
                    exprs[nid] = SymExpr::make_select(std::move(cond), std::move(tv), std::move(fv));
                }
                continue;
            }

            if (n.op == SsaOp::PsbStore && n.args.size() >= 2) {
                uint32_t fau_off = 0;
                if (!try_extract_fau_dst_offset(pa.val(n.args[0]), &fau_off)) continue;

                SsaId val_id = n.args[1];
                if (val_id < vec_exprs.size() && !vec_exprs[val_id].empty()) {
                    for (size_t i = 0; i < vec_exprs[val_id].size(); ++i) {
                        if (vec_exprs[val_id][i]) {
                            summary[fau_off + static_cast<uint32_t>(i * 4u)] = vec_exprs[val_id][i];
                        }
                    }
                } else if (auto e = get_expr(val_id)) {
                    summary[fau_off] = e;
                }
                continue;
            }
        }
    }

    // Backfill any scalar pointer-analysis PilotFauWrites not captured by vector/scalar exprs.
    for (const auto& kv : pa.pilot_fau_writes) {
        if (summary.count(kv.first)) continue;
        int32_t rack = -1, desc = -1;
        std::string sym;
        int64_t off = 0;
        bool off_known = false;
        if (try_extract_buffer_addr(kv.second.value, &rack, &desc, &sym, &off, &off_known)) {
            summary[kv.first] = SymExpr::make_buffer_addr(rack, desc, sym, off, off_known);
        } else {
            uint64_t cv = 0;
            if (TrackedValDomain::try_as_const_u64(kv.second.value, &cv)) {
                summary[kv.first] = SymExpr::make_const_u32(static_cast<uint32_t>(cv));
            }
        }
    }

    return summary;
}

void seed_fcst_into_fau_summary(const std::map<uint32_t, uint32_t>& fcst_words,
                                FauSummaryMap& summary,
                                bool overwrite) {
    for (const auto& kv : fcst_words) {
        if (overwrite || !summary.count(kv.first)) {
            summary[kv.first] = SymExpr::make_const_u32(kv.second);
        }
    }
}

namespace {

/// Materializes `SymExpr` DAGs into `bb0` of `main_ssa` with structural memoization
/// so shared UBO loads and Pilot predicates are emitted at most once.
class SymExprMaterializer {
public:
    explicit SymExprMaterializer(ValhallSsaModule& mod) : mod_(mod) {
        entry_tok_ = find_or_emit_entry_token();
    }

    SsaId materialize_u32(const std::shared_ptr<const SymExpr>& e,
                          const DeFauDevirtOptions& opts,
                          bool* out_is_const,
                          bool* out_is_load,
                          bool* out_is_expr) {
        SsaId raw = materialize_raw(e, opts, out_is_const, out_is_load, out_is_expr);
        if (raw == kInvalidSsa) return kInvalidSsa;
        return ensure_type(raw, ValType::U32);
    }

    SsaId materialize_addr64(const std::shared_ptr<const SymExpr>& e) {
        if (!e || e->kind != SymExpr::Kind::BufferAddr) return kInvalidSsa;
        std::string key = canonical_buf_key(e->rack, e->desc_idx, e->symbol_name);
        SsaId base = get_or_emit_buf_base(key);
        if (e->offset_known && e->byte_offset != 0) {
            SsaId off_c = emit_const_u64(static_cast<uint64_t>(e->byte_offset));
            return emit_node(SsaOp::IAdd64, ValType::U64, {base, off_c});
        }
        return base;
    }

    void commit_to_entry_block() {
        if (emitted_.empty() || mod_.blocks.empty()) return;
        auto& nodes = mod_.blocks[0].nodes;
        size_t insert_pos = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i] == entry_tok_) {
                insert_pos = i + 1;
                break;
            }
        }
        nodes.insert(nodes.begin() + static_cast<std::ptrdiff_t>(insert_pos),
                     emitted_.begin(), emitted_.end());
        emitted_.clear();
    }

private:
    ValhallSsaModule& mod_;
    SsaId entry_tok_ = kInvalidSsa;
    std::vector<SsaId> emitted_;
    std::unordered_map<std::string, SsaId> memo_;
    std::unordered_map<std::string, SsaId> buf_bases_;

    SsaId find_or_emit_entry_token() {
        if (!mod_.blocks.empty()) {
            for (SsaId nid : mod_.blocks[0].nodes) {
                const auto& n = mod_.get(nid);
                if (n.op == SsaOp::Nop && n.type == ValType::Token) return nid;
            }
        }
        return emit_node(SsaOp::Nop, ValType::Token, {});
    }

    SsaId emit_node(SsaOp op, ValType ty, std::vector<SsaId> args,
                    uint32_t imm32 = 0, uint64_t imm64 = 0, std::string aux = "") {
        SsaNode n{};
        n.id = static_cast<SsaId>(mod_.nodes.size());
        n.op = op;
        n.type = ty;
        n.imm_u32 = imm32;
        n.imm_u64 = imm64;
        n.args = std::move(args);
        n.aux_str = std::move(aux);
        mod_.nodes.push_back(std::move(n));
        SsaId id = static_cast<SsaId>(mod_.nodes.size() - 1);
        emitted_.push_back(id);
        return id;
    }

    SsaId emit_const_u32(uint32_t v) {
        return emit_node(SsaOp::ConstU32, ValType::U32, {}, v);
    }

    SsaId emit_const_u64(uint64_t v) {
        return emit_node(SsaOp::ConstU64, ValType::U64, {}, 0, v);
    }

    SsaId get_or_emit_buf_base(const std::string& key) {
        auto it = buf_bases_.find(key);
        if (it != buf_bases_.end()) return it->second;
        SsaId id = emit_node(SsaOp::LoadSidebandU64, ValType::U64, {}, 0x10u, 0, key);
        buf_bases_[key] = id;
        return id;
    }

    SsaId ensure_type(SsaId id, ValType target) {
        ValType cur = mod_.get(id).type;
        if (cur == target) return id;
        if (cur == ValType::Bool && target == ValType::U32) {
            return emit_node(SsaOp::Select, ValType::U32,
                             {id, emit_const_u32(0xFFFFFFFFu), emit_const_u32(0u)});
        }
        if (cur == ValType::U32 && target == ValType::Bool) {
            return emit_node(SsaOp::ICmpNe, ValType::Bool, {id, emit_const_u32(0u)});
        }
        return emit_node(SsaOp::Bitcast, target, {id});
    }

    SsaId materialize_raw(const std::shared_ptr<const SymExpr>& e,
                          const DeFauDevirtOptions& opts,
                          bool* out_is_const,
                          bool* out_is_load,
                          bool* out_is_expr) {
        if (!e) return kInvalidSsa;
        std::string key = e->format();
        auto mit = memo_.find(key);
        if (mit != memo_.end()) return mit->second;

        SsaId res = kInvalidSsa;
        switch (e->kind) {
            case SymExpr::Kind::ConstU32:
                if (!opts.materialize_fcst_constants) return kInvalidSsa;
                res = emit_const_u32(e->const_u32);
                if (out_is_const) *out_is_const = true;
                break;

            case SymExpr::Kind::BufferScalarLoad: {
                if (!opts.materialize_pilot_loads) return kInvalidSsa;
                std::string bkey = canonical_buf_key(e->rack, e->desc_idx, e->symbol_name);
                SsaId base = get_or_emit_buf_base(bkey);
                SsaId addr = base;
                if (e->byte_offset != 0) {
                    SsaId off_c = emit_const_u64(static_cast<uint64_t>(e->byte_offset));
                    addr = emit_node(SsaOp::IAdd64, ValType::U64, {base, off_c});
                }
                SsaId ld = emit_node(SsaOp::PsbLoad, ValType::TupleValToken,
                                     {addr, entry_tok_}, e->elem_bytes, 0, bkey);
                entry_tok_ = emit_node(SsaOp::TupleExtract, ValType::Token, {ld}, 1u);
                res = emit_node(SsaOp::TupleExtract, ValType::U32, {ld}, 0u);
                if (out_is_load) *out_is_load = true;
                break;
            }

            case SymExpr::Kind::UnaryOp: {
                if (!opts.materialize_pilot_exprs) return kInvalidSsa;
                SsaId a = materialize_raw(e->a, opts, nullptr, nullptr, nullptr);
                if (a == kInvalidSsa) return kInvalidSsa;
                ValType op_ty = is_float_op(e->op) ? ValType::F32
                              : (e->op == SsaOp::LogicalNot ? ValType::Bool : ValType::U32);
                a = ensure_type(a, op_ty);
                res = emit_node(e->op, op_ty, {a});
                if (out_is_expr) *out_is_expr = true;
                break;
            }

            case SymExpr::Kind::BinaryOp: {
                if (!opts.materialize_pilot_exprs) return kInvalidSsa;
                SsaId a = materialize_raw(e->a, opts, nullptr, nullptr, nullptr);
                SsaId b = materialize_raw(e->b, opts, nullptr, nullptr, nullptr);
                if (a == kInvalidSsa || b == kInvalidSsa) return kInvalidSsa;
                ValType in_ty = is_float_op(e->op) ? ValType::F32
                              : ((e->op == SsaOp::LogicalAnd || e->op == SsaOp::LogicalOr)
                                     ? ValType::Bool
                                     : ValType::U32);
                ValType out_ty = is_bool_producing_op(e->op) ? ValType::Bool : in_ty;
                a = ensure_type(a, in_ty);
                b = ensure_type(b, in_ty);
                res = emit_node(e->op, out_ty, {a, b});
                if (out_is_expr) *out_is_expr = true;
                break;
            }

            case SymExpr::Kind::SelectOp: {
                if (!opts.materialize_pilot_exprs) return kInvalidSsa;
                SsaId c = materialize_raw(e->a, opts, nullptr, nullptr, nullptr);
                SsaId tv = materialize_raw(e->b, opts, nullptr, nullptr, nullptr);
                SsaId fv = materialize_raw(e->c, opts, nullptr, nullptr, nullptr);
                if (c == kInvalidSsa || tv == kInvalidSsa || fv == kInvalidSsa) return kInvalidSsa;
                c = ensure_type(c, ValType::Bool);
                tv = ensure_type(tv, ValType::U32);
                fv = ensure_type(fv, ValType::U32);
                res = emit_node(SsaOp::Select, ValType::U32, {c, tv, fv});
                if (out_is_expr) *out_is_expr = true;
                break;
            }

            case SymExpr::Kind::BufferAddr:
            case SymExpr::Kind::FauSlotRef:
                return kInvalidSsa;
        }

        if (res != kInvalidSsa) {
            memo_[key] = res;
        }
        return res;
    }
};

}  // namespace

DeFauDevirtResult run_ssa_defau_devirt(ValhallSsaModule& main_ssa,
                                       const FauSummaryMap& fau_summary,
                                       const DeFauDevirtOptions& opts) {
    DeFauDevirtResult result;
    if (main_ssa.blocks.empty()) return result;

    SymExprMaterializer mat(main_ssa);
    const size_t orig_node_count = main_ssa.nodes.size();
    std::vector<SsaId> remap(orig_node_count, kInvalidSsa);
    for (size_t i = 0; i < orig_node_count; ++i) remap[i] = static_cast<SsaId>(i);
    std::vector<uint8_t> dead(orig_node_count, 0);
    bool any_remapped = false;

    // 1. Rewrite 64-bit Pack64(LoadFau(lo), LoadFau(hi)) resolving to a BufferAddr.
    if (opts.materialize_buffer_addrs) {
        for (const auto& blk : main_ssa.blocks) {
            for (SsaId nid : blk.nodes) {
                if (nid >= orig_node_count) continue;
                const auto& n = main_ssa.get(nid);
                if (n.op != SsaOp::Pack64 || n.args.size() < 2) continue;
                const auto& lo = main_ssa.get(n.args[0]);
                const auto& hi = main_ssa.get(n.args[1]);
                if (lo.op != SsaOp::LoadFau || hi.op != SsaOp::LoadFau) continue;
                uint32_t lo_off = lo.imm_u32 * 4u;
                uint32_t hi_off = hi.imm_u32 * 4u;
                if (hi_off != lo_off + 4u) continue;
                auto it = fau_summary.find(lo_off);
                if (it == fau_summary.end() || !it->second ||
                    it->second->kind != SymExpr::Kind::BufferAddr) {
                    continue;
                }
                SsaId addr_id = mat.materialize_addr64(it->second);
                if (addr_id != kInvalidSsa) {
                    remap[nid] = addr_id;
                    dead[nid] = 1;
                    any_remapped = true;
                    ++result.fau_addrs_materialized;
                    ++result.stats.nodes_replaced;
                }
            }
        }
    }

    // 2. Rewrite 32-bit LoadFau(w) nodes resolving to ConstU32, BufferScalarLoad, or SymExpr.
    for (const auto& blk : main_ssa.blocks) {
        for (SsaId nid : blk.nodes) {
            if (nid >= orig_node_count || dead[nid]) continue;
            auto& n = main_ssa.get_mut(nid);
            if (n.op != SsaOp::LoadFau) continue;
            uint32_t byte_off = n.imm_u32 * 4u;

            auto it = fau_summary.find(byte_off);
            if (it == fau_summary.end() || !it->second) {
                // Skip fallback warning if this LoadFau is only the hi/lo half of a remapped BufferAddr.
                uint32_t aligned = byte_off & ~7u;
                auto pit = fau_summary.find(aligned);
                if (pit != fau_summary.end() && pit->second &&
                    pit->second->kind == SymExpr::Kind::BufferAddr) {
                    continue;
                }
                if (opts.record_fallbacks) {
                    result.fallbacks.push_back(FallbackRecord{
                        n.orig_pc, nid, n.op, FallbackReason::UnresolvedFauSlot,
                        "FAU+0x" + [&]() {
                            char b[16];
                            std::snprintf(b, sizeof(b), "%x", byte_off);
                            return std::string(b);
                        }()});
                }
                continue;
            }

            if (it->second->kind == SymExpr::Kind::BufferAddr) {
                continue;
            }

            // Fast path: rewrite ConstU32 directly in place without creating extra nodes.
            if (it->second->kind == SymExpr::Kind::ConstU32 && opts.materialize_fcst_constants) {
                n.op = SsaOp::ConstU32;
                n.type = ValType::U32;
                n.imm_u32 = it->second->const_u32;
                n.imm_u32_2 = 0;
                n.args.clear();
                n.aux_str.clear();
                n.effects = {};
                ++result.fau_consts_materialized;
                ++result.stats.nodes_folded;
                continue;
            }

            bool is_const = false, is_load = false, is_expr = false;
            SsaId repl = mat.materialize_u32(it->second, opts, &is_const, &is_load, &is_expr);
            if (repl != kInvalidSsa) {
                remap[nid] = repl;
                dead[nid] = 1;
                any_remapped = true;
                if (is_const) ++result.fau_consts_materialized;
                else if (is_load) ++result.fau_loads_materialized;
                else if (is_expr) ++result.fau_exprs_materialized;
                ++result.stats.nodes_replaced;
            }
        }
    }

    mat.commit_to_entry_block();
    if (any_remapped) {
        remap.resize(main_ssa.nodes.size(), kInvalidSsa);
        for (size_t i = orig_node_count; i < main_ssa.nodes.size(); ++i) {
            remap[i] = static_cast<SsaId>(i);
        }
        dead.resize(main_ssa.nodes.size(), 0);
        apply_value_remap(main_ssa, remap);
        result.stats.nodes_removed += erase_dead_nodes(main_ssa, dead);
    }

    return result;
}

}  // namespace valhall::dataflow
