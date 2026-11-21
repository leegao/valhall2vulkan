#include "ssa_fact_prover.h"

#include "transfer.h"
#include "worklist.h"

#include <unordered_map>

namespace valhall::dataflow {

namespace {

using compiler::kInvalidSsa;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValhallSsaModule;
using compiler::ValType;
using tables::SsaOp;

bool is_subgroup_lane_id_node(const SsaNode& n) {
    return n.op == SsaOp::LoadBuiltinInput &&
           (n.aux_str == "gl_SubgroupInvocationID" || n.imm_u32 == 0x29u);
}

bool is_const_u32_val(const ValhallSsaModule& mod, SsaId id, uint32_t* out) {
    if (id == kInvalidSsa || id >= mod.nodes.size()) return false;
    const auto& n = mod.get(id);
    if (n.op == SsaOp::ConstU32 || n.op == SsaOp::ConstS32) {
        if (out) *out = n.imm_u32;
        return true;
    }
    return false;
}

/// Forward `KnownBitsDomain` transfer visitor built directly on `SsaDomainTransferVisitor`.
class KnownBitsTransferVisitor : public SsaDomainTransferVisitor<KnownBitsDomain> {
public:
    using Base = SsaDomainTransferVisitor<KnownBitsDomain>;

    KnownBitsTransferVisitor(const ValhallSsaModule& mod,
                             State* st,
                             bool block_lane_id_is_zero)
        : Base(st), mod_(mod), block_lane_id_is_zero_(block_lane_id_is_zero) {}

    void on_const(SsaId dst, const SsaNode& node) override {
        if (node.op == SsaOp::ConstU32 || node.op == SsaOp::ConstS32) {
            assign(dst, KnownBitsDomain::make_const_u64(node.imm_u32));
        } else if (node.op == SsaOp::ConstU64) {
            assign(dst, KnownBitsDomain::make_const_u64(node.imm_u64));
        } else if (node.op == SsaOp::ConstBool) {
            assign(dst, KnownBitsDomain::make_const_u64(node.imm_u32 ? 1u : 0u));
        } else {
            assign(dst, KnownBitsDomain::top());
        }
    }

    void on_load_builtin(SsaId dst, uint32_t builtin_id, const SsaNode& node) override {
        if (is_subgroup_lane_id_node(node)) {
            if (block_lane_id_is_zero_) {
                assign(dst, KnownBitsDomain::make_const_u64(0u));
            } else {
                assign(dst, KnownBitsDomain::make_u32_bound(15u));
            }
            return;
        }
        (void)builtin_id;
        assign(dst, KnownBitsDomain::top());
    }

    void on_other(SsaId dst, const SsaNode& node) override {
        if (node.op == SsaOp::BfeU && node.args.size() >= 3) {
            assign(dst, KnownBitsDomain::bfe_u(eval(node.args[0]),
                                               eval(node.args[1]),
                                               eval(node.args[2])));
            return;
        }
        if (node.op == SsaOp::Bfi && node.args.size() >= 4) {
            assign(dst, KnownBitsDomain::bfi(eval(node.args[0]),
                                             eval(node.args[1]),
                                             eval(node.args[2]),
                                             eval(node.args[3])));
            return;
        }
        if (node.op == SsaOp::SubgroupBallot) {
            assign(dst, KnownBitsDomain::make_u32_bound(0xFFFFu));
            return;
        }
        if (node.op == SsaOp::Bitcast && !node.args.empty()) {
            assign(dst, eval(node.args[0]));
            return;
        }
        if (node.op == SsaOp::Select && node.args.size() >= 3) {
            assign(dst, KnownBitsDomain::join(eval(node.args[1]), eval(node.args[2])));
            return;
        }
        (void)mod_;
        assign(dst, KnownBitsDomain::top());
    }

private:
    const ValhallSsaModule& mod_;
    bool block_lane_id_is_zero_ = false;
};

/// Computes blocks where `gl_SubgroupInvocationID == 0` is guaranteed on every incoming edge
/// (e.g. the fallthrough block of `BRANCH lane_id.w0, skip_bb`).
std::vector<uint8_t> compute_lane_id_zero_blocks(const ValhallSsaModule& mod) {
    const size_t nblks = mod.blocks.size();
    std::vector<uint8_t> is_zero(nblks, 0);
    if (nblks == 0) return is_zero;

    // Build incoming edge predicates: (pred_bb, succ_bb) -> does edge guarantee lane_id == 0?
    std::vector<std::vector<bool>> edge_zero(nblks);
    for (size_t b = 0; b < nblks; ++b) {
        const auto& blk = mod.blocks[b];
        if (!blk.is_conditional_branch || blk.branch_cond == kInvalidSsa ||
            blk.branch_cond >= mod.nodes.size()) {
            continue;
        }
        const auto& cond = mod.get(blk.branch_cond);
        bool is_ne = (cond.op == SsaOp::ICmpNe);
        bool is_eq = (cond.op == SsaOp::ICmpEq);
        if ((!is_ne && !is_eq) || cond.args.size() < 2) continue;

        auto checks_lane_zero = [&](SsaId a, SsaId z) -> bool {
            uint32_t cv = 1;
            return a < mod.nodes.size() && is_subgroup_lane_id_node(mod.get(a)) &&
                   is_const_u32_val(mod, z, &cv) && cv == 0u;
        };
        if (!checks_lane_zero(cond.args[0], cond.args[1]) &&
            !checks_lane_zero(cond.args[1], cond.args[0])) {
            continue;
        }
        int32_t zero_succ = is_ne ? blk.fallthrough_block : blk.branch_target_block;
        if (zero_succ >= 0 && static_cast<size_t>(zero_succ) < nblks && mod.blocks[zero_succ].preds.size() == 1 &&
            mod.blocks[zero_succ].preds[0] == b) {
            is_zero[zero_succ] = 1;
        }
    }
    return is_zero;
}

}  // namespace

FactProverResult run_ssa_fact_prover(ValhallSsaModule& mod,
                                     const FactProverOptions& opts) {
    FactProverResult result;
    if (mod.blocks.empty()) return result;

    const std::vector<uint8_t> lane_zero_blks = opts.enable_path_sensitive_refinement
        ? compute_lane_id_zero_blocks(mod)
        : std::vector<uint8_t>(mod.blocks.size(), 0);

    // 1. Forward KnownBitsDomain fixpoint pass via `SsaDomainTransferVisitor` + `dispatch_ssa_transfer`.
    AbstractState<KnownBitsDomain> st(mod.nodes.size());
    for (size_t b = 0; b < mod.blocks.size(); ++b) {
        KnownBitsTransferVisitor visitor(mod, &st, lane_zero_blks[b] != 0);
        for (SsaId nid : mod.blocks[b].nodes) {
            dispatch_ssa_transfer(mod.get(nid), visitor);
        }
    }

    result.known_bits.resize(mod.nodes.size(), KnownBitsDomain::top());
    for (size_t i = 0; i < mod.nodes.size(); ++i) {
        result.known_bits[i] = st.val(static_cast<SsaId>(i));
    }

    // 2. Algebraic & extensional-equality discharge pass.
    std::vector<SsaId> remap(mod.nodes.size(), kInvalidSsa);
    for (size_t i = 0; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
    std::vector<uint8_t> dead(mod.nodes.size(), 0);
    bool any_remapped = false;

    for (size_t b = 0; b < mod.blocks.size(); ++b) {
        bool lane_zero = (lane_zero_blks[b] != 0);
        for (SsaId nid : mod.blocks[b].nodes) {
            auto& n = mod.get_mut(nid);

            // Rule A: Path-sensitive constant folding inside refined blocks (e.g. lane_id == 0 -> 16 | 0 = 16).
            if (lane_zero && n.op != SsaOp::ConstU32 && n.op != SsaOp::ConstS32 &&
                n.op != SsaOp::ConstBool && n.op != SsaOp::ConstF32 &&
                (n.type == ValType::U32 || n.type == ValType::S32) &&
                (is_subgroup_lane_id_node(n) || n.op == SsaOp::BitOr || n.op == SsaOp::BitAnd ||
                 n.op == SsaOp::BitXor || n.op == SsaOp::IAdd || n.op == SsaOp::ISub ||
                 n.op == SsaOp::Copy)) {
                uint64_t cv = 0;
                if (KnownBitsDomain::try_as_const_u64(result.known_bits[nid], &cv)) {
                    n.op = (n.type == ValType::S32) ? SsaOp::ConstS32 : SsaOp::ConstU32;
                    n.imm_u32 = static_cast<uint32_t>(cv);
                    n.args.clear();
                    n.aux_str.clear();
                    n.effects = {};
                    ++result.path_consts_folded;
                    ++result.stats.nodes_folded;
                    continue;
                }
            }

            // Rule A2: Fold ICmpEq / ICmpNe on constants inside path-refined blocks & collapse Select.
            if (lane_zero && (n.op == SsaOp::ICmpEq || n.op == SsaOp::ICmpNe) && n.args.size() >= 2) {
                SsaId a0 = resolve_remap(remap, n.args[0]);
                SsaId a1 = resolve_remap(remap, n.args[1]);
                uint32_t c0 = 0, c1 = 0;
                if (is_const_u32_val(mod, a0, &c0) && is_const_u32_val(mod, a1, &c1)) {
                    bool eq = (n.op == SsaOp::ICmpEq) ? (c0 == c1) : (c0 != c1);
                    n.op = SsaOp::ConstBool;
                    n.type = ValType::Bool;
                    n.imm_u32 = eq ? 1u : 0u;
                    n.args.clear();
                    ++result.path_consts_folded;
                    ++result.stats.nodes_folded;
                    continue;
                }
            }

            if (n.op == SsaOp::Select && n.args.size() >= 3) {
                SsaId c = resolve_remap(remap, n.args[0]);
                if (c < mod.nodes.size() && mod.get(c).op == SsaOp::ConstBool) {
                    SsaId chosen = resolve_remap(remap, mod.get(c).imm_u32 ? n.args[1] : n.args[2]);
                    remap[nid] = chosen;
                    dead[nid] = 1;
                    any_remapped = true;
                    ++result.csel_ladders_collapsed;
                    ++result.stats.nodes_replaced;
                    continue;
                }
            }

            // Rule B: Eliminate identity & round-trip Bitcast<T>(Bitcast<U>(x)) -> x.
            if (opts.eliminate_roundtrip_bitcasts && n.op == SsaOp::Bitcast && !n.args.empty()) {
                SsaId src = resolve_remap(remap, n.args[0]);
                if (src < mod.nodes.size()) {
                    const auto& sn = mod.get(src);
                    if (sn.type == n.type) {
                        remap[nid] = src;
                        dead[nid] = 1;
                        any_remapped = true;
                        ++result.bitcasts_eliminated;
                        ++result.stats.nodes_replaced;
                        continue;
                    }
                    if (sn.op == SsaOp::Bitcast && !sn.args.empty()) {
                        SsaId inner = resolve_remap(remap, sn.args[0]);
                        if (inner < mod.nodes.size() && mod.get(inner).type == n.type) {
                            remap[nid] = inner;
                            dead[nid] = 1;
                            any_remapped = true;
                            ++result.bitcasts_eliminated;
                            ++result.stats.nodes_replaced;
                            continue;
                        }
                    }
                }
            }

            // Rule C: Canonicalize FCmpUnordNe(x, x) -> IsNan(x).
            if (opts.canonicalize_isnan && n.op == SsaOp::FCmpUnordNe && n.args.size() >= 2) {
                SsaId a0 = resolve_remap(remap, n.args[0]);
                SsaId a1 = resolve_remap(remap, n.args[1]);
                if (a0 == a1) {
                    n.op = SsaOp::IsNan;
                    n.args = {a0};
                    ++result.isnan_canonicalized;
                    ++result.stats.nodes_replaced;
                    continue;
                }
            }

            // Rule D: Unwrap Valhall boolean masks:
            //   ICmpNe(Select(p, c_nz, 0), 0) -> p
            //   ICmpEq(Select(p, 0xFFFFFFFF, 0), 0xFFFFFFFF) -> p
            //   ICmpEq(Select(p, c_nz, 0), 0) -> LogicalNot(p)
            if (opts.unwrap_bool_masks && (n.op == SsaOp::ICmpNe || n.op == SsaOp::ICmpEq) &&
                n.args.size() >= 2) {
                SsaId a0 = resolve_remap(remap, n.args[0]);
                SsaId a1 = resolve_remap(remap, n.args[1]);
                uint32_t rhs_c = 0;
                if (a0 < mod.nodes.size() && mod.get(a0).op == SsaOp::Select &&
                    mod.get(a0).args.size() >= 3 && is_const_u32_val(mod, a1, &rhs_c)) {
                    const auto& sel = mod.get(a0);
                    uint32_t tc = 0, fc = 0;
                    if (is_const_u32_val(mod, resolve_remap(remap, sel.args[1]), &tc) &&
                        is_const_u32_val(mod, resolve_remap(remap, sel.args[2]), &fc) &&
                        tc != 0u && fc == 0u) {
                        SsaId pred = resolve_remap(remap, sel.args[0]);
                        if ((n.op == SsaOp::ICmpNe && rhs_c == 0u) ||
                            (n.op == SsaOp::ICmpEq && rhs_c == tc)) {
                            remap[nid] = pred;
                            dead[nid] = 1;
                            any_remapped = true;
                            ++result.bool_masks_unwrapped;
                            ++result.stats.nodes_replaced;
                            continue;
                        }
                        if (n.op == SsaOp::ICmpEq && rhs_c == 0u) {
                            n.op = SsaOp::LogicalNot;
                            n.args = {pred};
                            ++result.bool_masks_unwrapped;
                            ++result.stats.nodes_replaced;
                            continue;
                        }
                    }
                }
            }

            // Rule E: BfeU(Bfi(base, ins, 16, 16), 0/16, 16) -> base / ins when range-safe.
            if (opts.simplify_bfi_bfe_16x2 && n.op == SsaOp::BfeU && n.args.size() >= 3) {
                SsaId src = resolve_remap(remap, n.args[0]);
                uint32_t off = 0, cnt = 0;
                if (src < mod.nodes.size() && mod.get(src).op == SsaOp::Bfi &&
                    mod.get(src).args.size() >= 4 &&
                    is_const_u32_val(mod, resolve_remap(remap, n.args[1]), &off) &&
                    is_const_u32_val(mod, resolve_remap(remap, n.args[2]), &cnt) && cnt == 16u) {
                    const auto& bfi = mod.get(src);
                    uint32_t boff = 0, bcnt = 0;
                    if (is_const_u32_val(mod, resolve_remap(remap, bfi.args[2]), &boff) &&
                        is_const_u32_val(mod, resolve_remap(remap, bfi.args[3]), &bcnt) &&
                        boff == 16u && bcnt == 16u) {
                        SsaId candidate = kInvalidSsa;
                        if (off == 0u) candidate = resolve_remap(remap, bfi.args[0]);
                        else if (off == 16u) candidate = resolve_remap(remap, bfi.args[1]);
                        if (candidate != kInvalidSsa &&
                            (opts.assume_valhall_tex16_range ||
                             (candidate < result.known_bits.size() &&
                              result.known_bits[candidate].upper16_zero()))) {
                            remap[nid] = candidate;
                            dead[nid] = 1;
                            any_remapped = true;
                            ++result.bfi_bfe_projected;
                            ++result.stats.nodes_replaced;
                            continue;
                        }
                    }
                }
            }
        }
    }

    if (any_remapped) {
        apply_value_remap(mod, remap);
        result.stats.nodes_removed += erase_dead_nodes(mod, dead);
    }

    return result;
}

}  // namespace valhall::dataflow
