#include "ssa_fp_opt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace valhall::dataflow {

namespace {

using compiler::kInvalidSsa;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValhallSsaModule;
using compiler::ValType;
using tables::SsaOp;

constexpr uint32_t kPosZeroBits = 0x00000000u;
constexpr uint32_t kNegZeroBits = 0x80000000u;
constexpr uint32_t kPosOneBits  = 0x3F800000u;
constexpr uint32_t kNegOneBits  = 0xBF800000u;
constexpr uint32_t kPosTwoBits  = 0x40000000u;
constexpr uint32_t kPosHalfBits = 0x3F000000u;

bool is_pos_zero(uint32_t b) { return b == kPosZeroBits; }
bool is_neg_zero(uint32_t b) { return b == kNegZeroBits; }
bool is_any_zero(uint32_t b) { return (b & 0x7FFFFFFFu) == 0u; }
bool is_pos_one(uint32_t b)  { return b == kPosOneBits; }
bool is_neg_one(uint32_t b)  { return b == kNegOneBits; }
bool is_pos_two(uint32_t b)  { return b == kPosTwoBits; }
bool is_pos_half(uint32_t b) { return b == kPosHalfBits; }

/// Returns true iff `bits` is a finite, normal, non-zero exact power of 2
/// (positive or negative) whose exact reciprocal is also normal.
bool is_exact_pow2_normal(uint32_t bits, uint32_t* out_rcp_bits) {
    uint32_t mant = bits & 0x007FFFFFu;
    uint32_t exp = (bits >> 23) & 0xFFu;
    if (mant != 0 || exp <= 1 || exp >= 254) return false;
    uint32_t sign = bits & 0x80000000u;
    uint32_t rcp_exp = 254u - exp;
    if (out_rcp_bits) *out_rcp_bits = sign | (rcp_exp << 23);
    return true;
}

bool is_f32_scalar(const SsaNode& n) {
    return n.type == ValType::F32;
}

SsaId emit_f32_const_before(ValhallSsaModule& mod, uint32_t blk, SsaId before, uint32_t bits) {
    SsaNode c{};
    c.op = SsaOp::ConstF32;
    c.type = ValType::F32;
    c.imm_u32 = bits;
    return insert_node_before(mod, blk, before, std::move(c));
}

SsaId emit_fneg_before(ValhallSsaModule& mod, uint32_t blk, SsaId before, SsaId x) {
    SsaNode neg{};
    neg.op = SsaOp::FNeg;
    neg.type = ValType::F32;
    neg.args = {x};
    return insert_node_before(mod, blk, before, std::move(neg));
}

SsaId emit_fabs_before(ValhallSsaModule& mod, uint32_t blk, SsaId before, SsaId x) {
    SsaNode ab{};
    ab.op = SsaOp::FAbs;
    ab.type = ValType::F32;
    ab.args = {x};
    return insert_node_before(mod, blk, before, std::move(ab));
}

SsaOp flip_fcmp_direction(SsaOp op) {
    switch (op) {
        case SsaOp::FCmpOrdLt: return SsaOp::FCmpOrdGt;
        case SsaOp::FCmpOrdLe: return SsaOp::FCmpOrdGe;
        case SsaOp::FCmpOrdGt: return SsaOp::FCmpOrdLt;
        case SsaOp::FCmpOrdGe: return SsaOp::FCmpOrdLe;
        default: return op;
    }
}

bool is_fcmp_op(SsaOp op) {
    return op == SsaOp::FCmpOrdEq || op == SsaOp::FCmpOrdNe ||
           op == SsaOp::FCmpOrdLt || op == SsaOp::FCmpOrdLe ||
           op == SsaOp::FCmpOrdGt || op == SsaOp::FCmpOrdGe ||
           op == SsaOp::FCmpUnordNe;
}

/// Checks if `id` is `FDiv(ConstF32(1.0), x)` (Valhall `FRCP` lowering) and outputs `x`.
bool match_rcp(const ValhallSsaModule& mod, SsaId id, SsaId* out_x) {
    if (id == kInvalidSsa || id >= mod.nodes.size()) return false;
    const auto& n = mod.get(id);
    uint32_t num_bits = 0;
    if (n.op == SsaOp::FDiv && n.args.size() == 2 &&
        try_get_const_f32_bits(mod, n.args[0], &num_bits) && is_pos_one(num_bits)) {
        if (out_x) *out_x = n.args[1];
        return true;
    }
    return false;
}

/// Canonicalizes commutative FP operand order: constants move to `args[1]`,
/// otherwise lower `SsaId` moves to `args[0]`. Returns true if swapped.
bool canonicalize_fp_commutativity(ValhallSsaModule& mod, SsaNode& n) {
    auto canonical_pair = [&](SsaId& a, SsaId& b) -> bool {
        uint32_t dummy = 0;
        bool ca = try_get_const_f32_bits(mod, a, &dummy);
        bool cb = try_get_const_f32_bits(mod, b, &dummy);
        if (ca && !cb) {
            std::swap(a, b);
            return true;
        }
        if (ca == cb && b < a) {
            std::swap(a, b);
            return true;
        }
        return false;
    };

    if ((n.op == SsaOp::FAdd || n.op == SsaOp::FMul ||
         n.op == SsaOp::FMin || n.op == SsaOp::FMax ||
         n.op == SsaOp::NMin || n.op == SsaOp::NMax ||
         n.op == SsaOp::FCmpOrdEq || n.op == SsaOp::FCmpOrdNe ||
         n.op == SsaOp::FCmpUnordNe) && n.args.size() == 2) {
        return canonical_pair(n.args[0], n.args[1]);
    }
    if (n.op == SsaOp::Fma && n.args.size() == 3) {
        return canonical_pair(n.args[0], n.args[1]);
    }
    return false;
}


SsaId unwrap_copies(const ValhallSsaModule& mod, SsaId id) {
    uint32_t depth = 0;
    while (id > 0 && id < mod.nodes.size() && depth < 16) {
        const auto& n = mod.get(id);
        if (n.op == SsaOp::Copy && n.args.size() == 1 && n.args[0] != kInvalidSsa) {
            id = n.args[0];
            ++depth;
            continue;
        }
        break;
    }
    return id;
}

/// Generic structural matcher for `FusionPatternNode` trees generated from
/// `specs/lowering/valhall_tiles.json`, allowing SSA-level verification passes
/// to reuse the exact same declarative rules as the Pass-2 tiler.
bool match_tile_pattern_ssa(const ValhallSsaModule& mod,
                            const tables::FusionPatternNode* pnodes,
                            uint16_t pnode_idx,
                            SsaId ssa_id,
                            std::vector<SsaId>& captured_leaves) {
    if (ssa_id == kInvalidSsa || ssa_id >= mod.nodes.size()) return false;
    const auto& pn = pnodes[pnode_idx];
    if (pn.is_leaf) {
        SsaId unwrapped = unwrap_copies(mod, ssa_id);
        size_t slot = static_cast<size_t>(pn.leaf_slot);
        if (slot >= captured_leaves.size()) captured_leaves.resize(slot + 1, kInvalidSsa);
        if (captured_leaves[slot] != kInvalidSsa && captured_leaves[slot] != unwrapped) {
            return false;
        }
        captured_leaves[slot] = unwrapped;
        return true;
    }

    ssa_id = unwrap_copies(mod, ssa_id);
    if (ssa_id == kInvalidSsa || ssa_id >= mod.nodes.size()) return false;
    const auto& sn = mod.get(ssa_id);
    if (sn.op != pn.ssa_op) return false;
    if (pn.has_imm && sn.imm_u32 != pn.imm_val) return false;
    if (pn.has_imm_offset && sn.imm_u32 != pn.imm_offset) return false;
    if (pn.has_imm_width && sn.imm_u32_2 != pn.imm_width) return false;
    if (sn.args.size() < pn.num_children) return false;

    for (uint8_t c = 0; c < pn.num_children; ++c) {
        if (pn.children[c] < 0) return false;
        if (!match_tile_pattern_ssa(mod, pnodes, static_cast<uint16_t>(pn.children[c]),
                                    sn.args[c], captured_leaves)) {
            return false;
        }
    }
    return true;
}

bool try_apply_tile_spec_rewrite(ValhallSsaModule& mod,
                                 SsaId nid,
                                 bool allow_fast_math,
                                 std::vector<SsaId>& remap,
                                 std::vector<uint8_t>& dead,
                                 bool& any_remapped,
                                 FpOptResult& result) {
    size_t num_tiles = 0, num_pnodes = 0;
    const auto* tiles = tables::get_fusion_tiles(&num_tiles);
    const auto* pnodes = tables::get_fusion_pattern_nodes(&num_pnodes);
    (void)num_pnodes;

    const SsaOp root_op = mod.get(nid).op;
    for (size_t ti = 0; ti < num_tiles; ++ti) {
        const auto& ft = tiles[ti];
        if (!ft.has_ssa_rewrite) continue;
        if (ft.requires_fast_math && !allow_fast_math) continue;
        if (pnodes[ft.root_node].ssa_op != root_op) continue;

        std::vector<SsaId> leaves;
        if (!match_tile_pattern_ssa(mod, pnodes, ft.root_node, nid, leaves)) {
            continue;
        }
        if (ft.ssa_rewrite_forward_leaf >= 0) {
            size_t slot = static_cast<size_t>(ft.ssa_rewrite_forward_leaf);
            if (slot >= leaves.size() || leaves[slot] == kInvalidSsa) continue;
            remap[nid] = resolve_remap(remap, leaves[slot]);
            dead[nid] = 1;
            any_remapped = true;
            ++result.stats.nodes_replaced;
        } else if (ft.ssa_rewrite_op != SsaOp::Nop) {
            std::vector<SsaId> new_args;
            new_args.reserve(ft.ssa_rewrite_num_args);
            bool ok = true;
            for (uint8_t a = 0; a < ft.ssa_rewrite_num_args; ++a) {
                size_t slot = ft.ssa_rewrite_args[a];
                if (slot >= leaves.size() || leaves[slot] == kInvalidSsa) {
                    ok = false;
                    break;
                }
                new_args.push_back(resolve_remap(remap, leaves[slot]));
            }
            if (!ok) continue;
            auto& n = mod.get_mut(nid);
            n.op = ft.ssa_rewrite_op;
            n.args = std::move(new_args);
            n.imm_u32 = 0;
            n.imm_u32_2 = 0;
            n.imm_u64 = 0;
            ++result.stats.nodes_replaced;
        } else {
            continue;
        }
        ++result.stats.fp_rewrites;
        ++result.rule_hits[std::string("tile:") + ft.name];
        return true;
    }
    return false;
}


/// Unwraps optional `FNeg` / `Copy` around an SSA node.
SsaId unwrap_fneg_or_copy(const ValhallSsaModule& mod, SsaId id) {
    for (int hop = 0; hop < 4 && id != kInvalidSsa && id < mod.nodes.size(); ++hop) {
        const auto& n = mod.get(id);
        if ((n.op == SsaOp::Copy || n.op == SsaOp::FNeg) && !n.args.empty()) {
            id = n.args[0];
        } else {
            break;
        }
    }
    return id;
}

/// Checks if `id` is a `ConstS32` or `ConstU32` with exact 32-bit value `expected`.
bool is_const_i32_val(const ValhallSsaModule& mod, SsaId id, uint32_t expected) {
    id = unwrap_copies(mod, id);
    if (id == kInvalidSsa || id >= mod.nodes.size()) return false;
    const auto& n = mod.get(id);
    return (n.op == SsaOp::ConstS32 || n.op == SsaOp::ConstU32) && n.imm_u32 == expected;
}

/// Checks if `id` is a `ConstF32` with exact IEEE-754 bit pattern `expected_bits`.
bool is_const_f32_bits(const ValhallSsaModule& mod, SsaId id, uint32_t expected_bits) {
    id = unwrap_copies(mod, id);
    if (id == kInvalidSsa || id >= mod.nodes.size()) return false;
    const auto& n = mod.get(id);
    return n.op == SsaOp::ConstF32 && n.imm_u32 == expected_bits;
}

/// Matches Valhall's `FREXPE.log + FADD_LSCALE + FLOGD + Fma` sequence:
///   Fma(diff, Select(FCmpOrdEq(m, 1.0), 1/ln(2), FDiv(Log2(m), diff)), ConvertSToF(e))
/// where `m` and `e` both derive from `Bitcast<U32>(x)`. Returns `x` if matched.
SsaId match_valhall_log2_fma(const ValhallSsaModule& mod, const SsaNode& fma) {
    if (fma.op != SsaOp::Fma || fma.args.size() != 3) return kInvalidSsa;
    SsaId e_f_id = unwrap_copies(mod, fma.args[2]);
    if (e_f_id == kInvalidSsa || mod.get(e_f_id).op != SsaOp::ConvertSToF || mod.get(e_f_id).args.empty()) {
        return kInvalidSsa;
    }
    auto try_pair = [&](SsaId diff_id, SsaId flogd_id) -> SsaId {
        diff_id = unwrap_copies(mod, diff_id);
        flogd_id = unwrap_copies(mod, flogd_id);
        if (diff_id == kInvalidSsa || flogd_id == kInvalidSsa) return kInvalidSsa;
        const auto& sel = mod.get(flogd_id);
        if (sel.op != SsaOp::Select || sel.args.size() != 3) return kInvalidSsa;
        // Check 0x3fb8aa3b (1 / ln(2)) on either branch and FDiv(Log2(m), diff) on the other.
        SsaId fdiv_id = kInvalidSsa;
        if (is_const_f32_bits(mod, sel.args[1], 0x3fb8aa3bu)) {
            fdiv_id = unwrap_copies(mod, sel.args[2]);
        } else if (is_const_f32_bits(mod, sel.args[2], 0x3fb8aa3bu)) {
            fdiv_id = unwrap_copies(mod, sel.args[1]);
        } else {
            return kInvalidSsa;
        }
        if (fdiv_id == kInvalidSsa || mod.get(fdiv_id).op != SsaOp::FDiv || mod.get(fdiv_id).args.size() != 2) {
            return kInvalidSsa;
        }
        const auto& fdiv = mod.get(fdiv_id);
        if (unwrap_copies(mod, fdiv.args[1]) != diff_id) return kInvalidSsa;
        SsaId log2_m_id = unwrap_copies(mod, fdiv.args[0]);
        if (log2_m_id == kInvalidSsa || mod.get(log2_m_id).op != SsaOp::Log2 || mod.get(log2_m_id).args.empty()) {
            return kInvalidSsa;
        }
        SsaId m_id = unwrap_copies(mod, mod.get(log2_m_id).args[0]);
        if (m_id == kInvalidSsa) return kInvalidSsa;
        // Trace `m` back to `Select(is_norm, Bitcast<F32>(Bfi(Bitcast<U32>(x), ..., 23, 8)), x)`
        // or directly `Bitcast<F32>(Bfi(Bitcast<U32>(x), ..., 23, 8))`.
        SsaId bfi_f_id = m_id;
        SsaId x_fallback = kInvalidSsa;
        if (mod.get(m_id).op == SsaOp::Select && mod.get(m_id).args.size() == 3) {
            bfi_f_id = unwrap_copies(mod, mod.get(m_id).args[1]);
            x_fallback = unwrap_copies(mod, mod.get(m_id).args[2]);
        }
        if (bfi_f_id == kInvalidSsa || mod.get(bfi_f_id).op != SsaOp::Bitcast || mod.get(bfi_f_id).args.empty()) {
            return kInvalidSsa;
        }
        SsaId bfi_id = unwrap_copies(mod, mod.get(bfi_f_id).args[0]);
        if (bfi_id == kInvalidSsa || mod.get(bfi_id).op != SsaOp::Bfi || mod.get(bfi_id).args.size() < 4) {
            return kInvalidSsa;
        }
        const auto& bfi = mod.get(bfi_id);
        if (!is_const_i32_val(mod, bfi.args[2], 23u) || !is_const_i32_val(mod, bfi.args[3], 8u)) {
            return kInvalidSsa;
        }
        SsaId x_u_id = unwrap_copies(mod, bfi.args[0]);
        if (x_u_id == kInvalidSsa || mod.get(x_u_id).op != SsaOp::Bitcast || mod.get(x_u_id).args.empty()) {
            return kInvalidSsa;
        }
        SsaId x_id = unwrap_copies(mod, mod.get(x_u_id).args[0]);
        if (x_id == kInvalidSsa || mod.get(x_id).type != ValType::F32) return kInvalidSsa;
        if (x_fallback != kInvalidSsa && x_fallback != x_id) return kInvalidSsa;
        // Also verify `e_f` extracts bits [30:23] from the same `x_u_id`.
        SsaId e_src = unwrap_copies(mod, mod.get(e_f_id).args[0]);
        bool found_exp_bfe = false;
        std::vector<SsaId> work = {e_src};
        for (size_t wi = 0; wi < work.size() && wi < 64; ++wi) {
            SsaId cur = unwrap_copies(mod, work[wi]);
            if (cur == kInvalidSsa || cur >= mod.nodes.size()) continue;
            const auto& cn = mod.get(cur);
            if (cn.op == SsaOp::BfeU && cn.args.size() >= 3 &&
                unwrap_copies(mod, cn.args[0]) == x_u_id &&
                is_const_i32_val(mod, cn.args[1], 23u) &&
                is_const_i32_val(mod, cn.args[2], 8u)) {
                found_exp_bfe = true;
                break;
            }
            for (SsaId a : cn.args) work.push_back(a);
        }
        return found_exp_bfe ? x_id : kInvalidSsa;
    };
    SsaId res = try_pair(fma.args[0], fma.args[1]);
    if (res == kInvalidSsa) res = try_pair(fma.args[1], fma.args[0]);
    return res;
}

/// Matches Valhall's Q8.24 `Exp2` wrapper:
///   Exp2(Ldexp(ConvertSToF(ConvertFToS(Ldexp(t, +24))), -24)) -> Exp2(t)
SsaId match_valhall_exp2_q824(const ValhallSsaModule& mod, const SsaNode& exp_n) {
    if (exp_n.op != SsaOp::Exp2 || exp_n.args.size() != 1) return kInvalidSsa;
    auto unwrap_pow2_scale = [&](SsaId id, uint32_t exp_u32, uint32_t f32_bits) -> SsaId {
        id = unwrap_copies(mod, id);
        if (id == kInvalidSsa) return kInvalidSsa;
        const auto& sn = mod.get(id);
        if (sn.op == SsaOp::Ldexp && sn.args.size() == 2 && is_const_i32_val(mod, sn.args[1], exp_u32)) {
            return unwrap_copies(mod, sn.args[0]);
        }
        if (sn.op == SsaOp::FMul && sn.args.size() == 2) {
            if (is_const_f32_bits(mod, sn.args[1], f32_bits)) return unwrap_copies(mod, sn.args[0]);
            if (is_const_f32_bits(mod, sn.args[0], f32_bits)) return unwrap_copies(mod, sn.args[1]);
        }
        return kInvalidSsa;
    };
    SsaId stof_id = unwrap_pow2_scale(exp_n.args[0], static_cast<uint32_t>(-24), 0x33800000u);
    if (stof_id == kInvalidSsa || mod.get(stof_id).op != SsaOp::ConvertSToF || mod.get(stof_id).args.empty()) {
        return kInvalidSsa;
    }
    SsaId ftos_id = unwrap_copies(mod, mod.get(stof_id).args[0]);
    if (ftos_id == kInvalidSsa || mod.get(ftos_id).op != SsaOp::ConvertFToS || mod.get(ftos_id).args.empty()) {
        return kInvalidSsa;
    }
    SsaId ld_pos = unwrap_copies(mod, mod.get(ftos_id).args[0]);
    if (ld_pos != kInvalidSsa && mod.get(ld_pos).op == SsaOp::FClamp && !mod.get(ld_pos).args.empty()) {
        ld_pos = unwrap_copies(mod, mod.get(ld_pos).args[0]);
    }
    SsaId t = unwrap_pow2_scale(ld_pos, 24u, 0x4b800000u);
    if (t != kInvalidSsa && mod.get(t).op == SsaOp::FClamp && !mod.get(t).args.empty()) {
        t = unwrap_copies(mod, mod.get(t).args[0]);
    }
    return (t != kInvalidSsa && mod.get(t).type == ValType::F32) ? t : kInvalidSsa;
}

/// Traces a Valhall `FSIN_TABLE.u6` / `FCOS_TABLE.u6` table lookup node `Sin(theta)` or `Cos(theta)`
/// back to the Cody-Waite quadrant reduction `q_f = Fma(x, 2/pi, 786432.0)`.
/// Returns `x` and sets `*out_qf = q_f` and `*out_is_cos` if matched.
SsaId match_valhall_trig_table(const ValhallSsaModule& mod,
                               SsaId tbl_id,
                               SsaId* out_qf,
                               bool* out_is_cos) {
    tbl_id = unwrap_copies(mod, tbl_id);
    if (tbl_id == kInvalidSsa || tbl_id >= mod.nodes.size()) return kInvalidSsa;
    const auto& tn = mod.get(tbl_id);
    if ((tn.op != SsaOp::Sin && tn.op != SsaOp::Cos) || tn.args.size() != 1) return kInvalidSsa;
    *out_is_cos = (tn.op == SsaOp::Cos);

    // theta = FMul(ConvertUToF(BitAnd(Bitcast<U32>(q_f), 0x3f)), 0x3dc90fdb /* pi/32 */)
    SsaId theta_id = unwrap_copies(mod, tn.args[0]);
    if (theta_id == kInvalidSsa || mod.get(theta_id).op != SsaOp::FMul || mod.get(theta_id).args.size() != 2) {
        return kInvalidSsa;
    }
    const auto& th = mod.get(theta_id);
    SsaId utof_id = kInvalidSsa;
    if (is_const_f32_bits(mod, th.args[1], 0x3dc90fdbu)) utof_id = unwrap_copies(mod, th.args[0]);
    else if (is_const_f32_bits(mod, th.args[0], 0x3dc90fdbu)) utof_id = unwrap_copies(mod, th.args[1]);
    else return kInvalidSsa;

    if (utof_id == kInvalidSsa || mod.get(utof_id).op != SsaOp::ConvertUToF || mod.get(utof_id).args.empty()) {
        return kInvalidSsa;
    }
    SsaId band_id = unwrap_copies(mod, mod.get(utof_id).args[0]);
    if (band_id == kInvalidSsa || mod.get(band_id).op != SsaOp::BitAnd || mod.get(band_id).args.size() != 2) {
        return kInvalidSsa;
    }
    const auto& ba = mod.get(band_id);
    SsaId q_u_id = kInvalidSsa;
    if (is_const_i32_val(mod, ba.args[1], 0x3fu)) q_u_id = unwrap_copies(mod, ba.args[0]);
    else if (is_const_i32_val(mod, ba.args[0], 0x3fu)) q_u_id = unwrap_copies(mod, ba.args[1]);
    else return kInvalidSsa;

    if (q_u_id == kInvalidSsa || mod.get(q_u_id).op != SsaOp::Bitcast || mod.get(q_u_id).args.empty()) {
        return kInvalidSsa;
    }
    SsaId q_f_id = unwrap_copies(mod, mod.get(q_u_id).args[0]);
    if (q_f_id == kInvalidSsa || mod.get(q_f_id).op != SsaOp::Fma || mod.get(q_f_id).args.size() != 3) {
        return kInvalidSsa;
    }
    const auto& qf = mod.get(q_f_id);
    // q_f = Fma(x, 2/pi [0x3f22f983], 786432.0 [0x49400000])
    if (!is_const_f32_bits(mod, qf.args[2], 0x49400000u)) return kInvalidSsa;
    SsaId x_id = kInvalidSsa;
    if (is_const_f32_bits(mod, qf.args[1], 0x3f22f983u)) x_id = unwrap_copies(mod, qf.args[0]);
    else if (is_const_f32_bits(mod, qf.args[0], 0x3f22f983u)) x_id = unwrap_copies(mod, qf.args[1]);
    else return kInvalidSsa;

    if (x_id == kInvalidSsa || mod.get(x_id).type != ValType::F32) return kInvalidSsa;
    *out_qf = q_f_id;
    return x_id;
}

/// Checks that `corr_id` is a Valhall Taylor correction term referencing the Cody-Waite
/// residual `r = Fma(FAdd(q_f, -786432.0), -pi/2, x)`.
bool correction_uses_cody_waite_residual(const ValhallSsaModule& mod,
                                         SsaId corr_id,
                                         SsaId q_f_id,
                                         SsaId x_id) {
    SsaId cur = unwrap_fneg_or_copy(mod, corr_id);
    if (cur != kInvalidSsa && mod.get(cur).op == SsaOp::FClamp && !mod.get(cur).args.empty()) {
        cur = unwrap_fneg_or_copy(mod, mod.get(cur).args[0]);
    }
    if (cur == kInvalidSsa || (mod.get(cur).op != SsaOp::Fma && mod.get(cur).op != SsaOp::FMul) || mod.get(cur).args.size() < 2) {
        return false;
    }
    const auto& fma = mod.get(cur);
    auto is_residual = [&](SsaId rid) -> bool {
        rid = unwrap_copies(mod, rid);
        if (rid == kInvalidSsa || mod.get(rid).op != SsaOp::Fma || mod.get(rid).args.size() != 3) {
            return false;
        }
        const auto& rn = mod.get(rid);
        if (unwrap_copies(mod, rn.args[2]) != x_id) return false;
        SsaId n_id = kInvalidSsa;
        if (is_const_f32_bits(mod, rn.args[1], 0xbfc90fdbu)) n_id = unwrap_copies(mod, rn.args[0]);
        else if (is_const_f32_bits(mod, rn.args[0], 0xbfc90fdbu)) n_id = unwrap_copies(mod, rn.args[1]);
        else return false;
        if (n_id == kInvalidSsa || mod.get(n_id).op != SsaOp::FAdd || mod.get(n_id).args.size() != 2) {
            return false;
        }
        const auto& na = mod.get(n_id);
        return (unwrap_copies(mod, na.args[0]) == q_f_id && is_const_f32_bits(mod, na.args[1], 0xc9400000u)) ||
               (unwrap_copies(mod, na.args[1]) == q_f_id && is_const_f32_bits(mod, na.args[0], 0xc9400000u));
    };
    return is_residual(fma.args[0]) || is_residual(fma.args[1]);
}

/// Canonicalizes Mali Vertex `gl_mali_ViewportTransform` epilogue stores:
///   - Case 1: If a block already stores clip-space `gl_Position` (from `XFB_Position`)
///     and later stores screen-space `gl_Position` with `w = FDiv(1.0, ...)`, drop the
///     redundant screen-space store so DCE removes the viewport transform epilogue.
///   - Case 2: If only the screen-space `gl_Position` store exists with
///     `VecConstruct4(Fma(inv_w, FMul(x, vp_sx), vp_ox), Fma(inv_w, FMul(y, vp_sy), vp_oy), sz, inv_w)`
///     where `inv_w = FDiv(1.0, w)`, reconstruct clip-space `VecConstruct4(x, y, z, w)`.
bool canonicalize_vertex_viewport_epilogue(ValhallSsaModule& mod, FpOptResult& result) {
    bool any_changed = false;
    auto is_screen_space_pos_vec = [&](SsaId vec_id, SsaId* out_inv_w) -> bool {
        vec_id = unwrap_copies(mod, vec_id);
        if (vec_id == kInvalidSsa || mod.get(vec_id).op != SsaOp::VecConstruct4 ||
            mod.get(vec_id).args.size() != 4) {
            return false;
        }
        SsaId w_id = unwrap_copies(mod, mod.get(vec_id).args[3]);
        if (w_id == kInvalidSsa || mod.get(w_id).op != SsaOp::FDiv || mod.get(w_id).args.size() != 2) {
            return false;
        }
        if (!is_const_f32_bits(mod, mod.get(w_id).args[0], 0x3f800000u)) return false;
        *out_inv_w = w_id;
        return true;
    };

    for (size_t b = 0; b < mod.blocks.size(); ++b) {
        std::vector<SsaId> pos_stores;
        for (SsaId nid : mod.blocks[b].nodes) {
            const auto& n = mod.get(nid);
            if (n.op == SsaOp::StoreBuiltinOutput && n.aux_str == "gl_Position" && n.args.size() >= 2) {
                pos_stores.push_back(nid);
            }
        }
        if (pos_stores.empty()) continue;

        // Case 1: Earlier clip-space gl_Position + later screen-space gl_Position epilogue.
        if (pos_stores.size() >= 2) {
            std::vector<SsaId> tok_remap(mod.nodes.size(), kInvalidSsa);
            for (size_t i = 0; i < tok_remap.size(); ++i) tok_remap[i] = static_cast<SsaId>(i);
            bool dropped = false;
            for (size_t idx = 1; idx < pos_stores.size(); ++idx) {
                SsaId sid = pos_stores[idx];
                const auto& st = mod.get(sid);
                SsaId inv_w = kInvalidSsa;
                if (is_screen_space_pos_vec(st.args[0], &inv_w)) {
                    tok_remap[sid] = st.args[1];
                    auto& mut = mod.get_mut(sid);
                    mut.op = SsaOp::Nop;
                    mut.args.clear();
                    mut.aux_str.clear();
                    mut.is_side_effect = false;
                    mut.is_barrier_or_cf = false;
                    ++result.rule_hits["vert_viewport_drop_redundant"];
                    ++result.stats.fp_rewrites;
                    ++result.stats.nodes_removed;
                    dropped = true;
                    any_changed = true;
                }
            }
            if (dropped) {
                apply_value_remap(mod, tok_remap);
            }
            continue;
        }

        // Case 2: Single screen-space gl_Position store; symbolically invert viewport transform.
        SsaId sid = pos_stores[0];
        SsaId vec_id = unwrap_copies(mod, mod.get(sid).args[0]);
        SsaId inv_w = kInvalidSsa;
        if (!is_screen_space_pos_vec(vec_id, &inv_w)) continue;
        SsaId orig_w = unwrap_copies(mod, mod.get(inv_w).args[1]);
        auto extract_clip_xy = [&](SsaId coord_id) -> SsaId {
            coord_id = unwrap_copies(mod, coord_id);
            if (coord_id == kInvalidSsa || mod.get(coord_id).op != SsaOp::Fma || mod.get(coord_id).args.size() != 3) {
                return kInvalidSsa;
            }
            const auto& fma = mod.get(coord_id);
            SsaId scaled = kInvalidSsa;
            if (unwrap_copies(mod, fma.args[0]) == inv_w) scaled = unwrap_copies(mod, fma.args[1]);
            else if (unwrap_copies(mod, fma.args[1]) == inv_w) scaled = unwrap_copies(mod, fma.args[0]);
            else return kInvalidSsa;
            if (scaled == kInvalidSsa || mod.get(scaled).op != SsaOp::FMul || mod.get(scaled).args.size() != 2) {
                return kInvalidSsa;
            }
            // Pick the operand of FMul(x, vp_scale) that is not a Bitcast of a UBO/push_const load.
            SsaId a0 = unwrap_copies(mod, mod.get(scaled).args[0]);
            SsaId a1 = unwrap_copies(mod, mod.get(scaled).args[1]);
            bool a0_bc = (a0 != kInvalidSsa && mod.get(a0).op == SsaOp::Bitcast);
            bool a1_bc = (a1 != kInvalidSsa && mod.get(a1).op == SsaOp::Bitcast);
            if (!a0_bc && a1_bc) return a0;
            if (a0_bc && !a1_bc) return a1;
            return a0;
        };
        auto extract_clip_z = [&](SsaId z_id) -> SsaId {
            z_id = unwrap_copies(mod, z_id);
            if (z_id == kInvalidSsa || mod.get(z_id).op != SsaOp::Fma || mod.get(z_id).args.size() != 3) {
                return kInvalidSsa;
            }
            SsaId ndc_z = unwrap_copies(mod, mod.get(z_id).args[0]);
            if (ndc_z != kInvalidSsa && mod.get(ndc_z).op == SsaOp::FClamp && !mod.get(ndc_z).args.empty()) {
                ndc_z = unwrap_copies(mod, mod.get(ndc_z).args[0]);
            }
            if (ndc_z == kInvalidSsa || mod.get(ndc_z).op != SsaOp::FMul || mod.get(ndc_z).args.size() != 2) {
                return kInvalidSsa;
            }
            if (unwrap_copies(mod, mod.get(ndc_z).args[0]) == inv_w) return unwrap_copies(mod, mod.get(ndc_z).args[1]);
            if (unwrap_copies(mod, mod.get(ndc_z).args[1]) == inv_w) return unwrap_copies(mod, mod.get(ndc_z).args[0]);
            return kInvalidSsa;
        };
        const auto& vn = mod.get(vec_id);
        SsaId cx = extract_clip_xy(vn.args[0]);
        SsaId cy = extract_clip_xy(vn.args[1]);
        SsaId cz = extract_clip_z(vn.args[2]);
        if (cx != kInvalidSsa && cy != kInvalidSsa && cz != kInvalidSsa && orig_w != kInvalidSsa) {
            SsaNode clip_v{};
            clip_v.id = static_cast<SsaId>(mod.nodes.size());
            clip_v.op = SsaOp::VecConstruct4;
            clip_v.type = ValType::V4F32;
            clip_v.args = {cx, cy, cz, orig_w};
            mod.nodes.push_back(std::move(clip_v));
            SsaId clip_id = static_cast<SsaId>(mod.nodes.size() - 1);
            auto& bnodes = mod.blocks[b].nodes;
            auto pos = std::find(bnodes.begin(), bnodes.end(), sid);
            bnodes.insert(pos, clip_id);
            mod.get_mut(sid).args[0] = clip_id;
            ++result.rule_hits["vert_viewport_invert_clip"];
            ++result.stats.fp_rewrites;
            ++result.stats.nodes_replaced;
            any_changed = true;
        }
    }
    return any_changed;
}

}  // namespace

FpOptResult run_ssa_fp_canonicalize(ValhallSsaModule& mod, const FpOptOptions& opts) {
    FpOptResult result;
    if (mod.blocks.empty()) return result;

    const bool fast = (opts.mode == FpOptMode::FastMath);
    const bool lift_transc = fast || opts.enable_transcendental_lifting;

    for (uint32_t iter = 0; iter < opts.max_iterations; ++iter) {
        bool changed = false;
        ++result.stats.iterations;

        std::vector<SsaId> remap(mod.nodes.size(), kInvalidSsa);
        for (size_t i = 0; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
        std::vector<uint8_t> dead(mod.nodes.size(), 0);
        bool any_remapped = false;

        auto ensure_remap = [&]() {
            size_t old_sz = remap.size();
            if (mod.nodes.size() > old_sz) {
                remap.resize(mod.nodes.size());
                for (size_t i = old_sz; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
                dead.resize(mod.nodes.size(), 0);
            }
        };

        auto hit = [&](const char* rule_name) {
            ++result.rule_hits[rule_name];
            ++result.stats.fp_rewrites;
            changed = true;
        };

        auto replace_with = [&](SsaId old_id, SsaId new_id, const char* rule_name) {
            ensure_remap();
            remap[old_id] = resolve_remap(remap, new_id);
            dead[old_id] = 1;
            any_remapped = true;
            ++result.stats.nodes_replaced;
            hit(rule_name);
        };

        auto rewrite_to_const_f32 = [&](SsaId nid, uint32_t bits, const char* rule_name) {
            auto& node = mod.get_mut(nid);
            node.op = SsaOp::ConstF32;
            node.type = ValType::F32;
            node.args.clear();
            node.imm_u32 = bits;
            ++result.stats.nodes_folded;
            hit(rule_name);
        };

        for (size_t b = 0; b < mod.blocks.size(); ++b) {
            std::vector<SsaId> snap = mod.blocks[b].nodes;
            for (SsaId nid : snap) {
                if (nid >= dead.size() || dead[nid]) continue;

                // Refresh args through current remap before matching.
                {
                    auto& n = mod.get_mut(nid);
                    for (auto& a : n.args) a = resolve_remap(remap, a);
                    if (canonicalize_fp_commutativity(mod, n)) {
                        changed = true;
                    }
                }

                if (opts.enable_tile_spec_rewrites) {
                    if (try_apply_tile_spec_rewrite(mod, nid, fast, remap, dead, any_remapped, result)) {
                        changed = true;
                        continue;
                    }
                }
                if (!opts.enable_cpp_rewrites) {
                    continue;
                }

                const SsaNode n = mod.get(nid);
                const uint32_t blk = static_cast<uint32_t>(b);

                // Sign-bit bitwise idioms -> FAbs / FNeg / FNeg(FAbs) (M5-M7, T1)
                // and CSEL Bitcast<F32>(Select(c, Bitcast<U32>(a), Bitcast<U32>(b))) -> Select<F32>(c, a, b)
                if (n.op == SsaOp::Bitcast && is_f32_scalar(n) && n.args.size() == 1) {
                    const auto& inner = mod.get(n.args[0]);
                    if (inner.op == SsaOp::Select && inner.args.size() == 3) {
                        auto unwrap_f32_branch = [&](SsaId bid, bool* needs_const, uint32_t* const_bits) -> SsaId {
                            *needs_const = false;
                            SsaId cur = resolve_remap(remap, bid);
                            const auto& bn = mod.get(cur);
                            if (bn.op == SsaOp::Bitcast && bn.args.size() == 1 &&
                                mod.get(bn.args[0]).type == ValType::F32) {
                                return resolve_remap(remap, bn.args[0]);
                            }
                            if (bn.op == SsaOp::ConstU32) {
                                *needs_const = true;
                                *const_bits = bn.imm_u32;
                                return cur;
                            }
                            return kInvalidSsa;
                        };
                        bool tc = false, fc = false;
                        uint32_t tb = 0, fb = 0;
                        SsaId ft = unwrap_f32_branch(inner.args[1], &tc, &tb);
                        SsaId ff = unwrap_f32_branch(inner.args[2], &fc, &fb);
                        if (ft != kInvalidSsa && ff != kInvalidSsa) {
                            SsaId cond_id = resolve_remap(remap, inner.args[0]);
                            if (tc) {
                                SsaNode cn{};
                                cn.id = static_cast<SsaId>(mod.nodes.size());
                                cn.op = SsaOp::ConstF32;
                                cn.type = ValType::F32;
                                cn.imm_u32 = tb;
                                mod.nodes.push_back(cn);
                                ft = cn.id;
                                auto& bnodes = mod.blocks[blk].nodes;
                                auto pos = std::find(bnodes.begin(), bnodes.end(), nid);
                                bnodes.insert(pos, ft);
                                ensure_remap();
                            }
                            if (fc) {
                                SsaNode cn{};
                                cn.id = static_cast<SsaId>(mod.nodes.size());
                                cn.op = SsaOp::ConstF32;
                                cn.type = ValType::F32;
                                cn.imm_u32 = fb;
                                mod.nodes.push_back(cn);
                                ff = cn.id;
                                auto& bnodes = mod.blocks[blk].nodes;
                                auto pos = std::find(bnodes.begin(), bnodes.end(), nid);
                                bnodes.insert(pos, ff);
                                ensure_remap();
                            }
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Select;
                            mut.type = ValType::F32;
                            mut.args = {cond_id, ft, ff};
                            hit("csel_bitcast_f32");
                            continue;
                        }
                    }
                    if ((inner.op == SsaOp::BitAnd || inner.op == SsaOp::BitXor || inner.op == SsaOp::BitOr) &&
                        inner.args.size() == 2) {
                        uint32_t mask = 0;
                        SsaId other = kInvalidSsa;
                        if (try_get_const_u32(mod, inner.args[1], &mask)) {
                            other = inner.args[0];
                        } else if (try_get_const_u32(mod, inner.args[0], &mask)) {
                            other = inner.args[1];
                        }
                        if (other != kInvalidSsa) {
                            const auto& on = mod.get(other);
                            if (on.op == SsaOp::Bitcast && on.args.size() == 1 &&
                                mod.get(on.args[0]).type == ValType::F32) {
                                SsaId fx = on.args[0];
                                if (inner.op == SsaOp::BitAnd && mask == 0x7FFFFFFFu) {
                                    auto& mut = mod.get_mut(nid);
                                    mut.op = SsaOp::FAbs;
                                    mut.args = {fx};
                                    hit("M5_bitand_fabs");
                                    continue;
                                }
                                if (inner.op == SsaOp::BitXor && mask == 0x80000000u) {
                                    auto& mut = mod.get_mut(nid);
                                    mut.op = SsaOp::FNeg;
                                    mut.args = {fx};
                                    hit("M6_bitxor_fneg");
                                    continue;
                                }
                                if (inner.op == SsaOp::BitOr && mask == 0x80000000u) {
                                    SsaId ab = emit_fabs_before(mod, blk, nid, fx);
                                    ensure_remap();
                                    auto& mut = mod.get_mut(nid);
                                    mut.op = SsaOp::FNeg;
                                    mut.args = {ab};
                                    hit("M7_bitor_neg_fabs");
                                    continue;
                                }
                            }
                        }
                    }
                }

                // FNeg rules (M1, M4, C9)
                if (n.op == SsaOp::FNeg && is_f32_scalar(n) && n.args.size() == 1) {
                    uint32_t cbits = 0;
                    if (try_get_const_f32_bits(mod, n.args[0], &cbits)) {
                        rewrite_to_const_f32(nid, cbits ^ 0x80000000u, "M4_fneg_const");
                        continue;
                    }
                    const auto& inner = mod.get(n.args[0]);
                    if (inner.op == SsaOp::FNeg && inner.args.size() == 1) {
                        replace_with(nid, inner.args[0], "M1_fneg_fneg");
                        continue;
                    }
                    // C9: FNeg(FMax/NMax(FNeg(a), FNeg(b))) -> FMin/NMin(a, b) (and dual)
                    if ((inner.op == SsaOp::FMax || inner.op == SsaOp::FMin ||
                         inner.op == SsaOp::NMax || inner.op == SsaOp::NMin) &&
                        inner.args.size() == 2) {
                        const auto& ia = mod.get(inner.args[0]);
                        const auto& ib = mod.get(inner.args[1]);
                        if (ia.op == SsaOp::FNeg && ib.op == SsaOp::FNeg &&
                            ia.args.size() == 1 && ib.args.size() == 1) {
                            auto& mut = mod.get_mut(nid);
                            if (inner.op == SsaOp::FMax) mut.op = SsaOp::FMin;
                            else if (inner.op == SsaOp::FMin) mut.op = SsaOp::FMax;
                            else if (inner.op == SsaOp::NMax) mut.op = SsaOp::NMin;
                            else mut.op = SsaOp::NMax;
                            mut.args = {ia.args[0], ib.args[0]};
                            hit("C9_demorgan_minmax");
                            continue;
                        }
                    }
                }

                // FAbs rules (M2, M3, M4, M8)
                if (n.op == SsaOp::FAbs && is_f32_scalar(n) && n.args.size() == 1) {
                    uint32_t cbits = 0;
                    if (try_get_const_f32_bits(mod, n.args[0], &cbits)) {
                        rewrite_to_const_f32(nid, cbits & 0x7FFFFFFFu, "M4_fabs_const");
                        continue;
                    }
                    const auto& inner = mod.get(n.args[0]);
                    if (inner.op == SsaOp::FAbs && inner.args.size() == 1) {
                        replace_with(nid, n.args[0], "M2_fabs_fabs");
                        continue;
                    }
                    if (inner.op == SsaOp::FNeg && inner.args.size() == 1) {
                        mod.get_mut(nid).args[0] = inner.args[0];
                        hit("M3_fabs_fneg");
                        continue;
                    }
                    if (inner.op == SsaOp::ConvertUToF) {
                        replace_with(nid, n.args[0], "M8_fabs_utof");
                        continue;
                    }
                }

                // Phase 3 transcendental lifting (Valhall Log2 / Q8.24 Exp2 / Sin / Cos)
                if (lift_transc && is_f32_scalar(n)) {
                    if (n.op == SsaOp::Fma && n.args.size() == 3) {
                        SsaId x_id = match_valhall_log2_fma(mod, n);
                        if (x_id != kInvalidSsa) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Log2;
                            mut.args = {resolve_remap(remap, x_id)};
                            hit("valhall_log2_lift");
                            continue;
                        }
                    }
                    if (n.op == SsaOp::Exp2 && n.args.size() == 1) {
                        SsaId t_id = match_valhall_exp2_q824(mod, n);
                        if (t_id != kInvalidSsa) {
                            auto& mut = mod.get_mut(nid);
                            mut.args = {resolve_remap(remap, t_id)};
                            hit("valhall_exp2_q824_lift");
                            continue;
                        }
                    }
                    if (n.op == SsaOp::FAdd && n.args.size() == 2) {
                        bool trig_lifted = false;
                        for (int side = 0; side < 2; ++side) {
                            SsaId q_f_id = kInvalidSsa;
                            bool is_cos = false;
                            SsaId x_id = match_valhall_trig_table(mod, n.args[side], &q_f_id, &is_cos);
                            if (x_id != kInvalidSsa &&
                                correction_uses_cody_waite_residual(mod, n.args[1 - side], q_f_id, x_id)) {
                                auto& mut = mod.get_mut(nid);
                                mut.op = is_cos ? SsaOp::Cos : SsaOp::Sin;
                                mut.args = {resolve_remap(remap, x_id)};
                                hit(is_cos ? "valhall_cos_table_lift" : "valhall_sin_table_lift");
                                trig_lifted = true;
                                break;
                            }
                        }
                        if (trig_lifted) continue;
                    }
                }

                // CLPER quad-shuffle derivative idiom:
                // FSub(SubgroupShuffle(x, BitXor(lid, 1|2)), x) -> DPdx(x) / DPdy(x)
                if (n.op == SsaOp::FSub && is_f32_scalar(n) && n.args.size() == 2) {
                    const auto& lhs = mod.get(n.args[0]);
                    if (lhs.op == SsaOp::SubgroupShuffle && lhs.args.size() >= 2 &&
                        lhs.args[0] == n.args[1]) {
                        const auto& lane_expr = mod.get(lhs.args[1]);
                        if (lane_expr.op == SsaOp::BitXor && lane_expr.args.size() == 2) {
                            uint32_t xor_mask = 0;
                            bool has_mask = try_get_const_u32(mod, lane_expr.args[1], &xor_mask) ||
                                            try_get_const_u32(mod, lane_expr.args[0], &xor_mask);
                            if (has_mask && (xor_mask == 1u || xor_mask == 2u)) {
                                auto& mut = mod.get_mut(nid);
                                mut.op = (xor_mask == 1u) ? SsaOp::DPdx : SsaOp::DPdy;
                                mut.args = {n.args[1]};
                                hit(xor_mask == 1u ? "clper_dpdx" : "clper_dpdy");
                                continue;
                            }
                        }
                    }
                }

                // FSub rules (I3, I4, I5, F1)
                if (n.op == SsaOp::FSub && is_f32_scalar(n) && n.args.size() == 2) {
                    uint32_t lb = 0, rb = 0;
                    bool lc = try_get_const_f32_bits(mod, n.args[0], &lb);
                    bool rc = try_get_const_f32_bits(mod, n.args[1], &rb);
                    if (rc && is_pos_zero(rb)) {
                        replace_with(nid, n.args[0], "I3_fsub_pos0");
                        continue;
                    }
                    if (lc && (is_neg_zero(lb) || (fast && is_pos_zero(lb)))) {
                        auto& mut = mod.get_mut(nid);
                        mut.op = SsaOp::FNeg;
                        mut.args = {n.args[1]};
                        hit(is_neg_zero(lb) ? "I4_neg0_fsub" : "I5_pos0_fsub");
                        continue;
                    }
                    if (fast && n.args[0] == n.args[1]) {
                        rewrite_to_const_f32(nid, kPosZeroBits, "I14_fsub_self");
                        continue;
                    }
                    if (opts.desugar_fsub) {
                        SsaId neg_b = emit_fneg_before(mod, blk, nid, n.args[1]);
                        ensure_remap();
                        auto& mut = mod.get_mut(nid);
                        mut.op = SsaOp::FAdd;
                        mut.args = {n.args[0], neg_b};
                        hit("F1_desugar_fsub");
                        continue;
                    }
                }

                // FAdd rules (I1, I2, I14, I15, F2, F3, F6, F7, F8, T10)
                if (n.op == SsaOp::FAdd && is_f32_scalar(n) && n.args.size() == 2) {
                    uint32_t rb = 0;
                    bool rc = try_get_const_f32_bits(mod, n.args[1], &rb);
                    if (rc && is_neg_zero(rb)) {
                        replace_with(nid, n.args[0], "I1_fadd_neg0");
                        continue;
                    }
                    if (fast && rc && is_pos_zero(rb)) {
                        replace_with(nid, n.args[0], "I2_fadd_pos0");
                        continue;
                    }
                    // I15 (T1): FAdd(x, x) -> FMul(x, 2.0)
                    if (n.args[0] == n.args[1]) {
                        SsaId c2 = emit_f32_const_before(mod, blk, nid, kPosTwoBits);
                        ensure_remap();
                        auto& mut = mod.get_mut(nid);
                        mut.op = SsaOp::FMul;
                        mut.args = {n.args[0], c2};
                        hit("I15_fadd_self_to_mul2");
                        continue;
                    }
                    // I14 (T2/fast): FAdd(x, FNeg(x)) -> 0.0
                    if (fast) {
                        const auto& na = mod.get(n.args[0]);
                        const auto& nb = mod.get(n.args[1]);
                        if ((nb.op == SsaOp::FNeg && nb.args.size() == 1 && nb.args[0] == n.args[0]) ||
                            (na.op == SsaOp::FNeg && na.args.size() == 1 && na.args[0] == n.args[1])) {
                            rewrite_to_const_f32(nid, kPosZeroBits, "I14_fadd_neg_self");
                            continue;
                        }
                    }
                    // F7 (T3/fast): FAdd(FAdd(a, c1), c2) -> FAdd(a, c1 + c2)
                    if (fast && rc) {
                        const auto& na = mod.get(n.args[0]);
                        uint32_t c1b = 0;
                        if (na.op == SsaOp::FAdd && na.args.size() == 2 &&
                            try_get_const_f32_bits(mod, na.args[1], &c1b)) {
                            SsaId inner_a = na.args[0];
                            uint32_t folded = f32_to_bits(bits_to_f32(c1b) + bits_to_f32(rb));
                            SsaId fc = emit_f32_const_before(mod, blk, nid, folded);
                            ensure_remap();
                            auto& mut = mod.get_mut(nid);
                            mut.args = {inner_a, fc};
                            hit("F7_fadd_reassoc_const");
                            continue;
                        }
                    }
                    // F2/F3 (T3/fast): FAdd(FMul(a, b), c) -> Fma(a, b, c)
                    if (fast) {
                        const auto& na = mod.get(n.args[0]);
                        const auto& nb = mod.get(n.args[1]);
                        if (na.op == SsaOp::FMul && na.args.size() == 2) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Fma;
                            mut.args = {na.args[0], na.args[1], n.args[1]};
                            hit("F2_fuse_fma");
                            continue;
                        }
                        if (nb.op == SsaOp::FMul && nb.args.size() == 2) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Fma;
                            mut.args = {nb.args[0], nb.args[1], n.args[0]};
                            hit("F2_fuse_fma");
                            continue;
                        }
                        if (na.op == SsaOp::FNeg && na.args.size() == 1 &&
                            mod.get(na.args[0]).op == SsaOp::FMul &&
                            mod.get(na.args[0]).args.size() == 2) {
                            SsaId m0 = mod.get(na.args[0]).args[0];
                            SsaId m1 = mod.get(na.args[0]).args[1];
                            SsaId neg_m0 = emit_fneg_before(mod, blk, nid, m0);
                            ensure_remap();
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Fma;
                            mut.args = {neg_m0, m1, n.args[1]};
                            hit("F3_fuse_neg_fma");
                            continue;
                        }
                        if (nb.op == SsaOp::FNeg && nb.args.size() == 1 &&
                            mod.get(nb.args[0]).op == SsaOp::FMul &&
                            mod.get(nb.args[0]).args.size() == 2) {
                            SsaId m0 = mod.get(nb.args[0]).args[0];
                            SsaId m1 = mod.get(nb.args[0]).args[1];
                            SsaId neg_m0 = emit_fneg_before(mod, blk, nid, m0);
                            ensure_remap();
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Fma;
                            mut.args = {neg_m0, m1, n.args[0]};
                            hit("F3_fuse_neg_fma");
                            continue;
                        }
                    }
                }

                // FMul rules (M11, I11, I12, I13, F7, R1, R7, R8, T8)
                if (n.op == SsaOp::FMul && is_f32_scalar(n) && n.args.size() == 2) {
                    auto is_valhall_persp_var = [&](SsaId id) -> bool {
                        if (id >= mod.nodes.size()) return false;
                        const auto& vn = mod.get(id);
                        if (vn.op != SsaOp::VecExtract || vn.args.empty()) return false;
                        const auto& base = mod.get(vn.args[0]);
                        return base.op == SsaOp::LoadFragVarying && !base.aux_bool && base.orig_inst_idx >= 0;
                    };
                    auto is_valhall_frag_w = [&](SsaId id) -> bool {
                        if (id >= mod.nodes.size()) return false;
                        const auto& wn = mod.get(id);
                        if (wn.op != SsaOp::VecExtract || wn.imm_u32 != 3u || wn.args.empty()) return false;
                        const auto& base = mod.get(wn.args[0]);
                        return base.op == SsaOp::LoadBuiltinInput && base.aux_str == "gl_FragCoord" && base.orig_inst_idx >= 0;
                    };
                    if (is_valhall_persp_var(n.args[0]) && is_valhall_frag_w(n.args[1])) {
                        replace_with(nid, n.args[0], "V1_split_persp_var_w");
                        continue;
                    }
                    if (is_valhall_persp_var(n.args[1]) && is_valhall_frag_w(n.args[0])) {
                        replace_with(nid, n.args[1], "V1_split_persp_var_w");
                        continue;
                    }
                    uint32_t rb = 0;
                    bool rc = try_get_const_f32_bits(mod, n.args[1], &rb);
                    if (rc && is_pos_one(rb)) {
                        replace_with(nid, n.args[0], "I11_fmul_1");
                        continue;
                    }
                    if (rc && is_neg_one(rb)) {
                        auto& mut = mod.get_mut(nid);
                        mut.op = SsaOp::FNeg;
                        mut.args = {n.args[0]};
                        hit("I12_fmul_neg1");
                        continue;
                    }
                    if (fast && rc && is_any_zero(rb)) {
                        rewrite_to_const_f32(nid, kPosZeroBits, "I13_fmul_0");
                        continue;
                    }
                    const auto& na = mod.get(n.args[0]);
                    const auto& nb = mod.get(n.args[1]);
                    if (na.op == SsaOp::FNeg && nb.op == SsaOp::FNeg &&
                        na.args.size() == 1 && nb.args.size() == 1) {
                        mod.get_mut(nid).args = {na.args[0], nb.args[0]};
                        hit("M11_fmul_neg_neg");
                        continue;
                    }
                    if (fast) {
                        // R1: FMul(a, FDiv(1.0, b)) -> FDiv(a, b) (Mali FRCP + FMUL)
                        SsaId denom = kInvalidSsa;
                        if (match_rcp(mod, n.args[1], &denom)) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::FDiv;
                            mut.args = {n.args[0], denom};
                            hit("R1_fmul_rcp");
                            continue;
                        }
                        if (match_rcp(mod, n.args[0], &denom)) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::FDiv;
                            mut.args = {n.args[1], denom};
                            hit("R1_fmul_rcp");
                            continue;
                        }
                        // R7: FMul(Sqrt(x), Sqrt(x)) -> FAbs(x)
                        if (na.op == SsaOp::Sqrt && nb.op == SsaOp::Sqrt &&
                            na.args.size() == 1 && nb.args.size() == 1 && na.args[0] == nb.args[0]) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::FAbs;
                            mut.args = {na.args[0]};
                            hit("R7_sqrt_sq");
                            continue;
                        }
                        // R8: FMul(x, InverseSqrt(x)) -> Sqrt(x)
                        if (nb.op == SsaOp::InverseSqrt && nb.args.size() == 1 && nb.args[0] == n.args[0]) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Sqrt;
                            mut.args = {n.args[0]};
                            hit("R8_mul_rsqrt");
                            continue;
                        }
                        if (na.op == SsaOp::InverseSqrt && na.args.size() == 1 && na.args[0] == n.args[1]) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Sqrt;
                            mut.args = {n.args[1]};
                            hit("R8_mul_rsqrt");
                            continue;
                        }
                        // T8: FMul(Exp2(a), Exp2(b)) -> Exp2(FAdd(a, b))
                        if (na.op == SsaOp::Exp2 && nb.op == SsaOp::Exp2 &&
                            na.args.size() == 1 && nb.args.size() == 1) {
                            SsaNode sum{};
                            sum.op = SsaOp::FAdd;
                            sum.type = ValType::F32;
                            sum.args = {na.args[0], nb.args[0]};
                            SsaId sum_id = insert_node_before(mod, blk, nid, std::move(sum));
                            ensure_remap();
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Exp2;
                            mut.args = {sum_id};
                            hit("T8_exp2_mul");
                            continue;
                        }
                        // F7: FMul(FMul(a, c1), c2) -> FMul(a, c1 * c2)
                        if (rc && na.op == SsaOp::FMul && na.args.size() == 2) {
                            uint32_t c1b = 0;
                            if (try_get_const_f32_bits(mod, na.args[1], &c1b)) {
                                SsaId inner_a = na.args[0];
                                uint32_t folded = f32_to_bits(bits_to_f32(c1b) * bits_to_f32(rb));
                                SsaId fc = emit_f32_const_before(mod, blk, nid, folded);
                                ensure_remap();
                                mod.get_mut(nid).args = {inner_a, fc};
                                hit("F7_fmul_reassoc_const");
                                continue;
                            }
                        }
                    }
                }

                // Fma rules (M12, I6, I7, I8, I9, I10)
                if (n.op == SsaOp::Fma && is_f32_scalar(n) && n.args.size() == 3) {
                    uint32_t b1 = 0, b2 = 0;
                    bool c1 = try_get_const_f32_bits(mod, n.args[1], &b1);
                    bool c2 = try_get_const_f32_bits(mod, n.args[2], &b2);

                    if (c2 && is_neg_zero(b2)) {
                        auto& mut = mod.get_mut(nid);
                        mut.op = SsaOp::FMul;
                        mut.args = {n.args[0], n.args[1]};
                        hit("I6_fma_neg0");
                        continue;
                    }
                    if (fast && c2 && is_pos_zero(b2)) {
                        auto& mut = mod.get_mut(nid);
                        mut.op = SsaOp::FMul;
                        mut.args = {n.args[0], n.args[1]};
                        hit("I7_fma_pos0");
                        continue;
                    }
                    if (c1 && is_pos_one(b1)) {
                        auto& mut = mod.get_mut(nid);
                        mut.op = SsaOp::FAdd;
                        mut.args = {n.args[0], n.args[2]};
                        hit("I8_fma_mul1");
                        continue;
                    }
                    if (c1 && is_neg_one(b1)) {
                        SsaId neg_a = emit_fneg_before(mod, blk, nid, n.args[0]);
                        ensure_remap();
                        auto& mut = mod.get_mut(nid);
                        mut.op = SsaOp::FAdd;
                        mut.args = {neg_a, n.args[2]};
                        hit("I9_fma_mul_neg1");
                        continue;
                    }
                    if (fast && c1 && is_any_zero(b1)) {
                        replace_with(nid, n.args[2], "I10_fma_mul0");
                        continue;
                    }
                    const auto& na = mod.get(n.args[0]);
                    const auto& nb = mod.get(n.args[1]);
                    if (na.op == SsaOp::FNeg && nb.op == SsaOp::FNeg &&
                        na.args.size() == 1 && nb.args.size() == 1) {
                        mod.get_mut(nid).args = {na.args[0], nb.args[0], n.args[2]};
                        hit("M12_fma_neg_neg");
                        continue;
                    }
                }

                // FDiv rules (R2, R3, R4, R5, R6)
                if (n.op == SsaOp::FDiv && is_f32_scalar(n) && n.args.size() == 2) {
                    uint32_t lb = 0, rb = 0;
                    bool lc = try_get_const_f32_bits(mod, n.args[0], &lb);
                    bool rc = try_get_const_f32_bits(mod, n.args[1], &rb);
                    if (rc && is_pos_one(rb)) {
                        replace_with(nid, n.args[0], "R6_fdiv_1");
                        continue;
                    }
                    if (rc && is_neg_one(rb)) {
                        auto& mut = mod.get_mut(nid);
                        mut.op = SsaOp::FNeg;
                        mut.args = {n.args[0]};
                        hit("R6_fdiv_neg1");
                        continue;
                    }
                    uint32_t rcp_bits = 0;
                    if (rc && is_exact_pow2_normal(rb, &rcp_bits)) {
                        SsaId rc_id = emit_f32_const_before(mod, blk, nid, rcp_bits);
                        ensure_remap();
                        auto& mut = mod.get_mut(nid);
                        mut.op = SsaOp::FMul;
                        mut.args = {n.args[0], rc_id};
                        hit("R5_fdiv_pow2");
                        continue;
                    }
                    if (fast && lc && is_pos_one(lb)) {
                        const auto& denom = mod.get(n.args[1]);
                        // R2: FDiv(1.0, InverseSqrt(x)) -> Sqrt(x) (Mali FRSQ + FRCP)
                        if (denom.op == SsaOp::InverseSqrt && denom.args.size() == 1) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Sqrt;
                            mut.args = {denom.args[0]};
                            hit("R2_rcp_rsqrt");
                            continue;
                        }
                        // R3: FDiv(1.0, Sqrt(x)) -> InverseSqrt(x)
                        if (denom.op == SsaOp::Sqrt && denom.args.size() == 1) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::InverseSqrt;
                            mut.args = {denom.args[0]};
                            hit("R3_rcp_sqrt");
                            continue;
                        }
                        // R4: FDiv(1.0, FDiv(1.0, x)) -> x
                        SsaId inner_x = kInvalidSsa;
                        if (match_rcp(mod, n.args[1], &inner_x)) {
                            replace_with(nid, inner_x, "R4_rcp_rcp");
                            continue;
                        }
                    }
                }

                // Ldexp rules (S1, S2, S3)
                if (n.op == SsaOp::Ldexp && is_f32_scalar(n) && n.args.size() == 2) {
                    uint32_t ku = 0;
                    if (try_get_const_u32(mod, n.args[1], &ku)) {
                        int32_t k = static_cast<int32_t>(ku);
                        if (k == 0) {
                            replace_with(nid, n.args[0], "S1_ldexp_0");
                            continue;
                        }
                        if (fast && k >= -126 && k <= 127) {
                            uint32_t pow2_bits = static_cast<uint32_t>(k + 127) << 23;
                            SsaId p2 = emit_f32_const_before(mod, blk, nid, pow2_bits);
                            ensure_remap();
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::FMul;
                            mut.args = {n.args[0], p2};
                            hit("S2_ldexp_to_fmul");
                            continue;
                        }
                    }
                }

                // Min / Max / Clamp lattice (C1-C8)
                if ((n.op == SsaOp::FMin || n.op == SsaOp::FMax ||
                     n.op == SsaOp::NMin || n.op == SsaOp::NMax) &&
                    is_f32_scalar(n) && n.args.size() == 2) {
                    if (n.args[0] == n.args[1]) {
                        replace_with(nid, n.args[0], "C4_minmax_self");
                        continue;
                    }
                    const bool is_min = (n.op == SsaOp::FMin || n.op == SsaOp::NMin);
                    const bool nan_suppress = (n.op == SsaOp::NMin || n.op == SsaOp::NMax);
                    const auto& na = mod.get(n.args[0]);
                    const auto& nb = mod.get(n.args[1]);

                    // C5: min(min(a, b), b) -> min(a, b)
                    if (na.op == n.op && na.args.size() == 2 &&
                        (na.args[0] == n.args[1] || na.args[1] == n.args[1])) {
                        replace_with(nid, n.args[0], "C5_minmax_absorb");
                        continue;
                    }
                    if (nb.op == n.op && nb.args.size() == 2 &&
                        (nb.args[0] == n.args[0] || nb.args[1] == n.args[0])) {
                        replace_with(nid, n.args[1], "C5_minmax_absorb");
                        continue;
                    }

                    // C1: NMin(NMax(x, lo), hi) -> FClamp(x, lo, hi)
                    if (is_min && (nan_suppress || fast)) {
                        SsaOp max_op = nan_suppress ? SsaOp::NMax : SsaOp::FMax;
                        uint32_t hi_b = 0, lo_b = 0;
                        if (try_get_const_f32_bits(mod, n.args[1], &hi_b) &&
                            na.op == max_op && na.args.size() == 2 &&
                            try_get_const_f32_bits(mod, na.args[1], &lo_b)) {
                            float fl = bits_to_f32(lo_b), fh = bits_to_f32(hi_b);
                            if (!std::isnan(fl) && !std::isnan(fh) && fl <= fh) {
                                auto& mut = mod.get_mut(nid);
                                mut.op = SsaOp::FClamp;
                                mut.args = {na.args[0], na.args[1], n.args[1]};
                                hit("C1_minmax_to_clamp");
                                continue;
                            }
                        }
                    }

                    // C3: NMax/NMin(FClamp(x, lo, hi), c) -> tightened FClamp
                    if (na.op == SsaOp::FClamp && na.args.size() == 3 && (nan_suppress || fast)) {
                        uint32_t lo_b = 0, hi_b = 0, cb = 0;
                        if (try_get_const_f32_bits(mod, na.args[1], &lo_b) &&
                            try_get_const_f32_bits(mod, na.args[2], &hi_b) &&
                            try_get_const_f32_bits(mod, n.args[1], &cb)) {
                            float fl = bits_to_f32(lo_b), fh = bits_to_f32(hi_b), fc = bits_to_f32(cb);
                            if (!std::isnan(fl) && !std::isnan(fh) && !std::isnan(fc)) {
                                if (!is_min && fc <= fl) {
                                    replace_with(nid, n.args[0], "C3_clamp_redundant_max");
                                    continue;
                                }
                                if (is_min && fc >= fh) {
                                    replace_with(nid, n.args[0], "C3_clamp_redundant_min");
                                    continue;
                                }
                            }
                        }
                    }

                    // C8: NMax(FAbs(a), 0.0) -> FAbs(a)
                    if (!is_min && (nan_suppress || fast) && na.op == SsaOp::FAbs) {
                        uint32_t cb = 0;
                        if (try_get_const_f32_bits(mod, n.args[1], &cb) && is_any_zero(cb)) {
                            replace_with(nid, n.args[0], "C8_max_fabs_0");
                            continue;
                        }
                    }
                }

                // C2: Nested FClamp(FClamp(x, lo1, hi1), lo2, hi2) -> FClamp(x, max(lo), min(hi))
                if (n.op == SsaOp::FClamp && is_f32_scalar(n) && n.args.size() == 3) {
                    const auto& inner = mod.get(n.args[0]);
                    if (inner.op == SsaOp::FClamp && inner.args.size() == 3) {
                        uint32_t lo1_b = 0, hi1_b = 0, lo2_b = 0, hi2_b = 0;
                        if (try_get_const_f32_bits(mod, inner.args[1], &lo1_b) &&
                            try_get_const_f32_bits(mod, inner.args[2], &hi1_b) &&
                            try_get_const_f32_bits(mod, n.args[1], &lo2_b) &&
                            try_get_const_f32_bits(mod, n.args[2], &hi2_b)) {
                            float lo1 = bits_to_f32(lo1_b), hi1 = bits_to_f32(hi1_b);
                            float lo2 = bits_to_f32(lo2_b), hi2 = bits_to_f32(hi2_b);
                            float nlo = std::fmax(lo1, lo2);
                            float nhi = std::fmin(hi1, hi2);
                            if (!std::isnan(nlo) && !std::isnan(nhi) && nlo <= nhi) {
                                SsaId inner_x = inner.args[0];
                                SsaId lo_id = emit_f32_const_before(mod, blk, nid, f32_to_bits(nlo));
                                SsaId hi_id = emit_f32_const_before(mod, blk, nid, f32_to_bits(nhi));
                                ensure_remap();
                                mod.get_mut(nid).args = {inner_x, lo_id, hi_id};
                                hit("C2_nested_fclamp");
                                continue;
                            }
                        }
                    }
                }

                // Transcendentals (T1-T6, FastMath)
                if (fast && is_f32_scalar(n)) {
                    if (n.op == SsaOp::Exp2 && n.args.size() == 1) {
                        const auto& arg = mod.get(n.args[0]);
                        if (arg.op == SsaOp::Log2 && arg.args.size() == 1) {
                            replace_with(nid, arg.args[0], "T1_exp2_log2");
                            continue;
                        }
                        if (arg.op == SsaOp::FMul && arg.args.size() == 2) {
                            const auto& m0 = mod.get(arg.args[0]);
                            const auto& m1 = mod.get(arg.args[1]);
                            if (m0.op == SsaOp::Log2 && m0.args.size() == 1) {
                                auto& mut = mod.get_mut(nid);
                                mut.op = SsaOp::Pow;
                                mut.args = {m0.args[0], arg.args[1]};
                                hit("T6_exp2_mul_log2_to_pow");
                                continue;
                            }
                            if (m1.op == SsaOp::Log2 && m1.args.size() == 1) {
                                auto& mut = mod.get_mut(nid);
                                mut.op = SsaOp::Pow;
                                mut.args = {m1.args[0], arg.args[0]};
                                hit("T6_exp2_mul_log2_to_pow");
                                continue;
                            }
                        }
                    } else if (n.op == SsaOp::Log2 && n.args.size() == 1) {
                        const auto& arg = mod.get(n.args[0]);
                        if (arg.op == SsaOp::Exp2 && arg.args.size() == 1) {
                            replace_with(nid, arg.args[0], "T1_log2_exp2");
                            continue;
                        }
                    } else if (n.op == SsaOp::Pow && n.args.size() == 2) {
                        uint32_t lb = 0, rb = 0;
                        bool lc = try_get_const_f32_bits(mod, n.args[0], &lb);
                        bool rc = try_get_const_f32_bits(mod, n.args[1], &rb);
                        if (rc && is_pos_one(rb)) {
                            replace_with(nid, n.args[0], "T2_pow_1");
                            continue;
                        }
                        if (rc && is_pos_two(rb)) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::FMul;
                            mut.args = {n.args[0], n.args[0]};
                            hit("T3_pow_2");
                            continue;
                        }
                        if (rc && is_pos_half(rb)) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Sqrt;
                            mut.args = {n.args[0]};
                            hit("T5_pow_half");
                            continue;
                        }
                        if (lc && is_pos_two(lb)) {
                            auto& mut = mod.get_mut(nid);
                            mut.op = SsaOp::Exp2;
                            mut.args = {n.args[1]};
                            hit("T4_pow_base2");
                            continue;
                        }
                    }
                }

                // FP comparisons (P1-P6, T1)
                if (is_fcmp_op(n.op) && n.args.size() == 2) {
                    const auto& na = mod.get(n.args[0]);
                    const auto& nb = mod.get(n.args[1]);
                    // P1: FCmp(-a, -b) -> FCmp(b, a)
                    if (na.op == SsaOp::FNeg && nb.op == SsaOp::FNeg &&
                        na.args.size() == 1 && nb.args.size() == 1) {
                        auto& mut = mod.get_mut(nid);
                        mut.args = {nb.args[0], na.args[0]};
                        hit("P1_fcmp_neg_neg");
                        continue;
                    }
                    // P2: FCmp(-a, c) -> FCmp_flipped(a, -c)
                    uint32_t rb = 0;
                    if (na.op == SsaOp::FNeg && na.args.size() == 1 &&
                        try_get_const_f32_bits(mod, n.args[1], &rb)) {
                        SsaId inner_a = na.args[0];
                        SsaId nc = emit_f32_const_before(mod, blk, nid, rb ^ 0x80000000u);
                        ensure_remap();
                        auto& mut = mod.get_mut(nid);
                        mut.op = flip_fcmp_direction(n.op);
                        mut.args = {inner_a, nc};
                        hit("P2_fcmp_neg_const");
                        continue;
                    }
                    // P3: FCmpOrdEq/UnordNe(FAbs(a), 0.0) -> FCmp*(a, 0.0)
                    if ((n.op == SsaOp::FCmpOrdEq || n.op == SsaOp::FCmpUnordNe) &&
                        na.op == SsaOp::FAbs && na.args.size() == 1 &&
                        try_get_const_f32_bits(mod, n.args[1], &rb) && is_any_zero(rb)) {
                        mod.get_mut(nid).args[0] = na.args[0];
                        hit("P3_fcmp_fabs_0");
                        continue;
                    }
                    // P4: FCmpOrdLt/Gt(FAdd(a, b), 0.0) -> FCmpOrdLt/Gt(a, FNeg(b))
                    if ((n.op == SsaOp::FCmpOrdLt || n.op == SsaOp::FCmpOrdGt) &&
                        na.op == SsaOp::FAdd && na.args.size() == 2 &&
                        try_get_const_f32_bits(mod, n.args[1], &rb) && is_any_zero(rb)) {
                        SsaId inner_a = na.args[0];
                        SsaId inner_b = na.args[1];
                        SsaId neg_b = emit_fneg_before(mod, blk, nid, inner_b);
                        ensure_remap();
                        mod.get_mut(nid).args = {inner_a, neg_b};
                        hit("P4_fcmp_fadd_0");
                        continue;
                    }
                }
            }
        }

        if (any_remapped) {
            apply_value_remap(mod, remap);
            result.stats.nodes_removed += erase_dead_nodes(mod, dead);
        }
        if (!changed) break;
    }

    if (opts.canonicalize_vertex_viewport && opts.enable_cpp_rewrites) {
        canonicalize_vertex_viewport_epilogue(mod, result);
    }
    return result;
}

}  // namespace valhall::dataflow
