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

}  // namespace

FpOptResult run_ssa_fp_canonicalize(ValhallSsaModule& mod, const FpOptOptions& opts) {
    FpOptResult result;
    if (mod.blocks.empty()) return result;

    const bool fast = (opts.mode == FpOptMode::FastMath);

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

                const SsaNode n = mod.get(nid);
                const uint32_t blk = static_cast<uint32_t>(b);

                // Sign-bit bitwise idioms -> FAbs / FNeg / FNeg(FAbs) (M5-M7, T1)
                if (n.op == SsaOp::Bitcast && is_f32_scalar(n) && n.args.size() == 1) {
                    const auto& inner = mod.get(n.args[0]);
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

    return result;
}

}  // namespace valhall::dataflow
