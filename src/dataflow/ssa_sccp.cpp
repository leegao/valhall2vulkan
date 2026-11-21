#include "ssa_sccp.h"

#include "ssa_dce.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <set>
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

enum class LatticeState : uint8_t {
    Undef = 0,
    Constant,
    Overdefined,
};

struct SccpVal {
    LatticeState state = LatticeState::Undef;
    ValType type = ValType::Void;
    uint64_t bits = 0;

    static SccpVal undef() { return SccpVal{}; }
    static SccpVal overdefined() {
        SccpVal v;
        v.state = LatticeState::Overdefined;
        return v;
    }
    static SccpVal make_u32(uint32_t x, ValType t = ValType::U32) {
        SccpVal v;
        v.state = LatticeState::Constant;
        v.type = t;
        v.bits = x;
        return v;
    }
    static SccpVal make_u64(uint64_t x, ValType t = ValType::U64) {
        SccpVal v;
        v.state = LatticeState::Constant;
        v.type = t;
        v.bits = x;
        return v;
    }
    static SccpVal make_bool(bool b) {
        SccpVal v;
        v.state = LatticeState::Constant;
        v.type = ValType::Bool;
        v.bits = b ? 1ULL : 0ULL;
        return v;
    }
    static SccpVal make_f32_bits(uint32_t b) {
        SccpVal v;
        v.state = LatticeState::Constant;
        v.type = ValType::F32;
        v.bits = b;
        return v;
    }

    bool is_undef() const { return state == LatticeState::Undef; }
    bool is_const() const { return state == LatticeState::Constant; }
    bool is_overdefined() const { return state == LatticeState::Overdefined; }

    bool operator==(const SccpVal& o) const {
        if (state != o.state) return false;
        if (state != LatticeState::Constant) return true;
        return bits == o.bits;
    }
    bool operator!=(const SccpVal& o) const { return !(*this == o); }
};

SccpVal meet_sccp(const SccpVal& a, const SccpVal& b) {
    if (a.is_undef()) return b;
    if (b.is_undef()) return a;
    if (a.is_overdefined() || b.is_overdefined()) return SccpVal::overdefined();
    if (a.bits == b.bits) return a;
    return SccpVal::overdefined();
}

uint32_t clz32(uint32_t x) {
    if (x == 0) return 32;
    return static_cast<uint32_t>(__builtin_clz(x));
}

uint32_t popcount32(uint32_t x) {
    return static_cast<uint32_t>(__builtin_popcount(x));
}

uint32_t bitreverse32(uint32_t x) {
    uint32_t r = 0;
    for (int i = 0; i < 32; ++i) {
        r = (r << 1) | ((x >> i) & 1u);
    }
    return r;
}

uint32_t rotl32(uint32_t x, uint32_t sh) {
    sh &= 31u;
    return sh == 0 ? x : ((x << sh) | (x >> (32u - sh)));
}

uint32_t rotr32(uint32_t x, uint32_t sh) {
    sh &= 31u;
    return sh == 0 ? x : ((x >> sh) | (x << (32u - sh)));
}

bool is_materializable_scalar_type(ValType t) {
    switch (t) {
        case ValType::Bool:
        case ValType::U32:
        case ValType::S32:
        case ValType::F32:
        case ValType::U64:
        case ValType::S64:
            return true;
        default:
            return false;
    }
}

bool is_const_op(SsaOp op) {
    return op == SsaOp::ConstBool || op == SsaOp::ConstU32 ||
           op == SsaOp::ConstS32 || op == SsaOp::ConstF32 ||
           op == SsaOp::ConstU64;
}

/// Evaluates a single non-Phi SSA node under the current SCCP lattice map.
SccpVal eval_node_sccp(const ValhallSsaModule& mod,
                       const SsaNode& n,
                       const std::vector<SccpVal>& lat,
                       bool fold_fp) {
    auto get = [&](size_t idx) -> SccpVal {
        if (idx >= n.args.size()) return SccpVal::overdefined();
        SsaId id = n.args[idx];
        if (id == kInvalidSsa || id >= lat.size()) return SccpVal::overdefined();
        return lat[id];
    };

    switch (n.op) {
        case SsaOp::ConstBool: return SccpVal::make_bool(n.imm_u32 != 0);
        case SsaOp::ConstU32:  return SccpVal::make_u32(n.imm_u32, ValType::U32);
        case SsaOp::ConstS32:  return SccpVal::make_u32(n.imm_u32, ValType::S32);
        case SsaOp::ConstF32:  return SccpVal::make_f32_bits(n.imm_u32);
        case SsaOp::ConstU64:  return SccpVal::make_u64(n.imm_u64, ValType::U64);

        case SsaOp::Copy: {
            SccpVal a = get(0);
            if (a.is_const()) a.type = n.type;
            return a;
        }
        case SsaOp::Bitcast: {
            SccpVal a = get(0);
            if (!a.is_const()) return a;
            if (!is_materializable_scalar_type(n.type)) return SccpVal::overdefined();
            a.type = n.type;
            return a;
        }
        case SsaOp::Select: {
            SccpVal c = get(0);
            SccpVal t = get(1);
            SccpVal f = get(2);
            if (n.args.size() >= 3 && n.args[1] == n.args[2]) return t;
            if (c.is_const()) {
                return c.bits ? t : f;
            }
            if (c.is_undef()) return SccpVal::undef();
            if (t.is_const() && f.is_const() && t.bits == f.bits) return t;
            return SccpVal::overdefined();
        }

        // Vector extract from VecConstruct*
        case SsaOp::VecExtract: {
            if (n.args.empty()) return SccpVal::overdefined();
            SsaId vec_id = n.args[0];
            if (vec_id == kInvalidSsa || vec_id >= mod.nodes.size()) return SccpVal::overdefined();
            const auto& vn = mod.get(vec_id);
            if ((vn.op == SsaOp::VecConstruct2 || vn.op == SsaOp::VecConstruct3 || vn.op == SsaOp::VecConstruct4) &&
                n.imm_u32 < vn.args.size()) {
                SsaId elem = vn.args[n.imm_u32];
                if (elem != kInvalidSsa && elem < lat.size()) {
                    return lat[elem];
                }
            }
            return SccpVal::overdefined();
        }

        // 64-bit Pack / Split / TupleExtract
        case SsaOp::Pack64: {
            SccpVal lo = get(0), hi = get(1);
            if (lo.is_overdefined() || hi.is_overdefined()) return SccpVal::overdefined();
            if (lo.is_undef() || hi.is_undef()) return SccpVal::undef();
            uint64_t v = (static_cast<uint64_t>(hi.bits & 0xFFFFFFFFULL) << 32) |
                         (lo.bits & 0xFFFFFFFFULL);
            return SccpVal::make_u64(v, n.type);
        }
        case SsaOp::TupleExtract: {
            if (n.args.empty()) return SccpVal::overdefined();
            SsaId tup_id = n.args[0];
            if (tup_id == kInvalidSsa || tup_id >= mod.nodes.size()) return SccpVal::overdefined();
            const auto& tn = mod.get(tup_id);
            if (tn.op == SsaOp::Split64 && !tn.args.empty()) {
                SsaId src64 = tn.args[0];
                if (src64 != kInvalidSsa && src64 < lat.size()) {
                    const auto& s = lat[src64];
                    if (!s.is_const()) return s;
                    uint32_t lane = (n.imm_u32 == 0)
                                        ? static_cast<uint32_t>(s.bits & 0xFFFFFFFFULL)
                                        : static_cast<uint32_t>((s.bits >> 32) & 0xFFFFFFFFULL);
                    return SccpVal::make_u32(lane, n.type);
                }
            }
            return SccpVal::overdefined();
        }
        case SsaOp::UConvert32To64: {
            SccpVal a = get(0);
            if (!a.is_const()) return a;
            return SccpVal::make_u64(a.bits & 0xFFFFFFFFULL, n.type);
        }
        case SsaOp::SConvert32To64: {
            SccpVal a = get(0);
            if (!a.is_const()) return a;
            int64_t sx = static_cast<int64_t>(static_cast<int32_t>(a.bits & 0xFFFFFFFFULL));
            return SccpVal::make_u64(static_cast<uint64_t>(sx), n.type);
        }
        case SsaOp::UConvert64To32: {
            SccpVal a = get(0);
            if (!a.is_const()) return a;
            return SccpVal::make_u32(static_cast<uint32_t>(a.bits & 0xFFFFFFFFULL), n.type);
        }

        // Boolean logic
        case SsaOp::LogicalNot: {
            SccpVal a = get(0);
            if (!a.is_const()) return a;
            return SccpVal::make_bool(a.bits == 0);
        }
        case SsaOp::LogicalAnd: {
            SccpVal a = get(0), b = get(1);
            if ((a.is_const() && a.bits == 0) || (b.is_const() && b.bits == 0)) {
                return SccpVal::make_bool(false);
            }
            if (a.is_overdefined() || b.is_overdefined()) return SccpVal::overdefined();
            if (a.is_undef() || b.is_undef()) return SccpVal::undef();
            return SccpVal::make_bool(a.bits != 0 && b.bits != 0);
        }
        case SsaOp::LogicalOr: {
            SccpVal a = get(0), b = get(1);
            if ((a.is_const() && a.bits != 0) || (b.is_const() && b.bits != 0)) {
                return SccpVal::make_bool(true);
            }
            if (a.is_overdefined() || b.is_overdefined()) return SccpVal::overdefined();
            if (a.is_undef() || b.is_undef()) return SccpVal::undef();
            return SccpVal::make_bool(a.bits != 0 || b.bits != 0);
        }

        // 32-bit integer unary ops
        case SsaOp::BitNot:
        case SsaOp::SNeg:
        case SsaOp::SAbs:
        case SsaOp::Clz:
        case SsaOp::BitCount:
        case SsaOp::BitReverse: {
            SccpVal a = get(0);
            if (!a.is_const()) return a;
            uint32_t x = static_cast<uint32_t>(a.bits);
            uint32_t r = 0;
            switch (n.op) {
                case SsaOp::BitNot:     r = ~x; break;
                case SsaOp::SNeg:       r = static_cast<uint32_t>(-static_cast<int32_t>(x)); break;
                case SsaOp::SAbs: {
                    int32_t sx = static_cast<int32_t>(x);
                    r = sx < 0 ? static_cast<uint32_t>(0u - x) : x;
                    break;
                }
                case SsaOp::Clz:        r = clz32(x); break;
                case SsaOp::BitCount:   r = popcount32(x); break;
                case SsaOp::BitReverse: r = bitreverse32(x); break;
                default: break;
            }
            return SccpVal::make_u32(r, n.type);
        }

        // 32-bit integer binary ops (including absorbing zero short-circuits)
        case SsaOp::IAdd:
        case SsaOp::ISub:
        case SsaOp::IMul:
        case SsaOp::BitAnd:
        case SsaOp::BitOr:
        case SsaOp::BitXor:
        case SsaOp::Shl:
        case SsaOp::ShrU:
        case SsaOp::ShrS:
        case SsaOp::RotL:
        case SsaOp::RotR:
        case SsaOp::UDiv:
        case SsaOp::SDiv:
        case SsaOp::UMod:
        case SsaOp::SRem:
        case SsaOp::SMod: {
            SccpVal a = get(0), b = get(1);
            if (n.op == SsaOp::IMul || n.op == SsaOp::BitAnd) {
                if ((a.is_const() && static_cast<uint32_t>(a.bits) == 0u) ||
                    (b.is_const() && static_cast<uint32_t>(b.bits) == 0u)) {
                    return SccpVal::make_u32(0u, n.type);
                }
            }
            if (n.op == SsaOp::BitOr) {
                if ((a.is_const() && static_cast<uint32_t>(a.bits) == 0xFFFFFFFFu) ||
                    (b.is_const() && static_cast<uint32_t>(b.bits) == 0xFFFFFFFFu)) {
                    return SccpVal::make_u32(0xFFFFFFFFu, n.type);
                }
            }
            if (n.args.size() == 2 && n.args[0] == n.args[1]) {
                if (n.op == SsaOp::ISub || n.op == SsaOp::BitXor) {
                    return SccpVal::make_u32(0u, n.type);
                }
            }
            if (a.is_overdefined() || b.is_overdefined()) return SccpVal::overdefined();
            if (a.is_undef() || b.is_undef()) return SccpVal::undef();

            uint32_t x = static_cast<uint32_t>(a.bits);
            uint32_t y = static_cast<uint32_t>(b.bits);
            int32_t sx = static_cast<int32_t>(x);
            int32_t sy = static_cast<int32_t>(y);
            uint32_t r = 0;
            switch (n.op) {
                case SsaOp::IAdd:   r = x + y; break;
                case SsaOp::ISub:   r = x - y; break;
                case SsaOp::IMul:   r = x * y; break;
                case SsaOp::BitAnd: r = x & y; break;
                case SsaOp::BitOr:  r = x | y; break;
                case SsaOp::BitXor: r = x ^ y; break;
                case SsaOp::Shl:    r = x << (y & 31u); break;
                case SsaOp::ShrU:   r = x >> (y & 31u); break;
                case SsaOp::ShrS:   r = static_cast<uint32_t>(sx >> (y & 31u)); break;
                case SsaOp::RotL:   r = rotl32(x, y); break;
                case SsaOp::RotR:   r = rotr32(x, y); break;
                case SsaOp::UDiv:
                    if (y == 0) return SccpVal::overdefined();
                    r = x / y;
                    break;
                case SsaOp::UMod:
                    if (y == 0) return SccpVal::overdefined();
                    r = x % y;
                    break;
                case SsaOp::SDiv:
                    if (sy == 0 || (sx == std::numeric_limits<int32_t>::min() && sy == -1)) {
                        return SccpVal::overdefined();
                    }
                    r = static_cast<uint32_t>(sx / sy);
                    break;
                case SsaOp::SRem:
                    if (sy == 0 || (sx == std::numeric_limits<int32_t>::min() && sy == -1)) {
                        return SccpVal::overdefined();
                    }
                    r = static_cast<uint32_t>(sx % sy);
                    break;
                case SsaOp::SMod: {
                    if (sy == 0 || (sx == std::numeric_limits<int32_t>::min() && sy == -1)) {
                        return SccpVal::overdefined();
                    }
                    int32_t rem = sx % sy;
                    if (rem != 0 && ((rem ^ sy) < 0)) rem += sy;
                    r = static_cast<uint32_t>(rem);
                    break;
                }
                default:
                    return SccpVal::overdefined();
            }
            return SccpVal::make_u32(r, n.type);
        }

        // Bitfield extract/insert
        case SsaOp::BfeU:
        case SsaOp::BfeS: {
            SccpVal a = get(0), o = get(1), w = get(2);
            if (a.is_overdefined() || o.is_overdefined() || w.is_overdefined()) return SccpVal::overdefined();
            if (a.is_undef() || o.is_undef() || w.is_undef()) return SccpVal::undef();
            uint32_t x = static_cast<uint32_t>(a.bits);
            uint32_t off = static_cast<uint32_t>(o.bits) & 31u;
            uint32_t width = static_cast<uint32_t>(w.bits);
            if (width == 0) return SccpVal::make_u32(0u, n.type);
            if (width >= 32) return SccpVal::make_u32(x, n.type);
            uint32_t mask = (1u << width) - 1u;
            uint32_t raw = (x >> off) & mask;
            if (n.op == SsaOp::BfeS && (raw & (1u << (width - 1u)))) {
                raw |= ~mask;
            }
            return SccpVal::make_u32(raw, n.type);
        }
        case SsaOp::Bfi: {
            SccpVal base = get(0), ins = get(1), o = get(2), w = get(3);
            if (base.is_overdefined() || ins.is_overdefined() ||
                o.is_overdefined() || w.is_overdefined()) {
                return SccpVal::overdefined();
            }
            if (base.is_undef() || ins.is_undef() || o.is_undef() || w.is_undef()) {
                return SccpVal::undef();
            }
            uint32_t b = static_cast<uint32_t>(base.bits);
            uint32_t iv = static_cast<uint32_t>(ins.bits);
            uint32_t off = static_cast<uint32_t>(o.bits) & 31u;
            uint32_t width = static_cast<uint32_t>(w.bits);
            if (width == 0) return SccpVal::make_u32(b, n.type);
            if (width >= 32) return SccpVal::make_u32(iv, n.type);
            uint32_t mask = ((1u << width) - 1u) << off;
            uint32_t res = (b & ~mask) | ((iv << off) & mask);
            return SccpVal::make_u32(res, n.type);
        }

        // 64-bit integer ops
        case SsaOp::BitNot64: {
            SccpVal a = get(0);
            if (!a.is_const()) return a;
            return SccpVal::make_u64(~a.bits, n.type);
        }
        case SsaOp::IAdd64:
        case SsaOp::ISub64:
        case SsaOp::IMul64:
        case SsaOp::BitAnd64:
        case SsaOp::BitOr64:
        case SsaOp::BitXor64:
        case SsaOp::Shl64:
        case SsaOp::ShrU64:
        case SsaOp::ShrS64: {
            SccpVal a = get(0), b = get(1);
            if (n.op == SsaOp::IMul64 || n.op == SsaOp::BitAnd64) {
                if ((a.is_const() && a.bits == 0ULL) || (b.is_const() && b.bits == 0ULL)) {
                    return SccpVal::make_u64(0ULL, n.type);
                }
            }
            if (a.is_overdefined() || b.is_overdefined()) return SccpVal::overdefined();
            if (a.is_undef() || b.is_undef()) return SccpVal::undef();
            uint64_t x = a.bits, y = b.bits, r = 0;
            switch (n.op) {
                case SsaOp::IAdd64:   r = x + y; break;
                case SsaOp::ISub64:   r = x - y; break;
                case SsaOp::IMul64:   r = x * y; break;
                case SsaOp::BitAnd64: r = x & y; break;
                case SsaOp::BitOr64:  r = x | y; break;
                case SsaOp::BitXor64: r = x ^ y; break;
                case SsaOp::Shl64:    r = x << (y & 63ULL); break;
                case SsaOp::ShrU64:   r = x >> (y & 63ULL); break;
                case SsaOp::ShrS64:   r = static_cast<uint64_t>(static_cast<int64_t>(x) >> (y & 63ULL)); break;
                default: break;
            }
            return SccpVal::make_u64(r, n.type);
        }

        // Integer comparisons
        case SsaOp::ICmpEq:
        case SsaOp::ICmpNe:
        case SsaOp::ICmpSLt:
        case SsaOp::ICmpSLe:
        case SsaOp::ICmpSGt:
        case SsaOp::ICmpSGe:
        case SsaOp::ICmpULt:
        case SsaOp::ICmpULe:
        case SsaOp::ICmpUGt:
        case SsaOp::ICmpUGe: {
            if (n.args.size() == 2 && n.args[0] == n.args[1]) {
                bool eq_true = (n.op == SsaOp::ICmpEq || n.op == SsaOp::ICmpSLe ||
                                n.op == SsaOp::ICmpSGe || n.op == SsaOp::ICmpULe ||
                                n.op == SsaOp::ICmpUGe);
                return SccpVal::make_bool(eq_true);
            }
            SccpVal a = get(0), b = get(1);
            if (a.is_overdefined() || b.is_overdefined()) return SccpVal::overdefined();
            if (a.is_undef() || b.is_undef()) return SccpVal::undef();
            uint32_t x = static_cast<uint32_t>(a.bits);
            uint32_t y = static_cast<uint32_t>(b.bits);
            int32_t sx = static_cast<int32_t>(x);
            int32_t sy = static_cast<int32_t>(y);
            bool r = false;
            switch (n.op) {
                case SsaOp::ICmpEq:  r = (x == y); break;
                case SsaOp::ICmpNe:  r = (x != y); break;
                case SsaOp::ICmpSLt: r = (sx < sy); break;
                case SsaOp::ICmpSLe: r = (sx <= sy); break;
                case SsaOp::ICmpSGt: r = (sx > sy); break;
                case SsaOp::ICmpSGe: r = (sx >= sy); break;
                case SsaOp::ICmpULt: r = (x < y); break;
                case SsaOp::ICmpULe: r = (x <= y); break;
                case SsaOp::ICmpUGt: r = (x > y); break;
                case SsaOp::ICmpUGe: r = (x >= y); break;
                default: break;
            }
            return SccpVal::make_bool(r);
        }

        // Floating-point constant folding (IEEE-754 binary32)
        case SsaOp::FAbs:
        case SsaOp::FNeg:
        case SsaOp::Floor:
        case SsaOp::Ceil:
        case SsaOp::Trunc:
        case SsaOp::RoundEven:
        case SsaOp::IsNan:
        case SsaOp::IsInf:
        case SsaOp::ConvertSToF:
        case SsaOp::ConvertUToF:
        case SsaOp::ConvertFToS:
        case SsaOp::ConvertFToU: {
            if (!fold_fp) return SccpVal::overdefined();
            SccpVal a = get(0);
            if (!a.is_const()) return a;
            uint32_t abits = static_cast<uint32_t>(a.bits);
            float fa = bits_to_f32(abits);
            switch (n.op) {
                case SsaOp::FAbs:        return SccpVal::make_f32_bits(abits & 0x7FFFFFFFu);
                case SsaOp::FNeg:        return SccpVal::make_f32_bits(abits ^ 0x80000000u);
                case SsaOp::Floor:       return SccpVal::make_f32_bits(f32_to_bits(std::floor(fa)));
                case SsaOp::Ceil:        return SccpVal::make_f32_bits(f32_to_bits(std::ceil(fa)));
                case SsaOp::Trunc:       return SccpVal::make_f32_bits(f32_to_bits(std::trunc(fa)));
                case SsaOp::RoundEven:   return SccpVal::make_f32_bits(f32_to_bits(std::nearbyint(fa)));
                case SsaOp::IsNan:       return SccpVal::make_bool(std::isnan(fa));
                case SsaOp::IsInf:       return SccpVal::make_bool(std::isinf(fa));
                case SsaOp::ConvertSToF: return SccpVal::make_f32_bits(f32_to_bits(static_cast<float>(static_cast<int32_t>(abits))));
                case SsaOp::ConvertUToF: return SccpVal::make_f32_bits(f32_to_bits(static_cast<float>(abits)));
                case SsaOp::ConvertFToS:
                    if (!std::isfinite(fa)) return SccpVal::overdefined();
                    return SccpVal::make_u32(static_cast<uint32_t>(static_cast<int32_t>(fa)), ValType::S32);
                case SsaOp::ConvertFToU:
                    if (!std::isfinite(fa) || fa < 0.0f) return SccpVal::overdefined();
                    return SccpVal::make_u32(static_cast<uint32_t>(fa), ValType::U32);
                default: break;
            }
            return SccpVal::overdefined();
        }

        case SsaOp::FAdd:
        case SsaOp::FSub:
        case SsaOp::FMul:
        case SsaOp::FDiv:
        case SsaOp::FMin:
        case SsaOp::FMax:
        case SsaOp::NMin:
        case SsaOp::NMax:
        case SsaOp::Ldexp:
        case SsaOp::FCmpOrdEq:
        case SsaOp::FCmpOrdNe:
        case SsaOp::FCmpOrdLt:
        case SsaOp::FCmpOrdLe:
        case SsaOp::FCmpOrdGt:
        case SsaOp::FCmpOrdGe:
        case SsaOp::FCmpUnordNe: {
            if (!fold_fp) return SccpVal::overdefined();
            SccpVal a = get(0), b = get(1);
            if (a.is_overdefined() || b.is_overdefined()) return SccpVal::overdefined();
            if (a.is_undef() || b.is_undef()) return SccpVal::undef();
            float fa = bits_to_f32(static_cast<uint32_t>(a.bits));
            float fb = bits_to_f32(static_cast<uint32_t>(b.bits));
            switch (n.op) {
                case SsaOp::FAdd: return SccpVal::make_f32_bits(f32_to_bits(fa + fb));
                case SsaOp::FSub: return SccpVal::make_f32_bits(f32_to_bits(fa - fb));
                case SsaOp::FMul: return SccpVal::make_f32_bits(f32_to_bits(fa * fb));
                case SsaOp::FDiv: return SccpVal::make_f32_bits(f32_to_bits(fa / fb));
                case SsaOp::FMin:
                    if (std::isnan(fa)) return SccpVal::make_f32_bits(static_cast<uint32_t>(a.bits));
                    if (std::isnan(fb)) return SccpVal::make_f32_bits(static_cast<uint32_t>(b.bits));
                    return SccpVal::make_f32_bits(fa < fb ? static_cast<uint32_t>(a.bits) : static_cast<uint32_t>(b.bits));
                case SsaOp::FMax:
                    if (std::isnan(fa)) return SccpVal::make_f32_bits(static_cast<uint32_t>(a.bits));
                    if (std::isnan(fb)) return SccpVal::make_f32_bits(static_cast<uint32_t>(b.bits));
                    return SccpVal::make_f32_bits(fa > fb ? static_cast<uint32_t>(a.bits) : static_cast<uint32_t>(b.bits));
                case SsaOp::NMin:
                    return SccpVal::make_f32_bits(f32_to_bits(std::fmin(fa, fb)));
                case SsaOp::NMax:
                    return SccpVal::make_f32_bits(f32_to_bits(std::fmax(fa, fb)));
                case SsaOp::Ldexp:
                    return SccpVal::make_f32_bits(f32_to_bits(std::ldexp(fa, static_cast<int>(static_cast<int32_t>(b.bits)))));
                case SsaOp::FCmpOrdEq:   return SccpVal::make_bool(fa == fb);
                case SsaOp::FCmpOrdNe:   return SccpVal::make_bool(!std::isnan(fa) && !std::isnan(fb) && fa != fb);
                case SsaOp::FCmpOrdLt:   return SccpVal::make_bool(fa < fb);
                case SsaOp::FCmpOrdLe:   return SccpVal::make_bool(fa <= fb);
                case SsaOp::FCmpOrdGt:   return SccpVal::make_bool(fa > fb);
                case SsaOp::FCmpOrdGe:   return SccpVal::make_bool(fa >= fb);
                case SsaOp::FCmpUnordNe: return SccpVal::make_bool(!(fa == fb));
                default: break;
            }
            return SccpVal::overdefined();
        }

        case SsaOp::Fma:
        case SsaOp::FClamp: {
            if (!fold_fp) return SccpVal::overdefined();
            SccpVal a = get(0), b = get(1), c = get(2);
            if (a.is_overdefined() || b.is_overdefined() || c.is_overdefined()) {
                return SccpVal::overdefined();
            }
            if (a.is_undef() || b.is_undef() || c.is_undef()) {
                return SccpVal::undef();
            }
            float fa = bits_to_f32(static_cast<uint32_t>(a.bits));
            float fb = bits_to_f32(static_cast<uint32_t>(b.bits));
            float fc = bits_to_f32(static_cast<uint32_t>(c.bits));
            if (n.op == SsaOp::Fma) {
                return SccpVal::make_f32_bits(f32_to_bits(std::fma(fa, fb, fc)));
            }
            float clamped = std::fmin(std::fmax(fa, fb), fc);
            return SccpVal::make_f32_bits(f32_to_bits(clamped));
        }

        default:
            return SccpVal::overdefined();
    }
}

void materialize_const_in_place(SsaNode& n, const SccpVal& v) {
    n.args.clear();
    n.phi_incomings.clear();
    n.imm_u32 = 0;
    n.imm_u32_2 = 0;
    n.imm_u64 = 0;
    n.aux_index = 0;
    n.aux_bool = false;
    n.aux_bool2 = false;
    n.aux_str.clear();
    switch (n.type) {
        case ValType::Bool:
            n.op = SsaOp::ConstBool;
            n.imm_u32 = v.bits ? 1u : 0u;
            break;
        case ValType::S32:
            n.op = SsaOp::ConstS32;
            n.imm_u32 = static_cast<uint32_t>(v.bits);
            break;
        case ValType::F32:
            n.op = SsaOp::ConstF32;
            n.imm_u32 = static_cast<uint32_t>(v.bits);
            break;
        case ValType::U64:
        case ValType::S64:
            n.op = SsaOp::ConstU64;
            n.imm_u64 = v.bits;
            break;
        case ValType::U32:
        default:
            n.op = SsaOp::ConstU32;
            n.imm_u32 = static_cast<uint32_t>(v.bits);
            break;
    }
}

}  // namespace

SsaPassStats run_ssa_sccp(ValhallSsaModule& mod, const SccpOptions& opts) {
    SsaPassStats stats;
    if (mod.blocks.empty()) return stats;

    const size_t num_blocks = mod.blocks.size();
    const size_t num_nodes = mod.nodes.size();

    // Build sparse def-use lists (both SSA users and blocks whose branch_cond uses a node).
    std::vector<std::vector<SsaId>> ssa_users(num_nodes);
    std::vector<std::vector<uint32_t>> cond_users(num_nodes);
    for (size_t b = 0; b < num_blocks; ++b) {
        const auto& blk = mod.blocks[b];
        for (SsaId pid : blk.phis) {
            for (const auto& inc : mod.get(pid).phi_incomings) {
                if (inc.value > 0 && inc.value < num_nodes) {
                    ssa_users[inc.value].push_back(pid);
                }
            }
        }
        for (SsaId nid : blk.nodes) {
            const auto& n = mod.get(nid);
            for (SsaId arg : n.args) {
                if (arg > 0 && arg < num_nodes) {
                    ssa_users[arg].push_back(nid);
                }
            }
            // VecExtract / TupleExtract depend on their underlying VecConstruct / Split64 operands.
            if (n.op == SsaOp::VecExtract && !n.args.empty()) {
                SsaId vid = n.args[0];
                if (vid > 0 && vid < num_nodes) {
                    const auto& vn = mod.get(vid);
                    if (n.imm_u32 < vn.args.size()) {
                        SsaId elem = vn.args[n.imm_u32];
                        if (elem > 0 && elem < num_nodes) ssa_users[elem].push_back(nid);
                    }
                }
            } else if (n.op == SsaOp::TupleExtract && !n.args.empty()) {
                SsaId tid = n.args[0];
                if (tid > 0 && tid < num_nodes) {
                    const auto& tn = mod.get(tid);
                    if (tn.op == SsaOp::Split64 && !tn.args.empty()) {
                        SsaId src64 = tn.args[0];
                        if (src64 > 0 && src64 < num_nodes) ssa_users[src64].push_back(nid);
                    }
                }
            }
        }
        if (blk.branch_cond > 0 && blk.branch_cond < num_nodes) {
            cond_users[blk.branch_cond].push_back(static_cast<uint32_t>(b));
        }
    }

    std::vector<SccpVal> lat(num_nodes, SccpVal::undef());
    std::vector<uint8_t> block_exec(num_blocks, 0);
    std::set<std::pair<uint32_t, uint32_t>> exec_edges;

    std::queue<std::pair<uint32_t, uint32_t>> cfg_wl;
    std::queue<SsaId> ssa_wl;

    auto mark_edge = [&](uint32_t from, uint32_t to) {
        if (to >= num_blocks) return;
        if (exec_edges.insert({from, to}).second) {
            cfg_wl.push({from, to});
        }
    };

    auto update_lat = [&](SsaId id, const SccpVal& nv) {
        if (id == kInvalidSsa || id >= num_nodes) return;
        SccpVal merged = meet_sccp(lat[id], nv);
        if (merged != lat[id]) {
            lat[id] = merged;
            ssa_wl.push(id);
        }
    };

    auto visit_phi = [&](SsaId pid) {
        const auto& phi = mod.get(pid);
        SccpVal acc = SccpVal::undef();
        for (const auto& inc : phi.phi_incomings) {
            if (exec_edges.count({inc.pred_block, phi.block_id})) {
                if (inc.value > 0 && inc.value < num_nodes) {
                    acc = meet_sccp(acc, lat[inc.value]);
                } else {
                    acc = SccpVal::overdefined();
                }
            }
        }
        update_lat(pid, acc);
    };

    auto visit_terminator = [&](uint32_t b) {
        if (!block_exec[b]) return;
        const auto& blk = mod.blocks[b];
        if (blk.is_terminated_by_end || blk.is_terminated_by_discard) return;
        if (blk.is_conditional_branch && blk.branch_cond > 0 && blk.branch_cond < num_nodes) {
            const auto& cv = lat[blk.branch_cond];
            if (cv.is_const()) {
                if (cv.bits != 0) {
                    if (blk.branch_target_block >= 0) mark_edge(b, static_cast<uint32_t>(blk.branch_target_block));
                } else {
                    if (blk.fallthrough_block >= 0) mark_edge(b, static_cast<uint32_t>(blk.fallthrough_block));
                }
                return;
            }
            if (cv.is_undef()) return;
        }
        for (uint32_t s : get_block_succs(blk)) {
            mark_edge(b, s);
        }
    };

    // Seed entry block 0 via synthetic sentinel edge (UINT32_MAX -> 0).
    block_exec[0] = 1;
    for (SsaId pid : mod.blocks[0].phis) visit_phi(pid);
    for (SsaId nid : mod.blocks[0].nodes) {
        update_lat(nid, eval_node_sccp(mod, mod.get(nid), lat, opts.fold_fp));
    }
    visit_terminator(0);

    while (!cfg_wl.empty() || !ssa_wl.empty()) {
        ++stats.iterations;
        while (!cfg_wl.empty()) {
            auto [from, to] = cfg_wl.front();
            cfg_wl.pop();
            (void)from;
            bool first_visit = !block_exec[to];
            block_exec[to] = 1;

            for (SsaId pid : mod.blocks[to].phis) {
                visit_phi(pid);
            }
            if (first_visit) {
                for (SsaId nid : mod.blocks[to].nodes) {
                    update_lat(nid, eval_node_sccp(mod, mod.get(nid), lat, opts.fold_fp));
                }
                visit_terminator(to);
            }
        }

        while (!ssa_wl.empty()) {
            SsaId def_id = ssa_wl.front();
            ssa_wl.pop();
            for (SsaId u : ssa_users[def_id]) {
                const auto& un = mod.get(u);
                if (un.block_id >= num_blocks || !block_exec[un.block_id]) continue;
                if (un.op == SsaOp::Phi) {
                    visit_phi(u);
                } else {
                    update_lat(u, eval_node_sccp(mod, un, lat, opts.fold_fp));
                }
            }
            for (uint32_t b : cond_users[def_id]) {
                visit_terminator(b);
            }
        }
    }

    // Materialize folded scalar constants.
    for (size_t b = 0; b < num_blocks; ++b) {
        if (!block_exec[b]) continue;
        auto& blk = mod.blocks[b];

        // Promote constant Phis by creating a constant node at the top of `blk.nodes`
        // and remapping the Phi ID to it.
        std::vector<SsaId> remap(mod.nodes.size(), kInvalidSsa);
        for (size_t i = 0; i < remap.size(); ++i) remap[i] = static_cast<SsaId>(i);
        std::vector<uint8_t> dead_phis(mod.nodes.size(), 0);
        uint32_t phi_const_count = 0;

        for (SsaId pid : blk.phis) {
            if (pid < lat.size() && lat[pid].is_const() &&
                is_materializable_scalar_type(mod.get(pid).type)) {
                SsaNode cnode{};
                cnode.type = mod.get(pid).type;
                cnode.orig_inst_idx = mod.get(pid).orig_inst_idx;
                cnode.orig_pc = mod.get(pid).orig_pc;
                materialize_const_in_place(cnode, lat[pid]);
                SsaId before = blk.nodes.empty() ? kInvalidSsa : blk.nodes.front();
                SsaId cid = insert_node_before(mod, static_cast<uint32_t>(b), before, std::move(cnode));
                remap.resize(mod.nodes.size(), kInvalidSsa);
                remap[cid] = cid;
                dead_phis.resize(mod.nodes.size(), 0);
                remap[pid] = cid;
                dead_phis[pid] = 1;
                ++phi_const_count;
                ++stats.nodes_folded;
                ++stats.phis_simplified;
            }
        }
        if (phi_const_count > 0) {
            apply_value_remap(mod, remap);
            stats.nodes_removed += erase_dead_nodes(mod, dead_phis);
        }

        for (SsaId nid : mod.blocks[b].nodes) {
            if (nid >= lat.size() || !lat[nid].is_const()) continue;
            auto& n = mod.get_mut(nid);
            if (is_const_op(n.op) || !is_pure_ssa_op(n, opts.fau_is_readonly)) continue;
            if (!is_materializable_scalar_type(n.type)) continue;
            materialize_const_in_place(n, lat[nid]);
            ++stats.nodes_folded;
        }
    }

    // Fold conditional branches with constant conditions.
    if (opts.fold_branches) {
        for (size_t b = 0; b < num_blocks; ++b) {
            if (!block_exec[b]) continue;
            auto& blk = mod.blocks[b];
            if (!blk.is_conditional_branch || blk.branch_cond == kInvalidSsa) continue;
            if (blk.branch_cond < lat.size() && lat[blk.branch_cond].is_const()) {
                bool taken = (lat[blk.branch_cond].bits != 0);
                int32_t target = taken ? blk.branch_target_block : blk.fallthrough_block;
                blk.is_conditional_branch = false;
                blk.is_unconditional_branch = true;
                blk.branch_cond = kInvalidSsa;
                blk.branch_target_block = target;
                blk.fallthrough_block = -1;
                ++stats.branches_folded;
            }
        }
    }

    stats.blocks_pruned += rebuild_cfg_preds_and_prune_phis(mod);
    stats.phis_simplified += simplify_trivial_phis(mod);

    if (opts.run_trailing_dce) {
        DceOptions dopts;
        dopts.fau_is_readonly = opts.fau_is_readonly;
        dopts.propagate_copies = true;
        dopts.prune_unreachable_blocks = false;
        SsaPassStats dce_stats = run_ssa_dce(mod, dopts);
        stats.nodes_removed += dce_stats.nodes_removed;
        stats.nodes_replaced += dce_stats.nodes_replaced;
        stats.phis_simplified += dce_stats.phis_simplified;
    }

    return stats;
}

}  // namespace valhall::dataflow
