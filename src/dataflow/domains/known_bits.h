#pragma once

#include "traits.h"

#include <cstdint>
#include <sstream>
#include <string>

namespace valhall::dataflow {

/// Per-bit certainty state for a 64-bit integer value.
/// Invariant: when `!is_bottom_val`, `(known_zero & known_one) == 0`.
struct KnownBitsValue {
    uint64_t known_zero = 0;
    uint64_t known_one = 0;
    bool is_bottom_val = true;

    bool is_bottom() const { return is_bottom_val; }
    bool is_top() const { return !is_bottom_val && known_zero == 0 && known_one == 0; }
    bool is_exact_const() const { return !is_bottom_val && (known_zero | known_one) == ~0ULL; }
    uint64_t const_u64() const { return known_one; }

    bool upper16_zero() const {
        return !is_bottom_val && (known_zero & 0xFFFFFFFFFFFF0000ULL) == 0xFFFFFFFFFFFF0000ULL;
    }

    bool upper32_zero() const {
        return !is_bottom_val && (known_zero & 0xFFFFFFFF00000000ULL) == 0xFFFFFFFF00000000ULL;
    }

    bool operator==(const KnownBitsValue& o) const {
        if (is_bottom_val != o.is_bottom_val) return false;
        if (is_bottom_val) return true;
        return known_zero == o.known_zero && known_one == o.known_one;
    }
};

/// 64-bit Known-Bits abstract domain (lattice height = 65).
/// Satisfies `ValidateLatticeDomain` and provides SFINAE domain operations
/// automatically dispatched by `SsaDomainTransferVisitor<KnownBitsDomain>`.
struct KnownBitsDomain {
    using Value = KnownBitsValue;

    static Value bottom() { return Value{0, 0, true}; }
    static Value top() { return Value{0, 0, false}; }

    static Value make_known(uint64_t kz, uint64_t ko) {
        if ((kz & ko) != 0) return bottom();
        return Value{kz, ko, false};
    }

    static Value make_const_u64(uint64_t c) {
        return Value{~c, c, false};
    }

    static bool try_as_const_u64(const Value& v, uint64_t* out) {
        if (!v.is_exact_const()) return false;
        if (out) *out = v.known_one;
        return true;
    }

    /// Creates a `KnownBitsValue` constraining all bits above `max_inclusive` to zero.
    static Value make_u32_bound(uint32_t max_inclusive) {
        uint64_t mask = 0;
        uint32_t m = max_inclusive;
        while (m != 0) {
            mask = (mask << 1) | 1ULL;
            m >>= 1;
        }
        return Value{~mask, 0, false};
    }

    static Value join(const Value& a, const Value& b) {
        if (a.is_bottom()) return b;
        if (b.is_bottom()) return a;
        return Value{a.known_zero & b.known_zero, a.known_one & b.known_one, false};
    }

    static Value meet(const Value& a, const Value& b) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        return make_known(a.known_zero | b.known_zero, a.known_one | b.known_one);
    }

    static bool leq(const Value& a, const Value& b) {
        if (a.is_bottom()) return true;
        if (b.is_bottom()) return false;
        return ((a.known_zero & b.known_zero) == b.known_zero) &&
               ((a.known_one & b.known_one) == b.known_one);
    }

    static bool equals(const Value& a, const Value& b) { return a == b; }

    static std::string format(const Value& v) {
        if (v.is_bottom()) return "Bottom";
        if (v.is_top()) return "Top";
        std::ostringstream oss;
        if (v.is_exact_const()) {
            oss << "0x" << std::hex << v.known_one << std::dec;
        } else {
            oss << "KB(z=0x" << std::hex << v.known_zero << ",o=0x" << v.known_one << std::dec << ")";
        }
        return oss.str();
    }

    static Value mask_width(const Value& v, bool is_64bit) {
        if (v.is_bottom() || is_64bit) return v;
        return make_known((v.known_zero & 0xFFFFFFFFULL) | 0xFFFFFFFF00000000ULL,
                          v.known_one & 0xFFFFFFFFULL);
    }

    static Value bit_and(const Value& a, const Value& b, bool is_64bit) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        return mask_width(make_known(a.known_zero | b.known_zero, a.known_one & b.known_one), is_64bit);
    }

    static Value bit_or(const Value& a, const Value& b, bool is_64bit) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        return mask_width(make_known(a.known_zero & b.known_zero, a.known_one | b.known_one), is_64bit);
    }

    static Value bit_xor(const Value& a, const Value& b, bool is_64bit) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t kz = (a.known_zero & b.known_zero) | (a.known_one & b.known_one);
        uint64_t ko = (a.known_zero & b.known_one) | (a.known_one & b.known_zero);
        return mask_width(make_known(kz, ko), is_64bit);
    }

    static Value bit_not(const Value& a, bool is_64bit) {
        if (a.is_bottom()) return bottom();
        return mask_width(make_known(a.known_one, a.known_zero), is_64bit);
    }

    static Value shift(const Value& a, const Value& b, ShiftKind kind, bool is_64bit) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t sh = 0;
        if (!try_as_const_u64(b, &sh)) return mask_width(top(), is_64bit);
        uint32_t width = is_64bit ? 64u : 32u;
        sh &= (width - 1u);
        Value am = mask_width(a, is_64bit);
        if (sh == 0) return am;
        switch (kind) {
            case ShiftKind::Shl: {
                uint64_t low_zeros = (1ULL << sh) - 1ULL;
                uint64_t kz = (am.known_zero << sh) | low_zeros;
                uint64_t ko = (am.known_one << sh);
                return mask_width(make_known(kz, ko), is_64bit);
            }
            case ShiftKind::ShrU: {
                uint64_t hi_zeros = ~((1ULL << (64u - sh)) - 1ULL);
                uint64_t kz = (am.known_zero >> sh) | hi_zeros;
                uint64_t ko = (am.known_one >> sh);
                return mask_width(make_known(kz, ko), is_64bit);
            }
            case ShiftKind::ShrS: {
                uint64_t sign_bit = is_64bit ? (1ULL << 63) : (1ULL << 31);
                uint64_t hi_mask = is_64bit ? ~((1ULL << (64u - sh)) - 1ULL)
                                            : (0xFFFFFFFFULL & ~((1ULL << (32u - sh)) - 1ULL));
                uint64_t kz = (am.known_zero >> sh) | ((am.known_zero & sign_bit) ? hi_mask : 0ULL);
                uint64_t ko = (am.known_one >> sh) | ((am.known_one & sign_bit) ? hi_mask : 0ULL);
                return mask_width(make_known(kz, ko), is_64bit);
            }
            default:
                return mask_width(top(), is_64bit);
        }
    }

    static Value iadd(const Value& a, const Value& b, bool is_64bit) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        Value am = mask_width(a, is_64bit);
        Value bm = mask_width(b, is_64bit);
        uint64_t min_sum = am.known_one + bm.known_one;
        uint64_t max_sum = (~am.known_zero) + (~bm.known_zero);
        uint64_t known_carries = ~(min_sum ^ max_sum);
        uint64_t known_bits = (am.known_zero | am.known_one) & (bm.known_zero | bm.known_one) & known_carries;
        uint64_t kz = known_bits & ~min_sum;
        uint64_t ko = known_bits & min_sum;
        return mask_width(make_known(kz, ko), is_64bit);
    }

    static Value isub(const Value& a, const Value& b, bool is_64bit) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t ca = 0, cb = 0;
        if (try_as_const_u64(a, &ca) && try_as_const_u64(b, &cb)) {
            return mask_width(make_const_u64(ca - cb), is_64bit);
        }
        return mask_width(top(), is_64bit);
    }

    static Value bfe_u(const Value& src, const Value& off_v, const Value& cnt_v) {
        if (src.is_bottom() || off_v.is_bottom() || cnt_v.is_bottom()) return bottom();
        uint64_t off = 0, cnt = 0;
        if (!try_as_const_u64(off_v, &off) || !try_as_const_u64(cnt_v, &cnt) ||
            off >= 32u || cnt == 0u || cnt > 32u) {
            return mask_width(top(), false);
        }
        uint64_t field_mask = (cnt == 32u) ? 0xFFFFFFFFULL : ((1ULL << cnt) - 1ULL);
        uint64_t kz = ((src.known_zero >> off) & field_mask) | ~field_mask;
        uint64_t ko = (src.known_one >> off) & field_mask;
        return make_known(kz, ko);
    }

    static Value bfi(const Value& base, const Value& ins, const Value& off_v, const Value& cnt_v) {
        if (base.is_bottom() || ins.is_bottom() || off_v.is_bottom() || cnt_v.is_bottom()) return bottom();
        uint64_t off = 0, cnt = 0;
        if (!try_as_const_u64(off_v, &off) || !try_as_const_u64(cnt_v, &cnt) ||
            off >= 32u || cnt == 0u || off + cnt > 32u) {
            return mask_width(top(), false);
        }
        uint64_t field_mask = ((cnt == 32u) ? 0xFFFFFFFFULL : ((1ULL << cnt) - 1ULL)) << off;
        uint64_t kz = (base.known_zero & ~field_mask) | ((ins.known_zero << off) & field_mask);
        uint64_t ko = (base.known_one & ~field_mask) | ((ins.known_one << off) & field_mask);
        return mask_width(make_known(kz, ko), false);
    }
};

static_assert(ValidateLatticeDomain<KnownBitsDomain>::valid, "KnownBitsDomain must satisfy LatticeDomain");

}  // namespace valhall::dataflow
