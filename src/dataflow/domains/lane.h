#pragma once

#include "traits.h"
#include <cstdint>
#include <string>
#include <utility>

namespace valhall::dataflow {

// LaneDomainWrapper<InnerDomainT> — Wrapper Adding 64-Bit Lane Ops & Lifting Base Ops

template <typename InnerDomainT>
struct LaneValue {
    static_assert(ValidateLatticeDomain<InnerDomainT>::valid, "");
    using InnerValue = typename InnerDomainT::Value;

    InnerValue inner = InnerDomainT::bottom();
    bool is_hi_half = false;

    LaneValue() = default;
    LaneValue(const InnerValue& v, bool hi = false) : inner(v), is_hi_half(hi) {}
    LaneValue(InnerValue&& v, bool hi = false) : inner(std::move(v)), is_hi_half(hi) {}

    bool is_bottom() const { return domain_equals<InnerDomainT>(inner, InnerDomainT::bottom()); }
    bool is_top() const { return domain_equals<InnerDomainT>(inner, InnerDomainT::top()); }
};

template <typename InnerDomainT>
struct LaneDomainWrapper
    : public PointwiseOpsMixin<LaneDomainWrapper<InnerDomainT>, LaneValue<InnerDomainT>, InnerDomainT> {
    static_assert(ValidateLatticeDomain<InnerDomainT>::valid, "");
    using InnerValue = typename InnerDomainT::Value;
    using Value = LaneValue<InnerDomainT>;

    static Value bottom() { return Value(InnerDomainT::bottom(), false); }
    static Value top() { return Value(InnerDomainT::top(), false); }

    static Value join(const Value& a, const Value& b) {
        if (a.is_bottom()) return b;
        if (b.is_bottom()) return a;
        if (a.is_top() || b.is_top()) return top();
        if (a.is_hi_half != b.is_hi_half) return top();
        return Value(InnerDomainT::join(a.inner, b.inner), a.is_hi_half);
    }

    static Value meet(const Value& a, const Value& b) {
        if (a.is_top()) return b;
        if (b.is_top()) return a;
        if (a.is_bottom() || b.is_bottom()) return bottom();
        if (a.is_hi_half != b.is_hi_half) return bottom();
        return Value(InnerDomainT::meet(a.inner, b.inner), a.is_hi_half);
    }

    static bool leq(const Value& a, const Value& b) {
        if (a.is_bottom() || b.is_top()) return true;
        if (b.is_bottom() || a.is_top()) return false;
        if (a.is_hi_half != b.is_hi_half) return false;
        return InnerDomainT::leq(a.inner, b.inner);
    }

    static bool equals(const Value& a, const Value& b) {
        if (a.is_bottom() && b.is_bottom()) return true;
        if (a.is_top() && b.is_top()) return true;
        return a.is_hi_half == b.is_hi_half && domain_equals<InnerDomainT>(a.inner, b.inner);
    }

    static std::string format(const Value& v) {
        std::string s = domain_format<InnerDomainT>(v.inner);
        if (v.is_hi_half && !v.is_bottom() && !v.is_top()) {
            s += ".hi";
        }
        return s;
    }

    static bool try_as_const_u64(const Value& v, uint64_t* out) {
        if constexpr (detail::has_const_helpers<InnerDomainT>::value) {
            if (!v.is_hi_half) return InnerDomainT::try_as_const_u64(v.inner, out);
        }
        (void)v; (void)out;
        return false;
    }

    static Value make_const_u64(uint64_t c) {
        if constexpr (detail::has_const_helpers<InnerDomainT>::value) {
            return Value(InnerDomainT::make_const_u64(c), false);
        }
        (void)c;
        return top();
    }

    template <typename Fn>
    static Value lift_unary_op(const Value& a, Fn&& fn) {
        if (a.is_bottom()) return bottom();
        InnerValue res = fn(a.inner);
        bool hi = a.is_hi_half &&
                  !domain_equals<InnerDomainT>(res, InnerDomainT::bottom()) &&
                  !domain_equals<InnerDomainT>(res, InnerDomainT::top());
        return Value(std::move(res), hi);
    }

    template <typename Fn>
    static Value lift_binary_op(const Value& a, const Value& b, Fn&& fn) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        InnerValue res = fn(a.inner, b.inner);
        if (domain_equals<InnerDomainT>(res, InnerDomainT::bottom()) ||
            domain_equals<InnerDomainT>(res, InnerDomainT::top())) {
            return Value(std::move(res), false);
        }
        bool hi = false;
        if (a.is_hi_half == b.is_hi_half) {
            hi = a.is_hi_half;
        } else if (a.is_hi_half && domain_equals<InnerDomainT>(res, a.inner)) {
            hi = true;
        } else if (b.is_hi_half && domain_equals<InnerDomainT>(res, b.inner)) {
            hi = true;
        }
        return Value(std::move(res), hi);
    }

    static Value split64(const Value& src64) { return src64; }

    static Value tuple_extract(const Value& tup, uint32_t elem_idx) {
        if (tup.is_bottom()) return bottom();
        if (tup.is_top()) return top();
        if (elem_idx == 0) return Value(tup.inner, false);
        if (elem_idx == 1) {
            if constexpr (detail::has_const_helpers<InnerDomainT>::value) {
                uint64_t c = 0;
                if (InnerDomainT::try_as_const_u64(tup.inner, &c)) {
                    return Value(InnerDomainT::make_const_u64(c >> 32), false);
                }
            }
            return Value(tup.inner, true);
        }
        return top();
    }

    static Value pack64(const Value& lo, const Value& hi) {
        if (lo.is_bottom() || hi.is_bottom()) return bottom();
        bool lo_const = false, hi_const = false;
        uint64_t clo = 0, chi = 0;
        if constexpr (detail::has_const_helpers<InnerDomainT>::value) {
            lo_const = InnerDomainT::try_as_const_u64(lo.inner, &clo);
            hi_const = InnerDomainT::try_as_const_u64(hi.inner, &chi);
            if (lo_const && hi_const) {
                uint64_t packed = ((chi & 0xFFFFFFFFULL) << 32) | (clo & 0xFFFFFFFFULL);
                return Value(InnerDomainT::make_const_u64(packed), false);
            }
        }
        // Preserve non-constant symbolic/pointer identities across split 32-bit halves,
        // while avoiding falsely treating `pack64(ConstInt, Top)` as `ConstInt`.
        if (!lo.is_top() && !lo_const) {
            return Value(lo.inner, false);
        }
        if (!hi.is_top() && !hi_const) {
            return Value(hi.inner, false);
        }
        return top();
    }
};


}  // namespace valhall::dataflow
