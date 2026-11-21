#pragma once

#include "traits.h"
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace valhall::dataflow {

// FiniteSetDomain<ElemDomainT, MaxK> — Bounded Finite-Set Powerset Lattice

template <typename ElemDomainT, size_t MaxK = 4>
struct FiniteSetValue {
    using ElemValue = typename ElemDomainT::Value;
    static_assert(ValidateLatticeDomain<ElemDomainT>::valid, "");
    static_assert(MaxK >= 1, "FiniteSetDomain MaxK must be >= 1");

    bool is_top_val = false;
    std::vector<ElemValue> items;  // Always deduplicated, contains neither Bottom nor Top

    FiniteSetValue() = default;

    // Implicit singleton lift
    FiniteSetValue(const ElemValue& single) { insert(single); }

    FiniteSetValue& operator=(const ElemValue& single) {
        is_top_val = false;
        items.clear();
        insert(single);
        return *this;
    }

    bool is_bottom() const { return !is_top_val && items.empty(); }
    bool is_top() const { return is_top_val; }
    bool is_singleton() const { return !is_top_val && items.size() == 1; }
    size_t size() const { return items.size(); }

    bool try_as_singleton(ElemValue* out) const {
        if (!is_singleton()) return false;
        if (out) *out = items[0];
        return true;
    }

    void insert(const ElemValue& e) {
        if (is_top_val) return;
        if (domain_equals<ElemDomainT>(e, ElemDomainT::bottom())) return;
        if (domain_equals<ElemDomainT>(e, ElemDomainT::top())) {
            is_top_val = true;
            items.clear();
            return;
        }
        for (const auto& existing : items) {
            if (domain_equals<ElemDomainT>(existing, e)) return;
        }
        if (items.size() >= MaxK) {
            is_top_val = true;
            items.clear();
            return;
        }
        items.push_back(e);
    }
};

template <typename ElemDomainT, size_t MaxK = 4>
struct FiniteSetDomain
    : public PointwiseOpsMixin<FiniteSetDomain<ElemDomainT, MaxK>,
                               FiniteSetValue<ElemDomainT, MaxK>,
                               ElemDomainT> {
    static_assert(ValidateLatticeDomain<ElemDomainT>::valid, "");
    using ElemValue = typename ElemDomainT::Value;
    using Value = FiniteSetValue<ElemDomainT, MaxK>;

    static Value bottom() { return Value{}; }
    static Value top() { Value v; v.is_top_val = true; return v; }

    static Value singleton(const ElemValue& e) { Value v; v.insert(e); return v; }
    static Value lift_value(const ElemValue& e) { return singleton(e); }

    template <typename Fn>
    static Value lift_unary(const Value& s, Fn&& fn) {
        if (s.is_bottom()) return bottom();
        if (s.is_top()) return top();
        Value out = bottom();
        for (const auto& e : s.items) {
            using Ret = std::decay_t<decltype(fn(e))>;
            if constexpr (std::is_same_v<Ret, Value>) {
                out = join(out, fn(e));
            } else {
                out.insert(fn(e));
            }
            if (out.is_top()) return top();
        }
        return out;
    }

    template <typename Fn>
    static Value lift_binary(const Value& a, const Value& b, Fn&& fn) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        if (a.is_top() || b.is_top()) return top();
        Value out = bottom();
        for (const auto& ea : a.items) {
            for (const auto& eb : b.items) {
                using Ret = std::decay_t<decltype(fn(ea, eb))>;
                if constexpr (std::is_same_v<Ret, Value>) {
                    out = join(out, fn(ea, eb));
                } else {
                    out.insert(fn(ea, eb));
                }
                if (out.is_top()) return top();
            }
        }
        return out;
    }

    // Domain-op lifting hooks used by PointwiseOpsMixin: propagates Top through
    // ElemDomainT so operations like `Ptr + Top -> Ptr(+?)` and `0 * Top -> 0`
    // retain full precision instead of collapsing prematurely to Top.
    template <typename Fn>
    static Value lift_unary_op(const Value& s, Fn&& fn) {
        if (s.is_bottom()) return bottom();
        if (s.is_top()) {
            Value out = bottom();
            out.insert(fn(ElemDomainT::top()));
            return out;
        }
        return lift_unary(s, std::forward<Fn>(fn));
    }

    template <typename Fn>
    static Value lift_binary_op(const Value& a, const Value& b, Fn&& fn) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        if (a.is_top() && b.is_top()) {
            Value out = bottom();
            out.insert(fn(ElemDomainT::top(), ElemDomainT::top()));
            return out;
        }
        if (!a.is_top() && b.is_top()) {
            return lift_unary(a, [&](const ElemValue& x) { return fn(x, ElemDomainT::top()); });
        }
        if (a.is_top() && !b.is_top()) {
            return lift_unary(b, [&](const ElemValue& y) { return fn(ElemDomainT::top(), y); });
        }
        return lift_binary(a, b, std::forward<Fn>(fn));
    }

    static Value join(const Value& a, const Value& b) {
        if (a.is_bottom()) return b;
        if (b.is_bottom()) return a;
        if (a.is_top() || b.is_top()) return top();
        Value out = a;
        for (const auto& eb : b.items) {
            out.insert(eb);
            if (out.is_top()) return top();
        }
        return out;
    }

    static Value meet(const Value& a, const Value& b) {
        if (a.is_top()) return b;
        if (b.is_top()) return a;
        if (a.is_bottom() || b.is_bottom()) return bottom();
        Value out = bottom();
        for (const auto& ea : a.items) {
            for (const auto& eb : b.items) {
                if (domain_equals<ElemDomainT>(ea, eb)) { out.insert(ea); break; }
            }
        }
        return out;
    }

    static bool leq(const Value& a, const Value& b) {
        if (a.is_bottom() || b.is_top()) return true;
        if (b.is_bottom() || a.is_top()) return false;
        for (const auto& ea : a.items) {
            bool covered = false;
            for (const auto& eb : b.items) {
                if (ElemDomainT::leq(ea, eb)) { covered = true; break; }
            }
            if (!covered) return false;
        }
        return true;
    }

    static bool equals(const Value& a, const Value& b) {
        if (a.is_top_val != b.is_top_val) return false;
        if (a.is_top_val) return true;
        if (a.items.size() != b.items.size()) return false;
        for (const auto& ea : a.items) {
            bool found = false;
            for (const auto& eb : b.items) {
                if (domain_equals<ElemDomainT>(ea, eb)) { found = true; break; }
            }
            if (!found) return false;
        }
        return true;
    }

    static std::string format(const Value& s) {
        if (s.is_bottom()) return "<bottom>";
        if (s.is_top()) return "<top>";
        if (s.items.size() == 1) return domain_format<ElemDomainT>(s.items[0]);
        std::ostringstream oss;
        oss << "{";
        for (size_t i = 0; i < s.items.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << domain_format<ElemDomainT>(s.items[i]);
        }
        oss << "}";
        return oss.str();
    }

    static bool try_as_const_u64(const Value& v, uint64_t* out) {
        if constexpr (detail::has_const_helpers<ElemDomainT>::value) {
            ElemValue single;
            if (v.try_as_singleton(&single)) return ElemDomainT::try_as_const_u64(single, out);
        }
        (void)v; (void)out;
        return false;
    }

    static Value make_const_u64(uint64_t c) {
        if constexpr (detail::has_const_helpers<ElemDomainT>::value) {
            return singleton(ElemDomainT::make_const_u64(c));
        }
        (void)c;
        return top();
    }
};


}  // namespace valhall::dataflow
