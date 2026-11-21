#pragma once

#include "mixins.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace valhall::dataflow {

// VectorDomain<ElemDomainT> — Pointwise Lattice over Dense Indexed Arrays

template <typename ElemDomainT>
struct VectorDomainValue {
    using ElemValue = typename ElemDomainT::Value;
    static_assert(ValidateLatticeDomain<ElemDomainT>::valid, "");

    std::vector<ElemValue> elems;
    bool is_top_val = false;

    explicit VectorDomainValue(size_t initial_size = 0)
        : elems(initial_size, ElemDomainT::bottom()) {}

    void ensure_size(size_t needed) {
        if (elems.size() < needed) {
            elems.resize(next_power_of_two(needed), ElemDomainT::bottom());
        }
    }

    ElemValue get(size_t idx) const {
        if (is_top_val) return ElemDomainT::top();
        return (idx < elems.size()) ? elems[idx] : ElemDomainT::bottom();
    }

    void set(size_t idx, const ElemValue& v) {
        is_top_val = false;
        ensure_size(idx + 1);
        elems[idx] = v;
    }

    bool join_at(size_t idx, const ElemValue& v) {
        if (is_top_val) return false;
        ensure_size(idx + 1);
        ElemValue joined = ElemDomainT::join(elems[idx], v);
        if (!domain_equals<ElemDomainT>(joined, elems[idx])) {
            elems[idx] = std::move(joined);
            return true;
        }
        return false;
    }
};

template <typename ElemDomainT>
struct VectorDomain
    : public PointwiseOpsMixin<VectorDomain<ElemDomainT>, VectorDomainValue<ElemDomainT>, ElemDomainT> {
    static_assert(ValidateLatticeDomain<ElemDomainT>::valid, "");
    using Value = VectorDomainValue<ElemDomainT>;

    static Value bottom() { return Value(0); }
    static Value top() { Value v; v.is_top_val = true; return v; }

    static bool join_into(Value& dst, const Value& other) {
        if (dst.is_top_val) return false;
        if (other.is_top_val) { dst = top(); return true; }
        bool changed = false;
        if (other.elems.size() > dst.elems.size()) {
            dst.ensure_size(other.elems.size());
        }
        for (size_t i = 0; i < other.elems.size(); ++i) {
            auto joined = ElemDomainT::join(dst.elems[i], other.elems[i]);
            if (!domain_equals<ElemDomainT>(joined, dst.elems[i])) {
                dst.elems[i] = std::move(joined);
                changed = true;
            }
        }
        return changed;
    }

    static bool meet_into(Value& dst, const Value& other) {
        if (other.is_top_val) return false;
        if (dst.is_top_val) { dst = other; return true; }
        bool changed = false;
        size_t n = std::min(dst.elems.size(), other.elems.size());
        for (size_t i = 0; i < n; ++i) {
            auto met = ElemDomainT::meet(dst.elems[i], other.elems[i]);
            if (!domain_equals<ElemDomainT>(met, dst.elems[i])) {
                dst.elems[i] = std::move(met);
                changed = true;
            }
        }
        for (size_t i = n; i < dst.elems.size(); ++i) {
            if (!domain_equals<ElemDomainT>(dst.elems[i], ElemDomainT::bottom())) {
                dst.elems[i] = ElemDomainT::bottom();
                changed = true;
            }
        }
        return changed;
    }

    static Value join(const Value& a, const Value& b) { Value out = a; join_into(out, b); return out; }
    static Value meet(const Value& a, const Value& b) { Value out = a; meet_into(out, b); return out; }

    static bool leq(const Value& a, const Value& b) {
        if (b.is_top_val) return true;
        if (a.is_top_val) return false;
        size_t max_n = std::max(a.elems.size(), b.elems.size());
        for (size_t i = 0; i < max_n; ++i) {
            if (!ElemDomainT::leq(a.get(i), b.get(i))) return false;
        }
        return true;
    }

    static bool equals(const Value& a, const Value& b) {
        if (a.is_top_val != b.is_top_val) return false;
        if (a.is_top_val) return true;
        size_t max_n = std::max(a.elems.size(), b.elems.size());
        for (size_t i = 0; i < max_n; ++i) {
            if (!domain_equals<ElemDomainT>(a.get(i), b.get(i))) return false;
        }
        return true;
    }

    static std::string format(const Value& v) {
        if (v.is_top_val) return "<top>";
        std::ostringstream oss;
        oss << "[";
        bool first = true;
        for (size_t i = 0; i < v.elems.size(); ++i) {
            if (domain_equals<ElemDomainT>(v.elems[i], ElemDomainT::bottom())) continue;
            if (!first) oss << ", ";
            oss << i << ":" << domain_format<ElemDomainT>(v.elems[i]);
            first = false;
        }
        oss << "]";
        return oss.str();
    }

    template <typename Fn>
    static Value apply_binary(const Value& a, const Value& b, Fn&& fn) {
        if (a.is_top_val || b.is_top_val) return top();
        size_t n = std::max(a.elems.size(), b.elems.size());
        Value out(n);
        for (size_t i = 0; i < n; ++i) out.elems[i] = fn(a.get(i), b.get(i));
        return out;
    }

    template <typename Fn>
    static Value apply_unary(const Value& a, Fn&& fn) {
        if (a.is_top_val) return top();
        Value out(a.elems.size());
        for (size_t i = 0; i < a.elems.size(); ++i) out.elems[i] = fn(a.elems[i]);
        return out;
    }

    template <typename Fn>
    static Value lift_binary_op(const Value& a, const Value& b, Fn&& fn) {
        return apply_binary(a, b, std::forward<Fn>(fn));
    }

    template <typename Fn>
    static Value lift_unary_op(const Value& a, Fn&& fn) {
        return apply_unary(a, std::forward<Fn>(fn));
    }
};

// MapDomain<KeyT, ValDomainT> — Pointwise Lattice over Sparse Maps

template <typename KeyT, typename ValDomainT>
struct MapDomainValue {
    using Val = typename ValDomainT::Value;
    static_assert(ValidateLatticeDomain<ValDomainT>::valid, "");

    std::map<KeyT, Val> entries;
    bool is_top_val = false;

    Val get(const KeyT& k) const {
        if (is_top_val) return ValDomainT::top();
        auto it = entries.find(k);
        return (it != entries.end()) ? it->second : ValDomainT::bottom();
    }

    void set(const KeyT& k, const Val& v) {
        is_top_val = false;
        if (domain_equals<ValDomainT>(v, ValDomainT::bottom())) {
            entries.erase(k);
        } else {
            entries[k] = v;
        }
    }

    bool join_at(const KeyT& k, const Val& v) {
        if (is_top_val) return false;
        auto it = entries.find(k);
        if (it == entries.end()) {
            if (!domain_equals<ValDomainT>(v, ValDomainT::bottom())) {
                entries[k] = v;
                return true;
            }
            return false;
        }
        Val joined = ValDomainT::join(it->second, v);
        if (!domain_equals<ValDomainT>(joined, it->second)) {
            it->second = std::move(joined);
            return true;
        }
        return false;
    }

    void clear() {
        entries.clear();
        is_top_val = false;
    }
};

template <typename KeyT, typename ValDomainT>
struct MapDomain
    : public PointwiseOpsMixin<MapDomain<KeyT, ValDomainT>, MapDomainValue<KeyT, ValDomainT>, ValDomainT> {
    static_assert(ValidateLatticeDomain<ValDomainT>::valid, "");
    using Value = MapDomainValue<KeyT, ValDomainT>;

    static Value bottom() { return Value{}; }
    static Value top() { Value v; v.is_top_val = true; return v; }

    static bool join_into(Value& dst, const Value& other) {
        if (dst.is_top_val) return false;
        if (other.is_top_val) { dst = top(); return true; }
        bool changed = false;
        for (const auto& kv : other.entries) {
            if (dst.join_at(kv.first, kv.second)) changed = true;
        }
        return changed;
    }

    static bool meet_into(Value& dst, const Value& other) {
        if (other.is_top_val) return false;
        if (dst.is_top_val) { dst = other; return true; }
        bool changed = false;
        for (auto it = dst.entries.begin(); it != dst.entries.end(); ) {
            auto oit = other.entries.find(it->first);
            if (oit == other.entries.end()) {
                it = dst.entries.erase(it);
                changed = true;
            } else {
                auto met = ValDomainT::meet(it->second, oit->second);
                if (domain_equals<ValDomainT>(met, ValDomainT::bottom())) {
                    it = dst.entries.erase(it);
                    changed = true;
                } else if (!domain_equals<ValDomainT>(met, it->second)) {
                    it->second = std::move(met);
                    changed = true;
                    ++it;
                } else {
                    ++it;
                }
            }
        }
        return changed;
    }

    static Value join(const Value& a, const Value& b) { Value out = a; join_into(out, b); return out; }
    static Value meet(const Value& a, const Value& b) { Value out = a; meet_into(out, b); return out; }

    static bool leq(const Value& a, const Value& b) {
        if (b.is_top_val) return true;
        if (a.is_top_val) return false;
        for (const auto& kv : a.entries) {
            if (!ValDomainT::leq(kv.second, b.get(kv.first))) return false;
        }
        return true;
    }

    static bool equals(const Value& a, const Value& b) {
        if (a.is_top_val != b.is_top_val) return false;
        if (a.is_top_val) return true;
        if (a.entries.size() != b.entries.size()) return false;
        for (const auto& kv : a.entries) {
            auto it = b.entries.find(kv.first);
            if (it == b.entries.end() || !domain_equals<ValDomainT>(kv.second, it->second)) return false;
        }
        return true;
    }

    template <typename Fn>
    static Value apply_binary(const Value& a, const Value& b, Fn&& fn) {
        if (a.is_top_val || b.is_top_val) return top();
        Value out;
        for (const auto& kv : a.entries) out.set(kv.first, fn(kv.second, b.get(kv.first)));
        for (const auto& kv : b.entries) {
            if (a.entries.find(kv.first) == a.entries.end()) {
                out.set(kv.first, fn(ValDomainT::bottom(), kv.second));
            }
        }
        return out;
    }

    template <typename Fn>
    static Value apply_unary(const Value& a, Fn&& fn) {
        if (a.is_top_val) return top();
        Value out;
        for (const auto& kv : a.entries) out.set(kv.first, fn(kv.second));
        return out;
    }

    template <typename Fn>
    static Value lift_binary_op(const Value& a, const Value& b, Fn&& fn) {
        return apply_binary(a, b, std::forward<Fn>(fn));
    }

    template <typename Fn>
    static Value lift_unary_op(const Value& a, Fn&& fn) {
        return apply_unary(a, std::forward<Fn>(fn));
    }
};

// TupleDomain<Domains...> — Variadic Pointwise Product Lattice

template <typename... Domains>
struct TupleDomain {
    static_assert((ValidateLatticeDomain<Domains>::valid && ...), "");
    using Value = std::tuple<typename Domains::Value...>;

    static Value bottom() { return Value{Domains::bottom()...}; }
    static Value top() { return Value{Domains::top()...}; }

    static Value join(const Value& a, const Value& b) {
        return join_impl(a, b, std::index_sequence_for<Domains...>{});
    }
    static Value meet(const Value& a, const Value& b) {
        return meet_impl(a, b, std::index_sequence_for<Domains...>{});
    }
    static bool leq(const Value& a, const Value& b) {
        return leq_impl(a, b, std::index_sequence_for<Domains...>{});
    }
    static bool equals(const Value& a, const Value& b) {
        return equals_impl(a, b, std::index_sequence_for<Domains...>{});
    }
    static std::string format(const Value& v) {
        return format_impl(v, std::index_sequence_for<Domains...>{});
    }

#define VALHALL_TUPLE_BINARY_OP(op_name, trait_name)                                                        \
    template <typename... Extra,                                                                            \
              bool En = (detail::trait_name##_v<Domains, Extra...> && ...),                                 \
              std::enable_if_t<En, int> = 0>                                                                \
    static Value op_name(const Value& a, const Value& b, Extra&&... extra) {                                \
        return op_name##_impl(a, b, std::index_sequence_for<Domains...>{}, std::forward<Extra>(extra)...);  \
    }

#define VALHALL_TUPLE_UNARY_OP(op_name, trait_name)                                                         \
    template <typename... Extra,                                                                            \
              bool En = (detail::trait_name##_v<Domains, Extra...> && ...),                                 \
              std::enable_if_t<En, int> = 0>                                                                \
    static Value op_name(const Value& a, Extra&&... extra) {                                                \
        return op_name##_impl(a, std::index_sequence_for<Domains...>{}, std::forward<Extra>(extra)...);     \
    }

    VALHALL_TUPLE_BINARY_OP(add, has_domain_add)
    VALHALL_TUPLE_BINARY_OP(sub, has_domain_sub)
    VALHALL_TUPLE_BINARY_OP(mul, has_domain_mul)
    VALHALL_TUPLE_BINARY_OP(div, has_domain_div)
    VALHALL_TUPLE_BINARY_OP(iadd, has_domain_iadd)
    VALHALL_TUPLE_BINARY_OP(isub, has_domain_isub)
    VALHALL_TUPLE_BINARY_OP(imul, has_domain_imul)
    VALHALL_TUPLE_BINARY_OP(idiv, has_domain_idiv)
    VALHALL_TUPLE_BINARY_OP(bit_and, has_domain_bit_and)
    VALHALL_TUPLE_BINARY_OP(bit_or, has_domain_bit_or)
    VALHALL_TUPLE_BINARY_OP(bit_xor, has_domain_bit_xor)
    VALHALL_TUPLE_BINARY_OP(shift, has_domain_shift)
    VALHALL_TUPLE_BINARY_OP(pack64, has_domain_pack64)

    VALHALL_TUPLE_UNARY_OP(bit_not, has_domain_bit_not)
    VALHALL_TUPLE_UNARY_OP(split64, has_domain_split64)
    VALHALL_TUPLE_UNARY_OP(tuple_extract, has_domain_tuple_extract)
    VALHALL_TUPLE_UNARY_OP(convert, has_domain_convert)

#undef VALHALL_TUPLE_BINARY_OP
#undef VALHALL_TUPLE_UNARY_OP

private:
    template <size_t... Is>
    static Value join_impl(const Value& a, const Value& b, std::index_sequence<Is...>) {
        return Value{Domains::join(std::get<Is>(a), std::get<Is>(b))...};
    }
    template <size_t... Is>
    static Value meet_impl(const Value& a, const Value& b, std::index_sequence<Is...>) {
        return Value{Domains::meet(std::get<Is>(a), std::get<Is>(b))...};
    }
    template <size_t... Is>
    static bool leq_impl(const Value& a, const Value& b, std::index_sequence<Is...>) {
        return (Domains::leq(std::get<Is>(a), std::get<Is>(b)) && ...);
    }
    template <size_t... Is>
    static bool equals_impl(const Value& a, const Value& b, std::index_sequence<Is...>) {
        return (domain_equals<Domains>(std::get<Is>(a), std::get<Is>(b)) && ...);
    }
    template <size_t... Is>
    static std::string format_impl(const Value& v, std::index_sequence<Is...>) {
        std::ostringstream oss;
        oss << "(";
        size_t idx = 0;
        ((oss << (idx++ > 0 ? ", " : "") << domain_format<Domains>(std::get<Is>(v))), ...);
        oss << ")";
        return oss.str();
    }

#define VALHALL_TUPLE_BINARY_IMPL(op_name)                                                                  \
    template <size_t... Is, typename... Extra>                                                              \
    static Value op_name##_impl(const Value& a, const Value& b, std::index_sequence<Is...>,                 \
                                Extra&&... extra) {                                                         \
        return Value{Domains::op_name(std::get<Is>(a), std::get<Is>(b), extra...)...};                      \
    }

#define VALHALL_TUPLE_UNARY_IMPL(op_name)                                                                   \
    template <size_t... Is, typename... Extra>                                                              \
    static Value op_name##_impl(const Value& a, std::index_sequence<Is...>, Extra&&... extra) {             \
        return Value{Domains::op_name(std::get<Is>(a), extra...)...};                                       \
    }

    VALHALL_TUPLE_BINARY_IMPL(add)
    VALHALL_TUPLE_BINARY_IMPL(sub)
    VALHALL_TUPLE_BINARY_IMPL(mul)
    VALHALL_TUPLE_BINARY_IMPL(div)
    VALHALL_TUPLE_BINARY_IMPL(iadd)
    VALHALL_TUPLE_BINARY_IMPL(isub)
    VALHALL_TUPLE_BINARY_IMPL(imul)
    VALHALL_TUPLE_BINARY_IMPL(idiv)
    VALHALL_TUPLE_BINARY_IMPL(bit_and)
    VALHALL_TUPLE_BINARY_IMPL(bit_or)
    VALHALL_TUPLE_BINARY_IMPL(bit_xor)
    VALHALL_TUPLE_BINARY_IMPL(shift)
    VALHALL_TUPLE_BINARY_IMPL(pack64)

    VALHALL_TUPLE_UNARY_IMPL(bit_not)
    VALHALL_TUPLE_UNARY_IMPL(split64)
    VALHALL_TUPLE_UNARY_IMPL(tuple_extract)
    VALHALL_TUPLE_UNARY_IMPL(convert)

#undef VALHALL_TUPLE_BINARY_IMPL
#undef VALHALL_TUPLE_UNARY_IMPL
};


}  // namespace valhall::dataflow
