#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace valhall::dataflow {

/// Shift / rotate direction for normalized `shift` domain and visitor operations.
enum class ShiftKind : uint8_t {
    Shl = 0,
    ShrU,
    ShrS,
    RotL,
    RotR,
};

// Compile-Time Lattice Interface Enforcement, Auto-Equals & Optional Op Traits

namespace detail {

template <typename D, typename = void>
struct has_value_type : std::false_type {};
template <typename D>
struct has_value_type<D, std::void_t<typename D::Value>> : std::true_type {};

template <typename D, typename = void>
struct has_bottom : std::false_type {};
template <typename D>
struct has_bottom<D, std::enable_if_t<std::is_convertible_v<decltype(D::bottom()), typename D::Value>>>
    : std::true_type {};

template <typename D, typename = void>
struct has_top : std::false_type {};
template <typename D>
struct has_top<D, std::enable_if_t<std::is_convertible_v<decltype(D::top()), typename D::Value>>>
    : std::true_type {};

template <typename D, typename = void>
struct has_join : std::false_type {};
template <typename D>
struct has_join<D, std::enable_if_t<std::is_convertible_v<
    decltype(D::join(std::declval<const typename D::Value&>(), std::declval<const typename D::Value&>())),
    typename D::Value>>> : std::true_type {};

template <typename D, typename = void>
struct has_meet : std::false_type {};
template <typename D>
struct has_meet<D, std::enable_if_t<std::is_convertible_v<
    decltype(D::meet(std::declval<const typename D::Value&>(), std::declval<const typename D::Value&>())),
    typename D::Value>>> : std::true_type {};

template <typename D, typename = void>
struct has_leq : std::false_type {};
template <typename D>
struct has_leq<D, std::enable_if_t<std::is_convertible_v<
    decltype(D::leq(std::declval<const typename D::Value&>(), std::declval<const typename D::Value&>())),
    bool>>> : std::true_type {};

template <typename D, typename = void>
struct has_equals : std::false_type {};
template <typename D>
struct has_equals<D, std::enable_if_t<std::is_convertible_v<
    decltype(D::equals(std::declval<const typename D::Value&>(), std::declval<const typename D::Value&>())),
    bool>>> : std::true_type {};

template <typename D, typename = void>
struct has_const_helpers : std::false_type {};
template <typename D>
struct has_const_helpers<
    D,
    std::void_t<
        decltype(D::try_as_const_u64(std::declval<const typename D::Value&>(), std::declval<uint64_t*>())),
        decltype(D::make_const_u64(std::declval<uint64_t>()))>> : std::true_type {};

template <typename D, typename = void>
struct has_format : std::false_type {};
template <typename D>
struct has_format<D, std::enable_if_t<std::is_convertible_v<
    decltype(D::format(std::declval<const typename D::Value&>())),
    std::string>>> : std::true_type {};

// Variadic optional operation traits supporting arbitrary trailing arguments (e.g. `is64`, `lookup_srt`).
#define VALHALL_DEFINE_BINARY_OP_TRAIT(trait_name, method_name)                                             \
    template <typename D, typename ArgsTuple, typename = void>                                              \
    struct trait_name##_args : std::false_type {};                                                          \
    template <typename D, typename... Extra>                                                                \
    struct trait_name##_args<D, std::tuple<Extra...>, std::enable_if_t<std::is_convertible_v<               \
        decltype(D::method_name(std::declval<const typename D::Value&>(),                                   \
                                std::declval<const typename D::Value&>(),                                   \
                                std::declval<Extra>()...)),                                                 \
        typename D::Value>>> : std::true_type {};                                                           \
    template <typename D, typename... Extra>                                                                \
    inline constexpr bool trait_name##_v = trait_name##_args<D, std::tuple<Extra...>>::value;

#define VALHALL_DEFINE_UNARY_OP_TRAIT(trait_name, method_name)                                              \
    template <typename D, typename ArgsTuple, typename = void>                                              \
    struct trait_name##_args : std::false_type {};                                                          \
    template <typename D, typename... Extra>                                                                \
    struct trait_name##_args<D, std::tuple<Extra...>, std::enable_if_t<std::is_convertible_v<               \
        decltype(D::method_name(std::declval<const typename D::Value&>(),                                   \
                                std::declval<Extra>()...)),                                                 \
        typename D::Value>>> : std::true_type {};                                                           \
    template <typename D, typename... Extra>                                                                \
    inline constexpr bool trait_name##_v = trait_name##_args<D, std::tuple<Extra...>>::value;

VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_add, add)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_sub, sub)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_mul, mul)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_div, div)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_iadd, iadd)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_isub, isub)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_imul, imul)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_idiv, idiv)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_bit_and, bit_and)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_bit_or, bit_or)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_bit_xor, bit_xor)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_shift, shift)
VALHALL_DEFINE_BINARY_OP_TRAIT(has_domain_pack64, pack64)

VALHALL_DEFINE_UNARY_OP_TRAIT(has_domain_bit_not, bit_not)
VALHALL_DEFINE_UNARY_OP_TRAIT(has_domain_split64, split64)
VALHALL_DEFINE_UNARY_OP_TRAIT(has_domain_tuple_extract, tuple_extract)

#undef VALHALL_DEFINE_BINARY_OP_TRAIT
#undef VALHALL_DEFINE_UNARY_OP_TRAIT

// Convenience legacy trait structs matching `SsaDomainTransferVisitor` default call signatures.
template <typename D> struct has_domain_iadd : has_domain_iadd_args<D, std::tuple<bool>> {};
template <typename D> struct has_domain_isub : has_domain_isub_args<D, std::tuple<bool>> {};
template <typename D> struct has_domain_imul : has_domain_imul_args<D, std::tuple<bool>> {};
template <typename D> struct has_domain_idiv : has_domain_idiv_args<D, std::tuple<bool>> {};
template <typename D> struct has_domain_bit_and : has_domain_bit_and_args<D, std::tuple<bool>> {};
template <typename D> struct has_domain_bit_or : has_domain_bit_or_args<D, std::tuple<bool>> {};
template <typename D> struct has_domain_bit_xor : has_domain_bit_xor_args<D, std::tuple<bool>> {};
template <typename D> struct has_domain_bit_not : has_domain_bit_not_args<D, std::tuple<bool>> {};
template <typename D> struct has_domain_shift : has_domain_shift_args<D, std::tuple<ShiftKind, bool>> {};
template <typename D> struct has_domain_pack64 : has_domain_pack64_args<D, std::tuple<>> {};
template <typename D> struct has_domain_split64 : has_domain_split64_args<D, std::tuple<>> {};
template <typename D> struct has_domain_tuple_extract : has_domain_tuple_extract_args<D, std::tuple<uint32_t>> {};

}  // namespace detail

/// Enforces the Lattice Domain contract at the point of instantiation with
/// human-readable compile-time error messages.
template <typename DomainT>
struct ValidateLatticeDomain {
    static_assert(detail::has_value_type<DomainT>::value,
                  "Dataflow DomainT must declare `using Value = ...;`");
    static_assert(detail::has_bottom<DomainT>::value,
                  "Dataflow DomainT must implement `static Value bottom();`");
    static_assert(detail::has_top<DomainT>::value,
                  "Dataflow DomainT must implement `static Value top();`");
    static_assert(detail::has_join<DomainT>::value,
                  "Dataflow DomainT must implement `static Value join(const Value&, const Value&);`");
    static_assert(detail::has_meet<DomainT>::value,
                  "Dataflow DomainT must implement `static Value meet(const Value&, const Value&);`");
    static_assert(detail::has_leq<DomainT>::value,
                  "Dataflow DomainT must implement `static bool leq(const Value&, const Value&);`");
    static constexpr bool valid = true;
};

/// Canonical equality helper for any Lattice Domain.
template <typename DomainT>
inline bool domain_equals(const typename DomainT::Value& a, const typename DomainT::Value& b) {
    static_assert(ValidateLatticeDomain<DomainT>::valid, "");
    if constexpr (detail::has_equals<DomainT>::value) {
        return DomainT::equals(a, b);
    } else {
        return DomainT::leq(a, b) && DomainT::leq(b, a);
    }
}

/// Human-readable diagnostic formatter for any Lattice Domain.
template <typename DomainT>
inline std::string domain_format(const typename DomainT::Value& v) {
    static_assert(ValidateLatticeDomain<DomainT>::valid, "");
    if constexpr (detail::has_format<DomainT>::value) {
        return DomainT::format(v);
    } else {
        if (domain_equals<DomainT>(v, DomainT::bottom())) return "<bottom>";
        if (domain_equals<DomainT>(v, DomainT::top())) return "<top>";
        return "<val>";
    }
}

inline size_t next_power_of_two(size_t n) {
    if (n <= 1) return 1;
    size_t p = 1;
    while (p < n && p <= (static_cast<size_t>(-1) >> 1)) {
        p <<= 1;
    }
    return (p >= n) ? p : n;
}

// CRTP Point-Wise Lifting Mixin for Single-Base Wrapper Domains
//
// Any wrapper domain (VectorDomain, MapDomain, FiniteSetDomain, LaneDomainWrapper)
// that provides `lift_unary_op(a, fn)` and `lift_binary_op(a, b, fn)` inherits
// automatic SFINAE-constrained point-wise lifts for every optional operation
// implemented by `BaseDomain` (forwarding arbitrary extra arguments like `is64`
// or `lookup_srt`).
template <typename Derived, typename ValueT, typename BaseDomain>
struct PointwiseOpsMixin {
#define VALHALL_LIFT_BINARY_OP(op_name, trait_name)                                                         \
    template <typename... Extra, typename D = BaseDomain,                                                   \
              std::enable_if_t<detail::trait_name##_v<D, Extra...>, int> = 0>                               \
    static ValueT op_name(const ValueT& a, const ValueT& b,                                                 \
                          Extra&&... extra) {                                      \
        return Derived::lift_binary_op(a, b, [&](const auto& x, const auto& y) {                            \
            return D::op_name(x, y, std::forward<Extra>(extra)...);                                         \
        });                                                                                                 \
    }

#define VALHALL_LIFT_UNARY_OP(op_name, trait_name)                                                          \
    template <typename... Extra, typename D = BaseDomain,                                                   \
              std::enable_if_t<detail::trait_name##_v<D, Extra...>, int> = 0>                               \
    static ValueT op_name(const ValueT& a,                                                                  \
                          Extra&&... extra) {                                      \
        return Derived::lift_unary_op(a, [&](const auto& x) {                                               \
            return D::op_name(x, std::forward<Extra>(extra)...);                                            \
        });                                                                                                 \
    }

    VALHALL_LIFT_BINARY_OP(add, has_domain_add)
    VALHALL_LIFT_BINARY_OP(sub, has_domain_sub)
    VALHALL_LIFT_BINARY_OP(mul, has_domain_mul)
    VALHALL_LIFT_BINARY_OP(div, has_domain_div)
    VALHALL_LIFT_BINARY_OP(iadd, has_domain_iadd)
    VALHALL_LIFT_BINARY_OP(isub, has_domain_isub)
    VALHALL_LIFT_BINARY_OP(imul, has_domain_imul)
    VALHALL_LIFT_BINARY_OP(idiv, has_domain_idiv)
    VALHALL_LIFT_BINARY_OP(bit_and, has_domain_bit_and)
    VALHALL_LIFT_BINARY_OP(bit_or, has_domain_bit_or)
    VALHALL_LIFT_BINARY_OP(bit_xor, has_domain_bit_xor)
    VALHALL_LIFT_BINARY_OP(shift, has_domain_shift)
    VALHALL_LIFT_BINARY_OP(pack64, has_domain_pack64)

    VALHALL_LIFT_UNARY_OP(bit_not, has_domain_bit_not)
    VALHALL_LIFT_UNARY_OP(split64, has_domain_split64)
    VALHALL_LIFT_UNARY_OP(tuple_extract, has_domain_tuple_extract)

#undef VALHALL_LIFT_BINARY_OP
#undef VALHALL_LIFT_UNARY_OP
};

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
struct VectorDomain : public PointwiseOpsMixin<VectorDomain<ElemDomainT>, VectorDomainValue<ElemDomainT>, ElemDomainT> {
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
struct MapDomain : public PointwiseOpsMixin<MapDomain<KeyT, ValDomainT>, MapDomainValue<KeyT, ValDomainT>, ValDomainT> {
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

#undef VALHALL_TUPLE_BINARY_IMPL
#undef VALHALL_TUPLE_UNARY_IMPL
};

// BitsetDomain — Compact Powerset Lattice for Liveness / Reaching Defs

struct BitsetValue {
    std::vector<uint64_t> words;
    bool is_top_val = false;

    explicit BitsetValue(size_t num_bits = 0)
        : words((num_bits + 63u) / 64u, 0ULL) {}

    void ensure_bits(size_t num_bits) {
        size_t need = (num_bits + 63u) / 64u;
        if (words.size() < need) words.resize(need, 0ULL);
    }

    void set(size_t bit) {
        is_top_val = false;
        ensure_bits(bit + 1);
        words[bit >> 6] |= (1ULL << (bit & 63u));
    }

    void clear(size_t bit) {
        is_top_val = false;
        if ((bit >> 6) < words.size()) {
            words[bit >> 6] &= ~(1ULL << (bit & 63u));
        }
    }

    bool test(size_t bit) const {
        if (is_top_val) return true;
        size_t w = bit >> 6;
        return (w < words.size()) && ((words[w] & (1ULL << (bit & 63u))) != 0);
    }

    void clear_all() {
        is_top_val = false;
        std::fill(words.begin(), words.end(), 0ULL);
    }

    void or_with(const BitsetValue& other) {
        if (is_top_val) return;
        if (other.is_top_val) { is_top_val = true; return; }
        if (other.words.size() > words.size()) words.resize(other.words.size(), 0ULL);
        for (size_t i = 0; i < other.words.size(); ++i) words[i] |= other.words[i];
    }

    void or_diff_with(const BitsetValue& other, const BitsetValue& minus) {
        if (is_top_val) return;
        if (other.words.size() > words.size()) words.resize(other.words.size(), 0ULL);
        for (size_t i = 0; i < other.words.size(); ++i) {
            uint64_t m = (i < minus.words.size()) ? minus.words[i] : 0ULL;
            words[i] |= (other.words[i] & ~m);
        }
    }
};

struct BitsetDomain {
    using Value = BitsetValue;

    static Value bottom() { return Value(0); }
    static Value top() { Value v; v.is_top_val = true; return v; }

    static Value join(const Value& a, const Value& b) {
        if (a.is_top_val || b.is_top_val) return top();
        Value out = a;
        out.or_with(b);
        return out;
    }

    static Value meet(const Value& a, const Value& b) {
        if (a.is_top_val) return b;
        if (b.is_top_val) return a;
        size_t n = std::min(a.words.size(), b.words.size());
        Value out;
        out.words.resize(n, 0ULL);
        for (size_t i = 0; i < n; ++i) out.words[i] = a.words[i] & b.words[i];
        return out;
    }

    static Value diff(const Value& a, const Value& b) {
        Value out;
        out.words.resize(a.words.size(), 0ULL);
        for (size_t i = 0; i < a.words.size(); ++i) {
            uint64_t bw = (i < b.words.size()) ? b.words[i] : 0ULL;
            out.words[i] = a.words[i] & ~bw;
        }
        return out;
    }

    static bool leq(const Value& a, const Value& b) {
        if (b.is_top_val) return true;
        if (a.is_top_val) return false;
        size_t max_w = std::max(a.words.size(), b.words.size());
        for (size_t i = 0; i < max_w; ++i) {
            uint64_t aw = (i < a.words.size()) ? a.words[i] : 0ULL;
            uint64_t bw = (i < b.words.size()) ? b.words[i] : 0ULL;
            if ((aw & ~bw) != 0ULL) return false;
        }
        return true;
    }

    static bool equals(const Value& a, const Value& b) {
        if (a.is_top_val != b.is_top_val) return false;
        if (a.is_top_val) return true;
        size_t max_w = std::max(a.words.size(), b.words.size());
        for (size_t i = 0; i < max_w; ++i) {
            uint64_t aw = (i < a.words.size()) ? a.words[i] : 0ULL;
            uint64_t bw = (i < b.words.size()) ? b.words[i] : 0ULL;
            if (aw != bw) return false;
        }
        return true;
    }
};

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
struct FiniteSetDomain : public PointwiseOpsMixin<FiniteSetDomain<ElemDomainT, MaxK>, FiniteSetValue<ElemDomainT, MaxK>, ElemDomainT> {
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

// UnionDomain<Domains...> — Tagged Disjoint-Sum Lattice with Automatic Cross-Domain Op Dispatch

struct UnionBottomTag {
    bool operator==(const UnionBottomTag&) const { return true; }
};

struct UnionTopTag {
    bool operator==(const UnionTopTag&) const { return true; }
};

namespace detail {

// Map a component value type T back to its owning Domain in Domains... (or void for Bottom/Top tags).
template <typename T, typename... Domains>
struct domain_for_value {
    using type = void;
};

template <typename T, typename Head, typename... Tail>
struct domain_for_value<T, Head, Tail...> {
    using type = std::conditional_t<
        std::is_same_v<T, typename Head::Value>,
        Head,
        typename domain_for_value<T, Tail...>::type>;
};

template <typename T, typename... Domains>
using domain_for_value_t = typename domain_for_value<std::decay_t<T>, Domains...>::type;

template <typename T, typename... Domains>
inline constexpr bool is_union_member_value_v = (std::is_same_v<std::decay_t<T>, typename Domains::Value> || ...);

// Optional external customization point for cross-domain binary operations.
template <typename OpTag, typename DomainA, typename DomainB, typename = void>
struct CrossDomainOp {};

// Helper to detect and invoke `Domain::op(a, b, extra...)` for arbitrary argument types A, B.
#define VALHALL_DEFINE_UNION_BINARY_OP_TAG(TagName, method_name)                                            \
    struct TagName {                                                                                        \
        template <typename D, typename A, typename B, typename ArgsTuple, typename = void>                  \
        struct has_method : std::false_type {};                                                             \
        template <typename D, typename A, typename B, typename... Extra>                                    \
        struct has_method<D, A, B, std::tuple<Extra...>, std::void_t<                                       \
            decltype(D::method_name(std::declval<const A&>(), std::declval<const B&>(),                     \
                                    std::declval<Extra>()...))>> : std::true_type {};                       \
        template <typename D, typename A, typename B, typename... Extra>                                    \
        static decltype(auto) invoke(const A& a, const B& b, Extra&&... extra) {                            \
            return D::method_name(a, b, std::forward<Extra>(extra)...);                                     \
        }                                                                                                   \
    };

#define VALHALL_DEFINE_UNION_UNARY_OP_TAG(TagName, method_name)                                             \
    struct TagName {                                                                                        \
        template <typename D, typename A, typename ArgsTuple, typename = void>                              \
        struct has_method : std::false_type {};                                                             \
        template <typename D, typename A, typename... Extra>                                                \
        struct has_method<D, A, std::tuple<Extra...>, std::void_t<                                          \
            decltype(D::method_name(std::declval<const A&>(), std::declval<Extra>()...))>>                  \
            : std::true_type {};                                                                            \
        template <typename D, typename A, typename... Extra>                                                \
        static decltype(auto) invoke(const A& a, Extra&&... extra) {                                        \
            return D::method_name(a, std::forward<Extra>(extra)...);                                        \
        }                                                                                                   \
    };

VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpAdd, add)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpSub, sub)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpMul, mul)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpDiv, div)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpIAdd, iadd)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpISub, isub)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpIMul, imul)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpIDiv, idiv)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpBitAnd, bit_and)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpBitOr, bit_or)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpBitXor, bit_xor)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpShift, shift)
VALHALL_DEFINE_UNION_BINARY_OP_TAG(OpPack64, pack64)

VALHALL_DEFINE_UNION_UNARY_OP_TAG(OpBitNot, bit_not)
VALHALL_DEFINE_UNION_UNARY_OP_TAG(OpSplit64, split64)
VALHALL_DEFINE_UNION_UNARY_OP_TAG(OpTupleExtract, tuple_extract)

#undef VALHALL_DEFINE_UNION_BINARY_OP_TAG
#undef VALHALL_DEFINE_UNION_UNARY_OP_TAG

template <typename OpTag, typename DA, typename DB, typename A, typename B, typename ArgsTuple, typename = void>
struct has_cross_op : std::false_type {};
template <typename OpTag, typename DA, typename DB, typename A, typename B, typename... Extra>
struct has_cross_op<OpTag, DA, DB, A, B, std::tuple<Extra...>, std::void_t<
    decltype(CrossDomainOp<OpTag, DA, DB>::apply(std::declval<const A&>(), std::declval<const B&>(),
                                                 std::declval<Extra>()...))>> : std::true_type {};

}  // namespace detail

template <typename... Domains>
struct UnionValue {
    static_assert((ValidateLatticeDomain<Domains>::valid && ...), "");
    using Variant = std::variant<UnionBottomTag, UnionTopTag, typename Domains::Value...>;

    Variant storage = UnionBottomTag{};

    UnionValue() = default;
    UnionValue(UnionBottomTag) : storage(UnionBottomTag{}) {}
    UnionValue(UnionTopTag) : storage(UnionTopTag{}) {}

    template <typename T,
              std::enable_if_t<detail::is_union_member_value_v<T, Domains...>, int> = 0>
    UnionValue(T&& v) {
        using D = detail::domain_for_value_t<T, Domains...>;
        if (domain_equals<D>(v, D::bottom())) {
            storage = UnionBottomTag{};
        } else if (domain_equals<D>(v, D::top())) {
            storage = UnionTopTag{};
        } else {
            storage = std::forward<T>(v);
        }
    }

    bool is_bottom() const { return std::holds_alternative<UnionBottomTag>(storage); }
    bool is_top() const { return std::holds_alternative<UnionTopTag>(storage); }

    template <typename DomainT>
    bool is() const {
        return std::holds_alternative<typename DomainT::Value>(storage);
    }

    template <typename DomainT>
    const typename DomainT::Value* get_if() const & {
        return std::get_if<typename DomainT::Value>(&storage);
    }

    template <typename DomainT>
    typename DomainT::Value* get_if() & {
        return std::get_if<typename DomainT::Value>(&storage);
    }

    // Disallow taking a raw pointer into a temporary UnionValue
    template <typename DomainT>
    const typename DomainT::Value* get_if() const && = delete;
};

template <typename... Domains>
struct UnionDomain {
    static_assert((ValidateLatticeDomain<Domains>::valid && ...), "");
    using Value = UnionValue<Domains...>;

    static Value bottom() { return Value(UnionBottomTag{}); }
    static Value top() { return Value(UnionTopTag{}); }

    static Value join(const Value& a, const Value& b) {
        if (a.is_bottom()) return b;
        if (b.is_bottom()) return a;
        if (a.is_top() || b.is_top()) return top();
        if (a.storage.index() != b.storage.index()) return top();
        return std::visit([](const auto& va, const auto& vb) -> Value {
            using TA = std::decay_t<decltype(va)>;
            using TB = std::decay_t<decltype(vb)>;
            if constexpr (std::is_same_v<TA, TB> && detail::is_union_member_value_v<TA, Domains...>) {
                using D = detail::domain_for_value_t<TA, Domains...>;
                return Value(D::join(va, vb));
            } else {
                return top();
            }
        }, a.storage, b.storage);
    }

    static Value meet(const Value& a, const Value& b) {
        if (a.is_top()) return b;
        if (b.is_top()) return a;
        if (a.is_bottom() || b.is_bottom()) return bottom();
        if (a.storage.index() != b.storage.index()) return bottom();
        return std::visit([](const auto& va, const auto& vb) -> Value {
            using TA = std::decay_t<decltype(va)>;
            using TB = std::decay_t<decltype(vb)>;
            if constexpr (std::is_same_v<TA, TB> && detail::is_union_member_value_v<TA, Domains...>) {
                using D = detail::domain_for_value_t<TA, Domains...>;
                return Value(D::meet(va, vb));
            } else {
                return bottom();
            }
        }, a.storage, b.storage);
    }

    static bool leq(const Value& a, const Value& b) {
        if (a.is_bottom() || b.is_top()) return true;
        if (b.is_bottom() || a.is_top()) return false;
        if (a.storage.index() != b.storage.index()) return false;
        return std::visit([](const auto& va, const auto& vb) -> bool {
            using TA = std::decay_t<decltype(va)>;
            using TB = std::decay_t<decltype(vb)>;
            if constexpr (std::is_same_v<TA, TB> && detail::is_union_member_value_v<TA, Domains...>) {
                using D = detail::domain_for_value_t<TA, Domains...>;
                return D::leq(va, vb);
            } else {
                return false;
            }
        }, a.storage, b.storage);
    }

    static bool equals(const Value& a, const Value& b) {
        if (a.storage.index() != b.storage.index()) return false;
        if (a.is_bottom() || a.is_top()) return true;
        return std::visit([](const auto& va, const auto& vb) -> bool {
            using TA = std::decay_t<decltype(va)>;
            using TB = std::decay_t<decltype(vb)>;
            if constexpr (std::is_same_v<TA, TB> && detail::is_union_member_value_v<TA, Domains...>) {
                using D = detail::domain_for_value_t<TA, Domains...>;
                return domain_equals<D>(va, vb);
            } else {
                return false;
            }
        }, a.storage, b.storage);
    }

    static std::string format(const Value& v) {
        return std::visit([](const auto& inner) -> std::string {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, UnionBottomTag>) return "<bottom>";
            else if constexpr (std::is_same_v<T, UnionTopTag>) return "<top>";
            else {
                using D = detail::domain_for_value_t<T, Domains...>;
                return domain_format<D>(inner);
            }
        }, v.storage);
    }

    static bool try_as_const_u64(const Value& v, uint64_t* out) {
        return std::visit([out](const auto& inner) -> bool {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (detail::is_union_member_value_v<T, Domains...>) {
                using D = detail::domain_for_value_t<T, Domains...>;
                if constexpr (detail::has_const_helpers<D>::value) {
                    return D::try_as_const_u64(inner, out);
                }
            }
            (void)out;
            return false;
        }, v.storage);
    }

    static Value make_const_u64(uint64_t c) {
        Value result = top();
        bool found = false;
        auto try_make = [&](auto tag) {
            using D = typename decltype(tag)::type;
            if (!found) {
                if constexpr (detail::has_const_helpers<D>::value) {
                    result = Value(D::make_const_u64(c));
                    found = true;
                }
            }
        };
        (try_make(type_tag<Domains>{}), ...);
        return result;
    }

    // Core generic binary dispatch: flows `OpTag` across matching or cross-domain
    // alternatives whenever `DA::op(a, b, ...)`, `DB::op(a, b, ...)`, or
    // `CrossDomainOp<OpTag, DA, DB>::apply(a, b, ...)` is defined.
    template <typename OpTag, typename... Extra>
    static Value dispatch_binary(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        return std::visit([&](const auto& va, const auto& vb) -> Value {
            using TA = std::decay_t<decltype(va)>;
            using TB = std::decay_t<decltype(vb)>;
            using DA = detail::domain_for_value_t<TA, Domains...>;
            using DB = detail::domain_for_value_t<TB, Domains...>;

            if constexpr (std::is_same_v<TA, UnionBottomTag> || std::is_same_v<TB, UnionBottomTag>) {
                return bottom();
            } else if constexpr (!std::is_void_v<DA> &&
                                 OpTag::template has_method<DA, TA, TB, std::tuple<Extra...>>::value) {
                return Value(OpTag::template invoke<DA>(va, vb, std::forward<Extra>(extra)...));
            } else if constexpr (!std::is_void_v<DA> && (sizeof...(Extra) > 0) &&
                                 OpTag::template has_method<DA, TA, TB, std::tuple<>>::value) {
                return Value(OpTag::template invoke<DA>(va, vb));
            } else if constexpr (!std::is_void_v<DB> && !std::is_same_v<DA, DB> &&
                                 OpTag::template has_method<DB, TA, TB, std::tuple<Extra...>>::value) {
                return Value(OpTag::template invoke<DB>(va, vb, std::forward<Extra>(extra)...));
            } else if constexpr (!std::is_void_v<DB> && !std::is_same_v<DA, DB> && (sizeof...(Extra) > 0) &&
                                 OpTag::template has_method<DB, TA, TB, std::tuple<>>::value) {
                return Value(OpTag::template invoke<DB>(va, vb));
            } else if constexpr (detail::has_cross_op<OpTag,
                                                      std::conditional_t<std::is_void_v<DA>, TA, DA>,
                                                      std::conditional_t<std::is_void_v<DB>, TB, DB>,
                                                      TA, TB, std::tuple<Extra...>>::value) {
                using CA = std::conditional_t<std::is_void_v<DA>, TA, DA>;
                using CB = std::conditional_t<std::is_void_v<DB>, TB, DB>;
                return Value(detail::CrossDomainOp<OpTag, CA, CB>::apply(
                    va, vb, std::forward<Extra>(extra)...));
            } else {
                return top();
            }
        }, a.storage, b.storage);
    }

    template <typename OpTag, typename... Extra>
    static Value dispatch_unary(const Value& a, Extra&&... extra) {
        if (a.is_bottom()) return bottom();
        if (a.is_top()) return top();
        return std::visit([&](const auto& va) -> Value {
            using TA = std::decay_t<decltype(va)>;
            using DA = detail::domain_for_value_t<TA, Domains...>;
            if constexpr (!std::is_void_v<DA> &&
                          OpTag::template has_method<DA, TA, std::tuple<Extra...>>::value) {
                return Value(OpTag::template invoke<DA>(va, std::forward<Extra>(extra)...));
            } else if constexpr (!std::is_void_v<DA> && (sizeof...(Extra) > 0) &&
                                 OpTag::template has_method<DA, TA, std::tuple<>>::value) {
                return Value(OpTag::template invoke<DA>(va));
            } else {
                return top();
            }
        }, a.storage);
    }

    // SFINAE-enabled algebraic & logical operations with universal lattice-agnostic
    // identity short-circuits (e.g. x + 0 == x, x * 1 == x, x & 0 == 0).
    template <typename... Extra,
              bool En = ((detail::has_domain_add_v<Domains, Extra...> || ...) || (detail::has_domain_add_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value add(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        if (try_as_const_u64(a, &c) && c == 0) return b;
        return dispatch_binary<detail::OpAdd>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_sub_v<Domains, Extra...> || ...) || (detail::has_domain_sub_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value sub(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        return dispatch_binary<detail::OpSub>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_mul_v<Domains, Extra...> || ...) || (detail::has_domain_mul_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value mul(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t ca = 0, cb = 0;
        bool ha = try_as_const_u64(a, &ca), hb = try_as_const_u64(b, &cb);
        if ((ha && ca == 0) || (hb && cb == 0)) return make_const_u64(0);
        if (hb && cb == 1) return a;
        if (ha && ca == 1) return b;
        return dispatch_binary<detail::OpMul>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_div_v<Domains, Extra...> || ...) || (detail::has_domain_div_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value div(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t cb = 0;
        if (try_as_const_u64(b, &cb) && cb == 1) return a;
        return dispatch_binary<detail::OpDiv>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_iadd_v<Domains, Extra...> || ...) || (detail::has_domain_iadd_v<Domains, bool> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value iadd(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        if (try_as_const_u64(a, &c) && c == 0) return b;
        return dispatch_binary<detail::OpIAdd>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_isub_v<Domains, Extra...> || ...) || (detail::has_domain_isub_v<Domains, bool> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value isub(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        return dispatch_binary<detail::OpISub>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_imul_v<Domains, Extra...> || ...) || (detail::has_domain_imul_v<Domains, bool> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value imul(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t ca = 0, cb = 0;
        bool ha = try_as_const_u64(a, &ca), hb = try_as_const_u64(b, &cb);
        if ((ha && ca == 0) || (hb && cb == 0)) return make_const_u64(0);
        if (hb && cb == 1) return a;
        if (ha && ca == 1) return b;
        return dispatch_binary<detail::OpIMul>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_idiv_v<Domains, Extra...> || ...) || (detail::has_domain_idiv_v<Domains, bool> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value idiv(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t cb = 0;
        if (try_as_const_u64(b, &cb) && cb == 1) return a;
        return dispatch_binary<detail::OpIDiv>(a, b, std::forward<Extra>(extra)...);
    }

    template <bool En = ((detail::has_domain_bit_and_v<Domains, bool> || ...) || (detail::has_domain_bit_and_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value bit_and(const Value& a, const Value& b, bool is64 = true) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t ca = 0, cb = 0;
        bool ha = try_as_const_u64(a, &ca), hb = try_as_const_u64(b, &cb);
        const uint64_t all_ones = is64 ? ~0ULL : 0xFFFFFFFFULL;
        if ((ha && ca == 0) || (hb && cb == 0)) return make_const_u64(0);
        if (hb && (cb & all_ones) == all_ones) return a;
        if (ha && (ca & all_ones) == all_ones) return b;
        return dispatch_binary<detail::OpBitAnd>(a, b, is64);
    }

    template <bool En = ((detail::has_domain_bit_or_v<Domains, bool> || ...) || (detail::has_domain_bit_or_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value bit_or(const Value& a, const Value& b, bool is64 = true) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        if (try_as_const_u64(a, &c) && c == 0) return b;
        return dispatch_binary<detail::OpBitOr>(a, b, is64);
    }

    template <bool En = ((detail::has_domain_bit_xor_v<Domains, bool> || ...) || (detail::has_domain_bit_xor_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value bit_xor(const Value& a, const Value& b, bool is64 = true) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        if (try_as_const_u64(a, &c) && c == 0) return b;
        return dispatch_binary<detail::OpBitXor>(a, b, is64);
    }

    template <bool En = ((detail::has_domain_bit_not_v<Domains, bool> || ...) || (detail::has_domain_bit_not_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value bit_not(const Value& a, bool is64 = true) {
        return dispatch_unary<detail::OpBitNot>(a, is64);
    }

    template <bool En = (detail::has_domain_shift_v<Domains, ShiftKind, bool> || ...),
              std::enable_if_t<En, int> = 0>
    static Value shift(const Value& a, const Value& b, ShiftKind kind, bool is64 = true) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t cb = 0;
        if (try_as_const_u64(b, &cb) && cb == 0) return a;
        return dispatch_binary<detail::OpShift>(a, b, kind, is64);
    }

    template <bool En = (detail::has_domain_pack64_v<Domains> || ...),
              std::enable_if_t<En, int> = 0>
    static Value pack64(const Value& lo, const Value& hi) {
        return dispatch_binary<detail::OpPack64>(lo, hi);
    }

    template <bool En = (detail::has_domain_split64_v<Domains> || ...),
              std::enable_if_t<En, int> = 0>
    static Value split64(const Value& src64) {
        return dispatch_unary<detail::OpSplit64>(src64);
    }

    template <bool En = (detail::has_domain_tuple_extract_v<Domains, uint32_t> || ...),
              std::enable_if_t<En, int> = 0>
    static Value tuple_extract(const Value& tup, uint32_t elem_idx) {
        return dispatch_unary<detail::OpTupleExtract>(tup, elem_idx);
    }

private:
    template <typename T>
    struct type_tag { using type = T; };
};

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
struct LaneDomainWrapper : public PointwiseOpsMixin<LaneDomainWrapper<InnerDomainT>, LaneValue<InnerDomainT>, InnerDomainT> {
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

    // Point-wise lifting hooks for PointwiseOpsMixin
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
