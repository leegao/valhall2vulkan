#pragma once

#include "traits.h"
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace valhall::dataflow {

// UnionDomain<Domains...> — Tagged Disjoint-Sum Lattice with Automatic Cross-Domain Op Dispatch

struct UnionBottomTag {
    bool operator==(const UnionBottomTag&) const { return true; }
};

struct UnionTopTag {
    bool operator==(const UnionTopTag&) const { return true; }
};

namespace detail {

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

template <typename OpTag, typename DomainA, typename DomainB, typename = void>
struct CrossDomainOp {};

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
VALHALL_DEFINE_UNION_UNARY_OP_TAG(OpConvert, convert)

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

    template <typename... Extra,
              bool En = ((detail::has_domain_add_v<Domains, Extra...> || ...) ||
                         (detail::has_domain_add_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value add(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        if (try_as_const_u64(a, &c) && c == 0) return b;
        return dispatch_binary<detail::OpAdd>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_sub_v<Domains, Extra...> || ...) ||
                         (detail::has_domain_sub_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value sub(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        return dispatch_binary<detail::OpSub>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_mul_v<Domains, Extra...> || ...) ||
                         (detail::has_domain_mul_v<Domains> || ...)),
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
              bool En = ((detail::has_domain_div_v<Domains, Extra...> || ...) ||
                         (detail::has_domain_div_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value div(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t cb = 0;
        if (try_as_const_u64(b, &cb) && cb == 1) return a;
        return dispatch_binary<detail::OpDiv>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_iadd_v<Domains, Extra...> || ...) ||
                         (detail::has_domain_iadd_v<Domains, bool> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value iadd(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        if (try_as_const_u64(a, &c) && c == 0) return b;
        return dispatch_binary<detail::OpIAdd>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_isub_v<Domains, Extra...> || ...) ||
                         (detail::has_domain_isub_v<Domains, bool> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value isub(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        return dispatch_binary<detail::OpISub>(a, b, std::forward<Extra>(extra)...);
    }

    template <typename... Extra,
              bool En = ((detail::has_domain_imul_v<Domains, Extra...> || ...) ||
                         (detail::has_domain_imul_v<Domains, bool> || ...)),
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
              bool En = ((detail::has_domain_idiv_v<Domains, Extra...> || ...) ||
                         (detail::has_domain_idiv_v<Domains, bool> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value idiv(const Value& a, const Value& b, Extra&&... extra) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t cb = 0;
        if (try_as_const_u64(b, &cb) && cb == 1) return a;
        return dispatch_binary<detail::OpIDiv>(a, b, std::forward<Extra>(extra)...);
    }

    template <bool En = ((detail::has_domain_bit_and_v<Domains, bool> || ...) ||
                         (detail::has_domain_bit_and_v<Domains> || ...)),
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

    template <bool En = ((detail::has_domain_bit_or_v<Domains, bool> || ...) ||
                         (detail::has_domain_bit_or_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value bit_or(const Value& a, const Value& b, bool is64 = true) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        if (try_as_const_u64(a, &c) && c == 0) return b;
        return dispatch_binary<detail::OpBitOr>(a, b, is64);
    }

    template <bool En = ((detail::has_domain_bit_xor_v<Domains, bool> || ...) ||
                         (detail::has_domain_bit_xor_v<Domains> || ...)),
              std::enable_if_t<En, int> = 0>
    static Value bit_xor(const Value& a, const Value& b, bool is64 = true) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        uint64_t c = 0;
        if (try_as_const_u64(b, &c) && c == 0) return a;
        if (try_as_const_u64(a, &c) && c == 0) return b;
        return dispatch_binary<detail::OpBitXor>(a, b, is64);
    }

    template <bool En = ((detail::has_domain_bit_not_v<Domains, bool> || ...) ||
                         (detail::has_domain_bit_not_v<Domains> || ...)),
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

    template <bool En = (detail::has_domain_convert_v<Domains, ConvertKind> || ...),
              std::enable_if_t<En, int> = 0>
    static Value convert(const Value& a, ConvertKind kind) {
        return dispatch_unary<detail::OpConvert>(a, kind);
    }

private:
    template <typename T>
    struct type_tag { using type = T; };
};


}  // namespace valhall::dataflow
