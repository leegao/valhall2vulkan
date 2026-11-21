#pragma once

#include "traits.h"
#include <cstdint>
#include <type_traits>
#include <utility>

namespace valhall::dataflow {

// Algebraic Action Mixins for Domain Composition

/// CRTP mixin expressing that `Derived` supports commutative & associative
/// additive shifts with `IntDomainT` (`base + z`, `z + base`, `base - z`), plus
/// dynamic-offset widening (`base + dynamic -> base + Top_Z`) for non-`ValueT`
/// operands. `Derived` implements `apply_int_delta(base, scalar, extra...)` and
/// `widen_unknown_offset(base)`.
template <typename Derived, typename ValueT, typename IntDomainT>
struct IntegralAdditiveActionMixin {
    using IntValue = typename IntDomainT::Value;
    using IntScalar = typename IntDomainT::Scalar;

    // Single templated `add` handling (ValueT, IntValue), (IntValue, ValueT),
    // (ValueT, OtherVal), and (OtherVal, ValueT) via enable_if. Using one overload
    // instead of multiple swapped-arg overloads avoids a clangd 21.1.8 segfault
    // during SFINAE overload resolution.
    template <typename A, typename B, typename... Extra,
              std::enable_if_t<
                  (std::is_same_v<std::decay_t<A>, ValueT> &&
                   (std::is_same_v<std::decay_t<B>, IntValue> ||
                    (!std::is_same_v<std::decay_t<B>, ValueT> &&
                     !std::is_same_v<std::decay_t<B>, IntValue>))) ||
                  (std::is_same_v<std::decay_t<A>, IntValue> &&
                   std::is_same_v<std::decay_t<B>, ValueT>) ||
                  (!std::is_same_v<std::decay_t<A>, ValueT> &&
                   !std::is_same_v<std::decay_t<A>, IntValue> &&
                   std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT add(const A& a, const B& b, Extra&&... extra) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT> &&
                      std::is_same_v<std::decay_t<B>, IntValue>) {
            const ValueT& base = a;
            const IntValue& delta = b;
            if (domain_equals<Derived>(base, Derived::bottom()) || delta.is_bottom()) return Derived::bottom();
            if (domain_equals<Derived>(base, Derived::top())) return Derived::top();
            if (delta.is_known()) {
                if (delta.value == 0) return base;
                return Derived::apply_int_delta(base, delta.value, std::forward<Extra>(extra)...);
            }
            return Derived::widen_unknown_offset(base);
        } else if constexpr (std::is_same_v<std::decay_t<A>, IntValue> &&
                             std::is_same_v<std::decay_t<B>, ValueT>) {
            const IntValue& delta = a;
            const ValueT& base = b;
            if (domain_equals<Derived>(base, Derived::bottom()) || delta.is_bottom()) return Derived::bottom();
            if (domain_equals<Derived>(base, Derived::top())) return Derived::top();
            if (delta.is_known()) {
                if (delta.value == 0) return base;
                return Derived::apply_int_delta(base, delta.value, std::forward<Extra>(extra)...);
            }
            return Derived::widen_unknown_offset(base);
        } else if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            // (ValueT, OtherVal) -> widen
            return Derived::widen_unknown_offset(a);
        } else {
            // (OtherVal, ValueT) -> widen
            return Derived::widen_unknown_offset(b);
        }
    }

    // Single templated `sub` handling (ValueT, IntValue) and (ValueT, OtherVal).
    template <typename A, typename B, typename... Extra,
              std::enable_if_t<std::is_same_v<std::decay_t<A>, ValueT> &&
                               (std::is_same_v<std::decay_t<B>, IntValue> ||
                                (!std::is_same_v<std::decay_t<B>, ValueT> &&
                                 !std::is_same_v<std::decay_t<B>, IntValue>)), int> = 0>
    static ValueT sub(const A& base, const B& delta, Extra&&... extra) {
        if constexpr (std::is_same_v<std::decay_t<B>, IntValue>) {
            if (domain_equals<Derived>(base, Derived::bottom()) || delta.is_bottom()) return Derived::bottom();
            if (domain_equals<Derived>(base, Derived::top())) return Derived::top();
            if (delta.is_known()) {
                if (delta.value == 0) return base;
                return Derived::apply_int_delta(base, static_cast<IntScalar>(-delta.value),
                                                std::forward<Extra>(extra)...);
            }
            return Derived::widen_unknown_offset(base);
        } else {
            return Derived::widen_unknown_offset(base);
        }
    }

    // Single templated `iadd` handling (ValueT, IntValue), (IntValue, ValueT),
    // (ValueT, OtherVal), and (OtherVal, ValueT) via enable_if.
    template <typename A, typename B,
              std::enable_if_t<(std::is_same_v<std::decay_t<A>, ValueT> &&
                                (std::is_same_v<std::decay_t<B>, IntValue> ||
                                 (!std::is_same_v<std::decay_t<B>, ValueT> &&
                                  !std::is_same_v<std::decay_t<B>, IntValue>))) ||
                               (std::is_same_v<std::decay_t<A>, IntValue> &&
                                std::is_same_v<std::decay_t<B>, ValueT>) ||
                               (!std::is_same_v<std::decay_t<A>, ValueT> &&
                                !std::is_same_v<std::decay_t<A>, IntValue> &&
                                std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT iadd(const A& a, const B& b, bool is64 = true) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT> &&
                      std::is_same_v<std::decay_t<B>, IntValue>) {
            return add(a, b);
        } else if constexpr (std::is_same_v<std::decay_t<A>, IntValue> &&
                             std::is_same_v<std::decay_t<B>, ValueT>) {
            return add(b, a);
        } else if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            return Derived::widen_unknown_offset(a);
        } else {
            return Derived::widen_unknown_offset(b);
        }
    }

    // Single templated `isub` handling (ValueT, IntValue) and (ValueT, OtherVal).
    template <typename A, typename B,
              std::enable_if_t<std::is_same_v<std::decay_t<A>, ValueT> &&
                               (std::is_same_v<std::decay_t<B>, IntValue> ||
                                (!std::is_same_v<std::decay_t<B>, ValueT> &&
                                 !std::is_same_v<std::decay_t<B>, IntValue>)), int> = 0>
    static ValueT isub(const A& base, const B& delta, bool is64 = true) {
        if constexpr (std::is_same_v<std::decay_t<B>, IntValue>) {
            return sub(base, delta);
        } else {
            return Derived::widen_unknown_offset(base);
        }
    }
};

/// CRTP mixin expressing that `Derived` supports commutative & associative
/// bitwise AND masking with `IntDomainT` (`base & mask`, `mask & base`).
/// `Derived` implements `apply_int_mask(base, scalar_mask)`.
template <typename Derived, typename ValueT, typename IntDomainT>
struct IntegralBitmaskActionMixin {
    using IntValue = typename IntDomainT::Value;

    // Single templated `bit_and` handling both (ValueT, IntValue) and (IntValue, ValueT)
    // argument orders via enable_if. Using one overload instead of two swapped-arg
    // overloads avoids a clangd 21.1.8 segfault during SFINAE overload resolution.
    template <typename A, typename B,
              std::enable_if_t<(std::is_same_v<std::decay_t<A>, ValueT> &&
                                std::is_same_v<std::decay_t<B>, IntValue>) ||
                               (std::is_same_v<std::decay_t<A>, IntValue> &&
                                std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT bit_and(const A& a, const B& b, bool is64 = true) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            const ValueT& base = a;
            const IntValue& mask = b;
            if (domain_equals<Derived>(base, Derived::bottom()) || mask.is_bottom()) return Derived::bottom();
            if (domain_equals<Derived>(base, Derived::top()) || !mask.is_known()) return Derived::top();
            const uint64_t all_ones = is64 ? ~0ULL : 0xFFFFFFFFULL;
            if ((static_cast<uint64_t>(mask.value) & all_ones) == all_ones) return base;
            if (mask.value == 0) return Derived::top();
            return Derived::apply_int_mask(base, mask.value);
        } else {
            const IntValue& mask = a;
            const ValueT& base = b;
            if (domain_equals<Derived>(base, Derived::bottom()) || mask.is_bottom()) return Derived::bottom();
            if (domain_equals<Derived>(base, Derived::top()) || !mask.is_known()) return Derived::top();
            const uint64_t all_ones = is64 ? ~0ULL : 0xFFFFFFFFULL;
            if ((static_cast<uint64_t>(mask.value) & all_ones) == all_ones) return base;
            if (mask.value == 0) return Derived::top();
            return Derived::apply_int_mask(base, mask.value);
        }
    }
};

/// CRTP mixin expressing that `Derived` can be arbitrarily combined under
/// integer arithmetic (`add`, `sub`, `mul`, `div`, `iadd`, `isub`, `imul`,
/// `idiv`) with `IntDomainT` and any other non-excluded domain value,
/// transforming the value via `Derived::combine_with_scalar(v)`.
/// Automatically propagates through `UnionDomain` via SFINAE dispatch.
template <typename Derived, typename ValueT, typename IntDomainT, typename... ExcludedValues>
struct IntegralCombinableMixin {
    template <typename OtherVal>
    static constexpr bool is_excluded_v = (std::is_same_v<std::decay_t<OtherVal>, ExcludedValues> || ...);

    // Single templated `add` handling (ValueT, OtherVal) and (OtherVal, ValueT) via enable_if.
    // Using one overload instead of two swapped-arg overloads avoids a clangd 21.1.8 segfault
    // during SFINAE overload resolution.
    template <typename A, typename B,
              std::enable_if_t<
                  (std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<B>) ||
                  (!std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<A> &&
                   std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT add(const A& a, const B& b) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            return Derived::combine_with_scalar(a);
        } else {
            return Derived::combine_with_scalar(b);
        }
    }

    // Single templated `sub` handling (ValueT, OtherVal) and (OtherVal, ValueT) via enable_if.
    template <typename A, typename B,
              std::enable_if_t<
                  (std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<B>) ||
                  (!std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<A> &&
                   std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT sub(const A& a, const B& b) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            return Derived::combine_with_scalar(a);
        } else {
            return Derived::combine_with_scalar(b);
        }
    }

    // Single templated `mul` handling (ValueT, OtherVal) and (OtherVal, ValueT) via enable_if.
    template <typename A, typename B,
              std::enable_if_t<
                  (std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<B>) ||
                  (!std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<A> &&
                   std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT mul(const A& a, const B& b) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            return Derived::combine_with_scalar(a);
        } else {
            return Derived::combine_with_scalar(b);
        }
    }

    // Single templated `div` handling (ValueT, OtherVal) and (OtherVal, ValueT) via enable_if.
    template <typename A, typename B,
              std::enable_if_t<
                  (std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<B>) ||
                  (!std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<A> &&
                   std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT div(const A& a, const B& b) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            return Derived::combine_with_scalar(a);
        } else {
            return Derived::combine_with_scalar(b);
        }
    }

    // Single templated `iadd` handling (ValueT, OtherVal) and (OtherVal, ValueT) via enable_if.
    template <typename A, typename B,
              std::enable_if_t<
                  (std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<B>) ||
                  (!std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<A> &&
                   std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT iadd(const A& a, const B& b, bool /*is64*/ = true) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            return Derived::combine_with_scalar(a);
        } else {
            return Derived::combine_with_scalar(b);
        }
    }

    // Single templated `isub` handling (ValueT, OtherVal) and (OtherVal, ValueT) via enable_if.
    template <typename A, typename B,
              std::enable_if_t<
                  (std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<B>) ||
                  (!std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<A> &&
                   std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT isub(const A& a, const B& b, bool /*is64*/ = true) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            return Derived::combine_with_scalar(a);
        } else {
            return Derived::combine_with_scalar(b);
        }
    }

    // Single templated `imul` handling (ValueT, OtherVal) and (OtherVal, ValueT) via enable_if.
    template <typename A, typename B,
              std::enable_if_t<
                  (std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<B>) ||
                  (!std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<A> &&
                   std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT imul(const A& a, const B& b, bool /*is64*/ = true) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            return Derived::combine_with_scalar(a);
        } else {
            return Derived::combine_with_scalar(b);
        }
    }

    // Single templated `idiv` handling (ValueT, OtherVal) and (OtherVal, ValueT) via enable_if.
    template <typename A, typename B,
              std::enable_if_t<
                  (std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<B>) ||
                  (!std::is_same_v<std::decay_t<A>, ValueT> && !is_excluded_v<A> &&
                   std::is_same_v<std::decay_t<B>, ValueT>), int> = 0>
    static ValueT idiv(const A& a, const B& b, bool /*is64*/ = true) {
        if constexpr (std::is_same_v<std::decay_t<A>, ValueT>) {
            return Derived::combine_with_scalar(a);
        } else {
            return Derived::combine_with_scalar(b);
        }
    }
};

// CRTP Point-Wise Lifting Mixin for Single-Base Wrapper Domains

template <typename Derived, typename ValueT, typename BaseDomain>
struct PointwiseOpsMixin {
#define VALHALL_LIFT_BINARY_OP(op_name, trait_name)                                                         \
    template <typename... Extra, typename D = BaseDomain,                                                   \
              std::enable_if_t<detail::trait_name##_v<D, Extra...>, int> = 0>                               \
    static ValueT op_name(const ValueT& a, const ValueT& b,                                                 \
                          Extra&&... extra) {                                                               \
        return Derived::lift_binary_op(a, b, [&](const auto& x, const auto& y) {                            \
            return D::op_name(x, y, std::forward<Extra>(extra)...);                                         \
        });                                                                                                 \
    }

#define VALHALL_LIFT_UNARY_OP(op_name, trait_name)                                                          \
    template <typename... Extra, typename D = BaseDomain,                                                   \
              std::enable_if_t<detail::trait_name##_v<D, Extra...>, int> = 0>                               \
    static ValueT op_name(const ValueT& a,                                                                  \
                          Extra&&... extra) {                                                               \
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
    VALHALL_LIFT_UNARY_OP(convert, has_domain_convert)

#undef VALHALL_LIFT_BINARY_OP
#undef VALHALL_LIFT_UNARY_OP
};


}  // namespace valhall::dataflow
