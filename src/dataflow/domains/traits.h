#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

namespace valhall::dataflow {

/// Shift / rotate direction for normalized `shift` domain and visitor operations.
enum class ShiftKind : uint8_t {
    Shl = 0,
    ShrU,
    ShrS,
    RotL,
    RotR,
};

/// Normalized scalar conversion kind for `convert` domain and visitor operations.
enum class ConvertKind : uint8_t {
    U32ToU64 = 0,
    S32ToS64,
    U64ToU32,
    FloatToSInt,
    FloatToUInt,
    SIntToFloat,
    UIntToFloat,
    Bitcast,
    Other,
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
VALHALL_DEFINE_UNARY_OP_TRAIT(has_domain_convert, convert)

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
template <typename D> struct has_domain_convert : has_domain_convert_args<D, std::tuple<ConvertKind>> {};

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

// FlatDomain<T> — Standard Flat Lattice Mixin (Bottom < {T} < Top, height = 2)


}  // namespace valhall::dataflow
