#pragma once

#include "traits.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace valhall::dataflow {

enum class FlatState : uint8_t {
    Bottom = 0,
    Known,
    Top,
};

template <typename T>
inline bool flat_payload_equals(const T& a, const T& b) {
    if constexpr (std::is_floating_point_v<T>) {
        // Bitwise comparison ensures NaN == NaN (lattice reflexivity) and distinguishes -0.0 / +0.0.
        return std::memcmp(&a, &b, sizeof(T)) == 0;
    } else {
        return a == b;
    }
}

template <typename T>
struct FlatValue {
    FlatState state = FlatState::Bottom;
    T value{};

    static FlatValue make_bottom() { return FlatValue{FlatState::Bottom, T{}}; }
    static FlatValue make_top() { return FlatValue{FlatState::Top, T{}}; }
    static FlatValue make(T v) { return FlatValue{FlatState::Known, std::move(v)}; }

    bool is_bottom() const { return state == FlatState::Bottom; }
    bool is_top() const { return state == FlatState::Top; }
    bool is_known() const { return state == FlatState::Known; }
};

template <typename T>
struct FlatDomain {
    using Scalar = T;
    using Value = FlatValue<T>;

    static Value bottom() { return Value::make_bottom(); }
    static Value top() { return Value::make_top(); }

    static Value join(const Value& a, const Value& b) {
        if (a.is_bottom()) return b;
        if (b.is_bottom()) return a;
        if (a.is_top() || b.is_top()) return top();
        return flat_payload_equals(a.value, b.value) ? a : top();
    }

    static Value meet(const Value& a, const Value& b) {
        if (a.is_top()) return b;
        if (b.is_top()) return a;
        if (a.is_bottom() || b.is_bottom()) return bottom();
        return flat_payload_equals(a.value, b.value) ? a : bottom();
    }

    static bool leq(const Value& a, const Value& b) {
        if (a.is_bottom() || b.is_top()) return true;
        if (b.is_bottom() || a.is_top()) return false;
        return flat_payload_equals(a.value, b.value);
    }

    static bool equals(const Value& a, const Value& b) {
        if (a.state != b.state) return false;
        return a.state != FlatState::Known || flat_payload_equals(a.value, b.value);
    }

    template <typename Fn>
    static Value lift_unary_known(const Value& a, Fn&& fn) {
        if (a.is_bottom()) return bottom();
        if (a.is_top()) return top();
        return Value::make(fn(a.value));
    }

    template <typename Fn>
    static Value lift_binary_known(const Value& a, const Value& b, Fn&& fn) {
        if (a.is_bottom() || b.is_bottom()) return bottom();
        if (a.is_top() || b.is_top()) return top();
        return Value::make(fn(a.value, b.value));
    }
};

// Standard IntegralDomain<IntT> — Flat Integer Constant Lattice with Full Arithmetic/Bitwise/Convert

template <typename IntT = int64_t>
struct IntegralDomain : public FlatDomain<IntT> {
    static_assert(std::is_integral_v<IntT>, "IntegralDomain requires an integral scalar type");
    using Base = FlatDomain<IntT>;
    using Scalar = IntT;
    using Value = typename Base::Value;

    static std::string format(const Value& v) {
        if (v.is_bottom()) return "<bottom>";
        if (v.is_top()) return "<top>";
        std::ostringstream oss;
        oss << "0x" << std::hex << static_cast<uint64_t>(v.value);
        return oss.str();
    }

    static bool try_as_const_u64(const Value& v, uint64_t* out) {
        if (v.is_known()) {
            if (out) *out = static_cast<uint64_t>(v.value);
            return true;
        }
        return false;
    }

    static Value make_const_u64(uint64_t c) {
        return Value::make(static_cast<IntT>(c));
    }

    static Value add(const Value& a, const Value& b) {
        return Base::lift_binary_known(a, b, [](IntT x, IntT y) {
            return static_cast<IntT>(static_cast<uint64_t>(x) + static_cast<uint64_t>(y));
        });
    }

    static Value sub(const Value& a, const Value& b) {
        return Base::lift_binary_known(a, b, [](IntT x, IntT y) {
            return static_cast<IntT>(static_cast<uint64_t>(x) - static_cast<uint64_t>(y));
        });
    }

    static Value mul(const Value& a, const Value& b) {
        if (a.is_bottom() || b.is_bottom()) return Base::bottom();
        if ((a.is_known() && a.value == 0) || (b.is_known() && b.value == 0)) return Value::make(0);
        return Base::lift_binary_known(a, b, [](IntT x, IntT y) {
            return static_cast<IntT>(static_cast<uint64_t>(x) * static_cast<uint64_t>(y));
        });
    }

    static Value div(const Value& a, const Value& b) {
        if (a.is_bottom() || b.is_bottom()) return Base::bottom();
        if (a.is_known() && b.is_known() && b.value != 0) {
            return Value::make(static_cast<IntT>(a.value / b.value));
        }
        return Base::top();
    }

    static Value iadd(const Value& a, const Value& b, bool /*is64*/ = true) { return add(a, b); }
    static Value isub(const Value& a, const Value& b, bool /*is64*/ = true) { return sub(a, b); }
    static Value imul(const Value& a, const Value& b, bool /*is64*/ = true) { return mul(a, b); }
    static Value idiv(const Value& a, const Value& b, bool /*is64*/ = true) { return div(a, b); }

    static Value bit_and(const Value& a, const Value& b, bool is64 = true) {
        if (a.is_bottom() || b.is_bottom()) return Base::bottom();
        if ((a.is_known() && a.value == 0) || (b.is_known() && b.value == 0)) return Value::make(0);
        const uint64_t all_ones = is64 ? ~0ULL : 0xFFFFFFFFULL;
        if (b.is_known() && (static_cast<uint64_t>(b.value) & all_ones) == all_ones) return a;
        if (a.is_known() && (static_cast<uint64_t>(a.value) & all_ones) == all_ones) return b;
        return Base::lift_binary_known(a, b, [](IntT x, IntT y) {
            return static_cast<IntT>(static_cast<uint64_t>(x) & static_cast<uint64_t>(y));
        });
    }

    static Value bit_or(const Value& a, const Value& b, bool /*is64*/ = true) {
        if (a.is_bottom() || b.is_bottom()) return Base::bottom();
        if (b.is_known() && b.value == 0) return a;
        if (a.is_known() && a.value == 0) return b;
        return Base::lift_binary_known(a, b, [](IntT x, IntT y) {
            return static_cast<IntT>(static_cast<uint64_t>(x) | static_cast<uint64_t>(y));
        });
    }

    static Value bit_xor(const Value& a, const Value& b, bool /*is64*/ = true) {
        if (a.is_bottom() || b.is_bottom()) return Base::bottom();
        if (b.is_known() && b.value == 0) return a;
        if (a.is_known() && a.value == 0) return b;
        return Base::lift_binary_known(a, b, [](IntT x, IntT y) {
            return static_cast<IntT>(static_cast<uint64_t>(x) ^ static_cast<uint64_t>(y));
        });
    }

    static Value bit_not(const Value& a, bool is64 = true) {
        const uint64_t mask = is64 ? ~0ULL : 0xFFFFFFFFULL;
        return Base::lift_unary_known(a, [mask](IntT x) {
            return static_cast<IntT>((~static_cast<uint64_t>(x)) & mask);
        });
    }

    static Value shift(const Value& a, const Value& b, ShiftKind kind, bool is64 = true) {
        if (a.is_bottom() || b.is_bottom()) return Base::bottom();
        if (b.is_known() && b.value == 0) return a;
        return Base::lift_binary_known(a, b, [kind, is64](IntT x, IntT y) {
            uint64_t ca = static_cast<uint64_t>(x);
            uint32_t sh = static_cast<uint32_t>(static_cast<uint64_t>(y) & (is64 ? 63u : 31u));
            uint64_t mask = is64 ? ~0ULL : 0xFFFFFFFFULL;
            uint64_t res = 0;
            switch (kind) {
                case ShiftKind::Shl:  res = (ca << sh) & mask; break;
                case ShiftKind::ShrU: res = (ca & mask) >> sh; break;
                case ShiftKind::ShrS:
                    if (is64) res = static_cast<uint64_t>(static_cast<int64_t>(ca) >> sh);
                    else res = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(ca) >> sh));
                    break;
                case ShiftKind::RotL: {
                    uint32_t w = is64 ? 64u : 32u;
                    res = (sh == 0) ? (ca & mask) : (((ca << sh) | ((ca & mask) >> (w - sh))) & mask);
                    break;
                }
                case ShiftKind::RotR: {
                    uint32_t w = is64 ? 64u : 32u;
                    res = (sh == 0) ? (ca & mask) : ((((ca & mask) >> sh) | (ca << (w - sh))) & mask);
                    break;
                }
            }
            return static_cast<IntT>(res);
        });
    }

    static Value convert(const Value& a, ConvertKind kind) {
        if (a.is_bottom()) return Base::bottom();
        if (a.is_top()) return Base::top();
        switch (kind) {
            case ConvertKind::S32ToS64:
                return Value::make(static_cast<IntT>(static_cast<int32_t>(a.value)));
            case ConvertKind::U32ToU64:
            case ConvertKind::U64ToU32:
                return Value::make(static_cast<IntT>(static_cast<uint32_t>(a.value)));
            case ConvertKind::Bitcast:
                return a;
            default:
                return Base::top();
        }
    }
};

// Standard FloatDomain<FloatT> — Flat Floating-Point Constant Lattice

template <typename FloatT = double>
struct FloatDomain : public FlatDomain<FloatT> {
    static_assert(std::is_floating_point_v<FloatT>, "FloatDomain requires a floating-point scalar type");
    using Base = FlatDomain<FloatT>;
    using Scalar = FloatT;
    using Value = typename Base::Value;

    static std::string format(const Value& v) {
        if (v.is_bottom()) return "<bottom>";
        if (v.is_top()) return "<top>";
        std::ostringstream oss;
        oss << v.value;
        return oss.str();
    }

    static Value add(const Value& a, const Value& b) {
        return Base::lift_binary_known(a, b, [](FloatT x, FloatT y) { return x + y; });
    }

    static Value sub(const Value& a, const Value& b) {
        return Base::lift_binary_known(a, b, [](FloatT x, FloatT y) { return x - y; });
    }

    static Value mul(const Value& a, const Value& b) {
        return Base::lift_binary_known(a, b, [](FloatT x, FloatT y) { return x * y; });
    }

    static Value div(const Value& a, const Value& b) {
        if (a.is_bottom() || b.is_bottom()) return Base::bottom();
        if (a.is_known() && b.is_known() && b.value != FloatT(0)) {
            return Value::make(a.value / b.value);
        }
        return Base::top();
    }

    static Value convert(const Value& a, ConvertKind kind) {
        if (a.is_bottom()) return Base::bottom();
        if (a.is_top()) return Base::top();
        if (kind == ConvertKind::Bitcast) return a;
        return Base::top();
    }
};


}  // namespace valhall::dataflow
