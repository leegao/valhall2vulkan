#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace valhall::dataflow {

// ============================================================================
// 1. Compile-Time Lattice Interface Enforcement & Auto-Equals
// ============================================================================

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

}  // namespace detail

/// Enforces the Lattice Domain contract at the point of instantiation with
/// human-readable compile-time error messages.
///
/// Required interface on `DomainT`:
///   using Value = ...;
///   static Value bottom();
///   static Value top();
///   static Value join(const Value& a, const Value& b);
///   static Value meet(const Value& a, const Value& b);
///   static bool  leq(const Value& a, const Value& b);
///
/// Optional interface on `DomainT`:
///   static bool  equals(const Value& a, const Value& b);
///     (Auto-implemented via `leq(a, b) && leq(b, a)` if omitted.)
///   static bool  try_as_const_u64(const Value& v, uint64_t* out);
///   static Value make_const_u64(uint64_t c);
///     (Enables built-in constant folding & algebraic extensional equalities
///      such as x+0 == x, x&~0 == x, x^x == 0 in `SsaDomainTransferVisitor`.)
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
/// Calls `DomainT::equals(a, b)` when provided; otherwise defaults to
/// antisymmetry: `DomainT::leq(a, b) && DomainT::leq(b, a)`.
template <typename DomainT>
inline bool domain_equals(const typename DomainT::Value& a, const typename DomainT::Value& b) {
    static_assert(ValidateLatticeDomain<DomainT>::valid, "");
    if constexpr (detail::has_equals<DomainT>::value) {
        return DomainT::equals(a, b);
    } else {
        return DomainT::leq(a, b) && DomainT::leq(b, a);
    }
}

/// Round `n` up to the next power of two (minimum 1) to guarantee amortized O(1)
/// growth and prevent adversarial O(N^2) reallocations in dense vector domains.
inline size_t next_power_of_two(size_t n) {
    if (n <= 1) return 1;
    size_t p = 1;
    while (p < n && p <= (static_cast<size_t>(-1) >> 1)) {
        p <<= 1;
    }
    return (p >= n) ? p : n;
}

// ============================================================================
// 2. VectorDomain<ElemDomainT> — Pointwise Lattice over Dense Indexed Arrays
// ============================================================================

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
struct VectorDomain {
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

    static Value join(const Value& a, const Value& b) {
        Value out = a;
        join_into(out, b);
        return out;
    }

    static Value meet(const Value& a, const Value& b) {
        Value out = a;
        meet_into(out, b);
        return out;
    }

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
};

// ============================================================================
// 3. MapDomain<KeyT, ValDomainT> — Pointwise Lattice over Sparse Maps
// ============================================================================

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
struct MapDomain {
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

    static Value join(const Value& a, const Value& b) {
        Value out = a;
        join_into(out, b);
        return out;
    }

    static Value meet(const Value& a, const Value& b) {
        Value out = a;
        meet_into(out, b);
        return out;
    }

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
};

// ============================================================================
// 4. TupleDomain<Domains...> — Variadic Pointwise Product Lattice
// ============================================================================

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
};

// ============================================================================
// 5. BitsetDomain — Compact Powerset Lattice for Liveness / Reaching Defs
// ============================================================================

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

    /// In-place set union: `this |= other`.
    void or_with(const BitsetValue& other) {
        if (is_top_val) return;
        if (other.is_top_val) { is_top_val = true; return; }
        if (other.words.size() > words.size()) words.resize(other.words.size(), 0ULL);
        for (size_t i = 0; i < other.words.size(); ++i) words[i] |= other.words[i];
    }

    /// In-place set union with difference: `this |= (other \ minus)`.
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

// ============================================================================
// 6. FiniteSetDomain<ElemDomainT, MaxK> — Bounded Finite-Set Powerset Lattice
// ============================================================================
// Tracks up to `MaxK` (default 4) distinct non-bottom, non-top elements before
// widening to `Top`. Provides `lift_value`, `try_as_singleton`, `lift_unary`,
// and `lift_binary` so pointwise transfer functions remain concise and readable.

template <typename ElemDomainT, size_t MaxK = 4>
struct FiniteSetValue {
    using ElemValue = typename ElemDomainT::Value;
    static_assert(ValidateLatticeDomain<ElemDomainT>::valid, "");
    static_assert(MaxK >= 1, "FiniteSetDomain MaxK must be >= 1");

    bool is_top_val = false;
    std::vector<ElemValue> items;  // Always deduplicated, contains neither Bottom nor Top

    bool is_bottom() const { return !is_top_val && items.empty(); }
    bool is_top() const { return is_top_val; }
    bool is_singleton() const { return !is_top_val && items.size() == 1; }
    size_t size() const { return items.size(); }

    /// Free lower: extracts the unique element when this set is a singleton.
    bool try_as_singleton(ElemValue* out) const {
        if (!is_singleton()) return false;
        if (out) *out = items[0];
        return true;
    }

    /// Insert an element into the bounded set (widening to Top if `e` is Top or
    /// cardinality exceeds `MaxK`).
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
struct FiniteSetDomain {
    static_assert(ValidateLatticeDomain<ElemDomainT>::valid, "");
    using ElemValue = typename ElemDomainT::Value;
    using Value = FiniteSetValue<ElemDomainT, MaxK>;

    static Value bottom() { return Value{}; }
    static Value top() { Value v; v.is_top_val = true; return v; }

    /// Free lift: promote a single `ElemValue` into a singleton `FiniteSetValue`.
    static Value singleton(const ElemValue& e) {
        Value v;
        v.insert(e);
        return v;
    }
    static Value lift_value(const ElemValue& e) { return singleton(e); }

    /// Lift a pointwise unary transfer function `fn(const ElemValue&) -> ElemValue`
    /// (or `-> Value`) across all elements in `s`.
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

    /// Lift a pointwise binary transfer function `fn(const ElemValue&, const ElemValue&) -> ElemValue`
    /// (or `-> Value`) across the Cartesian product `a × b`.
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
                if (domain_equals<ElemDomainT>(ea, eb)) {
                    out.insert(ea);
                    break;
                }
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
                if (ElemDomainT::leq(ea, eb)) {
                    covered = true;
                    break;
                }
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

    // Forward constant-folding helpers when `ElemDomainT` supports them
    static bool try_as_const_u64(const Value& v, uint64_t* out) {
        if constexpr (detail::has_const_helpers<ElemDomainT>::value) {
            ElemValue single;
            if (v.try_as_singleton(&single)) {
                return ElemDomainT::try_as_const_u64(single, out);
            }
        }
        (void)v;
        (void)out;
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
