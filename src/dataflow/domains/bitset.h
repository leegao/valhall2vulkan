#pragma once

#include "traits.h"
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace valhall::dataflow {

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


}  // namespace valhall::dataflow
