#pragma once

#include "valhall_ssa.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace valhall::dataflow {

/// Generic abstract state parameterized by a lattice domain `DomainT`.
///
/// `DomainT` must provide:
///   using Value = ...;
///   static Value bottom();
///   static Value top();
///   static Value join(const Value& a, const Value& b);
///   static Value meet(const Value& a, const Value& b);
///   static bool  leq(const Value& a, const Value& b);
///   static bool  equals(const Value& a, const Value& b);
///
/// Optionally, `DomainT` may provide:
///   static bool try_as_const_u64(const Value& v, uint64_t* out);
///   static Value make_const_u64(uint64_t v);
///
/// In SSA form, every `SsaId` has a single static definition, so the register
/// file is represented by an indexed environment `ssa_vals[SsaId] -> Value`.
/// Mutable side-channel state (such as Pilot FAU uniform memory written via
/// `PsbStore` and symbolic memory cells) is tracked in `fau_mem` and `sym_mem`.
template <typename DomainT>
struct AbstractState {
    using Value = typename DomainT::Value;

    /// Abstract value of every SSA value (1-indexed; index 0 is kInvalidSsa).
    std::vector<Value> ssa_vals;

    /// Abstract FAU uniform memory (byte offset -> abstract value).
    std::map<uint32_t, Value> fau_mem;

    /// Abstract symbolic memory locations (e.g. "SRT[4]+0x00" -> abstract value).
    std::map<std::string, Value> sym_mem;

    explicit AbstractState(size_t num_ssa_nodes = 1)
        : ssa_vals(num_ssa_nodes > 0 ? num_ssa_nodes : 1, DomainT::bottom()) {}

    void ensure_size(size_t num_ssa_nodes) {
        if (ssa_vals.size() < num_ssa_nodes) {
            ssa_vals.resize(num_ssa_nodes, DomainT::bottom());
        }
    }

    Value val(compiler::SsaId id) const {
        if (id == compiler::kInvalidSsa || id >= ssa_vals.size()) {
            return DomainT::bottom();
        }
        return ssa_vals[id];
    }

    void set_val(compiler::SsaId id, const Value& v) {
        if (id == compiler::kInvalidSsa) return;
        if (id >= ssa_vals.size()) {
            ssa_vals.resize(static_cast<size_t>(id) + 1, DomainT::bottom());
        }
        ssa_vals[id] = v;
    }

    /// Monotone update: `ssa_vals[id] = join(ssa_vals[id], v)`. Returns true if changed.
    bool join_val(compiler::SsaId id, const Value& v) {
        if (id == compiler::kInvalidSsa) return false;
        if (id >= ssa_vals.size()) {
            ssa_vals.resize(static_cast<size_t>(id) + 1, DomainT::bottom());
        }
        Value joined = DomainT::join(ssa_vals[id], v);
        if (!DomainT::equals(joined, ssa_vals[id])) {
            ssa_vals[id] = std::move(joined);
            return true;
        }
        return false;
    }

    Value fau(uint32_t byte_off) const {
        auto it = fau_mem.find(byte_off);
        return (it != fau_mem.end()) ? it->second : DomainT::bottom();
    }

    void set_fau(uint32_t byte_off, const Value& v) {
        fau_mem[byte_off] = v;
    }

    bool join_fau(uint32_t byte_off, const Value& v) {
        auto it = fau_mem.find(byte_off);
        if (it == fau_mem.end()) {
            if (!DomainT::equals(v, DomainT::bottom())) {
                fau_mem[byte_off] = v;
                return true;
            }
            return false;
        }
        Value joined = DomainT::join(it->second, v);
        if (!DomainT::equals(joined, it->second)) {
            it->second = std::move(joined);
            return true;
        }
        return false;
    }

    Value sym(const std::string& key) const {
        auto it = sym_mem.find(key);
        return (it != sym_mem.end()) ? it->second : DomainT::bottom();
    }

    void set_sym(const std::string& key, const Value& v) {
        sym_mem[key] = v;
    }

    /// Pointwise lattice join: `this = this ⊔ other`. Returns true if state changed.
    bool join_with(const AbstractState& other) {
        bool changed = false;
        if (other.ssa_vals.size() > ssa_vals.size()) {
            ssa_vals.resize(other.ssa_vals.size(), DomainT::bottom());
        }
        for (size_t i = 1; i < other.ssa_vals.size(); ++i) {
            Value joined = DomainT::join(ssa_vals[i], other.ssa_vals[i]);
            if (!DomainT::equals(joined, ssa_vals[i])) {
                ssa_vals[i] = std::move(joined);
                changed = true;
            }
        }
        for (const auto& kv : other.fau_mem) {
            if (join_fau(kv.first, kv.second)) changed = true;
        }
        for (const auto& kv : other.sym_mem) {
            auto it = sym_mem.find(kv.first);
            if (it == sym_mem.end()) {
                if (!DomainT::equals(kv.second, DomainT::bottom())) {
                    sym_mem[kv.first] = kv.second;
                    changed = true;
                }
            } else {
                Value joined = DomainT::join(it->second, kv.second);
                if (!DomainT::equals(joined, it->second)) {
                    it->second = std::move(joined);
                    changed = true;
                }
            }
        }
        return changed;
    }

    /// Pointwise lattice meet: `this = this ⊓ other`. Returns true if state changed.
    bool meet_with(const AbstractState& other) {
        bool changed = false;
        size_t n = std::min(ssa_vals.size(), other.ssa_vals.size());
        for (size_t i = 1; i < n; ++i) {
            Value met = DomainT::meet(ssa_vals[i], other.ssa_vals[i]);
            if (!DomainT::equals(met, ssa_vals[i])) {
                ssa_vals[i] = std::move(met);
                changed = true;
            }
        }
        for (size_t i = n; i < ssa_vals.size(); ++i) {
            if (!DomainT::equals(ssa_vals[i], DomainT::bottom())) {
                ssa_vals[i] = DomainT::bottom();
                changed = true;
            }
        }
        for (auto it = fau_mem.begin(); it != fau_mem.end(); ) {
            auto oit = other.fau_mem.find(it->first);
            if (oit == other.fau_mem.end()) {
                it = fau_mem.erase(it);
                changed = true;
            } else {
                Value met = DomainT::meet(it->second, oit->second);
                if (!DomainT::equals(met, it->second)) {
                    it->second = std::move(met);
                    changed = true;
                }
                ++it;
            }
        }
        for (auto it = sym_mem.begin(); it != sym_mem.end(); ) {
            auto oit = other.sym_mem.find(it->first);
            if (oit == other.sym_mem.end()) {
                it = sym_mem.erase(it);
                changed = true;
            } else {
                Value met = DomainT::meet(it->second, oit->second);
                if (!DomainT::equals(met, it->second)) {
                    it->second = std::move(met);
                    changed = true;
                }
                ++it;
            }
        }
        return changed;
    }

    bool equals(const AbstractState& other) const {
        size_t max_n = std::max(ssa_vals.size(), other.ssa_vals.size());
        for (size_t i = 1; i < max_n; ++i) {
            Value a = (i < ssa_vals.size()) ? ssa_vals[i] : DomainT::bottom();
            Value b = (i < other.ssa_vals.size()) ? other.ssa_vals[i] : DomainT::bottom();
            if (!DomainT::equals(a, b)) return false;
        }
        if (fau_mem.size() != other.fau_mem.size()) return false;
        for (const auto& kv : fau_mem) {
            auto oit = other.fau_mem.find(kv.first);
            if (oit == other.fau_mem.end() || !DomainT::equals(kv.second, oit->second)) return false;
        }
        if (sym_mem.size() != other.sym_mem.size()) return false;
        for (const auto& kv : sym_mem) {
            auto oit = other.sym_mem.find(kv.first);
            if (oit == other.sym_mem.end() || !DomainT::equals(kv.second, oit->second)) return false;
        }
        return true;
    }
};

}  // namespace valhall::dataflow
