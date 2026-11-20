#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

namespace valhall::dataflow {

/// Generic abstract machine state over a lattice domain `DomainT`:
///   - `DomainT::Value` : abstract value type
///   - `DomainT::bottom()`, `DomainT::top()`
///   - `DomainT::join(a, b)`, `DomainT::meet(a, b)`
///   - `DomainT::leq(a, b)`, `DomainT::equals(a, b)`
///
/// Encapsulates Valhall's 64x32-bit GPR file (`r0..r63`), ISA live-range kill
/// markers (from `^` discard-last-use annotations), FAU uniform memory, and
/// symbolic buffer/scratch memory.
template <typename DomainT>
struct AbstractState {
    using Value = typename DomainT::Value;

    std::array<Value, 64> regs{};
    std::array<bool, 64> reg_killed{};
    std::map<uint32_t, Value> fau_mem;                         // FAU byte offset -> abstract value
    std::map<std::pair<std::string, int64_t>, Value> sym_mem;  // (base_symbol, byte_offset) -> abstract value

    AbstractState() {
        regs.fill(DomainT::bottom());
        reg_killed.fill(false);
    }

    static AbstractState bottom() {
        return AbstractState{};
    }

    const Value& get_reg(uint8_t r) const {
        return regs[r & 63u];
    }

    void set_reg(uint8_t r, const Value& v) {
        uint8_t idx = r & 63u;
        regs[idx] = v;
        reg_killed[idx] = false;
    }

    void set_reg_pair(uint8_t base, const Value& v) {
        set_reg(base, v);
        if (base + 1u < 64u) set_reg(static_cast<uint8_t>(base + 1u), v);
    }

    void set_reg_range(uint8_t base, uint8_t count, const Value& v) {
        for (uint8_t i = 0; i < count; ++i) {
            if (base + i < 64u) set_reg(static_cast<uint8_t>(base + i), v);
        }
    }

    void kill_reg(uint8_t r) {
        uint8_t idx = r & 63u;
        reg_killed[idx] = true;
    }

    bool is_reg_killed(uint8_t r) const {
        return reg_killed[r & 63u];
    }

    Value get_fau(uint32_t byte_off) const {
        auto it = fau_mem.find(byte_off);
        return (it != fau_mem.end()) ? it->second : DomainT::bottom();
    }

    void set_fau(uint32_t byte_off, const Value& v) {
        fau_mem[byte_off] = v;
    }

    Value get_sym_mem(const std::string& base, int64_t off) const {
        auto it = sym_mem.find({base, off});
        return (it != sym_mem.end()) ? it->second : DomainT::bottom();
    }

    void set_sym_mem(const std::string& base, int64_t off, const Value& v) {
        sym_mem[{base, off}] = v;
    }

    /// Pointwise lattice join (`this = this ⊔ other`). Returns true if state changed.
    bool join_with(const AbstractState& other) {
        bool changed = false;
        for (size_t i = 0; i < 64; ++i) {
            Value joined = DomainT::join(regs[i], other.regs[i]);
            if (!DomainT::equals(joined, regs[i])) {
                regs[i] = std::move(joined);
                changed = true;
            }
            bool k = reg_killed[i] && other.reg_killed[i];
            if (k != reg_killed[i]) {
                reg_killed[i] = k;
                changed = true;
            }
        }
        for (const auto& kv : other.fau_mem) {
            auto it = fau_mem.find(kv.first);
            if (it == fau_mem.end()) {
                fau_mem[kv.first] = kv.second;
                changed = true;
            } else {
                Value joined = DomainT::join(it->second, kv.second);
                if (!DomainT::equals(joined, it->second)) {
                    it->second = std::move(joined);
                    changed = true;
                }
            }
        }
        for (const auto& kv : other.sym_mem) {
            auto it = sym_mem.find(kv.first);
            if (it == sym_mem.end()) {
                sym_mem[kv.first] = kv.second;
                changed = true;
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

    /// Pointwise lattice meet (`this = this ⊓ other`).
    void meet_with(const AbstractState& other) {
        for (size_t i = 0; i < 64; ++i) {
            regs[i] = DomainT::meet(regs[i], other.regs[i]);
            reg_killed[i] = reg_killed[i] || other.reg_killed[i];
        }
        for (auto it = fau_mem.begin(); it != fau_mem.end(); ) {
            auto oit = other.fau_mem.find(it->first);
            if (oit == other.fau_mem.end()) {
                it = fau_mem.erase(it);
            } else {
                it->second = DomainT::meet(it->second, oit->second);
                ++it;
            }
        }
        for (auto it = sym_mem.begin(); it != sym_mem.end(); ) {
            auto oit = other.sym_mem.find(it->first);
            if (oit == other.sym_mem.end()) {
                it = sym_mem.erase(it);
            } else {
                it->second = DomainT::meet(it->second, oit->second);
                ++it;
            }
        }
    }

    bool equals(const AbstractState& other) const {
        for (size_t i = 0; i < 64; ++i) {
            if (!DomainT::equals(regs[i], other.regs[i])) return false;
            if (reg_killed[i] != other.reg_killed[i]) return false;
        }
        if (fau_mem.size() != other.fau_mem.size()) return false;
        for (const auto& kv : fau_mem) {
            auto it = other.fau_mem.find(kv.first);
            if (it == other.fau_mem.end() || !DomainT::equals(kv.second, it->second)) return false;
        }
        if (sym_mem.size() != other.sym_mem.size()) return false;
        for (const auto& kv : sym_mem) {
            auto it = other.sym_mem.find(kv.first);
            if (it == other.sym_mem.end() || !DomainT::equals(kv.second, it->second)) return false;
        }
        return true;
    }
};

}  // namespace valhall::dataflow
