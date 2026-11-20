#pragma once

#include "domains.h"
#include "memory_domain.h"
#include "valhall_ssa.h"

#include <cstdint>

namespace valhall::dataflow {

/// Generic SSA abstract state parameterized by a lattice domain `DomainT`.
///
/// Composes three prepared domain combinators from `domains.h`:
///   1. `ssa_vals` (`VectorDomain<DomainT>`):
///      Maps each `SsaId` (1-indexed) to its abstract value. Because SSA
///      guarantees a single static definition per `SsaId`, every path joining
///      at a merge block carries consistent values for all dominating `SsaId`s,
///      and divergent definitions are merged explicitly via `Phi` nodes.
///   2. `fau_mem` (`MapDomain<uint32_t, DomainT>`):
///      Models the FAU_0 uniform bank (byte offset -> abstract value). In Main
///      shaders, FAU_0 is **read-only (RO)**—populated once before dispatch by
///      the Pilot shader and driver RLOC/FCST tables—which makes locating and
///      analyzing FAU uniform loads flow-insensitive in Main shaders. In Pilot
///      shaders, `PsbStore` to `FauDstBase + off` writes into `fau_mem`.
///   3. `sym_mem` (`MapDomain<SymbolicMemoryLoc, DomainT>`):
///      Models disjoint symbolic memory regions (`SrtTable`, `Buffer`, `Tls`,
///      `Wls`, `DataSection`) keyed by `SymbolicMemoryLoc`.
template <typename DomainT>
struct AbstractState {
    static_assert(ValidateLatticeDomain<DomainT>::valid, "");
    using Value = typename DomainT::Value;
    using SsaEnvDomain = VectorDomain<DomainT>;
    using FauMemDomain = MapDomain<uint32_t, DomainT>;
    using SymMemDomain = MapDomain<SymbolicMemoryLoc, DomainT>;

    typename SsaEnvDomain::Value ssa_vals;
    typename FauMemDomain::Value fau_mem;
    typename SymMemDomain::Value sym_mem;

    explicit AbstractState(size_t num_ssa_nodes = 1)
        : ssa_vals(num_ssa_nodes > 0 ? num_ssa_nodes : 1) {}

    void ensure_size(size_t num_ssa_nodes) {
        ssa_vals.ensure_size(num_ssa_nodes);
    }

    Value val(compiler::SsaId id) const {
        if (id == compiler::kInvalidSsa) return DomainT::bottom();
        return ssa_vals.get(id);
    }

    void set_val(compiler::SsaId id, const Value& v) {
        if (id == compiler::kInvalidSsa) return;
        ssa_vals.set(id, v);
    }

    /// Monotone update: `ssa_vals[id] = join(ssa_vals[id], v)`. Returns true if changed.
    bool join_val(compiler::SsaId id, const Value& v) {
        if (id == compiler::kInvalidSsa) return false;
        return ssa_vals.join_at(id, v);
    }

    Value fau(uint32_t byte_off) const { return fau_mem.get(byte_off); }
    void set_fau(uint32_t byte_off, const Value& v) { fau_mem.set(byte_off, v); }
    bool join_fau(uint32_t byte_off, const Value& v) { return fau_mem.join_at(byte_off, v); }

    Value sym(const SymbolicMemoryLoc& loc) const { return sym_mem.get(loc); }
    void set_sym(const SymbolicMemoryLoc& loc, const Value& v) { sym_mem.set(loc, v); }
    bool join_sym(const SymbolicMemoryLoc& loc, const Value& v) { return sym_mem.join_at(loc, v); }

    /// Pointwise lattice join: `this = this ⊔ other`. Returns true if state changed.
    bool join_with(const AbstractState& other) {
        bool c1 = SsaEnvDomain::join_into(ssa_vals, other.ssa_vals);
        bool c2 = FauMemDomain::join_into(fau_mem, other.fau_mem);
        bool c3 = SymMemDomain::join_into(sym_mem, other.sym_mem);
        return c1 || c2 || c3;
    }

    /// Pointwise lattice meet: `this = this ⊓ other`. Returns true if state changed.
    bool meet_with(const AbstractState& other) {
        bool c1 = SsaEnvDomain::meet_into(ssa_vals, other.ssa_vals);
        bool c2 = FauMemDomain::meet_into(fau_mem, other.fau_mem);
        bool c3 = SymMemDomain::meet_into(sym_mem, other.sym_mem);
        return c1 || c2 || c3;
    }

    bool equals(const AbstractState& other) const {
        return SsaEnvDomain::equals(ssa_vals, other.ssa_vals) &&
               FauMemDomain::equals(fau_mem, other.fau_mem) &&
               SymMemDomain::equals(sym_mem, other.sym_mem);
    }
};

}  // namespace valhall::dataflow
