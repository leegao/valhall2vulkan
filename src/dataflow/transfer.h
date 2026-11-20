#pragma once

#include "abstract_domain.h"
#include "liveness.h"
#include "valhall_ssa.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <type_traits>
#include <vector>

namespace valhall::dataflow {

/// Shift / rotate direction for normalized `on_shift` dispatch.
enum class ShiftKind : uint8_t {
    Shl = 0,
    ShrU,
    ShrS,
    RotL,
    RotR,
};

/// Abstract SSA transfer-function visitor base class.
///
/// Groups the 134 `tables::SsaOp` enum values into normalized semantic transfer
/// hooks (e.g. `BitAnd` + `BitAnd64` -> `on_and(dst, lhs, rhs, is_64bit, node)`,
/// `IAdd` + `IAdd64` -> `on_iadd(dst, lhs, rhs, is_64bit, node)`, `Copy` +
/// `Bitcast` -> `on_copy(dst, src, node)`). Every hook receives the full
/// `const compiler::SsaNode&` so specialized analyses can still inspect
/// op-specific metadata (`imm_u32`, `imm_u64`, `aux_index`, `aux_str`, `type`).
class SsaTransferVisitorBase {
public:
    virtual ~SsaTransferVisitorBase() = default;

    // Constants & moves / aliases
    virtual void on_const(compiler::SsaId dst, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_copy(compiler::SsaId dst, compiler::SsaId src, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_phi(compiler::SsaId dst, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_select(compiler::SsaId dst, compiler::SsaId cond,
                           compiler::SsaId true_val, compiler::SsaId false_val,
                           const compiler::SsaNode& node) { on_other(dst, node); }

    // Integer arithmetic
    virtual void on_iadd(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                         bool is_64bit, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_isub(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                         bool is_64bit, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_imul(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                         bool is_64bit, const compiler::SsaNode& node) { on_other(dst, node); }

    // Bitwise logic & shifts
    virtual void on_and(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                        bool is_64bit, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_or(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                       bool is_64bit, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_xor(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                        bool is_64bit, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_not(compiler::SsaId dst, compiler::SsaId src,
                        bool is_64bit, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_shift(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                          ShiftKind kind, bool is_64bit, const compiler::SsaNode& node) { on_other(dst, node); }

    // 64-bit packing / splitting / tuples / conversions / vectors
    virtual void on_pack64(compiler::SsaId dst, compiler::SsaId lo, compiler::SsaId hi,
                           const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_split64(compiler::SsaId dst, compiler::SsaId src64,
                            const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_tuple_extract(compiler::SsaId dst, compiler::SsaId tuple_val, uint32_t elem_idx,
                                  const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_convert(compiler::SsaId dst, compiler::SsaId src,
                            const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_vec(compiler::SsaId dst, const compiler::SsaNode& node) { on_other(dst, node); }

    // Shader inputs, FAU uniforms, sideband, memory, textures, outputs, atomics
    virtual void on_load_fau(compiler::SsaId dst, uint32_t fau_word_idx,
                             const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_load_builtin(compiler::SsaId dst, uint32_t builtin_id,
                                 const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_load_sideband(compiler::SsaId dst, uint32_t member_idx, bool is_64bit,
                                  const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_psb_load(compiler::SsaId dst, compiler::SsaId addr, uint32_t elem_bytes,
                             const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_psb_store(compiler::SsaId addr, compiler::SsaId val, uint32_t elem_bytes,
                              const compiler::SsaNode& node) { on_other(node.id, node); }
    virtual void on_load_vert_attr(compiler::SsaId dst, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_load_frag_varying(compiler::SsaId dst, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_tex_sample(compiler::SsaId dst, const compiler::SsaNode& node) { on_other(dst, node); }
    virtual void on_store_output(const compiler::SsaNode& node) { on_other(node.id, node); }
    virtual void on_atomic(compiler::SsaId dst, const compiler::SsaNode& node) { on_other(dst, node); }

    // Fallback for all other ops (FP math, comparisons, barriers, branches, Nop, etc.)
    virtual void on_other(compiler::SsaId /*dst*/, const compiler::SsaNode& /*node*/) {}

    // Optional liveness hook: fired after an SSA node when liveness reports `killed_id` is last-used
    virtual void on_ssa_killed(compiler::SsaId /*killed_id*/, const compiler::SsaNode& /*at_node*/) {}
};

/// Normalized dispatcher from any `compiler::SsaNode` into `SsaTransferVisitorBase`.
void dispatch_ssa_transfer(const compiler::SsaNode& node, SsaTransferVisitorBase& visitor);

namespace detail {

template <typename T, typename = void>
struct has_const_helpers : std::false_type {};

template <typename T>
struct has_const_helpers<
    T,
    std::void_t<
        decltype(T::try_as_const_u64(std::declval<const typename T::Value&>(), std::declval<uint64_t*>())),
        decltype(T::make_const_u64(std::declval<uint64_t>()))>> : std::true_type {};

}  // namespace detail

/// Domain-aware SSA transfer visitor providing built-in **extensional equalities**:
///
/// 1. `Copy` / `Bitcast`: `dst = state.val(src)` (default state copy/alias).
/// 2. `Phi`: `dst = ⊔ { state.val(v) | (_, v) ∈ phi_incomings }`.
/// 3. `Select(c, a, b)`: if `a == b`, exact copy; otherwise `val(a) ⊔ val(b)`.
/// 4. Algebraic identities (when `DomainT` exposes `try_as_const_u64`/`make_const_u64`):
///      - `x + 0 == 0 + x == x`
///      - `x - 0 == x`, `x - x == 0`
///      - `x * 1 == 1 * x == x`, `x * 0 == 0 * x == 0`
///      - `x | 0 == 0 | x == x`, `x | x == x`
///      - `x ^ 0 == 0 ^ x == x`, `x ^ x == 0`
///      - `x & ~0 == ~0 & x == x`, `x & x == x`, `x & 0 == 0 & x == 0`
///      - `x << 0 == x >> 0 == x`
///
/// Subclasses override whichever semantic hooks they want to refine and can fall
/// back to `SsaDomainTransferVisitor<DomainT>::on_*` to inherit these equalities.
template <typename DomainT>
class SsaDomainTransferVisitor : public SsaTransferVisitorBase {
public:
    using Value = typename DomainT::Value;
    using State = AbstractState<DomainT>;

    explicit SsaDomainTransferVisitor(State* st = nullptr) : state_(st) {}

    void set_state(State* st) { state_ = st; }
    State* state() const { return state_; }

    Value eval(compiler::SsaId id) const {
        return state_ ? state_->val(id) : DomainT::bottom();
    }

    void assign(compiler::SsaId dst, const Value& v) {
        if (state_ && dst != compiler::kInvalidSsa) {
            state_->set_val(dst, v);
        }
    }

    static bool try_const(const Value& v, uint64_t* out) {
        if constexpr (detail::has_const_helpers<DomainT>::value) {
            return DomainT::try_as_const_u64(v, out);
        }
        (void)v;
        (void)out;
        return false;
    }

    static Value make_const(uint64_t c) {
        if constexpr (detail::has_const_helpers<DomainT>::value) {
            return DomainT::make_const_u64(c);
        }
        (void)c;
        return DomainT::top();
    }

    // --- Built-in extensional equalities ---

    void on_copy(compiler::SsaId dst, compiler::SsaId src, const compiler::SsaNode&) override {
        assign(dst, eval(src));
    }

    void on_phi(compiler::SsaId dst, const compiler::SsaNode& node) override {
        Value acc = DomainT::bottom();
        for (const auto& inc : node.phi_incomings) {
            if (inc.value == dst) continue; // ignore self-loop backedge
            acc = DomainT::join(acc, eval(inc.value));
        }
        assign(dst, acc);
    }

    void on_select(compiler::SsaId dst, compiler::SsaId /*cond*/,
                   compiler::SsaId true_val, compiler::SsaId false_val,
                   const compiler::SsaNode&) override {
        if (true_val == false_val) {
            assign(dst, eval(true_val));
        } else {
            assign(dst, DomainT::join(eval(true_val), eval(false_val)));
        }
    }

    void on_iadd(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool /*is_64bit*/, const compiler::SsaNode&) override {
        Value a = eval(lhs), b = eval(rhs);
        uint64_t ca = 0, cb = 0;
        bool ha = try_const(a, &ca), hb = try_const(b, &cb);
        if (ha && hb) { assign(dst, make_const(ca + cb)); return; }
        if (hb && cb == 0) { assign(dst, a); return; }
        if (ha && ca == 0) { assign(dst, b); return; }
        assign(dst, DomainT::top());
    }

    void on_isub(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool /*is_64bit*/, const compiler::SsaNode&) override {
        if (lhs != compiler::kInvalidSsa && lhs == rhs) {
            assign(dst, make_const(0));
            return;
        }
        Value a = eval(lhs), b = eval(rhs);
        uint64_t ca = 0, cb = 0;
        bool ha = try_const(a, &ca), hb = try_const(b, &cb);
        if (ha && hb) { assign(dst, make_const(ca - cb)); return; }
        if (hb && cb == 0) { assign(dst, a); return; }
        assign(dst, DomainT::top());
    }

    void on_imul(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool /*is_64bit*/, const compiler::SsaNode&) override {
        Value a = eval(lhs), b = eval(rhs);
        uint64_t ca = 0, cb = 0;
        bool ha = try_const(a, &ca), hb = try_const(b, &cb);
        if (ha && hb) { assign(dst, make_const(ca * cb)); return; }
        if ((ha && ca == 0) || (hb && cb == 0)) { assign(dst, make_const(0)); return; }
        if (hb && cb == 1) { assign(dst, a); return; }
        if (ha && ca == 1) { assign(dst, b); return; }
        assign(dst, DomainT::top());
    }

    void on_and(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                bool is_64bit, const compiler::SsaNode&) override {
        if (lhs != compiler::kInvalidSsa && lhs == rhs) {
            assign(dst, eval(lhs));
            return;
        }
        Value a = eval(lhs), b = eval(rhs);
        uint64_t ca = 0, cb = 0;
        bool ha = try_const(a, &ca), hb = try_const(b, &cb);
        const uint64_t all_ones = is_64bit ? ~0ULL : 0xFFFFFFFFULL;
        if (ha && hb) { assign(dst, make_const(ca & cb)); return; }
        if ((ha && ca == 0) || (hb && cb == 0)) { assign(dst, make_const(0)); return; }
        if (hb && (cb & all_ones) == all_ones) { assign(dst, a); return; }
        if (ha && (ca & all_ones) == all_ones) { assign(dst, b); return; }
        assign(dst, DomainT::top());
    }

    void on_or(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
               bool /*is_64bit*/, const compiler::SsaNode&) override {
        if (lhs != compiler::kInvalidSsa && lhs == rhs) {
            assign(dst, eval(lhs));
            return;
        }
        Value a = eval(lhs), b = eval(rhs);
        uint64_t ca = 0, cb = 0;
        bool ha = try_const(a, &ca), hb = try_const(b, &cb);
        if (ha && hb) { assign(dst, make_const(ca | cb)); return; }
        if (hb && cb == 0) { assign(dst, a); return; }
        if (ha && ca == 0) { assign(dst, b); return; }
        assign(dst, DomainT::top());
    }

    void on_xor(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                bool /*is_64bit*/, const compiler::SsaNode&) override {
        if (lhs != compiler::kInvalidSsa && lhs == rhs) {
            assign(dst, make_const(0));
            return;
        }
        Value a = eval(lhs), b = eval(rhs);
        uint64_t ca = 0, cb = 0;
        bool ha = try_const(a, &ca), hb = try_const(b, &cb);
        if (ha && hb) { assign(dst, make_const(ca ^ cb)); return; }
        if (hb && cb == 0) { assign(dst, a); return; }
        if (ha && ca == 0) { assign(dst, b); return; }
        assign(dst, DomainT::top());
    }

    void on_not(compiler::SsaId dst, compiler::SsaId src,
                bool is_64bit, const compiler::SsaNode&) override {
        uint64_t c = 0;
        if (try_const(eval(src), &c)) {
            uint64_t mask = is_64bit ? ~0ULL : 0xFFFFFFFFULL;
            assign(dst, make_const((~c) & mask));
            return;
        }
        assign(dst, DomainT::top());
    }

    void on_shift(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                  ShiftKind kind, bool is_64bit, const compiler::SsaNode&) override {
        Value a = eval(lhs), b = eval(rhs);
        uint64_t ca = 0, cb = 0;
        bool ha = try_const(a, &ca), hb = try_const(b, &cb);
        if (hb && cb == 0) { assign(dst, a); return; }
        if (ha && hb) {
            uint32_t sh = static_cast<uint32_t>(cb & (is_64bit ? 63u : 31u));
            uint64_t mask = is_64bit ? ~0ULL : 0xFFFFFFFFULL;
            uint64_t res = 0;
            switch (kind) {
                case ShiftKind::Shl:  res = (ca << sh) & mask; break;
                case ShiftKind::ShrU: res = (ca & mask) >> sh; break;
                case ShiftKind::ShrS:
                    if (is_64bit) res = static_cast<uint64_t>(static_cast<int64_t>(ca) >> sh);
                    else res = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(ca) >> sh));
                    break;
                case ShiftKind::RotL: {
                    uint32_t w = is_64bit ? 64u : 32u;
                    res = (sh == 0) ? (ca & mask) : (((ca << sh) | ((ca & mask) >> (w - sh))) & mask);
                    break;
                }
                case ShiftKind::RotR: {
                    uint32_t w = is_64bit ? 64u : 32u;
                    res = (sh == 0) ? (ca & mask) : ((((ca & mask) >> sh) | (ca << (w - sh))) & mask);
                    break;
                }
            }
            assign(dst, make_const(res));
            return;
        }
        assign(dst, DomainT::top());
    }

    void on_other(compiler::SsaId dst, const compiler::SsaNode& node) override {
        if (dst != compiler::kInvalidSsa && node.type != compiler::ValType::Void &&
            node.type != compiler::ValType::Token) {
            assign(dst, DomainT::top());
        }
    }

protected:
    State* state_ = nullptr;
};

/// Result of forward SSA CFG worklist dataflow analysis.
template <typename DomainT>
struct SsaDataflowResult {
    bool converged = false;
    uint32_t iterations = 0;

    /// Unified converged abstract state across the entire SSA module
    /// (since every SsaId has a unique definition, `final_state.val(id)` gives
    /// its fixpoint abstract value).
    AbstractState<DomainT> final_state;
};

/// Forward chaotic-iteration worklist fixpoint solver over a `ValhallSsaModule`.
///
/// Exploits SSA single-static-assignment: `ssa_vals` is stored ONCE in O(|V|)
/// space (never replicated per block), while mutable side-channel maps
/// (`fau_mem`, `sym_mem`) are joined per basic block. This scales smoothly to
/// 100KB+ shaders with tens of thousands of SSA nodes and hundreds of blocks.
template <typename DomainT>
SsaDataflowResult<DomainT> run_ssa_forward_dataflow(
    const compiler::ValhallSsaModule& mod,
    const AbstractState<DomainT>& entry_state,
    SsaDomainTransferVisitor<DomainT>& visitor,
    const SsaLivenessResult* liveness = nullptr,
    uint32_t max_iterations = 64) {
    using Value = typename DomainT::Value;
    SsaDataflowResult<DomainT> res;
    const size_t num_blocks = mod.blocks.size();
    const size_t num_nodes = mod.nodes.size();

    res.final_state = entry_state;
    res.final_state.ensure_size(num_nodes);

    if (num_blocks == 0) {
        res.converged = true;
        return res;
    }

    // Per-block exit side-channel maps only (small: typically 0..20 FAU entries)
    std::vector<std::map<uint32_t, Value>> block_out_fau(num_blocks);
    std::vector<std::map<std::string, Value>> block_out_sym(num_blocks);
    std::vector<bool> visited(num_blocks, false);

    auto map_equals = [](const auto& a, const auto& b) -> bool {
        if (a.size() != b.size()) return false;
        for (const auto& kv : a) {
            auto it = b.find(kv.first);
            if (it == b.end() || !DomainT::equals(kv.second, it->second)) return false;
        }
        return true;
    };

    auto join_map = [](auto& dst, const auto& src) {
        for (const auto& kv : src) {
            auto it = dst.find(kv.first);
            if (it == dst.end()) {
                if (!DomainT::equals(kv.second, DomainT::bottom())) dst[kv.first] = kv.second;
            } else {
                it->second = DomainT::join(it->second, kv.second);
            }
        }
    };

    std::deque<uint32_t> worklist;
    std::vector<bool> in_worklist(num_blocks, false);
    for (uint32_t b = 0; b < num_blocks; ++b) {
        worklist.push_back(b);
        in_worklist[b] = true;
    }

    const size_t max_visits = static_cast<size_t>(max_iterations) * std::max<size_t>(1, num_blocks);
    size_t visits = 0;

    while (!worklist.empty() && visits < max_visits) {
        uint32_t b = worklist.front();
        worklist.pop_front();
        in_worklist[b] = false;
        ++visits;

        const auto& blk = mod.blocks[b];

        // Join mutable side-channel maps from visited predecessors
        res.final_state.fau_mem.clear();
        res.final_state.sym_mem.clear();
        if (b == 0) {
            res.final_state.fau_mem = entry_state.fau_mem;
            res.final_state.sym_mem = entry_state.sym_mem;
        }
        for (uint32_t pred : blk.preds) {
            if (pred < num_blocks && visited[pred]) {
                join_map(res.final_state.fau_mem, block_out_fau[pred]);
                join_map(res.final_state.sym_mem, block_out_sym[pred]);
            }
        }

        visitor.set_state(&res.final_state);
        bool ssa_changed = false;

        // 1. Evaluate Phi nodes at block entry (monotone join into shared ssa_vals)
        for (compiler::SsaId pid : blk.phis) {
            if (pid == compiler::kInvalidSsa || pid >= num_nodes) continue;
            Value prev = res.final_state.val(pid);
            dispatch_ssa_transfer(mod.get(pid), visitor);
            Value next = DomainT::join(prev, res.final_state.val(pid));
            if (!DomainT::equals(next, prev)) {
                res.final_state.set_val(pid, next);
                ssa_changed = true;
            } else {
                res.final_state.set_val(pid, prev);
            }
        }

        // 2. Evaluate body nodes in topological/program order
        for (compiler::SsaId nid : blk.nodes) {
            if (nid == compiler::kInvalidSsa || nid >= num_nodes) continue;
            Value prev = res.final_state.val(nid);
            const auto& node = mod.get(nid);
            dispatch_ssa_transfer(node, visitor);
            Value next = DomainT::join(prev, res.final_state.val(nid));
            if (!DomainT::equals(next, prev)) {
                res.final_state.set_val(nid, next);
                ssa_changed = true;
            } else {
                res.final_state.set_val(nid, prev);
            }
            if (liveness && nid < liveness->killed_after_node.size()) {
                for (compiler::SsaId kid : liveness->killed_after_node[nid]) {
                    visitor.on_ssa_killed(kid, node);
                }
            }
        }

        bool mem_changed = !visited[b] ||
                           !map_equals(res.final_state.fau_mem, block_out_fau[b]) ||
                           !map_equals(res.final_state.sym_mem, block_out_sym[b]);
        visited[b] = true;
        if (ssa_changed || mem_changed) {
            block_out_fau[b] = res.final_state.fau_mem;
            block_out_sym[b] = res.final_state.sym_mem;
            auto enqueue = [&](int32_t succ) {
                if (succ >= 0 && static_cast<size_t>(succ) < num_blocks && !in_worklist[succ]) {
                    worklist.push_back(static_cast<uint32_t>(succ));
                    in_worklist[succ] = true;
                }
            };
            enqueue(blk.fallthrough_block);
            if (blk.branch_target_block != blk.fallthrough_block) {
                enqueue(blk.branch_target_block);
            }
        }
    }

    res.converged = worklist.empty();
    res.iterations = static_cast<uint32_t>((visits + num_blocks - 1) / std::max<size_t>(1, num_blocks));

    // Merge all visited block exit side-channel maps into final_state
    res.final_state.fau_mem = entry_state.fau_mem;
    res.final_state.sym_mem = entry_state.sym_mem;
    for (size_t b = 0; b < num_blocks; ++b) {
        if (visited[b]) {
            join_map(res.final_state.fau_mem, block_out_fau[b]);
            join_map(res.final_state.sym_mem, block_out_sym[b]);
        }
    }
    return res;
}

}  // namespace valhall::dataflow
