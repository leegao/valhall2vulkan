#pragma once

#include "abstract_domain.h"
#include "domains.h"
#include "liveness.h"
#include "valhall_ssa.h"
#include "worklist.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace valhall::dataflow {

/// High-level semantic classification of `tables::SsaOp`.
/// Decouples op categorization from transfer dispatch so `dispatch_ssa_transfer`
/// switches on self-contained semantic kinds rather than 134 raw opcodes.
enum class SsaOpKind : uint8_t {
    Const = 0,
    Copy,
    Phi,
    Select,
    IntAdd,
    IntSub,
    IntMul,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shift,
    Pack64,
    Split64,
    TupleExtract,
    Convert,
    Vec,
    LoadFau,
    LoadBuiltin,
    LoadSideband,
    PsbLoad,
    PsbStore,
    LoadVertAttr,
    LoadFragVarying,
    TexSample,
    StoreOutput,
    Atomic,
    Other,
};

/// Shift / rotate direction for normalized `on_shift` dispatch.
enum class ShiftKind : uint8_t {
    Shl = 0,
    ShrU,
    ShrS,
    RotL,
    RotR,
};

/// Classify any `tables::SsaOp` into its semantic `SsaOpKind`.
SsaOpKind classify_ssa_op(tables::SsaOp op);

/// Returns true if `op` is a 64-bit integer / bitwise / shift / sideband op.
bool is_ssa_op_64bit(tables::SsaOp op);

/// Returns the `ShiftKind` for shift/rotate ops (defaults to `ShiftKind::Shl`).
ShiftKind classify_ssa_shift_kind(tables::SsaOp op);

/// Abstract SSA transfer-function visitor base class.
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

/// Domain-aware SSA transfer visitor providing built-in **extensional equalities**.
template <typename DomainT>
class SsaDomainTransferVisitor : public SsaTransferVisitorBase {
public:
    static_assert(ValidateLatticeDomain<DomainT>::valid, "");
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
            if (inc.value == dst) continue;
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

/// Result of directional SSA CFG worklist dataflow analysis.
template <typename DomainT>
struct SsaDataflowResult {
    static_assert(ValidateLatticeDomain<DomainT>::valid, "");
    bool converged = false;
    uint32_t iterations = 0;
    AbstractState<DomainT> final_state;
};

/// Directional chaotic-iteration fixpoint solver over a `ValhallSsaModule`,
/// backed by the domain-independent `run_cfg_worklist()` in `worklist.cpp`.
///
/// Exploits SSA single-static-assignment: `ssa_vals` is stored ONCE in O(|V|)
/// space, while mutable side-channel maps (`fau_mem`, `sym_mem`) are joined per
/// basic block (`O(|V| + |B| * |mu|)` memory).
template <typename DomainT>
SsaDataflowResult<DomainT> run_ssa_dataflow(
    const compiler::ValhallSsaModule& mod,
    DataflowDirection dir,
    const AbstractState<DomainT>& entry_state,
    SsaDomainTransferVisitor<DomainT>& visitor,
    const SsaLivenessResult* liveness = nullptr,
    uint32_t max_iterations = 64) {
    static_assert(ValidateLatticeDomain<DomainT>::valid, "");
    using Value = typename DomainT::Value;
    using FauMemDomain = typename AbstractState<DomainT>::FauMemDomain;
    using SymMemDomain = typename AbstractState<DomainT>::SymMemDomain;

    SsaDataflowResult<DomainT> res;
    const size_t num_blocks = mod.blocks.size();
    const size_t num_nodes = mod.nodes.size();

    res.final_state = entry_state;
    res.final_state.ensure_size(num_nodes);

    if (num_blocks == 0) {
        res.converged = true;
        return res;
    }

    std::vector<typename FauMemDomain::Value> block_out_fau(num_blocks);
    std::vector<typename SymMemDomain::Value> block_out_sym(num_blocks);
    std::vector<bool> visited(num_blocks, false);

    auto eval_node = [&](compiler::SsaId id) -> bool {
        if (id == compiler::kInvalidSsa || id >= num_nodes) return false;
        Value prev = res.final_state.val(id);
        const auto& node = mod.get(id);
        dispatch_ssa_transfer(node, visitor);
        Value next = DomainT::join(prev, res.final_state.val(id));
        if (!domain_equals<DomainT>(next, prev)) {
            res.final_state.set_val(id, next);
            return true;
        }
        res.final_state.set_val(id, prev);
        return false;
    };

    WorklistResult wr = run_cfg_worklist(
        mod,
        dir,
        [&](uint32_t b) -> bool {
            const auto& blk = mod.blocks[b];

            // Join side-channel memory maps from incoming edges
            res.final_state.fau_mem.clear();
            res.final_state.sym_mem.clear();
            if (dir == DataflowDirection::Forward) {
                if (b == 0) {
                    res.final_state.fau_mem = entry_state.fau_mem;
                    res.final_state.sym_mem = entry_state.sym_mem;
                }
                for (uint32_t pred : blk.preds) {
                    if (pred < num_blocks && visited[pred]) {
                        FauMemDomain::join_into(res.final_state.fau_mem, block_out_fau[pred]);
                        SymMemDomain::join_into(res.final_state.sym_mem, block_out_sym[pred]);
                    }
                }
            } else {
                auto join_succ = [&](int32_t s) {
                    if (s >= 0 && static_cast<size_t>(s) < num_blocks && visited[s]) {
                        FauMemDomain::join_into(res.final_state.fau_mem, block_out_fau[s]);
                        SymMemDomain::join_into(res.final_state.sym_mem, block_out_sym[s]);
                    }
                };
                join_succ(blk.fallthrough_block);
                if (blk.branch_target_block != blk.fallthrough_block) join_succ(blk.branch_target_block);
                if (blk.fallthrough_block < 0 && blk.branch_target_block < 0) {
                    res.final_state.fau_mem = entry_state.fau_mem;
                    res.final_state.sym_mem = entry_state.sym_mem;
                }
            }

            visitor.set_state(&res.final_state);
            bool ssa_changed = false;

            if (dir == DataflowDirection::Forward) {
                for (compiler::SsaId pid : blk.phis) {
                    if (eval_node(pid)) ssa_changed = true;
                }
                for (compiler::SsaId nid : blk.nodes) {
                    if (eval_node(nid)) ssa_changed = true;
                    if (liveness && nid < liveness->killed_after_node.size()) {
                        for (compiler::SsaId kid : liveness->killed_after_node[nid]) {
                            visitor.on_ssa_killed(kid, mod.get(nid));
                        }
                    }
                }
            } else {
                for (auto it = blk.nodes.rbegin(); it != blk.nodes.rend(); ++it) {
                    if (eval_node(*it)) ssa_changed = true;
                }
                for (auto it = blk.phis.rbegin(); it != blk.phis.rend(); ++it) {
                    if (eval_node(*it)) ssa_changed = true;
                }
            }

            bool mem_changed = !visited[b] ||
                               !FauMemDomain::equals(res.final_state.fau_mem, block_out_fau[b]) ||
                               !SymMemDomain::equals(res.final_state.sym_mem, block_out_sym[b]);
            visited[b] = true;
            if (ssa_changed || mem_changed) {
                block_out_fau[b] = res.final_state.fau_mem;
                block_out_sym[b] = res.final_state.sym_mem;
                return true;
            }
            return false;
        },
        max_iterations);

    res.converged = wr.converged;
    res.iterations = wr.iterations;

    // Merge all visited block exit side-channel maps into final_state
    res.final_state.fau_mem = entry_state.fau_mem;
    res.final_state.sym_mem = entry_state.sym_mem;
    for (size_t b = 0; b < num_blocks; ++b) {
        if (visited[b]) {
            FauMemDomain::join_into(res.final_state.fau_mem, block_out_fau[b]);
            SymMemDomain::join_into(res.final_state.sym_mem, block_out_sym[b]);
        }
    }
    return res;
}

/// Convenience wrapper for forward SSA dataflow analysis.
template <typename DomainT>
SsaDataflowResult<DomainT> run_ssa_forward_dataflow(
    const compiler::ValhallSsaModule& mod,
    const AbstractState<DomainT>& entry_state,
    SsaDomainTransferVisitor<DomainT>& visitor,
    const SsaLivenessResult* liveness = nullptr,
    uint32_t max_iterations = 64) {
    return run_ssa_dataflow<DomainT>(
        mod, DataflowDirection::Forward, entry_state, visitor, liveness, max_iterations);
}

}  // namespace valhall::dataflow
