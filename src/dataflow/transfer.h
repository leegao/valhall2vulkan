#pragma once

#include "abstract_domain.h"
#include "valhall_cfg.h"
#include "valhall_isa_generated.h"

#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace valhall::dataflow {

/// Normalized operand passed to grouped Valhall transfer functions.
struct AbstractOperand {
    enum class Kind : uint8_t {
        None = 0,
        Register,
        FauUniform,
        SmallConstant,
        FauSpecial,
        Immediate,
    };

    Kind kind = Kind::None;
    bool is_64bit = false;
    uint8_t reg_num = 0;
    bool discard_last_use = false;
    uint8_t fau_page = 0;
    uint32_t fau_byte_offset = 0;
    uint64_t imm_u64 = 0;
    int64_t imm_s64 = 0;
    std::string symbolic_name;
    std::string swizzle_mod;
    bool abs = false;
    bool neg = false;
    bool not_mod = false;

    static AbstractOperand from_decoded(const DecodedSource& s);
    static AbstractOperand from_imm(uint64_t u64, int64_t s64);
    static AbstractOperand from_reg(uint8_t reg, bool is64 = false);

    bool is_valid() const { return kind != Kind::None; }
    bool is_reg() const { return kind == Kind::Register; }
    bool is_const() const { return kind == Kind::Immediate || kind == Kind::SmallConstant; }
    bool is_const_zero() const { return is_const() && imm_u64 == 0u; }
    bool is_const_all_ones() const {
        return is_const() && (imm_u64 == 0xFFFFFFFFULL || imm_u64 == 0xFFFFFFFFFFFFFFFFULL);
    }
    bool same_reg(const AbstractOperand& other) const {
        return kind == Kind::Register && other.kind == Kind::Register &&
               reg_num == other.reg_num && is_64bit == other.is_64bit &&
               swizzle_mod == other.swizzle_mod && !not_mod && !other.not_mod &&
               !neg && !other.neg && !abs && !other.abs;
    }
};

/// Normalized destination (covers both ALU dest_reg and Load/Message staging registers).
struct AbstractDest {
    bool present = false;
    uint8_t base_reg = 0;
    uint8_t reg_count = 1;  // 1 for 32-bit, 2 for 64-bit, 1..4 for staging vectors
    bool is_64bit = false;

    static AbstractDest from_alu(const DecodedValhallInst& insn);
    static AbstractDest from_staging(const DecodedStaging& sr);
};

enum class ShiftKind : uint8_t {
    LogicalLeft = 0,
    LogicalRight = 1,
    ArithmeticRight = 2,
    Rotate = 3,
};

/// Abstract interpretation visitor interface for grouped Valhall ISA operations.
/// Every Valhall mnemonic is normalized into a clean semantic group while still
/// passing `const DecodedValhallInst& insn` for concrete field/modifier inspection.
class TransferVisitorBase {
public:
    virtual ~TransferVisitorBase() = default;

    // Called before/after each instruction dispatch
    virtual void on_inst_begin(const DecodedValhallInst& /*insn*/) {}
    virtual void on_inst_end(const DecodedValhallInst& /*insn*/) {}

    // Live-range kill hook (driven directly by ISA `^` discard_last_use bits)
    virtual void on_reg_killed(uint8_t /*reg*/, const DecodedValhallInst& /*insn*/) {}

    // Bitwise (AND/AND_IMM/LSHIFT_AND..., OR/OR_IMM/..., XOR/XOR_IMM/..., NOT, shifts)
    virtual void on_and(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) = 0;
    virtual void on_or(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) = 0;
    virtual void on_xor(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) = 0;
    virtual void on_not(const AbstractDest& dst, const AbstractOperand& src, const DecodedValhallInst& insn) = 0;
    virtual void on_shift(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& amount, ShiftKind kind, const DecodedValhallInst& insn) = 0;

    // Integer arithmetic
    virtual void on_iadd(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) = 0;
    virtual void on_isub(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) = 0;
    virtual void on_imul(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) = 0;
    virtual void on_shadd(const AbstractDest& dst, const AbstractOperand& base, const AbstractOperand& index, uint32_t shift, const DecodedValhallInst& insn) = 0;
    virtual void on_iunary(const AbstractDest& dst, const AbstractOperand& src, const DecodedValhallInst& insn) = 0;

    // Floating-point & SFU
    virtual void on_fadd(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) = 0;
    virtual void on_fmul(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) = 0;
    virtual void on_fma(const AbstractDest& dst, const AbstractOperand& a, const AbstractOperand& b, const AbstractOperand& c, const DecodedValhallInst& insn) = 0;
    virtual void on_fminmax(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) = 0;
    virtual void on_funary(const AbstractDest& dst, const AbstractOperand& src, const DecodedValhallInst& insn) = 0;

    // Move, Convert, Compare, Select
    virtual void on_mov(const AbstractDest& dst, const AbstractOperand& src, const DecodedValhallInst& insn) = 0;
    virtual void on_convert(const AbstractDest& dst, const AbstractOperand& src, const DecodedValhallInst& insn) = 0;
    virtual void on_cmp(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) = 0;
    virtual void on_csel(const AbstractDest& dst, const AbstractOperand& cond0, const AbstractOperand& cond1,
                         const AbstractOperand& true_val, const AbstractOperand& false_val, const DecodedValhallInst& insn) = 0;

    // PC-relative & Descriptor Address Calculation
    virtual void on_adr(const AbstractDest& dst, int64_t target_pc, const DecodedValhallInst& insn) = 0;
    virtual void on_lea(const AbstractDest& dst, const AbstractOperand& idx, uint32_t index, uint32_t table, const DecodedValhallInst& insn) = 0;

    // Memory & Atomics
    virtual void on_load(const AbstractDest& dst, const AbstractOperand& base, int64_t offset, const DecodedValhallInst& insn) = 0;
    virtual void on_store(const DecodedStaging& src, const AbstractOperand& base, int64_t offset, const DecodedValhallInst& insn) = 0;
    virtual void on_atomic(const AbstractDest& dst, const DecodedStaging& src, const AbstractOperand& base, int64_t offset, const DecodedValhallInst& insn) = 0;

    // Graphics, Texture & Fragment Output
    virtual void on_ld_attr(const AbstractDest& dst, uint32_t index, uint32_t table, const DecodedValhallInst& insn) = 0;
    virtual void on_ld_var(const AbstractDest& dst, uint32_t index, uint32_t table, const DecodedValhallInst& insn) = 0;
    virtual void on_tex(const AbstractDest& dst, uint32_t tex_idx, uint32_t smp_idx, const DecodedValhallInst& insn) = 0;
    virtual void on_blend(const DecodedStaging& src, int32_t rt_idx, const DecodedValhallInst& insn) = 0;

    // Control flow & fallback
    virtual void on_branch(const AbstractOperand& cond, int64_t target_pc, const DecodedValhallInst& insn) = 0;
    virtual void on_nop_or_barrier(const DecodedValhallInst& insn) = 0;
    virtual void on_unhandled(const AbstractDest& dst, const DecodedValhallInst& insn) = 0;
};

/// Dispatch a decoded Valhall instruction into the normalized transfer visitor,
/// automatically normalizing mnemonic variants and firing `on_reg_killed` for `^` operands.
void dispatch_valhall_transfer(const DecodedValhallInst& insn, TransferVisitorBase& visitor);

/// Generic domain-backed transfer visitor providing sensible extensional-equality defaults:
///   - `on_mov(dst, src)` copies/aliases `eval_operand(src)` into `dst`
///   - `x + 0`, `x - 0`, `x | 0`, `x ^ 0`, `x & ~0`, `x & x`, `x | x` copy `x`
///   - `on_csel(dst, ..., t, f)` joins `eval_operand(t)` and `eval_operand(f)`
///   - `on_reg_killed(r)` marks `state_.kill_reg(r)`
///   - All other writes default to `DomainT::top()` so subclasses only override what they refine.
template <typename DomainT>
class DomainTransferVisitor : public TransferVisitorBase {
public:
    using State = AbstractState<DomainT>;
    using Value = typename DomainT::Value;

    explicit DomainTransferVisitor(State* state) : state_(state) {}

    void set_state(State* state) { state_ = state; }
    State& state() { return *state_; }
    const State& state() const { return *state_; }

    /// Resolve an AbstractOperand to its abstract value in the current state.
    virtual Value eval_operand(const AbstractOperand& op) const {
        if (!state_) return DomainT::top();
        switch (op.kind) {
            case AbstractOperand::Kind::Register:
                return state_->get_reg(op.reg_num);
            case AbstractOperand::Kind::FauUniform:
                return state_->get_fau(op.fau_byte_offset);
            default:
                return DomainT::top();
        }
    }

    void write_dest(const AbstractDest& dst, const Value& v) {
        if (!state_ || !dst.present) return;
        state_->set_reg_range(dst.base_reg, dst.reg_count, v);
    }

    void on_reg_killed(uint8_t reg, const DecodedValhallInst& /*insn*/) override {
        if (state_) state_->kill_reg(reg);
    }

    // --- Extensional equalities ---
    void on_mov(const AbstractDest& dst, const AbstractOperand& src, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, eval_operand(src));
    }

    void on_and(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& /*insn*/) override {
        if (rhs.is_const_all_ones() || lhs.same_reg(rhs)) {
            write_dest(dst, eval_operand(lhs));
        } else if (lhs.is_const_all_ones()) {
            write_dest(dst, eval_operand(rhs));
        } else {
            write_dest(dst, DomainT::top());
        }
    }

    void on_or(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& /*insn*/) override {
        if (rhs.is_const_zero() || lhs.same_reg(rhs)) {
            write_dest(dst, eval_operand(lhs));
        } else if (lhs.is_const_zero()) {
            write_dest(dst, eval_operand(rhs));
        } else {
            write_dest(dst, DomainT::top());
        }
    }

    void on_xor(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& /*insn*/) override {
        if (rhs.is_const_zero()) {
            write_dest(dst, eval_operand(lhs));
        } else if (lhs.is_const_zero()) {
            write_dest(dst, eval_operand(rhs));
        } else {
            write_dest(dst, DomainT::top());
        }
    }

    void on_not(const AbstractDest& dst, const AbstractOperand& /*src*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }

    void on_shift(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& amount, ShiftKind /*kind*/, const DecodedValhallInst& /*insn*/) override {
        if (amount.is_const_zero()) write_dest(dst, eval_operand(lhs));
        else write_dest(dst, DomainT::top());
    }

    void on_iadd(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& /*insn*/) override {
        if (rhs.is_const_zero()) write_dest(dst, eval_operand(lhs));
        else if (lhs.is_const_zero()) write_dest(dst, eval_operand(rhs));
        else write_dest(dst, DomainT::top());
    }

    void on_isub(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& /*insn*/) override {
        if (rhs.is_const_zero()) write_dest(dst, eval_operand(lhs));
        else write_dest(dst, DomainT::top());
    }

    void on_imul(const AbstractDest& dst, const AbstractOperand& /*lhs*/, const AbstractOperand& /*rhs*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }

    void on_shadd(const AbstractDest& dst, const AbstractOperand& base, const AbstractOperand& index, uint32_t /*shift*/, const DecodedValhallInst& /*insn*/) override {
        if (index.is_const_zero()) write_dest(dst, eval_operand(base));
        else write_dest(dst, DomainT::top());
    }

    void on_iunary(const AbstractDest& dst, const AbstractOperand& /*src*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }

    void on_fadd(const AbstractDest& dst, const AbstractOperand& /*lhs*/, const AbstractOperand& /*rhs*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_fmul(const AbstractDest& dst, const AbstractOperand& /*lhs*/, const AbstractOperand& /*rhs*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_fma(const AbstractDest& dst, const AbstractOperand& /*a*/, const AbstractOperand& /*b*/, const AbstractOperand& /*c*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_fminmax(const AbstractDest& dst, const AbstractOperand& /*lhs*/, const AbstractOperand& /*rhs*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_funary(const AbstractDest& dst, const AbstractOperand& /*src*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_convert(const AbstractDest& dst, const AbstractOperand& /*src*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_cmp(const AbstractDest& dst, const AbstractOperand& /*lhs*/, const AbstractOperand& /*rhs*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }

    void on_csel(const AbstractDest& dst, const AbstractOperand& /*cond0*/, const AbstractOperand& /*cond1*/,
                 const AbstractOperand& true_val, const AbstractOperand& false_val, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::join(eval_operand(true_val), eval_operand(false_val)));
    }

    void on_adr(const AbstractDest& dst, int64_t /*target_pc*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_lea(const AbstractDest& dst, const AbstractOperand& /*idx*/, uint32_t /*index*/, uint32_t /*table*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_load(const AbstractDest& dst, const AbstractOperand& /*base*/, int64_t /*offset*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_store(const DecodedStaging& /*src*/, const AbstractOperand& /*base*/, int64_t /*offset*/, const DecodedValhallInst& /*insn*/) override {}
    void on_atomic(const AbstractDest& dst, const DecodedStaging& /*src*/, const AbstractOperand& /*base*/, int64_t /*offset*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_ld_attr(const AbstractDest& dst, uint32_t /*index*/, uint32_t /*table*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_ld_var(const AbstractDest& dst, uint32_t /*index*/, uint32_t /*table*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_tex(const AbstractDest& dst, uint32_t /*tex_idx*/, uint32_t /*smp_idx*/, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }
    void on_blend(const DecodedStaging& /*src*/, int32_t /*rt_idx*/, const DecodedValhallInst& /*insn*/) override {}
    void on_branch(const AbstractOperand& /*cond*/, int64_t /*target_pc*/, const DecodedValhallInst& /*insn*/) override {}
    void on_nop_or_barrier(const DecodedValhallInst& /*insn*/) override {}
    void on_unhandled(const AbstractDest& dst, const DecodedValhallInst& /*insn*/) override {
        write_dest(dst, DomainT::top());
    }

protected:
    State* state_ = nullptr;
};

/// Result of running forward worklist abstract interpretation over a decoded shader.
template <typename DomainT>
struct DataflowFixpointResult {
    using State = AbstractState<DomainT>;
    std::vector<State> pre_states;   // Abstract state immediately BEFORE instruction[i]
    std::vector<State> post_states;  // Abstract state immediately AFTER instruction[i]
    size_t iterations = 0;
};

/// CFG-aware forward worklist abstract interpretation driver.
/// Joins incoming states at basic-block merge points until fixpoint (bounded by max_iters).
template <typename DomainT, typename VisitorT>
DataflowFixpointResult<DomainT> run_forward_dataflow(
    const std::vector<DecodedValhallInst>& instructions,
    const AbstractState<DomainT>& entry_state,
    VisitorT& visitor,
    size_t max_iters = 64) {
    using State = AbstractState<DomainT>;
    DataflowFixpointResult<DomainT> res;
    res.pre_states.resize(instructions.size(), State::bottom());
    res.post_states.resize(instructions.size(), State::bottom());
    if (instructions.empty()) return res;

    cfg::ControlFlowGraph g = cfg::build_cfg(instructions);
    if (g.blocks.empty()) return res;

    // Map PC -> original instruction index
    std::map<uint64_t, size_t> pc_to_idx;
    for (size_t i = 0; i < instructions.size(); ++i) {
        pc_to_idx[instructions[i].pc] = i;
    }

    const size_t nblk = g.blocks.size();
    std::vector<State> blk_in(nblk, State::bottom());
    std::vector<bool> visited(nblk, false);
    std::vector<bool> in_queue(nblk, false);
    std::deque<size_t> worklist;

    blk_in[0] = entry_state;
    visited[0] = true;
    worklist.push_back(0);
    in_queue[0] = true;

    while (!worklist.empty() && res.iterations < max_iters * nblk) {
        size_t bidx = worklist.front();
        worklist.pop_front();
        in_queue[bidx] = false;
        ++res.iterations;

        State cur = blk_in[bidx];
        const auto& blk = g.blocks[bidx];
        for (size_t gi : blk.inst_indices) {
            if (gi >= g.instructions.size()) continue;
            const auto& insn = g.instructions[gi];
            auto it = pc_to_idx.find(insn.pc);
            size_t orig_idx = (it != pc_to_idx.end()) ? it->second : gi;
            if (orig_idx < res.pre_states.size()) {
                res.pre_states[orig_idx] = cur;
            }
            visitor.set_state(&cur);
            dispatch_valhall_transfer(insn, visitor);
            if (orig_idx < res.post_states.size()) {
                res.post_states[orig_idx] = cur;
            }
        }

        int32_t succs_arr[2] = { blk.fallthrough_block, blk.branch_target_block };
        for (int32_t succ_i : succs_arr) {
            if (succ_i < 0) continue;
            uint32_t succ = static_cast<uint32_t>(succ_i);
            if (succ >= nblk) continue;
            if (!visited[succ]) {
                blk_in[succ] = cur;
                visited[succ] = true;
                if (!in_queue[succ]) {
                    worklist.push_back(succ);
                    in_queue[succ] = true;
                }
            } else if (blk_in[succ].join_with(cur)) {
                if (!in_queue[succ]) {
                    worklist.push_back(succ);
                    in_queue[succ] = true;
                }
            }
        }
    }

    return res;
}

}  // namespace valhall::dataflow
