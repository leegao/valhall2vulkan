#pragma once
#include "valhall_isa_generated.h"
#include "valhall_cfg.h"
#include "valhall_tables_generated.h"
#include "valhall_effects.h"
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace valhall::compiler {

enum class ValType : uint8_t {
    Void = 0,
    Bool,
    U32,
    S32,
    F32,
    U64,
    S64,
    V2F32,
    V2U32,
    V2S32,
    V4F32,
    V4U32,
    V4S32,
    V3U32,
    V2Bool,
    TupleU32x2,
    TupleValToken,
    Token,
};

using SsaId = uint32_t;
constexpr SsaId kInvalidSsa = 0;

struct PhiIncoming {
    uint32_t pred_block = 0;
    SsaId value = kInvalidSsa;
};

struct SsaNode {
    SsaId id = kInvalidSsa;
    tables::SsaOp op = tables::SsaOp::Nop;
    ValType type = ValType::Void;
    uint32_t block_id = 0;
    bool is_barrier_or_cf = false;
    bool is_side_effect = false;

    std::vector<SsaId> args;
    std::vector<PhiIncoming> phi_incomings;

    // Constant / immediate / auxiliary metadata
    uint64_t imm_u64 = 0;
    uint32_t imm_u32 = 0;
    uint32_t imm_u32_2 = 0;
    uint32_t aux_index = 0;
    bool aux_bool = false;
    bool aux_bool2 = false;
    std::string aux_str;

    // Effect contribution from lowering
    tables::EffectMask effects{};
};

struct SsaBlock {
    uint32_t id = 0;
    std::vector<SsaId> phis;
    std::vector<SsaId> nodes; // Topological/program order within block

    // CFG edges (preserving structured CFG metadata from cfg::BasicBlock)
    std::vector<uint32_t> preds;
    int32_t branch_target_block = -1;
    int32_t fallthrough_block = -1;
    bool is_conditional_branch = false;
    bool is_unconditional_branch = false;
    bool is_terminated_by_end = false;
    bool is_terminated_by_discard = false;

    bool is_loop_header = false;
    int32_t loop_backedge_block = -1;
    int32_t loop_merge_block = -1;
    int32_t selection_merge_block = -1;

    // Condition metadata for conditional branch
    SsaId branch_cond = kInvalidSsa;
};

struct ValhallSsaModule {
    std::vector<SsaBlock> blocks;
    std::vector<SsaNode> nodes; // 1-indexed (nodes[0] is dummy)
    std::vector<uint32_t> use_counts;

    // Register state during Pass 0/1 construction: reg_defs[block_id][reg] -> SsaId
    std::vector<std::array<SsaId, 64>> block_in_regs;
    std::vector<std::array<SsaId, 64>> block_out_regs;
    std::vector<SsaId> block_in_token;
    std::vector<SsaId> block_out_token;

    ValhallSsaModule() {
        nodes.emplace_back(); // 0 = kInvalidSsa
        use_counts.push_back(0);
    }

    SsaId add_node(SsaNode n) {
        SsaId id = static_cast<SsaId>(nodes.size());
        n.id = id;
        nodes.push_back(std::move(n));
        use_counts.push_back(0);
        return id;
    }

    const SsaNode& get(SsaId id) const { return nodes[id]; }
    SsaNode& get_mut(SsaId id) { return nodes[id]; }

    void recompute_use_counts();
};

// Sub-block splitting on cfg::ControlFlowGraph for Pass 0
// Splits blocks at instructions marked splits_block in valhall_lowering.json
// (excluding block-terminal branches which already end blocks).
struct SplitCfgResult {
    std::vector<SsaBlock> blocks;
    // Per SsaBlock: list of instructions from cfg.instructions
    std::vector<std::vector<DecodedValhallInst>> block_insts;
};

SplitCfgResult split_cfg_subblocks(const cfg::ControlFlowGraph& cfg);

// Mem2reg / Dominance-Frontier SSA promotion helper over r0..r63 and effect tokens
class SsaBuilder {
public:
    explicit SsaBuilder(ValhallSsaModule& mod) : mod_(mod) {}

    SsaId current_reg(uint32_t blk_id, uint8_t reg) const;
    void set_reg(uint32_t blk_id, uint8_t reg, SsaId val);

    SsaId current_token(uint32_t blk_id) const;
    void set_token(uint32_t blk_id, SsaId tok);

    SsaId emit(uint32_t blk_id, SsaNode n);
    SsaId emit_const_u32(uint32_t blk_id, uint32_t v);
    SsaId emit_const_s32(uint32_t blk_id, int32_t v);
    SsaId emit_const_f32_bits(uint32_t blk_id, uint32_t bits);
    SsaId emit_const_u64(uint32_t blk_id, uint64_t v);
    SsaId emit_const_bool(uint32_t blk_id, bool v);
    SsaId emit_bitcast(uint32_t blk_id, ValType target_t, SsaId val);

    // Finalize Phi nodes across all blocks using iterative dominance-frontier / Braun-style resolution
    void seal_all_phis();

private:
    ValhallSsaModule& mod_;
};

} // namespace valhall::compiler
