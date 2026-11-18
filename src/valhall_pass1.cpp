#include "valhall_pass1.h"
#include "valhall_pass1_handlers.h"
#include "valhall_operands.h"
#include "valhall_tables_generated.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace valhall::compiler {

namespace {

#include "valhall_pass1_lowering_generated.inc"

inline bool pass1_verbose_enabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("VALHALL_PASS1_VERBOSE");
        return v && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

inline const char* lowering_kind_name(tables::LoweringKind k) {
    switch (k) {
        case tables::LoweringKind::DirectSsaOp: return "DirectSsaOp";
        case tables::LoweringKind::DynamicSsaOp: return "DynamicSsaOp";
        case tables::LoweringKind::Handler: return "Handler";
        case tables::LoweringKind::InlineOps: return "InlineOps";
    }
    return "Unknown";
}

void lower_instruction(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins) {
    size_t cand_count = 0;
    const uint16_t* cand_indices = tables::lookup_lowering_rule_candidates(ins.spec_index, &cand_count);
    const tables::LoweringRuleSpec* rule = nullptr;
    const uint32_t arch_bit = (ctx.opts.arch < 32) ? (1u << ctx.opts.arch) : 0u;
    for (size_t i = 0; i < cand_count; ++i) {
        const auto* cand = tables::get_lowering_rule(cand_indices[i]);
        if (!cand) continue;
        if ((cand->arch_mask & arch_bit) == 0) continue;
        if (cand->has_when && !eval_rule_when_generated(ctx, blk, ins, cand)) continue;
        rule = cand;
        break;
    }
    if (!rule) {
        std::cerr << "[VALHALL_PASS1][ERROR] Missing matching lowering rule for spec_index=" << ins.spec_index
                  << " arch=v" << static_cast<unsigned>(ctx.opts.arch)
                  << " pc=0x" << std::hex << ins.pc << std::dec << "\n";
        ctx.res.error_message = "Missing lowering rule for spec_index=" + std::to_string(ins.spec_index);
        return;
    }

    if (pass1_verbose_enabled()) {
        std::cerr << "[VALHALL_PASS1] blk=" << blk
                  << " pc=0x" << std::hex << ins.pc << std::dec
                  << " spec=" << ins.spec_index
                  << " rule=" << rule->rule_index << " (" << rule->mnemonic << ")"
                  << " kind=" << lowering_kind_name(rule->kind)
                  << " target=" << (rule->handler_name ? rule->handler_name : "")
                  << "\n";
    }

    if (!lower_rule_generated(ctx, blk, ins, rule)) {
        std::cerr << "[VALHALL_PASS1][ERROR] Unhandled rule_index=" << rule->rule_index
                  << " mnemonic=" << rule->mnemonic
                  << " pc=0x" << std::hex << ins.pc << std::dec << "\n";
        ctx.res.error_message = std::string("Unhandled lowering rule_index=") + std::to_string(rule->rule_index) +
                                " (" + rule->mnemonic + ")";
    }
}

} // namespace

Pass1Result run_pass1_lowering(
    const cfg::ControlFlowGraph& primary_cfg,
    const cfg::ControlFlowGraph& secondary_cfg,
    const ValhallTranslateOptions& opts) {
    Pass1Result res;
    res.is_fragment_stage = (opts.stage == ShaderStage::Fragment);
    res.is_compute_or_pilot = (opts.stage == ShaderStage::Compute || opts.stage == ShaderStage::Pilot);
    res.local_size_x = (opts.stage == ShaderStage::Pilot) ? 1u : std::max(1u, opts.local_size_x);
    res.local_size_y = (opts.stage == ShaderStage::Pilot) ? 1u : std::max(1u, opts.local_size_y);
    res.local_size_z = (opts.stage == ShaderStage::Pilot) ? 1u : std::max(1u, opts.local_size_z);

    // Sub-block splitting for primary (and optional secondary Varying EBIN)
    SplitCfgResult pri = split_cfg_subblocks(primary_cfg);
    SplitCfgResult sec;
    bool has_secondary = (opts.stage == ShaderStage::Vertex && !secondary_cfg.blocks.empty());
    if (has_secondary) {
        sec = split_cfg_subblocks(secondary_cfg);
    }

    if (pri.blocks.empty()) {
        res.ok = true;
        return res;
    }

    uint32_t sec_offset = static_cast<uint32_t>(pri.blocks.size());
    std::vector<std::vector<DecodedValhallInst>> all_insts = std::move(pri.block_insts);
    res.ssa.blocks = std::move(pri.blocks);

    if (has_secondary && !sec.blocks.empty()) {
        for (uint32_t i = 0; i < sec_offset; ++i) {
            if (res.ssa.blocks[i].is_terminated_by_end) {
                res.ssa.blocks[i].is_terminated_by_end = false;
                res.ssa.blocks[i].fallthrough_block = static_cast<int32_t>(sec_offset);
                sec.blocks[0].preds.push_back(i);
            }
        }
        for (size_t j = 0; j < sec.blocks.size(); ++j) {
            auto sb_blk = sec.blocks[j];
            sb_blk.id += sec_offset;
            if (j > 0) {
                for (auto& p : sb_blk.preds) p += sec_offset;
            }
            if (sb_blk.branch_target_block >= 0) sb_blk.branch_target_block += static_cast<int32_t>(sec_offset);
            if (sb_blk.fallthrough_block >= 0) sb_blk.fallthrough_block += static_cast<int32_t>(sec_offset);
            if (sb_blk.loop_backedge_block >= 0) sb_blk.loop_backedge_block += static_cast<int32_t>(sec_offset);
            if (sb_blk.loop_merge_block >= 0) sb_blk.loop_merge_block += static_cast<int32_t>(sec_offset);
            if (sb_blk.selection_merge_block >= 0) sb_blk.selection_merge_block += static_cast<int32_t>(sec_offset);
            res.ssa.blocks.push_back(std::move(sb_blk));
            all_insts.push_back(std::move(sec.block_insts[j]));
        }
        for (auto& b : res.ssa.blocks) b.preds.clear();
        for (uint32_t i = 0; i < res.ssa.blocks.size(); ++i) {
            const auto& b = res.ssa.blocks[i];
            if (b.branch_target_block >= 0) res.ssa.blocks[b.branch_target_block].preds.push_back(i);
            if (b.fallthrough_block >= 0 && b.fallthrough_block != b.branch_target_block) {
                res.ssa.blocks[b.fallthrough_block].preds.push_back(i);
            }
        }
    }

    size_t num_blks = res.ssa.blocks.size();
    res.ssa.block_in_regs.resize(num_blks);
    res.ssa.block_out_regs.resize(num_blks);
    res.ssa.block_in_token.assign(num_blks, kInvalidSsa);
    res.ssa.block_out_token.assign(num_blks, kInvalidSsa);

    Pass1Context ctx(res.ssa, opts, res);

    // Initialize block 0 registers to 0 and initial effect token
    SsaId zero_u32 = ctx.sb.emit_const_u32(0, 0u);
    for (uint32_t r = 0; r < 64; ++r) {
        res.ssa.block_in_regs[0][r] = zero_u32;
        res.ssa.block_out_regs[0][r] = zero_u32;
    }
    SsaNode init_tok{};
    init_tok.op = tables::SsaOp::Nop;
    init_tok.type = ValType::Token;
    SsaId tok0 = ctx.sb.emit(0, std::move(init_tok));
    res.ssa.block_in_token[0] = tok0;
    res.ssa.block_out_token[0] = tok0;

    // Emit stage preamble in block 0
    ctx.emit_stage_preamble(0, primary_cfg.instructions);

    // Lower all blocks in program order
    for (uint32_t b = 0; b < num_blks; ++b) {
        if (b > 0) {
            const auto& blk = res.ssa.blocks[b];
            bool single_forward_pred = (blk.preds.size() == 1 && blk.preds[0] < b);
            if (single_forward_pred) {
                uint32_t p = blk.preds[0];
                res.ssa.block_in_regs[b] = res.ssa.block_out_regs[p];
                res.ssa.block_out_regs[b] = res.ssa.block_out_regs[p];
                res.ssa.block_in_token[b] = res.ssa.block_out_token[p];
                res.ssa.block_out_token[b] = res.ssa.block_out_token[p];
            } else {
                for (uint32_t r = 0; r < 64; ++r) {
                    SsaNode phi{};
                    phi.op = tables::SsaOp::Phi;
                    phi.type = ValType::U32;
                    phi.aux_index = r;
                    phi.aux_bool = false;
                    SsaId pid = ctx.sb.emit(b, std::move(phi));
                    res.ssa.block_in_regs[b][r] = pid;
                    res.ssa.block_out_regs[b][r] = pid;
                }
                SsaNode tphi{};
                tphi.op = tables::SsaOp::Phi;
                tphi.type = ValType::Token;
                tphi.aux_bool = true;
                SsaId tpid = ctx.sb.emit(b, std::move(tphi));
                res.ssa.block_in_token[b] = tpid;
                res.ssa.block_out_token[b] = tpid;
            }
        }

        // Corner case: Vertex Position+Varying EBIN fusion reloads r59..r61 at Varying EBIN entry
        if (has_secondary && b == sec_offset) {
            SsaId vi = ctx.emit_builtin_input(b, spirv::BuiltIn::VertexIndex, ValType::U32, "gl_VertexIndex");
            SsaId ii = ctx.emit_builtin_input(b, spirv::BuiltIn::InstanceIndex, ValType::U32, "gl_InstanceIndex");
            ctx.sb.set_reg(b, 59, vi);
            ctx.sb.set_reg(b, 60, vi);
            ctx.sb.set_reg(b, 61, ii);
        }

        for (const auto& ins : all_insts[b]) {
            lower_instruction(ctx, b, ins);
            if (!res.error_message.empty()) {
                return res;
            }
        }
    }

    ctx.sb.seal_all_phis();
    res.ok = true;
    return res;
}

} // namespace valhall::compiler
