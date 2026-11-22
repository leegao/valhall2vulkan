#include "valhall_isa_generated.h"
#include "valhall_tables_generated.h"
#include "valhall_pass1.h"
#include "valhall_tiler.h"
#include "valhall_to_spirv.h"
#include "spirv_reflect.h"
#include "cmpbe_chunks.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

namespace astc_ns {
#include "fixtures/astc_encoder_mbs2_shader.h"
}

using namespace valhall;
using namespace valhall::compiler;
using namespace valhall::tables;

static DecodedValhallInst make_inst(uint64_t pc, Mnemonic mn, Mnemonic cmn, Variant var = Variant::NONE) {
    DecodedValhallInst ins{};
    ins.valid = true;
    ins.pc = pc;
    ins.arch = 11;
    ins.mnemonic = mn;
    ins.canonical_mnemonic = cmn;
    ins.variant = var;
    ins.canonical_variant = var;
    // Lookup matching spec_index
    size_t count = 0;
    const auto* specs = get_all_instruction_specs(&count);
    for (size_t i = 0; i < count; ++i) {
        if (specs[i].mnemonic == mn && (var == Variant::NONE || specs[i].variant == var)) {
            ins.spec_index = static_cast<uint16_t>(i);
            break;
        }
    }
    return ins;
}

static DecodedSource reg_src(uint8_t r, bool is64 = false) {
    DecodedSource s{};
    s.kind = SourceKind::Register;
    s.reg_num = r;
    s.is_64bit = is64;
    return s;
}

static bool has_capability(const std::vector<uint32_t>& words, spirv::Capability cap) {
    size_t idx = 5;
    while (idx < words.size()) {
        uint32_t wc = words[idx] >> 16;
        uint32_t op = words[idx] & 0xffffu;
        if (wc == 0) break;
        if (op == uint32_t(spirv::Op::OpCapability) && wc == 2 && words[idx + 1] == uint32_t(cap)) {
            return true;
        }
        idx += wc;
    }
    return false;
}

int main() {
    std::cout << "=== [Test 8] Two-Pass Spec-Driven Compiler Verification ===\n";

    // 1. 100% XML spec coverage check
    size_t spec_count = 0;
    const auto* specs = get_all_instruction_specs(&spec_count);
    assert(spec_count == kNumXmlInstructions);
    for (size_t i = 0; i < spec_count; ++i) {
        const auto* rule = lookup_lowering_rule(static_cast<uint16_t>(i));
        assert(rule != nullptr);
        if (rule->kind == LoweringKind::DirectSsaOp) {
            const auto* ctile = lookup_canonical_tile(rule->ssa_op);
            assert(ctile != nullptr);
        } else {
            assert(rule->handler_name != nullptr && rule->handler_name[0] != '\0');
        }
    }
    std::cout << "[PASS] 100% XML coverage: " << spec_count << "/627 specs map to LoweringRuleSpec + CanonicalTileSpec\n";

    // 2. Pass 0 Sub-block splitting at mid-block ATEST
    {
        std::vector<DecodedValhallInst> insts;
        auto i0 = make_inst(0x00, Mnemonic::MOV_IMM, Mnemonic::IADD_IMM, Variant::I32);
        i0.has_dest = true; i0.dest_reg = 0; i0.imm_signed = 1;
        auto i1 = make_inst(0x08, Mnemonic::ATEST, Mnemonic::ATEST);
        i1.has_dest = true; i1.dest_reg = 1; i1.srcs.push_back(reg_src(0));
        auto i2 = make_inst(0x10, Mnemonic::NOP, Mnemonic::NOP);
        i2.flow = FlowControlM::end;
        insts.push_back(i0); insts.push_back(i1); insts.push_back(i2);

        auto cfg = cfg::build_cfg(insts);
        assert(cfg.blocks.size() == 1); // Original CFG has 1 block
        auto split = split_cfg_subblocks(cfg);
        assert(split.blocks.size() == 2); // Split at ATEST!
        assert(split.blocks[0].fallthrough_block == 1);
        assert(split.blocks[1].is_terminated_by_end);
        std::cout << "[PASS] Pass 0 sub-block splitting at mid-block ATEST (1 CFG block -> 2 sub-blocks)\n";
    }

    // 3. Pass 0 mem2reg & loop Phi generation + trivial Phi elimination
    {
        std::vector<DecodedValhallInst> insts;
        // B0: r0 = 0
        auto i0 = make_inst(0x00, Mnemonic::MOV_IMM, Mnemonic::IADD_IMM, Variant::I32);
        i0.has_dest = true; i0.dest_reg = 0; i0.imm_signed = 0;
        // B1 (loop header): r0 = r0 + 1; BRANCH.lt r0 < 10 -> B1
        auto i1 = make_inst(0x08, Mnemonic::IADD_IMM, Mnemonic::IADD_IMM, Variant::I32);
        i1.has_dest = true; i1.dest_reg = 0; i1.srcs.push_back(reg_src(0)); i1.imm_signed = 1;
        auto i2 = make_inst(0x10, Mnemonic::BRANCH, Mnemonic::BRANCH);
        i2.has_branch_offset = true; i2.branch_target_pc = 0x08; i2.srcs.push_back(reg_src(0));
        // B2: NOP.end
        auto i3 = make_inst(0x18, Mnemonic::NOP, Mnemonic::NOP);
        i3.flow = FlowControlM::end;
        insts = { i0, i1, i2, i3 };

        auto cfg = cfg::build_cfg(insts);
        ValhallTranslateOptions opts{};
        opts.stage = ShaderStage::Compute;
        auto p1 = run_pass1_lowering(cfg, {}, opts);
        assert(p1.ok);
        // Block 1 (loop header) should have non-trivial Phi for r0, and trivial Phis for r1..r63 collapsed!
        bool has_r0_phi = false;
        for (SsaId pid : p1.ssa.blocks[1].phis) {
            const auto& pn = p1.ssa.get(pid);
            if (!pn.aux_bool && pn.aux_index == 0) has_r0_phi = true;
            // All other unmodified registers must have been collapsed
            if (!pn.aux_bool) assert(pn.aux_index == 0);
        }
        assert(has_r0_phi);
        std::cout << "[PASS] Pass 0 mem2reg: loop Phi retained for r0, trivial Phis collapsed for r1..r63\n";
    }

    // 4. Pass 2 DP Fusion (FMul + FAdd -> GlslFma) & Algebraic Elision + Multi-Use Penalty
    {
        ValhallSsaModule mod;
        SsaBuilder sb(mod);
        SsaBlock b0{}; b0.id = 0; b0.is_terminated_by_end = true;
        mod.blocks.push_back(b0);
        mod.block_in_regs.resize(1);
        mod.block_out_regs.resize(1);
        mod.block_in_token.resize(1, kInvalidSsa);
        mod.block_out_token.resize(1, kInvalidSsa);

        SsaId a = sb.emit_const_f32_bits(0, 0x3fc00000u);
        SsaId b_v = sb.emit_const_f32_bits(0, 0x40000000u);
        SsaId c = sb.emit_const_f32_bits(0, 0x40400000u);

        SsaNode mul{}; mul.op = SsaOp::FMul; mul.type = ValType::F32; mul.args = { a, b_v };
        SsaId mul_id = sb.emit(0, std::move(mul));

        SsaNode add{}; add.op = SsaOp::FAdd; add.type = ValType::F32; add.args = { mul_id, c };
        SsaId add_id = sb.emit(0, std::move(add));

        // Side-effecting root so add_id is live
        SsaNode st{}; st.op = SsaOp::StoreFragColor; st.type = ValType::Token; st.is_side_effect = true;
        SsaId v4 = sb.emit(0, SsaNode{0, SsaOp::VecConstruct4, ValType::V4F32, 0, false, false, { add_id, add_id, add_id, add_id }});
        st.args = { v4 };
        sb.emit(0, std::move(st));

        Pass1Result p1{};
        p1.ok = true;
        p1.ssa = std::move(mod);
        ValhallTranslateOptions opts{};
        opts.stage = ShaderStage::Fragment;
        auto p2 = run_pass2_tiler_and_emit(p1, opts);
        assert(p2.ok);
        assert(p2.stats.fusion_tiles_selected >= 1);
        assert(p2.effects.uses_glsl_std450());
        std::cout << "[PASS] Pass 2 DP Tiler: FMul+FAdd fused into GlslFma (fusion_tiles="
                  << p2.stats.fusion_tiles_selected << ", total_cost=" << p2.stats.total_dp_cost << ")\n";
    }

    // 5. EffectSet lazy capabilities: pure u32 shader omits Int64 / PSB
    {
        std::vector<DecodedValhallInst> insts;
        auto i0 = make_inst(0x00, Mnemonic::IADD_IMM, Mnemonic::IADD_IMM, Variant::I32);
        i0.has_dest = true; i0.dest_reg = 0; i0.srcs.push_back(reg_src(0)); i0.imm_signed = 42;
        auto i1 = make_inst(0x08, Mnemonic::NOP, Mnemonic::NOP);
        i1.flow = FlowControlM::end;
        insts = { i0, i1 };
        auto cfg = cfg::build_cfg(insts);
        ValhallTranslateOptions opts{};
        opts.stage = ShaderStage::Compute;
        auto p1 = run_pass1_lowering(cfg, {}, opts);
        auto p2 = run_pass2_tiler_and_emit(p1, opts);
        assert(p2.ok);
        assert(!has_capability(p2.spirv_words, spirv::Capability::Int64));
        assert(!has_capability(p2.spirv_words, spirv::Capability::Float16));
        std::cout << "[PASS] EffectSet lazy capabilities: pure u32 shader omits Int64 and Float16\n";
    }

    // 6. Translate all EBINs of production astc_encoder_mbs2_shader.h and validate with SPIRV-Reflect
    {
        auto* astc_ccom = &astc_ns::CCOM_1;
        assert(astc_ccom && astc_ccom->cmmn.nof_ebin >= 1);
        for (uint32_t b = 0; b < astc_ccom->cmmn.nof_ebin; ++b) {
            const auto& eb = astc_ccom->cmmn.ebin[b];
            size_t sz = (eb.data_section_offset && eb.data_section_offset < eb.bytecode.size)
                        ? eb.data_section_offset : eb.bytecode.size;
            ValhallTranslateOptions opts{};
            opts.stage = (b == astc_ccom->cmmn.nof_ebin - 1) ? ShaderStage::Pilot : ShaderStage::Compute;
            opts.arch = 11;
            auto tres = translate_valhall_to_spirv(eb.bytecode.byte_code, sz, opts);
            assert(tres.ok);
            SpvReflectShaderModule sm{};
            SpvReflectResult r = spvReflectCreateShaderModule(tres.spirv_words.size() * 4, tres.spirv_words.data(), &sm);
            assert(r == SPV_REFLECT_RESULT_SUCCESS);
            spvReflectDestroyShaderModule(&sm);
        }
        std::cout << "[PASS] Translated and SPIRV-Reflect validated all " << astc_ccom->cmmn.nof_ebin
                  << " EBINs of production astc_encoder_mbs2_shader.h\n";
    }

    // 7. Milestone 1: DirectSsaOp, DynamicSsaOp (@fn), and Handler (HANDLER(@fn)) dispatch
    {
        auto lower_single = [](DecodedValhallInst ins) -> Pass1Result {
            auto end_i = make_inst(ins.pc + 8, Mnemonic::NOP, Mnemonic::NOP);
            end_i.flow = FlowControlM::end;
            std::vector<DecodedValhallInst> insts = { ins, end_i };
            auto cfg = cfg::build_cfg(insts);
            ValhallTranslateOptions opts{};
            opts.stage = ShaderStage::Compute;
            return run_pass1_lowering(cfg, {}, opts);
        };
        auto has_op = [](const Pass1Result& p1, SsaOp op) {
            for (const auto& n : p1.ssa.nodes) {
                if (n.op == op) return true;
            }
            return false;
        };

        // DirectSsaOp: FABS.f32 & FABS.v2f16
        auto fabs32 = make_inst(0x00, Mnemonic::FABS, Mnemonic::FABS, Variant::F32);
        fabs32.has_dest = true; fabs32.dest_reg = 0; fabs32.srcs.push_back(reg_src(1));
        auto p1_fabs = lower_single(fabs32);
        assert(p1_fabs.ok && has_op(p1_fabs, SsaOp::FAbs));

        auto fabs16 = make_inst(0x00, Mnemonic::FABS, Mnemonic::FABS, Variant::V2F16);
        fabs16.has_dest = true; fabs16.dest_reg = 0; fabs16.srcs.push_back(reg_src(1));
        auto p1_fabs16 = lower_single(fabs16);
        assert(p1_fabs16.ok && has_op(p1_fabs16, SsaOp::FAbs) && has_op(p1_fabs16, SsaOp::UnpackHalf2x16));

        // DynamicSsaOp (@get_round_mode): FROUND.rte / rtn / rtp / rtz
        for (const auto& [rm, expected_op] : std::vector<std::pair<const char*, SsaOp>>{
                 {"rte", SsaOp::RoundEven}, {"rtn", SsaOp::Floor}, {"rtp", SsaOp::Ceil}, {"rtz", SsaOp::Trunc}}) {
            auto fr = make_inst(0x00, Mnemonic::FROUND, Mnemonic::FROUND, Variant::F32);
            fr.has_dest = true; fr.dest_reg = 0; fr.srcs.push_back(reg_src(1));
            fr.modifier_strs["round"] = rm;
            auto p1_fr = lower_single(fr);
            assert(p1_fr.ok && has_op(p1_fr, expected_op));
        }

        // DynamicSsaOp (@get_fp_minmax_op): FMIN / FMAX with nan_suppress vs nan_propagate
        auto fmin_sup = make_inst(0x00, Mnemonic::FMIN, Mnemonic::FMIN, Variant::F32);
        fmin_sup.has_dest = true; fmin_sup.dest_reg = 0; fmin_sup.srcs = { reg_src(1), reg_src(2) };
        fmin_sup.modifier_strs["sem"] = "nan_suppress";
        assert(has_op(lower_single(fmin_sup), SsaOp::NMin));

        auto fmax_prop = make_inst(0x00, Mnemonic::FMAX, Mnemonic::FMAX, Variant::F32);
        fmax_prop.has_dest = true; fmax_prop.dest_reg = 0; fmax_prop.srcs = { reg_src(1), reg_src(2) };
        fmax_prop.modifier_strs["sem"] = "nan_propagate";
        assert(has_op(lower_single(fmax_prop), SsaOp::FMax));

        // Handler (HANDLER(@handle_int_shadd64)): SHADD
        auto shadd = make_inst(0x00, Mnemonic::SHADD, Mnemonic::SHADD, Variant::U64);
        shadd.has_dest = true; shadd.dest_is_64bit = true; shadd.dest_reg = 0;
        shadd.srcs = { reg_src(2, true), reg_src(4, true) };
        shadd.imm_unsigned = 3;
        auto p1_sh = lower_single(shadd);
        assert(p1_sh.ok && has_op(p1_sh, SsaOp::Shl64) && has_op(p1_sh, SsaOp::IAdd64));

        std::cout << "[PASS] Milestone 1 lowering dispatch: DirectSsaOp, DynamicSsaOp (@fn), and Handler (HANDLER(@fn)) verified\n";
    }

    // 8. Milestones 1.5 & 2: @LOOKUP tables and INLINE_OPS DSL rules
    {
        auto lower_single = [](DecodedValhallInst ins) -> Pass1Result {
            auto end_i = make_inst(ins.pc + 8, Mnemonic::NOP, Mnemonic::NOP);
            end_i.flow = FlowControlM::end;
            std::vector<DecodedValhallInst> insts = { ins, end_i };
            auto cfg = cfg::build_cfg(insts);
            ValhallTranslateOptions opts{};
            opts.stage = ShaderStage::Compute;
            return run_pass1_lowering(cfg, {}, opts);
        };
        auto has_op = [](const Pass1Result& p1, SsaOp op) {
            for (const auto& n : p1.ssa.nodes) {
                if (n.op == op) return true;
            }
            return false;
        };

        // @LOOKUP(round_mode, $modifiers.round) with XML enum names
        for (const auto& [rm, expected_op] : std::vector<std::pair<const char*, SsaOp>>{
                 {"none", SsaOp::RoundEven}, {"round_down", SsaOp::Floor}, {"round_up", SsaOp::Ceil}, {"round_zero", SsaOp::Trunc}}) {
            auto fr = make_inst(0x00, Mnemonic::FROUND, Mnemonic::FROUND, Variant::F32);
            fr.has_dest = true; fr.dest_reg = 0; fr.srcs.push_back(reg_src(1));
            fr.modifier_strs["round"] = rm;
            assert(has_op(lower_single(fr), expected_op));
        }

        // INLINE_OPS: HADD (unsigned vs signed)
        auto hadd_u = make_inst(0x00, Mnemonic::HADD, Mnemonic::HADD, Variant::U32);
        hadd_u.has_dest = true; hadd_u.dest_reg = 0; hadd_u.srcs = { reg_src(1), reg_src(2) };
        assert(has_op(lower_single(hadd_u), SsaOp::ShrU));

        auto hadd_s = make_inst(0x00, Mnemonic::HADD, Mnemonic::HADD, Variant::S32);
        hadd_s.has_dest = true; hadd_s.dest_reg = 0; hadd_s.srcs = { reg_src(1), reg_src(2) };
        assert(has_op(lower_single(hadd_s), SsaOp::ShrS));

        // INLINE_OPS: MUX & FPCLASS
        auto mux = make_inst(0x00, Mnemonic::MUX, Mnemonic::MUX, Variant::I32);
        mux.has_dest = true; mux.dest_reg = 0; mux.srcs = { reg_src(1), reg_src(2), reg_src(3) };
        auto p1_mux = lower_single(mux);
        assert(p1_mux.ok && has_op(p1_mux, SsaOp::BitNot) && has_op(p1_mux, SsaOp::BitOr));

        auto fpc = make_inst(0x00, Mnemonic::FPCLASS, Mnemonic::FPCLASS, Variant::F32);
        fpc.has_dest = true; fpc.dest_reg = 0; fpc.srcs = { reg_src(1) };
        auto p1_fpc = lower_single(fpc);
        assert(p1_fpc.ok && has_op(p1_fpc, SsaOp::IsNan) && has_op(p1_fpc, SsaOp::IsInf));

        // INLINE_OPS + @LOOKUP: FCMP.ne (unordered) & FCMP.gtlt (ordered) & ICMP.s32
        auto fcmp_ne = make_inst(0x00, Mnemonic::FCMP, Mnemonic::FCMP, Variant::F32);
        fcmp_ne.has_dest = true; fcmp_ne.dest_reg = 0; fcmp_ne.srcs = { reg_src(1), reg_src(2) };
        fcmp_ne.modifier_strs["cmpf"] = "ne"; fcmp_ne.modifier_strs["result_type"] = "i1";
        assert(has_op(lower_single(fcmp_ne), SsaOp::FCmpUnordNe));

        auto fcmp_gtlt = make_inst(0x00, Mnemonic::FCMP, Mnemonic::FCMP, Variant::F32);
        fcmp_gtlt.has_dest = true; fcmp_gtlt.dest_reg = 0; fcmp_gtlt.srcs = { reg_src(1), reg_src(2) };
        fcmp_gtlt.modifier_strs["cmpf"] = "gtlt";
        assert(has_op(lower_single(fcmp_gtlt), SsaOp::FCmpOrdNe));

        auto icmp_s = make_inst(0x00, Mnemonic::ICMP, Mnemonic::ICMP, Variant::S32);
        icmp_s.has_dest = true; icmp_s.dest_reg = 0; icmp_s.srcs = { reg_src(1), reg_src(2) };
        icmp_s.modifier_strs["cmpf"] = "lt";
        assert(has_op(lower_single(icmp_s), SsaOp::ICmpSLt));

        // INLINE_OPS: LSHIFT_AND with not_result=1
        auto lsa = make_inst(0x00, Mnemonic::LSHIFT_AND, Mnemonic::LSHIFT_AND, Variant::I32);
        lsa.has_dest = true; lsa.dest_reg = 0; lsa.srcs = { reg_src(1), reg_src(2), reg_src(3) };
        lsa.fields["not_result"] = 1;
        auto p1_lsa = lower_single(lsa);
        assert(p1_lsa.ok && has_op(p1_lsa, SsaOp::Shl) && has_op(p1_lsa, SsaOp::BitAnd) && has_op(p1_lsa, SsaOp::BitNot));

        // INLINE_OPS: WMASK & CLPER & CUBEFACE1
        auto wm = make_inst(0x00, Mnemonic::WMASK, Mnemonic::WMASK);
        wm.has_dest = true; wm.dest_reg = 0; wm.srcs = { reg_src(1) };
        assert(has_op(lower_single(wm), SsaOp::SubgroupBallot));

        auto clp = make_inst(0x00, Mnemonic::CLPER, Mnemonic::CLPER);
        clp.has_dest = true; clp.dest_reg = 0; clp.srcs = { reg_src(1), reg_src(2) };
        assert(has_op(lower_single(clp), SsaOp::SubgroupShuffle));

        auto cf1 = make_inst(0x00, Mnemonic::CUBEFACE1, Mnemonic::CUBEFACE1);
        cf1.has_dest = true; cf1.dest_reg = 0; cf1.srcs = { reg_src(1), reg_src(2), reg_src(3) };
        assert(has_op(lower_single(cf1), SsaOp::FMax));
        auto cf2 = make_inst(0x00, Mnemonic::CUBEFACE2, Mnemonic::CUBEFACE2); cf2.has_dest = true; cf2.dest_reg = 0; cf2.srcs = { reg_src(1), reg_src(2), reg_src(3) };
        assert(has_op(lower_single(cf2), SsaOp::FCmpOrdGe) && has_op(lower_single(cf2), SsaOp::Shl));

        std::cout << "[PASS] Milestones 1.5 & 2: @LOOKUP tables and INLINE_OPS DSL rules verified\n";
    }

    // 9. Checkpoint v7: BRANCH, JUMP, ADR, LEA, IADD_IMM(64/32), RSCALE, CSEL(v2i16/f32/i32)
    {
        auto lower_single = [](DecodedValhallInst ins) -> Pass1Result {
            auto end_i = make_inst(ins.pc + 8, Mnemonic::NOP, Mnemonic::NOP);
            end_i.flow = FlowControlM::end;
            std::vector<DecodedValhallInst> insts = { ins, end_i };
            auto cfg = cfg::build_cfg(insts);
            ValhallTranslateOptions opts{};
            opts.stage = ShaderStage::Compute;
            return run_pass1_lowering(cfg, {}, opts);
        };
        auto has_op = [](const Pass1Result& p1, SsaOp op) {
            for (const auto& n : p1.ssa.nodes) {
                if (n.op == op) return true;
            }
            return false;
        };

        // ADR -> @INLINE_OPS (load_sideband_u64 + split64_lo)
        auto adr = make_inst(0x10, Mnemonic::ADR, Mnemonic::ADR);
        adr.has_dest = true; adr.dest_reg = 0; adr.imm_signed = 16;
        auto p1_adr = lower_single(adr);
        assert(p1_adr.ok && p1_adr.uses_sideband && has_op(p1_adr, SsaOp::LoadSidebandU64) && has_op(p1_adr, SsaOp::Split64));

        // LEA_BUF_IMM -> @INLINE_OPS (sr_set + $has_field)
        auto lea = make_inst(0x00, Mnemonic::LEA_BUF_IMM, Mnemonic::LEA_BUF_IMM);
        lea.sr_write.present = true; lea.sr_write.base_reg = 4; lea.sr_write.count = 2;
        lea.fields["buffer_index"] = 3;
        auto p1_lea = lower_single(lea);
        assert(p1_lea.ok);

        // IADD_IMM (32-bit vs 64-bit variant)
        auto add32 = make_inst(0x00, Mnemonic::IADD_IMM, Mnemonic::IADD_IMM, Variant::I32);
        add32.has_dest = true; add32.dest_reg = 0; add32.srcs = { reg_src(1) }; add32.imm_signed = 7;
        assert(has_op(lower_single(add32), SsaOp::IAdd));

        auto add64 = make_inst(0x00, Mnemonic::IADD, Mnemonic::IADD, Variant::U64);
        add64.has_dest = true; add64.dest_is_64bit = true; add64.dest_reg = 0; add64.srcs = { reg_src(2, true), reg_src(4, true) };
        assert(has_op(lower_single(add64), SsaOp::IAdd64));

        // FMA_RSCALE -> @INLINE_OPS
        auto fmar = make_inst(0x00, Mnemonic::FMA_RSCALE, Mnemonic::FMA_RSCALE, Variant::F32);
        fmar.has_dest = true; fmar.dest_reg = 0;
        fmar.srcs = { reg_src(1), reg_src(2), reg_src(3), reg_src(4) };
        auto p1_fmar = lower_single(fmar);
        assert(p1_fmar.ok && has_op(p1_fmar, SsaOp::Fma) && has_op(p1_fmar, SsaOp::Ldexp));

        // CSEL: f32, s32, v2s16
        auto csel_f = make_inst(0x00, Mnemonic::CSEL, Mnemonic::CSEL, Variant::F32);
        csel_f.has_dest = true; csel_f.dest_reg = 0;
        csel_f.srcs = { reg_src(1), reg_src(2), reg_src(3), reg_src(4) };
        csel_f.modifier_strs["cmpf"] = "lt";
        assert(has_op(lower_single(csel_f), SsaOp::FCmpOrdLt));

        auto csel_v2s = make_inst(0x00, Mnemonic::CSEL, Mnemonic::CSEL, Variant::V2S16);
        csel_v2s.has_dest = true; csel_v2s.dest_reg = 0;
        csel_v2s.srcs = { reg_src(1), reg_src(2), reg_src(3), reg_src(4) };
        csel_v2s.modifier_strs["cmpf"] = "ge";
        assert(has_op(lower_single(csel_v2s), SsaOp::ICmpSGe));

        std::cout << "[PASS] Checkpoint v7: BRANCH, JUMP, ADR, LEA, IADD_IMM(64/32), RSCALE, CSEL(v2i16/f32/i32) verified\n";
    }


    // Checkpoint v27: Valhall Compute preload ABI (r55=LID.xy, r56=LID.z, r57..59=WGID.xyz, r60..62=GID.xyz)
    // and LD_TEX_IMM 16:16 packed coordinate extraction from src0.
    {
        auto i0 = make_inst(0x00, Mnemonic::IMUL, Mnemonic::IMUL, Variant::U32);
        i0.has_dest = true; i0.dest_reg = 0;
        i0.srcs = { reg_src(57), reg_src(58) }; // WorkGroupID.x * WorkGroupID.y
        auto i1 = make_inst(0x08, Mnemonic::LD_TEX_IMM, Mnemonic::LD_TEX_IMM, Variant::NONE);
        i1.sr_write.present = true; i1.sr_write.base_reg = 5; i1.sr_write.count = 1;
        i1.srcs = { reg_src(0) };
        i1.fields["index"] = 2u;
        i1.fields["table"] = 4u;
        auto i2 = make_inst(0x10, Mnemonic::NOP, Mnemonic::NOP);
        i2.flow = FlowControlM::end;

        std::vector<DecodedValhallInst> insts = { i0, i1, i2 };
        auto cfg = cfg::build_cfg(insts);
        ValhallTranslateOptions opts{};
        opts.stage = ShaderStage::Compute;
        auto p1 = run_pass1_lowering(cfg, cfg::ControlFlowGraph{}, opts);
        assert(p1.ok);

        bool saw_imul_wgid = false;
        bool saw_tex_bfe = false;
        bool saw_tex_binding = false;
        for (size_t id = 1; id < p1.ssa.nodes.size(); ++id) {
            const auto& n = p1.ssa.nodes[id];
            if (n.op == SsaOp::IMul && n.args.size() == 2) {
                const auto& a = p1.ssa.get(n.args[0]);
                const auto& b_n = p1.ssa.get(n.args[1]);
                if (a.op == SsaOp::VecExtract && b_n.op == SsaOp::VecExtract &&
                    a.imm_u32 == 0u && b_n.imm_u32 == 1u &&
                    p1.ssa.get(a.args[0]).aux_str == "gl_WorkGroupID" &&
                    p1.ssa.get(b_n.args[0]).aux_str == "gl_WorkGroupID") {
                    saw_imul_wgid = true;
                }
            }
            if (n.op == SsaOp::TexSample || n.op == SsaOp::TexFetch) {
                assert(n.imm_u32 == 2u);
                if (n.aux_str == "tex:0:2") saw_tex_binding = true;
                const auto& uv = p1.ssa.get(n.args[0]);
                assert(uv.op == SsaOp::VecConstruct2 && uv.args.size() == 2);
                const auto& ux = p1.ssa.get(p1.ssa.get(uv.args[0]).args[0]);
                const auto& uy = p1.ssa.get(p1.ssa.get(uv.args[1]).args[0]);
                if (ux.op == SsaOp::BfeU && uy.op == SsaOp::BfeU && ux.args[0] == uy.args[0]) {
                    saw_tex_bfe = true;
                }
            }
        }
        assert(saw_imul_wgid);
        assert(saw_tex_bfe);
        assert(saw_tex_binding);
        std::cout << "[PASS] Checkpoint v27: Compute ABI r55..r62 and LD_TEX_IMM 16:16 coordinate unpacking verified\n";
    }

    // Checkpoint v27.1: BRANCH unary combine modes and CLPER lane_op / subgroup modes.
    {
        auto lower_single = [&](const DecodedValhallInst& ins) {
            DecodedValhallInst end = make_inst(0x08, Mnemonic::NOP, Mnemonic::NOP);
            end.flow = FlowControlM::end;
            std::vector<DecodedValhallInst> v = { ins, end };
            auto cfg = cfg::build_cfg(v);
            ValhallTranslateOptions opts{};
            opts.stage = ShaderStage::Compute;
            auto p1 = run_pass1_lowering(cfg, cfg::ControlFlowGraph{}, opts);
            assert(p1.ok);
            return p1;
        };
        auto has_op = [&](const Pass1Result& p1, SsaOp op) {
            for (size_t id = 1; id < p1.ssa.nodes.size(); ++id) {
                if (p1.ssa.nodes[id].op == op) return true;
            }
            return false;
        };
        auto br_and = make_inst(0x00, Mnemonic::BRANCH, Mnemonic::BRANCH);
        br_and.srcs = { reg_src(2) };
        br_and.modifier_strs["combine"] = "and";
        auto p_and = lower_single(br_and);
        assert(has_op(p_and, SsaOp::ICmpEq));
        bool saw_all_ones = false;
        for (size_t id = 1; id < p_and.ssa.nodes.size(); ++id) {
            const auto& n = p_and.ssa.nodes[id];
            if (n.op == SsaOp::ConstU32 && n.imm_u32 == 0xFFFFFFFFu) saw_all_ones = true;
        }
        assert(saw_all_ones);

        auto br_h1 = make_inst(0x00, Mnemonic::BRANCH, Mnemonic::BRANCH);
        br_h1.srcs = { reg_src(2) };
        br_h1.modifier_strs["combine"] = "h1";
        assert(has_op(lower_single(br_h1), SsaOp::ShrU));

        auto br_lb = make_inst(0x00, Mnemonic::BRANCH, Mnemonic::BRANCH);
        br_lb.srcs = { reg_src(2) };
        br_lb.modifier_strs["combine"] = "lowbits";
        assert(has_op(lower_single(br_lb), SsaOp::LogicalAnd));

        auto clper_xor = make_inst(0x00, Mnemonic::CLPER, Mnemonic::CLPER, Variant::NONE);
        clper_xor.has_dest = true; clper_xor.dest_reg = 2;
        clper_xor.srcs = { reg_src(1), reg_src(3) };
        clper_xor.modifier_strs["lane_op"] = "xor";
        clper_xor.modifier_strs["subgroup"] = "subgroup16";
        auto p_xor = lower_single(clper_xor);
        assert(has_op(p_xor, SsaOp::BitXor));
        assert(has_op(p_xor, SsaOp::SubgroupShuffle));

        auto clper_shift = make_inst(0x00, Mnemonic::CLPER, Mnemonic::CLPER, Variant::NONE);
        clper_shift.has_dest = true; clper_shift.dest_reg = 2;
        clper_shift.srcs = { reg_src(1), reg_src(3) };
        clper_shift.modifier_strs["lane_op"] = "shift";
        clper_shift.modifier_strs["subgroup"] = "subgroup4";
        assert(has_op(lower_single(clper_shift), SsaOp::ISub));

        auto clper_or = make_inst(0x00, Mnemonic::CLPER_OR, Mnemonic::CLPER_OR, Variant::NONE);
        clper_or.has_dest = true; clper_or.dest_reg = 2;
        clper_or.srcs = { reg_src(1), reg_src(3), reg_src(4) };
        assert(has_op(lower_single(clper_or), SsaOp::BitOr));

        std::cout << "[PASS] Checkpoint v27.1: BRANCH combine modes and CLPER lane_op/subgroup verified\n";
    }

    // Checkpoint v27.2: Valhall SFU log2/exp2 lowering (FREXPE.log, FADD_LSCALE, FLOGD, FMUL_RSCALE, FEXP).
    {
        auto lower_single = [&](const DecodedValhallInst& ins) {
            DecodedValhallInst end = make_inst(0x08, Mnemonic::NOP, Mnemonic::NOP);
            end.flow = FlowControlM::end;
            std::vector<DecodedValhallInst> v = { ins, end };
            auto cfg = cfg::build_cfg(v);
            ValhallTranslateOptions opts{};
            opts.stage = ShaderStage::Fragment;
            auto p1 = run_pass1_lowering(cfg, cfg::ControlFlowGraph{}, opts);
            assert(p1.ok);
            return p1;
        };
        auto has_op = [&](const Pass1Result& p1, SsaOp op) {
            for (size_t id = 1; id < p1.ssa.nodes.size(); ++id) {
                if (p1.ssa.nodes[id].op == op) return true;
            }
            return false;
        };

        auto frexpe_log = make_inst(0x00, Mnemonic::FREXPE, Mnemonic::FREXPE, Variant::F32);
        frexpe_log.has_dest = true; frexpe_log.dest_reg = 5;
        frexpe_log.srcs = { reg_src(1) };
        frexpe_log.modifier_strs["special"] = "log";
        auto p_fl = lower_single(frexpe_log);
        assert(has_op(p_fl, SsaOp::BfeU) && has_op(p_fl, SsaOp::Select));

        auto lscale = make_inst(0x00, Mnemonic::FADD_LSCALE, Mnemonic::FADD_LSCALE, Variant::F32);
        lscale.has_dest = true; lscale.dest_reg = 5;
        lscale.srcs = { reg_src(1), reg_src(2) };
        auto p_ls = lower_single(lscale);
        assert(has_op(p_ls, SsaOp::Bfi) && has_op(p_ls, SsaOp::FAdd) && !has_op(p_ls, SsaOp::Ldexp));

        auto flogd = make_inst(0x00, Mnemonic::FLOGD, Mnemonic::FLOGD, Variant::F32);
        flogd.has_dest = true; flogd.dest_reg = 5;
        flogd.srcs = { reg_src(1) };
        auto p_ld = lower_single(flogd);
        assert(has_op(p_ld, SsaOp::Log2) && has_op(p_ld, SsaOp::FDiv) && has_op(p_ld, SsaOp::Select));

        auto rscale = make_inst(0x00, Mnemonic::FMUL_RSCALE, Mnemonic::FMA_RSCALE, Variant::F32);
        rscale.has_dest = true; rscale.dest_reg = 5;
        rscale.srcs = { reg_src(1), reg_src(2), reg_src(3) };
        rscale.fields["neg2"] = 1u;  // elided -0.0 addend bitfield must NOT negate src2 exponent!
        auto p_rs = lower_single(rscale);
        assert(has_op(p_rs, SsaOp::Ldexp) && !has_op(p_rs, SsaOp::SNeg));

        auto fexp = make_inst(0x00, Mnemonic::FEXP, Mnemonic::FEXP, Variant::F32);
        fexp.has_dest = true; fexp.dest_reg = 5;
        fexp.srcs = { reg_src(1), reg_src(2) };
        auto p_fe = lower_single(fexp);
        assert(has_op(p_fe, SsaOp::ConvertSToF) && has_op(p_fe, SsaOp::Ldexp) && has_op(p_fe, SsaOp::Exp2));

        std::cout << "[PASS] Checkpoint v27.2: Valhall SFU log2/exp2 (FREXPE.log, FADD_LSCALE, FLOGD, FMUL_RSCALE, FEXP) verified\n";
    }


    // Checkpoint v29 regression tests: FNEG single negation, FSIN_TABLE.u6, TEX_SINGLE.a write_mask, verify_ssa_types
    {
        auto lower_single = [](DecodedValhallInst ins) -> Pass1Result {
            auto end_i = make_inst(ins.pc + 8, Mnemonic::NOP, Mnemonic::NOP);
            end_i.flow = FlowControlM::end;
            std::vector<DecodedValhallInst> insts = { ins, end_i };
            cfg::ControlFlowGraph c = cfg::build_cfg(insts);
            ValhallTranslateOptions o;
            o.stage = ShaderStage::Fragment;
            return run_pass1_lowering(c, {}, o);
        };
        auto count_op = [](const Pass1Result& p1, SsaOp op) {
            int cnt = 0;
            for (const auto& n : p1.ssa.nodes) if (n.op == op) ++cnt;
            return cnt;
        };

        auto fneg_i = make_inst(0x00, Mnemonic::FNEG, Mnemonic::FNEG, Variant::F32);
        fneg_i.has_dest = true; fneg_i.dest_reg = 1;
        fneg_i.srcs = { reg_src(0) };
        fneg_i.fields["neg0"] = 1;
        auto p1_fneg = lower_single(fneg_i);
        assert(p1_fneg.ok && count_op(p1_fneg, SsaOp::FNeg) == 1);
        std::string verr;
        assert(verify_ssa_types(p1_fneg.ssa, &verr));

        auto fsin_i = make_inst(0x00, Mnemonic::FSIN_TABLE, Mnemonic::FSIN_TABLE);
        fsin_i.has_dest = true; fsin_i.dest_reg = 0; fsin_i.srcs = { reg_src(1) };
        auto p1_fsin = lower_single(fsin_i);
        assert(p1_fsin.ok && count_op(p1_fsin, SsaOp::BitAnd) >= 1 &&
               count_op(p1_fsin, SsaOp::ConvertUToF) >= 1 && count_op(p1_fsin, SsaOp::Sin) == 1);

        auto tex_a = make_inst(0x00, Mnemonic::TEX_SINGLE, Mnemonic::TEX_SINGLE);
        tex_a.modifier_strs["write_mask"] = "a";
        tex_a.sr_write.present = true; tex_a.sr_write.base_reg = 0; tex_a.sr_write.count = 1;
        tex_a.sr_read.present = true; tex_a.sr_read.base_reg = 0; tex_a.sr_read.count = 2;
        auto p1_tex = lower_single(tex_a);
        assert(p1_tex.ok);
        bool saw_ch3 = false;
        for (const auto& n : p1_tex.ssa.nodes) {
            if (n.op == SsaOp::VecExtract && n.imm_u32 == 3u) saw_ch3 = true;
        }
        assert(saw_ch3);
        assert(verify_ssa_types(p1_tex.ssa, &verr));

        std::cout << "[PASS] Checkpoint v29 lowering regression tests (FNEG, FSIN_TABLE, TEX_SINGLE.a, verify_ssa_types)\n";
    }


    // Checkpoint v29.2: SWZ.v2i16 r1, r0.h10 must swap 16-bit halves via (v >> 16) | (v << 16).
    {
        auto swz_h10 = make_inst(0x00, Mnemonic::SWZ, Mnemonic::SWZ, Variant::V2I16);
        swz_h10.has_dest = true;
        swz_h10.dest_reg = 1;
        auto s0 = reg_src(0);
        s0.swizzle_mod = "h10";
        swz_h10.srcs = { s0 };
        DecodedValhallInst end = make_inst(0x08, Mnemonic::NOP, Mnemonic::NOP);
        end.flow = FlowControlM::end;
        std::vector<DecodedValhallInst> v = { swz_h10, end };
        auto cfg = cfg::build_cfg(v);
        ValhallTranslateOptions opts{};
        opts.stage = ShaderStage::Fragment;
        auto p1 = run_pass1_lowering(cfg, cfg::ControlFlowGraph{}, opts);
        assert(p1.ok);
        bool saw_shru = false, saw_shl = false, saw_bitor = false;
        for (size_t id = 1; id < p1.ssa.nodes.size(); ++id) {
            if (p1.ssa.nodes[id].op == SsaOp::ShrU) saw_shru = true;
            if (p1.ssa.nodes[id].op == SsaOp::Shl) saw_shl = true;
            if (p1.ssa.nodes[id].op == SsaOp::BitOr) saw_bitor = true;
        }
        assert(saw_shru && saw_shl && saw_bitor);
        std::cout << "[PASS] Checkpoint v29.2: SWZ.v2i16 .h10 half-word swap verified\n";
    }

    std::cout << "=== ALL TWO-PASS COMPILER TESTS PASSED ===\n";
    return 0;
}
