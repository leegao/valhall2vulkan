#include "valhall_pass1_handlers.h"
#include <algorithm>
#include <cstring>

namespace valhall::compiler {

using tables::SsaOp;

SsaId Pass1Context::emit_op(uint32_t blk, SsaOp op, ValType vt, std::vector<SsaId> args,
                              tables::EffectMask eff, bool barrier, bool side_eff) {
    SsaNode n{};
    n.op = op;
    n.type = vt;
    n.args = std::move(args);
    n.effects = eff;
    n.is_barrier_or_cf = barrier;
    n.is_side_effect = side_eff;
    return sb.emit(blk, std::move(n));
}

SsaId Pass1Context::emit_load_fau(uint32_t blk, uint32_t word_idx) {
    SsaNode n{};
    n.op = SsaOp::LoadFau;
    n.type = ValType::U32;
    n.imm_u32 = word_idx & 63u;
    n.effects.flags |= EffectSet::kFlagUsesSideband;
    res.uses_sideband = true;
    return sb.emit(blk, std::move(n));
}

SsaId Pass1Context::emit_load_sideband_u64(uint32_t blk, uint32_t member_idx) {
    SsaNode n{};
    n.op = SsaOp::LoadSidebandU64;
    n.type = ValType::U64;
    n.imm_u32 = member_idx;
    n.effects.flags |= EffectSet::kFlagUsesSideband;
    n.effects.cap_bits |= EffectSet::kCapInt64;
    res.uses_sideband = true;
    return sb.emit(blk, std::move(n));
}

SsaId Pass1Context::emit_builtin_input(uint32_t blk, spirv::BuiltIn bi, ValType vt, const char* name, bool flat) {
    SsaNode n{};
    n.op = SsaOp::LoadBuiltinInput;
    n.type = vt;
    n.imm_u32 = static_cast<uint32_t>(bi);
    n.aux_bool = flat;
    n.aux_str = name;
    return sb.emit(blk, std::move(n));
}

void Pass1Context::emit_split64(uint32_t blk, SsaId val_u64, SsaId* lo, SsaId* hi) {
    SsaNode sp{};
    sp.op = SsaOp::Split64;
    sp.type = ValType::TupleU32x2;
    sp.args = { val_u64 };
    sp.effects.cap_bits |= EffectSet::kCapInt64;
    SsaId tup = sb.emit(blk, std::move(sp));

    SsaNode e0{}; e0.op = SsaOp::TupleExtract; e0.type = ValType::U32; e0.imm_u32 = 0; e0.args = { tup };
    SsaNode e1{}; e1.op = SsaOp::TupleExtract; e1.type = ValType::U32; e1.imm_u32 = 1; e1.args = { tup };
    *lo = sb.emit(blk, std::move(e0));
    *hi = sb.emit(blk, std::move(e1));
}

void Pass1Context::emit_stage_preamble(uint32_t blk, const std::vector<DecodedValhallInst>& insts) {
    if (opts.stage == ShaderStage::Pilot) {
        SsaId fau_va = emit_load_sideband_u64(blk, 3);
        SsaId srt_va = emit_load_sideband_u64(blk, 2);
        SsaId f0 = 0, f1 = 0, s0 = 0, s1 = 0;
        emit_split64(blk, fau_va, &f0, &f1);
        emit_split64(blk, srt_va, &s0, &s1);
        sb.set_reg(blk, 0, f0);
        sb.set_reg(blk, 1, f1);
        sb.set_reg(blk, 2, s0);
        sb.set_reg(blk, 3, s1);
    } else if (opts.stage == ShaderStage::Compute) {
        bool reads_wg_regs = false;
        for (const auto& ins : insts) {
            for (const auto& s : ins.srcs) {
                if (s.kind == SourceKind::Register && s.reg_num >= 55 && s.reg_num <= 57) {
                    reads_wg_regs = true;
                }
            }
        }
        SsaId wgid = emit_builtin_input(blk, spirv::BuiltIn::WorkgroupId, ValType::V3U32, "gl_WorkGroupID");
        for (uint32_t c = 0; c < 3; ++c) {
            SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::U32; ex.imm_u32 = c; ex.args = { wgid };
            sb.set_reg(blk, 55 + c, sb.emit(blk, std::move(ex)));
        }
        auto bi = reads_wg_regs ? spirv::BuiltIn::LocalInvocationId : spirv::BuiltIn::GlobalInvocationId;
        const char* bname = reads_wg_regs ? "gl_LocalInvocationID" : "gl_GlobalInvocationID";
        SsaId tid = emit_builtin_input(blk, bi, ValType::V3U32, bname);
        for (uint32_t c = 0; c < 3; ++c) {
            SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::U32; ex.imm_u32 = c; ex.args = { tid };
            sb.set_reg(blk, 60 + c, sb.emit(blk, std::move(ex)));
        }
    } else if (opts.stage == ShaderStage::Vertex) {
        SsaId vi = emit_builtin_input(blk, spirv::BuiltIn::VertexIndex, ValType::U32, "gl_VertexIndex");
        SsaId ii = emit_builtin_input(blk, spirv::BuiltIn::InstanceIndex, ValType::U32, "gl_InstanceIndex");
        sb.set_reg(blk, 59, vi);
        sb.set_reg(blk, 60, vi);
        sb.set_reg(blk, 61, ii);
    } else if (opts.stage == ShaderStage::Fragment) {
        SsaId fc = emit_builtin_input(blk, spirv::BuiltIn::FragCoord, ValType::V4F32, "gl_FragCoord");
        SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = 2; ex.args = { fc };
        SsaId fz = sb.emit(blk, std::move(ex));
        sb.set_reg(blk, 59, sb.emit_bitcast(blk, ValType::U32, fz));
        sb.set_reg(blk, 60, sb.emit_const_u32(blk, 1u));
    }
}

SsaId Pass1Context::eval_float_cmp(uint32_t blk, const std::string& cmpf, SsaId a, SsaId b) {
    SsaOp op = SsaOp::FCmpOrdEq;
    if (cmpf == "gt") op = SsaOp::FCmpOrdGt;
    else if (cmpf == "ge") op = SsaOp::FCmpOrdGe;
    else if (cmpf == "ne") op = SsaOp::FCmpUnordNe;
    else if (cmpf == "lt") op = SsaOp::FCmpOrdLt;
    else if (cmpf == "le") op = SsaOp::FCmpOrdLe;
    else if (cmpf == "gtlt") op = SsaOp::FCmpOrdNe;
    else if (cmpf == "total") op = SsaOp::FCmpTotal;
    return emit_op(blk, op, ValType::Bool, { a, b });
}

SsaId Pass1Context::eval_int_cmp(uint32_t blk, const std::string& cmpf, SsaId a, SsaId b, bool is_signed) {
    SsaOp op = SsaOp::ICmpEq;
    if (cmpf == "ne") op = SsaOp::ICmpNe;
    else if (is_signed) {
        a = sb.emit_bitcast(blk, ValType::S32, a);
        b = sb.emit_bitcast(blk, ValType::S32, b);
        if (cmpf == "gt") op = SsaOp::ICmpSGt;
        else if (cmpf == "ge") op = SsaOp::ICmpSGe;
        else if (cmpf == "lt") op = SsaOp::ICmpSLt;
        else if (cmpf == "le") op = SsaOp::ICmpSLe;
    } else {
        if (cmpf == "gt") op = SsaOp::ICmpUGt;
        else if (cmpf == "ge") op = SsaOp::ICmpUGe;
        else if (cmpf == "lt") op = SsaOp::ICmpULt;
        else if (cmpf == "le") op = SsaOp::ICmpULe;
    }
    return emit_op(blk, op, ValType::Bool, { a, b });
}

SsaId Pass1Context::format_cmp_result(uint32_t blk, const DecodedValhallInst& ins, SsaId cond_bool) {
    std::string rtype = "m1";
    auto it = ins.modifier_strs.find("result_type");
    if (it != ins.modifier_strs.end()) rtype = it->second;
    uint32_t tval = 0xFFFFFFFFu;
    if (rtype == "i1") tval = 1u;
    else if (rtype == "f1") tval = 0x3F800000u;
    return emit_op(blk, SsaOp::Select, ValType::U32,
                   { cond_bool, sb.emit_const_u32(blk, tval), sb.emit_const_u32(blk, 0u) });
}

SsaId Pass1Context::eval_branch_cond(uint32_t blk, const DecodedValhallInst& ins) {
    if (ins.srcs.empty()) return sb.emit_const_bool(blk, true);
    SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
    SsaId cond = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { s0, sb.emit_const_u32(blk, 0u) });

    std::string comb = "none";
    auto it_c = ins.modifier_strs.find("combine");
    if (it_c != ins.modifier_strs.end()) comb = it_c->second;
    if (ins.srcs.size() >= 2 && comb != "none") {
        SsaId s1 = fetch_src_int32(sb, blk, ins, 1, false);
        if (comb == "eq") {
            cond = emit_op(blk, SsaOp::ICmpEq, ValType::Bool, { s0, s1 });
        } else {
            SsaId c1 = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { s1, sb.emit_const_u32(blk, 0u) });
            cond = emit_op(blk, (comb == "and") ? SsaOp::LogicalAnd : SsaOp::LogicalOr, ValType::Bool, { cond, c1 });
        }
    }
    auto it_not = ins.fields.find("not");
    if (it_not != ins.fields.end() && it_not->second != 0) {
        cond = emit_op(blk, SsaOp::LogicalNot, ValType::Bool, { cond });
    }
    return cond;
}

// ============================================================================
// Remaining Complex Host-Side Handlers (@HANDLER(fn))
// ============================================================================

void handle_int_idpadd(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    std::string var = variant_to_string(ins.canonical_variant);
    bool s0_signed = (var.find("s") == 0);
    bool s1_signed = s0_signed;
    if (ins.fields.count("unsigned0") && ins.fields.at("unsigned0") != 0) s0_signed = false;
    if (ins.fields.count("unsigned1") && ins.fields.at("unsigned1") != 0) s1_signed = false;
    SsaId a4 = fetch_src_v4i8(ctx.sb, blk, ins, 0, s0_signed);
    SsaId b4 = fetch_src_v4i8(ctx.sb, blk, ins, 1, s1_signed);
    if (s0_signed != s1_signed) {
        if (!s0_signed) a4 = ctx.sb.emit_bitcast(blk, ValType::V4S32, a4);
        if (!s1_signed) b4 = ctx.sb.emit_bitcast(blk, ValType::V4S32, b4);
    }
    bool acc_s = (s0_signed || s1_signed);
    SsaId acc = (ins.srcs.size() >= 3) ? fetch_src_int32(ctx.sb, blk, ins, 2, acc_s) : ctx.sb.emit_const_u32(blk, 0u);
    if (acc_s) acc = ctx.sb.emit_bitcast(blk, ValType::S32, acc);
    ValType elem_t = acc_s ? ValType::S32 : ValType::U32;
    for (uint32_t i = 0; i < 4; ++i) {
        SsaNode ea{}; ea.op = SsaOp::VecExtract; ea.type = elem_t; ea.imm_u32 = i; ea.args = { a4 };
        SsaNode eb{}; eb.op = SsaOp::VecExtract; eb.type = elem_t; eb.imm_u32 = i; eb.args = { b4 };
        SsaId prod = ctx.emit_op(blk, SsaOp::IMul, elem_t, { ctx.sb.emit(blk, std::move(ea)), ctx.sb.emit(blk, std::move(eb)) });
        acc = ctx.emit_op(blk, SsaOp::IAdd, elem_t, { acc, prod });
    }
    store_dest_u32(ctx.sb, blk, ins, acc_s ? ctx.sb.emit_bitcast(blk, ValType::U32, acc) : acc);
}

static void handle_mem_load_impl(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins,
                                 const tables::LoweringRuleSpec* rule, bool is_pka) {
    SsaId base_addr = fetch_src_u64(ctx.sb, blk, ins, 0, false);
    SsaId byte_addr = base_addr;
    if (ins.has_imm && ins.imm_signed != 0) {
        byte_addr = ctx.emit_op(blk, SsaOp::IAdd64, ValType::U64,
                                { base_addr, ctx.sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_signed)) },
                                rule->effects);
    }
    uint8_t sr = ins.sr_write.present ? ins.sr_write.base_reg : 0;
    uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 1;
    std::string mwidth = "i32";
    auto it_mw = ins.modifier_strs.find("memory_width");
    if (it_mw != ins.modifier_strs.end()) mwidth = it_mw->second;

    uint32_t elem_bytes = 4;
    if (mwidth == "i8") elem_bytes = 1;
    else if (mwidth == "i16") elem_bytes = 2;

    SsaId tok = ctx.sb.current_token(blk);
    if (elem_bytes < 4) {
        SsaNode ld{};
        ld.op = SsaOp::PsbLoad; ld.type = ValType::TupleValToken;
        ld.imm_u32 = elem_bytes; ld.aux_bool = is_pka;
        ld.args = { byte_addr, tok }; ld.effects = rule->effects;
        ld.is_barrier_or_cf = true; ld.is_side_effect = true;
        SsaId tup = ctx.sb.emit(blk, std::move(ld));
        SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
        SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
        ctx.sb.set_reg(blk, sr, ctx.sb.emit(blk, std::move(ev)));
        ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(et)));
        return;
    }
    for (uint8_t w = 0; w < cnt; ++w) {
        SsaId w_addr = (w == 0) ? byte_addr
                                : ctx.emit_op(blk, SsaOp::IAdd64, ValType::U64,
                                              { byte_addr, ctx.sb.emit_const_u64(blk, uint64_t(w) * 4ULL) },
                                              rule->effects);
        SsaNode ld{};
        ld.op = SsaOp::PsbLoad; ld.type = ValType::TupleValToken;
        ld.imm_u32 = 4u; ld.aux_bool = is_pka;
        ld.args = { w_addr, tok }; ld.effects = rule->effects;
        ld.is_barrier_or_cf = true; ld.is_side_effect = true;
        SsaId tup = ctx.sb.emit(blk, std::move(ld));
        SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
        SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
        ctx.sb.set_reg(blk, sr + w, ctx.sb.emit(blk, std::move(ev)));
        tok = ctx.sb.emit(blk, std::move(et));
    }
    ctx.sb.set_token(blk, tok);
}

void handle_mem_load(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    handle_mem_load_impl(ctx, blk, ins, rule, false);
}

void handle_mem_load_pka(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    handle_mem_load_impl(ctx, blk, ins, rule, true);
}

void handle_mem_store(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    std::string access = "none";
    auto it_acc = ins.modifier_strs.find("access");
    if (it_acc != ins.modifier_strs.end()) access = it_acc->second;
    uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
    uint8_t cnt = ins.sr_read.present ? std::max<uint8_t>(1, ins.sr_read.count) : 1;

    if (ctx.opts.stage == ShaderStage::Vertex && access == "istream") {
        SsaId r8_f  = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, sr + 0));
        SsaId r9_f  = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, sr + 1));
        SsaId r10_f = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, sr + 2));
        SsaId r11_f = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, sr + 3));

        SsaId sx = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.emit_load_fau(blk, 0));
        SsaId sy = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.emit_load_fau(blk, 1));
        SsaId sz = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.emit_load_fau(blk, 2));
        SsaId oz = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.emit_load_fau(blk, 3));
        SsaId ox = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.emit_load_fau(blk, 4));
        SsaId oy = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.emit_load_fau(blk, 5));

        SsaId has_vp = ctx.emit_op(blk, SsaOp::FCmpOrdNe, ValType::Bool, { sx, ctx.sb.emit_const_f32_bits(blk, 0u) });
        SsaId wc = ctx.emit_op(blk, SsaOp::FDiv, ValType::F32, { ctx.sb.emit_const_f32_bits(blk, 0x3f800000u), r11_f });
        SsaId dx = ctx.emit_op(blk, SsaOp::FSub, ValType::F32, { r8_f, ox });
        SsaId dy = ctx.emit_op(blk, SsaOp::FSub, ValType::F32, { r9_f, oy });
        SsaId dz = ctx.emit_op(blk, SsaOp::FSub, ValType::F32, { r10_f, oz });
        SsaId xc = ctx.emit_op(blk, SsaOp::FMul, ValType::F32, { ctx.emit_op(blk, SsaOp::FDiv, ValType::F32, { dx, sx }), wc });
        SsaId yc = ctx.emit_op(blk, SsaOp::FMul, ValType::F32, { ctx.emit_op(blk, SsaOp::FDiv, ValType::F32, { dy, sy }), wc });
        SsaId zc = ctx.emit_op(blk, SsaOp::FMul, ValType::F32, { ctx.emit_op(blk, SsaOp::FDiv, ValType::F32, { dz, sz }), wc });

        SsaId fx = ctx.emit_op(blk, SsaOp::Select, ValType::F32, { has_vp, xc, r8_f });
        SsaId fy = ctx.emit_op(blk, SsaOp::Select, ValType::F32, { has_vp, yc, r9_f });
        SsaId fz = ctx.emit_op(blk, SsaOp::Select, ValType::F32, { has_vp, zc, r10_f });
        SsaId fw = ctx.emit_op(blk, SsaOp::Select, ValType::F32, { has_vp, wc, r11_f });
        SsaId pos_vec = ctx.emit_op(blk, SsaOp::VecConstruct4, ValType::V4F32, { fx, fy, fz, fw });

        SsaNode st{};
        st.op = SsaOp::StoreBuiltinOutput; st.type = ValType::Token;
        st.imm_u32 = static_cast<uint32_t>(spirv::BuiltIn::Position);
        st.aux_str = "gl_Position";
        st.args = { pos_vec, ctx.sb.current_token(blk) };
        st.is_barrier_or_cf = true; st.is_side_effect = true;
        ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(st)));
        return;
    }
    if (ctx.opts.stage == ShaderStage::Vertex && access == "estream") {
        uint32_t var_loc = static_cast<uint32_t>(std::max<int64_t>(0, ins.imm_signed) / 16);
        std::vector<SsaId> comps(4);
        for (uint8_t c = 0; c < 4; ++c) {
            if (c < cnt) comps[c] = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, sr + c));
            else comps[c] = ctx.sb.emit_const_f32_bits(blk, (c == 3) ? 0x3f800000u : 0u);
        }
        SsaId vec = ctx.emit_op(blk, SsaOp::VecConstruct4, ValType::V4F32, std::move(comps));
        SsaNode st{};
        st.op = SsaOp::StoreVertVarying; st.type = ValType::Token;
        st.imm_u32 = var_loc;
        st.args = { vec, ctx.sb.current_token(blk) };
        st.is_barrier_or_cf = true; st.is_side_effect = true;
        ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(st)));
        return;
    }

    SsaId base_addr = fetch_src_u64(ctx.sb, blk, ins, 0, false);
    SsaId byte_addr = base_addr;
    if (ins.has_imm && ins.imm_signed != 0) {
        byte_addr = ctx.emit_op(blk, SsaOp::IAdd64, ValType::U64,
                                { base_addr, ctx.sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_signed)) },
                                rule->effects);
    }
    std::string mwidth = "i32";
    auto it_mw = ins.modifier_strs.find("memory_width");
    if (it_mw != ins.modifier_strs.end()) mwidth = it_mw->second;
    uint32_t elem_bytes = (mwidth == "i8") ? 1u : (mwidth == "i16") ? 2u : 4u;

    SsaId tok = ctx.sb.current_token(blk);
    if (elem_bytes < 4) {
        SsaNode st{};
        st.op = SsaOp::PsbStore; st.type = ValType::Token; st.imm_u32 = elem_bytes;
        st.args = { byte_addr, ctx.sb.current_reg(blk, sr), tok };
        st.effects = rule->effects; st.is_barrier_or_cf = true; st.is_side_effect = true;
        ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(st)));
        return;
    }
    for (uint8_t w = 0; w < cnt; ++w) {
        SsaId w_addr = (w == 0) ? byte_addr
                                : ctx.emit_op(blk, SsaOp::IAdd64, ValType::U64,
                                              { byte_addr, ctx.sb.emit_const_u64(blk, uint64_t(w) * 4ULL) },
                                              rule->effects);
        SsaNode st{};
        st.op = SsaOp::PsbStore; st.type = ValType::Token; st.imm_u32 = 4u;
        st.args = { w_addr, ctx.sb.current_reg(blk, sr + w), tok };
        st.effects = rule->effects; st.is_barrier_or_cf = true; st.is_side_effect = true;
        tok = ctx.sb.emit(blk, std::move(st));
    }
    ctx.sb.set_token(blk, tok);
}

static SsaId compute_atomic_addr(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    SsaId base_addr = fetch_src_u64(ctx.sb, blk, ins, 0, false);
    if (ins.has_imm && ins.imm_signed != 0) {
        return ctx.emit_op(blk, SsaOp::IAdd64, ValType::U64,
                           { base_addr, ctx.sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_signed)) },
                           rule->effects);
    }
    return base_addr;
}

void handle_mem_atom1(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    SsaId byte_addr = compute_atomic_addr(ctx, blk, ins, rule);
    SsaId tok = ctx.sb.current_token(blk);
    std::string aop = "ainc";
    auto it = ins.modifier_strs.find("atom1_opc");
    if (it != ins.modifier_strs.end()) aop = it->second;
    SsaId delta = ctx.sb.emit_const_u32(blk, (aop == "adec") ? 0xFFFFFFFFu : 1u);
    SsaNode at{};
    at.op = SsaOp::AtomicRmw; at.type = ValType::TupleValToken;
    at.aux_str = "aadd"; at.aux_bool = false;
    at.args = { byte_addr, delta, tok };
    at.effects = rule->effects; at.is_barrier_or_cf = true; at.is_side_effect = true;
    SsaId tup = ctx.sb.emit(blk, std::move(at));
    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
    SsaId old_v = ctx.sb.emit(blk, std::move(ev));
    ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(et)));
    if (ins.sr_write.present) ctx.sb.set_reg(blk, ins.sr_write.base_reg, old_v);
}

void handle_mem_axchg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    SsaId byte_addr = compute_atomic_addr(ctx, blk, ins, rule);
    uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
    SsaId tok = ctx.sb.current_token(blk);
    SsaNode at{};
    at.op = SsaOp::AtomicExchange; at.type = ValType::TupleValToken;
    at.args = { byte_addr, ctx.sb.current_reg(blk, sr), tok };
    at.effects = rule->effects; at.is_barrier_or_cf = true; at.is_side_effect = true;
    SsaId tup = ctx.sb.emit(blk, std::move(at));
    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
    SsaId old_v = ctx.sb.emit(blk, std::move(ev));
    ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(et)));
    if (ins.sr_write.present) ctx.sb.set_reg(blk, ins.sr_write.base_reg, old_v);
}

void handle_mem_acmpxchg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    SsaId byte_addr = compute_atomic_addr(ctx, blk, ins, rule);
    uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
    SsaId tok = ctx.sb.current_token(blk);
    SsaId cmp_v = ctx.sb.current_reg(blk, sr);
    SsaId xchg_v = ctx.sb.current_reg(blk, sr + 1);
    SsaNode at{};
    at.op = SsaOp::AtomicCmpXchg; at.type = ValType::TupleValToken;
    at.args = { byte_addr, xchg_v, cmp_v, tok };
    at.effects = rule->effects; at.is_barrier_or_cf = true; at.is_side_effect = true;
    SsaId tup = ctx.sb.emit(blk, std::move(at));
    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
    SsaId old_v = ctx.sb.emit(blk, std::move(ev));
    ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(et)));
    if (ins.sr_write.present) ctx.sb.set_reg(blk, ins.sr_write.base_reg, old_v);
}

void handle_mem_atomic(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    SsaId byte_addr = compute_atomic_addr(ctx, blk, ins, rule);
    std::string var = variant_to_string(ins.canonical_variant);
    bool is_64 = (var.find("64") != std::string::npos) || ins.dest_is_64bit ||
                 (ins.sr_write.present && ins.sr_write.count >= 2) ||
                 (ins.sr_read.present && ins.sr_read.count >= 2 && ins.canonical_mnemonic != Mnemonic::ACMPXCHG);
    uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
    SsaId tok = ctx.sb.current_token(blk);
    SsaId arg = is_64
        ? ctx.emit_op(blk, SsaOp::Pack64, ValType::U64, { ctx.sb.current_reg(blk, sr), ctx.sb.current_reg(blk, sr + 1) }, rule->effects)
        : ctx.sb.current_reg(blk, sr);
    std::string aop = "aadd";
    auto it = ins.modifier_strs.find("atom_opc");
    if (it != ins.modifier_strs.end()) aop = it->second;
    SsaNode at{};
    at.op = SsaOp::AtomicRmw; at.type = ValType::TupleValToken;
    at.aux_str = aop; at.aux_bool = is_64;
    at.args = { byte_addr, arg, tok };
    at.effects = rule->effects; at.is_barrier_or_cf = true; at.is_side_effect = true;
    SsaId tup = ctx.sb.emit(blk, std::move(at));
    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = is_64 ? ValType::U64 : ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
    SsaId old_v = ctx.sb.emit(blk, std::move(ev));
    ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(et)));
    if (ins.sr_write.present) {
        if (is_64) {
            SsaId lo = 0, hi = 0;
            ctx.emit_split64(blk, old_v, &lo, &hi);
            ctx.sb.set_reg(blk, ins.sr_write.base_reg, lo);
            ctx.sb.set_reg(blk, ins.sr_write.base_reg + 1, hi);
        } else {
            ctx.sb.set_reg(blk, ins.sr_write.base_reg, old_v);
        }
    }
}

void handle_vert_attr(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    uint32_t attr_idx = 0;
    auto it = ins.fields.find("attribute_index");
    if (it != ins.fields.end()) attr_idx = it->second;
    SsaNode ld{};
    ld.op = SsaOp::LoadVertAttr; ld.type = ValType::V4F32; ld.imm_u32 = attr_idx;
    ld.is_barrier_or_cf = true;
    SsaId vec = ctx.sb.emit(blk, std::move(ld));
    uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
    uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
    for (uint8_t c = 0; c < cnt && c < 4; ++c) {
        SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = { vec };
        ctx.sb.set_reg(blk, base_r + c, ctx.sb.emit_bitcast(blk, ValType::U32, ctx.sb.emit(blk, std::move(ex))));
    }
}

void handle_frag_var(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    bool is_flat = (ins.canonical_mnemonic == Mnemonic::LD_VAR_FLAT_IMM ||
                    ins.canonical_mnemonic == Mnemonic::LD_VAR_FLAT);
    uint32_t var_loc = 0;
    auto it = ins.fields.find("index");
    if (it != ins.fields.end()) var_loc = it->second / 16u;
    SsaNode ld{};
    ld.op = SsaOp::LoadFragVarying; ld.type = ValType::V4F32;
    ld.imm_u32 = var_loc; ld.aux_bool = is_flat;
    ld.is_barrier_or_cf = true;
    SsaId vec = ctx.sb.emit(blk, std::move(ld));
    uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
    uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
    for (uint8_t c = 0; c < cnt && c < 4; ++c) {
        SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = { vec };
        ctx.sb.set_reg(blk, base_r + c, ctx.sb.emit_bitcast(blk, ValType::U32, ctx.sb.emit(blk, std::move(ex))));
    }
}

void handle_tex_var_sample(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    SsaNode ld{};
    ld.op = SsaOp::LoadFragVarying; ld.type = ValType::V4F32; ld.imm_u32 = 0; ld.aux_bool = false;
    ld.is_barrier_or_cf = true;
    SsaId uv4 = ctx.sb.emit(blk, std::move(ld));
    SsaNode exu{}; exu.op = SsaOp::VecExtract; exu.type = ValType::F32; exu.imm_u32 = 0; exu.args = { uv4 };
    SsaNode exv{}; exv.op = SsaOp::VecExtract; exv.type = ValType::F32; exv.imm_u32 = 1; exv.args = { uv4 };
    SsaId uv = ctx.emit_op(blk, SsaOp::VecConstruct2, ValType::V2F32,
                           { ctx.sb.emit(blk, std::move(exu)), ctx.sb.emit(blk, std::move(exv)) });

    SsaNode ts{};
    ts.op = SsaOp::TexSample; ts.type = ValType::TupleValToken;
    ts.imm_u32 = 0; ts.aux_bool = (ctx.opts.stage == ShaderStage::Fragment); ts.aux_bool2 = false;
    ts.args = { uv, ctx.sb.current_token(blk) };
    ts.is_barrier_or_cf = true; ts.is_side_effect = true;
    SsaId tup = ctx.sb.emit(blk, std::move(ts));
    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::V4F32; ev.imm_u32 = 0; ev.args = { tup };
    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
    SsaId sampled = ctx.sb.emit(blk, std::move(ev));
    ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(et)));

    uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
    uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
    for (uint8_t c = 0; c < cnt && c < 4; ++c) {
        SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = { sampled };
        ctx.sb.set_reg(blk, base_r + c, ctx.sb.emit_bitcast(blk, ValType::U32, ctx.sb.emit(blk, std::move(ex))));
    }
}

void handle_tex_sample(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    uint8_t coord_r = ins.sr_read.present ? ins.sr_read.base_reg : 0;
    SsaId u = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, coord_r));
    SsaId v = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, coord_r + 1));
    SsaId uv = ctx.emit_op(blk, SsaOp::VecConstruct2, ValType::V2F32, { u, v });
    bool is_gather = (ins.canonical_mnemonic == Mnemonic::TEX_GATHER || ins.mnemonic == Mnemonic::TEX_GATHER);
    uint32_t gcomp = 0;
    if (is_gather) {
        auto it_gc = ins.fields.find("gather_component");
        if (it_gc != ins.fields.end()) gcomp = it_gc->second & 3u;
    }
    SsaNode ts{};
    ts.op = SsaOp::TexSample; ts.type = ValType::TupleValToken;
    ts.imm_u32 = 0; ts.imm_u32_2 = gcomp;
    ts.aux_bool = (ctx.opts.stage == ShaderStage::Fragment && !is_gather);
    ts.aux_bool2 = is_gather;
    ts.args = { uv, ctx.sb.current_token(blk) };
    ts.is_barrier_or_cf = true; ts.is_side_effect = true;
    SsaId tup = ctx.sb.emit(blk, std::move(ts));
    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::V4F32; ev.imm_u32 = 0; ev.args = { tup };
    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
    SsaId sampled = ctx.sb.emit(blk, std::move(ev));
    ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(et)));

    uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
    uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
    for (uint8_t c = 0; c < cnt && c < 4; ++c) {
        SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = { sampled };
        ctx.sb.set_reg(blk, base_r + c, ctx.sb.emit_bitcast(blk, ValType::U32, ctx.sb.emit(blk, std::move(ex))));
    }
}

void handle_frag_atest(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    SsaId cov = (ins.srcs.size() >= 1) ? fetch_src_int32(ctx.sb, blk, ins, 0, false) : ctx.sb.emit_const_u32(blk, 1u);
    if (ins.has_dest) store_dest_u32(ctx.sb, blk, ins, cov);
    if (ctx.opts.stage == ShaderStage::Fragment && !ins.srcs.empty()) {
        SsaId is_zero = ctx.emit_op(blk, SsaOp::ICmpEq, ValType::Bool, { cov, ctx.sb.emit_const_u32(blk, 0u) });
        SsaNode k{};
        k.op = SsaOp::Kill; k.type = ValType::Void; k.aux_bool = true;
        k.args = { is_zero, ctx.sb.current_token(blk) };
        k.is_barrier_or_cf = true; k.is_side_effect = true;
        ctx.sb.emit(blk, std::move(k));
    }
}

void handle_frag_zs_emit(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
    SsaId z_f32 = ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, sr));
    SsaNode st{};
    st.op = SsaOp::StoreBuiltinOutput; st.type = ValType::Token;
    st.imm_u32 = static_cast<uint32_t>(spirv::BuiltIn::FragDepth);
    st.aux_str = "gl_FragDepth";
    st.args = { z_f32, ctx.sb.current_token(blk) };
    st.effects = rule->effects; st.is_barrier_or_cf = true; st.is_side_effect = true;
    ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(st)));
}

void handle_frag_blend(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
    uint32_t rt_loc = 0;
    if (!ins.srcs.empty() && ins.srcs[0].symbolic_name.find("blend_descriptor_") == 0) {
        char c = ins.srcs[0].symbolic_name.back();
        if (c >= '0' && c <= '7') rt_loc = static_cast<uint32_t>(c - '0');
    }
    SsaId col = ctx.emit_op(blk, SsaOp::VecConstruct4, ValType::V4F32, {
        ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, sr + 0)),
        ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, sr + 1)),
        ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, sr + 2)),
        ctx.sb.emit_bitcast(blk, ValType::F32, ctx.sb.current_reg(blk, sr + 3)),
    });
    SsaNode st{};
    st.op = SsaOp::StoreFragColor; st.type = ValType::Token; st.imm_u32 = rt_loc;
    st.args = { col, ctx.sb.current_token(blk) };
    st.is_barrier_or_cf = true; st.is_side_effect = true;
    ctx.sb.set_token(blk, ctx.sb.emit(blk, std::move(st)));
}

} // namespace valhall::compiler
