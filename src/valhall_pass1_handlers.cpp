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
// Uniform Host ABI Functions (@fn(...) in @INLINE_OPS)
// ============================================================================

namespace abi {

static inline bool is_v2f16_helper(const DecodedValhallInst& ins) {
    if (ins.canonical_variant == Variant::V2F16 || ins.variant == Variant::V2F16) return true;
    return std::strcmp(variant_to_string(ins.canonical_variant), "v2f16") == 0;
}

bool has_src0(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.srcs.size() >= 1; }
bool has_src1(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.srcs.size() >= 2; }
bool has_src2(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.srcs.size() >= 3; }
bool has_src3(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.srcs.size() >= 4; }
bool has_dest(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.has_dest; }
bool is_v2f16(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return is_v2f16_helper(ins); }
bool is_signed(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    std::string var = variant_to_string(ins.canonical_variant);
    return (var.find("s") == 0 || var.find("v2s") == 0 || var.find("v4s") == 0);
}
bool is_64bit(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.dest_is_64bit; }
bool dest_is_64bit(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.dest_is_64bit; }
bool src0_swz_h1(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return get_src_swz(ins, 0) == "h1"; }
bool is_frag_stage(Pass1Context& ctx, uint32_t, const DecodedValhallInst&, const tables::LoweringRuleSpec*) { return ctx.opts.stage == ShaderStage::Fragment; }
bool is_vert_stage(Pass1Context& ctx, uint32_t, const DecodedValhallInst&, const tables::LoweringRuleSpec*) { return ctx.opts.stage == ShaderStage::Vertex; }
bool has_imm_nonzero(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.has_imm && ins.imm_signed != 0; }
bool sr_write_present(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.sr_write.present; }
bool is_64_atom(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    std::string var = variant_to_string(ins.canonical_variant);
    return (var.find("64") != std::string::npos) || ins.dest_is_64bit ||
           (ins.sr_write.present && ins.sr_write.count >= 2) ||
           (ins.sr_read.present && ins.sr_read.count >= 2 && ins.canonical_mnemonic != Mnemonic::ACMPXCHG);
}
bool is_gather(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ins.canonical_mnemonic == Mnemonic::TEX_GATHER || ins.mnemonic == Mnemonic::TEX_GATHER;
}
bool is_flat_var(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ins.canonical_mnemonic == Mnemonic::LD_VAR_FLAT_IMM || ins.canonical_mnemonic == Mnemonic::LD_VAR_FLAT;
}
bool has_not_result(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    auto it = ins.fields.find("not_result"); return it != ins.fields.end() && it->second != 0;
}
bool has_not(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    auto it = ins.fields.find("not"); return it != ins.fields.end() && it->second != 0;
}

uint32_t sr_write_count(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ins.sr_write.present ? std::max<uint32_t>(1u, ins.sr_write.count) : 1u;
}
uint32_t sr_write_count_max4(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ins.sr_write.present ? std::min<uint32_t>(4u, std::max<uint32_t>(1u, ins.sr_write.count)) : 4u;
}
uint32_t sr_read_count(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ins.sr_read.present ? std::max<uint32_t>(1u, ins.sr_read.count) : 1u;
}
uint32_t src0_swz_h1_u32(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return (get_src_swz(ins, 0) == "h1") ? 1u : 0u;
}
SsaId branch_target_pc_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    uint64_t tgt = static_cast<uint64_t>(ins.has_branch_offset ? ins.branch_target_pc : (int64_t(ins.pc) + 8 + ins.imm_signed));
    return ctx.sb.emit_const_u64(blk, tgt);
}

bool mod_active(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* key) {
    auto it = ins.modifier_strs.find(key);
    return it != ins.modifier_strs.end() && !it->second.empty() && it->second != "none";
}
bool mod_eq(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* key, const char* val) {
    auto it = ins.modifier_strs.find(key);
    std::string cur = (it != ins.modifier_strs.end()) ? it->second : "";
    return cur == val;
}
bool mod_ne(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* key, const char* val) {
    auto it = ins.modifier_strs.find(key);
    std::string cur = (it != ins.modifier_strs.end()) ? it->second : "";
    return cur != val;
}
bool has_field(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* name) {
    return ins.fields.count(name) > 0;
}
bool field_nz(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* name) {
    auto it = ins.fields.find(name);
    return it != ins.fields.end() && it->second != 0;
}
SsaId field_u32_mul16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* name) {
    auto it = ins.fields.find(name);
    uint32_t v = (it != ins.fields.end()) ? it->second : 0u;
    return ctx.sb.emit_const_u32(blk, v * 16u);
}
bool is_zero(Pass1Context&, uint32_t, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t v) { return v == 0u; }
bool lt_u32(Pass1Context&, uint32_t, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t a, uint32_t b) { return a < b; }

SsaId u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t v) {
    return ctx.sb.emit_const_u32(blk, v);
}
SsaId s32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, int32_t v) {
    return ctx.sb.emit_const_s32(blk, v);
}
SsaId u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint64_t v) {
    return ctx.sb.emit_const_u64(blk, v);
}
SsaId f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, float v) {
    uint32_t bits = 0; std::memcpy(&bits, &v, sizeof(float));
    return ctx.sb.emit_const_f32_bits(blk, bits);
}
SsaId f32_bits(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t bits) {
    return ctx.sb.emit_const_f32_bits(blk, bits);
}
SsaId const_bool(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, bool v) {
    return ctx.sb.emit_const_bool(blk, v);
}
SsaId fp_zero(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    if (!is_v2f16_helper(ins)) return ctx.sb.emit_const_f32_bits(blk, 0u);
    SsaId s = ctx.sb.emit_const_f32_bits(blk, 0u);
    return ctx.emit_op(blk, SsaOp::VecConstruct2, ValType::V2F32, { s, s });
}
SsaId token(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*) {
    return ctx.sb.current_token(blk);
}
SsaId raw0(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return fetch_raw_src_u32(ctx.sb, blk, ins, 0);
}
SsaId imm_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_u32(blk, static_cast<uint32_t>(ins.imm_unsigned));
}
SsaId imm_s32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_u32(blk, static_cast<uint32_t>(ins.imm_signed));
}
SsaId imm_f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_f32_bits(blk, static_cast<uint32_t>(ins.imm_unsigned));
}
SsaId imm_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_signed));
}
SsaId imm_u64_mask63(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_unsigned & 63u));
}
SsaId u64_mul4(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t w) {
    return ctx.sb.emit_const_u64(blk, uint64_t(w) * 4ULL);
}
SsaId f32_alpha_or_zero(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t c) {
    return ctx.sb.emit_const_f32_bits(blk, (c == 3u) ? 0x3f800000u : 0u);
}

SsaId load_sideband_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t member_idx) {
    return ctx.emit_load_sideband_u64(blk, member_idx);
}

SsaId split64_lo(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, SsaId val_u64) {
    SsaId lo = 0, hi = 0;
    ctx.emit_split64(blk, val_u64, &lo, &hi);
    return lo;
}

SsaId load_fau(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t word_idx) {
    return ctx.emit_load_fau(blk, word_idx);
}

SsaId bitcast(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, ValType vt, SsaId val) {
    return ctx.sb.emit_bitcast(blk, vt, val);
}

SsaId extract(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, ValType vt, SsaId vec, uint32_t idx) {
    SsaNode ex{};
    ex.op = SsaOp::VecExtract;
    ex.type = vt;
    ex.imm_u32 = idx;
    ex.args = { vec };
    return ctx.sb.emit(blk, std::move(ex));
}

SsaId shuffle2(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, ValType vt, SsaId a, SsaId b, uint32_t i0, uint32_t i1) {
    SsaNode sh{};
    sh.op = SsaOp::VecShuffle;
    sh.type = vt;
    sh.args = { a, b };
    sh.imm_u32 = i0;
    sh.imm_u32_2 = i1;
    return ctx.sb.emit(blk, std::move(sh));
}

SsaId bfi(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, SsaId a, SsaId b, uint32_t off, uint32_t w) {
    SsaNode n{};
    n.op = SsaOp::Bfi;
    n.type = ValType::U32;
    n.imm_u32 = off;
    n.imm_u32_2 = w;
    n.args = { a, b, ctx.sb.emit_const_u32(blk, off), ctx.sb.emit_const_u32(blk, w) };
    return ctx.sb.emit(blk, std::move(n));
}

void set_token(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, SsaId tok) {
    ctx.sb.set_token(blk, tok);
}

void set_branch_cond(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, SsaId cond) {
    ctx.mod.blocks[blk].branch_cond = cond;
}

SsaId sr_get(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t offset) {
    uint8_t base = ins.sr_read.present ? ins.sr_read.base_reg : 0;
    return ctx.sb.current_reg(blk, static_cast<uint8_t>(base + offset));
}

void sr_set(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t offset, SsaId val) {
    uint8_t base = ins.sr_write.present ? ins.sr_write.base_reg : 0;
    ctx.sb.set_reg(blk, static_cast<uint8_t>(base + offset), val);
}

SsaId tuple_val(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, ValType vt, SsaId tup) {
    SsaNode ev{};
    ev.op = SsaOp::TupleExtract;
    ev.type = vt;
    ev.imm_u32 = 0;
    ev.args = { tup };
    return ctx.sb.emit(blk, std::move(ev));
}

SsaId tuple_tok(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, SsaId tup) {
    SsaNode et{};
    et.op = SsaOp::TupleExtract;
    et.type = ValType::Token;
    et.imm_u32 = 1;
    et.args = { tup };
    return ctx.sb.emit(blk, std::move(et));
}

SsaId psb_load(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec* rule, SsaId addr, uint32_t elem_bytes, bool is_pka, SsaId tok) {
    SsaNode ld{};
    ld.op = SsaOp::PsbLoad;
    ld.type = ValType::TupleValToken;
    ld.imm_u32 = elem_bytes;
    ld.aux_bool = is_pka;
    ld.args = { addr, tok };
    ld.effects = rule->effects;
    ld.is_barrier_or_cf = true;
    ld.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(ld));
}

SsaId psb_store(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec* rule, SsaId addr, SsaId val, uint32_t elem_bytes, SsaId tok) {
    SsaNode st{};
    st.op = SsaOp::PsbStore;
    st.type = ValType::Token;
    st.imm_u32 = elem_bytes;
    st.args = { addr, val, tok };
    st.effects = rule->effects;
    st.is_barrier_or_cf = true;
    st.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(st));
}

SsaId atomic_rmw(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId addr, SsaId arg, bool is_64, SsaId tok) {
    std::string aop = "aadd";
    auto it = ins.modifier_strs.find("atom_opc");
    if (it != ins.modifier_strs.end()) aop = it->second;
    SsaNode at{};
    at.op = SsaOp::AtomicRmw;
    at.type = ValType::TupleValToken;
    at.aux_str = aop;
    at.aux_bool = is_64;
    at.args = { addr, arg, tok };
    at.effects = rule->effects;
    at.is_barrier_or_cf = true;
    at.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(at));
}

SsaId atomic_rmw_aadd(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec* rule, SsaId addr, SsaId arg, bool is_64, SsaId tok) {
    SsaNode at{};
    at.op = SsaOp::AtomicRmw;
    at.type = ValType::TupleValToken;
    at.aux_str = "aadd";
    at.aux_bool = is_64;
    at.args = { addr, arg, tok };
    at.effects = rule->effects;
    at.is_barrier_or_cf = true;
    at.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(at));
}

SsaId atomic_xchg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec* rule, SsaId addr, SsaId arg, SsaId tok) {
    SsaNode at{};
    at.op = SsaOp::AtomicExchange;
    at.type = ValType::TupleValToken;
    at.args = { addr, arg, tok };
    at.effects = rule->effects;
    at.is_barrier_or_cf = true;
    at.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(at));
}

SsaId atomic_cmpxchg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec* rule, SsaId addr, SsaId xchg, SsaId cmp, SsaId tok) {
    SsaNode at{};
    at.op = SsaOp::AtomicCmpXchg;
    at.type = ValType::TupleValToken;
    at.args = { addr, xchg, cmp, tok };
    at.effects = rule->effects;
    at.is_barrier_or_cf = true;
    at.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(at));
}

SsaId load_vert_attr(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    uint32_t attr_idx = 0;
    auto it = ins.fields.find("attribute_index");
    if (it != ins.fields.end()) attr_idx = it->second;
    SsaNode ld{};
    ld.op = SsaOp::LoadVertAttr;
    ld.type = ValType::V4F32;
    ld.imm_u32 = attr_idx;
    ld.is_barrier_or_cf = true;
    return ctx.sb.emit(blk, std::move(ld));
}

SsaId load_frag_var(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, bool is_flat) {
    uint32_t var_loc = 0;
    auto it = ins.fields.find("index");
    if (it != ins.fields.end()) var_loc = it->second / 16u;
    SsaNode ld{};
    ld.op = SsaOp::LoadFragVarying;
    ld.type = ValType::V4F32;
    ld.imm_u32 = var_loc;
    ld.aux_bool = is_flat;
    ld.is_barrier_or_cf = true;
    return ctx.sb.emit(blk, std::move(ld));
}

SsaId load_frag_var_idx(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t var_loc, bool is_flat) {
    SsaNode ld{};
    ld.op = SsaOp::LoadFragVarying;
    ld.type = ValType::V4F32;
    ld.imm_u32 = var_loc;
    ld.aux_bool = is_flat;
    ld.is_barrier_or_cf = true;
    return ctx.sb.emit(blk, std::move(ld));
}

SsaId tex_sample(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, SsaId uv, bool is_gather, SsaId tok) {
    uint32_t gcomp = 0;
    if (is_gather) {
        auto it_gc = ins.fields.find("gather_component");
        if (it_gc != ins.fields.end()) gcomp = it_gc->second & 3u;
    }
    SsaNode ts{};
    ts.op = SsaOp::TexSample;
    ts.type = ValType::TupleValToken;
    ts.imm_u32 = 0;
    ts.imm_u32_2 = gcomp;
    ts.aux_bool = (ctx.opts.stage == ShaderStage::Fragment && !is_gather);
    ts.aux_bool2 = is_gather;
    ts.args = { uv, tok };
    ts.is_barrier_or_cf = true;
    ts.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(ts));
}

SsaId store_builtin_pos(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, SsaId pos_vec, SsaId tok) {
    SsaNode st{};
    st.op = SsaOp::StoreBuiltinOutput;
    st.type = ValType::Token;
    st.imm_u32 = static_cast<uint32_t>(spirv::BuiltIn::Position);
    st.aux_str = "gl_Position";
    st.args = { pos_vec, tok };
    st.is_barrier_or_cf = true;
    st.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(st));
}

SsaId store_builtin_depth(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec* rule, SsaId z_f32, SsaId tok) {
    SsaNode st{};
    st.op = SsaOp::StoreBuiltinOutput;
    st.type = ValType::Token;
    st.imm_u32 = static_cast<uint32_t>(spirv::BuiltIn::FragDepth);
    st.aux_str = "gl_FragDepth";
    st.args = { z_f32, tok };
    st.effects = rule->effects;
    st.is_barrier_or_cf = true;
    st.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(st));
}

SsaId store_vert_varying(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, SsaId vec, SsaId tok) {
    uint32_t var_loc = static_cast<uint32_t>(std::max<int64_t>(0, ins.imm_signed) / 16);
    SsaNode st{};
    st.op = SsaOp::StoreVertVarying;
    st.type = ValType::Token;
    st.imm_u32 = var_loc;
    st.args = { vec, tok };
    st.is_barrier_or_cf = true;
    st.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(st));
}

SsaId store_frag_color(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, SsaId col, SsaId tok) {
    uint32_t rt_loc = 0;
    if (!ins.srcs.empty() && ins.srcs[0].symbolic_name.find("blend_descriptor_") == 0) {
        char c = ins.srcs[0].symbolic_name.back();
        if (c >= '0' && c <= '7') rt_loc = static_cast<uint32_t>(c - '0');
    }
    SsaNode st{};
    st.op = SsaOp::StoreFragColor;
    st.type = ValType::Token;
    st.imm_u32 = rt_loc;
    st.args = { col, tok };
    st.is_barrier_or_cf = true;
    st.is_side_effect = true;
    return ctx.sb.emit(blk, std::move(st));
}

void kill_cond(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, SsaId cond, SsaId tok) {
    SsaNode k{};
    k.op = SsaOp::Kill;
    k.type = ValType::Void;
    k.aux_bool = true;
    k.args = { cond, tok };
    k.is_barrier_or_cf = true;
    k.is_side_effect = true;
    ctx.sb.emit(blk, std::move(k));
}

} // namespace abi

// ============================================================================
// Remaining Complex Host-Side Handler (@HANDLER(fn))
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

} // namespace valhall::compiler
