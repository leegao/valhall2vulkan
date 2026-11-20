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
    SsaId s0 = fetch_src_int32(*this, blk, ins, nullptr, 0, false);
    SsaId cond = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { s0, sb.emit_const_u32(blk, 0u) });

    std::string comb = "none";
    auto it_c = ins.modifier_strs.find("combine");
    if (it_c != ins.modifier_strs.end()) comb = it_c->second;
    if (ins.srcs.size() >= 2 && comb != "none") {
        SsaId s1 = fetch_src_int32(*this, blk, ins, nullptr, 1, false);
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

// TODO(v17.x): Verify that [[verifier::z3(...)]] short-circuit semantics on abi:: helpers
// are equivalent to the actual C++ implementation bodies.
namespace abi {

static inline bool is_v2f16_helper(const DecodedValhallInst& ins) {
    if (ins.canonical_variant == Variant::V2F16 || ins.variant == Variant::V2F16) return true;
    return std::strcmp(variant_to_string(ins.canonical_variant), "v2f16") == 0;
}

[[verifier::z3("InstLoweringContext", "len(this.srcs) >= 1")]] bool has_src0(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.srcs.size() >= 1; }
[[verifier::z3("InstLoweringContext", "len(this.srcs) >= 2")]] bool has_src1(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.srcs.size() >= 2; }
[[verifier::z3("InstLoweringContext", "len(this.srcs) >= 3")]] bool has_src2(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.srcs.size() >= 3; }
[[verifier::z3("InstLoweringContext", "len(this.srcs) >= 4")]] bool has_src3(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.srcs.size() >= 4; }
[[verifier::z3("InstLoweringContext", "this.has_dest")]] bool has_dest(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.has_dest; }
bool is_v2f16(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return is_v2f16_helper(ins); }
bool is_signed(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    std::string var = variant_to_string(ins.canonical_variant);
    return (var.find("s") == 0 || var.find("v2s") == 0 || var.find("v4s") == 0);
}
[[verifier::z3("InstLoweringContext", "this.dest_is_64bit")]] bool is_64bit(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.dest_is_64bit; }
[[verifier::z3("InstLoweringContext", "this.dest_is_64bit")]] bool dest_is_64bit(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.dest_is_64bit; }
[[verifier::z3("InstLoweringContext", "this.stage == 'Fragment'")]] bool is_frag_stage(Pass1Context& ctx, uint32_t, const DecodedValhallInst&, const tables::LoweringRuleSpec*) { return ctx.opts.stage == ShaderStage::Fragment; }
[[verifier::z3("InstLoweringContext", "this.stage == 'Vertex'")]] bool is_vert_stage(Pass1Context& ctx, uint32_t, const DecodedValhallInst&, const tables::LoweringRuleSpec*) { return ctx.opts.stage == ShaderStage::Vertex; }
[[verifier::z3("InstLoweringContext", "this.has_imm and this.imm_signed != 0")]] bool has_imm_nonzero(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.has_imm && ins.imm_signed != 0; }
[[verifier::z3("InstLoweringContext", "this.sr_write_present")]] bool sr_write_present(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) { return ins.sr_write.present; }
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
[[verifier::z3("InstLoweringContext", "this.fields.get('not_result', 0) != 0")]] bool has_not_result(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    auto it = ins.fields.find("not_result"); return it != ins.fields.end() && it->second != 0;
}
[[verifier::z3("InstLoweringContext", "this.fields.get('not', 0) != 0")]] bool has_not(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    auto it = ins.fields.find("not"); return it != ins.fields.end() && it->second != 0;
}

[[verifier::z3("InstLoweringContext", "max(1, this.sr_write_count) if this.sr_write_present else 1")]] uint32_t sr_write_count(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ins.sr_write.present ? std::max<uint32_t>(1u, ins.sr_write.count) : 1u;
}
[[verifier::z3("InstLoweringContext", "min(4, max(1, this.sr_write_count)) if this.sr_write_present else 4")]] uint32_t sr_write_count_max4(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ins.sr_write.present ? std::min<uint32_t>(4u, std::max<uint32_t>(1u, ins.sr_write.count)) : 4u;
}
[[verifier::z3("InstLoweringContext", "max(1, this.sr_read_count) if this.sr_read_present else 1")]] uint32_t sr_read_count(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ins.sr_read.present ? std::max<uint32_t>(1u, ins.sr_read.count) : 1u;
}
uint32_t src0_swz_h1_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    return (abi::get_src_swz(ctx, blk, ins, rule, 0) == "h1") ? 1u : 0u;
}
SsaId branch_target_pc_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    uint64_t tgt = static_cast<uint64_t>(ins.has_branch_offset ? ins.branch_target_pc : (int64_t(ins.pc) + 8 + ins.imm_signed));
    return ctx.sb.emit_const_u64(blk, tgt);
}

[[verifier::z3("InstLoweringContext", "bool(this.modifiers.get(args[0], '')) and this.modifiers.get(args[0], '') != 'none'")]] bool mod_active(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* key) {
    auto it = ins.modifier_strs.find(key);
    return it != ins.modifier_strs.end() && !it->second.empty() && it->second != "none";
}
[[verifier::z3("InstLoweringContext", "this.modifiers.get(args[0], '') == args[1]")]] bool mod_eq(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* key, const char* val) {
    auto it = ins.modifier_strs.find(key);
    std::string cur = (it != ins.modifier_strs.end()) ? it->second : "";
    return cur == val;
}
[[verifier::z3("InstLoweringContext", "this.modifiers.get(args[0], '') != args[1]")]] bool mod_ne(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* key, const char* val) {
    auto it = ins.modifier_strs.find(key);
    std::string cur = (it != ins.modifier_strs.end()) ? it->second : "";
    return cur != val;
}
[[verifier::z3("InstLoweringContext", "args[0] in this.fields")]] bool has_field(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* name) {
    return ins.fields.count(name) > 0;
}
[[verifier::z3("InstLoweringContext", "this.fields.get(args[0], 0) != 0")]] bool field_nz(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* name) {
    auto it = ins.fields.find(name);
    return it != ins.fields.end() && it->second != 0;
}
SsaId field_u32_mul16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, const char* name) {
    auto it = ins.fields.find(name);
    uint32_t v = (it != ins.fields.end()) ? it->second : 0u;
    return ctx.sb.emit_const_u32(blk, v * 16u);
}
[[verifier::z3("InstLoweringContext", "args[0] == 0")]] bool is_zero(Pass1Context&, uint32_t, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t v) { return v == 0u; }
[[verifier::z3("InstLoweringContext", "args[0] < args[1]")]] bool lt_u32(Pass1Context&, uint32_t, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t a, uint32_t b) { return a < b; }

[[verifier::z3("SymbolicSsaModule", "this.emit_const_u32(args[0])")]] SsaId u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t v) {
    return ctx.sb.emit_const_u32(blk, v);
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_s32(args[0])")]] SsaId s32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, int32_t v) {
    return ctx.sb.emit_const_s32(blk, v);
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_u64(args[0])")]] SsaId u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint64_t v) {
    return ctx.sb.emit_const_u64(blk, v);
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_f32(args[0])")]] SsaId f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, float v) {
    uint32_t bits = 0; std::memcpy(&bits, &v, sizeof(float));
    return ctx.sb.emit_const_f32_bits(blk, bits);
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_f32_bits(args[0])")]] SsaId f32_bits(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t bits) {
    return ctx.sb.emit_const_f32_bits(blk, bits);
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_bool(args[0])")]] SsaId const_bool(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, bool v) {
    return ctx.sb.emit_const_bool(blk, v);
}
[[verifier::z3("SymbolicSsaModule", "this.current_token()")]] SsaId token(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*) {
    return ctx.sb.current_token(blk);
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_u32(ins.imm_unsigned)")]] SsaId imm_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_u32(blk, static_cast<uint32_t>(ins.imm_unsigned));
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_u32(ins.imm_signed)")]] SsaId imm_s32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_u32(blk, static_cast<uint32_t>(ins.imm_signed));
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_f32_bits(ins.imm_unsigned)")]] SsaId imm_f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_f32_bits(blk, static_cast<uint32_t>(ins.imm_unsigned));
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_u64(ins.imm_signed)")]] SsaId imm_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_signed));
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_u64(ins.imm_unsigned & 63)")]] SsaId imm_u64_mask63(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_unsigned & 63u));
}
[[verifier::z3("SymbolicSsaModule", "this.emit_const_u64(args[0] * 4)")]] SsaId u64_mul4(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t w) {
    return ctx.sb.emit_const_u64(blk, uint64_t(w) * 4ULL);
}
SsaId f32_alpha_or_zero(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t c) {
    return ctx.sb.emit_const_f32_bits(blk, (c == 3u) ? 0x3f800000u : 0u);
}

SsaId load_sideband_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t member_idx) {
    return ctx.emit_load_sideband_u64(blk, member_idx);
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

[[verifier::z3("SymbolicSsaModule", "this.set_token(args[0])")]] void set_token(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, SsaId tok) {
    ctx.sb.set_token(blk, tok);
}

[[verifier::z3("SymbolicSsaModule", "this.set_branch_cond(args[0])")]] void set_branch_cond(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, SsaId cond) {
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
    auto it = ins.fields.find("varying_index");
    if (it != ins.fields.end()) {
        var_loc = it->second;
    } else if (auto it2 = ins.fields.find("index"); it2 != ins.fields.end()) {
        var_loc = it2->second;
    }
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

void handle_int_idpadd(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    std::string var = variant_to_string(ins.canonical_variant);
    bool s0_signed = (var.find("s") == 0);
    bool s1_signed = s0_signed;
    if (ins.fields.count("unsigned0") && ins.fields.at("unsigned0") != 0) s0_signed = false;
    if (ins.fields.count("unsigned1") && ins.fields.at("unsigned1") != 0) s1_signed = false;
    SsaId a4 = fetch_src_v4i8(ctx, blk, ins, rule, 0, s0_signed);
    SsaId b4 = fetch_src_v4i8(ctx, blk, ins, rule, 1, s1_signed);
    if (s0_signed != s1_signed) {
        if (!s0_signed) a4 = ctx.sb.emit_bitcast(blk, ValType::V4S32, a4);
        if (!s1_signed) b4 = ctx.sb.emit_bitcast(blk, ValType::V4S32, b4);
    }
    bool acc_s = (s0_signed || s1_signed);
    SsaId acc = (ins.srcs.size() >= 3) ? fetch_src_int32(ctx, blk, ins, rule, 2, acc_s) : ctx.sb.emit_const_u32(blk, 0u);
    if (acc_s) acc = ctx.sb.emit_bitcast(blk, ValType::S32, acc);
    ValType elem_t = acc_s ? ValType::S32 : ValType::U32;
    for (uint32_t i = 0; i < 4; ++i) {
        SsaNode ea{}; ea.op = SsaOp::VecExtract; ea.type = elem_t; ea.imm_u32 = i; ea.args = { a4 };
        SsaNode eb{}; eb.op = SsaOp::VecExtract; eb.type = elem_t; eb.imm_u32 = i; eb.args = { b4 };
        SsaId prod = ctx.emit_op(blk, SsaOp::IMul, elem_t, { ctx.sb.emit(blk, std::move(ea)), ctx.sb.emit(blk, std::move(eb)) });
        acc = ctx.emit_op(blk, SsaOp::IAdd, elem_t, { acc, prod });
    }
    store_dest_u32(ctx, blk, ins, rule, acc_s ? ctx.sb.emit_bitcast(blk, ValType::U32, acc) : acc);
}


namespace abi {

std::string get_src_swz(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    // For AND/OR/XOR/NOT aliases of SHIFT_BITWISE_I32, lanes1 in the raw encoding belongs
    // to the omitted shift-amount operand (k0.b0), not to ins.srcs[1].
    bool is_shift_bitwise_alias_src1 = (src_idx == 1) && (
        ins.mnemonic == Mnemonic::AND || ins.canonical_mnemonic == Mnemonic::AND ||
        ins.mnemonic == Mnemonic::OR  || ins.canonical_mnemonic == Mnemonic::OR  ||
        ins.mnemonic == Mnemonic::XOR || ins.canonical_mnemonic == Mnemonic::XOR ||
        ins.mnemonic == Mnemonic::NOT || ins.canonical_mnemonic == Mnemonic::NOT);
    if (is_shift_bitwise_alias_src1) {
        return "";
    }
    if (src_idx < ins.srcs.size() && !ins.srcs[src_idx].swizzle_mod.empty()) {
        return ins.srcs[src_idx].swizzle_mod;
    }
    std::string idx_str = std::to_string(src_idx);
    for (const char* prefix : {"widen", "swz", "lanes"}) {
        auto it = ins.modifier_strs.find(std::string(prefix) + idx_str);
        if (it != ins.modifier_strs.end() && it->second != "none") {
            return it->second;
        }
    }
    return "";
}

std::string get_dest_mod(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    if (!ins.dest_mod.empty()) return ins.dest_mod;
    for (const char* k : {"dest_width", "dest_width_narrow"}) {
        auto it = ins.modifier_strs.find(k);
        if (it != ins.modifier_strs.end() && it->second != "none") {
            return it->second;
        }
    }
    return "";
}

bool get_src_abs(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    if (src_idx < ins.srcs.size() && ins.srcs[src_idx].abs) return true;
    auto it = ins.fields.find("abs" + std::to_string(src_idx));
    return (it != ins.fields.end() && it->second != 0);
}

bool get_src_neg(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    if (src_idx < ins.srcs.size() && ins.srcs[src_idx].neg) return true;
    auto it = ins.fields.find("neg" + std::to_string(src_idx));
    return (it != ins.fields.end() && it->second != 0);
}

bool get_src_not(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    if (src_idx < ins.srcs.size() && ins.srcs[src_idx].not_mod) return true;
    auto it = ins.fields.find("not" + std::to_string(src_idx));
    return (it != ins.fields.end() && it->second != 0);
}

bool src_swz_eq(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx, const char* str) {
    return abi::get_src_swz(ctx, blk, ins, rule, src_idx) == str;
}

[[verifier::z3("InstLoweringContext", "this.dest_mod == args[0]")]] bool dest_mod_eq(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, const char* str) {
    return abi::get_dest_mod(ctx, blk, ins, rule) == str;
}

[[verifier::z3("InstLoweringContext", "args[0] >= len(this.srcs)")]] bool src_oob(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx >= ins.srcs.size();
}

[[verifier::z3("InstLoweringContext", "args[0] < len(this.srcs) and this.srcs[args[0]].kind == 'Register'")]] bool src_is_reg(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx < ins.srcs.size() && ins.srcs[src_idx].kind == SourceKind::Register;
}

[[verifier::z3("InstLoweringContext", "args[0] < len(this.srcs) and this.srcs[args[0]].kind == 'FauUniform'")]] bool src_is_fau_uniform(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx < ins.srcs.size() && ins.srcs[src_idx].kind == SourceKind::FauUniform;
}

[[verifier::z3("InstLoweringContext", "args[0] < len(this.srcs) and this.srcs[args[0]].kind == 'SmallConstant'")]] bool src_is_small_const(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx < ins.srcs.size() && ins.srcs[src_idx].kind == SourceKind::SmallConstant;
}

[[verifier::z3("InstLoweringContext", "args[0] < len(this.srcs) and this.srcs[args[0]].kind == 'FauSpecial' and this.srcs[args[0]].symbolic_name == 'pc'")]] bool src_is_fau_pc(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx < ins.srcs.size() &&
           ins.srcs[src_idx].kind == SourceKind::FauSpecial &&
           ins.srcs[src_idx].symbolic_name == "pc";
}

[[verifier::z3("InstLoweringContext", "args[0] < len(this.srcs) and this.srcs[args[0]].kind == 'FauSpecial' and this.srcs[args[0]].symbolic_name == 'lane_id'")]] bool src_is_fau_lane_id(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx < ins.srcs.size() &&
           ins.srcs[src_idx].kind == SourceKind::FauSpecial &&
           ins.srcs[src_idx].symbolic_name == "lane_id";
}

[[verifier::z3("InstLoweringContext", "args[0] < len(this.srcs) and this.srcs[args[0]].kind == 'FauSpecial' and this.srcs[args[0]].symbolic_name == 'lane_width'")]] bool src_is_fau_lane_width(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx < ins.srcs.size() &&
           ins.srcs[src_idx].kind == SourceKind::FauSpecial &&
           ins.srcs[src_idx].symbolic_name == "lane_width";
}

[[verifier::z3("InstLoweringContext", "args[0] < len(this.srcs) and this.srcs[args[0]].word_select")]] bool src_word_select(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx < ins.srcs.size() && ins.srcs[src_idx].word_select;
}

[[verifier::z3("InstLoweringContext", "args[0] < len(this.srcs) and this.srcs[args[0]].is_64bit")]] bool src_is_64bit(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx < ins.srcs.size() && ins.srcs[src_idx].is_64bit;
}

[[verifier::z3("InstLoweringContext", "this.srcs[args[0]].reg_num")]] uint32_t src_reg_num(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return ins.srcs[src_idx].reg_num;
}

[[verifier::z3("InstLoweringContext", "this.srcs[args[0]].reg_num & 62")]] uint32_t src_reg_base(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return ins.srcs[src_idx].reg_num & 62u;
}

[[verifier::z3("InstLoweringContext", "this.dest_reg & 63")]] uint32_t dest_reg_num(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ins.dest_reg & 63u;
}

[[verifier::z3("InstLoweringContext", "this.dest_reg & 62")]] uint32_t dest_reg_base(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ins.dest_reg & 62u;
}

[[verifier::z3("SymbolicSsaModule", "this.current_reg(ins.srcs[args[0]].reg_num)")]] SsaId src_read_reg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return ctx.sb.current_reg(blk, ins.srcs[src_idx].reg_num);
}

[[verifier::z3("SymbolicSsaModule", "this.current_reg((ins.srcs[args[0]].reg_num & 62) + args[1])")]] SsaId src_read_reg_base(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx, uint32_t off) {
    uint8_t base = ins.srcs[src_idx].reg_num & 62u;
    return ctx.sb.current_reg(blk, static_cast<uint8_t>(base + off));
}

[[verifier::z3("SymbolicSsaModule", "this.emit_const_u32(ins.srcs[args[0]].imm_bits)")]] SsaId src_emit_imm_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return ctx.sb.emit_const_u32(blk, ins.srcs[src_idx].imm_bits);
}

[[verifier::z3("SymbolicSsaModule", "this.emit_const_u64(ins.srcs[args[0]].imm_bits)")]] SsaId src_emit_imm_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return ctx.sb.emit_const_u64(blk, uint64_t(ins.srcs[src_idx].imm_bits));
}

[[verifier::z3("SymbolicSsaModule", "this.emit_const_u64(ins.pc)")]] SsaId ins_pc_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.emit_const_u64(blk, ins.pc);
}

[[verifier::z3("SymbolicSsaModule", "this.current_reg(ins.dest_reg & 63)")]] SsaId dest_read_reg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    return ctx.sb.current_reg(blk, ins.dest_reg & 63u);
}

[[verifier::z3("SymbolicSsaModule", "this.set_reg(args[0], args[1])")]] void dest_set_reg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t d, SsaId val) {
    ctx.sb.set_reg(blk, static_cast<uint8_t>(d), val);
}

SsaId emit_lane_id(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*) {
    SsaNode bi{};
    bi.op = tables::SsaOp::LoadBuiltinInput;
    bi.type = ValType::U32;
    bi.imm_u32 = static_cast<uint32_t>(spirv::BuiltIn::SubgroupLocalInvocationId);
    bi.aux_bool = true;
    bi.aux_str = "gl_SubgroupInvocationID";
    return ctx.sb.emit(blk, std::move(bi));
}

SsaId emit_lane_width(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*) {
    SsaNode sb32{};
    sb32.op = tables::SsaOp::LoadSidebandU32;
    sb32.type = ValType::U32;
    sb32.imm_u32 = 5;
    sb32.imm_u32_2 = 3;
    sb32.effects.flags |= EffectSet::kFlagUsesSideband;
    return ctx.sb.emit(blk, std::move(sb32));
}

uint32_t add_u32(Pass1Context&, uint32_t, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t a, uint32_t b) {
    return a + b;
}

uint32_t mul8(Pass1Context&, uint32_t, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t a) {
    return a * 8u;
}

} // namespace abi

} // namespace valhall::compiler
