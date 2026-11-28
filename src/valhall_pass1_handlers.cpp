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
    n.imm_u32 = word_idx & 127u;
    n.effects.flags |= EffectSet::kFlagUsesSideband;
    res.uses_sideband = true;
    return sb.emit(blk, std::move(n));
}

SsaId Pass1Context::emit_load_sideband_u64(uint32_t blk, uint32_t member_idx) {
    SsaNode n{};
    n.op = SsaOp::LoadSidebandU64;
    n.type = ValType::U64;
    n.imm_u32 = member_idx;
    if (member_idx == 8u) {
        n.aux_str = "gl_SamplePosition";
    }
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
        (void)insts;
        // Valhall compute hardware preload ABI (aux_idx preload mask = r48 + bit):
        //   r55: LocalInvocationID.xy packed as v2u16 (h0 = x, h1 = y)
        //   r56: LocalInvocationID.z (h0 = z)
        //   r57..r59: WorkGroupID.xyz
        //   r60..r62: GlobalInvocationID.xyz
        SsaId lid = emit_builtin_input(blk, spirv::BuiltIn::LocalInvocationId, ValType::V3U32, "gl_LocalInvocationID");
        SsaNode lx_n{}; lx_n.op = SsaOp::VecExtract; lx_n.type = ValType::U32; lx_n.imm_u32 = 0; lx_n.args = { lid };
        SsaNode ly_n{}; ly_n.op = SsaOp::VecExtract; ly_n.type = ValType::U32; ly_n.imm_u32 = 1; ly_n.args = { lid };
        SsaNode lz_n{}; lz_n.op = SsaOp::VecExtract; lz_n.type = ValType::U32; lz_n.imm_u32 = 2; lz_n.args = { lid };
        SsaId lx = sb.emit(blk, std::move(lx_n));
        SsaId ly = sb.emit(blk, std::move(ly_n));
        SsaId lz = sb.emit(blk, std::move(lz_n));
        SsaId c16 = sb.emit_const_u32(blk, 16u);
        SsaNode lxy{}; lxy.op = SsaOp::Bfi; lxy.type = ValType::U32;
        lxy.imm_u32 = 16u; lxy.imm_u32_2 = 16u;
        lxy.args = { lx, ly, c16, c16 };
        sb.set_reg(blk, 55, sb.emit(blk, std::move(lxy)));
        sb.set_reg(blk, 56, lz);

        SsaId wgid = emit_builtin_input(blk, spirv::BuiltIn::WorkgroupId, ValType::V3U32, "gl_WorkGroupID");
        for (uint32_t c = 0; c < 3; ++c) {
            SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::U32; ex.imm_u32 = c; ex.args = { wgid };
            sb.set_reg(blk, 57 + c, sb.emit(blk, std::move(ex)));
        }

        SsaId gid = emit_builtin_input(blk, spirv::BuiltIn::GlobalInvocationId, ValType::V3U32, "gl_GlobalInvocationID");
        for (uint32_t c = 0; c < 3; ++c) {
            SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::U32; ex.imm_u32 = c; ex.args = { gid };
            sb.set_reg(blk, 60 + c, sb.emit(blk, std::move(ex)));
        }
    } else if (opts.stage == ShaderStage::Vertex) {
        SsaId vi = emit_builtin_input(blk, spirv::BuiltIn::VertexIndex, ValType::U32, "gl_VertexIndex");
        SsaId ii = emit_builtin_input(blk, spirv::BuiltIn::InstanceIndex, ValType::U32, "gl_InstanceIndex");
        sb.set_reg(blk, 59, vi);
        sb.set_reg(blk, 60, vi);
        sb.set_reg(blk, 61, ii);
    } else if (opts.stage == ShaderStage::Fragment) {
        // Valhall Fragment ABI: r59 holds packed integer pixel coordinates (uint16(y) << 16 | uint16(x)),
        // r60 holds initial sample coverage mask, and r61 holds active/helper flag (.h0) + sample ID ([21:16]).
        SsaId fc = emit_builtin_input(blk, spirv::BuiltIn::FragCoord, ValType::V4F32, "gl_FragCoord");
        SsaNode ex_x{}; ex_x.op = SsaOp::VecExtract; ex_x.type = ValType::F32; ex_x.imm_u32 = 0; ex_x.args = { fc };
        SsaNode ex_y{}; ex_y.op = SsaOp::VecExtract; ex_y.type = ValType::F32; ex_y.imm_u32 = 1; ex_y.args = { fc };
        SsaId fx = emit_op(blk, SsaOp::ConvertFToU, ValType::U32, { emit_op(blk, SsaOp::Floor, ValType::F32, { sb.emit(blk, std::move(ex_x)) }) });
        SsaId fy = emit_op(blk, SsaOp::ConvertFToU, ValType::U32, { emit_op(blk, SsaOp::Floor, ValType::F32, { sb.emit(blk, std::move(ex_y)) }) });
        SsaId packed_xy = emit_op(blk, SsaOp::Bfi, ValType::U32, { fx, fy, sb.emit_const_u32(blk, 16u), sb.emit_const_u32(blk, 16u) });
        sb.set_reg(blk, 59, packed_xy);
        sb.set_reg(blk, 60, sb.emit_const_u32(blk, 1u));
        SsaId helper = emit_builtin_input(blk, spirv::BuiltIn::HelperInvocation, ValType::Bool, "HelperInvocation");
        SsaId active = emit_op(blk, SsaOp::Select, ValType::U32, { helper, sb.emit_const_u32(blk, 0u), sb.emit_const_u32(blk, 1u) });
        SsaId sid = emit_builtin_input(blk, spirv::BuiltIn::SampleId, ValType::U32, "gl_SampleID");
        SsaId r61_val = emit_op(blk, SsaOp::Bfi, ValType::U32, { active, sid, sb.emit_const_u32(blk, 16u), sb.emit_const_u32(blk, 16u) });
        sb.set_reg(blk, 61, r61_val);
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

    std::string comb = "none";
    auto it_c = ins.modifier_strs.find("combine");
    if (it_c != ins.modifier_strs.end()) comb = it_c->second;

    SsaId cond = kInvalidSsa;
    if (comb == "and") {
        cond = emit_op(blk, SsaOp::ICmpEq, ValType::Bool, { s0, sb.emit_const_u32(blk, 0xFFFFFFFFu) });
    } else if (comb == "h0") {
        SsaId lo = emit_op(blk, SsaOp::BitAnd, ValType::U32, { s0, sb.emit_const_u32(blk, 0xFFFFu) });
        cond = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { lo, sb.emit_const_u32(blk, 0u) });
    } else if (comb == "h1") {
        SsaId hi = emit_op(blk, SsaOp::ShrU, ValType::U32, { s0, sb.emit_const_u32(blk, 16u) });
        cond = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { hi, sb.emit_const_u32(blk, 0u) });
    } else if (comb == "lowbits") {
        SsaId mask = sb.emit_const_u32(blk, 0x001FFFF8u);
        SsaId lb = emit_op(blk, SsaOp::BitAnd, ValType::U32, { s0, mask });
        SsaId nz = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { lb, sb.emit_const_u32(blk, 0u) });
        SsaId nf = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { lb, mask });
        cond = emit_op(blk, SsaOp::LogicalAnd, ValType::Bool, { nz, nf });
    } else {
        cond = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { s0, sb.emit_const_u32(blk, 0u) });
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

static uint32_t map_tex_write_mask_channel(const DecodedValhallInst& ins, uint32_t sr_idx) {
    auto it = ins.modifier_strs.find("write_mask");
    if (it == ins.modifier_strs.end() || it->second.empty()) return sr_idx;
    const std::string& mask = it->second;
    uint32_t seen = 0;
    for (char c : mask) {
        uint32_t ch = 0xFFu;
        if (c == 'r' || c == 'R') ch = 0u;
        else if (c == 'g' || c == 'G') ch = 1u;
        else if (c == 'b' || c == 'B') ch = 2u;
        else if (c == 'a' || c == 'A') ch = 3u;
        if (ch != 0xFFu) {
            if (seen == sr_idx) return ch;
            ++seen;
        }
    }
    return sr_idx;
}

SsaId extract(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, ValType vt, SsaId vec, uint32_t idx) {
    SsaNode ex{};
    ValType src_ty = (vec < ctx.res.ssa.nodes.size()) ? ctx.res.ssa.get(vec).type : ValType::Void;
    ex.op = (src_ty == ValType::TupleU32x2 || src_ty == ValType::TupleValToken)
            ? SsaOp::TupleExtract : SsaOp::VecExtract;
    ex.type = vt;
    uint32_t eff_idx = idx;
    bool is_tex_result = (vec < ctx.res.ssa.nodes.size() &&
                          ctx.res.ssa.get(vec).op == SsaOp::TupleExtract);
    if (ex.op == SsaOp::VecExtract && src_ty == ValType::V4F32 && is_tex_result &&
        rule && (rule->op_class == tables::OpClass::TexSample ||
                 rule->op_class == tables::OpClass::TexVarSample)) {
        eff_idx = map_tex_write_mask_channel(ins, idx);
        if ((ins.mnemonic == Mnemonic::TEX_GATHER ||
             ins.mnemonic == Mnemonic::VAR_TEX_GATHER) && eff_idx < 4u) {
            // Valhall gather4 writes hardware channels (r,g,b,a) = SPIR-V (w,z,x,y) = (3,2,0,1).
            static constexpr uint32_t kGather4Perm[4] = { 3u, 2u, 0u, 1u };
            eff_idx = kGather4Perm[eff_idx];
        }
    }
    ex.imm_u32 = eff_idx;
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
    uint8_t dst_reg = static_cast<uint8_t>(base + offset);
    // BUG-020: LOAD.i16/i8 with extend=none and lane=h0/h1/b0..b3 inserts into the target
    // subword lane of dst_reg while preserving all other bits of dst_reg.
    if (offset == 0u && ins.mnemonic == Mnemonic::LOAD) {
        auto it_ext = ins.modifier_strs.find("extend");
        auto it_lane = ins.modifier_strs.find("lane");
        if ((it_ext == ins.modifier_strs.end() || it_ext->second == "none") &&
            it_lane != ins.modifier_strs.end()) {
            const std::string& ln = it_lane->second;
            uint32_t bit_off = 0u, bit_w = 0u;
            if (ln == "h0") { bit_off = 0u; bit_w = 16u; }
            else if (ln == "h1") { bit_off = 16u; bit_w = 16u; }
            else if (ln == "b0") { bit_off = 0u; bit_w = 8u; }
            else if (ln == "b1") { bit_off = 8u; bit_w = 8u; }
            else if (ln == "b2") { bit_off = 16u; bit_w = 8u; }
            else if (ln == "b3") { bit_off = 24u; bit_w = 8u; }
            if (bit_w != 0u) {
                SsaId prev = ctx.sb.current_reg(blk, dst_reg);
                if (prev == kInvalidSsa) prev = ctx.sb.emit_const_u32(blk, 0u);
                val = ctx.emit_op(blk, SsaOp::Bfi, ValType::U32, {
                    prev, val, ctx.sb.emit_const_u32(blk, bit_off), ctx.sb.emit_const_u32(blk, bit_w)
                });
            }
        }
    }
    ctx.sb.set_reg(blk, dst_reg, val);
}

SsaId tuple_val(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, ValType vt, SsaId tup) {
    if (tup != kInvalidSsa && ctx.mod.get(tup).type == ValType::U32) {
        return tup;
    }
    SsaNode ev{};
    ev.op = SsaOp::TupleExtract;
    ev.type = vt;
    ev.imm_u32 = 0;
    ev.args = { tup };
    return ctx.sb.emit(blk, std::move(ev));
}

SsaId tuple_tok(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, SsaId tup) {
    if (tup != kInvalidSsa && ctx.mod.get(tup).type == ValType::U32) {
        return ctx.sb.current_token(blk);
    }
    SsaNode et{};
    et.op = SsaOp::TupleExtract;
    et.type = ValType::Token;
    et.imm_u32 = 1;
    et.args = { tup };
    return ctx.sb.emit(blk, std::move(et));
}


void lea_pka(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId off32) {
    std::string buf_sym;
    SsaId dyn_desc_idx = kInvalidSsa;
    if (ins.srcs.size() >= 2) {
        if (ins.srcs[1].kind == SourceKind::FauUniform) {
            uint32_t word_off = static_cast<uint32_t>(ins.srcs[1].fau_index64) * 8u +
                                (ins.srcs[1].word_select ? 4u : 0u);
            auto it = ctx.opts.fau_pka_buf_names.find(word_off);
            if (it != ctx.opts.fau_pka_buf_names.end() && !it->second.empty()) {
                buf_sym = it->second;
            } else {
                auto fit = ctx.opts.fau_fcst_words.find(word_off);
                if (fit != ctx.opts.fau_fcst_words.end()) {
                    auto sit = ctx.opts.srt_pka_buf_names.find(fit->second);
                    if (sit != ctx.opts.srt_pka_buf_names.end() && !sit->second.empty()) {
                        buf_sym = sit->second;
                    }
                }
            }
        } else if (ins.srcs[1].kind == SourceKind::SmallConstant) {
            auto sit = ctx.opts.srt_pka_buf_names.find(ins.srcs[1].imm_bits);
            if (sit != ctx.opts.srt_pka_buf_names.end() && !sit->second.empty()) {
                buf_sym = sit->second;
            }
        } else if (ins.srcs[1].kind == SourceKind::Register) {
            SsaId start_id = ctx.sb.current_reg(blk, ins.srcs[1].reg_num);
            std::vector<SsaId> work;
            std::vector<SsaId> seen;
            if (start_id != kInvalidSsa) work.push_back(start_id);
            bool has_const = false;
            bool conflict = false;
            uint32_t packed_desc = 0u;
            while (!work.empty() && !conflict && seen.size() < 32u) {
                SsaId cur = work.back();
                work.pop_back();
                if (cur == kInvalidSsa) continue;
                bool dup = false;
                for (SsaId v : seen) { if (v == cur) { dup = true; break; } }
                if (dup) continue;
                seen.push_back(cur);
                const auto& dn = ctx.res.ssa.get(cur);
                if ((dn.op == SsaOp::Copy || dn.op == SsaOp::Bitcast ||
                     dn.op == SsaOp::SubgroupShuffle) && !dn.args.empty()) {
                    work.push_back(dn.args[0]);
                } else if (dn.op == SsaOp::IAdd && dn.args.size() == 2u) {
                    auto resolve_base = [&](SsaId cand) -> bool {
                        if (cand == kInvalidSsa || cand >= ctx.res.ssa.nodes.size()) return false;
                        const auto& cn = ctx.res.ssa.get(cand);
                        uint32_t dval = 0u;
                        if (cn.op == SsaOp::LoadFau) {
                            auto fit = ctx.opts.fau_fcst_words.find(cn.imm_u32 * 4u);
                            if (fit == ctx.opts.fau_fcst_words.end()) return false;
                            dval = fit->second;
                        } else if (cn.op == SsaOp::ConstU32) {
                            dval = cn.imm_u32;
                        } else {
                            return false;
                        }
                        return ctx.opts.srt_pka_buf_names.count(dval) > 0u;
                    };
                    if (resolve_base(dn.args[0])) {
                        work.push_back(dn.args[0]);
                        dyn_desc_idx = dn.args[1];
                    } else if (resolve_base(dn.args[1])) {
                        work.push_back(dn.args[1]);
                        dyn_desc_idx = dn.args[0];
                    } else {
                        conflict = true;
                    }
                } else if (dn.op == SsaOp::LoadFau) {
                    uint32_t byte_off = dn.imm_u32 * 4u;
                    auto fit = ctx.opts.fau_fcst_words.find(byte_off);
                    if (fit != ctx.opts.fau_fcst_words.end()) {
                        if (!has_const) { has_const = true; packed_desc = fit->second; }
                        else if (packed_desc != fit->second) conflict = true;
                    } else {
                        conflict = true;
                    }
                } else if (dn.op == SsaOp::ConstU32) {
                    if (!has_const) { has_const = true; packed_desc = dn.imm_u32; }
                    else if (packed_desc != dn.imm_u32) conflict = true;
                } else {
                    conflict = true;
                }
            }
            if (has_const && !conflict) {
                auto it = ctx.opts.srt_pka_buf_names.find(packed_desc);
                if (it != ctx.opts.srt_pka_buf_names.end() && !it->second.empty()) {
                    buf_sym = it->second;
                }
            }
        }
    }
    if (!buf_sym.empty()) {
        SsaNode base_n{};
        base_n.op = SsaOp::LoadSidebandU64;
        base_n.type = ValType::U64;
        base_n.imm_u32 = 0x10u;
        base_n.aux_str = buf_sym;
        base_n.effects.flags |= EffectSet::kFlagUsesSideband;
        ctx.res.uses_sideband = true;
        SsaId base_id = ctx.sb.emit(blk, std::move(base_n));
        if (dyn_desc_idx != kInvalidSsa) {
            SsaId idx64 = ctx.emit_op(blk, SsaOp::UConvert32To64, ValType::U64, { dyn_desc_idx });
            SsaId sh24 = ctx.sb.emit_const_u64(blk, 24ull);
            SsaId stride = ctx.emit_op(blk, SsaOp::Shl64, ValType::U64, { idx64, sh24 });
            base_id = ctx.emit_op(blk, SsaOp::IAdd64, ValType::U64, { base_id, stride });
        }
        SsaId off64 = ctx.emit_op(blk, SsaOp::UConvert32To64, ValType::U64, { off32 });
        SsaId eff = ctx.emit_op(blk, SsaOp::IAdd64, ValType::U64, { base_id, off64 });
        SsaId lo = kInvalidSsa, hi = kInvalidSsa;
        ctx.emit_split64(blk, eff, &lo, &hi);
        sr_set(ctx, blk, ins, rule, 0u, lo);
        sr_set(ctx, blk, ins, rule, 1u, hi);
        return;
    }
    sr_set(ctx, blk, ins, rule, 0u, off32);
    sr_set(ctx, blk, ins, rule, 1u, ctx.sb.emit_const_u32(blk, 0xEEEEEEEEu));
}

SsaId psb_load(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId addr, uint32_t elem_bytes, bool is_pka, SsaId tok) {
    SsaId eff_addr = addr;
    std::string buf_sym;
    SsaId dyn_desc_idx = kInvalidSsa;
    if (is_pka && ins.srcs.size() >= 2) {
        if (ins.pc == ctx.last_pka_pc && ctx.last_pka_table_hi != kInvalidSsa) {
            return ctx.last_pka_table_hi;
        }
        if (ins.pc != ctx.last_pka_pc) {
            ctx.last_pka_table_hi = kInvalidSsa;
        }
        // Check for SRT descriptor-table load: LD_PKA.i64 with mode descriptor 0x3e000000 | rack.
        if (ins.pc != ctx.last_pka_pc && elem_bytes == 4u &&
            ins.srcs[1].kind == SourceKind::FauUniform) {
            uint32_t word_off = static_cast<uint32_t>(ins.srcs[1].fau_index64) * 8u +
                                (ins.srcs[1].word_select ? 4u : 0u);
            auto fit = ctx.opts.fau_fcst_words.find(word_off);
            if (fit != ctx.opts.fau_fcst_words.end() && (fit->second & 0xFE000000u) == 0x3E000000u) {
                uint32_t rack = fit->second & 0xFFu;
                // Unwrap UConvert32To64 / Copy / Bitcast on addr to recover base_off (+ dynamic bidx<<5).
                SsaId cur = addr;
                for (int depth = 0; depth < 6 && cur != kInvalidSsa; ++depth) {
                    const auto& n = ctx.mod.get(cur);
                    if ((n.op == SsaOp::UConvert32To64 || n.op == SsaOp::SConvert32To64 ||
                         n.op == SsaOp::Copy || n.op == SsaOp::Bitcast) && !n.args.empty()) {
                        cur = n.args[0];
                    } else {
                        break;
                    }
                }
                bool has_base_off = false;
                uint32_t base_off = 0u;
                SsaId dyn_off = kInvalidSsa;
                if (cur != kInvalidSsa) {
                    const auto& n = ctx.mod.get(cur);
                    if (n.op == SsaOp::ConstU32) {
                        has_base_off = true;
                        base_off = n.imm_u32;
                    } else if (n.op == SsaOp::IAdd && n.args.size() >= 2) {
                        const auto& a0 = ctx.mod.get(n.args[0]);
                        const auto& a1 = ctx.mod.get(n.args[1]);
                        if (a1.op == SsaOp::ConstU32) {
                            has_base_off = true;
                            base_off = a1.imm_u32;
                            dyn_off = n.args[0];
                        } else if (a0.op == SsaOp::ConstU32) {
                            has_base_off = true;
                            base_off = a0.imm_u32;
                            dyn_off = n.args[1];
                        }
                    }
                }
                if (has_base_off && (base_off & 0x1Fu) == 8u) {
                    uint32_t desc_idx = base_off >> 5u;
                    uint32_t packed_desc = (rack << 24u) | (desc_idx & 0x00FFFFFFu);
                    auto sit = ctx.opts.srt_pka_buf_names.find(packed_desc);
                    if (sit != ctx.opts.srt_pka_buf_names.end() && !sit->second.empty()) {
                        SsaNode base_n{};
                        base_n.op = SsaOp::LoadSidebandU64;
                        base_n.type = ValType::U64;
                        base_n.imm_u32 = 0x10u;
                        base_n.aux_str = sit->second;
                        base_n.effects.flags |= EffectSet::kFlagUsesSideband;
                        ctx.res.uses_sideband = true;
                        SsaId ptr64 = ctx.sb.emit(blk, std::move(base_n));
                        if (dyn_off != kInvalidSsa) {
                            SsaId bidx32 = ctx.emit_op(blk, SsaOp::ShrU, ValType::U32,
                                                       { dyn_off, ctx.sb.emit_const_u32(blk, 5u) });
                            SsaId bidx64 = ctx.emit_op(blk, SsaOp::UConvert32To64, ValType::U64, { bidx32 });
                            SsaId slice_off = ctx.emit_op(blk, SsaOp::Shl64, ValType::U64,
                                                          { bidx64, ctx.sb.emit_const_u64(blk, 24u) });
                            ptr64 = ctx.emit_op(blk, SsaOp::IAdd64, ValType::U64, { ptr64, slice_off });
                        }
                        SsaId lo = kInvalidSsa, hi = kInvalidSsa;
                        ctx.emit_split64(blk, ptr64, &lo, &hi);
                        ctx.last_pka_pc = ins.pc;
                        ctx.last_pka_buf_sym = sit->second;
                        ctx.last_pka_table_hi = hi;
                        return lo;
                    }
                }
            }
        }
        if (ins.pc == ctx.last_pka_pc) {
            buf_sym = ctx.last_pka_buf_sym;
        } else if (ins.srcs[1].kind == SourceKind::FauUniform) {
            uint32_t word_off = static_cast<uint32_t>(ins.srcs[1].fau_index64) * 8u +
                                (ins.srcs[1].word_select ? 4u : 0u);
            auto it = ctx.opts.fau_pka_buf_names.find(word_off);
            if (it != ctx.opts.fau_pka_buf_names.end() && !it->second.empty()) {
                buf_sym = it->second;
            }
        } else if (ins.srcs[1].kind == SourceKind::SmallConstant) {
                auto sit = ctx.opts.srt_pka_buf_names.find(ins.srcs[1].imm_bits);
                if (sit != ctx.opts.srt_pka_buf_names.end() && !sit->second.empty()) {
                    buf_sym = sit->second;
                }
            } else if (ins.srcs[1].kind == SourceKind::Register) {
            // Walk Copy/Bitcast/Phi chains (including cross-block Phi merges) to recover
            // a unique ConstU32 packed SRT descriptor `(rack << 24) | desc_idx`.
            SsaId start_id = ctx.sb.current_reg(blk, ins.srcs[1].reg_num);
            std::vector<SsaId> work;
            std::vector<SsaId> seen;
            if (start_id != kInvalidSsa) work.push_back(start_id);
            bool has_const = false;
            bool conflict = false;
            uint32_t packed_desc = 0u;
            while (!work.empty() && !conflict && seen.size() < 32u) {
                SsaId cur = work.back();
                work.pop_back();
                if (cur == kInvalidSsa) continue;
                bool dup = false;
                for (SsaId v : seen) { if (v == cur) { dup = true; break; } }
                if (dup) continue;
                seen.push_back(cur);
                const auto& dn = ctx.res.ssa.get(cur);
                if ((dn.op == SsaOp::Copy || dn.op == SsaOp::Bitcast ||
                     dn.op == SsaOp::SubgroupShuffle) && !dn.args.empty()) {
                    work.push_back(dn.args[0]);
                } else if (dn.op == SsaOp::IAdd && dn.args.size() == 2u) {
                    auto resolve_base = [&](SsaId cand) -> bool {
                        if (cand == kInvalidSsa || cand >= ctx.res.ssa.nodes.size()) return false;
                        const auto& cn = ctx.res.ssa.get(cand);
                        uint32_t dval = 0u;
                        if (cn.op == SsaOp::LoadFau) {
                            auto fit = ctx.opts.fau_fcst_words.find(cn.imm_u32 * 4u);
                            if (fit == ctx.opts.fau_fcst_words.end()) return false;
                            dval = fit->second;
                        } else if (cn.op == SsaOp::ConstU32) {
                            dval = cn.imm_u32;
                        } else {
                            return false;
                        }
                        return ctx.opts.srt_pka_buf_names.count(dval) > 0u;
                    };
                    if (resolve_base(dn.args[0])) {
                        work.push_back(dn.args[0]);
                        dyn_desc_idx = dn.args[1];
                    } else if (resolve_base(dn.args[1])) {
                        work.push_back(dn.args[1]);
                        dyn_desc_idx = dn.args[0];
                    } else {
                        conflict = true;
                    }
                } else if (dn.op == SsaOp::Phi) {
                    bool any_inc = false;
                    for (const auto& inc : dn.phi_incomings) {
                        if (inc.value != kInvalidSsa) { work.push_back(inc.value); any_inc = true; }
                    }
                    for (SsaId a : dn.args) {
                        if (a != kInvalidSsa) { work.push_back(a); any_inc = true; }
                    }
                    // Pass 1 seals Phi incomings after all blocks are lowered; walk already-lowered
                    // forward predecessors via `block_out_regs` to recover loop-invariant descriptors.
                    if (!any_inc && dn.block_id < ctx.res.ssa.blocks.size()) {
                        for (uint32_t pred : ctx.res.ssa.blocks[dn.block_id].preds) {
                            if (pred < dn.block_id) {
                                uint32_t phi_reg = (dn.aux_index < 64u) ? dn.aux_index : ins.srcs[1].reg_num;
                                SsaId pv = ctx.sb.current_reg(pred, phi_reg);
                                if (pv != kInvalidSsa) work.push_back(pv);
                            }
                        }
                    }
                } else if (dn.op == SsaOp::LoadFau) {
                    uint32_t byte_off = dn.imm_u32 * 4u;
                    auto fit = ctx.opts.fau_fcst_words.find(byte_off);
                    if (fit != ctx.opts.fau_fcst_words.end()) {
                        if (!has_const) {
                            has_const = true;
                            packed_desc = fit->second;
                        } else if (packed_desc != fit->second) {
                            conflict = true;
                        }
                    } else {
                        conflict = true;
                    }
                } else if (dn.op == SsaOp::ConstU32) {
                    if (!has_const) {
                        has_const = true;
                        packed_desc = dn.imm_u32;
                    } else if (packed_desc != dn.imm_u32) {
                        conflict = true;
                    }
                } else {
                    conflict = true;
                }
            }
            if (has_const && !conflict) {
                auto it = ctx.opts.srt_pka_buf_names.find(packed_desc);
                if (it != ctx.opts.srt_pka_buf_names.end() && !it->second.empty()) {
                    buf_sym = it->second;
                }
            }
        }
        if (ins.pc != ctx.last_pka_pc) {
            ctx.last_pka_pc = ins.pc;
            ctx.last_pka_buf_sym = buf_sym;
        }
        if (!buf_sym.empty()) {
            SsaNode base_n{};
            base_n.op = SsaOp::LoadSidebandU64;
            base_n.type = ValType::U64;
            base_n.imm_u32 = 0x10u;
            base_n.aux_str = buf_sym;
            base_n.effects.flags |= EffectSet::kFlagUsesSideband;
            ctx.res.uses_sideband = true;
            SsaId base_id = ctx.sb.emit(blk, std::move(base_n));
            if (dyn_desc_idx != kInvalidSsa) {
                SsaId idx64 = ctx.emit_op(blk, SsaOp::UConvert32To64, ValType::U64, { dyn_desc_idx });
                SsaId sh24 = ctx.sb.emit_const_u64(blk, 24ull);
                SsaId stride = ctx.emit_op(blk, SsaOp::Shl64, ValType::U64, { idx64, sh24 });
                base_id = ctx.emit_op(blk, SsaOp::IAdd64, ValType::U64, { base_id, stride });
            }
            eff_addr = ctx.emit_op(blk, SsaOp::IAdd64, ValType::U64, { base_id, addr });
        }
    }
    SsaNode ld{};
    ld.op = SsaOp::PsbLoad;
    ld.type = ValType::TupleValToken;
    ld.imm_u32 = elem_bytes;
    ld.aux_bool = is_pka;
    ld.aux_str = std::move(buf_sym);
    ld.args = { eff_addr, tok };
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

SsaId load_vert_attr(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    uint32_t attr_idx = 0;
    auto it = ins.fields.find("attribute_index");
    if (it != ins.fields.end()) attr_idx = it->second;
    if (auto mit = ctx.opts.vert_attr_idx_to_loc.find(attr_idx);
        mit != ctx.opts.vert_attr_idx_to_loc.end()) {
        attr_idx = mit->second;
    }
    SsaNode ld{};
    ld.op = SsaOp::LoadVertAttr;
    ld.type = ValType::V4F32;
    ld.imm_u32 = attr_idx;
    if (ins.mnemonic == Mnemonic::LD_ATTR && !ins.srcs.empty()) {
        uint32_t dyn_si = static_cast<uint32_t>(ins.srcs.size() - 1u);
        SsaId raw_idx = fetch_src_int32(ctx, blk, ins, rule, dyn_si, false);
        SsaId masked = ctx.emit_op(blk, SsaOp::BitAnd, ValType::U32, {raw_idx, ctx.sb.emit_const_u32(blk, 0xFFFFu)});
        ld.args = { masked };
    }
    ld.is_barrier_or_cf = true;
    return ctx.sb.emit(blk, std::move(ld));
}

SsaId load_frag_var(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, bool is_flat) {
    uint32_t var_loc = 0;
    auto it = ins.fields.find("varying_index");
    if (it != ins.fields.end()) {
        var_loc = it->second;
    } else if (auto it2 = ins.fields.find("index"); it2 != ins.fields.end()) {
        var_loc = it2->second;
    }
    if (auto mit = ctx.opts.frag_varying_addr_to_loc.find(var_loc);
        mit != ctx.opts.frag_varying_addr_to_loc.end()) {
        var_loc = mit->second;
    }
    SsaNode ld{};
    ld.op = SsaOp::LoadFragVarying;
    ld.type = ValType::V4F32;
    ld.imm_u32 = var_loc;
    ld.aux_bool = is_flat;
    if ((ins.mnemonic == Mnemonic::LD_VAR || ins.mnemonic == Mnemonic::LD_VAR_FLAT) && !ins.srcs.empty()) {
        uint32_t dyn_si = static_cast<uint32_t>(ins.srcs.size() - 1u);
        SsaId raw_idx = fetch_src_int32(ctx, blk, ins, rule, dyn_si, false);
        SsaId masked = ctx.emit_op(blk, SsaOp::BitAnd, ValType::U32, {raw_idx, ctx.sb.emit_const_u32(blk, 0xFFFFu)});
        ld.args = { masked };
    }
    ld.is_barrier_or_cf = true;
    return ctx.sb.emit(blk, std::move(ld));
}

SsaId load_frag_var_idx(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, uint32_t var_loc, bool is_flat) {
    if (auto mit = ctx.opts.frag_varying_addr_to_loc.find(var_loc);
        mit != ctx.opts.frag_varying_addr_to_loc.end()) {
        var_loc = mit->second;
    }
    SsaNode ld{};
    ld.op = SsaOp::LoadFragVarying;
    ld.type = ValType::V4F32;
    ld.imm_u32 = var_loc;
    ld.aux_bool = is_flat;
    ld.is_barrier_or_cf = true;
    return ctx.sb.emit(blk, std::move(ld));
}

SsaId load_frag_special(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*) {
    // isa-v9-v14.xml encodes LD_VAR_SPECIAL's selector in field "varying_name" (type VaryingNameM).
    // Fall back to "index" only for synthetic/legacy instruction encodings.
    auto varying_name = VaryingNameM::frag_w;
    if (auto it = ins.fields.find("varying_name"); it != ins.fields.end()) {
        varying_name = static_cast<VaryingNameM>(it->second);
    } else if (auto it = ins.fields.find("index"); it != ins.fields.end()) {
        varying_name = static_cast<VaryingNameM>(it->second);
    }

    switch (varying_name) {
        case VaryingNameM::frag_w:
        case VaryingNameM::frag_z: {
            SsaId fc = ctx.emit_builtin_input(blk, spirv::BuiltIn::FragCoord, ValType::V4F32, "gl_FragCoord", false);
            SsaNode ex{};
            ex.op = SsaOp::VecExtract;
            ex.type = ValType::F32;
            ex.imm_u32 = (varying_name == VaryingNameM::frag_w) ? 3u : 2u;
            ex.aux_bool = (varying_name == VaryingNameM::frag_w &&
                           ((ctx.opts.is_panfrost && !ctx.opts.has_frag_noperspective_varyings) ||
                            (!ins.srcs.empty() && ins.srcs[0].kind == SourceKind::Register)));
            ex.args = { fc };
            return ctx.sb.emit(blk, std::move(ex));
        }
        case VaryingNameM::bary: {
            // Arch v14+ VaryingNameM::bary -> SPV_KHR_fragment_shader_barycentric BuiltIn BaryCoordKHR (5286).
            SsaId bc = ctx.emit_builtin_input(blk, static_cast<spirv::BuiltIn>(5286u), ValType::V3U32, "gl_BaryCoordEXT", false);
            SsaNode ex{};
            ex.op = SsaOp::VecExtract;
            ex.type = ValType::F32;
            ex.imm_u32 = 0u;
            ex.args = { bc };
            return ctx.sb.emit(blk, std::move(ex));
        }
        case VaryingNameM::point:
        default: {
            SsaId pc = ctx.emit_builtin_input(blk, spirv::BuiltIn::PointCoord, ValType::V2F32, "gl_PointCoord", false);
            SsaNode ex{};
            ex.op = SsaOp::VecExtract;
            ex.type = ValType::F32;
            ex.imm_u32 = 0u;
            ex.args = { pc };
            return ctx.sb.emit(blk, std::move(ex));
        }
    }
}

SsaId tex_sample(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, SsaId uv, bool is_gather, SsaId tok) {
    uint32_t gcomp = 0;
    if (is_gather) {
        auto it_gc = ins.fields.find("gather_component");
        if (it_gc != ins.fields.end()) gcomp = it_gc->second & 3u;
    }
    uint32_t tex_idx = 0;
    bool has_tex_idx = false;
    if (auto it_idx = ins.fields.find("index"); it_idx != ins.fields.end()) {
        tex_idx = it_idx->second;
        has_tex_idx = true;
    } else if (auto it_tidx = ins.fields.find("texture_index"); it_tidx != ins.fields.end()) {
        tex_idx = it_tidx->second;
        has_tex_idx = true;
    } else if (!ins.srcs.empty() && ins.srcs[0].kind == SourceKind::FauUniform) {
        uint32_t fau_off = static_cast<uint32_t>(ins.srcs[0].fau_index64) * 8u;
        auto it_fau = ctx.opts.fau_tex_handle_indices.find(fau_off);
        if (it_fau != ctx.opts.fau_tex_handle_indices.end()) {
            tex_idx = it_fau->second;
            has_tex_idx = true;
        }
    } else if (!ins.srcs.empty() && ins.srcs[0].kind == SourceKind::Register) {
        SsaId hi_id = ctx.sb.current_reg(blk, (ins.srcs[0].reg_num & 62u) + 1u);
        if (hi_id != kInvalidSsa) {
            const auto& hn = ctx.mod.get(hi_id);
            if (hn.op == SsaOp::ConstU32 && (hn.imm_u32 >> 24) == 0x04u) {
                tex_idx = hn.imm_u32 & 0xFFFFu;
                has_tex_idx = true;
            }
        }
        if (!has_tex_idx) {
            SsaId lo_id = ctx.sb.current_reg(blk, ins.srcs[0].reg_num & 62u);
            if (lo_id != kInvalidSsa) {
                const auto& ln = ctx.mod.get(lo_id);
                if (ln.op == SsaOp::ConstU32) {
                    uint32_t cand = (ln.imm_u32 >> 16) & 0x7FFu;
                    for (const auto& s : ctx.opts.samplers) {
                        if (s.tex_index == cand) {
                            tex_idx = cand;
                            has_tex_idx = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    SsaNode ts{};
    bool is_grad = (ins.mnemonic == Mnemonic::TEX_GRADIENT ||
                    ins.mnemonic == Mnemonic::VAR_TEX_GRADIENT ||
                    ins.mnemonic == Mnemonic::VAR_TEX_BUF_GRADIENT);
    bool is_fetch = (ins.mnemonic == Mnemonic::TEX_FETCH ||
                     ins.mnemonic == Mnemonic::LD_TEX ||
                     ins.mnemonic == Mnemonic::LD_TEX_IMM);
    bool is_dref = false;
    if (auto it = ins.modifier_strs.find("compare_enable"); it != ins.modifier_strs.end() && it->second == "compare") {
        is_dref = true;
    }
    bool is_explicit_lod = false;
    bool is_zero_lod = false;
    bool is_grdesc = false;
    if (auto it = ins.modifier_strs.find("lod_mode"); it != ins.modifier_strs.end()) {
        if (it->second == "explicit") is_explicit_lod = true;
        else if (it->second == "grdesc") is_grdesc = true;
        else if (it->second == "zero" || it->second == "none") is_zero_lod = true;
    }
    if (!is_explicit_lod && ctx.opts.stage != ShaderStage::Fragment) {
        is_zero_lod = true;
    }
    if (is_grad) ts.op = SsaOp::ImageQueryLod;
    else if (is_fetch) ts.op = SsaOp::TexFetch;
    else if (is_gather) ts.op = SsaOp::TexGather;
    else if (is_dref) ts.op = SsaOp::TexSampleDref;
    else if (is_explicit_lod || is_zero_lod) ts.op = SsaOp::TexSampleLod;
    else ts.op = SsaOp::TexSample;
    bool is_cube = false;
    if (auto it = ins.modifier_strs.find("dimensionality"); it != ins.modifier_strs.end() && it->second == "cube") {
        is_cube = true;
    } else if (auto it = ins.modifier_strs.find("dimension"); it != ins.modifier_strs.end() && it->second == "cube") {
        is_cube = true;
    }
    ts.type = ValType::TupleValToken;
    ts.imm_u32 = tex_idx;
    ts.imm_u32_2 = (is_cube && (ts.op == SsaOp::TexSample || ts.op == SsaOp::TexSampleLod)) ? 1u : gcomp;
    if (!has_tex_idx && ctx.opts.samplers.size() == 1u) {
        tex_idx = ctx.opts.samplers[0].tex_index;
        has_tex_idx = true;
    }
    if (has_tex_idx) {
        uint32_t d_set = 0;
        uint32_t binding = tex_idx;
        for (const auto& s : ctx.opts.samplers) {
            if (s.tex_index == tex_idx) {
                d_set = s.descriptor_set;
                binding = s.binding;
                break;
            }
        }
        ts.aux_str = "tex:" + std::to_string(d_set) + ":" + std::to_string(binding);
    }
    ts.aux_bool = (ctx.opts.stage == ShaderStage::Fragment && !is_gather);
    ts.aux_bool2 = is_gather;
    bool has_texel_offset = false;
    if (auto it = ins.modifier_strs.find("texel_offset");
        it != ins.modifier_strs.end() && it->second != "none") {
        has_texel_offset = true;
    }
    ts.args = { uv };
    uint32_t sr_cnt = abi::sr_read_count(ctx, blk, ins, nullptr);
    uint32_t max_sr = std::min<uint32_t>(sr_cnt, 8u);
    uint32_t extra_end = (has_texel_offset && max_sr > 2u) ? (max_sr - 1u) : max_sr;
    bool resolved_grdesc = false;
    if (is_grdesc && extra_end >= 4u) {
        SsaId g0 = abi::sr_get(ctx, blk, ins, nullptr, 2u);
        SsaId cur = g0;
        auto is_zero_const = [&](SsaId id) -> bool {
            if (id == kInvalidSsa || id >= ctx.mod.nodes.size()) return false;
            const auto& cn = ctx.mod.get(id);
            return (cn.op == SsaOp::ConstU32 && cn.imm_u32 == 0u) ||
                   (cn.op == SsaOp::ConstU64 && cn.imm_u64 == 0u);
        };
        for (int depth = 0; depth < 12 && cur != kInvalidSsa; ++depth) {
            if (ctx.grad_desc_map.count(cur)) break;
            const auto& n = ctx.mod.get(cur);
            if ((n.op == SsaOp::Bitcast || n.op == SsaOp::Copy ||
                 n.op == SsaOp::VecExtract || n.op == SsaOp::TupleExtract ||
                 n.op == SsaOp::Split64 || n.op == SsaOp::Pack64) && !n.args.empty()) {
                cur = n.args[0];
            } else if ((n.op == SsaOp::IAdd64 || n.op == SsaOp::IAdd) && n.args.size() == 2u) {
                if (is_zero_const(n.args[1])) {
                    cur = n.args[0];
                } else if (is_zero_const(n.args[0])) {
                    cur = n.args[1];
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        auto git = ctx.grad_desc_map.find(cur);
        if (git != ctx.grad_desc_map.end()) {
            ts.op = SsaOp::TexSampleLod;
            for (SsaId gv : git->second) {
                ts.args.push_back(gv);
            }
            resolved_grdesc = true;
        }
    }
    if (!resolved_grdesc && extra_end > 2u &&
        ins.mnemonic != Mnemonic::LD_TEX && ins.mnemonic != Mnemonic::LD_TEX_IMM) {
        for (uint32_t sr_i = 2u; sr_i < extra_end; ++sr_i) {
            SsaId sr_val = abi::sr_get(ctx, blk, ins, nullptr, sr_i);
            if (is_explicit_lod && !is_dref && sr_i == extra_end - 1u) {
                SsaId lod_s16 = ctx.emit_op(blk, SsaOp::BfeS, ValType::S32,
                                            {sr_val, ctx.sb.emit_const_u32(blk, 0u),
                                             ctx.sb.emit_const_u32(blk, 16u)});
                SsaId lod_f = ctx.emit_op(blk, SsaOp::ConvertSToF, ValType::F32, {lod_s16});
                SsaId inv256 = ctx.sb.emit_const_f32_bits(blk, 0x3b800000u);
                sr_val = ctx.emit_op(blk, SsaOp::FMul, ValType::F32, {lod_f, inv256});
            }
            if (is_cube && sr_i == 2u) {
                sr_val = ctx.sb.emit_bitcast(blk, ValType::F32, sr_val);
            }
            ts.args.push_back(sr_val);
        }
    }
    if ((ts.op == SsaOp::TexSampleLod && is_zero_lod && !is_explicit_lod) ||
        (ins.mnemonic == Mnemonic::TEX_FETCH && ts.args.size() == 1u)) {
        ts.args.push_back(ctx.sb.emit_const_u32(blk, 0u));
    }
    if (has_texel_offset && max_sr > 2u &&
        ins.mnemonic != Mnemonic::LD_TEX && ins.mnemonic != Mnemonic::LD_TEX_IMM) {
        SsaId packed_off = abi::sr_get(ctx, blk, ins, nullptr, max_sr - 1u);
        SsaId zero_u32 = ctx.sb.emit_const_u32(blk, 0u);
        SsaId eight_u32 = ctx.sb.emit_const_u32(blk, 8u);
        SsaId dx = ctx.emit_op(blk, SsaOp::BfeS, ValType::S32, {packed_off, zero_u32, eight_u32});
        SsaId dy = ctx.emit_op(blk, SsaOp::BfeS, ValType::S32, {packed_off, eight_u32, eight_u32});
        ts.args.push_back(dx);
        ts.args.push_back(dy);
    }
    ts.args.push_back(tok);
    ts.is_barrier_or_cf = true;
    ts.is_side_effect = true;
    SsaNode ts_template = ts;
    SsaId ts_id = ctx.sb.emit(blk, std::move(ts));
    if (ins.mnemonic == Mnemonic::TEX_DUAL || ins.mnemonic == Mnemonic::VAR_TEX_DUAL) {
        uint32_t pmask = 0xFu;
        if (auto it_pm = ins.fields.find("primary_write_mask"); it_pm != ins.fields.end() && it_pm->second != 0u) {
            pmask = it_pm->second & 0xFu;
        }
        uint32_t smask = 0xFu;
        if (auto it_sm = ins.fields.find("secondary_write_mask"); it_sm != ins.fields.end() && it_sm->second != 0u) {
            smask = it_sm->second & 0xFu;
        }
        SsaNode pval{};
        pval.op = SsaOp::TupleExtract;
        pval.type = ValType::V4F32;
        pval.imm_u32 = 0u;
        pval.args = { ts_id };
        SsaId pval_id = ctx.sb.emit(blk, std::move(pval));

        SsaNode ptok{};
        ptok.op = SsaOp::TupleExtract;
        ptok.type = ValType::Token;
        ptok.imm_u32 = 1u;
        ptok.args = { ts_id };
        SsaId ptok_id = ctx.sb.emit(blk, std::move(ptok));

        uint32_t sr_out = 0u;
        for (uint32_t c = 0u; c < 4u; ++c) {
            if ((pmask & (1u << c)) == 0u) continue;
            SsaNode ex{};
            ex.op = SsaOp::VecExtract;
            ex.type = ValType::F32;
            ex.imm_u32 = c;
            ex.args = { pval_id };
            SsaId f_id = ctx.sb.emit(blk, std::move(ex));
            abi::sr_set(ctx, blk, ins, nullptr, sr_out++, ctx.sb.emit_bitcast(blk, ValType::U32, f_id));
        }

        uint32_t sec_tex_idx = 0u;
        bool has_sec_tex = false;
        if (!ins.srcs.empty() && ins.srcs[0].kind == SourceKind::FauUniform) {
            uint32_t fau_off2 = static_cast<uint32_t>(ins.srcs[0].fau_index64) * 8u + 4u;
            auto it_fau2 = ctx.opts.fau_tex_handle_indices.find(fau_off2);
            if (it_fau2 != ctx.opts.fau_tex_handle_indices.end()) {
                sec_tex_idx = it_fau2->second;
                has_sec_tex = true;
            }
        } else if (!ins.srcs.empty() && ins.srcs[0].kind == SourceKind::Register) {
            SsaId hi_id2 = ctx.sb.current_reg(blk, (ins.srcs[0].reg_num & 62u) + 1u);
            if (hi_id2 != kInvalidSsa) {
                const auto& hn2 = ctx.mod.get(hi_id2);
                if (hn2.op == SsaOp::ConstU32) {
                    uint32_t cand = (hn2.imm_u32 >> 16) & 0x7FFu;
                    for (const auto& s : ctx.opts.samplers) {
                        if (s.tex_index == cand) {
                            sec_tex_idx = cand;
                            has_sec_tex = true;
                            break;
                        }
                    }
                }
            }
        }
        SsaNode ts2 = std::move(ts_template);
        ts2.imm_u32 = sec_tex_idx;
        ts2.aux_str.clear();
        if (has_sec_tex) {
            uint32_t d_set2 = 0u;
            uint32_t binding2 = sec_tex_idx;
            for (const auto& s : ctx.opts.samplers) {
                if (s.tex_index == sec_tex_idx) {
                    d_set2 = s.descriptor_set;
                    binding2 = s.binding;
                    break;
                }
            }
            ts2.aux_str = "tex:" + std::to_string(d_set2) + ":" + std::to_string(binding2);
        }
        if (!ts2.args.empty()) ts2.args.back() = ptok_id;
        SsaId ts2_id = ctx.sb.emit(blk, std::move(ts2));

        SsaNode sval{};
        sval.op = SsaOp::TupleExtract;
        sval.type = ValType::V4F32;
        sval.imm_u32 = 0u;
        sval.args = { ts2_id };
        SsaId sval_id = ctx.sb.emit(blk, std::move(sval));
        for (uint32_t c = 0u; c < 4u; ++c) {
            if ((smask & (1u << c)) == 0u) continue;
            SsaNode ex{};
            ex.op = SsaOp::VecExtract;
            ex.type = ValType::F32;
            ex.imm_u32 = c;
            ex.args = { sval_id };
            SsaId f_id = ctx.sb.emit(blk, std::move(ex));
            abi::sr_set(ctx, blk, ins, nullptr, sr_out++, ctx.sb.emit_bitcast(blk, ValType::U32, f_id));
        }
        return ts2_id;
    }
    if (is_grad && sr_cnt >= 4u) {
        auto it_d = ins.modifier_strs.find("coordinate_or_derivative");
        if (it_d != ins.modifier_strs.end() && it_d->second == "derivative") {
            // Valhall TEX_GRADIENT.derivative SRs: (ddx.x, ddy.x, ddx.y, ddy.y) -> SPIR-V (ddx.x, ddx.y, ddy.x, ddy.y).
            SsaId ddx_x = abi::sr_get(ctx, blk, ins, nullptr, 0u);
            SsaId ddy_x = abi::sr_get(ctx, blk, ins, nullptr, 1u);
            SsaId ddx_y = abi::sr_get(ctx, blk, ins, nullptr, 2u);
            SsaId ddy_y = abi::sr_get(ctx, blk, ins, nullptr, 3u);
            ctx.grad_desc_map[ts_id] = { ddx_x, ddx_y, ddy_x, ddy_y };
        }
    }
    return ts_id;
}

SsaId lea_tex(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, SsaId xy, SsaId z_layer) {
    uint32_t tex_idx = 0;
    bool has_tex_idx = false;
    if (auto it_idx = ins.fields.find("index"); it_idx != ins.fields.end()) {
        tex_idx = it_idx->second;
        has_tex_idx = true;
    } else if (auto it_tidx = ins.fields.find("texture_index"); it_tidx != ins.fields.end()) {
        tex_idx = it_tidx->second;
        has_tex_idx = true;
    } else if (!ins.srcs.empty() && ins.srcs[0].kind == SourceKind::FauUniform) {
        uint32_t fau_off = static_cast<uint32_t>(ins.srcs[0].fau_index64) * 8u;
        auto it_fau = ctx.opts.fau_tex_handle_indices.find(fau_off);
        if (it_fau != ctx.opts.fau_tex_handle_indices.end()) {
            tex_idx = it_fau->second;
            has_tex_idx = true;
        }
    }
    if (!has_tex_idx && ctx.opts.samplers.size() == 1u) {
        tex_idx = ctx.opts.samplers[0].tex_index;
        has_tex_idx = true;
    }
    SsaNode n{};
    n.op = SsaOp::ImageTexelAddr;
    n.type = ValType::U64;
    n.imm_u32 = tex_idx;
    if (has_tex_idx) {
        uint32_t d_set = 0;
        uint32_t binding = tex_idx;
        for (const auto& s : ctx.opts.samplers) {
            if (s.tex_index == tex_idx) {
                d_set = s.descriptor_set;
                binding = s.binding;
                break;
            }
        }
        n.aux_str = "tex:" + std::to_string(d_set) + ":" + std::to_string(binding);
    }
    n.args = { xy, z_layer };
    n.is_barrier_or_cf = true;
    return ctx.sb.emit(blk, std::move(n));
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

SsaId store_vert_varying(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec, SsaId tok) {
    // Valhall Vertex Position shaders write gl_Position to the Position FIFO via STORE.estream k0, #0x1.
    if (!ins.srcs.empty() && ins.srcs[0].kind == SourceKind::SmallConstant && ins.imm_signed == 1) {
        return store_builtin_pos(ctx, blk, ins, rule, vec, tok);
    }
    uint32_t var_loc = static_cast<uint32_t>(std::max<int64_t>(0, ins.imm_signed) / 16);
    if (ins.mnemonic == Mnemonic::STORE && !ctx.opts.idvs_varying_offsets.empty()) {
        if (auto vit = ctx.opts.idvs_varying_offsets.find(ins.imm_signed);
            vit != ctx.opts.idvs_varying_offsets.end()) {
            const auto& vspec = vit->second;
            if (vspec.is_position) {
                return store_builtin_pos(ctx, blk, ins, rule, vec, tok);
            }
            if (vspec.is_point_size) {
                SsaNode ex0{}; ex0.op = SsaOp::VecExtract; ex0.type = ValType::F32; ex0.imm_u32 = 0; ex0.args = { vec };
                SsaId ps_f32 = ctx.sb.emit(blk, std::move(ex0));
                if (ins.variant == Variant::I16) {
                    SsaId raw_u32 = ctx.sb.emit_bitcast(blk, ValType::U32, ps_f32);
                    SsaId up2 = ctx.emit_op(blk, SsaOp::UnpackHalf2x16, ValType::V2F32, { raw_u32 });
                    SsaNode ex_ps{}; ex_ps.op = SsaOp::VecExtract; ex_ps.type = ValType::F32; ex_ps.imm_u32 = 0; ex_ps.args = { up2 };
                    ps_f32 = ctx.sb.emit(blk, std::move(ex_ps));
                }
                SsaNode st{};
                st.op = SsaOp::StoreBuiltinOutput;
                st.type = ValType::Token;
                st.imm_u32 = static_cast<uint32_t>(spirv::BuiltIn::PointSize);
                st.aux_str = "gl_PointSize";
                st.args = { ps_f32, tok };
                st.is_barrier_or_cf = true;
                st.is_side_effect = true;
                return ctx.sb.emit(blk, std::move(st));
            }
            if (vspec.user_location >= 0) {
                var_loc = static_cast<uint32_t>(vspec.user_location);
                if (vspec.ncomps > 0u && vspec.ncomps < 4u) {
                    SsaId zf = ctx.sb.emit_const_f32_bits(blk, 0u);
                    std::vector<SsaId> lanes(4, zf);
                    for (uint32_t c = 0; c < vspec.ncomps; ++c) {
                        SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = { vec };
                        lanes[c] = ctx.sb.emit(blk, std::move(ex));
                    }
                    SsaNode vc{}; vc.op = SsaOp::VecConstruct4; vc.type = ValType::V4F32; vc.args = std::move(lanes);
                    vec = ctx.sb.emit(blk, std::move(vc));
                }
            }
        }
    }
    if (ins.mnemonic == Mnemonic::ST_CVT) {
        uint32_t attr_idx = 0xFFFFFFFFu;
        if (!ins.srcs.empty() && ins.srcs[0].kind == SourceKind::Register) {
            SsaId addr_lo = ctx.sb.current_reg(blk, ins.srcs[0].reg_num & 63u);
            if (addr_lo < ctx.res.ssa.nodes.size() && ctx.res.ssa.get(addr_lo).op == SsaOp::ConstU32) {
                attr_idx = ctx.res.ssa.get(addr_lo).imm_u32 / 16u;
            }
        }
        bool is_f16 = false;
        if (auto it = ins.modifier_strs.find("register_format"); it != ins.modifier_strs.end() && it->second == "f16") {
            is_f16 = true;
        }
        std::string sout_name;
        if (attr_idx != 0xFFFFFFFFu) {
            if (auto sit = ctx.opts.vert_sout_attr_names.find(attr_idx); sit != ctx.opts.vert_sout_attr_names.end()) {
                sout_name = sit->second;
            }
        }
        if (sout_name.find("PointSize") != std::string::npos || (is_f16 && sout_name.empty())) {
            SsaNode ex0{}; ex0.op = SsaOp::VecExtract; ex0.type = ValType::F32; ex0.imm_u32 = 0; ex0.args = { vec };
            SsaId raw_u32 = ctx.sb.emit_bitcast(blk, ValType::U32, ctx.sb.emit(blk, std::move(ex0)));
            SsaId up2 = ctx.emit_op(blk, SsaOp::UnpackHalf2x16, ValType::V2F32, { raw_u32 });
            SsaNode ex_ps{}; ex_ps.op = SsaOp::VecExtract; ex_ps.type = ValType::F32; ex_ps.imm_u32 = 0; ex_ps.args = { up2 };
            SsaId ps_f32 = ctx.sb.emit(blk, std::move(ex_ps));
            SsaNode st{};
            st.op = SsaOp::StoreBuiltinOutput;
            st.type = ValType::Token;
            st.imm_u32 = static_cast<uint32_t>(spirv::BuiltIn::PointSize);
            st.aux_str = "gl_PointSize";
            st.args = { ps_f32, tok };
            st.is_barrier_or_cf = true;
            st.is_side_effect = true;
            return ctx.sb.emit(blk, std::move(st));
        }
        if (sout_name.find("PositionFifoAttributes") != std::string::npos) {
            return tok;
        }
        if (sout_name.find("XFB_Position") != std::string::npos) {
            return store_builtin_pos(ctx, blk, ins, rule, vec, tok);
        }
        if (attr_idx != 0xFFFFFFFFu) {
            var_loc = attr_idx;
        }
    }
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
    } else if (!ins.srcs.empty() && ins.srcs[0].kind == SourceKind::FauUniform) {
        auto it = ctx.opts.fau_blend_rt_indices.find(static_cast<uint32_t>(ins.srcs[0].fau_index64));
        if (it != ctx.opts.fau_blend_rt_indices.end()) {
            rt_loc = it->second;
        }
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
    bool s0_signed = (var.find("s") != std::string::npos);
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
    // BRANCH handles its combine modifier (and/h0/h1/lowbits) explicitly in its lowering rule;
    // suppress src0 swizzle here so .h1 is not shifted right by 16 twice.
    if (src_idx == 0 && (ins.mnemonic == Mnemonic::BRANCH || ins.canonical_mnemonic == Mnemonic::BRANCH)) {
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
    if ((ins.mnemonic == Mnemonic::FMUL_RSCALE || ins.canonical_mnemonic == Mnemonic::FMUL_RSCALE ||
         ins.mnemonic == Mnemonic::FADD_RSCALE || ins.canonical_mnemonic == Mnemonic::FADD_RSCALE) && src_idx == 2u) return false;
    if ((ins.mnemonic == Mnemonic::FABS || ins.canonical_mnemonic == Mnemonic::FABS) && src_idx == 0u) return false;
    if (src_idx < ins.srcs.size() && ins.srcs[src_idx].abs) return true;
    auto it = ins.fields.find("abs" + std::to_string(src_idx));
    return (it != ins.fields.end() && it->second != 0);
}

bool get_src_neg(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    if ((ins.mnemonic == Mnemonic::FMUL_RSCALE || ins.canonical_mnemonic == Mnemonic::FMUL_RSCALE ||
         ins.mnemonic == Mnemonic::FADD_RSCALE || ins.canonical_mnemonic == Mnemonic::FADD_RSCALE) && src_idx == 2u) return false;
    if ((ins.mnemonic == Mnemonic::FNEG || ins.canonical_mnemonic == Mnemonic::FNEG) && src_idx == 0u) return false;
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

[[verifier::z3("InstLoweringContext", "args[0] < len(this.srcs) and this.srcs[args[0]].kind == 'FauSpecial' and this.srcs[args[0]].symbolic_name == 'thread_local_pointer'")]] bool src_is_fau_tls(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx < ins.srcs.size() &&
           ins.srcs[src_idx].kind == SourceKind::FauSpecial &&
           ins.srcs[src_idx].symbolic_name == "thread_local_pointer";
}

[[verifier::z3("InstLoweringContext", "args[0] < len(this.srcs) and this.srcs[args[0]].kind == 'FauSpecial' and this.srcs[args[0]].symbolic_name == 'workgroup_local_pointer'")]] bool src_is_fau_wls(Pass1Context&, uint32_t, const DecodedValhallInst& ins, const tables::LoweringRuleSpec*, uint32_t src_idx) {
    return src_idx < ins.srcs.size() &&
           ins.srcs[src_idx].kind == SourceKind::FauSpecial &&
           ins.srcs[src_idx].symbolic_name == "workgroup_local_pointer";
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
