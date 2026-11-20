// AUTO-GENERATED FILE — DO NOT EDIT.
// Generated from specs/lowering/valhall_lowering.json (operands + inline_ops host_functions) by tools/gen_valhall_tables.py

#include "valhall_operands.h"
#include "valhall_pass1_handlers.h"

namespace valhall::compiler {

std::string get_src_swz(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx) {
    return abi::get_src_swz(ctx, blk, ins, rule, src_idx);
}

std::string get_dest_mod(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    return abi::get_dest_mod(ctx, blk, ins, rule);
}

SsaId fetch_raw_src_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx) {
    if (abi::src_oob(ctx, blk, ins, rule, src_idx)) {
        return abi::u32(ctx, blk, ins, rule, 0);
    }
    if (abi::src_is_reg(ctx, blk, ins, rule, src_idx)) {
        return abi::src_read_reg(ctx, blk, ins, rule, src_idx);
    }
    if (abi::src_is_fau_uniform(ctx, blk, ins, rule, src_idx)) {
        return abi::load_fau(ctx, blk, ins, rule, abi::src_reg_num(ctx, blk, ins, rule, src_idx));
    }
    if (abi::src_is_small_const(ctx, blk, ins, rule, src_idx)) {
        return abi::src_emit_imm_u32(ctx, blk, ins, rule, src_idx);
    }
    if (abi::src_is_fau_pc(ctx, blk, ins, rule, src_idx)) {
        SsaId pc_base = abi::load_sideband_u64(ctx, blk, ins, rule, 1);
        SsaId pc_off = abi::ins_pc_u64(ctx, blk, ins, rule);
        SsaNode _n1{};
        _n1.op = tables::SsaOp::IAdd64;
        _n1.type = ValType::U64;
        _n1.args = { pc_base, pc_off };
        _n1.effects.cap_bits |= EffectSet::kCapInt64;
        SsaId cur_pc = ctx.sb.emit(blk, std::move(_n1));
        if (abi::src_word_select(ctx, blk, ins, rule, src_idx)) {
            return abi::split64_hi(ctx, blk, ins, rule, cur_pc);
        }
        return abi::split64_lo(ctx, blk, ins, rule, cur_pc);
    }
    if (abi::src_is_fau_lane_id(ctx, blk, ins, rule, src_idx)) {
        return abi::emit_lane_id(ctx, blk, ins, rule);
    }
    if (abi::src_is_fau_lane_width(ctx, blk, ins, rule, src_idx)) {
        return abi::emit_lane_width(ctx, blk, ins, rule);
    }
    return abi::load_fau(ctx, blk, ins, rule, abi::src_reg_num(ctx, blk, ins, rule, src_idx));
}

SsaId fetch_src_int32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx, bool is_signed) {
    SsaId val = fetch_raw_src_u32(ctx, blk, ins, rule, src_idx);
    if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b0")) {
        val = abi::extract_bits(ctx, blk, ins, rule, val, 0, 8, is_signed);
    } else if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b1")) {
        val = abi::extract_bits(ctx, blk, ins, rule, val, 8, 8, is_signed);
    } else if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b2")) {
        val = abi::extract_bits(ctx, blk, ins, rule, val, 16, 8, is_signed);
    } else if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b3")) {
        val = abi::extract_bits(ctx, blk, ins, rule, val, 24, 8, is_signed);
    } else if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h0")) {
        val = abi::extract_bits(ctx, blk, ins, rule, val, 0, 16, is_signed);
    } else if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h1")) {
        val = abi::extract_bits(ctx, blk, ins, rule, val, 16, 16, is_signed);
    } else if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h00")) {
        SsaId h = abi::extract_bits(ctx, blk, ins, rule, val, 0, 16, is_signed);
        SsaNode _n2{};
        _n2.op = tables::SsaOp::Shl;
        _n2.type = ValType::U32;
        _n2.args = { h, abi::u32(ctx, blk, ins, rule, 16) };
        SsaId hi = ctx.sb.emit(blk, std::move(_n2));
        SsaNode _n3{};
        _n3.op = tables::SsaOp::BitOr;
        _n3.type = ValType::U32;
        _n3.args = { h, hi };
        val = ctx.sb.emit(blk, std::move(_n3));
    } else if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h11")) {
        SsaId h = abi::extract_bits(ctx, blk, ins, rule, val, 16, 16, is_signed);
        SsaNode _n4{};
        _n4.op = tables::SsaOp::Shl;
        _n4.type = ValType::U32;
        _n4.args = { h, abi::u32(ctx, blk, ins, rule, 16) };
        SsaId hi = ctx.sb.emit(blk, std::move(_n4));
        SsaNode _n5{};
        _n5.op = tables::SsaOp::BitOr;
        _n5.type = ValType::U32;
        _n5.args = { h, hi };
        val = ctx.sb.emit(blk, std::move(_n5));
    } else if (abi::src_swz_is_b_rep(ctx, blk, ins, rule, src_idx)) {
        uint32_t boff = abi::src_swz_b_rep_off(ctx, blk, ins, rule, src_idx);
        SsaNode _n6{};
        _n6.op = tables::SsaOp::BfeU;
        _n6.type = ValType::U32;
        _n6.args = { val, abi::u32(ctx, blk, ins, rule, boff), abi::u32(ctx, blk, ins, rule, 8) };
        SsaId bk = ctx.sb.emit(blk, std::move(_n6));
        SsaNode _n7{};
        _n7.op = tables::SsaOp::IMul;
        _n7.type = ValType::U32;
        _n7.args = { bk, abi::u32(ctx, blk, ins, rule, 0x01010101u) };
        val = ctx.sb.emit(blk, std::move(_n7));
    }
    if (abi::get_src_abs(ctx, blk, ins, rule, src_idx)) {
        SsaId sval = abi::bitcast(ctx, blk, ins, rule, ValType::S32, val);
        SsaNode _n8{};
        _n8.op = tables::SsaOp::SAbs;
        _n8.type = ValType::S32;
        _n8.args = { sval };
        _n8.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId ab = ctx.sb.emit(blk, std::move(_n8));
        val = abi::bitcast(ctx, blk, ins, rule, ValType::U32, ab);
    }
    if (abi::get_src_neg(ctx, blk, ins, rule, src_idx)) {
        SsaId sval2 = abi::bitcast(ctx, blk, ins, rule, ValType::S32, val);
        SsaNode _n9{};
        _n9.op = tables::SsaOp::SNeg;
        _n9.type = ValType::S32;
        _n9.args = { sval2 };
        SsaId ng = ctx.sb.emit(blk, std::move(_n9));
        val = abi::bitcast(ctx, blk, ins, rule, ValType::U32, ng);
    }
    if (abi::get_src_not(ctx, blk, ins, rule, src_idx)) {
        SsaNode _n10{};
        _n10.op = tables::SsaOp::BitNot;
        _n10.type = ValType::U32;
        _n10.args = { val };
        val = ctx.sb.emit(blk, std::move(_n10));
    }
    return val;
}

SsaId fetch_src_f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx) {
    SsaId raw_u32 = fetch_raw_src_u32(ctx, blk, ins, rule, src_idx);
    SsaId fval = abi::bitcast(ctx, blk, ins, rule, ValType::F32, raw_u32);
    if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h0")) {
        SsaNode _n11{};
        _n11.op = tables::SsaOp::UnpackHalf2x16;
        _n11.type = ValType::V2F32;
        _n11.args = { raw_u32 };
        _n11.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vec = ctx.sb.emit(blk, std::move(_n11));
        fval = abi::extract(ctx, blk, ins, rule, ValType::F32, vec, 0);
    } else if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h1")) {
        SsaNode _n12{};
        _n12.op = tables::SsaOp::UnpackHalf2x16;
        _n12.type = ValType::V2F32;
        _n12.args = { raw_u32 };
        _n12.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vec = ctx.sb.emit(blk, std::move(_n12));
        fval = abi::extract(ctx, blk, ins, rule, ValType::F32, vec, 1);
    }
    if (abi::get_src_abs(ctx, blk, ins, rule, src_idx)) {
        SsaNode _n13{};
        _n13.op = tables::SsaOp::FAbs;
        _n13.type = ValType::F32;
        _n13.args = { fval };
        _n13.effects.flags |= EffectSet::kFlagUsesGlsl450;
        fval = ctx.sb.emit(blk, std::move(_n13));
    }
    if (abi::get_src_neg(ctx, blk, ins, rule, src_idx)) {
        SsaNode _n14{};
        _n14.op = tables::SsaOp::FNeg;
        _n14.type = ValType::F32;
        _n14.args = { fval };
        fval = ctx.sb.emit(blk, std::move(_n14));
    }
    return fval;
}

SsaId fetch_src_v2f16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx) {
    SsaId raw_u32 = fetch_raw_src_u32(ctx, blk, ins, rule, src_idx);
    SsaNode _n15{};
    _n15.op = tables::SsaOp::UnpackHalf2x16;
    _n15.type = ValType::V2F32;
    _n15.args = { raw_u32 };
    _n15.effects.flags |= EffectSet::kFlagUsesGlsl450;
    SsaId vec = ctx.sb.emit(blk, std::move(_n15));
    vec = abi::apply_v2_swizzle(ctx, blk, ins, rule, vec, ValType::V2F32, src_idx);
    if (abi::get_src_abs(ctx, blk, ins, rule, src_idx)) {
        SsaNode _n16{};
        _n16.op = tables::SsaOp::FAbs;
        _n16.type = ValType::V2F32;
        _n16.args = { vec };
        _n16.effects.flags |= EffectSet::kFlagUsesGlsl450;
        vec = ctx.sb.emit(blk, std::move(_n16));
    }
    if (abi::get_src_neg(ctx, blk, ins, rule, src_idx)) {
        SsaNode _n17{};
        _n17.op = tables::SsaOp::FNeg;
        _n17.type = ValType::V2F32;
        _n17.args = { vec };
        vec = ctx.sb.emit(blk, std::move(_n17));
    }
    return vec;
}

SsaId fetch_src_v2i16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx, bool is_signed) {
    SsaId raw_u32 = fetch_raw_src_u32(ctx, blk, ins, rule, src_idx);
    if (is_signed) {
        SsaId in_s = abi::bitcast(ctx, blk, ins, rule, ValType::S32, raw_u32);
        SsaNode _n18{};
        _n18.op = tables::SsaOp::BfeS;
        _n18.type = ValType::S32;
        _n18.args = { in_s, abi::u32(ctx, blk, ins, rule, 0), abi::u32(ctx, blk, ins, rule, 16) };
        SsaId l0 = ctx.sb.emit(blk, std::move(_n18));
        SsaNode _n19{};
        _n19.op = tables::SsaOp::BfeS;
        _n19.type = ValType::S32;
        _n19.args = { in_s, abi::u32(ctx, blk, ins, rule, 16), abi::u32(ctx, blk, ins, rule, 16) };
        SsaId l1 = ctx.sb.emit(blk, std::move(_n19));
        SsaNode _n20{};
        _n20.op = tables::SsaOp::VecConstruct2;
        _n20.type = ValType::V2S32;
        _n20.args = { l0, l1 };
        SsaId vec = ctx.sb.emit(blk, std::move(_n20));
        return abi::apply_v2_swizzle(ctx, blk, ins, rule, vec, ValType::V2S32, src_idx);
    }
    SsaNode _n21{};
    _n21.op = tables::SsaOp::BfeU;
    _n21.type = ValType::U32;
    _n21.args = { raw_u32, abi::u32(ctx, blk, ins, rule, 0), abi::u32(ctx, blk, ins, rule, 16) };
    SsaId l0 = ctx.sb.emit(blk, std::move(_n21));
    SsaNode _n22{};
    _n22.op = tables::SsaOp::BfeU;
    _n22.type = ValType::U32;
    _n22.args = { raw_u32, abi::u32(ctx, blk, ins, rule, 16), abi::u32(ctx, blk, ins, rule, 16) };
    SsaId l1 = ctx.sb.emit(blk, std::move(_n22));
    SsaNode _n23{};
    _n23.op = tables::SsaOp::VecConstruct2;
    _n23.type = ValType::V2U32;
    _n23.args = { l0, l1 };
    SsaId vec = ctx.sb.emit(blk, std::move(_n23));
    return abi::apply_v2_swizzle(ctx, blk, ins, rule, vec, ValType::V2U32, src_idx);
}

SsaId fetch_src_v4i8(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx, bool is_signed) {
    SsaId raw_u32 = fetch_raw_src_u32(ctx, blk, ins, rule, src_idx);
    if (is_signed) {
        SsaId in_s = abi::bitcast(ctx, blk, ins, rule, ValType::S32, raw_u32);
        SsaNode _n24{};
        _n24.op = tables::SsaOp::BfeS;
        _n24.type = ValType::S32;
        _n24.args = { in_s, abi::u32(ctx, blk, ins, rule, 0), abi::u32(ctx, blk, ins, rule, 8) };
        SsaId l0 = ctx.sb.emit(blk, std::move(_n24));
        SsaNode _n25{};
        _n25.op = tables::SsaOp::BfeS;
        _n25.type = ValType::S32;
        _n25.args = { in_s, abi::u32(ctx, blk, ins, rule, 8), abi::u32(ctx, blk, ins, rule, 8) };
        SsaId l1 = ctx.sb.emit(blk, std::move(_n25));
        SsaNode _n26{};
        _n26.op = tables::SsaOp::BfeS;
        _n26.type = ValType::S32;
        _n26.args = { in_s, abi::u32(ctx, blk, ins, rule, 16), abi::u32(ctx, blk, ins, rule, 8) };
        SsaId l2 = ctx.sb.emit(blk, std::move(_n26));
        SsaNode _n27{};
        _n27.op = tables::SsaOp::BfeS;
        _n27.type = ValType::S32;
        _n27.args = { in_s, abi::u32(ctx, blk, ins, rule, 24), abi::u32(ctx, blk, ins, rule, 8) };
        SsaId l3 = ctx.sb.emit(blk, std::move(_n27));
        SsaNode _n28{};
        _n28.op = tables::SsaOp::VecConstruct4;
        _n28.type = ValType::V4U32;
        _n28.args = { l0, l1, l2, l3 };
        return ctx.sb.emit(blk, std::move(_n28));
    }
    SsaNode _n29{};
    _n29.op = tables::SsaOp::BfeU;
    _n29.type = ValType::U32;
    _n29.args = { raw_u32, abi::u32(ctx, blk, ins, rule, 0), abi::u32(ctx, blk, ins, rule, 8) };
    SsaId l0 = ctx.sb.emit(blk, std::move(_n29));
    SsaNode _n30{};
    _n30.op = tables::SsaOp::BfeU;
    _n30.type = ValType::U32;
    _n30.args = { raw_u32, abi::u32(ctx, blk, ins, rule, 8), abi::u32(ctx, blk, ins, rule, 8) };
    SsaId l1 = ctx.sb.emit(blk, std::move(_n30));
    SsaNode _n31{};
    _n31.op = tables::SsaOp::BfeU;
    _n31.type = ValType::U32;
    _n31.args = { raw_u32, abi::u32(ctx, blk, ins, rule, 16), abi::u32(ctx, blk, ins, rule, 8) };
    SsaId l2 = ctx.sb.emit(blk, std::move(_n31));
    SsaNode _n32{};
    _n32.op = tables::SsaOp::BfeU;
    _n32.type = ValType::U32;
    _n32.args = { raw_u32, abi::u32(ctx, blk, ins, rule, 24), abi::u32(ctx, blk, ins, rule, 8) };
    SsaId l3 = ctx.sb.emit(blk, std::move(_n32));
    SsaNode _n33{};
    _n33.op = tables::SsaOp::VecConstruct4;
    _n33.type = ValType::V4U32;
    _n33.args = { l0, l1, l2, l3 };
    return ctx.sb.emit(blk, std::move(_n33));
}

SsaId fetch_src_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx, bool sign_extend_subword) {
    if (abi::src_oob(ctx, blk, ins, rule, src_idx)) {
        return abi::u64(ctx, blk, ins, rule, 0);
    }
    if (abi::src_is_reg(ctx, blk, ins, rule, src_idx)) {
        if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "w0")) {
            return abi::widen_to_u64(ctx, blk, ins, rule, abi::src_read_reg_base(ctx, blk, ins, rule, src_idx, 0), sign_extend_subword);
        }
        if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "w1")) {
            return abi::widen_to_u64(ctx, blk, ins, rule, abi::src_read_reg_base(ctx, blk, ins, rule, src_idx, 1), sign_extend_subword);
        }
        if (!(abi::src_is_64bit(ctx, blk, ins, rule, src_idx))) {
            return abi::widen_to_u64(ctx, blk, ins, rule, abi::src_read_reg(ctx, blk, ins, rule, src_idx), sign_extend_subword);
        }
        return abi::emit_pack_u64(ctx, blk, ins, rule, abi::src_read_reg_base(ctx, blk, ins, rule, src_idx, 0), abi::src_read_reg_base(ctx, blk, ins, rule, src_idx, 1));
    } else if (abi::src_is_fau_uniform(ctx, blk, ins, rule, src_idx)) {
        uint32_t base = abi::src_reg_base(ctx, blk, ins, rule, src_idx);
        if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "w0")) {
            return abi::widen_to_u64(ctx, blk, ins, rule, abi::load_fau(ctx, blk, ins, rule, base), sign_extend_subword);
        }
        if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "w1")) {
            return abi::widen_to_u64(ctx, blk, ins, rule, abi::load_fau(ctx, blk, ins, rule, abi::add_u32(ctx, blk, ins, rule, base, 1)), sign_extend_subword);
        }
        if (!(abi::src_is_64bit(ctx, blk, ins, rule, src_idx))) {
            return abi::widen_to_u64(ctx, blk, ins, rule, abi::load_fau(ctx, blk, ins, rule, abi::src_reg_num(ctx, blk, ins, rule, src_idx)), sign_extend_subword);
        }
        return abi::emit_pack_u64(ctx, blk, ins, rule, abi::load_fau(ctx, blk, ins, rule, base), abi::load_fau(ctx, blk, ins, rule, abi::add_u32(ctx, blk, ins, rule, base, 1)));
    } else if (abi::src_is_small_const(ctx, blk, ins, rule, src_idx)) {
        return abi::src_emit_imm_u64(ctx, blk, ins, rule, src_idx);
    } else if (abi::src_is_fau_pc(ctx, blk, ins, rule, src_idx)) {
        SsaId pc_base = abi::load_sideband_u64(ctx, blk, ins, rule, 1);
        SsaNode _n34{};
        _n34.op = tables::SsaOp::IAdd64;
        _n34.type = ValType::U64;
        _n34.args = { pc_base, abi::ins_pc_u64(ctx, blk, ins, rule) };
        _n34.effects.cap_bits |= EffectSet::kCapInt64;
        return ctx.sb.emit(blk, std::move(_n34));
    } else if (abi::src_is_fau_tls(ctx, blk, ins, rule, src_idx)) {
        return abi::load_sideband_u64(ctx, blk, ins, rule, 6);
    } else if (abi::src_is_fau_wls(ctx, blk, ins, rule, src_idx)) {
        return abi::load_sideband_u64(ctx, blk, ins, rule, 7);
    } else {
        uint32_t base = abi::src_reg_base(ctx, blk, ins, rule, src_idx);
        return abi::emit_pack_u64(ctx, blk, ins, rule, abi::load_fau(ctx, blk, ins, rule, base), abi::load_fau(ctx, blk, ins, rule, abi::add_u32(ctx, blk, ins, rule, base, 1)));
    }
}

void store_dest_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val_u32) {
    if (!(abi::has_dest(ctx, blk, ins, rule))) {
        return;
    }
    uint32_t d = abi::dest_reg_num(ctx, blk, ins, rule);
    if (abi::dest_mod_eq(ctx, blk, ins, rule, "h0")) {
        SsaId bfi_id = abi::bfi(ctx, blk, ins, rule, abi::dest_read_reg(ctx, blk, ins, rule), val_u32, 0, 16);
        abi::dest_set_reg(ctx, blk, ins, rule, d, bfi_id);
    } else if (abi::dest_mod_eq(ctx, blk, ins, rule, "h1")) {
        SsaId bfi_id = abi::bfi(ctx, blk, ins, rule, abi::dest_read_reg(ctx, blk, ins, rule), val_u32, 16, 16);
        abi::dest_set_reg(ctx, blk, ins, rule, d, bfi_id);
    } else if (abi::dest_mod_is_byte(ctx, blk, ins, rule)) {
        uint32_t off = abi::dest_mod_byte_off(ctx, blk, ins, rule);
        SsaId bfi_id = abi::bfi(ctx, blk, ins, rule, abi::dest_read_reg(ctx, blk, ins, rule), val_u32, off, 8);
        abi::dest_set_reg(ctx, blk, ins, rule, d, bfi_id);
    } else {
        abi::dest_set_reg(ctx, blk, ins, rule, d, val_u32);
    }
}

void store_dest_f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val_f32) {
    if (!(abi::has_dest(ctx, blk, ins, rule))) {
        return;
    }
    val_f32 = abi::apply_f32_clamp(ctx, blk, ins, rule, val_f32);
    if ((abi::dest_mod_eq(ctx, blk, ins, rule, "h0") || abi::dest_mod_eq(ctx, blk, ins, rule, "h1"))) {
        SsaNode _n35{};
        _n35.op = tables::SsaOp::VecConstruct2;
        _n35.type = ValType::V2F32;
        _n35.args = { val_f32, val_f32 };
        SsaId v2 = ctx.sb.emit(blk, std::move(_n35));
        SsaNode _n36{};
        _n36.op = tables::SsaOp::PackHalf2x16;
        _n36.type = ValType::U32;
        _n36.args = { v2 };
        _n36.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId packed = ctx.sb.emit(blk, std::move(_n36));
        uint32_t d = abi::dest_reg_num(ctx, blk, ins, rule);
        if (abi::dest_mod_eq(ctx, blk, ins, rule, "h1")) {
            SsaId bfi_id = abi::bfi(ctx, blk, ins, rule, abi::dest_read_reg(ctx, blk, ins, rule), packed, 16, 16);
            abi::dest_set_reg(ctx, blk, ins, rule, d, bfi_id);
        } else {
            SsaId bfi_id = abi::bfi(ctx, blk, ins, rule, abi::dest_read_reg(ctx, blk, ins, rule), packed, 0, 16);
            abi::dest_set_reg(ctx, blk, ins, rule, d, bfi_id);
        }
        return;
    }
    SsaId bits = abi::bitcast(ctx, blk, ins, rule, ValType::U32, val_f32);
    abi::dest_set_reg(ctx, blk, ins, rule, abi::dest_reg_num(ctx, blk, ins, rule), bits);
}

void store_dest_v2f16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec2_f32) {
    if (!(abi::has_dest(ctx, blk, ins, rule))) {
        return;
    }
    vec2_f32 = abi::apply_v2f16_clamp(ctx, blk, ins, rule, vec2_f32);
    SsaNode _n37{};
    _n37.op = tables::SsaOp::PackHalf2x16;
    _n37.type = ValType::U32;
    _n37.args = { vec2_f32 };
    _n37.effects.flags |= EffectSet::kFlagUsesGlsl450;
    SsaId pk = ctx.sb.emit(blk, std::move(_n37));
    store_dest_u32(ctx, blk, ins, rule, pk);
}

void store_dest_v2i16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec2_i32, bool is_signed) {
    if (!(abi::has_dest(ctx, blk, ins, rule))) {
        return;
    }
    SsaId e0 = abi::extract_lane_u32(ctx, blk, ins, rule, vec2_i32, 0, is_signed);
    SsaId e1 = abi::extract_lane_u32(ctx, blk, ins, rule, vec2_i32, 1, is_signed);
    SsaId p0 = abi::bfi(ctx, blk, ins, rule, abi::u32(ctx, blk, ins, rule, 0), e0, 0, 16);
    SsaId p1 = abi::bfi(ctx, blk, ins, rule, p0, e1, 16, 16);
    store_dest_u32(ctx, blk, ins, rule, p1);
}

void store_dest_v4i8(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec4_i32, bool is_signed) {
    if (!(abi::has_dest(ctx, blk, ins, rule))) {
        return;
    }
    SsaId acc = abi::u32(ctx, blk, ins, rule, 0);
    for (uint32_t i = 0; i < 4; ++i) {
        SsaId ei = abi::extract_lane_u32(ctx, blk, ins, rule, vec4_i32, i, is_signed);
        acc = abi::bfi(ctx, blk, ins, rule, acc, ei, abi::mul8(ctx, blk, ins, rule, i), 8);
    }
    store_dest_u32(ctx, blk, ins, rule, acc);
}

void store_dest_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val_u64) {
    if (!(abi::has_dest(ctx, blk, ins, rule))) {
        return;
    }
    SsaId lo = abi::split64_lo(ctx, blk, ins, rule, val_u64);
    SsaId hi = abi::split64_hi(ctx, blk, ins, rule, val_u64);
    uint32_t d = abi::dest_reg_base(ctx, blk, ins, rule);
    abi::dest_set_reg(ctx, blk, ins, rule, d, lo);
    abi::dest_set_reg(ctx, blk, ins, rule, abi::add_u32(ctx, blk, ins, rule, d, 1), hi);
}

namespace abi {

bool src0_swz_h1(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    return abi::src_swz_eq(ctx, blk, ins, rule, 0, "h1");
}

SsaId fp_zero(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    if (abi::is_v2f16(ctx, blk, ins, rule)) {
        SsaId s = abi::f32_bits(ctx, blk, ins, rule, 0);
        SsaNode _n38{};
        _n38.op = tables::SsaOp::VecConstruct2;
        _n38.type = ValType::V2F32;
        _n38.args = { s, s };
        SsaId v = ctx.sb.emit(blk, std::move(_n38));
        return v;
    }
    return abi::f32_bits(ctx, blk, ins, rule, 0);
}

SsaId raw0(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    return fetch_raw_src_u32(ctx, blk, ins, rule, 0);
}

SsaId split64_lo(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val_u64) {
    SsaNode _n39{};
    _n39.op = tables::SsaOp::Split64;
    _n39.type = ValType::TupleU32x2;
    _n39.args = { val_u64 };
    _n39.effects.cap_bits |= EffectSet::kCapInt64;
    SsaId tup = ctx.sb.emit(blk, std::move(_n39));
    return abi::tuple_val(ctx, blk, ins, rule, ValType::U32, tup);
}

SsaId extract_bits(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val, uint32_t offset, uint32_t count, bool is_signed) {
    if (is_signed) {
        SsaId sval = abi::bitcast(ctx, blk, ins, rule, ValType::S32, val);
        SsaNode _n40{};
        _n40.op = tables::SsaOp::BfeS;
        _n40.type = ValType::S32;
        _n40.args = { sval, abi::u32(ctx, blk, ins, rule, offset), abi::u32(ctx, blk, ins, rule, count) };
        SsaId ext = ctx.sb.emit(blk, std::move(_n40));
        return abi::bitcast(ctx, blk, ins, rule, ValType::U32, ext);
    }
    SsaNode _n41{};
    _n41.op = tables::SsaOp::BfeU;
    _n41.type = ValType::U32;
    _n41.args = { val, abi::u32(ctx, blk, ins, rule, offset), abi::u32(ctx, blk, ins, rule, count) };
    return ctx.sb.emit(blk, std::move(_n41));
}

SsaId widen_to_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId w32, bool sext) {
    if (sext) {
        SsaId s32 = abi::bitcast(ctx, blk, ins, rule, ValType::S32, w32);
        SsaNode _n42{};
        _n42.op = tables::SsaOp::SConvert32To64;
        _n42.type = ValType::S64;
        _n42.args = { s32 };
        _n42.effects.cap_bits |= EffectSet::kCapInt64;
        SsaId cvt = ctx.sb.emit(blk, std::move(_n42));
        return abi::bitcast(ctx, blk, ins, rule, ValType::U64, cvt);
    }
    SsaNode _n43{};
    _n43.op = tables::SsaOp::UConvert32To64;
    _n43.type = ValType::U64;
    _n43.args = { w32 };
    _n43.effects.cap_bits |= EffectSet::kCapInt64;
    return ctx.sb.emit(blk, std::move(_n43));
}

SsaId apply_v2_swizzle(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec, ValType vt, uint32_t src_idx) {
    if ((abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h00") || abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h0"))) {
        return abi::shuffle2(ctx, blk, ins, rule, vt, vec, vec, 0, 0);
    } else if ((abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h11") || abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h1"))) {
        return abi::shuffle2(ctx, blk, ins, rule, vt, vec, vec, 1, 1);
    } else if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "h10")) {
        return abi::shuffle2(ctx, blk, ins, rule, vt, vec, vec, 1, 0);
    }
    return vec;
}

SsaId apply_f32_clamp(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val_f32) {
    if (abi::has_clamp_0_1(ctx, blk, ins, rule)) {
        SsaNode _n44{};
        _n44.op = tables::SsaOp::FClamp;
        _n44.type = ValType::F32;
        _n44.args = { val_f32, abi::f32_bits(ctx, blk, ins, rule, 0), abi::f32_bits(ctx, blk, ins, rule, 0x3f800000u) };
        _n44.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return ctx.sb.emit(blk, std::move(_n44));
    } else if (abi::has_clamp_m1_1(ctx, blk, ins, rule)) {
        SsaNode _n45{};
        _n45.op = tables::SsaOp::FClamp;
        _n45.type = ValType::F32;
        _n45.args = { val_f32, abi::f32_bits(ctx, blk, ins, rule, 0xbf800000u), abi::f32_bits(ctx, blk, ins, rule, 0x3f800000u) };
        _n45.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return ctx.sb.emit(blk, std::move(_n45));
    } else if (abi::has_clamp_0_inf(ctx, blk, ins, rule)) {
        SsaNode _n46{};
        _n46.op = tables::SsaOp::FMax;
        _n46.type = ValType::F32;
        _n46.args = { val_f32, abi::f32_bits(ctx, blk, ins, rule, 0) };
        _n46.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return ctx.sb.emit(blk, std::move(_n46));
    }
    return val_f32;
}

SsaId apply_v2f16_clamp(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec2_f32) {
    if (abi::has_clamp_0_1(ctx, blk, ins, rule)) {
        SsaId lo = abi::f32_bits(ctx, blk, ins, rule, 0);
        SsaId hi = abi::f32_bits(ctx, blk, ins, rule, 0x3f800000u);
        SsaNode _n47{};
        _n47.op = tables::SsaOp::VecConstruct2;
        _n47.type = ValType::V2F32;
        _n47.args = { lo, lo };
        _n47.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vlo = ctx.sb.emit(blk, std::move(_n47));
        SsaNode _n48{};
        _n48.op = tables::SsaOp::VecConstruct2;
        _n48.type = ValType::V2F32;
        _n48.args = { hi, hi };
        _n48.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vhi = ctx.sb.emit(blk, std::move(_n48));
        SsaNode _n49{};
        _n49.op = tables::SsaOp::FClamp;
        _n49.type = ValType::F32;
        _n49.args = { vec2_f32, vlo, vhi };
        _n49.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return ctx.sb.emit(blk, std::move(_n49));
    } else if (abi::has_clamp_m1_1(ctx, blk, ins, rule)) {
        SsaId lo = abi::f32_bits(ctx, blk, ins, rule, 0xbf800000u);
        SsaId hi = abi::f32_bits(ctx, blk, ins, rule, 0x3f800000u);
        SsaNode _n50{};
        _n50.op = tables::SsaOp::VecConstruct2;
        _n50.type = ValType::V2F32;
        _n50.args = { lo, lo };
        _n50.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vlo = ctx.sb.emit(blk, std::move(_n50));
        SsaNode _n51{};
        _n51.op = tables::SsaOp::VecConstruct2;
        _n51.type = ValType::V2F32;
        _n51.args = { hi, hi };
        _n51.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vhi = ctx.sb.emit(blk, std::move(_n51));
        SsaNode _n52{};
        _n52.op = tables::SsaOp::FClamp;
        _n52.type = ValType::F32;
        _n52.args = { vec2_f32, vlo, vhi };
        _n52.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return ctx.sb.emit(blk, std::move(_n52));
    } else if (abi::has_clamp_0_inf(ctx, blk, ins, rule)) {
        SsaId z = abi::f32_bits(ctx, blk, ins, rule, 0);
        SsaNode _n53{};
        _n53.op = tables::SsaOp::VecConstruct2;
        _n53.type = ValType::V2F32;
        _n53.args = { z, z };
        _n53.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vz = ctx.sb.emit(blk, std::move(_n53));
        SsaNode _n54{};
        _n54.op = tables::SsaOp::FMax;
        _n54.type = ValType::F32;
        _n54.args = { vec2_f32, vz };
        _n54.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return ctx.sb.emit(blk, std::move(_n54));
    }
    return vec2_f32;
}

SsaId extract_lane_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec, uint32_t idx, bool is_signed) {
    if (is_signed) {
        SsaId ex = abi::extract(ctx, blk, ins, rule, ValType::S32, vec, idx);
        return abi::bitcast(ctx, blk, ins, rule, ValType::U32, ex);
    }
    return abi::extract(ctx, blk, ins, rule, ValType::U32, vec, idx);
}

bool src_swz_is_b_rep(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx) {
    if ((((abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b0000") || abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b1111")) || abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b2222")) || abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b3333"))) {
        return true;
    }
    return false;
}

uint32_t src_swz_b_rep_off(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx) {
    if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b1111")) {
        return 8;
    }
    if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b2222")) {
        return 16;
    }
    if (abi::src_swz_eq(ctx, blk, ins, rule, src_idx, "b3333")) {
        return 24;
    }
    return 0;
}

bool dest_mod_is_byte(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    if ((((abi::dest_mod_eq(ctx, blk, ins, rule, "b0") || abi::dest_mod_eq(ctx, blk, ins, rule, "b1")) || abi::dest_mod_eq(ctx, blk, ins, rule, "b2")) || abi::dest_mod_eq(ctx, blk, ins, rule, "b3"))) {
        return true;
    }
    return false;
}

uint32_t dest_mod_byte_off(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    if (abi::dest_mod_eq(ctx, blk, ins, rule, "b1")) {
        return 8;
    }
    if (abi::dest_mod_eq(ctx, blk, ins, rule, "b2")) {
        return 16;
    }
    if (abi::dest_mod_eq(ctx, blk, ins, rule, "b3")) {
        return 24;
    }
    return 0;
}

bool has_clamp_0_1(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    return abi::mod_eq(ctx, blk, ins, rule, "clamp", "clamp_0_1");
}

bool has_clamp_m1_1(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    return abi::mod_eq(ctx, blk, ins, rule, "clamp", "clamp_m1_1");
}

bool has_clamp_0_inf(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {
    return abi::mod_eq(ctx, blk, ins, rule, "clamp", "clamp_0_inf");
}

SsaId emit_pack_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId lo_u32, SsaId hi_u32) {
    SsaNode _n55{};
    _n55.op = tables::SsaOp::Pack64;
    _n55.type = ValType::U32;
    _n55.args = { lo_u32, hi_u32 };
    _n55.effects.cap_bits |= EffectSet::kCapInt64;
    return ctx.sb.emit(blk, std::move(_n55));
}

SsaId split64_hi(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val_u64) {
    SsaNode _n56{};
    _n56.op = tables::SsaOp::Split64;
    _n56.type = ValType::TupleU32x2;
    _n56.args = { val_u64 };
    _n56.effects.cap_bits |= EffectSet::kCapInt64;
    SsaId tup = ctx.sb.emit(blk, std::move(_n56));
    return abi::extract(ctx, blk, ins, rule, ValType::U32, tup, 1);
}

} // namespace abi

} // namespace valhall::compiler
