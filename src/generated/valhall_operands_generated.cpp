// AUTO-GENERATED FILE — DO NOT EDIT.
// Generated from specs/lowering/valhall_lowering.json (operands + inline_ops host_functions) by tools/gen_valhall_tables.py

#include "valhall_operands.h"
#include "valhall_pass1_handlers.h"

namespace valhall::compiler {

static SsaId _helper_extract_bits(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val, uint32_t offset, uint32_t count, bool is_signed) {
    if (is_signed) {
        SsaId sval = operands_host::bitcast(sb, blk_id, ValType::S32, val);
        SsaNode _n1{};
        _n1.op = tables::SsaOp::BfeS;
        _n1.type = ValType::S32;
        _n1.args = { sval, operands_host::const_u32(sb, blk_id, offset), operands_host::const_u32(sb, blk_id, count) };
        SsaId ext = sb.emit(blk_id, std::move(_n1));
        return operands_host::bitcast(sb, blk_id, ValType::U32, ext);
    }
    SsaNode _n2{};
    _n2.op = tables::SsaOp::BfeU;
    _n2.type = ValType::U32;
    _n2.args = { val, operands_host::const_u32(sb, blk_id, offset), operands_host::const_u32(sb, blk_id, count) };
    return sb.emit(blk_id, std::move(_n2));
}

static SsaId _helper_widen_to_u64(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId w32, bool sext) {
    if (sext) {
        SsaId s32 = operands_host::bitcast(sb, blk_id, ValType::S32, w32);
        SsaNode _n3{};
        _n3.op = tables::SsaOp::SConvert32To64;
        _n3.type = ValType::S64;
        _n3.args = { s32 };
        _n3.effects.cap_bits |= EffectSet::kCapInt64;
        SsaId cvt = sb.emit(blk_id, std::move(_n3));
        return operands_host::bitcast(sb, blk_id, ValType::U64, cvt);
    }
    SsaNode _n4{};
    _n4.op = tables::SsaOp::UConvert32To64;
    _n4.type = ValType::U64;
    _n4.args = { w32 };
    _n4.effects.cap_bits |= EffectSet::kCapInt64;
    return sb.emit(blk_id, std::move(_n4));
}

static SsaId _helper_apply_v2_swizzle(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec, ValType vt, uint32_t src_idx) {
    if ((operands_host::src_swz_eq(ins, src_idx, "h00") || operands_host::src_swz_eq(ins, src_idx, "h0"))) {
        return operands_host::shuffle2(sb, blk_id, vt, vec, vec, 0, 0);
    } else if ((operands_host::src_swz_eq(ins, src_idx, "h11") || operands_host::src_swz_eq(ins, src_idx, "h1"))) {
        return operands_host::shuffle2(sb, blk_id, vt, vec, vec, 1, 1);
    } else if (operands_host::src_swz_eq(ins, src_idx, "h10")) {
        return operands_host::shuffle2(sb, blk_id, vt, vec, vec, 1, 0);
    }
    return vec;
}

static SsaId _helper_apply_f32_clamp(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val_f32) {
    if (operands_host::has_clamp_0_1(ins)) {
        SsaNode _n5{};
        _n5.op = tables::SsaOp::FClamp;
        _n5.type = ValType::F32;
        _n5.args = { val_f32, operands_host::const_f32_bits(sb, blk_id, 0), operands_host::const_f32_bits(sb, blk_id, 0x3f800000u) };
        _n5.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return sb.emit(blk_id, std::move(_n5));
    } else if (operands_host::has_clamp_m1_1(ins)) {
        SsaNode _n6{};
        _n6.op = tables::SsaOp::FClamp;
        _n6.type = ValType::F32;
        _n6.args = { val_f32, operands_host::const_f32_bits(sb, blk_id, 0xbf800000u), operands_host::const_f32_bits(sb, blk_id, 0x3f800000u) };
        _n6.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return sb.emit(blk_id, std::move(_n6));
    } else if (operands_host::has_clamp_0_inf(ins)) {
        SsaNode _n7{};
        _n7.op = tables::SsaOp::FMax;
        _n7.type = ValType::F32;
        _n7.args = { val_f32, operands_host::const_f32_bits(sb, blk_id, 0) };
        _n7.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return sb.emit(blk_id, std::move(_n7));
    }
    return val_f32;
}

static SsaId _helper_apply_v2f16_clamp(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec2_f32) {
    if (operands_host::has_clamp_0_1(ins)) {
        SsaId lo = operands_host::const_f32_bits(sb, blk_id, 0);
        SsaId hi = operands_host::const_f32_bits(sb, blk_id, 0x3f800000u);
        SsaNode _n8{};
        _n8.op = tables::SsaOp::VecConstruct2;
        _n8.type = ValType::V2F32;
        _n8.args = { lo, lo };
        _n8.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vlo = sb.emit(blk_id, std::move(_n8));
        SsaNode _n9{};
        _n9.op = tables::SsaOp::VecConstruct2;
        _n9.type = ValType::V2F32;
        _n9.args = { hi, hi };
        _n9.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vhi = sb.emit(blk_id, std::move(_n9));
        SsaNode _n10{};
        _n10.op = tables::SsaOp::FClamp;
        _n10.type = ValType::F32;
        _n10.args = { vec2_f32, vlo, vhi };
        _n10.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return sb.emit(blk_id, std::move(_n10));
    } else if (operands_host::has_clamp_m1_1(ins)) {
        SsaId lo = operands_host::const_f32_bits(sb, blk_id, 0xbf800000u);
        SsaId hi = operands_host::const_f32_bits(sb, blk_id, 0x3f800000u);
        SsaNode _n11{};
        _n11.op = tables::SsaOp::VecConstruct2;
        _n11.type = ValType::V2F32;
        _n11.args = { lo, lo };
        _n11.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vlo = sb.emit(blk_id, std::move(_n11));
        SsaNode _n12{};
        _n12.op = tables::SsaOp::VecConstruct2;
        _n12.type = ValType::V2F32;
        _n12.args = { hi, hi };
        _n12.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vhi = sb.emit(blk_id, std::move(_n12));
        SsaNode _n13{};
        _n13.op = tables::SsaOp::FClamp;
        _n13.type = ValType::F32;
        _n13.args = { vec2_f32, vlo, vhi };
        _n13.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return sb.emit(blk_id, std::move(_n13));
    } else if (operands_host::has_clamp_0_inf(ins)) {
        SsaId z = operands_host::const_f32_bits(sb, blk_id, 0);
        SsaNode _n14{};
        _n14.op = tables::SsaOp::VecConstruct2;
        _n14.type = ValType::V2F32;
        _n14.args = { z, z };
        _n14.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vz = sb.emit(blk_id, std::move(_n14));
        SsaNode _n15{};
        _n15.op = tables::SsaOp::FMax;
        _n15.type = ValType::F32;
        _n15.args = { vec2_f32, vz };
        _n15.effects.flags |= EffectSet::kFlagUsesGlsl450;
        return sb.emit(blk_id, std::move(_n15));
    }
    return vec2_f32;
}

static SsaId _helper_extract_lane_u32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec, uint32_t idx, bool is_signed) {
    if (is_signed) {
        SsaId ex = operands_host::extract(sb, blk_id, ValType::S32, vec, idx);
        return operands_host::bitcast(sb, blk_id, ValType::U32, ex);
    }
    return operands_host::extract(sb, blk_id, ValType::U32, vec, idx);
}

std::string get_src_swz(const DecodedValhallInst& ins, size_t src_idx) {
    return operands_host::get_src_swz(ins, src_idx);
}

std::string get_dest_mod(const DecodedValhallInst& ins) {
    return operands_host::get_dest_mod(ins);
}

SsaId fetch_raw_src_u32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx) {
    if (operands_host::src_oob(ins, src_idx)) {
        return operands_host::const_u32(sb, blk_id, 0);
    }
    if (operands_host::src_is_reg(ins, src_idx)) {
        return operands_host::src_read_reg(sb, blk_id, ins, src_idx);
    }
    if (operands_host::src_is_fau_uniform(ins, src_idx)) {
        return operands_host::emit_load_fau_u32(sb, blk_id, operands_host::src_reg_num(ins, src_idx));
    }
    if (operands_host::src_is_small_const(ins, src_idx)) {
        return operands_host::src_emit_imm_u32(sb, blk_id, ins, src_idx);
    }
    if (operands_host::src_is_fau_pc(ins, src_idx)) {
        SsaId pc_base = operands_host::emit_load_sideband_u64(sb, blk_id, 1);
        SsaId pc_off = operands_host::ins_pc_u64(sb, blk_id, ins);
        SsaNode _n16{};
        _n16.op = tables::SsaOp::IAdd64;
        _n16.type = ValType::U64;
        _n16.args = { pc_base, pc_off };
        _n16.effects.cap_bits |= EffectSet::kCapInt64;
        SsaId cur_pc = sb.emit(blk_id, std::move(_n16));
        if (operands_host::src_word_select(ins, src_idx)) {
            return operands_host::split64_hi(sb, blk_id, cur_pc);
        }
        return operands_host::split64_lo(sb, blk_id, cur_pc);
    }
    if (operands_host::src_is_fau_lane_id(ins, src_idx)) {
        return operands_host::emit_lane_id(sb, blk_id);
    }
    if (operands_host::src_is_fau_lane_width(ins, src_idx)) {
        return operands_host::emit_lane_width(sb, blk_id);
    }
    return operands_host::emit_load_fau_u32(sb, blk_id, operands_host::src_reg_num(ins, src_idx));
}

SsaId fetch_src_int32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool is_signed) {
    SsaId val = fetch_raw_src_u32(sb, blk_id, ins, src_idx);
    if (operands_host::src_swz_eq(ins, src_idx, "b0")) {
        val = _helper_extract_bits(sb, blk_id, ins, val, 0, 8, is_signed);
    } else if (operands_host::src_swz_eq(ins, src_idx, "b1")) {
        val = _helper_extract_bits(sb, blk_id, ins, val, 8, 8, is_signed);
    } else if (operands_host::src_swz_eq(ins, src_idx, "b2")) {
        val = _helper_extract_bits(sb, blk_id, ins, val, 16, 8, is_signed);
    } else if (operands_host::src_swz_eq(ins, src_idx, "b3")) {
        val = _helper_extract_bits(sb, blk_id, ins, val, 24, 8, is_signed);
    } else if (operands_host::src_swz_eq(ins, src_idx, "h0")) {
        val = _helper_extract_bits(sb, blk_id, ins, val, 0, 16, is_signed);
    } else if (operands_host::src_swz_eq(ins, src_idx, "h1")) {
        val = _helper_extract_bits(sb, blk_id, ins, val, 16, 16, is_signed);
    } else if (operands_host::src_swz_eq(ins, src_idx, "h00")) {
        SsaId h = _helper_extract_bits(sb, blk_id, ins, val, 0, 16, is_signed);
        SsaNode _n17{};
        _n17.op = tables::SsaOp::Shl;
        _n17.type = ValType::U32;
        _n17.args = { h, operands_host::const_u32(sb, blk_id, 16) };
        SsaId hi = sb.emit(blk_id, std::move(_n17));
        SsaNode _n18{};
        _n18.op = tables::SsaOp::BitOr;
        _n18.type = ValType::U32;
        _n18.args = { h, hi };
        val = sb.emit(blk_id, std::move(_n18));
    } else if (operands_host::src_swz_eq(ins, src_idx, "h11")) {
        SsaId h = _helper_extract_bits(sb, blk_id, ins, val, 16, 16, is_signed);
        SsaNode _n19{};
        _n19.op = tables::SsaOp::Shl;
        _n19.type = ValType::U32;
        _n19.args = { h, operands_host::const_u32(sb, blk_id, 16) };
        SsaId hi = sb.emit(blk_id, std::move(_n19));
        SsaNode _n20{};
        _n20.op = tables::SsaOp::BitOr;
        _n20.type = ValType::U32;
        _n20.args = { h, hi };
        val = sb.emit(blk_id, std::move(_n20));
    } else if (operands_host::src_swz_is_b_rep(ins, src_idx)) {
        uint32_t boff = operands_host::src_swz_b_rep_off(ins, src_idx);
        SsaNode _n21{};
        _n21.op = tables::SsaOp::BfeU;
        _n21.type = ValType::U32;
        _n21.args = { val, operands_host::const_u32(sb, blk_id, boff), operands_host::const_u32(sb, blk_id, 8) };
        SsaId bk = sb.emit(blk_id, std::move(_n21));
        SsaNode _n22{};
        _n22.op = tables::SsaOp::IMul;
        _n22.type = ValType::U32;
        _n22.args = { bk, operands_host::const_u32(sb, blk_id, 0x01010101u) };
        val = sb.emit(blk_id, std::move(_n22));
    }
    if (operands_host::get_src_abs(ins, src_idx)) {
        SsaId sval = operands_host::bitcast(sb, blk_id, ValType::S32, val);
        SsaNode _n23{};
        _n23.op = tables::SsaOp::SAbs;
        _n23.type = ValType::S32;
        _n23.args = { sval };
        _n23.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId ab = sb.emit(blk_id, std::move(_n23));
        val = operands_host::bitcast(sb, blk_id, ValType::U32, ab);
    }
    if (operands_host::get_src_neg(ins, src_idx)) {
        SsaId sval2 = operands_host::bitcast(sb, blk_id, ValType::S32, val);
        SsaNode _n24{};
        _n24.op = tables::SsaOp::SNeg;
        _n24.type = ValType::S32;
        _n24.args = { sval2 };
        SsaId ng = sb.emit(blk_id, std::move(_n24));
        val = operands_host::bitcast(sb, blk_id, ValType::U32, ng);
    }
    if (operands_host::get_src_not(ins, src_idx)) {
        SsaNode _n25{};
        _n25.op = tables::SsaOp::BitNot;
        _n25.type = ValType::U32;
        _n25.args = { val };
        val = sb.emit(blk_id, std::move(_n25));
    }
    return val;
}

SsaId fetch_src_f32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx) {
    SsaId raw_u32 = fetch_raw_src_u32(sb, blk_id, ins, src_idx);
    SsaId fval = operands_host::bitcast(sb, blk_id, ValType::F32, raw_u32);
    if (operands_host::src_swz_eq(ins, src_idx, "h0")) {
        SsaNode _n26{};
        _n26.op = tables::SsaOp::UnpackHalf2x16;
        _n26.type = ValType::V2F32;
        _n26.args = { raw_u32 };
        _n26.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vec = sb.emit(blk_id, std::move(_n26));
        fval = operands_host::extract(sb, blk_id, ValType::F32, vec, 0);
    } else if (operands_host::src_swz_eq(ins, src_idx, "h1")) {
        SsaNode _n27{};
        _n27.op = tables::SsaOp::UnpackHalf2x16;
        _n27.type = ValType::V2F32;
        _n27.args = { raw_u32 };
        _n27.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vec = sb.emit(blk_id, std::move(_n27));
        fval = operands_host::extract(sb, blk_id, ValType::F32, vec, 1);
    }
    if (operands_host::get_src_abs(ins, src_idx)) {
        SsaNode _n28{};
        _n28.op = tables::SsaOp::FAbs;
        _n28.type = ValType::F32;
        _n28.args = { fval };
        _n28.effects.flags |= EffectSet::kFlagUsesGlsl450;
        fval = sb.emit(blk_id, std::move(_n28));
    }
    if (operands_host::get_src_neg(ins, src_idx)) {
        SsaNode _n29{};
        _n29.op = tables::SsaOp::FNeg;
        _n29.type = ValType::F32;
        _n29.args = { fval };
        fval = sb.emit(blk_id, std::move(_n29));
    }
    return fval;
}

SsaId fetch_src_v2f16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx) {
    SsaId raw_u32 = fetch_raw_src_u32(sb, blk_id, ins, src_idx);
    SsaNode _n30{};
    _n30.op = tables::SsaOp::UnpackHalf2x16;
    _n30.type = ValType::V2F32;
    _n30.args = { raw_u32 };
    _n30.effects.flags |= EffectSet::kFlagUsesGlsl450;
    SsaId vec = sb.emit(blk_id, std::move(_n30));
    vec = _helper_apply_v2_swizzle(sb, blk_id, ins, vec, ValType::V2F32, src_idx);
    if (operands_host::get_src_abs(ins, src_idx)) {
        SsaNode _n31{};
        _n31.op = tables::SsaOp::FAbs;
        _n31.type = ValType::V2F32;
        _n31.args = { vec };
        _n31.effects.flags |= EffectSet::kFlagUsesGlsl450;
        vec = sb.emit(blk_id, std::move(_n31));
    }
    if (operands_host::get_src_neg(ins, src_idx)) {
        SsaNode _n32{};
        _n32.op = tables::SsaOp::FNeg;
        _n32.type = ValType::V2F32;
        _n32.args = { vec };
        vec = sb.emit(blk_id, std::move(_n32));
    }
    return vec;
}

SsaId fetch_src_v2i16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool is_signed) {
    SsaId raw_u32 = fetch_raw_src_u32(sb, blk_id, ins, src_idx);
    if (is_signed) {
        SsaId in_s = operands_host::bitcast(sb, blk_id, ValType::S32, raw_u32);
        SsaNode _n33{};
        _n33.op = tables::SsaOp::BfeS;
        _n33.type = ValType::S32;
        _n33.args = { in_s, operands_host::const_u32(sb, blk_id, 0), operands_host::const_u32(sb, blk_id, 16) };
        SsaId l0 = sb.emit(blk_id, std::move(_n33));
        SsaNode _n34{};
        _n34.op = tables::SsaOp::BfeS;
        _n34.type = ValType::S32;
        _n34.args = { in_s, operands_host::const_u32(sb, blk_id, 16), operands_host::const_u32(sb, blk_id, 16) };
        SsaId l1 = sb.emit(blk_id, std::move(_n34));
        SsaNode _n35{};
        _n35.op = tables::SsaOp::VecConstruct2;
        _n35.type = ValType::V2S32;
        _n35.args = { l0, l1 };
        SsaId vec = sb.emit(blk_id, std::move(_n35));
        return _helper_apply_v2_swizzle(sb, blk_id, ins, vec, ValType::V2S32, src_idx);
    }
    SsaNode _n36{};
    _n36.op = tables::SsaOp::BfeU;
    _n36.type = ValType::U32;
    _n36.args = { raw_u32, operands_host::const_u32(sb, blk_id, 0), operands_host::const_u32(sb, blk_id, 16) };
    SsaId l0 = sb.emit(blk_id, std::move(_n36));
    SsaNode _n37{};
    _n37.op = tables::SsaOp::BfeU;
    _n37.type = ValType::U32;
    _n37.args = { raw_u32, operands_host::const_u32(sb, blk_id, 16), operands_host::const_u32(sb, blk_id, 16) };
    SsaId l1 = sb.emit(blk_id, std::move(_n37));
    SsaNode _n38{};
    _n38.op = tables::SsaOp::VecConstruct2;
    _n38.type = ValType::V2U32;
    _n38.args = { l0, l1 };
    SsaId vec = sb.emit(blk_id, std::move(_n38));
    return _helper_apply_v2_swizzle(sb, blk_id, ins, vec, ValType::V2U32, src_idx);
}

SsaId fetch_src_v4i8(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool is_signed) {
    SsaId raw_u32 = fetch_raw_src_u32(sb, blk_id, ins, src_idx);
    if (is_signed) {
        SsaId in_s = operands_host::bitcast(sb, blk_id, ValType::S32, raw_u32);
        SsaNode _n39{};
        _n39.op = tables::SsaOp::BfeS;
        _n39.type = ValType::S32;
        _n39.args = { in_s, operands_host::const_u32(sb, blk_id, 0), operands_host::const_u32(sb, blk_id, 8) };
        SsaId l0 = sb.emit(blk_id, std::move(_n39));
        SsaNode _n40{};
        _n40.op = tables::SsaOp::BfeS;
        _n40.type = ValType::S32;
        _n40.args = { in_s, operands_host::const_u32(sb, blk_id, 8), operands_host::const_u32(sb, blk_id, 8) };
        SsaId l1 = sb.emit(blk_id, std::move(_n40));
        SsaNode _n41{};
        _n41.op = tables::SsaOp::BfeS;
        _n41.type = ValType::S32;
        _n41.args = { in_s, operands_host::const_u32(sb, blk_id, 16), operands_host::const_u32(sb, blk_id, 8) };
        SsaId l2 = sb.emit(blk_id, std::move(_n41));
        SsaNode _n42{};
        _n42.op = tables::SsaOp::BfeS;
        _n42.type = ValType::S32;
        _n42.args = { in_s, operands_host::const_u32(sb, blk_id, 24), operands_host::const_u32(sb, blk_id, 8) };
        SsaId l3 = sb.emit(blk_id, std::move(_n42));
        SsaNode _n43{};
        _n43.op = tables::SsaOp::VecConstruct4;
        _n43.type = ValType::V4U32;
        _n43.args = { l0, l1, l2, l3 };
        return sb.emit(blk_id, std::move(_n43));
    }
    SsaNode _n44{};
    _n44.op = tables::SsaOp::BfeU;
    _n44.type = ValType::U32;
    _n44.args = { raw_u32, operands_host::const_u32(sb, blk_id, 0), operands_host::const_u32(sb, blk_id, 8) };
    SsaId l0 = sb.emit(blk_id, std::move(_n44));
    SsaNode _n45{};
    _n45.op = tables::SsaOp::BfeU;
    _n45.type = ValType::U32;
    _n45.args = { raw_u32, operands_host::const_u32(sb, blk_id, 8), operands_host::const_u32(sb, blk_id, 8) };
    SsaId l1 = sb.emit(blk_id, std::move(_n45));
    SsaNode _n46{};
    _n46.op = tables::SsaOp::BfeU;
    _n46.type = ValType::U32;
    _n46.args = { raw_u32, operands_host::const_u32(sb, blk_id, 16), operands_host::const_u32(sb, blk_id, 8) };
    SsaId l2 = sb.emit(blk_id, std::move(_n46));
    SsaNode _n47{};
    _n47.op = tables::SsaOp::BfeU;
    _n47.type = ValType::U32;
    _n47.args = { raw_u32, operands_host::const_u32(sb, blk_id, 24), operands_host::const_u32(sb, blk_id, 8) };
    SsaId l3 = sb.emit(blk_id, std::move(_n47));
    SsaNode _n48{};
    _n48.op = tables::SsaOp::VecConstruct4;
    _n48.type = ValType::V4U32;
    _n48.args = { l0, l1, l2, l3 };
    return sb.emit(blk_id, std::move(_n48));
}

SsaId fetch_src_u64(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool sign_extend_subword) {
    if (operands_host::src_oob(ins, src_idx)) {
        return operands_host::const_u64(sb, blk_id, 0);
    }
    if (operands_host::src_is_reg(ins, src_idx)) {
        if (operands_host::src_swz_eq(ins, src_idx, "w0")) {
            return _helper_widen_to_u64(sb, blk_id, ins, operands_host::src_read_reg_base(sb, blk_id, ins, src_idx, 0), sign_extend_subword);
        }
        if (operands_host::src_swz_eq(ins, src_idx, "w1")) {
            return _helper_widen_to_u64(sb, blk_id, ins, operands_host::src_read_reg_base(sb, blk_id, ins, src_idx, 1), sign_extend_subword);
        }
        if (!(operands_host::src_is_64bit(ins, src_idx))) {
            return _helper_widen_to_u64(sb, blk_id, ins, operands_host::src_read_reg(sb, blk_id, ins, src_idx), sign_extend_subword);
        }
        return operands_host::emit_pack_u64(sb, blk_id, operands_host::src_read_reg_base(sb, blk_id, ins, src_idx, 0), operands_host::src_read_reg_base(sb, blk_id, ins, src_idx, 1));
    } else if (operands_host::src_is_fau_uniform(ins, src_idx)) {
        uint32_t base = operands_host::src_reg_base(ins, src_idx);
        if (operands_host::src_swz_eq(ins, src_idx, "w0")) {
            return _helper_widen_to_u64(sb, blk_id, ins, operands_host::emit_load_fau_u32(sb, blk_id, base), sign_extend_subword);
        }
        if (operands_host::src_swz_eq(ins, src_idx, "w1")) {
            return _helper_widen_to_u64(sb, blk_id, ins, operands_host::emit_load_fau_u32(sb, blk_id, operands_host::add_u32(base, 1)), sign_extend_subword);
        }
        if (!(operands_host::src_is_64bit(ins, src_idx))) {
            return _helper_widen_to_u64(sb, blk_id, ins, operands_host::emit_load_fau_u32(sb, blk_id, operands_host::src_reg_num(ins, src_idx)), sign_extend_subword);
        }
        return operands_host::emit_pack_u64(sb, blk_id, operands_host::emit_load_fau_u32(sb, blk_id, base), operands_host::emit_load_fau_u32(sb, blk_id, operands_host::add_u32(base, 1)));
    } else if (operands_host::src_is_small_const(ins, src_idx)) {
        return operands_host::src_emit_imm_u64(sb, blk_id, ins, src_idx);
    } else if (operands_host::src_is_fau_pc(ins, src_idx)) {
        SsaId pc_base = operands_host::emit_load_sideband_u64(sb, blk_id, 1);
        SsaNode _n49{};
        _n49.op = tables::SsaOp::IAdd64;
        _n49.type = ValType::U64;
        _n49.args = { pc_base, operands_host::ins_pc_u64(sb, blk_id, ins) };
        _n49.effects.cap_bits |= EffectSet::kCapInt64;
        return sb.emit(blk_id, std::move(_n49));
    } else {
        uint32_t base = operands_host::src_reg_base(ins, src_idx);
        return operands_host::emit_pack_u64(sb, blk_id, operands_host::emit_load_fau_u32(sb, blk_id, base), operands_host::emit_load_fau_u32(sb, blk_id, operands_host::add_u32(base, 1)));
    }
}

void store_dest_u32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val_u32) {
    if (!(operands_host::has_dest(ins))) {
        return;
    }
    uint32_t d = operands_host::dest_reg_num(ins);
    if (operands_host::dest_mod_eq(ins, "h0")) {
        SsaId bfi_id = operands_host::bfi(sb, blk_id, operands_host::dest_read_reg(sb, blk_id, ins), val_u32, 0, 16);
        operands_host::dest_set_reg(sb, blk_id, d, bfi_id);
    } else if (operands_host::dest_mod_eq(ins, "h1")) {
        SsaId bfi_id = operands_host::bfi(sb, blk_id, operands_host::dest_read_reg(sb, blk_id, ins), val_u32, 16, 16);
        operands_host::dest_set_reg(sb, blk_id, d, bfi_id);
    } else if (operands_host::dest_mod_is_byte(ins)) {
        uint32_t off = operands_host::dest_mod_byte_off(ins);
        SsaId bfi_id = operands_host::bfi(sb, blk_id, operands_host::dest_read_reg(sb, blk_id, ins), val_u32, off, 8);
        operands_host::dest_set_reg(sb, blk_id, d, bfi_id);
    } else {
        operands_host::dest_set_reg(sb, blk_id, d, val_u32);
    }
}

void store_dest_f32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val_f32) {
    if (!(operands_host::has_dest(ins))) {
        return;
    }
    val_f32 = _helper_apply_f32_clamp(sb, blk_id, ins, val_f32);
    if ((operands_host::dest_mod_eq(ins, "h0") || operands_host::dest_mod_eq(ins, "h1"))) {
        SsaNode _n50{};
        _n50.op = tables::SsaOp::VecConstruct2;
        _n50.type = ValType::V2F32;
        _n50.args = { val_f32, val_f32 };
        SsaId v2 = sb.emit(blk_id, std::move(_n50));
        SsaNode _n51{};
        _n51.op = tables::SsaOp::PackHalf2x16;
        _n51.type = ValType::U32;
        _n51.args = { v2 };
        _n51.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId packed = sb.emit(blk_id, std::move(_n51));
        uint32_t d = operands_host::dest_reg_num(ins);
        if (operands_host::dest_mod_eq(ins, "h1")) {
            SsaId bfi_id = operands_host::bfi(sb, blk_id, operands_host::dest_read_reg(sb, blk_id, ins), packed, 16, 16);
            operands_host::dest_set_reg(sb, blk_id, d, bfi_id);
        } else {
            SsaId bfi_id = operands_host::bfi(sb, blk_id, operands_host::dest_read_reg(sb, blk_id, ins), packed, 0, 16);
            operands_host::dest_set_reg(sb, blk_id, d, bfi_id);
        }
        return;
    }
    SsaId bits = operands_host::bitcast(sb, blk_id, ValType::U32, val_f32);
    operands_host::dest_set_reg(sb, blk_id, operands_host::dest_reg_num(ins), bits);
}

void store_dest_v2f16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec2_f32) {
    if (!(operands_host::has_dest(ins))) {
        return;
    }
    vec2_f32 = _helper_apply_v2f16_clamp(sb, blk_id, ins, vec2_f32);
    SsaNode _n52{};
    _n52.op = tables::SsaOp::PackHalf2x16;
    _n52.type = ValType::U32;
    _n52.args = { vec2_f32 };
    _n52.effects.flags |= EffectSet::kFlagUsesGlsl450;
    SsaId pk = sb.emit(blk_id, std::move(_n52));
    store_dest_u32(sb, blk_id, ins, pk);
}

void store_dest_v2i16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec2_i32, bool is_signed) {
    if (!(operands_host::has_dest(ins))) {
        return;
    }
    SsaId e0 = _helper_extract_lane_u32(sb, blk_id, ins, vec2_i32, 0, is_signed);
    SsaId e1 = _helper_extract_lane_u32(sb, blk_id, ins, vec2_i32, 1, is_signed);
    SsaId p0 = operands_host::bfi(sb, blk_id, operands_host::const_u32(sb, blk_id, 0), e0, 0, 16);
    SsaId p1 = operands_host::bfi(sb, blk_id, p0, e1, 16, 16);
    store_dest_u32(sb, blk_id, ins, p1);
}

void store_dest_v4i8(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec4_i32, bool is_signed) {
    if (!(operands_host::has_dest(ins))) {
        return;
    }
    SsaId acc = operands_host::const_u32(sb, blk_id, 0);
    for (uint32_t i = 0; i < 4; ++i) {
        SsaId ei = _helper_extract_lane_u32(sb, blk_id, ins, vec4_i32, i, is_signed);
        acc = operands_host::bfi(sb, blk_id, acc, ei, operands_host::mul8(i), 8);
    }
    store_dest_u32(sb, blk_id, ins, acc);
}

void store_dest_u64(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val_u64) {
    if (!(operands_host::has_dest(ins))) {
        return;
    }
    SsaId lo = operands_host::split64_lo(sb, blk_id, val_u64);
    SsaId hi = operands_host::split64_hi(sb, blk_id, val_u64);
    uint32_t d = operands_host::dest_reg_base(ins);
    operands_host::dest_set_reg(sb, blk_id, d, lo);
    operands_host::dest_set_reg(sb, blk_id, operands_host::add_u32(d, 1), hi);
}

} // namespace valhall::compiler
