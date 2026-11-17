#include <cstring>
#include "valhall_operands.h"
#include <algorithm>

namespace valhall::compiler {

std::string get_src_swz(const DecodedValhallInst& ins, size_t src_idx) {
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

static bool get_src_abs(const DecodedValhallInst& ins, size_t src_idx) {
    if (src_idx < ins.srcs.size() && ins.srcs[src_idx].abs) return true;
    auto it = ins.fields.find("abs" + std::to_string(src_idx));
    return (it != ins.fields.end() && it->second != 0);
}

static bool get_src_neg(const DecodedValhallInst& ins, size_t src_idx) {
    if (src_idx < ins.srcs.size() && ins.srcs[src_idx].neg) return true;
    auto it = ins.fields.find("neg" + std::to_string(src_idx));
    return (it != ins.fields.end() && it->second != 0);
}

static bool get_src_not(const DecodedValhallInst& ins, size_t src_idx) {
    if (src_idx < ins.srcs.size() && ins.srcs[src_idx].not_mod) return true;
    auto it = ins.fields.find("not" + std::to_string(src_idx));
    return (it != ins.fields.end() && it->second != 0);
}

std::string get_dest_mod(const DecodedValhallInst& ins) {
    if (!ins.dest_mod.empty()) return ins.dest_mod;
    for (const char* k : {"dest_width", "dest_width_narrow"}) {
        auto it = ins.modifier_strs.find(k);
        if (it != ins.modifier_strs.end() && it->second != "none") {
            return it->second;
        }
    }
    return "";
}

static SsaId emit_load_fau_u32(SsaBuilder& sb, uint32_t blk_id, uint32_t word_idx) {
    SsaNode n{};
    n.op = tables::SsaOp::LoadFau;
    n.type = ValType::U32;
    n.imm_u32 = word_idx & 63u;
    n.effects.flags |= EffectSet::kFlagUsesSideband;
    return sb.emit(blk_id, std::move(n));
}

static SsaId emit_load_sideband_u64(SsaBuilder& sb, uint32_t blk_id, uint32_t member_idx) {
    SsaNode n{};
    n.op = tables::SsaOp::LoadSidebandU64;
    n.type = ValType::U64;
    n.imm_u32 = member_idx;
    n.effects.flags |= EffectSet::kFlagUsesSideband;
    n.effects.cap_bits |= EffectSet::kCapInt64;
    return sb.emit(blk_id, std::move(n));
}

static SsaId emit_pack_u64(SsaBuilder& sb, uint32_t blk_id, SsaId lo_u32, SsaId hi_u32) {
    SsaNode n{};
    n.op = tables::SsaOp::Pack64;
    n.type = ValType::U64;
    n.args = { lo_u32, hi_u32 };
    n.effects.cap_bits |= EffectSet::kCapInt64;
    return sb.emit(blk_id, std::move(n));
}

static void emit_split_u64(SsaBuilder& sb, uint32_t blk_id, SsaId val_u64, SsaId* out_lo, SsaId* out_hi) {
    SsaNode sp{};
    sp.op = tables::SsaOp::Split64;
    sp.type = ValType::TupleU32x2;
    sp.args = { val_u64 };
    sp.effects.cap_bits |= EffectSet::kCapInt64;
    SsaId tup = sb.emit(blk_id, std::move(sp));

    SsaNode ex0{};
    ex0.op = tables::SsaOp::TupleExtract;
    ex0.type = ValType::U32;
    ex0.imm_u32 = 0;
    ex0.args = { tup };
    *out_lo = sb.emit(blk_id, std::move(ex0));

    SsaNode ex1{};
    ex1.op = tables::SsaOp::TupleExtract;
    ex1.type = ValType::U32;
    ex1.imm_u32 = 1;
    ex1.args = { tup };
    *out_hi = sb.emit(blk_id, std::move(ex1));
}

SsaId fetch_raw_src_u32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx) {
    if (src_idx >= ins.srcs.size()) return sb.emit_const_u32(blk_id, 0u);
    const auto& s = ins.srcs[src_idx];
    switch (s.kind) {
        case SourceKind::Register:
            return sb.current_reg(blk_id, s.reg_num);
        case SourceKind::FauUniform:
            return emit_load_fau_u32(sb, blk_id, s.reg_num);
        case SourceKind::SmallConstant:
            return sb.emit_const_u32(blk_id, s.imm_bits);
        case SourceKind::FauSpecial: {
            if (s.symbolic_name == "pc") {
                SsaId pc_base = emit_load_sideband_u64(sb, blk_id, 1);
                SsaId pc_off = sb.emit_const_u64(blk_id, ins.pc);
                SsaNode add{};
                add.op = tables::SsaOp::IAdd64;
                add.type = ValType::U64;
                add.args = { pc_base, pc_off };
                add.effects.cap_bits |= EffectSet::kCapInt64;
                SsaId cur_pc = sb.emit(blk_id, std::move(add));
                SsaId lo = 0, hi = 0;
                emit_split_u64(sb, blk_id, cur_pc, &lo, &hi);
                return s.word_select ? hi : lo;
            }
            if (s.symbolic_name == "lane_id") {
                SsaNode bi{};
                bi.op = tables::SsaOp::LoadBuiltinInput;
                bi.type = ValType::U32;
                bi.imm_u32 = static_cast<uint32_t>(spirv::BuiltIn::SubgroupLocalInvocationId);
                bi.aux_bool = true; // flat
                bi.aux_str = "gl_SubgroupInvocationID";
                return sb.emit(blk_id, std::move(bi));
            }
            if (s.symbolic_name == "lane_width") {
                SsaNode sb32{};
                sb32.op = tables::SsaOp::LoadSidebandU32;
                sb32.type = ValType::U32;
                sb32.imm_u32 = 5;
                sb32.imm_u32_2 = 3;
                sb32.effects.flags |= EffectSet::kFlagUsesSideband;
                return sb.emit(blk_id, std::move(sb32));
            }
            return emit_load_fau_u32(sb, blk_id, s.reg_num);
        }
    }
    return sb.emit_const_u32(blk_id, 0u);
}

SsaId fetch_src_int32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool is_signed) {
    SsaId val = fetch_raw_src_u32(sb, blk_id, ins, src_idx);
    std::string swz = get_src_swz(ins, src_idx);

    auto extract_bits = [&](uint32_t offset, uint32_t count) -> SsaId {
        if (is_signed) {
            SsaId sval = sb.emit_bitcast(blk_id, ValType::S32, val);
            SsaNode bfe{};
            bfe.op = tables::SsaOp::BfeS;
            bfe.type = ValType::S32;
            bfe.args = { sval, sb.emit_const_u32(blk_id, offset), sb.emit_const_u32(blk_id, count) };
            SsaId ext = sb.emit(blk_id, std::move(bfe));
            return sb.emit_bitcast(blk_id, ValType::U32, ext);
        } else {
            SsaNode bfe{};
            bfe.op = tables::SsaOp::BfeU;
            bfe.type = ValType::U32;
            bfe.args = { val, sb.emit_const_u32(blk_id, offset), sb.emit_const_u32(blk_id, count) };
            return sb.emit(blk_id, std::move(bfe));
        }
    };

    if (swz == "b0") val = extract_bits(0, 8);
    else if (swz == "b1") val = extract_bits(8, 8);
    else if (swz == "b2") val = extract_bits(16, 8);
    else if (swz == "b3") val = extract_bits(24, 8);
    else if (swz == "h0") val = extract_bits(0, 16);
    else if (swz == "h1") val = extract_bits(16, 16);
    else if (swz == "h00" || swz == "h11") {
        SsaId h = extract_bits(swz == "h11" ? 16u : 0u, 16u);
        SsaNode sh{};
        sh.op = tables::SsaOp::Shl;
        sh.type = ValType::U32;
        sh.args = { h, sb.emit_const_u32(blk_id, 16u) };
        SsaId hi = sb.emit(blk_id, std::move(sh));
        SsaNode bor{};
        bor.op = tables::SsaOp::BitOr;
        bor.type = ValType::U32;
        bor.args = { h, hi };
        val = sb.emit(blk_id, std::move(bor));
    } else if (swz == "b0000" || swz == "b1111" || swz == "b2222" || swz == "b3333") {
        uint32_t byte_idx = static_cast<uint32_t>(swz[1] - '0');
        SsaNode bfe{};
        bfe.op = tables::SsaOp::BfeU;
        bfe.type = ValType::U32;
        bfe.args = { val, sb.emit_const_u32(blk_id, byte_idx * 8u), sb.emit_const_u32(blk_id, 8u) };
        SsaId bk = sb.emit(blk_id, std::move(bfe));
        SsaNode mul{};
        mul.op = tables::SsaOp::IMul;
        mul.type = ValType::U32;
        mul.args = { bk, sb.emit_const_u32(blk_id, 0x01010101u) };
        val = sb.emit(blk_id, std::move(mul));
    }

    if (get_src_abs(ins, src_idx)) {
        SsaId sval = sb.emit_bitcast(blk_id, ValType::S32, val);
        SsaNode ab{};
        ab.op = tables::SsaOp::SAbs;
        ab.type = ValType::S32;
        ab.args = { sval };
        ab.effects.flags |= EffectSet::kFlagUsesGlsl450;
        val = sb.emit_bitcast(blk_id, ValType::U32, sb.emit(blk_id, std::move(ab)));
    }
    if (get_src_neg(ins, src_idx)) {
        SsaId sval = sb.emit_bitcast(blk_id, ValType::S32, val);
        SsaNode ng{};
        ng.op = tables::SsaOp::SNeg;
        ng.type = ValType::S32;
        ng.args = { sval };
        val = sb.emit_bitcast(blk_id, ValType::U32, sb.emit(blk_id, std::move(ng)));
    }
    if (get_src_not(ins, src_idx)) {
        SsaNode nt{};
        nt.op = tables::SsaOp::BitNot;
        nt.type = ValType::U32;
        nt.args = { val };
        val = sb.emit(blk_id, std::move(nt));
    }
    return val;
}

SsaId fetch_src_f32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx) {
    SsaId raw_u32 = fetch_raw_src_u32(sb, blk_id, ins, src_idx);
    std::string swz = get_src_swz(ins, src_idx);
    SsaId fval = kInvalidSsa;
    if (swz == "h0" || swz == "h1") {
        SsaNode unp{};
        unp.op = tables::SsaOp::UnpackHalf2x16;
        unp.type = ValType::V2F32;
        unp.args = { raw_u32 };
        unp.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId vec = sb.emit(blk_id, std::move(unp));

        SsaNode ex{};
        ex.op = tables::SsaOp::VecExtract;
        ex.type = ValType::F32;
        ex.imm_u32 = (swz == "h1") ? 1u : 0u;
        ex.args = { vec };
        fval = sb.emit(blk_id, std::move(ex));
    } else {
        fval = sb.emit_bitcast(blk_id, ValType::F32, raw_u32);
    }

    if (get_src_abs(ins, src_idx)) {
        SsaNode ab{};
        ab.op = tables::SsaOp::FAbs;
        ab.type = ValType::F32;
        ab.args = { fval };
        ab.effects.flags |= EffectSet::kFlagUsesGlsl450;
        fval = sb.emit(blk_id, std::move(ab));
    }
    if (get_src_neg(ins, src_idx)) {
        SsaNode ng{};
        ng.op = tables::SsaOp::FNeg;
        ng.type = ValType::F32;
        ng.args = { fval };
        fval = sb.emit(blk_id, std::move(ng));
    }
    return fval;
}

SsaId fetch_src_v2f16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx) {
    SsaId raw_u32 = fetch_raw_src_u32(sb, blk_id, ins, src_idx);
    SsaNode unp{};
    unp.op = tables::SsaOp::UnpackHalf2x16;
    unp.type = ValType::V2F32;
    unp.args = { raw_u32 };
    unp.effects.flags |= EffectSet::kFlagUsesGlsl450;
    SsaId vec = sb.emit(blk_id, std::move(unp));

    std::string swz = get_src_swz(ins, src_idx);
    if (swz == "h00" || swz == "h0" || swz == "h11" || swz == "h1" || swz == "h10") {
        SsaNode sh{};
        sh.op = tables::SsaOp::VecShuffle;
        sh.type = ValType::V2F32;
        sh.args = { vec, vec };
        if (swz == "h00" || swz == "h0") { sh.imm_u32 = 0; sh.imm_u32_2 = 0; }
        else if (swz == "h11" || swz == "h1") { sh.imm_u32 = 1; sh.imm_u32_2 = 1; }
        else { sh.imm_u32 = 1; sh.imm_u32_2 = 0; }
        vec = sb.emit(blk_id, std::move(sh));
    }

    if (get_src_abs(ins, src_idx)) {
        SsaNode ab{};
        ab.op = tables::SsaOp::FAbs;
        ab.type = ValType::V2F32;
        ab.args = { vec };
        ab.effects.flags |= EffectSet::kFlagUsesGlsl450;
        vec = sb.emit(blk_id, std::move(ab));
    }
    if (get_src_neg(ins, src_idx)) {
        SsaNode ng{};
        ng.op = tables::SsaOp::FNeg;
        ng.type = ValType::V2F32;
        ng.args = { vec };
        vec = sb.emit(blk_id, std::move(ng));
    }
    return vec;
}

SsaId fetch_src_v2i16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool is_signed) {
    SsaId raw_u32 = fetch_raw_src_u32(sb, blk_id, ins, src_idx);
    ValType elem_t = is_signed ? ValType::S32 : ValType::U32;
    ValType v2t = is_signed ? ValType::V2S32 : ValType::V2U32;
    SsaId in_val = is_signed ? sb.emit_bitcast(blk_id, ValType::S32, raw_u32) : raw_u32;
    tables::SsaOp bfe_op = is_signed ? tables::SsaOp::BfeS : tables::SsaOp::BfeU;

    SsaNode b0{};
    b0.op = bfe_op; b0.type = elem_t;
    b0.args = { in_val, sb.emit_const_u32(blk_id, 0u), sb.emit_const_u32(blk_id, 16u) };
    SsaId l0 = sb.emit(blk_id, std::move(b0));

    SsaNode b1{};
    b1.op = bfe_op; b1.type = elem_t;
    b1.args = { in_val, sb.emit_const_u32(blk_id, 16u), sb.emit_const_u32(blk_id, 16u) };
    SsaId l1 = sb.emit(blk_id, std::move(b1));

    SsaNode vc{};
    vc.op = tables::SsaOp::VecConstruct2; vc.type = v2t; vc.args = { l0, l1 };
    SsaId vec = sb.emit(blk_id, std::move(vc));

    std::string swz = get_src_swz(ins, src_idx);
    if (swz == "h00" || swz == "h0" || swz == "h11" || swz == "h1" || swz == "h10") {
        SsaNode sh{};
        sh.op = tables::SsaOp::VecShuffle; sh.type = v2t; sh.args = { vec, vec };
        if (swz == "h00" || swz == "h0") { sh.imm_u32 = 0; sh.imm_u32_2 = 0; }
        else if (swz == "h11" || swz == "h1") { sh.imm_u32 = 1; sh.imm_u32_2 = 1; }
        else { sh.imm_u32 = 1; sh.imm_u32_2 = 0; }
        vec = sb.emit(blk_id, std::move(sh));
    }
    return vec;
}

SsaId fetch_src_v4i8(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool is_signed) {
    SsaId raw_u32 = fetch_raw_src_u32(sb, blk_id, ins, src_idx);
    ValType elem_t = is_signed ? ValType::S32 : ValType::U32;
    ValType v4t = is_signed ? ValType::V4S32 : ValType::V4U32;
    SsaId in_val = is_signed ? sb.emit_bitcast(blk_id, ValType::S32, raw_u32) : raw_u32;
    tables::SsaOp bfe_op = is_signed ? tables::SsaOp::BfeS : tables::SsaOp::BfeU;

    std::vector<SsaId> lanes(4);
    for (uint32_t i = 0; i < 4; ++i) {
        SsaNode b{};
        b.op = bfe_op; b.type = elem_t;
        b.args = { in_val, sb.emit_const_u32(blk_id, i * 8u), sb.emit_const_u32(blk_id, 8u) };
        lanes[i] = sb.emit(blk_id, std::move(b));
    }
    SsaNode vc{};
    vc.op = tables::SsaOp::VecConstruct4; vc.type = v4t; vc.args = std::move(lanes);
    return sb.emit(blk_id, std::move(vc));
}

SsaId fetch_src_u64(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool sign_extend_subword) {
    if (src_idx >= ins.srcs.size()) return sb.emit_const_u64(blk_id, 0ULL);
    const auto& s = ins.srcs[src_idx];
    std::string swz = get_src_swz(ins, src_idx);

    auto widen = [&](SsaId w32, bool sext) -> SsaId {
        if (sext) {
            SsaId s32 = sb.emit_bitcast(blk_id, ValType::S32, w32);
            SsaNode cvt{};
            cvt.op = tables::SsaOp::SConvert32To64;
            cvt.type = ValType::S64;
            cvt.args = { s32 };
            cvt.effects.cap_bits |= EffectSet::kCapInt64;
            return sb.emit_bitcast(blk_id, ValType::U64, sb.emit(blk_id, std::move(cvt)));
        } else {
            SsaNode cvt{};
            cvt.op = tables::SsaOp::UConvert32To64;
            cvt.type = ValType::U64;
            cvt.args = { w32 };
            cvt.effects.cap_bits |= EffectSet::kCapInt64;
            return sb.emit(blk_id, std::move(cvt));
        }
    };

    if (s.kind == SourceKind::Register) {
        uint8_t base = s.reg_num & 62u;
        if (swz == "w0") return widen(sb.current_reg(blk_id, base), sign_extend_subword);
        if (swz == "w1") return widen(sb.current_reg(blk_id, base + 1), sign_extend_subword);
        if (!s.is_64bit) return widen(sb.current_reg(blk_id, s.reg_num), sign_extend_subword);
        return emit_pack_u64(sb, blk_id, sb.current_reg(blk_id, base), sb.current_reg(blk_id, base + 1));
    } else if (s.kind == SourceKind::FauUniform) {
        uint8_t base = s.reg_num & 62u;
        if (swz == "w0") return widen(emit_load_fau_u32(sb, blk_id, base), sign_extend_subword);
        if (swz == "w1") return widen(emit_load_fau_u32(sb, blk_id, base + 1), sign_extend_subword);
        if (!s.is_64bit) return widen(emit_load_fau_u32(sb, blk_id, s.reg_num), sign_extend_subword);
        return emit_pack_u64(sb, blk_id, emit_load_fau_u32(sb, blk_id, base), emit_load_fau_u32(sb, blk_id, base + 1));
    } else if (s.kind == SourceKind::SmallConstant) {
        return sb.emit_const_u64(blk_id, uint64_t(s.imm_bits));
    } else if (s.kind == SourceKind::FauSpecial) {
        if (s.symbolic_name == "pc") {
            SsaId pc_base = emit_load_sideband_u64(sb, blk_id, 1);
            SsaNode add{};
            add.op = tables::SsaOp::IAdd64;
            add.type = ValType::U64;
            add.args = { pc_base, sb.emit_const_u64(blk_id, ins.pc) };
            add.effects.cap_bits |= EffectSet::kCapInt64;
            return sb.emit(blk_id, std::move(add));
        }
        uint8_t base = s.reg_num & 62u;
        return emit_pack_u64(sb, blk_id, emit_load_fau_u32(sb, blk_id, base), emit_load_fau_u32(sb, blk_id, base + 1));
    }
    return sb.emit_const_u64(blk_id, 0ULL);
}

void store_dest_u32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val_u32) {
    if (!ins.has_dest) return;
    uint8_t d = ins.dest_reg & 63u;
    std::string dm = get_dest_mod(ins);
    if (dm == "h0" || dm == "h1") {
        uint32_t off = (dm == "h1") ? 16u : 0u;
        SsaNode bfi{};
        bfi.op = tables::SsaOp::Bfi;
        bfi.type = ValType::U32;
        bfi.imm_u32 = off;
        bfi.imm_u32_2 = 16u;
        bfi.args = { sb.current_reg(blk_id, d), val_u32, sb.emit_const_u32(blk_id, off), sb.emit_const_u32(blk_id, 16u) };
        sb.set_reg(blk_id, d, sb.emit(blk_id, std::move(bfi)));
    } else if (dm == "b0" || dm == "b1" || dm == "b2" || dm == "b3") {
        uint32_t off = static_cast<uint32_t>(dm[1] - '0') * 8u;
        SsaNode bfi{};
        bfi.op = tables::SsaOp::Bfi;
        bfi.type = ValType::U32;
        bfi.imm_u32 = off;
        bfi.imm_u32_2 = 8u;
        bfi.args = { sb.current_reg(blk_id, d), val_u32, sb.emit_const_u32(blk_id, off), sb.emit_const_u32(blk_id, 8u) };
        sb.set_reg(blk_id, d, sb.emit(blk_id, std::move(bfi)));
    } else {
        sb.set_reg(blk_id, d, val_u32);
    }
}

void store_dest_f32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val_f32) {
    if (!ins.has_dest) return;
    auto it_clamp = ins.modifier_strs.find("clamp");
    if (it_clamp != ins.modifier_strs.end()) {
        const std::string& cl = it_clamp->second;
        if (cl == "clamp_0_1" || cl == "clamp_m1_1") {
            float lo = (cl == "clamp_m1_1") ? -1.0f : 0.0f;
            uint32_t lo_bits = 0, hi_bits = 0x3f800000u;
            std::memcpy(&lo_bits, &lo, sizeof(float));
            SsaNode cm{};
            cm.op = tables::SsaOp::FClamp;
            cm.type = ValType::F32;
            cm.args = { val_f32, sb.emit_const_f32_bits(blk_id, lo_bits), sb.emit_const_f32_bits(blk_id, hi_bits) };
            cm.effects.flags |= EffectSet::kFlagUsesGlsl450;
            val_f32 = sb.emit(blk_id, std::move(cm));
        } else if (cl == "clamp_0_inf") {
            SsaNode mx{};
            mx.op = tables::SsaOp::FMax;
            mx.type = ValType::F32;
            mx.args = { val_f32, sb.emit_const_f32_bits(blk_id, 0u) };
            mx.effects.flags |= EffectSet::kFlagUsesGlsl450;
            val_f32 = sb.emit(blk_id, std::move(mx));
        }
    }

    std::string dm = get_dest_mod(ins);
    if (dm == "h0" || dm == "h1") {
        SsaNode vc{};
        vc.op = tables::SsaOp::VecConstruct2;
        vc.type = ValType::V2F32;
        vc.args = { val_f32, val_f32 };
        SsaId v2 = sb.emit(blk_id, std::move(vc));

        SsaNode pk{};
        pk.op = tables::SsaOp::PackHalf2x16;
        pk.type = ValType::U32;
        pk.args = { v2 };
        pk.effects.flags |= EffectSet::kFlagUsesGlsl450;
        SsaId packed = sb.emit(blk_id, std::move(pk));

        uint8_t d = ins.dest_reg & 63u;
        uint32_t off = (dm == "h1") ? 16u : 0u;
        SsaNode bfi{};
        bfi.op = tables::SsaOp::Bfi;
        bfi.type = ValType::U32;
        bfi.imm_u32 = off;
        bfi.imm_u32_2 = 16u;
        bfi.args = { sb.current_reg(blk_id, d), packed, sb.emit_const_u32(blk_id, off), sb.emit_const_u32(blk_id, 16u) };
        sb.set_reg(blk_id, d, sb.emit(blk_id, std::move(bfi)));
        return;
    }

    SsaId bits = sb.emit_bitcast(blk_id, ValType::U32, val_f32);
    sb.set_reg(blk_id, ins.dest_reg & 63u, bits);
}

void store_dest_v2f16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec2_f32) {
    if (!ins.has_dest) return;
    auto it_clamp = ins.modifier_strs.find("clamp");
    if (it_clamp != ins.modifier_strs.end()) {
        const std::string& cl = it_clamp->second;
        if (cl == "clamp_0_1" || cl == "clamp_m1_1") {
            float lo_f = (cl == "clamp_m1_1") ? -1.0f : 0.0f;
            uint32_t lo_bits = 0, hi_bits = 0x3f800000u;
            std::memcpy(&lo_bits, &lo_f, sizeof(float));
            SsaId lo_s = sb.emit_const_f32_bits(blk_id, lo_bits);
            SsaId hi_s = sb.emit_const_f32_bits(blk_id, hi_bits);
            SsaNode vlo{}; vlo.op = tables::SsaOp::VecConstruct2; vlo.type = ValType::V2F32; vlo.args = { lo_s, lo_s };
            SsaNode vhi{}; vhi.op = tables::SsaOp::VecConstruct2; vhi.type = ValType::V2F32; vhi.args = { hi_s, hi_s };
            SsaNode cm{};
            cm.op = tables::SsaOp::FClamp; cm.type = ValType::V2F32;
            cm.args = { vec2_f32, sb.emit(blk_id, std::move(vlo)), sb.emit(blk_id, std::move(vhi)) };
            cm.effects.flags |= EffectSet::kFlagUsesGlsl450;
            vec2_f32 = sb.emit(blk_id, std::move(cm));
        } else if (cl == "clamp_0_inf") {
            SsaId z = sb.emit_const_f32_bits(blk_id, 0u);
            SsaNode vz{}; vz.op = tables::SsaOp::VecConstruct2; vz.type = ValType::V2F32; vz.args = { z, z };
            SsaNode mx{};
            mx.op = tables::SsaOp::FMax; mx.type = ValType::V2F32;
            mx.args = { vec2_f32, sb.emit(blk_id, std::move(vz)) };
            mx.effects.flags |= EffectSet::kFlagUsesGlsl450;
            vec2_f32 = sb.emit(blk_id, std::move(mx));
        }
    }

    SsaNode pk{};
    pk.op = tables::SsaOp::PackHalf2x16;
    pk.type = ValType::U32;
    pk.args = { vec2_f32 };
    pk.effects.flags |= EffectSet::kFlagUsesGlsl450;
    store_dest_u32(sb, blk_id, ins, sb.emit(blk_id, std::move(pk)));
}

void store_dest_v2i16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec2_i32, bool is_signed) {
    if (!ins.has_dest) return;
    ValType elem_t = is_signed ? ValType::S32 : ValType::U32;
    SsaNode ex0{}; ex0.op = tables::SsaOp::VecExtract; ex0.type = elem_t; ex0.imm_u32 = 0; ex0.args = { vec2_i32 };
    SsaNode ex1{}; ex1.op = tables::SsaOp::VecExtract; ex1.type = elem_t; ex1.imm_u32 = 1; ex1.args = { vec2_i32 };
    SsaId e0 = sb.emit_bitcast(blk_id, ValType::U32, sb.emit(blk_id, std::move(ex0)));
    SsaId e1 = sb.emit_bitcast(blk_id, ValType::U32, sb.emit(blk_id, std::move(ex1)));

    SsaNode b0{};
    b0.op = tables::SsaOp::Bfi; b0.type = ValType::U32; b0.imm_u32 = 0; b0.imm_u32_2 = 16;
    b0.args = { sb.emit_const_u32(blk_id, 0u), e0, sb.emit_const_u32(blk_id, 0u), sb.emit_const_u32(blk_id, 16u) };
    SsaId p0 = sb.emit(blk_id, std::move(b0));

    SsaNode b1{};
    b1.op = tables::SsaOp::Bfi; b1.type = ValType::U32; b1.imm_u32 = 16; b1.imm_u32_2 = 16;
    b1.args = { p0, e1, sb.emit_const_u32(blk_id, 16u), sb.emit_const_u32(blk_id, 16u) };
    store_dest_u32(sb, blk_id, ins, sb.emit(blk_id, std::move(b1)));
}

void store_dest_v4i8(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec4_i32, bool is_signed) {
    if (!ins.has_dest) return;
    ValType elem_t = is_signed ? ValType::S32 : ValType::U32;
    SsaId acc = sb.emit_const_u32(blk_id, 0u);
    for (uint32_t i = 0; i < 4; ++i) {
        SsaNode ex{}; ex.op = tables::SsaOp::VecExtract; ex.type = elem_t; ex.imm_u32 = i; ex.args = { vec4_i32 };
        SsaId ei = sb.emit_bitcast(blk_id, ValType::U32, sb.emit(blk_id, std::move(ex)));
        SsaNode bi{};
        bi.op = tables::SsaOp::Bfi; bi.type = ValType::U32; bi.imm_u32 = i * 8u; bi.imm_u32_2 = 8u;
        bi.args = { acc, ei, sb.emit_const_u32(blk_id, i * 8u), sb.emit_const_u32(blk_id, 8u) };
        acc = sb.emit(blk_id, std::move(bi));
    }
    store_dest_u32(sb, blk_id, ins, acc);
}

void store_dest_u64(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val_u64) {
    if (!ins.has_dest) return;
    SsaId lo = 0, hi = 0;
    emit_split_u64(sb, blk_id, val_u64, &lo, &hi);
    uint8_t d = ins.dest_reg & 62u;
    sb.set_reg(blk_id, d, lo);
    sb.set_reg(blk_id, d + 1, hi);
}

} // namespace valhall::compiler
