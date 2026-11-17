#include "valhall_to_spirv.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace valhall {

using namespace valhall::spirv;

namespace {

struct TranslatorContext {
    SpirvBuilder b;
    ValhallTranslateOptions opts;
    uint32_t reg_vars[64] = {0};

    // Sideband buffer variable & member info
    uint32_t sideband_struct_t = 0;
    uint32_t sideband_var_id = 0;

    // Graphics interface variables
    std::map<uint32_t, uint32_t> vert_attr_inputs_;      // location -> OpVariable Input vec4
    std::map<uint32_t, uint32_t> vert_varying_outputs_;  // location -> OpVariable Output vec4
    std::map<uint32_t, uint32_t> frag_varying_inputs_;   // location -> OpVariable Input vec4
    std::map<uint32_t, uint32_t> frag_color_outputs_;    // location -> OpVariable Output vec4
    std::map<uint32_t, uint32_t> sampled_image_vars_;    // index -> OpVariable UniformConstant SampledImage
    std::map<uint32_t, uint32_t> sampled_image_types_;   // index -> OpTypeSampledImage

    TranslatorContext(ExecutionModel model, const ValhallTranslateOptions& o)
        : b(model, true), opts(o) {}

    uint32_t load_reg_u32(uint8_t r) {
        return b.emit_load(b.type_u32(), reg_vars[r & 63]);
    }

    void store_reg_u32(uint8_t r, uint32_t val_u32) {
        b.emit_store(reg_vars[r & 63], val_u32);
    }

    uint32_t load_fau_u32(uint32_t word_idx) {
        uint32_t ptr_u32 = b.type_pointer(StorageClass::StorageBuffer, b.type_u32());
        uint32_t ac = b.emit_access_chain(ptr_u32, sideband_var_id,
                                          { b.const_u32(0), b.const_u32(word_idx & 63u) });
        return b.emit_load(b.type_u32(), ac);
    }

    uint32_t load_sideband_u64(uint32_t member_idx) {
        uint32_t ptr_u64 = b.type_pointer(StorageClass::StorageBuffer, b.type_u64());
        uint32_t ac = b.emit_access_chain(ptr_u64, sideband_var_id, { b.const_u32(member_idx) });
        return b.emit_load(b.type_u64(), ac);
    }

    uint32_t pack_u64(uint32_t lo_u32, uint32_t hi_u32) {
        uint32_t v2u32 = b.type_vector(b.type_u32(), 2);
        uint32_t vec = b.emit_typed_inst(Op::OpCompositeConstruct, v2u32, { lo_u32, hi_u32 });
        return b.emit_bitcast(b.type_u64(), vec);
    }

    void unpack_u64(uint32_t val_u64, uint32_t* out_lo, uint32_t* out_hi) {
        uint32_t v2u32 = b.type_vector(b.type_u32(), 2);
        uint32_t vec = b.emit_bitcast(v2u32, val_u64);
        *out_lo = b.emit_typed_inst(Op::OpCompositeExtract, b.type_u32(), { vec, 0u });
        *out_hi = b.emit_typed_inst(Op::OpCompositeExtract, b.type_u32(), { vec, 1u });
    }

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

    bool get_src_abs(const DecodedValhallInst& ins, size_t src_idx) {
        if (src_idx < ins.srcs.size() && ins.srcs[src_idx].abs) return true;
        if (src_idx == 0 && ins.mnemonic == Mnemonic::FABS) return true;
        auto it = ins.fields.find("abs" + std::to_string(src_idx));
        return (it != ins.fields.end() && it->second != 0);
    }

    bool get_src_neg(const DecodedValhallInst& ins, size_t src_idx) {
        if (src_idx < ins.srcs.size() && ins.srcs[src_idx].neg) return true;
        if (src_idx == 0 && ins.mnemonic == Mnemonic::FNEG) return true;
        auto it = ins.fields.find("neg" + std::to_string(src_idx));
        return (it != ins.fields.end() && it->second != 0);
    }

    bool get_src_not(const DecodedValhallInst& ins, size_t src_idx) {
        if (src_idx < ins.srcs.size() && ins.srcs[src_idx].not_mod) return true;
        auto it = ins.fields.find("not" + std::to_string(src_idx));
        return (it != ins.fields.end() && it->second != 0);
    }

    uint32_t load_raw_src_u32(const DecodedValhallInst& ins, size_t src_idx) {
        if (src_idx >= ins.srcs.size()) return b.const_u32(0);
        const auto& s = ins.srcs[src_idx];
        switch (s.kind) {
            case SourceKind::Register:
                return load_reg_u32(s.reg_num);
            case SourceKind::FauUniform:
                return load_fau_u32(s.reg_num);
            case SourceKind::SmallConstant:
                return b.const_u32(s.imm_bits);
            case SourceKind::FauSpecial: {
                if (s.symbolic_name == "pc") {
                    uint32_t pc_base = load_sideband_u64(1);
                    uint32_t cur_pc = b.emit_typed_inst(Op::OpIAdd, b.type_u64(),
                                                        { pc_base, b.const_u64(ins.pc) });
                    uint32_t lo = 0, hi = 0;
                    unpack_u64(cur_pc, &lo, &hi);
                    return s.word_select ? hi : lo;
                }
                if (s.symbolic_name == "lane_id") {
                    uint32_t var = b.get_or_create_builtin_input(BuiltIn::SubgroupLocalInvocationId, b.type_u32(), "gl_SubgroupInvocationID", true);
                    return b.emit_load(b.type_u32(), var);
                }
                if (s.symbolic_name == "lane_width") {
                    uint32_t ptr_u32 = b.type_pointer(StorageClass::StorageBuffer, b.type_u32());
                    uint32_t ac = b.emit_access_chain(ptr_u32, sideband_var_id, { b.const_u32(5), b.const_u32(3) });
                    return b.emit_load(b.type_u32(), ac);
                }
                return load_fau_u32(s.reg_num);
            }
        }
        return b.const_u32(0);
    }

    uint32_t load_src_int32(const DecodedValhallInst& ins, size_t src_idx, bool is_signed = false) {
        uint32_t val = load_raw_src_u32(ins, src_idx);
        std::string swz = get_src_swz(ins, src_idx);

        auto extract_bits = [&](uint32_t offset, uint32_t count) -> uint32_t {
            if (is_signed) {
                uint32_t sval = b.emit_bitcast(b.type_s32(), val);
                uint32_t ext = b.emit_typed_inst(Op::OpBitFieldSExtract, b.type_s32(),
                                                 { sval, b.const_u32(offset), b.const_u32(count) });
                return b.emit_bitcast(b.type_u32(), ext);
            } else {
                return b.emit_typed_inst(Op::OpBitFieldUExtract, b.type_u32(),
                                         { val, b.const_u32(offset), b.const_u32(count) });
            }
        };

        if (swz == "b0") val = extract_bits(0, 8);
        else if (swz == "b1") val = extract_bits(8, 8);
        else if (swz == "b2") val = extract_bits(16, 8);
        else if (swz == "b3") val = extract_bits(24, 8);
        else if (swz == "h0") val = extract_bits(0, 16);
        else if (swz == "h1") val = extract_bits(16, 16);
        else if (swz == "h00") {
            uint32_t h0 = b.emit_typed_inst(Op::OpBitFieldUExtract, b.type_u32(), { val, b.const_u32(0), b.const_u32(16) });
            uint32_t hi = b.emit_typed_inst(Op::OpShiftLeftLogical, b.type_u32(), { h0, b.const_u32(16) });
            val = b.emit_typed_inst(Op::OpBitwiseOr, b.type_u32(), { h0, hi });
        } else if (swz == "h11") {
            uint32_t h1 = b.emit_typed_inst(Op::OpBitFieldUExtract, b.type_u32(), { val, b.const_u32(16), b.const_u32(16) });
            uint32_t hi = b.emit_typed_inst(Op::OpShiftLeftLogical, b.type_u32(), { h1, b.const_u32(16) });
            val = b.emit_typed_inst(Op::OpBitwiseOr, b.type_u32(), { h1, hi });
        } else if (swz == "b0000" || swz == "b1111" || swz == "b2222" || swz == "b3333") {
            uint32_t byte_idx = uint32_t(swz[1] - '0');
            uint32_t bk = b.emit_typed_inst(Op::OpBitFieldUExtract, b.type_u32(), { val, b.const_u32(byte_idx * 8), b.const_u32(8) });
            val = b.emit_typed_inst(Op::OpIMul, b.type_u32(), { bk, b.const_u32(0x01010101u) });
        }

        if (get_src_abs(ins, src_idx)) {
            uint32_t sval = b.emit_bitcast(b.type_s32(), val);
            uint32_t abs_s = b.emit_glsl_ext(GlslStd450::SAbs, b.type_s32(), { sval });
            val = b.emit_bitcast(b.type_u32(), abs_s);
        }
        if (get_src_neg(ins, src_idx)) {
            uint32_t sval = b.emit_bitcast(b.type_s32(), val);
            uint32_t neg_s = b.emit_typed_inst(Op::OpSNegate, b.type_s32(), { sval });
            val = b.emit_bitcast(b.type_u32(), neg_s);
        }
        if (get_src_not(ins, src_idx)) {
            val = b.emit_typed_inst(Op::OpNot, b.type_u32(), { val });
        }
        return val;
    }

    uint32_t load_src_f32(const DecodedValhallInst& ins, size_t src_idx) {
        uint32_t raw_u32 = load_raw_src_u32(ins, src_idx);
        std::string swz = get_src_swz(ins, src_idx);
        uint32_t fval = 0;
        if (swz == "h0" || swz == "h1") {
            uint32_t v2f = b.type_vector(b.type_f32(), 2);
            uint32_t unpacked = b.emit_glsl_ext(GlslStd450::UnpackHalf2x16, v2f, { raw_u32 });
            fval = b.emit_typed_inst(Op::OpCompositeExtract, b.type_f32(), { unpacked, (swz == "h1") ? 1u : 0u });
        } else {
            fval = b.emit_bitcast(b.type_f32(), raw_u32);
        }
        if (get_src_abs(ins, src_idx)) {
            fval = b.emit_glsl_ext(GlslStd450::FAbs, b.type_f32(), { fval });
        }
        if (get_src_neg(ins, src_idx)) {
            fval = b.emit_typed_inst(Op::OpFNegate, b.type_f32(), { fval });
        }
        return fval;
    }

    uint32_t load_src_u64(const DecodedValhallInst& ins, size_t src_idx, bool sign_extend_subword = false) {
        if (src_idx >= ins.srcs.size()) return b.const_u64(0);
        const auto& s = ins.srcs[src_idx];
        std::string swz = get_src_swz(ins, src_idx);

        auto widen_u32_to_u64 = [&](uint32_t w32, bool sext) -> uint32_t {
            if (sext) {
                uint32_t s32 = b.emit_bitcast(b.type_s32(), w32);
                uint32_t s64 = b.emit_typed_inst(Op::OpSConvert, b.type_s64(), { s32 });
                return b.emit_bitcast(b.type_u64(), s64);
            } else {
                return b.emit_typed_inst(Op::OpUConvert, b.type_u64(), { w32 });
            }
        };

        if (s.kind == SourceKind::Register) {
            uint8_t base = s.reg_num & 62u;
            if (swz == "w0") {
                return widen_u32_to_u64(load_reg_u32(base), sign_extend_subword);
            } else if (swz == "w1") {
                return widen_u32_to_u64(load_reg_u32(base + 1), sign_extend_subword);
            }
            if (!s.is_64bit) {
                return widen_u32_to_u64(load_reg_u32(s.reg_num), sign_extend_subword);
            }
            return pack_u64(load_reg_u32(base), load_reg_u32(base + 1));
        } else if (s.kind == SourceKind::FauUniform) {
            uint8_t base = s.reg_num & 62u;
            if (swz == "w0") {
                return widen_u32_to_u64(load_fau_u32(base), sign_extend_subword);
            } else if (swz == "w1") {
                return widen_u32_to_u64(load_fau_u32(base + 1), sign_extend_subword);
            }
            if (!s.is_64bit) {
                return widen_u32_to_u64(load_fau_u32(s.reg_num), sign_extend_subword);
            }
            return pack_u64(load_fau_u32(base), load_fau_u32(base + 1));
        } else if (s.kind == SourceKind::SmallConstant) {
            uint64_t c64 = uint64_t(s.imm_bits) | (uint64_t(s.imm_bits) << 32);
            return b.const_u64(c64);
        } else if (s.kind == SourceKind::FauSpecial) {
            if (s.symbolic_name == "pc") {
                uint32_t pc_base = load_sideband_u64(1);
                return b.emit_typed_inst(Op::OpIAdd, b.type_u64(), { pc_base, b.const_u64(ins.pc) });
            }
            uint8_t base = s.reg_num & 62u;
            return pack_u64(load_fau_u32(base), load_fau_u32(base + 1));
        }
        return b.const_u64(0);
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

    void write_dest_u32(const DecodedValhallInst& ins, uint32_t val_u32) {
        if (!ins.has_dest) return;
        uint8_t d = ins.dest_reg & 63u;
        std::string dm = get_dest_mod(ins);
        if (dm == "h0") {
            uint32_t old_v = load_reg_u32(d);
            uint32_t res = b.emit_typed_inst(Op::OpBitFieldInsert, b.type_u32(),
                                             { old_v, val_u32, b.const_u32(0), b.const_u32(16) });
            store_reg_u32(d, res);
        } else if (dm == "h1") {
            uint32_t old_v = load_reg_u32(d);
            uint32_t res = b.emit_typed_inst(Op::OpBitFieldInsert, b.type_u32(),
                                             { old_v, val_u32, b.const_u32(16), b.const_u32(16) });
            store_reg_u32(d, res);
        } else if (dm == "b0" || dm == "b1" || dm == "b2" || dm == "b3") {
            uint32_t bit_off = uint32_t(dm[1] - '0') * 8u;
            uint32_t old_v = load_reg_u32(d);
            uint32_t res = b.emit_typed_inst(Op::OpBitFieldInsert, b.type_u32(),
                                             { old_v, val_u32, b.const_u32(bit_off), b.const_u32(8) });
            store_reg_u32(d, res);
        } else {
            store_reg_u32(d, val_u32);
        }
    }

    void write_dest_f32(const DecodedValhallInst& ins, uint32_t val_f32) {
        if (!ins.has_dest) return;
        // Apply float clamp modifier if present
        auto it_clamp = ins.modifier_strs.find("clamp");
        if (it_clamp != ins.modifier_strs.end()) {
            const std::string& cl = it_clamp->second;
            if (cl == "clamp_0_1") {
                val_f32 = b.emit_glsl_ext(GlslStd450::FClamp, b.type_f32(),
                                          { val_f32, b.const_f32(0.0f), b.const_f32(1.0f) });
            } else if (cl == "clamp_m1_1") {
                val_f32 = b.emit_glsl_ext(GlslStd450::FClamp, b.type_f32(),
                                          { val_f32, b.const_f32(-1.0f), b.const_f32(1.0f) });
            } else if (cl == "clamp_0_inf") {
                val_f32 = b.emit_glsl_ext(GlslStd450::FMax, b.type_f32(),
                                          { val_f32, b.const_f32(0.0f) });
            }
        }

        std::string dm = get_dest_mod(ins);
        if (dm == "h0" || dm == "h1") {
            uint32_t v2f = b.type_vector(b.type_f32(), 2);
            uint32_t vec = b.emit_typed_inst(Op::OpCompositeConstruct, v2f, { val_f32, val_f32 });
            uint32_t packed = b.emit_glsl_ext(GlslStd450::PackHalf2x16, b.type_u32(), { vec });
            uint8_t d = ins.dest_reg & 63u;
            uint32_t old_v = load_reg_u32(d);
            uint32_t res = b.emit_typed_inst(Op::OpBitFieldInsert, b.type_u32(),
                                             { old_v, packed, b.const_u32((dm == "h1") ? 16u : 0u), b.const_u32(16) });
            store_reg_u32(d, res);
            return;
        }

        uint32_t bits = b.emit_bitcast(b.type_u32(), val_f32);
        store_reg_u32(ins.dest_reg & 63u, bits);
    }

    void write_dest_u64(const DecodedValhallInst& ins, uint32_t val_u64) {
        if (!ins.has_dest) return;
        uint32_t lo = 0, hi = 0;
        unpack_u64(val_u64, &lo, &hi);
        uint8_t d = ins.dest_reg & 62u;
        store_reg_u32(d, lo);
        store_reg_u32(d + 1, hi);
    }

    uint32_t eval_float_cmp(const std::string& cmpf, uint32_t a_f32, uint32_t b_f32) {
        uint32_t bt = b.type_bool();
        if (cmpf == "eq") return b.emit_typed_inst(Op::OpFOrdEqual, bt, { a_f32, b_f32 });
        if (cmpf == "gt") return b.emit_typed_inst(Op::OpFOrdGreaterThan, bt, { a_f32, b_f32 });
        if (cmpf == "ge") return b.emit_typed_inst(Op::OpFOrdGreaterThanEqual, bt, { a_f32, b_f32 });
        if (cmpf == "ne") return b.emit_typed_inst(Op::OpFUnordNotEqual, bt, { a_f32, b_f32 });
        if (cmpf == "lt") return b.emit_typed_inst(Op::OpFOrdLessThan, bt, { a_f32, b_f32 });
        if (cmpf == "le") return b.emit_typed_inst(Op::OpFOrdLessThanEqual, bt, { a_f32, b_f32 });
        if (cmpf == "gtlt") return b.emit_typed_inst(Op::OpFOrdNotEqual, bt, { a_f32, b_f32 });
        if (cmpf == "total") {
            uint32_t nan_a = b.emit_typed_inst(Op::OpIsNan, bt, { a_f32 });
            uint32_t nan_b = b.emit_typed_inst(Op::OpIsNan, bt, { b_f32 });
            uint32_t any_nan = b.emit_typed_inst(Op::OpLogicalOr, bt, { nan_a, nan_b });
            return b.emit_typed_inst(Op::OpLogicalNot, bt, { any_nan });
        }
        return b.emit_typed_inst(Op::OpFOrdEqual, bt, { a_f32, b_f32 });
    }

    uint32_t eval_int_cmp(const std::string& cmpf, uint32_t a_u32, uint32_t b_u32, bool is_signed) {
        uint32_t bt = b.type_bool();
        if (cmpf == "eq") return b.emit_typed_inst(Op::OpIEqual, bt, { a_u32, b_u32 });
        if (cmpf == "ne") return b.emit_typed_inst(Op::OpINotEqual, bt, { a_u32, b_u32 });
        if (is_signed) {
            uint32_t sa = b.emit_bitcast(b.type_s32(), a_u32);
            uint32_t sb = b.emit_bitcast(b.type_s32(), b_u32);
            if (cmpf == "gt") return b.emit_typed_inst(Op::OpSGreaterThan, bt, { sa, sb });
            if (cmpf == "ge") return b.emit_typed_inst(Op::OpSGreaterThanEqual, bt, { sa, sb });
            if (cmpf == "lt") return b.emit_typed_inst(Op::OpSLessThan, bt, { sa, sb });
            if (cmpf == "le") return b.emit_typed_inst(Op::OpSLessThanEqual, bt, { sa, sb });
        } else {
            if (cmpf == "gt") return b.emit_typed_inst(Op::OpUGreaterThan, bt, { a_u32, b_u32 });
            if (cmpf == "ge") return b.emit_typed_inst(Op::OpUGreaterThanEqual, bt, { a_u32, b_u32 });
            if (cmpf == "lt") return b.emit_typed_inst(Op::OpULessThan, bt, { a_u32, b_u32 });
            if (cmpf == "le") return b.emit_typed_inst(Op::OpULessThanEqual, bt, { a_u32, b_u32 });
        }
        return b.emit_typed_inst(Op::OpIEqual, bt, { a_u32, b_u32 });
    }

    uint32_t format_cmp_result(const DecodedValhallInst& ins, uint32_t cond_bool) {
        std::string rtype = "m1";
        auto it = ins.modifier_strs.find("result_type");
        if (it != ins.modifier_strs.end()) rtype = it->second;
        uint32_t true_val = b.const_u32(0xFFFFFFFFu);
        if (rtype == "i1") true_val = b.const_u32(1u);
        else if (rtype == "f1") true_val = b.const_u32(0x3F800000u);
        return b.emit_select(b.type_u32(), cond_bool, true_val, b.const_u32(0u));
    }

    uint32_t get_or_create_vert_attr(uint32_t loc) {
        auto it = vert_attr_inputs_.find(loc);
        if (it != vert_attr_inputs_.end()) return it->second;
        uint32_t v4f = b.type_vector(b.type_f32(), 4);
        uint32_t ptr_t = b.type_pointer(StorageClass::Input, v4f);
        uint32_t var_id = b.add_global_var(ptr_t, StorageClass::Input, "in_attr_" + std::to_string(loc), true);
        b.decorate(var_id, Decoration::Location, { loc });
        vert_attr_inputs_[loc] = var_id;
        return var_id;
    }

    uint32_t get_or_create_vert_varying(uint32_t loc) {
        auto it = vert_varying_outputs_.find(loc);
        if (it != vert_varying_outputs_.end()) return it->second;
        uint32_t v4f = b.type_vector(b.type_f32(), 4);
        uint32_t ptr_t = b.type_pointer(StorageClass::Output, v4f);
        uint32_t var_id = b.add_global_var(ptr_t, StorageClass::Output, "out_var_" + std::to_string(loc), true);
        b.decorate(var_id, Decoration::Location, { loc });
        vert_varying_outputs_[loc] = var_id;
        return var_id;
    }

    uint32_t get_or_create_frag_varying(uint32_t loc, bool flat = false) {
        auto it = frag_varying_inputs_.find(loc);
        if (it != frag_varying_inputs_.end()) return it->second;
        uint32_t v4f = b.type_vector(b.type_f32(), 4);
        uint32_t ptr_t = b.type_pointer(StorageClass::Input, v4f);
        uint32_t var_id = b.add_global_var(ptr_t, StorageClass::Input, "in_var_" + std::to_string(loc), true);
        b.decorate(var_id, Decoration::Location, { loc });
        if (flat) b.decorate(var_id, Decoration::Flat);
        frag_varying_inputs_[loc] = var_id;
        return var_id;
    }

    uint32_t get_or_create_frag_color_output(uint32_t loc) {
        auto it = frag_color_outputs_.find(loc);
        if (it != frag_color_outputs_.end()) return it->second;
        uint32_t v4f = b.type_vector(b.type_f32(), 4);
        uint32_t ptr_t = b.type_pointer(StorageClass::Output, v4f);
        uint32_t var_id = b.add_global_var(ptr_t, StorageClass::Output, "out_color_" + std::to_string(loc), true);
        b.decorate(var_id, Decoration::Location, { loc });
        frag_color_outputs_[loc] = var_id;
        return var_id;
    }

    uint32_t get_or_create_sampled_image(uint32_t idx, Dim dim = Dim::Dim2D, bool arrayed = false, bool shadow = false, uint32_t* out_si_type = nullptr) {
        auto it = sampled_image_vars_.find(idx);
        if (it != sampled_image_vars_.end()) {
            if (out_si_type) *out_si_type = sampled_image_types_[idx];
            return it->second;
        }
        uint32_t d_set = 0;
        uint32_t binding = 1 + idx;
        for (const auto& s : opts.samplers) {
            if (s.tex_index == idx) {
                d_set = s.descriptor_set;
                binding = s.binding;
                dim = s.dim;
                arrayed = s.arrayed;
                shadow = s.shadow;
                break;
            }
        }
        uint32_t img_t = b.type_image(b.type_f32(), dim, shadow ? 1u : 0u, arrayed ? 1u : 0u, 0u, 1u, 0u);
        uint32_t si_t = b.type_sampled_image(img_t);
        uint32_t ptr_t = b.type_pointer(StorageClass::UniformConstant, si_t);
        uint32_t var_id = b.add_global_var(ptr_t, StorageClass::UniformConstant, "u_sampler_" + std::to_string(idx), false);
        b.decorate(var_id, Decoration::DescriptorSet, { d_set });
        b.decorate(var_id, Decoration::Binding, { binding });
        sampled_image_vars_[idx] = var_id;
        sampled_image_types_[idx] = si_t;
        if (out_si_type) *out_si_type = si_t;
        return var_id;
    }

    void emit_instruction(const DecodedValhallInst& ins) {
        Mnemonic mn = ins.mnemonic;
        Mnemonic cmn = ins.canonical_mnemonic;
        std::string var = variant_to_string(ins.canonical_variant);

        // 0. NOP / WAIT / BARRIER / DISCARD
        if (cmn == Mnemonic::NOP || mn == Mnemonic::NOP) {
            return;
        }
        if (cmn == Mnemonic::BARRIER || mn == Mnemonic::BARRIER) {
            // Workgroup execution + memory barrier
            b.emit_inst(Op::OpControlBarrier,
                        { b.const_u32(2u), b.const_u32(2u), b.const_u32(0x108u) });
            return;
        }
        if (cmn == Mnemonic::DISCARD || mn == Mnemonic::DISCARD) {
            b.emit_kill();
            return;
        }

        // 1. ADR (PC-relative address calculation: rd = (pc + offset) & 0xFFFFFFFF)
        if (mn == Mnemonic::ADR) {
            uint32_t pc_base = load_sideband_u64(1);
            int64_t tgt_pc = ins.has_branch_offset ? ins.branch_target_pc : (int64_t(ins.pc) + ins.imm_signed);
            uint32_t full_addr = b.emit_typed_inst(Op::OpIAdd, b.type_u64(),
                                                   { pc_base, b.const_u64(static_cast<uint64_t>(tgt_pc)) });
            uint32_t lo = 0, hi = 0;
            unpack_u64(full_addr, &lo, &hi);
            write_dest_u32(ins, lo);
            return;
        }

        // 2. MOV / MOV_IMM /NOP-like moves
        if (mn == Mnemonic::MOV || cmn == Mnemonic::MOV) {
            uint32_t s0 = load_src_int32(ins, 0, false);
            write_dest_u32(ins, s0);
            return;
        }
        if (mn == Mnemonic::MOV_IMM) {
            uint32_t imm = b.const_u32(static_cast<uint32_t>(ins.imm_signed));
            write_dest_u32(ins, imm);
            return;
        }

        // 3. Floating-Point Arithmetic (FADD, FADD_IMM, FMUL, FMA, FMIN, FMAX, FABS, FNEG, FSAT, FROUND, FFRACT)
        if (mn == Mnemonic::FABS) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t res = b.emit_glsl_ext(GlslStd450::FAbs, b.type_f32(), { s0 });
            write_dest_f32(ins, res);
            return;
        }
        if (mn == Mnemonic::FNEG) {
            uint32_t s0 = load_src_f32(ins, 0);
            write_dest_f32(ins, s0);
            return;
        }
        if (false) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t res = b.emit_glsl_ext(GlslStd450::FClamp, b.type_f32(),
                                           { s0, b.const_f32(0.0f), b.const_f32(1.0f) });
            write_dest_f32(ins, res);
            return;
        }
        if (mn == Mnemonic::FMUL) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t s1 = load_src_f32(ins, 1);
            uint32_t res = b.emit_typed_inst(Op::OpFMul, b.type_f32(), { s0, s1 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FMA || mn == Mnemonic::FMA) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t s1 = load_src_f32(ins, 1);
            uint32_t s2 = (ins.srcs.size() >= 3) ? load_src_f32(ins, 2) : b.const_f32(0.0f);
            uint32_t res = b.emit_glsl_ext(GlslStd450::Fma, b.type_f32(), { s0, s1, s2 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FADD || mn == Mnemonic::FADD) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t s1 = load_src_f32(ins, 1);
            uint32_t res = b.emit_typed_inst(Op::OpFAdd, b.type_f32(), { s0, s1 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FADD_IMM || mn == Mnemonic::FADD_IMM) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t s1 = b.const_f32_bits(static_cast<uint32_t>(ins.imm_unsigned));
            uint32_t res = b.emit_typed_inst(Op::OpFAdd, b.type_f32(), { s0, s1 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FMIN || mn == Mnemonic::FMIN) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t s1 = load_src_f32(ins, 1);
            uint32_t res = b.emit_glsl_ext(GlslStd450::FMin, b.type_f32(), { s0, s1 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FMAX || mn == Mnemonic::FMAX) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t s1 = load_src_f32(ins, 1);
            uint32_t res = b.emit_glsl_ext(GlslStd450::FMax, b.type_f32(), { s0, s1 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FROUND || mn == Mnemonic::FROUND) {
            uint32_t s0 = load_src_f32(ins, 0);
            std::string rmode = "rte";
            auto it = ins.modifier_strs.find("round");
            if (it != ins.modifier_strs.end()) rmode = it->second;
            GlslStd450 op = GlslStd450::RoundEven;
            if (rmode == "rtn" || rmode == "round_neg_inf") op = GlslStd450::Floor;
            else if (rmode == "rtp" || rmode == "round_pos_inf") op = GlslStd450::Ceil;
            else if (rmode == "rtz" || rmode == "round_zero") op = GlslStd450::Trunc;
            uint32_t res = b.emit_glsl_ext(op, b.type_f32(), { s0 });
            write_dest_f32(ins, res);
            return;
        }

        // 4. Special Function Unit (FRCP, FRSQ, FSQRT, FSIN_TABLE, FCOS_TABLE, FEXP, FLOG, FREXP, FLOGD)
        if (cmn == Mnemonic::FRCP || mn == Mnemonic::FRCP || cmn == Mnemonic::FRCP_APPROX) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t res = b.emit_typed_inst(Op::OpFDiv, b.type_f32(), { b.const_f32(1.0f), s0 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FRSQ || mn == Mnemonic::FRSQ || cmn == Mnemonic::FRSQ_APPROX) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t res = b.emit_glsl_ext(GlslStd450::InverseSqrt, b.type_f32(), { s0 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FSQRT || mn == Mnemonic::FSQRT) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t res = b.emit_glsl_ext(GlslStd450::Sqrt, b.type_f32(), { s0 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FSIN_TABLE || mn == Mnemonic::FSIN_TABLE) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t arg = b.emit_typed_inst(Op::OpFMul, b.type_f32(), { s0, b.const_f32(6.283185307179586f) });
            uint32_t res = b.emit_glsl_ext(GlslStd450::Sin, b.type_f32(), { arg });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FCOS_TABLE || mn == Mnemonic::FCOS_TABLE) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t arg = b.emit_typed_inst(Op::OpFMul, b.type_f32(), { s0, b.const_f32(6.283185307179586f) });
            uint32_t res = b.emit_glsl_ext(GlslStd450::Cos, b.type_f32(), { arg });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FEXP_TABLE || mn == Mnemonic::FEXP_TABLE || cmn == Mnemonic::FEXP) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t res = b.emit_glsl_ext(GlslStd450::Exp2, b.type_f32(), { s0 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::FLOG_TABLE || mn == Mnemonic::FLOG_TABLE || cmn == Mnemonic::FLOGD) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t res = b.emit_glsl_ext(GlslStd450::Log2, b.type_f32(), { s0 });
            write_dest_f32(ins, res);
            return;
        }

        // 5. Type Conversions
        if (cmn == Mnemonic::U32_TO_F32 || mn == Mnemonic::U32_TO_F32) {
            uint32_t s0 = load_src_int32(ins, 0, false);
            uint32_t res = b.emit_typed_inst(Op::OpConvertUToF, b.type_f32(), { s0 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::S32_TO_F32 || mn == Mnemonic::S32_TO_F32) {
            uint32_t s0 = load_src_int32(ins, 0, true);
            uint32_t s32 = b.emit_bitcast(b.type_s32(), s0);
            uint32_t res = b.emit_typed_inst(Op::OpConvertSToF, b.type_f32(), { s32 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::F32_TO_U32 || mn == Mnemonic::F32_TO_U32) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t res = b.emit_typed_inst(Op::OpConvertFToU, b.type_u32(), { s0 });
            write_dest_u32(ins, res);
            return;
        }
        if (cmn == Mnemonic::F32_TO_S32 || mn == Mnemonic::F32_TO_S32) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t s32 = b.emit_typed_inst(Op::OpConvertFToS, b.type_s32(), { s0 });
            uint32_t res = b.emit_bitcast(b.type_u32(), s32);
            write_dest_u32(ins, res);
            return;
        }
        if (cmn == Mnemonic::F16_TO_F32 || mn == Mnemonic::F16_TO_F32) {
            uint32_t s0 = load_src_f32(ins, 0);
            write_dest_f32(ins, s0);
            return;
        }
        if (cmn == Mnemonic::F32_TO_F16 || mn == Mnemonic::F32_TO_F16) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t s1 = (ins.srcs.size() >= 2) ? load_src_f32(ins, 1) : s0;
            uint32_t v2f = b.type_vector(b.type_f32(), 2);
            uint32_t vec = b.emit_typed_inst(Op::OpCompositeConstruct, v2f, { s0, s1 });
            uint32_t packed = b.emit_glsl_ext(GlslStd450::PackHalf2x16, b.type_u32(), { vec });
            write_dest_u32(ins, packed);
            return;
        }
        if (cmn == Mnemonic::U16_TO_F32 || mn == Mnemonic::U16_TO_F32 ||
            cmn == Mnemonic::U8_TO_F32  || mn == Mnemonic::U8_TO_F32) {
            uint32_t s0 = load_src_int32(ins, 0, false);
            uint32_t res = b.emit_typed_inst(Op::OpConvertUToF, b.type_f32(), { s0 });
            write_dest_f32(ins, res);
            return;
        }
        if (cmn == Mnemonic::S16_TO_F32 || mn == Mnemonic::S16_TO_F32 ||
            cmn == Mnemonic::S8_TO_F32  || mn == Mnemonic::S8_TO_F32) {
            uint32_t s0 = load_src_int32(ins, 0, true);
            uint32_t s32 = b.emit_bitcast(b.type_s32(), s0);
            uint32_t res = b.emit_typed_inst(Op::OpConvertSToF, b.type_f32(), { s32 });
            write_dest_f32(ins, res);
            return;
        }

        // 6. Integer Arithmetic (IADD, IADD_IMM, ISUB, IMUL, IMAD, SHADD, IABS, IMIN, IMAX)
        if (cmn == Mnemonic::IADD_IMM || mn == Mnemonic::IADD_IMM) {
            if (ins.dest_is_64bit) {
                uint32_t s0 = load_src_u64(ins, 0);
                uint32_t imm = b.const_u64(static_cast<uint64_t>(ins.imm_signed));
                uint32_t res = b.emit_typed_inst(Op::OpIAdd, b.type_u64(), { s0, imm });
                write_dest_u64(ins, res);
            } else {
                uint32_t s0 = load_src_int32(ins, 0, false);
                uint32_t imm = b.const_u32(static_cast<uint32_t>(ins.imm_signed));
                uint32_t res = b.emit_typed_inst(Op::OpIAdd, b.type_u32(), { s0, imm });
                write_dest_u32(ins, res);
            }
            return;
        }
        if (cmn == Mnemonic::IADD || mn == Mnemonic::IADD) {
            if (ins.dest_is_64bit) {
                bool is_s = (var.find("s64") != std::string::npos);
                uint32_t s0 = load_src_u64(ins, 0, is_s);
                uint32_t s1 = load_src_u64(ins, 1, is_s);
                uint32_t res = b.emit_typed_inst(Op::OpIAdd, b.type_u64(), { s0, s1 });
                write_dest_u64(ins, res);
            } else {
                bool is_s = (var.find("s") == 0);
                uint32_t s0 = load_src_int32(ins, 0, is_s);
                uint32_t s1 = load_src_int32(ins, 1, is_s);
                uint32_t res = b.emit_typed_inst(Op::OpIAdd, b.type_u32(), { s0, s1 });
                write_dest_u32(ins, res);
            }
            return;
        }
        if (cmn == Mnemonic::ISUB || mn == Mnemonic::ISUB) {
            if (ins.dest_is_64bit) {
                bool is_s = (var.find("s64") != std::string::npos);
                uint32_t s0 = load_src_u64(ins, 0, is_s);
                uint32_t s1 = load_src_u64(ins, 1, is_s);
                uint32_t res = b.emit_typed_inst(Op::OpISub, b.type_u64(), { s0, s1 });
                write_dest_u64(ins, res);
            } else {
                bool is_s = (var.find("s") == 0);
                uint32_t s0 = load_src_int32(ins, 0, is_s);
                uint32_t s1 = load_src_int32(ins, 1, is_s);
                uint32_t res = b.emit_typed_inst(Op::OpISub, b.type_u32(), { s0, s1 });
                write_dest_u32(ins, res);
            }
            return;
        }
        if (cmn == Mnemonic::IMUL || mn == Mnemonic::IMUL) {
            bool is_s = (var.find("s") == 0);
            uint32_t s0 = load_src_int32(ins, 0, is_s);
            uint32_t s1 = load_src_int32(ins, 1, is_s);
            uint32_t res = b.emit_typed_inst(Op::OpIMul, b.type_u32(), { s0, s1 });
            write_dest_u32(ins, res);
            return;
        }
        if (cmn == Mnemonic::SHADD || mn == Mnemonic::SHADD) {
            bool is_s = (var.find("s64") != std::string::npos);
            uint32_t base64 = load_src_u64(ins, 0, false);
            uint32_t idx64  = load_src_u64(ins, 1, is_s);
            uint32_t sh_amt = b.const_u64(static_cast<uint64_t>(ins.imm_unsigned & 63u));
            uint32_t shifted = b.emit_typed_inst(Op::OpShiftLeftLogical, b.type_u64(), { idx64, sh_amt });
            uint32_t res64 = b.emit_typed_inst(Op::OpIAdd, b.type_u64(), { base64, shifted });
            write_dest_u64(ins, res64);
            return;
        }
        if (cmn == Mnemonic::IABS || mn == Mnemonic::IABS) {
            uint32_t s0 = load_src_int32(ins, 0, true);
            uint32_t s32 = b.emit_bitcast(b.type_s32(), s0);
            uint32_t res_s = b.emit_glsl_ext(GlslStd450::SAbs, b.type_s32(), { s32 });
            write_dest_u32(ins, b.emit_bitcast(b.type_u32(), res_s));
            return;
        }
        if (cmn == Mnemonic::HADD || mn == Mnemonic::HADD) {
            bool is_s = (var.find("s") == 0);
            uint32_t s0 = load_src_int32(ins, 0, is_s);
            uint32_t s1 = load_src_int32(ins, 1, is_s);
            uint32_t sum = b.emit_typed_inst(Op::OpIAdd, b.type_u32(), { s0, s1 });
            uint32_t res = b.emit_typed_inst(Op::OpShiftRightLogical, b.type_u32(), { sum, b.const_u32(1u) });
            write_dest_u32(ins, res);
            return;
        }
        if (cmn == Mnemonic::U16_TO_U32 || mn == Mnemonic::U16_TO_U32 ||
            cmn == Mnemonic::U8_TO_U32  || mn == Mnemonic::U8_TO_U32) {
            write_dest_u32(ins, load_src_int32(ins, 0, false));
            return;
        }
        if (cmn == Mnemonic::S16_TO_S32 || mn == Mnemonic::S16_TO_S32 ||
            cmn == Mnemonic::S8_TO_S32  || mn == Mnemonic::S8_TO_S32) {
            write_dest_u32(ins, load_src_int32(ins, 0, true));
            return;
        }
        if (cmn == Mnemonic::V2F32_TO_V2F16 || mn == Mnemonic::V2F32_TO_V2F16) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t s1 = (ins.srcs.size() >= 2) ? load_src_f32(ins, 1) : s0;
            uint32_t v2f = b.type_vector(b.type_f32(), 2);
            uint32_t vec = b.emit_typed_inst(Op::OpCompositeConstruct, v2f, { s0, s1 });
            uint32_t packed = b.emit_glsl_ext(GlslStd450::PackHalf2x16, b.type_u32(), { vec });
            write_dest_u32(ins, packed);
            return;
        }
        if (cmn == Mnemonic::MKVEC || mn == Mnemonic::MKVEC || cmn == Mnemonic::SWZ || mn == Mnemonic::SWZ) {
            uint32_t s0 = load_src_int32(ins, 0, false);
            if (ins.srcs.size() >= 2) {
                uint32_t s1 = load_src_int32(ins, 1, false);
                s0 = b.emit_typed_inst(Op::OpBitFieldInsert, b.type_u32(), { s0, s1, b.const_u32(16u), b.const_u32(16u) });
            }
            write_dest_u32(ins, s0);
            return;
        }

        // 7. Bitwise & Shift Instructions
        if (mn == Mnemonic::AND_IMM || mn == Mnemonic::OR_IMM || mn == Mnemonic::XOR_IMM) {
            uint32_t s0 = load_src_int32(ins, 0, false);
            uint32_t imm = b.const_u32(static_cast<uint32_t>(ins.imm_unsigned));
            Op op = Op::OpBitwiseAnd;
            if (mn == Mnemonic::OR_IMM) op = Op::OpBitwiseOr;
            else if (mn == Mnemonic::XOR_IMM) op = Op::OpBitwiseXor;
            uint32_t res = b.emit_typed_inst(op, b.type_u32(), { s0, imm });
            write_dest_u32(ins, res);
            return;
        }
        if (ins.category == SemanticCategory::BITWISE_SHIFT ||
            mn == Mnemonic::AND || mn == Mnemonic::OR || mn == Mnemonic::XOR ||
            mn == Mnemonic::LSHIFT || mn == Mnemonic::RSHIFT || mn == Mnemonic::ARSHIFT ||
            cmn == Mnemonic::LSHIFT_AND || cmn == Mnemonic::LSHIFT_OR || cmn == Mnemonic::LSHIFT_XOR ||
            cmn == Mnemonic::RSHIFT_AND || cmn == Mnemonic::RSHIFT_OR || cmn == Mnemonic::RSHIFT_XOR ||
            cmn == Mnemonic::ARSHIFT_AND || cmn == Mnemonic::ARSHIFT_OR || cmn == Mnemonic::ARSHIFT_XOR) {
            if (cmn == Mnemonic::CLZ || mn == Mnemonic::CLZ) {
                uint32_t s0 = load_src_int32(ins, 0, false);
                uint32_t msb = b.emit_glsl_ext(GlslStd450::FindUMsb, b.type_s32(), { s0 });
                uint32_t msb_u = b.emit_bitcast(b.type_u32(), msb);
                uint32_t is_zero = b.emit_typed_inst(Op::OpIEqual, b.type_bool(), { s0, b.const_u32(0) });
                uint32_t sub31 = b.emit_typed_inst(Op::OpISub, b.type_u32(), { b.const_u32(31), msb_u });
                uint32_t res = b.emit_select(b.type_u32(), is_zero, b.const_u32(32), sub31);
                write_dest_u32(ins, res);
                return;
            }
            if (cmn == Mnemonic::POPCOUNT || mn == Mnemonic::POPCOUNT) {
                uint32_t s0 = load_src_int32(ins, 0, false);
                uint32_t res = b.emit_typed_inst(Op::OpBitCount, b.type_u32(), { s0 });
                write_dest_u32(ins, res);
                return;
            }
            if (cmn == Mnemonic::BITREV || mn == Mnemonic::BITREV) {
                uint32_t s0 = load_src_int32(ins, 0, false);
                uint32_t res = b.emit_typed_inst(Op::OpBitReverse, b.type_u32(), { s0 });
                write_dest_u32(ins, res);
                return;
            }

            // General Valhall SHIFT_BITWISE family:
            // On 2-source aliases (AND, OR, XOR, LSHIFT, RSHIFT, ARSHIFT) vs 3-source (LSHIFT_OR, etc.):
            uint32_t s0 = load_src_int32(ins, 0, false);
            if (mn == Mnemonic::AND || mn == Mnemonic::OR || mn == Mnemonic::XOR) {
                uint32_t s1 = load_src_int32(ins, 1, false);
                Op bop = Op::OpBitwiseOr;
                if (mn == Mnemonic::AND) bop = Op::OpBitwiseAnd;
                else if (mn == Mnemonic::XOR) bop = Op::OpBitwiseXor;
                uint32_t res = b.emit_typed_inst(bop, b.type_u32(), { s0, s1 });
                auto it_nr = ins.fields.find("not_result");
                if (it_nr != ins.fields.end() && it_nr->second != 0) {
                    res = b.emit_typed_inst(Op::OpNot, b.type_u32(), { res });
                }
                write_dest_u32(ins, res);
                return;
            }

            uint32_t sh_raw = load_src_int32(ins, 1, false);
            uint32_t sh_amt = b.emit_typed_inst(Op::OpBitwiseAnd, b.type_u32(), { sh_raw, b.const_u32(31u) });
            uint32_t shifted = s0;
            if (cmn == Mnemonic::ARSHIFT_AND || cmn == Mnemonic::ARSHIFT_OR || cmn == Mnemonic::ARSHIFT_XOR || mn == Mnemonic::ARSHIFT) {
                uint32_t s0_s = b.emit_bitcast(b.type_s32(), s0);
                uint32_t sh_s = b.emit_typed_inst(Op::OpShiftRightArithmetic, b.type_s32(), { s0_s, sh_amt });
                shifted = b.emit_bitcast(b.type_u32(), sh_s);
            } else if (cmn == Mnemonic::RSHIFT_AND || cmn == Mnemonic::RSHIFT_OR || cmn == Mnemonic::RSHIFT_XOR || mn == Mnemonic::RSHIFT) {
                shifted = b.emit_typed_inst(Op::OpShiftRightLogical, b.type_u32(), { s0, sh_amt });
            } else {
                shifted = b.emit_typed_inst(Op::OpShiftLeftLogical, b.type_u32(), { s0, sh_amt });
            }

            uint32_t res = shifted;
            if (ins.srcs.size() >= 3) {
                uint32_t s2 = load_src_int32(ins, 2, false);
                Op bop = Op::OpBitwiseOr;
                if (cmn == Mnemonic::LSHIFT_AND || cmn == Mnemonic::RSHIFT_AND || cmn == Mnemonic::ARSHIFT_AND) bop = Op::OpBitwiseAnd;
                else if (cmn == Mnemonic::LSHIFT_XOR || cmn == Mnemonic::RSHIFT_XOR || cmn == Mnemonic::ARSHIFT_XOR) bop = Op::OpBitwiseXor;
                res = b.emit_typed_inst(bop, b.type_u32(), { shifted, s2 });
            }
            auto it_nr = ins.fields.find("not_result");
            if (it_nr != ins.fields.end() && it_nr->second != 0) {
                res = b.emit_typed_inst(Op::OpNot, b.type_u32(), { res });
            }
            write_dest_u32(ins, res);
            return;
        }

        // 8. Comparisons & Conditional Selects (FCMP, FCMP_OR, FCMP_AND, ICMP, ICMP_OR, ICMP_AND, CSEL, MUX)
        if (cmn == Mnemonic::FCMP || cmn == Mnemonic::FCMP_OR || cmn == Mnemonic::FCMP_AND ||
            mn == Mnemonic::FCMP || mn == Mnemonic::FCMP_OR || mn == Mnemonic::FCMP_AND) {
            uint32_t s0 = load_src_f32(ins, 0);
            uint32_t s1 = load_src_f32(ins, 1);
            std::string cmpf = "eq";
            auto it = ins.modifier_strs.find("cmpf");
            if (it != ins.modifier_strs.end()) cmpf = it->second;
            uint32_t cond = eval_float_cmp(cmpf, s0, s1);
            uint32_t mask = format_cmp_result(ins, cond);
            if (ins.srcs.size() >= 3 && mn != Mnemonic::FCMP) {
                uint32_t s2 = load_src_int32(ins, 2, false);
                Op comb = (cmn == Mnemonic::FCMP_AND || mn == Mnemonic::FCMP_AND) ? Op::OpBitwiseAnd : Op::OpBitwiseOr;
                mask = b.emit_typed_inst(comb, b.type_u32(), { mask, s2 });
            }
            write_dest_u32(ins, mask);
            return;
        }
        if (cmn == Mnemonic::ICMP || cmn == Mnemonic::ICMP_OR || cmn == Mnemonic::ICMP_AND ||
            mn == Mnemonic::ICMP || mn == Mnemonic::ICMP_OR || mn == Mnemonic::ICMP_AND) {
            bool is_s = (var.find("s") == 0);
            uint32_t s0 = load_src_int32(ins, 0, is_s);
            uint32_t s1 = load_src_int32(ins, 1, is_s);
            std::string cmpf = "eq";
            auto it = ins.modifier_strs.find("cmpf");
            if (it != ins.modifier_strs.end()) cmpf = it->second;
            uint32_t cond = eval_int_cmp(cmpf, s0, s1, is_s);
            uint32_t mask = format_cmp_result(ins, cond);
            if (ins.srcs.size() >= 3 && mn != Mnemonic::ICMP) {
                uint32_t s2 = load_src_int32(ins, 2, false);
                Op comb = (cmn == Mnemonic::ICMP_AND || mn == Mnemonic::ICMP_AND) ? Op::OpBitwiseAnd : Op::OpBitwiseOr;
                mask = b.emit_typed_inst(comb, b.type_u32(), { mask, s2 });
            }
            write_dest_u32(ins, mask);
            return;
        }
        if (cmn == Mnemonic::CSEL || mn == Mnemonic::CSEL) {
            std::string cmpf = "ne";
            auto it = ins.modifier_strs.find("cmpf");
            if (it != ins.modifier_strs.end()) cmpf = it->second;
            bool is_float = (var.find("f") == 0);
            bool is_signed = (var.find("s") == 0);
            uint32_t cond = 0;
            if (is_float) {
                uint32_t a = load_src_f32(ins, 0);
                uint32_t c = load_src_f32(ins, 1);
                cond = eval_float_cmp(cmpf, a, c);
            } else {
                uint32_t a = load_src_int32(ins, 0, is_signed);
                uint32_t c = load_src_int32(ins, 1, is_signed);
                cond = eval_int_cmp(cmpf, a, c, is_signed);
            }
            uint32_t t_val = load_src_int32(ins, 2, false);
            uint32_t f_val = load_src_int32(ins, 3, false);
            uint32_t res = b.emit_select(b.type_u32(), cond, t_val, f_val);
            write_dest_u32(ins, res);
            return;
        }
        if (cmn == Mnemonic::MUX || mn == Mnemonic::MUX) {
            uint32_t a = load_src_int32(ins, 0, false);
            uint32_t c = load_src_int32(ins, 1, false);
            uint32_t mask = load_src_int32(ins, 2, false);
            uint32_t not_m = b.emit_typed_inst(Op::OpNot, b.type_u32(), { mask });
            uint32_t p0 = b.emit_typed_inst(Op::OpBitwiseAnd, b.type_u32(), { a, mask });
            uint32_t p1 = b.emit_typed_inst(Op::OpBitwiseAnd, b.type_u32(), { c, not_m });
            uint32_t res = b.emit_typed_inst(Op::OpBitwiseOr, b.type_u32(), { p0, p1 });
            write_dest_u32(ins, res);
            return;
        }

        // 9. Subgroup / Quad Derivative Ops (CLPER, CLPER_OR,WMASK)
        if (cmn == Mnemonic::CLPER || cmn == Mnemonic::CLPER_OR || mn == Mnemonic::CLPER) {
            uint32_t s0 = load_src_int32(ins, 0, false);
            uint32_t s1 = load_src_int32(ins, 1, false);
            uint32_t lane = b.emit_typed_inst(Op::OpBitwiseAnd, b.type_u32(), { s1, b.const_u32(31u) });
            uint32_t res = b.emit_typed_inst(Op::OpGroupNonUniformShuffle, b.type_u32(),
                                             { b.const_u32(3u), s0, lane });
            write_dest_u32(ins, res);
            return;
        }

        // 10. Vertex & Varying Instructions (LD_ATTR_IMM, LEA_BUF_IMM, LD_VAR_IMM, LD_VAR, LD_VAR_SPECIAL)
        if (cmn == Mnemonic::LD_ATTR_IMM || mn == Mnemonic::LD_ATTR_IMM ||
            cmn == Mnemonic::LD_ATTR || mn == Mnemonic::LD_ATTR) {
            uint32_t attr_idx = 0;
            auto it = ins.fields.find("attribute_index");
            if (it != ins.fields.end()) attr_idx = it->second;
            uint32_t var_id = get_or_create_vert_attr(attr_idx);
            uint32_t v4f = b.type_vector(b.type_f32(), 4);
            uint32_t vec = b.emit_load(v4f, var_id);
            uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
            uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
            for (uint8_t c = 0; c < cnt && c < 4; ++c) {
                uint32_t comp_f = b.emit_typed_inst(Op::OpCompositeExtract, b.type_f32(), { vec, uint32_t(c) });
                store_reg_u32(base_r + c, b.emit_bitcast(b.type_u32(), comp_f));
            }
            return;
        }
        if (cmn == Mnemonic::LEA_BUF_IMM || mn == Mnemonic::LEA_BUF_IMM) {
            // Write synthetic 64-bit buffer address for estream/istream or SRT buffer
            uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
            uint32_t buf_idx = 0;
            auto it_bi = ins.fields.find("buffer_index");
            if (it_bi != ins.fields.end()) buf_idx = it_bi->second;
            store_reg_u32(base_r, buf_idx * 16u);
            store_reg_u32(base_r + 1, 0xEEEEEEEEu);
            return;
        }
        if (cmn == Mnemonic::LD_VAR_IMM || mn == Mnemonic::LD_VAR_IMM ||
            cmn == Mnemonic::LD_VAR_FLAT_IMM || mn == Mnemonic::LD_VAR_FLAT_IMM ||
            cmn == Mnemonic::LD_VAR_BUF_IMM || mn == Mnemonic::LD_VAR_BUF_IMM ||
            cmn == Mnemonic::LD_VAR_SPECIAL || mn == Mnemonic::LD_VAR_SPECIAL) {
            bool is_flat = (cmn == Mnemonic::LD_VAR_FLAT_IMM || mn == Mnemonic::LD_VAR_FLAT_IMM);
            uint32_t var_loc = 0;
            auto it = ins.fields.find("index");
            if (it != ins.fields.end()) var_loc = it->second / 16u;
            uint32_t var_id = get_or_create_frag_varying(var_loc, is_flat);
            uint32_t v4f = b.type_vector(b.type_f32(), 4);
            uint32_t vec = b.emit_load(v4f, var_id);
            uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
            uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
            for (uint8_t c = 0; c < cnt && c < 4; ++c) {
                uint32_t comp_f = b.emit_typed_inst(Op::OpCompositeExtract, b.type_f32(), { vec, uint32_t(c) });
                store_reg_u32(base_r + c, b.emit_bitcast(b.type_u32(), comp_f));
            }
            return;
        }

        // 11. Texture & Fused Varying-Texture Instructions (VAR_TEX_*, TEX_SINGLE, TEX_FETCH, TEX_GATHER, TEX_GRADIENT)
        if (cmn == Mnemonic::VAR_TEX_SINGLE || mn == Mnemonic::VAR_TEX_SINGLE ||
            cmn == Mnemonic::VAR_TEX_GATHER || mn == Mnemonic::VAR_TEX_GATHER ||
            cmn == Mnemonic::VAR_TEX_GRADIENT || mn == Mnemonic::VAR_TEX_GRADIENT ||
            cmn == Mnemonic::VAR_TEX_DUAL || mn == Mnemonic::VAR_TEX_DUAL) {
            // Fused Varying Interpolate (from in_var_0) + Texture Sample (from u_sampler_0)
            uint32_t var_id = get_or_create_frag_varying(0, false);
            uint32_t v4f = b.type_vector(b.type_f32(), 4);
            uint32_t v2f = b.type_vector(b.type_f32(), 2);
            uint32_t uv4 = b.emit_load(v4f, var_id);
            uint32_t u = b.emit_typed_inst(Op::OpCompositeExtract, b.type_f32(), { uv4, 0u });
            uint32_t v = b.emit_typed_inst(Op::OpCompositeExtract, b.type_f32(), { uv4, 1u });
            uint32_t uv = b.emit_typed_inst(Op::OpCompositeConstruct, v2f, { u, v });

            uint32_t si_type = 0;
            uint32_t samp_var = get_or_create_sampled_image(0, Dim::Dim2D, false, false, &si_type);
            uint32_t si_val = b.emit_load(si_type, samp_var);

            uint32_t sampled = 0;
            if (opts.stage == ShaderStage::Fragment) {
                sampled = b.emit_typed_inst(Op::OpImageSampleImplicitLod, v4f, { si_val, uv });
            } else {
                sampled = b.emit_typed_inst(Op::OpImageSampleExplicitLod, v4f,
                                            { si_val, uv, ImageOperandsMask::Lod, b.const_f32(0.0f) });
            }
            uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
            uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
            for (uint8_t c = 0; c < cnt && c < 4; ++c) {
                uint32_t comp_f = b.emit_typed_inst(Op::OpCompositeExtract, b.type_f32(), { sampled, uint32_t(c) });
                store_reg_u32(base_r + c, b.emit_bitcast(b.type_u32(), comp_f));
            }
            return;
        }
        if (cmn == Mnemonic::TEX_SINGLE || mn == Mnemonic::TEX_SINGLE ||
            cmn == Mnemonic::TEX_FETCH  || mn == Mnemonic::TEX_FETCH ||
            cmn == Mnemonic::TEX_GATHER || mn == Mnemonic::TEX_GATHER ||
            cmn == Mnemonic::TEX_GRADIENT || mn == Mnemonic::TEX_GRADIENT) {
            uint8_t coord_r = ins.sr_read.present ? ins.sr_read.base_reg : 0;
            uint32_t u = b.emit_bitcast(b.type_f32(), load_reg_u32(coord_r));
            uint32_t v = b.emit_bitcast(b.type_f32(), load_reg_u32(coord_r + 1));
            uint32_t v2f = b.type_vector(b.type_f32(), 2);
            uint32_t v4f = b.type_vector(b.type_f32(), 4);
            uint32_t uv = b.emit_typed_inst(Op::OpCompositeConstruct, v2f, { u, v });

            uint32_t si_type = 0;
            uint32_t samp_var = get_or_create_sampled_image(0, Dim::Dim2D, false, false, &si_type);
            uint32_t si_val = b.emit_load(si_type, samp_var);

            uint32_t sampled = 0;
            if (opts.stage == ShaderStage::Fragment && cmn != Mnemonic::TEX_GATHER) {
                sampled = b.emit_typed_inst(Op::OpImageSampleImplicitLod, v4f, { si_val, uv });
            } else if (cmn == Mnemonic::TEX_GATHER || mn == Mnemonic::TEX_GATHER) {
                uint32_t comp = 0;
                auto it_gc = ins.fields.find("gather_component");
                if (it_gc != ins.fields.end()) comp = it_gc->second & 3u;
                sampled = b.emit_typed_inst(Op::OpImageGather, v4f, { si_val, uv, b.const_u32(comp) });
            } else {
                sampled = b.emit_typed_inst(Op::OpImageSampleExplicitLod, v4f,
                                            { si_val, uv, ImageOperandsMask::Lod, b.const_f32(0.0f) });
            }
            uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
            uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
            for (uint8_t c = 0; c < cnt && c < 4; ++c) {
                uint32_t comp_f = b.emit_typed_inst(Op::OpCompositeExtract, b.type_f32(), { sampled, uint32_t(c) });
                store_reg_u32(base_r + c, b.emit_bitcast(b.type_u32(), comp_f));
            }
            return;
        }

        // 12. Fragment Output Instructions (ATEST, ZS_EMIT, BLEND)
        if (cmn == Mnemonic::ATEST || mn == Mnemonic::ATEST) {
            if (ins.has_dest) {
                uint32_t cov = (ins.srcs.size() >= 1) ? load_src_int32(ins, 0, false) : b.const_u32(1u);
                write_dest_u32(ins, cov);
            }
            return;
        }
        if (cmn == Mnemonic::ZS_EMIT || mn == Mnemonic::ZS_EMIT) {
            uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
            uint32_t z_f32 = b.emit_bitcast(b.type_f32(), load_reg_u32(sr));
            uint32_t depth_var = b.get_or_create_builtin_output(BuiltIn::FragDepth, b.type_f32(), "gl_FragDepth");
            b.emit_store(depth_var, z_f32);
            return;
        }
        if (cmn == Mnemonic::BLEND || mn == Mnemonic::BLEND) {
            uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
            uint32_t rt_loc = 0;
            if (!ins.srcs.empty() && ins.srcs[0].symbolic_name.find("blend_descriptor_") == 0) {
                char c = ins.srcs[0].symbolic_name.back();
                if (c >= '0' && c <= '7') rt_loc = uint32_t(c - '0');
            }
            uint32_t r = b.emit_bitcast(b.type_f32(), load_reg_u32(sr + 0));
            uint32_t g = b.emit_bitcast(b.type_f32(), load_reg_u32(sr + 1));
            uint32_t bl = b.emit_bitcast(b.type_f32(), load_reg_u32(sr + 2));
            uint32_t a = b.emit_bitcast(b.type_f32(), load_reg_u32(sr + 3));
            uint32_t v4f = b.type_vector(b.type_f32(), 4);
            uint32_t col = b.emit_typed_inst(Op::OpCompositeConstruct, v4f, { r, g, bl, a });
            uint32_t out_var = get_or_create_frag_color_output(rt_loc);
            b.emit_store(out_var, col);
            return;
        }

        // 13. Memory Loads (LOAD.*, LD_BUFFER_*)
        if (cmn == Mnemonic::LOAD || mn == Mnemonic::LOAD ||
            cmn == Mnemonic::LD_CHECKED || cmn == Mnemonic::LD_CHECKED_IMM ||
            cmn == Mnemonic::LD_CVT || cmn == Mnemonic::LD_PKA) {
            uint32_t base_addr = load_src_u64(ins, 0, false);
            uint32_t byte_addr = base_addr;
            if (ins.has_imm && ins.imm_signed != 0) {
                byte_addr = b.emit_typed_inst(Op::OpIAdd, b.type_u64(),
                                              { base_addr, b.const_u64(static_cast<uint64_t>(ins.imm_signed)) });
            }
            uint8_t sr = ins.sr_write.present ? ins.sr_write.base_reg : 0;
            uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 1;

            std::string mwidth = "i32";
            auto it_mw = ins.modifier_strs.find("memory_width");
            if (it_mw != ins.modifier_strs.end()) mwidth = it_mw->second;

            if (mwidth == "i8") {
                uint32_t p8 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u8());
                uint32_t ptr = b.emit_u_to_ptr(p8, byte_addr);
                uint32_t v8 = b.emit_load(b.type_u8(), ptr, 1u);
                uint32_t v32 = b.emit_typed_inst(Op::OpUConvert, b.type_u32(), { v8 });
                store_reg_u32(sr, v32);
                return;
            }
            if (mwidth == "i16") {
                uint32_t p16 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u16());
                uint32_t ptr = b.emit_u_to_ptr(p16, byte_addr);
                uint32_t v16 = b.emit_load(b.type_u16(), ptr, 2u);
                uint32_t v32 = b.emit_typed_inst(Op::OpUConvert, b.type_u32(), { v16 });
                store_reg_u32(sr, v32);
                return;
            }

            uint32_t p32 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u32());
            for (uint8_t w = 0; w < cnt; ++w) {
                uint32_t w_addr = (w == 0)
                    ? byte_addr
                    : b.emit_typed_inst(Op::OpIAdd, b.type_u64(), { byte_addr, b.const_u64(uint64_t(w) * 4ULL) });
                uint32_t ptr = b.emit_u_to_ptr(p32, w_addr);
                uint32_t val = b.emit_load(b.type_u32(), ptr, 4u);
                store_reg_u32(sr + w, val);
            }
            return;
        }

        // 14. Memory Stores (STORE.* including Vertex istream / estream)
        if (cmn == Mnemonic::STORE || mn == Mnemonic::STORE) {
            std::string access = "none";
            auto it_acc = ins.modifier_strs.find("access");
            if (it_acc != ins.modifier_strs.end()) access = it_acc->second;
            uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
            uint8_t cnt = ins.sr_read.present ? std::max<uint8_t>(1, ins.sr_read.count) : 1;

            if (opts.stage == ShaderStage::Vertex && access == "istream") {
                // Position EBIN output -> gl_Position
                // Check if gl_mali_ViewportTransform (u0.w0 != 0.0f) was applied in the shader:
                // In malioc's Position EBIN:
                //   r11 = 1.0 / w_clip
                //   r8  = (x_clip / w_clip) * u0.w0 + u2.w0
                //   r9  = (y_clip / w_clip) * u0.w1 + u2.w1
                //   r10 = (z_clip / w_clip) * u1.w0 + u1.w1
                // Inverting with the exact FAU values u0..u2 recovers exact clip-space gl_Position!
                uint32_t r8_f  = b.emit_bitcast(b.type_f32(), load_reg_u32(sr + 0));
                uint32_t r9_f  = b.emit_bitcast(b.type_f32(), load_reg_u32(sr + 1));
                uint32_t r10_f = b.emit_bitcast(b.type_f32(), load_reg_u32(sr + 2));
                uint32_t r11_f = b.emit_bitcast(b.type_f32(), load_reg_u32(sr + 3));

                uint32_t sx = b.emit_bitcast(b.type_f32(), load_fau_u32(0));
                uint32_t sy = b.emit_bitcast(b.type_f32(), load_fau_u32(1));
                uint32_t sz = b.emit_bitcast(b.type_f32(), load_fau_u32(2));
                uint32_t oz = b.emit_bitcast(b.type_f32(), load_fau_u32(3));
                uint32_t ox = b.emit_bitcast(b.type_f32(), load_fau_u32(4));
                uint32_t oy = b.emit_bitcast(b.type_f32(), load_fau_u32(5));

                uint32_t has_vp = b.emit_typed_inst(Op::OpFOrdNotEqual, b.type_bool(), { sx, b.const_f32(0.0f) });
                uint32_t wc = b.emit_typed_inst(Op::OpFDiv, b.type_f32(), { b.const_f32(1.0f), r11_f });
                uint32_t dx = b.emit_typed_inst(Op::OpFSub, b.type_f32(), { r8_f, ox });
                uint32_t dy = b.emit_typed_inst(Op::OpFSub, b.type_f32(), { r9_f, oy });
                uint32_t dz = b.emit_typed_inst(Op::OpFSub, b.type_f32(), { r10_f, oz });
                uint32_t xc = b.emit_typed_inst(Op::OpFMul, b.type_f32(),
                                                { b.emit_typed_inst(Op::OpFDiv, b.type_f32(), { dx, sx }), wc });
                uint32_t yc = b.emit_typed_inst(Op::OpFMul, b.type_f32(),
                                                { b.emit_typed_inst(Op::OpFDiv, b.type_f32(), { dy, sy }), wc });
                uint32_t zc = b.emit_typed_inst(Op::OpFMul, b.type_f32(),
                                                { b.emit_typed_inst(Op::OpFDiv, b.type_f32(), { dz, sz }), wc });

                uint32_t final_x = b.emit_select(b.type_f32(), has_vp, xc, r8_f);
                uint32_t final_y = b.emit_select(b.type_f32(), has_vp, yc, r9_f);
                uint32_t final_z = b.emit_select(b.type_f32(), has_vp, zc, r10_f);
                uint32_t final_w = b.emit_select(b.type_f32(), has_vp, wc, r11_f);

                uint32_t v4f = b.type_vector(b.type_f32(), 4);
                uint32_t pos_vec = b.emit_typed_inst(Op::OpCompositeConstruct, v4f,
                                                     { final_x, final_y, final_z, final_w });
                uint32_t pos_var = b.get_or_create_builtin_output(BuiltIn::Position, v4f, "gl_Position");
                b.emit_store(pos_var, pos_vec);
                return;
            }
            if (opts.stage == ShaderStage::Vertex && access == "estream") {
                // Varying EBIN output -> out_var_k
                uint32_t var_loc = static_cast<uint32_t>(std::max<int64_t>(0, ins.imm_signed) / 16);
                uint32_t comps[4];
                for (uint8_t c = 0; c < 4; ++c) {
                    if (c < cnt) {
                        comps[c] = b.emit_bitcast(b.type_f32(), load_reg_u32(sr + c));
                    } else {
                        comps[c] = b.const_f32((c == 3) ? 1.0f : 0.0f);
                    }
                }
                uint32_t v4f = b.type_vector(b.type_f32(), 4);
                uint32_t vec = b.emit_typed_inst(Op::OpCompositeConstruct, v4f,
                                                 { comps[0], comps[1], comps[2], comps[3] });
                uint32_t out_var = get_or_create_vert_varying(var_loc);
                b.emit_store(out_var, vec);
                return;
            }

            // Standard GPU memory store via PhysicalStorageBuffer64
            uint32_t base_addr = load_src_u64(ins, 0, false);
            uint32_t byte_addr = base_addr;
            if (ins.has_imm && ins.imm_signed != 0) {
                byte_addr = b.emit_typed_inst(Op::OpIAdd, b.type_u64(),
                                              { base_addr, b.const_u64(static_cast<uint64_t>(ins.imm_signed)) });
            }
            std::string mwidth = "i32";
            auto it_mw = ins.modifier_strs.find("memory_width");
            if (it_mw != ins.modifier_strs.end()) mwidth = it_mw->second;

            if (mwidth == "i8") {
                uint32_t p8 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u8());
                uint32_t ptr = b.emit_u_to_ptr(p8, byte_addr);
                uint32_t v8 = b.emit_typed_inst(Op::OpUConvert, b.type_u8(), { load_reg_u32(sr) });
                b.emit_store(ptr, v8, 1u);
                return;
            }
            if (mwidth == "i16") {
                uint32_t p16 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u16());
                uint32_t ptr = b.emit_u_to_ptr(p16, byte_addr);
                uint32_t v16 = b.emit_typed_inst(Op::OpUConvert, b.type_u16(), { load_reg_u32(sr) });
                b.emit_store(ptr, v16, 2u);
                return;
            }

            uint32_t p32 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u32());
            for (uint8_t w = 0; w < cnt; ++w) {
                uint32_t w_addr = (w == 0)
                    ? byte_addr
                    : b.emit_typed_inst(Op::OpIAdd, b.type_u64(), { byte_addr, b.const_u64(uint64_t(w) * 4ULL) });
                uint32_t ptr = b.emit_u_to_ptr(p32, w_addr);
                b.emit_store(ptr, load_reg_u32(sr + w), 4u);
            }
            return;
        }

        // 15. Memory Atomics (ATOM, ATOM_RETURN, ATOM1, ATOM1_RETURN, AXCHG, ACMPXCHG)
        if (ins.category == SemanticCategory::MEMORY_ATOMIC ||
            cmn == Mnemonic::ATOM || cmn == Mnemonic::ATOM_RETURN ||
            cmn == Mnemonic::ATOM1 || cmn == Mnemonic::ATOM1_RETURN ||
            cmn == Mnemonic::AXCHG || cmn == Mnemonic::ACMPXCHG) {
            uint32_t base_addr = load_src_u64(ins, 0, false);
            uint32_t byte_addr = base_addr;
            if (ins.has_imm && ins.imm_signed != 0) {
                byte_addr = b.emit_typed_inst(Op::OpIAdd, b.type_u64(),
                                              { base_addr, b.const_u64(static_cast<uint64_t>(ins.imm_signed)) });
            }
            uint32_t p32 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u32());
            uint32_t ptr = b.emit_u_to_ptr(p32, byte_addr);
            uint32_t scope_dev = b.const_u32(1u);
            uint32_t sem_relaxed = b.const_u32(0u);

            if (cmn == Mnemonic::ATOM1 || cmn == Mnemonic::ATOM1_RETURN || mn == Mnemonic::ATOM1 || mn == Mnemonic::ATOM1_RETURN) {
                std::string aop = "ainc";
                auto it = ins.modifier_strs.find("atom1_opc");
                if (it != ins.modifier_strs.end()) aop = it->second;
                uint32_t delta = (aop == "adec") ? b.const_u32(0xFFFFFFFFu) : b.const_u32(1u);
                uint32_t old_v = b.emit_typed_inst(Op::OpAtomicIAdd, b.type_u32(),
                                                   { ptr, scope_dev, sem_relaxed, delta });
                if (ins.sr_write.present) {
                    store_reg_u32(ins.sr_write.base_reg, old_v);
                }
                return;
            }
            if (cmn == Mnemonic::AXCHG || mn == Mnemonic::AXCHG) {
                uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
                uint32_t val = load_reg_u32(sr);
                uint32_t old_v = b.emit_typed_inst(Op::OpAtomicExchange, b.type_u32(),
                                                   { ptr, scope_dev, sem_relaxed, val });
                if (ins.sr_write.present) {
                    store_reg_u32(ins.sr_write.base_reg, old_v);
                }
                return;
            }
            if (cmn == Mnemonic::ACMPXCHG || mn == Mnemonic::ACMPXCHG) {
                uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
                uint32_t cmp_v = load_reg_u32(sr);
                uint32_t xchg_v = load_reg_u32(sr + 1);
                uint32_t old_v = b.emit_typed_inst(Op::OpAtomicCompareExchange, b.type_u32(),
                                                   { ptr, scope_dev, sem_relaxed, sem_relaxed, xchg_v, cmp_v });
                if (ins.sr_write.present) {
                    store_reg_u32(ins.sr_write.base_reg, old_v);
                }
                return;
            }

            // General ATOM / ATOM_RETURN
            uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
            uint32_t arg = load_reg_u32(sr);
            std::string aop = "aadd";
            auto it = ins.modifier_strs.find("atom_opc");
            if (it != ins.modifier_strs.end()) aop = it->second;

            Op spv_aop = Op::OpAtomicIAdd;
            if (aop == "asmin") spv_aop = Op::OpAtomicSMin;
            else if (aop == "aumin") spv_aop = Op::OpAtomicUMin;
            else if (aop == "asmax") spv_aop = Op::OpAtomicSMax;
            else if (aop == "aumax") spv_aop = Op::OpAtomicUMax;
            else if (aop == "aand") spv_aop = Op::OpAtomicAnd;
            else if (aop == "aor") spv_aop = Op::OpAtomicOr;
            else if (aop == "axor") spv_aop = Op::OpAtomicXor;

            uint32_t old_v = b.emit_typed_inst(spv_aop, b.type_u32(),
                                               { ptr, scope_dev, sem_relaxed, arg });
            if (ins.sr_write.present) {
                store_reg_u32(ins.sr_write.base_reg, old_v);
            }
            return;
        }
    }

    uint32_t eval_branch_condition(const DecodedValhallInst& ins) {
        if (ins.srcs.empty()) {
            return b.const_bool(true);
        }
        uint32_t s0 = load_src_int32(ins, 0, false);
        uint32_t cond = b.emit_typed_inst(Op::OpINotEqual, b.type_bool(), { s0, b.const_u32(0u) });

        std::string comb = "none";
        auto it_c = ins.modifier_strs.find("combine");
        if (it_c != ins.modifier_strs.end()) comb = it_c->second;
        if (ins.srcs.size() >= 2 && comb != "none") {
            uint32_t s1 = load_src_int32(ins, 1, false);
            if (comb == "eq") {
                cond = b.emit_typed_inst(Op::OpIEqual, b.type_bool(), { s0, s1 });
            } else {
                uint32_t c1 = b.emit_typed_inst(Op::OpINotEqual, b.type_bool(), { s1, b.const_u32(0u) });
                Op lop = (comb == "and") ? Op::OpLogicalAnd : Op::OpLogicalOr;
                cond = b.emit_typed_inst(lop, b.type_bool(), { cond, c1 });
            }
        }

        // Check negate / branchz eq
        auto it_not = ins.fields.find("not");
        bool negate = (it_not != ins.fields.end() && it_not->second != 0);

        if (negate) {
            cond = b.emit_typed_inst(Op::OpLogicalNot, b.type_bool(), { cond });
        }
        return cond;
    }

    void emit_cfg_blocks(const cfg::ControlFlowGraph& cfg, uint32_t exit_block_override = 0) {
        if (cfg.blocks.empty()) {
            if (exit_block_override != 0) b.emit_branch(exit_block_override);
            else b.emit_return();
            return;
        }

        // Allocate SPIR-V label IDs for all blocks
        std::vector<uint32_t> blk_labels(cfg.blocks.size());
        for (size_t i = 0; i < cfg.blocks.size(); ++i) {
            blk_labels[i] = b.alloc_id();
        }
        // Jump from current setup block into block 0
        b.emit_branch(blk_labels[0]);

        for (size_t i = 0; i < cfg.blocks.size(); ++i) {
            const auto& blk = cfg.blocks[i];
            b.emit_label(blk_labels[i]);

            // If this block is a loop header, emit OpLoopMerge before the branch
            size_t num_insts = blk.inst_indices.size();
            for (size_t k = 0; k < num_insts; ++k) {
                size_t ins_idx = blk.inst_indices[k];
                const auto& ins = cfg.instructions[ins_idx];
                bool is_last = (k + 1 == num_insts);
                if (is_last && (blk.is_conditional_branch || blk.is_unconditional_branch)) {
                    break;
                }
                emit_instruction(ins);
            }

            // Emit block terminator
            if (blk.is_terminated_by_discard) {
                // OpKill was already emitted by DISCARD
                continue;
            }
            if (blk.is_unconditional_branch && blk.branch_target_block >= 0 &&
                static_cast<size_t>(blk.branch_target_block) < blk_labels.size()) {
                b.emit_branch(blk_labels[blk.branch_target_block]);
                continue;
            }
            if (blk.is_conditional_branch && blk.branch_target_block >= 0 &&
                static_cast<size_t>(blk.branch_target_block) < blk_labels.size() &&
                blk.fallthrough_block >= 0 &&
                static_cast<size_t>(blk.fallthrough_block) < blk_labels.size()) {
                const auto& br_ins = cfg.instructions[blk.inst_indices.back()];
                uint32_t cond = eval_branch_condition(br_ins);
                uint32_t tgt_lbl = blk_labels[blk.branch_target_block];
                uint32_t fall_lbl = blk_labels[blk.fallthrough_block];

                if (blk.is_loop_header && blk.loop_backedge_block >= 0 && blk.loop_merge_block >= 0) {
                    b.emit_loop_merge(blk_labels[blk.loop_merge_block],
                                      blk_labels[blk.loop_backedge_block]);
                } else if (blk.selection_merge_block >= 0 &&
                           static_cast<size_t>(blk.selection_merge_block) < blk_labels.size()) {
                    b.emit_selection_merge(blk_labels[blk.selection_merge_block]);
                }
                b.emit_branch_conditional(cond, tgt_lbl, fall_lbl);
                continue;
            }
            if (blk.is_terminated_by_end) {
                if (exit_block_override != 0) {
                    b.emit_branch(exit_block_override);
                } else {
                    b.emit_return();
                }
                continue;
            }
            if (blk.fallthrough_block >= 0 &&
                static_cast<size_t>(blk.fallthrough_block) < blk_labels.size()) {
                b.emit_branch(blk_labels[blk.fallthrough_block]);
            } else {
                if (exit_block_override != 0) b.emit_branch(exit_block_override);
                else b.emit_return();
            }
        }
    }
};

} // namespace

ValhallTranslationResult translate_valhall_to_spirv(
    const uint8_t* code_bytes,
    size_t code_size_bytes,
    const ValhallTranslateOptions& opts) {
    ValhallTranslationResult res;
    if (!code_bytes || code_size_bytes < 8) {
        res.error_message = "Empty or truncated Valhall instruction stream";
        return res;
    }

    auto raw_insts = decode_valhall_shader(code_bytes, code_size_bytes, opts.arch, opts.fau32);
    res.primary_cfg = cfg::build_cfg(raw_insts);

    std::ostringstream dis_oss;
    dis_oss << "// Stage: " << int(opts.stage) << " (" << res.primary_cfg.instructions.size() << " executable instructions)\n";
    for (const auto& ins : res.primary_cfg.instructions) {
        dis_oss << "0x" << std::hex << std::setw(4) << std::setfill('0') << ins.pc
                << ": " << ins.disassembly << "\n";
    }

    if (opts.secondary_vertex_code && opts.secondary_vertex_code_size >= 8) {
        auto sec_raw = decode_valhall_shader(opts.secondary_vertex_code, opts.secondary_vertex_code_size, opts.arch, opts.fau32);
        res.secondary_cfg = cfg::build_cfg(sec_raw);
        dis_oss << "// Secondary Vertex Varying EBIN (" << res.secondary_cfg.instructions.size() << " instructions)\n";
        for (const auto& ins : res.secondary_cfg.instructions) {
            dis_oss << "0x" << std::hex << std::setw(4) << std::setfill('0') << ins.pc
                    << ": " << ins.disassembly << "\n";
        }
    }
    res.disassembly_listing = dis_oss.str();

    ExecutionModel model = ExecutionModel::GLCompute;
    if (opts.stage == ShaderStage::Vertex) model = ExecutionModel::Vertex;
    else if (opts.stage == ShaderStage::Fragment) model = ExecutionModel::Fragment;

    TranslatorContext ctx(model, opts);
    auto& b = ctx.b;

    if (model == ExecutionModel::GLCompute) {
        uint32_t lx = (opts.stage == ShaderStage::Pilot) ? 1u : std::max(1u, opts.local_size_x);
        uint32_t ly = (opts.stage == ShaderStage::Pilot) ? 1u : std::max(1u, opts.local_size_y);
        uint32_t lz = (opts.stage == ShaderStage::Pilot) ? 1u : std::max(1u, opts.local_size_z);
        b.add_execution_mode(ExecutionMode::LocalSize, { lx, ly, lz });
    } else if (model == ExecutionModel::Fragment) {
        b.add_execution_mode(ExecutionMode::OriginUpperLeft);
        for (const auto& ins : res.primary_cfg.instructions) {
            if (ins.canonical_mnemonic == Mnemonic::ZS_EMIT || ins.mnemonic == Mnemonic::ZS_EMIT) {
                b.add_execution_mode(ExecutionMode::DepthReplacing);
                break;
            }
        }
    }

    // Declare ValhallSidebandData StorageBuffer at (opts.sideband_set, opts.sideband_binding)
    uint32_t arr_fau_t = b.type_array(b.type_u32(), b.const_u32(64), 4);
    uint32_t arr_blend_t = b.type_array(b.type_u64(), b.const_u32(4), 8);
    uint32_t v4u32_t = b.type_vector(b.type_u32(), 4);
    ctx.sideband_struct_t = b.type_struct(
        { arr_fau_t, b.type_u64(), b.type_u64(), b.type_u64(), arr_blend_t, v4u32_t, v4u32_t },
        "ValhallSidebandBuffer");
    b.decorate(ctx.sideband_struct_t, Decoration::Block);
    b.member_decorate(ctx.sideband_struct_t, 0, Decoration::Offset, { 0 });
    b.member_decorate(ctx.sideband_struct_t, 1, Decoration::Offset, { 256 });
    b.member_decorate(ctx.sideband_struct_t, 2, Decoration::Offset, { 264 });
    b.member_decorate(ctx.sideband_struct_t, 3, Decoration::Offset, { 272 });
    b.member_decorate(ctx.sideband_struct_t, 4, Decoration::Offset, { 280 });
    b.member_decorate(ctx.sideband_struct_t, 5, Decoration::Offset, { 312 });
    b.member_decorate(ctx.sideband_struct_t, 6, Decoration::Offset, { 328 });

    uint32_t sb_ptr_t = b.type_pointer(StorageClass::StorageBuffer, ctx.sideband_struct_t);
    ctx.sideband_var_id = b.add_global_var(sb_ptr_t, StorageClass::StorageBuffer, "u_sideband", false);
    b.decorate(ctx.sideband_var_id, Decoration::DescriptorSet, { opts.sideband_set });
    b.decorate(ctx.sideband_var_id, Decoration::Binding, { opts.sideband_binding });
    b.decorate(ctx.sideband_var_id, Decoration::NonWritable);

    // Begin main()
    b.begin_main_function();

    // Allocate 64 scalar 32-bit registers r0..r63 in the entry block
    for (uint32_t r = 0; r < 64; ++r) {
        ctx.reg_vars[r] = b.add_local_var(b.type_u32(), "r" + std::to_string(r));
        b.emit_store(ctx.reg_vars[r], b.const_u32(0u));
    }

    // Stage-specific register preloading
    if (opts.stage == ShaderStage::Pilot) {
        // [r0:r1] = pilot_fau_dst_va, [r2:r3] = srt_base_va
        uint32_t fau_va = ctx.load_sideband_u64(3);
        uint32_t srt_va = ctx.load_sideband_u64(2);
        uint32_t f_lo = 0, f_hi = 0, s_lo = 0, s_hi = 0;
        ctx.unpack_u64(fau_va, &f_lo, &f_hi);
        ctx.unpack_u64(srt_va, &s_lo, &s_hi);
        ctx.store_reg_u32(0, f_lo);
        ctx.store_reg_u32(1, f_hi);
        ctx.store_reg_u32(2, s_lo);
        ctx.store_reg_u32(3, s_hi);
    } else if (opts.stage == ShaderStage::Compute) {
        // Check if shader reads r55..r57 (WorkgroupId) directly
        bool reads_wg_regs = false;
        for (const auto& ins : res.primary_cfg.instructions) {
            for (const auto& s : ins.srcs) {
                if (s.kind == SourceKind::Register && (s.reg_num >= 55 && s.reg_num <= 57)) {
                    reads_wg_regs = true;
                }
            }
        }
        uint32_t v3u = b.type_vector(b.type_u32(), 3);
        uint32_t wgid_var = b.get_or_create_builtin_input(BuiltIn::WorkgroupId, v3u, "gl_WorkGroupID");
        uint32_t wgid = b.emit_load(v3u, wgid_var);
        ctx.store_reg_u32(55, b.emit_typed_inst(Op::OpCompositeExtract, b.type_u32(), { wgid, 0u }));
        ctx.store_reg_u32(56, b.emit_typed_inst(Op::OpCompositeExtract, b.type_u32(), { wgid, 1u }));
        ctx.store_reg_u32(57, b.emit_typed_inst(Op::OpCompositeExtract, b.type_u32(), { wgid, 2u }));

        BuiltIn id_bi = reads_wg_regs ? BuiltIn::LocalInvocationId : BuiltIn::GlobalInvocationId;
        const char* id_name = reads_wg_regs ? "gl_LocalInvocationID" : "gl_GlobalInvocationID";
        uint32_t tid_var = b.get_or_create_builtin_input(id_bi, v3u, id_name);
        uint32_t tid = b.emit_load(v3u, tid_var);
        ctx.store_reg_u32(60, b.emit_typed_inst(Op::OpCompositeExtract, b.type_u32(), { tid, 0u }));
        ctx.store_reg_u32(61, b.emit_typed_inst(Op::OpCompositeExtract, b.type_u32(), { tid, 1u }));
        ctx.store_reg_u32(62, b.emit_typed_inst(Op::OpCompositeExtract, b.type_u32(), { tid, 2u }));
    } else if (opts.stage == ShaderStage::Vertex) {
        uint32_t vi_var = b.get_or_create_builtin_input(BuiltIn::VertexIndex, b.type_u32(), "gl_VertexIndex");
        uint32_t ii_var = b.get_or_create_builtin_input(BuiltIn::InstanceIndex, b.type_u32(), "gl_InstanceIndex");
        uint32_t vi = b.emit_load(b.type_u32(), vi_var);
        uint32_t ii = b.emit_load(b.type_u32(), ii_var);
        ctx.store_reg_u32(59, vi);
        ctx.store_reg_u32(60, vi);
        ctx.store_reg_u32(61, ii);
    } else if (opts.stage == ShaderStage::Fragment) {
        uint32_t v4f = b.type_vector(b.type_f32(), 4);
        uint32_t fc_var = b.get_or_create_builtin_input(BuiltIn::FragCoord, v4f, "gl_FragCoord");
        uint32_t fc = b.emit_load(v4f, fc_var);
        uint32_t fz = b.emit_typed_inst(Op::OpCompositeExtract, b.type_f32(), { fc, 2u });
        ctx.store_reg_u32(59, b.emit_bitcast(b.type_u32(), fz));
        ctx.store_reg_u32(60, b.const_u32(1u));
    }

    if (opts.stage == ShaderStage::Vertex && !res.secondary_cfg.blocks.empty()) {
        // Fuse Position EBIN (primary_cfg) -> Varying EBIN (secondary_cfg)
        uint32_t varying_entry_lbl = b.alloc_id();
        ctx.emit_cfg_blocks(res.primary_cfg, varying_entry_lbl);
        b.emit_label(varying_entry_lbl);
        // Re-initialize r59, r60, r61 for Varying EBIN
        uint32_t vi_var = b.get_or_create_builtin_input(BuiltIn::VertexIndex, b.type_u32(), "gl_VertexIndex");
        uint32_t ii_var = b.get_or_create_builtin_input(BuiltIn::InstanceIndex, b.type_u32(), "gl_InstanceIndex");
        uint32_t vi = b.emit_load(b.type_u32(), vi_var);
        uint32_t ii = b.emit_load(b.type_u32(), ii_var);
        ctx.store_reg_u32(59, vi);
        ctx.store_reg_u32(60, vi);
        ctx.store_reg_u32(61, ii);
        ctx.emit_cfg_blocks(res.secondary_cfg, 0);
    } else {
        ctx.emit_cfg_blocks(res.primary_cfg, 0);
    }

    b.end_main_function();
    res.spirv_words = b.assemble();
    res.ok = true;
    return res;
}

} // namespace valhall
