#pragma once
#include "valhall_pass1.h"
#include "valhall_operands.h"
#include "valhall_tables_generated.h"
#include <cstdint>
#include <string>
#include <vector>

namespace valhall::compiler {

struct Pass1Context {
    ValhallSsaModule& mod;
    SsaBuilder sb;
    const ValhallTranslateOptions& opts;
    Pass1Result& res;

    Pass1Context(ValhallSsaModule& m, const ValhallTranslateOptions& o, Pass1Result& r)
        : mod(m), sb(m), opts(o), res(r) {}

    SsaId emit_op(uint32_t blk, tables::SsaOp op, ValType vt, std::vector<SsaId> args,
                  tables::EffectMask eff = {}, bool barrier = false, bool side_eff = false);
    SsaId emit_load_fau(uint32_t blk, uint32_t word_idx);
    SsaId emit_load_sideband_u64(uint32_t blk, uint32_t member_idx);
    SsaId emit_builtin_input(uint32_t blk, spirv::BuiltIn bi, ValType vt, const char* name, bool flat = false);
    void emit_split64(uint32_t blk, SsaId val_u64, SsaId* lo, SsaId* hi);
    void emit_stage_preamble(uint32_t blk, const std::vector<DecodedValhallInst>& insts);

    SsaId eval_float_cmp(uint32_t blk, const std::string& cmpf, SsaId a, SsaId b);
    SsaId eval_int_cmp(uint32_t blk, const std::string& cmpf, SsaId a, SsaId b, bool is_signed);
    SsaId format_cmp_result(uint32_t blk, const DecodedValhallInst& ins, SsaId cond_bool);
    SsaId eval_branch_cond(uint32_t blk, const DecodedValhallInst& ins);
};

// Dynamic SsaOp selectors (@fn in valhall_lowering.json)
tables::SsaOp get_round_mode(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins,
                             ValType vt, const std::vector<SsaId>& args, tables::EffectMask effects);
tables::SsaOp get_fp_minmax_op(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins,
                               ValType vt, const std::vector<SsaId>& args, tables::EffectMask effects);

// Full instruction handlers (HANDLER(@fn) in valhall_lowering.json)
void handle_cf_branch(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_cf_jump(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_sys_barrier(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_sys_discard(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_sys_gclk(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

void handle_alu_adr(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_alu_mov_imm(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_alu_vec_pack(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

void handle_fp_imm(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_fma(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_flush(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_lscale(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_rscale(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_rcp(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_trig_turn(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_sfu2(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_frexp(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_rcbrt(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_class(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_fp_cube(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

void handle_int_imm(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_int_ternary(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_int_hadd(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_int_idpadd(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_int_shadd64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

void handle_bit_logic(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_bit_logic64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_bit_imm(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_bit_shift(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_bit_shift64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_bit_shift_logic(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_bit_shift_logic64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_bit_rot_logic(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_bit_rot_logic64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_bit_mux(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

void handle_cmp_float(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_cmp_int(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_cmp_csel(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

void handle_cvt_f16_to_f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_cvt_f32_to_f16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_cvt_f16_to_s32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_cvt_f16_to_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_cvt_v2i8_to_v2f16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_cvt_v2i8_to_v2i16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

void handle_subgroup_wmask(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_subgroup_clper(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

void handle_mem_lea(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_mem_load(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_mem_load_pka(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_mem_store(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_mem_atomic(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_mem_atom1(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_mem_acmpxchg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_mem_axchg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

void handle_vert_attr(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_frag_var(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_tex_var_sample(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_tex_sample(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_frag_atest(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_frag_zs_emit(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
void handle_frag_blend(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

} // namespace valhall::compiler
