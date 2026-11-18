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

// Uniform Host ABI Functions (@fn, $fn = @fn(), #fn = @fn(...) in valhall_lowering.json["host_functions"])
// ABI: RET fn(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, ARGS...)
namespace abi {

// 0-arg Metavariables ($FN -> @FN())
bool has_src0(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool has_src1(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool has_src2(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool has_src3(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool has_dest(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool is_v2f16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool is_signed(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool is_64bit(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool dest_is_64bit(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool src0_swz_h1(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool is_frag_stage(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool is_vert_stage(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool has_imm_nonzero(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool sr_write_present(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool is_64_atom(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool is_gather(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool is_flat_var(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool has_not_result(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
bool has_not(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

uint32_t sr_write_count(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
uint32_t sr_write_count_max4(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
uint32_t sr_read_count(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
uint32_t src0_swz_h1_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
SsaId branch_target_pc_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

// Parameterized Predicates & Field Helpers (@fn)
bool mod_active(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, const char* key);
bool mod_eq(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, const char* key, const char* val);
bool mod_ne(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, const char* key, const char* val);
bool has_field(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, const char* name);
bool field_nz(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, const char* name);
SsaId field_u32_mul16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, const char* name);
bool is_zero(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t v);
bool lt_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t a, uint32_t b);

// Constant / Immediate Constructors (#FN -> @FN)
SsaId u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t v);
SsaId s32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, int32_t v);
SsaId u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint64_t v);
SsaId f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, float v);
SsaId f32_bits(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t bits);
SsaId const_bool(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, bool v);
SsaId fp_zero(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
SsaId token(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
SsaId raw0(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
SsaId imm_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
SsaId imm_s32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
SsaId imm_f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
SsaId imm_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
SsaId imm_u64_mask63(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
SsaId u64_mul4(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t w);
SsaId f32_alpha_or_zero(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t c);

// SSA / Staging / Sideband / Stage I/O ABI Functions (@fn)
SsaId load_sideband_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t member_idx);
SsaId split64_lo(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val_u64);
SsaId load_fau(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t word_idx);
SsaId bitcast(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, ValType vt, SsaId val);
SsaId extract(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, ValType vt, SsaId vec, uint32_t idx);
SsaId shuffle2(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, ValType vt, SsaId a, SsaId b, uint32_t i0, uint32_t i1);
SsaId bfi(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId a, SsaId b, uint32_t off, uint32_t w);
void set_token(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId tok);
void set_branch_cond(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId cond);

SsaId sr_get(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t offset);
void sr_set(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t offset, SsaId val);
SsaId tuple_val(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, ValType vt, SsaId tup);
SsaId tuple_tok(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId tup);

SsaId psb_load(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId addr, uint32_t elem_bytes, bool is_pka, SsaId tok);
SsaId psb_store(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId addr, SsaId val, uint32_t elem_bytes, SsaId tok);
SsaId atomic_rmw(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId addr, SsaId arg, bool is_64, SsaId tok);
SsaId atomic_rmw_aadd(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId addr, SsaId arg, bool is_64, SsaId tok);
SsaId atomic_xchg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId addr, SsaId arg, SsaId tok);
SsaId atomic_cmpxchg(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId addr, SsaId xchg, SsaId cmp, SsaId tok);

SsaId load_vert_attr(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);
SsaId load_frag_var(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, bool is_flat);
SsaId load_frag_var_idx(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t var_loc, bool is_flat);
SsaId tex_sample(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId uv, bool is_gather, SsaId tok);

SsaId store_builtin_pos(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId pos_vec, SsaId tok);
SsaId store_builtin_depth(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId z_f32, SsaId tok);
SsaId store_vert_varying(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec, SsaId tok);
SsaId store_frag_color(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId col, SsaId tok);
void kill_cond(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId cond, SsaId tok);

} // namespace abi

// Remaining host-side handler (@HANDLER(fn) in valhall_lowering.json)
void handle_int_idpadd(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);


// Operand-lowering host ABI helpers (@host functions for specs/lowering/valhall_operands.json)
namespace operands_host {

std::string get_src_swz(const DecodedValhallInst& ins, size_t src_idx);
std::string get_dest_mod(const DecodedValhallInst& ins);
bool get_src_abs(const DecodedValhallInst& ins, size_t src_idx);
bool get_src_neg(const DecodedValhallInst& ins, size_t src_idx);
bool get_src_not(const DecodedValhallInst& ins, size_t src_idx);

bool src_swz_eq(const DecodedValhallInst& ins, size_t src_idx, const char* s);
bool src_swz_is_b_rep(const DecodedValhallInst& ins, size_t src_idx);
uint32_t src_swz_b_rep_off(const DecodedValhallInst& ins, size_t src_idx);
bool dest_mod_eq(const DecodedValhallInst& ins, const char* s);
bool dest_mod_is_byte(const DecodedValhallInst& ins);
uint32_t dest_mod_byte_off(const DecodedValhallInst& ins);

bool has_clamp_0_1(const DecodedValhallInst& ins);
bool has_clamp_m1_1(const DecodedValhallInst& ins);
bool has_clamp_0_inf(const DecodedValhallInst& ins);

bool src_oob(const DecodedValhallInst& ins, size_t src_idx);
bool src_is_reg(const DecodedValhallInst& ins, size_t src_idx);
bool src_is_fau_uniform(const DecodedValhallInst& ins, size_t src_idx);
bool src_is_small_const(const DecodedValhallInst& ins, size_t src_idx);
bool src_is_fau_pc(const DecodedValhallInst& ins, size_t src_idx);
bool src_is_fau_lane_id(const DecodedValhallInst& ins, size_t src_idx);
bool src_is_fau_lane_width(const DecodedValhallInst& ins, size_t src_idx);
bool src_word_select(const DecodedValhallInst& ins, size_t src_idx);
bool src_is_64bit(const DecodedValhallInst& ins, size_t src_idx);
uint32_t src_reg_num(const DecodedValhallInst& ins, size_t src_idx);
uint32_t src_reg_base(const DecodedValhallInst& ins, size_t src_idx);

bool has_dest(const DecodedValhallInst& ins);
uint32_t dest_reg_num(const DecodedValhallInst& ins);
uint32_t dest_reg_base(const DecodedValhallInst& ins);

SsaId src_read_reg(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx);
SsaId src_read_reg_base(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, uint32_t off);
SsaId src_emit_imm_u32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx);
SsaId src_emit_imm_u64(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx);
SsaId ins_pc_u64(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins);
SsaId dest_read_reg(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins);
void dest_set_reg(SsaBuilder& sb, uint32_t blk_id, uint32_t d, SsaId val);

SsaId emit_load_fau_u32(SsaBuilder& sb, uint32_t blk_id, uint32_t word_idx);
SsaId emit_load_sideband_u64(SsaBuilder& sb, uint32_t blk_id, uint32_t member_idx);
SsaId emit_pack_u64(SsaBuilder& sb, uint32_t blk_id, SsaId lo_u32, SsaId hi_u32);
SsaId split64_lo(SsaBuilder& sb, uint32_t blk_id, SsaId val_u64);
SsaId split64_hi(SsaBuilder& sb, uint32_t blk_id, SsaId val_u64);
SsaId emit_lane_id(SsaBuilder& sb, uint32_t blk_id);
SsaId emit_lane_width(SsaBuilder& sb, uint32_t blk_id);

SsaId const_u32(SsaBuilder& sb, uint32_t blk_id, uint32_t v);
SsaId const_u64(SsaBuilder& sb, uint32_t blk_id, uint64_t v);
SsaId const_f32_bits(SsaBuilder& sb, uint32_t blk_id, uint32_t bits);
SsaId bitcast(SsaBuilder& sb, uint32_t blk_id, ValType vt, SsaId val);
SsaId extract(SsaBuilder& sb, uint32_t blk_id, ValType vt, SsaId vec, uint32_t idx);
SsaId shuffle2(SsaBuilder& sb, uint32_t blk_id, ValType vt, SsaId a, SsaId b, uint32_t i0, uint32_t i1);
SsaId bfi(SsaBuilder& sb, uint32_t blk_id, SsaId a, SsaId b, uint32_t off, uint32_t w);
uint32_t add_u32(uint32_t a, uint32_t b);
uint32_t mul8(uint32_t a);

} // namespace operands_host

} // namespace valhall::compiler
