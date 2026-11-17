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

// Uniform Host ABI Functions (@fn(...) in @INLINE_OPS, typed in valhall_lowering.json["host_functions"])
// ABI: RET fn(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, ARGS...)
namespace abi {

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

} // namespace valhall::compiler
