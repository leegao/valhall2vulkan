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

// Remaining complex host-side handlers (@HANDLER(fn) in valhall_lowering.json)
void handle_int_idpadd(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

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
