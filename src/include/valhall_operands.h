#pragma once
#include "valhall_ssa.h"
#include "valhall_isa_generated.h"
#include <cstdint>
#include <string>

namespace valhall::tables {
struct LoweringRuleSpec;
}

namespace valhall::compiler {

struct Pass1Context;

// Compositional Operand & Writeback Combinators (ValhallSSA level)
// Strict fetch order: Source Resolution -> Subword Swizzle/Widen -> abs -> neg -> not
// Strict store order: clamp -> pack -> narrow dest_mod (Bfi / Split64)
// Uniform Pass1 ABI: (Pass1Context&, uint32_t blk, const DecodedValhallInst&, const tables::LoweringRuleSpec*, ...)

SsaId fetch_raw_src_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx);
SsaId fetch_src_int32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx, bool is_signed = false);
SsaId fetch_src_f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx);
SsaId fetch_src_v2f16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx);
SsaId fetch_src_v2i16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx, bool is_signed);
SsaId fetch_src_v4i8(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx, bool is_signed);
SsaId fetch_src_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx, bool sign_extend_subword = false);

void store_dest_u32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val_u32);
void store_dest_f32(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val_f32);
void store_dest_v2f16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec2_f32);
void store_dest_v2i16(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec2_i32, bool is_signed);
void store_dest_v4i8(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId vec4_i32, bool is_signed);
void store_dest_u64(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, SsaId val_u64);

std::string get_src_swz(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, uint32_t src_idx);
std::string get_dest_mod(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule);

} // namespace valhall::compiler
