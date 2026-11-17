#pragma once
#include "valhall_ssa.h"
#include "valhall_isa_generated.h"
#include <cstdint>
#include <string>

namespace valhall::compiler {

// Compositional Operand & Writeback Combinators (ValhallSSA level)
// Strict fetch order: Source Resolution -> Subword Swizzle/Widen -> abs -> neg -> not
// Strict store order: clamp -> pack -> narrow dest_mod (Bfi / Split64)

SsaId fetch_raw_src_u32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx);
SsaId fetch_src_int32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool is_signed = false);
SsaId fetch_src_f32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx);
SsaId fetch_src_v2f16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx);
SsaId fetch_src_v2i16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool is_signed);
SsaId fetch_src_v4i8(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool is_signed);
SsaId fetch_src_u64(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, size_t src_idx, bool sign_extend_subword = false);

void store_dest_u32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val_u32);
void store_dest_f32(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val_f32);
void store_dest_v2f16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec2_f32);
void store_dest_v2i16(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec2_i32, bool is_signed);
void store_dest_v4i8(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId vec4_i32, bool is_signed);
void store_dest_u64(SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, SsaId val_u64);

std::string get_src_swz(const DecodedValhallInst& ins, size_t src_idx);
std::string get_dest_mod(const DecodedValhallInst& ins);

} // namespace valhall::compiler
