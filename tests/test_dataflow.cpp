#include "abstract_domain.h"
#include "liveness.h"
#include "pointer_analysis.h"
#include "transfer.h"
#include "valhall_ssa.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace valhall::dataflow;
using namespace valhall::compiler;
using namespace valhall::tables;

namespace {

SsaId emit_op(ValhallSsaModule& mod, uint32_t blk, SsaOp op, ValType ty,
              std::vector<SsaId> args = {}, uint32_t imm32 = 0, uint64_t imm64 = 0,
              std::string aux_str = "") {
    SsaNode n{};
    n.op = op;
    n.type = ty;
    n.block_id = blk;
    n.args = std::move(args);
    n.imm_u32 = imm32;
    n.imm_u64 = imm64;
    n.aux_str = std::move(aux_str);
    SsaId id = mod.add_node(std::move(n));
    if (op == SsaOp::Phi) {
        mod.blocks[blk].phis.push_back(id);
    } else {
        mod.blocks[blk].nodes.push_back(id);
    }
    return id;
}

void test_tracked_val_lattice_laws() {
    auto bot = TrackedValDomain::bottom();
    auto top = TrackedValDomain::top();
    auto c42 = TrackedVal::make_const(42);
    auto c99 = TrackedVal::make_const(99);
    auto b0 = TrackedVal::make_buffer_addr(4, 0, "bufA", 0, false, true);
    auto b16 = TrackedVal::make_buffer_addr(4, 0, "bufA", 16, false, true);
    auto b_unk = TrackedVal::make_buffer_addr(4, 0, "bufA", 0, false, false);
    auto b_other = TrackedVal::make_buffer_addr(4, 1, "bufB", 0, false, true);

    // Bottom / Top identities
    assert(TrackedValDomain::equals(TrackedValDomain::join(bot, c42), c42));
    assert(TrackedValDomain::equals(TrackedValDomain::join(c42, bot), c42));
    assert(TrackedValDomain::equals(TrackedValDomain::join(top, c42), top));
    assert(TrackedValDomain::equals(TrackedValDomain::meet(top, c42), c42));
    assert(TrackedValDomain::equals(TrackedValDomain::meet(bot, c42), bot));

    // Partial order & widening on same base pointer
    assert(TrackedValDomain::leq(bot, b0));
    assert(TrackedValDomain::leq(b0, b_unk));
    assert(TrackedValDomain::leq(b_unk, top));
    assert(!TrackedValDomain::leq(b_unk, b0));

    auto j_same_base = TrackedValDomain::join(b0, b16);
    assert(j_same_base.kind == TrackedVal::Kind::BufferAddr);
    assert(j_same_base.symbol_name == "bufA");
    assert(!j_same_base.offset_known);
    assert(TrackedValDomain::equals(j_same_base, b_unk));

    // Distinct constants / distinct bases join to Top and meet to Bottom
    assert(TrackedValDomain::join(c42, c99).is_top());
    assert(TrackedValDomain::join(b0, b_other).is_top());
    assert(TrackedValDomain::meet(c42, c99).is_bottom());
    assert(TrackedValDomain::meet(b0, b_unk).offset_known);
    assert(TrackedValDomain::meet(b0, b_unk).byte_offset == 0);

    std::cout << "[PASS] test_tracked_val_lattice_laws\n";
}

void test_ssa_grouped_dispatch_and_extensional_equalities() {
    ValhallSsaModule mod;
    SsaBlock b0{}; b0.id = 0; b0.is_terminated_by_end = true;
    mod.blocks.push_back(b0);

    SsaId c0 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 0);
    SsaId c1 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 1);
    SsaId call1 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, ~0ULL);
    SsaId srt = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 2); // SrtBase

    // Extensional equalities:
    SsaId cp = emit_op(mod, 0, SsaOp::Copy, ValType::U64, {srt});
    SsaId add0 = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {cp, c0});
    SsaId or0 = emit_op(mod, 0, SsaOp::BitOr64, ValType::U64, {add0, c0});
    SsaId xor0 = emit_op(mod, 0, SsaOp::BitXor64, ValType::U64, {or0, c0});
    SsaId and_all1 = emit_op(mod, 0, SsaOp::BitAnd64, ValType::U64, {xor0, call1});
    SsaId and_self = emit_op(mod, 0, SsaOp::BitAnd64, ValType::U64, {and_all1, and_all1});
    SsaId or_self = emit_op(mod, 0, SsaOp::BitOr64, ValType::U64, {and_self, and_self});
    SsaId shl0 = emit_op(mod, 0, SsaOp::Shl64, ValType::U64, {or_self, c0});
    SsaId mul1 = emit_op(mod, 0, SsaOp::IMul64, ValType::U64, {shl0, c1});
    SsaId sel_same = emit_op(mod, 0, SsaOp::Select, ValType::U64, {c1, mul1, mul1});

    // Zero identities:
    SsaId sub_self = emit_op(mod, 0, SsaOp::ISub64, ValType::U64, {srt, srt});
    SsaId xor_self = emit_op(mod, 0, SsaOp::BitXor64, ValType::U64, {srt, srt});
    SsaId and0 = emit_op(mod, 0, SsaOp::BitAnd64, ValType::U64, {srt, c0});

    PointerAnalysisEnv env;
    env.role = PointerAnalysisEnv::StageRole::Pilot;
    auto res = run_ssa_pointer_analysis(mod, env);
    assert(res.converged);

    for (SsaId id : {cp, add0, or0, xor0, and_all1, and_self, or_self, shl0, mul1, sel_same}) {
        auto v = res.final_state.val(id);
        assert(v.kind == TrackedVal::Kind::SrtBase);
        assert(v.offset_known && v.byte_offset == 0);
    }
    for (SsaId z : {sub_self, xor_self, and0}) {
        auto v = res.final_state.val(z);
        assert(v.kind == TrackedVal::Kind::ConstInt && v.const_val == 0);
    }

    std::cout << "[PASS] test_ssa_grouped_dispatch_and_extensional_equalities\n";
}

void test_ssa_diamond_cfg_phi_and_liveness() {
    // Diamond CFG: B0 -> {B1, B2} -> B3
    // B0: srt = SrtBase, off16 = 16, off32 = 32, cond = ConstBool(1)
    // B1: p1 = IAdd64(srt, off16) -> SrtBase+0x10
    // B2: p2 = IAdd64(srt, off32) -> SrtBase+0x20
    // B3: pm = Phi((B1, p1), (B2, p2)) -> SrtBase+? (same base, offset widened!)
    ValhallSsaModule mod;
    mod.blocks.resize(4);
    for (uint32_t i = 0; i < 4; ++i) mod.blocks[i].id = i;

    mod.blocks[0].fallthrough_block = 1;
    mod.blocks[0].branch_target_block = 2;
    mod.blocks[0].is_conditional_branch = true;
    mod.blocks[1].preds = {0};
    mod.blocks[1].fallthrough_block = 3;
    mod.blocks[2].preds = {0};
    mod.blocks[2].fallthrough_block = 3;
    mod.blocks[3].preds = {1, 2};
    mod.blocks[3].is_terminated_by_end = true;

    SsaId cond = emit_op(mod, 0, SsaOp::ConstBool, ValType::Bool, {}, 1);
    SsaId srt = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 2);
    SsaId off16 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 16);
    SsaId off32 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 32);
    mod.blocks[0].branch_cond = cond;

    SsaId p1 = emit_op(mod, 1, SsaOp::IAdd64, ValType::U64, {srt, off16});
    SsaId p2 = emit_op(mod, 2, SsaOp::IAdd64, ValType::U64, {srt, off32});

    SsaId pm = emit_op(mod, 3, SsaOp::Phi, ValType::U64);
    mod.get_mut(pm).phi_incomings = {{1, p1}, {2, p2}};
    SsaId zero = emit_op(mod, 3, SsaOp::ConstU64, ValType::U64, {}, 0, 0);
    SsaId use_pm = emit_op(mod, 3, SsaOp::IAdd64, ValType::U64, {pm, zero});

    // 1. Verify backwards SSA liveness with Phi edge-use semantics
    SsaLivenessResult live = run_ssa_liveness(mod);
    assert(live.converged);
    // p1 is live-out of B1 (for Phi in B3), NOT live-in of B3 (because Phi defines pm, not uses p1 inside B3!)
    assert(live.is_live_out(1, p1));
    assert(!live.is_live_out(2, p1));
    assert(live.is_live_out(2, p2));
    assert(!live.is_live_out(1, p2));
    assert(!live.is_live_in(3, p1));
    assert(!live.is_live_in(3, p2));
    // srt is live-out of B0 and live-in of both B1 and B2, and killed at p1 in B1 and p2 in B2
    assert(live.is_live_out(0, srt));
    assert(live.is_live_in(1, srt));
    assert(live.is_live_in(2, srt));
    assert(live.is_killed_at(p1, srt));
    assert(live.is_killed_at(p2, srt));
    assert(live.is_killed_at(use_pm, pm));

    // 2. Verify forward SSA pointer analysis joins p1 (+0x10) and p2 (+0x20) at Phi pm -> SrtBase+?
    PointerAnalysisEnv env;
    env.role = PointerAnalysisEnv::StageRole::Pilot;
    auto df = run_ssa_pointer_analysis(mod, env, &live);
    assert(df.converged);
    assert(df.final_state.val(p1).kind == TrackedVal::Kind::SrtBase && df.final_state.val(p1).byte_offset == 16);
    assert(df.final_state.val(p2).kind == TrackedVal::Kind::SrtBase && df.final_state.val(p2).byte_offset == 32);
    auto vpm = df.final_state.val(pm);
    assert(vpm.kind == TrackedVal::Kind::SrtBase);
    assert(!vpm.offset_known);
    assert(df.final_state.val(use_pm).kind == TrackedVal::Kind::SrtBase);

    std::cout << "[PASS] test_ssa_diamond_cfg_phi_and_liveness\n";
}

void test_ssa_pilot_srt_chasing_and_fau_store() {
    ValhallSsaModule mod;
    SsaBlock b0{}; b0.id = 0; b0.is_terminated_by_end = true;
    mod.blocks.push_back(b0);

    SsaId tok = emit_op(mod, 0, SsaOp::Nop, ValType::Token);
    SsaId fau_base = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 3);
    SsaId srt_base = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 2);
    SsaId c64 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 0x40); // rack 4
    SsaId srt_r4 = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {srt_base, c64});
    SsaId ld_rack = emit_op(mod, 0, SsaOp::PsbLoad, ValType::TupleValToken, {srt_r4, tok}, 8);
    SsaId rack_ptr = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U64, {ld_rack}, 0);

    // Load buffer VA at desc 0, field +0x08
    SsaId c8 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 8);
    SsaId desc0_va_addr = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {rack_ptr, c8});
    SsaId ld_buf = emit_op(mod, 0, SsaOp::PsbLoad, ValType::TupleValToken, {desc0_va_addr, tok}, 8);
    SsaId buf_va = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U64, {ld_buf}, 0);

    // Load buffer size at desc 0, field +0x04
    SsaId c4 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 4);
    SsaId desc0_sz_addr = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {rack_ptr, c4});
    SsaId ld_sz = emit_op(mod, 0, SsaOp::PsbLoad, ValType::TupleValToken, {desc0_sz_addr, tok}, 4);
    SsaId buf_sz = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U32, {ld_sz}, 0);

    // Store buffer VA into FAU_0 + 0x00 and buffer size into FAU_0 + 0x08
    emit_op(mod, 0, SsaOp::PsbStore, ValType::Token, {fau_base, buf_va, tok}, 8);
    SsaId fau_8 = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {fau_base, c8});
    emit_op(mod, 0, SsaOp::PsbStore, ValType::Token, {fau_8, buf_sz, tok}, 4);

    PointerAnalysisEnv env;
    env.role = PointerAnalysisEnv::StageRole::Pilot;
    env.lookup_srt_symbol = [](uint8_t rack, uint32_t desc) -> std::string {
        if (rack == 4 && desc == 0) return "my_ssbo";
        return "";
    };

    auto df = run_ssa_pointer_analysis(mod, env);
    assert(df.converged);
    assert(df.final_state.val(rack_ptr).kind == TrackedVal::Kind::SrtRackPtr);
    assert(df.final_state.val(rack_ptr).rack == 4);
    assert(df.final_state.val(buf_va).kind == TrackedVal::Kind::BufferAddr);
    assert(df.final_state.val(buf_va).symbol_name == "my_ssbo");
    assert(df.final_state.val(buf_sz).kind == TrackedVal::Kind::BufferSize);
    assert(df.final_state.val(buf_sz).symbol_name == "my_ssbo");

    assert(df.pilot_fau_writes.count(0x00) == 1);
    assert(df.pilot_fau_writes.at(0x00).value.kind == TrackedVal::Kind::BufferAddr);
    assert(df.pilot_fau_writes.at(0x00).value.symbol_name == "my_ssbo");
    assert(df.pilot_fau_writes.at(0x00).byte_size == 8);

    assert(df.pilot_fau_writes.count(0x08) == 1);
    assert(df.pilot_fau_writes.at(0x08).value.kind == TrackedVal::Kind::BufferSize);
    assert(df.pilot_fau_writes.at(0x08).byte_size == 4);

    std::cout << "[PASS] test_ssa_pilot_srt_chasing_and_fau_store\n";
}

void test_ssa_pack64_split64_and_fau_seed() {
    ValhallSsaModule mod;
    SsaBlock b0{}; b0.id = 0; b0.is_terminated_by_end = true;
    mod.blocks.push_back(b0);

    // Main compute shader loads 64-bit buffer VA from FAU u0 (words 0 and 1) and packs into 64-bit ptr
    SsaId w0 = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 0);
    SsaId w1 = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 1);
    SsaId ptr = emit_op(mod, 0, SsaOp::Pack64, ValType::U64, {w0, w1});
    SsaId off = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 16);
    SsaId ptr16 = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {ptr, off});
    SsaId sp = emit_op(mod, 0, SsaOp::Split64, ValType::TupleU32x2, {ptr16});
    SsaId lo = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U32, {sp}, 0);
    SsaId hi = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U32, {sp}, 1);

    PointerAnalysisEnv env;
    env.role = PointerAnalysisEnv::StageRole::MainCompute;
    env.initial_fau[0x00] = TrackedVal::make_buffer_addr(4, 0, "input_buf", 0, false, true);

    auto df = run_ssa_pointer_analysis(mod, env);
    assert(df.converged);
    auto vp = df.final_state.val(ptr16);
    assert(vp.kind == TrackedVal::Kind::BufferAddr);
    assert(vp.symbol_name == "input_buf");
    assert(vp.offset_known && vp.byte_offset == 16);
    assert(!df.final_state.val(lo).is_hi_half);
    assert(df.final_state.val(hi).is_hi_half);

    std::cout << "[PASS] test_ssa_pack64_split64_and_fau_seed\n";
}

}  // namespace

int main() {
    std::cout << "=== [Test] SSA-Level Valhall Dataflow, Liveness & Pointer Analysis ===\n";
    test_tracked_val_lattice_laws();
    test_ssa_grouped_dispatch_and_extensional_equalities();
    test_ssa_diamond_cfg_phi_and_liveness();
    test_ssa_pilot_srt_chasing_and_fau_store();
    test_ssa_pack64_split64_and_fau_seed();
    std::cout << "=== ALL SSA DATAFLOW TESTS PASSED ===\n";
    return 0;
}
