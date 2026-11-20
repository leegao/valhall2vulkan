#include "abstract_domain.h"
#include "domains.h"
#include "liveness.h"
#include "memory_domain.h"
#include "pointer_analysis.h"
#include "tracked_val.h"
#include "transfer.h"
#include "valhall_ssa.h"
#include "worklist.h"

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

// Minimal domain omitting `equals()` to verify auto-equals via `leq(a, b) && leq(b, a)`
struct SignDomain {
    enum class Value : uint8_t { Bottom = 0, Neg, Zero, Pos, Top };
    static Value bottom() { return Value::Bottom; }
    static Value top() { return Value::Top; }
    static bool leq(const Value& a, const Value& b) {
        return a == Value::Bottom || b == Value::Top || a == b;
    }
    static Value join(const Value& a, const Value& b) {
        if (a == Value::Bottom) return b;
        if (b == Value::Bottom) return a;
        return (a == b) ? a : Value::Top;
    }
    static Value meet(const Value& a, const Value& b) {
        if (a == Value::Top) return b;
        if (b == Value::Top) return a;
        return (a == b) ? a : Value::Bottom;
    }
};

void test_domain_combinators_and_memory_disjointness() {
    static_assert(ValidateLatticeDomain<SignDomain>::valid, "");
    assert(domain_equals<SignDomain>(SignDomain::Value::Pos, SignDomain::Value::Pos));
    assert(!domain_equals<SignDomain>(SignDomain::Value::Pos, SignDomain::Value::Neg));

    // VectorDomain power-of-2 growth + join/meet
    VectorDomain<SignDomain>::Value va, vb;
    va.set(5, SignDomain::Value::Pos);
    assert(va.elems.size() == 8); // next_power_of_two(6) == 8
    vb.set(5, SignDomain::Value::Neg);
    vb.set(2, SignDomain::Value::Zero);
    auto vj = VectorDomain<SignDomain>::join(va, vb);
    assert(vj.get(5) == SignDomain::Value::Top);
    assert(vj.get(2) == SignDomain::Value::Zero);
    assert(vj.get(99) == SignDomain::Value::Bottom);

    // MapDomain
    MapDomain<uint32_t, SignDomain>::Value ma, mb;
    ma.set(0x00, SignDomain::Value::Pos);
    mb.set(0x00, SignDomain::Value::Pos);
    mb.set(0x08, SignDomain::Value::Neg);
    auto mj = MapDomain<uint32_t, SignDomain>::join(ma, mb);
    auto mm = MapDomain<uint32_t, SignDomain>::meet(ma, mb);
    assert(mj.get(0x00) == SignDomain::Value::Pos && mj.get(0x08) == SignDomain::Value::Neg);
    assert(mm.get(0x00) == SignDomain::Value::Pos && mm.get(0x08) == SignDomain::Value::Bottom);

    // TupleDomain<SignDomain, BitsetDomain>
    using ProdDom = TupleDomain<SignDomain, BitsetDomain>;
    BitsetValue b1(16), b2(16);
    b1.set(1); b2.set(3);
    auto p1 = ProdDom::Value{SignDomain::Value::Pos, b1};
    auto p2 = ProdDom::Value{SignDomain::Value::Pos, b2};
    auto pj = ProdDom::join(p1, p2);
    assert(std::get<0>(pj) == SignDomain::Value::Pos);
    assert(std::get<1>(pj).test(1) && std::get<1>(pj).test(3));
    assert(ProdDom::leq(p1, pj) && ProdDom::leq(p2, pj));

    // FiniteSetDomain<TrackedValDomain, 4>: implicit singleton lift + up to 4 distinct values, then Top
    TrackedValSet s1 = TrackedVal::make_const(10); // implicit converting ctor!
    auto s2 = TrackedValSetDomain::lift_value(TrackedVal::make_const(20));
    auto s12 = TrackedValSetDomain::join(s1, s2);
    assert(s12.size() == 2 && !s12.is_top());
    TrackedVal single;
    assert(!s12.try_as_singleton(&single));
    assert(s1.try_as_singleton(&single) && single.const_val == 10);

    // Lift unary (+1) across {10, 20} -> {11, 21}
    auto s12_plus1 = TrackedValSetDomain::lift_unary(s12, [](const TrackedVal& v) {
        return TrackedVal::make_const(v.const_val + 1);
    });
    assert(s12_plus1.size() == 2);
    auto s3 = TrackedValSetDomain::lift_value(TrackedVal::make_const(30));
    auto s4 = TrackedValSetDomain::lift_value(TrackedVal::make_const(40));
    auto s5 = TrackedValSetDomain::lift_value(TrackedVal::make_const(50));
    auto s1234 = TrackedValSetDomain::join(TrackedValSetDomain::join(s12, s3), s4);
    assert(s1234.size() == 4 && !s1234.is_top());
    auto s12345 = TrackedValSetDomain::join(s1234, s5);
    assert(s12345.is_top()); // > 4 elements widens to Top!

    // SymbolicMemoryLoc disjointness
    auto srt0 = SymbolicMemoryLoc::make_srt(0, 0);
    auto srt4 = SymbolicMemoryLoc::make_srt(4, 0);
    auto buf0 = SymbolicMemoryLoc::make_buffer(4, 0, 0, true, "buf0");
    auto buf1 = SymbolicMemoryLoc::make_buffer(4, 1, 0, true, "buf1");
    auto buf0_off16 = SymbolicMemoryLoc::make_buffer(4, 0, 16, true, "buf0");
    auto buf0_unk = SymbolicMemoryLoc::make_buffer(4, 0, 0, false, "buf0");
    auto tls0 = SymbolicMemoryLoc::make_tls(0);
    assert(!srt0.may_alias(srt4));
    assert(!srt4.may_alias(buf0));
    assert(!buf0.may_alias(buf1));
    assert(!buf0.may_alias(buf0_off16));
    assert(buf0.may_alias(buf0_unk));
    assert(!buf0.may_alias(tls0));
    assert(buf0.with_added_offset(16) == buf0_off16);
    assert(buf0.with_unknown_offset() == buf0_unk);
    assert(buf0_off16.with_masked_offset(0xFF) == buf0_off16);

    // SymbolicMemoryLoc::deref()
    auto srt_root_r4 = SymbolicMemoryLoc::make_srt(-1, 0x40);
    auto d_rack = srt_root_r4.deref(8);
    assert(d_rack.kind == MemoryDerefResult::Kind::SrtRackPtr && d_rack.table_id == 4);
    auto d_va = SymbolicMemoryLoc::make_srt(4, 0x08).deref(8);
    assert(d_va.kind == MemoryDerefResult::Kind::BufferAddr && d_va.table_id == 4 && d_va.entry_id == 0);
    auto d_sz = SymbolicMemoryLoc::make_srt(4, 0x04).deref(4);
    assert(d_sz.kind == MemoryDerefResult::Kind::BufferSize && d_sz.table_id == 4 && d_sz.entry_id == 0);

    // TrackedValSetOps & SidebandValue
    TrackedValSet sb = TrackedVal::make_sideband(5);
    TrackedValSet c0 = TrackedVal::make_const(0);
    TrackedValSet c4 = TrackedVal::make_const(4);
    TrackedValSet c20 = TrackedVal::make_const(20);
    assert(collapse_tracked_val_set(TrackedValSetOps::add(sb, c0)).kind == TrackedVal::Kind::SidebandValue);
    assert(TrackedValSetOps::add(sb, c4).is_top());
    assert(collapse_tracked_val_set(TrackedValSetOps::div(c20, c4)).const_val == 5);
    assert(collapse_tracked_val_set(TrackedValSetOps::mul(c4, c20)).const_val == 80);

    std::cout << "[PASS] test_domain_combinators_and_memory_disjointness\n";
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

    assert(TrackedValDomain::equals(TrackedValDomain::join(bot, c42), c42));
    assert(TrackedValDomain::equals(TrackedValDomain::join(c42, bot), c42));
    assert(TrackedValDomain::equals(TrackedValDomain::join(top, c42), top));
    assert(TrackedValDomain::equals(TrackedValDomain::meet(top, c42), c42));
    assert(TrackedValDomain::equals(TrackedValDomain::meet(bot, c42), bot));

    assert(TrackedValDomain::leq(bot, b0));
    assert(TrackedValDomain::leq(b0, b_unk));
    assert(TrackedValDomain::leq(b_unk, top));
    assert(!TrackedValDomain::leq(b_unk, b0));

    auto j_same_base = TrackedValDomain::join(b0, b16);
    assert(j_same_base.kind == TrackedVal::Kind::BufferAddr);
    assert(j_same_base.symbol_name == "bufA");
    assert(!j_same_base.offset_known);
    assert(TrackedValDomain::equals(j_same_base, b_unk));

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
    SsaId srt = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 2);

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

    SsaId sub_self = emit_op(mod, 0, SsaOp::ISub64, ValType::U64, {srt, srt});
    SsaId xor_self = emit_op(mod, 0, SsaOp::BitXor64, ValType::U64, {srt, srt});
    SsaId and0 = emit_op(mod, 0, SsaOp::BitAnd64, ValType::U64, {srt, c0});

    PointerAnalysisEnv env;
    env.role = PointerAnalysisEnv::StageRole::Pilot;
    auto res = run_ssa_pointer_analysis(mod, env);
    assert(res.converged);

    for (SsaId id : {cp, add0, or0, xor0, and_all1, and_self, or_self, shl0, mul1, sel_same}) {
        auto v = res.val(id);
        assert(v.kind == TrackedVal::Kind::SrtBase);
        assert(v.offset_known && v.byte_offset == 0);
    }
    for (SsaId z : {sub_self, xor_self, and0}) {
        auto v = res.val(z);
        assert(v.kind == TrackedVal::Kind::ConstInt && v.const_val == 0);
    }

    std::cout << "[PASS] test_ssa_grouped_dispatch_and_extensional_equalities\n";
}

void test_ssa_diamond_cfg_phi_finiteset_and_liveness() {
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

    SsaId tok = emit_op(mod, 0, SsaOp::Nop, ValType::Token);
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
    // Dereference non-singleton set {SRT_0+0x10, SRT_0+0x20} -> must widen to Top!
    SsaId ld_non_single = emit_op(mod, 3, SsaOp::PsbLoad, ValType::TupleValToken, {use_pm, tok}, 8);

    // 1. Verify backwards SSA liveness (BitsetDomain + run_cfg_worklist(Backward))
    SsaLivenessResult live = run_ssa_liveness(mod);
    assert(live.converged);
    assert(live.is_live_out(1, p1));
    assert(!live.is_live_out(2, p1));
    assert(live.is_live_out(2, p2));
    assert(!live.is_live_out(1, p2));
    assert(!live.is_live_in(3, p1));
    assert(!live.is_live_in(3, p2));
    assert(live.is_live_out(0, srt));
    assert(live.is_live_in(1, srt));
    assert(live.is_live_in(2, srt));
    assert(live.is_killed_at(p1, srt));
    assert(live.is_killed_at(p2, srt));
    assert(live.is_killed_at(use_pm, pm));

    // 2. Verify FiniteSetDomain preserves exact 2-element set {SRT_0+0x10, SRT_0+0x20} at Phi pm,
    //    collapses to SrtBase+? via df.val(pm), and widens non-singleton PsbLoad to Top!
    PointerAnalysisEnv env;
    env.role = PointerAnalysisEnv::StageRole::Pilot;
    auto df = run_ssa_pointer_analysis(mod, env, &live);
    assert(df.converged);
    assert(df.val_set(pm).size() == 2);
    auto vpm = df.val(pm);
    assert(vpm.kind == TrackedVal::Kind::SrtBase && !vpm.offset_known);
    assert(df.val_set(ld_non_single).is_top());

    std::cout << "[PASS] test_ssa_diamond_cfg_phi_finiteset_and_liveness\n";
}

void test_ssa_pilot_srt_chasing_and_fau_store() {
    ValhallSsaModule mod;
    SsaBlock b0{}; b0.id = 0; b0.is_terminated_by_end = true;
    mod.blocks.push_back(b0);

    SsaId tok = emit_op(mod, 0, SsaOp::Nop, ValType::Token);
    SsaId fau_base = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 3);
    SsaId srt_base = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 2);
    SsaId c64 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 0x40);
    SsaId srt_r4 = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {srt_base, c64});
    SsaId ld_rack = emit_op(mod, 0, SsaOp::PsbLoad, ValType::TupleValToken, {srt_r4, tok}, 8);
    SsaId rack_ptr = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U64, {ld_rack}, 0);

    SsaId c8 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 8);
    SsaId desc0_va_addr = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {rack_ptr, c8});
    SsaId ld_buf = emit_op(mod, 0, SsaOp::PsbLoad, ValType::TupleValToken, {desc0_va_addr, tok}, 8);
    SsaId buf_va = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U64, {ld_buf}, 0);

    SsaId c4 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 4);
    SsaId desc0_sz_addr = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {rack_ptr, c4});
    SsaId ld_sz = emit_op(mod, 0, SsaOp::PsbLoad, ValType::TupleValToken, {desc0_sz_addr, tok}, 4);
    SsaId buf_sz = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U32, {ld_sz}, 0);

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
    assert(df.val(rack_ptr).kind == TrackedVal::Kind::SrtRackPtr && df.val(rack_ptr).rack == 4);
    assert(df.val(buf_va).kind == TrackedVal::Kind::BufferAddr && df.val(buf_va).symbol_name == "my_ssbo");
    assert(df.val(buf_sz).kind == TrackedVal::Kind::BufferSize && df.val(buf_sz).symbol_name == "my_ssbo");
    assert(df.pilot_fau_writes.count(0x00) == 1 && df.pilot_fau_writes.at(0x00).value.symbol_name == "my_ssbo");
    assert(df.pilot_fau_writes.count(0x08) == 1 && df.pilot_fau_writes.at(0x08).value.kind == TrackedVal::Kind::BufferSize);

    std::cout << "[PASS] test_ssa_pilot_srt_chasing_and_fau_store\n";
}

void test_ssa_pack64_split64_and_fau_seed() {
    ValhallSsaModule mod;
    SsaBlock b0{}; b0.id = 0; b0.is_terminated_by_end = true;
    mod.blocks.push_back(b0);

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
    auto vp = df.val(ptr16);
    assert(vp.kind == TrackedVal::Kind::BufferAddr && vp.symbol_name == "input_buf");
    assert(vp.offset_known && vp.byte_offset == 16);
    assert(!df.val(lo).is_hi_half && df.val(hi).is_hi_half);

    std::cout << "[PASS] test_ssa_pack64_split64_and_fau_seed\n";
}

}  // namespace

int main() {
    std::cout << "=== [Test] Valhall SSA Dataflow Combinators, Worklist, Liveness & Pointer Analysis ===\n";
    test_domain_combinators_and_memory_disjointness();
    test_tracked_val_lattice_laws();
    test_ssa_grouped_dispatch_and_extensional_equalities();
    test_ssa_diamond_cfg_phi_finiteset_and_liveness();
    test_ssa_pilot_srt_chasing_and_fau_store();
    test_ssa_pack64_split64_and_fau_seed();
    std::cout << "=== ALL SSA DATAFLOW TESTS PASSED ===\n";
    return 0;
}
