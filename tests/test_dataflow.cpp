#include "abstract_domain.h"
#include "domains.h"
#include "liveness.h"
#include "memory_domain.h"
#include "pointer_analysis.h"
#include "ssa_defau_devirt.h"
#include "tracked_val.h"
#include "transfer.h"
#include "valhall_ssa.h"
#include "worklist.h"

#include <cstdlib>

#define CHECK(cond)                                                                           \
    do {                                                                                      \
        if (!(cond)) {                                                                        \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n";      \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)
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
    static_assert(ValidateLatticeDomain<ConstIntDomain>::valid, "");
    static_assert(ValidateLatticeDomain<PointerDomain>::valid, "");
    static_assert(ValidateLatticeDomain<SymbolValueDomain>::valid, "");
    static_assert(ValidateLatticeDomain<ShaderAtomDomain>::valid, "");
    static_assert(ValidateLatticeDomain<TrackedValUnionDomain>::valid, "");
    static_assert(ValidateLatticeDomain<TrackedValDomain>::valid, "");
    static_assert(ValidateLatticeDomain<TrackedValSetDomain>::valid, "");

    // SFINAE verification: ops only exist on wrappers when base domain implements them
    static_assert(!detail::has_domain_iadd<SignDomain>::value, "");
    static_assert(!detail::has_domain_iadd<VectorDomain<SignDomain>>::value, "");
    static_assert(!detail::has_domain_iadd<MapDomain<uint32_t, SignDomain>>::value, "");
    static_assert(!detail::has_domain_iadd<FiniteSetDomain<SignDomain, 4>>::value, "");
    static_assert(!detail::has_domain_iadd<LaneDomainWrapper<SignDomain>>::value, "");
    static_assert(!detail::has_domain_iadd<UnionDomain<SignDomain, BitsetDomain>>::value, "");
    static_assert(!detail::has_domain_iadd<TupleDomain<SignDomain, BitsetDomain>>::value, "");

    static_assert(detail::has_domain_iadd<TrackedValUnionDomain>::value, "");
    static_assert(detail::has_domain_iadd<TrackedValDomain>::value, "");
    static_assert(detail::has_domain_iadd<TrackedValSetDomain>::value, "");
    static_assert(detail::has_domain_add_v<TrackedValSetDomain, const SrtSymbolLookupFn&>, "");
    static_assert(detail::has_domain_sub_v<TrackedValSetDomain, const SrtSymbolLookupFn&>, "");
    static_assert(detail::has_domain_mul_v<TrackedValSetDomain>, "");
    static_assert(detail::has_domain_div_v<TrackedValSetDomain>, "");
    static_assert(detail::has_domain_bit_and_v<TrackedValSetDomain>, "");
    static_assert(detail::has_domain_bit_and_v<TrackedValSetDomain, bool>, "");
    static_assert(detail::has_domain_pack64<TrackedValDomain>::value, "");
    static_assert(detail::has_domain_pack64<TrackedValSetDomain>::value, "");
    static_assert(detail::has_domain_tuple_extract<TrackedValSetDomain>::value, "");
    static_assert(ValidateLatticeDomain<IntegralDomain<int32_t>>::valid, "");
    static_assert(ValidateLatticeDomain<FloatDomain<double>>::valid, "");
    static_assert(ValidateLatticeDomain<FloatDomain<float>>::valid, "");
    static_assert(!detail::has_domain_convert<SignDomain>::value, "");
    static_assert(!detail::has_domain_convert<FiniteSetDomain<SignDomain, 4>>::value, "");
    static_assert(detail::has_domain_convert<IntegralDomain<int64_t>>::value, "");
    static_assert(detail::has_domain_convert<FloatDomain<double>>::value, "");
    static_assert(detail::has_domain_convert<TrackedValUnionDomain>::value, "");
    static_assert(detail::has_domain_convert<TrackedValDomain>::value, "");
    static_assert(detail::has_domain_convert<TrackedValSetDomain>::value, "");

    CHECK(domain_equals<SignDomain>(SignDomain::Value::Pos, SignDomain::Value::Pos));
    CHECK(!domain_equals<SignDomain>(SignDomain::Value::Pos, SignDomain::Value::Neg));

    // VectorDomain power-of-2 growth + join/meet
    VectorDomain<SignDomain>::Value va, vb;
    va.set(5, SignDomain::Value::Pos);
    CHECK(va.elems.size() == 8);
    vb.set(5, SignDomain::Value::Neg);
    vb.set(2, SignDomain::Value::Zero);
    auto vj = VectorDomain<SignDomain>::join(va, vb);
    CHECK(vj.get(5) == SignDomain::Value::Top);
    CHECK(vj.get(2) == SignDomain::Value::Zero);
    CHECK(vj.get(99) == SignDomain::Value::Bottom);

    // MapDomain
    MapDomain<uint32_t, SignDomain>::Value ma, mb;
    ma.set(0x00, SignDomain::Value::Pos);
    mb.set(0x00, SignDomain::Value::Pos);
    mb.set(0x08, SignDomain::Value::Neg);
    auto mj = MapDomain<uint32_t, SignDomain>::join(ma, mb);
    auto mm = MapDomain<uint32_t, SignDomain>::meet(ma, mb);
    CHECK(mj.get(0x00) == SignDomain::Value::Pos && mj.get(0x08) == SignDomain::Value::Neg);
    CHECK(mm.get(0x00) == SignDomain::Value::Pos && mm.get(0x08) == SignDomain::Value::Bottom);

    // TupleDomain<SignDomain, BitsetDomain>
    using ProdDom = TupleDomain<SignDomain, BitsetDomain>;
    BitsetValue b1(16), b2(16);
    b1.set(1); b2.set(3);
    auto p1 = ProdDom::Value{SignDomain::Value::Pos, b1};
    auto p2 = ProdDom::Value{SignDomain::Value::Pos, b2};
    auto pj = ProdDom::join(p1, p2);
    CHECK(std::get<0>(pj) == SignDomain::Value::Pos);
    CHECK(std::get<1>(pj).test(1) && std::get<1>(pj).test(3));
    CHECK(ProdDom::leq(p1, pj) && ProdDom::leq(p2, pj));

    // FiniteSetDomain<TrackedValDomain, 4>: implicit singleton lift + up to 4 distinct values, then Top
    TrackedValSet s1 = TrackedValDomain::make_const(10);
    auto s2 = TrackedValSetDomain::lift_value(TrackedValDomain::make_const(20));
    auto s12 = TrackedValSetDomain::join(s1, s2);
    CHECK(s12.size() == 2 && !s12.is_top());
    TrackedVal single;
    CHECK(!s12.try_as_singleton(&single));
    CHECK(s1.try_as_singleton(&single));
    CHECK(single.inner.get_if<ConstIntDomain>()->value == 10);

    // Lift unary (+1) across {10, 20} -> {11, 21}
    auto s12_plus1 = TrackedValSetDomain::lift_unary(s12, [](const TrackedVal& v) {
        return TrackedValDomain::make_const(v.inner.get_if<ConstIntDomain>()->value + 1);
    });
    CHECK(s12_plus1.size() == 2);
    auto s3 = TrackedValSetDomain::lift_value(TrackedValDomain::make_const(30));
    auto s4 = TrackedValSetDomain::lift_value(TrackedValDomain::make_const(40));
    auto s5 = TrackedValSetDomain::lift_value(TrackedValDomain::make_const(50));
    auto s1234 = TrackedValSetDomain::join(TrackedValSetDomain::join(s12, s3), s4);
    CHECK(s1234.size() == 4 && !s1234.is_top());
    auto s12345 = TrackedValSetDomain::join(s1234, s5);
    CHECK(s12345.is_top());

    // SymbolicMemoryLoc disjointness
    auto srt0 = SymbolicMemoryLoc::make_srt(0, 0);
    auto srt4 = SymbolicMemoryLoc::make_srt(4, 0);
    auto buf0 = SymbolicMemoryLoc::make_buffer(4, 0, 0, true, "buf0");
    auto buf1 = SymbolicMemoryLoc::make_buffer(4, 1, 0, true, "buf1");
    auto buf0_off16 = SymbolicMemoryLoc::make_buffer(4, 0, 16, true, "buf0");
    auto buf0_unk = SymbolicMemoryLoc::make_buffer(4, 0, 0, false, "buf0");
    auto tls0 = SymbolicMemoryLoc::make_tls(0);
    CHECK(!srt0.may_alias(srt4));
    CHECK(!srt4.may_alias(buf0));
    CHECK(!buf0.may_alias(buf1));
    CHECK(!buf0.may_alias(buf0_off16));
    CHECK(buf0.may_alias(buf0_unk));
    CHECK(!buf0.may_alias(tls0));
    CHECK(buf0.with_added_offset(16) == buf0_off16);
    CHECK(buf0.with_unknown_offset() == buf0_unk);
    CHECK(buf0_off16.with_masked_offset(0xFF) == buf0_off16);

    // SymbolicMemoryLoc::deref()
    auto srt_root_r4 = SymbolicMemoryLoc::make_srt(-1, 0x40);
    auto d_rack = srt_root_r4.deref(8);
    CHECK(d_rack.kind == MemoryDerefResult::Kind::SrtRackPtr && d_rack.table_id == 4);
    auto d_va = SymbolicMemoryLoc::make_srt(4, 0x08).deref(8);
    CHECK(d_va.kind == MemoryDerefResult::Kind::BufferAddr && d_va.table_id == 4 && d_va.entry_id == 0);
    auto d_sz = SymbolicMemoryLoc::make_srt(4, 0x04).deref(4);
    CHECK(d_sz.kind == MemoryDerefResult::Kind::BufferSize && d_sz.table_id == 4 && d_sz.entry_id == 0);

    // Auto-lifted FiniteSetDomain arithmetic ops (replacing old TrackedValSetOps)
    TrackedValSet sb = TrackedValDomain::make_sideband(5);
    TrackedValSet c0 = TrackedValDomain::make_const(0);
    TrackedValSet c4 = TrackedValDomain::make_const(4);
    TrackedValSet c20 = TrackedValDomain::make_const(20);
    auto sb_add0 = collapse_tracked_val_set(TrackedValSetDomain::add(sb, c0));
    CHECK(sb_add0.inner.get_if<ShaderAtomDomain>()->value.kind == ShaderAtomDomain::Kind::SidebandValue);
    CHECK(format_tracked_val(sb_add0) == "sideband.local_size_xyz");
    CHECK(TrackedValSetDomain::add(sb, c4).is_top());
    CHECK(TrackedValDomain::equals(collapse_tracked_val_set(TrackedValSetDomain::div(c20, c4)), TrackedValDomain::make_const(5)));
    CHECK(TrackedValDomain::equals(collapse_tracked_val_set(TrackedValSetDomain::mul(c4, c20)), TrackedValDomain::make_const(80)));

    std::cout << "[PASS] test_domain_combinators_and_memory_disjointness\n";
}

void test_tracked_val_lattice_laws_and_union_dispatch() {
    auto bot = TrackedValDomain::bottom();
    auto top = TrackedValDomain::top();
    auto c42 = TrackedValDomain::make_const(42);
    auto c99 = TrackedValDomain::make_const(99);
    auto b0 = TrackedValDomain::make_buffer_addr(4, 0, "bufA", 0, false, true);
    auto b16 = TrackedValDomain::make_buffer_addr(4, 0, "bufA", 16, false, true);
    auto b_unk = TrackedValDomain::make_buffer_addr(4, 0, "bufA", 0, false, false);
    auto b_other = TrackedValDomain::make_buffer_addr(4, 1, "bufB", 0, false, true);
    auto abi_tid = TrackedValDomain::make_abi("gl_GlobalInvocationID.x");
    auto sym = TrackedValDomain::make_symbol_value(4, 0, "ubo.scale", false);

    // Core lattice laws via LaneDomainWrapper<UnionDomain<...>>
    CHECK(TrackedValDomain::equals(TrackedValDomain::join(bot, c42), c42));
    CHECK(TrackedValDomain::equals(TrackedValDomain::join(c42, bot), c42));
    CHECK(TrackedValDomain::equals(TrackedValDomain::join(top, c42), top));
    CHECK(TrackedValDomain::equals(TrackedValDomain::meet(top, c42), c42));
    CHECK(TrackedValDomain::equals(TrackedValDomain::meet(bot, c42), bot));

    CHECK(TrackedValDomain::leq(bot, b0));
    CHECK(TrackedValDomain::leq(b0, b_unk));
    CHECK(TrackedValDomain::leq(b_unk, top));
    CHECK(!TrackedValDomain::leq(b_unk, b0));

    auto j_same_base = TrackedValDomain::join(b0, b16);
    const auto* j_ptr = j_same_base.inner.get_if<PointerDomain>();
    CHECK(j_ptr && j_ptr->kind == PointerVal::Kind::BufferAddr);
    CHECK(j_ptr->symbol_name == "bufA" && !j_ptr->offset_known);
    CHECK(TrackedValDomain::equals(j_same_base, b_unk));

    CHECK(TrackedValDomain::join(c42, c99).is_top());
    CHECK(TrackedValDomain::join(b0, b_other).is_top());
    CHECK(TrackedValDomain::join(c42, b0).is_top());
    CHECK(TrackedValDomain::meet(c42, c99).is_bottom());
    CHECK(TrackedValDomain::meet(c42, b0).is_bottom());
    auto m_ptr = TrackedValDomain::meet(b0, b_unk);
    CHECK(m_ptr.inner.get_if<PointerDomain>()->offset_known);
    CHECK(m_ptr.inner.get_if<PointerDomain>()->byte_offset == 0);

    // Cross-domain arithmetic dispatch in UnionDomain:
    // Ptr + ConstInt and ConstInt + Ptr (commutative)
    auto p_plus_16 = TrackedValDomain::add(b0, TrackedValDomain::make_const(16));
    CHECK(TrackedValDomain::equals(p_plus_16, b16));
    auto c16_plus_p = TrackedValDomain::add(TrackedValDomain::make_const(16), b0);
    CHECK(TrackedValDomain::equals(c16_plus_p, b16));

    // Ptr - ConstInt works, but ConstInt - Ptr is disallowed -> Top
    auto p_minus_16 = TrackedValDomain::sub(b16, TrackedValDomain::make_const(16));
    CHECK(TrackedValDomain::equals(p_minus_16, b0));
    CHECK(TrackedValDomain::sub(TrackedValDomain::make_const(16), b0).is_top());

    // Same-base Ptr - Ptr -> ConstInt(delta); different-base Ptr - Ptr -> Top
    auto ptr_diff = TrackedValDomain::sub(b16, b0);
    CHECK(ptr_diff.inner.get_if<ConstIntDomain>()->value == 16);
    CHECK(TrackedValDomain::sub(b16, b_other).is_top());

    // Ptr + dynamic non-pointer (ABI / Top / Symbol) widens offset to +? while keeping base identity
    CHECK(TrackedValDomain::equals(TrackedValDomain::add(b0, abi_tid), b_unk));
    CHECK(TrackedValDomain::equals(TrackedValDomain::add(abi_tid, b0), b_unk));
    CHECK(TrackedValDomain::equals(TrackedValDomain::add(b0, top), b_unk));
    CHECK(TrackedValDomain::equals(TrackedValDomain::add(sym, b0), b_unk));
    CHECK(TrackedValDomain::equals(TrackedValDomain::sub(b0, abi_tid), b_unk));

    // Ptr + Ptr is nonsensical and yields Top (no longer arbitrarily returning LHS)
    CHECK(TrackedValDomain::add(b0, b_other).is_top());
    CHECK(TrackedValDomain::add(b0, b16).is_top());

    // SymbolValue + ConstInt(0) preserves non-derived; SymbolValue + ConstInt(4) marks derived
    auto sym_plus0 = TrackedValDomain::add(sym, TrackedValDomain::make_const(0));
    CHECK(!sym_plus0.inner.get_if<SymbolValueDomain>()->derived);
    auto sym_plus4 = TrackedValDomain::add(sym, TrackedValDomain::make_const(4));
    CHECK(sym_plus4.inner.get_if<SymbolValueDomain>()->derived);

    // Merged bit_and(a, b, is64 = true) handles both 32-bit and 64-bit all-ones masks
    auto mask32 = TrackedValDomain::make_const(0xFFFFFFFFLL);
    CHECK(TrackedValDomain::equals(TrackedValDomain::bit_and(b16, mask32, false), b16));
    CHECK(TrackedValDomain::equals(TrackedValDomain::bit_and(b16, TrackedValDomain::make_const(~0LL)), b16));
    auto b_masked = TrackedValDomain::bit_and(b16, TrackedValDomain::make_const(0x0F));
    CHECK(b_masked.inner.get_if<PointerDomain>()->byte_offset == 0);

    // FiniteSetDomain auto-lifted add with SrtSymbolLookupFn callback & Top propagation
    SrtSymbolLookupFn lookup = [](uint8_t rack, uint32_t desc) -> std::string {
        return (rack == 4 && desc == 2) ? "ssbo_2" : "";
    };
    TrackedValSet rack4 = TrackedValDomain::make_srt_rack(4, 0, true);
    TrackedValSet desc2_set = TrackedValSetDomain::add(rack4, TrackedValDomain::make_const(64), lookup);
    TrackedVal desc2 = collapse_tracked_val_set(desc2_set);
    const auto* dptr = desc2.inner.get_if<PointerDomain>();
    CHECK(dptr && dptr->kind == PointerVal::Kind::DescriptorAddr && dptr->desc_idx == 2 && dptr->symbol_name == "ssbo_2");

    TrackedValSet buf_plus_top = TrackedValSetDomain::add(TrackedValSet(b0), TrackedValSetDomain::top());
    CHECK(TrackedValDomain::equals(collapse_tracked_val_set(buf_plus_top), b_unk));

    // Debug formatter exposes active union alternative and .hi lane tag
    CHECK(format_tracked_val_debug(c42) == "[ConstInt] 0x2a");
    CHECK(format_tracked_val_debug(b16).rfind("[Pointer] &bufA+0x10", 0) == 0);
    CHECK(format_tracked_val_debug(TrackedValDomain::tuple_extract(b16, 1)).find(".hi") != std::string::npos);

    std::cout << "[PASS] test_tracked_val_lattice_laws_and_union_dispatch\n";
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
    CHECK(res.converged);

    for (SsaId id : {cp, add0, or0, xor0, and_all1, and_self, or_self, shl0, mul1, sel_same}) {
        auto v = res.val(id);
        const auto* ptr = v.inner.get_if<PointerDomain>();
        CHECK(ptr && ptr->kind == PointerVal::Kind::SrtBase);
        CHECK(ptr->offset_known && ptr->byte_offset == 0);
    }
    for (SsaId z : {sub_self, xor_self, and0}) {
        auto v = res.val(z);
        const auto* ci = v.inner.get_if<ConstIntDomain>();
        CHECK(ci && ci->is_known() && ci->value == 0);
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
    SsaId ld_non_single = emit_op(mod, 3, SsaOp::PsbLoad, ValType::TupleValToken, {use_pm, tok}, 8);

    SsaLivenessResult live = run_ssa_liveness(mod);
    CHECK(live.converged);
    CHECK(live.is_live_out(1, p1));
    CHECK(!live.is_live_out(2, p1));
    CHECK(live.is_live_out(2, p2));
    CHECK(!live.is_live_out(1, p2));
    CHECK(!live.is_live_in(3, p1));
    CHECK(!live.is_live_in(3, p2));
    CHECK(live.is_live_out(0, srt));
    CHECK(live.is_live_in(1, srt));
    CHECK(live.is_live_in(2, srt));
    CHECK(live.is_killed_at(p1, srt));
    CHECK(live.is_killed_at(p2, srt));
    CHECK(live.is_killed_at(use_pm, pm));

    PointerAnalysisEnv env;
    env.role = PointerAnalysisEnv::StageRole::Pilot;
    auto df = run_ssa_pointer_analysis(mod, env, &live);
    CHECK(df.converged);
    CHECK(df.val_set(pm).size() == 2);
    auto vpm = df.val(pm);
    const auto* vpm_ptr = vpm.inner.get_if<PointerDomain>();
    CHECK(vpm_ptr && vpm_ptr->kind == PointerVal::Kind::SrtBase && !vpm_ptr->offset_known);
    CHECK(df.val_set(ld_non_single).is_top());

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
    CHECK(df.converged);
    auto v_rack = df.val(rack_ptr);
    const auto* rp = v_rack.inner.get_if<PointerDomain>();
    CHECK(rp && rp->kind == PointerVal::Kind::SrtRackPtr && rp->rack == 4);
    auto v_buf = df.val(buf_va);
    const auto* bv = v_buf.inner.get_if<PointerDomain>();
    CHECK(bv && bv->kind == PointerVal::Kind::BufferAddr && bv->symbol_name == "my_ssbo");
    auto v_sz = df.val(buf_sz);
    const auto* bs = v_sz.inner.get_if<ShaderAtomDomain>();
    CHECK(bs && bs->is_known() && bs->value.kind == ShaderAtomDomain::Kind::BufferSize && bs->value.symbol_name == "my_ssbo");
    CHECK(df.pilot_fau_writes.count(0x00) == 1 &&
           df.pilot_fau_writes.at(0x00).value.inner.get_if<PointerDomain>()->symbol_name == "my_ssbo");
    CHECK(df.pilot_fau_writes.count(0x08) == 1 &&
           df.pilot_fau_writes.at(0x08).value.inner.get_if<ShaderAtomDomain>()->value.kind == ShaderAtomDomain::Kind::BufferSize);

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
    env.initial_fau[0x00] = TrackedValDomain::make_buffer_addr(4, 0, "input_buf", 0, false, true);

    auto df = run_ssa_pointer_analysis(mod, env);
    CHECK(df.converged);
    auto vp = df.val(ptr16);
    const auto* vptr = vp.inner.get_if<PointerDomain>();
    CHECK(vptr && vptr->kind == PointerVal::Kind::BufferAddr && vptr->symbol_name == "input_buf");
    CHECK(vptr->offset_known && vptr->byte_offset == 16);
    CHECK(!df.val(lo).is_hi_half && df.val(hi).is_hi_half);

    std::cout << "[PASS] test_ssa_pack64_split64_and_fau_seed\n";
}

void test_compound_op_delegation_lane_wrapper_and_atomic_exchange() {
    // Compound domain pointwise op delegation across VectorDomain, MapDomain, TupleDomain, FiniteSetDomain, LaneDomainWrapper
    VectorDomain<TrackedValDomain>::Value va, vb;
    va.set(0, TrackedValDomain::make_const(10));
    vb.set(0, TrackedValDomain::make_const(32));
    auto vsum = VectorDomain<TrackedValDomain>::iadd(va, vb, false);
    CHECK(TrackedValDomain::equals(vsum.get(0), TrackedValDomain::make_const(42)));
    auto vmul = VectorDomain<TrackedValDomain>::mul(va, vb);
    CHECK(TrackedValDomain::equals(vmul.get(0), TrackedValDomain::make_const(320)));

    MapDomain<uint32_t, TrackedValDomain>::Value ma, mb;
    ma.set(1, TrackedValDomain::make_const(0xF0));
    mb.set(1, TrackedValDomain::make_const(0x3C));
    auto mand = MapDomain<uint32_t, TrackedValDomain>::bit_and(ma, mb, false);
    CHECK(TrackedValDomain::equals(mand.get(1), TrackedValDomain::make_const(0x30)));
    auto mand_def = MapDomain<uint32_t, TrackedValDomain>::bit_and(ma, mb);
    CHECK(TrackedValDomain::equals(mand_def.get(1), TrackedValDomain::make_const(0x30)));

    using PairDom = TupleDomain<TrackedValDomain, TrackedValDomain>;
    auto tmul = PairDom::imul({TrackedValDomain::make_const(3), TrackedValDomain::make_const(5)},
                              {TrackedValDomain::make_const(7), TrackedValDomain::make_const(9)}, false);
    CHECK(std::get<0>(tmul).inner.get_if<ConstIntDomain>()->value == 21);
    CHECK(std::get<1>(tmul).inner.get_if<ConstIntDomain>()->value == 45);
    auto tsub = PairDom::sub({TrackedValDomain::make_const(20), TrackedValDomain::make_const(10)},
                             {TrackedValDomain::make_const(8), TrackedValDomain::make_const(3)});
    CHECK(std::get<0>(tsub).inner.get_if<ConstIntDomain>()->value == 12);
    CHECK(std::get<1>(tsub).inner.get_if<ConstIntDomain>()->value == 7);

    auto fsh = TrackedValSetDomain::shift(TrackedValDomain::make_const(3), TrackedValDomain::make_const(4),
                                          ShiftKind::Shl, false);
    CHECK(TrackedValDomain::equals(collapse_tracked_val_set(fsh), TrackedValDomain::make_const(48)));

    // LaneDomainWrapper<ConstIntDomain> & TrackedValDomain (= LaneDomainWrapper<TrackedValUnionDomain>)
    auto lv = TrackedValDomain::make_const(0x0000000200000005LL);
    auto sp = TrackedValDomain::split64(lv);
    auto lo = TrackedValDomain::tuple_extract(sp, 0);
    auto hi = TrackedValDomain::tuple_extract(sp, 1);
    CHECK(lo.inner.get_if<ConstIntDomain>()->value == 0x0000000200000005LL);
    CHECK(hi.inner.get_if<ConstIntDomain>()->value == 2 && !hi.is_hi_half);
    auto repacked = TrackedValDomain::pack64(TrackedValDomain::make_const(5),
                                             TrackedValDomain::make_const(2));
    CHECK(repacked.inner.get_if<ConstIntDomain>()->value == 0x0000000200000005LL);

    // pack64(ConstInt, Top) must widen to Top rather than falsely returning ConstInt
    CHECK(TrackedValDomain::pack64(TrackedValDomain::make_const(5), TrackedValDomain::top()).is_top());
    // pack64(Pointer, Top) preserves the pointer identity
    auto ptr_lo = TrackedValDomain::make_buffer_addr(4, 0, "p", 0);
    CHECK(TrackedValDomain::equals(TrackedValDomain::pack64(ptr_lo, TrackedValDomain::top()), ptr_lo));

    // Default AtomicExchange transfer in SSA pointer analysis
    ValhallSsaModule mod;
    SsaBlock b0{}; b0.id = 0; b0.is_terminated_by_end = true;
    mod.blocks.push_back(b0);
    SsaId tok = emit_op(mod, 0, SsaOp::Nop, ValType::Token);
    SsaId w0 = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 0);
    SsaId w1 = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 1);
    SsaId buf = emit_op(mod, 0, SsaOp::Pack64, ValType::U64, {w0, w1});
    SsaId c7 = emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 7);
    SsaId c9 = emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 9);
    emit_op(mod, 0, SsaOp::PsbStore, ValType::Token, {buf, c7, tok}, 4);
    SsaId ax = emit_op(mod, 0, SsaOp::AtomicExchange, ValType::TupleValToken, {buf, c9, tok});
    SsaId old_v = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U32, {ax}, 0);

    PointerAnalysisEnv env;
    env.role = PointerAnalysisEnv::StageRole::MainCompute;
    env.initial_fau[0x00] = TrackedValDomain::make_buffer_addr(4, 0, "counter_buf", 0, false, true);
    auto df = run_ssa_pointer_analysis(mod, env);
    CHECK(df.converged);
    CHECK(!df.val_set(old_v).is_bottom() && !df.val_set(old_v).is_top());

    std::cout << "[PASS] test_compound_op_delegation_lane_wrapper_and_atomic_exchange\n";
}


void test_flat_float_domains_algebraic_mixins_and_auto_convert() {
    // FloatDomain<double> backed by FlatDomain<double> (including NaN-safe lattice reflexivity)
    using FDom = FloatDomain<double>;
    auto f2 = FDom::Value::make(2.5);
    auto f4 = FDom::Value::make(4.0);
    auto f0 = FDom::Value::make(0.0);
    CHECK(FDom::equals(FDom::add(f2, f4), FDom::Value::make(6.5)));
    CHECK(FDom::equals(FDom::sub(f4, f2), FDom::Value::make(1.5)));
    CHECK(FDom::equals(FDom::mul(f2, f4), FDom::Value::make(10.0)));
    CHECK(FDom::equals(FDom::div(f4, f2), FDom::Value::make(1.6)));
    CHECK(FDom::div(f4, f0).is_top());
    CHECK(FDom::join(f2, f4).is_top());
    CHECK(FDom::meet(f2, f4).is_bottom());
    auto fnan = FDom::Value::make(std::nan(""));
    CHECK(FDom::equals(fnan, fnan));
    CHECK(FDom::leq(fnan, fnan));

    // UnionDomain<IntegralDomain<int64_t>, FloatDomain<double>> flows ops independently per variant
    using NumUnion = UnionDomain<IntegralDomain<int64_t>, FloatDomain<double>>;
    auto ni = NumUnion::add(NumUnion::Value(IntegralDomain<int64_t>::Value::make(10)),
                            NumUnion::Value(IntegralDomain<int64_t>::Value::make(32)));
    CHECK(ni.get_if<IntegralDomain<int64_t>>()->value == 42);
    auto nf = NumUnion::mul(NumUnion::Value(FDom::Value::make(3.0)),
                            NumUnion::Value(FDom::Value::make(2.5)));
    CHECK(FDom::equals(*nf.get_if<FDom>(), FDom::Value::make(7.5)));
    CHECK(NumUnion::add(ni, nf).is_top());

    // Automatic on_convert delegation in SsaDomainTransferVisitor<TrackedValSetDomain>
    // (SsaPointerTransferVisitor no longer overrides on_convert!)
    ValhallSsaModule mod;
    SsaBlock b0{}; b0.id = 0; b0.is_terminated_by_end = true;
    mod.blocks.push_back(b0);

    SsaId neg4_u32 = emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 0xFFFFFFFCu);
    SsaId big_u64 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 0x0000000200000010ULL);
    SsaId s_ext = emit_op(mod, 0, SsaOp::SConvert32To64, ValType::U64, {neg4_u32});
    SsaId u_ext = emit_op(mod, 0, SsaOp::UConvert32To64, ValType::U64, {neg4_u32});
    SsaId trunc = emit_op(mod, 0, SsaOp::UConvert64To32, ValType::U32, {big_u64});
    SsaId to_f = emit_op(mod, 0, SsaOp::ConvertSToF, ValType::F32, {neg4_u32});
    SsaId srt = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 2);
    SsaId srt_conv = emit_op(mod, 0, SsaOp::UConvert32To64, ValType::U64, {srt});

    PointerAnalysisEnv env;
    env.role = PointerAnalysisEnv::StageRole::Pilot;
    auto df = run_ssa_pointer_analysis(mod, env);
    CHECK(df.converged);
    CHECK(TrackedValDomain::equals(df.val(s_ext), TrackedValDomain::make_const(-4)));
    CHECK(TrackedValDomain::equals(df.val(u_ext), TrackedValDomain::make_const(0x00000000FFFFFFFCLL)));
    CHECK(TrackedValDomain::equals(df.val(trunc), TrackedValDomain::make_const(0x10)));
    CHECK(df.val(to_f).is_top());
    CHECK(TrackedValDomain::equals(df.val(srt_conv), TrackedValDomain::make_srt_base(0, true)));

    std::cout << "[PASS] test_flat_float_domains_algebraic_mixins_and_auto_convert\n";
}

void test_sym_expr_domain_and_mixins() {
    auto buf = SymExprDomain::from_expr(SymExpr::make_buffer_addr(4, 1, "ubo:0:0", 0x10, true));
    auto off = IntegralDomain<uint64_t>::Value::make(0xc0);
    auto shifted = SymExprDomain::add(buf, off);
    CHECK(shifted.is_known() && shifted.value.expr->byte_offset == 0xd0 && shifted.value.expr->offset_known);

    auto masked = SymExprDomain::bit_and(shifted, IntegralDomain<uint64_t>::Value::make(0x00FFFFFFULL));
    CHECK(SymExprDomain::equals(masked, shifted));

    auto widened = SymExprDomain::add(buf, IntegralDomain<uint64_t>::top());
    CHECK(widened.is_known() && !widened.value.expr->offset_known);

    auto joined = SymExprDomain::join(buf, shifted);
    CHECK(joined.is_top());
    CHECK(SymExprDomain::equals(SymExprDomain::join(buf, SymExprDomain::bottom()), buf));

    // Verify bounded depth cap (kMaxSymExprDepth).
    auto e = SymExpr::make_const_u32(1u);
    for (uint8_t i = 0; i < kMaxSymExprDepth; ++i) {
        e = SymExpr::make_unary(SsaOp::FNeg, e);
        CHECK(e != nullptr);
    }
    CHECK(SymExpr::make_unary(SsaOp::FNeg, e) == nullptr);
    std::cout << "[PASS] test_sym_expr_domain_and_mixins\n";
}

void test_known_bits_domain() {
    auto c16 = KnownBitsDomain::make_const_u64(16u);
    auto lane = KnownBitsDomain::make_u32_bound(15u);
    CHECK(lane.upper16_zero());
    CHECK(!lane.is_exact_const());

    auto or_val = KnownBitsDomain::bit_or(c16, lane, false);
    CHECK((or_val.known_one & 16u) == 16u);
    CHECK((or_val.known_zero & ~31ULL) == ~31ULL);

    auto lane0 = KnownBitsDomain::meet(lane, KnownBitsDomain::make_const_u64(0u));
    CHECK(lane0.is_exact_const() && lane0.const_u64() == 0u);
    auto or0 = KnownBitsDomain::bit_or(c16, lane0, false);
    CHECK(or0.is_exact_const() && or0.const_u64() == 16u);

    auto packed = KnownBitsDomain::bfi(KnownBitsDomain::make_const_u64(0x1234u),
                                       KnownBitsDomain::make_const_u64(0x5678u),
                                       KnownBitsDomain::make_const_u64(16u),
                                       KnownBitsDomain::make_const_u64(16u));
    CHECK(packed.is_exact_const() && packed.const_u64() == 0x56781234ULL);
    auto hi = KnownBitsDomain::bfe_u(packed, KnownBitsDomain::make_const_u64(16u),
                                     KnownBitsDomain::make_const_u64(16u));
    CHECK(hi.is_exact_const() && hi.const_u64() == 0x5678ULL);

    auto conflict = KnownBitsDomain::meet(KnownBitsDomain::make_const_u64(1u),
                                          KnownBitsDomain::make_const_u64(2u));
    CHECK(conflict.is_bottom());
    std::cout << "[PASS] test_known_bits_domain\n";
}

}  // namespace

void test_pilot_bfe_const_fold_and_deep_expr() {
    ValhallSsaModule mod;
    mod.blocks.emplace_back();
    mod.blocks[0].id = 0;
    auto add = [&](SsaOp op, ValType ty, std::vector<SsaId> args = {},
                   uint32_t imm32 = 0, uint64_t imm64 = 0) -> SsaId {
        SsaNode n{};
        n.op = op;
        n.type = ty;
        n.block_id = 0;
        n.args = std::move(args);
        n.imm_u32 = imm32;
        n.imm_u64 = imm64;
        SsaId id = mod.add_node(std::move(n));
        mod.blocks[0].nodes.push_back(id);
        return id;
    };

    SsaId tok = add(SsaOp::Nop, ValType::Token);
    SsaId fau_dst = add(SsaOp::LoadSidebandU64, ValType::U64, {}, 0);
    SsaId srt_base = add(SsaOp::LoadSidebandU64, ValType::U64, {}, 1);
    SsaId ld_rack = add(SsaOp::PsbLoad, ValType::TupleValToken, {srt_base, tok}, 8);
    SsaId rack_ptr = add(SsaOp::TupleExtract, ValType::U64, {ld_rack}, 0);
    SsaId c8 = add(SsaOp::ConstU64, ValType::U64, {}, 0, 8);
    SsaId desc_addr = add(SsaOp::IAdd64, ValType::U64, {rack_ptr, c8});
    SsaId ld_buf = add(SsaOp::PsbLoad, ValType::TupleValToken, {desc_addr, tok}, 8);
    SsaId buf_ptr = add(SsaOp::TupleExtract, ValType::U64, {ld_buf}, 0);
    SsaId ld_fmt = add(SsaOp::PsbLoad, ValType::TupleValToken, {buf_ptr, tok}, 4);
    SsaId fmt = add(SsaOp::TupleExtract, ValType::U32, {ld_fmt}, 0);

    // BfeU(0xc0b0a090, 0, 8) -> 0x90, then ICmpEq(fmt, 0x90) stored to FAU+0x1c.
    SsaId packed_k = add(SsaOp::ConstU32, ValType::U32, {}, 0xc0b0a090u);
    SsaId c0 = add(SsaOp::ConstU32, ValType::U32, {}, 0u);
    SsaId c8_u32 = add(SsaOp::ConstU32, ValType::U32, {}, 8u);
    SsaId byte0 = add(SsaOp::BfeU, ValType::U32, {packed_k, c0, c8_u32});
    SsaId eq = add(SsaOp::ICmpEq, ValType::Bool, {fmt, byte0});
    SsaId off1c = add(SsaOp::ConstU64, ValType::U64, {}, 0, 0x1c);
    SsaId dst1c = add(SsaOp::IAdd64, ValType::U64, {fau_dst, off1c});
    add(SsaOp::PsbStore, ValType::Token, {dst1c, eq, tok}, 4);

    PointerAnalysisEnv env;
    env.role = PointerAnalysisEnv::StageRole::Pilot;
    env.fau_dst_sideband_member = 0;
    env.srt_base_sideband_member = 1;
    env.lookup_srt_symbol = [](uint8_t, uint32_t) { return std::string("_2862_registers"); };

    FauSummaryMap summary = run_pilot_fau_summary(mod, env);
    CHECK(summary.count(0x1c) == 1);
    CHECK(summary[0x1c] != nullptr);
    CHECK(summary[0x1c]->kind == SymExpr::Kind::BinaryOp);
    CHECK(summary[0x1c]->op == SsaOp::ICmpEq);
    CHECK(summary[0x1c]->b != nullptr && summary[0x1c]->b->kind == SymExpr::Kind::ConstU32);
    CHECK(summary[0x1c]->b->const_u32 == 0x90u);
    std::cout << "[PASS] test_pilot_bfe_const_fold_and_deep_expr\n";
}

static void test_defau_distinct_sideband_ids_and_ternary_quaternary_symexpr() {
    ValhallSsaModule mod;
    mod.blocks.resize(1);
    mod.blocks[0].id = 0;

    auto add = [&](SsaOp op, ValType ty, std::vector<SsaId> args,
                   uint32_t imm32 = 0, uint64_t imm64 = 0, std::string aux = "") -> SsaId {
        SsaNode n{};
        n.id = static_cast<SsaId>(mod.nodes.size());
        n.op = op;
        n.type = ty;
        n.imm_u32 = imm32;
        n.imm_u64 = imm64;
        n.args = std::move(args);
        n.aux_str = std::move(aux);
        mod.nodes.push_back(std::move(n));
        SsaId id = static_cast<SsaId>(mod.nodes.size() - 1);
        mod.blocks[0].nodes.push_back(id);
        return id;
    };

    add(SsaOp::Nop, ValType::Token, {});
    // Two 64-bit buffer address pairs + two 32-bit FAU expressions (Fma + Bfi)
    SsaId f0 = add(SsaOp::LoadFau, ValType::U32, {}, 0u);  // FAU+0x00
    SsaId f1 = add(SsaOp::LoadFau, ValType::U32, {}, 1u);  // FAU+0x04
    SsaId addr0 = add(SsaOp::Pack64, ValType::U64, {f0, f1});
    SsaId f2 = add(SsaOp::LoadFau, ValType::U32, {}, 2u);  // FAU+0x08
    SsaId f3 = add(SsaOp::LoadFau, ValType::U32, {}, 3u);  // FAU+0x0c
    SsaId addr1 = add(SsaOp::Pack64, ValType::U64, {f2, f3});
    SsaId f4 = add(SsaOp::LoadFau, ValType::U32, {}, 4u);  // FAU+0x10 -> Fma
    SsaId f5 = add(SsaOp::LoadFau, ValType::U32, {}, 5u);  // FAU+0x14 -> Bfi
    (void)addr0;
    (void)addr1;
    (void)f4;
    (void)f5;

    FauSummaryMap summary;
    summary[0x00u] = SymExpr::make_buffer_addr(4, 1, "ssbo:0:1", 0, true);
    summary[0x08u] = SymExpr::make_buffer_addr(4, 0, "ssbo:0:0", 0, true);
    auto pc_ld = SymExpr::make_buffer_load(0, 0, "push_const", 0x04, 4);
    auto half = SymExpr::make_const_u32(0x3f000000u);
    summary[0x10u] = SymExpr::make_ternary(SsaOp::Fma, pc_ld, half, half);
    summary[0x14u] = SymExpr::make_quaternary(
        SsaOp::Bfi, pc_ld, SymExpr::make_const_u32(1u),
        SymExpr::make_const_u32(16u), SymExpr::make_const_u32(16u));

    DeFauDevirtResult res = run_ssa_defau_devirt(mod, summary);
    CHECK(res.fau_addrs_materialized == 2u);
    CHECK(res.fau_exprs_materialized == 2u);
    CHECK(res.fallbacks.empty());

    // Verify all 3 LoadSidebandU64 nodes have strictly distinct imm_u32 slots (>= 0x10).
    std::map<std::string, uint32_t> slot_by_key;
    std::set<uint32_t> seen_slots;
    bool has_fma = false;
    bool has_bfi = false;
    for (SsaId nid : mod.blocks[0].nodes) {
        const auto& n = mod.get(nid);
        if (n.op == SsaOp::LoadSidebandU64) {
            CHECK(n.imm_u32 >= 0x10u);
            CHECK(seen_slots.insert(n.imm_u32).second);
            slot_by_key[n.aux_str] = n.imm_u32;
        } else if (n.op == SsaOp::Fma) {
            has_fma = true;
        } else if (n.op == SsaOp::Bfi) {
            has_bfi = true;
        }
    }
    CHECK(slot_by_key.size() == 3u);
    CHECK(slot_by_key.count("ssbo:0:1") == 1u);
    CHECK(slot_by_key.count("ssbo:0:0") == 1u);
    CHECK(slot_by_key.count("push_const") == 1u);
    CHECK(has_fma);
    CHECK(has_bfi);
    std::cout << "[PASS] test_defau_distinct_sideband_ids_and_ternary_quaternary_symexpr\n";
}

int main() {
    std::cout << "=== [Test] Valhall SSA Dataflow Combinators, Worklist, Liveness & Pointer Analysis ===\n";
    test_domain_combinators_and_memory_disjointness();
    test_tracked_val_lattice_laws_and_union_dispatch();
    test_ssa_grouped_dispatch_and_extensional_equalities();
    test_ssa_diamond_cfg_phi_finiteset_and_liveness();
    test_ssa_pilot_srt_chasing_and_fau_store();
    test_ssa_pack64_split64_and_fau_seed();
    test_compound_op_delegation_lane_wrapper_and_atomic_exchange();
    test_flat_float_domains_algebraic_mixins_and_auto_convert();
    test_sym_expr_domain_and_mixins();
    test_known_bits_domain();
    test_pilot_bfe_const_fold_and_deep_expr();
    test_defau_distinct_sideband_ids_and_ternary_quaternary_symexpr();
    std::cout << "=== ALL SSA DATAFLOW TESTS PASSED ===\n";
    return 0;
}
