#include "ssa_cse.h"
#include "ssa_dce.h"
#include "ssa_fp_opt.h"
#include "ssa_buffer_symbolize.h"
#include "ssa_spill_mem2reg.h"
#include "ssa_sroa.h"
#include "ssa_opt_pipeline.h"
#include "ssa_defau_devirt.h"
#include "ssa_fact_prover.h"
#include "ssa_opt_utils.h"
#include "ssa_sccp.h"
#include "valhall_ssa.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

using namespace valhall::dataflow;
using namespace valhall::compiler;
using namespace valhall::tables;

namespace {

SsaId emit_op(ValhallSsaModule& mod, uint32_t blk, SsaOp op, ValType ty,
              std::vector<SsaId> args = {}, uint32_t imm32 = 0, uint64_t imm64 = 0,
              std::string aux_str = "", bool side_effect = false) {
    SsaNode n{};
    n.op = op;
    n.type = ty;
    n.block_id = blk;
    n.args = std::move(args);
    n.imm_u32 = imm32;
    n.imm_u64 = imm64;
    n.aux_str = std::move(aux_str);
    n.is_side_effect = side_effect;
    SsaId id = mod.add_node(std::move(n));
    if (op == SsaOp::Phi) {
        mod.blocks[blk].phis.push_back(id);
    } else {
        mod.blocks[blk].nodes.push_back(id);
    }
    return id;
}

uint32_t count_live_nodes(const ValhallSsaModule& mod) {
    uint32_t cnt = 0;
    for (const auto& blk : mod.blocks) {
        cnt += static_cast<uint32_t>(blk.phis.size() + blk.nodes.size());
    }
    return cnt;
}

uint32_t count_op(const ValhallSsaModule& mod, SsaOp op) {
    uint32_t cnt = 0;
    for (const auto& blk : mod.blocks) {
        for (SsaId id : blk.phis) {
            if (mod.get(id).op == op) ++cnt;
        }
        for (SsaId id : blk.nodes) {
            if (mod.get(id).op == op) ++cnt;
        }
    }
    return cnt;
}

void test_dce_basic_and_unreachable() {
    ValhallSsaModule mod;
    mod.blocks.resize(2);
    mod.blocks[0].id = 0;
    mod.blocks[0].is_terminated_by_end = true;
    mod.blocks[1].id = 1; // Unreachable block
    mod.blocks[1].is_terminated_by_end = true;

    SsaId a = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 0);
    SsaId b = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 1);
    SsaId live_sum = emit_op(mod, 0, SsaOp::IAdd, ValType::U32, {a, b});
    SsaId dead_mul = emit_op(mod, 0, SsaOp::IMul, ValType::U32, {live_sum, b});
    (void)emit_op(mod, 0, SsaOp::BitXor, ValType::U32, {dead_mul, a});
    SsaId copy_sum = emit_op(mod, 0, SsaOp::Copy, ValType::U32, {live_sum});
    SsaId out = emit_op(mod, 0, SsaOp::StoreFragColor, ValType::Void, {copy_sum}, 0, 0, "", true);

    // Dead computation in unreachable block 1
    (void)emit_op(mod, 1, SsaOp::ConstU32, ValType::U32, {}, 42);

    SsaPassStats st = run_ssa_dce(mod);
    CHECK(st.blocks_pruned == 1);
    CHECK(st.nodes_removed >= 2);
    CHECK(mod.blocks[1].nodes.empty());
    CHECK(mod.get(out).args[0] == live_sum); // Copy propagated
    CHECK(count_op(mod, SsaOp::IMul) == 0);
    CHECK(count_op(mod, SsaOp::BitXor) == 0);
    CHECK(count_op(mod, SsaOp::StoreFragColor) == 1);
    std::cout << "[PASS] test_dce_basic_and_unreachable\n";
}

void test_cse_dominator_scoping_and_commutativity() {
    // Diamond CFG: 0 -> {1, 2} -> 3
    ValhallSsaModule mod;
    mod.blocks.resize(4);
    for (uint32_t i = 0; i < 4; ++i) mod.blocks[i].id = i;
    mod.blocks[0].is_conditional_branch = true;
    mod.blocks[0].branch_target_block = 1;
    mod.blocks[0].fallthrough_block = 2;
    mod.blocks[1].preds = {0};
    mod.blocks[1].fallthrough_block = 3;
    mod.blocks[2].preds = {0};
    mod.blocks[2].fallthrough_block = 3;
    mod.blocks[3].preds = {1, 2};
    mod.blocks[3].is_terminated_by_end = true;

    SsaId cond = emit_op(mod, 0, SsaOp::LoadBuiltinInput, ValType::Bool, {}, 1);
    mod.blocks[0].branch_cond = cond;
    SsaId x = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 0);
    SsaId y = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 1);
    SsaId dom_add = emit_op(mod, 0, SsaOp::IAdd, ValType::U32, {x, y});
    (void)dom_add;

    // Block 1 computes commuted `y + x` (dominated by dom_add -> eliminated)
    // and a sibling-only `x * y`.
    SsaId b1_add = emit_op(mod, 1, SsaOp::IAdd, ValType::U32, {y, x});
    SsaId b1_mul = emit_op(mod, 1, SsaOp::IMul, ValType::U32, {x, y});
    (void)emit_op(mod, 1, SsaOp::StoreBuiltinOutput, ValType::Void, {b1_add}, 0, 0, "", true);

    // Block 2 also computes `x * y` — NOT dominated by block 1, must NOT be CSE'd!
    SsaId b2_mul = emit_op(mod, 2, SsaOp::IMul, ValType::U32, {x, y});

    // Block 3 has two identical Phis merging (b1_mul, b2_mul).
    SsaId phi1 = emit_op(mod, 3, SsaOp::Phi, ValType::U32);
    mod.get_mut(phi1).phi_incomings = {{1, b1_mul}, {2, b2_mul}};
    SsaId phi2 = emit_op(mod, 3, SsaOp::Phi, ValType::U32);
    mod.get_mut(phi2).phi_incomings = {{2, b2_mul}, {1, b1_mul}};
    SsaId fin = emit_op(mod, 3, SsaOp::IAdd, ValType::U32, {phi1, phi2});
    (void)emit_op(mod, 3, SsaOp::StoreFragColor, ValType::Void, {fin}, 0, 0, "", true);

    SsaPassStats st = run_ssa_cse(mod);
    CHECK(st.nodes_replaced == 2); // b1_add -> dom_add, phi2 -> phi1
    CHECK(st.phis_simplified == 1);
    CHECK(count_op(mod, SsaOp::IMul) == 2); // Sibling branches preserved!
    CHECK(mod.get(fin).args[0] == phi1 && mod.get(fin).args[1] == phi1);
    std::cout << "[PASS] test_cse_dominator_scoping_and_commutativity\n";
}

void test_sccp_const_fold_and_branch_pruning() {
    // Diamond CFG where branch condition folds to true -> block 2 pruned & Phi folded.
    ValhallSsaModule mod;
    mod.blocks.resize(4);
    for (uint32_t i = 0; i < 4; ++i) mod.blocks[i].id = i;
    mod.blocks[0].is_conditional_branch = true;
    mod.blocks[0].branch_target_block = 1;
    mod.blocks[0].fallthrough_block = 2;
    mod.blocks[1].preds = {0};
    mod.blocks[1].fallthrough_block = 3;
    mod.blocks[2].preds = {0};
    mod.blocks[2].fallthrough_block = 3;
    mod.blocks[3].preds = {1, 2};
    mod.blocks[3].is_terminated_by_end = true;

    SsaId c10 = emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 10);
    SsaId c20 = emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 20);
    SsaId c30 = emit_op(mod, 0, SsaOp::IAdd, ValType::U32, {c10, c20});
    SsaId cmp = emit_op(mod, 0, SsaOp::ICmpEq, ValType::Bool, {c30, emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 30)});
    mod.blocks[0].branch_cond = cmp;

    // Taken branch (block 1): computes 3.0f * 4.0f + 1.0f = 13.0f
    SsaId f3 = emit_op(mod, 1, SsaOp::ConstF32, ValType::F32, {}, f32_to_bits(3.0f));
    SsaId f4 = emit_op(mod, 1, SsaOp::ConstF32, ValType::F32, {}, f32_to_bits(4.0f));
    SsaId f1 = emit_op(mod, 1, SsaOp::ConstF32, ValType::F32, {}, f32_to_bits(1.0f));
    SsaId f13 = emit_op(mod, 1, SsaOp::Fma, ValType::F32, {f3, f4, f1});

    // Dead fallthrough branch (block 2): dynamic unknown value
    SsaId unk = emit_op(mod, 2, SsaOp::LoadFau, ValType::F32, {}, 5);

    // Merge block (block 3): Phi(1: f13, 2: unk) should fold to constant 13.0f!
    SsaId phi = emit_op(mod, 3, SsaOp::Phi, ValType::F32);
    mod.get_mut(phi).phi_incomings = {{1, f13}, {2, unk}};
    SsaId store = emit_op(mod, 3, SsaOp::StoreFragColor, ValType::Void, {phi}, 0, 0, "", true);

    SsaPassStats st = run_ssa_sccp(mod);
    CHECK(st.branches_folded == 1);
    CHECK(st.blocks_pruned == 1);
    CHECK(!mod.blocks[0].is_conditional_branch);
    CHECK(mod.blocks[0].branch_target_block == 1);
    CHECK(mod.blocks[2].nodes.empty());
    uint32_t out_bits = 0;
    CHECK(try_get_const_f32_bits(mod, mod.get(store).args[0], &out_bits));
    CHECK(bits_to_f32(out_bits) == 13.0f);
    std::cout << "[PASS] test_sccp_const_fold_and_branch_pruning\n";
}

void test_spill_mem2reg_diamond_and_loop() {
    // Diamond CFG: block 0 stores x to TLS+0x10; block 1 stores y to TLS+0x10;
    // block 2 does not store; block 3 loads TLS+0x10 -> must insert Phi(1: y, 2: x)!
    ValhallSsaModule mod;
    mod.blocks.resize(4);
    for (uint32_t i = 0; i < 4; ++i) mod.blocks[i].id = i;
    mod.blocks[0].is_conditional_branch = true;
    mod.blocks[0].branch_target_block = 1;
    mod.blocks[0].fallthrough_block = 2;
    mod.blocks[1].preds = {0};
    mod.blocks[1].fallthrough_block = 3;
    mod.blocks[2].preds = {0};
    mod.blocks[2].fallthrough_block = 3;
    mod.blocks[3].preds = {1, 2};
    mod.blocks[3].is_terminated_by_end = true;

    SsaId tok0 = emit_op(mod, 0, SsaOp::ConstU32, ValType::Token, {}, 0);
    SsaId cond = emit_op(mod, 0, SsaOp::LoadBuiltinInput, ValType::Bool, {}, 1);
    mod.blocks[0].branch_cond = cond;
    SsaId tls = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 6);
    SsaId c16 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 16);
    SsaId addr = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {tls, c16});
    SsaId x = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 0);
    SsaId st0 = emit_op(mod, 0, SsaOp::PsbStore, ValType::Token, {addr, x, tok0}, 4, 0, "", true);

    SsaId y = emit_op(mod, 1, SsaOp::LoadFau, ValType::U32, {}, 1);
    SsaId st1 = emit_op(mod, 1, SsaOp::PsbStore, ValType::Token, {addr, y, st0}, 4, 0, "", true);

    SsaId tok_phi = emit_op(mod, 3, SsaOp::Phi, ValType::Token);
    mod.get_mut(tok_phi).phi_incomings = {{1, st1}, {2, st0}};
    SsaId ld = emit_op(mod, 3, SsaOp::PsbLoad, ValType::TupleValToken, {addr, tok_phi}, 4);
    SsaId ld_val = emit_op(mod, 3, SsaOp::TupleExtract, ValType::U32, {ld}, 0);
    SsaId out = emit_op(mod, 3, SsaOp::StoreFragColor, ValType::Void, {ld_val}, 0, 0, "", true);

    SsaPassStats st = run_ssa_spill_mem2reg(mod);
    CHECK(st.loads_promoted == 1);
    CHECK(st.stores_eliminated == 2);
    CHECK(count_op(mod, SsaOp::PsbLoad) == 0);
    CHECK(count_op(mod, SsaOp::PsbStore) == 0);

    SsaId out_arg = mod.get(out).args[0];
    CHECK(mod.get(out_arg).op == SsaOp::Phi);
    CHECK(mod.get(out_arg).phi_incomings.size() == 2);
    bool saw_x = false, saw_y = false;
    for (const auto& inc : mod.get(out_arg).phi_incomings) {
        if (inc.pred_block == 1 && inc.value == y) saw_y = true;
        if (inc.pred_block == 2 && inc.value == x) saw_x = true;
    }
    CHECK(saw_x && saw_y);
    std::cout << "[PASS] test_spill_mem2reg_diamond_and_loop\n";
}

void test_sroa_fixed_size_struct() {
    // Store Pack64(f0, f1) at TLS+0x00 (8B), then load two 32-bit fields at TLS+0x00 and TLS+0x04.
    // SROA splits the 8B store into two 4B stores and mem2reg promotes both fields to f0 and f1!
    ValhallSsaModule mod;
    mod.blocks.resize(1);
    mod.blocks[0].id = 0;
    mod.blocks[0].is_terminated_by_end = true;

    SsaId tok0 = emit_op(mod, 0, SsaOp::ConstU32, ValType::Token, {}, 0);
    SsaId tls = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 6);
    SsaId c4 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 4);
    SsaId addr4 = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {tls, c4});

    SsaId f0 = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 0);
    SsaId f1 = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 1);
    SsaId pk = emit_op(mod, 0, SsaOp::Pack64, ValType::U64, {f0, f1});
    SsaId st64 = emit_op(mod, 0, SsaOp::PsbStore, ValType::Token, {tls, pk, tok0}, 8, 0, "", true);

    SsaId ld0 = emit_op(mod, 0, SsaOp::PsbLoad, ValType::TupleValToken, {tls, st64}, 4);
    SsaId v0 = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U32, {ld0}, 0);
    SsaId t0 = emit_op(mod, 0, SsaOp::TupleExtract, ValType::Token, {ld0}, 1);
    SsaId ld1 = emit_op(mod, 0, SsaOp::PsbLoad, ValType::TupleValToken, {addr4, t0}, 4);
    SsaId v1 = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U32, {ld1}, 0);
    SsaId sum = emit_op(mod, 0, SsaOp::IAdd, ValType::U32, {v0, v1});
    (void)emit_op(mod, 0, SsaOp::StoreFragColor, ValType::Void, {sum}, 0, 0, "", true);

    SsaPassStats st = run_ssa_sroa(mod);
    CHECK(st.structs_scalarized == 1);
    CHECK(st.loads_promoted == 2);
    CHECK(st.stores_eliminated == 2);
    CHECK(count_op(mod, SsaOp::PsbLoad) == 0);
    CHECK(count_op(mod, SsaOp::PsbStore) == 0);
    CHECK(mod.get(sum).args[0] == f0);
    CHECK(mod.get(sum).args[1] == f1);
    std::cout << "[PASS] test_sroa_fixed_size_struct\n";
}

void test_buffer_symbolization_fixed_stride() {
    // Compute shader: addr = ssbo:0:1 + (tid << 4) + 4 (stride 16, field offset 4)
    // Store val to addr, then reload from ssbo:0:1 + (tid * 16) + 4 -> same symbolic slot!
    ValhallSsaModule mod;
    mod.blocks.resize(1);
    mod.blocks[0].id = 0;
    mod.blocks[0].is_terminated_by_end = true;

    SsaId tok0 = emit_op(mod, 0, SsaOp::ConstU32, ValType::Token, {}, 0);
    SsaId base = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 0, 0, "ssbo:0:1");
    SsaId tid = emit_op(mod, 0, SsaOp::LoadBuiltinInput, ValType::U32, {}, 28);
    SsaId c4_sh = emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 4);
    SsaId off_shl = emit_op(mod, 0, SsaOp::Shl, ValType::U32, {tid, c4_sh});
    SsaId off_shl64 = emit_op(mod, 0, SsaOp::UConvert32To64, ValType::U64, {off_shl});
    SsaId c4_64 = emit_op(mod, 0, SsaOp::ConstU64, ValType::U64, {}, 0, 4);
    SsaId addr_a = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {base, off_shl64}), c4_64});

    SsaId c16 = emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 16);
    SsaId off_mul = emit_op(mod, 0, SsaOp::IMul, ValType::U32, {tid, c16});
    SsaId off_mul64 = emit_op(mod, 0, SsaOp::UConvert32To64, ValType::U64, {off_mul});
    SsaId addr_b = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {base, off_mul64}), c4_64});

    SsaId val = emit_op(mod, 0, SsaOp::LoadFau, ValType::U32, {}, 2);
    SsaId st_node = emit_op(mod, 0, SsaOp::PsbStore, ValType::Token, {addr_a, val, tok0}, 4, 0, "", true);
    SsaId ld_node = emit_op(mod, 0, SsaOp::PsbLoad, ValType::TupleValToken, {addr_b, st_node}, 4);
    SsaId ld_val = emit_op(mod, 0, SsaOp::TupleExtract, ValType::U32, {ld_node}, 0);
    SsaId out = emit_op(mod, 0, SsaOp::StoreFragColor, ValType::Void, {ld_val}, 0, 0, "", true);

    BufferSymbolizeResult res = run_ssa_symbolize_buffers(mod);
    CHECK(res.accesses.size() == 2);
    CHECK(res.accesses[0].sym_slot_tag == "sym:ssbo:0:1[stride=16,off=4]");
    CHECK(res.accesses[1].sym_slot_tag == "sym:ssbo:0:1[stride=16,off=4]");
    CHECK(res.stats.loads_promoted == 1);
    CHECK(mod.get(st_node).aux_str == "sym:ssbo:0:1[stride=16,off=4]");
    CHECK(mod.get(st_node).imm_u32_2 == 16);
    CHECK(mod.get(st_node).aux_index == 4);
    CHECK(mod.get(st_node).aux_bool == true);
    // Redundant reload forwarded directly to `val`!
    CHECK(mod.get(out).args[0] == val);
    CHECK(count_op(mod, SsaOp::PsbLoad) == 0);
    std::cout << "[PASS] test_buffer_symbolization_fixed_stride\n";
}

void test_fp_canonicalizer_strict_and_fastmath() {
    // 1. StrictIEEE: strip Valhall `-0.0` FMA/FADD carriers, double neg, clamp intersection,
    //    and ensure unsafe FastMath rules (x + +0.0, FRCP+FMUL) do NOT fire.
    {
        ValhallSsaModule mod;
        mod.blocks.resize(1);
        mod.blocks[0].id = 0;
        mod.blocks[0].is_terminated_by_end = true;

        SsaId a = emit_op(mod, 0, SsaOp::LoadFau, ValType::F32, {}, 0);
        SsaId b = emit_op(mod, 0, SsaOp::LoadFau, ValType::F32, {}, 1);
        SsaId neg0 = emit_op(mod, 0, SsaOp::ConstF32, ValType::F32, {}, 0x80000000u);
        SsaId pos0 = emit_op(mod, 0, SsaOp::ConstF32, ValType::F32, {}, 0x00000000u);
        SsaId pos1 = emit_op(mod, 0, SsaOp::ConstF32, ValType::F32, {}, 0x3F800000u);
        SsaId neg1 = emit_op(mod, 0, SsaOp::ConstF32, ValType::F32, {}, 0xBF800000u);

        // Mali FMUL lowering: Fma(a, b, -0.0) -> FMul(a, b)
        SsaId fma_mul = emit_op(mod, 0, SsaOp::Fma, ValType::F32, {a, b, neg0});
        // Mali clamp carrier: FAdd(fma_mul, -0.0) -> fma_mul
        SsaId add_neg0 = emit_op(mod, 0, SsaOp::FAdd, ValType::F32, {fma_mul, neg0});
        // Double FNeg: --add_neg0 -> add_neg0
        SsaId nn = emit_op(mod, 0, SsaOp::FNeg, ValType::F32, {emit_op(mod, 0, SsaOp::FNeg, ValType::F32, {add_neg0})});
        // Nested clamp intersection: clamp(clamp(nn, -1, 1), 0, 1) -> clamp(nn, 0, 1)
        SsaId cl1 = emit_op(mod, 0, SsaOp::FClamp, ValType::F32, {nn, neg1, pos1});
        SsaId cl2 = emit_op(mod, 0, SsaOp::FClamp, ValType::F32, {cl1, pos0, pos1});
        // Unsafe in StrictIEEE: x + +0.0 must remain FAdd!
        SsaId add_pos0 = emit_op(mod, 0, SsaOp::FAdd, ValType::F32, {cl2, pos0});
        SsaId out = emit_op(mod, 0, SsaOp::StoreFragColor, ValType::Void, {add_pos0}, 0, 0, "", true);

        FpOptOptions fopts;
        fopts.mode = FpOptMode::StrictIEEE;
        FpOptResult res = run_ssa_fp_canonicalize(mod, fopts);

        CHECK(res.rule_hits["tile:elide_fma_neg_zero"] == 1);
        CHECK(res.rule_hits["tile:elide_fadd_neg_zero_r"] + res.rule_hits["tile:elide_fadd_neg_zero_l"] == 1);
        CHECK(res.rule_hits["tile:elide_fneg_fneg"] == 1);
        CHECK(res.rule_hits["C2_nested_fclamp"] == 1);
        CHECK(res.rule_hits.count("I2_fadd_pos0") == 0);
        CHECK(mod.get(fma_mul).op == SsaOp::FMul);
        CHECK(mod.get(out).args[0] == add_pos0);
        CHECK(mod.get(add_pos0).op == SsaOp::FAdd);
    }

    // 2. FastMath: Mali FRCP+FMUL -> FDiv, FRSQ+FRCP -> Sqrt, FMUL_RSCALE Ldexp(-1) -> FMul(0.5),
    //    Exp2(Log2(x)*y) -> Pow(x, y), FAdd(FMul(a,b), c) -> Fma(a,b,c).
    {
        ValhallSsaModule mod;
        mod.blocks.resize(1);
        mod.blocks[0].id = 0;
        mod.blocks[0].is_terminated_by_end = true;

        SsaId a = emit_op(mod, 0, SsaOp::LoadFau, ValType::F32, {}, 0);
        SsaId b = emit_op(mod, 0, SsaOp::LoadFau, ValType::F32, {}, 1);
        SsaId c = emit_op(mod, 0, SsaOp::LoadFau, ValType::F32, {}, 2);
        SsaId one = emit_op(mod, 0, SsaOp::ConstF32, ValType::F32, {}, 0x3F800000u);
        SsaId m1 = emit_op(mod, 0, SsaOp::ConstS32, ValType::S32, {}, static_cast<uint32_t>(-1));

        // Mali a / b: FRCP(b) + FMUL(a, rcp)
        SsaId rcp_b = emit_op(mod, 0, SsaOp::FDiv, ValType::F32, {one, b});
        SsaId div_ab = emit_op(mod, 0, SsaOp::FMul, ValType::F32, {a, rcp_b});
        // Mali sqrt(a): FRSQ(a) + FRCP
        SsaId rsq_a = emit_op(mod, 0, SsaOp::InverseSqrt, ValType::F32, {a});
        SsaId sqrt_a = emit_op(mod, 0, SsaOp::FDiv, ValType::F32, {one, rsq_a});
        // Mali FMUL_RSCALE(..., -1) -> Ldexp(..., -1)
        SsaId scaled = emit_op(mod, 0, SsaOp::Ldexp, ValType::F32, {div_ab, m1});
        // FMA fusion: scaled + sqrt_a * c
        SsaId prod = emit_op(mod, 0, SsaOp::FMul, ValType::F32, {sqrt_a, c});
        SsaId fused = emit_op(mod, 0, SsaOp::FAdd, ValType::F32, {prod, scaled});
        (void)emit_op(mod, 0, SsaOp::StoreFragColor, ValType::Void, {fused}, 0, 0, "", true);

        FpOptOptions fopts;
        fopts.mode = FpOptMode::FastMath;
        FpOptResult res = run_ssa_fp_canonicalize(mod, fopts);

        CHECK(res.rule_hits["tile:fast_fuse_fmul_rcp_r"] + res.rule_hits["tile:fast_fuse_fmul_rcp_l"] == 1);
        CHECK(res.rule_hits["tile:fast_fuse_rcp_rsqrt"] == 1);
        CHECK(res.rule_hits["S2_ldexp_to_fmul"] == 1);
        CHECK(res.rule_hits["F2_fuse_fma"] == 1);
        CHECK(mod.get(div_ab).op == SsaOp::FDiv);
        CHECK(mod.get(sqrt_a).op == SsaOp::Sqrt);
        CHECK(mod.get(scaled).op == SsaOp::FMul);
        CHECK(mod.get(fused).op == SsaOp::Fma);
    }
    std::cout << "[PASS] test_fp_canonicalizer_strict_and_fastmath\n";
}

void test_full_opt_pipeline() {
    ValhallSsaModule mod;
    mod.blocks.resize(1);
    mod.blocks[0].id = 0;
    mod.blocks[0].is_terminated_by_end = true;

    SsaId tok0 = emit_op(mod, 0, SsaOp::ConstU32, ValType::Token, {}, 0);
    SsaId tls = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 6);
    SsaId x = emit_op(mod, 0, SsaOp::LoadFau, ValType::F32, {}, 0);
    SsaId neg0 = emit_op(mod, 0, SsaOp::ConstF32, ValType::F32, {}, 0x80000000u);
    SsaId one = emit_op(mod, 0, SsaOp::ConstF32, ValType::F32, {}, 0x3F800000u);

    // Spill x to TLS+0, reload it, wrap in Fma(x, 1.0, -0.0) + dead math, write to SSBO[tid*4].
    SsaId st_tls = emit_op(mod, 0, SsaOp::PsbStore, ValType::Token, {tls, x, tok0}, 4, 0, "", true);
    SsaId ld_tls = emit_op(mod, 0, SsaOp::PsbLoad, ValType::TupleValToken, {tls, st_tls}, 4);
    SsaId reloaded = emit_op(mod, 0, SsaOp::TupleExtract, ValType::F32, {ld_tls}, 0);
    SsaId tok1 = emit_op(mod, 0, SsaOp::TupleExtract, ValType::Token, {ld_tls}, 1);

    SsaId fma_id = emit_op(mod, 0, SsaOp::Fma, ValType::F32, {reloaded, one, neg0});
    (void)emit_op(mod, 0, SsaOp::FMul, ValType::F32, {x, one}); // Dead

    SsaId ssbo = emit_op(mod, 0, SsaOp::LoadSidebandU64, ValType::U64, {}, 0, 0, "ssbo:0:0");
    SsaId tid = emit_op(mod, 0, SsaOp::LoadBuiltinInput, ValType::U32, {}, 28);
    SsaId off = emit_op(mod, 0, SsaOp::UConvert32To64, ValType::U64, {emit_op(mod, 0, SsaOp::Shl, ValType::U32, {tid, emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 2)})});
    SsaId dst_addr = emit_op(mod, 0, SsaOp::IAdd64, ValType::U64, {ssbo, off});
    SsaId ssbo_st = emit_op(mod, 0, SsaOp::PsbStore, ValType::Token, {dst_addr, fma_id, tok1}, 4, 0, "", true);

    SsaOptPipelineOptions popts;
    popts.fp_mode = FpOptMode::StrictIEEE;
    popts.record_trace = true;
    SsaOptPipelineResult pres = run_ssa_opt_pipeline(mod, popts);

    CHECK(pres.total_stats.loads_promoted == 1);
    CHECK(pres.total_stats.stores_eliminated == 1);
    CHECK(pres.symbolized_accesses.size() == 1);
    CHECK(pres.symbolized_accesses[0].sym_slot_tag == "sym:ssbo:0:0[stride=4,off=0]");
    CHECK(mod.get(ssbo_st).args[1] == x); // Fma(reloaded, 1.0, -0.0) reduced all the way to x!
    CHECK(count_live_nodes(mod) < 12);
    CHECK(!pres.trace.empty());
    std::cout << "[PASS] test_full_opt_pipeline (rounds=" << pres.rounds_executed
              << ", remaining_nodes=" << count_live_nodes(mod) << ")\n";
}

}  // namespace


void test_analysis_cache_and_invalidation() {
    ValhallSsaModule mod;
    mod.blocks.emplace_back();
    mod.blocks[0].id = 0;
    SsaId c1 = emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 7u);
    SsaId c2 = emit_op(mod, 0, SsaOp::ConstU32, ValType::U32, {}, 5u);
    SsaId sum = emit_op(mod, 0, SsaOp::IAdd, ValType::U32, {c1, c2});
    emit_op(mod, 0, SsaOp::StoreFragColor, ValType::Void, {sum}, 0u, 0u, "", true);
    mod.blocks[0].is_terminated_by_end = true;

    const DominatorTree& dt1 = get_or_build_dominator_tree(mod);
    const DominatorTree& dt2 = get_or_build_dominator_tree(mod);
    CHECK(&dt1 == &dt2);
    const auto* cache = get_ssa_analysis_cache(mod);
    CHECK(cache != nullptr && cache->dom_tree_builds == 1);

    const PointerAnalysisResult& pa1 = get_or_run_pointer_analysis(mod);
    const PointerAnalysisResult& pa2 = get_or_run_pointer_analysis(mod);
    CHECK(&pa1 == &pa2);
    CHECK(cache->ptr_analysis_runs == 1);

    uint32_t folded_sum = 0;
    CHECK(try_get_const_u32(mod, sum, &folded_sum));
    CHECK(folded_sum == 12u);

    SsaNode extra{};
    extra.op = SsaOp::ConstU32;
    extra.type = ValType::U32;
    extra.imm_u32 = 99u;
    insert_node_before(mod, 0, kInvalidSsa, std::move(extra));
    CHECK(get_ssa_analysis_cache(mod) == nullptr);
    std::cout << "[PASS] test_analysis_cache_and_invalidation\n";
}

void test_fp_dual_toggles_tile_vs_cpp() {
    auto build_sample = []() {
        ValhallSsaModule mod;
        mod.blocks.emplace_back();
        mod.blocks[0].id = 0;
        SsaId a = emit_op(mod, 0, SsaOp::LoadFau, ValType::F32, {}, 0u);
        SsaId b = emit_op(mod, 0, SsaOp::LoadFau, ValType::F32, {}, 1u);
        SsaId neg0 = emit_op(mod, 0, SsaOp::ConstF32, ValType::F32, {}, f32_to_bits(-0.0f));
        SsaId fma = emit_op(mod, 0, SsaOp::Fma, ValType::F32, {a, b, neg0});
        emit_op(mod, 0, SsaOp::StoreFragColor, ValType::Void, {fma}, 0u, 0u, "", true);
        mod.blocks[0].is_terminated_by_end = true;
        return mod;
    };

    {
        ValhallSsaModule mod = build_sample();
        FpOptOptions fopts;
        fopts.mode = FpOptMode::StrictIEEE;
        fopts.enable_tile_spec_rewrites = true;
        fopts.enable_cpp_rewrites = false;
        FpOptResult res = run_ssa_fp_canonicalize(mod, fopts);
        CHECK(res.rule_hits["tile:elide_fma_neg_zero"] == 1);
        CHECK(res.rule_hits["I6_fma_neg0"] == 0);
    }

    {
        ValhallSsaModule mod = build_sample();
        FpOptOptions fopts;
        fopts.mode = FpOptMode::StrictIEEE;
        fopts.enable_tile_spec_rewrites = false;
        fopts.enable_cpp_rewrites = true;
        FpOptResult res = run_ssa_fp_canonicalize(mod, fopts);
        CHECK(res.rule_hits["tile:elide_fma_neg_zero"] == 0);
        CHECK(res.rule_hits["I6_fma_neg0"] == 1);
    }

    {
        std::vector<SsaId> args = {9u, 3u};
        canonicalize_commutative_args(SsaOp::IAdd, args);
        CHECK(args[0] == 3u && args[1] == 9u);

        std::vector<SsaId> fma_args = {7u, 2u, 1u};
        canonicalize_commutative_args(SsaOp::Fma, fma_args);
        CHECK(fma_args[0] == 2u && fma_args[1] == 7u && fma_args[2] == 1u);
    }
    std::cout << "[PASS] test_fp_dual_toggles_tile_vs_cpp\n";
}

void test_defau_devirt_fcst_loads_exprs_and_fallbacks() {
    ValhallSsaModule mod;
    SsaBlock blk{}; blk.id = 0; mod.blocks.push_back(blk);
    mod.nodes.push_back(SsaNode{});

    auto emit = [&](SsaOp op, ValType ty, std::vector<SsaId> args, uint32_t imm = 0) -> SsaId {
        SsaNode n{};
        n.id = static_cast<SsaId>(mod.nodes.size());
        n.op = op;
        n.type = ty;
        n.imm_u32 = imm;
        n.args = std::move(args);
        mod.nodes.push_back(std::move(n));
        SsaId id = static_cast<SsaId>(mod.nodes.size() - 1);
        mod.blocks[0].nodes.push_back(id);
        return id;
    };

    SsaId tok = emit(SsaOp::Nop, ValType::Token, {});
    (void)tok;
    SsaId fau_fcst = emit(SsaOp::LoadFau, ValType::U32, {}, 8u);   // FAU+0x20 -> FCST 0x0ffe0000
    SsaId fau_ubo = emit(SsaOp::LoadFau, ValType::U32, {}, 2u);    // FAU+0x08 -> ubo:0:0+0xd0
    SsaId fau_pred = emit(SsaOp::LoadFau, ValType::U32, {}, 0u);   // FAU+0x00 -> Select(ubo==1.0f, ~0, 0)
    SsaId fau_lo = emit(SsaOp::LoadFau, ValType::U32, {}, 6u);     // FAU+0x18 -> &ssbo:0:1
    SsaId fau_hi = emit(SsaOp::LoadFau, ValType::U32, {}, 7u);
    SsaId addr64 = emit(SsaOp::Pack64, ValType::U64, {fau_lo, fau_hi});
    SsaId fau_unk = emit(SsaOp::LoadFau, ValType::U32, {}, 30u);   // Unresolved -> fallback
    emit(SsaOp::BitAnd, ValType::U32, {fau_fcst, fau_ubo});
    emit(SsaOp::BitOr, ValType::U32, {fau_pred, fau_unk});
    emit(SsaOp::PsbLoad, ValType::TupleValToken, {addr64, tok}, 4u);

    FauSummaryMap summary;
    seed_fcst_into_fau_summary({{0x20u, 0x0ffe0000u}}, summary);
    auto ubo_ld = SymExpr::make_buffer_load(4, 0, "ubo:0:0", 0xd0, 4);
    summary[0x08u] = ubo_ld;
    auto cmp = SymExpr::make_binary(SsaOp::FCmpOrdEq, ubo_ld, SymExpr::make_const_u32(0x3f800000u));
    summary[0x00u] = SymExpr::make_select(cmp, SymExpr::make_const_u32(0xFFFFFFFFu), SymExpr::make_const_u32(0u));
    summary[0x18u] = SymExpr::make_buffer_addr(4, 1, "ssbo:0:1", 0, true);

    auto res = run_ssa_defau_devirt(mod, summary);
    CHECK(res.fau_consts_materialized == 1u);
    CHECK(res.fau_loads_materialized == 1u);
    CHECK(res.fau_exprs_materialized == 1u);
    CHECK(res.fau_addrs_materialized == 1u);
    CHECK(res.fallbacks.size() == 1u);
    CHECK(res.fallbacks[0].reason == FallbackReason::UnresolvedFauSlot);
    CHECK(mod.get(fau_fcst).op == SsaOp::ConstU32 && mod.get(fau_fcst).imm_u32 == 0x0ffe0000u);
    std::cout << "[PASS] test_defau_devirt_fcst_loads_exprs_and_fallbacks\n";
}

void test_fact_prover_bitcasts_isnan_masks_bfi_and_path_csel() {
    ValhallSsaModule mod;
    SsaBlock b0{}; b0.id = 0; b0.is_conditional_branch = true; b0.branch_target_block = 2; b0.fallthrough_block = 1;
    SsaBlock b1{}; b1.id = 1; b1.preds = {0}; b1.is_unconditional_branch = true; b1.branch_target_block = 2;
    SsaBlock b2{}; b2.id = 2; b2.preds = {0, 1}; b2.is_terminated_by_end = true;
    mod.blocks = {b0, b1, b2};
    mod.nodes.push_back(SsaNode{});

    auto emit = [&](uint32_t blk, SsaOp op, ValType ty, std::vector<SsaId> args,
                    uint32_t imm = 0, std::string aux = "") -> SsaId {
        SsaNode n{};
        n.id = static_cast<SsaId>(mod.nodes.size());
        n.op = op;
        n.type = ty;
        n.imm_u32 = imm;
        n.args = std::move(args);
        n.aux_str = std::move(aux);
        mod.nodes.push_back(std::move(n));
        SsaId id = static_cast<SsaId>(mod.nodes.size() - 1);
        mod.blocks[blk].nodes.push_back(id);
        return id;
    };

    // bb0: round-trip bitcast, FCmpUnordNe(x, x), m1 mask unwrap, Bfi/BfeU 16:16, BRANCH lane_id != 0
    SsaId c0 = emit(0, SsaOp::ConstU32, ValType::U32, {}, 0u);
    SsaId c16 = emit(0, SsaOp::ConstU32, ValType::U32, {}, 16u);
    SsaId call1 = emit(0, SsaOp::ConstU32, ValType::U32, {}, 0xFFFFFFFFu);
    SsaId f0 = emit(0, SsaOp::ConstF32, ValType::F32, {}, 0x3f800000u);
    SsaId u_rt = emit(0, SsaOp::Bitcast, ValType::U32, {f0});
    SsaId f_rt = emit(0, SsaOp::Bitcast, ValType::F32, {u_rt});
    SsaId nan_cmp = emit(0, SsaOp::FCmpUnordNe, ValType::Bool, {f_rt, f_rt});
    SsaId m1 = emit(0, SsaOp::Select, ValType::U32, {nan_cmp, call1, c0});
    SsaId unwrapped = emit(0, SsaOp::ICmpNe, ValType::Bool, {m1, c0});
    (void)unwrapped;
    SsaId packed = emit(0, SsaOp::Bfi, ValType::U32, {c0, c16, c16, c16});
    SsaId hi16 = emit(0, SsaOp::BfeU, ValType::U32, {packed, c16, c16});
    (void)hi16;
    SsaId lid0 = emit(0, SsaOp::LoadBuiltinInput, ValType::U32, {}, 0x29u, "gl_SubgroupInvocationID");
    SsaId br_c = emit(0, SsaOp::ICmpNe, ValType::Bool, {lid0, c0});
    mod.blocks[0].branch_cond = br_c;

    // bb1 (fallthrough where lane_id == 0): 16 | lane_id == 1 -> false -> collapse Select
    SsaId lid1 = emit(1, SsaOp::LoadBuiltinInput, ValType::U32, {}, 0x29u, "gl_SubgroupInvocationID");
    SsaId or16 = emit(1, SsaOp::BitOr, ValType::U32, {c16, lid1});
    SsaId c1 = emit(1, SsaOp::ConstU32, ValType::U32, {}, 1u);
    SsaId eq1 = emit(1, SsaOp::ICmpEq, ValType::Bool, {or16, c1});
    SsaId sel = emit(1, SsaOp::Select, ValType::U32, {eq1, c1, c0});
    (void)sel;

    auto res = run_ssa_fact_prover(mod);
    CHECK(res.bitcasts_eliminated >= 1u);
    CHECK(res.isnan_canonicalized == 1u);
    CHECK(res.bool_masks_unwrapped == 1u);
    CHECK(res.bfi_bfe_projected == 1u);
    CHECK(res.path_consts_folded >= 2u);
    CHECK(res.csel_ladders_collapsed == 1u);
    CHECK(mod.get(nan_cmp).op == SsaOp::IsNan);
    std::cout << "[PASS] test_fact_prover_bitcasts_isnan_masks_bfi_and_path_csel\n";
}

int main() {
    test_dce_basic_and_unreachable();
    test_cse_dominator_scoping_and_commutativity();
    test_sccp_const_fold_and_branch_pruning();
    test_spill_mem2reg_diamond_and_loop();
    test_sroa_fixed_size_struct();
    test_buffer_symbolization_fixed_stride();
    test_fp_canonicalizer_strict_and_fastmath();
    test_full_opt_pipeline();
    test_analysis_cache_and_invalidation();
    test_fp_dual_toggles_tile_vs_cpp();
    test_defau_devirt_fcst_loads_exprs_and_fallbacks();
    test_fact_prover_bitcasts_isnan_masks_bfi_and_path_csel();
    std::cout << "ALL SSA OPT TESTS PASSED!\n";
    return 0;
}
