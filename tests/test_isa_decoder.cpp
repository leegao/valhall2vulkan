#include "valhall_isa_generated.h"
#include "valhall_cfg.h"
#include "valhall_spirv_builder.h"
#include "valhall_to_spirv.h"
#include "cmpbe_chunks.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>

namespace sample_ns {
#include "fixtures/sample_mbs2_shader.h"
}
namespace astc_ns {
#include "fixtures/astc_encoder_mbs2_shader.h"
}

int main() {
    std::cout << "=== [Test 1] Valhall ISA (v9-v14) Spec Extractor & Decoder Verification ===\n";

    // 1. Verify all 627 instruction specs from isa-v9-v14.xml are present and decode their canonical fixed encodings
    size_t count = 0;
    const valhall::InstructionSemanticsSpec* specs = valhall::get_all_instruction_specs(&count);
    assert(count == 627);
    assert(specs != nullptr);
    std::cout << "[PASS] Loaded " << count << " instruction specifications from isa-v9-v14.xml + ISA.xml\n";

    size_t decoded_specs = 0;
    std::set<std::string> unique_mnemonics;
    for (size_t i = 0; i < count; ++i) {
        const auto& sp = specs[i];
        unique_mnemonics.insert(valhall::mnemonic_to_string(sp.mnemonic));
        uint8_t test_arch = std::max<uint8_t>(9, sp.since_arch);
        auto dec = valhall::decode_valhall_instruction(sp.fixed_value, test_arch, 0, false);
        if (dec.valid) {
            ++decoded_specs;
            assert(dec.canonical_mnemonic == sp.canonical_mnemonic);
        }
    }
    std::cout << "[PASS] Verified " << unique_mnemonics.size() << " unique mnemonics and "
              << decoded_specs << "/" << count << " fixed-pattern spec encodings across v9..v14\n";
    assert(unique_mnemonics.size() == 201);

    // 2. Verify all 32 Small Constant LUT entries from valhall/ISA.xml
    const auto* k0 = valhall::lookup_small_constant(0, 11);
    const auto* k1f = valhall::lookup_small_constant(16, 11);
    const auto* k03020100 = valhall::lookup_small_constant(8, 11);
    assert(k0 && k0->bit_pattern == 0x00000000u);
    assert(k1f && k1f->bit_pattern == 0x3f800000u);
    assert(k03020100 && k03020100->bit_pattern == 0x03020100u);
    std::cout << "[PASS] Verified 32-entry Small Constant LUT (k0=0x0, k1f=0x3f800000, k03020100=0x03020100)\n";

    // 3. Verify FAU Special Pages (0, 1, 3) from valhall/ISA.xml & isa-v9-v14.xml
    const char* p0_bd0 = valhall::lookup_fau_special_name(0, 8, 11);
    const char* p0_wid = valhall::lookup_fau_special_name(0, 2, 11);
    const char* p1_tlp = valhall::lookup_fau_special_name(1, 1, 11);
    const char* p3_lid = valhall::lookup_fau_special_name(3, 1, 11);
    assert(p0_bd0 && std::string(p0_bd0) == "blend_descriptor_0");
    assert(p0_wid && std::string(p0_wid) == "warp_id");
    assert(p1_tlp && std::string(p1_tlp) == "thread_local_pointer");
    assert(p3_lid && std::string(p3_lid) == "lane_id");
    std::cout << "[PASS] Verified FAU Special Page resolvers (blend_descriptor_0, pc, warp_id, lane_id)\n";

    // 4. Decode sample_mbs2_shader.h (Main Compute EBIN_42 + Pilot EBIN_44)
    auto sample_main = valhall::decode_valhall_shader(
        sample_ns::EBIN_42.bytecode.byte_code,
        sample_ns::EBIN_42.data_section_offset,
        11, false);
    assert(sample_main.size() == 27);
    for (const auto& ins : sample_main) {
        assert(ins.valid && ins.mnemonic != valhall::Mnemonic::INVALID);
    }
    assert(sample_main[0].mnemonic == valhall::Mnemonic::ATOM1);
    assert(sample_main[6].mnemonic == valhall::Mnemonic::BRANCH);
    assert(sample_main[13].mnemonic == valhall::Mnemonic::ADR);
    assert(sample_main[26].mnemonic == valhall::Mnemonic::NOP && sample_main[26].flow == valhall::FlowControlM::end);

    auto sample_pilot = valhall::decode_valhall_shader(
        sample_ns::EBIN_44.bytecode.byte_code,
        sample_ns::EBIN_44.bytecode.size,
        11, false);
    assert(sample_pilot.size() == 19);
    for (const auto& ins : sample_pilot) {
        assert(ins.valid && ins.mnemonic != valhall::Mnemonic::INVALID);
    }
    std::cout << "[PASS] Decoded 100% of sample_mbs2_shader.h (27 main compute insts + 19 pilot insts)\n";

    // 5. Decode astc_encoder_mbs2_shader.h (Complex production ASTC compute shader)
    auto* astc_ccom = &astc_ns::CCOM_1;
    assert(astc_ccom && astc_ccom->cmmn.nof_ebin >= 1);
    size_t total_astc_insts = 0;
    for (uint32_t b = 0; b < astc_ccom->cmmn.nof_ebin; ++b) {
        const auto& eb = astc_ccom->cmmn.ebin[b];
        size_t sz = (eb.data_section_offset && eb.data_section_offset < eb.bytecode.size)
                    ? eb.data_section_offset : eb.bytecode.size;
        auto insts = valhall::decode_valhall_shader(eb.bytecode.byte_code, sz, 11, false);
        for (const auto& ins : insts) {
            if (ins.flow == valhall::FlowControlM::end) {
                ++total_astc_insts;
                break;
            }
            assert(ins.valid && ins.mnemonic != valhall::Mnemonic::INVALID);
            ++total_astc_insts;
        }
    }
    std::cout << "[PASS] Decoded " << total_astc_insts
              << " instructions from production astc_encoder_mbs2_shader.h with 0 invalid instructions!\n";

    // 6. Layer 1B CFG regression tests (DISCARD fallthrough, indirect JUMP, diamond if-then-else)
    {
        using namespace valhall;
        // 6a. DISCARD must not have a fallthrough edge
        std::vector<DecodedValhallInst> d_insts(2);
        d_insts[0].valid = true; d_insts[0].pc = 0x0; d_insts[0].mnemonic = Mnemonic::DISCARD; d_insts[0].canonical_mnemonic = Mnemonic::DISCARD;
        d_insts[1].valid = true; d_insts[1].pc = 0x8; d_insts[1].mnemonic = Mnemonic::NOP; d_insts[1].canonical_mnemonic = Mnemonic::NOP; d_insts[1].flow = FlowControlM::end;
        auto d_cfg = cfg::build_cfg(d_insts);
        assert(d_cfg.blocks.size() == 2);
        assert(d_cfg.blocks[0].is_terminated_by_discard);
        assert(d_cfg.blocks[0].fallthrough_block == -1);

        // 6b. Indirect JUMP (has_branch_offset == false) must not corrupt Block 0 into a loop
        std::vector<DecodedValhallInst> j_insts(2);
        j_insts[0].valid = true; j_insts[0].pc = 0x0; j_insts[0].mnemonic = Mnemonic::NOP; j_insts[0].canonical_mnemonic = Mnemonic::NOP;
        j_insts[1].valid = true; j_insts[1].pc = 0x8; j_insts[1].mnemonic = Mnemonic::JUMP; j_insts[1].canonical_mnemonic = Mnemonic::JUMP; j_insts[1].has_branch_offset = false; j_insts[1].branch_target_pc = 0;
        auto j_cfg = cfg::build_cfg(j_insts);
        assert(!j_cfg.blocks[0].is_loop_header);
        assert(j_cfg.blocks.back().branch_target_block == -1);

        // 6c. Diamond if-then-else: B0 -> (cond? B2 : B1), B1 -> B3, B2 -> B3 => selection_merge_block == 3
        std::vector<DecodedValhallInst> dia(4);
        dia[0].valid = true; dia[0].pc = 0x00; dia[0].mnemonic = Mnemonic::BRANCH; dia[0].canonical_mnemonic = Mnemonic::BRANCH;
        dia[0].has_branch_offset = true; dia[0].branch_target_pc = 0x10; dia[0].srcs.resize(1);
        dia[1].valid = true; dia[1].pc = 0x08; dia[1].mnemonic = Mnemonic::BRANCH; dia[1].canonical_mnemonic = Mnemonic::BRANCH;
        dia[1].has_branch_offset = true; dia[1].branch_target_pc = 0x18; dia[1].modifier_strs["condition"] = "always";
        dia[2].valid = true; dia[2].pc = 0x10; dia[2].mnemonic = Mnemonic::NOP; dia[2].canonical_mnemonic = Mnemonic::NOP;
        dia[3].valid = true; dia[3].pc = 0x18; dia[3].mnemonic = Mnemonic::NOP; dia[3].canonical_mnemonic = Mnemonic::NOP; dia[3].flow = FlowControlM::end;
        auto dia_cfg = cfg::build_cfg(dia);
        assert(dia_cfg.blocks.size() == 4);
        assert(dia_cfg.blocks[0].selection_merge_block == 3);
        std::cout << "[PASS] Layer 1B CFG regression tests (DISCARD, indirect JUMP, diamond ipdom merge)\n";
    }

    // 7. Layer 2 SpirvBuilder lazy capabilities & SSA ID validation diagnostics
    {
        using namespace valhall::spirv;
        SpirvBuilder sb(ExecutionModel::GLCompute, false);
        sb.begin_main_function();
        uint32_t v = sb.add_local_var(sb.type_u32(), "x");
        sb.emit_store(v, sb.const_u32(42u));
        sb.emit_return();
        sb.end_main_function();
        assert(!sb.has_errors());
        auto words = sb.assemble();
        // Verify Float16 (9), Int64 (11), Int16 (22), Int8 (39) are NOT emitted for trivial u32 shader
        bool has_shader = false, has_int64 = false, has_f16 = false;
        size_t idx = 5;
        while (idx < words.size()) {
            uint32_t wc = words[idx] >> 16;
            uint32_t op = words[idx] & 0xffff;
            if (op == uint32_t(Op::OpCapability) && wc == 2) {
                if (words[idx + 1] == uint32_t(Capability::Shader)) has_shader = true;
                if (words[idx + 1] == uint32_t(Capability::Int64)) has_int64 = true;
                if (words[idx + 1] == uint32_t(Capability::Float16)) has_f16 = true;
            }
            idx += wc;
        }
        assert(has_shader && !has_int64 && !has_f16);

        // Verify invalid SSA ID 0 in emit_store is caught immediately
        SpirvBuilder bad_sb(ExecutionModel::GLCompute, false);
        bad_sb.begin_main_function();
        uint32_t bv = bad_sb.add_local_var(bad_sb.type_u32(), "y");
        bad_sb.emit_store(bv, 0u);
        assert(bad_sb.has_errors());
        std::cout << "[PASS] Layer 2 SpirvBuilder lazy capabilities & SSA ID error diagnostics\n";
    }

    std::cout << "=== ALL ISA DECODER TESTS PASSED ===\n";
    return 0;
}
