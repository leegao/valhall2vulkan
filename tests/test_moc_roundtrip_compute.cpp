#include "valhall_isa_generated.h"
#include "valhall_csf_descriptors_generated.h"
#include "valhall_to_spirv.h"
#include "valhall_csf_pipeline.h"
#include "cmpbe_chunks.h"
#include "spirv2valhall.h"
#include "spirv_reflect.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace sample_ns {
#include "fixtures/sample_mbs2_shader.h"
}

using namespace valhall;
using namespace valhall::csf;
using namespace valhall::pipeline;

static std::vector<uint32_t> read_spirv_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    assert(ifs.good());
    size_t sz = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<uint32_t> words(sz / 4);
    ifs.read(reinterpret_cast<char*>(words.data()), sz);
    return words;
}

static void validate_with_spirv_reflect(const std::vector<uint32_t>& spirv_words, const char* label) {
    SpvReflectShaderModule mod{};
    SpvReflectResult res = spvReflectCreateShaderModule(
        spirv_words.size() * sizeof(uint32_t), spirv_words.data(), &mod);
    if (res != SPV_REFLECT_RESULT_SUCCESS) {
        std::cerr << "[FAIL] SPIRV-Reflect validation failed for " << label << " (code=" << res << ")\n";
        std::abort();
    }
    std::cout << "[PASS] SPIRV-Reflect validated " << label
              << " (entry=\"" << mod.entry_point_name << "\", stage=" << mod.shader_stage
              << ", words=" << spirv_words.size() << ")\n";
    spvReflectDestroyShaderModule(&mod);
}

int main() {
    std::cout << "=== [Test 3] End-to-End Compute Roundtrip (SPIR-V -> Malioc MSB2/SPD/SRT -> Valhall -> SPIR-V -> Lavapipe) ===\n";

    ValhallVulkanEngine vk;
    std::string err;
    if (!vk.init(&err)) {
        std::cerr << "Vulkan init failed: " << err << "\n";
        return 1;
    }
    std::cout << "[INFO] Initialized Vulkan device: " << vk.device_name() << "\n";

    // ------------------------------------------------------------------------
    // Step 1: Compile sample.comp with glslang -> Reference SPIR-V
    //         AND run malioc with libmoc_hook.so to verify live MBS2 extraction!
    // ------------------------------------------------------------------------
    const std::string sample_comp_path = "tests/fixtures/sample.comp";
#ifndef TEST_BUILD_DIR
#define TEST_BUILD_DIR "build"
#endif
#ifndef MOC_HOOK_PATH
#define MOC_HOOK_PATH "build/libmoc_hook.so"
#endif
    const std::string ref_spv_path = std::string(TEST_BUILD_DIR) + "/sample_ref.spv";
    const std::string glslang_bin = "mali_offline_compiler/external/glslang";
    const std::string malioc_bin  = "mali_offline_compiler/malioc";
    const std::string hook_so     = MOC_HOOK_PATH;
    const std::string live_dump_h = std::string(TEST_BUILD_DIR) + "/live_sample_mbs2.h";

    {
        std::string cmd1 = glslang_bin + " -V --target-env vulkan1.2 -S comp " + sample_comp_path + " -o " + ref_spv_path + " > /dev/null";
        int rc1 = std::system(cmd1.c_str());
        assert(rc1 == 0);

        valhall::spirv2valhall::ExtractOptions s2v_opts;
        s2v_opts.core = "Mali-G615";
        s2v_opts.malioc_path = malioc_bin;
        s2v_opts.moc_hook_path = hook_so;
        s2v_opts.dump_header_path = live_dump_h;

        valhall::spirv2valhall::ExtractResult s2v_res;
        std::string s2v_err;
        bool ok = valhall::spirv2valhall::extract_valhall_from_spirv(ref_spv_path, s2v_opts, &s2v_res, &s2v_err);
        if (!ok) {
            std::cerr << "[FAIL] spirv2valhall extraction failed: " << s2v_err << "\n";
            std::abort();
        }
        assert(s2v_res.stage_info.stage == valhall::spirv2valhall::ShaderStage::Compute);
        assert(!s2v_res.ebins.empty());
        assert(s2v_res.ebins[0].invalid_count == 0);

        std::ifstream f_live(live_dump_h);
        assert(f_live.good());
        std::cout << "[PASS] Compiled sample.comp to reference SPIR-V and extracted live MBS2 via tools/spirv2valhall ("
                  << s2v_res.ebins.size() << " EBINs, " << s2v_res.ebins[0].instructions.size() << " insts)\n";
    }

    auto ref_spirv = read_spirv_file(ref_spv_path);
    validate_with_spirv_reflect(ref_spirv, "Reference sample.comp SPIR-V");

    // ------------------------------------------------------------------------
    // Step 2: Execute Reference SPIR-V on Lavapipe
    //   Grid: 8x8 workgroup, u_TextureDimensions_0 = (64, 64) -> 16x16 threads active (256 elements)
    // ------------------------------------------------------------------------
    constexpr uint32_t kNumElems = 256;
    std::vector<uint32_t> host_input(kNumElems);
    for (uint32_t i = 0; i < kNumElems; ++i) {
        host_input[i] = i % 61u; // valid index into packed_trit_data_0[61]
    }
    std::vector<uint32_t> zero_out(kNumElems, 0u);
    std::vector<uint32_t> zero_dbg(64, 0u);
    int32_t push_dims[2] = { 64, 64 }; // tex_coord_0 = gid * 4 < 64 => gid.x < 16, gid.y < 16

    GpuBuffer ref_in_buf  = vk.alloc_buffer(kNumElems * sizeof(uint32_t), host_input.data(), kNumElems * sizeof(uint32_t));
    GpuBuffer ref_out_buf = vk.alloc_buffer(kNumElems * sizeof(uint32_t), zero_out.data(), kNumElems * sizeof(uint32_t));
    GpuBuffer ref_dbg_buf = vk.alloc_buffer(64 * sizeof(uint32_t), zero_dbg.data(), 64 * sizeof(uint32_t));

    // In sample.comp: binding 0 = input_0, binding 1 = output_0, binding 2 = _DEBUG_LINE_BUFFER_0
    bool ref_ok = vk.dispatch_spirv_compute(
        ref_spirv,
        { ref_in_buf.buffer, ref_out_buf.buffer, ref_dbg_buf.buffer },
        push_dims, sizeof(push_dims),
        2, 2, 1, &err);
    assert(ref_ok);
    std::cout << "[PASS] Executed Reference SPIR-V on Lavapipe (out[0]=" << ref_out_buf.words()[0]
              << ", out[1]=" << ref_out_buf.words()[1]
              << ", out[255]=" << ref_out_buf.words()[255]
              << ", dbg[49]=" << ref_dbg_buf.words()[49] << ")\n";

    // ------------------------------------------------------------------------
    // Step 3: Set up v11.xml Descriptors (SRT, Buffer descriptors, SPD, FAU, COMPUTESR)
    //         and execute the Translated Valhall Pilot Shader (EBIN_44) + Main Compute Shader (EBIN_42)!
    // ------------------------------------------------------------------------
    GpuBuffer val_in_buf  = vk.alloc_buffer(kNumElems * sizeof(uint32_t), host_input.data(), kNumElems * sizeof(uint32_t));
    GpuBuffer val_out_buf = vk.alloc_buffer(kNumElems * sizeof(uint32_t), zero_out.data(), kNumElems * sizeof(uint32_t));
    GpuBuffer val_dbg_buf = vk.alloc_buffer(64 * sizeof(uint32_t), zero_dbg.data(), 64 * sizeof(uint32_t));
    GpuBuffer val_pc_buf  = vk.alloc_buffer(sizeof(push_dims), push_dims, sizeof(push_dims));

    // Build v11.xml Buffer descriptors:
    //   1) Push Constants Buffer descriptor at push_desc_buf
    //   2) Descriptor Set 0 Buffer table at dset0_buf (Binding 0 = input_0, Binding 1 = output_0, Binding 2 = _DEBUG_LINE_BUFFER_0)
    Buffer push_desc{};
    push_desc.type = DescriptorType::Buffer;
    push_desc.size = sizeof(push_dims);
    push_desc.address = val_pc_buf.va();
    uint32_t push_desc_words[Buffer::kWordCount] = {0};
    push_desc.pack(push_desc_words);
    GpuBuffer push_desc_gpu = vk.alloc_buffer(sizeof(push_desc_words), push_desc_words, sizeof(push_desc_words));

    uint32_t dset0_words[3 * Buffer::kWordCount] = {0};
    {
        Buffer b0{};
        b0.type = DescriptorType::Buffer;
        b0.size = kNumElems * sizeof(uint32_t);
        b0.address = val_in_buf.va();
        b0.pack(dset0_words + 0 * Buffer::kWordCount);

        Buffer b1{};
        b1.type = DescriptorType::Buffer;
        b1.size = kNumElems * sizeof(uint32_t);
        b1.address = val_out_buf.va();
        b1.pack(dset0_words + 1 * Buffer::kWordCount);

        Buffer b2{};
        b2.type = DescriptorType::Buffer;
        b2.size = 64 * sizeof(uint32_t);
        b2.address = val_dbg_buf.va();
        b2.pack(dset0_words + 2 * Buffer::kWordCount);
    }
    GpuBuffer dset0_gpu = vk.alloc_buffer(sizeof(dset0_words), dset0_words, sizeof(dset0_words));

    // Build SRT:
    //   Offset +0x00: pointer to push_desc_gpu
    //   Offset +0x40: pointer to dset0_gpu
    uint64_t srt_qwords[16] = {0};
    srt_qwords[0] = push_desc_gpu.va(); // SRT + 0x00
    srt_qwords[8] = dset0_gpu.va();     // SRT + 0x40
    GpuBuffer srt_gpu = vk.alloc_buffer(sizeof(srt_qwords), srt_qwords, sizeof(srt_qwords));

    // Allocate FAU buffer (64 words = 256 bytes)
    GpuBuffer fau_gpu = vk.alloc_buffer(256);

    // 3a. Translate & Execute Pilot Shader (EBIN_44) on Lavapipe!
    bool pilot_ok = vk.execute_valhall_pilot_shader(
        sample_ns::EBIN_44.bytecode.byte_code,
        sample_ns::EBIN_44.bytecode.size,
        fau_gpu.va(),
        srt_gpu.va(),
        11,
        &err);
    if (!pilot_ok) {
        std::cerr << "Pilot shader execution failed: " << err << "\n";
        return 1;
    }

    // Verify that the translated Pilot Shader EBIN_44 populated FAU on the GPU!
    uint64_t fau_u0 = *reinterpret_cast<const uint64_t*>(fau_gpu.bytes() + 0x00);
    uint64_t fau_u1 = *reinterpret_cast<const uint64_t*>(fau_gpu.bytes() + 0x08);
    uint64_t fau_u2 = *reinterpret_cast<const uint64_t*>(fau_gpu.bytes() + 0x10);
    uint32_t fau_u3_w0 = *reinterpret_cast<const uint32_t*>(fau_gpu.bytes() + 0x18);
    assert(fau_u0 == val_dbg_buf.va());
    assert(fau_u1 == val_in_buf.va());
    assert(fau_u2 == val_out_buf.va());
    assert(fau_u3_w0 == 16u); // 64 / 4 = 16!
    std::cout << "[PASS] Translated Valhall Pilot Shader (EBIN_44) executed on Lavapipe and populated FAU from v11.xml SRT/Buffer descriptors (u3.w0=" << fau_u3_w0 << ")\n";

    // 3b. Apply MBS2 RLOC_10 (pushConstants at FAU + 0x1c) and FCST_40 (0x40501062 at FAU + 0x24)
    std::memcpy(fau_gpu.bytes() + sample_ns::RLOC_10.reg_offset, push_dims, sample_ns::RLOC_10.size);
    *reinterpret_cast<uint32_t*>(fau_gpu.bytes() + sample_ns::FCST_40.reg_offset) = sample_ns::FCST_40.value;

    // 3c. Upload full EBIN_42 bytecode (instructions + embedded packed_trit_data_0[61] LUT at 0x100) to GPU memory
    //     and build v11.xml ShaderProgram (SPD) descriptor!
    GpuBuffer shader_code_gpu = vk.alloc_buffer(
        sample_ns::EBIN_42.bytecode.size,
        sample_ns::EBIN_42.bytecode.byte_code,
        sample_ns::EBIN_42.bytecode.size);

    ShaderProgram spd{};
    spd.type = DescriptorType::Shader;
    spd.stage = csf::ShaderStage::Compute;
    spd.register_allocation = ShaderRegisterAllocation::V_32_Per_Thread;
    spd.binary = shader_code_gpu.va();
    uint32_t spd_words[ShaderProgram::kWordCount] = {0};
    spd.pack(spd_words);
    GpuBuffer spd_gpu = vk.alloc_buffer(sizeof(spd_words), spd_words, sizeof(spd_words));

    // 3d. Populate v11.xml COMPUTE SR staging register file & CS RUN_COMPUTE command
    CsfStagingRegisterFile comp_sr{};
    comp_sr.set64(COMPUTESR::SRT_0, srt_gpu.va());
    comp_sr.set64(COMPUTESR::FAU_0, fau_gpu.va() | (uint64_t(8) << 56));
    comp_sr.set64(COMPUTESR::SPD_0, spd_gpu.va());
    // WG_SIZE: local_size = (8, 8, 1) -> (7 << 0) | (7 << 10) | (0 << 20)
    comp_sr.set32(COMPUTESR::WG_SIZE, (7u << 0) | (7u << 10) | (0u << 20));
    comp_sr.set32(COMPUTESR::JOB_SIZE_X, 2u);
    comp_sr.set32(COMPUTESR::JOB_SIZE_Y, 2u);
    comp_sr.set32(COMPUTESR::JOB_SIZE_Z, 1u);

    CSRUNCOMPUTE run_cmd{};
    run_cmd.srt_select = 0;
    run_cmd.spd_select = 0;
    run_cmd.tsd_select = 0;
    run_cmd.fau_select = 0;

    std::vector<uint32_t> translated_spirv;
    std::string disasm;
    bool exec_ok = vk.execute_cs_run_compute(comp_sr, run_cmd, &translated_spirv, &disasm, &err);
    if (!exec_ok) {
        std::cerr << "execute_cs_run_compute failed: " << err << "\n";
        return 1;
    }
    validate_with_spirv_reflect(translated_spirv, "Translated Valhall Compute EBIN_42 SPIR-V");

    // ------------------------------------------------------------------------
    // Step 4: Verify 100% Bit-Exact Equivalence between Reference SPIR-V and Translated Valhall SPIR-V!
    // ------------------------------------------------------------------------
    for (uint32_t i = 0; i < kNumElems; ++i) {
        uint32_t expected = ref_out_buf.words()[i];
        uint32_t actual   = val_out_buf.words()[i];
        if (expected != actual) {
            std::cerr << "[FAIL] Mismatch at output[" << i << "]: expected=" << expected
                      << " actual=" << actual << "\n";
            return 1;
        }
    }
    // Also verify atomic debug line counters at indices 34, 38, 42, 46, 49
    for (uint32_t idx : {34u, 38u, 42u, 46u, 49u}) {
        uint32_t exp_cnt = ref_dbg_buf.words()[idx];
        uint32_t act_cnt = val_dbg_buf.words()[idx];
        assert(exp_cnt == act_cnt && exp_cnt == 256u);
    }
    std::cout << "[PASS] 100% BIT-EXACT EQUIVALENCE across all " << kNumElems
              << " compute outputs AND all 5 atomic debug counters (256/256 threads)!\n";
    std::cout << "=== ALL COMPUTE ROUNDTRIP TESTS PASSED ===\n";
    return 0;
}
