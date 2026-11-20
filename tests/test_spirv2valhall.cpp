#include "spirv2valhall.h"
#include "cmpbe_chunks.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace valhall::spirv2valhall;

#ifndef TEST_BUILD_DIR
#define TEST_BUILD_DIR "build"
#endif
#ifndef SPIRV2VALHALL_BIN_PATH
#define SPIRV2VALHALL_BIN_PATH "build/spirv2valhall"
#endif

static std::vector<uint32_t> make_synthetic_spv(uint32_t exec_model, bool declare_f64, bool use_f64) {
    std::vector<uint32_t> w = {
        0x07230203u,  // Magic
        0x00010300u,  // Version 1.3
        0u,           // Generator
        16u,          // Bound
        0u,           // Schema
    };
    // OpCapability Shader (1)
    w.push_back((2u << 16) | 17u);
    w.push_back(1u);
    if (declare_f64) {
        // OpCapability Float64 (10)
        w.push_back((2u << 16) | 17u);
        w.push_back(10u);
    }
    // OpEntryPoint <exec_model> %1 "main"
    w.push_back((5u << 16) | 15u);
    w.push_back(exec_model);
    w.push_back(1u);
    w.push_back(0x6E69616Du);  // "main"
    w.push_back(0x00000000u);  // NUL terminator
    if (use_f64) {
        // OpTypeFloat %2 64
        w.push_back((3u << 16) | 22u);
        w.push_back(2u);
        w.push_back(64u);
    }
    return w;
}

static bool compile_glsl_to_spv(const std::string& stage, const std::string& glsl, const std::string& out_spv) {
    std::string src_path = out_spv + "." + stage;
    {
        std::ofstream ofs(src_path);
        ofs << glsl;
    }
    std::string cmd = "mali_offline_compiler/external/glslang -V --target-env vulkan1.2 -S " +
                      stage + " " + src_path + " -o " + out_spv + " > /dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    std::remove(src_path.c_str());
    return rc == 0;
}

int main() {
    std::cout << "=== [Test] tools/spirv2valhall Unit & End-to-End Extraction Tests ===\n";

    // 1. Arch inference from core names
    assert(infer_arch_from_core("Mali-G78") == 9);
    assert(infer_arch_from_core("Mali-G710") == 10);
    assert(infer_arch_from_core("Mali-G615") == 11);
    assert(infer_arch_from_core("Mali-G720") == 12);
    assert(infer_arch_from_core("Immortalis-G925") == 14);

    // 2. Stage auto-inference on synthetic SPIR-V streams + error paths
    {
        SpirvStageInfo info{};
        std::string err;
        assert(infer_spirv_stage(make_synthetic_spv(0, false, false), &info, &err));
        assert(info.stage == ShaderStage::Vertex && info.malioc_flag == "--vertex" && info.entry_point == "main");

        assert(infer_spirv_stage(make_synthetic_spv(4, false, false), &info, &err));
        assert(info.stage == ShaderStage::Fragment && info.malioc_flag == "--fragment");

        assert(infer_spirv_stage(make_synthetic_spv(5, false, false), &info, &err));
        assert(info.stage == ShaderStage::Compute && info.malioc_flag == "--compute");

        std::vector<uint32_t> bad_magic = {0xDEADBEEFu, 0, 0, 0, 0};
        assert(!infer_spirv_stage(bad_magic, &info, &err));
    }

    // 3. Unused capability stripping (strip when OpTypeFloat 64 absent, preserve when present)
    {
        bool stripped = false;
        auto unused_f64 = make_synthetic_spv(5, true, false);
        auto cleaned = strip_unused_malioc_capabilities(unused_f64, &stripped);
        assert(stripped);
        assert(cleaned.size() == unused_f64.size() - 2);

        bool stripped2 = false;
        auto used_f64 = make_synthetic_spv(5, true, true);
        auto preserved = strip_unused_malioc_capabilities(used_f64, &stripped2);
        assert(!stripped2);
        assert(preserved.size() == used_f64.size());
    }
    std::cout << "[PASS] Stage auto-inference, arch inference, and unused-capability stripping verified\n";

    // 4. End-to-end live extraction via spirv2valhall_lib on Compute, Vertex, and Fragment SPIR-V
    const std::string comp_glsl = R"(
#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) buffer Buf { float data[]; } buf;
void main() {
    uint i = gl_GlobalInvocationID.x;
    buf.data[i] = buf.data[i] * 2.0 + 1.0;
}
)";
    const std::string vert_glsl = R"(
#version 450
layout(location = 0) in vec4 in_pos;
void main() {
    gl_Position = in_pos * 2.0;
}
)";
    const std::string frag_glsl = R"(
#version 450
layout(location = 0) in vec4 v_col;
layout(location = 0) out vec4 out_col;
void main() {
    out_col = clamp(v_col * 0.5 + 0.25, 0.0, 1.0);
}
)";

    const std::string comp_spv = std::string(TEST_BUILD_DIR) + "/s2v_test_comp.spv";
    const std::string vert_spv = std::string(TEST_BUILD_DIR) + "/s2v_test_vert.spv";
    const std::string frag_spv = std::string(TEST_BUILD_DIR) + "/s2v_test_frag.spv";
    assert(compile_glsl_to_spv("comp", comp_glsl, comp_spv));
    assert(compile_glsl_to_spv("vert", vert_glsl, vert_spv));
    assert(compile_glsl_to_spv("frag", frag_glsl, frag_spv));

    ExtractOptions opts;
    opts.core = "Mali-G710";

    auto check_live = [&](const std::string& spv, ShaderStage expected_stage, const std::string& expected_flag) {
        ExtractResult res;
        std::string err;
        bool ok = extract_valhall_from_spirv(spv, opts, &res, &err);
        if (!ok) {
            std::cerr << "[FAIL] extract_valhall_from_spirv(" << spv << "): " << err << "\n";
            std::abort();
        }
        assert(res.stage_info.stage == expected_stage);
        assert(res.stage_info.malioc_flag == expected_flag);
        assert(res.arch == 10);
        assert(!res.ebins.empty());
        size_t total_insts = 0;
        for (const auto& eb : res.ebins) {
            assert(eb.invalid_count == 0);
            total_insts += eb.instructions.size();
        }
        assert(total_insts > 0);
        std::cout << "[PASS] Extracted " << expected_flag << " (" << res.ebins.size()
                  << " EBINs, " << total_insts << " valid Valhall insts)\n";
    };

    check_live(comp_spv, ShaderStage::Compute, "--compute");
    check_live(vert_spv, ShaderStage::Vertex, "--vertex");
    check_live(frag_spv, ShaderStage::Fragment, "--fragment");

    // 5. Verify spirv2valhall CLI executable end-to-end (no stage flag passed!)
    const std::string cli_hdr = std::string(TEST_BUILD_DIR) + "/s2v_cli_frag.h";
    std::string cli_cmd = std::string(SPIRV2VALHALL_BIN_PATH) + " " + frag_spv +
                          " -c Mali-G720 -o " + cli_hdr + " --quiet";
    int cli_rc = std::system(cli_cmd.c_str());
    assert(cli_rc == 0);
    std::ifstream ifs(cli_hdr);
    assert(ifs.good());
    std::cout << "[PASS] spirv2valhall CLI extracted Mali-G720 fragment shader with auto-inferred stage\n";

    // Confirm cmpbe_chunks.h from tools/spirv2valhall/ is directly usable
    cmpbe_chunk_EBIN dummy_ebin{};
    (void)dummy_ebin;
    return 0;
}
