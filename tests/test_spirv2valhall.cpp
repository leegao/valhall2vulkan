#include "spirv2valhall.h"
#include "cmpbe_chunks.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
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

static std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path);
    assert(ifs.good());
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

int main() {
    std::cout << "=== [Test] tools/spirv2valhall In-Process Direct, SSA Enrichment, Pilot & --dump-spirv Tests ===\n";

    // 1. Arch inference, stage auto-inference, and unused-capability stripping
    assert(infer_arch_from_core("Mali-G78") == 9);
    assert(infer_arch_from_core("Mali-G710") == 10);
    assert(infer_arch_from_core("Mali-G615") == 11);
    assert(infer_arch_from_core("Mali-G720") == 12);
    assert(infer_arch_from_core("Immortalis-G925") == 14);

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

    // 2. End-to-end live in-process extraction + SSA enrichment + --dump-spirv on Compute, Vertex, Fragment
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
    opts.lift_to_spirv = true;

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
        if (expected_stage == ShaderStage::Compute) {
            assert(res.pilot_analysis.found_pilot);
            assert(!res.pilot_analysis.fau_map.empty());
            assert(res.lifted_spirv.size() >= 2);  // Main Compute + Pilot
        } else {
            assert(!res.lifted_spirv.empty());
        }
        for (const auto& lsh : res.lifted_spirv) {
            assert(lsh.ok);
            assert(lsh.reflect_valid);
            assert(lsh.disassembly.find("OpEntryPoint") != std::string::npos);
            assert(lsh.disassembly.find("OpFunctionEnd") != std::string::npos);
        }
        if (expected_stage == ShaderStage::Fragment) {
            assert(res.lifted_spirv[0].disassembly.find("_11_v_col") != std::string::npos);
        }
        if (expected_stage == ShaderStage::Vertex) {
            assert(res.lifted_spirv[0].disassembly.find("_17_in_pos") != std::string::npos);
        }

        // Verify enriched disassembly runs cleanly via SSA-level pointer analysis
        std::string dis = format_disassembly(res, false, true);
        assert(!dis.empty());
        if (expected_stage == ShaderStage::Compute) {
            assert(dis.find("[MAIN COMPUTE SHADER]") != std::string::npos);
            assert(dis.find("[PILOT SHADER]") != std::string::npos);
            assert(dis.find("gl_GlobalInvocationID") != std::string::npos);
        }
        std::cout << "[PASS] In-process direct extracted & lifted " << expected_flag << " (" << res.ebins.size()
                  << " EBINs, " << total_insts << " Valhall insts, "
                  << res.lifted_spirv.size() << " symbol-named SPIR-V module(s))\n";
    };

    check_live(comp_spv, ShaderStage::Compute, "--compute");
    check_live(vert_spv, ShaderStage::Vertex, "--vertex");
    check_live(frag_spv, ShaderStage::Fragment, "--fragment");

    // 3. Verify optional in-process MBS2 C header emission & clean bailout when libMali-Gxx is missing
    {
        ExtractOptions c_opts;
        c_opts.core = "Mali-G710";
        c_opts.dump_header_path = std::string(TEST_BUILD_DIR) + "/s2v_inproc_emit.h";
        ExtractResult r1;
        std::string err;
        assert(extract_valhall_from_spirv(comp_spv, c_opts, &r1, &err));
        assert(r1.dump_header_text.find("cmpbe_chunk_MBS2 MBS2") != std::string::npos);

        ExtractOptions bad_opts;
        bad_opts.mali_lib_path = "/nonexistent/libMali-Gxx_missing.so";
        ExtractResult r_bad;
        std::string bad_err;
        assert(!extract_valhall_from_spirv(comp_spv, bad_opts, &r_bad, &bad_err));
        assert(bad_err.find("not readable") != std::string::npos);
        std::cout << "[PASS] Verified in-process MBS2 C header emission & clean bailout on missing libMali-Gxx\n";
    }

    // 4. Verify spirv2valhall CLI executable end-to-end with --dump-spirv & --out-spirv
    const std::string cli_hdr = std::string(TEST_BUILD_DIR) + "/s2v_cli_comp.h";
    const std::string cli_spv_pfx = std::string(TEST_BUILD_DIR) + "/s2v_cli_lifted";
    const std::string cli_out_txt = std::string(TEST_BUILD_DIR) + "/s2v_cli_comp_out.txt";
    std::string cli_cmd = std::string(SPIRV2VALHALL_BIN_PATH) + " " + comp_spv +
                          " -c Mali-G710 -o " + cli_hdr +
                          " --dump-spirv --out-spirv " + cli_spv_pfx +
                          " > " + cli_out_txt;
    int cli_rc = std::system(cli_cmd.c_str());
    assert(cli_rc == 0);
    std::string cli_out = read_text_file(cli_out_txt);
    assert(cli_out.find("CSF Execution Context & Descriptor Graph") != std::string::npos);
    assert(cli_out.find("[MAIN COMPUTE SHADER]") != std::string::npos);
    assert(cli_out.find("[PILOT SHADER]") != std::string::npos);
    assert(cli_out.find("Lifted SPIR-V Dump (via valhall2spirv)") != std::string::npos);
    assert(cli_out.find("spirv-reflect=PASS") != std::string::npos);
    std::ifstream spv_out0(cli_spv_pfx + "_0.spv", std::ios::binary);
    assert(spv_out0.good());
    std::cout << "[PASS] spirv2valhall CLI verified end-to-end with context dump, pilot shader, and --dump-spirv\n";

    cmpbe_chunk_EBIN dummy_ebin{};
    (void)dummy_ebin;
    return 0;
}
