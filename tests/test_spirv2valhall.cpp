#include "spirv2valhall.h"
#include "json_lite.h"
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
    std::cout << "=== [Test] tools/spirv2valhall Modular, AST, Context, Enrichment, Pilot & --dump-spirv Tests ===\n";

    // 1. json_lite unit tests
    {
        JsonValue jv;
        std::string jerr;
        assert(parse_json(R"({"a": 42, "b": [true, null, "hi\n"], "c": {"k": -7}})", &jv, &jerr));
        assert(jv.is_object());
        assert(jv.get_int("a") == 42);
        assert(jv.get("b")->arr_val.size() == 3);
        assert(jv.get("b")->arr_val[2].str_val == "hi\n");
        assert(jv.get("c")->get_int("k") == -7);

        JsonValue bad;
        assert(!parse_json("{unterminated", &bad, &jerr));
        assert(!jerr.empty());
    }
    std::cout << "[PASS] json_lite parser verified\n";

    // 2. Arch inference, stage auto-inference, and unused-capability stripping
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

    // 3. Direct clang_ast_dumper MBS2 parsing, Pilot pointer chasing, Context dump & Enrichment on fixture
    {
        std::string fixture_hdr = read_text_file("tests/fixtures/sample_mbs2_shader.h");
        Mbs2Module mod;
        std::vector<ExtractedEbin> ebins;
        Mbs2AstParseOptions aopts;
        aopts.disassemble = true;
        std::string err;
        bool ok = parse_mbs2_header_with_clang_ast(fixture_hdr, 11, aopts, &mod, &ebins, &err);
        if (!ok) {
            std::cerr << "[FAIL] parse_mbs2_header_with_clang_ast: " << err << "\n";
            std::abort();
        }
        assert(mod.version == 48);
        assert(mod.stage_chunk == "CCOM");
        assert(mod.has_kern && mod.kern.name == "main");
        assert(mod.kern.wg_x == 8 && mod.kern.wg_y == 8 && mod.kern.wg_z == 1);
        assert(mod.cmmn.suni.size() == 1);
        assert(mod.cmmn.s3bo.size() == 3);
        assert(mod.cmmn.fcsts.size() == 1);
        assert(ebins.size() == 2);
        assert(ebins[0].role == EbinRole::MainCompute);
        assert(ebins[0].code_size == 216);
        assert(ebins[0].data_section.size() == 288);
        assert(ebins[1].role == EbinRole::Pilot);
        assert(ebins[1].code_size == 152);

        // Pilot SRT -> Resource -> Buffer -> FAU pointer chasing
        PilotAnalysisResult pilot = analyze_pilot_shaders(mod, ebins);
        assert(pilot.found_pilot);
        assert(pilot.pilot_ebin_index == 1);
        assert(pilot.fau_map.count(0x00) && pilot.fau_map[0x00].symbol_name == "_12__DEBUG_LINE_BUFFER_0");
        assert(pilot.fau_map.count(0x08) && pilot.fau_map[0x08].symbol_name == "_83_input_0");
        assert(pilot.fau_map.count(0x10) && pilot.fau_map[0x10].symbol_name == "_77_output_0");
        assert(pilot.fau_map.count(0x18) && pilot.fau_map[0x18].expression.find("u_TextureDimensions_0.x") != std::string::npos);

        // Context dump (everything minus raw EBIN bytecode)
        std::string ctx = format_csf_execution_context(mod, pilot);
        assert(ctx.find("CSF Execution Context & Descriptor Graph") != std::string::npos);
        assert(ctx.find("WG_SIZE=(8, 8, 1)") != std::string::npos);
        assert(ctx.find("_43_pushConstants_0") != std::string::npos);
        assert(ctx.find("_83_input_0") != std::string::npos);
        assert(ctx.find("SRT_0 + 0x40") != std::string::npos);
        assert(ctx.find("0x40501062") != std::string::npos);

        // Enriched disassembly + ABI clobber check
        auto main_notes = enrich_ebin_instructions(mod, ebins[0], pilot);
        assert(main_notes.size() == ebins[0].instructions.size());
        // PC 0x0018 (idx 3) reads r60 before clobber -> tagged GlobalInvocationID.x
        assert(main_notes[3].find("r60 = GlobalInvocationID.x") != std::string::npos);
        // PC 0x0068 (idx 13) is ADR r60, 0x100 -> tagged data_section+0x28 and clobbers r60!
        assert(main_notes[13].find("data_section+0x28") != std::string::npos);
        // PC 0x0088 (idx 17) reads clobbered r60 -> MUST NOT be tagged GlobalInvocationID.x, MUST be tagged data_section!
        assert(main_notes[17].find("GlobalInvocationID") == std::string::npos);
        assert(main_notes[17].find("data_section+0x28") != std::string::npos);
        // PC 0x00a0 (idx 20) reads u4.w1 -> tagged FCST 0x40501062
        assert(main_notes[20].find("0x40501062") != std::string::npos);

        SpirvStageInfo sinfo{ShaderStage::Compute, 5, "main", "--compute"};
        std::string dis = format_enriched_disassembly(sinfo, 11, mod, ebins, pilot, false, true);
        assert(dis.find("[MAIN COMPUTE SHADER]") != std::string::npos);
        assert(dis.find("[PILOT SHADER]") != std::string::npos);
        assert(dis.find("chase SRT[4]->Buf[0].address (_83_input_0)") != std::string::npos);
    }
    std::cout << "[PASS] clang_ast_dumper MBS2 parsing, Pilot chasing, CSF context dump, and ABI-clobber enrichment verified\n";

    // 4. End-to-end live extraction + --dump-spirv roundtrip on Compute, Vertex, and Fragment SPIR-V
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
        assert(res.used_inproc_direct);  // Default Auto uses direct in-process libMali-Gxx!
        if (expected_stage == ShaderStage::Fragment) {
            assert(res.lifted_spirv[0].disassembly.find("_11_v_col") != std::string::npos);
        }
        if (expected_stage == ShaderStage::Vertex) {
            assert(res.lifted_spirv[0].disassembly.find("_17_in_pos") != std::string::npos);
        }
        std::cout << "[PASS] In-process direct extracted & lifted " << expected_flag << " (" << res.ebins.size()
                  << " EBINs, " << total_insts << " Valhall insts, "
                  << res.lifted_spirv.size() << " symbol-named SPIR-V module(s))\n";
    };

    check_live(comp_spv, ShaderStage::Compute, "--compute");
    check_live(vert_spv, ShaderStage::Vertex, "--vertex");
    check_live(frag_spv, ShaderStage::Fragment, "--fragment");

    // 4b. Verify in-process C header emission + SubprocessAst (in-process libclang_ast_dumper_lib.a)
    {
        ExtractOptions inproc_c_opts;
        inproc_c_opts.core = "Mali-G710";
        inproc_c_opts.backend = ExtractBackend::InProcDirect;
        inproc_c_opts.dump_header_path = std::string(TEST_BUILD_DIR) + "/s2v_inproc_emit.h";
        ExtractResult r1;
        std::string err;
        assert(extract_valhall_from_spirv(comp_spv, inproc_c_opts, &r1, &err));
        assert(r1.used_inproc_direct);
        assert(r1.dump_header_text.find("cmpbe_chunk_MBS2 MBS2") != std::string::npos);

        ExtractOptions ast_opts;
        ast_opts.core = "Mali-G710";
        ast_opts.backend = ExtractBackend::SubprocessAst;
        ExtractResult r2;
        assert(extract_valhall_from_spirv(frag_spv, ast_opts, &r2, &err));
        assert(!r2.used_inproc_direct);
        assert(r2.mbs2.ast_global_count >= 20);
        assert(r2.ebins.size() == 1);
        std::cout << "[PASS] Verified both InProcDirect (live cmpbe_chunk_MBS2* + C header) and in-process clang_ast_dumper_lib\n";
    }

    // 5. Verify spirv2valhall CLI executable end-to-end with --dump-spirv & --out-spirv
    const std::string cli_hdr = std::string(TEST_BUILD_DIR) + "/s2v_cli_frag.h";
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
