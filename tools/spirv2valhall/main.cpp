#include "spirv2valhall.h"
#include "equiv_bundle.h"
#include "spirv2ssa.h"
#include "valhall_cfg.h"
#include "valhall_pass1.h"
#include "valhall_ssa_json.h"
#include "ssa_defau_devirt.h"
#include "ssa_fact_prover.h"
#include "ssa_opt_pipeline.h"
#include "isa_enricher.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

static void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " <input.spv> [options]\n"
        << "  Automatically infers SPIR-V shader stage, compiles in-process via\n"
        << "  libMali-Gxx (reading the live cmpbe_chunk_MBS2* directly), dumps the\n"
        << "  CSF Execution Context & Descriptor Graph, and prints MBS2-enriched\n"
        << "  Valhall disassembly (including Pilot Shaders for compute shaders).\n\n"
        << "Options:\n"
        << "  -c, --core <CORE>           Mali GPU core (default: Mali-G710)\n"
        << "  -a, --arch <N>              Valhall arch version (default: inferred from core)\n"
        << "  -o, --out-header <FILE>     Persist decompiled MBS2 C header (cmpbe_chunks.h format)\n"
        << "  --out-bin <PREFIX>          Write raw EBIN bytecode to <PREFIX>_0.bin, ...\n"
        << "  --dump-spirv                Lift Valhall ISA back to SPIR-V via valhall2spirv and dump\n"
        << "  --out-spirv <PREFIX>        Write lifted SPIR-V binary(ies) to <PREFIX>_0.spv, ...\n"
        << "  --dump-ssa                  Lower Valhall ISA to ValhallSSA and print SSA instead of ISA\n"
        << "  --dump-ssa-json <FILE>      Write lowered (and optionally optimized) ValhallSSA JSON to FILE\n"
        << "  --direct-ssa                Lower input .spv directly via spirv2ssa (bypassing libMali) and output SSA\n"
        << "  --dump-equiv-json <FILE>    Write dual ValhallSSA equivalence bundle JSON\n"
        << "  --opt-ssa                   Run verification SSA optimization pipeline on ValhallSSA / spirv2ssa\n"
        << "  --opt-fp-mode <strict|fast> FP algebraic canonicalization mode (default: strict)\n"
        << "  --no-fp-tile-rewrites       Disable declarative valhall_tiles.json FP rewrites in SSA opt\n"
        << "  --no-fp-cpp-rewrites        Disable procedural C++ FP rewrites in SSA opt\n"
        << "  --no-context                Omit the CSF Execution Context & Descriptor Graph header\n"
        << "  --context-only              Print only the CSF Execution Context & Descriptor Graph\n"
        << "  --no-enrich                 Print raw Valhall disassembly without MBS2/ABI/SRT tags\n"
        << "  --mali-lib <PATH>           Path to libMali-Gxx_r55p0-00rel0.so\n"
        << "  --no-strip-caps             Do not strip unused Float64/Interlock capabilities\n"
        << "  --mods                      Include decoded modifier key=value pairs\n"
        << "  --quiet                     Do not print text output to stdout\n"
        << "  -v, --verbose               Print stage inference and compilation diagnostics\n"
        << "  -h, --help                  Show this help message\n";
}

static valhall::ShaderStage to_compiler_stage(valhall::spirv2valhall::ShaderStage st) {
    using valhall::spirv2valhall::ShaderStage;
    switch (st) {
        case ShaderStage::Vertex: return valhall::ShaderStage::Vertex;
        case ShaderStage::Fragment: return valhall::ShaderStage::Fragment;
        case ShaderStage::Compute:
        default: return valhall::ShaderStage::Compute;
    }
}

int main(int argc, char** argv) {
    valhall::spirv2valhall::ExtractOptions opts;
    std::string spv_path;
    std::string out_bin_prefix;
    std::string out_spirv_prefix;
    std::string dump_equiv_json_path;
    std::string dump_ssa_json_path;
    valhall::spirv2valhall::EquivBundleOptions equiv_opts;
    bool dump_ssa = false;
    bool run_defau = false;
    bool dump_devirt_fallbacks = false;
    bool direct_ssa = false;
    bool include_mods = false;
    bool show_context = true;
    bool context_only = false;
    bool enrich = true;
    bool dump_spirv = false;
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-c" || arg == "--core") && i + 1 < argc) {
            opts.core = argv[++i];
        } else if ((arg == "-a" || arg == "--arch") && i + 1 < argc) {
            opts.arch = static_cast<uint8_t>(std::atoi(argv[++i]));
        } else if ((arg == "-o" || arg == "--out-header") && i + 1 < argc) {
            opts.dump_header_path = argv[++i];
        } else if (arg == "--out-bin" && i + 1 < argc) {
            out_bin_prefix = argv[++i];
        } else if (arg == "--dump-spirv") {
            dump_spirv = true;
            opts.lift_to_spirv = true;
        } else if (arg == "--out-spirv" && i + 1 < argc) {
            out_spirv_prefix = argv[++i];
            opts.lift_to_spirv = true;
        } else if (arg == "--dump-ssa") {
            dump_ssa = true;
        } else if (arg == "--dump-ssa-json" && i + 1 < argc) {
            dump_ssa_json_path = argv[++i];
        } else if (arg == "--direct-ssa") {
            direct_ssa = true;
        } else if (arg == "--dump-equiv-json" && i + 1 < argc) {
            dump_equiv_json_path = argv[++i];
        } else if (arg == "--opt-ssa") {
            equiv_opts.run_ssa_opt = true;
        } else if (arg == "--defau") {
            run_defau = true;
        } else if (arg == "--dump-devirt-fallbacks") {
            run_defau = true;
            dump_devirt_fallbacks = true;
        } else if (arg == "--opt-fp-mode" && i + 1 < argc) {
            std::string m = argv[++i];
            equiv_opts.fp_fast_math = (m == "fast" || m == "fast_math");
        } else if (arg == "--no-fp-tile-rewrites") {
            equiv_opts.enable_fp_tile_rewrites = false;
        } else if (arg == "--no-fp-cpp-rewrites") {
            equiv_opts.enable_fp_cpp_rewrites = false;
        } else if (arg == "--no-context") {
            show_context = false;
        } else if (arg == "--context-only") {
            context_only = true;
            show_context = true;
        } else if (arg == "--no-enrich") {
            enrich = false;
        } else if (arg == "--mali-lib" && i + 1 < argc) {
            opts.mali_lib_path = argv[++i];
        } else if (arg == "--no-strip-caps") {
            opts.strip_unused_caps = false;
        } else if (arg == "--mods") {
            include_mods = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        } else if (!arg.empty() && arg[0] != '-' && spv_path.empty()) {
            spv_path = arg;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    if (spv_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    // Direct SPIR-V -> ValhallSSA mode (bypasses libMali compilation).
    if (direct_ssa) {
        std::vector<uint32_t> spv_words;
        std::string rerr;
        if (!valhall::spirv2valhall::read_spirv_words(spv_path, &spv_words, &rerr)) {
            std::cerr << "[spirv2valhall] ERROR: " << rerr << "\n";
            return 1;
        }
        valhall::spirv2ssa::Spirv2SsaOptions sopts;
        sopts.tolerant = true;
        sopts.run_ssa_opt = equiv_opts.run_ssa_opt;
        sopts.ssa_opt_fp_fast_math = equiv_opts.fp_fast_math;
        sopts.ssa_opt_fp_tile_rewrites = equiv_opts.enable_fp_tile_rewrites;
        sopts.ssa_opt_fp_cpp_rewrites = equiv_opts.enable_fp_cpp_rewrites;

        auto dres = valhall::spirv2ssa::lower_spirv_to_valhall_ssa(spv_words, sopts);
        if (!dres.ok) {
            std::cerr << "[spirv2valhall] ERROR: direct spirv2ssa failed: " << dres.error_message << "\n";
            return 1;
        }
        if (!dump_ssa_json_path.empty()) {
            std::ofstream ofs(dump_ssa_json_path, std::ios::binary);
            if (!ofs.good()) {
                std::cerr << "[spirv2valhall] ERROR: Failed to write " << dump_ssa_json_path << "\n";
                return 1;
            }
            ofs << valhall::compiler::serialize_ssa_module_json(dres.ssa);
        }
        if (!quiet) {
            std::cout << valhall::compiler::format_ssa_module_text(dres.ssa);
        }
        return 0;
    }

    valhall::spirv2valhall::ExtractResult res;
    std::string err;
    if (!valhall::spirv2valhall::extract_valhall_from_spirv(spv_path, opts, &res, &err)) {
        std::cerr << "[spirv2valhall] ERROR: " << err << "\n";
        return 1;
    }

    if (!out_bin_prefix.empty()) {
        for (const auto& eb : res.ebins) {
            std::string bin_path = out_bin_prefix + "_" + std::to_string(eb.index) + ".bin";
            std::ofstream ofs(bin_path, std::ios::binary);
            if (!ofs.good()) {
                std::cerr << "[spirv2valhall] ERROR: Failed to write " << bin_path << "\n";
                return 1;
            }
            ofs.write(reinterpret_cast<const char*>(eb.bytecode.data()),
                      static_cast<std::streamsize>(eb.bytecode.size()));
        }
    }

    if (!out_spirv_prefix.empty()) {
        for (size_t i = 0; i < res.lifted_spirv.size(); ++i) {
            const auto& sh = res.lifted_spirv[i];
            if (!sh.ok || sh.spirv_words.empty()) continue;
            std::string out_path = out_spirv_prefix + "_" + std::to_string(i) + ".spv";
            std::ofstream ofs(out_path, std::ios::binary);
            if (!ofs.good()) {
                std::cerr << "[spirv2valhall] ERROR: Failed to write " << out_path << "\n";
                return 1;
            }
            ofs.write(reinterpret_cast<const char*>(sh.spirv_words.data()),
                      static_cast<std::streamsize>(sh.spirv_words.size() * sizeof(uint32_t)));
        }
    }

    if (!dump_equiv_json_path.empty()) {
        std::string bundle_json;
        std::string berr;
        if (!valhall::spirv2valhall::build_equivalence_bundle_json(spv_path, res, &bundle_json, &berr, equiv_opts)) {
            std::cerr << "[spirv2valhall] ERROR: " << berr << "\n";
            return 1;
        }
        std::ofstream ofs(dump_equiv_json_path, std::ios::binary);
        if (!ofs.good()) {
            std::cerr << "[spirv2valhall] ERROR: Failed to write " << dump_equiv_json_path << "\n";
            return 1;
        }
        ofs << bundle_json;
    }

    if (dump_ssa || !dump_ssa_json_path.empty()) {
        using valhall::spirv2valhall::EbinRole;
        using valhall::spirv2valhall::ExtractedEbin;
        using valhall::spirv2valhall::ShaderStage;
        const ExtractedEbin* primary_eb = nullptr;
        const ExtractedEbin* secondary_eb = nullptr;
        if (res.stage_info.stage == ShaderStage::Vertex) {
            for (const auto& eb : res.ebins) {
                if (eb.role == EbinRole::VertexVarying && !secondary_eb) secondary_eb = &eb;
                else if (!primary_eb) primary_eb = &eb;
            }
        } else {
            for (const auto& eb : res.ebins) {
                if (eb.role != EbinRole::Pilot) {
                    primary_eb = &eb;
                    break;
                }
            }
        }
        if (!primary_eb) {
            std::cerr << "[spirv2valhall] ERROR: No primary EBIN found for --dump-ssa\n";
            return 1;
        }

        valhall::ValhallTranslateOptions vopts;
        vopts.stage = to_compiler_stage(res.stage_info.stage);
        vopts.arch = res.arch;
        vopts.tolerant_ssa = true;
        vopts.run_ssa_opt = run_defau ? false : equiv_opts.run_ssa_opt;
        vopts.ssa_opt_fp_fast_math = equiv_opts.fp_fast_math;
        vopts.ssa_opt_fp_tile_rewrites = equiv_opts.enable_fp_tile_rewrites;
        vopts.ssa_opt_fp_cpp_rewrites = equiv_opts.enable_fp_cpp_rewrites;
        if (res.mbs2.has_kern && res.mbs2.kern.has_workgroup_size) {
            vopts.local_size_x = res.mbs2.kern.wg_x;
            vopts.local_size_y = res.mbs2.kern.wg_y;
            vopts.local_size_z = res.mbs2.kern.wg_z;
        }
        for (const auto& sym : res.mbs2.cmmn.sinp) {
            if (sym.rack_index == 11u && sym.location != 0xFFFFu) {
                vopts.frag_varying_addr_to_loc[sym.address] = sym.location;
            }
        }
        for (const auto& sym : res.mbs2.cmmn.sout) {
            if (sym.rack_index == 3u && !sym.name.empty()) {
                vopts.vert_sout_attr_names[sym.address] = sym.name;
            }
        }
        std::map<uint32_t, uint32_t> simg_addrs;
        for (const auto& sym : res.mbs2.cmmn.simg) {
            valhall::CombinedSamplerBindingSpec spec;
            spec.tex_index = sym.address;
            spec.descriptor_set = (sym.rack_index >= 68u) ? (sym.rack_index - 68u)
                                : (sym.rack_index >= 4u ? (sym.rack_index - 4u) : 0u);
            spec.binding = (sym.binding != 0xFFFFu) ? sym.binding : sym.address;
            vopts.samplers.push_back(spec);
            simg_addrs[sym.address] = spec.binding;
        }
        {
            std::map<std::pair<uint32_t, uint32_t>, std::string> pka_buf_by_rack_addr;
            auto add_pka = [&](const std::vector<valhall::spirv2valhall::Mbs2Symbol>& syms, bool is_ssbo) {
                for (const auto& sym : syms) {
                    std::string bname = sym.name;
                    if (!is_ssbo && sym.rack_index == 0u && sym.binding == 0xFFFFu) {
                        bname = "push_const";
                    } else if (sym.binding != 0xFFFFu) {
                        uint32_t d_set = (sym.rack_index >= 4u) ? (sym.rack_index - 4u) : 0u;
                        bname = std::string(is_ssbo ? "ssbo:" : "ubo:") +
                                std::to_string(d_set) + ":" + std::to_string(sym.binding);
                    }
                    pka_buf_by_rack_addr[{sym.rack_index, sym.address}] = std::move(bname);
                }
            };
            add_pka(res.mbs2.cmmn.suni, false);
            add_pka(res.mbs2.cmmn.s3bo, true);
            std::map<uint32_t, uint32_t> fcst_all;
            for (const auto& f : res.mbs2.cmmn.fcsts) fcst_all[f.reg_offset] = f.value;
            if (primary_eb->meta.has_dbug) {
                for (const auto& f : primary_eb->meta.dbug.fcsts) fcst_all[f.reg_offset] = f.value;
            }
            for (const auto& kv : fcst_all) {
                uint32_t rack = (kv.second >> 24u) & 0xFFu;
                uint32_t addr = kv.second & 0xFFFFu;
                auto it = pka_buf_by_rack_addr.find({rack, addr});
                if (it != pka_buf_by_rack_addr.end()) {
                    vopts.fau_pka_buf_names[kv.first] = it->second;
                }
            }
        }
        if (!simg_addrs.empty()) {
            std::map<uint32_t, uint32_t> fcst_map;
            for (const auto& f : res.mbs2.cmmn.fcsts) fcst_map[f.reg_offset] = f.value;
            if (primary_eb->meta.has_dbug) {
                for (const auto& f : primary_eb->meta.dbug.fcsts) fcst_map[f.reg_offset] = f.value;
            }
            for (const auto& kv : fcst_map) {
                if ((kv.first & 7u) != 0u) continue;
                auto hi_it = fcst_map.find(kv.first + 4u);
                if (hi_it == fcst_map.end()) continue;
                if (kv.second == 0xFFFFFFFFu) {
                    uint32_t desc_addr = hi_it->second & 0xFFFFu;
                    if (simg_addrs.count(desc_addr)) {
                        vopts.fau_tex_handle_indices[kv.first] = desc_addr;
                    }
                }
            }
        }

        valhall::cfg::ControlFlowGraph pri_cfg = valhall::cfg::build_cfg(primary_eb->instructions);
        valhall::cfg::ControlFlowGraph sec_cfg;
        if (secondary_eb) {
            sec_cfg = valhall::cfg::build_cfg(secondary_eb->instructions);
        }
        valhall::compiler::Pass1Result p1 = valhall::compiler::run_pass1_lowering(pri_cfg, sec_cfg, vopts);
        if (!p1.ok) {
            std::cerr << "[spirv2valhall] ERROR: valhall_pass1 failed: " << p1.error_message << "\n";
            return 1;
        }
        if (run_defau) {
            auto fau_summary = valhall::spirv2valhall::build_fau_summary_for_ebin(res.mbs2, res.ebins, res.arch, *primary_eb);
            auto df_res = valhall::dataflow::run_ssa_defau_devirt(p1.ssa, fau_summary);
            if (dump_devirt_fallbacks) {
                std::cerr << "[defau] consts=" << df_res.fau_consts_materialized
                          << " loads=" << df_res.fau_loads_materialized
                          << " exprs=" << df_res.fau_exprs_materialized
                          << " addrs=" << df_res.fau_addrs_materialized
                          << " fallbacks=" << df_res.fallbacks.size() << "\n";
                for (const auto& fb : df_res.fallbacks) {
                    std::cerr << "  [fallback] pc=0x" << std::hex << fb.valhall_pc
                              << " %" << std::dec << fb.ssa_id << " " << fb.detail << "\n";
                }
            }
            auto fp_pre = valhall::dataflow::run_ssa_fact_prover(p1.ssa);
            if (equiv_opts.run_ssa_opt) {
                valhall::dataflow::SsaOptPipelineOptions popts;
                popts.fp_mode = equiv_opts.fp_fast_math ? valhall::dataflow::FpOptMode::FastMath
                                                        : valhall::dataflow::FpOptMode::StrictIEEE;
                popts.enable_fp_tile_rewrites = equiv_opts.enable_fp_tile_rewrites;
                popts.enable_fp_cpp_rewrites = equiv_opts.enable_fp_cpp_rewrites;
                valhall::dataflow::run_ssa_opt_pipeline(p1.ssa, popts);
                auto fp_post = valhall::dataflow::run_ssa_fact_prover(p1.ssa);
                valhall::dataflow::run_ssa_dce(p1.ssa);
                if (dump_devirt_fallbacks) {
                    std::cerr << "[fact_prover] bitcasts=" << (fp_pre.bitcasts_eliminated + fp_post.bitcasts_eliminated)
                              << " isnan=" << (fp_pre.isnan_canonicalized + fp_post.isnan_canonicalized)
                              << " bool_masks=" << (fp_pre.bool_masks_unwrapped + fp_post.bool_masks_unwrapped)
                              << " bfi_bfe=" << (fp_pre.bfi_bfe_projected + fp_post.bfi_bfe_projected)
                              << " path_consts=" << (fp_pre.path_consts_folded + fp_post.path_consts_folded)
                              << " csel_ladders=" << (fp_pre.csel_ladders_collapsed + fp_post.csel_ladders_collapsed)
                              << "\n";
                }
            }
        }
        if (!dump_ssa_json_path.empty()) {
            std::ofstream ofs(dump_ssa_json_path, std::ios::binary);
            if (!ofs.good()) {
                std::cerr << "[spirv2valhall] ERROR: Failed to write " << dump_ssa_json_path << "\n";
                return 1;
            }
            ofs << valhall::compiler::serialize_ssa_module_json(p1.ssa);
        }
        if (dump_ssa && !quiet) {
            std::cout << valhall::compiler::format_ssa_module_text(p1.ssa);
        }
        if (dump_ssa) {
            return 0;
        }
    }

    if (!quiet) {
        if (show_context) {
            std::cout << valhall::spirv2valhall::format_context(res);
        }
        if (!context_only) {
            std::cout << valhall::spirv2valhall::format_disassembly(res, include_mods, enrich);
        }
        if (dump_spirv) {
            std::cout << valhall::spirv2valhall::format_lifted_spirv_dump(res.lifted_spirv);
        }
    }

    return 0;
}
