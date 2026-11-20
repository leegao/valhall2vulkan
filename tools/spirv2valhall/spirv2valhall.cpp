#include "spirv2valhall.h"

#ifdef HAVE_MALIOC_INPROC
#include "malioc_inproc.h"
#endif

#include <cstdio>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace valhall::spirv2valhall {

bool parse_mbs2_header_ebins(
    const std::string& header_text,
    uint8_t arch,
    bool disassemble,
    std::vector<ExtractedEbin>* out,
    std::string* err) {
    Mbs2AstParseOptions aopts;
    aopts.disassemble = disassemble;
    Mbs2Module dummy_mod;
    return parse_mbs2_header_with_clang_ast(header_text, arch, aopts, &dummy_mod, out, err);
}

bool extract_valhall_from_spirv(
    const std::string& spv_path,
    const ExtractOptions& opts,
    ExtractResult* out,
    std::string* err) {
    if (!out) {
        if (err) *err = "Null ExtractResult output pointer";
        return false;
    }
    *out = ExtractResult{};

    // 1. Inspect input SPIR-V & infer stage + arch
    std::vector<uint32_t> words;
    if (!read_spirv_words(spv_path, &words, err)) return false;
    if (!infer_spirv_stage(words, &out->stage_info, err)) return false;

    out->arch = (opts.arch != 0) ? opts.arch : infer_arch_from_core(opts.core);

    // 2. Optionally sanitize unused capabilities that malioc rejects
    std::vector<uint32_t> effective_words = words;
    if (opts.strip_unused_caps) {
        bool stripped = false;
        effective_words = strip_unused_malioc_capabilities(words, &stripped);
        out->stripped_caps = stripped;
    }

    if (opts.verbose) {
        std::cerr << "[spirv2valhall] stage=" << out->stage_info.malioc_flag
                  << " entry=\"" << out->stage_info.entry_point << "\""
                  << " core=" << opts.core << " arch=v" << static_cast<int>(out->arch)
                  << " stripped_caps=" << (out->stripped_caps ? "yes" : "no") << "\n";
    }

    bool parsed_direct = false;

#ifdef HAVE_MALIOC_INPROC
    // 3a. Fast direct in-process path: call libMali-Gxx directly and read live cmpbe_chunk_MBS2*!
    bool try_inproc = (opts.backend == ExtractBackend::InProcDirect) ||
                      (opts.backend == ExtractBackend::Auto &&
                       opts.clang_ast_dumper_path.empty() &&
                       malioc_inproc::is_inproc_malioc_available(opts.mali_lib_path));
    if (try_inproc) {
        malioc_inproc::InProcCompileOptions iopts;
        iopts.core = opts.core;
        iopts.shader_type = out->stage_info.malioc_flag;
        iopts.entry_point = out->stage_info.entry_point;
        iopts.mali_lib_path = opts.mali_lib_path;
        iopts.emit_c_header = !opts.dump_header_path.empty();
        iopts.verbose = opts.verbose;

        malioc_inproc::InProcCompileResult ires;
        std::string ierr;
        bool read_ok = false;
        std::string read_err;
        bool cok = malioc_inproc::compile_spirv_in_process(
            effective_words,
            iopts,
            [&](const cmpbe_chunk_MBS2* live_mbs2) {
                if (!read_ok && live_mbs2) {
                    read_ok = read_mbs2_struct_direct(
                        live_mbs2, out->arch, opts.disassemble, &out->mbs2, &out->ebins, &read_err);
                }
            },
            &ires,
            &ierr);

        if (cok && read_ok) {
            parsed_direct = true;
            out->used_inproc_direct = true;
            out->dump_header_text = std::move(ires.mbs2_c_header);
            if (!opts.dump_header_path.empty() && !out->dump_header_text.empty()) {
                std::ofstream ofs(opts.dump_header_path, std::ios::binary);
                ofs << out->dump_header_text;
            }
        } else if (opts.backend == ExtractBackend::InProcDirect) {
            if (err) *err = !cok ? ierr : read_err;
            return false;
        } else if (opts.verbose) {
            std::cerr << "[spirv2valhall] in-process malioc fallback (" << (!cok ? ierr : read_err) << ")\n";
        }
    }
#endif

    // 3b. Subprocess malioc + LD_PRELOAD=libmoc_hook.so -> in-process clang_ast_dumper_lib
    if (!parsed_direct) {
        std::string effective_spv = spv_path;
        std::string temp_spv_path;
        if (out->stripped_caps) {
            char tmpl[] = "/tmp/spirv2valhall_in_XXXXXX";
            int fd = mkstemp(tmpl);
            if (fd < 0) {
                if (err) *err = "Failed to create temporary sanitized SPIR-V file";
                return false;
            }
            close(fd);
            temp_spv_path = std::string(tmpl) + ".spv";
            std::remove(tmpl);
            std::ofstream ofs(temp_spv_path, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(effective_words.data()),
                      static_cast<std::streamsize>(effective_words.size() * sizeof(uint32_t)));
            ofs.close();
            effective_spv = temp_spv_path;
        }

        MocRunOptions mopts;
        mopts.core = opts.core;
        mopts.stage_flag = out->stage_info.malioc_flag;
        mopts.malioc_path = opts.malioc_path;
        mopts.moc_hook_path = opts.moc_hook_path;
        mopts.dump_header_path = opts.dump_header_path;
        mopts.verbose = opts.verbose;

        MocRunResult mres;
        bool moc_ok = run_malioc_with_hook(effective_spv, mopts, &mres, err);
        if (!temp_spv_path.empty()) std::remove(temp_spv_path.c_str());
        if (!moc_ok) return false;

        out->dump_header_text = std::move(mres.dump_header_text);
        out->malioc_output = std::move(mres.malioc_output);

        Mbs2AstParseOptions aopts;
        aopts.clang_ast_dumper_path = opts.clang_ast_dumper_path;
        aopts.disassemble = opts.disassemble;
        aopts.verbose = opts.verbose;
        if (!parse_mbs2_header_with_clang_ast(out->dump_header_text, out->arch, aopts,
                                              &out->mbs2, &out->ebins, err)) {
            return false;
        }
    }

    // 4. Analyze pilot shader(s) via decoupled src/dataflow pointer analysis
    out->pilot_analysis = analyze_pilot_shaders(out->mbs2, out->ebins);

    if (opts.verbose) {
        std::cerr << "[spirv2valhall] parsed MBS2 v" << out->mbs2.version
                  << " (" << out->mbs2.stage_chunk << ", backend="
                  << (out->used_inproc_direct ? "inproc-direct" : "moc+clang-ast") << "): "
                  << out->ebins.size() << " EBIN(s), pilot="
                  << (out->pilot_analysis.found_pilot ? "yes" : "no")
                  << ", pilot_fau_entries=" << out->pilot_analysis.fau_map.size() << "\n";
    }

    // 5. Optionally lift Valhall ISA back to SPIR-V via valhall2spirv (with MBS2 symbol bindings!)
    if (opts.lift_to_spirv) {
        std::string lift_err;
        if (!lift_ebins_to_spirv(out->stage_info, out->arch, out->mbs2, out->ebins,
                                 &out->lifted_spirv, &out->pilot_analysis, &lift_err)) {
            if (err) *err = lift_err;
            return false;
        }
    }

    return true;
}

std::string format_context(const ExtractResult& res) {
    return format_csf_execution_context(res.mbs2, res.pilot_analysis);
}

std::string format_disassembly(const ExtractResult& res, bool include_modifiers, bool enrich) {
    return format_enriched_disassembly(res.stage_info, res.arch, res.mbs2, res.ebins,
                                       res.pilot_analysis, include_modifiers, enrich);
}

}  // namespace valhall::spirv2valhall
