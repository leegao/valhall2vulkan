#include "spirv2valhall.h"

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
    std::string effective_spv = spv_path;
    std::string temp_spv_path;
    if (opts.strip_unused_caps) {
        bool stripped = false;
        std::vector<uint32_t> cleaned = strip_unused_malioc_capabilities(words, &stripped);
        out->stripped_caps = stripped;
        if (stripped) {
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
            ofs.write(reinterpret_cast<const char*>(cleaned.data()),
                      static_cast<std::streamsize>(cleaned.size() * sizeof(uint32_t)));
            ofs.close();
            effective_spv = temp_spv_path;
        }
    }

    if (opts.verbose) {
        std::cerr << "[spirv2valhall] stage=" << out->stage_info.malioc_flag
                  << " entry=\"" << out->stage_info.entry_point << "\""
                  << " core=" << opts.core << " arch=v" << static_cast<int>(out->arch)
                  << " stripped_caps=" << (out->stripped_caps ? "yes" : "no") << "\n";
    }

    // 3. Run malioc + libmoc_hook.so to dump MBS2 C header
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

    // 4. Parse full MBS2 graph + EBINs using tools/clang_ast_dumper
    Mbs2AstParseOptions aopts;
    aopts.clang_ast_dumper_path = opts.clang_ast_dumper_path;
    aopts.disassemble = opts.disassemble;
    aopts.verbose = opts.verbose;
    if (!parse_mbs2_header_with_clang_ast(out->dump_header_text, out->arch, aopts,
                                          &out->mbs2, &out->ebins, err)) {
        return false;
    }

    // 5. Analyze pilot shader(s) to chase SRT -> Resource -> Buffer -> FAU bindings
    out->pilot_analysis = analyze_pilot_shaders(out->mbs2, out->ebins);

    if (opts.verbose) {
        std::cerr << "[spirv2valhall] parsed MBS2 v" << out->mbs2.version
                  << " (" << out->mbs2.stage_chunk << "): "
                  << out->ebins.size() << " EBIN(s), pilot="
                  << (out->pilot_analysis.found_pilot ? "yes" : "no")
                  << ", pilot_fau_entries=" << out->pilot_analysis.fau_map.size() << "\n";
    }

    // 6. Optionally lift Valhall ISA back to SPIR-V via valhall2spirv
    if (opts.lift_to_spirv) {
        std::string lift_err;
        if (!lift_ebins_to_spirv(out->stage_info, out->arch, out->mbs2, out->ebins,
                                 &out->lifted_spirv, &lift_err)) {
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
