#include "spirv2valhall.h"
#include "malioc_inproc.h"

#include <fstream>
#include <iostream>

namespace valhall::spirv2valhall {

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

    // 3. Direct in-process compilation via libMali-Gxx (`malioc_inproc`) -> live cmpbe_chunk_MBS2*
    if (!malioc_inproc::is_inproc_malioc_available(opts.mali_lib_path)) {
        if (err) {
            *err = "In-process Mali offline compiler library not readable: " + opts.mali_lib_path +
                   " (specify --mali-lib <PATH> to libMali-Gxx_r55p0-00rel0.so)";
        }
        return false;
    }

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

    if (!cok || !read_ok) {
        if (err) *err = !cok ? ierr : read_err;
        return false;
    }

    out->dump_header_text = std::move(ires.mbs2_c_header);
    if (!opts.dump_header_path.empty() && !out->dump_header_text.empty()) {
        std::ofstream ofs(opts.dump_header_path, std::ios::binary);
        ofs << out->dump_header_text;
    }

    // 4. Analyze pilot shader(s) via decoupled SSA-level src/dataflow pointer analysis
    out->pilot_analysis = analyze_pilot_shaders(out->mbs2, out->ebins);

    if (opts.verbose) {
        std::cerr << "[spirv2valhall] parsed MBS2 v" << out->mbs2.version
                  << " (" << out->mbs2.stage_chunk << ", backend=inproc-direct): "
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
