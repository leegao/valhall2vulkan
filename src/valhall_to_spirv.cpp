#include "valhall_to_spirv.h"
#include "valhall_pass1.h"
#include "valhall_tiler.h"
#include <sstream>
#include <iomanip>

namespace valhall {

ValhallTranslationResult translate_valhall_to_spirv(
    const uint8_t* code_bytes,
    size_t code_size_bytes,
    const ValhallTranslateOptions& opts) {
    ValhallTranslationResult res;
    if (!code_bytes || code_size_bytes < 8) {
        res.error_message = "Empty or truncated Valhall instruction stream";
        return res;
    }

    auto raw_insts = decode_valhall_shader(code_bytes, code_size_bytes, opts.arch, opts.fau32);
    res.primary_cfg = cfg::build_cfg(raw_insts);

    std::ostringstream dis_oss;
    dis_oss << "// Stage: " << int(opts.stage) << " (" << res.primary_cfg.instructions.size() << " executable instructions)\n";
    for (const auto& ins : res.primary_cfg.instructions) {
        dis_oss << "0x" << std::hex << std::setw(4) << std::setfill('0') << ins.pc
                << ": " << ins.disassembly << "\n";
    }

    if (opts.secondary_vertex_code && opts.secondary_vertex_code_size >= 8) {
        auto sec_raw = decode_valhall_shader(opts.secondary_vertex_code, opts.secondary_vertex_code_size, opts.arch, opts.fau32);
        res.secondary_cfg = cfg::build_cfg(sec_raw);
        dis_oss << "// Secondary Vertex Varying EBIN (" << res.secondary_cfg.instructions.size() << " instructions)\n";
        for (const auto& ins : res.secondary_cfg.instructions) {
            dis_oss << "0x" << std::hex << std::setw(4) << std::setfill('0') << ins.pc
                    << ": " << ins.disassembly << "\n";
        }
    }
    res.disassembly_listing = dis_oss.str();

    // Pass 1: Spec-driven lowering (valhall_lowering.json) -> ValhallSSA (Pass 0 sub-block split + mem2reg)
    auto pass1 = compiler::run_pass1_lowering(res.primary_cfg, res.secondary_cfg, opts);
    if (!pass1.ok) {
        res.ok = false;
        res.error_message = pass1.error_message;
        return res;
    }

    // Pass 2: Bounded DP DAG tiler (valhall_tiles.json) + EffectSet lattice join + single-pass SPIR-V emission
    auto pass2 = compiler::run_pass2_tiler_and_emit(pass1, opts);
    if (!pass2.ok) {
        res.ok = false;
        res.error_message = pass2.error_message;
        return res;
    }

    res.spirv_words = std::move(pass2.spirv_words);
    res.ok = true;
    return res;
}

} // namespace valhall
