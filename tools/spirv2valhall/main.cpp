#include "spirv2valhall.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

static void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " <input.spv> [options]\n"
        << "  Automatically infers SPIR-V shader stage, invokes malioc + libmoc_hook.so,\n"
        << "  parses cmpbe_chunk_MBS2 via tools/clang_ast_dumper, dumps the CSF Execution\n"
        << "  Context & Descriptor Graph, and prints MBS2-enriched Valhall disassembly\n"
        << "  (including Pilot Shaders for compute shaders).\n\n"
        << "Options:\n"
        << "  -c, --core <CORE>           Mali GPU core (default: Mali-G710)\n"
        << "  -a, --arch <N>              Valhall arch version (default: inferred from core)\n"
        << "  -o, --out-header <FILE>     Persist raw MBS2 C header (cmpbe_chunks.h format)\n"
        << "  --out-bin <PREFIX>          Write raw EBIN bytecode to <PREFIX>_0.bin, ...\n"
        << "  --dump-spirv                Lift Valhall ISA back to SPIR-V via valhall2spirv and dump\n"
        << "  --out-spirv <PREFIX>        Write lifted SPIR-V binary(ies) to <PREFIX>_0.spv, ...\n"
        << "  --no-context                Omit the CSF Execution Context & Descriptor Graph header\n"
        << "  --context-only              Print only the CSF Execution Context & Descriptor Graph\n"
        << "  --no-enrich                 Print raw Valhall disassembly without MBS2/ABI/SRT tags\n"
        << "  --malioc <PATH>             Path to malioc binary (default: mali_offline_compiler/malioc)\n"
        << "  --moc-hook <PATH>           Path to libmoc_hook.so\n"
        << "  --clang-ast-dumper <PATH>   Path to clang_ast_dumper binary\n"
        << "  --no-strip-caps             Do not strip unused Float64/Interlock capabilities\n"
        << "  --mods                      Include decoded modifier key=value pairs\n"
        << "  --quiet                     Do not print text output to stdout\n"
        << "  -v, --verbose               Print stage inference, malioc, and AST dumper diagnostics\n"
        << "  -h, --help                  Show this help message\n";
}

int main(int argc, char** argv) {
    valhall::spirv2valhall::ExtractOptions opts;
    std::string spv_path;
    std::string out_bin_prefix;
    std::string out_spirv_prefix;
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
        } else if (arg == "--no-context") {
            show_context = false;
        } else if (arg == "--context-only") {
            context_only = true;
            show_context = true;
        } else if (arg == "--no-enrich") {
            enrich = false;
        } else if (arg == "--malioc" && i + 1 < argc) {
            opts.malioc_path = argv[++i];
        } else if (arg == "--moc-hook" && i + 1 < argc) {
            opts.moc_hook_path = argv[++i];
        } else if (arg == "--clang-ast-dumper" && i + 1 < argc) {
            opts.clang_ast_dumper_path = argv[++i];
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
