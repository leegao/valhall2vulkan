#include "spirv2valhall.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

static void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " <input.spv> [options]\n"
        << "  Automatically infers SPIR-V shader stage, invokes malioc + libmoc_hook.so,\n"
        << "  and extracts/disassembles Valhall ISA bytecode.\n\n"
        << "Options:\n"
        << "  -c, --core <CORE>       Mali GPU core (default: Mali-G710)\n"
        << "  -a, --arch <N>          Valhall arch version (default: inferred from core)\n"
        << "  -o, --out-header <FILE> Persist raw MBS2 C header (cmpbe_chunks.h format)\n"
        << "  --out-bin <PREFIX>      Write raw EBIN bytecode to <PREFIX>_0.bin, ...\n"
        << "  --malioc <PATH>         Path to malioc binary (default: mali_offline_compiler/malioc)\n"
        << "  --moc-hook <PATH>       Path to libmoc_hook.so\n"
        << "  --no-strip-caps         Do not strip unused Float64/Interlock capabilities\n"
        << "  --mods                  Include decoded modifier key=value pairs\n"
        << "  --quiet                 Do not print disassembly to stdout\n"
        << "  -v, --verbose           Print stage inference and malioc command diagnostics\n"
        << "  -h, --help              Show this help message\n";
}

int main(int argc, char** argv) {
    valhall::spirv2valhall::ExtractOptions opts;
    std::string spv_path;
    std::string out_bin_prefix;
    bool include_mods = false;
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
        } else if (arg == "--malioc" && i + 1 < argc) {
            opts.malioc_path = argv[++i];
        } else if (arg == "--moc-hook" && i + 1 < argc) {
            opts.moc_hook_path = argv[++i];
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

    if (!quiet) {
        std::cout << valhall::spirv2valhall::format_disassembly(res, include_mods);
    }
    return 0;
}
