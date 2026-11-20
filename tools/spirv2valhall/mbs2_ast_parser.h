#pragma once

#include "mbs2_types.h"
#include "valhall_isa_generated.h"

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

struct ExtractedEbin {
    uint32_t index = 0;
    EbinRole role = EbinRole::Unknown;
    uint32_t variant_flags = 0;
    uint32_t code_size = 0;
    std::vector<uint8_t> bytecode;      // Executable Valhall ISA bytes (code_size)
    std::vector<uint8_t> data_section;  // Optional embedded constant/LUT section after data_section_offset
    Mbs2EbinMeta meta;
    std::vector<valhall::DecodedValhallInst> instructions;
    size_t invalid_count = 0;
};

struct Mbs2AstParseOptions {
    std::string clang_ast_dumper_path;  // empty = $SPIRV2VALHALL_CLANG_AST_DUMPER or CMake default
    std::string cmpbe_include_dir;      // empty = CMake default tools/spirv2valhall
    std::string clang_resource_dir;     // empty = $CLANG_RESOURCE_DIR or CMake default
    bool disassemble = true;
    bool verbose = false;
};

/// Resolve the path to clang_ast_dumper binary.
std::string resolve_clang_ast_dumper_path(const std::string& override_path = "");

/// Parse a moc_hook-dumped MBS2 C header via tools/clang_ast_dumper (--globals --strict),
/// populating both the full Mbs2Module metadata graph and decoded ExtractedEbin list.
bool parse_mbs2_header_with_clang_ast(
    const std::string& header_text,
    uint8_t arch,
    const Mbs2AstParseOptions& opts,
    Mbs2Module* out_module,
    std::vector<ExtractedEbin>* out_ebins,
    std::string* err = nullptr);

/// Parse an already-generated clang_ast_dumper JSON document into Mbs2Module + ExtractedEbins.
bool parse_mbs2_from_ast_json(
    const std::string& ast_json_text,
    uint8_t arch,
    bool disassemble,
    Mbs2Module* out_module,
    std::vector<ExtractedEbin>* out_ebins,
    std::string* err = nullptr);

}  // namespace valhall::spirv2valhall
