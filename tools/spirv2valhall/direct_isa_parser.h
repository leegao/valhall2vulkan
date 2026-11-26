#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

struct DirectIsaCase {
    std::string case_id;
    uint8_t arch = 11;
    std::string stage = "compute";  // "compute", "fragment", "vertex"
    std::vector<uint64_t> words;
    // Optional sideband FAU uniform overrides: (u_word_index, u32_value).
    std::vector<std::pair<uint32_t, uint32_t>> fau_words;
};

// Parse either a `.tsv` batch file (`case_id\tarch\tstage\tw0,w1,...[\tu0=0x...,u1=0x...]`)
// or a `.valhall` text stream (`@case`, `@arch`, `@stage`, `@fau`, and 64-bit hex words).
bool parse_direct_isa_file(const std::string& path,
                           std::vector<DirectIsaCase>* out_cases,
                           std::string* out_err);

// Decode each DirectIsaCase via `decode_valhall_shader`, build CFG, lower via
// `run_pass1_lowering`, and write one JSON result line per case to stdout.
int run_direct_isa_file(const std::string& path);

}  // namespace valhall::spirv2valhall
