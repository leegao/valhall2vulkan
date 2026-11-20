#pragma once

#include "cmpbe_chunks.h"
#include "mbs2_ast_parser.h"
#include "mbs2_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

/// Convert a live in-memory `cmpbe_chunk_MBS2*` (obtained directly from libMali-Gxx
/// via `malioc_inproc`) into `Mbs2Module` and decoded `ExtractedEbin`s —
/// bypassing C serialization, `clang_ast_dumper`, and JSON parsing altogether.
bool read_mbs2_struct_direct(
    const cmpbe_chunk_MBS2* mbs2,
    uint8_t arch,
    bool disassemble,
    Mbs2Module* out_module,
    std::vector<ExtractedEbin>* out_ebins,
    std::string* err = nullptr);

}  // namespace valhall::spirv2valhall
