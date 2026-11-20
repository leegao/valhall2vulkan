#pragma once

#include "cmpbe_chunks.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace valhall::malioc_inproc {

struct InProcCompileOptions {
    std::string core = "Mali-G710";
    std::string revision = "r0p0";
    std::string shader_type = "compute";  // "compute", "vertex", "fragment", "geometry", "tessellation_control", "tessellation_evaluation"
    std::string entry_point = "main";
    std::string mali_lib_path = "mali_offline_compiler/graphics/libMali-Gxx_r55p0-00rel0.so";
    bool emit_c_header = false;           // true = also call cmpbe_v2_deserialize_MBS2_to_C in-process
    bool verbose = false;
};

struct InProcCompileResult {
    bool ok = false;
    std::string error_message;
    std::string mbs2_c_header;            // Populated when opts.emit_c_header == true
    size_t shader_count = 0;
};

/// Compile a SPIR-V binary directly in-process via libMali-Gxx_r55p0-00rel0.so,
/// invoking `on_mbs2` with the live `const cmpbe_chunk_MBS2*` struct pointer directly
/// in memory (zero subprocess, zero LD_PRELOAD, zero clang_ast_dumper!), and optionally
/// emitting the decompiled MBS2 C header string as well.
bool compile_spirv_in_process(
    const std::vector<uint32_t>& spirv_words,
    const InProcCompileOptions& opts,
    const std::function<void(const cmpbe_chunk_MBS2* mbs2)>& on_mbs2,
    InProcCompileResult* out,
    std::string* err = nullptr);

/// Check whether the in-process libMali-Gxx compiler library is loadable on this host.
bool is_inproc_malioc_available(
    const std::string& mali_lib_path = "mali_offline_compiler/graphics/libMali-Gxx_r55p0-00rel0.so");

}  // namespace valhall::malioc_inproc
