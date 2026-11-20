#pragma once

#include <string>
#include <vector>

namespace valhall::spirv2valhall {

struct MocRunOptions {
    std::string core = "Mali-G710";
    std::string stage_flag;         // e.g. "--compute", "--vertex", "--fragment"
    std::string malioc_path = "mali_offline_compiler/malioc";
    std::string moc_hook_path;      // empty = $SPIRV2VALHALL_MOC_HOOK or build default
    std::string dump_header_path;   // optional path to persist generated MBS2 C header
    bool verbose = false;
};

struct MocRunResult {
    std::string dump_header_text;
    std::string malioc_output;
    std::string resolved_moc_hook_path;
};

/// POSIX single-quote shell escaping helper.
std::string shell_quote(const std::string& s);

/// Resolve the path to libmoc_hook.so from options, environment, or CMake default.
std::string resolve_moc_hook_path(const std::string& override_path);

/// Invoke malioc under LD_PRELOAD=libmoc_hook.so on a SPIR-V file and return the dumped MBS2 C header.
bool run_malioc_with_hook(
    const std::string& spv_path,
    const MocRunOptions& opts,
    MocRunResult* out,
    std::string* err = nullptr);

}  // namespace valhall::spirv2valhall
