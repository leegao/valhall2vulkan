#include "moc_runner.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>

namespace valhall::spirv2valhall {

std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

std::string resolve_moc_hook_path(const std::string& override_path) {
    if (!override_path.empty()) return override_path;
    const char* env_hook = std::getenv("SPIRV2VALHALL_MOC_HOOK");
    if (env_hook && env_hook[0] != '\0') return env_hook;
#ifdef SPIRV2VALHALL_DEFAULT_MOC_HOOK_PATH
    return SPIRV2VALHALL_DEFAULT_MOC_HOOK_PATH;
#else
    return "build/libmoc_hook.so";
#endif
}

bool run_malioc_with_hook(
    const std::string& spv_path,
    const MocRunOptions& opts,
    MocRunResult* out,
    std::string* err) {
    if (!out) {
        if (err) *err = "Null MocRunResult output pointer";
        return false;
    }
    *out = MocRunResult{};
    out->resolved_moc_hook_path = resolve_moc_hook_path(opts.moc_hook_path);

    std::string hdr_path = opts.dump_header_path;
    bool remove_hdr = false;
    if (hdr_path.empty()) {
        char tmpl[] = "/tmp/spirv2valhall_mbs2_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd < 0) {
            if (err) *err = "Failed to create temporary MBS2 header file";
            return false;
        }
        close(fd);
        hdr_path = tmpl;
        remove_hdr = true;
    }

    std::string cmd = "MOC_DUMP_PATH=" + shell_quote(hdr_path) +
                      " LD_PRELOAD=" + shell_quote(out->resolved_moc_hook_path) + " " +
                      shell_quote(opts.malioc_path) +
                      " --vulkan --spirv -c " + shell_quote(opts.core) + " " +
                      opts.stage_flag + " " +
                      shell_quote(spv_path) + " 2>&1";

    if (opts.verbose) {
        std::cerr << "[spirv2valhall/moc] cmd: " << cmd << "\n";
    }

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        if (remove_hdr) std::remove(hdr_path.c_str());
        if (err) *err = "Failed to launch malioc subprocess";
        return false;
    }
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        out->malioc_output += buf;
    }
    int rc = pclose(pipe);

    if (rc != 0) {
        if (remove_hdr) std::remove(hdr_path.c_str());
        if (err) {
            *err = "malioc failed (exit=" + std::to_string(rc) + ") for " + spv_path +
                   " (" + opts.stage_flag + "):\n" + out->malioc_output;
        }
        return false;
    }

    std::ifstream hfs(hdr_path);
    if (!hfs.good()) {
        if (remove_hdr) std::remove(hdr_path.c_str());
        if (err) *err = "moc_hook did not produce MBS2 dump at " + hdr_path;
        return false;
    }
    std::ostringstream oss;
    oss << hfs.rdbuf();
    out->dump_header_text = oss.str();
    hfs.close();

    if (remove_hdr) std::remove(hdr_path.c_str());
    return true;
}

}  // namespace valhall::spirv2valhall
