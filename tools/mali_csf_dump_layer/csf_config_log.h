#pragma once
// ============================================================================
// csf_config_log.h
// Runtime configuration (Android system properties + env vars) and structured
// diagnostic logging for VK_LAYER_MALI_csf_dump.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mali_csf_dump {

enum class LogLevel : int {
    Error = 0,
    Warn  = 1,
    Info  = 2,
    Debug = 3,
};

struct LayerConfig {
    bool enabled = true;
    uint32_t target_frame = 1u;
    std::string dump_dir;
    LogLevel log_level = LogLevel::Warn;
    uint32_t max_draws = 0u;           // 0 = unlimited draws/dispatches in target frame
    size_t max_shader_bytes = 65536u;  // Upper bound for dynamic Valhall shader scan
};

std::string get_config_str(const char* prop_name, const char* env_name, const char* def_val);
bool is_dump_enabled();
uint32_t get_target_frame();
std::string get_dump_dir();
LogLevel get_log_level();
uint32_t get_max_draws();
size_t get_max_shader_bytes();
LayerConfig read_layer_config();

// Structured log emitter (routes to __android_log_print on Android if available,
// else stderr) + records recent warnings/errors for post-mortem diagnostics.
void csf_log(LogLevel level, const char* component, const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;

// Diagnostics ring buffer snapshot for metadata.json / diagnostics.log
std::vector<std::string> get_recent_diagnostics();
void clear_recent_diagnostics();

} // namespace mali_csf_dump
