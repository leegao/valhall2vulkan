#include "csf_config_log.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <mutex>

namespace mali_csf_dump {

namespace {

using PropGetFn = int (*)(const char*, char*);
using AndroidLogPrintFn = int (*)(int, const char*, const char*, ...);

constexpr size_t kMaxDiagnosticsEntries = 256u;
struct DiagGlobals {
    std::mutex mu;
    std::deque<std::string> ring;
};

DiagGlobals& diag_globals() {
    static DiagGlobals g;
    return g;
}

const char* level_tag(LogLevel lvl) {
    switch (lvl) {
        case LogLevel::Error: return "ERROR";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Debug: return "DEBUG";
    }
    return "INFO";
}

#ifdef __ANDROID__
int to_android_prio(LogLevel lvl) {
    // ANDROID_LOG_DEBUG=3, INFO=4, WARN=5, ERROR=6
    switch (lvl) {
        case LogLevel::Error: return 6;
        case LogLevel::Warn:  return 5;
        case LogLevel::Info:  return 4;
        case LogLevel::Debug: return 3;
    }
    return 4;
}
#endif

} // namespace

std::string get_config_str(const char* prop_name, const char* env_name, const char* def_val) {
    static PropGetFn sys_prop_get =
        reinterpret_cast<PropGetFn>(dlsym(RTLD_DEFAULT, "__system_property_get"));
    if (sys_prop_get && prop_name) {
        char buf[256] = {0};
        if (sys_prop_get(prop_name, buf) > 0 && buf[0] != '\0') {
            return std::string(buf);
        }
    }
    if (env_name) {
        if (const char* e = std::getenv(env_name)) {
            if (e[0] != '\0') return std::string(e);
        }
    }
    return def_val ? std::string(def_val) : std::string();
}

bool is_dump_enabled() {
    std::string v = get_config_str("debug.mali.csf_dump.enable", "MALI_CSF_DUMP_ENABLE", "1");
    return v == "1" || v == "true" || v == "TRUE" || v == "yes";
}

uint32_t get_target_frame() {
    std::string v = get_config_str("debug.mali.csf_dump.frame", "MALI_CSF_DUMP_FRAME", "1");
    char* endp = nullptr;
    unsigned long f = std::strtoul(v.c_str(), &endp, 10);
    return (endp && endp != v.c_str()) ? static_cast<uint32_t>(f) : 1u;
}

std::string get_dump_dir() {
#ifdef __ANDROID__
    const char* def_dir = "/data/local/tmp/mali_csf_dump";
#else
    const char* def_dir = "/tmp/mali_csf_dump";
#endif
    return get_config_str("debug.mali.csf_dump.dir", "MALI_CSF_DUMP_DIR", def_dir);
}

LogLevel get_log_level() {
    std::string v = get_config_str("debug.mali.csf_dump.log_level", "MALI_CSF_DUMP_LOG_LEVEL", "1");
    if (v == "0" || v == "error" || v == "ERROR") return LogLevel::Error;
    if (v == "1" || v == "warn"  || v == "WARN")  return LogLevel::Warn;
    if (v == "2" || v == "info"  || v == "INFO")  return LogLevel::Info;
    if (v == "3" || v == "debug" || v == "DEBUG" || v == "trace") return LogLevel::Debug;
    return LogLevel::Warn;
}

uint32_t get_max_draws() {
    std::string v = get_config_str("debug.mali.csf_dump.max_draws", "MALI_CSF_DUMP_MAX_DRAWS", "0");
    char* endp = nullptr;
    unsigned long n = std::strtoul(v.c_str(), &endp, 10);
    return (endp && endp != v.c_str()) ? static_cast<uint32_t>(n) : 0u;
}

size_t get_max_shader_bytes() {
    std::string v = get_config_str("debug.mali.csf_dump.max_shader_bytes",
                                   "MALI_CSF_DUMP_MAX_SHADER_BYTES", "65536");
    char* endp = nullptr;
    unsigned long n = std::strtoul(v.c_str(), &endp, 10);
    if (endp && endp != v.c_str() && n >= 64ul && n <= (16ul << 20)) {
        return static_cast<size_t>(n);
    }
    return 65536u;
}

LayerConfig read_layer_config() {
    LayerConfig cfg;
    cfg.enabled = is_dump_enabled();
    cfg.target_frame = get_target_frame();
    cfg.dump_dir = get_dump_dir();
    cfg.log_level = get_log_level();
    cfg.max_draws = get_max_draws();
    cfg.max_shader_bytes = get_max_shader_bytes();
    return cfg;
}

void csf_log(LogLevel level, const char* component, const char* fmt, ...) {
    char msg[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    const char* comp = component ? component : "core";
    const char* tag = level_tag(level);

    if (level == LogLevel::Error || level == LogLevel::Warn) {
        std::lock_guard<std::mutex> lk(diag_globals().mu);
        if (diag_globals().ring.size() >= kMaxDiagnosticsEntries) {
            diag_globals().ring.pop_front();
        }
        diag_globals().ring.emplace_back(std::string("[") + tag + "][" + comp + "] " + msg);
    }

    if (static_cast<int>(level) > static_cast<int>(get_log_level())) {
        return;
    }

#ifdef __ANDROID__
    static AndroidLogPrintFn android_log =
        reinterpret_cast<AndroidLogPrintFn>(dlsym(RTLD_DEFAULT, "__android_log_print"));
    if (android_log) {
        android_log(to_android_prio(level), "MaliCsfDump", "[%s] %s", comp, msg);
        return;
    }
#endif
    std::fprintf(stderr, "[MALI_CSF_DUMP][%s][%s] %s\n", tag, comp, msg);
}

std::vector<std::string> get_recent_diagnostics() {
    std::lock_guard<std::mutex> lk(diag_globals().mu);
    return std::vector<std::string>(diag_globals().ring.begin(), diag_globals().ring.end());
}

void clear_recent_diagnostics() {
    std::lock_guard<std::mutex> lk(diag_globals().mu);
    diag_globals().ring.clear();
}

} // namespace mali_csf_dump
