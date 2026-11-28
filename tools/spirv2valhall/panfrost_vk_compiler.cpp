#include "panfrost_vk_compiler.h"
#include "panfrost_vk_compile.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <unordered_map>

namespace fs = std::filesystem;

namespace valhall::spirv2valhall {

namespace {

bool read_file_bytes(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        if (err) *err = "Failed to open file: " + path;
        return false;
    }
    out->assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return true;
}

bool read_file_text(const std::string& path, std::string* out, std::string* err) {
    std::ifstream ifs(path);
    if (!ifs) {
        if (err) *err = "Failed to open file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    *out = ss.str();
    return true;
}

/// Minimal SPIR-V metadata needed to complement `panfrost_vk_compile --info-json`
/// (specifically compute LocalSize and SSBO vs. UBO storage class per (set, binding)).
struct SpvMeta {
    std::string entry_point = "main";
    valhall::ShaderStage stage = valhall::ShaderStage::Compute;
    uint32_t local_size[3] = {1, 1, 1};
    /// Maps ((set << 16) | binding) -> storage class (12 = StorageBuffer, 2 = Uniform).
    std::unordered_map<uint64_t, uint32_t> binding_storage_class;
    std::vector<uint32_t> vert_input_locations;
};

bool parse_spv_metadata(const std::vector<uint32_t>& words, SpvMeta* out, std::string* err) {
    if (words.size() < 5 || words[0] != 0x07230203u) {
        if (err) *err = "Invalid SPIR-V magic/header";
        return false;
    }
    bool found_ep = false;
    std::unordered_map<uint32_t, uint32_t> desc_set;
    std::unordered_map<uint32_t, uint32_t> binding;
    std::unordered_map<uint32_t, uint32_t> var_sc;
    std::unordered_map<uint32_t, uint32_t> ptr_to_pointee;
    std::unordered_map<uint32_t, bool> is_buffer_block;
    std::unordered_map<uint32_t, uint32_t> var_location;
    std::unordered_map<uint32_t, bool> is_builtin;
    std::unordered_map<uint32_t, uint32_t> var_pointee;
    std::unordered_map<uint32_t, uint32_t> type_slots;
    std::unordered_map<uint32_t, uint32_t> const_u32_val;

    size_t i = 5;
    while (i < words.size()) {
        uint16_t wc = static_cast<uint16_t>(words[i] >> 16);
        uint16_t op = static_cast<uint16_t>(words[i] & 0xFFFFu);
        if (wc == 0 || i + wc > words.size()) break;
        if (op == 15 && wc >= 4 && !found_ep) {  // OpEntryPoint
            uint32_t model = words[i + 1];
            if (model == 0) {
                out->stage = valhall::ShaderStage::Vertex;
            } else if (model == 4) {
                out->stage = valhall::ShaderStage::Fragment;
            } else if (model == 5) {
                out->stage = valhall::ShaderStage::Compute;
            } else {
                if (err) {
                    *err = "Unsupported SPIR-V execution model " + std::to_string(model) +
                           " for Panfrost Vulkan compiler (supported: Vertex, Fragment, GLCompute)";
                }
                return false;
            }
            const char* name_ptr = reinterpret_cast<const char*>(&words[i + 3]);
            size_t max_len = (wc - 3) * 4;
            size_t len = strnlen(name_ptr, max_len);
            out->entry_point.assign(name_ptr, len);
            found_ep = true;
        } else if (op == 16 && wc >= 6 && words[i + 2] == 17) {  // OpExecutionMode LocalSize
            out->local_size[0] = words[i + 3];
            out->local_size[1] = words[i + 4];
            out->local_size[2] = words[i + 5];
        } else if (op == 71 && wc >= 3) {  // OpDecorate
            uint32_t target = words[i + 1];
            uint32_t dec = words[i + 2];
            if (dec == 34 && wc >= 4) desc_set[target] = words[i + 3];  // DescriptorSet
            else if (dec == 33 && wc >= 4) binding[target] = words[i + 3];  // Binding
            else if (dec == 3) is_buffer_block[target] = true;  // BufferBlock (legacy SSBO in Uniform SC)
            else if (dec == 30 && wc >= 4) var_location[target] = words[i + 3];  // Location
            else if (dec == 11) is_builtin[target] = true;  // BuiltIn
        } else if (op == 43 && wc >= 4) {  // OpConstant
            const_u32_val[words[i + 2]] = words[i + 3];
        } else if (op == 24 && wc >= 4) {  // OpTypeMatrix
            type_slots[words[i + 1]] = words[i + 3];
        } else if (op == 28 && wc >= 4) {  // OpTypeArray
            uint32_t elem_slots = type_slots.count(words[i + 2]) ? type_slots[words[i + 2]] : 1u;
            uint32_t len = const_u32_val.count(words[i + 3]) ? const_u32_val[words[i + 3]] : 1u;
            if (len > 0u && len <= 64u) type_slots[words[i + 1]] = elem_slots * len;
        } else if (op == 32 && wc >= 4) {  // OpTypePointer
            ptr_to_pointee[words[i + 1]] = words[i + 3];
        } else if (op == 59 && wc >= 4) {  // OpVariable
            uint32_t res_type = words[i + 1];
            uint32_t res_id = words[i + 2];
            uint32_t sc = words[i + 3];
            if (sc == 2 && ptr_to_pointee.count(res_type) && is_buffer_block[ptr_to_pointee[res_type]]) {
                sc = 12;  // Uniform + BufferBlock -> SSBO
            }
            var_sc[res_id] = sc;
            if (ptr_to_pointee.count(res_type)) var_pointee[res_id] = ptr_to_pointee[res_type];
        }
        i += wc;
    }
    if (!found_ep) {
        if (err) *err = "No OpEntryPoint found in SPIR-V module";
        return false;
    }
    for (const auto& [var_id, sc] : var_sc) {
        if (sc != 12 && sc != 2) continue;
        auto bit = binding.find(var_id);
        if (bit == binding.end()) continue;
        uint32_t ds = desc_set.count(var_id) ? desc_set[var_id] : 0u;
        uint32_t bd = bit->second;
        uint64_t sb_key = (static_cast<uint64_t>(ds) << 32) | bd;
        out->binding_storage_class[sb_key] = sc;
    }
    if (out->stage == valhall::ShaderStage::Vertex) {
        std::set<uint32_t> locs;
        for (const auto& [var_id, sc] : var_sc) {
            if (sc != 1u || is_builtin[var_id]) continue;  // Input storage class
            auto lit = var_location.find(var_id);
            if (lit == var_location.end()) continue;
            uint32_t slots = 1u;
            auto pit = var_pointee.find(var_id);
            if (pit != var_pointee.end() && type_slots.count(pit->second)) {
                slots = type_slots[pit->second];
            }
            for (uint32_t s = 0; s < slots; ++s) locs.insert(lit->second + s);
        }
        out->vert_input_locations.assign(locs.begin(), locs.end());
    }
    return true;
}

/// Lightweight JSON DOM sufficient to parse `panfrost_vk_compile --info-json`.
struct JsonVal {
    enum class Kind { Null, Bool, Int, Str, Array, Object };
    Kind kind = Kind::Null;
    bool bval = false;
    int64_t ival = 0;
    std::string sval;
    std::vector<JsonVal> arr;
    std::map<std::string, JsonVal> obj;

    const JsonVal* get(const std::string& k) const {
        if (kind != Kind::Object) return nullptr;
        auto it = obj.find(k);
        return (it != obj.end()) ? &it->second : nullptr;
    }
    uint32_t as_u32(uint32_t def = 0) const {
        return (kind == Kind::Int && ival >= 0) ? static_cast<uint32_t>(ival) : def;
    }
    bool as_bool(bool def = false) const {
        return (kind == Kind::Bool) ? bval : def;
    }
    std::string as_str(const std::string& def = "") const {
        return (kind == Kind::Str) ? sval : def;
    }
};

class MiniJsonParser {
public:
    explicit MiniJsonParser(const std::string& s) : s_(s) {}

    bool parse(JsonVal* out) {
        skip_ws();
        if (!parse_val(out)) return false;
        skip_ws();
        return pos_ == s_.size();
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) {
            ++pos_;
        }
    }

    bool match(const char* lit) {
        size_t n = std::strlen(lit);
        if (pos_ + n <= s_.size() && s_.compare(pos_, n, lit) == 0) {
            pos_ += n;
            return true;
        }
        return false;
    }

    bool parse_string(std::string* out) {
        if (pos_ >= s_.size() || s_[pos_] != '"') return false;
        ++pos_;
        out->clear();
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') return true;
            if (c == '\\' && pos_ < s_.size()) {
                char e = s_[pos_++];
                switch (e) {
                    case '"': out->push_back('"'); break;
                    case '\\': out->push_back('\\'); break;
                    case '/': out->push_back('/'); break;
                    case 'n': out->push_back('\n'); break;
                    case 'r': out->push_back('\r'); break;
                    case 't': out->push_back('\t'); break;
                    default: out->push_back(e); break;
                }
            } else {
                out->push_back(c);
            }
        }
        return false;
    }

    bool parse_number(JsonVal* out) {
        size_t start = pos_;
        if (pos_ < s_.size() && s_[pos_] == '-') ++pos_;
        while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        if (pos_ < s_.size() && (s_[pos_] == '.' || s_[pos_] == 'e' || s_[pos_] == 'E')) {
            while (pos_ < s_.size() &&
                   (std::isalnum(static_cast<unsigned char>(s_[pos_])) ||
                    s_[pos_] == '.' || s_[pos_] == '+' || s_[pos_] == '-')) {
                ++pos_;
            }
        }
        if (pos_ == start) return false;
        out->kind = JsonVal::Kind::Int;
        out->ival = std::strtoll(s_.substr(start, pos_ - start).c_str(), nullptr, 10);
        return true;
    }

    bool parse_val(JsonVal* out) {
        skip_ws();
        if (pos_ >= s_.size()) return false;
        char c = s_[pos_];
        if (c == '{') {
            ++pos_;
            out->kind = JsonVal::Kind::Object;
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == '}') {
                ++pos_;
                return true;
            }
            while (true) {
                skip_ws();
                std::string key;
                if (!parse_string(&key)) return false;
                skip_ws();
                if (pos_ >= s_.size() || s_[pos_] != ':') return false;
                ++pos_;
                JsonVal val;
                if (!parse_val(&val)) return false;
                out->obj[std::move(key)] = std::move(val);
                skip_ws();
                if (pos_ < s_.size() && s_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                if (pos_ < s_.size() && s_[pos_] == '}') {
                    ++pos_;
                    return true;
                }
                return false;
            }
        } else if (c == '[') {
            ++pos_;
            out->kind = JsonVal::Kind::Array;
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ']') {
                ++pos_;
                return true;
            }
            while (true) {
                JsonVal val;
                if (!parse_val(&val)) return false;
                out->arr.push_back(std::move(val));
                skip_ws();
                if (pos_ < s_.size() && s_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                if (pos_ < s_.size() && s_[pos_] == ']') {
                    ++pos_;
                    return true;
                }
                return false;
            }
        } else if (c == '"') {
            out->kind = JsonVal::Kind::Str;
            return parse_string(&out->sval);
        } else if (match("true")) {
            out->kind = JsonVal::Kind::Bool;
            out->bval = true;
            return true;
        } else if (match("false")) {
            out->kind = JsonVal::Kind::Bool;
            out->bval = false;
            return true;
        } else if (match("null")) {
            out->kind = JsonVal::Kind::Null;
            return true;
        } else if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parse_number(out);
        }
        return false;
    }
};

valhall::ShaderStage parse_stage_str(const std::string& s, valhall::ShaderStage fallback) {
    if (s == "vertex" || s == "vert") return valhall::ShaderStage::Vertex;
    if (s == "fragment" || s == "frag") return valhall::ShaderStage::Fragment;
    if (s == "compute" || s == "comp") return valhall::ShaderStage::Compute;
    return fallback;
}

/// Invoke the statically linked `panfrost_vk_compile()` directly in-process,
/// capturing stderr into `stderr_out`. Because Mesa's `pan_use_kraid()` caches
/// `PAN_USE_KRAID` once per process via `DEBUG_GET_ONCE_FLAGS_OPTION`, we track
/// the latched backend and only fork a child if a single process switches
/// between Kraid and Bifrost mid-process. HACKY
bool run_panfrost_vk_static(const std::string& spv_path,
                            const std::string& out_bin,
                            const std::string& out_json,
                            const PanfrostVkCompileOptions& opts,
                            int* rc_out,
                            int* term_sig_out,
                            std::string* stderr_out,
                            std::string* err) {
    struct panfrost_vk_compile_options c_opts {};
    c_opts.in_path = spv_path.c_str();
    c_opts.out_path = out_bin.c_str();
    c_opts.arch = (opts.arch != 0) ? opts.arch : 11u;
    c_opts.entry_override = nullptr;
    c_opts.info_json_path = out_json.c_str();
    c_opts.stage_override = -1;
    c_opts.robust = opts.robust;
    c_opts.no_idvs = opts.no_idvs;
    c_opts.promote_imm = !opts.no_promote_imm;
    c_opts.clamp_version = !opts.no_clamp_version;
    c_opts.dump_nir = false;
    c_opts.disasm = false;
    c_opts.core_max_id = opts.core_max_id;

    // Resolve desired PAN_USE_KRAID value ("all" for Kraid/Auto, "0" for Bifrost).
    const bool want_bifrost = (opts.backend == PanfrostVkBackend::Bifrost);
    const char* desired_env = want_bifrost ? "0" : "all";

    static bool has_latched = false;
    static bool latched_bifrost = false;

    const bool need_fork_for_backend_switch =
        has_latched && (latched_bifrost != want_bifrost);

    if (need_fork_for_backend_switch) {
        int err_pipe[2] = {-1, -1};
        if (::pipe(err_pipe) != 0) {
            if (err) *err = "pipe() failed for static panfrost_vk_compile backend switch";
            return false;
        }
        pid_t pid = ::fork();
        if (pid < 0) {
            ::close(err_pipe[0]);
            ::close(err_pipe[1]);
            if (err) *err = "fork() failed for static panfrost_vk_compile backend switch";
            return false;
        }
        if (pid == 0) {
            ::close(err_pipe[0]);
            ::dup2(err_pipe[1], STDERR_FILENO);
            ::close(err_pipe[1]);
            ::setenv("PAN_USE_KRAID", desired_env, 1);
            int rc = ::panfrost_vk_compile(&c_opts);
            std::fflush(stderr);
            ::_exit(rc == 0 ? 0 : 1);
        }
        ::close(err_pipe[1]);
        stderr_out->clear();
        char buf[512];
        ssize_t n = 0;
        while ((n = ::read(err_pipe[0], buf, sizeof(buf))) > 0) {
            stderr_out->append(buf, static_cast<size_t>(n));
        }
        ::close(err_pipe[0]);
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        *rc_out = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        *term_sig_out = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
        return true;
    }

    if (!has_latched) {
        if (opts.backend != PanfrostVkBackend::Auto) {
            ::setenv("PAN_USE_KRAID", desired_env, 1);
        } else if (!std::getenv("PAN_USE_KRAID")) {
            ::setenv("PAN_USE_KRAID", "all", 1);
        }
        has_latched = true;
        latched_bifrost = want_bifrost;
    }

    // Redirect stderr to a temp file during the in-process call so Mesa diagnostics
    // are captured cleanly without polluting JSON stdout/stderr.
    std::string err_path = out_json + ".stderr";
    std::fflush(stderr);
    int saved_stderr = ::dup(STDERR_FILENO);
    int err_fd = ::open(err_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (saved_stderr >= 0 && err_fd >= 0) {
        ::dup2(err_fd, STDERR_FILENO);
        ::close(err_fd);
    } else if (err_fd >= 0) {
        ::close(err_fd);
    }

    int rc = ::panfrost_vk_compile(&c_opts);
    std::fflush(stderr);

    if (saved_stderr >= 0) {
        ::dup2(saved_stderr, STDERR_FILENO);
        ::close(saved_stderr);
    }
    read_file_text(err_path, stderr_out, nullptr);
    if (opts.verbose && !stderr_out->empty()) {
        std::cerr << *stderr_out;
    }

    *rc_out = rc;
    *term_sig_out = 0;
    return true;
}

}  // namespace

PanfrostVkCompileOutput compile_vulkan_spv_via_panfrost(
    const std::string& spv_path,
    const PanfrostVkCompileOptions& opts) {
    PanfrostVkCompileOutput out;
    out.arch = (opts.arch != 0) ? opts.arch : 11;

    std::vector<uint8_t> raw_bytes;
    if (!read_file_bytes(spv_path, &raw_bytes, &out.error)) {
        return out;
    }
    if (raw_bytes.size() < 20 || (raw_bytes.size() % 4) != 0) {
        out.error = "Invalid SPIR-V binary size: " + spv_path;
        return out;
    }
    std::vector<uint32_t> spv_words(raw_bytes.size() / 4);
    std::memcpy(spv_words.data(), raw_bytes.data(), raw_bytes.size());

    SpvMeta meta;
    if (!parse_spv_metadata(spv_words, &meta, &out.error)) {
        return out;
    }
    out.entry_point = meta.entry_point;
    out.stage = meta.stage;
    out.local_size[0] = meta.local_size[0];
    out.local_size[1] = meta.local_size[1];
    out.local_size[2] = meta.local_size[2];
    out.vert_input_locations = meta.vert_input_locations;

    char tmpl[] = "/tmp/panfrost_vk_XXXXXX";
    char* td_c = ::mkdtemp(tmpl);
    if (!td_c) {
        out.error = "mkdtemp() failed for panfrost_vk_compile";
        return out;
    }
    fs::path td(td_c);
    auto cleanup = [&]() {
        std::error_code ec;
        fs::remove_all(td, ec);
    };

    fs::path out_bin = td / "shader.bin";
    fs::path out_json = td / "shader_info.json";

    if (opts.verbose) {
        std::cerr << "[panfrost-vk] running static panfrost_vk_compile on " << spv_path << "\n";
    }

    int exit_code = -1;
    int term_sig = 0;
    std::string stderr_text;
    if (!run_panfrost_vk_static(spv_path, out_bin.string(), out_json.string(),
                                opts, &exit_code, &term_sig, &stderr_text, &out.error)) {
        cleanup();
        return out;
    }

    if (term_sig != 0) {
        out.error = "panfrost_vk_compile crashed with signal " + std::to_string(term_sig);
        if (!stderr_text.empty()) out.error += ": " + stderr_text;
        cleanup();
        return out;
    }
    if (exit_code != 0) {
        out.error = "panfrost_vk_compile failed (rc=" + std::to_string(exit_code) + ")";
        if (!stderr_text.empty()) out.error += ": " + stderr_text;
        cleanup();
        return out;
    }

    if (!read_file_bytes(out_bin.string(), &out.ebin_bytes, &out.error) || out.ebin_bytes.empty()) {
        if (out.error.empty()) out.error = "panfrost_vk_compile produced empty binary";
        cleanup();
        return out;
    }

    std::string info_text;
    if (!read_file_text(out_json.string(), &info_text, &out.error)) {
        cleanup();
        return out;
    }

    if (opts.verbose) {
        std::cerr << "[panfrost-vk] info_json:\n" << info_text << "\n";
    }
    JsonVal root;
    MiniJsonParser parser(info_text);
    if (!parser.parse(&root) || root.kind != JsonVal::Kind::Object) {
        out.error = "Failed to parse panfrost_vk_compile --info-json output";
        cleanup();
        return out;
    }

    if (const auto* v = root.get("arch")) out.arch = static_cast<uint8_t>(v->as_u32(out.arch));
    if (const auto* v = root.get("stage")) out.stage = parse_stage_str(v->as_str(), out.stage);
    if (const auto* v = root.get("entry")) out.entry_point = v->as_str(out.entry_point);
    if (const auto* v = root.get("backend")) out.backend_used = v->as_str();
    if (const auto* v = root.get("idvs")) out.idvs = v->as_bool(true);
    if (const auto* v = root.get("robust")) out.robust = v->as_bool(false);
    if (const auto* v = root.get("merge_workgroups")) out.merge_workgroups = v->as_bool(false);
    if (const auto* v = root.get("instrs")) out.instrs = v->as_u32(0);
    if (const auto* v = root.get("work_reg_count")) out.work_reg_count = v->as_u32(32);
    if (const auto* v = root.get("tls_size")) out.tls_size = v->as_u32(0);
    if (const auto* v = root.get("wls_size")) out.wls_size = v->as_u32(0);
    if (const auto* v = root.get("constant_data_size")) out.constant_data_size = v->as_u32(0);
    if (const auto* v = root.get("secondary_enable")) out.secondary_enable = v->as_bool(false);
    if (const auto* v = root.get("secondary_offset")) out.secondary_offset = v->as_u32(0);

    // Trim trailing all-zero 64-bit cache-line alignment padding only when there is no
    // trailing .rodata constant pool (otherwise trailing zero constants would be truncated).
    if (out.constant_data_size == 0u) {
        while (out.ebin_bytes.size() > 8u && (out.ebin_bytes.size() % 8u) == 0u) {
            uint64_t tail = 0;
            std::memcpy(&tail, out.ebin_bytes.data() + out.ebin_bytes.size() - 8u, 8u);
            if (tail != 0u) break;
            out.ebin_bytes.resize(out.ebin_bytes.size() - 8u);
        }
    }

    if (const auto* fau = root.get("fau")) {
        if (const auto* v = fau->get("sysval_slots")) out.fau_sysval_slots = v->as_u32(0);
        if (const auto* v = fau->get("driver_slots")) out.fau_driver_slots = v->as_u32(0);
        if (const auto* v = fau->get("reserved_words")) out.fau_reserved_words = v->as_u32(0);
        if (const auto* v = fau->get("total_words")) out.fau_total_words = v->as_u32(0);
        if (const auto* arr = fau->get("used_sysval_words"); arr && arr->kind == JsonVal::Kind::Array) {
            for (const auto& el : arr->arr) out.used_sysval_words.push_back(el.as_u32());
        }
        if (const auto* arr = fau->get("used_push_const_words"); arr && arr->kind == JsonVal::Kind::Array) {
            for (const auto& el : arr->arr) out.used_push_const_words.push_back(el.as_u32());
        }
        if (const auto* imms = fau->get("immediates"); imms && imms->kind == JsonVal::Kind::Array) {
            for (const auto& item : imms->arr) {
                const auto* wv = item.get("word");
                const auto* vv = item.get("value");
                if (!wv || !vv) continue;
                uint32_t word_idx = wv->as_u32();
                uint32_t val = static_cast<uint32_t>(std::strtoul(vv->as_str().c_str(), nullptr, 0));
                out.fau_const_words[word_idx] = val;
            }
        }
    }

    if (const auto* descs = root.get("descriptors"); descs && descs->kind == JsonVal::Kind::Array) {
        for (const auto& item : descs->arr) {
            PanfrostVkDescriptorInfo di;
            if (const auto* v = item.get("set")) di.set = v->as_u32();
            if (const auto* v = item.get("table")) di.table = v->as_u32(di.set + 1u);
            if (const auto* v = item.get("binding")) di.binding = v->as_u32();
            if (const auto* v = item.get("kind")) di.kind = v->as_str();
            if (const auto* v = item.get("desc_idx")) di.desc_idx = v->as_u32();
            if (const auto* v = item.get("desc_count")) di.desc_count = v->as_u32(1);
            out.descriptors.push_back(di);

            if (di.kind == "buffer") {
                uint64_t sb_key = (static_cast<uint64_t>(di.set) << 32) | di.binding;
                uint32_t sc = 12u;  // Default to StorageBuffer if metadata didn't classify it
                auto sit = meta.binding_storage_class.find(sb_key);
                if (sit != meta.binding_storage_class.end()) sc = sit->second;
                std::string prefix = (sc == 12u) ? "ssbo:" : "ubo:";
                std::string canonical = prefix + std::to_string(di.set) + ":" + std::to_string(di.binding);
                uint32_t count = (di.desc_count > 0 && di.desc_count <= 64u) ? di.desc_count : 1u;
                for (uint32_t elem = 0; elem < count; ++elem) {
                    uint32_t packed = (di.table << 24) | ((di.desc_idx + elem) & 0xFFFFFFu);
                    out.srt_pka_buf_names[packed] = canonical;
                }
            }
        }
    }

    if (const auto* vars = root.get("varyings"); vars && vars->kind == JsonVal::Kind::Array) {
        for (const auto& item : vars->arr) {
            PanfrostVkVaryingInfo vi;
            if (const auto* v = item.get("index")) vi.index = v->as_u32();
            if (const auto* v = item.get("location")) vi.location = v->as_u32();
            if (const auto* v = item.get("user_location"); v && v->kind == JsonVal::Kind::Int) {
                vi.user_location = static_cast<int32_t>(v->ival);
            }
            if (const auto* v = item.get("name")) vi.name = v->as_str();
            if (const auto* v = item.get("offset"); v && v->kind == JsonVal::Kind::Int) {
                vi.offset = static_cast<int32_t>(v->ival);
            }
            if (const auto* v = item.get("ncomps")) vi.ncomps = v->as_u32();
            if (const auto* v = item.get("section")) vi.section = v->as_u32();
            out.varyings.push_back(std::move(vi));
        }
    }

    cleanup();
    out.ok = true;
    return out;
}

}  // namespace valhall::spirv2valhall
