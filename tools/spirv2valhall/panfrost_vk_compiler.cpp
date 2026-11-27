#include "panfrost_vk_compiler.h"

#include <fcntl.h>
#include <sys/stat.h>
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
#include <unordered_map>

namespace fs = std::filesystem;

namespace valhall::spirv2valhall {

namespace {

bool is_executable_file(const std::string& path) {
    if (path.empty()) return false;
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return false;
    if (!S_ISREG(st.st_mode)) return false;
    return ::access(path.c_str(), X_OK) == 0;
}

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

/// Spawn `bin` with `argv` via fork/execv, capturing stderr into `stderr_out`.
bool run_subprocess(const std::string& bin,
                    const std::vector<std::string>& args,
                    int* exit_code_out,
                    int* term_sig_out,
                    std::string* stderr_out,
                    std::string* err) {
    int err_pipe[2] = {-1, -1};
    if (::pipe(err_pipe) != 0) {
        if (err) *err = "pipe() failed for panfrost_vk_compile";
        return false;
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(err_pipe[0]);
        ::close(err_pipe[1]);
        if (err) *err = "fork() failed for panfrost_vk_compile";
        return false;
    }

    if (pid == 0) {
        ::close(err_pipe[0]);
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(err_pipe[1]);
        int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDOUT_FILENO);
            ::close(devnull);
        }
        std::vector<char*> cargv;
        cargv.reserve(args.size() + 1);
        for (const auto& a : args) {
            cargv.push_back(const_cast<char*>(a.c_str()));
        }
        cargv.push_back(nullptr);
        ::execv(bin.c_str(), cargv.data());
        std::fprintf(stderr, "execv(%s) failed: %s\n", bin.c_str(), std::strerror(errno));
        ::_exit(127);
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

    *exit_code_out = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    *term_sig_out = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
    return true;
}

}  // namespace

std::string find_panfrost_vk_compiler(const std::string& explicit_path) {
    if (!explicit_path.empty() && is_executable_file(explicit_path)) {
        return explicit_path;
    }
    if (const char* env = std::getenv("PANFROST_VK_COMPILE_BIN")) {
        if (is_executable_file(env)) return env;
    }
    fs::path cur = fs::current_path();
    for (int depth = 0; depth < 5; ++depth) {
        fs::path cand = cur / "third_party" / "prebuilts" / "panfrost" / "panfrost_vk_compile";
        if (is_executable_file(cand.string())) return cand.string();
        if (!cur.has_parent_path() || cur == cur.parent_path()) break;
        cur = cur.parent_path();
    }
    if (const char* path_env = std::getenv("PATH")) {
        std::istringstream iss(path_env);
        std::string dir;
        while (std::getline(iss, dir, ':')) {
            if (dir.empty()) continue;
            fs::path cand = fs::path(dir) / "panfrost_vk_compile";
            if (is_executable_file(cand.string())) return cand.string();
        }
    }
    return "";
}

PanfrostVkCompileOutput compile_vulkan_spv_via_panfrost(
    const std::string& spv_path,
    const PanfrostVkCompileOptions& opts) {
    PanfrostVkCompileOutput out;
    out.arch = (opts.arch != 0) ? opts.arch : 11;

    std::string bin = find_panfrost_vk_compiler(opts.panfrost_vk_bin);
    if (bin.empty()) {
        out.error = "panfrost_vk_compile not found (checked --panfrost-bin, PANFROST_VK_COMPILE_BIN, "
                    "third_party/prebuilts/panfrost/panfrost_vk_compile, and PATH)";
        return out;
    }

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

    std::vector<std::string> argv;
    argv.push_back(bin);
    if (opts.backend == PanfrostVkBackend::Kraid) {
        argv.push_back("--kraid");
    } else if (opts.backend == PanfrostVkBackend::Bifrost) {
        argv.push_back("--no-kraid");
    }
    if (opts.robust) argv.push_back("--robust");
    if (opts.no_idvs) argv.push_back("--no-idvs");
    if (opts.no_promote_imm) argv.push_back("--no-promote-imm");
    if (opts.no_clamp_version) argv.push_back("--no-clamp-version");
    if (opts.core_max_id != 0) {
        argv.push_back("--core-max-id");
        argv.push_back(std::to_string(opts.core_max_id));
    }
    argv.push_back("--info-json");
    argv.push_back(out_json.string());
    argv.push_back(spv_path);
    argv.push_back(out_bin.string());

    if (opts.verbose) {
        std::cerr << "[panfrost-vk] running:";
        for (const auto& a : argv) std::cerr << " " << a;
        std::cerr << "\n";
    }

    int exit_code = -1;
    int term_sig = 0;
    std::string stderr_text;
    if (!run_subprocess(bin, argv, &exit_code, &term_sig, &stderr_text, &out.error)) {
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
    // Trim trailing all-zero 64-bit cache-line alignment padding words emitted by Panfrost.
    while (out.ebin_bytes.size() > 8u && (out.ebin_bytes.size() % 8u) == 0u) {
        uint64_t tail = 0;
        std::memcpy(&tail, out.ebin_bytes.data() + out.ebin_bytes.size() - 8u, 8u);
        if (tail != 0u) break;
        out.ebin_bytes.resize(out.ebin_bytes.size() - 8u);
    }

    std::string info_text;
    if (!read_file_text(out_json.string(), &info_text, &out.error)) {
        cleanup();
        return out;
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

    cleanup();
    out.ok = true;
    return out;
}

}  // namespace valhall::spirv2valhall
