#include "spirv2valhall.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <unistd.h>

namespace valhall::spirv2valhall {

namespace {

constexpr uint32_t kSpirvMagic = 0x07230203u;
constexpr uint16_t kSpvOpExtension = 10u;
constexpr uint16_t kSpvOpEntryPoint = 15u;
constexpr uint16_t kSpvOpCapability = 17u;
constexpr uint16_t kSpvOpTypeFloat = 22u;
constexpr uint16_t kSpvOpBeginInvocationInterlockEXT = 5364u;
constexpr uint16_t kSpvOpEndInvocationInterlockEXT = 5365u;
constexpr uint32_t kSpvCapFloat64 = 10u;
constexpr uint32_t kSpvCapFragmentShaderPixelInterlockEXT = 5378u;

bool read_file_bytes(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.good()) {
        if (err) *err = "Failed to open file: " + path;
        return false;
    }
    std::streamsize sz = ifs.tellg();
    if (sz < 0) {
        if (err) *err = "Failed to determine file size: " + path;
        return false;
    }
    ifs.seekg(0);
    out->resize(static_cast<size_t>(sz));
    if (sz > 0 && !ifs.read(reinterpret_cast<char*>(out->data()), sz)) {
        if (err) *err = "Failed to read file: " + path;
        return false;
    }
    return true;
}

bool read_spirv_words(const std::string& path, std::vector<uint32_t>* out, std::string* err) {
    std::vector<uint8_t> raw;
    if (!read_file_bytes(path, &raw, err)) return false;
    if (raw.size() < 20 || (raw.size() % 4) != 0) {
        if (err) *err = "Invalid SPIR-V file size (" + std::to_string(raw.size()) + " bytes): " + path;
        return false;
    }
    out->resize(raw.size() / 4);
    std::memcpy(out->data(), raw.data(), raw.size());
    if ((*out)[0] != kSpirvMagic) {
        if (err) *err = "Invalid SPIR-V magic number in " + path;
        return false;
    }
    return true;
}

std::string extract_spv_string(const uint32_t* words, size_t count) {
    std::string s;
    s.reserve(count * 4);
    for (size_t i = 0; i < count; ++i) {
        uint32_t w = words[i];
        for (int b = 0; b < 4; ++b) {
            char c = static_cast<char>((w >> (b * 8)) & 0xFFu);
            if (c == '\0') return s;
            s.push_back(c);
        }
    }
    return s;
}

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

std::vector<uint8_t> decode_c_string_literals(const std::string& combined) {
    std::vector<uint8_t> out;
    out.reserve(combined.size());
    size_t i = 0;
    const size_t n = combined.size();
    while (i < n) {
        if (combined[i] == '\\' && i + 1 < n) {
            char c = combined[i + 1];
            if (c == 'x') {
                unsigned val = 0;
                size_t j = i + 2;
                int digits = 0;
                while (j < n && digits < 2 && std::isxdigit(static_cast<unsigned char>(combined[j]))) {
                    char h = combined[j];
                    val = (val << 4) | (h >= '0' && h <= '9' ? (h - '0') : (std::tolower(static_cast<unsigned char>(h)) - 'a' + 10));
                    ++j;
                    ++digits;
                }
                out.push_back(static_cast<uint8_t>(val & 0xFFu));
                i = j;
            } else if (c >= '0' && c <= '7') {
                unsigned val = 0;
                size_t j = i + 1;
                int digits = 0;
                while (j < n && digits < 3 && combined[j] >= '0' && combined[j] <= '7') {
                    val = (val << 3) | static_cast<unsigned>(combined[j] - '0');
                    ++j;
                    ++digits;
                }
                out.push_back(static_cast<uint8_t>(val & 0xFFu));
                i = j;
            } else {
                switch (c) {
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case '\\': out.push_back('\\'); break;
                    case '"': out.push_back('"'); break;
                    case '\'': out.push_back('\''); break;
                    case 'a': out.push_back('\a'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'v': out.push_back('\v'); break;
                    default: out.push_back(static_cast<uint8_t>(c)); break;
                }
                i += 2;
            }
        } else {
            out.push_back(static_cast<uint8_t>(combined[i]));
            ++i;
        }
    }
    return out;
}

}  // namespace

uint8_t infer_arch_from_core(const std::string& core) {
    std::string lo = core;
    std::transform(lo.begin(), lo.end(), lo.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lo.find("g925") != std::string::npos || lo.find("g725") != std::string::npos || lo.find("g625") != std::string::npos) {
        return 14;
    }
    if (lo.find("g720") != std::string::npos || lo.find("g620") != std::string::npos) {
        return 12;
    }
    if (lo.find("g715") != std::string::npos || lo.find("g615") != std::string::npos) {
        return 11;
    }
    if (lo.find("g710") != std::string::npos || lo.find("g610") != std::string::npos ||
        lo.find("g510") != std::string::npos || lo.find("g310") != std::string::npos) {
        return 10;
    }
    if (lo.find("g78") != std::string::npos || lo.find("g77") != std::string::npos ||
        lo.find("g68") != std::string::npos || lo.find("g57") != std::string::npos) {
        return 9;
    }
    return 10;
}

bool infer_spirv_stage(const std::vector<uint32_t>& words, SpirvStageInfo* out, std::string* err) {
    if (!out) {
        if (err) *err = "Null SpirvStageInfo output pointer";
        return false;
    }
    if (words.size() < 5 || words[0] != kSpirvMagic) {
        if (err) *err = "Invalid SPIR-V stream header";
        return false;
    }

    size_t pos = 5;
    while (pos < words.size()) {
        uint32_t w = words[pos];
        uint32_t wc = w >> 16;
        uint16_t op = static_cast<uint16_t>(w & 0xFFFFu);
        if (wc == 0 || pos + wc > words.size()) {
            if (err) *err = "Malformed SPIR-V instruction length at word " + std::to_string(pos);
            return false;
        }
        if (op == kSpvOpEntryPoint && wc >= 4) {
            uint32_t em = words[pos + 1];
            out->execution_model = em;
            out->entry_point = extract_spv_string(&words[pos + 3], wc - 3);
            switch (em) {
                case 0: out->stage = ShaderStage::Vertex; out->malioc_flag = "--vertex"; return true;
                case 1: out->stage = ShaderStage::TessControl; out->malioc_flag = "--tessellation_control"; return true;
                case 2: out->stage = ShaderStage::TessEval; out->malioc_flag = "--tessellation_evaluation"; return true;
                case 3: out->stage = ShaderStage::Geometry; out->malioc_flag = "--geometry"; return true;
                case 4: out->stage = ShaderStage::Fragment; out->malioc_flag = "--fragment"; return true;
                case 5: out->stage = ShaderStage::Compute; out->malioc_flag = "--compute"; return true;
                case 5313: out->stage = ShaderStage::RayGen; out->malioc_flag = "--ray_generation"; return true;
                case 5314: out->stage = ShaderStage::Intersection; out->malioc_flag = "--ray_intersection"; return true;
                case 5315: out->stage = ShaderStage::AnyHit; out->malioc_flag = "--ray_any_hit"; return true;
                case 5316: out->stage = ShaderStage::ClosestHit; out->malioc_flag = "--ray_closest_hit"; return true;
                case 5317: out->stage = ShaderStage::Miss; out->malioc_flag = "--ray_miss"; return true;
                case 5318: out->stage = ShaderStage::Callable; out->malioc_flag = "--ray_callable"; return true;
                default:
                    if (err) *err = "Unsupported SPIR-V ExecutionModel: " + std::to_string(em);
                    return false;
            }
        }
        pos += wc;
    }

    if (err) *err = "No OpEntryPoint found in SPIR-V binary";
    return false;
}

bool infer_spirv_stage_from_file(const std::string& spv_path, SpirvStageInfo* out, std::string* err) {
    std::vector<uint32_t> words;
    if (!read_spirv_words(spv_path, &words, err)) return false;
    return infer_spirv_stage(words, out, err);
}

std::vector<uint32_t> strip_unused_malioc_capabilities(const std::vector<uint32_t>& words, bool* stripped_any) {
    if (stripped_any) *stripped_any = false;
    if (words.size() < 5 || words[0] != kSpirvMagic) return words;

    bool uses_f64 = false;
    bool uses_interlock = false;
    size_t pos = 5;
    while (pos < words.size()) {
        uint32_t wc = words[pos] >> 16;
        uint16_t op = static_cast<uint16_t>(words[pos] & 0xFFFFu);
        if (wc == 0 || pos + wc > words.size()) break;
        if (op == kSpvOpTypeFloat && wc >= 3 && words[pos + 2] == 64u) {
            uses_f64 = true;
        } else if (op == kSpvOpBeginInvocationInterlockEXT || op == kSpvOpEndInvocationInterlockEXT) {
            uses_interlock = true;
        }
        pos += wc;
    }

    std::vector<uint32_t> out;
    out.reserve(words.size());
    out.insert(out.end(), words.begin(), words.begin() + 5);
    pos = 5;
    while (pos < words.size()) {
        uint32_t wc = words[pos] >> 16;
        uint16_t op = static_cast<uint16_t>(words[pos] & 0xFFFFu);
        if (wc == 0 || pos + wc > words.size()) break;

        bool skip = false;
        if (op == kSpvOpCapability && wc == 2) {
            uint32_t cap = words[pos + 1];
            if ((cap == kSpvCapFloat64 && !uses_f64) ||
                (cap == kSpvCapFragmentShaderPixelInterlockEXT && !uses_interlock)) {
                skip = true;
            }
        } else if (op == kSpvOpExtension && !uses_interlock && wc >= 2) {
            std::string ext = extract_spv_string(&words[pos + 1], wc - 1);
            if (ext.find("fragment_shader_interlock") != std::string::npos) {
                skip = true;
            }
        }

        if (skip) {
            if (stripped_any) *stripped_any = true;
        } else {
            out.insert(out.end(), words.begin() + static_cast<std::ptrdiff_t>(pos),
                       words.begin() + static_cast<std::ptrdiff_t>(pos + wc));
        }
        pos += wc;
    }
    return out;
}

bool parse_mbs2_header_ebins(
    const std::string& header_text,
    uint8_t arch,
    bool disassemble,
    std::vector<ExtractedEbin>* out,
    std::string* err) {
    if (!out) {
        if (err) *err = "Null EBIN output vector";
        return false;
    }
    out->clear();

    // 1. Parse all `static const uint8_t b_N[] = "..." "...";` byte arrays
    std::map<std::string, std::vector<uint8_t>> blobs;
    size_t scan = 0;
    const std::string b_prefix = "static const uint8_t b_";
    while ((scan = header_text.find(b_prefix, scan)) != std::string::npos) {
        size_t name_start = scan + std::strlen("static const uint8_t ");
        size_t bracket = header_text.find("[]", name_start);
        if (bracket == std::string::npos) break;
        std::string bname = header_text.substr(name_start, bracket - name_start);
        size_t eq = header_text.find('=', bracket);
        if (eq == std::string::npos) break;
        size_t p = eq + 1;
        std::string combined;
        while (p < header_text.size()) {
            while (p < header_text.size() && std::isspace(static_cast<unsigned char>(header_text[p]))) ++p;
            if (p >= header_text.size() || header_text[p] == ';') break;
            if (header_text[p] == '"') {
                ++p;
                while (p < header_text.size()) {
                    if (header_text[p] == '\\' && p + 1 < header_text.size()) {
                        combined.push_back(header_text[p]);
                        combined.push_back(header_text[p + 1]);
                        p += 2;
                    } else if (header_text[p] == '"') {
                        ++p;
                        break;
                    } else {
                        combined.push_back(header_text[p]);
                        ++p;
                    }
                }
            } else {
                ++p;
            }
        }
        blobs[bname] = decode_c_string_literals(combined);
        scan = p;
    }

    // 2. Parse `static const cmpbe_chunk_OBJC OBJC_N = { size, (uint8_t*)b_M };`
    struct ObjcRef { size_t size; std::string bname; };
    std::map<std::string, ObjcRef> objcs;
    {
        std::regex re_objc(R"(static const cmpbe_chunk_OBJC (OBJC_\d+)\s*=\s*\{\s*(\d+)\s*,\s*\(uint8_t\*\)(b_\d+)\s*\};)");
        auto begin = std::sregex_iterator(header_text.begin(), header_text.end(), re_objc);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            objcs[(*it)[1].str()] = ObjcRef{static_cast<size_t>(std::stoull((*it)[2].str())), (*it)[3].str()};
        }
    }

    // 3. Parse each `static const cmpbe_chunk_EBIN EBIN_N = { ... };`
    {
        std::regex re_ebin(R"(static const cmpbe_chunk_EBIN EBIN_\d+\s*=\s*\{([^}]+)\};)");
        std::regex re_objc_use(R"(OBJC_\d+)");
        std::regex re_u32(R"(\(uint32_t\)\s*0x([0-9a-fA-F]+))");
        std::regex re_variant(R"(\(cmpbe_chunk_FSHA_variant\)\s*0x([0-9a-fA-F]+))");

        auto begin = std::sregex_iterator(header_text.begin(), header_text.end(), re_ebin);
        auto end = std::sregex_iterator();
        uint32_t idx = 0;
        for (auto it = begin; it != end; ++it) {
            std::string body = (*it)[1].str();
            std::smatch om;
            if (!std::regex_search(body, om, re_objc_use)) continue;
            auto oit = objcs.find(om.str());
            if (oit == objcs.end()) continue;
            auto bit = blobs.find(oit->second.bname);
            if (bit == blobs.end()) continue;

            uint32_t code_sz = static_cast<uint32_t>(bit->second.size());
            std::smatch u32m;
            if (std::regex_search(body, u32m, re_u32)) {
                uint32_t parsed = static_cast<uint32_t>(std::stoul(u32m[1].str(), nullptr, 16));
                if (parsed != 0 && parsed != 0xFFFFFFFFu && parsed <= bit->second.size()) {
                    code_sz = parsed;
                }
            }
            code_sz = (code_sz / 8u) * 8u;

            uint32_t vflags = 0;
            std::smatch vm;
            if (std::regex_search(body, vm, re_variant)) {
                vflags = static_cast<uint32_t>(std::stoul(vm[1].str(), nullptr, 16));
            }

            ExtractedEbin eb;
            eb.index = idx++;
            eb.variant_flags = vflags;
            eb.code_size = code_sz;
            eb.bytecode.assign(bit->second.begin(), bit->second.begin() + code_sz);
            if (disassemble) {
                for (size_t off = 0; off + 8 <= eb.bytecode.size(); off += 8) {
                    uint64_t raw = 0;
                    std::memcpy(&raw, &eb.bytecode[off], 8);
                    auto dec = valhall::decode_valhall_instruction(raw, arch, off);
                    if (!dec.valid) eb.invalid_count++;
                    eb.instructions.push_back(std::move(dec));
                }
            }
            out->push_back(std::move(eb));
        }
    }

    if (out->empty()) {
        if (err) *err = "No EBIN bytecode sections found in moc_hook MBS2 dump";
        return false;
    }
    return true;
}

bool extract_valhall_from_spirv(
    const std::string& spv_path,
    const ExtractOptions& opts,
    ExtractResult* out,
    std::string* err) {
    if (!out) {
        if (err) *err = "Null ExtractResult output pointer";
        return false;
    }
    *out = ExtractResult{};

    std::vector<uint32_t> words;
    if (!read_spirv_words(spv_path, &words, err)) return false;
    if (!infer_spirv_stage(words, &out->stage_info, err)) return false;

    out->arch = (opts.arch != 0) ? opts.arch : infer_arch_from_core(opts.core);

    std::string effective_spv = spv_path;
    std::string temp_spv_path;
    if (opts.strip_unused_caps) {
        bool stripped = false;
        std::vector<uint32_t> cleaned = strip_unused_malioc_capabilities(words, &stripped);
        out->stripped_caps = stripped;
        if (stripped) {
            char tmpl[] = "/tmp/spirv2valhall_in_XXXXXX";
            int fd = mkstemp(tmpl);
            if (fd < 0) {
                if (err) *err = "Failed to create temporary sanitized SPIR-V file";
                return false;
            }
            close(fd);
            temp_spv_path = std::string(tmpl) + ".spv";
            std::remove(tmpl);
            std::ofstream ofs(temp_spv_path, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(cleaned.data()),
                      static_cast<std::streamsize>(cleaned.size() * sizeof(uint32_t)));
            ofs.close();
            effective_spv = temp_spv_path;
        }
    }

    std::string hook_so = opts.moc_hook_path;
    if (hook_so.empty()) {
        const char* env_hook = std::getenv("SPIRV2VALHALL_MOC_HOOK");
        if (env_hook && env_hook[0] != '\0') {
            hook_so = env_hook;
        } else {
#ifdef SPIRV2VALHALL_DEFAULT_MOC_HOOK_PATH
            hook_so = SPIRV2VALHALL_DEFAULT_MOC_HOOK_PATH;
#else
            hook_so = "build/libmoc_hook.so";
#endif
        }
    }

    std::string hdr_path = opts.dump_header_path;
    bool remove_hdr = false;
    if (hdr_path.empty()) {
        char tmpl[] = "/tmp/spirv2valhall_mbs2_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd < 0) {
            if (!temp_spv_path.empty()) std::remove(temp_spv_path.c_str());
            if (err) *err = "Failed to create temporary MBS2 header file";
            return false;
        }
        close(fd);
        hdr_path = tmpl;
        remove_hdr = true;
    }

    std::string cmd = "MOC_DUMP_PATH=" + shell_quote(hdr_path) +
                      " LD_PRELOAD=" + shell_quote(hook_so) + " " +
                      shell_quote(opts.malioc_path) +
                      " --vulkan --spirv -c " + shell_quote(opts.core) + " " +
                      out->stage_info.malioc_flag + " " +
                      shell_quote(effective_spv) + " 2>&1";

    if (opts.verbose) {
        std::cerr << "[spirv2valhall] stage=" << out->stage_info.malioc_flag
                  << " entry=\"" << out->stage_info.entry_point << "\""
                  << " core=" << opts.core << " arch=v" << static_cast<int>(out->arch)
                  << " stripped_caps=" << (out->stripped_caps ? "yes" : "no") << "\n";
        std::cerr << "[spirv2valhall] cmd: " << cmd << "\n";
    }

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        if (!temp_spv_path.empty()) std::remove(temp_spv_path.c_str());
        if (remove_hdr) std::remove(hdr_path.c_str());
        if (err) *err = "Failed to launch malioc subprocess";
        return false;
    }
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        out->malioc_output += buf;
    }
    int rc = pclose(pipe);

    if (!temp_spv_path.empty()) std::remove(temp_spv_path.c_str());

    if (rc != 0) {
        if (remove_hdr) std::remove(hdr_path.c_str());
        if (err) {
            *err = "malioc failed (exit=" + std::to_string(rc) + ") for " + spv_path +
                   " (" + out->stage_info.malioc_flag + "):\n" + out->malioc_output;
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

    return parse_mbs2_header_ebins(out->dump_header_text, out->arch, opts.disassemble, &out->ebins, err);
}

std::string format_disassembly(const ExtractResult& res, bool include_modifiers) {
    std::ostringstream oss;
    oss << "; spirv2valhall stage=" << res.stage_info.malioc_flag
        << " entry=\"" << res.stage_info.entry_point << "\""
        << " arch=v" << static_cast<int>(res.arch)
        << " ebins=" << res.ebins.size() << "\n";
    for (const auto& eb : res.ebins) {
        oss << "; EBIN[" << eb.index << "] variant=0x" << std::hex << eb.variant_flags
            << std::dec << " code_size=" << eb.code_size
            << " insts=" << eb.instructions.size()
            << " invalid=" << eb.invalid_count << "\n";
        for (const auto& dec : eb.instructions) {
            char hdr[64];
            std::snprintf(hdr, sizeof(hdr), "  0x%04llx: %016llx  ",
                          static_cast<unsigned long long>(dec.pc),
                          static_cast<unsigned long long>(dec.raw_word));
            oss << hdr << (dec.valid ? dec.disassembly : std::string("INVALID"));
            if (include_modifiers && dec.valid && !dec.modifier_strs.empty()) {
                oss << "  [";
                bool first = true;
                for (const auto& kv : dec.modifier_strs) {
                    if (!first) oss << ",";
                    oss << kv.first << "=" << kv.second;
                    first = false;
                }
                oss << "]";
            }
            oss << "\n";
        }
    }
    return oss.str();
}

}  // namespace valhall::spirv2valhall
