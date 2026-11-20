#include "spirv_inspect.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>

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

}  // namespace

const char* shader_stage_name(ShaderStage stage) {
    switch (stage) {
        case ShaderStage::Vertex: return "Vertex";
        case ShaderStage::TessControl: return "TessControl";
        case ShaderStage::TessEval: return "TessEval";
        case ShaderStage::Geometry: return "Geometry";
        case ShaderStage::Fragment: return "Fragment";
        case ShaderStage::Compute: return "Compute";
        case ShaderStage::RayGen: return "RayGen";
        case ShaderStage::AnyHit: return "AnyHit";
        case ShaderStage::ClosestHit: return "ClosestHit";
        case ShaderStage::Miss: return "Miss";
        case ShaderStage::Intersection: return "Intersection";
        case ShaderStage::Callable: return "Callable";
        default: return "Unknown";
    }
}

bool read_spirv_words(const std::string& path, std::vector<uint32_t>* out, std::string* err) {
    if (!out) {
        if (err) *err = "Null SPIR-V output vector";
        return false;
    }
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

uint8_t infer_arch_from_core(const std::string& core) {
    std::string lo = core;
    std::transform(lo.begin(), lo.end(), lo.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
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

}  // namespace valhall::spirv2valhall
