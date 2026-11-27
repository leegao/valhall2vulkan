#include "panfrost_compiler.h"
#include "panfrost_vk_compiler.h"

#include "spirv2ssa.h"
#include "ssa_defau_devirt.h"
#include "ssa_opt_pipeline.h"
#include "valhall_cfg.h"
#include "valhall_isa_generated.h"
#include "valhall_pass1.h"
#include "valhall_ssa_json.h"
#include "valhall_to_spirv.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;

namespace valhall::spirv2valhall {

namespace {

constexpr uint32_t kSpvMagic = 0x07230203u;

// SPIR-V opcodes used by the Vulkan -> OpenCL SPIR-V translator.
constexpr uint16_t kOpName = 5;
constexpr uint16_t kOpMemberName = 6;
constexpr uint16_t kOpExtInstImport = 11;
constexpr uint16_t kOpExtInst = 12;
constexpr uint16_t kOpMemoryModel = 14;
constexpr uint16_t kOpEntryPoint = 15;
constexpr uint16_t kOpExecutionMode = 16;
constexpr uint16_t kOpCapability = 17;
constexpr uint16_t kOpTypeVoid = 19;
constexpr uint16_t kOpTypeInt = 21;
constexpr uint16_t kOpTypeFloat = 22;
constexpr uint16_t kOpTypeVector = 23;
constexpr uint16_t kOpTypeArray = 28;
constexpr uint16_t kOpTypeRuntimeArray = 29;
constexpr uint16_t kOpTypeStruct = 30;
constexpr uint16_t kOpTypePointer = 32;
constexpr uint16_t kOpTypeFunction = 33;
constexpr uint16_t kOpFunction = 54;
constexpr uint16_t kOpFunctionParameter = 55;
constexpr uint16_t kOpVariable = 59;
constexpr uint16_t kOpAccessChain = 65;
constexpr uint16_t kOpInBoundsAccessChain = 66;
constexpr uint16_t kOpPtrAccessChain = 67;
constexpr uint16_t kOpInBoundsPtrAccessChain = 70;
constexpr uint16_t kOpDecorate = 71;
constexpr uint16_t kOpMemberDecorate = 72;
constexpr uint16_t kOpExtension = 10;

// SPIR-V constants.
constexpr uint32_t kCapShader = 1;
constexpr uint32_t kCapAddresses = 4;
constexpr uint32_t kCapLinkage = 5;
constexpr uint32_t kCapKernel = 6;
constexpr uint32_t kCapInt64 = 11;
constexpr uint32_t kCapStorageBuffer16BitAccess = 4433;
constexpr uint32_t kCapUniformAndStorageBuffer16BitAccess = 4434;
constexpr uint32_t kCapStorageBuffer8BitAccess = 4448;
constexpr uint32_t kCapPhysicalStorageBufferAddresses = 5347;

constexpr uint32_t kAddrPhysical64 = 2;
constexpr uint32_t kMemOpenCL = 2;
constexpr uint32_t kExecModelKernel = 6;
constexpr uint32_t kExecModelGLCompute = 5;

constexpr uint32_t kScInput = 1;
constexpr uint32_t kScUniform = 2;
constexpr uint32_t kScCrossWorkgroup = 5;
constexpr uint32_t kScStorageBuffer = 12;

constexpr uint32_t kDecBlock = 2;
constexpr uint32_t kDecBufferBlock = 3;
constexpr uint32_t kDecArrayStride = 6;
constexpr uint32_t kDecBuiltIn = 11;
constexpr uint32_t kDecRestrict = 19;
constexpr uint32_t kDecAliased = 20;
constexpr uint32_t kDecCoherent = 22;
constexpr uint32_t kDecNonWritable = 24;
constexpr uint32_t kDecNonReadable = 25;
constexpr uint32_t kDecBinding = 33;
constexpr uint32_t kDecDescriptorSet = 34;
constexpr uint32_t kDecOffset = 35;
constexpr uint32_t kDecLinkageAttributes = 41;
constexpr uint32_t kLinkageExport = 0;

struct SpvInstRaw {
    uint16_t op = 0;
    std::vector<uint32_t> words;  // full instruction including word[0]
};

std::string decode_spv_string(const std::vector<uint32_t>& words, size_t start_word) {
    std::string out;
    for (size_t i = start_word; i < words.size(); ++i) {
        uint32_t w = words[i];
        for (int b = 0; b < 4; ++b) {
            char c = static_cast<char>((w >> (b * 8)) & 0xFFu);
            if (c == '\0') return out;
            out.push_back(c);
        }
    }
    return out;
}

void append_spv_string(std::vector<uint32_t>& words, const std::string& s) {
    size_t len = s.size() + 1;
    size_t nwords = (len + 3) / 4;
    for (size_t wi = 0; wi < nwords; ++wi) {
        uint32_t w = 0;
        for (int b = 0; b < 4; ++b) {
            size_t idx = wi * 4 + b;
            uint8_t ch = (idx < s.size()) ? static_cast<uint8_t>(s[idx]) : 0u;
            w |= (static_cast<uint32_t>(ch) << (b * 8));
        }
        words.push_back(w);
    }
}

std::vector<uint32_t> make_inst(uint16_t op, const std::vector<uint32_t>& operands) {
    std::vector<uint32_t> out;
    out.reserve(1 + operands.size());
    out.push_back((static_cast<uint32_t>(1 + operands.size()) << 16) | op);
    out.insert(out.end(), operands.begin(), operands.end());
    return out;
}

uint32_t map_glsl450_to_opencl_std(uint32_t glsl_op) {
    switch (glsl_op) {
        case 1:  return 53;   // Round -> OpenCLstd_Rint
        case 2:  return 53;   // RoundEven -> OpenCLstd_Rint
        case 3:  return 66;   // Trunc -> OpenCLstd_Trunc
        case 4:  return 23;   // FAbs -> OpenCLstd_Fabs
        case 5:  return 141;  // SAbs -> OpenCLstd_SAbs
        case 6:  return 103;  // FSign -> OpenCLstd_Sign
        case 8:  return 25;   // Floor -> OpenCLstd_Floor
        case 9:  return 12;   // Ceil -> OpenCLstd_Ceil
        case 10: return 30;   // Fract -> OpenCLstd_Fract
        case 11: return 100;  // Radians -> OpenCLstd_Radians
        case 12: return 96;   // Degrees -> OpenCLstd_Degrees
        case 13: return 57;   // Sin -> OpenCLstd_Sin
        case 14: return 14;   // Cos -> OpenCLstd_Cos
        case 15: return 62;   // Tan -> OpenCLstd_Tan
        case 16: return 3;    // Asin -> OpenCLstd_Asin
        case 17: return 0;    // Acos -> OpenCLstd_Acos (0 is valid, handled specially if needed)
        case 18: return 6;    // Atan -> OpenCLstd_Atan
        case 25: return 7;    // Atan2 -> OpenCLstd_Atan2
        case 26: return 48;   // Pow -> OpenCLstd_Pow
        case 27: return 19;   // Exp -> OpenCLstd_Exp
        case 28: return 37;   // Log -> OpenCLstd_Log
        case 29: return 20;   // Exp2 -> OpenCLstd_Exp2
        case 30: return 38;   // Log2 -> OpenCLstd_Log2
        case 31: return 61;   // Sqrt -> OpenCLstd_Sqrt
        case 32: return 56;   // InverseSqrt -> OpenCLstd_Rsqrt
        case 37: return 28;   // FMin -> OpenCLstd_Fmin
        case 38: return 159;  // UMin -> OpenCLstd_UMin
        case 39: return 158;  // SMin -> OpenCLstd_SMin
        case 40: return 27;   // FMax -> OpenCLstd_Fmax
        case 41: return 157;  // UMax -> OpenCLstd_UMax
        case 42: return 156;  // SMax -> OpenCLstd_SMax
        case 43: return 95;   // FClamp -> OpenCLstd_FClamp
        case 44: return 150;  // UClamp -> OpenCLstd_UClamp
        case 45: return 149;  // SClamp -> OpenCLstd_SClamp
        case 46: return 99;   // FMix -> OpenCLstd_Mix
        case 48: return 101;  // Step -> OpenCLstd_Step
        case 49: return 102;  // SmoothStep -> OpenCLstd_Smoothstep
        case 50: return 26;   // Fma -> OpenCLstd_Fma
        case 52: return 34;   // Ldexp -> OpenCLstd_Ldexp
        case 79: return 28;   // NMin -> OpenCLstd_Fmin
        case 80: return 27;   // NMax -> OpenCLstd_Fmax
        case 81: return 95;   // NClamp -> OpenCLstd_FClamp
        default: return 0xFFFFFFFFu;
    }
}

bool read_file_bytes(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        if (err) *err = "Failed to open file: " + path;
        return false;
    }
    ifs.seekg(0, std::ios::end);
    std::streamoff sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (sz < 0) {
        if (err) *err = "Failed to stat file: " + path;
        return false;
    }
    out->resize(static_cast<size_t>(sz));
    if (sz > 0) {
        ifs.read(reinterpret_cast<char*>(out->data()), sz);
    }
    return true;
}

bool read_file_text(const std::string& path, std::string* out, std::string* err) {
    std::ifstream ifs(path);
    if (!ifs) {
        if (err) *err = "Failed to open text file: " + path;
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    *out = oss.str();
    return true;
}

std::string find_compat_preload_lib(const std::string& panfrost_bin) {
    if (const char* env_p = std::getenv("PANFROST_LD_PRELOAD"); env_p && env_p[0] != '\0') {
        if (fs::exists(env_p)) return env_p;
    }
    // Check sibling spirv-tools/libspv_compat.so (used on Debian 12 / glibc 2.36 sandboxes).
    std::error_code ec;
    fs::path p = fs::absolute(panfrost_bin, ec);
    for (int up = 0; up < 4 && !p.empty(); ++up) {
        fs::path cand = p / "third_party/prebuilts/spirv-tools/libspv_compat.so";
        if (fs::exists(cand, ec)) return cand.string();
        fs::path cand2 = p / "spirv_tools_bin/libspv_compat.so";
        if (fs::exists(cand2, ec)) return cand2.string();
        p = p.parent_path();
    }
    if (fs::exists("third_party/prebuilts/spirv-tools/libspv_compat.so", ec)) {
        return fs::absolute("third_party/prebuilts/spirv-tools/libspv_compat.so", ec).string();
    }
    return "";
}

}  // namespace

std::string find_panfrost_compiler(const std::string& explicit_path) {
    std::error_code ec;
    auto is_exec = [&](const fs::path& p) -> bool {
        return !p.empty() && fs::is_regular_file(p, ec) && ::access(p.c_str(), X_OK) == 0;
    };

    if (!explicit_path.empty() && is_exec(explicit_path)) {
        return explicit_path;
    }
    if (const char* env_bin = std::getenv("PANFROST_COMPILE_BIN"); env_bin && is_exec(env_bin)) {
        return env_bin;
    }

    // Search relative to CWD and upward (handles running from build/ or repo root).
    fs::path cwd = fs::current_path(ec);
    for (int up = 0; up < 4 && !cwd.empty(); ++up) {
        fs::path cand = cwd / "third_party/prebuilts/panfrost/panfrost_compile";
        if (is_exec(cand)) return cand.string();
        cwd = cwd.parent_path();
    }

    // Search PATH.
    if (const char* path_env = std::getenv("PATH")) {
        std::istringstream iss(path_env);
        std::string dir;
        while (std::getline(iss, dir, ':')) {
            fs::path cand = fs::path(dir) / "panfrost_compile";
            if (is_exec(cand)) return cand.string();
        }
    }
    return "";
}

bool translate_vulkan_spv_to_opencl_spv(
    const std::vector<uint32_t>& vk_words,
    std::vector<uint32_t>* cl_words,
    std::string* err,
    std::vector<PanfrostArgBinding>* out_bindings) {
    if (vk_words.size() < 5 || vk_words[0] != kSpvMagic) {
        if (err) *err = "Invalid SPIR-V binary header";
        return false;
    }

    uint32_t bound = vk_words[3];
    std::vector<SpvInstRaw> insts;
    size_t pos = 5;
    while (pos < vk_words.size()) {
        uint32_t hdr = vk_words[pos];
        uint16_t wc = static_cast<uint16_t>(hdr >> 16);
        uint16_t op = static_cast<uint16_t>(hdr & 0xFFFFu);
        if (wc == 0 || pos + wc > vk_words.size()) {
            if (err) *err = "Truncated SPIR-V instruction stream";
            return false;
        }
        SpvInstRaw raw;
        raw.op = op;
        raw.words.assign(vk_words.begin() + pos, vk_words.begin() + pos + wc);
        insts.push_back(std::move(raw));
        pos += wc;
    }

    // Analyze types, entry point, buffers, and builtins.
    uint32_t entry_fn = 0;
    std::string entry_name = "main";
    bool is_compute = false;
    uint32_t glsl_ext_id = 0;

    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> ptr_types;  // tid -> (sc, pointee)
    std::unordered_map<uint32_t, std::vector<uint32_t>> struct_members;
    std::unordered_map<uint32_t, uint32_t> rarray_elem;
    std::unordered_map<uint32_t, uint32_t> fn_ret_type;
    std::unordered_set<uint32_t> buffer_block_structs;
    std::unordered_set<uint32_t> unwrapped_structs;
    std::unordered_set<uint32_t> unwrapped_rarrays;
    std::unordered_map<uint32_t, uint32_t> var_builtin;
    std::unordered_map<uint32_t, uint32_t> var_set;
    std::unordered_map<uint32_t, uint32_t> var_binding;

    struct BufferVarInfo {
        uint32_t var_id = 0;
        uint32_t orig_ptr_ty = 0;
        uint32_t elem_ty = 0;
        uint32_t cw_ptr_ty = 0;
        bool unwrapped_single_rarray = false;
        std::string param_name;
    };
    std::vector<BufferVarInfo> buf_vars;
    std::unordered_map<uint32_t, size_t> buf_var_idx;
    std::unordered_set<uint32_t> input_builtin_vars;
    std::unordered_map<uint32_t, std::string> existing_names;

    for (const auto& ins : insts) {
        if (ins.op == kOpEntryPoint && ins.words.size() >= 4) {
            if (ins.words[1] == kExecModelGLCompute) {
                is_compute = true;
                entry_fn = ins.words[2];
                entry_name = decode_spv_string(ins.words, 3);
            }
        } else if (ins.op == kOpExtInstImport && ins.words.size() >= 3) {
            std::string nm = decode_spv_string(ins.words, 2);
            if (nm == "GLSL.std.450") glsl_ext_id = ins.words[1];
        } else if (ins.op == kOpTypePointer && ins.words.size() == 4) {
            ptr_types[ins.words[1]] = {ins.words[2], ins.words[3]};
        } else if (ins.op == kOpTypeStruct && ins.words.size() >= 2) {
            std::vector<uint32_t> mems(ins.words.begin() + 2, ins.words.end());
            struct_members[ins.words[1]] = std::move(mems);
        } else if (ins.op == kOpTypeRuntimeArray && ins.words.size() == 3) {
            rarray_elem[ins.words[1]] = ins.words[2];
        } else if (ins.op == kOpTypeFunction && ins.words.size() >= 3) {
            fn_ret_type[ins.words[1]] = ins.words[2];
        } else if (ins.op == kOpDecorate && ins.words.size() >= 3) {
            if (ins.words[2] == kDecBlock || ins.words[2] == kDecBufferBlock) {
                buffer_block_structs.insert(ins.words[1]);
            } else if (ins.words[2] == kDecBuiltIn && ins.words.size() >= 4) {
                var_builtin[ins.words[1]] = ins.words[3];
            } else if (ins.words[2] == kDecDescriptorSet && ins.words.size() >= 4) {
                var_set[ins.words[1]] = ins.words[3];
            } else if (ins.words[2] == kDecBinding && ins.words.size() >= 4) {
                var_binding[ins.words[1]] = ins.words[3];
            }
        } else if (ins.op == kOpName && ins.words.size() >= 3) {
            existing_names[ins.words[1]] = decode_spv_string(ins.words, 2);
        }
    }

    if (!is_compute || entry_fn == 0) {
        if (err) {
            *err = "Panfrost Vulkan translation (--panfrost) currently requires a GLCompute SPIR-V shader "
                   "(use --panfrost-cl for OpenCL SPIR-V kernels)";
        }
        return false;
    }

    // Identify global StorageBuffer / Uniform-BufferBlock variables and Input BuiltIn variables.
    for (const auto& ins : insts) {
        if (ins.op != kOpVariable || ins.words.size() < 4) continue;
        uint32_t ptr_ty = ins.words[1];
        uint32_t var_id = ins.words[2];
        uint32_t sc = ins.words[3];
        if (sc == kScStorageBuffer ||
            (sc == kScUniform && ptr_types.count(ptr_ty) &&
             buffer_block_structs.count(ptr_types[ptr_ty].second))) {
            BufferVarInfo bi;
            bi.var_id = var_id;
            bi.orig_ptr_ty = ptr_ty;
            uint32_t pointee = ptr_types[ptr_ty].second;
            bi.elem_ty = pointee;
            auto sm_it = struct_members.find(pointee);
            if (sm_it != struct_members.end() && sm_it->second.size() == 1) {
                uint32_t m0 = sm_it->second[0];
                auto ra_it = rarray_elem.find(m0);
                if (ra_it != rarray_elem.end()) {
                    bi.elem_ty = ra_it->second;
                    bi.unwrapped_single_rarray = true;
                    unwrapped_structs.insert(pointee);
                    unwrapped_rarrays.insert(m0);
                }
            }
            bi.cw_ptr_ty = bound++;
            bi.param_name = "buf_" + std::to_string(buf_vars.size());
            if (out_bindings) {
                PanfrostArgBinding ab;
                ab.arg_idx = static_cast<uint32_t>(buf_vars.size());
                ab.fau_offset = 24u + ab.arg_idx * 8u;
                ab.descriptor_set = var_set.count(var_id) ? var_set[var_id] : 0u;
                ab.binding = var_binding.count(var_id) ? var_binding[var_id] : ab.arg_idx;
                ab.is_ssbo = (sc == kScStorageBuffer) ||
                             (sc == kScUniform && buffer_block_structs.count(pointee));
                ab.key = std::string(ab.is_ssbo ? "ssbo:" : "ubo:") +
                         std::to_string(ab.descriptor_set) + ":" + std::to_string(ab.binding);
                out_bindings->push_back(std::move(ab));
            }
            buf_var_idx[var_id] = buf_vars.size();
            buf_vars.push_back(std::move(bi));
        } else if (sc == kScInput && var_builtin.count(var_id)) {
            input_builtin_vars.insert(var_id);
        }
    }

    uint32_t new_kernel_fn_ty = bound++;
    uint32_t void_ty = 0;
    for (const auto& ins : insts) {
        if (ins.op == kOpTypeVoid && ins.words.size() == 2) {
            void_ty = ins.words[1];
            break;
        }
    }
    if (void_ty == 0) {
        if (err) *err = "SPIR-V module missing OpTypeVoid";
        return false;
    }

    // Rebuild SPIR-V in canonical section order.
    std::vector<uint32_t> out;
    out.push_back(kSpvMagic);
    out.push_back(0x00010200u);  // SPIR-V 1.2 (OpenCL 2.2)
    out.push_back(vk_words[2]);
    out.push_back(0);            // bound (patched at end)
    out.push_back(0);

    auto emit = [&](const std::vector<uint32_t>& w) {
        out.insert(out.end(), w.begin(), w.end());
    };

    // 1. Capabilities
    std::set<uint32_t> emitted_caps = {kCapAddresses, kCapLinkage, kCapKernel, kCapInt64};
    for (uint32_t c : emitted_caps) {
        emit(make_inst(kOpCapability, {c}));
    }
    for (const auto& ins : insts) {
        if (ins.op != kOpCapability || ins.words.size() < 2) continue;
        uint32_t c = ins.words[1];
        if ((c != 9u && c != 22u && c != 39u) || emitted_caps.count(c)) {
            continue;
        }
        emitted_caps.insert(c);
        emit(ins.words);
    }

    // 2. ExtInstImport
    for (const auto& ins : insts) {
        if (ins.op != kOpExtInstImport) continue;
        if (ins.words[1] == glsl_ext_id) {
            std::vector<uint32_t> ops = {glsl_ext_id};
            append_spv_string(ops, "OpenCL.std");
            emit(make_inst(kOpExtInstImport, ops));
        } else {
            emit(ins.words);
        }
    }

    // 3. MemoryModel
    emit(make_inst(kOpMemoryModel, {kAddrPhysical64, kMemOpenCL}));

    // 4. EntryPoint
    {
        std::vector<uint32_t> ep_ops = {kExecModelKernel, entry_fn};
        append_spv_string(ep_ops, entry_name);
        for (uint32_t iv : input_builtin_vars) {
            ep_ops.push_back(iv);
        }
        emit(make_inst(kOpEntryPoint, ep_ops));
    }

    // 5. ExecutionMode
    for (const auto& ins : insts) {
        if (ins.op == kOpExecutionMode) emit(ins.words);
    }

    // 6. Debug names (preserve existing + guarantee kernel & param names for panfrost_compile)
    {
        std::vector<uint32_t> nm_ops = {entry_fn};
        append_spv_string(nm_ops, entry_name);
        emit(make_inst(kOpName, nm_ops));
        for (const auto& bv : buf_vars) {
            std::vector<uint32_t> p_ops = {bv.var_id};
            append_spv_string(p_ops, bv.param_name);
            emit(make_inst(kOpName, p_ops));
        }
    }

    // 7. Decorations (export kernel, keep BuiltIn, strip Vulkan-only layout/binding decorations)
    {
        std::vector<uint32_t> dec_ops = {entry_fn, kDecLinkageAttributes};
        append_spv_string(dec_ops, entry_name);
        dec_ops.push_back(kLinkageExport);
        emit(make_inst(kOpDecorate, dec_ops));
    }
    for (const auto& ins : insts) {
        if (ins.op == kOpDecorate && ins.words.size() >= 3) {
            uint32_t target = ins.words[1];
            uint32_t dec = ins.words[2];
            if (dec == kDecBlock || dec == kDecBufferBlock || dec == kDecArrayStride ||
                dec == kDecBinding || dec == kDecDescriptorSet || dec == kDecOffset ||
                dec == kDecNonWritable || dec == kDecNonReadable || dec == kDecCoherent ||
                dec == kDecRestrict || dec == kDecAliased) {
                continue;
            }
            if (buf_var_idx.count(target)) continue;
            emit(ins.words);
        } else if (ins.op == kOpMemberDecorate) {
            continue;
        }
    }

    // 8. Types, constants, global variables (excluding removed SSBO globals)
    bool in_functions = false;
    bool emitted_extra_types = false;
    auto emit_extra_types = [&]() {
        if (emitted_extra_types) return;
        emitted_extra_types = true;
        for (const auto& bv : buf_vars) {
            emit(make_inst(kOpTypePointer, {bv.cw_ptr_ty, kScCrossWorkgroup, bv.elem_ty}));
        }
        std::vector<uint32_t> fn_ops = {new_kernel_fn_ty, void_ty};
        for (const auto& bv : buf_vars) {
            fn_ops.push_back(bv.cw_ptr_ty);
        }
        emit(make_inst(kOpTypeFunction, fn_ops));
    };

    for (const auto& ins : insts) {
        if (ins.op == kOpCapability || ins.op == kOpExtension || ins.op == kOpExtInstImport ||
            ins.op == kOpMemoryModel || ins.op == kOpEntryPoint || ins.op == kOpExecutionMode ||
            ins.op == kOpName || ins.op == kOpMemberName ||
            ins.op == kOpDecorate || ins.op == kOpMemberDecorate) {
            continue;
        }

        if (ins.op == kOpFunction) {
            emit_extra_types();
            in_functions = true;
            if (ins.words.size() >= 5 && ins.words[2] == entry_fn) {
                std::vector<uint32_t> w = ins.words;
                w[4] = new_kernel_fn_ty;
                emit(w);
                for (const auto& bv : buf_vars) {
                    emit(make_inst(kOpFunctionParameter, {bv.cw_ptr_ty, bv.var_id}));
                }
                continue;
            }
            emit(ins.words);
            continue;
        }

        if (!in_functions) {
            // Skip removed SSBO OpVariable declarations.
            if (ins.op == kOpVariable && ins.words.size() >= 4 && buf_var_idx.count(ins.words[2])) {
                continue;
            }
            // Drop unwrapped OpTypeRuntimeArray and single-member BufferBlock structs.
            if (ins.op == kOpTypeRuntimeArray && unwrapped_rarrays.count(ins.words[1])) continue;
            if (ins.op == kOpTypeStruct && unwrapped_structs.count(ins.words[1])) continue;
            // Rewrite StorageBuffer/Uniform-BufferBlock OpTypePointer -> CrossWorkgroup.
            if (ins.op == kOpTypePointer && ins.words.size() == 4) {
                uint32_t sc = ins.words[2];
                uint32_t pt = ins.words[3];
                if (unwrapped_structs.count(pt) || unwrapped_rarrays.count(pt)) continue;
                if (sc == kScStorageBuffer || sc == kScUniform) {
                    emit(make_inst(kOpTypePointer, {ins.words[1], kScCrossWorkgroup, pt}));
                    continue;
                }
            }
            emit(ins.words);
            continue;
        }

        // Inside function body:
        if ((ins.op == kOpAccessChain || ins.op == kOpInBoundsAccessChain) && ins.words.size() >= 4) {
            uint32_t base_id = ins.words[3];
            auto bit = buf_var_idx.find(base_id);
            if (bit != buf_var_idx.end()) {
                const auto& bv = buf_vars[bit->second];
                std::vector<uint32_t> ops = {ins.words[1], ins.words[2], base_id};
                size_t idx_start = bv.unwrapped_single_rarray ? 5u : 4u;
                for (size_t k = idx_start; k < ins.words.size(); ++k) {
                    ops.push_back(ins.words[k]);
                }
                if (ops.size() == 3) {
                    // Direct pointer to element 0 -> OpInBoundsPtrAccessChain needs element index,
                    // fallback to OpCopyObject (opcode 83).
                    emit(make_inst(83, {ins.words[1], ins.words[2], base_id}));
                } else {
                    emit(make_inst(kOpInBoundsPtrAccessChain, ops));
                }
                continue;
            }
        } else if (ins.op == kOpExtInst && ins.words.size() >= 5 && ins.words[3] == glsl_ext_id) {
            uint32_t cl_op = map_glsl450_to_opencl_std(ins.words[4]);
            if (cl_op == 0xFFFFFFFFu) {
                if (err) {
                    *err = "Unsupported GLSL.std.450 opcode " + std::to_string(ins.words[4]) +
                           " in --panfrost Vulkan->OpenCL translator";
                }
                return false;
            }
            std::vector<uint32_t> w = ins.words;
            w[4] = cl_op;
            emit(w);
            continue;
        }

        emit(ins.words);
    }

    out[3] = bound;
    *cl_words = std::move(out);
    return true;
}

bool parse_panfrost_c_output(
    const std::string& c_source,
    const std::string& h_source,
    uint8_t arch,
    std::vector<PanfrostKernelBinary>* out_kernels,
    std::string* err) {
    out_kernels->clear();

    // Match: static const uint32_t <name>[<count>] = { ... };
    std::regex arr_re(
        R"(static\s+const\s+uint32_t\s+([A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*=\s*\{([^}]*)\}\s*;)");
    auto begin = std::sregex_iterator(c_source.begin(), c_source.end(), arr_re);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        std::string arr_name = (*it)[1].str();
        std::string body = (*it)[3].str();
        std::vector<uint32_t> words;
        const char* p = body.c_str();
        while (*p) {
            while (*p && !std::isxdigit(static_cast<unsigned char>(*p))) ++p;
            if (!*p) break;
            char* endp = nullptr;
            unsigned long v = std::strtoul(p, &endp, 0);
            if (endp == p) {
                ++p;
                continue;
            }
            words.push_back(static_cast<uint32_t>(v));
            p = endp;
        }
        if (words.size() < 8) continue;

        // struct bifrost_precompiled_kernel_info ends with:
        //   uint32_t local_size_x, local_size_y, local_size_z, binary_size;
        // followed immediately by binary_size bytes of Valhall machine code.
        size_t header_words = 0;
        uint32_t bin_size = 0;
        // Fast check at standard Mesa 26.3 layout (276 words = 1104 bytes), then scan fallback.
        if (words.size() > 276 && words[275] > 0 && (words[275] % 8u) == 0u &&
            276u + ((words[275] + 3u) / 4u) == words.size()) {
            header_words = 276;
            bin_size = words[275];
        } else {
            for (size_t hw = 4; hw < words.size(); ++hw) {
                uint32_t cand_sz = words[hw - 1];
                if (cand_sz > 0 && (cand_sz % 8u) == 0u &&
                    hw + ((cand_sz + 3u) / 4u) == words.size()) {
                    header_words = hw;
                    bin_size = cand_sz;
                    break;
                }
            }
        }
        if (header_words == 0 || bin_size == 0) {
            if (err) *err = "Could not locate binary_size in panfrost_compile blob: " + arr_name;
            return false;
        }

        PanfrostKernelBinary kb;
        kb.name = arr_name;
        kb.arch = arch;
        kb.work_reg_count = (words.size() > 1) ? words[1] : 32u;
        kb.local_size[0] = words[header_words - 4];
        kb.local_size[1] = words[header_words - 3];
        kb.local_size[2] = words[header_words - 2];
        kb.header_text = h_source;

        size_t bin_words = words.size() - header_words;
        // Trim trailing all-zero 64-bit alignment padding after the shader's final instruction.
        while (bin_words >= 2 && words[header_words + bin_words - 1] == 0u &&
               words[header_words + bin_words - 2] == 0u) {
            bin_words -= 2;
        }
        kb.ebin_bytes.resize(bin_words * 4);
        std::memcpy(kb.ebin_bytes.data(), words.data() + header_words, kb.ebin_bytes.size());
        out_kernels->push_back(std::move(kb));
    }

    if (out_kernels->empty()) {
        if (err) *err = "No compiled kernel arrays found in panfrost_compile C output";
        return false;
    }
    return true;
}

PanfrostCompileResult compile_with_panfrost(
    const std::string& spv_path,
    PanfrostMode mode,
    uint8_t arch,
    const std::string& panfrost_bin,
    bool verbose,
    PanfrostVkBackend vk_backend,
    bool vk_robust,
    bool vk_no_idvs) {
    PanfrostCompileResult res;

    if (mode == PanfrostMode::Vulkan) {
        PanfrostVkCompileOptions vk_opts;
        vk_opts.arch = (arch >= 9 && arch <= 15) ? arch : 11;
        vk_opts.backend = vk_backend;
        vk_opts.robust = vk_robust;
        vk_opts.no_idvs = vk_no_idvs;
        vk_opts.panfrost_vk_bin = panfrost_bin;
        vk_opts.verbose = verbose;
        PanfrostVkCompileOutput vko = compile_vulkan_spv_via_panfrost(spv_path, vk_opts);
        if (!vko.ok) {
            res.error = vko.error;
            return res;
        }
        PanfrostKernelBinary kb;
        kb.name = vko.entry_point;
        kb.stage = vko.stage;
        kb.arch = vko.arch;
        kb.local_size[0] = vko.local_size[0];
        kb.local_size[1] = vko.local_size[1];
        kb.local_size[2] = vko.local_size[2];
        kb.work_reg_count = vko.work_reg_count;
        kb.tls_size = vko.tls_size;
        kb.wls_size = vko.wls_size;
        kb.ebin_bytes = std::move(vko.ebin_bytes);
        res.backend_used = vko.backend_used;
        res.idvs = vko.idvs;
        res.robust = vko.robust;
        res.merge_workgroups = vko.merge_workgroups;
        res.fau_sysval_slots = vko.fau_sysval_slots;
        res.fau_driver_slots = vko.fau_driver_slots;
        res.fau_reserved_words = vko.fau_reserved_words;
        res.fau_total_words = vko.fau_total_words;
        res.used_sysval_words = std::move(vko.used_sysval_words);
        res.used_push_const_words = std::move(vko.used_push_const_words);
        res.kernels.push_back(std::move(kb));
        res.fau_const_words = std::move(vko.fau_const_words);
        res.srt_pka_buf_names = std::move(vko.srt_pka_buf_names);
        res.ok = true;
        return res;
    }

    if (arch < 6 || arch > 14) arch = 10;
    res.backend_used = "opencl";

    std::string bin = find_panfrost_compiler(panfrost_bin);
    if (bin.empty()) {
        res.error = "panfrost_compile not found (checked --panfrost-bin, PANFROST_COMPILE_BIN, "
                    "third_party/prebuilts/panfrost/panfrost_compile, PATH)";
        return res;
    }

    std::string tmpl = "/tmp/spirv2valhall_panfrost_XXXXXX";
    std::vector<char> tmpl_buf(tmpl.begin(), tmpl.end());
    tmpl_buf.push_back('\0');
    char* td_c = ::mkdtemp(tmpl_buf.data());
    if (!td_c) {
        res.error = "Failed to create temporary directory for panfrost_compile";
        return res;
    }
    fs::path td(td_c);
    auto cleanup = [&]() {
        std::error_code ec;
        fs::remove_all(td, ec);
    };

    std::string input_cl_spv = spv_path;
    if (mode == PanfrostMode::OpenCL) {
        // Detect if input is a Vulkan GLCompute shader; if so, translate to OpenCL SPIR-V first.
        std::vector<uint8_t> raw_bytes;
        if (read_file_bytes(spv_path, &raw_bytes, nullptr) && raw_bytes.size() >= 20 &&
            (raw_bytes.size() % 4) == 0) {
            std::vector<uint32_t> words(raw_bytes.size() / 4);
            std::memcpy(words.data(), raw_bytes.data(), raw_bytes.size());
            bool is_vk_compute = false;
            size_t wi = 5;
            while (wi < words.size()) {
                uint16_t wc = static_cast<uint16_t>(words[wi] >> 16);
                uint16_t op = static_cast<uint16_t>(words[wi] & 0xFFFFu);
                if (wc == 0 || wi + wc > words.size()) break;
                if (op == 15 && wc >= 4) {
                    is_vk_compute = (words[wi + 1] == 5);  // SpvExecutionModelGLCompute
                    break;
                }
                wi += wc;
            }
            if (is_vk_compute) {
                std::vector<uint32_t> cl_words;
                if (!translate_vulkan_spv_to_opencl_spv(words, &cl_words, &res.error, &res.arg_bindings)) {
                    cleanup();
                    return res;
                }
                fs::path cl_path = td / "translated_cl.spv";
                std::ofstream ofs(cl_path, std::ios::binary);
                ofs.write(reinterpret_cast<const char*>(cl_words.data()),
                          static_cast<std::streamsize>(cl_words.size() * 4));
                ofs.close();
                input_cl_spv = cl_path.string();
            }
        }
    }

    fs::path out_h = td / "panfrost_out.h";
    fs::path out_c = td / "panfrost_out.c";
    fs::path err_log = td / "panfrost_stderr.txt";

    std::string preload = find_compat_preload_lib(bin);
    std::ostringstream cmd;
    if (!preload.empty()) {
        cmd << "LD_PRELOAD=\"" << preload << "${LD_PRELOAD:+:$LD_PRELOAD}\" ";
    }
    cmd << "\"" << bin << "\" panlib " << static_cast<int>(arch)
        << " \"" << input_cl_spv << "\" \"" << out_h.string() << "\" \"" << out_c.string() << "\""
        << " 2>\"" << err_log.string() << "\"";

    if (verbose) {
        std::cerr << "[panfrost] running: " << cmd.str() << "\n";
    }

    int rc = std::system(cmd.str().c_str());
    std::string stderr_text;
    read_file_text(err_log.string(), &stderr_text, nullptr);
    if (rc != 0) {
        res.error = "panfrost_compile failed (rc=" + std::to_string(rc) + ")";
        if (!stderr_text.empty()) {
            res.error += ": " + stderr_text;
        }
        cleanup();
        return res;
    }

    std::string c_src, h_src;
    if (!read_file_text(out_c.string(), &c_src, &res.error) ||
        !read_file_text(out_h.string(), &h_src, &res.error)) {
        cleanup();
        return res;
    }

    if (!parse_panfrost_c_output(c_src, h_src, arch, &res.kernels, &res.error)) {
        cleanup();
        return res;
    }

    cleanup();
    res.ok = true;
    return res;
}


static void devirtualize_panfrost_fau(
    valhall::compiler::ValhallSsaModule& ssa,
    const std::vector<PanfrostArgBinding>& bindings) {
    if (ssa.blocks.empty()) return;
    // Map FAU word_idx -> (key, is_hi) for 64-bit buffer pointers at offset 24 + 8*i.
    std::unordered_map<uint32_t, std::pair<std::string, bool>> word_to_buf;
    for (const auto& ab : bindings) {
        uint32_t w_lo = ab.fau_offset / 4u;
        word_to_buf[w_lo] = {ab.key, false};
        word_to_buf[w_lo + 1u] = {ab.key, true};
    }

    // Check if any LoadFau references a bound buffer or num_workgroups (words 0, 1, 2).
    std::unordered_map<std::string, std::pair<valhall::compiler::SsaId, valhall::compiler::SsaId>> buf_lo_hi;
    valhall::compiler::SsaId nwg_id = valhall::compiler::kInvalidSsa;
    std::vector<valhall::compiler::SsaId> entry_prefix;

    auto emit_entry = [&](valhall::compiler::SsaNode n) -> valhall::compiler::SsaId {
        n.id = static_cast<valhall::compiler::SsaId>(ssa.nodes.size());
        n.block_id = 0;
        ssa.nodes.push_back(std::move(n));
        entry_prefix.push_back(ssa.nodes.back().id);
        return ssa.nodes.back().id;
    };

    auto get_buf_lo_hi = [&](const std::string& key) -> std::pair<valhall::compiler::SsaId, valhall::compiler::SsaId> {
        auto it = buf_lo_hi.find(key);
        if (it != buf_lo_hi.end()) return it->second;
        valhall::compiler::SsaNode sb{};
        sb.op = valhall::tables::SsaOp::LoadSidebandU64;
        sb.type = valhall::compiler::ValType::U64;
        sb.imm_u32 = 0x10u;
        sb.aux_str = key;
        auto base = emit_entry(std::move(sb));

        valhall::compiler::SsaNode sp{};
        sp.op = valhall::tables::SsaOp::Split64;
        sp.type = valhall::compiler::ValType::TupleU32x2;
        sp.args = {base};
        auto sp_id = emit_entry(std::move(sp));

        valhall::compiler::SsaNode lo{};
        lo.op = valhall::tables::SsaOp::TupleExtract;
        lo.type = valhall::compiler::ValType::U32;
        lo.imm_u32 = 0;
        lo.args = {sp_id};
        auto lo_id = emit_entry(std::move(lo));

        valhall::compiler::SsaNode hi{};
        hi.op = valhall::tables::SsaOp::TupleExtract;
        hi.type = valhall::compiler::ValType::U32;
        hi.imm_u32 = 1;
        hi.args = {sp_id};
        auto hi_id = emit_entry(std::move(hi));

        buf_lo_hi[key] = {lo_id, hi_id};
        return {lo_id, hi_id};
    };

    auto get_nwg = [&]() -> valhall::compiler::SsaId {
        if (nwg_id != valhall::compiler::kInvalidSsa) return nwg_id;
        valhall::compiler::SsaNode bn{};
        bn.op = valhall::tables::SsaOp::LoadBuiltinInput;
        bn.type = valhall::compiler::ValType::V3U32;
        bn.imm_u32 = 24u;  // SpvBuiltInNumWorkgroups
        bn.aux_str = "gl_NumWorkGroups";
        nwg_id = emit_entry(std::move(bn));
        return nwg_id;
    };

    size_t orig_count = ssa.nodes.size();
    for (size_t i = 0; i < orig_count; ++i) {
        if (ssa.nodes[i].op != valhall::tables::SsaOp::LoadFau) continue;
        uint32_t widx = ssa.nodes[i].imm_u32;
        auto bit = word_to_buf.find(widx);
        if (bit != word_to_buf.end()) {
            auto [lo_id, hi_id] = get_buf_lo_hi(bit->second.first);
            ssa.nodes[i].op = valhall::tables::SsaOp::Copy;
            ssa.nodes[i].args = {bit->second.second ? hi_id : lo_id};
            ssa.nodes[i].imm_u32 = 0;
            ssa.nodes[i].aux_str = bit->second.first;
        } else if (widx < 3u) {
            auto nw = get_nwg();
            ssa.nodes[i].op = valhall::tables::SsaOp::VecExtract;
            ssa.nodes[i].type = valhall::compiler::ValType::U32;
            ssa.nodes[i].args = {nw};
            ssa.nodes[i].imm_u32 = widx;
        }
    }

    if (!entry_prefix.empty()) {
        auto& b0_nodes = ssa.blocks[0].nodes;
        b0_nodes.insert(b0_nodes.begin(), entry_prefix.begin(), entry_prefix.end());
    }
}


static void devirtualize_panfrost_vk_fau(
    valhall::compiler::ValhallSsaModule& ssa,
    const PanfrostCompileResult& cres,
    const PanfrostKernelBinary& kb) {
    if (ssa.blocks.empty()) return;

    valhall::dataflow::FauSummaryMap summary;
    std::unordered_map<uint32_t, uint32_t> nwg_word_to_comp;

    // Decode panvk compacted sysval FAU slots (64-bit slots packed in ascending order).
    const uint32_t sysval_count = static_cast<uint32_t>(cres.used_sysval_words.size());
    for (uint32_t i = 0; i < sysval_count; ++i) {
        uint32_t orig_slot = cres.used_sysval_words[i];
        uint32_t fau_byte = i * 8u;
        uint32_t fau_word = i * 2u;
        if (kb.stage == valhall::ShaderStage::Compute) {
            // panvk_compute_sysvals: slot 0={base.x,base.y}, slot 1={base.z,_pad},
            // slot 5={num_wg.x,num_wg.y}, slot 6={num_wg.z,local_sz.x}, slot 7={local_sz.y,local_sz.z}
            if (orig_slot == 0u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(0u);
                summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_const_u32(0u);
            } else if (orig_slot == 1u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(0u);
            } else if (orig_slot == 5u) {
                nwg_word_to_comp[fau_word] = 0u;
                nwg_word_to_comp[fau_word + 1u] = 1u;
            } else if (orig_slot == 6u) {
                nwg_word_to_comp[fau_word] = 2u;
                summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_const_u32(kb.local_size[0]);
            } else if (orig_slot == 7u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(kb.local_size[1]);
                summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_const_u32(kb.local_size[2]);
            }
        }
    }

    // Rewrite num_workgroups sysval loads to VecExtract(LoadBuiltinInput(NumWorkgroups), comp).
    if (!nwg_word_to_comp.empty()) {
        valhall::compiler::SsaId nwg_id = valhall::compiler::kInvalidSsa;
        std::vector<valhall::compiler::SsaId> entry_prefix;
        size_t orig_count = ssa.nodes.size();
        for (size_t idx = 0; idx < orig_count; ++idx) {
            if (ssa.nodes[idx].op != valhall::tables::SsaOp::LoadFau) continue;
            auto it = nwg_word_to_comp.find(ssa.nodes[idx].imm_u32);
            if (it == nwg_word_to_comp.end()) continue;
            if (nwg_id == valhall::compiler::kInvalidSsa) {
                valhall::compiler::SsaNode bn{};
                bn.id = static_cast<valhall::compiler::SsaId>(ssa.nodes.size());
                bn.block_id = 0;
                bn.op = valhall::tables::SsaOp::LoadBuiltinInput;
                bn.type = valhall::compiler::ValType::V3U32;
                bn.imm_u32 = 24u;  // SpvBuiltInNumWorkgroups
                bn.aux_str = "gl_NumWorkGroups";
                ssa.nodes.push_back(std::move(bn));
                nwg_id = ssa.nodes.back().id;
                entry_prefix.push_back(nwg_id);
            }
            ssa.nodes[idx].op = valhall::tables::SsaOp::VecExtract;
            ssa.nodes[idx].type = valhall::compiler::ValType::U32;
            ssa.nodes[idx].args = {nwg_id};
            ssa.nodes[idx].imm_u32 = it->second;
            ssa.nodes[idx].aux_str.clear();
        }
        if (!entry_prefix.empty()) {
            auto& b0 = ssa.blocks[0].nodes;
            b0.insert(b0.begin(), entry_prefix.begin(), entry_prefix.end());
        }
    }

    // Decode panvk compacted user push-constant FAU slots into canonical "push_const" loads.
    for (uint32_t j = 0; j < cres.used_push_const_words.size(); ++j) {
        uint32_t pc_slot = cres.used_push_const_words[j];
        uint32_t fau_byte = (sysval_count + j) * 8u;
        summary[fau_byte] = valhall::dataflow::SymExpr::make_buffer_load(-1, -1, "push_const", pc_slot * 8u, 4u);
        summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_buffer_load(-1, -1, "push_const", pc_slot * 8u + 4u, 4u);
    }

    // Seed promoted FAU immediate constants.
    for (const auto& [w, val] : cres.fau_const_words) {
        summary[w * 4u] = valhall::dataflow::SymExpr::make_const_u32(val);
    }

    if (!summary.empty()) {
        valhall::dataflow::DeFauDevirtOptions dopt;
        dopt.record_fallbacks = false;
        valhall::dataflow::run_ssa_defau_devirt(ssa, summary, dopt);
    }
}

static const char* stage_name_str(valhall::ShaderStage st) {
    switch (st) {
        case valhall::ShaderStage::Vertex: return "Vertex";
        case valhall::ShaderStage::Fragment: return "Fragment";
        case valhall::ShaderStage::Compute:
        default: return "Compute";
    }
}

int run_panfrost_pipeline(const std::string& spv_path, const PanfrostRunOptions& opts) {
    uint8_t arch = (opts.arch != 0) ? opts.arch : (opts.mode == PanfrostMode::Vulkan ? 11 : 10);
    PanfrostCompileResult cres = compile_with_panfrost(
        spv_path, opts.mode, arch, opts.panfrost_bin, opts.verbose,
        opts.vk_backend, opts.vk_robust, opts.vk_no_idvs);
    if (cres.ok && !cres.kernels.empty()) {
        arch = cres.kernels[0].arch;
    }
    if (!cres.ok) {
        std::cerr << "spirv2valhall (--panfrost): " << cres.error << "\n";
        return 1;
    }

    const char* mode_label = (opts.mode == PanfrostMode::OpenCL) ? "panfrost-cl" : "panfrost";

    if (!opts.out_bin_prefix.empty()) {
        for (size_t i = 0; i < cres.kernels.size(); ++i) {
            std::string path = opts.out_bin_prefix + "_" + std::to_string(i) + ".bin";
            std::ofstream ofs(path, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(cres.kernels[i].ebin_bytes.data()),
                      static_cast<std::streamsize>(cres.kernels[i].ebin_bytes.size()));
        }
    }

    // Lower primary kernel to ValhallSSA if requested.
    bool need_ssa = opts.dump_ssa || !opts.dump_ssa_json_path.empty() ||
                    !opts.dump_equiv_json_path.empty() || opts.dump_spirv ||
                    !opts.out_spirv_prefix.empty();

    valhall::compiler::Pass1Result primary_p1;
    if (!cres.kernels.empty() && need_ssa) {
        const auto& kb0 = cres.kernels[0];
        valhall::ValhallTranslateOptions vopts{};
        vopts.stage = kb0.stage;
        vopts.arch = arch;
        vopts.local_size_x = kb0.local_size[0];
        vopts.local_size_y = kb0.local_size[1];
        vopts.local_size_z = kb0.local_size[2];
        vopts.srt_pka_buf_names = cres.srt_pka_buf_names;
        for (const auto& [w, val] : cres.fau_const_words) {
            uint32_t byte_off = w * 4u;
            vopts.fau_fcst_words[byte_off] = val;
            auto sit = cres.srt_pka_buf_names.find(val);
            if (sit != cres.srt_pka_buf_names.end()) {
                vopts.fau_pka_buf_names[byte_off] = sit->second;
            }
        }
        std::vector<valhall::DecodedValhallInst> pri_insts;
        for (size_t i = 0; i + 8 <= kb0.ebin_bytes.size(); i += 8) {
            uint64_t word = 0;
            std::memcpy(&word, kb0.ebin_bytes.data() + i, 8);
            pri_insts.push_back(valhall::decode_valhall_instruction(word, arch, i));
        }
        valhall::cfg::ControlFlowGraph pri_cfg = valhall::cfg::build_cfg(pri_insts);
        valhall::cfg::ControlFlowGraph sec_cfg;
        primary_p1 = valhall::compiler::run_pass1_lowering(pri_cfg, sec_cfg, vopts);
        if (primary_p1.ok && !cres.arg_bindings.empty()) {
            devirtualize_panfrost_fau(primary_p1.ssa, cres.arg_bindings);
        }
        if (primary_p1.ok && opts.mode == PanfrostMode::Vulkan) {
            devirtualize_panfrost_vk_fau(primary_p1.ssa, cres, kb0);
        }
        if (opts.equiv_opts.run_ssa_opt && primary_p1.ok) {
            valhall::dataflow::SsaOptPipelineOptions oopt;
            oopt.fp_mode = opts.equiv_opts.fp_fast_math ? valhall::dataflow::FpOptMode::FastMath : valhall::dataflow::FpOptMode::StrictIEEE;
            oopt.enable_fp_tile_rewrites = opts.equiv_opts.enable_fp_tile_rewrites;
            oopt.enable_fp_cpp_rewrites = opts.equiv_opts.enable_fp_cpp_rewrites;
            valhall::dataflow::run_ssa_opt_pipeline(primary_p1.ssa, oopt);
        }
    }

    if (!opts.dump_ssa_json_path.empty()) {
        std::ofstream ofs(opts.dump_ssa_json_path);
        if (!ofs) {
            std::cerr << "spirv2valhall: failed to write " << opts.dump_ssa_json_path << "\n";
            return 1;
        }
        ofs << valhall::compiler::serialize_ssa_module_json(primary_p1.ssa);
    }

    if (!opts.dump_equiv_json_path.empty()) {
        valhall::spirv2ssa::Spirv2SsaResult direct_res;
        if (opts.mode == PanfrostMode::Vulkan) {
            std::vector<uint8_t> raw_bytes;
            std::string rerr;
            if (read_file_bytes(spv_path, &raw_bytes, &rerr) && (raw_bytes.size() % 4) == 0) {
                std::vector<uint32_t> spv_words(raw_bytes.size() / 4);
                std::memcpy(spv_words.data(), raw_bytes.data(), raw_bytes.size());
                valhall::spirv2ssa::Spirv2SsaOptions sopts;
                direct_res = valhall::spirv2ssa::lower_spirv_to_valhall_ssa(spv_words, sopts);
                if (opts.equiv_opts.run_ssa_opt && direct_res.ok) {
                    valhall::dataflow::SsaOptPipelineOptions oopt;
                    oopt.fp_mode = opts.equiv_opts.fp_fast_math ? valhall::dataflow::FpOptMode::FastMath : valhall::dataflow::FpOptMode::StrictIEEE;
                    oopt.enable_fp_tile_rewrites = opts.equiv_opts.enable_fp_tile_rewrites;
                    oopt.enable_fp_cpp_rewrites = opts.equiv_opts.enable_fp_cpp_rewrites;
                    valhall::dataflow::run_ssa_opt_pipeline(direct_res.ssa, oopt);
                }
            }
        }
        std::ostringstream pc_js;
        pc_js << "{";
        if (!cres.kernels.empty()) {
            const auto& kb0 = cres.kernels[0];
            size_t n_insts = kb0.ebin_bytes.size() / 8;
            bool first_pc = true;
            for (size_t i = 0; i < n_insts; ++i) {
                uint64_t word = 0;
                std::memcpy(&word, kb0.ebin_bytes.data() + i * 8, 8);
                if (word == 0u) continue;
                auto di = valhall::decode_valhall_instruction(word, arch, i * 8);
                if (!first_pc) pc_js << ",";
                first_pc = false;
                pc_js << "\"" << (i * 8) << "\":\"";
                for (char ch : di.disassembly) {
                    if (ch == '\\' || ch == '"') pc_js << '\\';
                    pc_js << ch;
                }
                pc_js << "\"";
            }
        }
        pc_js << "}";
        std::ostringstream ab_js;
        ab_js << "[";
        for (size_t i = 0; i < cres.arg_bindings.size(); ++i) {
            const auto& ab = cres.arg_bindings[i];
            if (i > 0) ab_js << ",";
            ab_js << "{\"arg_idx\":" << ab.arg_idx
                  << ",\"fau_offset\":" << ab.fau_offset
                  << ",\"set\":" << ab.descriptor_set
                  << ",\"binding\":" << ab.binding
                  << ",\"key\":\"" << ab.key << "\"}";
        }
        ab_js << "]";
        std::ostringstream fau_js;
        fau_js << "{";
        bool first_fau = true;
        for (const auto& [w, val] : cres.fau_const_words) {
            if (!first_fau) fau_js << ",";
            first_fau = false;
            fau_js << "\"" << w << "\":" << val;
        }
        fau_js << "}";
        std::ostringstream pka_js;
        pka_js << "{";
        bool first_pka = true;
        for (const auto& [pid, name] : cres.srt_pka_buf_names) {
            if (!first_pka) pka_js << ",";
            first_pka = false;
            pka_js << "\"" << pid << "\":\"" << name << "\"";
        }
        pka_js << "}";
        const char* st_str = cres.kernels.empty() ? "Compute" : stage_name_str(cres.kernels[0].stage);
        std::ostringstream js;
        js << "{\n"
           << "  \"compiler\": \"" << mode_label << "\",\n"
           << "  \"panfrost_backend\": \"" << cres.backend_used << "\",\n"
           << "  \"panfrost_idvs\": " << (cres.idvs ? "true" : "false") << ",\n"
           << "  \"panfrost_robust\": " << (cres.robust ? "true" : "false") << ",\n"
           << "  \"panfrost_merge_workgroups\": " << (cres.merge_workgroups ? "true" : "false") << ",\n"
           << "  \"panfrost_fau_sysval_slots\": " << cres.fau_sysval_slots << ",\n"
           << "  \"panfrost_fau_driver_slots\": " << cres.fau_driver_slots << ",\n"
           << "  \"panfrost_fau_reserved_words\": " << cres.fau_reserved_words << ",\n"
           << "  \"panfrost_fau_total_words\": " << cres.fau_total_words << ",\n"
           << "  \"stage\": \"" << st_str << "\",\n"
           << "  \"arch\": " << static_cast<int>(arch) << ",\n"
           << "  \"mbs2_context\": {\"fau_fcst_words\": " << fau_js.str()
           << ", \"srt_pka_buf_names\": " << pka_js.str() << "},\n"
           << "  \"local_size\": ["
           << (cres.kernels.empty() ? 1 : cres.kernels[0].local_size[0]) << ","
           << (cres.kernels.empty() ? 1 : cres.kernels[0].local_size[1]) << ","
           << (cres.kernels.empty() ? 1 : cres.kernels[0].local_size[2]) << "],\n"
           << "  \"panfrost_arg_bindings\": " << ab_js.str() << ",\n"
           << "  \"pc_disasm\": " << pc_js.str() << ",\n"
           << "  \"direct_ok\": " << (direct_res.ok ? "true" : "false") << ",\n"
           << "  \"valhall_ok\": " << (primary_p1.ok ? "true" : "false") << ",\n"
           << "  \"direct_ssa\": " << valhall::compiler::serialize_ssa_module_json(direct_res.ssa) << ",\n"
           << "  \"valhall_ssa\": " << valhall::compiler::serialize_ssa_module_json(primary_p1.ssa) << "\n"
           << "}\n";
        std::ofstream ofs(opts.dump_equiv_json_path);
        if (!ofs) {
            std::cerr << "spirv2valhall: failed to write " << opts.dump_equiv_json_path << "\n";
            return 1;
        }
        ofs << js.str();
    }

    if (opts.dump_spirv || !opts.out_spirv_prefix.empty()) {
        for (size_t i = 0; i < cres.kernels.size(); ++i) {
            const auto& kb = cres.kernels[i];
            valhall::ValhallTranslateOptions vopts{};
            vopts.stage = kb.stage;
            vopts.arch = arch;
            vopts.local_size_x = kb.local_size[0];
            vopts.local_size_y = kb.local_size[1];
            vopts.local_size_z = kb.local_size[2];
            auto tr = valhall::translate_valhall_to_spirv(kb.ebin_bytes.data(), kb.ebin_bytes.size(), vopts);
            if (!tr.ok) {
                std::cerr << "spirv2valhall (--panfrost): SPIR-V lift failed for " << kb.name
                          << ": " << tr.error_message << "\n";
                return 1;
            }
            if (!opts.out_spirv_prefix.empty()) {
                std::string path = opts.out_spirv_prefix + "_" + std::to_string(i) + ".spv";
                std::ofstream ofs(path, std::ios::binary);
                ofs.write(reinterpret_cast<const char*>(tr.spirv_words.data()),
                          static_cast<std::streamsize>(tr.spirv_words.size() * 4));
            }
            if (opts.dump_spirv && !opts.quiet) {
                std::cout << "; lifted SPIR-V for " << kb.name << " (" << tr.spirv_words.size() << " words)\n";
            }
        }
    }

    if (!opts.quiet) {
        if (opts.dump_ssa) {
            std::cout << "; spirv2valhall --" << mode_label << " --dump-ssa arch=v"
                      << static_cast<int>(arch) << " kernel=\"" << cres.kernels[0].name << "\"\n\n"
                      << valhall::compiler::format_ssa_module_text(primary_p1.ssa);
        } else {
            std::cout << "; spirv2valhall compiler=" << mode_label
                      << " arch=v" << static_cast<int>(arch)
                      << " kernels=" << cres.kernels.size() << "\n";
            for (size_t ki = 0; ki < cres.kernels.size(); ++ki) {
                const auto& kb = cres.kernels[ki];
                size_t n_insts = kb.ebin_bytes.size() / 8;
                size_t invalid_cnt = 0;
                std::ostringstream body;
                for (size_t i = 0; i < n_insts; ++i) {
                    uint64_t word = 0;
                    std::memcpy(&word, kb.ebin_bytes.data() + i * 8, 8);
                    if (word == 0u) {
                        body << "  0x" << std::hex << std::setw(4) << std::setfill('0') << (i * 8)
                             << ": " << std::setw(16) << word << std::dec << "  ; <align_pad>\n";
                        continue;
                    }
                    valhall::DecodedValhallInst inst = valhall::decode_valhall_instruction(word, arch, i * 8);
                    if (!inst.valid) ++invalid_cnt;
                    body << "  0x" << std::hex << std::setw(4) << std::setfill('0') << (i * 8)
                         << ": " << std::setw(16) << word << std::dec << "  "
                         << inst.disassembly << "\n";
                }
                std::cout << "\n; === PANFROST KERNEL[" << ki << "] \"" << kb.name
                          << "\" wg=(" << kb.local_size[0] << "," << kb.local_size[1] << ","
                          << kb.local_size[2] << ") regs=" << kb.work_reg_count
                          << " code_size=" << kb.ebin_bytes.size()
                          << " insts=" << n_insts << " invalid=" << invalid_cnt << " ===\n"
                          << body.str();
            }
        }
    }

    return 0;
}

}  // namespace valhall::spirv2valhall
