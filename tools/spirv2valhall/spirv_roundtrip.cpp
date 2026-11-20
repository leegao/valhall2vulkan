#include "spirv_roundtrip.h"
#include "valhall_to_spirv.h"
#include "spirv_reflect.h"

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>

namespace valhall::spirv2valhall {

namespace {

std::string decode_spv_literal_string(const uint32_t* words, size_t max_words, size_t* out_words_consumed) {
    std::string s;
    for (size_t i = 0; i < max_words; ++i) {
        uint32_t w = words[i];
        for (int b = 0; b < 4; ++b) {
            char c = static_cast<char>((w >> (b * 8)) & 0xFFu);
            if (c == '\0') {
                if (out_words_consumed) *out_words_consumed = i + 1;
                return s;
            }
            s.push_back(c);
        }
    }
    if (out_words_consumed) *out_words_consumed = max_words;
    return s;
}

const char* spv_capability_name(uint32_t c) {
    switch (c) {
        case 0: return "Matrix";
        case 1: return "Shader";
        case 9: return "Float16";
        case 10: return "Float64";
        case 11: return "Int64";
        case 22: return "Int16";
        case 39: return "Int8";
        case 5347: return "PhysicalStorageBufferAddresses";
        default: return nullptr;
    }
}

const char* spv_exec_model_name(uint32_t em) {
    switch (em) {
        case 0: return "Vertex";
        case 1: return "TessellationControl";
        case 2: return "TessellationEvaluation";
        case 3: return "Geometry";
        case 4: return "Fragment";
        case 5: return "GLCompute";
        default: return nullptr;
    }
}

const char* spv_exec_mode_name(uint32_t em) {
    switch (em) {
        case 7: return "OriginUpperLeft";
        case 17: return "LocalSize";
        case 21: return "DepthReplacing";
        default: return nullptr;
    }
}

const char* spv_storage_class_name(uint32_t sc) {
    switch (sc) {
        case 0: return "UniformConstant";
        case 1: return "Input";
        case 2: return "Uniform";
        case 3: return "Output";
        case 4: return "Workgroup";
        case 6: return "Private";
        case 7: return "Function";
        case 9: return "PushConstant";
        case 12: return "StorageBuffer";
        case 5349: return "PhysicalStorageBuffer";
        default: return nullptr;
    }
}

const char* spv_decoration_name(uint32_t dec) {
    switch (dec) {
        case 2: return "Block";
        case 3: return "BufferBlock";
        case 6: return "ArrayStride";
        case 11: return "BuiltIn";
        case 14: return "Flat";
        case 24: return "NonWritable";
        case 30: return "Location";
        case 33: return "Binding";
        case 34: return "DescriptorSet";
        case 35: return "Offset";
        default: return nullptr;
    }
}

const char* spv_builtin_name(uint32_t bi) {
    switch (bi) {
        case 0: return "Position";
        case 1: return "PointSize";
        case 15: return "FragCoord";
        case 22: return "FragDepth";
        case 26: return "WorkgroupId";
        case 27: return "LocalInvocationId";
        case 28: return "GlobalInvocationId";
        case 42: return "VertexIndex";
        case 43: return "InstanceIndex";
        default: return nullptr;
    }
}

const char* spv_glsl450_name(uint32_t op) {
    switch (op) {
        case 1: return "Round";
        case 2: return "RoundEven";
        case 3: return "Trunc";
        case 4: return "FAbs";
        case 5: return "SAbs";
        case 8: return "Floor";
        case 9: return "Ceil";
        case 10: return "Fract";
        case 13: return "Sin";
        case 14: return "Cos";
        case 26: return "Pow";
        case 29: return "Exp2";
        case 30: return "Log2";
        case 31: return "Sqrt";
        case 32: return "InverseSqrt";
        case 37: return "FMin";
        case 38: return "UMin";
        case 39: return "SMin";
        case 40: return "FMax";
        case 41: return "UMax";
        case 42: return "SMax";
        case 43: return "FClamp";
        case 44: return "UClamp";
        case 45: return "SClamp";
        case 46: return "FMix";
        case 50: return "Fma";
        case 58: return "PackHalf2x16";
        case 62: return "UnpackHalf2x16";
        case 73: return "FindILsb";
        case 74: return "FindSMsb";
        case 75: return "FindUMsb";
        default: return nullptr;
    }
}

const char* spv_opcode_name(uint16_t op) {
    switch (op) {
        case 0: return "OpNop";
        case 1: return "OpUndef";
        case 3: return "OpSource";
        case 5: return "OpName";
        case 6: return "OpMemberName";
        case 10: return "OpExtension";
        case 11: return "OpExtInstImport";
        case 12: return "OpExtInst";
        case 14: return "OpMemoryModel";
        case 15: return "OpEntryPoint";
        case 16: return "OpExecutionMode";
        case 17: return "OpCapability";
        case 19: return "OpTypeVoid";
        case 20: return "OpTypeBool";
        case 21: return "OpTypeInt";
        case 22: return "OpTypeFloat";
        case 23: return "OpTypeVector";
        case 24: return "OpTypeMatrix";
        case 25: return "OpTypeImage";
        case 26: return "OpTypeSampler";
        case 27: return "OpTypeSampledImage";
        case 28: return "OpTypeArray";
        case 29: return "OpTypeRuntimeArray";
        case 30: return "OpTypeStruct";
        case 32: return "OpTypePointer";
        case 33: return "OpTypeFunction";
        case 41: return "OpConstantTrue";
        case 42: return "OpConstantFalse";
        case 43: return "OpConstant";
        case 44: return "OpConstantComposite";
        case 46: return "OpConstantNull";
        case 54: return "OpFunction";
        case 55: return "OpFunctionParameter";
        case 56: return "OpFunctionEnd";
        case 57: return "OpFunctionCall";
        case 59: return "OpVariable";
        case 61: return "OpLoad";
        case 62: return "OpStore";
        case 65: return "OpAccessChain";
        case 71: return "OpDecorate";
        case 72: return "OpMemberDecorate";
        case 79: return "OpVectorShuffle";
        case 80: return "OpCompositeConstruct";
        case 81: return "OpCompositeExtract";
        case 82: return "OpCompositeInsert";
        case 87: return "OpImageSampleImplicitLod";
        case 88: return "OpImageSampleExplicitLod";
        case 95: return "OpImageFetch";
        case 109: return "OpConvertFToU";
        case 110: return "OpConvertFToS";
        case 111: return "OpConvertSToF";
        case 112: return "OpConvertUToF";
        case 113: return "OpUConvert";
        case 114: return "OpSConvert";
        case 115: return "OpFConvert";
        case 121: return "OpConvertPtrToU";
        case 122: return "OpConvertUToPtr";
        case 124: return "OpBitcast";
        case 126: return "OpSNegate";
        case 127: return "OpFNegate";
        case 128: return "OpIAdd";
        case 129: return "OpFAdd";
        case 130: return "OpISub";
        case 131: return "OpFSub";
        case 132: return "OpIMul";
        case 133: return "OpFMul";
        case 134: return "OpUDiv";
        case 135: return "OpSDiv";
        case 136: return "OpFDiv";
        case 137: return "OpUMod";
        case 138: return "OpSRem";
        case 139: return "OpSMod";
        case 140: return "OpFRem";
        case 148: return "OpDot";
        case 151: return "OpIAddCarry";
        case 152: return "OpISubBorrow";
        case 153: return "OpUMulExtended";
        case 154: return "OpSMulExtended";
        case 164: return "OpLogicalEqual";
        case 165: return "OpLogicalNotEqual";
        case 166: return "OpLogicalOr";
        case 167: return "OpLogicalAnd";
        case 168: return "OpLogicalNot";
        case 169: return "OpSelect";
        case 170: return "OpIEqual";
        case 171: return "OpINotEqual";
        case 172: return "OpUGreaterThan";
        case 173: return "OpSGreaterThan";
        case 174: return "OpUGreaterThanEqual";
        case 175: return "OpSGreaterThanEqual";
        case 176: return "OpULessThan";
        case 177: return "OpSLessThan";
        case 178: return "OpULessThanEqual";
        case 179: return "OpSLessThanEqual";
        case 180: return "OpFOrdEqual";
        case 181: return "OpFUnordEqual";
        case 182: return "OpFOrdNotEqual";
        case 183: return "OpFUnordNotEqual";
        case 184: return "OpFOrdLessThan";
        case 185: return "OpFUnordLessThan";
        case 186: return "OpFOrdGreaterThan";
        case 187: return "OpFUnordGreaterThan";
        case 188: return "OpFOrdLessThanEqual";
        case 189: return "OpFUnordLessThanEqual";
        case 190: return "OpFOrdGreaterThanEqual";
        case 191: return "OpFUnordGreaterThanEqual";
        case 194: return "OpShiftRightLogical";
        case 195: return "OpShiftRightArithmetic";
        case 196: return "OpShiftLeftLogical";
        case 197: return "OpBitwiseOr";
        case 198: return "OpBitwiseXor";
        case 199: return "OpBitwiseAnd";
        case 200: return "OpNot";
        case 201: return "OpBitFieldInsert";
        case 202: return "OpBitFieldSExtract";
        case 203: return "OpBitFieldUExtract";
        case 204: return "OpBitReverse";
        case 205: return "OpBitCount";
        case 207: return "OpDPdx";
        case 208: return "OpDPdy";
        case 209: return "OpFwidth";
        case 224: return "OpControlBarrier";
        case 225: return "OpMemoryBarrier";
        case 227: return "OpAtomicLoad";
        case 228: return "OpAtomicStore";
        case 229: return "OpAtomicExchange";
        case 230: return "OpAtomicCompareExchange";
        case 234: return "OpAtomicIAdd";
        case 235: return "OpAtomicISub";
        case 236: return "OpAtomicSMin";
        case 237: return "OpAtomicUMin";
        case 238: return "OpAtomicSMax";
        case 239: return "OpAtomicUMax";
        case 240: return "OpAtomicAnd";
        case 241: return "OpAtomicOr";
        case 242: return "OpAtomicXor";
        case 245: return "OpPhi";
        case 246: return "OpLoopMerge";
        case 247: return "OpSelectionMerge";
        case 248: return "OpLabel";
        case 249: return "OpBranch";
        case 250: return "OpBranchConditional";
        case 251: return "OpSwitch";
        case 252: return "OpKill";
        case 253: return "OpReturn";
        case 254: return "OpReturnValue";
        case 255: return "OpUnreachable";
        case 4416: return "OpDemoteToHelperInvocation";
        default: return nullptr;
    }
}

bool op_has_type_and_result(uint16_t op) {
    if (op == 12 || op == 41 || op == 42 || op == 43 || op == 44 || op == 46 ||
        op == 54 || op == 55 || op == 57 || op == 59 || op == 61 || op == 65 ||
        op == 79 || op == 80 || op == 81 || op == 82 || op == 87 || op == 88 ||
        op == 95 || op == 245 || op == 1) {
        return true;
    }
    if (op >= 109 && op <= 209) return true;
    if (op >= 227 && op <= 242 && op != 228) return true;
    return false;
}

bool op_has_result_only(uint16_t op) {
    return (op >= 19 && op <= 33) || op == 11 || op == 248;
}

std::string id_str(uint32_t id, const std::map<uint32_t, std::string>& names) {
    auto it = names.find(id);
    if (it != names.end() && !it->second.empty()) {
        return "%" + it->second;
    }
    return "%" + std::to_string(id);
}

void validate_with_reflect(LiftedSpirvShader* sh) {
    if (!sh || !sh->ok || sh->spirv_words.empty()) return;
    SpvReflectShaderModule mod{};
    SpvReflectResult rr = spvReflectCreateShaderModule(
        sh->spirv_words.size() * sizeof(uint32_t), sh->spirv_words.data(), &mod);
    if (rr == SPV_REFLECT_RESULT_SUCCESS) {
        sh->reflect_valid = true;
        sh->reflect_entry_point = mod.entry_point_name ? mod.entry_point_name : "";
        sh->reflect_shader_stage = static_cast<uint32_t>(mod.shader_stage);
        spvReflectDestroyShaderModule(&mod);
    } else {
        sh->reflect_valid = false;
    }
}

}  // namespace

std::string format_spirv_disassembly(const std::vector<uint32_t>& words) {
    if (words.size() < 5 || words[0] != 0x07230203u) {
        return "; <invalid SPIR-V binary>\n";
    }

    // Pass 1: collect OpName symbols for readable IDs
    std::map<uint32_t, std::string> names;
    {
        size_t pos = 5;
        while (pos < words.size()) {
            uint32_t wc = words[pos] >> 16;
            uint16_t op = static_cast<uint16_t>(words[pos] & 0xFFFFu);
            if (wc == 0 || pos + wc > words.size()) break;
            if (op == 5 && wc >= 3) {
                names[words[pos + 1]] = decode_spv_literal_string(&words[pos + 2], wc - 2, nullptr);
            }
            pos += wc;
        }
    }

    std::ostringstream oss;
    uint32_t ver = words[1];
    oss << "; SPIR-V\n";
    oss << "; Version: " << ((ver >> 16) & 0xFFu) << "." << ((ver >> 8) & 0xFFu) << "\n";
    oss << "; Generator: 0x" << std::hex << words[2] << std::dec << "\n";
    oss << "; Bound: " << words[3] << "\n";
    oss << "; Schema: " << words[4] << "\n";

    size_t pos = 5;
    while (pos < words.size()) {
        uint32_t wc = words[pos] >> 16;
        uint16_t op = static_cast<uint16_t>(words[pos] & 0xFFFFu);
        if (wc == 0 || pos + wc > words.size()) break;

        const char* name = spv_opcode_name(op);
        std::string op_name = name ? std::string(name) : ("Op#" + std::to_string(op));

        size_t arg_start = 1;
        if (op_has_type_and_result(op) && wc >= 3) {
            std::string lhs = id_str(words[pos + 2], names) + " = ";
            oss << "  " << std::left << std::setw(18) << lhs << op_name << " " << id_str(words[pos + 1], names);
            arg_start = 3;
        } else if (op_has_result_only(op) && wc >= 2) {
            std::string lhs = id_str(words[pos + 1], names) + " = ";
            oss << "  " << std::left << std::setw(18) << lhs << op_name;
            arg_start = 2;
        } else {
            oss << "  " << std::string(18, ' ') << op_name;
            arg_start = 1;
        }

        if (op == 17 && wc == 2) {
            const char* cn = spv_capability_name(words[pos + 1]);
            oss << " " << (cn ? std::string(cn) : std::to_string(words[pos + 1]));
        } else if (op == 14 && wc == 3) {
            oss << " " << (words[pos + 1] == 5348 ? "PhysicalStorageBuffer64" : (words[pos + 1] == 0 ? "Logical" : std::to_string(words[pos + 1])))
                << " " << (words[pos + 2] == 1 ? "GLSL450" : std::to_string(words[pos + 2]));
        } else if (op == 5 && wc >= 3) {
            oss << " " << id_str(words[pos + 1], names) << " \""
                << decode_spv_literal_string(&words[pos + 2], wc - 2, nullptr) << "\"";
        } else if (op == 6 && wc >= 4) {
            oss << " " << id_str(words[pos + 1], names) << " " << words[pos + 2] << " \""
                << decode_spv_literal_string(&words[pos + 3], wc - 3, nullptr) << "\"";
        } else if ((op == 10 || op == 11) && wc >= arg_start + 1) {
            oss << " \"" << decode_spv_literal_string(&words[pos + arg_start], wc - arg_start, nullptr) << "\"";
        } else if (op == 15 && wc >= 4) {
            size_t consumed = 0;
            std::string ep = decode_spv_literal_string(&words[pos + 3], wc - 3, &consumed);
            const char* emn = spv_exec_model_name(words[pos + 1]);
            oss << " " << (emn ? std::string(emn) : std::to_string(words[pos + 1]))
                << " " << id_str(words[pos + 2], names) << " \"" << ep << "\"";
            for (size_t i = 3 + consumed; i < wc; ++i) {
                oss << " " << id_str(words[pos + i], names);
            }
        } else if (op == 16 && wc >= 3) {
            const char* emn = spv_exec_mode_name(words[pos + 2]);
            oss << " " << id_str(words[pos + 1], names)
                << " " << (emn ? std::string(emn) : std::to_string(words[pos + 2]));
            for (size_t i = 3; i < wc; ++i) oss << " " << words[pos + i];
        } else if (op == 71 && wc >= 3) {
            uint32_t dec = words[pos + 2];
            const char* dn = spv_decoration_name(dec);
            oss << " " << id_str(words[pos + 1], names)
                << " " << (dn ? std::string(dn) : std::to_string(dec));
            for (size_t i = 3; i < wc; ++i) {
                if (dec == 11 && i == 3) {
                    const char* bn = spv_builtin_name(words[pos + i]);
                    oss << " " << (bn ? std::string(bn) : std::to_string(words[pos + i]));
                } else {
                    oss << " " << words[pos + i];
                }
            }
        } else if (op == 72 && wc >= 4) {
            uint32_t dec = words[pos + 3];
            const char* dn = spv_decoration_name(dec);
            oss << " " << id_str(words[pos + 1], names) << " " << words[pos + 2]
                << " " << (dn ? std::string(dn) : std::to_string(dec));
            for (size_t i = 4; i < wc; ++i) oss << " " << words[pos + i];
        } else if (op == 32 && wc == 4) {
            // OpTypePointer %res StorageClass %pointee
            const char* scn = spv_storage_class_name(words[pos + 2]);
            oss << " " << (scn ? std::string(scn) : std::to_string(words[pos + 2]))
                << " " << id_str(words[pos + 3], names);
        } else if (op == 59 && wc >= 4) {
            // OpVariable %type %res StorageClass [Init]
            const char* scn = spv_storage_class_name(words[pos + 3]);
            oss << " " << (scn ? std::string(scn) : std::to_string(words[pos + 3]));
            for (size_t i = 4; i < wc; ++i) oss << " " << id_str(words[pos + i], names);
        } else if (op == 54 && wc == 5) {
            // OpFunction %ret %res Control %func_type
            oss << " " << (words[pos + 3] == 0 ? "None" : std::to_string(words[pos + 3]))
                << " " << id_str(words[pos + 4], names);
        } else if (op == 12 && wc >= 5) {
            // OpExtInst %type %res %set instr operands...
            const char* gn = spv_glsl450_name(words[pos + 4]);
            oss << " " << id_str(words[pos + 3], names)
                << " " << (gn ? std::string(gn) : std::to_string(words[pos + 4]));
            for (size_t i = 5; i < wc; ++i) oss << " " << id_str(words[pos + i], names);
        } else if (op == 81 && wc >= 4) {
            // OpCompositeExtract %type %res %composite idx0...
            oss << " " << id_str(words[pos + 3], names);
            for (size_t i = 4; i < wc; ++i) oss << " " << words[pos + i];
        } else if (op == 82 && wc >= 5) {
            // OpCompositeInsert %type %res %obj %composite idx0...
            oss << " " << id_str(words[pos + 3], names) << " " << id_str(words[pos + 4], names);
            for (size_t i = 5; i < wc; ++i) oss << " " << words[pos + i];
        } else if (op == 79 && wc >= 5) {
            // OpVectorShuffle %type %res %v1 %v2 comps...
            oss << " " << id_str(words[pos + 3], names) << " " << id_str(words[pos + 4], names);
            for (size_t i = 5; i < wc; ++i) oss << " " << words[pos + i];
        } else if (op == 43 && wc >= 4) {
            for (size_t i = 3; i < wc; ++i) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), " 0x%x", words[pos + i]);
                oss << buf;
            }
        } else if (op == 21 || op == 22 || op == 23) {
            for (size_t i = arg_start; i < wc; ++i) {
                if (op == 23 && i == arg_start) oss << " " << id_str(words[pos + i], names);
                else oss << " " << words[pos + i];
            }
        } else if ((op == 61 && wc >= 5) || (op == 62 && wc >= 4)) {
            // OpLoad / OpStore with optional MemoryAccess + Alignment literals
            size_t mem_idx = (op == 61) ? 4 : 3;
            for (size_t i = arg_start; i < wc; ++i) {
                if (i >= mem_idx) oss << " " << words[pos + i];
                else oss << " " << id_str(words[pos + i], names);
            }
        } else if (op == 246 || op == 247) {
            // OpLoopMerge / OpSelectionMerge
            for (size_t i = arg_start; i < wc; ++i) {
                if (i == wc - 1) oss << " " << words[pos + i];
                else oss << " " << id_str(words[pos + i], names);
            }
        } else {
            for (size_t i = arg_start; i < wc; ++i) {
                oss << " " << id_str(words[pos + i], names);
            }
        }
        oss << "\n";
        pos += wc;
    }
    return oss.str();
}

bool lift_ebins_to_spirv(
    const SpirvStageInfo& stage_info,
    uint8_t arch,
    const Mbs2Module& mod,
    const std::vector<ExtractedEbin>& ebins,
    std::vector<LiftedSpirvShader>* out_shaders,
    std::string* err) {
    if (!out_shaders) {
        if (err) *err = "Null LiftedSpirvShader output vector";
        return false;
    }
    out_shaders->clear();

    auto run_one = [&](const std::string& label, EbinRole role,
                       const uint8_t* code, size_t code_sz,
                       const valhall::ValhallTranslateOptions& topts) {
        LiftedSpirvShader sh;
        sh.label = label;
        sh.role = role;
        auto tres = valhall::translate_valhall_to_spirv(code, code_sz, topts);
        sh.ok = tres.ok;
        sh.error_message = tres.error_message;
        if (tres.ok) {
            sh.spirv_words = std::move(tres.spirv_words);
            validate_with_reflect(&sh);
            sh.disassembly = format_spirv_disassembly(sh.spirv_words);
        }
        out_shaders->push_back(std::move(sh));
    };

    if (stage_info.stage == ShaderStage::Compute) {
        for (const auto& eb : ebins) {
            valhall::ValhallTranslateOptions topts;
            topts.arch = arch;
            if (eb.role == EbinRole::Pilot) {
                topts.stage = valhall::ShaderStage::Pilot;
                topts.local_size_x = 1;
                topts.local_size_y = 1;
                topts.local_size_z = 1;
                run_one("PILOT SHADER (EBIN[" + std::to_string(eb.index) + "])",
                        eb.role, eb.bytecode.data(), eb.bytecode.size(), topts);
            } else {
                topts.stage = valhall::ShaderStage::Compute;
                if (mod.has_kern && mod.kern.has_workgroup_size) {
                    topts.local_size_x = mod.kern.wg_x;
                    topts.local_size_y = mod.kern.wg_y;
                    topts.local_size_z = mod.kern.wg_z;
                }
                run_one("MAIN COMPUTE SHADER (EBIN[" + std::to_string(eb.index) + "])",
                        eb.role, eb.bytecode.data(), eb.bytecode.size(), topts);
            }
        }
    } else if (stage_info.stage == ShaderStage::Vertex) {
        const ExtractedEbin* pos_eb = nullptr;
        const ExtractedEbin* var_eb = nullptr;
        for (const auto& eb : ebins) {
            if (eb.role == EbinRole::VertexVarying && !var_eb) var_eb = &eb;
            else if (!pos_eb) pos_eb = &eb;
        }
        if (pos_eb) {
            valhall::ValhallTranslateOptions topts;
            topts.stage = valhall::ShaderStage::Vertex;
            topts.arch = arch;
            if (var_eb) {
                topts.secondary_vertex_code = var_eb->bytecode.data();
                topts.secondary_vertex_code_size = var_eb->bytecode.size();
            }
            for (const auto& inp : mod.cmmn.sinp) {
                valhall::VertexAttributeBindingSpec spec;
                spec.location = (inp.location != 0xFFFFu) ? inp.location : inp.address;
                if (inp.type && inp.type->has_tpge) {
                    spec.components = std::max<uint32_t>(1u, inp.type->tpge.vector_size);
                    spec.is_integer = (inp.type->tpge.scalar_type == 2 || inp.type->tpge.scalar_type == 3);
                    spec.is_signed = (inp.type->tpge.scalar_type == 2);
                }
                topts.vertex_attributes.push_back(spec);
            }
            for (const auto& outp : mod.cmmn.sout) {
                valhall::VaryingSlotSpec vspec;
                vspec.location = (outp.location != 0xFFFFu) ? outp.location : outp.address;
                if (outp.type && outp.type->has_tpge) {
                    vspec.components = std::max<uint32_t>(1u, outp.type->tpge.vector_size);
                    vspec.is_integer = (outp.type->tpge.scalar_type == 2 || outp.type->tpge.scalar_type == 3);
                }
                topts.varyings.push_back(vspec);
            }
            run_one("VERTEX SHADER (EBIN[" + std::to_string(pos_eb->index) + "])",
                    pos_eb->role, pos_eb->bytecode.data(), pos_eb->bytecode.size(), topts);
        }
    } else if (stage_info.stage == ShaderStage::Fragment) {
        for (const auto& eb : ebins) {
            valhall::ValhallTranslateOptions topts;
            topts.stage = valhall::ShaderStage::Fragment;
            topts.arch = arch;
            if (mod.has_cfra && !mod.cfra.output_vars.empty()) {
                topts.num_color_attachments = static_cast<uint32_t>(mod.cfra.output_vars.size());
            } else if (!mod.cmmn.sout.empty()) {
                topts.num_color_attachments = static_cast<uint32_t>(mod.cmmn.sout.size());
            }
            for (const auto& inp : mod.cmmn.sinp) {
                valhall::VaryingSlotSpec vspec;
                vspec.location = (inp.location != 0xFFFFu) ? inp.location : inp.address;
                if (inp.type && inp.type->has_tpge) {
                    vspec.components = std::max<uint32_t>(1u, inp.type->tpge.vector_size);
                    vspec.is_integer = (inp.type->tpge.scalar_type == 2 || inp.type->tpge.scalar_type == 3);
                }
                topts.varyings.push_back(vspec);
            }
            run_one("FRAGMENT SHADER (EBIN[" + std::to_string(eb.index) + "])",
                    eb.role, eb.bytecode.data(), eb.bytecode.size(), topts);
        }
    } else {
        if (err) {
            *err = std::string("valhall2spirv lifting is currently supported for Compute, Vertex, and Fragment stages (got ") +
                   shader_stage_name(stage_info.stage) + ")";
        }
        return false;
    }

    bool any_ok = false;
    for (const auto& s : *out_shaders) {
        if (s.ok) any_ok = true;
    }
    if (!any_ok) {
        if (err) {
            *err = out_shaders->empty() ? "No EBINs available to lift to SPIR-V"
                                        : out_shaders->front().error_message;
        }
        return false;
    }
    return true;
}

std::string format_lifted_spirv_dump(const std::vector<LiftedSpirvShader>& shaders) {
    std::ostringstream oss;
    oss << "\n; ============================================================================\n";
    oss << "; Lifted SPIR-V Dump (via valhall2spirv)\n";
    oss << "; ============================================================================\n";
    for (const auto& sh : shaders) {
        oss << "\n; --- " << sh.label << " ---\n";
        if (!sh.ok) {
            oss << "; [LIFT FAILED] " << sh.error_message << "\n";
            continue;
        }
        oss << "; [valhall2spirv] words=" << sh.spirv_words.size()
            << " (" << (sh.spirv_words.size() * 4) << " bytes)"
            << " spirv-reflect=" << (sh.reflect_valid ? "PASS" : "FAIL");
        if (sh.reflect_valid) {
            oss << " entry=\"" << sh.reflect_entry_point << "\""
                << " stage_bits=0x" << std::hex << sh.reflect_shader_stage << std::dec;
        }
        oss << "\n";
        oss << sh.disassembly;
    }
    return oss.str();
}

}  // namespace valhall::spirv2valhall
