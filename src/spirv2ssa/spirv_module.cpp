#include "spirv_module.h"

#include <algorithm>

namespace valhall::spirv2ssa {

namespace {

constexpr uint32_t kSpirvMagic = 0x07230203u;

// Core SPIR-V opcodes used during module parsing.
constexpr uint16_t kOpUndef = 1;
constexpr uint16_t kOpName = 5;
constexpr uint16_t kOpExtInstImport = 11;
constexpr uint16_t kOpEntryPoint = 15;
constexpr uint16_t kOpExecutionMode = 16;
constexpr uint16_t kOpTypeVoid = 19;
constexpr uint16_t kOpTypeBool = 20;
constexpr uint16_t kOpTypeInt = 21;
constexpr uint16_t kOpTypeFloat = 22;
constexpr uint16_t kOpTypeVector = 23;
constexpr uint16_t kOpTypeMatrix = 24;
constexpr uint16_t kOpTypeImage = 25;
constexpr uint16_t kOpTypeSampler = 26;
constexpr uint16_t kOpTypeSampledImage = 27;
constexpr uint16_t kOpTypeArray = 28;
constexpr uint16_t kOpTypeRuntimeArray = 29;
constexpr uint16_t kOpTypeStruct = 30;
constexpr uint16_t kOpTypePointer = 32;
constexpr uint16_t kOpTypeFunction = 33;
constexpr uint16_t kOpConstantTrue = 41;
constexpr uint16_t kOpConstantFalse = 42;
constexpr uint16_t kOpConstant = 43;
constexpr uint16_t kOpConstantComposite = 44;
constexpr uint16_t kOpConstantNull = 46;
constexpr uint16_t kOpSpecConstantTrue = 48;
constexpr uint16_t kOpSpecConstantFalse = 49;
constexpr uint16_t kOpSpecConstant = 50;
constexpr uint16_t kOpSpecConstantComposite = 51;
constexpr uint16_t kOpFunction = 54;
constexpr uint16_t kOpFunctionParameter = 55;
constexpr uint16_t kOpFunctionEnd = 56;
constexpr uint16_t kOpVariable = 59;
constexpr uint16_t kOpDecorate = 71;
constexpr uint16_t kOpMemberDecorate = 72;
constexpr uint16_t kOpLabel = 248;

// Decoration codes
constexpr uint32_t kDecSpecId = 1;
constexpr uint32_t kDecBlock = 2;
constexpr uint32_t kDecBufferBlock = 3;
constexpr uint32_t kDecRowMajor = 4;
constexpr uint32_t kDecArrayStride = 6;
constexpr uint32_t kDecMatrixStride = 7;
constexpr uint32_t kDecBuiltIn = 11;
constexpr uint32_t kDecFlat = 14;
constexpr uint32_t kDecNonWritable = 24;
constexpr uint32_t kDecLocation = 30;
constexpr uint32_t kDecComponent = 31;
constexpr uint32_t kDecBinding = 33;
constexpr uint32_t kDecDescriptorSet = 34;
constexpr uint32_t kDecOffset = 35;

constexpr uint32_t kExecModeLocalSize = 17;

std::string decode_literal_string(const uint32_t* words, size_t max_words, size_t* out_consumed = nullptr) {
    std::string s;
    for (size_t i = 0; i < max_words; ++i) {
        uint32_t w = words[i];
        for (int b = 0; b < 4; ++b) {
            char c = static_cast<char>((w >> (b * 8)) & 0xFFu);
            if (c == '\0') {
                if (out_consumed) *out_consumed = i + 1;
                return s;
            }
            s.push_back(c);
        }
    }
    if (out_consumed) *out_consumed = max_words;
    return s;
}

void apply_decoration(SpirvDecorationInfo& info, uint32_t dec, const uint32_t* args, size_t nargs) {
    switch (dec) {
        case kDecSpecId:
            if (nargs >= 1) info.spec_id = static_cast<int32_t>(args[0]);
            break;
        case kDecBlock:
            info.block = true;
            break;
        case kDecBufferBlock:
            info.buffer_block = true;
            break;
        case kDecArrayStride:
            if (nargs >= 1) info.array_stride = args[0];
            break;
        case kDecBuiltIn:
            if (nargs >= 1) info.builtin = static_cast<int32_t>(args[0]);
            break;
        case kDecFlat:
            info.flat = true;
            break;
        case kDecNonWritable:
            info.non_writable = true;
            break;
        case kDecLocation:
            if (nargs >= 1) info.location = static_cast<int32_t>(args[0]);
            break;
        case kDecComponent:
            if (nargs >= 1) info.component = args[0];
            break;
        case kDecBinding:
            if (nargs >= 1) info.binding = static_cast<int32_t>(args[0]);
            break;
        case kDecDescriptorSet:
            if (nargs >= 1) info.descriptor_set = static_cast<int32_t>(args[0]);
            break;
        case kDecOffset:
            if (nargs >= 1) info.offset = args[0];
            break;
        default:
            break;
    }
}

} // namespace

const SpirvType* SpirvModule::get_type(uint32_t id) const {
    auto it = types.find(id);
    return (it != types.end()) ? &it->second : nullptr;
}

const SpirvConstant* SpirvModule::get_const(uint32_t id) const {
    auto it = constants.find(id);
    return (it != constants.end()) ? &it->second : nullptr;
}

const SpirvVariable* SpirvModule::get_global_var(uint32_t id) const {
    auto it = global_vars.find(id);
    return (it != global_vars.end()) ? &it->second : nullptr;
}

const SpirvFunction* SpirvModule::get_function(uint32_t id) const {
    auto it = functions.find(id);
    return (it != functions.end()) ? &it->second : nullptr;
}

uint32_t SpirvModule::struct_member_offset(const SpirvType& st, uint32_t member_idx) const {
    if (member_idx < st.member_offsets.size() && st.member_offsets[member_idx] != 0xFFFFFFFFu) {
        return st.member_offsets[member_idx];
    }
    uint32_t off = 0;
    for (uint32_t i = 0; i < member_idx && i < st.member_types.size(); ++i) {
        off += type_byte_size(st.member_types[i]);
    }
    return off;
}

uint32_t SpirvModule::type_byte_size(uint32_t type_id) const {
    const SpirvType* t = get_type(type_id);
    if (!t) return 4u;
    switch (t->kind) {
        case SpirvTypeKind::Bool:
            return 4u;
        case SpirvTypeKind::Int:
        case SpirvTypeKind::Float:
            return std::max<uint32_t>(1u, t->width / 8u);
        case SpirvTypeKind::Vector:
            return type_byte_size(t->elem_type) * std::max<uint32_t>(1u, t->count);
        case SpirvTypeKind::Matrix:
            return type_byte_size(t->elem_type) * std::max<uint32_t>(1u, t->count);
        case SpirvTypeKind::Array: {
            uint32_t stride = (t->array_stride != 0) ? t->array_stride : type_byte_size(t->elem_type);
            return stride * t->count;
        }
        case SpirvTypeKind::Struct: {
            if (t->member_types.empty()) return 0u;
            uint32_t last = static_cast<uint32_t>(t->member_types.size() - 1);
            return struct_member_offset(*t, last) + type_byte_size(t->member_types[last]);
        }
        case SpirvTypeKind::Pointer:
            return 8u;
        default:
            return 4u;
    }
}

bool parse_spirv_module(
    const std::vector<uint32_t>& words,
    SpirvModule* out,
    std::string* err) {
    if (!out) {
        if (err) *err = "Null SpirvModule output pointer";
        return false;
    }
    *out = SpirvModule{};

    if (words.size() < 5 || words[0] != kSpirvMagic) {
        if (err) *err = "Invalid SPIR-V header or magic number";
        return false;
    }

    out->version = words[1];
    out->bound = words[3];

    SpirvFunction* cur_func = nullptr;
    SpirvBasicBlock* cur_blk = nullptr;

    size_t idx = 5;
    while (idx < words.size()) {
        uint32_t header = words[idx];
        uint16_t wc = static_cast<uint16_t>(header >> 16);
        uint16_t op = static_cast<uint16_t>(header & 0xFFFFu);
        if (wc == 0 || idx + wc > words.size()) {
            if (err) *err = "Truncated or malformed SPIR-V instruction at word " + std::to_string(idx);
            return false;
        }
        const uint32_t* w = &words[idx];

        if (cur_func != nullptr) {
            if (op == kOpFunctionEnd) {
                cur_func = nullptr;
                cur_blk = nullptr;
            } else if (op == kOpFunctionParameter && wc >= 3) {
                cur_func->params.push_back({w[2], w[1]});
            } else if (op == kOpLabel && wc >= 2) {
                cur_func->blocks.emplace_back();
                cur_blk = &cur_func->blocks.back();
                cur_blk->label_id = w[1];
            } else if (cur_blk != nullptr) {
                SpirvInst inst;
                inst.opcode = op;
                inst.words.assign(w, w + wc);
                cur_blk->insts.push_back(std::move(inst));
            }
            idx += wc;
            continue;
        }

        switch (op) {
            case kOpName:
                if (wc >= 3) {
                    out->names[w[1]] = decode_literal_string(w + 2, wc - 2);
                }
                break;

            case kOpExtInstImport:
                if (wc >= 3) {
                    out->ext_inst_imports[w[1]] = decode_literal_string(w + 2, wc - 2);
                }
                break;

            case kOpEntryPoint:
                if (wc >= 4) {
                    out->entry_point.execution_model = w[1];
                    out->entry_point.func_id = w[2];
                    size_t name_words = 0;
                    out->entry_point.name = decode_literal_string(w + 3, wc - 3, &name_words);
                    for (size_t k = 3 + name_words; k < wc; ++k) {
                        out->entry_point.interface_ids.push_back(w[k]);
                    }
                }
                break;

            case kOpExecutionMode:
                if (wc >= 6 && w[1] == out->entry_point.func_id && w[2] == kExecModeLocalSize) {
                    out->entry_point.local_size_x = std::max(1u, w[3]);
                    out->entry_point.local_size_y = std::max(1u, w[4]);
                    out->entry_point.local_size_z = std::max(1u, w[5]);
                }
                break;

            case kOpDecorate:
                if (wc >= 3) {
                    apply_decoration(out->decorations[w[1]], w[2], w + 3, wc - 3);
                }
                break;

            case kOpMemberDecorate:
                if (wc >= 4) {
                    auto& md = out->member_decorations[{w[1], w[2]}];
                    if (w[3] == kDecRowMajor) {
                        md.flat = true; // reuse flag temporarily; transferred cleanly below
                    } else if (w[3] == kDecMatrixStride && wc >= 5) {
                        md.array_stride = w[4];
                    } else {
                        apply_decoration(md, w[3], w + 4, wc - 4);
                    }
                }
                break;

            case kOpTypeVoid:
                if (wc >= 2) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Void;
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeBool:
                if (wc >= 2) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Bool;
                    t.width = 1;
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeInt:
                if (wc >= 4) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Int;
                    t.width = w[2];
                    t.is_signed = (w[3] != 0);
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeFloat:
                if (wc >= 3) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Float;
                    t.width = w[2];
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeVector:
                if (wc >= 4) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Vector;
                    t.elem_type = w[2];
                    t.count = w[3];
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeMatrix:
                if (wc >= 4) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Matrix;
                    t.elem_type = w[2];
                    t.count = w[3];
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeImage:
                if (wc >= 7) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Image;
                    t.image_sampled_type = w[2];
                    t.image_dim = w[3];
                    t.image_depth = w[4];
                    t.image_arrayed = w[5];
                    t.image_ms = w[6];
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeSampler:
                if (wc >= 2) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Sampler;
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeSampledImage:
                if (wc >= 3) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::SampledImage;
                    t.elem_type = w[2];
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeArray:
                if (wc >= 4) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Array;
                    t.elem_type = w[2];
                    t.length_id = w[3];
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeRuntimeArray:
                if (wc >= 3) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::RuntimeArray;
                    t.elem_type = w[2];
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeStruct:
                if (wc >= 2) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Struct;
                    for (uint16_t k = 2; k < wc; ++k) {
                        t.member_types.push_back(w[k]);
                    }
                    out->types[t.id] = std::move(t);
                }
                break;

            case kOpTypePointer:
                if (wc >= 4) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Pointer;
                    t.storage_class = w[2];
                    t.pointee_type = w[3];
                    out->types[t.id] = t;
                }
                break;

            case kOpTypeFunction:
                if (wc >= 3) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Function;
                    t.elem_type = w[2];
                    for (uint16_t k = 3; k < wc; ++k) {
                        t.member_types.push_back(w[k]);
                    }
                    out->types[t.id] = std::move(t);
                }
                break;

            case kOpConstantTrue:
            case kOpSpecConstantTrue:
                if (wc >= 3) {
                    SpirvConstant c{};
                    c.id = w[2];
                    c.type_id = w[1];
                    c.kind = SpirvConstKind::Bool;
                    c.scalar_bits = 1;
                    c.is_spec_const = (op == kOpSpecConstantTrue);
                    out->constants[c.id] = c;
                }
                break;

            case kOpConstantFalse:
            case kOpSpecConstantFalse:
                if (wc >= 3) {
                    SpirvConstant c{};
                    c.id = w[2];
                    c.type_id = w[1];
                    c.kind = SpirvConstKind::Bool;
                    c.scalar_bits = 0;
                    c.is_spec_const = (op == kOpSpecConstantFalse);
                    out->constants[c.id] = c;
                }
                break;

            case kOpConstant:
            case kOpSpecConstant:
                if (wc >= 4) {
                    SpirvConstant c{};
                    c.id = w[2];
                    c.type_id = w[1];
                    c.is_spec_const = (op == kOpSpecConstant);
                    if (wc >= 5) {
                        c.kind = SpirvConstKind::Scalar64;
                        c.scalar_bits = uint64_t(w[3]) | (uint64_t(w[4]) << 32);
                    } else {
                        c.kind = SpirvConstKind::Scalar32;
                        c.scalar_bits = w[3];
                    }
                    out->constants[c.id] = c;
                }
                break;

            case kOpConstantComposite:
            case kOpSpecConstantComposite:
                if (wc >= 3) {
                    SpirvConstant c{};
                    c.id = w[2];
                    c.type_id = w[1];
                    c.kind = SpirvConstKind::Composite;
                    c.is_spec_const = (op == kOpSpecConstantComposite);
                    for (uint16_t k = 3; k < wc; ++k) {
                        c.constituents.push_back(w[k]);
                    }
                    out->constants[c.id] = std::move(c);
                }
                break;

            case kOpConstantNull:
                if (wc >= 3) {
                    SpirvConstant c{};
                    c.id = w[2];
                    c.type_id = w[1];
                    c.kind = SpirvConstKind::Null;
                    out->constants[c.id] = c;
                }
                break;

            case kOpUndef:
                if (wc >= 3) {
                    SpirvConstant c{};
                    c.id = w[2];
                    c.type_id = w[1];
                    c.kind = SpirvConstKind::Undef;
                    out->constants[c.id] = c;
                }
                break;

            case kOpVariable:
                if (wc >= 4) {
                    SpirvVariable v{};
                    v.id = w[2];
                    v.ptr_type_id = w[1];
                    v.storage_class = w[3];
                    if (wc >= 5) v.initializer_id = w[4];
                    out->global_vars[v.id] = v;
                }
                break;

            case kOpFunction:
                if (wc >= 5) {
                    SpirvFunction f{};
                    f.id = w[2];
                    f.return_type_id = w[1];
                    f.func_type_id = w[4];
                    auto [it, _] = out->functions.emplace(f.id, std::move(f));
                    cur_func = &it->second;
                    cur_blk = nullptr;
                }
                break;

            default:
                break;
        }
        idx += wc;
    }

    // Resolve decorations onto types, constants, and global variables.
    for (auto& [tid, t] : out->types) {
        if (auto it = out->decorations.find(tid); it != out->decorations.end()) {
            t.array_stride = it->second.array_stride;
            t.is_block = it->second.block;
            t.is_buffer_block = it->second.buffer_block;
        }
        if (t.kind == SpirvTypeKind::Array && t.length_id != 0) {
            if (const auto* c = out->get_const(t.length_id)) {
                t.count = static_cast<uint32_t>(c->scalar_bits);
            }
        }
        if (t.kind == SpirvTypeKind::Struct) {
            size_t n = t.member_types.size();
            t.member_offsets.assign(n, 0xFFFFFFFFu);
            t.member_builtins.assign(n, -1);
            t.member_matrix_stride.assign(n, 0u);
            t.member_row_major.assign(n, false);
            for (size_t m = 0; m < n; ++m) {
                auto mit = out->member_decorations.find({tid, static_cast<uint32_t>(m)});
                if (mit != out->member_decorations.end()) {
                    t.member_offsets[m] = mit->second.offset;
                    t.member_builtins[m] = mit->second.builtin;
                    t.member_matrix_stride[m] = mit->second.array_stride;
                    t.member_row_major[m] = mit->second.flat;
                }
            }
        }
    }

    for (auto& [vid, v] : out->global_vars) {
        if (const auto* pt = out->get_type(v.ptr_type_id); pt && pt->kind == SpirvTypeKind::Pointer) {
            v.pointee_type_id = pt->pointee_type;
        }
        if (auto it = out->decorations.find(vid); it != out->decorations.end()) {
            v.dec = it->second;
        }
        if (auto it = out->names.find(vid); it != out->names.end()) {
            v.name = it->second;
        }
    }

    for (auto& [cid, c] : out->constants) {
        if (c.is_spec_const) {
            if (auto it = out->decorations.find(cid); it != out->decorations.end() && it->second.spec_id >= 0) {
                c.spec_id = static_cast<uint32_t>(it->second.spec_id);
            }
        }
    }

    if (out->entry_point.func_id == 0) {
        if (err) *err = "SPIR-V module has no OpEntryPoint";
        return false;
    }
    return true;
}

} // namespace valhall::spirv2ssa
