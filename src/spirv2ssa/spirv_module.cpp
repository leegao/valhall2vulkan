#include "spirv_module.h"
#include "valhall_spirv_parser_generated.h"

#include <algorithm>

namespace valhall::spirv2ssa {

namespace {

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
    switch (static_cast<SpvDecoration>(dec)) {
        case SpvDecorationSpecId:
            if (nargs >= 1) info.spec_id = static_cast<int32_t>(args[0]);
            break;
        case SpvDecorationBlock:
            info.block = true;
            break;
        case SpvDecorationBufferBlock:
            info.buffer_block = true;
            break;
        case SpvDecorationArrayStride:
            if (nargs >= 1) info.array_stride = args[0];
            break;
        case SpvDecorationBuiltIn:
            if (nargs >= 1) info.builtin = static_cast<int32_t>(args[0]);
            break;
        case SpvDecorationFlat:
            info.flat = true;
            break;
        case SpvDecorationNonWritable:
            info.non_writable = true;
            break;
        case SpvDecorationLocation:
            if (nargs >= 1) info.location = static_cast<int32_t>(args[0]);
            break;
        case SpvDecorationComponent:
            if (nargs >= 1) info.component = args[0];
            break;
        case SpvDecorationBinding:
            if (nargs >= 1) info.binding = static_cast<int32_t>(args[0]);
            break;
        case SpvDecorationDescriptorSet:
            if (nargs >= 1) info.descriptor_set = static_cast<int32_t>(args[0]);
            break;
        case SpvDecorationOffset:
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

    if (words.size() < 5 || words[0] != SpvMagicNumber) {
        if (err) *err = "Invalid SPIR-V header or magic number";
        return false;
    }

    out->version = words[1];
    out->bound = words[3];

    SpirvFunction* cur_func = nullptr;
    SpirvBasicBlock* cur_blk = nullptr;

    size_t idx = 5;
    while (idx < words.size()) {
        spirv_parser::DecodedSpvInst dec{};
        if (!spirv_parser::decode_spv_instruction(&words[idx], words.size() - idx, &dec)) {
            if (err) *err = "Truncated or malformed SPIR-V instruction at word " + std::to_string(idx);
            return false;
        }
        const uint32_t* w = dec.words;
        const uint16_t wc = dec.word_count;

        if (cur_func != nullptr) {
            if (dec.opcode == SpvOpFunctionEnd) {
                cur_func = nullptr;
                cur_blk = nullptr;
            } else if (dec.opcode == SpvOpFunctionParameter && wc >= 3) {
                cur_func->params.push_back({dec.result_id, dec.result_type});
            } else if (dec.opcode == SpvOpLabel && wc >= 2) {
                cur_func->blocks.emplace_back();
                cur_blk = &cur_func->blocks.back();
                cur_blk->label_id = w[1];
            } else if (cur_blk != nullptr) {
                SpirvInst inst;
                inst.opcode = static_cast<uint16_t>(dec.opcode);
                inst.words.assign(w, w + wc);
                cur_blk->insts.push_back(std::move(inst));
            }
            idx += wc;
            continue;
        }

        switch (dec.opcode) {
            case SpvOpName:
                if (wc >= 3) {
                    out->names[w[1]] = decode_literal_string(w + 2, wc - 2);
                }
                break;

            case SpvOpExtInstImport:
                if (wc >= 3) {
                    out->ext_inst_imports[w[1]] = decode_literal_string(w + 2, wc - 2);
                }
                break;

            case SpvOpEntryPoint:
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

            case SpvOpExecutionMode:
                if (wc >= 6 && w[1] == out->entry_point.func_id &&
                    static_cast<SpvExecutionMode>(w[2]) == SpvExecutionModeLocalSize) {
                    out->entry_point.local_size_x = std::max(1u, w[3]);
                    out->entry_point.local_size_y = std::max(1u, w[4]);
                    out->entry_point.local_size_z = std::max(1u, w[5]);
                }
                break;

            case SpvOpDecorate:
                if (wc >= 3) {
                    apply_decoration(out->decorations[w[1]], w[2], w + 3, wc - 3);
                }
                break;

            case SpvOpMemberDecorate:
                if (wc >= 4) {
                    auto& md = out->member_decorations[{w[1], w[2]}];
                    if (static_cast<SpvDecoration>(w[3]) == SpvDecorationRowMajor) {
                        md.flat = true;
                    } else if (static_cast<SpvDecoration>(w[3]) == SpvDecorationMatrixStride && wc >= 5) {
                        md.array_stride = w[4];
                    } else {
                        apply_decoration(md, w[3], w + 4, wc - 4);
                    }
                }
                break;

            case SpvOpTypeVoid:
                if (wc >= 2) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Void;
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeBool:
                if (wc >= 2) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Bool;
                    t.width = 1;
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeInt:
                if (wc >= 4) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Int;
                    t.width = w[2];
                    t.is_signed = (w[3] != 0);
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeFloat:
                if (wc >= 3) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Float;
                    t.width = w[2];
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeVector:
                if (wc >= 4) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Vector;
                    t.elem_type = w[2];
                    t.count = w[3];
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeMatrix:
                if (wc >= 4) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Matrix;
                    t.elem_type = w[2];
                    t.count = w[3];
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeImage:
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

            case SpvOpTypeSampler:
                if (wc >= 2) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Sampler;
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeSampledImage:
                if (wc >= 3) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::SampledImage;
                    t.elem_type = w[2];
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeArray:
                if (wc >= 4) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Array;
                    t.elem_type = w[2];
                    t.length_id = w[3];
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeRuntimeArray:
                if (wc >= 3) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::RuntimeArray;
                    t.elem_type = w[2];
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeStruct:
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

            case SpvOpTypePointer:
                if (wc >= 4) {
                    SpirvType t{};
                    t.id = w[1];
                    t.kind = SpirvTypeKind::Pointer;
                    t.storage_class = w[2];
                    t.pointee_type = w[3];
                    out->types[t.id] = t;
                }
                break;

            case SpvOpTypeFunction:
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

            case SpvOpConstantTrue:
            case SpvOpSpecConstantTrue:
                if (wc >= 3) {
                    SpirvConstant c{};
                    c.id = dec.result_id;
                    c.type_id = dec.result_type;
                    c.kind = SpirvConstKind::Bool;
                    c.scalar_bits = 1;
                    c.is_spec_const = (dec.opcode == SpvOpSpecConstantTrue);
                    out->constants[c.id] = c;
                }
                break;

            case SpvOpConstantFalse:
            case SpvOpSpecConstantFalse:
                if (wc >= 3) {
                    SpirvConstant c{};
                    c.id = dec.result_id;
                    c.type_id = dec.result_type;
                    c.kind = SpirvConstKind::Bool;
                    c.scalar_bits = 0;
                    c.is_spec_const = (dec.opcode == SpvOpSpecConstantFalse);
                    out->constants[c.id] = c;
                }
                break;

            case SpvOpConstant:
            case SpvOpSpecConstant:
                if (wc >= 4) {
                    SpirvConstant c{};
                    c.id = dec.result_id;
                    c.type_id = dec.result_type;
                    c.is_spec_const = (dec.opcode == SpvOpSpecConstant);
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

            case SpvOpConstantComposite:
            case SpvOpSpecConstantComposite:
                if (wc >= 3) {
                    SpirvConstant c{};
                    c.id = dec.result_id;
                    c.type_id = dec.result_type;
                    c.kind = SpirvConstKind::Composite;
                    c.is_spec_const = (dec.opcode == SpvOpSpecConstantComposite);
                    for (uint16_t k = 3; k < wc; ++k) {
                        c.constituents.push_back(w[k]);
                    }
                    out->constants[c.id] = std::move(c);
                }
                break;

            case SpvOpSpecConstantOp:
                if (wc >= 4) {
                    SpirvConstant c{};
                    c.id = dec.result_id;
                    c.type_id = dec.result_type;
                    c.is_spec_const = true;
                    const auto sub_op = static_cast<SpvOp>(w[3]);
                    auto get_u64 = [&](size_t wi) -> uint64_t {
                        if (wi >= wc) return 0u;
                        auto it = out->constants.find(w[wi]);
                        return (it != out->constants.end()) ? it->second.scalar_bits : 0u;
                    };
                    uint64_t a = get_u64(4), b = get_u64(5);
                    uint64_t rv = 0u;
                    switch (sub_op) {
                        case SpvOpIAdd: rv = a + b; break;
                        case SpvOpISub: rv = a - b; break;
                        case SpvOpIMul: rv = a * b; break;
                        case SpvOpUDiv: rv = (b != 0) ? (a / b) : 0u; break;
                        case SpvOpSDiv: {
                            int64_t sa = static_cast<int32_t>(a), sb = static_cast<int32_t>(b);
                            rv = (sb != 0) ? static_cast<uint64_t>(sa / sb) : 0u;
                            break;
                        }
                        case SpvOpUMod: rv = (b != 0) ? (a % b) : 0u; break;
                        case SpvOpSRem: {
                            int64_t sa = static_cast<int32_t>(a), sb = static_cast<int32_t>(b);
                            rv = (sb != 0) ? static_cast<uint64_t>(sa % sb) : 0u;
                            break;
                        }
                        case SpvOpSMod: {
                            int64_t sa = static_cast<int32_t>(a), sb = static_cast<int32_t>(b);
                            if (sb != 0) {
                                int64_t r = sa % sb;
                                if (r != 0 && ((r < 0) != (sb < 0))) r += sb;
                                rv = static_cast<uint64_t>(r);
                            }
                            break;
                        }
                        case SpvOpShiftLeftLogical: rv = a << (b & 63u); break;
                        case SpvOpShiftRightLogical: rv = a >> (b & 63u); break;
                        case SpvOpShiftRightArithmetic: rv = static_cast<uint64_t>(static_cast<int32_t>(a) >> (b & 31u)); break;
                        case SpvOpBitwiseAnd: rv = a & b; break;
                        case SpvOpBitwiseOr: rv = a | b; break;
                        case SpvOpBitwiseXor: rv = a ^ b; break;
                        case SpvOpNot: rv = ~a; break;
                        case SpvOpSNegate: rv = static_cast<uint64_t>(-static_cast<int32_t>(a)); break;
                        case SpvOpSConvert: rv = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a))); break;
                        case SpvOpUConvert: rv = a; break;
                        case SpvOpIEqual: rv = (a == b) ? 1u : 0u; break;
                        case SpvOpINotEqual: rv = (a != b) ? 1u : 0u; break;
                        case SpvOpULessThan: rv = (a < b) ? 1u : 0u; break;
                        case SpvOpSLessThan: rv = (static_cast<int32_t>(a) < static_cast<int32_t>(b)) ? 1u : 0u; break;
                        case SpvOpUGreaterThan: rv = (a > b) ? 1u : 0u; break;
                        case SpvOpSGreaterThan: rv = (static_cast<int32_t>(a) > static_cast<int32_t>(b)) ? 1u : 0u; break;
                        case SpvOpLogicalAnd: rv = ((a != 0) && (b != 0)) ? 1u : 0u; break;
                        case SpvOpLogicalOr: rv = ((a != 0) || (b != 0)) ? 1u : 0u; break;
                        case SpvOpLogicalNot: rv = (a == 0) ? 1u : 0u; break;
                        case SpvOpSelect: rv = (a != 0) ? b : get_u64(6); break;
                        default: rv = 0u; break;
                    }
                    const SpirvType* rt = out->get_type(c.type_id);
                    if (rt && rt->kind == SpirvTypeKind::Bool) {
                        c.kind = SpirvConstKind::Bool;
                        c.scalar_bits = (rv != 0) ? 1u : 0u;
                    } else if (rt && rt->width == 64) {
                        c.kind = SpirvConstKind::Scalar64;
                        c.scalar_bits = rv;
                    } else {
                        c.kind = SpirvConstKind::Scalar32;
                        c.scalar_bits = rv & 0xFFFFFFFFu;
                    }
                    out->constants[c.id] = c;
                }
                break;

            case SpvOpConstantNull:
                if (wc >= 3) {
                    SpirvConstant c{};
                    c.id = dec.result_id;
                    c.type_id = dec.result_type;
                    c.kind = SpirvConstKind::Null;
                    out->constants[c.id] = c;
                }
                break;

            case SpvOpUndef:
                if (wc >= 3) {
                    SpirvConstant c{};
                    c.id = dec.result_id;
                    c.type_id = dec.result_type;
                    c.kind = SpirvConstKind::Undef;
                    out->constants[c.id] = c;
                }
                break;

            case SpvOpVariable:
                if (wc >= 4) {
                    SpirvVariable v{};
                    v.id = dec.result_id;
                    v.ptr_type_id = dec.result_type;
                    v.storage_class = w[3];
                    if (wc >= 5) v.initializer_id = w[4];
                    out->global_vars[v.id] = v;
                }
                break;

            case SpvOpFunction:
                if (wc >= 5) {
                    SpirvFunction f{};
                    f.id = dec.result_id;
                    f.return_type_id = dec.result_type;
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
