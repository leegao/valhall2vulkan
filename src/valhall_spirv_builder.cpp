#include "valhall_spirv_builder.h"
#include <algorithm>
#include <cstring>

namespace valhall::spirv {

SpirvBuilder::SpirvBuilder(ExecutionModel model, bool enable_bda)
    : exec_model_(model), enable_bda_(enable_bda) {
    glsl_ext_id_ = alloc_id();
    main_func_id_ = alloc_id();
    entry_label_id_ = alloc_id();

    add_capability(Capability::Shader);

    if (enable_bda_) {
        add_capability(Capability::PhysicalStorageBufferAddresses);
        add_extension("SPV_KHR_physical_storage_buffer");
    }
}

void SpirvBuilder::append_op(std::vector<uint32_t>& buf, Op opcode, const std::vector<uint32_t>& args) {
    uint32_t word_count = static_cast<uint32_t>(1 + args.size());
    buf.push_back((word_count << 16) | static_cast<uint32_t>(opcode));
    buf.insert(buf.end(), args.begin(), args.end());
}

std::vector<uint32_t> SpirvBuilder::encode_string(const std::string& s) {
    std::vector<uint32_t> words;
    size_t len = s.size() + 1; // include NUL terminator
    size_t num_words = (len + 3) / 4;
    words.resize(num_words, 0u);
    std::memcpy(words.data(), s.c_str(), s.size());
    return words;
}

void SpirvBuilder::validate_id(uint32_t id, const char* op_name, const char* operand_name) {
    if (id == 0 || id >= next_id_) {
        errors_.push_back(std::string("Invalid SSA ID for ") + operand_name + " in " + op_name +
                          ": " + std::to_string(id) + " (bound=" + std::to_string(next_id_) + ")");
    }
}

void SpirvBuilder::add_capability(Capability cap) {
    if (std::find(capabilities_.begin(), capabilities_.end(), cap) == capabilities_.end()) {
        capabilities_.push_back(cap);
    }
}

void SpirvBuilder::add_extension(const std::string& ext) {
    if (std::find(extensions_.begin(), extensions_.end(), ext) == extensions_.end()) {
        extensions_.push_back(ext);
    }
}

void SpirvBuilder::add_execution_mode(ExecutionMode mode, const std::vector<uint32_t>& literals) {
    op_ExecutionMode(main_func_id_, mode, literals);
}

void SpirvBuilder::set_name(uint32_t id, const std::string& name) {
    if (name.empty()) return;
    op_Name(id, name);
}

void SpirvBuilder::set_member_name(uint32_t struct_type_id, uint32_t member_idx, const std::string& name) {
    if (name.empty()) return;
    op_MemberName(struct_type_id, member_idx, name);
}

void SpirvBuilder::decorate(uint32_t target_id, Decoration dec, const std::vector<uint32_t>& literals) {
    op_Decorate(target_id, dec, literals);
}

void SpirvBuilder::member_decorate(uint32_t struct_type_id, uint32_t member_idx, Decoration dec, const std::vector<uint32_t>& literals) {
    op_MemberDecorate(struct_type_id, member_idx, dec, literals);
}

uint32_t SpirvBuilder::type_void() {
    if (void_type_id_ == 0) void_type_id_ = op_TypeVoid();
    return void_type_id_;
}

uint32_t SpirvBuilder::type_bool() {
    if (bool_type_id_ == 0) bool_type_id_ = op_TypeBool();
    return bool_type_id_;
}

uint32_t SpirvBuilder::type_int(uint32_t width, bool is_signed) {
    auto key = std::make_pair(width, is_signed);
    auto it = int_types_.find(key);
    if (it != int_types_.end()) return it->second;
    if (width == 64) add_capability(Capability::Int64);
    else if (width == 16) add_capability(Capability::Int16);
    else if (width == 8) add_capability(Capability::Int8);
    uint32_t id = op_TypeInt(width, is_signed ? 1u : 0u);
    int_types_[key] = id;
    return id;
}

uint32_t SpirvBuilder::type_float(uint32_t width) {
    auto it = float_types_.find(width);
    if (it != float_types_.end()) return it->second;
    if (width == 16) add_capability(Capability::Float16);
    else if (width == 64) add_capability(Capability::Float64);
    uint32_t id = op_TypeFloat(width);
    float_types_[width] = id;
    return id;
}

uint32_t SpirvBuilder::type_vector(uint32_t elem_type_id, uint32_t count) {
    if (count == 1) return elem_type_id;
    auto key = std::make_pair(elem_type_id, count);
    auto it = vector_types_.find(key);
    if (it != vector_types_.end()) return it->second;
    uint32_t id = op_TypeVector(elem_type_id, count);
    vector_types_[key] = id;
    return id;
}

uint32_t SpirvBuilder::type_matrix(uint32_t col_type_id, uint32_t col_count) {
    auto key = std::make_pair(col_type_id, col_count);
    auto it = matrix_types_.find(key);
    if (it != matrix_types_.end()) return it->second;
    uint32_t id = op_TypeMatrix(col_type_id, col_count);
    matrix_types_[key] = id;
    return id;
}

uint32_t SpirvBuilder::type_array(uint32_t elem_type_id, uint32_t length_const_id, uint32_t stride_bytes) {
    auto key = std::make_tuple(elem_type_id, length_const_id, stride_bytes);
    auto it = array_types_.find(key);
    if (it != array_types_.end()) return it->second;
    uint32_t id = op_TypeArray(elem_type_id, length_const_id);
    if (stride_bytes > 0) {
        decorate(id, Decoration::ArrayStride, { stride_bytes });
    }
    array_types_[key] = id;
    return id;
}

uint32_t SpirvBuilder::type_runtime_array(uint32_t elem_type_id, uint32_t stride_bytes) {
    auto key = std::make_pair(elem_type_id, stride_bytes);
    auto it = runtime_array_types_.find(key);
    if (it != runtime_array_types_.end()) return it->second;
    uint32_t id = op_TypeRuntimeArray(elem_type_id);
    if (stride_bytes > 0) {
        decorate(id, Decoration::ArrayStride, { stride_bytes });
    }
    runtime_array_types_[key] = id;
    return id;
}

uint32_t SpirvBuilder::type_struct(const std::vector<uint32_t>& member_types, const std::string& debug_name) {
    uint32_t id = op_TypeStruct(member_types);
    if (!debug_name.empty()) set_name(id, debug_name);
    return id;
}

uint32_t SpirvBuilder::type_pointer(StorageClass sc, uint32_t pointee_type_id) {
    auto key = std::make_pair(static_cast<uint32_t>(sc), pointee_type_id);
    auto it = pointer_types_.find(key);
    if (it != pointer_types_.end()) return it->second;
    uint32_t id = op_TypePointer(sc, pointee_type_id);
    pointer_types_[key] = id;
    return id;
}

uint32_t SpirvBuilder::type_function(uint32_t return_type_id, const std::vector<uint32_t>& param_types) {
    std::vector<uint32_t> key = { return_type_id };
    key.insert(key.end(), param_types.begin(), param_types.end());
    auto it = func_types_.find(key);
    if (it != func_types_.end()) return it->second;
    uint32_t id = op_TypeFunction(return_type_id, param_types);
    func_types_[key] = id;
    return id;
}

uint32_t SpirvBuilder::type_image(uint32_t sampled_type_id, Dim dim, uint32_t depth, uint32_t arrayed, uint32_t ms, uint32_t sampled, uint32_t format) {
    std::vector<uint32_t> key = { sampled_type_id, static_cast<uint32_t>(dim), depth, arrayed, ms, sampled, format };
    auto it = image_types_.find(key);
    if (it != image_types_.end()) return it->second;
    if (dim == Dim::Dim1D) {
        add_capability(sampled == 2 ? Capability::Image1D : Capability::Sampled1D);
    }
    uint32_t id = op_TypeImage(sampled_type_id, dim, depth, arrayed, ms, sampled, static_cast<ImageFormat>(format));
    image_types_[key] = id;
    return id;
}

uint32_t SpirvBuilder::type_sampler() {
    if (sampler_type_id_ == 0) sampler_type_id_ = op_TypeSampler();
    return sampler_type_id_;
}

uint32_t SpirvBuilder::type_sampled_image(uint32_t image_type_id) {
    auto it = sampled_image_types_.find(image_type_id);
    if (it != sampled_image_types_.end()) return it->second;
    uint32_t id = op_TypeSampledImage(image_type_id);
    sampled_image_types_[image_type_id] = id;
    return id;
}

uint32_t SpirvBuilder::const_bool(bool val) {
    uint32_t btype = type_bool();
    if (val) {
        if (const_true_id_ == 0) const_true_id_ = op_ConstantTrue(btype);
        return const_true_id_;
    } else {
        if (const_false_id_ == 0) const_false_id_ = op_ConstantFalse(btype);
        return const_false_id_;
    }
}

uint32_t SpirvBuilder::const_u32(uint32_t val) {
    auto it = const_u32_map_.find(val);
    if (it != const_u32_map_.end()) return it->second;
    uint32_t id = op_Constant(type_u32(), { val });
    const_u32_map_[val] = id;
    return id;
}

uint32_t SpirvBuilder::const_s32(int32_t val) {
    auto it = const_s32_map_.find(val);
    if (it != const_s32_map_.end()) return it->second;
    uint32_t id = op_Constant(type_s32(), { static_cast<uint32_t>(val) });
    const_s32_map_[val] = id;
    return id;
}

uint32_t SpirvBuilder::const_f32_bits(uint32_t bits) {
    auto it = const_f32_map_.find(bits);
    if (it != const_f32_map_.end()) return it->second;
    uint32_t id = op_Constant(type_f32(), { bits });
    const_f32_map_[bits] = id;
    return id;
}

uint32_t SpirvBuilder::const_f32(float val) {
    uint32_t bits = 0;
    std::memcpy(&bits, &val, sizeof(float));
    return const_f32_bits(bits);
}

uint32_t SpirvBuilder::const_u64(uint64_t val) {
    auto it = const_u64_map_.find(val);
    if (it != const_u64_map_.end()) return it->second;
    uint32_t id = op_Constant(type_u64(),
                              { static_cast<uint32_t>(val & 0xFFFFFFFFULL), static_cast<uint32_t>(val >> 32) });
    const_u64_map_[val] = id;
    return id;
}

uint32_t SpirvBuilder::const_u16(uint16_t val) {
    auto it = const_u16_map_.find(val);
    if (it != const_u16_map_.end()) return it->second;
    uint32_t id = op_Constant(type_u16(), { static_cast<uint32_t>(val) });
    const_u16_map_[val] = id;
    return id;
}

uint32_t SpirvBuilder::const_s64(int64_t val) {
    auto it = const_s64_map_.find(val);
    if (it != const_s64_map_.end()) return it->second;
    uint64_t uval = static_cast<uint64_t>(val);
    uint32_t id = op_Constant(type_s64(),
                              { static_cast<uint32_t>(uval & 0xFFFFFFFFULL), static_cast<uint32_t>(uval >> 32) });
    const_s64_map_[val] = id;
    return id;
}

uint32_t SpirvBuilder::const_s16(int16_t val) {
    auto it = const_s16_map_.find(val);
    if (it != const_s16_map_.end()) return it->second;
    uint32_t id = op_Constant(type_s16(), { static_cast<uint32_t>(static_cast<uint16_t>(val)) });
    const_s16_map_[val] = id;
    return id;
}

uint32_t SpirvBuilder::const_u8(uint8_t val) {
    auto it = const_u8_map_.find(val);
    if (it != const_u8_map_.end()) return it->second;
    uint32_t id = op_Constant(type_u8(), { static_cast<uint32_t>(val) });
    const_u8_map_[val] = id;
    return id;
}

uint32_t SpirvBuilder::const_s8(int8_t val) {
    auto it = const_s8_map_.find(val);
    if (it != const_s8_map_.end()) return it->second;
    uint32_t id = op_Constant(type_s8(), { static_cast<uint32_t>(static_cast<uint8_t>(val)) });
    const_s8_map_[val] = id;
    return id;
}

uint32_t SpirvBuilder::const_f16(uint16_t bits) {
    auto it = const_f16_map_.find(bits);
    if (it != const_f16_map_.end()) return it->second;
    uint32_t id = op_Constant(type_f16(), { static_cast<uint32_t>(bits) });
    const_f16_map_[bits] = id;
    return id;
}

uint32_t SpirvBuilder::const_f64(double val) {
    uint64_t bits = 0;
    std::memcpy(&bits, &val, sizeof(double));
    auto it = const_f64_map_.find(bits);
    if (it != const_f64_map_.end()) return it->second;
    uint32_t id = op_Constant(type_f64(),
                              { static_cast<uint32_t>(bits & 0xFFFFFFFFULL), static_cast<uint32_t>(bits >> 32) });
    const_f64_map_[bits] = id;
    return id;
}

uint32_t SpirvBuilder::const_composite(uint32_t type_id, const std::vector<uint32_t>& constituents) {
    std::vector<uint32_t> key = { type_id };
    key.insert(key.end(), constituents.begin(), constituents.end());
    auto it = const_composite_map_.find(key);
    if (it != const_composite_map_.end()) return it->second;
    uint32_t id = op_ConstantComposite(type_id, constituents);
    const_composite_map_[key] = id;
    return id;
}

uint32_t SpirvBuilder::const_null(uint32_t type_id) {
    auto it = const_null_map_.find(type_id);
    if (it != const_null_map_.end()) return it->second;
    uint32_t id = op_ConstantNull(type_id);
    const_null_map_[type_id] = id;
    return id;
}

uint32_t SpirvBuilder::add_global_var(uint32_t ptr_type_id, StorageClass sc, const std::string& name, bool add_to_interface) {
    uint32_t id = op_Variable(ptr_type_id, sc);
    if (!name.empty()) set_name(id, name);
    if (add_to_interface || sc == StorageClass::Input || sc == StorageClass::Output) {
        if (std::find(entry_interface_ids_.begin(), entry_interface_ids_.end(), id) == entry_interface_ids_.end()) {
            entry_interface_ids_.push_back(id);
        }
    }
    return id;
}

uint32_t SpirvBuilder::add_local_var(uint32_t value_type_id, const std::string& name) {
    uint32_t ptr_t = type_pointer(StorageClass::Function, value_type_id);
    uint32_t id = op_Variable(ptr_t, StorageClass::Function);
    if (!name.empty()) set_name(id, name);
    return id;
}

uint32_t SpirvBuilder::get_or_create_builtin_input(BuiltIn builtin, uint32_t value_type_id, const std::string& name, bool flat_decor) {
    uint32_t key = static_cast<uint32_t>(builtin);
    auto it = builtin_inputs_.find(key);
    if (it != builtin_inputs_.end()) return it->second;
    uint32_t ptr_t = type_pointer(StorageClass::Input, value_type_id);
    uint32_t var_id = add_global_var(ptr_t, StorageClass::Input, name, true);
    decorate(var_id, Decoration::BuiltIn, { key });
    if (flat_decor && exec_model_ == ExecutionModel::Fragment) {
        decorate(var_id, Decoration::Flat);
    }
    builtin_inputs_[key] = var_id;
    return var_id;
}

uint32_t SpirvBuilder::get_or_create_builtin_output(BuiltIn builtin, uint32_t value_type_id, const std::string& name) {
    uint32_t key = static_cast<uint32_t>(builtin);
    auto it = builtin_outputs_.find(key);
    if (it != builtin_outputs_.end()) return it->second;
    uint32_t ptr_t = type_pointer(StorageClass::Output, value_type_id);
    uint32_t var_id = add_global_var(ptr_t, StorageClass::Output, name, true);
    decorate(var_id, Decoration::BuiltIn, { key });
    builtin_outputs_[key] = var_id;
    return var_id;
}

void SpirvBuilder::begin_main_function() {
    set_name(main_func_id_, "main");
    current_label_id_ = entry_label_id_;
    in_entry_block_ = true;
}

void SpirvBuilder::end_main_function() {
    in_entry_block_ = false;
}

uint32_t SpirvBuilder::emit_label(uint32_t label_id) {
    if (label_id == 0) return op_Label();
    in_entry_block_ = false;
    current_label_id_ = label_id;
    append_op(body_words_, Op::OpLabel, { label_id });
    return label_id;
}

uint32_t SpirvBuilder::emit_phi(uint32_t result_type_id, size_t num_incomings) {
    uint32_t id = alloc_id();
    phi_word_offsets_[id] = body_words_.size();
    std::vector<uint32_t> args(2 + num_incomings * 2, 0u);
    args[0] = result_type_id;
    args[1] = id;
    append_op(body_words_, Op::OpPhi, args);
    return id;
}

void SpirvBuilder::patch_phi(uint32_t phi_id, const std::vector<std::pair<uint32_t, uint32_t>>& incomings) {
    auto it = phi_word_offsets_.find(phi_id);
    if (it == phi_word_offsets_.end()) return;
    size_t off = it->second;
    for (size_t i = 0; i < incomings.size(); ++i) {
        body_words_[off + 3 + i * 2] = incomings[i].first;
        body_words_[off + 4 + i * 2] = incomings[i].second;
    }
}

uint32_t SpirvBuilder::emit_glsl_ext(GlslStd450 ext_op, uint32_t result_type_id, const std::vector<uint32_t>& operands) {
    uint32_t id = alloc_id();
    std::vector<uint32_t> args = { result_type_id, id, glsl_ext_id_, static_cast<uint32_t>(ext_op) };
    args.insert(args.end(), operands.begin(), operands.end());
    append_op(body_words_, Op::OpExtInst, args);
    return id;
}

uint32_t SpirvBuilder::emit_load(uint32_t result_type_id, uint32_t ptr_id, uint32_t align_bytes) {
    if (align_bytes > 0) {
        return op_Load(result_type_id, ptr_id, MemoryAccessMask::Aligned, { align_bytes });
    }
    return op_Load(result_type_id, ptr_id);
}

void SpirvBuilder::emit_store(uint32_t ptr_id, uint32_t val_id, uint32_t align_bytes) {
    if (align_bytes > 0) {
        op_Store(ptr_id, val_id, MemoryAccessMask::Aligned, { align_bytes });
    } else {
        op_Store(ptr_id, val_id);
    }
}

std::vector<uint32_t> SpirvBuilder::assemble() const {
    // Ensure void() function type exists
    SpirvBuilder* self = const_cast<SpirvBuilder*>(this);
    uint32_t void_t = self->type_void();
    uint32_t main_fn_type = self->type_function(void_t);

    std::vector<uint32_t> out;
    out.reserve(512 + types_consts_globals_words_.size() + local_vars_words_.size() + body_words_.size());

    // 1. SPIR-V 5-word Header
    out.push_back(kSpvMagicNumber);
    out.push_back(kSpvVersion13);
    out.push_back(kSpvGeneratorId);
    out.push_back(self->next_id_);
    out.push_back(0u);

    // 2. OpCapability
    for (Capability cap : capabilities_) {
        append_op(out, Op::OpCapability, { static_cast<uint32_t>(cap) });
    }

    // 3. OpExtension
    for (const auto& ext : extensions_) {
        append_op(out, Op::OpExtension, encode_string(ext));
    }

    // 4. OpExtInstImport "GLSL.std.450"
    {
        std::vector<uint32_t> args = { glsl_ext_id_ };
        auto s = encode_string("GLSL.std.450");
        args.insert(args.end(), s.begin(), s.end());
        append_op(out, Op::OpExtInstImport, args);
    }

    // 5. OpMemoryModel
    AddressingModel addr_model = enable_bda_ ? AddressingModel::PhysicalStorageBuffer64 : AddressingModel::Logical;
    append_op(out, Op::OpMemoryModel, { static_cast<uint32_t>(addr_model), static_cast<uint32_t>(MemoryModel::GLSL450) });

    // 6. OpEntryPoint
    {
        std::vector<uint32_t> args = { static_cast<uint32_t>(exec_model_), main_func_id_ };
        auto name_words = encode_string("main");
        args.insert(args.end(), name_words.begin(), name_words.end());
        args.insert(args.end(), entry_interface_ids_.begin(), entry_interface_ids_.end());
        append_op(out, Op::OpEntryPoint, args);
    }

    // 7. OpExecutionMode
    out.insert(out.end(), execution_modes_words_.begin(), execution_modes_words_.end());

    // 8. Debug OpName / OpMemberName
    out.insert(out.end(), debug_names_words_.begin(), debug_names_words_.end());

    // 9. Annotations (OpDecorate / OpMemberDecorate)
    out.insert(out.end(), decorations_words_.begin(), decorations_words_.end());

    // 10. Types, Constants, Global Variables
    out.insert(out.end(), types_consts_globals_words_.begin(), types_consts_globals_words_.end());

    // 11. Main Function
    append_op(out, Op::OpFunction, { void_t, main_func_id_, 0u, main_fn_type });
    append_op(out, Op::OpLabel, { entry_label_id_ });
    out.insert(out.end(), local_vars_words_.begin(), local_vars_words_.end());
    out.insert(out.end(), body_words_.begin(), body_words_.end());
    append_op(out, Op::OpFunctionEnd, {});

    return out;
}

} // namespace valhall::spirv
