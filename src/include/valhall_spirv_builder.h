#pragma once
#include "valhall_spirv_builder_generated.h"
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <tuple>

namespace valhall::spirv {

class SpirvBuilder {
public:
    explicit SpirvBuilder(ExecutionModel model, bool enable_bda = true);

    uint32_t alloc_id() { return next_id_++; }
    uint32_t glsl_ext_id() const { return glsl_ext_id_; }
    ExecutionModel execution_model() const { return exec_model_; }

    // Capabilities & Extensions
    void add_capability(Capability cap);
    void add_extension(const std::string& ext);
    void add_execution_mode(ExecutionMode mode, const std::vector<uint32_t>& literals = {});

    // Debug names & decorations
    void set_name(uint32_t id, const std::string& name);
    void set_member_name(uint32_t struct_type_id, uint32_t member_idx, const std::string& name);
    void decorate(uint32_t target_id, Decoration dec, const std::vector<uint32_t>& literals = {});
    void member_decorate(uint32_t struct_type_id, uint32_t member_idx, Decoration dec, const std::vector<uint32_t>& literals = {});

    // Deduplicated Types
    uint32_t type_void();
    uint32_t type_bool();
    uint32_t type_int(uint32_t width, bool is_signed);
    uint32_t type_u8()  { return type_int(8, false); }
    uint32_t type_s8()  { return type_int(8, true); }
    uint32_t type_u16() { return type_int(16, false); }
    uint32_t type_s16() { return type_int(16, true); }
    uint32_t type_u32() { return type_int(32, false); }
    uint32_t type_s32() { return type_int(32, true); }
    uint32_t type_u64() { return type_int(64, false); }
    uint32_t type_s64() { return type_int(64, true); }
    uint32_t type_float(uint32_t width);
    uint32_t type_f16() { return type_float(16); }
    uint32_t type_f32() { return type_float(32); }
    uint32_t type_f64() { return type_float(64); }
    uint32_t type_vector(uint32_t elem_type_id, uint32_t count);
    uint32_t type_matrix(uint32_t col_type_id, uint32_t col_count);
    uint32_t type_array(uint32_t elem_type_id, uint32_t length_const_id, uint32_t stride_bytes = 0);
    uint32_t type_runtime_array(uint32_t elem_type_id, uint32_t stride_bytes = 0);
    uint32_t type_struct(const std::vector<uint32_t>& member_types, const std::string& debug_name = "");
    uint32_t type_pointer(StorageClass sc, uint32_t pointee_type_id);
    uint32_t type_function(uint32_t return_type_id, const std::vector<uint32_t>& param_types = {});
    uint32_t type_image(uint32_t sampled_type_id, Dim dim, uint32_t depth, uint32_t arrayed, uint32_t ms, uint32_t sampled, uint32_t format = 0);
    uint32_t type_sampler();
    uint32_t type_sampled_image(uint32_t image_type_id);

    // Deduplicated Constants
    uint32_t const_bool(bool val);
    uint32_t const_u32(uint32_t val);
    uint32_t const_s32(int32_t val);
    uint32_t const_f32(float val);
    uint32_t const_f32_bits(uint32_t bits);
    uint32_t const_u64(uint64_t val);
    uint32_t const_s64(int64_t val);
    uint32_t const_u16(uint16_t val);
    uint32_t const_s16(int16_t val);
    uint32_t const_u8(uint8_t val);
    uint32_t const_s8(int8_t val);
    uint32_t const_f16(uint16_t bits);
    uint32_t const_f64(double val);
    uint32_t const_composite(uint32_t type_id, const std::vector<uint32_t>& constituents);
    uint32_t const_null(uint32_t type_id);

    // Global & Local Variables
    uint32_t add_global_var(uint32_t ptr_type_id, StorageClass sc, const std::string& name = "", bool add_to_interface = false);
    uint32_t add_local_var(uint32_t value_type_id, const std::string& name = "");

    // BuiltIn Input/Output variable helpers
    uint32_t get_or_create_builtin_input(BuiltIn builtin, uint32_t value_type_id, const std::string& name, bool flat_decor = false);
    uint32_t get_or_create_builtin_output(BuiltIn builtin, uint32_t value_type_id, const std::string& name);

    // Function & Block Emission
    void begin_main_function();
    void end_main_function();
    uint32_t emit_label(uint32_t label_id = 0);
    uint32_t emit_phi(uint32_t result_type_id, size_t num_incomings);
    void patch_phi(uint32_t phi_id, const std::vector<std::pair<uint32_t, uint32_t>>& incomings);
    uint32_t current_label() const { return current_label_id_; }

    // Generic GLSL.std.450 extended-instruction emitter
    uint32_t emit_glsl_ext(GlslStd450 ext_op, uint32_t result_type_id, const std::vector<uint32_t>& operands);

    // Load/store helpers that fold in alignment handling
    uint32_t emit_load(uint32_t result_type_id, uint32_t ptr_id, uint32_t align_bytes = 0);
    void emit_store(uint32_t ptr_id, uint32_t val_id, uint32_t align_bytes = 0);

    // Auto-generated weak-linkage emitters for all 879 SPIR-V Core Grammar ops
    VALHALL_SPIRV_BUILDER_GENERATED_OPS

    // Validation diagnostics accumulated during emission
    bool has_errors() const { return !errors_.empty(); }
    const std::vector<std::string>& errors() const { return errors_; }

    // Finalize binary SPIR-V words
    std::vector<uint32_t> assemble() const;

private:
    ExecutionModel exec_model_;
    bool enable_bda_;
    uint32_t next_id_ = 1;
    uint32_t glsl_ext_id_ = 0;
    uint32_t main_func_id_ = 0;
    uint32_t entry_label_id_ = 0;
    uint32_t current_label_id_ = 0;
    bool in_entry_block_ = false;

    std::vector<Capability> capabilities_;
    std::vector<std::string> extensions_;
    std::vector<uint32_t> entry_interface_ids_;
    std::vector<uint32_t> execution_modes_words_;
    std::vector<uint32_t> debug_names_words_;
    std::vector<uint32_t> decorations_words_;
    std::vector<uint32_t> types_consts_globals_words_;
    std::vector<uint32_t> local_vars_words_;
    std::vector<uint32_t> body_words_;
    std::map<uint32_t, size_t> phi_word_offsets_;

    // Deduplication maps
    uint32_t void_type_id_ = 0;
    uint32_t bool_type_id_ = 0;
    uint32_t sampler_type_id_ = 0;
    std::map<std::pair<uint32_t, bool>, uint32_t> int_types_;
    std::map<uint32_t, uint32_t> float_types_;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> vector_types_;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> matrix_types_;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> array_types_;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> runtime_array_types_;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> pointer_types_;
    std::map<std::vector<uint32_t>, uint32_t> func_types_;
    std::map<std::vector<uint32_t>, uint32_t> image_types_;
    std::map<uint32_t, uint32_t> sampled_image_types_;

    uint32_t const_true_id_ = 0;
    uint32_t const_false_id_ = 0;
    std::map<uint32_t, uint32_t> const_u32_map_;
    std::map<int32_t, uint32_t> const_s32_map_;
    std::map<uint32_t, uint32_t> const_f32_map_;
    std::map<uint64_t, uint32_t> const_u64_map_;
    std::map<int64_t, uint32_t> const_s64_map_;
    std::map<uint16_t, uint32_t> const_u16_map_;
    std::map<int16_t, uint32_t> const_s16_map_;
    std::map<uint8_t, uint32_t> const_u8_map_;
    std::map<int8_t, uint32_t> const_s8_map_;
    std::map<uint16_t, uint32_t> const_f16_map_;
    std::map<uint64_t, uint32_t> const_f64_map_;
    std::vector<std::string> errors_;
    std::map<std::vector<uint32_t>, uint32_t> const_composite_map_;
    std::map<uint32_t, uint32_t> const_null_map_;
    std::map<uint32_t, uint32_t> builtin_inputs_;
    std::map<uint32_t, uint32_t> builtin_outputs_;

    static void append_op(std::vector<uint32_t>& buf, Op opcode, const std::vector<uint32_t>& args);
    static std::vector<uint32_t> encode_string(const std::string& s);
    void validate_id(uint32_t id, const char* op_name, const char* operand_name);
};

} // namespace valhall::spirv
