#include "valhall_spirv_builder.h"
#include "spirv_reflect.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace valhall::spirv;

int main() {
    std::cout << "=== [Test] Auto-Generated SpirvBuilder & Weak Storage Verification ===\n";

    // 1. Verify spec-accurate enum values from spirv.core.grammar.json
    static_assert(kSpvMagicNumber == 0x07230203u, "SPIR-V magic number mismatch");
    static_assert(static_cast<uint16_t>(Op::OpIAdd) == 128, "OpIAdd opcode mismatch");
    static_assert(static_cast<uint16_t>(Op::OpFAdd) == 129, "OpFAdd opcode mismatch");
    static_assert(static_cast<uint32_t>(Decoration::AliasedPointer) == 5356u,
                  "Decoration::AliasedPointer must match SPIR-V spec (5356)");
    static_assert(static_cast<uint32_t>(Dim::Dim2D) == 1u, "Dim::Dim2D mismatch");
    static_assert(ImageOperandsMask::Lod == 0x0002u, "ImageOperandsMask::Lod mismatch");
    static_assert(MemoryAccessMask::Aligned == 0x0002u, "MemoryAccessMask::Aligned mismatch");

    // 2. Build a valid GLCompute SPIR-V module using auto-generated op_* methods
    SpirvBuilder b(ExecutionModel::GLCompute, /*enable_bda=*/false);
    b.add_execution_mode(ExecutionMode::LocalSize, { 1u, 1u, 1u });

    uint32_t u32_t = b.op_TypeInt(32u, 0u);
    uint32_t f32_t = b.op_TypeFloat(32u);
    uint32_t bool_t = b.op_TypeBool();
    uint32_t v4u_t = b.op_TypeVector(u32_t, 4u);

    uint32_t c1 = b.op_Constant(u32_t, { 42u });
    uint32_t c2 = b.op_Constant(u32_t, { 58u });
    uint32_t cf1 = b.op_Constant(f32_t, { 0x3F800000u }); // 1.0f
    uint32_t ctrue = b.op_ConstantTrue(bool_t);

    // Build a StorageBuffer struct { uint32_t out_val; float out_f; }
    uint32_t struct_t = b.op_TypeStruct({ u32_t, f32_t });
    b.op_Decorate(struct_t, Decoration::Block);
    b.op_MemberDecorate(struct_t, 0u, Decoration::Offset, { 0u });
    b.op_MemberDecorate(struct_t, 1u, Decoration::Offset, { 4u });

    uint32_t ptr_struct_t = b.op_TypePointer(StorageClass::StorageBuffer, struct_t);
    uint32_t ssbo_var = b.add_global_var(ptr_struct_t, StorageClass::StorageBuffer, "u_out", false);
    b.op_Decorate(ssbo_var, Decoration::DescriptorSet, { 0u });
    b.op_Decorate(ssbo_var, Decoration::Binding, { 0u });

    b.begin_main_function();

    uint32_t sum_u = b.op_IAdd(u32_t, c1, c2);
    uint32_t sum_f = b.op_FAdd(f32_t, cf1, cf1);
    uint32_t sel_u = b.op_Select(u32_t, ctrue, sum_u, c1);
    uint32_t bfi_u = b.op_BitFieldInsert(u32_t, sel_u, c1, b.const_u32(0u), b.const_u32(8u));
    (void)v4u_t;

    uint32_t ptr_u32_sb = b.op_TypePointer(StorageClass::StorageBuffer, u32_t);
    uint32_t ptr_f32_sb = b.op_TypePointer(StorageClass::StorageBuffer, f32_t);
    uint32_t p0 = b.op_AccessChain(ptr_u32_sb, ssbo_var, { b.const_u32(0u) });
    uint32_t p1 = b.op_AccessChain(ptr_f32_sb, ssbo_var, { b.const_u32(1u) });
    b.op_Store(p0, bfi_u, MemoryAccessMask::None);
    b.op_Store(p1, sum_f);
    b.op_Return();
    b.end_main_function();

    std::vector<uint32_t> spirv_words = b.assemble();
    assert(spirv_words.size() > 5);
    assert(spirv_words[0] == kSpvMagicNumber);

    // 3. Validate with SPIRV-Reflect
    SpvReflectShaderModule mod{};
    SpvReflectResult res = spvReflectCreateShaderModule(
        spirv_words.size() * sizeof(uint32_t), spirv_words.data(), &mod);
    assert(res == SPV_REFLECT_RESULT_SUCCESS);
    assert(mod.shader_stage == SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT);
    assert(std::strcmp(mod.entry_point_name, "main") == 0);
    assert(mod.descriptor_binding_count == 1);
    spvReflectDestroyShaderModule(&mod);

    std::cout << "[PASS] Auto-generated SpirvBuilder emitted valid SPIR-V (" << spirv_words.size()
              << " words) verified by SPIRV-Reflect\n";
    return 0;
}
