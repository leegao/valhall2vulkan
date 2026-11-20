#include "spirv2ssa.h"
#include "valhall_spirv_builder.h"
#include "valhall_spirv_parser_generated.h"
#include "valhall_ssa_json.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace valhall;
using namespace valhall::spirv;
using namespace valhall::spirv2ssa;
using tables::SsaOp;

namespace {

uint32_t count_ops(const compiler::ValhallSsaModule& mod, SsaOp op) {
    uint32_t c = 0;
    for (size_t i = 1; i < mod.nodes.size(); ++i) {
        if (mod.nodes[i].op == op) ++c;
    }
    return c;
}

std::vector<uint32_t> make_compute_shader_spv() {
    SpirvBuilder b(ExecutionModel::GLCompute, /*enable_bda=*/false);
    b.add_execution_mode(ExecutionMode::LocalSize, {1u, 1u, 1u});

    uint32_t t_u32 = b.op_TypeInt(32u, 0u);
    uint32_t t_v3u32 = b.op_TypeVector(t_u32, 3u);
    uint32_t t_ptr_in_v3u32 = b.op_TypePointer(StorageClass::Input, t_v3u32);
    uint32_t t_rta_u32 = b.op_TypeRuntimeArray(t_u32);
    b.op_Decorate(t_rta_u32, Decoration::ArrayStride, {4u});
    uint32_t t_ssbo = b.op_TypeStruct({t_rta_u32});
    b.op_Decorate(t_ssbo, Decoration::Block);
    b.op_MemberDecorate(t_ssbo, 0u, Decoration::Offset, {0u});
    uint32_t t_ptr_ssbo = b.op_TypePointer(StorageClass::StorageBuffer, t_ssbo);
    uint32_t t_ptr_ssbo_u32 = b.op_TypePointer(StorageClass::StorageBuffer, t_u32);

    uint32_t v_gid = b.add_global_var(t_ptr_in_v3u32, StorageClass::Input, "gl_GlobalInvocationID", true);
    b.op_Decorate(v_gid, Decoration::BuiltIn, {static_cast<uint32_t>(BuiltIn::GlobalInvocationId)});
    uint32_t v_buf = b.add_global_var(t_ptr_ssbo, StorageClass::StorageBuffer, "u_buf", false);
    b.op_Decorate(v_buf, Decoration::DescriptorSet, {0u});
    b.op_Decorate(v_buf, Decoration::Binding, {1u});

    uint32_t c_0 = b.const_u32(0u);
    uint32_t c_2 = b.const_u32(2u);
    uint32_t c_10 = b.const_u32(10u);

    b.begin_main_function();
    uint32_t gid_val = b.op_Load(t_v3u32, v_gid);
    uint32_t gid_x = b.op_CompositeExtract(t_u32, gid_val, {0u});
    uint32_t mul = b.op_IMul(t_u32, gid_x, c_2);
    uint32_t add = b.op_IAdd(t_u32, mul, c_10);
    uint32_t elem_ptr = b.op_AccessChain(t_ptr_ssbo_u32, v_buf, {c_0, gid_x});
    b.op_Store(elem_ptr, add);
    b.op_Return();
    b.end_main_function();
    return b.assemble();
}

std::vector<uint32_t> make_branching_frag_spv() {
    SpirvBuilder b(ExecutionModel::Fragment, /*enable_bda=*/false);
    b.add_execution_mode(ExecutionMode::OriginUpperLeft, {});

    uint32_t t_bool = b.op_TypeBool();
    uint32_t t_f32 = b.op_TypeFloat(32u);
    uint32_t t_v4f32 = b.op_TypeVector(t_f32, 4u);
    uint32_t t_ptr_in = b.op_TypePointer(StorageClass::Input, t_v4f32);
    uint32_t t_ptr_out = b.op_TypePointer(StorageClass::Output, t_v4f32);
    uint32_t t_ptr_fn_f32 = b.op_TypePointer(StorageClass::Function, t_f32);

    uint32_t v_in = b.add_global_var(t_ptr_in, StorageClass::Input, "v_in", true);
    b.op_Decorate(v_in, Decoration::Location, {0u});
    uint32_t v_out = b.add_global_var(t_ptr_out, StorageClass::Output, "v_out", true);
    b.op_Decorate(v_out, Decoration::Location, {0u});

    uint32_t c_half = b.const_f32(0.5f);
    uint32_t c_one = b.const_f32(1.0f);
    uint32_t c_two = b.const_f32(2.0f);

    b.begin_main_function();
    uint32_t local_x = b.op_Variable(t_ptr_fn_f32, StorageClass::Function);
    uint32_t lbl_then = b.alloc_id();
    uint32_t lbl_else = b.alloc_id();
    uint32_t lbl_merge = b.alloc_id();

    uint32_t inp = b.op_Load(t_v4f32, v_in);
    uint32_t x = b.op_CompositeExtract(t_f32, inp, {0u});
    uint32_t cond = b.op_FOrdGreaterThan(t_bool, x, c_half);
    b.op_SelectionMerge(lbl_merge, SelectionControlMask::None);
    b.op_BranchConditional(cond, lbl_then, lbl_else);

    b.emit_label(lbl_then);
    uint32_t x2 = b.op_FMul(t_f32, x, c_two);
    b.op_Store(local_x, x2);
    b.op_Branch(lbl_merge);

    b.emit_label(lbl_else);
    uint32_t x1 = b.op_FAdd(t_f32, x, c_one);
    b.op_Store(local_x, x1);
    b.op_Branch(lbl_merge);

    b.emit_label(lbl_merge);
    uint32_t merged_x = b.op_Load(t_f32, local_x);
    uint32_t out_vec = b.op_CompositeConstruct(t_v4f32, {merged_x, merged_x, merged_x, c_one});
    b.op_Store(v_out, out_vec);
    b.op_Return();
    b.end_main_function();
    return b.assemble();
}

std::vector<uint32_t> make_switch_compute_spv() {
    SpirvBuilder b(ExecutionModel::GLCompute, /*enable_bda=*/false);
    b.add_execution_mode(ExecutionMode::LocalSize, {1u, 1u, 1u});

    uint32_t t_u32 = b.op_TypeInt(32u, 0u);
    uint32_t t_v3u32 = b.op_TypeVector(t_u32, 3u);
    uint32_t t_ptr_in = b.op_TypePointer(StorageClass::Input, t_v3u32);
    uint32_t t_ptr_fn = b.op_TypePointer(StorageClass::Function, t_u32);
    uint32_t t_rta = b.op_TypeRuntimeArray(t_u32);
    b.op_Decorate(t_rta, Decoration::ArrayStride, {4u});
    uint32_t t_ssbo = b.op_TypeStruct({t_rta});
    b.op_Decorate(t_ssbo, Decoration::Block);
    b.op_MemberDecorate(t_ssbo, 0u, Decoration::Offset, {0u});
    uint32_t t_ptr_ssbo = b.op_TypePointer(StorageClass::StorageBuffer, t_ssbo);
    uint32_t t_ptr_ssbo_u32 = b.op_TypePointer(StorageClass::StorageBuffer, t_u32);

    uint32_t v_gid = b.add_global_var(t_ptr_in, StorageClass::Input, "gl_GlobalInvocationID", true);
    b.op_Decorate(v_gid, Decoration::BuiltIn, {static_cast<uint32_t>(BuiltIn::GlobalInvocationId)});
    uint32_t v_buf = b.add_global_var(t_ptr_ssbo, StorageClass::StorageBuffer, "u_buf", false);
    b.op_Decorate(v_buf, Decoration::DescriptorSet, {0u});
    b.op_Decorate(v_buf, Decoration::Binding, {0u});

    b.begin_main_function();
    uint32_t local_res = b.op_Variable(t_ptr_fn, StorageClass::Function);
    uint32_t lbl_c0 = b.alloc_id();
    uint32_t lbl_c1 = b.alloc_id();
    uint32_t lbl_def = b.alloc_id();
    uint32_t lbl_merge = b.alloc_id();

    uint32_t gid = b.op_Load(t_v3u32, v_gid);
    uint32_t x = b.op_CompositeExtract(t_u32, gid, {0u});
    b.op_SelectionMerge(lbl_merge, SelectionControlMask::None);
    b.op_Switch(x, lbl_def, {0u, lbl_c0, 1u, lbl_c1});

    b.emit_label(lbl_c0);
    b.op_Store(local_res, b.const_u32(100u));
    b.op_Branch(lbl_merge);

    b.emit_label(lbl_c1);
    b.op_Store(local_res, b.const_u32(200u));
    b.op_Branch(lbl_merge);

    b.emit_label(lbl_def);
    b.op_Store(local_res, b.const_u32(300u));
    b.op_Branch(lbl_merge);

    b.emit_label(lbl_merge);
    uint32_t out_v = b.op_Load(t_u32, local_res);
    uint32_t ptr = b.op_AccessChain(t_ptr_ssbo_u32, v_buf, {b.const_u32(0u), b.const_u32(0u)});
    b.op_Store(ptr, out_v);
    b.op_Return();
    b.end_main_function();
    return b.assemble();
}

void test_generated_parser_tables() {
    assert(std::strcmp(spirv_parser::spv_op_name(SpvOpIAdd), "OpIAdd") == 0);
    assert(std::strcmp(spirv_parser::spv_op_name(SpvOpSwitch), "OpSwitch") == 0);
    const auto* info = spirv_parser::lookup_spv_op_info(SpvOpBranchConditional);
    assert(info != nullptr && info->is_terminator && !info->has_result_id);
    assert(std::strcmp(tables::ssa_op_to_string(SsaOp::PsbStore), "PsbStore") == 0);
    std::cout << "[PASS] test_generated_parser_tables\n";
}

void test_compute_lowering() {
    auto spv = make_compute_shader_spv();
    auto res = lower_spirv_to_valhall_ssa(spv);
    assert(res.ok);
    assert(res.stage == ShaderStage::Compute);
    assert(res.ssa.blocks.size() == 1);
    assert(count_ops(res.ssa, SsaOp::LoadBuiltinInput) >= 1);
    assert(count_ops(res.ssa, SsaOp::IMul) == 1);
    assert(count_ops(res.ssa, SsaOp::IAdd) == 1);
    assert(count_ops(res.ssa, SsaOp::PsbStore) == 1);
    assert(res.unsupported_ops.empty());

    std::string json = compiler::serialize_ssa_module_json(res.ssa);
    assert(json.find("\"PsbStore\"") != std::string::npos);
    assert(json.find("\"ssbo:0:1\"") != std::string::npos);
    std::cout << "[PASS] test_compute_lowering (" << res.ssa.nodes.size() - 1 << " SSA nodes)\n";
}

void test_branching_frag_mem2reg() {
    auto spv = make_branching_frag_spv();
    auto res = lower_spirv_to_valhall_ssa(spv);
    assert(res.ok);
    assert(res.stage == ShaderStage::Fragment);
    assert(res.ssa.blocks.size() == 4);
    assert(count_ops(res.ssa, SsaOp::LoadFragVarying) == 1);
    assert(count_ops(res.ssa, SsaOp::StoreFragColor) == 1);
    assert(count_ops(res.ssa, SsaOp::Phi) >= 1);
    assert(res.unsupported_ops.empty());
    std::cout << "[PASS] test_branching_frag_mem2reg (" << res.ssa.blocks.size() << " blocks, "
              << count_ops(res.ssa, SsaOp::Phi) << " Phis)\n";
}

void test_switch_desugaring() {
    auto spv = make_switch_compute_spv();
    auto res = lower_spirv_to_valhall_ssa(spv);
    assert(res.ok);
    assert(res.unsupported_ops.empty());
    assert(count_ops(res.ssa, SsaOp::ICmpEq) == 2);
    assert(count_ops(res.ssa, SsaOp::PsbStore) == 1);
    std::cout << "[PASS] test_switch_desugaring (" << res.ssa.blocks.size() << " blocks)\n";
}

} // namespace

int main() {
    test_generated_parser_tables();
    test_compute_lowering();
    test_branching_frag_mem2reg();
    test_switch_desugaring();
    std::cout << "All spirv2ssa tests passed!\n";
    return 0;
}
