#include "abstract_domain.h"
#include "pointer_analysis.h"
#include "transfer.h"
#include "valhall_isa_generated.h"
#include "valhall_to_spirv.h"
#include "spirv_reflect.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace valhall;
using namespace valhall::dataflow;

namespace {

// Custom recording visitor verifying grouped dispatch (AND + AND_IMM -> on_and, etc.)
struct DispatchRecorder : public TransferVisitorBase {
    int and_calls = 0;
    int or_calls = 0;
    int iadd_calls = 0;
    int shadd_calls = 0;
    int mov_calls = 0;
    int adr_calls = 0;
    int load_calls = 0;
    int store_calls = 0;
    int kill_calls = 0;

    void on_reg_killed(uint8_t, const DecodedValhallInst&) override { ++kill_calls; }
    void on_and(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override { ++and_calls; }
    void on_or(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override { ++or_calls; }
    void on_xor(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_not(const AbstractDest&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_shift(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, ShiftKind, const DecodedValhallInst&) override {}
    void on_iadd(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override { ++iadd_calls; }
    void on_isub(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_imul(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_shadd(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, uint32_t, const DecodedValhallInst&) override { ++shadd_calls; }
    void on_iunary(const AbstractDest&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_fadd(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_fmul(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_fma(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_fminmax(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_funary(const AbstractDest&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_mov(const AbstractDest&, const AbstractOperand&, const DecodedValhallInst&) override { ++mov_calls; }
    void on_convert(const AbstractDest&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_cmp(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_csel(const AbstractDest&, const AbstractOperand&, const AbstractOperand&, const AbstractOperand&, const AbstractOperand&, const DecodedValhallInst&) override {}
    void on_adr(const AbstractDest&, int64_t, const DecodedValhallInst&) override { ++adr_calls; }
    void on_lea(const AbstractDest&, const AbstractOperand&, uint32_t, uint32_t, const DecodedValhallInst&) override {}
    void on_load(const AbstractDest&, const AbstractOperand&, int64_t, const DecodedValhallInst&) override { ++load_calls; }
    void on_store(const DecodedStaging&, const AbstractOperand&, int64_t, const DecodedValhallInst&) override { ++store_calls; }
    void on_atomic(const AbstractDest&, const DecodedStaging&, const AbstractOperand&, int64_t, const DecodedValhallInst&) override {}
    void on_ld_attr(const AbstractDest&, uint32_t, uint32_t, const DecodedValhallInst&) override {}
    void on_ld_var(const AbstractDest&, uint32_t, uint32_t, const DecodedValhallInst&) override {}
    void on_tex(const AbstractDest&, uint32_t, uint32_t, const DecodedValhallInst&) override {}
    void on_blend(const DecodedStaging&, int32_t, const DecodedValhallInst&) override {}
    void on_branch(const AbstractOperand&, int64_t, const DecodedValhallInst&) override {}
    void on_nop_or_barrier(const DecodedValhallInst&) override {}
    void on_unhandled(const AbstractDest&, const DecodedValhallInst&) override {}
};

}  // namespace

int main() {
    std::cout << "=== [Test] Valhall Abstract Interpretation & Pointer Dataflow Framework ===\n";

    // 1. Verify TrackedValDomain lattice laws (bottom, top, join, meet, leq, idempotence, commutativity)
    {
        TrackedVal bot = TrackedValDomain::bottom();
        TrackedVal top = TrackedValDomain::top();
        TrackedVal c42 = TrackedVal::make_const(42);
        TrackedVal c99 = TrackedVal::make_const(99);
        TrackedVal buf_fixed = TrackedVal::make_buffer_addr(4, 0, "input_buf", 0, false);
        TrackedVal buf_off8 = TrackedVal::make_buffer_addr(4, 0, "input_buf", 8, false);
        TrackedVal buf_idx = TrackedVal::make_buffer_addr(4, 0, "input_buf", 0, true);
        TrackedVal buf_other = TrackedVal::make_buffer_addr(4, 1, "output_buf", 0, false);

        assert(TrackedValDomain::leq(bot, c42));
        assert(TrackedValDomain::leq(c42, top));
        assert(TrackedValDomain::equals(TrackedValDomain::join(bot, c42), c42));
        assert(TrackedValDomain::equals(TrackedValDomain::join(c42, c42), c42));
        assert(TrackedValDomain::equals(TrackedValDomain::join(c42, c99), top));
        assert(TrackedValDomain::equals(TrackedValDomain::meet(top, c42), c42));
        assert(TrackedValDomain::equals(TrackedValDomain::meet(c42, c99), bot));

        // Same buffer at different offsets joins to dynamically indexed BufferAddr!
        TrackedVal joined_buf = TrackedValDomain::join(buf_fixed, buf_off8);
        assert(TrackedValDomain::equals(joined_buf, buf_idx));
        assert(TrackedValDomain::leq(buf_fixed, buf_idx));
        assert(TrackedValDomain::equals(TrackedValDomain::meet(buf_fixed, buf_idx), buf_fixed));
        // Distinct buffers join to Top
        assert(TrackedValDomain::equals(TrackedValDomain::join(buf_fixed, buf_other), top));

        // SymbolValue direct <= derived
        TrackedVal sym_direct = TrackedVal::make_symbol_value(0, 0, "u_Dims.x", false);
        TrackedVal sym_derived = TrackedVal::make_symbol_value(0, 0, "u_Dims.x", true);
        assert(TrackedValDomain::leq(sym_direct, sym_derived));
        assert(TrackedValDomain::equals(TrackedValDomain::join(sym_direct, sym_derived), sym_derived));
        assert(TrackedValDomain::equals(TrackedValDomain::meet(sym_direct, sym_derived), sym_direct));
    }
    std::cout << "[PASS] TrackedValDomain lattice laws (join, meet, leq, generalization) verified\n";

    // 2. Verify grouped transfer dispatch: AND + AND_IMM -> on_and, MOV_IMM -> on_mov, ADR -> on_adr, kill `^`
    {
        // Real Valhall words from sample_ref.spv and s2v_test_comp.spv:
        uint64_t w_and_imm = 0x0118fd00ffffff7dULL;  // AND_IMM.i32 r61, r61^, #0xffffff (v11)
        uint64_t w_and     = 0x00b4c3011044c043ULL;  // AND.i32 r3, r3^, r4^ (v10)
        uint64_t w_mov_imm = 0x0910ff00000003c0ULL;  // MOV_IMM.i32.wait0 r63, #0x3 (v11)
        uint64_t w_adr     = 0x0710fc00000090feULL;  // ADR r60, 0x100 (v11, pc=0x68)
        uint64_t w_load    = 0x006084f468004042ULL;  // LOAD.i64.slot1 @r4:r5, [r2^:r3^], #0x40 (v11)
        uint64_t w_store   = 0x2061460468000000ULL;  // STORE.i64.slot1.wait2 @r6:r7, [r0:r1], #0x0 (v11)
        uint64_t w_shadd   = 0x01a3c00020270282ULL;  // SHADD.u64 [r0:r1], u1, [r2:r3].w0, #0x2 (v11)

        DispatchRecorder rec;
        dispatch_valhall_transfer(decode_valhall_instruction(w_and_imm, 11, 0x18), rec);
        dispatch_valhall_transfer(decode_valhall_instruction(w_and, 10, 0x10), rec);
        dispatch_valhall_transfer(decode_valhall_instruction(w_mov_imm, 11, 0x10), rec);
        dispatch_valhall_transfer(decode_valhall_instruction(w_adr, 11, 0x68), rec);
        dispatch_valhall_transfer(decode_valhall_instruction(w_load, 11, 0x08), rec);
        dispatch_valhall_transfer(decode_valhall_instruction(w_store, 11, 0x60), rec);
        dispatch_valhall_transfer(decode_valhall_instruction(w_shadd, 11, 0x58), rec);

        assert(rec.and_calls == 2);  // BOTH AND and AND_IMM dispatched to on_and!
        assert(rec.mov_calls == 1);
        assert(rec.adr_calls == 1);
        assert(rec.load_calls == 1);
        assert(rec.store_calls == 1);
        assert(rec.shadd_calls == 1);
        assert(rec.kill_calls >= 2); // r4^ on AND and r2^:r3^ on LOAD fired on_reg_killed!
    }
    std::cout << "[PASS] Grouped Valhall transfer dispatch (AND/AND_IMM, MOV_IMM, ADR, LOAD, STORE, SHADD, ^ kills) verified\n";

    // 3. Verify extensional equalities in DomainTransferVisitor (MOV alias, IADD +0, AND ~0, CSEL join)
    {
        PointerState st;
        TrackedVal ptr = TrackedVal::make_buffer_addr(4, 0, "my_ssbo", 0, false);
        st.set_reg(10, ptr);

        DomainTransferVisitor<TrackedValDomain> vis(&st);
        DecodedValhallInst dummy{};
        dummy.valid = true;

        // MOV r11, r10 -> aliases ptr
        vis.on_mov(AbstractDest{true, 11, 1, false}, AbstractOperand::from_reg(10), dummy);
        assert(TrackedValDomain::equals(st.get_reg(11), ptr));

        // IADD r12, r11, 0 -> identity copy
        vis.on_iadd(AbstractDest{true, 12, 1, false}, AbstractOperand::from_reg(11), AbstractOperand::from_imm(0, 0), dummy);
        assert(TrackedValDomain::equals(st.get_reg(12), ptr));

        // AND r13, r12, 0xFFFFFFFF -> identity copy
        vis.on_and(AbstractDest{true, 13, 1, false}, AbstractOperand::from_reg(12), AbstractOperand::from_imm(0xFFFFFFFFULL, -1), dummy);
        assert(TrackedValDomain::equals(st.get_reg(13), ptr));

        // CSEL r14, cond, 0, r10, r15 (where r15 is same buffer @ +16) -> joins to indexed BufferAddr!
        st.set_reg(15, TrackedVal::make_buffer_addr(4, 0, "my_ssbo", 16, false));
        vis.on_csel(AbstractDest{true, 14, 1, false}, AbstractOperand::from_reg(0), AbstractOperand::from_imm(0, 0),
                    AbstractOperand::from_reg(10), AbstractOperand::from_reg(15), dummy);
        assert(st.get_reg(14).kind == TrackedVal::Kind::BufferAddr && st.get_reg(14).is_indexed);
    }
    std::cout << "[PASS] Extensional equalities (MOV alias, +0, &~0, CSEL join) verified\n";

    // 4. Verify optional ValhallSymbolBindings in valhall2spirv (OpName emission)
    {
        // Minimal 5-instruction compute shader (from s2v_test_comp.spv):
        //   0x00: SHADD.u64 [r0:r1], u0, [r60^:r61^].w0, #0x2
        //   0x08: LOAD.i32.slot0.wait0 @r2, [r0:r1], #0x0
        //   0x10: FMA.f32 r2, r2^, k0f2f, k1f
        //   0x18: STORE.i32.slot0.end @r2, [r0^:r1^], #0x0
        const uint64_t code_words[4] = {
            0x01a3c00020277c80ULL,
            0x0860828218000000ULL,
            0x00b2c20000d0dc42ULL,
            0x7861420218000040ULL,
        };
        ValhallTranslateOptions topts;
        topts.stage = ShaderStage::Compute;
        topts.arch = 10;
        topts.symbol_bindings.sideband_var_name = "my_custom_sideband";
        topts.symbol_bindings.fau_word_names[0] = "fau_my_ssbo_lo";
        topts.symbol_bindings.fau_word_names[4] = "fau_my_ssbo_hi";

        auto tres = translate_valhall_to_spirv(
            reinterpret_cast<const uint8_t*>(code_words), sizeof(code_words), topts);
        assert(tres.ok);
        SpvReflectShaderModule mod{};
        assert(spvReflectCreateShaderModule(tres.spirv_words.size() * 4, tres.spirv_words.data(), &mod) == SPV_REFLECT_RESULT_SUCCESS);
        spvReflectDestroyShaderModule(&mod);

        // Verify OpName strings appear in the emitted SPIR-V words
        std::string raw(reinterpret_cast<const char*>(tres.spirv_words.data()), tres.spirv_words.size() * 4);
        assert(raw.find("my_custom_sideband") != std::string::npos);
        assert(raw.find("fau_my_ssbo_lo") != std::string::npos);
        assert(raw.find("fau_my_ssbo_hi") != std::string::npos);
    }
    std::cout << "[PASS] Optional ValhallSymbolBindings OpName emission in valhall2spirv verified\n";

    return 0;
}
