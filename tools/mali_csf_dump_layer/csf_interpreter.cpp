#include "csf_interpreter.h"
#include "csf_config_log.h"
#include "csf_kbase_uapi.h"
#include "valhall_csf_descriptors_generated.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>

namespace mali_csf_dump {

namespace {

inline uint64_t read_reg64(const CsfQueueState& q, uint32_t reg) {
    if (reg + 1u >= kCsfMaxRegs) return 0ULL;
    return static_cast<uint64_t>(q.regs[reg]) |
           (static_cast<uint64_t>(q.regs[reg + 1u]) << 32);
}

inline void write_reg64(CsfQueueState& q, uint32_t reg, uint64_t val) {
    if (reg + 1u >= kCsfMaxRegs) return;
    q.regs[reg] = static_cast<uint32_t>(val & 0xFFFFFFFFULL);
    q.regs[reg + 1u] = static_cast<uint32_t>(val >> 32);
}

inline uint32_t bfext_u32(uint32_t v, uint32_t pos, uint32_t width) {
    if (pos >= 32u || width == 0u) return 0u;
    uint32_t shifted = v >> pos;
    if (width >= 32u) return shifted;
    return shifted & ((1u << width) - 1u);
}

inline uint32_t bfext_s32(uint32_t v, uint32_t pos, uint32_t width) {
    if (pos >= 32u || width == 0u) return 0u;
    uint32_t u = bfext_u32(v, pos, width);
    if (width < 32u && (u & (1u << (width - 1u))) != 0u) {
        u |= ~((1u << width) - 1u);
    }
    return u;
}

inline uint32_t bfins_32(uint32_t base, uint32_t ins, uint32_t pos, uint32_t width) {
    if (pos >= 32u || width == 0u) return base;
    uint32_t w = std::min(width, 32u - pos);
    uint32_t mask = (w >= 32u) ? 0xFFFFFFFFu : (((1u << w) - 1u) << pos);
    return (base & ~mask) | ((ins << pos) & mask);
}

inline uint64_t bfext_u64(uint64_t v, uint32_t pos, uint32_t width) {
    if (pos >= 64u || width == 0u) return 0ULL;
    uint64_t shifted = v >> pos;
    if (width >= 64u) return shifted;
    return shifted & ((1ULL << width) - 1ULL);
}

inline uint64_t bfext_s64(uint64_t v, uint32_t pos, uint32_t width) {
    if (pos >= 64u || width == 0u) return 0ULL;
    uint64_t u = bfext_u64(v, pos, width);
    if (width < 64u && (u & (1ULL << (width - 1u))) != 0ULL) {
        u |= ~((1ULL << width) - 1ULL);
    }
    return u;
}

inline uint64_t bfins_64(uint64_t base, uint64_t ins, uint32_t pos, uint32_t width) {
    if (pos >= 64u || width == 0u) return base;
    uint32_t w = std::min(width, 64u - pos);
    uint64_t mask = (w >= 64u) ? ~0ULL : (((1ULL << w) - 1ULL) << pos);
    return (base & ~mask) | ((ins << pos) & mask);
}

inline uint32_t eval_logic_op32(valhall::csf::CSLogicOp mode, uint32_t a, uint32_t b) {
    switch (mode) {
        case valhall::csf::CSLogicOp::CLEAR:    return 0u;
        case valhall::csf::CSLogicOp::AND:      return a & b;
        case valhall::csf::CSLogicOp::AND_A_NB: return a & ~b;
        case valhall::csf::CSLogicOp::MOV_A:    return a;
        case valhall::csf::CSLogicOp::AND_NA_B: return (~a) & b;
        case valhall::csf::CSLogicOp::MOV_B:    return b;
        case valhall::csf::CSLogicOp::XOR:      return a ^ b;
        case valhall::csf::CSLogicOp::OR:       return a | b;
        case valhall::csf::CSLogicOp::NOR:      return ~(a | b);
        case valhall::csf::CSLogicOp::XNOR:     return ~(a ^ b);
        case valhall::csf::CSLogicOp::NOT_B:    return ~b;
        case valhall::csf::CSLogicOp::OR_A_NB:  return a | ~b;
        case valhall::csf::CSLogicOp::NOT_A:    return ~a;
        case valhall::csf::CSLogicOp::OR_NA_B:  return (~a) | b;
        case valhall::csf::CSLogicOp::NAND:     return ~(a & b);
        case valhall::csf::CSLogicOp::SET:      return 0xFFFFFFFFu;
    }
    return a;
}

inline bool eval_branch_cond(valhall::csf::CSCondition cond, int32_t val) {
    switch (cond) {
        case valhall::csf::CSCondition::Lequal:  return val <= 0;
        case valhall::csf::CSCondition::Greater: return val > 0;
        case valhall::csf::CSCondition::Equal:   return val == 0;
        case valhall::csf::CSCondition::Nequal:  return val != 0;
        case valhall::csf::CSCondition::Less:    return val < 0;
        case valhall::csf::CSCondition::Gequal:  return val >= 0;
        case valhall::csf::CSCondition::Always:  return true;
    }
    return false;
}

struct StreamFrame {
    const uint64_t* words = nullptr;
    size_t count = 0;
    size_t pc_idx = 0;
};

} // namespace

size_t compute_user_io_insert_offset(const CsfQueueState& q, size_t page_size) {
    // Page 0 = doorbell, Page 1 = user input page (CS_INSERT), Page 2 = user output page.
    // On kbase < 1.35 (per-queue USER_IO), CS_INSERT is always at page_size + 0.
    // On kbase >= 1.35 (CSG-shared USER_IO), CS_INSERT is strided by csi_index * 16.
    const size_t csi_off = q.csg_shared_user_io
                               ? (static_cast<size_t>(q.csi_index) * kCsUserIoCsiStride135)
                               : 0u;
    return page_size + csi_off;
}

CsfExecResult execute_csf_stream(CsfQueueState& q,
                                 const VaTracker& va,
                                 const uint64_t* words,
                                 size_t count,
                                 const CsfDispatchCallbacks& cb,
                                 uint32_t max_steps) {
    CsfExecResult res{};
    if (!words || count == 0) return res;

    StreamFrame stack[kCsfMaxCallDepth + 1u];
    uint32_t sp = 0;
    stack[0] = StreamFrame{ words, count, 0u };

    while (true) {
        StreamFrame& cur = stack[sp];
        if (cur.pc_idx >= cur.count) {
            if (sp == 0) break;
            --sp;
            continue;
        }
        if (res.steps >= max_steps) {
            res.hit_step_limit = true;
            csf_log(LogLevel::Warn, "csf",
                    "Queue 0x%" PRIx64 " hit step limit (%u steps); stopping stream walk",
                    q.buffer_gpu_addr, max_steps);
            break;
        }

        const uint64_t insn = cur.words[cur.pc_idx];
        cur.pc_idx++;
        res.steps++;
        q.insns_executed++;

        const uint32_t op_raw = static_cast<uint32_t>((insn >> 56) & 0xFFu);
        const auto op = static_cast<valhall::csf::CSOpcode>(op_raw);

        switch (op_raw) {
            // --- No-ops, waits, resource/scoreboard/sync/heap control ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::NOP):
            case 0x80u: // ENOP / NOP_TRAP
            case static_cast<uint32_t>(valhall::csf::CSOpcode::WAIT):
            case 5u:    // RUN_TILING (v10)
            case static_cast<uint32_t>(valhall::csf::CSOpcode::RUN_FULLSCREEN):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::FINISH_TILING):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::FINISH_FRAGMENT):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SET_SB_ENTRY):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::PROGRESS_WAIT):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SET_EXCEPTION_HANDLER):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::NEXT_SB_ENTRY):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SET_STATE):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SET_STATE_IMM32):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SHARED_SB_INC):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SHARED_SB_DEC):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::REQ_RESOURCE):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::FLUSH_CACHE2):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SYNC_ADD32):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SYNC_SET32):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SYNC_WAIT32):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::ERROR_BARRIER):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::HEAP_SET):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::HEAP_OPERATION):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::TRACE_POINT):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SYNC_ADD64):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SYNC_SET64):
            case static_cast<uint32_t>(valhall::csf::CSOpcode::SYNC_WAIT64):
                break;

            // --- Immediate moves ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::MOVE48): {
                const auto m = valhall::csf::CSMOVE48::unpack_u64(insn);
                if (m.destination + 1u < kCsfMaxRegs) {
                    q.regs[m.destination] = static_cast<uint32_t>(m.immediate & 0xFFFFFFFFULL);
                    q.regs[m.destination + 1u] = static_cast<uint32_t>((m.immediate >> 32) & 0xFFFFu);
                } else {
                    q.fault_count++;
                }
                break;
            }
            case static_cast<uint32_t>(valhall::csf::CSOpcode::MOVE32): {
                const auto m = valhall::csf::CSMOVE32::unpack_u64(insn);
                if (m.destination < kCsfMaxRegs) {
                    q.regs[m.destination] = m.immediate;
                } else {
                    q.fault_count++;
                }
                break;
            }

            // --- ARITH_IMM32 (v10 ADD_IMMEDIATE32 when sub_op == 0, v13+ full sub-opcodes) ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::ADD_IMMEDIATE32): {
                const uint32_t imm32 = static_cast<uint32_t>(insn & 0xFFFFFFFFULL);
                const uint32_t sub_op = static_cast<uint32_t>((insn >> 32) & 0xFFu);
                const uint32_t src = static_cast<uint32_t>((insn >> 40) & 0xFFu);
                const uint32_t dst = static_cast<uint32_t>((insn >> 48) & 0xFFu);
                if (src >= kCsfMaxRegs || dst >= kCsfMaxRegs) {
                    q.fault_count++;
                    break;
                }
                const uint32_t sv = q.regs[src];
                uint32_t dv = sv;
                switch (sub_op) {
                    case 0u: // ADD_IMM32
                        dv = static_cast<uint32_t>(static_cast<int32_t>(sv) + static_cast<int32_t>(imm32));
                        break;
                    case 1u: // LSHIFT_IMM32
                        dv = sv << (imm32 & 31u);
                        break;
                    case 2u: // RSHIFT_IMM_U32
                        dv = sv >> (imm32 & 31u);
                        break;
                    case 3u: // RSHIFT_IMM_S32
                        dv = static_cast<uint32_t>(static_cast<int32_t>(sv) >> (imm32 & 31u));
                        break;
                    case 4u: // BFEXT_U32
                        dv = bfext_u32(sv, imm32 & 0xFFu, (imm32 >> 8) & 0xFFu);
                        break;
                    case 5u: // BFEXT_S32
                        dv = bfext_s32(sv, imm32 & 0xFFu, (imm32 >> 8) & 0xFFu);
                        break;
                    case 6u: // BFINS_IMM32
                        dv = bfins_32(sv, (imm32 >> 16) & 0xFFFFu, imm32 & 0xFFu, (imm32 >> 8) & 0xFFu);
                        break;
                    case 7u: // UMIN_IMM32
                        dv = std::min(sv, imm32);
                        break;
                    default:
                        q.unknown_opcodes++;
                        break;
                }
                q.regs[dst] = dv;
                break;
            }

            // --- ARITH_IMM64 (v10 ADD_IMMEDIATE64 when sub_op == 0, v13+ full sub-opcodes) ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::ADD_IMMEDIATE64): {
                const uint32_t imm32 = static_cast<uint32_t>(insn & 0xFFFFFFFFULL);
                const uint32_t sub_op = static_cast<uint32_t>((insn >> 32) & 0xFFu);
                const uint32_t src = static_cast<uint32_t>((insn >> 40) & 0xFFu);
                const uint32_t dst = static_cast<uint32_t>((insn >> 48) & 0xFFu);
                if (src + 1u >= kCsfMaxRegs || dst + 1u >= kCsfMaxRegs) {
                    q.fault_count++;
                    break;
                }
                const uint64_t sv = read_reg64(q, src);
                uint64_t dv = sv;
                switch (sub_op) {
                    case 0u: // ADD_IMM64
                        dv = static_cast<uint64_t>(static_cast<int64_t>(sv) +
                                                   static_cast<int64_t>(static_cast<int32_t>(imm32)));
                        break;
                    case 1u: // LSHIFT_IMM64
                        dv = sv << (imm32 & 63u);
                        break;
                    case 2u: // RSHIFT_IMM_U64
                        dv = sv >> (imm32 & 63u);
                        break;
                    case 3u: // RSHIFT_IMM_S64
                        dv = static_cast<uint64_t>(static_cast<int64_t>(sv) >> (imm32 & 63u));
                        break;
                    case 4u: // BFEXT_U64
                        dv = bfext_u64(sv, imm32 & 0xFFu, (imm32 >> 8) & 0xFFu);
                        break;
                    case 5u: // BFEXT_S64
                        dv = bfext_s64(sv, imm32 & 0xFFu, (imm32 >> 8) & 0xFFu);
                        break;
                    case 6u: // BFINS_IMM64
                        dv = bfins_64(sv, (imm32 >> 16) & 0xFFFFu, imm32 & 0xFFu, (imm32 >> 8) & 0xFFu);
                        break;
                    case 7u: // UMIN_IMM64
                        dv = std::min<uint64_t>(sv, static_cast<uint64_t>(imm32));
                        break;
                    default:
                        q.unknown_opcodes++;
                        break;
                }
                write_reg64(q, dst, dv);
                break;
            }

            // --- ARITH_REG32 (opcode 18: v10 UMIN32 when sub_op == 0, v13+ full sub-opcodes) ---
            case static_cast<uint32_t>(valhall::csf::V13CSOpcode::ARITH_REG32): {
                const uint32_t sub_op = static_cast<uint32_t>((insn >> 24) & 0xFFu);
                const uint32_t src1 = static_cast<uint32_t>((insn >> 32) & 0xFFu);
                const uint32_t src0 = static_cast<uint32_t>((insn >> 40) & 0xFFu);
                const uint32_t dst = static_cast<uint32_t>((insn >> 48) & 0xFFu);
                if (src0 >= kCsfMaxRegs || src1 >= kCsfMaxRegs || dst >= kCsfMaxRegs) {
                    q.fault_count++;
                    break;
                }
                const uint32_t a = q.regs[src0];
                const uint32_t b = q.regs[src1];
                uint32_t dv = a;
                switch (sub_op) {
                    case 0u: dv = std::min(a, b); break; // UMIN32
                    case 1u: dv = a + b; break;          // ADD32
                    case 2u: dv = a - b; break;          // SUB32
                    case 3u: dv = a << (b & 31u); break; // LSHIFT32
                    case 4u: dv = a >> (b & 31u); break; // RSHIFT_U32
                    case 5u: dv = static_cast<uint32_t>(static_cast<int32_t>(a) >> (b & 31u)); break; // RSHIFT_S32
                    case 6u: // BFINS32
                        dv = bfins_32(a, b, static_cast<uint32_t>(insn & 0xFFu),
                                      static_cast<uint32_t>((insn >> 8) & 0xFFu));
                        break;
                    default:
                        q.unknown_opcodes++;
                        break;
                }
                q.regs[dst] = dv;
                break;
            }

            // --- ARITH_REG64 (opcode 57) ---
            case static_cast<uint32_t>(valhall::csf::V13CSOpcode::ARITH_REG64): {
                const uint32_t sub_op = static_cast<uint32_t>((insn >> 24) & 0xFFu);
                const uint32_t src1 = static_cast<uint32_t>((insn >> 32) & 0xFFu);
                const uint32_t src0 = static_cast<uint32_t>((insn >> 40) & 0xFFu);
                const uint32_t dst = static_cast<uint32_t>((insn >> 48) & 0xFFu);
                if (src0 + 1u >= kCsfMaxRegs || src1 + 1u >= kCsfMaxRegs || dst + 1u >= kCsfMaxRegs) {
                    q.fault_count++;
                    break;
                }
                const uint64_t a = read_reg64(q, src0);
                const uint64_t b = read_reg64(q, src1);
                uint64_t dv = a;
                switch (sub_op) {
                    case 0u: dv = std::min(a, b); break; // UMIN64
                    case 1u: dv = a + b; break;          // ADD64
                    case 2u: dv = a - b; break;          // SUB64
                    case 3u: dv = a << (b & 63u); break; // LSHIFT64
                    case 4u: dv = a >> (b & 63u); break; // RSHIFT_U64
                    case 5u: dv = static_cast<uint64_t>(static_cast<int64_t>(a) >> (b & 63u)); break; // RSHIFT_S64
                    case 6u: // BFINS64
                        dv = bfins_64(a, b, static_cast<uint32_t>(insn & 0xFFu),
                                      static_cast<uint32_t>((insn >> 8) & 0xFFu));
                        break;
                    default:
                        q.unknown_opcodes++;
                        break;
                }
                write_reg64(q, dst, dv);
                break;
            }

            // --- LOGIC_OP32 (opcode 19) ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::LOGIC_OP32): {
                const auto l = valhall::csf::CSLOGICOP32::unpack_u64(insn);
                if (l.source_0 >= kCsfMaxRegs || l.source_1 >= kCsfMaxRegs || l.destination >= kCsfMaxRegs) {
                    q.fault_count++;
                    break;
                }
                const uint32_t a = q.regs[l.source_0];
                const uint32_t b = (l.index == valhall::csf::CSLogicOpIndex::Index)
                                       ? (1u << (q.regs[l.source_1] & 31u))
                                       : q.regs[l.source_1];
                q.regs[l.destination] = eval_logic_op32(l.mode, a, b);
                break;
            }

            // --- LOAD_MULTIPLE (opcode 20) ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::LOAD_MULTIPLE): {
                const auto lm = valhall::csf::CSLOADMULTIPLE::unpack_u64(insn);
                if (lm.address + 1u >= kCsfMaxRegs) {
                    q.fault_count++;
                    break;
                }
                const uint64_t base_va = static_cast<uint64_t>(
                    static_cast<int64_t>(read_reg64(q, lm.address)) + static_cast<int64_t>(lm.offset));
                for (uint32_t b = 0; b < 16u; ++b) {
                    if ((lm.mask & (1u << b)) == 0u) continue;
                    const uint32_t reg_idx = lm.base_register + b;
                    if (reg_idx >= kCsfMaxRegs) {
                        q.fault_count++;
                        break;
                    }
                    const uint64_t word_va = (base_va + static_cast<uint64_t>(b) * 4ULL) & 0x0000FFFFFFFFFFFFULL;
                    auto sit = q.shadow_mem.find(word_va);
                    if (sit != q.shadow_mem.end()) {
                        q.regs[reg_idx] = sit->second;
                    } else if (CpuSpan span = va.gpu_to_cpu_span(word_va, 4u)) {
                        uint32_t val = 0;
                        std::memcpy(&val, span.ptr, sizeof(val));
                        q.regs[reg_idx] = val;
                    } else {
                        csf_log(LogLevel::Debug, "csf",
                                "LOAD_MULTIPLE unmapped VA 0x%" PRIx64 " -> r%u", word_va, reg_idx);
                    }
                }
                break;
            }

            // --- STORE_MULTIPLE (opcode 21, non-destructive per-queue shadow memory) ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::STORE_MULTIPLE): {
                const auto sm = valhall::csf::CSSTOREMULTIPLE::unpack_u64(insn);
                if (sm.address + 1u >= kCsfMaxRegs) {
                    q.fault_count++;
                    break;
                }
                const uint64_t base_va = static_cast<uint64_t>(
                    static_cast<int64_t>(read_reg64(q, sm.address)) + static_cast<int64_t>(sm.offset));
                for (uint32_t b = 0; b < 16u; ++b) {
                    if ((sm.mask & (1u << b)) == 0u) continue;
                    const uint32_t reg_idx = sm.base_register + b;
                    if (reg_idx >= kCsfMaxRegs) {
                        q.fault_count++;
                        break;
                    }
                    const uint64_t word_va = (base_va + static_cast<uint64_t>(b) * 4ULL) & 0x0000FFFFFFFFFFFFULL;
                    q.shadow_mem[word_va] = q.regs[reg_idx];
                }
                break;
            }

            // --- BRANCH (opcode 22) ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::BRANCH): {
                const auto br = valhall::csf::CSBRANCH::unpack_u64(insn);
                if (br.value >= kCsfMaxRegs) {
                    q.fault_count++;
                    break;
                }
                const int32_t v = static_cast<int32_t>(q.regs[br.value]);
                if (eval_branch_cond(br.condition, v)) {
                    q.branches_taken++;
                    // cur.pc_idx already points to next instruction (pc + 1).
                    // Branch offset is signed count of 64-bit instructions relative to next instruction.
                    const int64_t target = static_cast<int64_t>(cur.pc_idx) + static_cast<int64_t>(br.offset);
                    if (target < 0 || static_cast<size_t>(target) > cur.count) {
                        q.fault_count++;
                        cur.pc_idx = cur.count;
                    } else {
                        cur.pc_idx = static_cast<size_t>(target);
                    }
                }
                break;
            }

            // --- CALL (opcode 32, adapted from lifted_csffw.c) ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::CALL): {
                const auto c = valhall::csf::CSCALL::unpack_u64(insn);
                if (c.address + 1u >= kCsfMaxRegs || c.length >= kCsfMaxRegs || sp >= kCsfMaxCallDepth) {
                    q.fault_count++;
                    break;
                }
                const uint64_t sub_va = read_reg64(q, c.address) & ~7ULL;
                const uint32_t sub_bytes = q.regs[c.length] & ~7u;
                if (sub_va != 0 && sub_bytes >= 8u && sub_bytes <= (4u << 20)) {
                    if (CpuSpan span = va.gpu_to_cpu_span(sub_va, sub_bytes)) {
                        q.calls_executed++;
                        stack[++sp] = StreamFrame{
                            reinterpret_cast<const uint64_t*>(span.ptr),
                            sub_bytes / 8u,
                            0u,
                        };
                    } else {
                        csf_log(LogLevel::Warn, "csf",
                                "CALL target VA 0x%" PRIx64 " (len=%u) unmapped", sub_va, sub_bytes);
                    }
                }
                break;
            }

            // --- JUMP (opcode 33, tail branch adapted from lifted_csffw.c) ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::JUMP): {
                const auto j = valhall::csf::CSJUMP::unpack_u64(insn);
                if (j.address + 1u >= kCsfMaxRegs || j.length >= kCsfMaxRegs) {
                    q.fault_count++;
                    break;
                }
                const uint64_t target_va = read_reg64(q, j.address) & ~7ULL;
                const uint32_t target_bytes = q.regs[j.length] & ~7u;
                if (target_va != 0 && target_bytes >= 8u && target_bytes <= (4u << 20)) {
                    if (CpuSpan span = va.gpu_to_cpu_span(target_va, target_bytes)) {
                        q.jumps_executed++;
                        cur = StreamFrame{
                            reinterpret_cast<const uint64_t*>(span.ptr),
                            target_bytes / 8u,
                            0u,
                        };
                    } else {
                        csf_log(LogLevel::Warn, "csf",
                                "JUMP target VA 0x%" PRIx64 " (len=%u) unmapped", target_va, target_bytes);
                        cur.pc_idx = cur.count;
                    }
                } else {
                    cur.pc_idx = cur.count;
                }
                break;
            }

            // --- STORE_STATE / PROT_REGION / PROGRESS_STORE / PROGRESS_LOAD (lifted_csffw.c) ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::STORE_STATE): {
                const auto ss = valhall::csf::CSSTORESTATE::unpack_u64(insn);
                if (ss.address + 1u < kCsfMaxRegs) {
                    const int16_t delta = static_cast<int16_t>(static_cast<uint16_t>(ss.offset & 0xFFF8));
                    uint64_t ptr = read_reg64(q, ss.address);
                    ptr = static_cast<uint64_t>(static_cast<int64_t>(ptr) + static_cast<int64_t>(delta));
                    write_reg64(q, ss.address, ptr);
                }
                break;
            }
            case static_cast<uint32_t>(valhall::csf::CSOpcode::PROT_REGION): {
                const auto pr = valhall::csf::CSPROTREGION::unpack_u64(insn);
                const size_t skip_insns = static_cast<size_t>(pr.size) / 8u;
                cur.pc_idx = std::min(cur.count, cur.pc_idx + skip_insns);
                break;
            }
            case static_cast<uint32_t>(valhall::csf::CSOpcode::PROGRESS_STORE): {
                const auto ps = valhall::csf::CSPROGRESSSTORE::unpack_u64(insn);
                if (ps.source + 1u < kCsfMaxRegs) {
                    q.iter_state64 = read_reg64(q, ps.source);
                }
                break;
            }
            case static_cast<uint32_t>(valhall::csf::CSOpcode::PROGRESS_LOAD): {
                const auto pl = valhall::csf::CSPROGRESSLOAD::unpack_u64(insn);
                if (pl.destination + 1u < kCsfMaxRegs) {
                    write_reg64(q, pl.destination, q.iter_state64);
                }
                break;
            }

            // --- Graphics & Compute Dispatch Opcodes ---
            case static_cast<uint32_t>(valhall::csf::CSOpcode::RUN_IDVS): {
                if (cb.on_run_idvs) cb.on_run_idvs(q, insn, /*is_idvs2=*/false);
                break;
            }
            case static_cast<uint32_t>(valhall::csf::V13CSOpcode::RUN_IDVS2): {
                if (cb.on_run_idvs) cb.on_run_idvs(q, insn, /*is_idvs2=*/true);
                break;
            }
            case static_cast<uint32_t>(valhall::csf::CSOpcode::RUN_FRAGMENT): {
                if (cb.on_run_fragment) {
                    cb.on_run_fragment(q, insn,
                                       static_cast<uint32_t>(valhall::csf::FRAGMENTSR::FBD_POINTER),
                                       /*is_frag2=*/false);
                }
                break;
            }
            case static_cast<uint32_t>(valhall::csf::V14CSOpcode::RUN_FRAGMENT2): {
                if (cb.on_run_fragment) {
                    cb.on_run_fragment(q, insn,
                                       static_cast<uint32_t>(valhall::csf::V14FRAGMENTSR::FBD_POINTER),
                                       /*is_frag2=*/true);
                }
                break;
            }
            case static_cast<uint32_t>(valhall::csf::CSOpcode::RUN_COMPUTE): {
                if (cb.on_run_compute) cb.on_run_compute(q, insn, /*is_indirect=*/false);
                break;
            }
            case static_cast<uint32_t>(valhall::csf::CSOpcode::RUN_COMPUTE_INDIRECT): {
                if (cb.on_run_compute) cb.on_run_compute(q, insn, /*is_indirect=*/true);
                break;
            }

            default:
                (void)op;
                q.unknown_opcodes++;
                csf_log(LogLevel::Debug, "csf",
                        "Unemulated CSF opcode %u (0x%02x) insn=0x%016" PRIx64,
                        op_raw, op_raw, insn);
                break;
        }
    }

    res.faulted = (q.fault_count > 0);
    return res;
}

CsfExecResult process_queue_ring(CsfQueueState& q,
                                 const VaTracker& va,
                                 uint64_t insert_off,
                                 const CsfDispatchCallbacks& cb,
                                 uint32_t max_steps) {
    CsfExecResult total{};
    if (q.buffer_size == 0 || insert_off <= q.last_extract) return total;

    CpuSpan ring_span = va.gpu_to_cpu_span(q.buffer_gpu_addr, q.buffer_size);
    if (!ring_span) {
        csf_log(LogLevel::Warn, "csf",
                "Queue ring buffer VA 0x%" PRIx64 " (size=%u) is unmapped",
                q.buffer_gpu_addr, q.buffer_size);
        return total;
    }

    const uint8_t* ring = reinterpret_cast<const uint8_t*>(ring_span.ptr);
    uint64_t cur = q.last_extract;
    while (cur < insert_off && total.steps < max_steps) {
        const size_t off = static_cast<size_t>(cur % q.buffer_size);
        const size_t avail = std::min<size_t>(q.buffer_size - off,
                                              static_cast<size_t>(insert_off - cur));
        if (avail < 8u) break;
        const size_t word_cnt = avail / 8u;
        CsfExecResult slice = execute_csf_stream(
            q, va, reinterpret_cast<const uint64_t*>(ring + off),
            word_cnt, cb, max_steps - total.steps);
        total.steps += slice.steps;
        total.hit_step_limit |= slice.hit_step_limit;
        total.faulted |= slice.faulted;
        cur += word_cnt * 8u;
    }
    q.last_extract = insert_off;
    return total;
}

} // namespace mali_csf_dump
