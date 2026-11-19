#include "csf_dumper.h"
#include "valhall_csf_descriptors_generated.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

namespace mali_csf_dump {

namespace {

constexpr uint64_t kVa48Mask = 0x0000FFFFFFFFFFFFULL;

void write_bin_file(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (f && data && size > 0) {
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
}

void ensure_dir(const std::string& dir) {
    std::string cur;
    for (size_t i = 0; i < dir.size(); ++i) {
        cur.push_back(dir[i]);
        if (dir[i] == '/' || i + 1 == dir.size()) {
            if (!cur.empty() && cur != "/") {
                ::mkdir(cur.c_str(), 0755);
            }
        }
    }
}

void dump_sr_u64(const std::string& path, const uint32_t* sr64) {
    uint64_t words[64] = {0};
    for (size_t i = 0; i < 64u; ++i) {
        words[i] = static_cast<uint64_t>(sr64[i]);
    }
    write_bin_file(path, words, sizeof(words));
}

inline uint64_t rd64_sr(const uint32_t* sr, uint32_t idx) {
    const uint32_t aligned = idx & 62u;
    return static_cast<uint64_t>(sr[aligned]) |
           (static_cast<uint64_t>(sr[aligned + 1u]) << 32);
}

// Dump a descriptor region up to `preferred_bytes`, safely clamped to available VA bytes
// (succeeds even for tight sub-allocations smaller than `preferred_bytes` as long as >= `min_bytes`).
size_t dump_va_clamped(const VaTracker& va,
                       uint64_t gpu_va,
                       size_t min_bytes,
                       size_t preferred_bytes,
                       const std::string& path) {
    CpuSpan span = va.gpu_to_cpu_span(gpu_va, min_bytes);
    if (!span) return 0u;
    const size_t n = std::min(span.available_bytes, preferred_bytes);
    write_bin_file(path, span.ptr, n);
    return n;
}

void dump_srt_and_buffers(const VaTracker& va,
                          const std::string& out_dir,
                          uint64_t raw_srt,
                          const char* srt_filename,
                          bool dump_resources) {
    const uint64_t srt_va = raw_srt & kVa48Mask & ~0x3FULL;
    if (srt_va == 0) return;

    uint32_t srt_count = static_cast<uint32_t>(raw_srt & 0x3FULL);
    const size_t srt_bytes = (srt_count > 0) ? (static_cast<size_t>(srt_count) * 16u) : 64u;
    if (srt_count == 0) srt_count = 4u;

    CpuSpan span = va.gpu_to_cpu_span(srt_va, 16u);
    if (!span) {
        csf_log(LogLevel::Debug, "dump", "SRT VA 0x%" PRIx64 " unmapped (%s)", srt_va, srt_filename);
        return;
    }

    const size_t dump_bytes = std::min(span.available_bytes, srt_bytes);
    write_bin_file(out_dir + "/" + srt_filename, span.ptr, dump_bytes);

    if (!dump_resources) return;
    const uint32_t actual_entries = std::min<uint32_t>(srt_count, static_cast<uint32_t>(dump_bytes / 16u));
    const uint32_t* srt_words = reinterpret_cast<const uint32_t*>(span.ptr);
    uint32_t buf_idx = 0;
    for (uint32_t r = 0; r < actual_entries; ++r) {
        const valhall::csf::Resource res = valhall::csf::Resource::unpack(srt_words + r * 4u);
        const uint64_t tbl_va = res.address & kVa48Mask;
        if (tbl_va >= 0x10000ULL && res.size > 0) {
            CpuSpan tp = va.gpu_to_cpu_span(tbl_va, 4u);
            if (tp) {
                const size_t n = std::min(tp.available_bytes, static_cast<size_t>(res.size));
                char name[64];
                std::snprintf(name, sizeof(name), "/srt_buf%u.bin", buf_idx++);
                write_bin_file(out_dir + name, tp.ptr, n);
            }
        }
    }
}

size_t dump_spd_and_shader(const VaTracker& va,
                           const std::string& out_dir,
                           uint64_t spd_va,
                           size_t max_shader_bytes,
                           const char* spd_name,
                           const char* bin_name) {
    if (spd_va == 0) return 0u;
    CpuSpan spd_span = va.gpu_to_cpu_span(spd_va, 32u);
    if (!spd_span) return 0u;

    write_bin_file(out_dir + "/" + spd_name, spd_span.ptr, 32u);
    const valhall::csf::ShaderProgram spd =
        valhall::csf::ShaderProgram::unpack(reinterpret_cast<const uint32_t*>(spd_span.ptr));
    const uint64_t code_va = spd.binary & kVa48Mask;
    if (code_va == 0) return 0u;

    CpuSpan code_span = va.gpu_to_cpu_span(code_va, 8u);
    if (!code_span) return 0u;

    const size_t code_bytes = estimate_valhall_shader_size(
        code_span.ptr, code_span.available_bytes, max_shader_bytes);
    write_bin_file(out_dir + "/" + bin_name, code_span.ptr, code_bytes);
    return code_bytes;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        if (static_cast<unsigned char>(c) >= 0x20) out.push_back(c);
    }
    return out;
}

} // namespace

size_t estimate_valhall_shader_size(const void* code_ptr,
                                    size_t available_bytes,
                                    size_t max_shader_bytes) {
    if (!code_ptr || available_bytes < 8u) return available_bytes;
    const size_t limit = std::min(available_bytes, std::max<size_t>(max_shader_bytes, 64u));
    const size_t max_words = limit / 8u;
    const uint64_t* words = reinterpret_cast<const uint64_t*>(code_ptr);

    // Scan for Valhall end-of-shader flow control: bits[62:59] == 15 (0xF).
    // Note: shaders can contain multiple branches/ends or trailing PC-relative LUTs,
    // so if an END word is followed immediately by non-zero non-NOP instructions or
    // padding up to a 64-byte alignment, we include the last END + 64-byte alignment.
    size_t last_end_word = 0;
    bool found_end = false;
    size_t zero_run = 0;
    for (size_t i = 0; i < max_words; ++i) {
        const uint64_t w = words[i];
        if (w == 0ULL) {
            zero_run++;
            // Two consecutive zero 64-bit words after an END signals end of shader + LUT
            if (found_end && zero_run >= 2u) {
                break;
            }
            continue;
        }
        zero_run = 0;
        const uint32_t flow = static_cast<uint32_t>((w >> 59) & 0xFu);
        if (flow == 15u) {
            found_end = true;
            last_end_word = i;
        }
    }

    if (found_end) {
        // Round up past last_end_word to 64-byte boundary (capturing short trailing LUTs),
        // bounded by limit.
        size_t end_bytes = (last_end_word + 1u) * 8u;
        size_t aligned = (end_bytes + 63u) & ~static_cast<size_t>(63u);
        return std::min(limit, std::max<size_t>(aligned, end_bytes));
    }

    // Fallback when no END encoding is found: dump up to min(limit, 4096) bytes
    return std::min<size_t>(limit, 4096u);
}

size_t compute_rt_writeback_bytes(uint32_t fb_width,
                                  uint32_t fb_height,
                                  uint32_t row_stride,
                                  uint32_t surface_stride,
                                  size_t available_bytes) {
    size_t desired = 0;
    if (surface_stride > 0) {
        desired = static_cast<size_t>(surface_stride);
    } else if (row_stride > 0 && fb_height > 0) {
        desired = static_cast<size_t>(row_stride) * static_cast<size_t>(fb_height);
    } else if (fb_width > 0 && fb_height > 0) {
        // Conservative RGBA8 fallback if descriptor strides are zero
        desired = static_cast<size_t>(fb_width) * static_cast<size_t>(fb_height) * 4u;
    }
    if (available_bytes > 0) {
        return (desired > 0) ? std::min(desired, available_bytes) : available_bytes;
    }
    return desired;
}

void CsfDumper::reset() {
    // Stateless helper; FrameCaptureState holds per-frame data.
}

void CsfDumper::ensure_frame_dir(FrameCaptureState& cap, const LayerConfig& cfg, uint32_t current_frame) {
    if (!cap.active) {
        cap.active = true;
        cap.frame_number = current_frame;
        cap.out_dir = cfg.dump_dir;
        ensure_dir(cap.out_dir);
    }
}

void CsfDumper::on_run_idvs(FrameCaptureState& cap,
                            const CsfQueueState& q,
                            const VaTracker& va,
                            const LayerConfig& cfg,
                            uint32_t current_frame,
                            uint64_t insn,
                            bool is_idvs2) {
    if (cfg.max_draws > 0 && cap.draws.size() >= cfg.max_draws) {
        return;
    }
    ensure_frame_dir(cap, cfg, current_frame);

    const uint32_t draw_idx = static_cast<uint32_t>(cap.draws.size());
    char rel[64];
    std::snprintf(rel, sizeof(rel), "draws/draw_%03u", draw_idx);
    const std::string draw_dir = cap.out_dir + "/" + rel;
    ensure_dir(draw_dir);

    auto dump_into = [&](const std::string& target_dir, CapturedDrawSummary* out_sum) {
        dump_sr_u64(target_dir + "/idvs_sr.bin", q.regs);
        write_bin_file(target_dir + "/csrunidvs_insn.bin", &insn, sizeof(insn));

        uint64_t raw_pos_srt = 0, raw_frag_srt = 0, raw_fau = 0, raw_frag_fau = 0;
        uint64_t pos_spd = 0, var_spd = 0, frag_spd = 0;
        uint64_t tsd_va = 0, tcd_va = 0, zsd_va = 0, bd_va = 0;

        if (is_idvs2) {
            raw_pos_srt  = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::V13IDVSSR::VERTEX_SRT));
            raw_frag_srt = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::V13IDVSSR::FRAGMENT_SRT));
            raw_fau      = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::V13IDVSSR::VERTEX_FAU));
            raw_frag_fau = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::V13IDVSSR::FRAGMENT_FAU));
            pos_spd      = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::V13IDVSSR::VERTEX_SPD)) & kVa48Mask;
            frag_spd     = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::V13IDVSSR::FRAGMENT_SPD)) & kVa48Mask;
            tsd_va       = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::V13IDVSSR::VERTEX_TSD)) & kVa48Mask;
            tcd_va       = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::V13IDVSSR::TILER_CTX)) & kVa48Mask;
            zsd_va       = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::V13IDVSSR::ZSD)) & kVa48Mask;
            bd_va        = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::V13IDVSSR::BLEND_DESC)) & kVa48Mask & ~0xFULL;
        } else {
            raw_pos_srt  = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::IDVSSR::SRT_0));
            raw_frag_srt = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::IDVSSR::SRT_2));
            raw_fau      = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::IDVSSR::FAU_0));
            pos_spd      = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::IDVSSR::SPD_0)) & kVa48Mask;
            var_spd      = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::IDVSSR::SPD_1)) & kVa48Mask;
            frag_spd     = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::IDVSSR::SPD_2)) & kVa48Mask;
            tsd_va       = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::IDVSSR::TSD_0)) & kVa48Mask;
            tcd_va       = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::IDVSSR::TILER_CTX)) & kVa48Mask;
            zsd_va       = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::IDVSSR::ZSD)) & kVa48Mask;
            bd_va        = rd64_sr(q.regs, static_cast<uint32_t>(valhall::csf::IDVSSR::BLEND_DESC)) & kVa48Mask & ~0xFULL;
        }

        dump_srt_and_buffers(va, target_dir, raw_pos_srt, "pos_srt.bin", /*dump_resources=*/true);
        dump_srt_and_buffers(va, target_dir, raw_frag_srt, "frag_srt.bin", /*dump_resources=*/false);

        const uint64_t fau_va = raw_fau & kVa48Mask;
        const size_t fau_bytes = std::max<size_t>(8u, static_cast<size_t>((raw_fau >> 56) & 0xFFu) * 8u);
        dump_va_clamped(va, fau_va, 8u, fau_bytes, target_dir + "/fau.bin");

        if (is_idvs2 && raw_frag_fau != 0) {
            const uint64_t frag_fau_va = raw_frag_fau & kVa48Mask;
            const size_t frag_fau_bytes = std::max<size_t>(8u, static_cast<size_t>((raw_frag_fau >> 56) & 0xFFu) * 8u);
            dump_va_clamped(va, frag_fau_va, 8u, frag_fau_bytes, target_dir + "/frag_fau.bin");
        }

        const size_t pos_sz = dump_spd_and_shader(va, target_dir, pos_spd, cfg.max_shader_bytes, "pos_spd.bin", "pos_shader.bin");
        const size_t var_sz = var_spd ? dump_spd_and_shader(va, target_dir, var_spd, cfg.max_shader_bytes, "var_spd.bin", "var_shader.bin") : 0u;
        const size_t frag_sz = dump_spd_and_shader(va, target_dir, frag_spd, cfg.max_shader_bytes, "frag_spd.bin", "frag_shader.bin");

        dump_va_clamped(va, tsd_va, 32u, 512u, target_dir + "/tsd.bin");
        dump_va_clamped(va, tcd_va, 64u, 256u, target_dir + "/tcd.bin");
        dump_va_clamped(va, zsd_va, 32u, 64u,  target_dir + "/zsd.bin");
        dump_va_clamped(va, bd_va,  16u, 64u,  target_dir + "/bd.bin");

        if (out_sum) {
            out_sum->pos_srt_va = raw_pos_srt & kVa48Mask & ~0x3FULL;
            out_sum->frag_srt_va = raw_frag_srt & kVa48Mask & ~0x3FULL;
            out_sum->fau_va = fau_va;
            out_sum->pos_shader_bytes = pos_sz;
            out_sum->var_shader_bytes = var_sz;
            out_sum->frag_shader_bytes = frag_sz;
        }
    };

    CapturedDrawSummary sum{};
    sum.draw_index = draw_idx;
    sum.is_idvs2 = is_idvs2;
    sum.queue_gpu_va = q.buffer_gpu_addr;
    sum.run_insn = insn;
    sum.sub_dir = rel;
    dump_into(draw_dir, &sum);

    // Mirror draw_000 into root out_dir for vkcube_frame1 backward compatibility
    if (!cap.has_primary_draw) {
        cap.has_primary_draw = true;
        cap.primary_is_idvs2 = is_idvs2;
        std::memcpy(cap.primary_idvs_sr, q.regs, sizeof(cap.primary_idvs_sr));
        dump_into(cap.out_dir, nullptr);
    }

    cap.draws.push_back(sum);
    csf_log(LogLevel::Info, "dump",
            "Captured %s #%u (queue=0x%" PRIx64 ", pos=%zuB, frag=%zuB)",
            is_idvs2 ? "RUN_IDVS2" : "RUN_IDVS", draw_idx,
            q.buffer_gpu_addr, sum.pos_shader_bytes, sum.frag_shader_bytes);
}

void CsfDumper::on_run_fragment(FrameCaptureState& cap,
                                const CsfQueueState& q,
                                const VaTracker& va,
                                const LayerConfig& cfg,
                                uint32_t current_frame,
                                uint64_t /*insn*/,
                                uint32_t fbd_reg_idx,
                                bool is_frag2) {
    if (cfg.max_draws > 0 && cap.fragments.size() >= cfg.max_draws) {
        return;
    }
    ensure_frame_dir(cap, cfg, current_frame);

    const uint32_t frag_idx = static_cast<uint32_t>(cap.fragments.size());
    char rel[64];
    std::snprintf(rel, sizeof(rel), "fragments/frag_%03u", frag_idx);
    const std::string frag_dir = cap.out_dir + "/" + rel;
    ensure_dir(frag_dir);

    CapturedFragmentSummary sum{};
    sum.frag_index = frag_idx;
    sum.is_frag2 = is_frag2;
    sum.queue_gpu_va = q.buffer_gpu_addr;
    sum.sub_dir = rel;

    uint32_t fbd_words[2] = { q.regs[fbd_reg_idx], q.regs[fbd_reg_idx + 1u] };
    const valhall::csf::FramebufferPointer fbp = valhall::csf::FramebufferPointer::unpack(fbd_words);
    sum.fbd_va = fbp.pointer & kVa48Mask;

    auto dump_into = [&](const std::string& target_dir, bool parse_summary) {
        dump_sr_u64(target_dir + "/frag_sr.bin", q.regs);
        if (CpuSpan span = va.gpu_to_cpu_span(sum.fbd_va, 128u)) {
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(span.ptr);
            const valhall::csf::FramebufferParameters fbd =
                valhall::csf::FramebufferParameters::unpack(reinterpret_cast<const uint32_t*>(ptr));
            const uint32_t rt_cnt = std::max<uint32_t>(1u, fbd.render_target_count);
            const size_t rt_offset = 128u + (fbp.zs_crc_extension_present ? 64u : 0u);
            const size_t exact_fbd_bytes = rt_offset + static_cast<size_t>(rt_cnt) * 64u;
            const size_t dump_bytes = std::min(span.available_bytes, exact_fbd_bytes);
            write_bin_file(target_dir + "/fbd.bin", span.ptr, dump_bytes);

            if (parse_summary) {
                sum.fbd_bytes = dump_bytes;
                sum.fb_width = fbd.width;
                sum.fb_height = fbd.height;
                sum.render_target_count = rt_cnt;
                if (rt_offset + 64u <= span.available_bytes) {
                    const valhall::csf::RenderTarget rtd =
                        valhall::csf::RenderTarget::unpack(reinterpret_cast<const uint32_t*>(ptr + rt_offset));
                    sum.rt_color_va = rtd.rgb.writeback_buffer.base & kVa48Mask;
                    sum.rt_row_stride = rtd.rgb.writeback_buffer.row_stride;
                    sum.rt_surface_stride = rtd.rgb.writeback_buffer.surface_stride;
                    CpuSpan rt_span = va.gpu_to_cpu_span(sum.rt_color_va, 4u);
                    sum.rt_bytes = compute_rt_writeback_bytes(
                        sum.fb_width, sum.fb_height, sum.rt_row_stride, sum.rt_surface_stride,
                        rt_span ? rt_span.available_bytes : 0u);
                }
            }
        }
    };

    dump_into(frag_dir, /*parse_summary=*/true);
    if (!cap.has_primary_frag) {
        cap.has_primary_frag = true;
        cap.primary_rt_color_va = sum.rt_color_va;
        cap.primary_fb_width = sum.fb_width;
        cap.primary_fb_height = sum.fb_height;
        cap.primary_rt_bytes = sum.rt_bytes;
        dump_into(cap.out_dir, /*parse_summary=*/false);
    }

    cap.fragments.push_back(sum);
    csf_log(LogLevel::Info, "dump",
            "Captured %s #%u (fb=%ux%u, fbd=%zuB, rt_va=0x%" PRIx64 ", rt_bytes=%zu)",
            is_frag2 ? "RUN_FRAGMENT2" : "RUN_FRAGMENT", frag_idx,
            sum.fb_width, sum.fb_height, sum.fbd_bytes, sum.rt_color_va, sum.rt_bytes);
}

void CsfDumper::on_run_compute(FrameCaptureState& cap,
                               const CsfQueueState& q,
                               const VaTracker& va,
                               const LayerConfig& cfg,
                               uint32_t current_frame,
                               uint64_t insn,
                               bool is_indirect) {
    if (cfg.max_draws > 0 && cap.computes.size() >= cfg.max_draws) {
        return;
    }
    ensure_frame_dir(cap, cfg, current_frame);

    const uint32_t comp_idx = static_cast<uint32_t>(cap.computes.size());
    char rel[64];
    std::snprintf(rel, sizeof(rel), "computes/compute_%03u", comp_idx);
    const std::string comp_dir = cap.out_dir + "/" + rel;
    ensure_dir(comp_dir);

    const valhall::csf::CSRUNCOMPUTE rc = valhall::csf::CSRUNCOMPUTE::unpack_u64(insn);
    const uint32_t srt_reg = static_cast<uint32_t>(valhall::csf::COMPUTESR::SRT_0) + rc.srt_select * 2u;
    const uint32_t fau_reg = static_cast<uint32_t>(valhall::csf::COMPUTESR::FAU_0) + rc.fau_select * 2u;
    const uint32_t spd_reg = static_cast<uint32_t>(valhall::csf::COMPUTESR::SPD_0) + rc.spd_select * 2u;
    const uint32_t tsd_reg = static_cast<uint32_t>(valhall::csf::COMPUTESR::TSD_0) + rc.tsd_select * 2u;

    dump_sr_u64(comp_dir + "/comp_sr.bin", q.regs);
    write_bin_file(comp_dir + "/csruncompute_insn.bin", &insn, sizeof(insn));

    const uint64_t raw_srt = rd64_sr(q.regs, srt_reg);
    const uint64_t raw_fau = rd64_sr(q.regs, fau_reg);
    const uint64_t spd_va  = rd64_sr(q.regs, spd_reg) & kVa48Mask;
    const uint64_t tsd_va  = rd64_sr(q.regs, tsd_reg) & kVa48Mask;

    dump_srt_and_buffers(va, comp_dir, raw_srt, "comp_srt.bin", /*dump_resources=*/true);

    const uint64_t fau_va = raw_fau & kVa48Mask;
    const size_t fau_bytes = std::max<size_t>(8u, static_cast<size_t>((raw_fau >> 56) & 0xFFu) * 8u);
    dump_va_clamped(va, fau_va, 8u, fau_bytes, comp_dir + "/fau.bin");

    const size_t shader_sz = dump_spd_and_shader(
        va, comp_dir, spd_va, cfg.max_shader_bytes, "comp_spd.bin", "comp_shader.bin");
    dump_va_clamped(va, tsd_va, 32u, 512u, comp_dir + "/tsd.bin");

    CapturedComputeSummary sum{};
    sum.compute_index = comp_idx;
    sum.is_indirect = is_indirect;
    sum.queue_gpu_va = q.buffer_gpu_addr;
    sum.run_insn = insn;
    sum.srt_va = raw_srt & kVa48Mask & ~0x3FULL;
    sum.fau_va = fau_va;
    sum.spd_va = spd_va;
    sum.tsd_va = tsd_va;
    sum.shader_bytes = shader_sz;
    sum.wg_size = q.regs[static_cast<uint32_t>(valhall::csf::COMPUTESR::WG_SIZE)];
    sum.job_offset[0] = q.regs[static_cast<uint32_t>(valhall::csf::COMPUTESR::JOB_OFFSET_X)];
    sum.job_offset[1] = q.regs[static_cast<uint32_t>(valhall::csf::COMPUTESR::JOB_OFFSET_Y)];
    sum.job_offset[2] = q.regs[static_cast<uint32_t>(valhall::csf::COMPUTESR::JOB_OFFSET_Z)];
    sum.job_size[0] = q.regs[static_cast<uint32_t>(valhall::csf::COMPUTESR::JOB_SIZE_X)];
    sum.job_size[1] = q.regs[static_cast<uint32_t>(valhall::csf::COMPUTESR::JOB_SIZE_Y)];
    sum.job_size[2] = q.regs[static_cast<uint32_t>(valhall::csf::COMPUTESR::JOB_SIZE_Z)];
    sum.sub_dir = rel;

    cap.computes.push_back(sum);
    csf_log(LogLevel::Info, "dump",
            "Captured %s #%u (queue=0x%" PRIx64 ", job=%ux%ux%u, shader=%zuB)",
            is_indirect ? "RUN_COMPUTE_INDIRECT" : "RUN_COMPUTE", comp_idx,
            q.buffer_gpu_addr, sum.job_size[0], sum.job_size[1], sum.job_size[2], shader_sz);
}

void CsfDumper::on_frame_present(FrameCaptureState& cap,
                                 const VaTracker& va,
                                 const std::string& detected_abi,
                                 uint16_t kbase_major,
                                 uint16_t kbase_minor) {
    if (!cap.active || !cap.has_any_capture()) return;

    // Dump render target writeback buffers after GPU completion / present
    for (const auto& frag : cap.fragments) {
        if (frag.rt_color_va == 0 || frag.rt_bytes == 0) continue;
        if (CpuSpan span = va.gpu_to_cpu_span(frag.rt_color_va, 4u)) {
            const size_t n = std::min(span.available_bytes, frag.rt_bytes);
            write_bin_file(cap.out_dir + "/" + frag.sub_dir + "/rt_color_after.bin", span.ptr, n);
            if (frag.frag_index == 0u) {
                write_bin_file(cap.out_dir + "/rt_color_after.bin", span.ptr, n);
            }
        }
    }

    write_metadata_json(cap, va, detected_abi, kbase_major, kbase_minor);

    // Emit diagnostics log if any warnings/errors were recorded
    const auto diags = get_recent_diagnostics();
    if (!diags.empty()) {
        std::ostringstream ds;
        for (const auto& line : diags) {
            ds << line << "\n";
        }
        std::string dstr = ds.str();
        write_bin_file(cap.out_dir + "/diagnostics.log", dstr.data(), dstr.size());
    }
}

void CsfDumper::write_metadata_json(const FrameCaptureState& cap,
                                    const VaTracker& va,
                                    const std::string& detected_abi,
                                    uint16_t kbase_major,
                                    uint16_t kbase_minor) {
    std::ostringstream js;
    js << "{\n"
       << "  \"frame\": " << cap.frame_number << ",\n"
       << "  \"driver_abi\": \"" << json_escape(detected_abi) << "\",\n"
       << "  \"kbase_version\": \"" << kbase_major << "." << kbase_minor << "\",\n"
       << "  \"csf_draw_op\": \"" << (cap.primary_is_idvs2 ? "RUN_IDVS2" : "RUN_IDVS") << "\",\n"
       << "  \"fb_width\": " << cap.primary_fb_width << ",\n"
       << "  \"fb_height\": " << cap.primary_fb_height << ",\n"
       << "  \"rt_color_va\": \"0x" << std::hex << cap.primary_rt_color_va << std::dec << "\",\n"
       << "  \"rt_color_bytes\": " << cap.primary_rt_bytes << ",\n"
       << "  \"draw_count\": " << cap.draws.size() << ",\n"
       << "  \"fragment_count\": " << cap.fragments.size() << ",\n"
       << "  \"compute_count\": " << cap.computes.size() << ",\n"
       << "  \"va_mappings_active\": " << va.mapping_count() << ",\n"
       << "  \"va_lookup_hits\": " << va.stats().lookup_hits << ",\n"
       << "  \"va_lookup_misses\": " << va.stats().lookup_misses << ",\n";

    // Legacy top-level primary draw fields for vkcube_frame1 tooling
    if (!cap.draws.empty()) {
        const auto& d0 = cap.draws[0];
        js << "  \"pos_srt_va\": \"0x" << std::hex << d0.pos_srt_va << "\",\n"
           << "  \"frag_srt_va\": \"0x" << d0.frag_srt_va << "\",\n"
           << "  \"fau_va\": \"0x" << d0.fau_va << std::dec << "\",\n";
    }
    if (!cap.fragments.empty()) {
        js << "  \"fbd_va\": \"0x" << std::hex << cap.fragments[0].fbd_va << std::dec << "\",\n";
    }

    js << "  \"draws\": [\n";
    for (size_t i = 0; i < cap.draws.size(); ++i) {
        const auto& d = cap.draws[i];
        js << "    {\"index\": " << d.draw_index
           << ", \"op\": \"" << (d.is_idvs2 ? "RUN_IDVS2" : "RUN_IDVS") << "\""
           << ", \"queue_va\": \"0x" << std::hex << d.queue_gpu_va << "\""
           << ", \"pos_shader_bytes\": " << std::dec << d.pos_shader_bytes
           << ", \"var_shader_bytes\": " << d.var_shader_bytes
           << ", \"frag_shader_bytes\": " << d.frag_shader_bytes
           << ", \"dir\": \"" << d.sub_dir << "\"}"
           << (i + 1 < cap.draws.size() ? "," : "") << "\n";
    }
    js << "  ],\n";

    js << "  \"fragments\": [\n";
    for (size_t i = 0; i < cap.fragments.size(); ++i) {
        const auto& f = cap.fragments[i];
        js << "    {\"index\": " << f.frag_index
           << ", \"op\": \"" << (f.is_frag2 ? "RUN_FRAGMENT2" : "RUN_FRAGMENT") << "\""
           << ", \"width\": " << f.fb_width
           << ", \"height\": " << f.fb_height
           << ", \"fbd_bytes\": " << f.fbd_bytes
           << ", \"rt_row_stride\": " << f.rt_row_stride
           << ", \"rt_surface_stride\": " << f.rt_surface_stride
           << ", \"rt_bytes\": " << f.rt_bytes
           << ", \"dir\": \"" << f.sub_dir << "\"}"
           << (i + 1 < cap.fragments.size() ? "," : "") << "\n";
    }
    js << "  ],\n";

    js << "  \"computes\": [\n";
    for (size_t i = 0; i < cap.computes.size(); ++i) {
        const auto& c = cap.computes[i];
        js << "    {\"index\": " << c.compute_index
           << ", \"indirect\": " << (c.is_indirect ? "true" : "false")
           << ", \"queue_va\": \"0x" << std::hex << c.queue_gpu_va << std::dec << "\""
           << ", \"wg_size\": " << c.wg_size
           << ", \"job_size\": [" << c.job_size[0] << ", " << c.job_size[1] << ", " << c.job_size[2] << "]"
           << ", \"shader_bytes\": " << c.shader_bytes
           << ", \"dir\": \"" << c.sub_dir << "\"}"
           << (i + 1 < cap.computes.size() ? "," : "") << "\n";
    }
    js << "  ]\n";
    js << "}\n";

    const std::string jstr = js.str();
    write_bin_file(cap.out_dir + "/metadata.json", jstr.data(), jstr.size());
}

} // namespace mali_csf_dump
