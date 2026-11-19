#pragma once
// ============================================================================
// csf_dumper.h
// Captures Mali CSF graphics draws (RUN_IDVS / RUN_IDVS2), fragment passes
// (RUN_FRAGMENT / RUN_FRAGMENT2), and compute dispatches (RUN_COMPUTE /
// RUN_COMPUTE_INDIRECT) with VA-bounded dynamic sizes, RTBuffer stride
// calculation, and vkcube_frame1 backward compatibility.
// ============================================================================

#include "csf_config_log.h"
#include "csf_interpreter.h"
#include "csf_va_tracker.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mali_csf_dump {

struct CapturedDrawSummary {
    uint32_t draw_index = 0;
    bool is_idvs2 = false;
    uint64_t queue_gpu_va = 0;
    uint64_t run_insn = 0;
    uint64_t pos_srt_va = 0;
    uint64_t frag_srt_va = 0;
    uint64_t fau_va = 0;
    size_t pos_shader_bytes = 0;
    size_t var_shader_bytes = 0;
    size_t frag_shader_bytes = 0;
    std::string sub_dir;
};

struct CapturedFragmentSummary {
    uint32_t frag_index = 0;
    bool is_frag2 = false;
    uint64_t queue_gpu_va = 0;
    uint64_t fbd_va = 0;
    size_t fbd_bytes = 0;
    uint32_t fb_width = 0;
    uint32_t fb_height = 0;
    uint32_t render_target_count = 0;
    uint64_t rt_color_va = 0;
    uint32_t rt_row_stride = 0;
    uint32_t rt_surface_stride = 0;
    size_t rt_bytes = 0;
    std::string sub_dir;
};

struct CapturedComputeSummary {
    uint32_t compute_index = 0;
    bool is_indirect = false;
    uint64_t queue_gpu_va = 0;
    uint64_t run_insn = 0;
    uint64_t srt_va = 0;
    uint64_t fau_va = 0;
    uint64_t spd_va = 0;
    uint64_t tsd_va = 0;
    size_t shader_bytes = 0;
    uint32_t wg_size = 0;
    uint32_t job_size[3] = {0, 0, 0};
    uint32_t job_offset[3] = {0, 0, 0};
    std::string sub_dir;
};

struct FrameCaptureState {
    bool active = false;
    uint32_t frame_number = 0;
    std::string out_dir;
    std::vector<CapturedDrawSummary> draws;
    std::vector<CapturedFragmentSummary> fragments;
    std::vector<CapturedComputeSummary> computes;

    // Primary draw/fragment snapshot for legacy vkcube_frame1 root compatibility
    bool has_primary_draw = false;
    bool primary_is_idvs2 = false;
    uint32_t primary_idvs_sr[64] = {0};
    bool has_primary_frag = false;
    uint64_t primary_rt_color_va = 0;
    uint32_t primary_fb_width = 0;
    uint32_t primary_fb_height = 0;
    size_t primary_rt_bytes = 0;

    bool has_any_capture() const {
        return !draws.empty() || !fragments.empty() || !computes.empty();
    }
};

// Estimate the byte length of a Valhall 64-bit shader binary by scanning for
// end-of-shader flow control (bits[62:59] == 15) within `available_bytes`.
size_t estimate_valhall_shader_size(const void* code_ptr,
                                    size_t available_bytes,
                                    size_t max_shader_bytes);

// Compute the byte size of an RGB render target writeback buffer from
// FramebufferParameters + RTBuffer stride fields, clamped to available mapping.
size_t compute_rt_writeback_bytes(uint32_t fb_width,
                                  uint32_t fb_height,
                                  uint32_t row_stride,
                                  uint32_t surface_stride,
                                  size_t available_bytes);

class CsfDumper {
public:
    void reset();

    void on_run_idvs(FrameCaptureState& cap,
                     const CsfQueueState& q,
                     const VaTracker& va,
                     const LayerConfig& cfg,
                     uint32_t current_frame,
                     uint64_t insn,
                     bool is_idvs2);

    void on_run_fragment(FrameCaptureState& cap,
                         const CsfQueueState& q,
                         const VaTracker& va,
                         const LayerConfig& cfg,
                         uint32_t current_frame,
                         uint64_t insn,
                         uint32_t fbd_reg_idx,
                         bool is_frag2);

    void on_run_compute(FrameCaptureState& cap,
                        const CsfQueueState& q,
                        const VaTracker& va,
                        const LayerConfig& cfg,
                        uint32_t current_frame,
                        uint64_t insn,
                        bool is_indirect);

    void on_frame_present(FrameCaptureState& cap,
                          const VaTracker& va,
                          const std::string& detected_abi,
                          uint16_t kbase_major,
                          uint16_t kbase_minor);

private:
    void ensure_frame_dir(FrameCaptureState& cap, const LayerConfig& cfg, uint32_t current_frame);
    void write_metadata_json(const FrameCaptureState& cap,
                             const VaTracker& va,
                             const std::string& detected_abi,
                             uint16_t kbase_major,
                             uint16_t kbase_minor);
};

} // namespace mali_csf_dump
