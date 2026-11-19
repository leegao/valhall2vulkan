#include "csf_layer_state.h"
#include "csf_config_log.h"

#include <unistd.h>

namespace mali_csf_dump {

LayerState& layer_state() {
    static LayerState s;
    return s;
}

void process_queue_kick_locked(LayerState& s, uint64_t buffer_gpu_addr) {
    auto it = s.queues_by_gpu_va.find(buffer_gpu_addr);
    if (it == s.queues_by_gpu_va.end()) return;
    CsfQueueState& q = it->second;
    if (q.buffer_size == 0) return;

    uint64_t insert_off = q.buffer_size;
    if (q.user_io_cpu != 0) {
        const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        const size_t ins_off = compute_user_io_insert_offset(q, page_sz);
        const volatile uint64_t* ins_ptr =
            reinterpret_cast<const volatile uint64_t*>(q.user_io_cpu + ins_off);
        insert_off = __atomic_load_n(ins_ptr, __ATOMIC_ACQUIRE);
    }

    const LayerConfig cfg = read_layer_config();
    const uint32_t cur_frame = s.current_frame.load();
    const bool capture_this_frame =
        cfg.enabled && !s.dump_done.load() && (cur_frame == cfg.target_frame);

    CsfDispatchCallbacks cb{};
    if (capture_this_frame) {
        cb.on_run_idvs = [&](const CsfQueueState& qs, uint64_t insn, bool is_idvs2) {
            s.dumper.on_run_idvs(s.capture, qs, s.va, cfg, cur_frame, insn, is_idvs2);
        };
        cb.on_run_fragment = [&](const CsfQueueState& qs, uint64_t insn, uint32_t fbd_reg, bool is_frag2) {
            s.dumper.on_run_fragment(s.capture, qs, s.va, cfg, cur_frame, insn, fbd_reg, is_frag2);
        };
        cb.on_run_compute = [&](const CsfQueueState& qs, uint64_t insn, bool is_indirect) {
            s.dumper.on_run_compute(s.capture, qs, s.va, cfg, cur_frame, insn, is_indirect);
        };
    }

    process_queue_ring(q, s.va, insert_off, cb);
}

void process_frame_present_locked(LayerState& s) {
    const LayerConfig cfg = read_layer_config();
    if (!cfg.enabled || s.dump_done.load()) return;

    const uint32_t cur_frame = s.current_frame.load();
    if (cur_frame == cfg.target_frame && s.capture.has_any_capture()) {
        s.dumper.on_frame_present(s.capture, s.va, s.detected_abi,
                                  s.kbase_ver_major, s.kbase_ver_minor);
        s.dump_done.store(true);
    }
    s.current_frame.fetch_add(1u);
}

void reset_layer_state_for_test(uint32_t target_frame) {
    LayerState& s = layer_state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.mali_fds.clear();
    s.va.clear();
    s.queues_by_gpu_va.clear();
    s.queues_by_mmap_handle.clear();
    s.kbase_ver_major = 0;
    s.kbase_ver_minor = 0;
    s.dump_done.store(false);
    s.current_frame.store(target_frame);
    s.capture = FrameCaptureState{};
    clear_recent_diagnostics();
}

} // namespace mali_csf_dump
