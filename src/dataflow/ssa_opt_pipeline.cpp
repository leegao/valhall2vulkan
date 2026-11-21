#include "ssa_opt_pipeline.h"

#include <sstream>

namespace valhall::dataflow {

namespace {

std::string format_round_summary(uint32_t round, const SsaPassStats& s) {
    std::ostringstream os;
    os << "round " << round
       << ": removed=" << s.nodes_removed
       << " replaced=" << s.nodes_replaced
       << " folded=" << s.nodes_folded
       << " branches=" << s.branches_folded
       << " phis=" << s.phis_simplified
       << " fp=" << s.fp_rewrites;
    return os.str();
}

}  // namespace

SsaOptPipelineResult run_ssa_opt_pipeline(compiler::ValhallSsaModule& mod,
                                          const SsaOptPipelineOptions& opts) {
    SsaOptPipelineResult res;
    if (mod.blocks.empty()) return res;

    if (opts.run_sroa) {
        SroaOptions sopts;
        sopts.tls_sideband_member = opts.tls_sideband_member;
        sopts.run_mem2reg_after_split = opts.run_spill_mem2reg;
        SsaPassStats st = run_ssa_sroa(mod, sopts);
        res.total_stats.merge(st);
        if (opts.record_trace && st.any_change()) {
            res.trace.push_back("sroa+mem2reg: structs=" + std::to_string(st.structs_scalarized) +
                                " loads=" + std::to_string(st.loads_promoted) +
                                " stores=" + std::to_string(st.stores_eliminated));
        }
    } else if (opts.run_spill_mem2reg) {
        SpillMem2RegOptions mopts;
        mopts.tls_sideband_member = opts.tls_sideband_member;
        SsaPassStats st = run_ssa_spill_mem2reg(mod, mopts);
        res.total_stats.merge(st);
        if (opts.record_trace && st.any_change()) {
            res.trace.push_back("mem2reg: loads=" + std::to_string(st.loads_promoted) +
                                " stores=" + std::to_string(st.stores_eliminated));
        }
    }

    if (opts.run_buffer_symbolization) {
        BufferSymbolizeOptions bopts;
        bopts.ptr_analysis = opts.ptr_analysis;
        bopts.forward_store_to_load = true;
        bopts.annotate_ssa_nodes = true;
        BufferSymbolizeResult bres = run_ssa_symbolize_buffers(mod, bopts);
        res.total_stats.merge(bres.stats);
        res.symbolized_accesses = std::move(bres.accesses);
        if (opts.record_trace && bres.stats.any_change()) {
            res.trace.push_back("sym_buf: symbolized=" +
                                std::to_string(bres.stats.buffer_accesses_symbolized) +
                                " forwarded=" + std::to_string(bres.stats.loads_promoted));
        }
    }

    for (uint32_t round = 0; round < opts.max_fixpoint_rounds; ++round) {
        SsaPassStats round_stats;
        ++res.rounds_executed;

        if (opts.run_sccp) {
            SccpOptions scopts;
            scopts.fau_is_readonly = opts.fau_is_readonly;
            scopts.run_trailing_dce = false;
            round_stats.merge(run_ssa_sccp(mod, scopts));
        }

        if (opts.run_fp_canonicalize) {
            FpOptOptions fopts;
            fopts.mode = opts.fp_mode;
            FpOptResult fres = run_ssa_fp_canonicalize(mod, fopts);
            round_stats.merge(fres.stats);
            for (const auto& [rule, cnt] : fres.rule_hits) {
                res.fp_rule_hits[rule] += cnt;
            }
        }

        if (opts.run_cse) {
            CseOptions copts;
            copts.fau_is_readonly = opts.fau_is_readonly;
            round_stats.merge(run_ssa_cse(mod, copts));
        }

        if (opts.run_dce) {
            DceOptions dopts;
            dopts.fau_is_readonly = opts.fau_is_readonly;
            round_stats.merge(run_ssa_dce(mod, dopts));
        }

        res.total_stats.merge(round_stats);
        if (opts.record_trace) {
            res.trace.push_back(format_round_summary(round, round_stats));
        }
        if (!round_stats.any_change()) {
            break;
        }
    }

    return res;
}

}  // namespace valhall::dataflow
