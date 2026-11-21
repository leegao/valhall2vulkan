#pragma once

#include "pointer_analysis.h"
#include "valhall_ssa.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace valhall::dataflow {

/// Telemetry counters collected by Valhall SSA optimization passes.
struct SsaPassStats {
    uint32_t nodes_removed = 0;
    uint32_t nodes_replaced = 0;
    uint32_t nodes_folded = 0;
    uint32_t branches_folded = 0;
    uint32_t phis_simplified = 0;
    uint32_t blocks_pruned = 0;
    uint32_t loads_promoted = 0;
    uint32_t stores_eliminated = 0;
    uint32_t structs_scalarized = 0;
    uint32_t buffer_accesses_symbolized = 0;
    uint32_t fp_rewrites = 0;
    uint32_t iterations = 0;

    void merge(const SsaPassStats& o) {
        nodes_removed += o.nodes_removed;
        nodes_replaced += o.nodes_replaced;
        nodes_folded += o.nodes_folded;
        branches_folded += o.branches_folded;
        phis_simplified += o.phis_simplified;
        blocks_pruned += o.blocks_pruned;
        loads_promoted += o.loads_promoted;
        stores_eliminated += o.stores_eliminated;
        structs_scalarized += o.structs_scalarized;
        buffer_accesses_symbolized += o.buffer_accesses_symbolized;
        fp_rewrites += o.fp_rewrites;
        iterations += o.iterations;
    }

    bool any_change() const {
        return nodes_removed > 0 || nodes_replaced > 0 || nodes_folded > 0 ||
               branches_folded > 0 || phis_simplified > 0 || blocks_pruned > 0 ||
               loads_promoted > 0 || stores_eliminated > 0 || structs_scalarized > 0 ||
               buffer_accesses_symbolized > 0 || fp_rewrites > 0;
    }
};

/// Dominator tree and RPO ordering over `compiler::ValhallSsaModule` blocks.
struct DominatorTree {
    static constexpr int32_t kNoIdom = -1;

    std::vector<uint8_t> reachable;
    std::vector<uint32_t> rpo_order;
    std::vector<int32_t> idom;
    std::vector<std::vector<uint32_t>> dom_children;
    std::vector<uint32_t> tin;
    std::vector<uint32_t> tout;

    /// Returns true iff block `a` dominates block `b` (both must be reachable).
    bool dominates(uint32_t a, uint32_t b) const {
        if (a >= reachable.size() || b >= reachable.size()) return false;
        if (!reachable[a] || !reachable[b]) return false;
        return tin[a] <= tin[b] && tout[b] <= tout[a];
    }
};

/// Module-owned cache for expensive SSA analyses (dominator tree and
/// tracked-value / pointer analysis). Stored via `ValhallSsaModule::analysis_cache`.
struct SsaModuleAnalysisCache {
    std::optional<DominatorTree> dom_tree;
    std::optional<PointerAnalysisResult> ptr_analysis;
    uint32_t dom_tree_builds = 0;
    uint32_t ptr_analysis_runs = 0;
};

/// Computes reachability, RPO, and Cooper-Harvey-Kennedy immediate dominators
/// rooted at block 0.
DominatorTree build_dominator_tree(const compiler::ValhallSsaModule& mod);

/// Returns a cached `DominatorTree` stored on `mod`, building it once on demand.
const DominatorTree& get_or_build_dominator_tree(const compiler::ValhallSsaModule& mod);

/// Returns a cached `PointerAnalysisResult` (`TrackedVal` lattice) stored on `mod`,
/// running `run_ssa_pointer_analysis` once on demand.
const PointerAnalysisResult& get_or_run_pointer_analysis(
    const compiler::ValhallSsaModule& mod,
    const PointerAnalysisEnv& env = {});

/// Returns the cached `PointerAnalysisResult` if already computed on `mod`, or nullptr.
const PointerAnalysisResult* get_cached_pointer_analysis(const compiler::ValhallSsaModule& mod);

/// Returns the underlying analysis cache struct (or nullptr if not yet populated).
const SsaModuleAnalysisCache* get_ssa_analysis_cache(const compiler::ValhallSsaModule& mod);

/// Invalidates all cached analyses attached to `mod`.
void invalidate_ssa_analyses(const compiler::ValhallSsaModule& mod);

/// Returns the distinct successor block IDs of `blk`.
std::vector<uint32_t> get_block_succs(const compiler::SsaBlock& blk);

/// Returns true iff `node` has no observable side effects, memory writes,
/// control-flow transfers, barriers, or subgroup/derivative lane coupling.
/// When `fau_is_readonly` is true, `LoadFau` is treated as pure.
bool is_pure_ssa_op(const compiler::SsaNode& node, bool fau_is_readonly = true);

/// Returns true iff `op` is a binary commutative operation whose two arguments
/// can be swapped without changing semantics.
bool is_commutative_ssa_op(tables::SsaOp op);

/// Canonicalizes commutative operand order in-place for binary commutative ops
/// (sorting `args[0..1]`) and `Fma(a, b, c)` (sorting multiplicative lanes `a, b`
/// while preserving addend `c`).
void canonicalize_commutative_args(tables::SsaOp op, std::vector<compiler::SsaId>& args);

/// Resolves a union-find / value substitution table to its canonical root.
compiler::SsaId resolve_remap(const std::vector<compiler::SsaId>& remap, compiler::SsaId id);

/// Rewrites all uses (`args`, `phi_incomings`, `branch_cond`, `block_out_*`)
/// according to `remap`, recomputes `mod.use_counts`, and invalidates cached analyses.
/// Returns the number of operand slots rewritten.
uint32_t apply_value_remap(compiler::ValhallSsaModule& mod,
                           const std::vector<compiler::SsaId>& remap);

/// Removes every node marked non-zero in `dead_mask` from all block `phis` and
/// `nodes` vectors, clears its operands to `SsaOp::Nop`, recomputes
/// `mod.use_counts`, and invalidates cached analyses. Returns the number of nodes erased.
uint32_t erase_dead_nodes(compiler::ValhallSsaModule& mod,
                          const std::vector<uint8_t>& dead_mask);

/// Rebuilds `blk.preds` for all blocks from `fallthrough_block` and
/// `branch_target_block` edges of reachable blocks, removes `phi_incomings`
/// whose predecessor is no longer in `blk.preds`, and invalidates cached analyses.
uint32_t rebuild_cfg_preds_and_prune_phis(compiler::ValhallSsaModule& mod);

/// Iteratively eliminates trivial Phi nodes (`phi(x, x, self) -> x`) and
/// applies the resulting substitutions. Returns the number of Phis eliminated.
uint32_t simplify_trivial_phis(compiler::ValhallSsaModule& mod);

/// Collapses `Copy` nodes and same-type `Bitcast` nodes to their source operand
/// (and folds any integer constants already proven by cached `TrackedVal` analysis).
/// Returns the number of nodes replaced.
uint32_t run_ssa_copy_prop(compiler::ValhallSsaModule& mod);

/// Extracts a 32-bit constant bit pattern if `id` resolves to a constant
/// (consulting cached `TrackedVal` analysis first, then unwrapping `Copy`/`Bitcast`).
bool try_get_const_u32(const compiler::ValhallSsaModule& mod, compiler::SsaId id, uint32_t* out);

/// Unwraps `Copy` and `Bitcast(F32, ConstU32)` chains and extracts an IEEE-754
/// binary32 bit pattern if `id` resolves to a constant float.
bool try_get_const_f32_bits(const compiler::ValhallSsaModule& mod, compiler::SsaId id, uint32_t* out_bits);

/// Extracts a 64-bit constant if `id` resolves to a constant
/// (consulting cached `TrackedVal` analysis first, then unwrapping `ConstU64`/`Pack64`).
bool try_get_const_u64(const compiler::ValhallSsaModule& mod, compiler::SsaId id, uint64_t* out);

/// Unwraps `Copy` chains and extracts a boolean constant if `id` resolves to
/// `ConstBool`.
bool try_get_const_bool(const compiler::ValhallSsaModule& mod, compiler::SsaId id, bool* out);

/// Bit-exact conversion helpers between `float` and IEEE-754 `uint32_t`.
uint32_t f32_to_bits(float v);
float bits_to_f32(uint32_t bits);

/// Inserts a newly constructed `SsaNode` into `blk_id` immediately before
/// `before_node_id` (or at the end of `blk.nodes` if `before_node_id` is
/// `kInvalidSsa` or not found), and invalidates cached analyses.
compiler::SsaId insert_node_before(compiler::ValhallSsaModule& mod,
                                   uint32_t blk_id,
                                   compiler::SsaId before_node_id,
                                   compiler::SsaNode node);

}  // namespace valhall::dataflow
