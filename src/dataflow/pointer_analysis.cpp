#include "pointer_analysis.h"

#include <algorithm>
#include <cstdio>

namespace valhall::dataflow {

namespace {

std::string hex_off(int64_t off) {
    char buf[32];
    if (off < 0) {
        std::snprintf(buf, sizeof(buf), "-0x%llx", static_cast<unsigned long long>(-off));
    } else {
        std::snprintf(buf, sizeof(buf), "+0x%llx", static_cast<unsigned long long>(off));
    }
    return buf;
}

/// Dereference a single `TrackedVal` pointer by delegating to `SymbolicMemoryLoc::deref()`,
/// then attaching MBS2 symbol names via `env` callbacks.
TrackedVal deref_pointer_single(const TrackedVal& addr, uint32_t elem_bytes, const PointerAnalysisEnv& env) {
    const PointerVal* ptr = addr.inner.get_if<PointerDomain>();
    if (!ptr || !ptr->is_valid_pointer()) return TrackedValDomain::top();
    MemoryDerefResult d = ptr->to_memory_loc().deref(elem_bytes);
    auto resolve_srt = [&](int32_t rack, int32_t desc, const std::string& fallback) -> std::string {
        if (!fallback.empty()) return fallback;
        if (env.lookup_srt_symbol && rack >= 0 && desc >= 0) {
            return env.lookup_srt_symbol(static_cast<uint8_t>(rack), static_cast<uint32_t>(desc));
        }
        return "";
    };

    switch (d.kind) {
        case MemoryDerefResult::Kind::SrtRackPtr:
            return TrackedValDomain::make_srt_rack(d.table_id, 0, d.offset_known);
        case MemoryDerefResult::Kind::BufferSize:
            return TrackedValDomain::make_buffer_size(d.table_id, d.entry_id, resolve_srt(d.table_id, d.entry_id, d.key));
        case MemoryDerefResult::Kind::BufferAddr:
            return TrackedValDomain::make_buffer_addr(d.table_id, d.entry_id, resolve_srt(d.table_id, d.entry_id, d.key),
                                                      0, false, true);
        case MemoryDerefResult::Kind::DescriptorAddr:
            return TrackedValDomain::make_descriptor(d.table_id, d.entry_id, resolve_srt(d.table_id, d.entry_id, d.key),
                                                     d.byte_offset, d.offset_known);
        case MemoryDerefResult::Kind::FauUniformRef:
            return TrackedValDomain::make_fau_uniform(static_cast<uint32_t>(d.byte_offset));
        case MemoryDerefResult::Kind::SymbolValue: {
            if (d.from_data_section) {
                std::string lbl = ".data";
                if (d.offset_known) lbl += hex_off(d.byte_offset);
                return TrackedValDomain::make_symbol_value(-1, -1, std::move(lbl), false);
            }
            std::string mem_sym = d.key;
            if (d.offset_known && d.byte_offset >= 0 && env.lookup_symbol_member &&
                d.table_id >= 0 && d.entry_id >= 0) {
                std::string m = env.lookup_symbol_member(
                    static_cast<uint8_t>(d.table_id),
                    static_cast<uint32_t>(d.entry_id),
                    static_cast<uint32_t>(d.byte_offset));
                if (!m.empty()) mem_sym = std::move(m);
            }
            return TrackedValDomain::make_symbol_value(d.table_id, d.entry_id, std::move(mem_sym), false, d.byte_offset, d.offset_known);
        }
        case MemoryDerefResult::Kind::Top:
        default:
            return TrackedValDomain::top();
    }
}

void append_unique_note(std::vector<std::string>& notes, const std::string& s) {
    if (s.empty()) return;
    if (std::find(notes.begin(), notes.end(), s) == notes.end()) {
        notes.push_back(s);
    }
}

/// SSA Pointer Transfer Visitor operating over `TrackedValSetDomain`.
///
/// `on_imul`, `on_and`, `on_or`, `on_xor`, `on_not`, `on_shift`, `on_pack64`,
/// `on_split64`, `on_tuple_extract`, and `on_convert` flow automatically through
/// `SsaDomainTransferVisitor<TrackedValSetDomain>` -> `FiniteSetDomain` ->
/// `LaneDomainWrapper` -> `UnionDomain<ConstInt, Pointer, Symbol, ShaderAtom>`.
class SsaPointerTransferVisitor : public SsaDomainTransferVisitor<TrackedValSetDomain> {
public:
    using SetDom = TrackedValSetDomain;

    SsaPointerTransferVisitor(
        const PointerAnalysisEnv& env,
        std::map<uint32_t, PilotFauWrite>* pilot_writes,
        std::vector<std::vector<std::string>>* ssa_notes)
        : env_(env), pilot_writes_(pilot_writes), ssa_notes_(ssa_notes) {}

    void on_const(compiler::SsaId dst, const compiler::SsaNode& node) override {
        using tables::SsaOp;
        switch (node.op) {
            case SsaOp::ConstU64:
                assign(dst, TrackedValDomain::make_const(static_cast<int64_t>(node.imm_u64)));
                break;
            case SsaOp::ConstS32:
                assign(dst, TrackedValDomain::make_const(static_cast<int64_t>(static_cast<int32_t>(node.imm_u32))));
                break;
            case SsaOp::ConstU32:
            case SsaOp::ConstBool:
            case SsaOp::ConstF32:
                assign(dst, TrackedValDomain::make_const(static_cast<int64_t>(node.imm_u32)));
                break;
            default:
                assign(dst, SetDom::top());
                break;
        }
    }

    void on_load_sideband(compiler::SsaId dst, uint32_t member_idx, bool /*is_64bit*/,
                          const compiler::SsaNode&) override {
        if (env_.is_fau_dst_va(member_idx)) {
            assign(dst, TrackedValDomain::make_fau_dst(0, true));
            return;
        }
        if (env_.is_srt_base_va(member_idx)) {
            assign(dst, TrackedValDomain::make_srt_base(0, true));
            return;
        }
        assign(dst, TrackedValDomain::make_sideband(member_idx));
    }

    void on_load_builtin(compiler::SsaId dst, uint32_t /*builtin_id*/,
                         const compiler::SsaNode& node) override {
        std::string lbl = node.aux_str.empty() ? "gl_BuiltIn" : node.aux_str;
        assign(dst, TrackedValDomain::make_abi(std::move(lbl)));
    }

    void on_load_fau(compiler::SsaId dst, uint32_t fau_word_idx,
                     const compiler::SsaNode&) override {
        uint32_t byte_off = fau_word_idx * 4u;
        if (state_) {
            TrackedValSet fv = state_->fau(byte_off);
            if (!fv.is_bottom()) {
                assign(dst, fv);
                return;
            }
            if ((byte_off & 4u) != 0) {
                TrackedValSet lo = state_->fau(byte_off & ~7u);
                if (!lo.is_bottom()) {
                    assign(dst, SetDom::lift_unary(lo, [](const TrackedVal& v) {
                        if (v.inner.is<PointerDomain>()) {
                            return TrackedValDomain::tuple_extract(v, 1);
                        }
                        return TrackedValDomain::top();
                    }));
                    return;
                }
            }
        }
        assign(dst, TrackedValDomain::make_fau_uniform(byte_off));
    }

    void on_vec(compiler::SsaId dst, const compiler::SsaNode& node) override {
        if (node.op == tables::SsaOp::VecExtract && !node.args.empty()) {
            uint32_t comp = node.imm_u32;
            assign(dst, SetDom::lift_unary(eval(node.args[0]), [comp](const TrackedVal& base) {
                if (const auto* atom = base.inner.get_if<ShaderAtomDomain>();
                    atom && atom->is_known() && atom->value.kind == ShaderAtomDomain::Kind::AbiPreload) {
                    static const char kComp[] = "xyzw";
                    std::string lbl = atom->value.label;
                    if (comp < 4) {
                        lbl.push_back('.');
                        lbl.push_back(kComp[comp]);
                    }
                    return TrackedValDomain::make_abi(std::move(lbl));
                }
                return base;
            }));
            return;
        }
        assign(dst, SetDom::top());
    }

    void on_iadd(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool /*is_64bit*/, const compiler::SsaNode&) override {
        TrackedValSet out = SetDom::add(eval(lhs), eval(rhs), env_.lookup_srt_symbol);
        assign(dst, out);
        TrackedVal single;
        if (out.try_as_singleton(&single) && single.inner.is<PointerDomain>()) {
            record_note(dst, "ptr " + format_tracked_val(single));
        }
    }

    void on_isub(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool /*is_64bit*/, const compiler::SsaNode&) override {
        if (lhs != compiler::kInvalidSsa && lhs == rhs) {
            assign(dst, TrackedValDomain::make_const(0));
            return;
        }
        assign(dst, SetDom::sub(eval(lhs), eval(rhs), env_.lookup_srt_symbol));
    }

    void on_psb_load(compiler::SsaId dst, compiler::SsaId addr, uint32_t elem_bytes,
                     const compiler::SsaNode&) override {
        TrackedValSet addr_set = eval(addr);
        TrackedVal single_addr;
        if (!addr_set.try_as_singleton(&single_addr)) {
            assign(dst, addr_set.is_bottom() ? SetDom::bottom() : SetDom::top());
            return;
        }
        TrackedVal loaded = deref_pointer_single(single_addr, elem_bytes, env_);
        assign(dst, loaded);
        if (!single_addr.is_bottom() && !single_addr.is_top()) {
            record_note(dst, "load " + format_tracked_val(single_addr) + " -> " + format_tracked_val(loaded));
        }
    }

    void on_psb_store(compiler::SsaId addr, compiler::SsaId val, uint32_t elem_bytes,
                      const compiler::SsaNode& node) override {
        TrackedValSet addr_set = eval(addr);
        TrackedValSet val_set = eval(val);
        TrackedVal single_addr;
        if (!addr_set.try_as_singleton(&single_addr)) {
            return;
        }
        const PointerVal* ptr = single_addr.inner.get_if<PointerDomain>();
        if (!ptr || !ptr->is_valid_pointer()) {
            return;
        }
        if (ptr->kind == PointerVal::Kind::FauDstBase &&
            ptr->offset_known && ptr->byte_offset >= 0) {
            uint32_t fau_off = static_cast<uint32_t>(ptr->byte_offset);
            if (state_) {
                state_->join_fau(fau_off, val_set);
            }
            TrackedVal collapsed = collapse_tracked_val_set(val_set);
            if (pilot_writes_ && !collapsed.is_bottom() && !collapsed.is_top()) {
                auto it = pilot_writes_->find(fau_off);
                if (it == pilot_writes_->end()) {
                    (*pilot_writes_)[fau_off] = PilotFauWrite{fau_off, elem_bytes, collapsed};
                } else {
                    it->second.value = TrackedValDomain::join(it->second.value, collapsed);
                    it->second.byte_size = std::max(it->second.byte_size, elem_bytes);
                }
            }
            record_note(node.id, "pilot store FAU+0x" + hex_off(fau_off).substr(3) + " <- " +
                                     format_tracked_val_set(val_set));
        } else {
            if (state_) {
                state_->join_sym(ptr->to_memory_loc(), val_set);
            }
            record_note(node.id, "store " + format_tracked_val(single_addr));
        }
    }

    void on_atomic_exchange(compiler::SsaId dst, compiler::SsaId addr,
                            compiler::SsaId val, const compiler::SsaNode& node) override {
        TrackedValSet addr_set = eval(addr);
        TrackedValSet val_set = eval(val);
        TrackedVal single_addr;
        const PointerVal* ptr = nullptr;
        if (!addr_set.try_as_singleton(&single_addr) ||
            !(ptr = single_addr.inner.get_if<PointerDomain>()) || !ptr->is_valid_pointer()) {
            assign(dst, SetDom::top());
            return;
        }
        SymbolicMemoryLoc loc = ptr->to_memory_loc();
        TrackedValSet old_val = state_ ? state_->sym(loc) : SetDom::bottom();
        if (old_val.is_bottom()) {
            old_val = deref_pointer_single(single_addr, node.aux_bool ? 8u : 4u, env_);
        }
        assign(dst, old_val);
        if (state_) {
            state_->join_sym(loc, val_set);
        }
        record_note(dst, "axchg " + format_tracked_val(single_addr));
    }

    void on_load_vert_attr(compiler::SsaId dst, const compiler::SsaNode& node) override {
        uint32_t loc = node.imm_u32;
        std::string sym = env_.lookup_input_symbol ? env_.lookup_input_symbol(loc) : "";
        TrackedVal v = TrackedValDomain::make_vertex_attr(loc, sym);
        assign(dst, v);
        record_note(dst, format_tracked_val(v));
    }

    void on_load_frag_varying(compiler::SsaId dst, const compiler::SsaNode& node) override {
        uint32_t loc = node.imm_u32;
        std::string sym = env_.lookup_input_symbol ? env_.lookup_input_symbol(loc) : "";
        TrackedVal v = TrackedValDomain::make_varying(loc, sym);
        assign(dst, v);
        record_note(dst, format_tracked_val(v));
    }

    void on_tex_sample(compiler::SsaId dst, const compiler::SsaNode& node) override {
        uint32_t img = node.imm_u32;
        uint32_t smp = node.imm_u32_2;
        std::string sym = env_.lookup_image_symbol ? env_.lookup_image_symbol(img) : "";
        TrackedVal v = TrackedValDomain::make_tex_sample(img, smp, sym);
        assign(dst, v);
        record_note(dst, format_tracked_val(v));
    }

    void on_store_output(const compiler::SsaNode& node) override {
        if (node.op == tables::SsaOp::StoreFragColor || node.op == tables::SsaOp::StoreVertVarying) {
            uint32_t loc = node.imm_u32;
            std::string sym = env_.lookup_output_symbol ? env_.lookup_output_symbol(loc) : "";
            TrackedVal v = TrackedValDomain::make_blend_desc(loc, sym);
            record_note(node.id, "out " + format_tracked_val(v));
        }
    }

private:
    void record_note(compiler::SsaId id, const std::string& note) {
        if (ssa_notes_ && id != compiler::kInvalidSsa && id < ssa_notes_->size()) {
            append_unique_note((*ssa_notes_)[id], note);
        }
    }

    const PointerAnalysisEnv& env_;
    std::map<uint32_t, PilotFauWrite>* pilot_writes_ = nullptr;
    std::vector<std::vector<std::string>>* ssa_notes_ = nullptr;
};

}  // namespace

PointerAnalysisResult run_ssa_pointer_analysis(
    const compiler::ValhallSsaModule& mod,
    const PointerAnalysisEnv& env,
    const SsaLivenessResult* liveness,
    uint32_t max_iterations) {
    PointerAnalysisResult out;
    out.ssa_notes.resize(mod.nodes.size());

    AbstractState<TrackedValSetDomain> entry(mod.nodes.size());
    for (const auto& kv : env.initial_fau) {
        entry.set_fau(kv.first, kv.second);
    }

    SsaPointerTransferVisitor visitor(env, &out.pilot_fau_writes, &out.ssa_notes);
    SsaDataflowResult<TrackedValSetDomain> df =
        run_ssa_forward_dataflow<TrackedValSetDomain>(mod, entry, visitor, liveness, max_iterations);

    out.converged = df.converged;
    out.iterations = df.iterations;
    out.final_state = std::move(df.final_state);
    if (liveness) {
        out.liveness = *liveness;
    }
    return out;
}

}  // namespace valhall::dataflow
