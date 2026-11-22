#pragma once

#include "flat.h"
#include "mixins.h"
#include "traits.h"
#include "valhall_tables_generated.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace valhall::dataflow {

/// Maximum AST depth for bounded symbolic expressions (guarantees finite lattice height).
static constexpr uint8_t kMaxSymExprDepth = 8;

/// Immutable, structurally-comparable symbolic expression node capturing
/// buffer base addresses, UBO/SSBO scalar loads, constants, and bounded pure
/// Pilot/Main ALU, bitfield, and comparison expressions.
struct SymExpr {
    enum class Kind : uint8_t {
        ConstU32 = 0,
        BufferAddr,
        BufferScalarLoad,
        FauSlotRef,
        UnaryOp,
        BinaryOp,
        SelectOp,
        TernaryOp,
        QuaternaryOp,
    };

    Kind kind = Kind::ConstU32;
    uint32_t const_u32 = 0;
    int32_t rack = -1;
    int32_t desc_idx = -1;
    std::string symbol_name;
    int64_t byte_offset = 0;
    bool offset_known = true;
    uint32_t elem_bytes = 4;
    uint32_t fau_offset = 0;
    tables::SsaOp op = tables::SsaOp::Nop;
    uint8_t depth = 0;
    std::shared_ptr<const SymExpr> a;
    std::shared_ptr<const SymExpr> b;
    std::shared_ptr<const SymExpr> c;
    std::shared_ptr<const SymExpr> d;

    static std::shared_ptr<const SymExpr> make_const_u32(uint32_t v) {
        auto e = std::make_shared<SymExpr>();
        e->kind = Kind::ConstU32;
        e->const_u32 = v;
        e->depth = 0;
        return e;
    }

    static std::shared_ptr<const SymExpr> make_buffer_addr(
        int32_t rack, int32_t desc_idx, std::string sym,
        int64_t off = 0, bool off_known = true) {
        auto e = std::make_shared<SymExpr>();
        e->kind = Kind::BufferAddr;
        e->rack = rack;
        e->desc_idx = desc_idx;
        e->symbol_name = std::move(sym);
        e->byte_offset = off;
        e->offset_known = off_known;
        e->depth = 0;
        return e;
    }

    static std::shared_ptr<const SymExpr> make_buffer_load(
        int32_t rack, int32_t desc_idx, std::string sym,
        int64_t byte_off, uint32_t bytes = 4) {
        auto e = std::make_shared<SymExpr>();
        e->kind = Kind::BufferScalarLoad;
        e->rack = rack;
        e->desc_idx = desc_idx;
        e->symbol_name = std::move(sym);
        e->byte_offset = byte_off;
        e->offset_known = true;
        e->elem_bytes = bytes;
        e->depth = 0;
        return e;
    }

    static std::shared_ptr<const SymExpr> make_fau_ref(uint32_t fau_off) {
        auto e = std::make_shared<SymExpr>();
        e->kind = Kind::FauSlotRef;
        e->fau_offset = fau_off;
        e->depth = 0;
        return e;
    }

    static std::shared_ptr<const SymExpr> make_unary(
        tables::SsaOp op, std::shared_ptr<const SymExpr> child) {
        if (!child || child->depth >= kMaxSymExprDepth) return nullptr;
        auto e = std::make_shared<SymExpr>();
        e->kind = Kind::UnaryOp;
        e->op = op;
        e->depth = static_cast<uint8_t>(child->depth + 1u);
        e->a = std::move(child);
        return e;
    }

    static std::shared_ptr<const SymExpr> make_binary(
        tables::SsaOp op,
        std::shared_ptr<const SymExpr> lhs,
        std::shared_ptr<const SymExpr> rhs) {
        if (!lhs || !rhs) return nullptr;
        uint8_t md = (lhs->depth > rhs->depth) ? lhs->depth : rhs->depth;
        if (md >= kMaxSymExprDepth) return nullptr;
        auto e = std::make_shared<SymExpr>();
        e->kind = Kind::BinaryOp;
        e->op = op;
        e->depth = static_cast<uint8_t>(md + 1u);
        e->a = std::move(lhs);
        e->b = std::move(rhs);
        return e;
    }

    static std::shared_ptr<const SymExpr> make_select(
        std::shared_ptr<const SymExpr> cond,
        std::shared_ptr<const SymExpr> tv,
        std::shared_ptr<const SymExpr> fv) {
        if (!cond || !tv || !fv) return nullptr;
        uint8_t md = cond->depth;
        if (tv->depth > md) md = tv->depth;
        if (fv->depth > md) md = fv->depth;
        if (md >= kMaxSymExprDepth) return nullptr;
        auto e = std::make_shared<SymExpr>();
        e->kind = Kind::SelectOp;
        e->op = tables::SsaOp::Select;
        e->depth = static_cast<uint8_t>(md + 1u);
        e->a = std::move(cond);
        e->b = std::move(tv);
        e->c = std::move(fv);
        return e;
    }

    static std::shared_ptr<const SymExpr> make_ternary(
        tables::SsaOp op,
        std::shared_ptr<const SymExpr> arg0,
        std::shared_ptr<const SymExpr> arg1,
        std::shared_ptr<const SymExpr> arg2) {
        if (!arg0 || !arg1 || !arg2) return nullptr;
        uint8_t md = arg0->depth;
        if (arg1->depth > md) md = arg1->depth;
        if (arg2->depth > md) md = arg2->depth;
        if (md >= kMaxSymExprDepth) return nullptr;
        auto e = std::make_shared<SymExpr>();
        e->kind = Kind::TernaryOp;
        e->op = op;
        e->depth = static_cast<uint8_t>(md + 1u);
        e->a = std::move(arg0);
        e->b = std::move(arg1);
        e->c = std::move(arg2);
        return e;
    }

    static std::shared_ptr<const SymExpr> make_quaternary(
        tables::SsaOp op,
        std::shared_ptr<const SymExpr> arg0,
        std::shared_ptr<const SymExpr> arg1,
        std::shared_ptr<const SymExpr> arg2,
        std::shared_ptr<const SymExpr> arg3) {
        if (!arg0 || !arg1 || !arg2 || !arg3) return nullptr;
        uint8_t md = arg0->depth;
        if (arg1->depth > md) md = arg1->depth;
        if (arg2->depth > md) md = arg2->depth;
        if (arg3->depth > md) md = arg3->depth;
        if (md >= kMaxSymExprDepth) return nullptr;
        auto e = std::make_shared<SymExpr>();
        e->kind = Kind::QuaternaryOp;
        e->op = op;
        e->depth = static_cast<uint8_t>(md + 1u);
        e->a = std::move(arg0);
        e->b = std::move(arg1);
        e->c = std::move(arg2);
        e->d = std::move(arg3);
        return e;
    }

    bool equals(const SymExpr& o) const {
        if (kind != o.kind) return false;
        switch (kind) {
            case Kind::ConstU32:
                return const_u32 == o.const_u32;
            case Kind::BufferAddr:
                return rack == o.rack && desc_idx == o.desc_idx &&
                       symbol_name == o.symbol_name &&
                       byte_offset == o.byte_offset && offset_known == o.offset_known;
            case Kind::BufferScalarLoad:
                return rack == o.rack && desc_idx == o.desc_idx &&
                       symbol_name == o.symbol_name &&
                       byte_offset == o.byte_offset && elem_bytes == o.elem_bytes;
            case Kind::FauSlotRef:
                return fau_offset == o.fau_offset;
            case Kind::UnaryOp:
                return op == o.op && a && o.a && a->equals(*o.a);
            case Kind::BinaryOp:
                return op == o.op && a && o.a && b && o.b &&
                       a->equals(*o.a) && b->equals(*o.b);
            case Kind::SelectOp:
                return a && o.a && b && o.b && c && o.c &&
                       a->equals(*o.a) && b->equals(*o.b) && c->equals(*o.c);
            case Kind::TernaryOp:
                return op == o.op && a && o.a && b && o.b && c && o.c &&
                       a->equals(*o.a) && b->equals(*o.b) && c->equals(*o.c);
            case Kind::QuaternaryOp:
                return op == o.op && a && o.a && b && o.b && c && o.c && d && o.d &&
                       a->equals(*o.a) && b->equals(*o.b) && c->equals(*o.c) && d->equals(*o.d);
        }
        return false;
    }

    std::string format() const {
        std::ostringstream oss;
        switch (kind) {
            case Kind::ConstU32:
                oss << "0x" << std::hex << const_u32 << std::dec;
                break;
            case Kind::BufferAddr:
                oss << "&" << (symbol_name.empty()
                                   ? ("buf:" + std::to_string(rack) + ":" + std::to_string(desc_idx))
                                   : symbol_name);
                if (!offset_known) oss << "+?";
                else if (byte_offset != 0) oss << "+0x" << std::hex << byte_offset << std::dec;
                break;
            case Kind::BufferScalarLoad:
                oss << "*(" << (symbol_name.empty()
                                    ? ("buf:" + std::to_string(rack) + ":" + std::to_string(desc_idx))
                                    : symbol_name)
                    << "+0x" << std::hex << byte_offset << std::dec << ")";
                break;
            case Kind::FauSlotRef:
                oss << "FAU+0x" << std::hex << fau_offset << std::dec;
                break;
            case Kind::UnaryOp:
                oss << tables::ssa_op_to_string(op) << "(" << (a ? a->format() : "?") << ")";
                break;
            case Kind::BinaryOp:
                oss << tables::ssa_op_to_string(op) << "("
                    << (a ? a->format() : "?") << ", "
                    << (b ? b->format() : "?") << ")";
                break;
            case Kind::SelectOp:
                oss << "Select(" << (a ? a->format() : "?") << ", "
                    << (b ? b->format() : "?") << ", "
                    << (c ? c->format() : "?") << ")";
                break;
            case Kind::TernaryOp:
                oss << tables::ssa_op_to_string(op) << "("
                    << (a ? a->format() : "?") << ", "
                    << (b ? b->format() : "?") << ", "
                    << (c ? c->format() : "?") << ")";
                break;
            case Kind::QuaternaryOp:
                oss << tables::ssa_op_to_string(op) << "("
                    << (a ? a->format() : "?") << ", "
                    << (b ? b->format() : "?") << ", "
                    << (c ? c->format() : "?") << ", "
                    << (d ? d->format() : "?") << ")";
                break;
        }
        return oss.str();
    }
};

struct SymExprVal {
    std::shared_ptr<const SymExpr> expr;

    bool operator==(const SymExprVal& o) const {
        if (expr == o.expr) return true;
        if (!expr || !o.expr) return false;
        return expr->equals(*o.expr);
    }
};

/// Bounded-depth symbolic expression flat domain reusing `FlatDomain`,
/// `IntegralAdditiveActionMixin`, and `IntegralBitmaskActionMixin`.
struct SymExprDomain
    : public FlatDomain<SymExprVal>,
      public IntegralAdditiveActionMixin<SymExprDomain, FlatValue<SymExprVal>, IntegralDomain<uint64_t>>,
      public IntegralBitmaskActionMixin<SymExprDomain, FlatValue<SymExprVal>, IntegralDomain<uint64_t>> {
    using Base = FlatDomain<SymExprVal>;
    using Value = typename Base::Value;

    static Value from_expr(std::shared_ptr<const SymExpr> e) {
        if (!e) return top();
        return Value::make(SymExprVal{std::move(e)});
    }

    static Value apply_int_delta(const Value& base, uint64_t delta, bool is_sub = false) {
        if (base.is_bottom()) return bottom();
        if (base.is_top() || !base.value.expr) return top();
        const SymExpr& e = *base.value.expr;
        if (e.kind == SymExpr::Kind::BufferAddr) {
            if (!e.offset_known) return base;
            int64_t sdelta = static_cast<int64_t>(delta);
            int64_t next_off = is_sub ? (e.byte_offset - sdelta) : (e.byte_offset + sdelta);
            return from_expr(SymExpr::make_buffer_addr(e.rack, e.desc_idx, e.symbol_name, next_off, true));
        }
        return top();
    }

    static Value widen_unknown_offset(const Value& base) {
        if (base.is_bottom()) return bottom();
        if (base.is_top() || !base.value.expr) return top();
        const SymExpr& e = *base.value.expr;
        if (e.kind == SymExpr::Kind::BufferAddr) {
            return from_expr(SymExpr::make_buffer_addr(e.rack, e.desc_idx, e.symbol_name, 0, false));
        }
        return top();
    }

    static Value apply_int_mask(const Value& base, uint64_t mask) {
        if (base.is_bottom()) return bottom();
        if (base.is_top() || !base.value.expr) return top();
        const SymExpr& e = *base.value.expr;
        // Preserve 48-bit canonical VA masks (Valhall AND.i32 hi, hi, 0x00FFFFFF).
        if (e.kind == SymExpr::Kind::BufferAddr && (mask == 0x00FFFFFFULL || mask == 0x0000FFFFFFFFFFFFULL)) {
            return base;
        }
        return top();
    }

    static std::string format(const Value& v) {
        if (v.is_bottom()) return "Bottom";
        if (v.is_top()) return "Top";
        return v.value.expr ? v.value.expr->format() : "NullSymExpr";
    }
};

static_assert(ValidateLatticeDomain<SymExprDomain>::valid, "SymExprDomain must satisfy LatticeDomain");

}  // namespace valhall::dataflow
