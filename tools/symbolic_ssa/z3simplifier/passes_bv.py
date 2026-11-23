#!/usr/bin/env python3
"""
BitVector and Boolean wrapper simplification passes for `z3simplifier`.

Addresses:
  - Issue 1 (Boolean & U32 `BVEC64` wrapper bloat):
      * `Extract(w-1, 0, ZeroExt(k, x))` -> `x` when `x.size() == w`
      * `Extract(w-1, 0, Concat(hi, lo))` -> `lo` when `lo.size() == w`
      * `Extract(w-1, 0, x)` -> `x` when `x.size() == w`
      * `Extract(hi, lo, If(c, t, f))` -> `If(c, Extract(hi, lo, t), Extract(hi, lo, f))`
        when `t` and `f` are BV constants (collapses `Extract(31, 0, If(b, 1, 0))`)
      * `If(b, 1, 0) != 0` and `0 != If(b, 1, 0)` -> `b`
      * `If(b, 1, 0) == 0` and `0 == If(b, 1, 0)` -> `Not(b)`
      * `If(b, 0, 1) != 0` -> `Not(b)`
      * `If(c, x, x)` -> `x`; `If(True, t, f)` -> `t`; `If(False, t, f)` -> `f`
      * `Not(Not(b))` -> `b`; `And(True, b)` -> `b`; `Or(False, b)` -> `b`
  - Issue 6 (Integer `BitVec` mul/shift/address normalization):
      * `x * 0` -> `0`, `x * 1` -> `x`
      * `x * (2^k)` -> `x << k` (eliminates 32/64-bit multiplier bit-blasting on
        UBO/SSBO stride multiplies)
      * `x + 0` -> `x`, `x - 0` -> `x`, `x - x` -> `0`
      * `(x + c1) + c2` -> `x + (c1 + c2)`
      * `(x << c1) << c2` -> `x << (c1 + c2)` (when `c1 + c2 < width`)
      * `x << 0` -> `x`, `LShR(x, 0)` -> `x`, `x >> 0` -> `x`
      * `x & 0` -> `0`, `x & ~0` -> `x`, `x | 0` -> `x`, `x ^ 0` -> `x`, `x ^ x` -> `0`
"""

from __future__ import annotations

from typing import Optional, Tuple

import z3

from tools.symbolic_ssa.z3simplifier.ast_utils import rewrite_dag
from tools.symbolic_ssa.z3simplifier.context import SimplificationContext
from tools.symbolic_ssa.z3simplifier.pass_base import Z3TermPass


def _is_bv_const(e: z3.ExprRef) -> Optional[int]:
    if isinstance(e, z3.BitVecNumRef):
        return int(e.as_long())
    return None


def _is_power_of_two(n: int) -> Optional[int]:
    if n > 0 and (n & (n - 1)) == 0:
        return n.bit_length() - 1
    return None


def _bv_wrapper_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    if not z3.is_app(node):
        return None
    kind = node.decl().kind()

    if kind == z3.Z3_OP_EXTRACT:
        hi, lo = node.params()
        width = hi - lo + 1
        src = node.arg(0)
        if lo == 0 and isinstance(src, z3.BitVecRef) and src.size() == width:
            return src
        if isinstance(src, z3.BitVecNumRef):
            v = (src.as_long() >> lo) & ((1 << width) - 1)
            return z3.BitVecVal(v, width)
        if z3.is_app(src):
            sk = src.decl().kind()
            if sk == z3.Z3_OP_ZERO_EXT and lo == 0:
                inner = src.arg(0)
                if isinstance(inner, z3.BitVecRef):
                    if inner.size() == width:
                        return inner
                    if inner.size() > width:
                        return z3.Extract(hi, 0, inner)
            elif sk == z3.Z3_OP_SIGN_EXT and lo == 0:
                inner = src.arg(0)
                if isinstance(inner, z3.BitVecRef) and inner.size() == width:
                    return inner
            elif sk == z3.Z3_OP_CONCAT and lo == 0:
                # In Z3 Concat(a_0, a_1, ..., a_{k-1}), a_{k-1} is the LSB part.
                lsb = src.arg(src.num_args() - 1)
                if isinstance(lsb, z3.BitVecRef) and lsb.size() == width:
                    return lsb
                if isinstance(lsb, z3.BitVecRef) and lsb.size() > width:
                    return z3.Extract(hi, 0, lsb)
            elif sk == z3.Z3_OP_ITE and src.num_args() == 3:
                c, t_arm, f_arm = src.arg(0), src.arg(1), src.arg(2)
                t_c = _is_bv_const(t_arm)
                f_c = _is_bv_const(f_arm)
                if t_c is not None and f_c is not None:
                    mask = (1 << width) - 1
                    nt = z3.BitVecVal((t_c >> lo) & mask, width)
                    nf = z3.BitVecVal((f_c >> lo) & mask, width)
                    if nt.eq(nf):
                        return nt
                    return z3.If(c, nt, nf)

    elif kind == z3.Z3_OP_ZERO_EXT and node.num_args() == 1:
        src = node.arg(0)
        if isinstance(src, z3.BitVecNumRef):
            k = node.params()[0]
            return z3.BitVecVal(src.as_long(), src.size() + k)

    elif kind == z3.Z3_OP_SIGN_EXT and node.num_args() == 1:
        src = node.arg(0)
        if isinstance(src, z3.BitVecNumRef):
            k = node.params()[0]
            w = src.size()
            val = src.as_signed_long()
            nw = w + k
            return z3.BitVecVal(val & ((1 << nw) - 1), nw)

    elif kind == z3.Z3_OP_CONCAT and node.num_args() >= 2:
        if all(isinstance(node.arg(i), z3.BitVecNumRef) for i in range(node.num_args())):
            acc = 0
            tot_w = 0
            for i in range(node.num_args()):
                arg = node.arg(i)
                aw = arg.size()
                acc = (acc << aw) | arg.as_long()
                tot_w += aw
            return z3.BitVecVal(acc, tot_w)

    elif kind == z3.Z3_OP_BV2INT and node.num_args() == 1:
        src = node.arg(0)
        if isinstance(src, z3.BitVecNumRef):
            return z3.IntVal(src.as_long())

    elif kind == z3.Z3_OP_INT2BV and node.num_args() == 1:
        src = node.arg(0)
        if isinstance(src, z3.IntNumRef) and isinstance(node, z3.BitVecRef):
            w = node.size()
            return z3.BitVecVal(src.as_long() & ((1 << w) - 1), w)

    elif kind == z3.Z3_OP_TO_REAL and node.num_args() == 1:
        src = node.arg(0)
        if isinstance(src, z3.IntNumRef):
            return z3.RealVal(src.as_long())

    elif kind == z3.Z3_OP_TO_INT and node.num_args() == 1:
        src = node.arg(0)
        if isinstance(src, z3.RatNumRef):
            num = src.numerator_as_long()
            den = src.denominator_as_long()
            return z3.IntVal(num // den)

    elif kind in (
        z3.Z3_OP_SLT, z3.Z3_OP_SLEQ, z3.Z3_OP_SGT, z3.Z3_OP_SGEQ,
        z3.Z3_OP_ULT, z3.Z3_OP_ULEQ, z3.Z3_OP_UGT, z3.Z3_OP_UGEQ,
    ) and node.num_args() == 2:
        a, b = node.arg(0), node.arg(1)
        if isinstance(a, z3.BitVecNumRef) and isinstance(b, z3.BitVecNumRef):
            if kind in (z3.Z3_OP_SLT, z3.Z3_OP_SLEQ, z3.Z3_OP_SGT, z3.Z3_OP_SGEQ):
                va, vb = a.as_signed_long(), b.as_signed_long()
                if kind == z3.Z3_OP_SLT: return z3.BoolVal(va < vb)
                if kind == z3.Z3_OP_SLEQ: return z3.BoolVal(va <= vb)
                if kind == z3.Z3_OP_SGT: return z3.BoolVal(va > vb)
                return z3.BoolVal(va >= vb)
            else:
                va, vb = a.as_long(), b.as_long()
                if kind == z3.Z3_OP_ULT: return z3.BoolVal(va < vb)
                if kind == z3.Z3_OP_ULEQ: return z3.BoolVal(va <= vb)
                if kind == z3.Z3_OP_UGT: return z3.BoolVal(va > vb)
                return z3.BoolVal(va >= vb)

    elif kind == z3.Z3_OP_ITE and node.num_args() == 3:
        c, t_arm, f_arm = node.arg(0), node.arg(1), node.arg(2)
        if z3.is_true(c):
            return t_arm
        if z3.is_false(c):
            return f_arm
        if t_arm.eq(f_arm):
            return t_arm
        if z3.is_true(t_arm) and z3.is_false(f_arm):
            return c
        if z3.is_false(t_arm) and z3.is_true(f_arm):
            return z3.Not(c)

    elif kind in (z3.Z3_OP_EQ, z3.Z3_OP_DISTINCT) and node.num_args() == 2:
        a, b = node.arg(0), node.arg(1)
        ca, cb = _is_bv_const(a), _is_bv_const(b)
        is_eq = kind == z3.Z3_OP_EQ
        if ca is not None and cb is not None:
            return z3.BoolVal(ca == cb if is_eq else ca != cb)
        # Unwrap `If(cond, 1, 0) != 0` -> `cond`, `If(cond, 1, 0) == 0` -> `Not(cond)`
        other, zero_const = (b, ca) if ca == 0 else ((a, cb) if cb == 0 else (None, None))
        if other is not None and zero_const == 0 and z3.is_app(other) and other.decl().kind() == z3.Z3_OP_ITE:
            oc, ot, of = other.arg(0), other.arg(1), other.arg(2)
            ot_c, of_c = _is_bv_const(ot), _is_bv_const(of)
            if ot_c is not None and of_c is not None:
                t_nz = ot_c != 0
                f_nz = of_c != 0
                if t_nz and not f_nz:
                    return z3.Not(oc) if is_eq else oc
                if not t_nz and f_nz:
                    return oc if is_eq else z3.Not(oc)
                if t_nz and f_nz:
                    return z3.BoolVal(not is_eq)
                return z3.BoolVal(is_eq)
        if a.eq(b):
            return z3.BoolVal(is_eq)

    elif kind == z3.Z3_OP_NOT and node.num_args() == 1:
        inner = node.arg(0)
        if z3.is_true(inner):
            return z3.BoolVal(False)
        if z3.is_false(inner):
            return z3.BoolVal(True)
        if z3.is_app(inner):
            ik = inner.decl().kind()
            if ik == z3.Z3_OP_NOT:
                return inner.arg(0)
            if ik == z3.Z3_OP_EQ and inner.num_args() == 2:
                # Simplify `Not(If(b,1,0) == 0)` if created directly
                a, b = inner.arg(0), inner.arg(1)
                ca, cb = _is_bv_const(a), _is_bv_const(b)
                other = b if ca == 0 else (a if cb == 0 else None)
                if other is not None and z3.is_app(other) and other.decl().kind() == z3.Z3_OP_ITE:
                    oc, ot, of = other.arg(0), other.arg(1), other.arg(2)
                    ot_c, of_c = _is_bv_const(ot), _is_bv_const(of)
                    if ot_c is not None and of_c is not None:
                        if ot_c != 0 and of_c == 0:
                            return oc
                        if ot_c == 0 and of_c != 0:
                            return z3.Not(oc)

    elif kind == z3.Z3_OP_AND:
        kept = []
        changed = False
        for i in range(node.num_args()):
            ch = node.arg(i)
            if z3.is_false(ch):
                return z3.BoolVal(False)
            if z3.is_true(ch):
                changed = True
                continue
            kept.append(ch)
        if changed:
            if not kept:
                return z3.BoolVal(True)
            return kept[0] if len(kept) == 1 else z3.And(*kept)

    elif kind == z3.Z3_OP_OR:
        kept = []
        changed = False
        for i in range(node.num_args()):
            ch = node.arg(i)
            if z3.is_true(ch):
                return z3.BoolVal(True)
            if z3.is_false(ch):
                changed = True
                continue
            kept.append(ch)
        if changed:
            if not kept:
                return z3.BoolVal(False)
            return kept[0] if len(kept) == 1 else z3.Or(*kept)

    return None


def _bv_algebraic_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    if not z3.is_app(node) or not isinstance(node, z3.BitVecRef):
        return None
    kind = node.decl().kind()
    w = node.size()
    mask = (1 << w) - 1

    if kind == z3.Z3_OP_BMUL and node.num_args() == 2:
        a, b = node.arg(0), node.arg(1)
        ca, cb = _is_bv_const(a), _is_bv_const(b)
        if ca is not None and cb is not None:
            return z3.BitVecVal((ca * cb) & mask, w)
        if ca == 0 or cb == 0:
            return z3.BitVecVal(0, w)
        if ca == 1:
            return b
        if cb == 1:
            return a
        if cb is not None:
            sh = _is_power_of_two(cb)
            if sh is not None and sh < w:
                return a << z3.BitVecVal(sh, w)
        if ca is not None:
            sh = _is_power_of_two(ca)
            if sh is not None and sh < w:
                return b << z3.BitVecVal(sh, w)

    elif kind == z3.Z3_OP_BADD and node.num_args() == 2:
        a, b = node.arg(0), node.arg(1)
        ca, cb = _is_bv_const(a), _is_bv_const(b)
        if ca is not None and cb is not None:
            return z3.BitVecVal((ca + cb) & mask, w)
        if ca == 0:
            return b
        if cb == 0:
            return a
        # Constant folding across nested additions: (x + c1) + c2 -> x + (c1 + c2)
        if cb is not None and z3.is_app(a) and a.decl().kind() == z3.Z3_OP_BADD and a.num_args() == 2:
            a0, a1 = a.arg(0), a.arg(1)
            ca1 = _is_bv_const(a1)
            if ca1 is not None:
                c_sum = (ca1 + cb) & mask
                return a0 if c_sum == 0 else (a0 + z3.BitVecVal(c_sum, w))

    elif kind == z3.Z3_OP_BSUB and node.num_args() == 2:
        a, b = node.arg(0), node.arg(1)
        ca, cb = _is_bv_const(a), _is_bv_const(b)
        if ca is not None and cb is not None:
            return z3.BitVecVal((ca - cb) & mask, w)
        if cb == 0:
            return a
        if a.eq(b):
            return z3.BitVecVal(0, w)

    elif kind == z3.Z3_OP_BSHL and node.num_args() == 2:
        a, b = node.arg(0), node.arg(1)
        ca, cb = _is_bv_const(a), _is_bv_const(b)
        if ca is not None and cb is not None:
            return z3.BitVecVal((ca << cb) & mask if cb < w else 0, w)
        if cb == 0:
            return a
        if cb is not None and cb >= w:
            return z3.BitVecVal(0, w)
        if cb is not None and z3.is_app(a) and a.decl().kind() == z3.Z3_OP_BSHL and a.num_args() == 2:
            ca1 = _is_bv_const(a.arg(1))
            if ca1 is not None:
                total = ca1 + cb
                if total >= w:
                    return z3.BitVecVal(0, w)
                return a.arg(0) << z3.BitVecVal(total, w)

    elif kind in (z3.Z3_OP_BLSHR, z3.Z3_OP_BASHR) and node.num_args() == 2:
        a, b = node.arg(0), node.arg(1)
        cb = _is_bv_const(b)
        if cb == 0:
            return a

    elif kind == z3.Z3_OP_BAND and node.num_args() == 2:
        a, b = node.arg(0), node.arg(1)
        ca, cb = _is_bv_const(a), _is_bv_const(b)
        if ca is not None and cb is not None:
            return z3.BitVecVal(ca & cb, w)
        if ca == 0 or cb == 0:
            return z3.BitVecVal(0, w)
        if ca == mask:
            return b
        if cb == mask:
            return a
        if a.eq(b):
            return a

    elif kind == z3.Z3_OP_BOR and node.num_args() == 2:
        a, b = node.arg(0), node.arg(1)
        ca, cb = _is_bv_const(a), _is_bv_const(b)
        if ca is not None and cb is not None:
            return z3.BitVecVal(ca | cb, w)
        if ca == 0:
            return b
        if cb == 0:
            return a
        if ca == mask or cb == mask:
            return z3.BitVecVal(mask, w)
        if a.eq(b):
            return a

    elif kind == z3.Z3_OP_BXOR and node.num_args() == 2:
        a, b = node.arg(0), node.arg(1)
        ca, cb = _is_bv_const(a), _is_bv_const(b)
        if ca is not None and cb is not None:
            return z3.BitVecVal(ca ^ cb, w)
        if ca == 0:
            return b
        if cb == 0:
            return a
        if a.eq(b):
            return z3.BitVecVal(0, w)

    return None


class BvWrapperPeepholePass(Z3TermPass):
    """Eliminates redundant BV64/BV32/Bool wrapper layers (`Extract`/`Concat`/`ZeroExt`/`If(b,1,0)`)."""

    name = "bv_wrapper_peephole"
    is_equivalence_preserving = True

    def rewrite_term(
        self,
        expr: z3.ExprRef,
        ctx: SimplificationContext,
    ) -> Tuple[z3.ExprRef, int]:
        return rewrite_dag(expr, _bv_wrapper_rule)


class BvAlgebraicPass(Z3TermPass):
    """Normalizes BitVector arithmetic/shift/bitwise expressions (incl. `x * 2^k -> x << k`)."""

    name = "bv_algebraic"
    is_equivalence_preserving = True

    def rewrite_term(
        self,
        expr: z3.ExprRef,
        ctx: SimplificationContext,
    ) -> Tuple[z3.ExprRef, int]:
        return rewrite_dag(expr, _bv_algebraic_rule)
