#!/usr/bin/env python3
"""
Layer-2 Pluggable Floating-Point & Bounded-Integer Proof-Obligation Discharger.

Architecture:
  Layer 1 (tools/verify/shader_equiv_verifier.py):
    - Whole-CFG topological DAG evaluation in QF_BV / QF_FP.
    - When exact IEEE-754 QF_FP times out on large FP graphs, Layer 1 submits
      structured FpObligation records to Layer 2 and falls back through real-
      arithmetic lowering equivalence (`fp_mode="real"`) and sound UF-FP.

  Layer 2 (this module):
    - FpProofDischargerRegistry routes each FpObligation through ordered backends:
        1. Z3FpExactBackend         -- fast bit-exact IEEE-754 Z3 QF_FP check.
        2. OrToolsCpSatBackend      -- translates bounded bitvector / quantized
                                       fixed-point / integer-index obligations into
                                       Google OR-Tools CP-SAT (`CpModel`).
        3. ExternalSmtProverBackend -- TODO(leegao@): subprocess adapter for
                                       Bitwuzla / CVC5 / Gappa / RealPaver / Flocq.
"""

from __future__ import annotations

import abc
import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple

import z3


@dataclasses.dataclass
class FpObligation:
    """A self-contained floating-point or bounded-integer equivalence proof obligation."""

    obligation_id: str
    output_name: str
    kind: str  # "bit_exact_fp", "ulp_bounded_fp", "fma_contraction", "uf_refinement", "bounded_int"
    lhs: z3.ExprRef
    rhs: z3.ExprRef
    preconditions: List[z3.BoolRef] = dataclasses.field(default_factory=list)
    ulp_bound: int = 0
    source_hint: str = ""

    def to_smt2(self) -> str:
        """Serialize this obligation's negation as a standalone SMT-LIB2 query."""
        s = z3.Solver()
        for pre in self.preconditions:
            s.add(pre)
        s.add(self.lhs != self.rhs)
        return s.to_smt2()


@dataclasses.dataclass
class DischargeResult:
    """Outcome of attempting to discharge an FpObligation with a backend."""

    status: str  # "DISCHARGED", "COUNTEREXAMPLE", "UNKNOWN", "UNAVAILABLE"
    backend_name: str
    elapsed_ms: float
    detail: str = ""


class FpProofBackend(abc.ABC):
    """Abstract interface for a Layer-2 proof-obligation discharger backend."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        ...

    @abc.abstractmethod
    def discharge(self, obligation: FpObligation, timeout_ms: int = 1000) -> DischargeResult:
        ...


class Z3FpExactBackend(FpProofBackend):
    """Built-in Z3 QF_FP backend for fast local bit-exact FP obligations."""

    @property
    def name(self) -> str:
        return "z3_fp_exact"

    def is_available(self) -> bool:
        return True

    def discharge(self, obligation: FpObligation, timeout_ms: int = 1000) -> DischargeResult:
        t0 = time.perf_counter()
        if obligation.ulp_bound != 0:
            return DischargeResult(
                status="UNKNOWN",
                backend_name=self.name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                detail="z3_fp_exact only discharges bit-exact (ulp_bound == 0) obligations",
            )
        if obligation.lhs.eq(obligation.rhs):
            return DischargeResult(
                status="DISCHARGED",
                backend_name=self.name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                detail="syntactically identical",
            )
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.set("rlimit", 500_000)
        for pre in obligation.preconditions:
            s.add(pre)
        s.add(obligation.lhs != obligation.rhs)
        rc = s.check()
        elapsed = (time.perf_counter() - t0) * 1000.0
        if rc == z3.unsat:
            return DischargeResult("DISCHARGED", self.name, elapsed, "Z3 QF_FP UNSAT")
        if rc == z3.sat:
            return DischargeResult("COUNTEREXAMPLE", self.name, elapsed, "Z3 QF_FP SAT")
        return DischargeResult("UNKNOWN", self.name, elapsed, "Z3 QF_FP timeout/rlimit")


class _CpSatUnsupported(Exception):
    """Raised when a Z3 AST contains an operator outside the CP-SAT integer fragment."""


class _Z3ToCpSatTranslator:
    """Translates bounded Z3 BitVec / Bool expressions into an OR-Tools CP-SAT CpModel."""

    # Keep CP-SAT domains well inside signed 64-bit integer range to avoid overflow.
    MAX_CP_VAL = (1 << 40) - 1

    def __init__(self, cp_model_mod: Any, var_bounds: Dict[str, Tuple[int, int]]):
        self.cp_model = cp_model_mod
        self.model = cp_model_mod.CpModel()
        self.var_bounds = var_bounds
        self._int_cache: Dict[int, Tuple[Any, int, int]] = {}  # z3_id -> (IntVar, lo, hi)
        self._bool_cache: Dict[int, Any] = {}
        self._named_vars: Dict[str, Any] = {}
        self._tmp_idx = 0

    def _fresh_int(self, lo: int, hi: int, prefix: str = "t") -> Any:
        self._tmp_idx += 1
        lo = max(-self.MAX_CP_VAL, lo)
        hi = min(self.MAX_CP_VAL, hi)
        if lo > hi:
            raise _CpSatUnsupported(f"Empty domain [{lo}, {hi}]")
        return self.model.NewIntVar(lo, hi, f"{prefix}_{self._tmp_idx}")

    def _fresh_bool(self, prefix: str = "b") -> Any:
        self._tmp_idx += 1
        return self.model.NewBoolVar(f"{prefix}_{self._tmp_idx}")

    def lower_bv(self, expr: z3.ExprRef) -> Tuple[Any, int, int]:
        """Lower a Z3 BitVec expression into (cp_IntVar, lo, hi)."""
        eid = expr.get_id()
        if eid in self._int_cache:
            return self._int_cache[eid]

        if isinstance(expr, z3.BitVecNumRef):
            val = expr.as_long()
            if abs(val) > self.MAX_CP_VAL:
                raise _CpSatUnsupported(f"Constant {val} exceeds CP-SAT domain")
            v = self.model.NewConstant(val)
            res = (v, val, val)
            self._int_cache[eid] = res
            return res

        if not z3.is_app(expr):
            raise _CpSatUnsupported("Non-application Z3 expression")

        kind = expr.decl().kind()
        ch = expr.children()

        # Free BitVec symbol
        if kind == z3.Z3_OP_UNINTERPRETED and len(ch) == 0:
            name = expr.decl().name()
            bw = expr.size()
            if name in self.var_bounds:
                lo, hi = self.var_bounds[name]
            elif bw <= 32:
                lo, hi = 0, (1 << bw) - 1
            else:
                raise _CpSatUnsupported(f"Unbounded {bw}-bit variable {name}")
            v = self.model.NewIntVar(lo, hi, name)
            self._named_vars[name] = v
            res = (v, lo, hi)
            self._int_cache[eid] = res
            return res

        bw = expr.size() if isinstance(expr, z3.BitVecRef) else 32
        mod_val = 1 << bw
        if bw > 32:
            raise _CpSatUnsupported(f"Bitwidth {bw} > 32")

        if kind == z3.Z3_OP_BADD and len(ch) == 2:
            va, la, ha = self.lower_bv(ch[0])
            vb, lb, hb = self.lower_bv(ch[1])
            s_lo, s_hi = la + lb, ha + hb
            if 0 <= s_lo and s_hi < mod_val:
                out = self._fresh_int(s_lo, s_hi, "add")
                self.model.Add(out == va + vb)
                res = (out, s_lo, s_hi)
            else:
                raw = self._fresh_int(s_lo, s_hi, "add_raw")
                self.model.Add(raw == va + vb)
                out = self._fresh_int(0, mod_val - 1, "add_mod")
                self.model.AddModuloEquality(out, raw, mod_val)
                res = (out, 0, mod_val - 1)
            self._int_cache[eid] = res
            return res

        if kind == z3.Z3_OP_BSUB and len(ch) == 2:
            va, la, ha = self.lower_bv(ch[0])
            vb, lb, hb = self.lower_bv(ch[1])
            d_lo, d_hi = la - hb, ha - lb
            if 0 <= d_lo and d_hi < mod_val:
                out = self._fresh_int(d_lo, d_hi, "sub")
                self.model.Add(out == va - vb)
                res = (out, d_lo, d_hi)
            else:
                raw = self._fresh_int(d_lo + mod_val, d_hi + mod_val, "sub_raw")
                self.model.Add(raw == va - vb + mod_val)
                out = self._fresh_int(0, mod_val - 1, "sub_mod")
                self.model.AddModuloEquality(out, raw, mod_val)
                res = (out, 0, mod_val - 1)
            self._int_cache[eid] = res
            return res

        if kind == z3.Z3_OP_BMUL and len(ch) == 2:
            va, la, ha = self.lower_bv(ch[0])
            vb, lb, hb = self.lower_bv(ch[1])
            corners = [la * lb, la * hb, ha * lb, ha * hb]
            p_lo, p_hi = min(corners), max(corners)
            if abs(p_lo) > self.MAX_CP_VAL or abs(p_hi) > self.MAX_CP_VAL:
                raise _CpSatUnsupported("Multiplication range exceeds CP-SAT safe limit")
            if 0 <= p_lo and p_hi < mod_val:
                out = self._fresh_int(p_lo, p_hi, "mul")
                self.model.AddMultiplicationEquality(out, [va, vb])
                res = (out, p_lo, p_hi)
            else:
                raw = self._fresh_int(p_lo, p_hi, "mul_raw")
                self.model.AddMultiplicationEquality(raw, [va, vb])
                out = self._fresh_int(0, mod_val - 1, "mul_mod")
                self.model.AddModuloEquality(out, raw, mod_val)
                res = (out, 0, mod_val - 1)
            self._int_cache[eid] = res
            return res

        if kind == z3.Z3_OP_BAND and len(ch) == 2:
            # Handle masking by constant (2^k - 1) -> x mod 2^k
            c_node, x_node = (ch[1], ch[0]) if isinstance(ch[1], z3.BitVecNumRef) else (ch[0], ch[1])
            if isinstance(c_node, z3.BitVecNumRef):
                mask = c_node.as_long()
                if mask == 0:
                    res = (self.model.NewConstant(0), 0, 0)
                    self._int_cache[eid] = res
                    return res
                if (mask & (mask + 1)) == 0:
                    vx, lx, hx = self.lower_bv(x_node)
                    if 0 <= lx and hx <= mask:
                        res = (vx, lx, hx)
                    else:
                        out = self._fresh_int(0, mask, "and_mask")
                        self.model.AddModuloEquality(out, vx, mask + 1)
                        res = (out, 0, mask)
                    self._int_cache[eid] = res
                    return res
            raise _CpSatUnsupported("General bitwise AND outside (2^k - 1) mask")

        if kind == z3.Z3_OP_BLSHR and len(ch) == 2 and isinstance(ch[1], z3.BitVecNumRef):
            sh = ch[1].as_long()
            vx, lx, hx = self.lower_bv(ch[0])
            if sh >= bw:
                res = (self.model.NewConstant(0), 0, 0)
            elif sh == 0:
                res = (vx, lx, hx)
            else:
                div = 1 << sh
                out = self._fresh_int(lx // div, hx // div, "lshr")
                self.model.AddDivisionEquality(out, vx, div)
                res = (out, lx // div, hx // div)
            self._int_cache[eid] = res
            return res

        if kind == z3.Z3_OP_BSHL and len(ch) == 2 and isinstance(ch[1], z3.BitVecNumRef):
            sh = ch[1].as_long()
            vx, lx, hx = self.lower_bv(ch[0])
            if sh >= bw:
                res = (self.model.NewConstant(0), 0, 0)
            elif sh == 0:
                res = (vx, lx, hx)
            else:
                mult = 1 << sh
                if hx * mult < mod_val and hx * mult <= self.MAX_CP_VAL:
                    out = self._fresh_int(lx * mult, hx * mult, "shl")
                    self.model.Add(out == vx * mult)
                    res = (out, lx * mult, hx * mult)
                else:
                    raise _CpSatUnsupported("Overflowing left shift")
            self._int_cache[eid] = res
            return res

        if kind == z3.Z3_OP_ZERO_EXT and len(ch) == 1:
            res = self.lower_bv(ch[0])
            self._int_cache[eid] = res
            return res

        if kind == z3.Z3_OP_EXTRACT and len(ch) == 1:
            hi_bit = expr.params()[0]
            lo_bit = expr.params()[1]
            vx, lx, hx = self.lower_bv(ch[0])
            width = hi_bit - lo_bit + 1
            mask = (1 << width) - 1
            if lo_bit == 0:
                if 0 <= lx and hx <= mask:
                    res = (vx, lx, hx)
                else:
                    out = self._fresh_int(0, mask, "ext")
                    self.model.AddModuloEquality(out, vx, mask + 1)
                    res = (out, 0, mask)
            else:
                shifted = self._fresh_int(0, hx >> lo_bit, "ext_sh")
                self.model.AddDivisionEquality(shifted, vx, 1 << lo_bit)
                out = self._fresh_int(0, mask, "ext_m")
                self.model.AddModuloEquality(out, shifted, mask + 1)
                res = (out, 0, mask)
            self._int_cache[eid] = res
            return res

        if kind == z3.Z3_OP_ITE and len(ch) == 3:
            bc = self.lower_bool(ch[0])
            vt, lt, ht = self.lower_bv(ch[1])
            vf, lf, hf = self.lower_bv(ch[2])
            out = self._fresh_int(min(lt, lf), max(ht, hf), "ite")
            self.model.Add(out == vt).OnlyEnforceIf(bc)
            self.model.Add(out == vf).OnlyEnforceIf(bc.Not())
            res = (out, min(lt, lf), max(ht, hf))
            self._int_cache[eid] = res
            return res

        raise _CpSatUnsupported(f"Unsupported Z3 BitVec op kind {kind} ({expr.decl().name()})")

    def lower_bool(self, expr: z3.ExprRef) -> Any:
        """Lower a Z3 Bool expression into a CP-SAT BoolVar."""
        eid = expr.get_id()
        if eid in self._bool_cache:
            return self._bool_cache[eid]

        if z3.is_true(expr):
            b = self._fresh_bool("true")
            self.model.Add(b == 1)
            self._bool_cache[eid] = b
            return b
        if z3.is_false(expr):
            b = self._fresh_bool("false")
            self.model.Add(b == 0)
            self._bool_cache[eid] = b
            return b

        kind = expr.decl().kind()
        ch = expr.children()

        if kind == z3.Z3_OP_UNINTERPRETED and len(ch) == 0:
            b = self.model.NewBoolVar(expr.decl().name())
            self._named_vars[expr.decl().name()] = b
            self._bool_cache[eid] = b
            return b

        if kind == z3.Z3_OP_NOT and len(ch) == 1:
            b = self.lower_bool(ch[0]).Not()
            self._bool_cache[eid] = b
            return b

        if kind == z3.Z3_OP_AND:
            subs = [self.lower_bool(c) for c in ch]
            out = self._fresh_bool("and")
            self.model.AddBoolAnd(subs).OnlyEnforceIf(out)
            self.model.AddBoolOr([s.Not() for s in subs]).OnlyEnforceIf(out.Not())
            self._bool_cache[eid] = out
            return out

        if kind == z3.Z3_OP_OR:
            subs = [self.lower_bool(c) for c in ch]
            out = self._fresh_bool("or")
            self.model.AddBoolOr(subs).OnlyEnforceIf(out)
            self.model.AddBoolAnd([s.Not() for s in subs]).OnlyEnforceIf(out.Not())
            self._bool_cache[eid] = out
            return out

        if kind in (z3.Z3_OP_EQ, z3.Z3_OP_ULT, z3.Z3_OP_ULEQ, z3.Z3_OP_UGT, z3.Z3_OP_UGEQ) and len(ch) == 2:
            if isinstance(ch[0], z3.BoolRef) and isinstance(ch[1], z3.BoolRef) and kind == z3.Z3_OP_EQ:
                ba, bb = self.lower_bool(ch[0]), self.lower_bool(ch[1])
                out = self._fresh_bool("beq")
                self.model.Add(ba == bb).OnlyEnforceIf(out)
                self.model.Add(ba != bb).OnlyEnforceIf(out.Not())
                self._bool_cache[eid] = out
                return out
            va, _, _ = self.lower_bv(ch[0])
            vb, _, _ = self.lower_bv(ch[1])
            out = self._fresh_bool("cmp")
            if kind == z3.Z3_OP_EQ:
                self.model.Add(va == vb).OnlyEnforceIf(out)
                self.model.Add(va != vb).OnlyEnforceIf(out.Not())
            elif kind == z3.Z3_OP_ULT:
                self.model.Add(va < vb).OnlyEnforceIf(out)
                self.model.Add(va >= vb).OnlyEnforceIf(out.Not())
            elif kind == z3.Z3_OP_ULEQ:
                self.model.Add(va <= vb).OnlyEnforceIf(out)
                self.model.Add(va > vb).OnlyEnforceIf(out.Not())
            elif kind == z3.Z3_OP_UGT:
                self.model.Add(va > vb).OnlyEnforceIf(out)
                self.model.Add(va <= vb).OnlyEnforceIf(out.Not())
            elif kind == z3.Z3_OP_UGEQ:
                self.model.Add(va >= vb).OnlyEnforceIf(out)
                self.model.Add(va < vb).OnlyEnforceIf(out.Not())
            self._bool_cache[eid] = out
            return out

        raise _CpSatUnsupported(f"Unsupported Z3 Bool op kind {kind} ({expr.decl().name()})")


def _extract_precondition_bounds(preconditions: List[z3.BoolRef]) -> Dict[str, Tuple[int, int]]:
    """Extract tight unsigned bounds [lo, hi] for free BitVec variables from preconditions."""
    bounds: Dict[str, Tuple[int, int]] = {}

    def _visit(e: z3.ExprRef) -> None:
        if not z3.is_app(e):
            return
        k = e.decl().kind()
        ch = e.children()
        if k == z3.Z3_OP_AND:
            for c in ch:
                _visit(c)
        elif k in (z3.Z3_OP_ULT, z3.Z3_OP_ULEQ) and len(ch) == 2:
            if z3.is_app(ch[0]) and ch[0].decl().kind() == z3.Z3_OP_UNINTERPRETED and len(ch[0].children()) == 0:
                if isinstance(ch[1], z3.BitVecNumRef):
                    name = ch[0].decl().name()
                    lim = ch[1].as_long() - (1 if k == z3.Z3_OP_ULT else 0)
                    lo, hi = bounds.get(name, (0, (1 << ch[0].size()) - 1))
                    bounds[name] = (lo, min(hi, lim))
        elif k in (z3.Z3_OP_UGT, z3.Z3_OP_UGEQ) and len(ch) == 2:
            if z3.is_app(ch[0]) and ch[0].decl().kind() == z3.Z3_OP_UNINTERPRETED and len(ch[0].children()) == 0:
                if isinstance(ch[1], z3.BitVecNumRef):
                    name = ch[0].decl().name()
                    lim = ch[1].as_long() + (1 if k == z3.Z3_OP_UGT else 0)
                    lo, hi = bounds.get(name, (0, (1 << ch[0].size()) - 1))
                    bounds[name] = (max(lo, lim), hi)

    for pre in preconditions:
        _visit(pre)
    return bounds


class OrToolsCpSatBackend(FpProofBackend):
    """Google OR-Tools CP-SAT backend for bounded bitvector / quantized fixed-point obligations."""

    @property
    def name(self) -> str:
        return "ortools_cpsat"

    def is_available(self) -> bool:
        try:
            import ortools.sat.python.cp_model  # noqa: F401
            return True
        except Exception:
            return False

    def discharge(self, obligation: FpObligation, timeout_ms: int = 1000) -> DischargeResult:
        t0 = time.perf_counter()
        if not self.is_available():
            return DischargeResult(
                status="UNAVAILABLE",
                backend_name=self.name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                detail="ortools package not installed",
            )
        from ortools.sat.python import cp_model

        try:
            var_bounds = _extract_precondition_bounds(obligation.preconditions)
            tr = _Z3ToCpSatTranslator(cp_model, var_bounds)
            for pre in obligation.preconditions:
                b_pre = tr.lower_bool(pre)
                tr.model.Add(b_pre == 1)
            lhs_v, _, _ = tr.lower_bv(obligation.lhs)
            rhs_v, _, _ = tr.lower_bv(obligation.rhs)
            if obligation.ulp_bound <= 0:
                tr.model.Add(lhs_v != rhs_v)
            else:
                # |lhs - rhs| > ulp_bound  <=>  (lhs - rhs > ulp) OR (rhs - lhs > ulp)
                b_hi = tr._fresh_bool("ulp_hi")
                b_lo = tr._fresh_bool("ulp_lo")
                tr.model.Add(lhs_v - rhs_v > obligation.ulp_bound).OnlyEnforceIf(b_hi)
                tr.model.Add(rhs_v - lhs_v > obligation.ulp_bound).OnlyEnforceIf(b_lo)
                tr.model.AddBoolOr([b_hi, b_lo])
        except _CpSatUnsupported as exc:
            return DischargeResult(
                status="UNKNOWN",
                backend_name=self.name,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                detail=f"Outside CP-SAT fragment: {exc}",
            )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.05, timeout_ms / 1000.0)
        solver.parameters.num_search_workers = 1
        status = solver.Solve(tr.model)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if status == cp_model.INFEASIBLE:
            return DischargeResult("DISCHARGED", self.name, elapsed, "OR-Tools CP-SAT INFEASIBLE")
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            witness = ", ".join(f"{k}={solver.Value(v)}" for k, v in list(tr._named_vars.items())[:6])
            return DischargeResult("COUNTEREXAMPLE", self.name, elapsed, f"OR-Tools CP-SAT SAT ({witness})")
        return DischargeResult("UNKNOWN", self.name, elapsed, f"OR-Tools CP-SAT status={status}")


class ExternalSmtProverBackend(FpProofBackend):
    """Pluggable external SMT / FP prover adapter (Bitwuzla / CVC5 / Gappa)."""

    def __init__(self, binary_name: str = "bitwuzla"):
        self._binary_name = binary_name

    @property
    def name(self) -> str:
        return f"external_{self._binary_name}"

    def is_available(self) -> bool:
        return False

    def discharge(self, obligation: FpObligation, timeout_ms: int = 1000) -> DischargeResult:
        t0 = time.perf_counter()
        # TODO(leegao@): Invoke external SMT-LIB2 QF_FPBV solver (Bitwuzla / CVC5)
        # or Gappa interval-error prover on `obligation.to_smt2()` with strict
        # subprocess timeout and parse `unsat` / `sat` / `unknown` responses.
        return DischargeResult(
            status="UNAVAILABLE",
            backend_name=self.name,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            detail=f"TODO(leegao@): External prover '{self._binary_name}' subprocess integration not yet implemented",
        )


class FpProofDischargerRegistry:
    """Routes FpObligations across ordered Layer-2 proof backends."""

    def __init__(self, backends: Optional[List[FpProofBackend]] = None):
        if backends is None:
            self.backends: List[FpProofBackend] = [
                Z3FpExactBackend(),
                OrToolsCpSatBackend(),
                ExternalSmtProverBackend("bitwuzla"),
            ]
        else:
            self.backends = list(backends)
        self.obligations: List[FpObligation] = []
        self.results: Dict[str, DischargeResult] = {}

    def register_backend(self, backend: FpProofBackend) -> None:
        self.backends.append(backend)

    def submit(
        self,
        obligation: FpObligation,
        timeout_ms: int = 1000,
        verbose: bool = False,
        log_fn=None,
    ) -> DischargeResult:
        """Record an obligation and attempt to discharge it across registered backends."""
        self.obligations.append(obligation)
        last_res = DischargeResult("UNAVAILABLE", "none", 0.0, "No backends registered")
        for backend in self.backends:
            res = backend.discharge(obligation, timeout_ms=timeout_ms)
            if verbose:
                msg = (
                    f"[equiv][layer2] ob={obligation.obligation_id} backend={backend.name} "
                    f"status={res.status} elapsed={res.elapsed_ms:.1f}ms detail={res.detail}"
                )
                if log_fn is not None:
                    log_fn(msg)
                else:
                    import sys as _sys
                    print(msg, file=_sys.stderr)
            last_res = res
            if res.status in ("DISCHARGED", "COUNTEREXAMPLE"):
                self.results[obligation.obligation_id] = res
                return res
        self.results[obligation.obligation_id] = last_res
        return last_res

    def summary(self) -> Dict[str, Any]:
        discharged = sum(1 for r in self.results.values() if r.status == "DISCHARGED")
        cex = sum(1 for r in self.results.values() if r.status == "COUNTEREXAMPLE")
        pending = len(self.obligations) - discharged - cex
        return {
            "total": len(self.obligations),
            "discharged": discharged,
            "counterexamples": cex,
            "pending": pending,
            "backends": [b.name for b in self.backends],
        }
