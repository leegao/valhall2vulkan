#!/usr/bin/env python3
"""
Layer-2 Pluggable Floating-Point Proof-Obligation Discharger.

Architecture:
  Layer 1 (tools/verify/shader_equiv_verifier.py):
    - Whole-CFG topological DAG evaluation in QF_BV / QF_FP.
    - When exact IEEE-754 QF_FP times out on large FP graphs (e.g. transcendental
      or multi-stage matrix/BCn/ASTC shaders), Layer 1 falls back to sound
      uninterpreted-function (UF) abstraction and emits structured FpObligation
      records for any remaining algebraic/rounding/ULP proof obligations.

  Layer 2 (this module):
    - Provides a pluggable backend registry (FpProofDischargerRegistry) that
      routes each FpObligation through registered proof backends in priority order:
        1. Z3FpExactBackend       -- fast bit-exact IEEE-754 Z3 QF_FP check for
                                     small local algebraic/identity lemmas.
        2. OrToolsCpSatBackend    -- TODO(leegao@): quantized/interval-bounded
                                     CP-SAT discharge for fixed-point & index-
                                     range FP lemmas.
        3. ExternalSmtProverBackend -- TODO(leegao@): subprocess adapter for
                                     Bitwuzla / CVC5 / Gappa / RealPaver / Flocq.
"""

from __future__ import annotations

import abc
import dataclasses
import time
from typing import Any, Dict, List, Optional

import z3


@dataclasses.dataclass
class FpObligation:
    """A self-contained floating-point equivalence or rounding proof obligation."""

    obligation_id: str
    output_name: str
    kind: str  # "bit_exact_fp", "ulp_bounded_fp", "fma_contraction", "uf_refinement"
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
    """Abstract interface for a Layer-2 FP proof-obligation discharger backend."""

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


class OrToolsCpSatBackend(FpProofBackend):
    """Pluggable OR-Tools CP-SAT backend for bounded fixed-point / integer-index FP lemmas."""

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
        # TODO(leegao@): Translate interval-bounded FP / quantized fixed-point
        # obligations (e.g., BC6/BC7 endpoint quantization, unorm8/unorm16 pack/unpack,
        # and integer-indexed LUT equivalence) into OR-Tools CP-SAT integer models
        # and verify zero counterexamples within the ULP tolerance.
        return DischargeResult(
            status="UNAVAILABLE",
            backend_name=self.name,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            detail="TODO(leegao@): OR-Tools CP-SAT FP-to-interval lowering not yet implemented",
        )


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

    def submit(self, obligation: FpObligation, timeout_ms: int = 1000) -> DischargeResult:
        """Record an obligation and attempt to discharge it across registered backends."""
        self.obligations.append(obligation)
        last_res = DischargeResult("UNAVAILABLE", "none", 0.0, "No backends registered")
        for backend in self.backends:
            res = backend.discharge(obligation, timeout_ms=timeout_ms)
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
