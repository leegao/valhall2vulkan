#!/usr/bin/env python3
"""
Tier-2 Z3 SMT Miter Harness for All-Paths Bit-Exact Equivalence (Task 3)
========================================================================
Connects:
  - Task 1: tools/verify/panfrost_z3_model.py   (Psi_pan)
  - Task 2: tools/verify/lowering_semantics_extractor.py (T_lowering)
  - Task 3: tools/verify/ssa_z3_eval.py         (Eval_Z3(T_lowering))

For any Valhall instruction (or all 627 specs in specs/kraid/isa-v9-v14.xml),
constructs the Z3 miter:
    Pre(x, m) && (Psi_pan(x, m) != Eval_Z3(T_lowering(x, m)))
and reports UNSAT (proved bit-exact equivalent), SAT (with concrete hex
counterexample), UF_ORACLE (uninterpreted hardware/memory/texture oracle), or UNKNOWN.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import z3

from tools.symbolic_ssa.z3simplifier import dsl

from tools.verify.clang_ast_evaluator import InstLoweringContext, SourceOperandDesc
from tools.verify.lowering_semantics_extractor import LoweringSemanticsExtractor
_BRANCH_MNEMS = {"BRANCH", "JUMP", "JUMP_EX"}
_DISCARD_MNEMS = {"DISCARD"}
_BARRIER_MNEMS = {"BARRIER"}
_STORE_MNEMS = {"STORE", "ST_MULTI", "ST_CHECKED", "ST_CHECKED_IMM", "ST_CVT"}


from tools.verify.panfrost_z3_model import (
    PANFROST_SEMANTICS,
    ValhallZ3State,
    VerificationLogger,
    eval_panfrost_instruction,
    verify_instruction_equivalence,
)
from tools.verify.ssa_z3_eval import eval_ssa_module_z3


@dataclasses.dataclass
class MiterResult:
    xml_index: int
    mnemonic: str
    variant: str
    rule_index: int
    rule_kind: str
    status: str  # UNSAT, SAT, UF_ORACLE, NO_DEST_EQUIV, UNKNOWN
    elapsed_ms: float
    counterexample: Optional[str] = None
    warnings: List[str] = dataclasses.field(default_factory=list)


def _contains_hw_uf(expr: Optional[z3.ExprRef]) -> bool:
    if expr is None:
        return False
    # Fast syntactic check on Z3 AST string for uninterpreted hw_ functions
    return "hw_" in str(expr)


def check_instruction_miter(
    ext: LoweringSemanticsExtractor,
    mnemonic: str,
    variant: str = "",
    modifiers: Optional[Dict[str, Any]] = None,
    fields: Optional[Dict[str, int]] = None,
    num_srcs: Optional[int] = None,
    xml_index: int = -1,
    timeout_ms: int = 3000,
    domain: Optional[Any] = None,
) -> MiterResult:
    """Run Tier-2 Z3 miter on a single Valhall instruction configuration."""
    t0 = time.perf_counter()
    mods = dict(modifiers or {})
    flds = dict(fields or {})
    rule = ext.match_rule(mnemonic, variant)
    assert rule is not None

    if num_srcs is None:
        # Look up from XML if available, else rule.src_types
        num_srcs = len(rule.src_types)
        for x in ext.xml_specs:
            if x["name"] == mnemonic and x["variant"] == variant:
                num_srcs = len(x["xml_srcs"])
                if xml_index < 0:
                    xml_index = x["xml_index"]
                break

    # Allocate non-aliasing symbolic registers:
    # src0 -> r0/r1, src1 -> r2/r3, src2 -> r4/r5, src3 -> r8/r9, dst -> r6/r7
    src_regs = [0, 2, 4, 8]
    dest_reg = 6
    raw_regs: List[Any] = [dsl.BitVec(f"r{i}", 32) for i in range(10)]

    # Determine 64-bit sources/dest
    is_64_dest = rule.dest_type == "u64" or variant.lower() in ("i64", "u64", "s64")
    pan_srcs: List[z3.BitVecRef] = []
    ctx_srcs: List[SourceOperandDesc] = []
    inputs: Dict[str, z3.ExprRef] = {}

    for i in range(num_srcs):
        rbase = src_regs[i]
        st = rule.src_types[i] if i < len(rule.src_types) else "u32"
        is_64_src = st in ("u64", "s64", "i64_by_var") or variant.lower() in ("i64", "u64", "s64")
        is_jump_addr = mnemonic in ("JUMP", "JUMP_EX") and ((num_srcs == 1 and i == 0) or (num_srcs >= 2 and i == 1))
        if (variant.lower() in ("i64", "u64", "s64") or (mnemonic in _STORE_MNEMS and i == 0) or is_jump_addr) and (PANFROST_SEMANTICS[mnemonic].__name__ != "_sem_memory" or (mnemonic in _STORE_MNEMS and i == 0)):
            s_expr = dsl.Concat(raw_regs[rbase + 1], raw_regs[rbase])
        else:
            s_expr = raw_regs[rbase]
        pan_srcs.append(dsl.lower(s_expr))
        inputs[f"src{i}"] = s_expr
        ctx_srcs.append(SourceOperandDesc(reg_num=rbase, is_64bit=is_64_src))

    dst_old = dsl.Concat(raw_regs[dest_reg + 1], raw_regs[dest_reg]) if is_64_dest else raw_regs[dest_reg]
    inputs["dst_old"] = dst_old

    if mnemonic in _STORE_MNEMS and len(pan_srcs) < 2:
        pan_srcs.append(dsl.lower(raw_regs[2]))
        inputs["sr0"] = raw_regs[2]

    # 1. Evaluate Psi_pan (Task 1)
    pan_mods = dict(mods)
    branch_off = int(pan_mods.get("offset", 16 if mnemonic == "BRANCH" else 0))
    if mnemonic == "BRANCH" and "offset" not in pan_mods:
        pan_mods["offset"] = branch_off
    pan_mods.update(flds)
    st_pan = ValhallZ3State.symbolic("miter")
    pan_res, st_pan_out, _ = eval_panfrost_instruction(
        mnemonic, variant, pan_srcs, modifiers=pan_mods, dst_old=dsl.lower(dst_old), state=st_pan
    )

    # 2. Extract & evaluate T_lowering -> SSA-IR -> Z3 (Tasks 2 & 3)
    str_mods = {k: str(v) for k, v in mods.items()}
    ins_ctx = InstLoweringContext(
        mnemonic=mnemonic,
        variant=variant,
        srcs=ctx_srcs,
        has_dest=True,
        dest_reg=dest_reg,
        dest_is_64bit=is_64_dest,
        modifiers=str_mods,
        fields=flds,
        sr_read_present=True,
        sr_read_base=2 if mnemonic in _STORE_MNEMS else 0,
        sr_read_count=1 if mnemonic in _STORE_MNEMS else 4,
        sr_write_present=True,
        sr_write_base=0,
        sr_write_count=4,
    )
    ssa_mod, tr = ext.lower_instruction(ins_ctx, xml_index=xml_index if xml_index >= 0 else None)
    ssa_eval = eval_ssa_module_z3(
        ssa_mod, raw_srcs=raw_regs, dst_old=dsl.lower(dst_old), dest_reg=dest_reg, state=st_pan, branch_offset=branch_off, domain=domain
    )

    if is_64_dest and dest_reg in ssa_eval.reg_writes and (dest_reg + 1) in ssa_eval.reg_writes:
        cand_res: Optional[Any] = dsl.Concat(
            ssa_eval.reg_writes[dest_reg + 1], ssa_eval.reg_writes[dest_reg]
        )
    elif dest_reg in ssa_eval.reg_writes:
        cand_res = ssa_eval.reg_writes[dest_reg]
    else:
        cand_res = None

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # No-dest / side-effect instructions: check known side-effect state transitions
    if pan_res is None and cand_res is None:
        state_checks: List[Tuple[str, z3.ExprRef, z3.ExprRef]] = []
        if mnemonic in _BRANCH_MNEMS:
            state_checks.append(("pc", st_pan_out.pc, ssa_eval.final_state.pc))
        if mnemonic in _DISCARD_MNEMS:
            state_checks.append(("discarded", st_pan_out.discarded, ssa_eval.final_state.discarded))
        if mnemonic in _BARRIER_MNEMS:
            state_checks.append(("barrier_epoch", st_pan_out.barrier_epoch, ssa_eval.final_state.barrier_epoch))
        if mnemonic in _STORE_MNEMS:
            state_checks.append(("mem_global", st_pan_out.mem_global, ssa_eval.final_state.mem_global))

        if not state_checks:
            return MiterResult(
                xml_index=xml_index,
                mnemonic=mnemonic,
                variant=variant,
                rule_index=tr.rule_index,
                rule_kind=tr.rule_kind,
                status="NO_DEST_EQUIV",
                elapsed_ms=elapsed_ms,
                warnings=tr.warnings,
            )

        final_st = "UNSAT"
        for field_name, pan_st_e, cand_st_e in state_checks:
            st_res, st_cex = verify_instruction_equivalence(
                f"{mnemonic}.{field_name}",
                variant,
                dsl.lower(pan_st_e),
                dsl.lower(cand_st_e),
                {k: dsl.lower(v) for k, v in inputs.items()},
                timeout_ms=timeout_ms,
            )
            if st_res == "SAT":
                return MiterResult(
                    xml_index=xml_index,
                    mnemonic=mnemonic,
                    variant=variant,
                    rule_index=tr.rule_index,
                    rule_kind=tr.rule_kind,
                    status="SAT",
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    counterexample=st_cex,
                    warnings=tr.warnings,
                )
            if st_res != "UNSAT":
                final_st = st_res
        return MiterResult(
            xml_index=xml_index,
            mnemonic=mnemonic,
            variant=variant,
            rule_index=tr.rule_index,
            rule_kind=tr.rule_kind,
            status=final_st,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            warnings=tr.warnings,
        )

    if pan_res is None or cand_res is None:
        return MiterResult(
            xml_index=xml_index,
            mnemonic=mnemonic,
            variant=variant,
            rule_index=tr.rule_index,
            rule_kind=tr.rule_kind,
            status="SAT",
            elapsed_ms=elapsed_ms,
            counterexample=f"Destination presence mismatch: pan_res={pan_res is not None}, cand_res={cand_res is not None}",
            warnings=tr.warnings,
        )

    if _contains_hw_uf(pan_res) or _contains_hw_uf(cand_res):
        return MiterResult(
            xml_index=xml_index,
            mnemonic=mnemonic,
            variant=variant,
            rule_index=tr.rule_index,
            rule_kind=tr.rule_kind,
            status="UF_ORACLE",
            elapsed_ms=elapsed_ms,
            warnings=tr.warnings,
        )

    status, cex = verify_instruction_equivalence(
        mnemonic,
        variant,
        dsl.lower(pan_res),
        dsl.lower(cand_res),
        {k: dsl.lower(v) for k, v in inputs.items()},
        timeout_ms=timeout_ms,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return MiterResult(
        xml_index=xml_index,
        mnemonic=mnemonic,
        variant=variant,
        rule_index=tr.rule_index,
        rule_kind=tr.rule_kind,
        status=status,
        elapsed_ms=elapsed_ms,
        counterexample=cex,
        warnings=tr.warnings,
    )


def run_all_627_miter(
    ext: Optional[LoweringSemanticsExtractor] = None,
    timeout_ms: int = 2500,
) -> Dict[str, Any]:
    """Run the Tier-2 Z3 miter across all 627 XML specs in isa-v9-v14.xml."""
    if ext is None:
        ext = LoweringSemanticsExtractor()
    results: List[MiterResult] = []
    counts: Dict[str, int] = {"UNSAT": 0, "SAT": 0, "UF_ORACLE": 0, "NO_DEST_EQUIV": 0, "UNKNOWN": 0}

    t0 = time.perf_counter()
    for x in ext.xml_specs:
        mr = check_instruction_miter(
            ext,
            mnemonic=x["name"],
            variant=x["variant"],
            num_srcs=len(x["xml_srcs"]),
            xml_index=x["xml_index"],
            timeout_ms=timeout_ms,
        )
        results.append(mr)
        counts[mr.status] = counts.get(mr.status, 0) + 1
    total_s = time.perf_counter() - t0
    return {
        "total_specs": len(results),
        "counts": counts,
        "elapsed_s": total_s,
        "results": results,
    }


if __name__ == "__main__":
    summary = run_all_627_miter()
    print(f"All-Paths Miter across {summary['total_specs']} XML specs in {summary['elapsed_s']:.2f}s:")
    print("  Counts:", summary["counts"])
    sat_items = [r for r in summary["results"] if r.status == "SAT"]
    print(f"  Sample SAT counterexamples ({len(sat_items)} total):")
    for r in sat_items:
        first_line = (r.counterexample or "").splitlines()[0] if r.counterexample else ""
        print(f"    [{r.xml_index:3d}] {r.mnemonic}.{r.variant} (rule #{r.rule_index}): {r.counterexample}")
