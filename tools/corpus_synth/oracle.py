"""
Coverage Oracle for Greedy Corpus Synthesis.

Supports two coverage targets:
  - `valhall-isa`: compiles the candidate SPIR-V through `spirv2valhall`
    (`libMali`) and extracts the set of unique Valhall hardware instruction
    mnemonics (with type/modifier suffixes, e.g. `FMA.f32`, `CSEL.u32.ne`,
    `LSHIFT_AND.i32`) from `mbs2_context.pc_disasm`.
  - `spirv-ops`: extracts the set of unique `(op_name, out_type)` pairs
    directly from the synthesized op sequence without invoking `libMali`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Set

from tools.corpus_synth.op_catalog import OpSpec


@dataclasses.dataclass
class OracleResult:
    """Result of evaluating a candidate SPIR-V shader against the coverage oracle."""

    ok: bool
    features: Set[str]
    bundle: Optional[Dict[str, Any]] = None
    error: str = ""


def _parse_valhall_mnemonic(disasm_line: str) -> Optional[str]:
    """Extract canonical opcode+modifier token (e.g. `FMA.f32`) from a Valhall disasm line."""
    line = disasm_line.strip()
    if not line:
        return None
    tok = line.split()[0]
    # Strip scheduling suffixes like `.wait0`, `.slot0`, `.barrier`, `.reconverge`
    parts = [
        p for p in tok.split(".")
        if not p.startswith(("wait", "slot", "barrier", "reconverge", "discard"))
    ]
    return ".".join(parts) if parts else tok


class CoverageOracle:
    """Evaluates synthesized SPIR-V binaries for Valhall-ISA or SPIR-V-op coverage."""

    def __init__(self, target: str = "valhall-isa", bin_path: str = "./build/spirv2valhall"):
        self.target = target
        self.bin_path = bin_path

    def evaluate(self, spv_bytes: bytes, ops: List[OpSpec]) -> OracleResult:
        if self.target == "spirv-ops":
            feats = {f"{spec.name}:{spec.out_type}" for spec in ops}
            return OracleResult(ok=True, features=feats)

        with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as spv_f:
            spv_f.write(spv_bytes)
            spv_path = spv_f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_f:
            json_path = json_f.name

        try:
            proc = subprocess.run(
                [self.bin_path, spv_path, "--quiet", "--dump-equiv-json", json_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                return OracleResult(ok=False, features=set(), error=proc.stderr.strip()[:200])
            with open(json_path, "r", encoding="utf-8") as fh:
                bundle = json.load(fh)
            if not bundle.get("direct_ok") or not bundle.get("valhall_ok"):
                return OracleResult(ok=False, features=set(), error="direct_ok/valhall_ok false")
            pc_disasm: Dict[str, str] = bundle.get("mbs2_context", {}).get("pc_disasm", {})
            feats: Set[str] = set()
            for line in pc_disasm.values():
                mn = _parse_valhall_mnemonic(line)
                if mn:
                    feats.add(mn)
            return OracleResult(ok=True, features=feats, bundle=bundle)
        except Exception as exc:
            return OracleResult(ok=False, features=set(), error=str(exc))
        finally:
            if os.path.exists(spv_path):
                os.remove(spv_path)
            if os.path.exists(json_path):
                os.remove(json_path)
