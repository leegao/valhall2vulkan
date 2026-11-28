"""SPIR-V assembly (.spvasm) <-> binary (.spv) I/O and bundle loading helpers."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from typing import Any


def find_spirv_tool(name: str) -> str | None:
    """Locate a SPIR-V Tools binary (e.g. spirv-as, spirv-dis, spirv-val, spirv-reduce, spirv-fuzz).

    Search order:
      1. `$SPIRV_TOOLS_DIR/<name>` when explicitly set in the environment.
      2. `<repo>/third_party/prebuilts/spirv-tools/<name>` (and `bin/<name>`) if present and executable.
      3. `PATH` via `shutil.which`.
    """
    env_dir = os.environ.get("SPIRV_TOOLS_DIR")
    if env_dir:
        cand = os.path.join(env_dir, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for rel in (
        os.path.join("third_party", "prebuilts", "spirv-tools", name),
        os.path.join("third_party", "prebuilts", "spirv-tools", "bin", name),
    ):
        cand = os.path.join(repo_root, rel)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return shutil.which(name)


def is_spvasm_path(path: str) -> bool:
    """Return True if path refers to a SPIR-V text assembly (.spvasm) file."""
    return path.lower().endswith(".spvasm")


def assemble_spvasm(spvasm_path: str, out_spv_path: str, target_env: str = "vulkan1.2") -> None:
    """Assemble a .spvasm text file into a binary .spv file via spirv-as."""
    tool = find_spirv_tool("spirv-as")
    if not tool:
        raise RuntimeError(
            "spirv-as not found on PATH (or SPIRV_TOOLS_DIR); required to consume .spvasm inputs"
        )
    cmd = [tool, "--target-env", target_env, spvasm_path, "-o", out_spv_path]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        # Retry with default target environment in case the assembly specifies an older/newer version.
        proc2 = subprocess.run([tool, spvasm_path, "-o", out_spv_path], check=False, capture_output=True, text=True, timeout=30)
        if proc2.returncode != 0:
            err = (proc.stderr or proc2.stderr or "").strip()[:300]
            raise RuntimeError(f"spirv-as failed on {spvasm_path}: {err}")


def disassemble_spv(spv_path: str) -> str:
    """Disassemble a binary .spv file into SPIR-V text assembly via spirv-dis."""
    tool = find_spirv_tool("spirv-dis")
    if not tool:
        raise RuntimeError(
            "spirv-dis not found on PATH (or SPIRV_TOOLS_DIR); required to emit .spvasm outputs"
        )
    proc = subprocess.run([tool, "--no-color", spv_path], check=False, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"spirv-dis failed on {spv_path}: {proc.stderr.strip()[:300]}")
    return proc.stdout


def validate_spv(spv_path: str, target_env: str = "vulkan1.3") -> bool:
    """Validate a binary .spv file via spirv-val (returns True if valid or spirv-val unavailable)."""
    tool = find_spirv_tool("spirv-val")
    if not tool:
        return True
    proc = subprocess.run([tool, "--target-env", target_env, spv_path], check=False, capture_output=True, text=True, timeout=30)
    if proc.returncode == 0:
        return True
    proc2 = subprocess.run([tool, spv_path], check=False, capture_output=True, text=True, timeout=30)
    return proc2.returncode == 0


@contextlib.contextmanager
def ensure_spv_path(path: str) -> Iterator[str]:
    """Yield a binary .spv path, assembling .spvasm inputs to a temporary .spv file."""
    if not is_spvasm_path(path):
        yield path
        return
    fd, tmp_spv = tempfile.mkstemp(suffix=".spv")
    os.close(fd)
    try:
        assemble_spvasm(path, tmp_spv)
        yield tmp_spv
    finally:
        if os.path.exists(tmp_spv):
            os.remove(tmp_spv)


def load_equiv_bundle(
    spv_or_spvasm_or_json: str,
    bin_path: str,
    timeout: int = 15,
    defau: bool = True,
    opt_ssa: bool = True,
) -> dict[str, Any]:
    """Load an equivalence bundle JSON from a .json, .spv, or .spvasm input."""
    if spv_or_spvasm_or_json.endswith(".json"):
        with open(spv_or_spvasm_or_json, "r", encoding="utf-8") as fh:
            return json.load(fh)
    with ensure_spv_path(spv_or_spvasm_or_json) as spv_path:
        fd, tmp_json = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            cmd = [bin_path, spv_path, "--quiet", "--dump-equiv-json", tmp_json]
            if defau:
                cmd.append("--defau")
            if opt_ssa:
                cmd.append("--opt-ssa")
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip()[:300])
            with open(tmp_json, "r", encoding="utf-8") as fh:
                bundle = json.load(fh)
            bundle["spv_path"] = spv_or_spvasm_or_json
            return bundle
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
