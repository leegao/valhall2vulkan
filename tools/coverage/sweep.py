"""Parallel equiv-fuzz sweep runner, cluster aggregator, and baseline differ."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import random
import re
import subprocess
import sys
from typing import Any


def _run_shader(
    group: str,
    path: str,
    *,
    fp_range: str,
    k_init: int,
    timeout_s: int,
) -> dict[str, Any]:
    name = os.path.basename(path)
    cmd = [
        sys.executable,
        "-m",
        "tools.coverage",
        "--canonicalize",
        "--k-init",
        str(k_init),
        "--domain",
        "udomain",
        "--fp-mode",
        "real",
        "--fp-range",
        fp_range,
        "--equiv-fuzz",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        out = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return {
            "group": group,
            "path": path,
            "name": name,
            "status": "TIMEOUT",
            "divs": 999,
            "suspect": "TIMEOUT",
            "signals": [],
        }

    status = "UNKNOWN"
    divs = 0
    suspect = ""
    signals: list[str] = []
    for line in out.splitlines():
        if "[EQUIV-FUZZ]" in line:
            m_st = re.search(r"status=(\S+)", line)
            if m_st:
                status = m_st.group(1)
            m_dv = re.search(r"divs=(\d+)", line)
            if m_dv:
                divs = int(m_dv.group(1))
        elif "[TRIAGE]" in line:
            m_su = re.search(r"suspect=(\S+)", line)
            if m_su:
                suspect = m_su.group(1)
            m_sg = re.search(r"signals=\[([^\]]*)\]", line)
            if m_sg and m_sg.group(1).strip():
                signals = [s.strip().strip("'\"") for s in m_sg.group(1).split(",")]

    return {
        "group": group,
        "path": path,
        "name": name,
        "status": status,
        "divs": divs,
        "suspect": suspect,
        "signals": signals,
    }


def _collect_targets(
    samples_dir: str,
    corpus_dir: str,
    corpus_count: int,
    seed: int,
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    if samples_dir and os.path.isdir(samples_dir):
        for p in sorted(glob.glob(os.path.join(samples_dir, "*.spv"))):
            targets.append(("samples", p))
    if corpus_dir and os.path.isdir(corpus_dir) and corpus_count > 0:
        corpus_all = sorted(glob.glob(os.path.join(corpus_dir, "*.spv")))
        picked = random.Random(seed).sample(corpus_all, min(corpus_count, len(corpus_all)))
        for p in picked:
            targets.append(("corpus50", p))
    return targets


def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel ValhallSSA equivalence sweep & cluster aggregator")
    ap.add_argument("--samples-dir", default="samples", help="Directory of sample .spv shaders")
    ap.add_argument("--corpus-dir", default="../spirv_corpus/spirv_samples", help="Directory of corpus .spv shaders")
    ap.add_argument("--corpus-count", type=int, default=50, help="Number of corpus shaders to sample")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic corpus sampling seed")
    ap.add_argument("--k-init", type=int, default=64, help="Initial concrete seed vectors")
    ap.add_argument("--fp-range", choices=["unconstrained", "vulkan"], default="unconstrained")
    ap.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    ap.add_argument("--timeout-s", type=int, default=30, help="Per-shader timeout in seconds")
    ap.add_argument("--json", dest="json_out", default=None, help="Write sweep results JSON to path")
    ap.add_argument("--compare", default=None, help="Compare against a baseline sweep JSON")
    ap.add_argument("--cluster", action="store_true", help="Print divergence cluster breakdown")
    args = ap.parse_args()

    targets = _collect_targets(args.samples_dir, args.corpus_dir, args.corpus_count, args.seed)
    if not targets:
        print("[ERROR] No .spv targets found", file=sys.stderr)
        return 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(
            ex.map(
                lambda item: _run_shader(
                    item[0],
                    item[1],
                    fp_range=args.fp_range,
                    k_init=args.k_init,
                    timeout_s=args.timeout_s,
                ),
                targets,
            )
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    total_eq = sum(1 for r in results if r["status"].startswith("EQUIV"))
    print(f"[SWEEP fp_range={args.fp_range}] total={total_eq}/{len(results)} EQUIV")
    for grp in ("samples", "corpus50"):
        sub = [r for r in results if r["group"] == grp]
        if sub:
            eq = sum(1 for r in sub if r["status"].startswith("EQUIV"))
            print(f"  {grp}: {eq}/{len(sub)} EQUIV ({100.0 * eq / len(sub):.1f}%)")

    if args.compare and os.path.isfile(args.compare):
        with open(args.compare, encoding="utf-8") as f:
            base = json.load(f)
        base_by_name = {(r.get("name") or os.path.basename(r["path"])): r for r in base}
        flipped_eq = []
        reduced = []
        regressed = []
        for r in results:
            b = base_by_name.get(r["name"])
            if not b:
                continue
            was_eq = b["status"].startswith("EQUIV")
            now_eq = r["status"].startswith("EQUIV")
            if not was_eq and now_eq:
                flipped_eq.append(f"{r['name'][:20]}({b['divs']}->0)")
            elif was_eq and not now_eq:
                regressed.append(f"{r['name'][:20]}(0->{r['divs']})")
            elif r["divs"] < b["divs"]:
                reduced.append(f"{r['name'][:20]}({b['divs']}->{r['divs']})")
        print(f"[DIFF vs {args.compare}] flipped_equiv={len(flipped_eq)} reduced={len(reduced)} regressed={len(regressed)}")
        if flipped_eq:
            print("  flipped_equiv:", ", ".join(flipped_eq))
        if reduced:
            print("  reduced:", ", ".join(reduced))
        if regressed:
            print("  regressed:", ", ".join(regressed))

    if args.cluster:
        clusters: Counter[tuple[str, str]] = Counter()
        for r in results:
            if not r["status"].startswith("EQUIV"):
                sig = r["signals"][0] if r["signals"] else ""
                clusters[(r["suspect"] or r["status"], sig)] += 1
        print("\n[CLUSTERS]")
        for (susp, sig), cnt in clusters.most_common():
            print(f"  {cnt:2d} | {susp:32s} | {sig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
