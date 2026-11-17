#!/usr/bin/env python3
"""
check_driver.py — Automated ELF + AArch64 ioctl analyzer & bringup
validator for mali_csf_dump_layer against arbitrary libGLES_mali*.so drivers.
"""
import argparse
import json
import re
import struct
import subprocess
from pathlib import Path

REQUIRED_HOOK_SYMS = {"open", "open64", "__open_2", "openat", "openat64", "close", "ioctl", "mmap", "mmap64", "munmap"}

KBASE_IOCTL_NAMES = {
    (0x80, 0): "VERSION_CHECK_JM",
    (0x80, 1): "SET_FLAGS",
    (0x80, 2): "JOB_SUBMIT (JM)",
    (0x80, 5): "MEM_ALLOC",
    (0x80, 6): "MEM_QUERY",
    (0x80, 7): "MEM_FREE",
    (0x80, 13): "GET_DDK_VERSION",
    (0x80, 21): "MEM_IMPORT",
    (0x80, 22): "MEM_ALIAS",
    (0x80, 36): "CS_QUEUE_REGISTER",
    (0x80, 37): "CS_QUEUE_KICK",
    (0x80, 38): "MEM_EXEC_INIT",
    (0x80, 39): "CS_QUEUE_BIND",
    (0x80, 40): "CS_QUEUE_REGISTER_EX",
    (0x80, 41): "CS_QUEUE_TERMINATE",
    (0x80, 42): "CS_QUEUE_GROUP_CREATE",
    (0x80, 43): "CS_QUEUE_GROUP_TERMINATE",
    (0x80, 44): "CS_EVENT_SIGNAL",
    (0x80, 45): "KCPU_QUEUE_CREATE",
    (0x80, 46): "KCPU_QUEUE_DELETE",
    (0x80, 47): "KCPU_QUEUE_ENQUEUE",
    (0x80, 48): "CS_TILER_HEAP_INIT",
    (0x80, 49): "CS_TILER_HEAP_TERM",
    (0x80, 50): "MEM_ALLOC_EX",
    (0x80, 52): "VERSION_CHECK_CSF",
}


def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def parse_sections(path: Path):
    out = run(["readelf", "-SW", str(path)])
    secs = {}
    for line in out.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)", line)
        if m:
            secs[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16), int(m.group(4), 16))
    return secs


def parse_dynamic(path: Path):
    out = run(["readelf", "-d", str(path)])
    dyn = {"needed": [], "flags": [], "soname": None, "has_jmprel": False, "has_rela": False, "has_relr": False}
    for line in out.splitlines():
        if "(NEEDED)" in line:
            m = re.search(r"\[([^\]]+)\]", line)
            if m:
                dyn["needed"].append(m.group(1))
        elif "(SONAME)" in line:
            m = re.search(r"\[([^\]]+)\]", line)
            if m:
                dyn["soname"] = m.group(1)
        elif "(FLAGS)" in line or "(FLAGS_1)" in line:
            dyn["flags"].append(line.strip())
        elif "(JMPREL)" in line:
            dyn["has_jmprel"] = True
        elif "(RELA)" in line:
            dyn["has_rela"] = True
        elif "(RELR)" in line:
            dyn["has_relr"] = True
    return dyn


def parse_relocs(path: Path):
    out = run(["readelf", "-r", str(path)])
    hooks = {}
    current_sec = "unknown"
    for line in out.splitlines():
        if line.startswith("Relocation section"):
            m = re.search(r"'([^']+)'", line)
            if m:
                current_sec = m.group(1)
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[2].startswith("R_AARCH64_"):
            sym = parts[4].split("@")[0]
            if sym in REQUIRED_HOOK_SYMS or sym == "dlopen":
                hooks[sym] = {
                    "got_offset": int(parts[0], 16),
                    "type": parts[2],
                    "section": current_sec,
                }
    return hooks


def find_ddk_version(data: bytes):
    matches = re.findall(rb"r\d{2}p\d+-\d{2}eac\d[a-zA-Z0-9._-]*", data)
    if matches:
        return matches[0].decode("ascii", errors="replace")
    return "unknown"


def decode_ioctl_sites(data: bytes, secs: dict, ioctl_got: int):
    if ".plt" not in secs or ".text" not in secs or ioctl_got is None:
        return []
    plt_va, plt_off, plt_sz = secs[".plt"]
    text_va, text_off, text_sz = secs[".text"]
    va_to_off = plt_off - plt_va

    ioctl_plt_va = None
    for off in range(plt_off, plt_off + plt_sz, 4):
        insn = struct.unpack_from("<I", data, off)[0]
        if (insn & 0x9F000000) == 0x90000000:  # ADRP
            immhi = (insn >> 5) & 0x7FFFF
            immlo = (insn >> 29) & 0x3
            imm = (immhi << 2) | immlo
            if imm & (1 << 20):
                imm -= 1 << 21
            cur_va = off - va_to_off
            page = (cur_va & ~0xFFF) + (imm << 12)
            if page == (ioctl_got & ~0xFFF):
                insn2 = struct.unpack_from("<I", data, off + 4)[0]
                if (insn2 & 0xFFC00000) == 0xF9400000:
                    imm12 = ((insn2 >> 10) & 0xFFF) << 3
                    if page + imm12 == ioctl_got:
                        ioctl_plt_va = cur_va
                        break
    if ioctl_plt_va is None:
        return []

    text_va_to_off = text_off - text_va
    reqs = {}
    for off in range(text_off, text_off + text_sz, 4):
        insn = struct.unpack_from("<I", data, off)[0]
        if (insn & 0xFC000000) == 0x94000000:  # BL
            imm26 = insn & 0x03FFFFFF
            if imm26 & (1 << 25):
                imm26 -= 1 << 26
            cur_va = off - text_va_to_off
            if cur_va + (imm26 << 2) == ioctl_plt_va:
                regs = {}
                for k in range(max(text_off, off - 64), off, 4):
                    ins = struct.unpack_from("<I", data, k)[0]
                    if (ins & 0x7F800000) == 0x52800000:
                        rd = ins & 0x1F
                        hw = (ins >> 21) & 0x3
                        imm16 = (ins >> 5) & 0xFFFF
                        regs[rd] = imm16 << (hw * 16)
                    elif (ins & 0x7F800000) == 0x72800000:
                        rd = ins & 0x1F
                        hw = (ins >> 21) & 0x3
                        imm16 = (ins >> 5) & 0xFFFF
                        cur = regs.get(rd, 0)
                        cur &= ~(0xFFFF << (hw * 16))
                        cur |= imm16 << (hw * 16)
                        regs[rd] = cur
                    elif (ins & 0x7FE0FFE0) == 0x2A0003E0:
                        rd = ins & 0x1F
                        rm = (ins >> 16) & 0x1F
                        if rm in regs:
                            regs[rd] = regs[rm]
                if 1 in regs:
                    w1 = regs[1]
                    reqs.setdefault(w1, 0)
                    reqs[w1] += 1

    out = []
    for req, count in sorted(reqs.items()):
        typ = (req >> 8) & 0xFF
        nr = req & 0xFF
        size = (req >> 16) & 0x3FFF
        out.append({
            "hex": f"0x{req:08x}",
            "type": f"0x{typ:02x}",
            "nr": nr,
            "size": size,
            "count": count,
            "name": KBASE_IOCTL_NAMES.get((typ, nr), "UNKNOWN"),
        })
    return out


def analyze_driver(path: Path):
    data = path.read_bytes()
    secs = parse_sections(path)
    dyn = parse_dynamic(path)
    hooks = parse_relocs(path)
    ddk_ver = find_ddk_version(data)
    ioctl_got = hooks["ioctl"]["got_offset"] if "ioctl" in hooks else None
    ioctls = decode_ioctl_sites(data, secs, ioctl_got)

    nrs = {i["nr"] for i in ioctls if i["type"] == "0x80"}
    has_csf = bool({36, 37, 39, 40} & nrs)
    has_jm = 2 in nrs
    mode = "CSF" if has_csf else ("JM" if has_jm else "UNKNOWN")

    missing_csf = sorted({36, 37, 39} - nrs) if mode == "CSF" else []
    has_get_ddk = 13 in nrs
    hook_sections = {v["section"] for v in hooks.values()}

    compatible = (
        mode == "CSF"
        and "ioctl" in hooks
        and bool({"open", "__open_2", "openat", "openat64"} & set(hooks.keys()))
        and bool({"mmap", "mmap64"} & set(hooks.keys()))
        and len(missing_csf) == 0
    )

    return {
        "file": path.name,
        "ddk_version": ddk_ver,
        "soname": dyn["soname"],
        "mode": mode,
        "compatible_with_csf_layer": compatible,
        "has_get_ddk_version_ioctl": has_get_ddk,
        "hooked_symbols": sorted(hooks.keys()),
        "hook_reloc_sections": sorted(hook_sections),
        "csf_ioctls": [i for i in ioctls if i["type"] == "0x80" and i["nr"] in (5, 7, 13, 36, 37, 39, 40, 50, 52)],
        "missing_required_csf_ioctls": missing_csf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drivers", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    results = [analyze_driver(p) for p in args.drivers]
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    for r in results:
        status = "PASS (CSF-ready)" if r["compatible_with_csf_layer"] else f"SKIP ({r['mode']})"
        print(f"[{status}] {r['file']} | DDK={r['ddk_version']} | hooks={','.join(r['hooked_symbols'])} | GET_DDK_VERSION={r['has_get_ddk_version_ioctl']}")
        for ic in r["csf_ioctls"]:
            print(f"    {ic['hex']} nr={ic['nr']:2d} size={ic['size']:2d} {ic['name']}")


if __name__ == "__main__":
    main()
