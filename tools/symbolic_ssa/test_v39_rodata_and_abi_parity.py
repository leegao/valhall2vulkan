"""Regression tests for Checkpoint v39:
  1. Embedded EBIN rodata (`mbs2_context.ebin_rodata`) serialization, concrete &
     symbolic constant/dynamic LUT reads, and `LoadSidebandU64(imm=1)` PC mapping.
  2. `ShaderEnvBase` concrete/symbolic ABI parity (`pilot_srt`, `ebin_rodata`,
     Q8.8 `gl_SamplePosition`, and 1 TB `REGION_STRIDE_HI` carry/borrow-safe
     `resolve_region_hi32` across signed `SHADD.s64`/`IADD.s64` offsets).
  3. End-to-end `samples/bcn_bc7.spv` bit-exact equivalence across all 8 BC7 modes
     (Modes 0..7), exercising `ebin_rodata`, `MKVEC.v2i8`, and Pilot FAU derivation.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest

import z3

from tools.symbolic_ssa.concrete_interpreter import ConcreteSsaInterpreter
from tools.symbolic_ssa.domains.shader_env.concrete_shader_env import ConcreteShaderEnv
from tools.symbolic_ssa.domains.shader_env.symbolic_shader_env import SymbolicShaderEnv
from tools.symbolic_ssa.z3utils import dsl


def _make_rodata_bundle(offset: int = 0x2C20, raw_bytes: bytes = b"\xcc\xcc\x00\x00\x88\x88\x00\x00") -> dict:
    return {
        "stage": "Compute",
        "local_size": [8, 8, 1],
        "direct_bindings": [
            {"kind": "PushConstant", "key": "push_const", "set": 0, "binding": 0},
            {"kind": "SSBO", "key": "ssbo:0:0", "set": 0, "binding": 0},
            {"kind": "SSBO", "key": "ssbo:0:1", "set": 0, "binding": 1},
        ],
        "direct_ssa": {"blocks": [], "nodes": []},
        "valhall_ssa": {"blocks": [], "nodes": []},
        "mbs2_context": {
            "ebin_rodata": {
                "offset": offset,
                "size": len(raw_bytes),
                "hex": raw_bytes.hex(),
            },
            "pilot_srt": [
                {"rack": 0, "addr": 0, "kind": "ubo", "key": "push_const"},
                {"rack": 4, "addr": 0, "kind": "ssbo", "key": "ssbo:0:0"},
                {"rack": 4, "addr": 1, "kind": "ssbo", "key": "ssbo:0:1"},
            ],
            "pilot_fau": [],
            "rlocs": [],
            "fcsts": [],
            "sinp": [],
            "sout": [],
        },
    }


class TestV39RodataAndAbiParity(unittest.TestCase):
    def test_ebin_rodata_concrete_and_symbolic_parity(self) -> None:
        raw = bytes.fromhex("cccc000088880000eeee0000c8ec0000")
        bundle = _make_rodata_bundle(offset=0x2C20, raw_bytes=raw)
        cenv = ConcreteShaderEnv(bundle, 0)
        senv = SymbolicShaderEnv(bundle, domain="udomain")

        expected_words = [0x0000CCCC, 0x00008888, 0x0000EEEE, 0x0000ECC8]
        ro_arr = senv.get_buf_mem("ebin_rodata")

        for idx, exp in enumerate(expected_words):
            off = 0x2C20 + idx * 4
            self.assertEqual(cenv.read_init_buf("ebin_rodata", off), exp)

            slot = senv.get_ro_buf_slot_value("ebin_rodata", off)
            self.assertIsNotNone(slot)
            self.assertEqual(senv.domain.const_int_val(slot), exp)

            dyn = z3.simplify(dsl.unwrap(senv.read_buf_u32(ro_arr, dsl.BitVecVal(off, 32))))
            self.assertTrue(z3.is_bv_value(dyn))
            self.assertEqual(dyn.as_long(), exp)

        # Out-of-bounds offset reads zero in both concrete and symbolic envs.
        self.assertEqual(cenv.read_init_buf("ebin_rodata", 0x2C00), 0)
        oob = z3.simplify(dsl.unwrap(senv.read_buf_u32(ro_arr, dsl.BitVecVal(0x2C00, 32))))
        self.assertEqual(oob.as_long(), 0)

    def test_region_stride_and_signed_borrow_resolution_parity(self) -> None:
        bundle = _make_rodata_bundle()
        cenv = ConcreteShaderEnv(bundle, 0)
        senv = SymbolicShaderEnv(bundle, domain="udomain")

        b_pc = cenv.get_buf_base("push_const")
        b_s0 = cenv.get_buf_base("ssbo:0:0")
        b_s1 = cenv.get_buf_base("ssbo:0:1")
        self.assertEqual(b_pc, senv.get_buf_base("push_const"))
        self.assertEqual(b_s0, senv.get_buf_base("ssbo:0:0"))
        self.assertEqual(b_s1, senv.get_buf_base("ssbo:0:1"))
        self.assertEqual((b_s0 - b_pc) >> 32, cenv.REGION_STRIDE_HI)

        # Simulate signed SHADD.s64 with negative 32-bit offset borrowing 1 from hi32.
        neg_off64 = (0xFFFFFFFF << 32) | 0x86CF4080
        addr_borrow = (b_s1 + neg_off64) & ((1 << 64) - 1)
        c_key, c_off = cenv.resolve_addr(addr_borrow)
        s_key, s_off, _ = senv.resolve_addr(dsl.BitVecVal(addr_borrow, 64))
        self.assertEqual(c_key, "ssbo:0:1")
        self.assertEqual(s_key, "ssbo:0:1")
        self.assertEqual(c_off, 0x86CF4080)
        self.assertEqual(s_off, 0x86CF4080)

        # Simulate positive unsigned carry (+3 in hi32) from shifted u32 offset.
        pos_carry = (b_s0 + (3 << 32) + 0x1234) & ((1 << 64) - 1)
        c_key2, c_off2 = cenv.resolve_addr(pos_carry)
        s_key2, s_off2, _ = senv.resolve_addr(dsl.BitVecVal(pos_carry, 64))
        self.assertEqual(c_key2, "ssbo:0:0")
        self.assertEqual(s_key2, "ssbo:0:0")
        self.assertEqual(c_off2, 0x1234)
        self.assertEqual(s_off2, 0x1234)

    def test_pilot_srt_parity_between_concrete_and_symbolic(self) -> None:
        bundle = _make_rodata_bundle()
        cenv = ConcreteShaderEnv(bundle, 0)
        senv = SymbolicShaderEnv(bundle, domain="udomain")

        # Rack 0 descriptor 0 (push_const): size at +4, base lo/hi at +8/+12.
        for off in (4, 8, 12):
            cv = cenv.read_init_buf("pilot_srt_rack_0", off)
            sv = senv.get_ro_buf_slot_value("pilot_srt_rack_0", off)
            self.assertIsNotNone(sv)
            self.assertEqual(cv, senv.domain.const_int_val(sv))
        self.assertEqual(cenv.read_init_buf("pilot_srt_rack_0", 4), 0x10000)

    def test_bcn_bc7_rodata_export_and_all_8_modes_bit_exact(self) -> None:
        spv = "samples/bcn_bc7.spv"
        bin_path = "./build/spirv2valhall"
        if not (os.path.exists(spv) and os.path.exists(bin_path)):
            self.skipTest("spirv2valhall binary or samples/bcn_bc7.spv not present")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name
        try:
            subprocess.run(
                [bin_path, spv, "--dump-equiv-json", tmp_json, "--quiet", "--opt-ssa", "--defau"],
                check=True,
            )
            with open(tmp_json, "r", encoding="utf-8") as fh:
                bundle = json.load(fh)
        finally:
            if os.path.exists(tmp_json):
                os.unlink(tmp_json)

        ro = bundle.get("mbs2_context", {}).get("ebin_rodata", {})
        self.assertEqual(ro.get("offset"), 0x2C20)
        self.assertEqual(ro.get("size"), 1376)
        self.assertEqual(len(ro.get("hex", "")), 1376 * 2)

        for mode_idx in range(8):
            mode_byte = 1 << mode_idx
            ov = {
                "bi_26_0": 0, "bi_26_1": 0, "bi_26_2": 0,
                "bi_27_0": 0, "bi_27_1": 0, "bi_27_2": 0,
                "bi_28_0": 0, "bi_28_1": 0, "bi_28_2": 0,
                "mem:push_const:0": 0,
                "mem:push_const:4": 16,
                "mem:push_const:8": 16,
                "mem:push_const:12": 1,
                "mem:push_const:16": 0,
                "mem:push_const:20": 16,
                "mem:push_const:24": 16,
                "mem:ssbo:0:1:0": mode_byte | 0x12345600,
                "mem:ssbo:0:1:4": 0x9ABCDEF0,
                "mem:ssbo:0:1:8": 0x13579BDF,
                "mem:ssbo:0:1:12": 0x2468ACE0,
            }
            ed = ConcreteShaderEnv(bundle, 0, overrides=ov)
            ev = ConcreteShaderEnv(bundle, 0, overrides=ov)
            _, od = ConcreteSsaInterpreter(bundle["direct_ssa"], is_valhall_side=False).evaluate(ed)
            _, ov_out = ConcreteSsaInterpreter(bundle["valhall_ssa"], is_valhall_side=True).evaluate(ev)
            self.assertIn("ssbo:0:0+0x0", od, msg=f"BC7 mode {mode_idx} missing direct write")
            self.assertEqual(
                od["ssbo:0:0+0x0"],
                ov_out.get("ssbo:0:0+0x0"),
                msg=f"BC7 mode {mode_idx} mismatch: direct={od['ssbo:0:0+0x0']:#x} valhall={ov_out.get('ssbo:0:0+0x0')}",
            )


if __name__ == "__main__":
    unittest.main()
