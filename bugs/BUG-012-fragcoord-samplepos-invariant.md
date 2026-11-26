# BUG-012 (`RC-13c`): `ConcreteShaderEnv` `gl_FragCoord.xy` Violated Vulkan `pixel + gl_SamplePosition` Invariant
- **Status:** FIXED in Checkpoint v48.2
- **Layer:** ShaderEnv (`tools/symbolic_ssa/domains/shader_env/concrete_shader_env.py:55`)
- **Targets (1):** `LD_VAR_SPECIAL`.
- **Root Cause & Fix:** Sampled `bi_15_0` / `bi_15_1` (`gl_FragCoord.xy`) as `float(pixel_xy) + gl_SamplePosition.xy` (`pixel_xy >= 0`), matching Valhall's `r59.h0/h1 + s16_to_f32(sample_table) * (1/256)`.
