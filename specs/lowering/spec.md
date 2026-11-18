# Valhall ISA Specification Format

## Overview

`specs/lowering/valhall_lowering.json` encodes the operational semantics of Arm Mali Valhall ISA as a syntax-directed translation into `ValhallSSA` (see `src/include/valhall_ssa.h`).

Every rule dispatches via one of four `LoweringKind` modes:
1. **`DirectSsaOp`**: `"ssa_op": "<SsaOp>"` (or `"Nop"` / `"Copy"`)
2. **`DynamicSsaOp`**: `"ssa_op": "@LOOKUP(<table>, <key_expr>)"`
3. **`InlineOps`**: `"ssa_op": "@INLINE_OPS"` with `"inline_ops": [ ... ]`
4. **`Handler`**: `"ssa_op": "@HANDLER(<fn>)"` (only used as fallback for `handle_int_idpadd`)

---

## 2. Concrete BNF Grammar for `@INLINE_OPS`

Here's the complete (ambiguous, so declarative) grammar for the InlineOps DSL used to specify the operational semantics of the ISA:

```bnf
<spec>          ::= "{" "schema_version": 4,
                        "host_functions": <host_fn_map>,
                        "lookup_tables": <lookup_map>,
                        "rules": [ <rule> ("," <rule>)* ] "}"

<host_fn_map>   ::= "{" ( <ident> ":" "{" "ret": <abi_ret>, "args": "[" <abi_arg_list> "]" "}" )* "}"
<abi_ret>       ::= "SsaId" | "void"
<abi_arg>       ::= "SsaId" | "ValType" | "uint32_t" | "bool"

<ssa_op_spec>   ::= <ident>
                  | "@LOOKUP(" <ident> "," <key_expr> ")"
                  | "@INLINE_OPS"
                  | "@HANDLER(" <ident> ")"

<inline_ops>    ::= "[" <stmt_str> ("," <stmt_str>)* "]"
<program>       ::= <stmt>*
<stmt>          ::= "%" <ident> ":" <val_type> "=" <expr> ";"
                  | "%" <ident> "=" <expr> ";"
                  | "@" <ident> "(" <arg_list>? ")" ";"
                  | <ssa_call> ";"
                  | "if" "(" <cond> ")" "{" <stmt>* "}" ( "else" "{" <stmt>* "}" )?
                  | "for" "(" "%" <ident> ":" "0" ".." <loop_bound> ")" "{" <stmt>* "}"
                  | "return" <atom>? ";"

<expr>          ::= <atom> | <ssa_call> | <lookup_call> | <host_call>
<ssa_call>      ::= <ident> "(" <atom_or_host_list>? ")" <node_flag>*
<lookup_call>   ::= "@LOOKUP(" <ident> "," <key_expr> ")" "(" <atom_or_host_list>? ")" <node_flag>*
<host_call>     ::= "@" <ident> "(" <host_arg_list>? ")"
<node_flag>     ::= "[barrier]" | "[side_effect]" | "[aux_bool=" <bool_expr> "]"

<cond>          ::= "!"? <cond_prim>
<cond_prim>     ::= "$has_src0" | "$has_src1" | "$has_src2" | "$has_src3" | "$has_dest"
                  | "$is_v2f16" | "$is_signed" | "$is_64bit" | "$dest_is_64bit"
                  | "$src0_swz_h1" | "$is_frag_stage" | "$is_vert_stage"
                  | "$has_imm_nonzero" | "$sr_write_present" | "$is_64_atom"
                  | "$is_gather" | "$is_flat_var"
                  | "@has_field(" <ident> | "@field_nz(" <ident>
                  | "@mod_active(" <ident> ")" | "@mod_eq(" <ident> "," <ident> ")" | "@mod_ne(" <ident> "," <ident> ")"
                  | "@is_zero(" "%" <ident> ")" | "@lt_u32(" <u32_expr> "," <u32_expr> ")"

<atom>          ::= "%" <digits> | "%" <ident> | "#raw0" | "#token" | "#fp_zero"
                  | "#imm_u32" | "#imm_s32" | "#imm_f32" | "#imm_u64" | "#imm_u64_mask63"
                  | "$branch_target_pc_u64" | "#const_bool(" ("true"|"false") ")"
                  | "#u32(" <int> ")" | "#s32(" <int> ")" | "#u64(" <int> ")" | "#f32(" <num> ")"
                  | "#u64_mul4(" "%" <ident> ")" | "#f32_alpha_or_zero(" "%" <ident> ")"
                  | "#field_u32(" <ident> ")" | "#field_u32_mul16(" <ident> ")"
                  | "#lookup_u32(" <ident> "," <key_expr> ")"
```

---

### 2.1 LL(1) Unambiguous Grammar (BNF)

Here's a machine executable (LL(1) with unambiguous first/follow sets) grammar as a reference to implement parsers. Please see `dsl_parser.py` for reference.

```bnf
<program>         ::= <stmt_list>

<stmt_list>       ::= <stmt> <stmt_list>
                    | ε

<stmt>            ::= VAR_REF <type_ann_opt> "=" <expr> ";"
                    | "@" <at_expr> ";"
                    | IDENT "(" <arg_list_opt> ")" <node_flag_list> ";"
                    | "if" "(" <cond> ")" "{" <stmt_list> "}" <else_opt>
                    | "for" "(" VAR_REF ":" INT_LIT ".." <u32_expr> ")" "{" <stmt_list> "}"
                    | "return" <expr_opt> ";"

<type_ann_opt>    ::= ":" IDENT
                    | ε

<else_opt>        ::= "else" "{" <stmt_list> "}"
                    | ε

<expr_opt>        ::= <expr>
                    | ε

<expr>            ::= SRC_REF
                    | VAR_REF
                    | "$" IDENT
                    | "#" <hash_tail>
                    | "@" <at_expr>
                    | IDENT "(" <arg_list_opt> ")" <node_flag_list>

<at_expr>         ::= "LOOKUP" "(" IDENT "," <key_expr> ")" "(" <arg_list_opt> ")" <node_flag_list>
                    | IDENT "(" <arg_list_opt> ")"

<hash_tail>       ::= "lookup_u32" "(" IDENT "," <key_expr> ")"
                    | "lookup_u32_raw" "(" IDENT "," <key_expr> ")"
                    | IDENT <call_suffix_opt>

<call_suffix_opt> ::= "(" <arg_list_opt> ")"
                    | ε

<node_flag_list>  ::= <node_flag> <node_flag_list>
                    | ε

<node_flag>       ::= "[" <node_flag_body> "]"

<node_flag_body>  ::= "barrier"
                    | "side_effect"
                    | "aux_bool" "=" <aux_bool_val>

<aux_bool_val>    ::= "true"
                    | "false"
                    | <cond>

<cond>            ::= "!" <cond_atom>
                    | <cond_atom>

<cond_atom>       ::= "$" IDENT
                    | "@" IDENT "(" <arg_list_opt> ")"

<arg_list_opt>    ::= <arg> <arg_list_tail>
                    | ε

<arg_list_tail>   ::= "," <arg> <arg_list_tail>
                    | ε

<arg>             ::= SRC_REF
                    | VAR_REF
                    | "$" IDENT
                    | "#" <hash_tail>
                    | "@" <at_expr>
                    | "!" <cond_atom>
                    | INT_LIT
                    | FLOAT_LIT
                    | STRING_LIT
                    | "true"
                    | "false"
                    | IDENT <ident_arg_tail>

<ident_arg_tail>  ::= "(" <arg_list_opt> ")" <node_flag_list>
                    | ε

<u32_expr>        ::= VAR_REF
                    | "$" IDENT
                    | "@" IDENT "(" <arg_list_opt> ")"
                    | "#" "lookup_u32_raw" "(" IDENT "," <key_expr> ")"
                    | INT_LIT

<key_expr>        ::= "$" "modifiers" "." IDENT
```

A bit of runtime "tree-filtering" is needed from this LL(1) parser to avoid certain error states in the language (the parsable language is a superset of the true language, but only slightly so).

---

## 3. Operational Semantics as a translation to `ValhallSSA`

Let our program state for block $b$ and instruction $\iota$ be:

$$\sigma = (\mathcal{M}, \mathcal{R}, \tau, \Gamma, \beta)$$

where:
- $\mathcal{M}$ is the ValhallSSA module (`ValhallSsaModule`), extended via $\mathcal{M}' = \mathcal{M} \oplus (b, n)$ returning fresh `SsaId` $v$,
- $\mathcal{R} : \text{Reg}[0..63] \to \text{SsaId}$ is the per-block register map (`sb.current_reg` / `sb.set_reg`),
- $\tau \in \text{SsaId}$ is the linear effect token (`sb.current_token` / `sb.set_token`),
- $\Gamma : \text{Var} \to (\text{SsaId} \cup \mathbb{N}_{32})$ is the lexical DSL variable environment,
- $\beta \in \text{SsaId}$ is `blocks[b].branch_cond`.

### Inference Rules

1. **Lazy Source Fetch (`%k`)**:
   $$\frac{v = \text{fetch\_src}_{\text{src\_types}[k]}(\mathcal{M}, b, \iota, k)}{\langle \%k, \sigma \rangle \Downarrow (v, \sigma')}$$

2. **Typed Binding & Reassignment**:
   $$\frac{\langle e, \sigma \rangle \Downarrow (v, \sigma')}{\langle \%x : T = e;\, \sigma \rangle \Downarrow \sigma'[\Gamma \mapsto \Gamma[x \mapsto v]]}$$

3. **SSA Node Emission (`Op(a_1, ..., a_n)[flags]`)**:
   $$\frac{\langle a_i, \sigma_{i-1} \rangle \Downarrow (v_i, \sigma_i) \quad \mathcal{M}' = \mathcal{M}_n \oplus (b, \text{Node}(\text{Op}, T, [v_1,\dots,v_n], \text{rule.effects}, \text{flags})) \mapsto v}{\langle \text{Op}(a_1, \dots, a_n), \sigma_0 \rangle \Downarrow (v, \sigma_n[\mathcal{M} \mapsto \mathcal{M}'])}$$

4. **Host ABI Call (`@fn(a_1, ..., a_m)`)**:
   $$\frac{\text{sig}(\text{fn}) = (\text{ret}, [\tau_1, \dots, \tau_m]) \quad \forall i.\, \langle a_i, \sigma_{i-1} \rangle : \tau_i \Downarrow (v_i, \sigma_i) \quad (r, \sigma') = \text{abi::fn}(\sigma_m, b, \iota, \text{rule}, v_1, \dots, v_m)}{\langle @\text{fn}(a_1, \dots, a_m), \sigma_0 \rangle \Downarrow (r, \sigma')}$$

5. **Bounded Loop (`for (%i : 0 .. B) { S }`)**:
   $$\frac{n = \llbracket B \rrbracket(\iota) \quad \forall k \in [0, n-1].\, \langle S, \sigma_k[\Gamma \mapsto \Gamma[i \mapsto k]] \rangle \Downarrow \sigma_{k+1}}{\langle \text{for } (\%i : 0 .. B)\{S\}, \sigma_0 \rangle \Downarrow \sigma_n}$$

6. **Conditional (`if (c) { S_1 } else { S_2 }`)**:
   $$\frac{\llbracket c \rrbracket(\iota, \sigma) = \text{true} \quad \langle S_1, \sigma \rangle \Downarrow \sigma'}{\langle \text{if }(c)\{S_1\}\text{ else }\{S_2\}, \sigma \rangle \Downarrow \sigma'} \qquad \frac{\llbracket c \rrbracket(\iota, \sigma) = \text{false} \quad \langle S_2, \sigma \rangle \Downarrow \sigma'}{\langle \text{if }(c)\{S_1\}\text{ else }\{S_2\}, \sigma \rangle \Downarrow \sigma'}$$

7. **Staging-Register Window (`for_sr_read` vs `for_sr_write`)**:
   - `$sr_read_count` $= \max(1, \iota.\text{sr\_read.count})$ bounds input staging registers $\mathcal{R}[\iota.\text{sr\_read.base} + k]$ read via `@sr_get(k)` (e.g., `STORE`, `TEX_SINGLE`, `ATOM`, `BLEND`).
   - `$sr_write_count` $= \max(1, \iota.\text{sr\_write.count})$ (and `$sr_write_count_max4` $= \min(4, \text{count})$) bounds output staging registers $\mathcal{R}[\iota.\text{sr\_write.base} + k]$ written via `@sr_set(k, v)` (e.g., `LOAD`, `LD_ATTR`, `LD_VAR`, `TEX_SINGLE`).
   - Instructions like `TEX_SINGLE` use **both** windows simultaneously (`sr_read` for `(u, v)` coordinates, `sr_write` for `RGBA` output).

---

## 4. References of free `%variables` / `$metavariables` / `#immediates` and `@functions`

All `@fn(...)` functions share the uniform C++ ABI:
`RET abi::fn(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule, ...)`

| Token | Kind / Signature | Semantics |
| :--- | :--- | :--- |
| `%0 .. %3` | Source operand | Lazy `fetch_src_*` using `src_types[k]` |
| `%var` | Local variable | `SsaId` or `uint32_t` loop index (`v_var`) |
| `#raw0` | Raw source | `fetch_raw_src_u32(ctx.sb, blk, ins, 0)` |
| `#imm_u32` / `#imm_s32` / `#imm_f32` | 32-bit immediate | `emit_const_u32` / `emit_const_f32_bits` |
| `#imm_u64` / `#imm_u64_mask63` | 64-bit immediate | `emit_const_u64(ins.imm_signed)` / `(ins.imm_unsigned & 63u)` |
| `#fp_zero` / `#token` | Context atoms | `0.0f` (`F32`/`V2F32`) / `ctx.sb.current_token(blk)` |
| `#u32(x)` / `#s32(x)` / `#u64(x)` / `#f32(x)` | Literal constants | `emit_const_*` |
| `#u64_mul4(%i)` / `#f32_alpha_or_zero(%i)` | Loop-indexed consts | `uint64_t(v_i)*4ULL` / `v_i==3 ? 1.0f : 0.0f` |
| `#field_u32(f)` / `@field_u32_mul16(f)` | Instruction field | `ins.fields.at(f)` / `ins.fields.at(f) * 16u` |
| `#lookup_u32(tbl, $modifiers.k)` | Table `SsaId` const | `emit_const_u32(lookup_tbl(mod))` |
| `#lookup_u32_raw(tbl, $modifiers.k)` | Table raw `uint32_t` | `lookup_tbl(mod)` (for `uint32_t` ABI args) |
| `$branch_target_pc_u64` | PC-relative target | `has_branch_offset ? branch_target_pc : pc + 8 + imm_signed` |
| `$has_src0..3`, `$has_dest` | Operand presence | `ins.srcs.size() >= k+1` / `ins.has_dest` |
| `$is_v2f16`, `$is_signed`, `$is_64bit`, `$dest_is_64bit` | Variant/width preds | Variant / `dest_is_64bit` checks |
| `$is_frag_stage`, `$is_vert_stage` | Stage predicates | `ctx.opts.stage == Fragment` / `Vertex` |
| `$sr_write_count`, `$sr_write_count_max4`, `$sr_read_count` | Loop bounds | Staging register window counts |
| --- | --- | --- |
| `@has_field(f`, `@field_nz(f` | Field predicates | `ins.fields.count(f) > 0` / `!= 0` |
| `@mod_active(k)`, `@mod_eq(k,v)`, `@mod_ne(k,v)` | Modifier predicates | `ins.modifier_strs` equality checks |
| `@load_sideband_u64` | `(uint32_t) -> SsaId` | Emits `LoadSidebandU64(member_idx)` and sets `uses_sideband` |
| `@split64_lo` | `(SsaId) -> SsaId` | Emits `Split64` and returns low 32-bit `U32` |
| `@load_fau` | `(uint32_t) -> SsaId` | Emits `LoadFau(word_idx & 63)` |
| `@bitcast` | `(ValType, SsaId) -> SsaId` | Emits `Bitcast` to LHS `ValType` |
| `@extract` | `(ValType, SsaId, uint32_t) -> SsaId` | Emits `VecExtract(vec, idx)` with LHS `ValType` |
| `@shuffle2` | `(ValType, SsaId, SsaId, uint32_t, uint32_t) -> SsaId` | Emits `VecShuffle(a, b, i0, i1)` |
| `@bfi` | `(SsaId, SsaId, uint32_t, uint32_t) -> SsaId` | Emits `Bfi(a, b, off, width)` |
| `@set_token` | `(SsaId) -> void` | Updates `sb.set_token(blk, tok)` |
| `@set_branch_cond` | `(SsaId) -> void` | Sets `mod.blocks[blk].branch_cond = cond` |
| `@sr_get` | `(uint32_t) -> SsaId` | Reads `sb.current_reg(blk, sr_read.base_reg + offset)` |
| `@sr_set` | `(uint32_t, SsaId) -> void` | Writes `sb.set_reg(blk, sr_write.base_reg + offset, val)` |
| `@tuple_val` | `(ValType, SsaId) -> SsaId` | Emits `TupleExtract(tup, 0)` with LHS `ValType` |
| `@tuple_tok` | `(SsaId) -> SsaId` | Emits `TupleExtract(tup, 1)` (`ValType::Token`) |
| `@psb_load` | `(SsaId, uint32_t, bool, SsaId) -> SsaId` | Emits `PsbLoad(addr, elem_bytes, is_pka, tok)` -> `TupleValToken` |
| `@psb_store` | `(SsaId, SsaId, uint32_t, SsaId) -> SsaId` | Emits `PsbStore(addr, val, elem_bytes, tok)` -> `Token` |
| `@atomic_rmw` | `(SsaId, SsaId, bool, SsaId) -> SsaId` | Emits `AtomicRmw(addr, arg, is_64, atom_opc, tok)` -> `TupleValToken` |
| `@atomic_rmw_aadd` | `(SsaId, SsaId, bool, SsaId) -> SsaId` | Emits `AtomicRmw(addr, arg, is_64, "aadd", tok)` -> `TupleValToken` |
| `@atomic_xchg` | `(SsaId, SsaId, SsaId) -> SsaId` | Emits `AtomicExchange(addr, arg, tok)` -> `TupleValToken` |
| `@atomic_cmpxchg` | `(SsaId, SsaId, SsaId, SsaId) -> SsaId` | Emits `AtomicCmpXchg(addr, xchg, cmp, tok)` -> `TupleValToken` |
| `@load_vert_attr` | `() -> SsaId` | Emits `LoadVertAttr(attribute_index)` -> `V4F32` |
| `@load_frag_var` | `(bool) -> SsaId` | Emits `LoadFragVarying(index / 16, is_flat)` -> `V4F32` |
| `@load_frag_var_idx` | `(uint32_t, bool) -> SsaId` | Emits `LoadFragVarying(var_loc, is_flat)` -> `V4F32` |
| `@tex_sample` | `(SsaId, bool, SsaId) -> SsaId` | Emits `TexSample(uv, is_gather, gather_component, tok)` -> `TupleValToken` |
| `@store_builtin_pos` | `(SsaId, SsaId) -> SsaId` | Emits `StoreBuiltinOutput(Position, pos_vec, tok)` -> `Token` |
| `@store_builtin_depth` | `(SsaId, SsaId) -> SsaId` | Emits `StoreBuiltinOutput(FragDepth, z_f32, tok)` -> `Token` |
| `@store_vert_varying` | `(SsaId, SsaId) -> SsaId` | Emits `StoreVertVarying(imm_signed / 16, vec, tok)` -> `Token` |
| `@store_frag_color` | `(SsaId, SsaId) -> SsaId` | Emits `StoreFragColor(blend_descriptor_N, col, tok)` -> `Token` |
| `@kill_cond` | `(SsaId, SsaId) -> void` | Emits conditional `Kill(cond, tok)` |
