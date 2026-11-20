# Formal Specification: Valhall SSA Dataflow & Pointer Analysis (`src/dataflow/`)

This document formalizes the abstract domain, complete lattice, CFG dataflow equations (forward pointer/resource analysis and backwards SSA liveness), and SSA transfer functions implemented in `src/dataflow/`.

---

## 1. SSA Program Representation (`src/ssa/valhall_ssa.h`)

A Valhall shader is lowered by Pass 1 (`valhall_pass1.cpp`) into a `ValhallSsaModule` $M = (B, V, E)$:
- $B = \{B_0, B_1, \dots, B_{k-1}\}$ is the set of basic blocks (`SsaBlock`), with entry block $B_0$ and directed CFG edges $E \subseteq B \times B$ given by `fallthrough_block` and `branch_target_block`.
- $V = \{1, 2, \dots, N\}$ is the set of SSA value identifiers (`SsaId`), where every $v \in V$ has a unique defining `SsaNode` $n_v$ in some block $B(v) \in B$.
- Each block $B_i$ consists of:
  1. A sequence of $\phi$-nodes $\Phi(B_i) \subseteq V$ at block entry, where each $\phi$-node $p \in \Phi(B_i)$ has incoming pairs $\{(P_1, v_1), \dots, (P_m, v_m)\}$ with $P_j \in \text{pred}(B_i)$.
  2. A sequence of body nodes $\text{Nodes}(B_i) \subseteq V$ in program order.
  3. An optional branch condition $\text{cond}(B_i) \in V \cup \{0\}$.
- Every lowered `SsaNode` carries `(orig_inst_idx, orig_pc)` mapping it back to its originating Valhall ISA instruction so SSA-level analysis results project cleanly onto source ISA disassembly.

---

## 2. Abstract Domain & Complete Lattice $(\mathcal{L}, \sqsubseteq, \sqcup, \sqcap, \bot, \top)$

### 2.1 Value Lattice $\mathcal{L}$ (`TrackedValDomain`)

Every SSA value is abstracted to an element of the bounded lattice $\mathcal{L}$ (`TrackedVal`):

$$\mathcal{L} = \{\bot, \top\} \;\cup\; \text{ConstInt}(\mathbb{Z}_{64}) \;\cup\; \text{Abi}(\text{Label}) \;\cup\; \mathcal{P} \;\cup\; \mathcal{R}$$

where:
- **Pointer-like kinds ($\mathcal{P}$)** carry a base identity $\beta$, a byte offset $\delta \in \mathbb{Z} \cup \{?\}$, a `derived` flag $d \in \{\text{false}, \text{true}\}$, and a split-half tag $h \in \{\text{lo}, \text{hi}\}$:
  $$\mathcal{P} = \{\text{FauDstBase}, \text{SrtBase}, \text{SrtRackPtr}(r), \text{DescriptorAddr}(r, i, s), \text{BufferAddr}(r, i, s), \text{DataSectionPtr}, \text{TlsPtr}, \text{WlsPtr}\}$$
- **Resource/value kinds ($\mathcal{R}$)** represent non-pointer shader resources:
  $$\mathcal{R} = \{\text{BufferSize}(r, i, s), \text{FauUniformRef}(o, s), \text{SymbolValue}(r, i, s, d), \text{VertexAttrValue}(\ell, s), \text{VaryingValue}(\ell, s), \text{TexSampleValue}(\text{img}, \text{smp}, s), \text{BlendDescRef}(\ell, s)\}$$

### 2.2 Partial Order $\sqsubseteq$ (`TrackedValDomain::leq`)

```text
                             ⊤ (Top)
            /           |               |             \
   ConstInt(c)    Abi(lbl)      P(β, off=?, d=true)    SymVal(s, d=true)
                                      |                       |
                                P(β, off=δ, d=false)   SymVal(s, d=false)
            \           |               |             /
                             ⊥ (Bottom)
```

Formally, $a \sqsubseteq b$ holds iff any of the following hold:
1. $a = \bot$ or $b = \top$ or $a = b$.
2. $a, b \in \mathcal{P}$ share the same base identity $\beta(a) = \beta(b)$ and split half $h(a) = h(b)$, and:
   - $a$ has a known offset ($\delta_a \in \mathbb{Z}$) while $b$ has an unknown offset ($\delta_b = ?$), with $d_a \implies d_b$; OR
   - $\delta_a = \delta_b$ and $d_a = \text{false}, d_b = \text{true}$.
3. $a, b \in \text{SymbolValue}$ share the same $(r, i, s)$ with $d_a = \text{false}, d_b = \text{true}$.

### 2.3 Join ($\sqcup$) and Meet ($\sqcap$)

$$a \sqcup b = \begin{cases}
b & \text{if } a = \bot \\
a & \text{if } b = \bot \\
\top & \text{if } a = \top \lor b = \top \\
a & \text{if } a = b \\
P(\beta, \delta_a \sqcup_\delta \delta_b, d_a \lor d_b) & \text{if } a, b \in \mathcal{P} \land \beta(a) = \beta(b) \land h(a) = h(b) \\
\text{SymbolValue}(r, i, s, d_a \lor d_b) & \text{if } a, b \in \text{SymbolValue} \land \text{same\_id}(a, b) \\
\top & \text{otherwise}
\end{cases}$$

where $\delta_a \sqcup_\delta \delta_b = \delta_a$ if $\delta_a = \delta_b \in \mathbb{Z}$, else $?$.

Dually, $a \sqcap b$ returns $b$ if $a = \top$, $a$ if $b = \top$, $\bot$ if either is $\bot$, $a$ if $a \sqsubseteq b$, $b$ if $b \sqsubseteq a$, and $\bot$ otherwise.

**Termination Guarantee:** Because every chain in $\mathcal{L}$ has length at most $h(\mathcal{L}) = 3$ ($\bot \sqsubset P(\beta, \delta, \text{false}) \sqsubset P(\beta, ?, \text{true}) \sqsubset \top$), the product lattice over $|V|$ SSA values and finite FAU offsets has finite height, so chaotic iteration of monotone transfer functions is guaranteed to converge to the least fixpoint.

---

## 3. Abstract State $\sigma$ (`AbstractState<DomainT>`)

$$\sigma = (\rho, \mu_{\text{FAU}}, \mu_{\text{sym}})$$

- $\rho : V \to \mathcal{L}$ maps each `SsaId` to its abstract value (initially $\lambda v.\,\bot$).
- $\mu_{\text{FAU}} : \mathbb{N} \to \mathcal{L}$ maps FAU uniform byte offsets to abstract values (seeded from `env.initial_fau` and updated by Pilot `PsbStore`s to `FauDstBase`).
- $\mu_{\text{sym}} : \text{String} \to \mathcal{L}$ maps symbolic memory locations to abstract values.

State join $\sigma_1 \sqcup \sigma_2$ and meet $\sigma_1 \sqcap \sigma_2$ are defined pointwise on $\rho$, $\mu_{\text{FAU}}$, and $\mu_{\text{sym}}$.

---

## 4. Forward CFG Dataflow Equations (`run_ssa_forward_dataflow`)

For every basic block $B \in \text{blocks}(M)$:

$$\text{IN}[B] = \begin{cases}
\sigma_{\text{entry}} \sqcup \bigsqcup_{P \in \text{pred}(B)} \text{OUT}[P] & \text{if } B = B_0 \\
\bigsqcup_{P \in \text{pred}(B)} \text{OUT}[P] & \text{if } B \neq B_0
\end{cases}$$

$$\text{OUT}[B] = \mathcal{F}_B(\text{IN}[B]), \qquad \mathcal{F}_B = \llbracket n_k \rrbracket^\# \circ \dots \circ \llbracket n_1 \rrbracket^\# \circ \llbracket \Phi(B) \rrbracket^\#$$

After fixpoint convergence ($\text{OUT}[B]^{(t+1)} = \text{OUT}[B]^{(t)}$ for all $B$), the unified module-wide SSA environment is:

$$\sigma_{\text{final}} = \bigsqcup_{B \in \text{blocks}(M)} \text{OUT}[B]$$

---

## 5. Backwards SSA Liveness Equations (`src/dataflow/liveness.{h,cpp}`)

SSA $\phi$-nodes evaluate their incoming operands along incoming CFG edges rather than inside the target block. For each block $B$:
- $\text{PHI\_DEFS}[B] = \{ p \mid p \in \Phi(B) \}$
- $\text{PHI\_USES}(B \to S) = \{ v \mid p \in \Phi(S),\; (B, v) \in \text{incomings}(p) \}$
- $\text{DEF}[B] = \{ n \mid n \in \text{Nodes}(B) \}$
- $\text{UE\_USE}[B]$ = operands used in $\text{Nodes}(B)$ or $\text{cond}(B)$ prior to any definition in $B$.

The backwards fixpoint equations are:

$$\text{LIVE\_OUT}[B] = \bigcup_{S \in \text{succ}(B)} \Big( \big(\text{LIVE\_IN}[S] \setminus \text{PHI\_DEFS}[S]\big) \cup \text{PHI\_USES}(B \to S) \Big)$$

$$\text{LIVE\_IN}[B] = \text{UE\_USE}[B] \cup \big(\text{LIVE\_OUT}[B] \setminus \text{DEF}[B]\big)$$

Walking $\text{Nodes}(B)$ in reverse from $\text{live} = \text{LIVE\_OUT}[B] \cup \{\text{cond}(B)\}$, an operand $u \in \text{args}(n)$ is **killed** at node $n$ iff $u \notin \text{live}_{\text{after}}(n)$:

$$\text{KILLED\_AFTER}[n] = \{ u \in \text{args}(n) \mid u \notin \text{live}_{\text{after}}(n) \}$$

---

## 6. SSA Transfer Functions $\llbracket \text{node} \rrbracket^\#(\sigma)$

### 6.1 Extensional Equalities (`SsaDomainTransferVisitor<DomainT>`)

| SSA Op Group | Transfer Equation $\rho'[d]$ |
|---|---|
| `Copy(x)`, `Bitcast(x)` | $\rho(x)$ |
| `Phi((P_1, v_1), \dots, (P_m, v_m))` | $\bigsqcup_{v_i \neq d} \rho(v_i)$ |
| `Select(c, a, b)` | $\rho(a)$ if $a = b$, else $\rho(a) \sqcup \rho(b)$ |
| `IAdd(x, y)` / `IAdd64(x, y)` | $\rho(x)$ if $\rho(y) = \text{Const}(0)$; $\rho(y)$ if $\rho(x) = \text{Const}(0)$; $\text{Const}(c_x + c_y)$ if both const |
| `ISub(x, y)` / `ISub64(x, y)` | $\text{Const}(0)$ if $x = y$; $\rho(x)$ if $\rho(y) = \text{Const}(0)$; $\text{Const}(c_x - c_y)$ if both const |
| `IMul(x, y)` / `IMul64(x, y)` | $\text{Const}(0)$ if either $0$; $\rho(x)$ if $\rho(y) = 1$; $\rho(y)$ if $\rho(x) = 1$ |
| `BitAnd(x, y)` / `BitAnd64(x, y)` | $\rho(x)$ if $x = y$ or $\rho(y) = \sim 0$; $\text{Const}(0)$ if either $0$ |
| `BitOr(x, y)` / `BitOr64(x, y)` | $\rho(x)$ if $x = y$ or $\rho(y) = 0$; $\rho(y)$ if $\rho(x) = 0$ |
| `BitXor(x, y)` / `BitXor64(x, y)` | $\text{Const}(0)$ if $x = y$; $\rho(x)$ if $\rho(y) = 0$; $\rho(y)$ if $\rho(x) = 0$ |
| `Shl`/`ShrU`/`ShrS`/`RotL`/`RotR(x, s)` | $\rho(x)$ if $\rho(s) = \text{Const}(0)$; constant-folded if both const |

### 6.2 Pointer, Resource & ABI Transfer Functions (`SsaPointerTransferVisitor`)

| SSA Op | Transfer Equation |
|---|---|
| `ConstU32/S32/U64/Bool/F32(c)` | $\rho'[d] = \text{ConstInt}(c)$ |
| `LoadSidebandU64(idx)` (Pilot) | $\rho'[d] = \text{FauDstBase}(0)$ if $\text{idx}=3$; $\text{SrtBase}(0)$ if $\text{idx}=2$ |
| `LoadBuiltinInput(bi, name)` | $\rho'[d] = \text{AbiPreload}(\text{name})$ |
| `VecExtract(v, c)` | If $\rho(v) = \text{AbiPreload}(\text{lbl})$, $\rho'[d] = \text{AbiPreload}(\text{lbl}.\text{comp}(c))$ |
| `LoadFau(w)` | $\rho'[d] = \mu_{\text{FAU}}(4w)$ if present; else $\mu_{\text{FAU}}(4w \land \sim 7)_{\text{hi}}$ if $4w \equiv 4 \pmod 8$; else $\text{FauUniformRef}(4w)$ |
| `Split64(x)` | $\rho'[d] = \rho(x)$ |
| `TupleExtract(t, i)` | $\rho'[d] = \rho(t)_{\text{lo}}$ ($i=0$); $\rho(t)_{\text{hi}}$ ($i=1$, or $\text{Const}(c \gg 32)$) |
| `Pack64(lo, hi)` | Reconstructs 64-bit pointer from $\rho(\text{lo})$ (clearing `is_hi_half`), or folds $\text{Const}((c_{\text{hi}} \ll 32) \mid c_{\text{lo}})$ |
| `IAdd/IAdd64(p, k)` ($p \in \mathcal{P}$) | If $\rho(k) = \text{Const}(\Delta)$: $p$ with offset $\delta + \Delta$ (promoting `SrtRackPtr(r, 32j)` $\to$ `DescriptorAddr(r, j)`). Otherwise: $p$ with $\delta = ?$. |
| `BitAnd/BitAnd64(p, m)` ($p \in \mathcal{P}$) | If $\rho(m) = \text{Const}(M \neq 0)$: preserves pointer base $p$ with $\delta' = \delta \land M$. |
| `PsbLoad(addr, sz)` | Dereferences $\rho(\text{addr})$:<br>• $\text{SrtBase}(16r) \to \text{SrtRackPtr}(r, 0)$<br>• $\text{SrtRackPtr}(r, 32j + f) \to \text{BufferSize}(r, j)$ ($f=4, \text{sz}=4$) or $\text{BufferAddr}(r, j, 0)$ ($f=8 \lor \text{sz}=8$)<br>• $\text{DescriptorAddr}(r, j, f) \to \text{BufferSize}$ ($f=4$) or $\text{BufferAddr}(r, j, 0)$<br>• $\text{BufferAddr}(r, j, \delta) \to \text{SymbolValue}(r, j, \text{member}(r, j, \delta))$<br>• $\text{FauDstBase}(\delta) \to \text{FauUniformRef}(\delta)$ |
| `PsbStore(addr, val, sz)` | If $\rho(\text{addr}) = \text{FauDstBase}(\delta)$ with known $\delta \ge 0$: updates $\mu'_{\text{FAU}}(\delta) = \mu_{\text{FAU}}(\delta) \sqcup \rho(\text{val})$ and records `PilotFauWrite{delta, sz, val}`. |
| `LoadVertAttr(loc)` | $\rho'[d] = \text{VertexAttrValue}(\text{loc}, \text{sym}(\text{loc}))$ |
| `LoadFragVarying(loc)` | $\rho'[d] = \text{VaryingValue}(\text{loc}, \text{sym}(\text{loc}))$ |
| `TexSample(img, smp)` | $\rho'[d] = \text{TexSampleValue}(\text{img}, \text{smp}, \text{sym}(\text{img}))$ |

---

## 7. Worked Example: Pilot Shader SRT Pointer Chasing in SSA

Consider a Mali Compute Pilot Shader loading SSBO binding 0 (`_7163_input_rgba8_buffer`, located at `SRT[4]`, descriptor `0`) and storing its 64-bit GPU VA to `FAU_0 + 0x00`:

```text
  %1 = LoadSidebandU64(3)              => ρ(%1) = FauDstBase(+0x0)
  %2 = LoadSidebandU64(2)              => ρ(%2) = SrtBase(+0x0)
  %3 = ConstU64(0x40)                  => ρ(%3) = ConstInt(0x40)
  %4 = IAdd64(%2, %3)                  => ρ(%4) = SrtBase(+0x40)          [rack = 0x40 / 16 = 4]
  %5 = PsbLoad(%4, 8)                  => ρ(%5) = SrtRackPtr(rack=4, +0)
  %6 = TupleExtract(%5, 0)             => ρ(%6) = SrtRackPtr(rack=4, +0)
  %7 = ConstU64(0x08)                  => ρ(%7) = ConstInt(0x08)
  %8 = IAdd64(%6, %7)                  => ρ(%8) = SrtRackPtr(rack=4, +8)  [desc=0, field=8]
  %9 = PsbLoad(%8, 8)                  => ρ(%9) = BufferAddr(rack=4, desc=0, "_7163_input_rgba8_buffer")
 %10 = TupleExtract(%9, 0)             => ρ(%10) = BufferAddr(rack=4, desc=0, "_7163_input_rgba8_buffer")
 %11 = PsbStore(%1, %10, 8)            => μ_FAU[0x00] = BufferAddr(4, 0, "_7163_input_rgba8_buffer")
```

The main compute shader then seeds `env.initial_fau[0x00] = BufferAddr(4, 0, "_7163_input_rgba8_buffer")`, so every `LoadFau(0)` + `LoadFau(1)` + `Pack64` immediately resolves to `&_7163_input_rgba8_buffer (SRT[4]->Buf[0].addr)` across the entire main shader CFG!
