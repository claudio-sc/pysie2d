# Slab-waveguide study (v1.2, roadmap §9 steps 1–3)

Throwaway prototypes behind the v1.2 code-spec
([waveguide-roadmap.md](../../waveguide-roadmap.md)). They are not package code
and not lint-checked (`docs/design/studies` is excluded from ruff). They import
only `pysie2d.layered` (its α sheet, `fresnel_r`, `_tau_half_space` and the path
constants). Run from this directory:

    VECLIB_MAXIMUM_THREADS=4 uv run --project ../../../.. python s3_green.py

Each script runs in under 10 s, except `s4_cutoff.py` (~3 min, QUADPACK references). Its output is committed beside it as `.txt`.

| Script | What it measures | Status |
|---|---|---|
| `sproto.py` | Shared code: `Stack` (stable recursion for R and τ, naive transfer matrix, dispersion function D), guided-pole bracketing in q and in γ, argument-principle counts in the q- and γ-planes, Cauchy-circle residues, the three-layer closed form, a path builder that takes a feature list, the path G, and two real-axis references (cosh and γ variables) | — |
| `s1_recursion.py` | Reductions to v0.9, lossless energy, the large-q asymptote, and where the naive form fails | done |
| `s2_poles.py` | A2: guided poles against the closed-form slab dispersion; mode counting; simple poles | done |
| `s3_green.py` | A3: reflected G on the feature-aware path against a real-axis reference with explicit pole indentation | done |
| `s4_cutoff.py` | Risk §7.1: poles at (β − k_out)/k = 1e-1 … 1e-8; mode counting with near-degenerate supermodes | done |

## Frame and conventions used here

As v0.9, extended downward. The cover (the background, ε = 1 relative) lies
above `z_top`. Layers j = 1..N run downward with thicknesses d_j and relative
permittivities ε_j. The bottom half-space starts at `z_bot = z_top − Σd`. The
downward wave is ∝ e^{−iαz}. Wavenumbers are background (`k = 2π n_clad/λ`).

- **Admittance** Y = α (TE, `pol = 2`, E_y) or α/ε (TM, `pol = 1`, H_y). One
  recursion serves both, and t = 1 + r = 2Y_a/(Y_a + Y_b) at every interface.
- **Outer sheets**: the package's vertical-cut `alpha` for the cover and the
  bottom. **Interior sheets**: R and τ are even in every interior α_j, so the
  prototype picks Im α_j ≥ 0, which makes every |e^{2iα_j d_j}| ≤ 1. The choice
  is discontinuous in q, but R is not, by evenness.
- **Recursion** (`Stack._gammas`): Γ at the top of each medium, from the bottom
  up: G_a = (r + Γ_b)/(1 + rΓ_b), and Γ_j = G_j·e^{2iα_j d_j}.
  τ = T/α₁ = [2/(Y₀+Y₁)]·Π_{i≥1} t_i e^{iα_i d_i} · Π_i 1/(1 + r_i Γ_{i+1}). It is
  finite at q = ±k₁ for the same reason as v1.1's τ.

## Results

### S1: the recursion (`s1_recursion.txt`)

- **N = 0 is bit-identical** to `fresnel_r` and `_tau_half_space`. Max |Δ| = 0
  for glass, Si and Ag substrates, TE and TM. This is the G7-style regression
  for the `HalfSpace` → `Stack(N = 0)` change.
- **Every layer equal to the cover** gives |R| ≤ 7e-16. α₁τ equals the free
  propagation e^{iα₁Σd} to 1e-15. Trap: the removable 0/0 at q = k₁ exactly gives
  NaN when ε = 1 throughout. The half-space has the same property. A path never
  samples it.
- **Lossless energy**, |R|² + (Re Y_b/Y₁)|α₁τ|² = 1 at propagating q: 2e-15 on
  SOI, a coupled double core and an air-bottom stack. τ is anchored this way
  independently of R.
- **Large-q asymptote**: R → r(cover | layer 1), with a remainder at round-off
  once e^{−2|Im α₁|d₁} < 1e-16.
- **The naive transfer matrix does not lose digits. It overflows.** The
  growing parts of the numerator and denominator cancel in the ratio, so it
  agrees with the recursion to 1e-14 (relative) until Σ_j |Im α_j|·d_j ≈ 709.
  Just before that it returns O(1) garbage, then NaN. On a real path the tail
  reaches q_max = 1.5·k_core + 38/z_min, so it fails at:

  | stack | gap 100 nm | gap 20 nm | gap 5 nm |
  |---|---|---|---|
  | SOI 220 nm | ok (q·Σd = 46) | ok (214) | **NaN** (841) |
  | SiN 400 nm | ok (81) | ok (385) | **NaN** (1525) |
  | Si/SiO₂/Si 220/300/220 | ok (156) | **NaN** (719) | **NaN** (2828) |

  **The test that fails on the naive form** (roadmap §4 item 1) is therefore
  R on the actual path for SOI at a 5 nm gap: finite and correct by the
  recursion, NaN by the transfer matrix.
- **Either cure is exact.** R is homogeneous of degree 0 in the transfer
  matrix, so scaling each layer's matrix by e^{−|Im α_j|d_j}, with the entries
  formed from bounded exponentials, also removes the overflow (`R_ref`). The
  package should take the recursion anyway, for three reasons: N = 0 is then
  bitwise v0.9, τ comes out of the same Γs, and the stack's reflection from below
  (item 6) is the same loop run upward.

### S2: guided poles, anchor A2 (`s2_poles.txt`)

Five three-layer slabs (symmetric, asymmetric with an air cover, a 2 µm
multimode core, a near-cutoff SiN) and a coupled double core.

- **Real roots of D agree with the textbook** κd = mπ + atan(p_c/κ) +
  atan(p_s/κ) to 0–5e-16 relative, TE and TM, up to 9 modes.
- **The counts agree three ways**: sign-change bracketing, the closed form, and
  the argument principle on D around the guided window. They also agree on the
  double core, which has no closed form (2 TE, 2 TM).
- **Every root is a simple pole of the stable R.** |1/R(β)| ≤ 5e-13,
  D'(β) ≠ 0 (smallest |D'|·k = 0.02), and Im D/D' ≤ 1e-16 at the root, so D is
  real on the window as claimed. The **residue** by a Cauchy circle on the
  recursion equals N(β)/D'(β) from the transfer matrix to 4e-15 (circle radius
  1e-2 to 1e-3·β; 1e-4·β is the finite-difference floor, 2e-13).
- SOI 220 nm in SiO₂ at 1550 nm: n_eff(TE₀) = 2.84778, n_eff(TM₀) = 2.05332.
- The near-cutoff cases occur naturally. Si 2000 TM₈ has (β − k_out)/k = 4e-3;
  SiN 400 TM₁ has 3.6e-2.

### S3: reflected G with guided poles, anchor A3 (`s3_green.txt`)

**The v0.9 path needs no change except the feature list.** Passing the guided
poles as features sets T = 1.5·β_max. The clearance rule then holds with room:
the worst pole is 0.67·δ from the hump, against the 0.2·δ rule. Node counts are
the v0.9 rule unchanged (M ≈ 2800 at D = 1500 nm, z_min = 20 nm).

The reference shares no ingredient with the path that could hide a common
error. It uses the scaled transfer matrix (not the recursion), and the
radiation-condition branch rule (not the vertical-cut sheet). It runs QUADPACK
on the real axis with θ/cosh substitutions (not Gauss–Legendre on a hump). Each
real pole is taken as a PV by the symmetric fold ∫₀ʰ[f(s_p+u) + f(s_p−u)]du plus
iπ·Res, for the path passing below +β. The fold uses no residue, so a wrong
residue cannot be absorbed into the PV.

Five stacks, TE and TM, 4 points (X, Z) from (0, 20) to (1500, 100) and
(40, 400) nm, G and ∂z′G:

- **Path vs reference: ≤ 4e-10 everywhere, typically 1e-11 to 1e-14.** The
  worst rows are the 9-mode Si 2000 TM.
- **That residual is the reference's floor.** It is flat from s = 1 to 2, and
  changing only the reference's fold half-width (0.45 → 0.1) moves the
  reference by the same amount: 4e-12 on SOI TM, 2.5e-10 on Si 2000 TM.
- **The path is path-independent to 2e-14**, comparing hump depth δ/2 with δ.
- s = 0.5 costs up to 1e-9 on the asymmetric air-cover stack, the largest T/δ.
  This is consistent with v0.9's node rule having no slack below s = 0.75.

Traps hit while building the reference, worth keeping for the spec's own
reference: quad's `weight="cauchy"` gave a 5e-4 error estimate on the pole panel,
which is why the fold is used. The tail limit must be extended in q, not in
s = arccosh(q/k). A transfer matrix must be scaled **before** forming cos,
because cos overflows first.

### S4: near cutoff and near degeneracy (`s4_cutoff.txt`)

**(a) Near cutoff.** One mode is placed at (β − k_out)/k = t, for
t = 1e-1 … 1e-8. β is chosen first, then the core thickness comes from the closed
form d = (mπ + atan(p_c/κ) + atan(p_s/κ))/κ. So the pole location is known by
construction, not found. Two cases: TE₁ and TM₁ of a symmetric SOI slab, where
k_out = k₁, and the fundamental mode of air|SiN|SiO₂, a true cutoff at k_bot.

- **The path does not notice.** G stays converged (s = 2 vs 1: ≤ 5e-14) and
  path-independent (δ/2 vs δ: ≤ 5e-14) at every t. M (2825–2841), T and the
  clearance (0.84–0.87·δ) are flat in t. Pole and branch point merge, but both
  stay 0.87·δ from the hump.
- **Path vs reference: ≤ 1e-9 for t ≥ 1e-6, ≤ 3e-8 at t = 1e-8**, and part (c)
  shows that this is the reference's spread.
- **The guided term fades continuously**: |Res R| ∝ t (symmetric) or ∝ √t
  (asymmetric), and the pole's share of |G| falls like √t in both, to 2e-3
  (symmetric) and 3e-4 to 1e-3 (asymmetric) at t = 1e-8.
  So near cutoff is benign for G. Whether it is benign for the
  power-normalised mode amplitude (P_m ∝ 1/γ grows) is S8's question.
- **The residue** from a circle of radius 0.3·(β − k_out), which must not enclose
  the branch point, agrees with N/D' to ε/t (6e-10 at t = 1e-8).
- **The q-grid finder misses near-cutoff modes.** It loses the mode at
  t ≤ 1e-4 (symmetric) and t ≤ 1e-6 (asymmetric). The q-plane winding count,
  whose box starts at k_out + 1e-6·width, loses it at 1e-6 to 1e-8. A missed
  mode is silent: G is still right and the S-matrix is a row short. **Bracketing
  in γ = √(q² − k_out²) from γ = 0 finds every one**, at exactly the constructed
  β. Near cutoff D is linear in γ, not in q, and D(γ = 0) is finite.
- **The cosh-variable reference fails as t → 0** (part c): at t = 1e-8, changing
  its fold half-width moves it by O(1). In the cosh variable the pole is t·k_out
  from the √ branch point at k_out, and in the asymmetric case also from the end
  of the [k₁, k_out] panel. `g_ref_gamma` integrates q ≥ k_out in γ, which
  makes the branch point a smooth endpoint and puts the pole √(2t)·k_out away.
  It also takes the upper half of [k₁, k_out] in y = √(k_out² − q²). It stays at
  ≤ 4e-8 under the same change. Both apparent path errors (8e-9 symmetric, 2e-4
  asymmetric) were the cosh reference.

**(b) Near degeneracy.** Si 220 | SiO₂ d_ox | Si 220 in SiO₂: the supermode
splitting falls like e^{−γ_ox d_ox}.

| d_ox (nm) | TE Δβ/β | TM Δβ/β | found by q-grid / γ-grid / γ-grid ×100 (TE; TM) | winding q / γ |
|---|---|---|---|---|
| 300 | 1.4e-2 | 6.9e-2 | 2/2/2; 2/2/2 | 2/2 |
| 1000 | 1.3e-5 | 1.1e-3 | **0/0**/2; 2/2/2 | 2/2 |
| 2000 | — | 3.0e-6 | **0/0/0**; **0/0**/2 | 2/2 |
| ≥ 3000 | — | — | **0/0/0**; **0/0/0** | 2/2 |

- **No bracketing grid sees a close pair.** Two roots in one cell give no sign
  change, and both modes vanish together. This is not a rounding problem.
- **The winding count stays right (2/2) throughout**, so the count detects
  what the bracketing misses.
- Beyond d_ox ≈ 3 µm the splitting (≈ e^{−30} and smaller) is below what D
  can resolve. The pair is a numerical double root, the case holomorphy §5
  flags (dD/dq ≠ 0). **The path is indifferent** (δ/2 vs δ ≤ 2e-14 wherever
  both roots were found). Only the per-mode residues are at stake.
- The γ-plane count has a sheet trap: Re γ > 0 is the Im α ≥ 0 sheet, which is
  not the code's vertical-cut sheet left of k_out. A box of height 0.05·γ_hi
  counted a leaky pole at q = 0.99655·k_out + 6e-5i·k (d_ox = 6 µm), which is
  improper on the code's sheet. A box of 0.01·γ_hi counts correctly.
- The open item from S1–S3 (D overflowing on a thick BOX) does not arise:
  γ_ox·d_ox is 60 at 6 µm, against an overflow at 709. **Degeneracy comes long
  before overflow.**

## Plan for the rest of the study

Items S4–S6 measure the §7 risks and finish roadmap §9 step 1's scope. S7–S9
are §9 step 2 (pin the §3 formulas), and S10 is step 3's floor for A5. Each
script states its check before it runs.

| # | Script | Question | Pass condition / output |
|---|---|---|---|
| S4 | `s4_cutoff.py` | Near cutoff and near degeneracy | **done**, see above |
| S5 | `s5_gap.py` | Small gaps (z_min = 2 nm to 10 nm) with poles present: M, accuracy, and timing of the banded GEMM against v0.9's t14 numbers | path vs reference ≤ 1e-10; M ∝ 1/z_min as in v0.9 |
| S6 | `s6_lossy.py` | Lossy core (Im n = 1e-3, 1e-1), a thin metal layer, and complex k at Q = 10, 3. Poles leave the axis; continue them from the lossless roots by Newton on D | path vs reference (lossy, real k), δ-independence (complex k), and Im β > 0 at real k |
| S7 | `s7_transmitted.py` | τ through the stack: the transmitted G below z_bot against a real-axis reference (extends v1.1 GT2) | ≤ the S3 floor |
| S8 | `s8_residue.py` | **Mode normalisation and the residue–overlap identity.** Res_β R against the closed form built from the normalised profile u_m(z) (the 1/ε weight in TM), and the overlap form of c_m^± | pinned formula, agreement at round-off on all S2 stacks |
| S9 | `s9_dipole.py` | **A4**: line-dipole emission into mode m. The residue amplitude against `∝ u_m(z₀)/P_m`, both polarisations; the far-x field of the path G against the residue term at large X | closed form at round-off; pins the residue, the normalisation and the phase reference |
| S10 | `s10_born.py` | **A5 floor**: small-circle S-parameters against the Born closed form, wired as a prototype on the package solver | the rad² rate and the gap dependence of the dressing corrections, measured |

The power balance (A6) and reciprocity (A7) need the assembled solver, so they
are gates of the v1.2 code-spec, not study items.

## Open after S1–S4

- **The mode finder, for the spec.** From S2 and S4: bracket D in γ from γ = 0,
  not in q. Check the bracket count against the argument-principle count in a
  thin γ box (or the q box) on every call. A shortfall means a close pair, so
  refine the cell the count points to, by subdividing the box. A pair that
  stays unresolved at round-off is a numerical double root.
- **What a near-degenerate pair means for S-parameters (a spec decision).**
  For two cores behind 1 µm of oxide (TE, Δβ/β = 1.3e-5) the beat length π/Δβ is
  already about 14 mm. At chip scale the supermode basis is meaningless and the
  core-localised one is physical. Options for the R9 code-spec: report the
  group's combined residue by one contour around the pair; switch to a localised
  basis; or warn and refuse. This is not measured further here.
- **A fixed hump suffices for real λ.** Whether it still does for slab QNMs is
  the low-priority later item (holomorphy §5), and it is not studied here.
