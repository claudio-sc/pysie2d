# Slab-waveguide study (v1.2, roadmap §9 steps 1–3)

Throwaway prototypes behind the v1.2 code-spec
([waveguide-roadmap.md](../../waveguide-roadmap.md)). They are not package code
and not lint-checked (`docs/design/studies` is excluded from ruff). They import
only `pysie2d.layered` (its α sheet, `fresnel_r`, `_tau_half_space` and the path
constants). Run from this directory:

    VECLIB_MAXIMUM_THREADS=4 uv run --project ../../../.. python s3_green.py

Each script runs in under 10 s. Its output is committed beside it as `.txt`.

| Script | What it measures | Status |
|---|---|---|
| `sproto.py` | Shared code: `Stack` (stable recursion for R and τ, naive transfer matrix, dispersion function D), guided-pole bracketing, argument-principle count, Cauchy-circle residues, the three-layer closed form, a path builder that takes a feature list, the path G, and the real-axis reference | — |
| `s1_recursion.py` | Reductions to v0.9, lossless energy, the large-q asymptote, and where the naive form fails | done |
| `s2_poles.py` | A2: guided poles against the closed-form slab dispersion; mode counting; simple poles | done |
| `s3_green.py` | A3: reflected G on the feature-aware path against a real-axis reference with explicit pole indentation | done |

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

## Plan for the rest of the study

Items S4–S6 measure the §7 risks and finish roadmap §9 step 1's scope. S7–S9
are §9 step 2 (pin the §3 formulas), and S10 is step 3's floor for A5. Each
script states its check before it runs.

| # | Script | Question | Pass condition / output |
|---|---|---|---|
| S4 | `s4_cutoff.py` | Near cutoff, β → k_out: does G on the path stay at the reference floor as (β−k_out)/k → 1e-6? What do T, δ, clearance and M do? | path vs reference at the S3 floor; M growth stated. A measured floor, not a gate (risk §7.1). |
| S5 | `s5_gap.py` | Small gaps (z_min = 2 nm to 10 nm) with poles present: M, accuracy, and timing of the banded GEMM against v0.9's t14 numbers | path vs reference ≤ 1e-10; M ∝ 1/z_min as in v0.9 |
| S6 | `s6_lossy.py` | Lossy core (Im n = 1e-3, 1e-1), a thin metal layer, and complex k at Q = 10, 3. Poles leave the axis; continue them from the lossless roots by Newton on D | path vs reference (lossy, real k), δ-independence (complex k), and Im β > 0 at real k |
| S7 | `s7_transmitted.py` | τ through the stack: the transmitted G below z_bot against a real-axis reference (extends v1.1 GT2) | ≤ the S3 floor |
| S8 | `s8_residue.py` | **Mode normalisation and the residue–overlap identity.** Res_β R against the closed form built from the normalised profile u_m(z) (the 1/ε weight in TM), and the overlap form of c_m^± | pinned formula, agreement at round-off on all S2 stacks |
| S9 | `s9_dipole.py` | **A4**: line-dipole emission into mode m. The residue amplitude against `∝ u_m(z₀)/P_m`, both polarisations; the far-x field of the path G against the residue term at large X | closed form at round-off; pins the residue, the normalisation and the phase reference |
| S10 | `s10_born.py` | **A5 floor**: small-circle S-parameters against the Born closed form, wired as a prototype on the package solver | the rad² rate and the gap dependence of the dressing corrections, measured |

The power balance (A6) and reciprocity (A7) need the assembled solver, so they
are gates of the v1.2 code-spec, not study items.

## Open after S1–S3

- **D as the mode solver's function.** D is real on the window and entire in
  the interior α_j. But it is the unscaled transfer matrix, so it overflows for
  a thick evanescent cladding layer *inside* the window, such as a buried oxide
  of several µm between two cores. The spec's mode solver should use the scaled
  D, which has the same roots. This is cheap to verify in S4.
- **A fixed hump suffices for real λ.** Whether it still does for slab QNMs is
  the low-priority later item (holomorphy §5), and it is not studied here.
