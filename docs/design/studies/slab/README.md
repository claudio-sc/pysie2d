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
| `s5_gap.py` | Risk §7.2: gaps down to 2 nm with poles present; M against v0.9; the cost of R(q) against N_layers | done |
| `s8_residue.py` | Mode normalisation: Res_β R = γ₁u_m(z_top)²/(β_m N_m), N_m = ∫p u_m² | done |
| `s9_dipole.py` | A4: the far-x guided sum, and the dipole power balance as an equality | done |
| `s6_lossy.py` | Lossy layers and complex k: pole continuation, path vs reference (lossy), path independence (complex k) | done |
| `s7_transmitted.py` | The transmitted G below the stack against a bottom-up shooting reference | done |
| `s10_born.py` | A5: small-circle S-matrix against first Born with the full stack G, TE | done |
| `s12_particle.py` | A particle over the stack, wired as v1.2: the overlap formula for c_m^±, dipole power balance, S energy, reciprocity | done |
| `s12b_inplane_particle.py` | In-plane dipole + particle: power balance with the source gradient by finite differences | done |
| `s11_inplane.py` | R18's anchor: in-plane dipoles as derivative sources, per-direction mode amplitudes, spin-locked one-way coupling, the power balance | done |
| `nearfield.py` | The field inside the layers and below; its checks (interface continuity, cover match at z_top, far-x guided sum in the core) run as `__main__` | done |
| `fig_data.py` | Data for the near-field showcase page (`python fig_data.py OUT.json`) | — |

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

### S5: small gaps (`s5_gap.txt`)

Gaps of 10, 5 and 2 nm (z_min = 2·gap), D = 300 nm, on SOI, SiN (2 modes) and
the double core, TE and TM. Points X ∈ {0, 50, 300} nm, Z ∈ {z_min, 4·z_min},
G and ∂z′G.

- **Path vs reference ≤ 5e-11 at every gap**, flat from s = 1 to 2 (the
  reference floor). The path is converged and independent of the hump depth to
  2e-14.
- **M is v0.9's plus 1–3 %**: 721/1249/2881 against 675/1227/2859 for a glass
  half-space at the same D and z_min. The poles add only the T = 1.5·β_max
  stretch. M ∝ 1/z_min, as in v0.9 (t14).
- The naive transfer matrix is **already NaN on the path at a 10 nm gap**
  for SiN and the double core, and at 5 nm for SOI. So S1's recursion is
  required, not optional.
- **R(q) on the whole path costs 0.05 ms (N = 0) to 0.8 ms (N = 21)**, linear
  in N. It is evaluated once per path and wavelength, against ~3.5 ms for one
  unbanded nn = 256 block-pair product. The stack adds nothing to the GEMM
  itself.

### S8: mode normalisation and the residue identity (`s8_residue.txt`)

Write the field equation as ∇·(p∇u) + k²w u = 0, with p = 1, w = ε (TE, E_y)
or p = 1/ε, w = 1 (TM, H_y). The guided modes are orthogonal under the
**bilinear** form ∫p u_m u_n dz = N_m δ_mn (no conjugate, so it continues to
complex ε). The 1D Green function of ∇·(p∇g) + (k²w − p q²)g = −δ then has the
pole

    g(z, z'; q) ⊃ u_m(z) u_m(z') / ((q² − β_m²) N_m).

In the cover, g_refl = (i/2)(R/α₁)e^{iα₁(h+h′)}, with α₁(β) = iγ₁ and
u_m = u_m(z_top)e^{−γ₁h}. Equating residues gives

    Res_{q=β_m} R = γ₁ u_m(z_top)² / (β_m N_m).

- **Verified to ≤ 4e-14** on six stacks, TE and TM, 16 modes per polarisation, including
  near cutoff. The left side is a Cauchy circle on the recursion. The right side
  is the (u, p u′) transfer profile with a Gauss–Legendre norm, and uses no R.
- **The TM 1/ε weight is load-bearing**: with the unweighted ∫u², the identity
  fails by 74 %.
- Power: the x-flux of a mode is ∝ β_m·∫p|u_m|² = β_m N_m for real profiles,
  so the residue is γ₁u_m(z_top)²/P_m up to the physical constants. This is
  the roadmap's `∝ u_m(z₀)/P_m`.

### S9: line-dipole emission into the modes, anchor A4 (`s9_dipole.txt`)

**(b) Dipole power balance, an equality.** The deformed path gives
Im G(r₀,r₀) = 1/4 + Im G_refl. Real-axis integrals of R and τ give

    P_up   = (1/π)∫₀^{π/2} |e^{−iα₁h₀} + R e^{iα₁h₀}|²/4 dθ
    P_down = (1/π)∫₀^{k_b} Re(Y_b)|τ|²|e^{iα₁h₀}|²/4 dq

The profiles of S8 give P_guided = Σ_m u_m(z₀)²/(2β_m N_m), which is the
iπδ(q − β) part of Im g, counted at ±β.

- **Im G = P_up + P_down + P_guided to ≤ 3.5e-15** on four stacks, TE and TM,
  at h₀ = 20, 100 and 400 nm. The guided share is up to 1/3 of the total
  (SOI TM at 20 nm: 0.117 of 0.354).
- No two terms share an integral. This is v1.1's G14 equality with the missing
  `P_guided` term in place, and A6 at the Green-function level. With a
  particle, the same balance needs only P_abs.

**(a) The far-x field.** The guided sum Σ_m (i/(2β_m N_m))u_m(z)u_m(z₀)e^{iβ_m|X|}
is the residue at +β, with the contour closed upward for X > 0.

- G_total − guided **decays on every stack**, to 1e-3–5e-2 of the guided
  terms at 40 µm. A wrong sign, a wrong factor i or a wrong phase reference
  would leave a remainder of order 2. The remainder is physics, not quadrature:
  the path matches the reference there (≤ 3e-11 at X = 10 and 20 µm).
- **The rate is X^{−3/2} on 6 of 8 cases** (slope −1.47 to −1.50 by 40 µm).
  The exceptions are **TM on SiO₂-clad stacks** (SOI, double core). Those decay
  more slowly: slope −0.6 at 5 µm, still −1.2 at 320 µm. The cause is not
  established. A search found no improper pole of D within |γ| < k of the
  branch point, so a slow leaky wave is not the explanation. Open.
- Normalise by Σ|terms|, not |Σ|: two supermodes beat, so |Σ| has near-zeros.

### S11: in-plane dipoles, R18's anchor (`s11_inplane.txt`)

A current J in the cover gives ∇·(p∇u) + k²u = −[∇×J]_y for TM (u = H_y).
An in-plane electric dipole is therefore a **derivative source**,
u = −(b·∇′)G(r, r₀) with **b ∝ (p_z, −p_x)**. The constant and the sign of
that map (iω, the cover ε) are pinned in the code-spec. The same form in TE
(u = E_y) is an in-plane *magnetic* dipole, and the algebra does not depend on
the polarisation. Applying −(b·∇′) to S8's guided pole gives

    c_m^± = (i/(2β_m N_m)) u_m(z₀) (±iβ_m b_x + γ₁ b_z),
    P_m^± = u_m(z₀)² |±iβ_m b_x + γ₁ b_z|² / (4β_m N_m),

so **c_m^− = 0 exactly when b_z = iβ_m b_x/γ₁**. That is the spin-locked
dipole: elliptical, with |b_z/b_x| = β/γ > 1. The power is
Im[(b*·∇)(b·∇′)G] at coincidence: k²|b|²/8 from the free part, plus the
reflected part with weight q²|b_x|² − α₁²|b_z|² (the odd cross terms vanish on
the symmetric path). The up/down integrals take source amplitudes
A↑ = i(qb_x + α₁b_z) and A↓ = i(qb_x − α₁b_z), and they **must run over the full
−k…k range**: they are not even in q once b_x b_z ≠ 0.

- **The power balance is an equality to ≤ 5e-15** for b = (1,0), (0,1),
  (1, 0.7−0.4i) and spin-locked, on SOI, air|Si|SiO₂ and SiN (2 modes), TE and
  TM, at h₀ = 20 and 150 nm.
- **The spin-locked dipole sends P⁺/(P⁺+P⁻) = 1.0000 one way** on single-mode
  slabs, at both heights. On the 2-mode SiN slab it is 0.88–0.95: the lock
  matches mode 0's γ₁, so mode 1 still couples both ways. Guided shares reach
  57–78 % at 20 nm, against 28–37 % for the scalar source (S9).
- **The far-x field matches the closed-form c_m^± in both directions**, with
  the remainder falling as in S9 (1e-1–1e-3 at 10 µm, 3e-2–9e-5 at 40 µm).
  For the spin-locked dipole the −x side carries no guided wave. A wrong ± in
  the closed form would leave an O(1) remainder there.

### S6: lossy layers and complex k (`s6_lossy.txt`)

- **Pole continuation** (`sproto.continue_poles`): a homotopy in 16 steps from the
  lossless stack at Re k to the target ε and k, with Newton on D and D′ from a
  Cauchy circle of radius 1e-4·|q|. It converges to |D/(D′β)| ≤ 3e-16 for
  Im n = 1e-3 and 5e-2, and for k(1 − 0.05i) and k(1 − i/6), TE and TM. Im β > 0
  with loss and Im β < 0 at Im k < 0, as holomorphy §5 says.
- **Lossy core at real k, path vs reference ≤ 5e-14.** With the poles off the
  axis the reference is plain QUADPACK with breakpoints at Re β.
- **Complex k:** at Q = 10, s = 2 and δ × 1.5 agree with the default to ≤ 4e-13.
  δ × 0.7 is worse (4e-10–8e-10), because the pole then sits closer to the hump.
  At Q = 3 the hump floor 3·|Im β| overrides the 6/D cap (δ·D ≈ 12): the path
  loses digits (3e-10 at δ × 1.5). That is v0.9's documented open item, which
  `SommerfeldPrecisionWarning` reports.

### S7: the transmitted G below the stack (`s7_transmitted.txt`)

- **Path vs reference ≤ 4e-11** on four stacks, TE and TM, inside the
  reference's own fold spread (≤ 3e-11). The path converges to 2e-13.
- **Trap:** the forward transfer-matrix T = aR + b cancels catastrophically on the
  tail. True T ~ e^{−qd} comes out of O(1) scaled terms, which gave 1e37. The
  reference shoots upward from the decaying bottom solution instead
  (T = 1/A_down at z_top), so the growth is the dominant direction.

### S12: a particle over the stack (`s12_particle.txt`, `s12b_inplane_particle.txt`)

The package's free-space `BIESolver.assemble`, plus the stack's reflected blocks
on the exterior rows (conventions §15's M1/M2). Applying S9's guided pole to
u_s = −(M1φ + M2χ) gives the **overlap formula**, which this pins:

    c_m^± = −(i/(2β_m N_m)) Σ_j h u_m(z_j) e^{∓iβ_m x_j} [(∓iβ_m dg_j + γ₁ df_j) φ_j + χ_j]

as the amplitude of u_m(z)e^{±iβ_m x}. A dipole adds its own (i/(2βN))u_m(z₀)e^{∓iβx₀}
(S9), or for an in-plane dipole (i/(2βN))u_m(z₀)(±iβb_x + γ₁b_z)e^{∓iβx₀} (S11). The
radiated powers come from the spectra of the total field, U = (i/2α₁)(u↑ + R d) above
and (i/2)τ d below, with u↑ and d the boundary sums of the representation:
P_up = (1/2π)∫|u↑ + R d|²/4 dθ and P_down = (1/2π)∫ Re(Y_b)|τ d|²/4 dq.

- **Dipole + particle power balance ≤ 1e-15**: a circle, a lossy circle, and a
  rotated (x-asymmetric) lossy triangle over SOI and over the 2-mode SiN slab,
  TE and TM.
- **Mode incidence:** Σ|S|² + P_up + P_down + P_abs = 1 to ≤ 4.4e-16, with
  power-normalised modes ũ = u/√(βN) and x_ref = 37 nm.
- **Reciprocity** t_{n←m}(L→R) = t_{m←n}(R→L) to ≤ 1.1e-16 on the asymmetric
  lossy triangle, and r_{nm} = r_{mn} on the 2-mode slab to ≤ 2e-17.
- **The far-x scattered field** (representation on the path) matches Σc u e^{±iβx}
  in both directions, with the remainder falling as in S9 (3e-2–5e-2 at 40 µm).
- **In-plane dipole + particle** (s12b): the right-hand side is
  −(b·∇′)(G₀ + G_refl) on the nodes. The source power is
  k²|b|²/8 + Im[(b*·∇)(b·∇′)G_refl] + Im[−(b*·∇)u_s](r₀), with ∇u_s by a 5-point
  difference. **The balance closes to the finite-difference floor**: 1e-12–3e-10
  at h = 0.5 nm and 6e-14–2e-11 at 0.25 nm, an h⁴ drop. Guided shares reach
  52–66 %.

### S10: the small-circle limit, anchor A5 (`s10_born.txt`)

TE, circle at (23, 60) nm, n = 2 in SiO₂. First Born with the **full** stack G,
u_s ≈ k²(ε − 1)πa²·G(r, r_p)u_inc(r_p), whose guided part gives S − δ in closed form.

- **The BIE S-matrix converges to Born**: 5.0e-4 at a = 2 nm on SOI, and
  1.2e-3 on the 2-mode SiN slab, all four entries and both directions.
- **The rate is (ka)²(A + B ln ka)**, the 2D self-interaction. Ratios per
  halving of a are 3.61, 3.63, 3.66 on SOI and 3.36, 3.43, 3.49 on SiN, against
  4 for a pure a² law.

## Plan for the rest of the study

Items S4–S6 measure the §7 risks and finish roadmap §9 step 1's scope. S7–S9
are §9 step 2 (pin the §3 formulas), and S10 is step 3's floor for A5. Each
script states its check before it runs.

| # | Script | Question | Pass condition / output |
|---|---|---|---|
| S4 | `s4_cutoff.py` | Near cutoff and near degeneracy | **done**, see above |
| S5 | `s5_gap.py` | Small gaps with poles present | **done**, see above |
| S6 | `s6_lossy.py` | Lossy layers and complex k | **done**, see above |
| S7 | `s7_transmitted.py` | The transmitted G below the stack | **done**, see above |
| S8 | `s8_residue.py` | Mode normalisation and the residue identity | **done**, see above |
| S9 | `s9_dipole.py` | A4: the dipole into modes; power balance | **done**, see above |
| S10 | `s10_born.py` | A5: small-circle Born limit | **done**, see above |

The power balance (A6) and reciprocity (A7) need the assembled solver, so they
are gates of the v1.2 code-spec, not study items.

## Open after S1–S5, S8, S9

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
- **The overlap form of c_m^±.** S9a fixes the guided part of G, so the
  amplitude a scatterer launches is c_m^± = (i/(2β_m N_m))·∫_Γ (the boundary
  densities against u_m(z′)e^{∓iβ_m x′}). The sign and density wiring of that
  integral is BIE wiring (φ, χ conventions). It is pinned by A6/A7 in the code-spec
  gates, not in a study.
- **The slow TM far-x rate** on SiO₂-clad stacks (S9a).
- **A fixed hump suffices for real λ.** Whether it still does for slab QNMs is
  the low-priority later item (holomorphy §5), and it is not studied here.
