# Inverse design for absorption tracking — study plan

Spike, branch `spike-interior-enhancement`. Study code only; nothing enters the
package until a stage below earns it. Decisions agreed with the owner
2026-10-28; reframed the same day from absolute Im ε to tracking (below).

## Goal

**Track changes of material absorption on one device over long periods**
(months), resolving δα = 0.1 dB/m. An absolute Im ε is *not* required. That
makes the radiation loss Q_rad, and anything else static, common-mode. What
must be rejected instead is everything that drifts:

- the real index, through the thermo-optic effect and through the
  Kramers–Kronig partner of the absorption change itself, with Q_rad and Q_c
  acting as gains;
- the bus coupling Q_c, through the gap: index drift, humidity, film growth;
- surface absorption (adsorbed water, oxide growth), which is a real absorption
  change but not a material one.

The readout is the loaded linewidth of a single high-Q resonance. With
1/Q_L = 1/Q_i + 1/Q_c and 1/Q_i = 1/Q_rad + 1/Q_abs, a change δε″ of the core
Im ε moves the loaded linewidth by a fraction

    δγ/γ = (Q_L/Q_i) · S · δε″,   S ≡ Q_i · ∂(1/Q_i)/∂ε″,

so at critical coupling (Q_L = Q_i/2) the resolution is
**δε″_min = 2 (σ_γ/γ) / S**, with σ_γ/γ the long-term linewidth repeatability.
The drift gain is T ≡ Q_i · ∂(1/Q_i)/∂n_core, and a temperature excursion ΔT
reads as δε″ = |T| (dn/dT) ΔT / S. Both are QNM quantities, available from
`QNMSolver` and `QNMResult.sensitivity` with no driven solve.

Units: ε″ = Im ε (absolute, `Material.epsi`) = n·α·λ₀/(2π), so
0.1 dB/m ↔ δε″ = 1.14·10⁻⁸ at n = 2.0.

## Fixed setup

| Item | Value |
|---|---|
| View | Top view of a SiN chip; bare indices, no EIM pretence |
| Indices | core n = 2.0, cladding SiO₂ n_clad = 1.45 |
| Polarisation | `pol = 1` (H out of plane, E in-plane — slab TE-like in top view) |
| Wavelength | λ_vac = 1550 nm (λ_bg ≈ 1069 nm) |
| Size | D = 12–18 µm (S0: ≈ 16 µm is the working point) |
| Q | loaded Q_L < 10⁶ |
| Baseline loss | 0.1–10 dB/cm; resolution target δα = 0.1 dB/m on top of it |
| Profile | r(θ) = r₀ + Σ_{m=1}^{N} a_m cos mθ, N ≤ 30 — cosine-only = mirror symmetry about the waveguide normal |
| Fabrication | critical dimension 100 nm (min radius of curvature and min bus gap) |
| Bus waveguide | single-mode strip at 1550 nm as an in-plane v1.2 `Stack`; several gaps as measurement channels |
| Parked | surface roughness; the anapole study (`../anapole/`) |

## Why not the driven interior intensity

The first version of this plan used J = (1/A)∫_Ω |E|² under plane-wave
incidence at λ₀ + iΔλ as a Lorentzian-averaged objective. Measured against
the closed-form interior Mie field of the n = 34 WGM (Q = 2.3·10³,
[s1_lorentz_counterexample.py](s1_lorentz_counterexample.py)), it fails three
ways:

1. **Wrong half-plane.** QNM poles sit at Im λ > 0 (conventions §8); the
   driven response is holomorphic for Im λ < 0. At λ₀ + iΔλ with Δλ = γ the
   evaluation lands on the pole (J = 2·10²⁴).
2. **|E|² is not holomorphic**, so the Lorentzian average of |E|² is not
   |E|² at λ₀ − iΔλ. The ratio is 1.2–1.6 and varies with Δλ, so an
   optimiser would be maximising the wrong thing. The identity holds for
   quantities that are linear in a Green function (an LDOS), not for
   intensities.
3. **Wrong drive.** A plane wave barely couples to a high-order WGM (peak
   ⟨J⟩ ≈ 3.5), while the measurement goes through a bus.

For tracking, the linewidth readout makes S and T, both QNM quantities, the
direct objective.

## Stages, checks, stop rules

**S0 — feasibility of the Q budget.** Q_rad of the circle's WGM ladder from
the analytic Mie poles; per-mode Q_abs from the lossy pole; the tracking
detection limit vs D.
*Check:* `QNMSolver` circle poles against the analytic ones.
*Stop rule:* the D range must contain a mode resolving 0.1 dB/m at
σ_γ/γ ≥ 10⁻³ and a thermal tolerance ≥ 1 K, within Q_L < 10⁶.

*Result (2026-10-28, [s0_qrad_budget.py](s0_qrad_budget.py) →
[.txt](s0_qrad_budget.txt)): feasible at D ≈ 16 µm, not at D ≤ 10 µm.*
p = 1 WGM ladder, 0.1 dB/cm baseline, critical coupling:

| D | n | Q_rad | Q_L | σ_γ/γ needed for 0.1 dB/m | ΔT tolerated |
|---|---|---|---|---|---|
| 9.7 µm | 34 | 2.3·10³ | 1.2·10³ | 2.9·10⁻⁶ | 13 mK |
| 12.8 µm | 46 | 4.8·10⁴ | 2.4·10⁴ | 6.3·10⁻⁵ | 0.2 K |
| 14.9 µm | 54 | 3.9·10⁵ | 1.8·10⁵ | 4.7·10⁻⁴ | 1.4 K |
| 15.9 µm | 58 | 1.1·10⁶ | 4.4·10⁵ | 1.2·10⁻³ | 3.6 K |
| 17.0 µm | 62 | 3.3·10⁶ | 8.8·10⁵ | 2.4·10⁻³ | 9.8 K |

- ΔT is the thermo-optic leak through Q_rad only, with dn/dT = 2.5·10⁻⁵ K⁻¹
  on the core (assumed literature value); Q_c's leak is S4's.
- Once Q_rad ≫ Q_abs the requirement saturates at σ_γ/γ ≈ δα/(2α₀): 5·10⁻³ at
  a 0.1 dB/cm baseline, 5·10⁻⁵ at 10 dB/cm. The baseline loss caps resolution
  as much as the size does.
- Q_abs is measured per mode (bulk κ in the core, pole re-polished):
  Q_abs ≈ Q_mat/Γ with Γ ≈ 0.9–0.95, so 3.7·10⁶ at 0.1 dB/cm. First order in
  κ to ≤ 2·10⁻⁴.
- Bare indices are optimistic: an EIM core index below 2.0 lowers Q_rad at
  every D, so a real device needs a larger D than this table.
- Check: `QNMSolver` finds both members of each ±n pair, matching the analytic
  pole to 1.6·10⁻¹² nm (n = 10, n_pts = 80) and 4.6·10⁻¹³ nm (n = 36,
  n_pts = 240), the contour floor. n = 36 first reaches it at n_pts = 200.

**S1 — the forward objective on the circle.** S and T at the 0.1 dB/cm
baseline, from `QNMSolver` + `QNMResult.sensitivity` (∂/∂`epsi`,
∂/∂`n_core`), against the analytic implicit derivative of the Mie pole,
dx/dm = −(∂D_n/∂m)/(∂D_n/∂x). That derivative is independent of the BIE
(rule 3). Modes n = 46, 58, 66 (D ≈ 12.8, 15.9, 18.0 µm).
*Check:* λ, ∂λ/∂ε″ and ∂λ/∂n_core at the floor set by the contour and the
central-difference step, each tolerance citing the measured floor.

*Result (2026-10-28, [s1_circle_sensitivity.py](s1_circle_sensitivity.py) →
[.txt](s1_circle_sensitivity.txt)): passes.* Worst member of the ±n pair, at
`step = 1e-5`:

| n | D | Q_i | S | T | λ err | S rel. err | T rel. err |
|---|---|---|---|---|---|---|---|
| 46 | 12.8 µm | 4.7·10⁴ | 1.10·10⁴ | −25.4 | 7·10⁻¹³ nm | 6·10⁻¹⁰ | 1.7·10⁻⁶ |
| 58 | 15.9 µm | 8.7·10⁵ | 2.06·10⁵ | −26.1 | 1·10⁻¹² nm | 1·10⁻⁹ | 3.6·10⁻⁵ |
| 66 | 18.0 µm | 2.7·10⁶ | 6.40·10⁵ | −11.3 | 5·10⁻¹³ nm | 2·10⁻⁹ | 2.8·10⁻⁴ |

- λ sits at the contour floor and is already converged on the lower rung of
  each n_pts pair (n_pts ≈ 5.8n suffices).
- The derivative error is the O(h²) truncation of the central difference
  inside `sensitivity`: a tenfold larger step makes it 100× larger in every
  case. The analytic reference agrees with a finite difference of its own
  polished root to ≤ 8·10⁻¹⁰.
- T is less accurate than S because it lives in Im dλ/dn_core, which is only
  6·10⁻⁴ to 4·10⁻⁶ of |dλ/dn_core|: the resonance shift dominates, and its
  error leaks in. 3·10⁻⁴ is ample for a drift gain.
- Physics: S/Q_i = Γ/Re ε, with Γ = 0.93–0.96. T falls from −26 to −11 at
  n = 66 because absorption takes over Q_i there (Q_rad = 9.8·10⁶ >
  Q_abs = 3.7·10⁶), and T only enters through Q_rad. Hence |T|/S ∝
  (∂ln Q_rad/∂n)/(Γ·Q_rad), so on the circle the drift ratio improves
  exponentially with D. S3 has to beat that at fixed D.
- The pair reproduces S0's tracking table: n = 58 resolves δε″ = 2·10⁻³/S =
  9.7·10⁻⁹ < 1.14·10⁻⁸ (0.1 dB/m) at σ_γ/γ = 10⁻³.

**S2 — gradient anchor.** dS/dp_m and dT/dp_m need ∂λ/∂p_m (sensitivity,
anchored in conventions §11) and the mixed ∂²λ/∂p_m∂ε″.

- A1: circle, m = 0 — S and T are radius-independent at the mode by scale
  covariance (§9), so dS/dr₀ = dT/dr₀ = 0 exactly.
- A2: circle, 1 ≤ m ≤ N — first-order degenerate perturbation theory couples
  the ±n doublet only through the m = 2n harmonic. With n ≈ 46–66 > N/2, every
  allowed harmonic should give dλ/da_m = 0 at the circle. **If confirmed, the
  circle is a stationary point of S and T**, and S3 cannot start from it with
  a gradient method.
- A2b: if A2 holds, the same rule makes the Hessian at the circle diagonal in
  the a_m (a cross term needs m + m′ = 2n), so 30 second differences of S and
  of R = |T|/S decide whether the circle is a local optimum.
- A3: a generic shape — adjoint vs central differences of S; tolerance = the
  measured FD floor.

*Result (2026-10-28, [s2_gradient_anchor.py](s2_gradient_anchor.py) →
[.txt](s2_gradient_anchor.txt), n = 46, n_pts = 260): all anchors pass, and
the circle is a local maximum of S.*

- A1: S and T over r₀(1 ± 10⁻³) agree to 6·10⁻¹¹ and 1.7·10⁻⁹, S1's floors.
- A2: max over m ≤ 30 of |dλ/da_m| is 1.3·10⁻¹¹ of |λ|/r₀ — the selection
  rule holds. Positive control m = 2n = 92: the adjoint splits the pair by
  ±0.033 nm/nm (0.136·|λ|/r₀), matching contour finite differences to
  4·10⁻⁸ at h = 10⁻³ and 4·10⁻³ nm, the contour floor over 2h.
- A2b: (1/S)·∂²S/∂a_m² < 0 for every m = 2…30, from −2·10⁻⁵ (m = 2) to
  −6·10⁻² nm⁻² (m = 5). m = 1 is flat (−3·10⁻¹¹), being a translation to
  first order. h = 0.5 and 1 nm agree to ≤ 2 %. **The circle is a strict local
  maximum of S** in the cosine family. R is a saddle: 19 of 30 directions lower
  it, but each costs S faster in log, with (ln R)″/(ln S)″ ≤ 0.85. The
  quiet harmonics m = 11, 18, 21, 24 come close to 1; most others sit at
  0.03–0.2. **No local direction improves both S and R.**
- A3, at a = {2: 30, 3: 20, 5: 12, 7: 8} nm: the adjoint dλ/da_m matches
  contour finite differences, with the gap falling as h² (m = 5:
  3.9·10⁻⁶ → 6.2·10⁻⁵ → 1.0·10⁻³ at h = 0.125, 0.5, 2 nm). The cheap route
  for dS/da and dT/da — predicted poles λ ± h·dλ/da, no contour — matches
  brute force by the same h² law, to ≤ 2·10⁻⁴ at h = 0.125 nm. One gradient
  costs five sensitivity calls per parameter instead of two contours
  (≈ 100 assemblies).
- This 0.5 % deformation alone cuts Q_i fivefold (4.7·10⁴ → 9.2·10³) and S
  likewise (1.1·10⁴ → 2.1·10³): S is steep away from the circle.

What it means for S3. Locally, maximising S is already solved by the circle,
and lowering R always costs S. Whether a deformation pays depends on which term
of the error budget dominates, δε″ ≈ √[(2σ_γ/γ)² + (|T|·(dn/dT)·ΔT)²]/S. At the
working point (n = 58, D ≈ 16 µm, 3.6 K tolerance) the budget is noise-limited
and the circle wins. At a smaller D, or with poor thermal control, it is
drift-limited. There, a quadratic extrapolation along m = 18 halves R for about
2.3× in S (a ≈ 120 nm). That is outside the perturbative range and has to be
checked, not trusted.

**S3 — design.** Maximise S at fixed D and baseline under the 100 nm critical
dimension, with |T|/S bounded, i.e. a thermal tolerance. The job the circle
cannot do is lower the drift gain T at fixed S.
*Stop rule:* if no shape improves |T|/S by more than a factor of 2 over the
best circle at equal S, say so. That is a result, not a reason to add
parameters.

*Decision (owner, 2026-10-29): S3 skipped.* D ≈ 16 µm is acceptable and
the stage is stabilised to ≤ 0.1 K. At n = 58 the thermo-optic leak through
Q_rad is then 0.1/3.6 of the 0.1 dB/m step, about 0.003 dB/m. The budget is
noise-limited, and S2 shows the circle is the local optimum there. The working
point is the circle, n = 58.

**S4 — drift rejection.** *Scoped with the owner 2026-10-29:* bus width
500 nm (single-mode below the 563 nm cutoff of this 2D model); readout =
through-port transmission dip; nuisance = **cladding index aging** δn_cl;
channels = **sister reference rings** on the same chip.

The dip gives Q_L and its depth, hence Q_i and Q_c separately, so bus-coupling
drift is measured rather than estimated. What leaks into Q_i is δn_cl through
Q_rad, with gain T_cl ≡ Q_i·∂(1/Q_i)/∂n_clad. A sister ring shares the film
and the cladding, so it sees the **same** δε″ and δn_cl. It separates them
only through a different ratio T_cl/S. Linewidth channels y_k = 1/Q_i,k with
σ(y_k) = 2(σ_γ/γ)/Q_i,k (the single-ring rule δε″_min = 2σ/S) give a 2×2
Fisher problem, and σ(δε″) with δn_cl free is [F⁻¹]₁₁.

- **S4a — main-ring tolerance.** T_cl at n = 58, by `QNMResult.sensitivity`
  in `n_clad` against the analytic dλ/dn_cl = λ/n_cl + (λ/x)(dx/dm)(m/n_cl).
  *Output:* the δn_cl that mimics 0.1 dB/m. If that exceeds any plausible
  aging, no reference ring is needed and S4 stops.
- **S4b — sister choice.** σ(δε″) vs the sister's order n_ref (p = 1 ladder),
  against the same pair with δn_cl known. *Output:* the best D_ref and the
  price of not knowing δn_cl.

*Result S4a–b (2026-10-29, [s4_cladding_drift.py](s4_cladding_drift.py) →
[.txt](s4_cladding_drift.txt); σ is 1σ, at σ_γ/γ = 10⁻³):*

- S4a: T_cl = +35.6 at n = 58 — raising n_cl lowers the contrast and Q_rad.
  **δn_cl = 6.6·10⁻⁵ mimics a 0.1 dB/m step**, so a single ring is only safe
  if the cladding index holds to better than that for months. The BIE
  dλ/dn_clad matches the analytic derivative to 8·10⁻¹⁰, and T_cl to ≤ 10⁻⁷.
- S4b, main n = 58 plus one sister, under Q_i ≤ 2·10⁶:

  | Sister | D_ref | σ, δn_cl free | σ, δn_cl known | Price |
  |---|---|---|---|---|
  | n = 46 | 12.8 µm | 0.129 dB/m | 0.085 dB/m | 1.5 |
  | n = 56 | 15.4 µm | 0.299 dB/m | 0.072 dB/m | 4.2 |
  | n = 62 | 16.9 µm | **0.083 dB/m** | 0.038 dB/m | 2.2 |

  One ring with δn_cl known gives 0.085 dB/m. **A second high-Q ring (n = 62)
  makes the measurement immune to cladding aging at no loss** relative to that
  single ring: both rings carry signal, and their T_cl/S differ by 2.7×
  (1.7·10⁻⁴ vs 6.3·10⁻⁵). A low-Q reference costs 1.5×. A near-twin
  (n = 56, 60) is ill-conditioned and costs 4×.
- Assumed: δε″ and δn_cl are uniform across the chip, and both rings are
  critically coupled with the same σ_γ/γ. A sister above n = 62 breaks the
  Q_L ceiling unless it is undercoupled, which changes its noise.

- **S4c — bus.** Q_c(g) of the main ring by `scatter_mode` on the 500 nm
  `Stack`, the critical gap, and ∂Q_c/∂n_cl as the error the Q_i/Q_c split
  must absorb. Scoped after S4a–b.

## Promotion

Nothing is expected to enter the package. If S3 needs a mixed eigenvalue
derivative ∂²λ/∂p∂ε″ often enough to justify an API, it would need A2 as a
library test.
