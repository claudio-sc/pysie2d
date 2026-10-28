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
- A3: a generic shape — adjoint vs central differences of S; tolerance = the
  measured FD floor.

**S3 — design.** Maximise S at fixed D and baseline under the 100 nm critical
dimension, with |T|/S bounded, i.e. a thermal tolerance. The job the circle
cannot do is lower the drift gain T at fixed S.
*Stop rule:* if no shape improves |T|/S by more than a factor of 2 over the
best circle at equal S, say so. That is a result, not a reason to add
parameters.

**S4 — drift rejection with the bus.** Bus-coupled S-parameters at several
gaps (`BIESolver.scatter_mode`; `QNMSolver` does not take a `Stack`). The
nuisances are the drifts of the real index, of Q_c and of surface absorption,
not Q_rad. Channels: the linewidth and the resonance shift of one mode; a
mode pair with contrasting Γ and T; and a bulk-weighted vs surface-weighted
pair (Ω vs Ω_δ, the interior minus a surface margin δ). Maximise the Fisher
information on δε″ with the drifts as nuisance parameters. Scoped in detail
after S3.

## Promotion

Nothing is expected to enter the package. If S3 needs a mixed eigenvalue
derivative ∂²λ/∂p∂ε″ often enough to justify an API, it would need A2 as a
library test.
