# Inverse design for interior field enhancement — study plan

Spike, branch `spike-interior-enhancement`. Study code only; nothing enters the
package until a stage below earns it. Decisions agreed with the owner
2026-10-28.

## Goal

Structures whose signals carry high information about **material loss** Im ε,
with the intrinsic radiation loss Q_rad fitted jointly as a nuisance parameter
(fabrication moves it off the model). Field enhancement is the stage-1 proxy:
at weak absorption ∂σ_abs/∂(Im ε) ∝ ∫_Ω |E|², so the volume-averaged interior
intensity *is* the absorption sensitivity. Separating Im ε from Q_rad needs two
channels with different weightings, since 1/Q = 1/Q_abs + 1/Q_rad + 1/Q_c adds;
that is stage 2.

## Fixed setup

| Item | Value |
|---|---|
| View | Top view of a SiN chip; bare indices, no EIM pretence |
| Indices | core n = 2.0, cladding SiO₂ n_clad = 1.45 |
| Polarisation | `pol = 1` (H out of plane, E in-plane — slab TE-like in top view) |
| Wavelength | λ_vac = 1550 nm (λ_bg ≈ 1069 nm) |
| Size | linear dimension 2–10 µm |
| Q | < 10⁶ |
| Loss range to resolve | 0.1–10 dB/cm (Q_abs ~ 10⁴–10⁶) |
| Profile | r(θ) = r₀ + Σ_{m=1}^{N} a_m cos mθ, N ≤ 30 — cosine-only = mirror symmetry about the waveguide normal |
| Fabrication | critical dimension 100 nm (min radius of curvature and min bus gap) |
| Bus waveguide | single-mode strip at 1550 nm as an in-plane v1.2 `Stack`; several gaps as measurement channels |
| Parked | surface roughness; the anapole study (`../anapole/`) |

## Objective

J(p) = (1/A) ∫_Ω |E|² dA, normalised by the incident intensity, evaluated at
complex λ̃ = λ₀ + iΔλ. For a field holomorphic in the upper half-plane the
Lorentzian average of width Δλ over real λ equals the value at λ̃ — one solve,
not a sweep (conventions §8: Im λ > 0 is the decaying side, so no pole is
crossed). **Δλ is swept**, not fixed: the optimum vs Δλ maps the Q/V trade-off
and stops the optimiser from running to an arbitrarily high-Q mode.

Extension (stage 1b): Ω_δ = {r ∈ Ω : dist(r, ∂Ω) > δ}, the interior minus a
surface margin, kept for when roughness returns — bulk-weighted vs
surface-weighted is the natural disentangling pair.

## Gradient: informal adjoint

Forward: M(λ, p)·u = f(p), u = (φ, χ); interior field E = Σ(λ, p)·u at
interior nodes. J is real, E complex, so J depends on u and ū.

One adjoint solve shared by all N parameters: Mᵀ v = ∂J/∂u, then

    dJ/dp_m = 2 Re[ vᵀ (∂f/∂p_m − ∂M/∂p_m · u) ] + ∂J/∂p_m|explicit

- **∂M/∂p_m by central differences of the discrete assembly** — one pair of
  assemblies per harmonic, **no extra solves**. Assembly is the cost
  (special-function bound), so this is ~2N assemblies + 2 solves per step;
  an analytic dM/dp is deferred until it is measured to matter. Step size
  from [shape-derivative-smoothness.md](../shape-derivative-smoothness.md).
- **The moving domain.** Ω depends on p, so ∂J/∂p_m|explicit carries the
  boundary term ∮ |E|² (δr_m · n̂) ds plus the dependence of the interior
  quadrature nodes. Dropping it is the classic error; anchor A1 catches it.

## Stages, checks, stop rules

**S0 — feasibility of the Q budget (do first).** At 0.1 dB/cm, Q_abs ≈ 10⁶,
at the Q ceiling. Measure Q_rad of the circle's WGM ladder (existing
`QNMSolver`) for D = 2–10 µm at n = 2.0/1.45, pol = 1.
*Check:* circle QNMs match the analytic Mie poles (existing anchor).
*Stop rule:* if no mode with Q_rad ≥ 3·Q_abs exists at the low-loss end for
D ≤ 10 µm, report the resolvable loss floor and narrow the range with the
owner before building anything else.

**S1 — forward objective on the circle.** J for plane-wave incidence vs the
closed form from interior Mie coefficients.
*Check:* agreement at round-off by nn ≈ 30–40 (Kress, spectral); the
tolerance cites the measured floor.

**S2 — gradient anchor, minimum effort.**
- A1: circle, m = 0 — dJ/dr₀ vs the analytic radius derivative of the Mie
  closed form. Independent (rule 3); exercises the moving-domain term.
- A2: generic shape, m ≥ 1 — adjoint vs central differences of J. Proves
  derivative consistency only; physics is already anchored by S1 and A1.
  Tolerance = the measured FD truncation floor at the chosen h.

**S3 — stage-1 design.** Plane wave for the gradient checks, the bus-waveguide
guided mode for the design itself. Maximise J under the 100 nm critical
dimension, sweep Δλ. *Output:* optimum J vs Δλ, and shapes.
*Stop rule:* if the optimum is a perturbed circle WGM at every Δλ (no gain
over the best circle beyond a factor 2), say so — that is a result, not a
reason to add parameters.

**S4 — stage-2 OED.** Unknowns (Im ε, Q_rad), channels: QNM linewidths of a
mode pair with contrasting confinement Γ, and S-parameters at several gaps
(Q_c varies with gap, Q_abs and Q_rad do not). Maximise det of the Fisher
information over 0.1–10 dB/cm. Scoped in detail after S3.

## Promotion

A driven shape-gradient API enters the package only if S3 shows the
adjoint is needed outside this study; it would then need A1 as a library test.
