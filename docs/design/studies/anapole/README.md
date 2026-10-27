# Anapole / dark-state study

Theory: eigen-eps `manuscripts/anapole-design-rules/note.tex` (anapoles as real
interior transmission eigenvalues, ITE). Scripts that import its `anapole_core.py`
or `pysie2d_crosscheck.py` find them at `../../../../../eigen-eps/manuscripts/anapole-design-rules/scripts`
or at `$EIGEN_EPS_SCRIPTS`, and need `--with mpmath`:

```bash
uv run --frozen --with mpmath python docs/design/studies/anapole/<script>.py
```

**Polarisation naming is swapped against the note** (conventions §1): pysie2d
`pol = 2` (`E_y`) is the note's **TM**, `pol = 1` (`H_y`) its **TE**.

## 1. ITE vs driven anapoles on a non-separable shape (2026-10-26)

Superellipse |x|ᵖ + |z|ᵖ = 1, X = 0.5. Driven by the four-plane-wave standing wave
of the ITE's symmetry class (A1 for pol 2, x-odd/z-even for pol 1); the driven
anapole is the zero of the class's lowest multipole coefficient.

| p | pol 2 ITE | driven c₀ = 0 | gap | pol 1 ITE | driven c₁ = 0 | gap |
| --- | --- | --- | --- | --- | --- | --- |
| 2 (circle) | 59.7601 | 59.7601 | 0 | 60.8227 | 60.8227 | 0 |
| 2.5 | 56.1648 | 55.4655 | −1.2 % | 56.8897 | 56.8717 | −0.03 % |
| 3 | 54.7271 | 52.8780 | −3.4 % | 54.9097 | 54.8570 | −0.10 % |
| 4 | 53.7838 | 49.9948 | −7.0 % | 53.1315 | 53.0108 | −0.23 % |

- On the circle the two coincide (`circle_check.py`: |c₀| = 8e-16, |c₁| = 1e-15).
- Off the circle they separate continuously (`trace.py`). The driven zeros are
  near-zeros with a floor (|c₀| = 5.6e-7, |c₁| = 1.1e-7 at p = 4), converged to
  10 digits in nn = 128–512 (`nncheck.py`).
- At its ITE the p = 4 pol-2 particle sits near a monopole *maximum*, and an ITE
  census over ε = 40–60 finds no other ITE near the driven zero (`census.py`).
- Reading: the ITE suppresses scattering of *its own* background field v, not of
  the plane wave; the projection zero is a complex condition met with one real
  parameter. Pole-hugging roots (pol 1) barely move, mid-branch roots (pol 2)
  move ~30× more. **For plane-wave darkness the objective must be driven
  σ_sca; ITEs are a starting guess.**

Scripts: `circle_check.py`, `ite_vs_driven.py` (+ `look.py`), `wide.py`,
`census.py`, `trace.py`, `nncheck.py`.

## 2. Dark-state campaign (2026-10-26)

Si (n = 3.476, lossless) in air, λ = 1550 nm, pol 2, plane wave along z.
Minimise c_sca/area over mirror-symmetric shapes r(θ) = R·exp(Σ aₘ cos mθ),
M = 10, under perimeter ≤ 10 interior wavelengths (4459 nm). **Area is fixed
per run** and swept (R_eq = 120–700 nm): with only an upper size bound,
c_sca/area → 0 by shrinking (Rayleigh, c_sca ∝ A²). Reference: the circle of the
same area. Anchor: the circle against Mie, 4e-16.

- `shapes.py`: shape builder and objective.
- `campaign.py`: SLSQP, central finite-difference gradient, 8 starts per area
  → `results/campaign.json`.
- `sweep.py`: 1000–2400 nm sweep of each best design and its circle, dips
  refined, depth / FWHM / Q → `results/sweep.json`.
- `dip_widths.py`: decade widths of the 1550 nm dip (detuning to ×10 and ×100
  above the minimum) → `results/dip_widths.json`. `plot_sweep.py` →
  `results/sweep.png`.

### Results

Converged: every best design agrees with nn = 1536 to ≤ 3e-5 at nn = 768
(nn = 192 in the optimiser was up to 8 % off on needle shapes, but the optima
keep their order of magnitude); the sweep's nn is chosen per design and each dip
is re-checked at 2·nn (≤ 4e-5). Circle vs Mie ≤ 1e-15 at 1000/1550/2400 nm.

| R_eq (nm) | c/A at 1550 (nm⁻¹) | × below circle | width ×10 (nm) | width ×100 (nm) | min curvature radius (nm) |
| --- | --- | --- | --- | --- | --- |
| 280 | 1.1e-5 | 368 | 31 | 113 | 3.3 |
| **300** | **1.4e-7** | **23 379** | **3.7** | **12.4** | 1.4 |
| 320 | 9.4e-7 | 3 563 | 9.4 | 31.5 | 16 |
| 440 | 2.6e-6 | 1 647 | 8.9 | 36.7 | 6.4 |
| 480 | 4.7e-6 | 1 016 | 16.1 | 58.0 | 15 |
| 620 | 7.1e-6 | 180 | 8.7 | 39.2 | 113 |
| 700 | 1.8e-4 | 6 | Fano, FWHM 0.2 nm | — | 79 |

- The deep dips are **broadband near-zeros, not resonances**: width(×100) /
  width(×10) ≈ 3.2–4 ≈ √10, i.e. c_sca ≈ c_min + κ(Δλ)², the square of a
  coefficient crossing zero linearly. Their sharpness on a ×10 scale is set by
  the depth floor; the linear FWHM to the shoulder is 100–500 nm (Q_dip 3–10).
- R_eq 700 instead parks 1550 nm on a narrow Fano antiresonance next to a
  high-Q mode: FWHM 0.2 nm, only ×5.6 below the background (zoomed at 0.1 nm).
  640–680 are shallow (×4–10 below the circle, linear FWHM 39–89 nm); not
  checked for the same mechanism.
- Many optima are needle-like (140–380 nm: r_max/r_min 6–171, curvature radius
  0–5 nm, coefficients at the ±0.6 bound); no curvature rule was imposed.
- Above ~360 nm usually only 1 of 8 starts reaches the best value: the landscape
  is rugged and deeper optima likely exist there.

### Multipole content (`multipole_data.py` → `results/multipoles.json`)

Per-order power c_sca,n = (4/k)(|c_n|² + |c_−n|²); the orders sum to `c_sca` (to
1e-3 at the deep dips, better elsewhere) and the circle comes from analytic Mie.
At 1550 nm the optimiser does not null one or two orders: **every low order is
suppressed at once and the residual sits in the highest excited order**.

| R_eq (nm) | n = 0 … 4 (c_sca,n/area, nm⁻¹) | residual order |
| --- | --- | --- |
| 300 | ≤ 7.7e-9 each | n = 5, 1.3e-7 |
| 440 | ≤ 1.3e-7 each | n = 5, 2.1e-6 |
| 620 | ≤ 6.4e-7 each | n = 6, 4.9e-6 |

Field maps at 1550 nm (total inside and out, scattered outside = total −
e^{−ikz}) are evaluated at nn = 1536 with a 5-node-spacing band masked.

## 3. Manufacturable dark states: CD = 200 nm (2026-10-26)

**Rule.** |κ| ≤ 1/ρ_min with **ρ_min = CD/2 = 100 nm** on convex and concave
boundary alike, enforced at 512 boundary samples as an SLSQP inequality.
Curvature alone allows two lobes to pinch a sub-CD neck or gap, so each result is
also checked morphologically: opening and closing with a disk of radius CD/2 on
a 2 nm raster must leave the shape unchanged (`width_gap_ok`). Every
unconstrained optimum from §2 below 400 nm fails the rule (ρ_min 0–5 nm).

**Search plan.**
1. *Preview* (`campaign_cd.py 270`): R_eq 300–700 nm step 50, starts = circle,
   §2 optimum shrunk toward the circle until feasible, two random smooth starts;
   hard wall-clock deadline, best feasible point kept.
2. *Ladder*: full R_eq ladder, 16+ starts per size (the §2 landscape needed
   them above 360 nm); continuation from neighbouring sizes' optima.
3. *Width/gap in the loop*: if the preview's winners fail the morphological
   check, add a smooth neck constraint (e.g. distance between mirrored
   boundary points ≥ CD) rather than rejecting after the fact.
4. *Fabrication tolerance*: re-optimise the best designs for the worst case over
   a uniform boundary offset of ±10 nm (erosion/dilation), since a quadratic
   near-zero is only as deep as the process is accurate.
5. *Then* sweep, multipoles and field maps as in §2, nn checked at 768.

### Preview result (270 s, 9 sizes × 4 starts, `results/campaign_cd.png`)

All 36 runs pass the width/gap check; ρ_min lands at 99.5–100 nm (the bound is
enforced at 512 samples, so it can dip ≤ 0.5 % between them). Best per size,
converged at nn = 768 (≤ 2e-13):

| R_eq (nm) | 300 | 350 | 400 | 450 | **500** | 550 | 600 | 650 | 700 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| × below circle | 1.1 | 2.3 | 10 | 23 | **316** | 82 | 16 | 4.7 | 6.0 |

- Below ~400 nm the CD rule leaves almost no freedom: the shapes stay near the
  circle and the darkness of §2 (×23 000 at 300 nm) is gone.
- **R_eq 500 nm: c/A = 1.9e-5, ×316**, a single dip at 1550 nm, ×10 width 25 nm,
  a gentle lobed shape. It beats the unconstrained §2 optimum at 500 nm (×56),
  so §2 was under-sampled there; one random start found it.
- Best runs come from different starts at different sizes: more starts per
  size are the cheapest next gain (plan step 2).

### R_eq = 500 nm (`campaign_500.py`, `showcase_500_data.py`)

Width/gap now in the loop: two boundary points more than π·CD/2 apart along the
curve must be ≥ CD apart (on a curve with ρ ≥ CD/2 this binds only on a real
neck or gap); the morphological check still passes on every result.
Fabrication bias: `shapes._offset` moves the boundary δ along the normal
(spectral derivatives); a circle biased by δ matches Mie for R + δ to 1e-15.

| design | c/A as drawn | × below circle | worst c/A, \|δ\| ≤ 10 nm | × below circle |
| --- | --- | --- | --- | --- |
| best nominal (6 of 32 starts) | 1.9e-5 | 316 | 1.4e-3 | 4.2 |
| tolerant (worst case over δ = 0, ±5, ±10) | 3.5e-4 | 17 | 4.2e-4 | 14 |

- 32 starts confirm the preview optimum: 6 converge to it, the next family is ×268.
- A ±10 nm bias costs the best nominal design ×75: its quadratic near-zero is
  narrow in shape space as well as in wavelength.
- Worst-case optimisation trades the deep nominal dip for a flat ×14–17 over
  the whole ±10 nm window. On a 1 nm bias grid its worst is 4.18e-4, slightly
  above the 4.11e-4 seen at the five optimised biases.
- **A bias detunes the dip, it does not fill it.** For the best nominal design
  δ = −10 / 0 / +10 nm moves the dip to 1513 / 1550 / 1587 nm (≈ 3.7 nm of
  wavelength per nm of bias) at nearly unchanged depth (3.2e-5 / 1.9e-5 /
  2.9e-5). The tolerant design buys flatness at fixed 1550 nm by widening the
  dip and centring it at 1527 nm. If the operating wavelength can follow the
  fabricated part, the nominal design keeps ×190–316 across the whole ±10 nm.
- Multipoles at 1550 nm: the nominal design holds every order ≤ 7e-6 (n = 1, 3
  at ~2e-8); the tolerant one leaves n = 0, 1 at ~1.5e-4.

## 4. Over the SiN slab: guided-drive anapoles and absorption (2026-10-27)

Stack and decisions: `../../anapole-extinction-handoff.md` §5 (E1–E8). TE0 of SiN
400 nm on SiO₂, air cover, pol 2, 1550 nm, n_eff = 1.715. All P are fractions of
the incident mode power; `balance()` ≤ 5e-15 on every solve; nn 384 moves every
quoted optimum by ≤ 7e-4.

**Anchors.**
- `wg_abs_check.py`: P_abs from the boundary densities equals k0²·Im ε·∫|u|² from a
  Bessel fit of the interior field, ratio 1 to 9e-15 (circle, 3 radii × 3 Im ε). This is
  the term the anapole note's reality proof isolates: absorption *is* the imaginary
  part of the interior condition.
- `wg_dark.py` (ε sweeps): the disk's exact TM channel anapoles (note, Eq. twopoints)
  at X = k0·500 nm: n = 0: 5.54, 14.32; n = 1: 7.05, 17.87; n = 2: 10.29, 23.56;
  n = 3: 14.29; n = 4: 18.94.

**Circle baseline** (`wg_baseline.py`, R 150–650, gap 50/100/200, Im ε 0.01–1). Best
P_abs 0.455 (R 450, gap 50, Im ε 0.3), with R ≈ 0 everywhere: a travelling-wave
absorber, consistent with the circle's degenerate even/odd pair, so the 50 %
single-resonance ceiling of handoff §3.1.2 does not bind it (not yet proved).

**The drive is a plane wave at complex angle** (`wg_evanescent.py`). e^{iβx−γz} has
cos α = n_eff, so order m gains ≈ (n_eff + √(n_eff²−1))^{|m|} = 3.1^{|m|}: on the circle
|m| = 2 grows ×47, |m| = 4 ×4000. The free-space dark design (§3) is dark only for its
own drive: along x it is 0.75 × the circle, not 1/316. Under the evanescent drive in
free space it is 1.5 × the circle; over the stack 3.55 ×. The drive explains part,
slab feedback the rest.

**Guided-drive anapole** (`wg_dark.py` phase 1, lossless, CD rules, 40 starts, 63 min
on one core). 30 of 40 starts reach the same floor, **P_scatt = 2.87e-3**, ×27 below
the circle (0.079), T = 0.997, R < 1e-4; the rest stall at 1.4–2.8e-2. A floor, not a zero.
- **Its Re ε sweep dips at ε = 12.10 ± 0.05, the Si value 12.08**, at the same 2.9e-3:
  the optimiser moved an anapole permittivity of the shape onto the material, which is
  the note's picture (shape sets the anapole ε; darkness = material ε on it). The
  next dips are an order shallower (1.5e-2 at 7.85, 17.75).
- The circle's sweep dips (4.35, 8.35, 10.80, 13.30, 15.25, 18.40, 21.60, 23.80) are
  shallow (2–5e-2) and sit 0.5–1.2 off the channel anapoles: under guided drive every
  channel is lit, and the dominant |m| = 2 zero (10.29) shows up at 10.80.

**Anapoles do not store field, so they do not absorb.** Field-energy enhancement
P_abs / (k0²·Im ε·∫|u_inc|²) over the particle:

| design | Im ε 0.01 | Im ε 1 |
| --- | --- | --- |
| R_eq 500 circle | 0.82 | 0.47 |
| guided-drive anapole | 0.91 | 0.51 |
| free-space dark design | 1.40 | 0.61 |

At an anapole the interior field has the incident field's boundary data (note §2: an
interior problem, no outgoing wave, no resonance), so ∫|u|² is the incident energy times
an O(1) energy ratio, the same kind of ratio as the note's c0. P_abs is Born-like:
slope 0.998 in Im ε, at most 0.068 at Im ε = 1. ΔP_scatt does not follow the (Im ε)² law
(slopes −0.3 to −1.7): the 2.9e-3 residual dominates, and loss first *lowers* it.

**Pareto front, Im ε free** (`wg_dark.py` phase 3; `wg_opt.py` had Im ε = 0.3):

| P_scatt cap | P_abs, Im ε 0.3 | P_abs, Im ε free | Im ε | enhancement |
| --- | --- | --- | --- | --- |
| 0.05 | 0.139 | 0.366 | 1.39 | 0.53 |
| 0.02 | 0.114 | 0.199 | 2.10 | 0.32 |
| 0.01 | 0.089 | 0.114 | 0.58 | — |
| 0.005 | — | 0.053 | 0.37 | — |
| 0.002 | — | infeasible: below the 2.87e-3 floor | | |

The gains come from raising Im ε (brute loss), not from field storage: enhancement
falls below 1 as loss screens the interior. Uncapped, P_abs 0.391 with P_scatt 0.27.

**Reading.** Under these rules (R_eq 500, gap 100, CD 200 nm, Si real part) "dark" and
"absorbing" pull apart: darkness is an interior matching condition that caps the stored
energy at O(1) × the incident energy, and absorption beyond Born needs a resonance, an
exterior problem that radiates. A dark absorber needs both at once, e.g. a resonance
of the interior (high Q, weakly coupled) with the anapole cancelling its radiation —
a hybrid anapole / bound-state route — or the two-parity route (handoff §4 step 5).
pysie2d only, not externally validated.

## 5. Option (a): resonances bright to the guide, dark to radiation (2026-10-27)

Coupled-mode theory, one resonance with decay rates γ_wg (guide), γ_r (radiation), γ_a
(absorption). At critical coupling γ_a = γ_wg + γ_r: **P_abs = η** for the disk's degenerate
±m pair (travelling wave, R ≈ 0; η/2 for a standing wave), **P_scatt = η(1 − η), T = (1 − η)²**,
η = γ_wg/(γ_wg + γ_r). A dark absorber is η → 1. Because the guided drive weights order m by
≈ 3.1^|m| over radiation (§4), η should rise with m without any shaping.

`wg_wgm.py`: circle, 1550 nm, gap 100, lossless radius scan 400–1000 nm, then Im ε sweep at each
extinction peak. Order from the interior field (Fourier on r = 0.5R, 0.85R); the own-radiation
multipole misreads high-m modes (non-resonant background dominates it).

| R (nm) | mode | Im ε opt | P_abs = η | P_scatt | η(1−η) | T | (1−η)² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 445.3 | m 4 | 0.13 | 0.575 | 0.238 | 0.244 | 0.187 | 0.181 |
| 531.8 | m 5 | 0.063 | 0.751 | 0.201 | 0.187 | 0.048 | 0.062 |
| 616.3 | m 6 | 0.040 | 0.848 | 0.138 | 0.129 | 0.014 | 0.023 |
| 699.4 | m 7 | 0.025 | 0.876 | 0.108 | 0.108 | 0.016 | 0.015 |
| 942.5 | m 10 | 0.010 | **0.969** | **0.028** | 0.030 | 0.003 | 0.001 |
| 956.5 | m 7, 2nd radial | 0.050 | 0.674 | 0.210 | 0.220 | 0.116 | 0.106 |

- **η rises monotonically with m on the fundamental WGM ladder**, as predicted. m = 10: 96.9 %
  of the mode absorbed, 2.8 % radiated, at Im ε = 0.01. nn 565 vs 1130 agree to 1e-6,
  balance 6e-15.
- The one-parameter CMT predicts P_scatt and T to ~10 % on every sharp line. It fails on the
  broad features (optimum Im ε ≥ 1.5, P_abs < 0.2), which are not single resonances: the
  discriminator is the law itself.
- P_abs > 0.5 with R ≈ 0: the 50 % standing-wave ceiling of handoff §3.1.2 does not bind a
  degenerate pair, as the travelling-wave form says.
- None of this uses an anapole: the darkness is the evanescent drive's order selectivity. At
  R_eq 500 (E7) only m ≤ 5 is available (η ≈ 0.75); an anapole-type cancellation of γ_r would
  have to lift that, or E7 relaxed.

## 6. Method (C) of the anapole note: eigenchannel anapoles from S (2026-10-27)

`wg_reactance.py`, free space, pol 2 (note TM), 1550 nm, R_eq 500. T-matrix in the
`pysie2d.multipole` basis (column n = drive by i^n J_n e^{inθ}), S = I + 2T, |n| ≤ 9, mirror
x → −x splits even/odd blocks. Eigenchannels followed in ε by eigenvector overlap (nearest-δ
matching hands crossings to the truncation channels at δ ~ 1e-8); a crossing is kept only if the
same channel's δ is small when re-solved at the interpolated root (a high-Q line wraps ±π/2
inside one 0.02 step and fakes a sign change: n = 5 at 18.51, n = 6 at 23.85 on the disk).

**Anchors.** |S†S − I| ≤ 2e-13 (disk), ≤ 1e-9 (shapes). Disk: all exact channel anapoles
recovered, 5.535, 7.053, 10.287, 14.288, 14.320, 17.869, 18.941, 23.564, 24.206; n = 0 only even.
Plane-wave z darkness of the free-space design from Σ_j |⟨v_j,a⟩|² sin²δ_j: 4.772 / 1.509e-2 =
**×316**, the campaign's number by an independent route.

**Correction (interior cross-check, `wg_interior_check.py`).** The interior method (anapole
note §5: DtN(√ε X) − DtN(X) singular, eigen-eps `pysie2d_crosscheck.dtn`, interior rows only)
agrees with the S sweep on the disk to ≤ 6e-5 at all nine roots. On the two shapes the S-sweep
roots were **tracking artefacts** (channel swaps at near-crossings): at them no strongly coupled
channel (≥ 30 % weight in |n| ≤ 3) is near δ = 0 (|δ| ≈ 0.1–0.2). At the interior roots the
strong channels come closest to zero but do not reach it (min |δ| 0.05 at 11.947, 0.015 at
14.149): off the circle an interior transmission eigenvalue is a near-zero of S, not a zero,
because its background field v does not extend to an incident wave (note §7).

| shape | interior eigenvalues ε, even | odd |
| --- | --- | --- |
| guided-drive anapole (§4) | 5.966 7.374 10.866 14.149 16.157 17.940 24.225 | 8.501 11.423 16.483 20.068 21.513 |
| free-space dark (§3) | 7.289 9.874 **11.947** 15.428 23.357 23.673 | 7.185 **12.584** 18.927 20.572 |

- **Neither dark design has an interior eigenvalue at Si (12.08).** Both are near-zeros of a
  projection, the note's "what not to fit": the drive lands on channels with small δ.
  Free-space design, z drive: weight 5.4 on an even channel at δ = +0.018 (nearest interior
  eigenvalue 11.947, 1 % below Si) and 0.47 at δ = +0.145; x drive lights a δ = −0.945 channel, hence not dark.
- **The guided-drive dark state is a stack effect.** Free-space evanescent drive (same
  complex-angle wave, no slab): 64.8 vs disk 107.9, only ×1.7 (direct BIE, confirmed by the
  T-matrix sum). Over the stack: ×27. The slab's reflected field is part of the cancellation,
  so the anapole note's interior picture (particle alone) does not describe it.

## 7. All-direction darkness: minimise ‖T‖² = Σ sin²δ (2026-10-27)

`wg_alldir.py` (16 starts, 4 cores), `wg_alldir_page_data.py`. Free space, Si, 1550 nm, R_eq 500,
CD rules. Mean over incidence angle of ‖T a(α)‖² equals ‖T‖² for every design (check).
Circle 4.772 (a stationary point: first-order deformations cancel in the angle average).
Best 0.858, ×5.6, reached by 4 starts; nn 384 agrees to 4 digits. Free-space z-dark design 2.17,
guided-dark design 3.74. Over the slab the all-direction optimum scatters P_scatt 0.136 of the
TE0 mode (circle 0.079): all-direction darkness in free space does not transfer to the guided
drive. Interior eigenvalues of the optimum: even 7.456 8.524 10.052 13.805 18.680; odd 12.546
12.896 16.032 18.413 (none at 12.08).
