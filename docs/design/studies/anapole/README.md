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
