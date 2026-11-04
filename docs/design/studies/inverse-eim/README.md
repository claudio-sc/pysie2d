# Inverse EIM — reproducing Høvik & Aksnes (2024)

Reproduction of J. Høvik and A. Aksnes, "Inverse Effective Index Method for
Two-Dimensional Simulations of Photonic Components", IEEE Photon. J. 16,
6600106 (2024), the method v1.4 adopts (roadmap R10, revised 2026-11-02).
Scope (owner, 2026-11-02): Figs. 2–5. Fig. 6 (a 5 µm ring) is out: pysie2d
particles cannot contain a hole.

Paper materials throughout: Si core 3.47 on SiO₂ 1.44 under air 1.0,
λ = 1550 nm. The 3D n_eff comes from MPB (`mpb_neff.py`, MEEP conda env); every
2D quantity comes from `Stack.guided_modes`.

| Step | Paper | Here | Status |
|---|---|---|---|
| R1 — Fig. 2, plain-EIM side view, 220 nm slab, TE | n_eff = 2.825 | 2.8247 | done |
| R2 — Fig. 3, inverse-EIM n_2D, 500 × 220 nm, TE | 2.763 | **2.7637** | done |
| R3 — Figs. 4–5, β error maps over the single-mode domain | ≤ 17 % EIM, ~100 % none, < 0.01 % inverse | 11–37 % EIM, 104–127 % none, ~1e-13 % inverse | done |

## R2

MPB, 500 × 220 nm, 4 × 4 µm cell. Second-order convergence from res 64
(steps 0.0131 → 0.0033 in TE); the oxide-clad strip at the same settings
extrapolates to 2.438, the commonly quoted ≈ 2.44.

| res (px/µm) | TE n_eff | TM n_eff |
|---|---|---|
| 32 | 2.35982 | 1.56492 |
| 64 | 2.37293 | 1.57239 |
| 128 | 2.37618 | 1.57395 |
| → ∞ (2nd order) | **2.3773** | **1.5745** |

The top-view slab is 500 nm wide; the 3D TE-like mode is pysie2d `pol = 1`.
n_2D depends on what the slab's two lateral sides are:

| Lateral claddings of the top-view slab | n_2D (TE) | 3D n_eff that gives 2.763 |
|---|---|---|
| air \| air (the physical sides of the core) | 2.7920 | 2.3450 |
| oxide \| oxide | 2.7371 | 2.4037 |
| **air \| oxide** (the side view's claddings) | **2.7637** | **2.3766** |

**Finding.** The paper's Fig. 3 uses its eq. (2) with the side view's two
claddings, n_cl = 1 and n_sub = 1.44, on the two *lateral* sides of the
top-view slab. Only that reading gives 2.763, to the published digits, and it
implies a 3D n_eff within 0.03 % of MPB's. Inverse EIM reproduces β whatever
the lateral claddings are, by construction. The choice changes the 2D mode's
lateral tails, so it changes the coupling to a scatterer, which is what the
v1.4 3D MEEP reference measures. It is a convention v1.4 has to pin.

## R3

`r3_sweep.py` (MPB, single core, 15.6 h, 2026-11-02/03) → `r3.jsonl`;
`r3_analyse.py` → `r3_errors.json`. Grid w 100–1000 × h 100–500 nm, 10 nm
steps: 3731 strips, 1434 single-mode (at most one guided band per y-parity),
1433 solved (one fails at cutoff), 83 excluded because the cell hit 12 µm.
That leaves 1125 TE and 789 TM strips.

Max (median) |Δβ/β| over the domain:

| Lateral claddings | TE none | TE EIM | TM none | TM EIM |
|---|---|---|---|---|
| air \| air | 104 % (42) | 11.3 % (2.2) | 126 % (49) | 21.8 % (6.5) |
| oxide \| oxide | 106 % (45) | 36.7 % (5.8) | 127 % (51) | 31.4 % (9.0) |
| air \| oxide | 105 % (43) | 26.6 % (4.2) | 126 % (50) | 26.9 % (7.9) |

Inverse EIM is ≤ 2e-13 % in every case: it reproduces β by construction, so
the paper's "< 0.01 %" is its own interpolation noise, and the column is a
check of the arithmetic, not of the physics.

**What is reproduced:** none ≈ 100 %, EIM errors at the tens-of-percent level
and largest at low confinement (median 2–5× higher in the less-confined half;
the maxima sit at n_eff ≈ 1.48–1.52, near cutoff), inverse EIM exact. **What
is not:** the paper's 17 % maximum lies between our conventions. Our domain
reaches closer to cutoff than the paper's (its 3496 count does not map onto
our 1914 strip–polarisation pairs), so the maxima do not discriminate the
lateral convention. Fig. 3 (R2) remains the evidence that the paper uses
air \| oxide.

## 3D reference and the comparison rule

**Fixed 2026-11-04, before any pysie2d number for this case was computed.**
Nothing below may change after the 2.5D side is seen; a change needs a new
dated entry and a statement that the comparison was already known.

**Case** (`validation/meep/run_strip3d.py`). The paper's air-clad strip,
500 × 220 nm Si (3.47) on SiO₂ (1.44) under air, TE₀, 31 λ in 1500–1650 nm. A
Si pillar beside it: r = 250 nm, 220 nm tall, on the oxide, 60 nm edge gap,
centred 560 nm from the strip axis. Runs: bare strip (`norm`) and `pillar`,
launched toward +x; the pillar is x-symmetric, so the other drive is not run.
Ladder res 25/30/40/50 px/µm, single core, 6.6 h (2026-11-03/04). Raw output
in `validation/_study/v14/` (not in git).

**Accepted systematic (owner, 2026-11-04).** With PML in the cell, MEEP's
plane mode solver, used by the source and the monitors, returns n_eff 2.297
(res 40) for TE₀ instead of 2.377 (MPB; MEEP's own solver in a periodic cell
without PML gives 2.376). The profile is still TE₀: overlap 0.9966 with the
correct one, Ey-dominant, 65 % of |E|² in the core, centred. The mismatch
cancels exactly in the bare-strip normalisation, which is pure TE₀. Radiated
field projected onto the distorted profile is the residual: bounded at a few
1e-3 in |t|² in the worst case, and measured below 3e-4 by the two monitor
planes (2.0 and 2.5 µm see different radiation content but agree to 2.9e-4
at res 50). Accepted with that measured bound, rather than rerunning with a
projection onto MPB's profile.

**Convergence.** The FDTD wave's own n_eff (phase advance between planes)
against MPB's 2.3773: errors 0.0234, 0.0197, 0.0122, 0.0074 at res 25, 30,
40, 50. The apparent order rises from ≈ 1 to ≈ 2.3, so res 25 and 30 are
pre-asymptotic and are not used.

**The rule.** For each observable X(λ) ∈ {|t|², P_rad = 1 − |t|² − |r|²,
complex t}:

1. MEEP value X_M = second-order Richardson extrapolation of res 40 and 50
   (order measured on n_eff above).
2. MEEP uncertainty U = |X_M(p = 2) − X_M(p = 1)| + f_norm + f_plane, where
   f_norm = max |1 − P_far/P_near| of the bare strip at res 50 (power not
   conserved by the guided mode, 5.8e-4) and f_plane = max |t(2.0 µm) −
   t(2.5 µm)| at res 50 (2.9e-4; it also bounds the accepted projection
   systematic). Maxed over λ, like v1.3.
3. pysie2d side: top view at `pol = 1` (3D TE-like, conventions §3 of the
   roadmap); strip = a 500 nm `Stack` layer of core index n_2D(λ), by inverse
   EIM from MPB's n_eff(λ) at every λ of the band (res 64 and 128,
   second-order extrapolated, as R2); pillar = a disc of r = 250 nm with the
   same n_2D (R22), on the cover side, centre 560 nm from the strip axis;
   nodes doubled until |S| moves < 1e-8. All three lateral conventions,
   air | air, oxide | oxide and air | oxide, are computed and reported; with
   air | oxide, the pillar side is air, as in 3D.
4. Reported: the model error E(λ) = X_2.5D − X_M, with U, and max |E| / U. The
   2.5D model "agrees within MEEP's uncertainty" only if |E| ≤ 3U at every λ
   (v1.3's factor). Otherwise E is the 2.5D model's error, reported as such,
   never as a tolerance of the code. |r|² (≈ 4e-4) is recorded but not
   judged if U exceeds half of it.
5. Which lateral convention v1.4 adopts as its default is an owner decision,
   taken with all three results on the table. The documentation will say
   that it was chosen after this comparison.

## Reproduce

    ~/miniforge3/envs/pysie2d-meep/bin/python mpb_neff.py 500 220 128
    uv run python docs/design/studies/inverse-eim/r3_analyse.py
