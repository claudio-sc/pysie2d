# Handoff: from dark particles to extinguishing a guided mode

Branch `dev-anapole-extinction`, cut from `v1.2-slab-waveguide` (PR #27, v1.2.0
releasing) on 2026-10-26. Merge `main` in once #27 lands. Everything here is study
work under `docs/design/`; nothing in `src/` changed.

**Goal of the next phase.** A particle coupled to a slab waveguide removes power
from the incident guided mode. The idea: drive the particle into an anapole-like
dark state, so it radiates little while storing a large internal field, and let
a lossy particle turn that field into absorption. This document says what we
have, what a probe already showed, and what is needed to state feasibility.

## 1. What exists

### 1.1 Free-space dark particles (study `docs/design/studies/anapole/`, README §1–3)

- **Polarisation names are swapped against the anapole literature**
  (conventions §1): pysie2d `pol = 2` (`E_y`) is the eigen-eps note's **TM**.
- **Interior transmission eigenvalues are not plane-wave anapoles off the
  circle.** On superellipses the driven multipole zero leaves the eigenvalue
  continuously: 7 % at p = 4 for pol 2, 0.23 % for pol 1. Darkness depends on
  the excitation, not only on the particle (README §1).
- **Dark states are many orders suppressed at once.** Si (n = 3.476) in air,
  pol 2, 1550 nm: the optimiser holds every order n ≤ 4 near zero and leaves the
  residual in the highest excited order. Best unconstrained: ×23 000 below the
  same-area circle at R_eq = 300 nm, but with 1 nm needles (README §2).
- **Under a 200 nm critical dimension** (radius of curvature ≥ 100 nm, width and
  gap ≥ 200 nm, both enforced in the optimiser): nothing below ~400 nm; best at
  R_eq = 500 nm, ×316 (README §3).
- **A ±10 nm boundary bias detunes the dip by ±37 nm** at nearly constant depth.
  At fixed 1550 nm a worst-case design holds ×14–17 across ±10 nm (README §3).
- Showcase page: "Dark Silicon Wires" artifact; rebuild with
  `build_page.py` after the `*_data.py` scripts.

### 1.2 v1.2 waveguide capability (slab-spec, behaviour table §8.4)

| Need | v1.2 | Note |
| --- | --- | --- |
| Guided-mode incidence on a particle | `BIESolver.scatter_mode(λ, mode, ±1, x_ref=)` | real λ, lossless stack (T2) |
| Lossy particle | `Material(epsi=…)` | the particle may be lossy; the stack may not |
| Full power account | `ScatterResult.mode_amplitudes(x_ref)` | T, R per mode, `p_up`, `p_down`, `p_abs`; `balance()` |
| Clusters | `ClusterBIESolver`, same calls | |
| Shape gradients | `mode_sensitivity` (central FD, frozen map) | 2 solves per parameter |
| Near field in the layers | `eval_field` | R19 |
| Multipoles over a stack | **no** | `NotImplementedError` |
| QNMs over a stack | **no** | R2, `ValueError` |
| Particle inside the core | **no** | R1 |
| 3-D / 2.5-D | **no** | cross-section only until v1.4 |

## 2. Probe (`studies/anapole/wg_probe.py`, 2026-10-26)

SiN slab (n = 2.0, 400 nm) on SiO₂, air cover, TE₀ (pol 2, n_eff = 1.715),
λ = 1550 nm. Si particle of R_eq = 500 nm, 100 nm above the slab, nn = 192.

| particle | Im ε | T | R | up + down | abs | balance |
| --- | --- | --- | --- | --- | --- | --- |
| circle | 0 | 0.921 | 0.000 | 0.079 | 0 | 7e-16 |
| circle | 0.05 | 0.918 | 0.000 | 0.073 | 0.009 | 2e-16 |
| free-space dark design | 0 | 0.699 | 0.021 | 0.280 | 0 | 2e-15 |
| free-space dark design | 0.05 | 0.703 | 0.019 | 0.249 | 0.030 | 2e-15 |

- **A solve costs ~0.1 s** (assembly ~0.01–0.04 s plus mode amplitudes 0.05 s).
  Optimisation over a stack is as affordable as in free space.
- **The free-space dark design is not dark over a waveguide.** It radiates 28 %
  of the mode's power, ×3.5 more than the circle. Two reasons, both expected
  from §1.1: the drive is the slab's evanescent tail (e^{iβx − γz}, β > k), not
  a plane wave, and the slab's reflected field acts back on the particle.
- **Absorption is small at Im ε = 0.05** (≤ 3 %). Nothing has been optimised
  for it yet.

## 3. What feasibility needs

### 3.1 Physics to settle first

1. **The figure of merit.** Mode extinction is 1 − T = R + P_rad + P_abs
   (+ other guided modes). "Remove power by absorption" means maximise P_abs
   with P_rad and R small. Decide which leftover is acceptable: reflected power
   is lost to the mode too, but is not dissipated.
2. **An absorption ceiling.** For a mirror-symmetric absorber on a single-mode
   two-port, one standing-wave resonance at critical coupling absorbs at most
   50 % (T = R = 25 %). Reaching 100 % needs an even and an odd resonance,
   degenerate and both critically coupled ("degenerate critical coupling",
   Piper, Liu & Fan, APL 104, 251110 (2014) — **verify** before relying on it).
   Our shapes are mirror-symmetric about x = 0, so this bound applies to them
   directly. It tells us whether one dark state can ever exceed 50 %, or
   whether we need two.
3. **What "anapole" means under guided drive.** The plane-wave definition does
   not carry over (§1.1, §2). The working definition should be the observable
   one: P_up + P_down → 0 while the internal field stays large. Whether that
   is reachable with finite coupling to the guide is the central open question.
4. **The absorber.** Si is lossless at 1550 nm. Absorption needs a lossy
   material (Ge, doped Si, a thin absorbing film) or a different wavelength.
   P_abs = k·Im ε·∫|E|² inside, so critical coupling sets an optimal Im ε for
   a given radiative coupling. Sweep Im ε rather than fixing it.

### 3.2 Code work, smallest first

1. **Study scripts only (no `src/` change):** sweeps and optimisation with
   `scatter_mode` + `mode_amplitudes`. Objective: maximise P_abs (or minimise
   T) with CD rules (`campaign_cd.py` constraints) and P_rad as a reported
   output, then as a penalty.
2. **Diagnosing darkness without multipoles over a stack:** use `p_up`,
   `p_down` and the far-field patterns (`far_field`, `far_field_down`) first.
   A particle-centred decomposition of the particle's own radiated field from
   its boundary densities (free-space Green function) is a small, separable
   addition if the far fields prove insufficient. It would be new physics
   (non-negotiable #3: needs an anchor, e.g. it must reproduce `multipoles()`
   exactly with no stack).
3. **Resonances:** QNMs over a stack are out of scope (R2). Read Q factors and
   critical coupling from driven spectra (Lorentzian fits of T, P_abs against
   λ and Im ε), with free-space QNMs as a guide only.

### 3.3 Validation anchors

- `balance()` to round-off on every solve (measured ≤ 2e-15 above).
- **P_abs two ways:** from `mode_amplitudes` (boundary flux) and from
  k·Im ε·∫|E|² over the particle interior (`eval_field`). Independent paths.
- **Small-particle limit:** Born scaling (slab-spec s10) for the extinction of
  a tiny lossy wire.
- **External:** MEEP is v1.3's milestone; a feasibility claim with numbers
  should be quoted as "pysie2d, not yet externally validated" until then.

## 4. Proposed plan (each step with its check)

1. **Owner decisions** (below). Check: none, but every later number depends on them.
2. **Baseline map:** circle radius × gap × Im ε, recording T, R, P_rad, P_abs.
   Check: balance ≤ 1e-13 everywhere; the circle's best P_abs compared with
   the 50 % single-resonance ceiling.
3. **Optimise for P_abs** under the CD rules at fixed Im ε, then sweep Im ε
   around the optimum. Check: nn 192 vs 384 on the optimum; P_abs two ways
   agree.
4. **Diagnose:** is the optimum dark (low P_rad, high ∫|E|²)? Field maps,
   far-field patterns, spectra. Check: the ratio P_abs/(Im ε·∫|E|²) is
   constant across designs (the identity), so a large P_abs really comes from
   field storage.
5. **Two-parity designs** if step 2–3 stall near 50 %: break the single-
   resonance limit with an even/odd pair (shape or a two-particle cluster).
6. **Robustness:** ±10 nm bias and λ detuning, as in README §3.

## 5. Owner decisions needed

1. **Stack:** SiN (as probed), SOI (220 nm Si on SiO₂) or another; and λ.
2. **Particle material and loss:** which absorber, which Im ε range.
3. **Placement:** gap above the slab (R1 keeps particles outside the core).
4. **Figure of merit:** P_abs alone, or 1 − T with P_rad and R limits.
5. **Single particle or clusters** for the two-parity route.

## 6. Where things are

- Study: `docs/design/studies/anapole/` (README is the record; large JSON
  outputs are gitignored and regenerable).
- Free-space objective, shapes, bias offsets: `shapes.py`.
- CD constraints and width/gap check: `campaign_cd.py`; in-loop neck
  constraint and the tolerance optimisation: `campaign_500.py`.
- Probe: `wg_probe.py`.
- Memory notes: `anapole-dark-state-study`, `artifact-svg-colour-tokens`
  (recurrent artifact plot bugs).
