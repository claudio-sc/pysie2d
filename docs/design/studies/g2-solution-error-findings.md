# G2, solution-error half — findings (2026-09-13)

Script: `g2_solution_error.py` (runs in ~4 s, single core). Closes the three
"Not established here" items of the 2026-09-12 G2 findings. Everything is run
through **production** code on the landed Kress path (commit 771bc9b):
`Parametrisation` for the maps, `Geometry.gielis(parametrisation=...)` and
`BIESolver.scatter`. The only thing the script builds itself is the hard-`np.clip`
density — production deliberately does not ship one — and it builds it from the
production smoothing (`_layer_bandwidth`) and the production series/Newton
machinery, so the clamp is the *only* difference between the two graded maps.

## Setup

- Shape: the G2 ladder's **baseline star**, `m = 4`, `n1 = 6`, `n2 = n3 = 12`,
  `rad = 200` nm (layer width 0.362 rad, `M = 18`, `4M = 72`).
- `n_core = 1.5`, `n_clad = 1`, **TE** (`pol = 2`). λ_vac = 600 nm for the main
  table, repeated at 450 and 900 nm; the map's own `wavelength_ref` moves with
  it (§13.2 — the map is fixed *per run*, never read from a solve).
- Observable **`qext`**, which is one forward far-field amplitude and so does
  not pass through the angular quadrature G5 has yet to fix (`qsca` does).
- Ladder `nn = 40, 60, 80, 100, 120, 160, 200`; **the same `nn` for every map**,
  which is what "at equal `nn`" in the gate means (G2 correction 6).
- Reference: **uniform-θ Kress at `nn = 800`**, per doc B (no closed form exists
  for a star, and an arc-length reference would bake one competitor into the
  answer).

**The reference's own floor is measured, not assumed:**
`|qext(640) − qext(800)|/qext = 1.3e-15` at 600 nm (1.1e-15 at 450, 7.4e-16 at
900). That is ~6 ulp of a quantity of order 4 — i.e. the reference is at
round-off, and every error below 1e-14 in this document would be reference
noise. The smallest number reported is 6.0e-10, five decades above it, so no
tolerance in this study is floor-limited.

**Rate metric.** Local algebraic order `p` is the wrong summary for a spectral
scheme and is also unusable here: `qext` error passes through zero as it
converges, so consecutive-rung `p` swings between −11 and +19 *on the same
ladder*. The reported rate is the least-squares slope `b` of `ln(err)` against
`nn` over the **resolved** rungs `nn ≥ 80`. The cut at 80 is not a round number:
it is the `nn ≥ 4M = 72` smoothing-bandwidth criterion of the 2026-09-12 table,
rounded up to the next ladder rung — below it the graded map is under-sampled
by its own construction and the comparison is not about accuracy.

## 1. Adaptive vs constant density at equal `nn` — a rate, at last

| map | err @ nn=80 | @ 120 | @ 200 | rate `b` (λ=600) | `b` @450 | `b` @900 |
|---|---|---|---|---|---|---|
| **uniform θ** (production default) | 2.12e-5 | 8.86e-7 | **6.01e-10** | **0.0901** | 0.0936 | 0.0923 |
| adaptive, shipped `α_eff = 0.2155` | 8.82e-6 | 6.73e-6 | 2.80e-8 | 0.0657 | 0.0748 | 0.0687 |
| uniform arc length (`σ ≡ 1`) | 4.09e-4 | 8.09e-5 | 2.92e-6 | 0.0416 | 0.0405 | 0.0433 |

Three statements, in descending confidence:

1. **Adaptive beats constant density, decisively, at equal `nn`** — provided
   "constant density" means uniform *arc length*, which is what the gate's own
   correction 6 defines the fair baseline to be. Factor **46×** at `nn = 80`,
   **12×** at 120, **104×** at 200, and a rate ratio `b` of 1.58 that holds at
   all three wavelengths (1.58 / 1.85 / 1.59). **This clause of the gate now
   passes, and the pre-Kress obstruction is genuinely gone**: the 2026-09-12
   orientation ladder found adaptive 1.4–1.6× *worse* because a first-order
   error is set by the worst-resolved node. Under Kress the error is set by the
   analyticity strip of `f∘w`, and the ordering reverses.

2. **Uniform θ still beats both**, by 1.37× in rate over adaptive and 46× in
   error at `nn = 200`. This is the same result conventions §13 already records
   for the near-corner shape and the spiky star, now confirmed on the baseline
   star with a rate rather than a single-`nn` error. Grading *narrows* the
   analyticity strip; grading *toward curvature* recovers part but not all of
   what it costs on this shape.

3. The ordering is not an artifact of one wavelength. It is reproduced at 450
   and 900 nm with `b` moving by less than 8 % for every map, which also gives
   the uncertainty on `b`: about ±0.004, i.e. the 0.0901 / 0.0657 / 0.0416
   separation is ~6 σ between adjacent rows.

## 2. Smooth clamp vs hard `np.clip` — the first solution-error number

Compared at **pinned `α = 0.5`**, because with `α` derived from the band the
clamp is never active and the two maps are the same map (2026-09-12 finding 3 —
reproduced here: the smooth-saturation `α_eff` at pinned α is 0.2155, identical
to the shipped value, and the two ladders agree digit for digit).

Two honest comparisons, because the two clamps do not realise the same contrast:

| map | realised `C` | @ nn=80 | @ 120 | @ 200 | rate `b` |
|---|---|---|---|---|---|
| smooth saturation (shipped formula) | 2.50 | 8.82e-6 | 6.73e-6 | 2.80e-8 | **0.0657** |
| smooth, contrast matched (`α_eff = 0.3841`) | 5.00 | 6.09e-3 | 5.54e-4 | 2.09e-5 | 0.0486 |
| **hard `np.clip`** | 5.00 | 1.00e-2 | 3.45e-3 | 2.23e-4 | **0.0318** |

**The smooth clamp measurably beats the hard one, on solution error, on both
readings**: 2.07× in rate and 8000× in error at `nn = 200` against the shipped
smooth map, and — the like-for-like reading, same realised contrast, same band,
only the clamp differing — **1.53× in rate and 10.7× in error at `nn = 200`**.
This clause of the gate now passes too.

**And there is a stronger, pre-solve version of the same statement.** The
hard-clipped `σγ` is C⁰, so its Fourier coefficients decay algebraically and
**never reach the round-off plateau the production constructor's doubling loop
waits for**: the loop runs to `N_f = 2^20` and raises. Measured spectral tail
(top quarter of the spectrum, relative to `a_0`) at `N_f = 4096`: smooth
**3.0e-17**, hard clip **1.3e-5**, against the production acceptance threshold
`4·eps = 8.9e-16`. The study script has to pin `HARD_N_F = 4096` to make the
hard clip representable at all. That is a twelve-decade separation with no
solve in it, and it is the argument I would put in the spec: the shipped smooth
saturation is not a refinement of `np.clip`, it is the thing that makes the
`Parametrisation` constructor terminate.

## 3. Is `α_eff` accuracy-optimal, or only band-filling?

Sweep of a **prescribed** `α_eff` (same smoothed `s̃`, same machinery, only the
exponent imposed), λ = 600 nm:

| `α_eff` | 0.00 | 0.10 | 0.15 | 0.20 | **0.2155** | 0.25 | 0.30 | 0.50 | 0.70 |
|---|---|---|---|---|---|---|---|---|---|
| rate `b` | 0.0416 | 0.0523 | 0.0561 | 0.0651 | **0.0658** | 0.0759 | 0.0574 | 0.0252 | 0.0341 |

**There is a genuine interior optimum, near `α_eff ≈ 0.20–0.25, and the
band-derived 0.2155 lands inside it.** The optimum is a broad plateau, and its
location is at the noise level of `b`: the best of `{0.20, 0.2155, 0.25}` is
0.25 at 600 nm, 0.20 at 450 nm, 0.25 at 900 nm, with the three values spanning
only 0.0651–0.0759 at 600 nm. So:

- `α_eff` derived from the R band is **not** merely band-filling on this shape —
  it sits on the accuracy plateau, and both over-grading (0.5, 0.7: `b` collapses
  to 0.025–0.034, *worse than no grading at all*) and under-grading (0.10) are
  measurably worse.
- But it is **not optimal by construction either**: it is optimal because the
  band `C = 5` happens to map to the plateau. Changing the band moves it —
  `C = 2 → α_eff = 0.123`, `C = 5 → 0.2155`, `C = 10 → 0.260`, `C = 20 → 0.293`
  — so `C = 20` would push `α_eff` off the top of the plateau and `C = 2` off
  the bottom. **The free knob is the band, not `α_eff`**, and nothing in the
  present construction ties the band to accuracy. I would record the result as
  "`α_eff` is a sound *transfer function* from the band; the band's calibration
  is unvalidated", not as "`α_eff` is right".

## 4. Gate status

**G2's pass criterion is now met, on the baseline star, with one caveat that
changes what the gate means.**

- "Adaptive beats const-density at equal `nn` on a high-curvature shape": **met**
  — 46–104× and a 1.58× rate advantage over uniform arc length, reproducible at
  three wavelengths. The superseding note in the study plan ("has now failed
  twice … the criterion is therefore not a gate on the density's construction")
  can be revised: under the *landed* Kress path, on a star rather than an
  ellipse, it passes.
- "Smooth clamp measurably beats the hard one": **met** — 1.53× in rate and
  10.7× in error at matched realised contrast, plus the twelve-decade,
  solve-free spectral-tail separation (3.0e-17 vs 1.3e-5) that makes the
  production constructor terminate at all.
- **Caveat, and it should stay attached to the gate:** the adaptive map still
  loses to the *production default*, uniform θ, by 1.37× in rate and 46× in
  error at `nn = 200`. So the gate passes against the baseline the gate names,
  and the density remains, on this shape, a scheme with no user: nobody should
  prefer it to the default here. Whether it acquires one is exactly G3's
  near-corner question, where the 2026-09-12 cost curve predicts the knee
  between `n1 = 6` (this shape) and `n1 = 12`. **I would not promote
  conventions §13's "still tentative" note to settled on the strength of this
  study** — it establishes that the density is *well constructed*, not that it
  is *worth selecting*.

## 5. What would change my mind

- A near-corner shape (`n1 = 20`, `n2 = n3 = 50`) where uniform θ's strip is
  narrow enough that grading overtakes it. That is the one measurement that
  would turn "well constructed" into "worth selecting", and it is G3's, not
  this study's — it needs `nn ≳ 4M = 376`, an order of magnitude more cost than
  anything here.
- A complex-λ (QNM) probe instead of a driven `qext`. Nothing here is real-only
  — the maps are λ-independent by construction (§13.2) and the assembly path is
  the production one, so complex `k` works — but every number above is at real
  `k`, and G4 has not run. I would not assume the map ranking transfers to a
  Beyn contour without measuring it.
- A second observable. `qext` at one λ is one scalar functional of the solution;
  a near-field or a pole position weights the boundary error differently, and
  the 1.37× uniform-θ margin is small enough that a different functional could
  plausibly reorder the top two rows. The 46× advantage over uniform arc length
  is not.
