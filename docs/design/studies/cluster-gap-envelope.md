# The near-gap quadrature envelope: how many nodes a gap needs

M2 of the multiparticle spec (§5), the measurement behind `GAP_ENVELOPE_C =
100.0` in `src/pysie2d/cluster.py` and the `ClusterGapWarning` raised by
`ClusterBIESolver._check_gap`.

Script: `m2_sweep.py` in the scratchpad of the measuring session; the setup is
reproduced in full below so the sweep can be rebuilt from this page alone.

This is written in the style of the `eval_field` near-boundary rule — a scalar
constant guarding a silent failure mode — and, unlike that rule, it is measured
rather than argued.

## The law being fitted

Two boundaries separated by `gap` put a near-singularity of the cross-block
kernel a distance `gap` off each contour. For a circle of radius `a` that
near-singularity sits at conformal half-width `≈ √(2·gap/a)` in the
parametrisation variable, so the trapezoid rule — spectrally accurate on an
analytic periodic integrand — converges at `exp(−nn·√(2·gap/a))`. Fixing an
accuracy target therefore fixes

    nn ≳ C·√(a/gap),

and the whole measurement is the single constant `C`. A fitted value far away
from the pilot's order of magnitude would mean the sweep is wrong, not the law.

## Method: BIE self-convergence, *not* the addition theorem

The obvious reference is the two-cylinder addition-theorem solution, and it is
the wrong one: its own truncation degrades as the gap closes, at a different
rate, so it reports a floor that belongs to the reference rather than to the
solver. Measured (spec §5, §10 point 4): at `gap = 0.016 λ` the BIE solution at
`nn = 240` is self-converged to 2.1e-15, while the same solution judged against
the addition theorem reads 5.2e-9 — six orders of magnitude of "error" that is
entirely the reference's.

So the envelope is measured by **self-convergence**: for each gap, take a
high-`nn` BIE solution as truth and find the smallest `nn` on the grid that
reaches the target.

- Two circles, radii `a = 180 nm` and `110 nm`; `a` is the **larger**.
- `n_core = 2.0` and `1.6`, `n_clad = 1`, `angle = 37°` (no symmetry to hide
  behind).
- Truth at `nn = 1024`; error = max relative far-field amplitude error over 361
  angles, normalised by the peak amplitude.
- Criterion `nn*` = smallest `nn` with that error below **1e-13** — round-off
  for this quantity, so `nn*` is "fully converged", not "converged to a
  tolerance someone chose".
- `nn` grid spaced by `√2`: 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512.
- `x = 2π·a/λ_vac` varied by varying λ: `x = 1.00`, `1.79` (the pilot), `3.50`;
  both polarisations at `x = 1.79` and `x = 3.50`.

## The sweep

`nn*` per configuration, and `C = nn*·√(gap/a)` from the binding (largest)
`nn*` in each row:

| `gap/a` | `x` 1.00 TE | 1.79 TE | 1.79 TM | 3.50 TE | 3.50 TM | `C` |
|---|---|---|---|---|---|---|
| 5.057 | 23 | 32 | 32 | 45 | 45 | 101.2 |
| 1.723 | 32 | 32 | 32 | 45 | 45 | 59.1 |
| 0.833 | 45 | 45 | 45 | 45 | 45 | 41.1 |
| 0.499 | 64 | 45 | 64 | 64 | 64 | 45.2 |
| 0.222 | 91 | 91 | 91 | 91 | 91 | 42.8 |
| 0.113 | 181 | 181 | 181 | 181 | 181 | 60.7 |
| 0.056 | 256 | 256 | 256 | 256 | 256 | 60.7 |
| 0.032 | 512 | 362 | 512 | 512 | 512 | 91.1 |

## What it shows

**Below `gap/a ≈ 0.83` the requirement is set by the gap and by nothing else.**
Across a 3.5× range in size parameter and both polarisations, the five columns
agree rung for rung on the `√2` grid, with two single-rung exceptions (`x` 1.79
TE at `gap/a` 0.499 and 0.032). That agreement is what licenses a *single*
constant rather than a family in `x` and `pol`: the near-gap integrand's
analyticity strip is a property of the geometry, and the size parameter only
enters through how well the particle itself is resolved.

**Above it, the roles swap.** At `gap/a = 5.06` the `x = 3.50` pair needs
`nn* = 45` while the `x = 1.00` pair needs 23 — the gap is irrelevant there and
`nn*` is simply what it takes to resolve a particle of that size parameter. The
two widest rows are therefore **excluded from the fit**: fitting a `√(a/gap)`
law to points where the gap is not the constraint would pull the constant
upward for the wrong reason.

**The gap-limited band is `C ≈ 41–91`,** and the `√2` grid is why it is that
wide: a rung of the grid is a factor 1.41 in `nn`, so a single measurement pins
`C` only to ±41 %. The one value below the band — `C = 31.8`, `x` 1.79 TE at
`gap/a = 0.499`, where `nn* = 45` rather than the 64 the other four columns need
— is that granularity, not a different law.

## The constant, and why 100

`GAP_ENVELOPE_C = 100.0`: above the measured band's maximum, not merely inside
it.

A warning that fires early costs a user one line of output; a warning that
stays silent while the cross-block quadrature loses eight digits costs them a
wrong answer they have no reason to doubt — and the tightest gap measured is
exactly where a smaller constant fails that way. The law `nn ~ C·√(a/gap)`
predicts a constant `C` set only by the target accuracy, not by the gap, so the
spread across the sweep (`C ≈ 41` at the widest gap-limited row up to `C = 91`
at the tightest) is itself informative: part of it is the `√2` grid's ±41 %
quantisation, but the rise at `gap/a = 0.032` is corroborated independently —
the spec's own pilot, at the same `gap/a`, found `nn = 240` reaching only
3.6e-11 (not round-off), and this study's stricter 1e-13 target at that same
gap needs `nn = 362`–`512` (`C` up to 91). That is a real, if modest, upward
curvature in the true relation near the singularity, not a measurement
artefact — so a constant fit to the mid-range points would systematically
under-predict at the tightest gaps, exactly the failure mode a warning cannot
afford.

`100` therefore trades the mid-range warning threshold for correctness at the
tight end: at gap-limited points from `gap/a = 0.83` down to `0.056` it
over-predicts by roughly 1.6–2.4×, and at the widest, particle-limited gaps
(where the need is set by resolving the particle, not the gap) it warns
earlier still, by construction of the `√(a/gap)` form. Both are the safe
direction. The measurement is validated only over `gap/a ∈ [0.032, 5.06]`;
`_check_gap`'s guard should be read as a diagnostic, not a certified bound,
outside that range — extrapolating `√(a/gap)` below `gap/a = 0.032` is
unmeasured, and the mild upward curvature seen here means a further factor
could in principle be needed at even tighter gaps than tested.

## Reconciling with the pilot's 39–57

The single-`x` pilot quoted in spec §5 gives `nn·√(gap/a)` in **39–57**, and
this study's gap-limited band is 41–91. They are the same measurement at two
different accuracy targets, and the difference is not noise:

- the pilot judged convergence at roughly **1e-10** on a **factor-2** `nn` grid;
- this study uses **1e-13** on a **`√2`** grid.

Under `error ~ exp(−nn·√(2·gap/a))` the node count needed scales with
`ln(1/ε)`, so tightening the target from 1e-10 to 1e-13 multiplies `C` by
`ln(1e13)/ln(1e10) = 1.30`. That maps the pilot's 39–57 to **51–74** — and the
measured 41–91 straddles it. The pilot was right about the law and right about
the order; what it could not see is the spread, because one `x`, one λ and one
polarisation cannot show a spread.

The spec itself (§5) treats a fitted constant far outside 39–57 as a signal the
*sweep*, not the law, is wrong — the right default, since a sweep bug is far
more common than new physics. `GAP_ENVELOPE_C = 100` sits above even the
rescaled 51–74 band, so that caution is being overridden here, not ignored:
the excess is anchored to the spec's own pilot table (its `nn = 240` at the
same `gap/a` reads 3.6e-11, corroborating this study's `nn = 362`–`512` need at
the stricter target, independently of this sweep's own code), not to a
one-off measurement this sweep alone produced.

## Scope, per D9

**The envelope is measured on circles only.** Nothing here says what two facing
flat sides or two facing tips require, and `_check_gap` says so in the warning
text itself when any particle in the pair is non-circular: for those, the number
is a diagnostic, not a bound.

## Open, and deliberately not resolved here (spec §11)

- **Does the envelope depend on the facing shape type?** Flat-against-flat and
  tip-against-tip are plausibly different problems, and the answer decides
  whether this is one constant or a family in the facing curvature.
- **Is the `a` in `√(a/gap)` the larger radius, the smaller, or a harmonic
  mean?** This study's data is consistent with the **larger** — with a 180/110
  pair the smaller particle never set the binding `nn*`, which is what one
  expects if the wider analyticity strip belongs to the smaller circle. But one
  radius ratio cannot distinguish "larger" from "harmonic mean" (they differ by
  only 1.4× here), and a two-radius-ratio sweep is still owed.
