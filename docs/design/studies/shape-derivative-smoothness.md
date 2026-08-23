# Shape-derivative smoothness: which `h` a central difference may use

Measured by `gate9_smoothness.py` in this directory, at commit `da4469d`.
Design point `m = 4`, `b₀ = 1.20`, `rad = 200 nm`, `n_core = 3`, `pol = 2`,
`λ = 700 + 8i` — complex λ on purpose, so the number describes the QNM path
rather than a real-wavenumber fast path.

`D(h) = [M(b₀+h) − M(b₀−h)] / (2h)`, deviation measured against `D(1e-6)`, the
most accurate step available.

| `h` | `n_pts` 120 | 200 | 400 | entries carrying it (200) |
|---|---|---|---|---|
| 1e-2 | 1.09e-3 | 1.09e-3 | 1.11e-3 | 0.952 |
| 1e-3 | 6.89e-5 | 8.29e-5 | 1.42e-4 | 0.487 |
| 1e-4 | 2.53e-5 | 3.24e-5 | 4.73e-5 | 0.051 |
| 1e-5 | 4.66e-9 | 5.60e-9 | 8.52e-9 | 0.000 |
| 1e-7 | 4.90e-8 | 6.18e-8 | 9.03e-8 | 0.000 |

## What it shows

**There is no `O(h²)` plateau across `[1e-4, 1e-3]`.** From 1e-3 to 1e-4 the
deviation falls by 2.6×, not the 100× a central difference's truncation term
would give; then from 1e-4 to 1e-5 it collapses by 5800×, far faster than any
truncation term can. A ladder that stalls and then falls off a cliff is not one
error source with a rate, it is two sources crossing over.

**The second source is the arc-length node placement, and the entry fraction
identifies it.** At `h = 1e-3` half the matrix carries the deviation; at 1e-4,
5 %; at 1e-5, none. That is a *count*, not a magnitude, decaying — the signature
of a discrete event, not of a Taylor remainder. Running the identical ladder on
the node angles alone, with no electromagnetics in it, reproduces the whole
shape (9.04e-5 → 4.51e-5 → 1.79e-5 → 1.26e-8, fraction 0.920 → 0.240 → 0.020 →
0.000), which localises it in the geometry.

The mechanism: inverting arc length to θ goes through `np.interp`, continuous
but only **piecewise** linear in the shape parameter. A node whose bracketing
cell differs between `b−h` and `b+h` contributes an O(1) error to the quotient,
and the number of such nodes is proportional to `h`. So the term is O(h) rather
than O(h²), it is not monotone in `h`, and it **grows with `n_pts`**
(2.53e-5 → 3.24e-5 → 4.73e-5 at `h = 1e-4`) because a finer boundary has more
cells to cross. Refining the discretisation makes this error worse, which is the
opposite of every other error in this package.

**The floor is ~1e-8, not 1e-9.** Subtractive cancellation is already visible at
`h = 1e-7` (6.2e-8 at `n_pts = 200`) and `h = 1e-6` is the best step measured.

## Consequence

The usable window is **`[1e-6, 1e-5]`**, bounded below by cancellation and above
by node re-placement, and it delivers ~8 digits. The window `[1e-4, 1e-3]` is
usable only to ~4 digits and its error does not shrink smoothly with `h` — so a
finite-difference-versus-adjoint check demanding *linearity in the step* across
that window is checking a region where the error is dominated by a discrete,
non-monotone term.

This contradicts the recorded reference measurement (clean `O(h²)`, ~6 digits,
window `[1e-4, 1e-3]`), which this script does not reproduce at **any** of the
three resolutions, `n_pts = 120` included. The disagreement is therefore not a
resolution effect and is not explained here; the metric used for the reference
is not recoverable from what was recorded. Both the ladder above and the
node-angle isolation are reproducible from this script.

Second derivatives remain forbidden regardless: the source of the O(h) term is
precisely a kinked first derivative.

## Per parameter: the two classes do not share a step size

Same ladder, `n_pts = 200`, taken in every parameter D12 quantifies over
(`m` excluded — it is categorical under D5 and never differentiated).
Relative L2 deviation from `D(1e-6)`, with the entry fraction carrying it:

| `h` | `a` | `b` | `n1` | `n2` | `n3` | `n_core` | `n_clad` |
|---|---|---|---|---|---|---|---|
| 1e-2 | 1.17e-3 | 1.09e-3 | 9.68e-5 | 3.59e-5 | 3.88e-5 | 2.20e-4 | 2.18e-4 |
| 1e-3 | 9.55e-5 | 8.29e-5 | 3.21e-5 | 2.53e-7 | 1.98e-6 | 2.20e-6 | 2.18e-6 |
| 1e-4 | 3.77e-5 | 3.24e-5 | 4.83e-8 | 3.51e-8 | 3.30e-8 | 2.21e-8 | 2.17e-8 |
| 1e-5 | 5.15e-9 | 5.60e-9 | 4.84e-8 | 3.58e-8 | 3.30e-8 | 4.76e-10 | 1.06e-9 |
| 1e-7 | 5.70e-8 | 6.18e-8 | 5.28e-7 | 4.11e-7 | 3.43e-7 | 3.38e-9 | 8.87e-9 |

**`n_core` and `n_clad` are the control, and they are textbook `O(h²)`:**
2.20e-4 → 2.20e-6 → 2.21e-8, exactly ×100 per decade over three decades, with
the entry fraction going to zero and staying there. They never touch the
arc-length inversion. That they behave perfectly on the same matrix, at the same
λ, through the same assembly is what rules out the assembly, the Hankel
evaluation and the complex arithmetic as the source of the geometric stall — the
only thing left different is the node placement.

**The stall tracks how strongly a parameter moves the nodes.** `a` and `b`
rescale the boundary directly and stall at `h = 1e-4` (frac 0.05); `n1` stalls a
decade earlier at 1e-3 (frac 0.052); `n2` and `n3` reach the floor by 1e-3
already. The ordering is the ordering of geometric leverage at this design
point, not a property of the parameter's name — so it will move with the design
point and cannot be tabulated once.

**Consequence for D12.** A single `h = 3e-4` for all seven parameters is not
supportable: it lands in the kink-dominated region for `a`, `b` and `n1`
(3–4e-5 relative, ~4 digits) while `n_core` and `n_clad` would deliver 2e-8 at
the same step. The step that serves *both* classes is **`h = 1e-5`**: 5.2e-9 and
5.6e-9 on `a` and `b`, 4.8e-10 and 1.1e-9 on the material pair, and the worst
parameter anywhere in the table at that step is `n1` at 4.8e-8. That is a
proposal, not a decision — D12 is locked and re-taking it is not this study's
call. What the study establishes is that the number currently in it is measured
to be the wrong one.

## With the node set frozen (D16): the O(h) term is gone

Same script, same design point, `parameter_sweep_frozen`, at commit `247492d`.
The only change is that `M(p₀±h)` is built on the θ nodes of `p₀` instead of
re-inverting arc length on each perturbed shape.

| `h` | `b`=1.20 | `b`=0.85 | `b`=1.50 | `b`=1.25, m=6 | `a`=1.00 | `n1`=1.60 | `n2`=3.00 |
|---|---|---|---|---|---|---|---|
| 1e-2 | 1.01e-3 | 1.13e-3 | 9.50e-4 | 8.11e-4 | 1.14e-3 | 9.51e-5 | 1.4e-5 |
| 1e-3 | 1.01e-5 | 1.13e-5 | 9.50e-6 | 8.11e-6 | 1.14e-5 | 9.51e-7 | 1.4e-7 |
| 1e-4 | 1.01e-7 | 1.13e-7 | 9.50e-8 | 8.10e-8 | 1.14e-7 | 1.15e-8 | 1.1e-8 |
| 1e-5 | 1.67e-9 | 1.76e-9 | 1.66e-9 | 1.49e-9 | 1.63e-9 | 6.40e-9 | 1.1e-8 |
| 1e-7 | 1.31e-8 | 1.53e-8 | 1.30e-8 | 1.30e-8 | 1.18e-8 | 6.47e-8 | 9.7e-8 |

**Exactly ×100 per decade**, over three decades, in every parameter, until the
cancellation floor takes over. Compare the unfrozen column for `b`=1.20 in the
first table: 8.29e-5 → 3.24e-5 (a factor of 2.6) against 1.01e-5 → 1.01e-7 here.
The stall is not reduced, it is absent — and the entry fraction carrying the
deviation now behaves like a truncation term should, spread over most of the
matrix at large `h` (0.85) and falling below the counting floor entirely by
`h = 1e-4`, rather than surviving as a shrinking minority of kinked nodes.

Two design points were excluded, both on grounds rather than convenience:

- **Odd `m` away from `a = b`** violates D5's closure condition. The arc-length
  inversion returns *coincident* nodes there — at `m = 3, b = 1.20` the minimum
  θ spacing is exactly 0.0 — which makes `_der_real_3` divide by zero and
  return **NaN second derivatives with nothing raised**. The prescribed-θ
  validator added in `247492d` rejects such a set; the unfrozen path accepts it.
  This is pre-existing and outside the catalogue's legal region, but it is a
  silent-NaN path and is recorded rather than fixed here.
- **`n1` at `a = b = 1, n2 = n3 = 2`** is an exact null direction: the bracket
  is `|cos|² + |sin|² ≡ 1`, so `r ≡ rad` for every `n1` and the shape is a
  circle. `∂M/∂n1` vanishes identically and a relative deviation divides noise
  by zero. Worth knowing as a null direction in its own right.

## Consequence for D12

With D16 in force the tradeoff is the classical one — truncation above,
cancellation below — and there is a genuine plateau:

- `h = 1e-5` is the best single step: worst parameter anywhere in the table is
  `n2` at 1.1e-8, and `a`, `b` reach 1.7e-9. Roughly **8 digits**.
- `h = 1e-4` is within a decade of it (worst 1.1e-7) and sits further from the
  cancellation floor, so it has more margin below.
- The originally locked `h = 3e-4` is *also* fine now — it would give ~1e-7 —
  which is worth stating plainly: the freeze, not the step size, was the fix.

`h = 1e-5` with the floor at `~1e-8` and truncation at `1e-7` a decade above
leaves about a decade of margin on each side. That margin, rather than the best
value at one design point, is the reason to prefer it: the truncation
coefficient scales with the parameter's geometric leverage, which moves across
the catalogue.

## Note: is refinement dangerous? (recorded, not investigated)

Conventional use of this BIE formulation treats aggressive boundary refinement
as numerically risky, on the grounds that shrinking `Δs` drives the off-diagonal
Hankel arguments `k·r_ij` towards the log singularity that the diagonal terms
handle analytically. **This is received intuition and not a measured property of
this code.** It has not been demonstrated here, and nothing in the ladders above
shows it: `R` up to 74 (`n_pts = 400` on the reference circle) behaves exactly
as the coarser cases do.

It is recorded because it would matter if true — it would put a *ceiling* on
refinement, and every convergence argument in this project assumes there is
none. The condition that would trigger a proper study: an accuracy or
conditioning result that gets **worse** with increasing `R` at fixed shape, or a
`cond(M)` that grows faster than the O(n_pts) the discretisation alone explains.
Absent that, refinement is treated as safe and the binding constraint on `n_pts`
is cost, not stability.
