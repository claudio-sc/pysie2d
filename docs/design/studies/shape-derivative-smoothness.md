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
