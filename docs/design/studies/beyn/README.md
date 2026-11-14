# Numerical performance of the contour eigensolver

Measurements on `pysie2d.beyn`: how accurate the extraction is for a given
contour, and what limits it. The solver's thresholds and defaults were set on
one test problem; this study asks which of them hold more widely.

These are study scripts: not package code. Run from the repo root:

    VECLIB_MAXIMUM_THREADS=1 uv run --frozen python docs/design/studies/beyn/s1_rank_rule.py

Outputs are in `results/`. Every script measures the extraction step alone, on
problems whose eigenvalues are known (`common.py`):

| Problem | Reference | Size |
|---|---|---|
| Synthetic pencil, linear and nonlinear: 5 poles inside the box, 4 just outside, 71 far away; random non-unitary eigenvector matrix | placed by hand | 80 |
| Pencil with unitary factors, nonlinear (the one in `tests/test_beyn.py`) | placed by hand | 40 |
| Circle, radius 200 nm, index 3, TE: one mode in a 25 × 25 nm box | analytic Mie poles | 128 |
| Same circle, TE and TM: a 600 × 49 nm box holding 11 and 9 modes | analytic Mie poles | 128 |
| Same box, absorbing core (index 3 + 0.05i), TE | analytic Mie poles | 128 |

At `n_pts = 64` the boundary discretisation of the circle is at round-off, so
the error measured against the Mie poles is the contour's.

## Findings

### 1. The rank rule was the accuracy floor, more often than the contour (S1)

The rule shipped up to v1.3.0 kept the singular values of `A₀` down to the last
drop of 1e3, and only above 1e-8 of the largest. A pole just outside the
contour enters `A₀` with a weight that falls as the contour is refined. Once
that weight passed below 1e-8 the pole was dropped, while still far above
round-off, and it then contaminated the eigenvalues inside at about its own
weight. The result got worse as nodes were added:

| Nodes | 64 | 96 | 128 | 192 | 256 |
|---|---|---|---|---|---|
| Synthetic linear pencil, shipped rule | 7.3e-14 | 2.1e-10 | 1.5e-8 | 6.3e-12 | 3.2e-8 |
| Same moments, new rule | 7.3e-14 | 5.2e-14 | 2.7e-14 | 3.6e-14 | 1.9e-14 |

The new rule traces the round-off plateau up from the smallest singular value
(each value within a factor `PLATEAU_GAP = 20` of the one below) and keeps
everything above it. Over 534 runs (seven problems, ten contour sizes, five
probe seeds, two probe counts):

| Rule | Runs with a false eigenvalue inside | Runs a digit worse than the best rule |
|---|---|---|
| shipped (cliff, floor 1e-8) | 0 | 142 |
| plateau, gap 5 | 2 | 1 |
| plateau, gap 10 | 0 | 2 |
| **plateau, gap 20** | 0 | 13 |
| plateau, gap 30 | 0 | 15 |
| plateau, gap 100 | 0 | 28 |

A false mode is the worse error, so the gap sits four times above the value at
which one first appeared.

The same effect explains the accuracy ladder quoted for the one-mode circle
box. With the rank cut out of the way that box no longer needs the default 48
nodes:

| Nodes per side | 4 | 6 | 12 | 24 |
|---|---|---|---|---|
| floor 1e-8, as `QNMSolver.modes` passes it today | 7.3e-12 | 3.5e-9 | 1.1e-11 | 1.3e-13 |
| floor at the plateau (`rank_tol = 0`) | 7.3e-12 | 9.0e-13 | 2.7e-13 | 1.3e-13 |

(Error in nm against the Mie pole, `n_pts = 40`. Under the shipped rule the
6-node entry was 7.0e-6.)

The estimate "round-off = ε/cancellation" holds on the circle (smallest
singular value at 0.25–0.73 of it once converged) and is 10–17 times too low on
the synthetic pencils, whose entries span many decades. That is why the floor
is read off the spectrum and the estimate is only a backstop.

### 2. An ellipse with the trapezoid rule needs about half the nodes (S2)

Same node count, the ellipse inscribed in the box against Gauss–Legendre on
its four sides. Worst and median error over the modes each contour encloses:

| Problem | Nodes | Rectangle, worst / median | Ellipse, worst / median |
|---|---|---|---|
| Synthetic linear | 16 | 1.5e-7 / 8.9e-8 | 2.5e-10 / 9.9e-11 |
| Synthetic nonlinear | 48 | no rank | 2.2e-12 / 8.7e-13 |
| Circle TE, wide box | 64 | 6.3 / 5.8e-11 | 1.6e-2 / 3.0e-12 |
| Circle TE, wide box | 128 | 1.1e-1 / 1.3e-12 | 4.9e-6 / 1.4e-12 |
| Circle TM, wide box | 64 | 1.6e-7 / 6.3e-11 | 9.9e-12 / 3.1e-12 |
| Circle TE, absorbing core | 128 | 2.5e-3 / 1.1e-12 | 4.2e-9 / 1.3e-12 |

The ellipse at `n` nodes is about as accurate as the rectangle at `2n`.

The two contours do not enclose the same region. The inscribed ellipse leaves
the corners out: of the 11 TE modes in the wide box it holds 9. The ellipse
through the corners was tried and dropped: it reaches outside the rectangle,
takes in 15–17 modes there, and saturated the 20-column probe.

### 3. The adaptive stop is reliable, and costs one doubling (S3)

The trapezoid rule nests, so `beyn_modes_adaptive` doubles the node count
reusing every solve, and stops when no eigenvalue moves by more than `tol`.

- The reported change was an upper bound on the true error in 13 of 14 runs
  and within 1 % of it in the other (ratios 0.99 to 7.8e3).
- No run reported a false or missing eigenvalue.
- The stop is tested on the previous level, so the final contour is one
  doubling finer than needed. Its node count is between 1× and 2× that of the
  smallest fixed rectangle that meets the same tolerance, chosen with
  hindsight. What it buys is that the node count is no longer a guess.

### 4. The mode that lags in the wide box is short of moments, not nodes (S4)

On the 11-mode TE box one mode, `n = 0`, converges far behind the rest. The
eigenvectors of a circle of a given order span two dimensions, and two moments
can separate only two poles there. A third pole of the same order cannot be
given a direction and leaks onto its neighbours. Using four moments in block
Hankel form (Beyn 2012, section 5), on the same solves:

| Nodes | 48 | 64 | 96 | 128 |
|---|---|---|---|---|
| `n = 0` error, two moments (shipped) | 21 | 6.3 | 1.0 | 1.1e-1 |
| `n = 0` error, four moments | 4.5e-8 | 1.7e-11 | 6.8e-13 | 3.2e-13 |

Six moments add nothing over four. This is not implemented in the package.

### 5. Proof of concept: interpolate `M`, integrate the interpolant (S5)

The node count is set by how close the poles of `M⁻¹` come to the contour, but
`M` itself is smooth across the box. S5 assembles `M` at `n` nodes of the
ellipse, fits a polynomial in λ through those matrices (Chebyshev basis on the
ellipse's focal segment), and runs a 512-node contour on the polynomial, which
costs a factorisation per node and no assembly.

Circle TE, wide box (9 modes in the ellipse, including the lagging `n = 0`):

| Assemblies | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|
| Fit error of the polynomial (relative) | 3.1e-5 | 6.1e-9 | 6.6e-13 | 1.6e-15 | 1.3e-15 |
| Worst mode, direct contour | no rank | 2.3 | 8.7e-1 | 1.6e-1 | 1.6e-2 |
| Worst mode, interpolated | 4.3e-3 | 1.0e-6 | 1.0e-10 | 9.6e-12 | 8.3e-12 |

The direct ellipse needs 256 nodes to pass what 32 assemblies reach this way
(S2: 2.5e-11 at 256). The fine contour also removes the lag of finding 4, since
at 512 nodes the offending pole's weight is negligible.

At `n_pts = 200` on one thread an assembly takes 75.5 ms, an evaluation of the
polynomial 0.6 ms and a factorisation with 20 probe solves 1.3 ms. 256 direct
nodes cost 19.7 s; 32 assemblies plus 512 interpolated nodes cost 3.7 s, a
factor 5.3.

Where it does not pay, and what is unknown:

- A small box. On the one-mode box 12 assemblies give 5.2e-12 interpolated
  and 1.2e-11 direct: the direct contour was already at the assembly count the
  fit needs.
- The eigenvalue error tracks the fit error, so the assembly count is set by
  how fast `M` varies across the box, about the box width in units of the
  wavelength times the particle's optical size. One geometry was measured.
- The absorbing-core box stalls at 6.0e-10 on its worst mode from 32
  assemblies on; the direct contour reaches 1.4e-11 at 256. Not diagnosed.
- Memory: the coefficients are `n` matrices (82 MB for 32 at N = 400). At
  N = 4000 that is 8 GB.
- The timing is one thread. The package threads the assemblies, which narrows
  the gap; the interpolated nodes would thread too, and were not.
- No test of a substrate, where `M` has more structure in λ.

### 6. The interpolation route as shipped (S6)

`beyn_modes_adaptive(interpolate=True)` puts S5 behind the adaptive stop: the
samples nest under doubling like the direct nodes, the polynomial is refitted
at each level in the ellipse's Faber basis (scaled Chebyshev polynomials; the
interpolation matrix has condition 2.3 to 2.6 on the wide box), and the
contour runs on 512 nodes of it. Both routes at `tol = 1e-9`, assemblies and
worst error on the ellipse, no missing or spurious mode in any row:

| Problem | Direct | Interpolated |
|---|---|---|
| Synthetic, linear | 32, 3.2e-14 | 32, 8.3e-14 |
| Synthetic, nonlinear | 128, 8.4e-13 | 32, 3.9e-12 |
| Unitary factors, nonlinear | 32, 2.5e-14 | 32, 5.6e-13 |
| Circle TE, one mode | 32, 6.1e-13 | 32, 1.1e-13 |
| Circle TE, wide box | 512, 4.8e-12 | 64, 2.0e-11 |
| Circle TM, wide box | 128, 2.4e-10 | 64, 1.9e-12 |
| Circle TE, lossy core | 512, 6.0e-10 | 64, 6.0e-10 |

The lossy core's 6.0e-10 is the same in both routes, so the stall S5 reported
is not the interpolation's. It sits under the tolerance, and was not
diagnosed further.

One trap, found while writing the tests: with a pole 0.05 outside the ellipse
and 1024 nodes on the polynomial, the pole's weight falls below round-off, but
a direction at 5e-13 is left in `A₀`. On `M` it sits within `PLATEAU_GAP` of
the plateau and is dropped. On the polynomial the plateau is lower (2e-14),
so it clears the gap and one level reports a noise eigenvalue just inside the
ellipse. It lands somewhere else at every level (2.9795, 2.9995, 2.9973), and
the stopping test rejects it. That is why the route is offered only through
the adaptive driver.

## What changed in the package

- `_detect_rank` and `beyn_poles`: the plateau rule; `rank_tol` now defaults
  to 0 in `beyn_poles` and `beyn_modes` and is an extra floor.
- `ellipse_contour_quad`, `beyn_poles(contour="ellipse")`,
  `beyn_modes_adaptive`.
- `beyn_modes_adaptive(interpolate=True)`, `ellipse_faber_basis`, and
  `beyn_modes(contour="ellipse")` (finding 6).
- `QNMSolver.modes` takes `contour="ellipse"`, `adaptive=True` and
  `interpolate=True`, opt-in, with `QNMResult.contour` deciding what
  `edge_margin` measures. The defaults are unchanged: `QNMSolver.modes` and
  `ClusterEigenEps.eigenpermittivities` still pass `rank_tol = 1e-8`, which
  keeps part of the old behaviour (first table row under finding 1).

## Not covered

- No shape but a circle, no substrate, no cluster. Whether the plateau has the
  same shape there is unmeasured.
- The gap of 20 sits between 5 (false modes) and the weakest direction that
  mattered, 106 above the plateau. Margins of 4 and 5, on seven problems.
- Finding 4 is an argument from symmetry, tested on one box. A shape without
  that symmetry should not show the lag; this was not run.
- Probe choice, refinement cost, and an independent extraction method are not
  part of this pass.
- Timings. Every comparison is in node counts, which is the number of matrix
  assemblies.
