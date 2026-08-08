# Finding quasi-normal modes

A quasi-normal mode (QNM) is a source-free solution of the scattering problem: a
complex wavelength where the boundary-integral operator `M(λ)` is singular.
`QNMSolver` finds **every** mode inside a rectangle of the complex λ-plane by
contour integration (Beyn's method) — no initial guess, no scan, no root
tracking.

```python
from pysie2d import Geometry, Material, QNMSolver

geom = Geometry.gielis(rad=200.0, n_pts=200, m=0)   # circle, radius 200 nm
mat = Material(n_core=3.0, n_clad=1.0, pol=2)       # pol=2 → TE
res = QNMSolver(geom, mat).modes(745 + 2j, 775 + 15j)

print(res.wavelengths)       # [760.32605922+7.77015403j 760.32605922+7.77015403j]
print(res.quality_factors)   # [48.92606095 48.92606095]
print(res.multiplicity)      # [2 2]
```

The cost is `4·n_quad_per_side` complex assemblies, **independent of how many
modes are inside**. Finding ten modes in one box costs the same as finding one.

Everything below assumes the running example: a circular cylinder of radius
200 nm, `n_core = 3.0` in vacuum, discretised at `n_pts = 200`.
`examples/qnm_spectrum.py` produces the whole spectrum figure from these pieces.

---

## 1. How to draw a box

The rectangle is given by two opposite corners, `z_lo` (bottom-left) and `z_hi`
(top-right), in **vacuum nanometres** — the same coordinate the modes come back
in.

**Both bounds are asserted, not merely documented:**

- **`Im λ > 0`.** Under the `exp(-iωt)` time convention a decaying mode has
  `Im ω < 0`, hence `Im λ > 0`. A box reaching down to `Im λ = 0` is searching
  for growing modes and raises.
- **`Re λ > 0`.** This keeps every Hankel argument off the `H^(1)` branch cut on
  the negative real axis, which is what makes `M(λ)` holomorphic on the
  rectangle — and holomorphy is the premise of the contour argument, not a
  detail.

Three things that catch people out:

**Do not expect conjugate pairs.** The reality condition here is `λ → −λ̄`, not
`λ → λ̄`, so a mode's mirror partner sits at *negative* `Re λ`, outside the
physical region entirely. Carrying real-eigenvalue intuition into this
non-Hermitian problem is the natural mistake.

**A box in `λ` is not a box in the size parameter `x`.** `λ = 2π·n_clad·a/x` is
a Möbius map: it does not carry corners to corners. `res.size_parameters` will
report your modes in `x`, but the *search* happens in λ, and any argument about
which poles are inside must be made in the coordinate the box was drawn in.

**Isolation in `Im λ` matters far more than width in `Re λ`.** This is the single
most useful rule of thumb. Take the TE `n=0` mode at `530.83 + 26.38j`. Its
neighbours in `Re λ` are close — TE `n=5` at `505.68 + 0.42j` and TE `n=2` at
`550.47 + 20.24j`, both roughly 20 nm away — so a box drawn generously in
`Re λ` is fine, but a box drawn generously *downward* in `Im λ` swallows the
`Q = 598` mode and the answer is no longer the single mode you asked for. The
working box is `Re λ ∈ [520, 545]`, `Im λ ∈ [15, 40]`: wide in `Re`, deliberately
floored well above `Im λ = 0.42`.

If you do not know where the modes are, the honest starting point is a coarse
box plus the diagnostics in §2 — or, for a circle, the analytic table in
`pysie2d.reference.mie.qnm_wavelengths`, which is the independent anchor this
feature was validated against.

`n_probe` (default 12) must exceed the number of modes inside the box, counting
rank leaked from poles just outside it. Exceeding it raises rather than silently
truncating.

### One wide box, or several narrow ones?

Both work, and the wide box is usually the better first move. **The cost is
`4·n_quad_per_side` assemblies whatever is inside the contour**, so finding
eleven modes costs no more than finding one — a wide box is not the expensive
option, it is the cheap one. `examples/qnm_wide_window.py` takes a single
rectangle spanning `Re λ ∈ [500, 1100]`, `Im λ ∈ [1, 50]` and returns all eleven
TE modes inside it from one call, agreeing with the seven hand-placed boxes of
`examples/qnm_spectrum.py` to ~1e-12 nm.

Two things change as the box grows:

- **Raise `n_probe`.** It must exceed the mode count *with multiplicity*, plus
  whatever leaks in from just outside. That example needs 11 + 3 and uses 20.
- **Raise `n_quad_per_side`.** This is the one regime where the quadrature, not
  `n_pts`, sets the accuracy — the contour is long, and nearby outside poles sit
  close relative to its size. On that box the worst-resolved mode's error falls
  `6.86 → 0.93 → 0.12 nm` at `16 → 24 → 32` nodes per side, while the other ten
  are already converged at 16.

You do not have to pay for the extra nodes: a coarse wide contour followed by
`refine()` (§3) is an equally valid strategy, and is exactly the case
refinement exists for. Where narrow boxes still earn their place is when you
want one specific mode and want the mode count itself to be the assertion.

---

## 2. How to tell the answer is trustworthy

Four diagnostics come back with every result. They describe **the quality of the
contour**, and they remain the first thing to check whether or not you go on to
refine.

| Field | Reads well when | What a bad value means |
|---|---|---|
| `edge_margin` (per mode) | ≳ 0.1 | Distance to the nearest box edge, as a fraction of the shorter side. Near zero, the contour passes close to the singularity and the quadrature may be unreliable. It is a **reason to check, not a verdict** — see the caveat below. |
| `sigma_ratio` (per mode) | ~1e-15 | `σ_min/σ_max` of `M` at the mode. The singularity measure itself, dimensionless on purpose — an absolute `σ_min` means nothing since it carries the scale of the operator. Large means you have not landed on a pole. |
| `cancellation` | small | How completely the contour integral cancelled. |
| `max_gap` / `sv_ratio` | large gap | The singular-value spectrum of the Beyn moment matrix and the largest ratio between consecutive values. This is where the rank decision lives; a clear gap means the mode count is unambiguous. |

On the well-drawn TE `n=0` box (`520+15j` to `545+40j`) those read
`edge_margin = 0.418`, `sigma_ratio = 1.5e-14`, `cancellation = 3.1e-02`,
`max_gap = 1.6e+07` — a clean extraction. Pushing the left edge in to 530 nm,
~0.5 nm from the pole, drops `edge_margin` to **0.018** and degrades
`sigma_ratio` to **3.0e-07**. That is exactly the regime the diagnostics exist to
warn about.

**A low `edge_margin` is not by itself a failure.** The two diagnostics answer
different questions, and `sigma_ratio` is the one that says whether you are on a
pole. In `examples/qnm_wide_window.py` the TE `n=4` mode sits 0.67 nm above the
box floor and reports `edge_margin = 0.014` — nominally alarming — yet
`sigma_ratio = 3.3e-15`, and the value agrees with its narrow-box counterpart to
1e-12 nm. It is genuinely resolved. Read `edge_margin` as a prompt to check
`sigma_ratio`, and if you are still unsure, `refine()` and see whether the mode
moves (§3).

### `rank` may legitimately exceed the mode count

`res.rank` is the rank detected in the moment matrix, and it can be **larger**
than `res.n_modes`. A pole just *outside* the contour leaks a rank direction in
through imperfect quadrature cancellation; its eigenvalue is then correctly
discarded by the in-contour filter. The well-drawn TE `n=0` box returns
`rank = 3` with `n_modes = 1`, because TE `n=5` and TE `n=2` are both just
outside. **This is not a bug** — do not force the two to agree.

### Degenerate modes come back twice

Every `n ≥ 1` mode of a circle is doubly degenerate through `exp(±inθ)`; only
`n = 0` is simple. Degenerate partners are reported as **separate, numerically
equal entries** with `multiplicity = 2`, never collapsed — the pair carries two
independent mode vectors, and merging them would make the count disagree with
the analytic one. The TE `n=3` example at the top of this page returns two rows
for that reason.

---

## 3. What `refine()` is and is not for

`QNMResult.refine()` polishes each mode with bordered Newton and returns a
**new** result, with the `converged` flag and `cond_jacobian` filled in.

**It is insurance, not accuracy, and it is opt-in for that reason.** On the
well-drawn TE `n=0` box:

```text
raw      λ = 530.455508 + 26.138381j   Re-λ error vs analytic = 0.3766 nm
refined  λ = 530.455508 + 26.138381j   Re-λ error vs analytic = 0.3766 nm
```

The two agree to every digit printed, and the error against the analytic pole is
unchanged. What refinement *did* do is drive `sigma_ratio` from `1.5e-14` to
`1.3e-17` and set `converged = True` — it confirmed the pole, it did not improve
it.

Where it earns its place is the coarse or badly placed contour. Using the
clipping box from §2 (`530+15j` to `560+40j`) at a coarse
`n_quad_per_side = 4`:

```text
raw      λ = 530.455462 + 26.138066j   edge_margin = 0.018,  sigma_ratio = 3.0e-07
refined  λ = 530.455508 + 26.138381j   converged = True,     sigma_ratio = 1.3e-17
```

The refined value is **bit-identical** to the well-drawn box's refined value —
the difference is exactly zero. That is the recovery path: a contour too coarse
or too badly placed to locate the pole cleanly still gets you there.

### On a degenerate pole it no-ops, and says so

Bordered Newton assumes a *simple* eigenvalue — a one-dimensional null space —
and a degenerate pole has a two-dimensional one, so its Jacobian is singular in
exact arithmetic. Rather than return a polished-looking number, `refine()` keeps
the contour estimate and records why in `cond_jacobian`. **Nothing raises.**

On the TE `n=3` pair: `cond_jacobian ≈ 4.1e15` and `1.1e15`, against
`DEGENERATE_COND = 1e12`, and `converged = False` for both. Compare against
`pysie2d.qnm.DEGENERATE_COND` rather than hard-coding the threshold — that is
why it is exported.

Read `cond_jacobian` alongside `multiplicity`: the two detect degeneracy by
independent means (eigenvalue spacing versus the conditioning of the actual
linear algebra), so a disagreement is informative. A `multiplicity` of 1 with a
singular Jacobian says a degenerate partner is missing — most likely clipped by
the box.

### Reading `converged`

Straight out of `.modes()`, `converged` is **all `False` and `cond_jacobian` all
`NaN`**. That means *no refinement was attempted*, not that anything failed.
`refine()` is the only thing that sets them.

---

## 4. What the accuracy actually is — and which knob to turn

At the anchor above, the extracted mode is off by **0.38 nm in `Re λ`** against
the analytic Mie pole. **All** of that is discretisation error in `n_pts`; none
of it is extraction:

- The contour integral is converged to ~1e-8 nm from 6 quadrature nodes per side
  upward. Raising `n_quad_per_side` past the default buys nothing.
- Refinement moves the answer by ~1e-8 nm, as shown above.
- Convergence is **first order in `n_pts`**, in both `Re λ` and `Im λ`.

**So if a mode is not accurate enough, raise `n_pts`.** Refining, or adding
contour nodes, is tuning the wrong knob — a common and expensive mistake here.

---

## 5. A worked example

Drawing a box around the TE `n=3` mode near `760.69 + 7.95j`. Its nearest TE
neighbours are at 690.51 and 1035.09 nm, so placement is forgiving in `Re λ`;
`Im λ ∈ [2, 15]` clears the mode comfortably on both sides.

```python
import numpy as np
from pysie2d import Geometry, Material, QNMSolver
from pysie2d.qnm import DEGENERATE_COND

geom = Geometry.gielis(rad=200.0, n_pts=200, m=0)
mat = Material(n_core=3.0, n_clad=1.0, pol=2)

res = QNMSolver(geom, mat).modes(745 + 2j, 775 + 15j, n_quad_per_side=6)
ref = res.refine()

print(f"modes found : {res.n_modes}  (rank {res.rank})")
print(f"wavelengths : {res.wavelengths}")
print(f"Q           : {res.quality_factors}")
print(f"multiplicity: {res.multiplicity}")
print(f"edge_margin : {res.edge_margin}")
print(f"sigma_ratio : {res.sigma_ratio}")
print(f"degenerate  : {ref.cond_jacobian > DEGENERATE_COND}")
```

Which prints:

```text
modes found : 2  (rank 2)
wavelengths : [760.32605922+7.77015403j 760.32605922+7.77015403j]
Q           : [48.92606095 48.92606095]
multiplicity: [2 2]
edge_margin : [0.443858 0.443858]
sigma_ratio : [2.26540737e-16 8.62101179e-16]
degenerate  : [ True  True]
```

**Is this trustworthy?** Yes, and here is the reading:

- `edge_margin = 0.44` — the mode sits nearly in the middle of the box; nothing
  is being clipped.
- `sigma_ratio ~ 1e-16` — `M` is singular to machine precision there. It is a
  genuine pole, not a numerical artefact.
- `rank = 2` equals `n_modes = 2`, so nothing leaked in from outside.
- `multiplicity = 2` and `cond_jacobian` above `DEGENERATE_COND` **agree**: this
  is the doubly degenerate `exp(±3iθ)` pair, exactly as theory requires for
  `n ≥ 1`. `converged` is `False` for both, which here means "degenerate,
  skipped by design", not "failed".
- Against the analytic Mie pole at `760.68665 + 7.94771j`, the answer is off by
  0.36 nm in `Re λ` — first-order discretisation at `n_pts = 200`, as expected.
  Raising `n_pts` is what shrinks it.

---

## 6. Limitations

The release ships with these, and they are design boundaries rather than bugs:

- **Homogeneous background only.** Slab or image backgrounds break the holomorphy
  argument the contour method rests on, which is precisely why they are out of
  scope.
- **Mode fields are not normalised.** `res.vectors` are raw `φ`/`χ` columns in
  the driven solver's layout, scaled to unit columns. A proper QNM norm is a
  separate piece of physics and is not implemented.
- **Refinement recovers a contour; it does not beat discretisation.** At a
  well-drawn box, accuracy is set by `n_pts` (§4).
- **Degenerate poles are found and counted, but not refined.** Since every
  `n ≥ 1` circle mode is degenerate, this is the common case, not the corner
  case.
- **`dM/dλ` is exact only for non-dispersive materials**, which makes `refine()`
  non-dispersive-only too. Extraction itself is unaffected.
- **The analytic anchor has a high-`Q` ceiling** of roughly `Q ≈ 1.5e4`: the
  seeder behind `reference.mie.qnm_size_parameters` cannot see a mode narrower
  than its grid's lowest row. This bounds the *validation* coverage, not the
  solver.

---

## See also

- `docs/conventions.md` §2 (vacuum wavelengths) and §8 (the QNM half-plane) —
  the normative statements.
- `examples/qnm_spectrum.py` — the spectrum figure, and seven worked boxes.
- `examples/qnm_wide_window.py` — the opposite strategy: one wide contour
  returning eleven modes in a single call, with the full diagnostic table
  printed and read.
- `pysie2d.reference.mie.qnm_wavelengths` — the analytic Mie poles of a circular
  cylinder, the independent anchor for everything above.
