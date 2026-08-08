# Performance — where the time goes, and what to do about it

Measured 2026-07-31 on Apple silicon, 8 cores, numpy 2.x on Accelerate,
Python 3.12, at commit `684ecd0` (branch `beyn-port`). Every number below was
measured unless labelled *(estimate)*. This document is about the **package**,
not about the Beyn port; `docs/design/beyn-port-status.md` links here rather
than repeating it.

§§1–4 were measured at **`nn ≤ 400`**. §5 re-measures at `nn` up to 3200 and
projects to 10 000; read it before sizing anything large, because the binding
constraint changes from time to **memory** well before the time model does.

## 1. The headline

**This is a special-function-bound code, not a linear-algebra-bound one.**

At complex λ — the QNM case, and the one that is slow —
`scipy.special.hankel1` is **97.9 % of one `BIESolver.assemble`** and
**95.5 % of an entire `QNMSolver.modes()` call**. Dense `O(n³)` linear algebra
is **2.4 %**. Any optimisation that does not touch Hankel evaluation, or the
number of times it happens, is optimising 2 % of the runtime.

This is not an artefact of the small `nn` it was measured at. §5.2 puts the
crossover where dense linear algebra overtakes Hankel evaluation at
**`nn ≈ 20 700`**, so the statement holds across the whole large-system range.

## 2. The profile

### Assembly, real vs complex wavenumber

| `nn` | real λ | complex λ | ratio |
|---|---|---|---|
| 200 | 3.95 ms | 93.7 ms | 23.7× |
| 300 | 9.08 ms | 209.6 ms | 23.1× |
| 400 | 16.7 ms | 375–403 ms | 22.4× |

The `_real_if_real` demotion in `kernels.py` — which routes real wavenumbers to
the Cephes branch of `hank0`/`hank1` — is worth **23×**, not the 11–13× its
docstring claims for the scalar function. Non-negotiable #1 says never simplify
the complex path away; this table is why the *real* path exists alongside it.

### Split of one assembly

| `nn` | mode | total | Hankel (4 calls) | scatter-fill (8 assignments) |
|---|---|---|---|---|
| 200 | real | 3.95 ms | 2.06 ms (52.1 %) | 0.85 ms (21.6 %) |
| 200 | complex | 93.9 ms | 91.9 ms (**97.9 %**) | 0.84 ms (0.9 %) |
| 300 | complex | 209.4 ms | 204.9 ms (**97.9 %**) | 1.90 ms (0.9 %) |
| 400 | complex | 402.9 ms | 364.0 ms (**90.3 %**) | 4.07 ms (1.0 %) |

`np.zeros` and `np.triu_indices` together are under 0.3 % at every size.

### Dense linear algebra, complex128, N = 2·nn

| N | `solve` (1 rhs) | `lu_factor` | `lu_solve` (12 rhs) | `svd(compute_uv=False)` |
|---|---|---|---|---|
| 400 | 2.36 ms | 2.33 ms | 0.21 ms | 37.1 ms |
| 600 | 6.83 ms | 8.39 ms | 0.38 ms | 101.8 ms |
| 800 | 14.2 ms | 13.9 ms | 0.68 ms | 246.8 ms |

### `QNMSolver.modes`, TE `n=0` anchor box, `nn = 200`, `n_quad_per_side = 12`

```
modes()                    4.980 s
  contour_moments (48)     4.635 s   (93.1 %)
    assemble               4.524 s   (97.6 % of contour_moments; 94.2 ms/node)
    LU + solve + accumulate  0.111 s ( 2.4 %)
  _sigma_ratio (1 mode)    0.133 s   ( 2.7 %) = 94.2 ms assemble + 38.1 ms SVD
n_quad_per_side = 6        2.530 s
```

**Correction to spec §6.3.** The full SVD in `qnm._sigma_ratio` is 2.7 % of a
`modes()` call, and its own dominant term is *another assembly* (94 ms), not the
SVD (38 ms). The spec's "497 ms vs 2–6 ms" was measured at a larger `nn`; at the
fixture resolution, inverse iteration would save 38 ms out of 4980. Correctly
parked in the status doc's §4.6, and smaller than it looks even there.

### The driven paths are not bottlenecks at all

`scatter` is 6.0 ms at `nn = 200` and 30 ms at `nn = 400`.
`relative_ldos_map` covers 14 400 source points at `nn = 300` in **0.80 s**,
because it LU-factorises once — the one structural win named in `CLAUDE.md`.

## 3. What to do — ranked by risk-adjusted return

### 3.1 Thread the contour loop — *accepted, first post-merge item*

**`scipy.special.hankel1` releases the GIL.** Measured:

| workers | 16 × 19 900 complex `hankel1` | 48 full assemblies (`nn = 200`) |
|---|---|---|
| serial | 218.3 ms | 4.474 s |
| 2 | 110.5 ms (1.97×) | 2.287 s (1.96×) |
| 4 | 58.2 ms (3.75×) | 1.227 s (3.65×) |
| 8 | 46.7 ms (4.67×) | **0.892 s (5.02×)** |

A `concurrent.futures.ThreadPoolExecutor` over the contour nodes in
`beyn.contour_moments` therefore gives **5.02× on 93 % of `modes()`** — a 4.98 s
call becomes roughly **1.4 s** *(estimate, composing measured parts)*. Stdlib
only, no new dependency, no pickling, shared memory. Peak memory at `nn = 400`
with 8 workers is ~200 MB *(estimate)*.

**Shared memory is also the catch, and §5.1 measures it: 8 workers takes peak
RSS from 3.56× to 28.2× the matrix.** At `nn = 400` that is the ~200 MB above
and nobody notices; at `nn = 4000` it is 29 GB and the run dies. The worker
count must therefore be **derived from a memory budget, not fixed at 8** — see
§5.5 C. The 5× itself reproduces on newer silicon (§5.1).

**It changes no number the solver produces** — the same calls in the same order
per node, merely concurrent. That is what makes it the lowest-risk change
available.

Decided: threading goes **inside `contour_moments`**, not in a caller-side loop
over search boxes. It helps the interactive one-box-at-a-time pattern and the
sweep pattern alike, and an outer loop can still be layered on top.

Two things to get right when building it:

- The LU-failure skip (`try/except (LinAlgError, ValueError)` around
  `lu_factor`) must survive per node, and `n_failed` must be accumulated
  thread-safely — return per-node status from the worker rather than mutating a
  counter.
- `warnings.warn` from a worker thread does not reliably reach the caller;
  collect and re-emit in the parent.

### 3.2 Structured Hankel evaluation — *parked, not rejected*

The code never evaluates Hankel functions at arbitrary complex points. It
evaluates them on `(one complex scalar k) × (a fixed real distance array r)` — a
single ray. Splitting the log branch point off analytically,

```
H₀^{(1)}(k r) = [1 + (2i/π)(ln(k/2)+γ)]·P(r²) + (2i/π)·ln(r)·P(r²) + (2i/π)·Q(r²)
  with  P(s) = J₀(k√s),  Q(s) = (π/2)Y₀(k√s) − (ln(k√s/2)+γ)J₀(k√s)
```

`P` and `Q` are **entire in `s = r²`**, so a Chebyshev interpolant converges
super-geometrically and the only branch evaluation left is one scalar `ln(k/2)`
— which `conventions.md` §8's `Re λ > 0` assertion already keeps on the
principal branch. Measured on the fixture (19 900 pairs, `rad = 200`,
`nn = 200`, λ = 530+26j), sampling `P`/`Q` from scipy at the Chebyshev nodes:

| degree | max rel. error vs `hankel1` | time | vs scipy Amos (22.9 ms) |
|---|---|---|---|
| 16 | 4.6e-15 | 1.39 ms | 16× |
| 20 | 7.0e-15 | 1.72 ms | **13×** |
| 40 | 2.5e-14 | 3.49 ms | 6.6× |

Degree scales with the **size parameter**, not with `nn`, so it improves
relatively as `nn` grows; at the core wavenumber (`nc = 3`) degree 20 still gives
8.0e-15. It multiplies with §3.1, so nothing is lost by deferring it.

**Why it is parked.** It is a from-scratch special-function implementation. It
changes every number the package produces at the 1e-15 level, needs its own
independent anchor under `CLAUDE.md` non-negotiable #3, and needs coverage across
the whole `(size parameter, n_core)` range the package claims. The probe covered
`H₀` on one circle and never touched the log-singularity diagonal terms or `H₁`
(which admits the same split plus an explicit `1/z²` term, *unmeasured*). Treat
it as a feature with a validation plan, not as an optimisation.

### 3.3 Rejected, with measurements

| Idea | Measured effect | Why |
|---|---|---|
| **Migrate to JAX** | **~1.02× ceiling** | See §4 |
| `vmap` over contour nodes | 1.0× on the dominant term | The 97.6 % that matters is a callback; `vmap` over it is either the loop you already have or one big scipy call |
| Batch Hankel across all contour nodes into one scipy call | 1.0× | Per-element cost is **flat**: 751 ns at n=100, 687 ns at n=955 200. There is no per-call overhead to amortise |
| `multiprocessing` instead of threads | ≤ threads, plus pickling | The hot function is already nogil, so processes add cost for nothing |
| Rewrite `kernels.py` fill functionally | −16 % best case | Fill is 0.9 % of a complex assembly |

## 4. The JAX evaluation — rejected 2026-07-31, re-affirmed 2026-08-08

Scoped seriously and rejected on measurement, not taste. Recorded so it is not
re-opened without new information. **Re-opened 2026-08-08** on the specific
question of large systems (`1000 < nn < 10000`), where a bigger working set and
a possible GPU path are the natural counter-arguments. The answer got *stronger*,
not weaker — §4.1.

**The blocker.** JAX has no Hankel function for complex argument, and no
`j0`/`y0`/`j1`/`y1` either; `jax.scipy.special`'s only Bessel export is
`bessel_jn` — first kind, **real** argument, integer order. Since complex
wavenumbers are non-negotiable #1, a JAX port must call back into the same Amos
routine at the same speed. Measured `jax.pure_callback` overhead under `jit`:
**1.5 %**. So the ceiling on migrating the QNM path is **~1.02×**, against the
5.02× that ten lines of stdlib threading already deliver.

Secondary findings, each independently disqualifying:

- **`pure_callback` does not support JVP** (`ValueError`), so autodiff through
  the assembly requires hand-registering `H₀' = −H₁` and `H₁' = H₀ − H₁/z` via
  `custom_jvp` — the same mathematical content as spec §6.1's identities. JAX
  would not have saved that derivation. It would also cost the same to evaluate:
  a JVP calls the same two Hankels the assembly already computes.
- **Holomorphic differentiation does work**, for the record: `jax.grad` raises on
  complex output unless `holomorphic=True`, and the right call for matrix output
  is `jax.jvp(M, (λ,), (1+0j,))`, verified against central difference to 4e-10.
  The machinery is the right shape; it has nothing to run on.
- **float32 by default.** `jax_enable_x64` is process-global and must be set
  before any array is created — a library has no business setting it in its
  host's process. And with x64 on, complex128 `solve` at N = 800 is 15.6 ms vs
  8.0 ms in complex64, forfeiting the usual advantage.
- **No bit-reproducibility.** XLA fuses and reassociates; there is no
  `np.array_equal` guarantee across `jit`/eager, backends, or versions. The
  derivative work's bit-identity acceptance test would be unachievable.
- **Dense linear algebra is slower than Accelerate here**: `solve` 0.83×/0.92×/
  1.03× and `svd` 0.77×/0.82×/0.85× at N = 400/600/800. And
  `jax.default_backend()` is `cpu` on Apple silicon — there is no GPU path on
  the development machine.
- **Dependency cost.** `jaxlib` alone is a 59.6 MiB wheel with a platform
  matrix, in a package whose stated dependencies are numpy + scipy and where
  `CLAUDE.md` calls a runtime dependency a scope decision.

**What the evaluation did confirm, and is worth keeping true:** `beyn.py` is the
most portable code in the repo — EM-free, driven by
`Callable[[complex], np.ndarray]`, naming no Hankel, wavelength, or material.
Keep it that way for its own sake. The single-conversion-point discipline from
the v0.4.0 convention fix also left every hot function as explicit array-in /
array-out with scalars passed as scalars, which is good structure independent of
any backend.

**Not worth doing to "keep the door open":** an `xp = numpy` array-namespace
indirection, functional-style rewrites of the in-place fill, or jit-shaped
control flow. Each costs readability or diagnostics today for a 1.02× that will
not be taken. The `try/except` around `lu_factor` and `_detect_rank`'s
data-dependent shape are *the design*, not obstacles.

### 4.1 Re-affirmed for large systems — 2026-08-08

The counter-argument to test was that `1000 < nn < 10000` changes the balance:
a bigger working set, a larger share of dense linear algebra, and a plausible
GPU path. Each fails on the numbers in §5.

**The original blocker is unchanged and now covers the entire range of
interest.** JAX still has no complex-argument Hankel. §5.2 puts the
Hankel-vs-linear-algebra crossover at `nn ≈ 20 700`, so across the whole
1000–10000 range a JAX port would `pure_callback` into the same Amos routine for
60–98 % of the runtime. The measured 1.02× ceiling holds at *every* size in
scope; large `nn` does not erode it, because the term that grows is the one JAX
cannot evaluate.

**On memory — the actual bottleneck — JAX is worse, not better.** The two fixes
that matter (§5.5 A and B) are in-place blocked fill and `overwrite_a=True`,
i.e. precisely the operations JAX's functional model removes. `.at[].set()`
under `jit` *may* fuse to an in-place update; that is not a guarantee to rest a
6.4 GB allocation on. Compounding it: `jax` preallocates 75 % of device memory
by default, and `jax_enable_x64` is process-global and must be set before any
array exists — a library has no business setting either in its host's process.
XLA fusion cannot compress a dense complex128 matrix; that allocation is
irreducible under any framework.

**The GPU argument does not rescue it here.** `jax.default_backend()` is `cpu`
on Apple silicon, so there is no GPU path on the development machine at all, and
unified memory means an accelerator would share the same 16 GB regardless. Even
on a discrete GPU, complex128 dense LU runs at 1/32–1/64 rate on non-datacentre
hardware — and §5.2 says that is the 2–30 % being accelerated, not the 70 %.

**What would change the verdict**, recorded so this stays falsifiable rather
than a matter of taste. All three must hold together:

1. the quadrature becomes spectral (§5.4), dropping `nn` by ~50× and moving the
   crossover so the code is genuinely linear-algebra-bound;
2. a fast direct or H-matrix solver replaces dense LU;
3. a CUDA machine with ≥ 40 GB is actually available.

Note that in that world the natural candidate is **CuPy, not JAX** — what is
wanted is BLAS/LAPACK on the GPU, not `jit` and autodiff. None of the three
holds today.

## 5. Large systems — `1000 < nn < 10000`

Measured 2026-08-08 on Apple M5, 10 cores, **16 GB unified memory**, numpy 2.5.1
on Accelerate, scipy 1.18.0, Python 3.12, branch `beyn-port`. All at complex λ.
§§1–4 above were measured at `nn ≤ 400`; this section re-measures to `nn = 3200`
and projects from fitted models.

**Headline: memory binds long before time does, and the wall arrives earlier
than §3.1 implies. Underneath that sits a numerical finding that matters more
than either — the discretisation is first-order, so large `nn` buys far less
accuracy than its cost suggests (§5.4).**

### 5.1 The memory model

Peak RSS of one `assemble()`, one fresh process per row so the high-water mark
is clean:

| `nn` | N = 2nn | matrix | peak (assembly) | peak/matrix | + `lu_factor` copy |
|---|---|---|---|---|---|
| 400 | 800 | 10.2 MB | 26.3 MB | 2.57× | 3.59× |
| 800 | 1600 | 41.0 MB | 103.8 MB | 2.54× | 3.54× |
| 1600 | 3200 | 163.8 MB | 423.5 MB | 2.58× | 3.59× |
| 2400 | 4800 | 368.6 MB | 951.1 MB | 2.58× | 3.58× |
| 3200 | 6400 | 655.4 MB | 1649.3 MB | 2.52× | 3.48× |

The ratio is flat, so the model is exact rather than fitted:

```
peak_bytes ≈ 3.55 × 64·nn² × n_workers   (≈ 227·nn²·W)
```

The 1.55× of temporaries above the matrix is the pair-array block in
`kernels.assemble_matrix`: `np.triu_indices` alone is two int64 arrays of
`nn²/2` — **8 GB at `nn = 10 000`, before any physics** — and `fi_fj`, `gi_gj`,
`cij`, `cji`, `r_tri`, the two argument arrays and the four complex Hankel
arrays are all live at once. The remaining 1.0× is `lu_factor`'s default
`overwrite_a=False` copy in `beyn.contour_moments`.

Thread scaling and its memory cost, 16 contour nodes at `nn = 800`
(assemble + `lu_factor` + `lu_solve` on 12 probe columns):

| workers | time | speedup | peak/matrix |
|---|---|---|---|
| serial | 15.70 s | 1.00× | 3.56× |
| 2 | 8.37 s | 1.88× | 7.66× |
| 4 | 4.61 s | 3.40× | 15.3× |
| 8 | 2.93 s | **5.36×** | 28.2× |
| 10 | 2.94 s | 5.33× | 30.7× |

§3.1's 5× reproduces on the M5 and saturates at 8 workers. But it is a
**memory-for-time trade**, and that is what §3.1 does not say. Where the wall
lands on 16 GB:

| `nn` | serial peak | 8 threads |
|---|---|---|
| 1000 | 0.23 GB | 1.8 GB |
| 2000 | 0.91 GB | 7.3 GB |
| 4000 | 3.6 GB | 29 GB ✗ |
| 6000 | 8.2 GB | ✗ |
| 8000 | 14.5 GB ✗ | ✗ |
| 10000 | 22.7 GB ✗ | ✗ |

**Serial ceiling ≈ `nn` 7000; with 8-way contour threading, ≈ `nn` 2500.**

### 5.2 The time model, and where the crossover is

```
assemble (complex λ)  ≈ 1.45e-6 · nn²   s   (0.061 s at nn=200 → 8.34 s at nn=2400)
lu_factor (N = 2nn)   ≈ 7.0e-11 · nn³   s   (2.29 s at N=6400)
```

Setting them equal puts the crossover at **`nn ≈ 20 700`**. At `nn = 10 000` one
contour node is 145 s of Hankel against 70 s of LU, so **§1's headline survives
the entire large-system range**: this stays special-function-bound, and
optimising the linear algebra is still optimising the small term.

Projected 48-node `modes()` call:

| `nn` | per node | serial | 8 threads | feasible? |
|---|---|---|---|---|
| 1000 | 1.5 s | 73 s | 14 s | yes |
| 2000 | 6.0 s | 4.8 min | 54 s | yes |
| 4000 | 27 s | 22 min | — | serial only |
| 10000 | 215 s | 2.9 h | — | **no — memory** |

### 5.3 Two diagnostics that are the wrong shape at scale

Both are O(N³) carrying a ~20× constant on top of an LU that is already
affordable:

| N | `lu_factor` | `svd` | `cond` |
|---|---|---|---|
| 1600 | 0.054 s | 0.612 s | 0.608 s |
| 4800 | 0.943 s | 19.88 s | 19.21 s |

- `qnm._sigma_ratio` takes a **full SVD per mode**, plus a redundant
  re-assembly. §2's correction parks this at 2.7 % — true at `nn = 200`, where
  assembly swamps it. At `nn = 4000` it is ~92 s + 27 s *per mode* against a
  22-minute contour: 35 % overhead on four modes. `σ_min` by inverse iteration
  on an LU is the right shape.
- `beyn.newton_refine` calls `np.linalg.cond(jac)` on the (N+1)² bordered
  Jacobian — a full SVD — unconditionally on iteration 0. The LAPACK 1-norm
  estimator (`zgecon`) off an LU gives the same decision at ~1× an LU. `jac` is
  also a second full matrix and `np.linalg.solve` copies it again.

Neither is urgent below `nn ≈ 2000`; both are mis-sized above it.

### 5.4 The finding that dominates the rest: the quadrature, not the hardware

Relative error of `qsca` against the Mie anchor, TE, `n_core = 3`, `rad = 200`,
λ = 600, with the far-field angular quadrature refined to `n_angles = 24 000` so
it does not contaminate the fit:

| `nn` | rel. error | observed order |
|---|---|---|
| 800 | 3.28e-3 | |
| 1600 | 1.675e-3 | **0.97** |

Clean **first order**, `err ≈ 2.7/nn`. So `nn = 1000` gives 2.7e-3 and
`nn = 10 000` gives 2.7e-4: **100× the memory and 240× the time for one decimal
digit**, and 1e-6 would need `nn ≈ 2.7 million`. The cap is the classical crude
log-singularity treatment — the `delt/(2e)·gamma` diagonal in `kernels.py` — not
the machine.

For a smooth closed curve under a periodic trigonometric parametrisation, which
is exactly this geometry, **Kress / Martensen-Kussmaul** quadrature splits
`H₀^{(1)}` into `ln|sin((θᵢ−θⱼ)/2)|` times an analytic factor plus an analytic
remainder, integrating the log part with exact weights and the rest by the
trapezoid rule. It is **spectrally convergent**: machine precision at
`nn ≈ 200–300` for this size parameter — some nine digits better than
`nn = 10 000` delivers today, at 1/50th the memory.

That dwarfs the 5× from threading and the 1.7× from memory hygiene. It has the
independent anchor non-negotiable #3 requires (Mie, already in-tree), and it
composes with the parked §3.2 Chebyshev-Hankel work rather than competing with
it. **It is a feature with a validation plan, not an optimisation** — but it
belongs on the roadmap ahead of any large-`nn` engineering, because it changes
what "large" means.

### 5.5 Ranked recommendations

**A. Blocked assembly — nearly free.** Process the upper triangle in row-blocks
(prototyped at block = 256, indices generated per block). Prototype measured
**bit-identical** output (`np.array_equal` → True) at +0.6 % time:

| `nn` | current peak | blocked peak |
|---|---|---|
| 1600 | 2.58× matrix | 1.68× |
| 2400 | 2.58× matrix | 1.43× |

The ratio improves with `nn` — temporaries are O(B·nn) against an O(nn²) matrix
— tending to ~1.05× at `nn = 10 000` *(extrapolated from the two measured
points)*.

**B. `lu_factor(..., overwrite_a=True)` in `beyn.contour_moments`.** One
keyword. The matrix is a fresh temporary from `m_builder` and nothing else holds
it. Removes a full matrix from peak.

A + B together take peak from **3.55× to ~2.1×** the matrix: the serial ceiling
moves from `nn ≈ 7000` to ≈ 9500, and the 8-thread ceiling from ≈ 2500 to
≈ 3400.

**C. Thread the contour loop (§3.1), with a memory-derived worker cap** rather
than a fixed 8: `W = clamp(1, ncpu, budget / (2.1 × 64·nn²))`. Without the cap
the change converts a working `nn = 4000` run into an OOM. Everything else in
§3.1 — per-node status returns, warnings collected and re-emitted in the parent
— stands unchanged.

**D. Re-shape the two O(N³) diagnostics of §5.3.** ~20× → ~1× an LU each.

**E. Then, and only if `nn ≫ 3000` is still needed: the quadrature (§5.4).**

**F. Structural, for later — block-circulant structure.** For a circle sampled
uniformly in θ, `r_ij` and both cross products depend only on `i−j`
(`cᵢⱼ = R²[cos(θᵢ−θⱼ) − 1]`), so **M is block-circulant**: O(nn) Hankel
evaluations, an O(nn log nn) FFT solve, O(nn) memory. An m-fold Gielis shape
gives block-circulant structure in m blocks. This is the only route that makes
`nn = 10⁵` thinkable, and it would make convergence studies against Mie
essentially free — but it is narrow (single smooth symmetric particle) and large.
Recorded, not proposed.

**Rejected: complex64.** It would halve memory, but `M(λ)` is deliberately
near-singular along the contour and it would break the rounding-level parity
test against `assemble_matrix_reference`. Wrong knob.
