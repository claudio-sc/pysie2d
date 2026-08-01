# Performance — where the time goes, and what to do about it

Measured 2026-07-31 on Apple silicon, 8 cores, numpy 2.x on Accelerate,
Python 3.12, at commit `684ecd0` (branch `beyn-port`). Every number below was
measured unless labelled *(estimate)*. This document is about the **package**,
not about the Beyn port; `docs/design/beyn-port-status.md` links here rather
than repeating it.

## 1. The headline

**This is a special-function-bound code, not a linear-algebra-bound one.**

At complex λ — the QNM case, and the one that is slow —
`scipy.special.hankel1` is **97.9 % of one `BIESolver.assemble`** and
**95.5 % of an entire `QNMSolver.modes()` call**. Dense `O(n³)` linear algebra
is **2.4 %**. Any optimisation that does not touch Hankel evaluation, or the
number of times it happens, is optimising 2 % of the runtime.

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

## 4. The JAX evaluation — rejected 2026-07-31

Scoped seriously and rejected on measurement, not taste. Recorded so it is not
re-opened without new information.

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
