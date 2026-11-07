# Cluster reduction study

Measurements behind [cluster-reduction-spec.md](../../cluster-reduction-spec.md).
The question: is the re-expansion in the ε-Beyn prototype a sound basis for a
package feature, and where does it break?

**Answer: route 1 is; routes 2 and 3 are not.** Route 1 is exact linear algebra
and held in every case tried. It pays at many permittivities per wavelength, and
for one-off solves only at large `P`.

These are throwaway scripts: not package code, not lint-checked. Run from the
repo root:

    VECLIB_MAXIMUM_THREADS=4 uv run --frozen python docs/design/studies/cluster-reduction/s1_cost.py

`s5`–`s8` were run with `VECLIB_MAXIMUM_THREADS=1`, because `beyn_modes` threads
the contour itself. Outputs are in `results/`. Timings are single runs on the
4-performance-core machine: read them as ratios, not benchmarks.

`prototype/` is the owner's `eps_beyn_prototype` (on `a14e407`), unmodified, with
its own README. `results/prototype_*.txt` are its cluster scripts re-run here.

## The method

Unknowns `x = [x_1 … x_P]`, each `x_p = [φ_p; χ_p]`. Conventions §14 gives the
block structure, and the off-diagonal blocks have zero lower halves:

    M(ε) = D(ε) + U·Y,   D = diag(M_p(ε)),   U·Y = neighbour coupling

`M_p` is particle `p` alone with its own substrate reflection. Only the interior
rows of `M_p` depend on ε. `U·Y` depends only on λ, the geometry and the
background. Compressing each particle's received coupling by SVD gives a small
matrix and the Woodbury solve:

    G(ε) = I + Y·D(ε)⁻¹·U          (size R₀ = Σ r_p, against N = 2·Σ nn_p)
    x = D⁻¹b − D⁻¹U·G⁻¹·Y·D⁻¹b

This is a numerical T-matrix: `D⁻¹` is what each particle answers alone, and the
SVD basis stands in for cylindrical harmonics.

**Why a cluster mode is not lost in G.** Woodbury also gives

    G⁻¹ = I − Y·M⁻¹·U

exactly. So the isolated-particle poles of `G` cancel in `G⁻¹`, which is the only
thing a Beyn contour integrates, and `G⁻¹` has poles only at cluster modes. With
`M⁻¹ ≈ v wᵀ/((ε−ε_c) N)`, the residue is `Y v wᵀ U / N`. It vanishes only if
`Y v = 0`, meaning the mode's field is zero on every neighbour's boundary. An
outgoing field that vanishes on a curve vanishes throughout the exterior, so
this needs a mode with no exterior field. The one remaining route is truncation:
a coupling below the SVD tolerance (1e-13 of the block's largest singular value).
`beyn_modes`' docstring warns that poles of M invalidate its rank argument; the
identity above is why that does not apply to `G`.

## Results

### Accuracy: exact to round-off

| Check | Result | Source |
|---|---|---|
| Plane wave, 4 mixed particles (shape, `nn`, material); free, glass, PEC, SiN; both pols | `x` 4e-14 to 1.4e-13, far field ≤ 1.6e-15 | `s5` H |
| Gap ladder to gap/a = 0.04, free and slab | `x` ≤ 4.7e-13 | `s1` A |
| `P` up to 64 with different radii | `x` ≤ 1.1e-13 | `s1` B |
| Polished eigenpermittivities, `G` vs `M` | ≤ 8e-16 TM circles (`s3`); ≤ 4.6e-15 triangles, free and slab (prototype) | `s3`, `prototype_reexp_*` |

Route 2 (isolated modes + quadratic leftover) misses eigenvalues by |Δε| of 0.008
to 0.5 in free space (median to worst, pitch 600 and 300), and its contour fails
outright over the slab. Route 3 (pure hybridisation) misses by 0.04 to 3.5, against
|ε| of 1 to 6. See `results/prototype_reexp_*.txt`.

### Cost per new ε at fixed λ

`nn = 100`, pitch 600 nm, complex ε so the interior rows take the Amos path.
"Today" is `_assemble` + solve, which is what `scatter` does on every call.

| P | Today, free | Today, slab | Dense, exterior cached | Route 1, distinct | Route 1, identical | Setup |
|---|---|---|---|---|---|---|
| 4 | 0.03 s | 0.18 s | 17 ms | 11 ms | 3 ms | 0.04 s |
| 8 | 0.09 s | 1.09 s | 74 ms | 23 ms | 5 ms | 0.12 s |
| 16 | 0.49 s | 8.6 s | ~400 ms | 49 ms | 10 ms | 0.42 s |
| 32 | 2.8 s | 66 s | 2.5 s | 0.12 s | 35 ms | 1.6 s |
| 64 | 27 s | not run | 26 s | 0.45 s | 180 ms | 7.2 s |

- **The prototype's speedups assume identical particles at one height.** Distinct
  particles get 1.5× at `P = 4` and 58× at `P = 64`, against 6× and 144×.
- **One-off solves** (a wavelength sweep): 9.6 s against 27 s at `P = 64`, 2.2 s
  against 2.8 s at `P = 32`, a slight loss at `P = 16`. Not a reason to change
  `scatter`.
- **Slab assembly grows 6–8× per doubling of `P`** and is 27× the solve at
  `P = 32`. That reverses the ranking in `performance.md` §6, measured in free
  space. Route 1 does not touch the first assembly; caching avoids repeats. This
  is a separate problem and is not addressed here.
- **Rank saturates:** `R₀/P` is 24, 27, 28, 28.4, 28.7 for `P` = 4 … 64. It is a
  property of the receiving particle.
- **Gain grows with particle size** (`s5` I): 2.7× at x = 2.6 and 27× at x = 42,
  with `r_max/2nn` falling from 0.47 to 0.05.
- **Truncating harder buys nothing** (`s1` C): tol 1e-4 gives only 1.3× per ε and
  costs nine digits. Keep 1e-13.
- **Memory:** `Y + U` is 380 MB against 2.6 GB at `P = 64`, but only if exterior
  rows are built one particle at a time. These probes built the dense matrix first.

### Cancellation near an isolated pole

`x = a − R·y` cancels when `D` is nearly singular but `M` is not (`s2` F, `s4`):

| \|ε−ε_iso\| | Error | ‖a‖/‖x‖ | After 1 refinement | After 2 |
|---|---|---|---|---|
| 1e-2 | 2.9e-13 | 0.46 | 1.1e-13 | 1.1e-13 |
| 1e-6 | 3.3e-11 | 2.4e3 | 1.1e-13 | 1.2e-13 |
| 1e-8 | 2.3e-9 | 2.4e5 | 1.6e-13 | 1.2e-13 |
| 1e-10 | 4.3e-7 | 2.4e7 | 1.7e-13 | 1.4e-13 |
| 1e-12 | 4.1e-5 | 2.4e9 | 1.5e-9 | 1.3e-13 |

**Correction, from the build.** The error tracks `1.5e-14·‖a‖/‖x‖` only for this random
right-hand side. A plane wave that does not drive the resonance keeps `‖a‖/‖x‖` at 0.26 at
every distance, while the error still reaches 5e-5, so a guard on that ratio never fires.
What does track it is `cond(G)`: about `5e-19·cond(G)` for both. The package therefore
refines every solve once and reads the error off the step (`s9_final_figures.py`).

Refinement must use the **full** residual: `b − M·x` has non-zero interior rows,
and the first attempt dropped them and appeared to show refinement failing. That
flawed run remains in `results/s3_anchor_refine_first_run.txt`, section F′.

### Anchors

| Anchor | Result | Source |
|---|---|---|
| Two circles, Graf addition theorem, TE | 9e-15 (d = 1000), 7.9e-14 (d = 400), 6.6e-11 (d = 300) | `s2` D, unpolished |
| Same, TM, polished | 7e-16 to 7.7e-16 (d = 1000), 2.9e-14 (d = 400), 3.1e-10 (d = 300) | `s3` |
| PEC mirror: every PEC mode in the circle + image pair | ≤ 1.9e-15, both pols, h = 200, 500 | `s8` L |
| SiN slab: pole fitted to passive driven solves vs ε-Beyn | 5.9e-7 to 1.3e-5 | `s8` M |

At d = 300 the limit is Graf's own truncation: order 10 against 12 differs by
1.3e-8 in TM. The slab fit needs no contour and no branch choice, and its driven
path is the one the roadmap's v1.3 milestone compared with MEEP.

**A transparent stack is not an anchor.** `_effective` turns it into no
background, so the difference is exactly 0.0 by construction (`s8` K): it cannot
fail.

### Trying to hide a mode

| Attempt | Result | Source |
|---|---|---|
| TE circles, d up to 3e5 nm | `M` and `G` find the same count | `s2` E |
| Whispering-gallery ladder at d = 6000, Im ε_iso down to 5.6e-13 | 4 = 4 modes, `M`−`G` ≤ 1.5e-14 | `s5` J |
| Graf roots at d = 2500 (10 roots) | `M` and `G` singular at every root, Newton roots agree; the 5 tight boxes where both return a count, Q up to 2.7e6, all give 2 = 2 | `s7` |

`s5` J showed `M` rank 4 against `G` rank 0 at d = 2500. `s6` found `G` and Graf
agree in all 20 rows with `M` the outlier: it is `M`'s noise in boxes far smaller
than the distance to the nearest pole. The evanescent split had pushed the modes
out of those boxes. In two wide boxes `G` succeeded where `M` reported no rank
gap, and in three more both failed, so `G` may be the more robust contour operator;
that is an observation, not a tested claim. The boxes in `s6` are themselves flawed (their top edge excludes
the modes), which is why `s7` exists.

**Not tested:** a high-Q or symmetry-dark mode over the slab.

### Existing behaviour found along the way

Both predate this work and neither is documented. The spec adds the notes;
neither is fixed.

- **Exterior irregular frequency** (`s3` G). At k_bg·a = j₀₁ the exterior rows
  lose rank for every ε, and `Q_ext` is wrong by 17 % (TE) and 33 % (TM), silently.
  The window is narrow: 5e-11 at δλ/λ = 1e-6, 5e-8 at 1e-9, 5e-5 at 1e-12.
- **False λ-QNM modes at H_n^{(1)} zeros** (`prototype/test_side.py`, `s9_false_qnm.py`).
  σmin/σmax is 2.8e-8, against 2.5e-3 two percent away, at Q ≈ 0.17, with the left
  null vector entirely in the interior rows. **`QNMSolver` does report it**, as a
  doubly degenerate 221.63 + 661.24i nm, for a circle of a = 100, n_core = 1.5, TE. No
  Mie denominator for n = 0…12 is near zero there; the true n = 2 pole is near
  264.8 + 52.4i.

## Scripts

| Script | What it measures |
|---|---|
| `s1_cost.py` | A rank vs gap, B cost vs `P`, C truncation |
| `s2_modes.py` | D Graf anchor, E far-separation hiding, F cancellation, G (crashes: wrong key; `s3` is the fixed version) |
| `s3_anchor_refine.py` | D′ polished TM anchor, G irregular frequency. Section F′ is removed from the script; see above |
| `s4_refine_full_rhs.py` | Refinement with the full residual |
| `s5_gaps.py` | H every background and pol, I rank vs size, J high-Q ladder |
| `s6_hidden_check.py` | The `s5` J discrepancy with widened boxes and Graf |
| `s7_hidden_decisive.py` | `M` and `G` at the true Graf roots |
| `s8_slab_anchor.py` | K transparent stack, L PEC mirror, M passive pole fit |
| `s9_refine_debug.py` | The build's cancellation diagnosis: plane wave vs random, `cond(G)`, the gap floor |
| `s9_final_figures.py`, `s9_final_gates.py`, `s9_final_gates2.py` | On the shipped code: the cost table and gap floor; the mode gate figures; the driven figures, glass fit and false QNM search |
| `s9_false_qnm.py` | `QNMSolver` at an `H_2^{(1)}` zero, against every Mie order |
| `s9_mutations.py` | Breaks the code on purpose; every mutation must fail a gate |

The `s9` scripts run against the package's `ClusterEigenEps`, from the repo root.
