# Kress–Martensen quadrature and the frozen-map API — code-spec (draft 1)

**Status:** draft 1, 13 Sep 2026. Not implemented. A verified reference
implementation of everything below is committed alongside as
[kress-spec.patch](kress-spec.patch), against branch `v0.6-quadrature` at
`e74c10f`.

**Who it is for:** the worker who lands v0.6 architecture items 1 (Kress on all
four blocks), 6 (`assemble_matrix_dwn`, QNM and sensitivity plumbing) and 7 (the
`theta=` API break). You do not need to derive anything. Every formula is
stated, every decision gives the evidence that settled it, and every tolerance
comes with the measurement behind it.

**Decisions it works within:** [v0.6-architecture.md](v0.6-architecture.md)
(doc A) §1–§5, [parametrisation-spec.md](parametrisation-spec.md) (landed), and
the four answers given on 13 Sep 2026 (§2). **Numbers:**
[pysie2d-quadrature-handoff.md](pysie2d-quadrature-handoff.md) §6 for the
mathematics; this document for everything measured on this machine.

---

## 0. How to execute this spec

Read §2, §3 and §9 before touching anything. The patch tells you *what* to
write; those sections tell you *why*, which is what you need when something
does not match.

### 0.1 The procedure

1. `git switch v0.6-quadrature && git pull`. Confirm `git log -1` is `e74c10f`
   or a docs-only descendant of it. If `src/` or `tests/` have changed since
   `e74c10f`, **stop and report**: the patch and every number below were
   produced against that tree.
2. `git apply --check docs/design/kress-spec.patch`, then
   `git apply docs/design/kress-spec.patch`.
3. Run the acceptance checklist, §8. Every item must pass as written.
4. Commit in the two commits of §7, **with exactly the messages given there**.
5. `git push`. PR #16 (draft) picks the commits up; CI runs on it.
6. Report back with the §8 checklist ticked and the CI link.

### 0.2 Rules that are not negotiable

- **Never widen a tolerance to make a test pass** (CLAUDE.md non-negotiable 4).
  If an assertion fails, or a measured value in §5 comes out more than 3× worse
  than the number quoted, stop and report both numbers. Do not re-measure and
  update the constant.
- **Do not edit anything listed out of scope in §1.2**, even where it is now
  visibly stale. §10 and §11 say who owns it.
- **Do not reformat or "tidy" code the patch does not touch.** `ruff format`
  must report the tree already formatted after the patch; if it wants to change
  a file, stop and report.
- **Do not merge to `main`.** v0.6 is a release, and merging is the owner's
  decision (CLAUDE.md, Git workflow).

### 0.3 If the patch does not apply

Transcribe from §3–§6. The patch is authoritative for exact text, and this
document for intent. Where they disagree, the patch is what was verified: report
the disagreement rather than choosing between them.

---

## 1. Scope

### 1.1 In

| # | Change | Where |
|---|---|---|
| 1 | Kress–Martensen product quadrature on all four blocks, both parities of `nn` | `kernels.py` |
| 2 | Its analytic wavenumber derivative, bit-identical matrix half | `kernels.py` |
| 3 | The loop reference rewritten as an independent Kress implementation | `kernels.py` |
| 4 | `Parametrisation.uniform_theta()`, the default node map | `parametrisation.py` |
| 5 | `Geometry` on a `Parametrisation`: arrays in `t`, `delt` derived, `theta=` → `parametrisation=` | `geometry.py` |
| 6 | Non-closing superformula rejected; v0.5 `theta` array refused with the migration in the message | `geometry.py` |
| 7 | `delt` dropped from the assembly primitives; narrowed to `float` in the field primitives | `kernels.py`, `solver.py`, `fields.py`, `green.py` |
| 8 | Frozen-map guard in `QNMResult.sensitivity`; docstrings whose measured claims are now false | `qnm.py` |
| 9 | `Parametrisation` exported | `__init__.py`, `tests/api_baseline.txt` |
| 10 | Every test whose premise, call or tolerance the change invalidates (§5) | `tests/` |
| 11 | `docs/conventions.md` §5, §9, §10, §11, §12, §13 | `docs/conventions.md` |

### 1.2 Out — do not touch

| Out | Why | Owner |
|---|---|---|
| Curvature-adaptive density as a default or a feature | Held pending G3 (doc A §4.1). `Parametrisation.gielis` stays callable as it is. | G3 |
| Far-field angular quadrature, and the `efficiencies` forward-index bug (F1) | Architecture item 5 / G5 | far-field spec |
| `README.md`, `CLAUDE.md`, `docs/qnm-guide.md`, `docs/design/performance.md`, `examples/` prose and figures | Item 8, "lands last" (doc A §8). **Merge blockers** — §11. | docs pass |
| Default `n_pts` (stays 200) | Deferred by decision (doc A §7) | owner |
| `Parametrisation.gielis` defects F4, F5 | Landed code, separate spec | parametrisation spec |
| `reference/mie.py` (F2) | Validation anchor; tests pass `n_max` explicitly instead | owner |
| `docs/design/studies/*.py` that import removed functions | Historical, excluded from CI; they stop running and that is accepted | — |
| `gielis-oed` | Pinned `<0.6`; migration is its own work | gielis-oed |

---

## 2. Decisions

| # | Decision | Evidence / reason |
|---|---|---|
| **D1** | **Default node map is uniform θ** (`Parametrisation.uniform_theta()`), not uniform arc length. | Study `studies/kress_default_map.py`, Appendix A. Once resolved, uniform θ is the most accurate map on every shape measured (rounded squares to `n = 20/50`, spiky stars to arm ratio 8). Arc length is ahead only at the coarsest resolutions: 2–7× at `nn = 40` on the rounded squares, and up to 30× at `nn = 120` on the arm-ratio-8 spike, where uniform θ's worst node has `R ≈ 4`. From `nn = 320` there uniform θ is spectral (5.8e-9 at 960) while arc length stalls near 3e-4. The arc-length map is **not** at fault: it matches an independent `scipy.quad` inversion to 2.7e-15. Composing with `w` narrows the analyticity strip (doc A §4.1). Uniform θ also needs no construction and no `n_core`, accepts non-analytic shapes that `Parametrisation.gielis` refuses (F4, F5), and is shape-independent, so every geometry is on a frozen map automatically. **Corrected 14 Sep 2026:** the low-`nn` arc-length advantage above is single-λ noise; at `nn` 50–300 on arm ratios 4–16 and three wavelengths no map is reliably better, and uniform θ reaches 1e-3 first where any map does (study plan, "Spiky stars at campaign resolution"). |
| **D2** | Keyword `parametrisation=`, not `theta=` with a new type. | Owner's answer. An old call fails loudly (`TypeError: unexpected keyword`), and the name says what is frozen. |
| **D3** | `delt` removed from `assemble_matrix`, `assemble_matrix_dwn`, `assemble_matrix_reference` and `Geometry.__init__`; `Geometry.delt` becomes a read-only property `2π/n_pts`. `far_field`/`eval_field` keep `delt: float`. | Owner's answer. Kress's weights presume `h = 2π/nn` exactly, so a free parameter could only be passed wrong. Keeping the field primitives' parameter keeps their call sites unchanged. |
| **D4** | `assemble_matrix_reference` rewritten as a Kress loop, not deleted. | Owner's answer. CLAUDE.md keeps it as a deliberate second implementation. It is made independent where vectorisation bugs hide: explicit pair loop, `W` rebuilt inline from `R`, Bessel functions always at complex argument. It caught a block-indexing error while this spec was being verified. |
| **D5** | Kress assembled as **shipped trapezoid entry + circulant correction** `K₁·W_d`, not as handoff §10's `A₁·R + A₂·h`. | Algebraically identical (§3.3), but needs no O(nn²) log array, leaves the M1/M3 diagonals exactly as shipped, and shares one cached `(nn,)` vector per resolution. |
| **D6** | Weights valid for **both parities** of `nn`. | The handoff formula is even-only. At odd `nn` it returns a plausible wrong matrix: a QNM displaced by **1.1 nm at `nn = 115`**, nothing raised. The suite uses odd `n_pts` (37, 33). |
| **D7** | Geometry arrays from the landed `_rderiv`/`_rderiv2` plus the chain rule, not from `parametrisation._radius_derivatives`. | Agree to ≤ 2.1e-15 on circle, ellipse, star and cusped star. `_rderiv` is what the Mie anchor already validates, and it masks the `0/0` at exact zeros that `_radius_derivatives` turns into `nan` for non-even exponents (e.g. `m = 0`, `n3 = 1`). |
| **D8** | Closure check in `Geometry.gielis`: integer `m`, and even `m` **or** (`a == b` and `n2 == n3`). | The v0.5 guard was the arc-length inversion noticing coincident nodes. On uniform θ nothing would notice, and a non-periodic `r(θ)` integrates a jump silently. The rule is derived in `_closes` and checked numerically: `m = 3, a = b, n2 = 4, n3 = 6` does not close. The v0.5 D5 note named only `a ≠ b`. |
| **D9** | Sensitivity guard kept, comparing `θ`, `w'`, `w''` exactly. Its rationale is rewritten. | Under a smooth map, rebuilding it at `b ± h` is second order too (rate 100.0 both ways), but `∂M/∂b` differs by **20 %** and `dλ/db` agrees only to 1.2e-10 (discretisation). The guard now protects "one discretisation", not "no O(h) term" (conventions §10). |
| **D10** | QNM and sensitivity tests run at `n_pts = 40`, `n_quad_per_side = 12`. Refinement tests deliberately use 6. | At 40 the TE n=0 discretisation error is 2.8e-14 nm, and 12 nodes per side put the contour at 1.1e-11. At 6 the contour is 7.0e-6, the case `refine()` now fixes. The whole suite runs in 20 s. |

---

## 3. The mathematics

Everything here is in the patch verbatim. Notation: `nn` nodes, `h = 2π/nn`,
`d = j − i` the node offset, `k` a wavenumber (background or core),
`r = |x_i − x_j|`, `z = k·r`, `γ = |dx/dt|`, `c_ij = (f_i − f_j)·ġ_j − (g_i − g_j)·ḟ_j`,
`deriv = ḟ·g̈ − f̈·ġ`, `γ_E` Euler's constant. Dots are `d/dt`.

### 3.1 Nodes and the parameter

    t_j = 2π(j + ½)/nn,   θ_j = w(t_j)

All geometry derivatives are with respect to `t` (§3.6). The offset `½` is
immaterial to the weights, which depend on `d` only.

### 3.2 Weights

    R_d = −(4π/nn) · Σ_{m=1}^{⌊(nn−1)/2⌋} cos(2πmd/nn)/m  −  [nn even]·(4π/nn²)·cos(πd)

    W_d = R_d − h·ln(4 sin²(πd/nn))    for d = 1 … nn−1
    W_0 = R_0

`R` is the unique vector with `Σ_d R_d cos(2πmd/nn) = −2π/m` for
`1 ≤ m ≤ ⌊nn/2⌋`, and `0` for `m = 0`. That is the independent test
(§5, `test_kernels`). For even `nn` it equals handoff §6.2 to 4.4e-16. Both
arrays are cached per `nn` (`functools.lru_cache`) and returned **read-only**.

### 3.3 Matrix entries, `i ≠ j`

For each wavenumber `k` (`k_bg` for rows `0:nn`, `k_core = nc·k_bg` for rows
`nn:2nn`):

    single_ij(k) = 0.25i·h·H₀(z)  −  W_d·J₀(z)/(4π)
    ratio_ij(k)  = 0.25i·h·H₁(z)/z  −  W_d·(J₁(z)/z)/(4π)
    double_ij(k) = k²·ratio_ij(k)

    M1[i,j] = double(k_bg)·c_ij        M2[i,j] = single(k_bg)
    M3[i,j] = double(k_core)·c_ij      M4[i,j] = η·single(k_core)

with `η = ε` for `pol = 1` (TM) and `1` for `pol = 2` (TE). The `(j, i)` entry uses
the same `single`/`double` value, with `c_ji = −(f_i − f_j)·ġ_i + (g_i − g_j)·ḟ_i`.

**Why this equals Kress** (handoff §6.3): an entry is `K₁·R_d + (K − K₁·L_d)·h`
with `L_d = ln(4 sin²(πd/nn))`. Collecting `K₁` gives `h·K + K₁·(R_d − h·L_d)`,
i.e. `h·K + K₁·W_d`. For the single layer `K = 0.25i·H₀(z)` and
`K₁ = −J₀(z)/(4π)`. For the double layer `K = 0.25i·k²·c·H₁(z)/z` and
`K₁ = −(k²/(4π))·c·J₁(z)/z`.

### 3.4 Diagonals

    M1[i,i] =  0.5 − h·deriv_i/(4π γ_i²)                              (unchanged)
    M3[i,i] = −(0.5 + h·deriv_i/(4π γ_i²))                            (unchanged)
    M2[i,i] = −W_0/(4π) + h·[0.25i − (γ_E + ln(k_bg·γ_i/2))/(2π)]
    M4[i,i] = η·{ −W_0/(4π) + h·[0.25i − (γ_E + ln(k_core·γ_i/2))/(2π)] }

The M1/M3 diagonals are identical to the shipped ones because the double layer's
`K₁` vanishes on the diagonal. `ln` is the principal branch, the same branch as
`H₀^{(1)}`, and holomorphic on the QNM half-plane of conventions §8.

### 3.5 Wavenumber derivative

With `C` standing for `J` or `H^{(1)}` alike:

    d/dk[k²·C₁(kr)/(kr)] = k·C₀(kr)          d/dk[C₀(kr)] = −k·r²·C₁(kr)/(kr)

so, per wavenumber,

    d single/dk = −k·r²·ratio          d double/dk = k·single

    dM1 = (d double/dk)(k_bg)·c        dM2 = (d single/dk)(k_bg)
    dM3 = nc·(d double/dk)(k_core)·c   dM4 = η·nc·(d single/dk)(k_core)

    dM1[i,i] = dM3[i,i] = 0
    dM2[i,i] = −h/(2π·k_bg)            dM4[i,i] = −η·nc·h/(2π·k_core)

`nc` is `d k_core/d k_bg`. The matrix half of `assemble_matrix_dwn` must be the
**same expressions in the same order** as `assemble_matrix`: the test is
`np.array_equal`.

### 3.6 Complex wavenumbers and the fast path

`_j0_h0(z)` / `_j1_h1(z)` return `(J, H)`:

- real `z`: `j = j0(z); return j, j + 1j*y0(z)` (Cephes; `J` is free)
- complex `z`: `jv(0, z), hankel1(0, z)` (Amos; `J` costs a second call)

`wnum_bg` and `ri·wnum_bg` go through `_real_if_real` first, as today.
**Keep the complex branch** (non-negotiable 1). Cost *(measured on the patched
code, complex λ, best of 7)*: Kress assembly is 1.44× / 1.45× / 1.45× the shipped
one at `nn = 40 / 60 / 200`, and the fused derivative 1.01× the matrix at all
three. Against that,
`nn` for a given accuracy drops by an order of magnitude or more (§5 numbers).

### 3.7 Geometry arrays in `t`

`_boundary_arrays(nodes, rad, a, b, m, n1, n2, n3, x0, z0)` evaluates `gielis`,
`_rderiv`, `_rderiv2` at `θ_j` (exactly as `_etoil_arc` did), forms `f′, g′, f″, g″`
in θ, then

    ḟ = f′·w′            f̈ = f″·w′² + f′·w″        (same for g)

with `w′, w″` from `Parametrisation.nodes(nn)`. On the identity map
`w′ = 1`, `w″ = 0` bit-for-bit.

### 3.8 Closure

    closes ⇔ m is an integer and (m even or (a == b and n2 == n3))

`|cos u|^n2/a^n2 + |sin u|^n3/b^n3` has period π in `u = mθ/4`, and π/2 exactly when
the swap `cos ↔ sin` leaves it unchanged. One turn advances `u` by `mπ/2`.

---

## 4. Code changes by file

Full text in the patch. What follows is what each change is and the trap it
avoids.

### 4.1 `src/pysie2d/kernels.py`

- Module docstring: Kress, `t`, and χ carrying the Jacobian of `t`.
- Imports add `functools` and `jv`.
- The Hankel helper section (`_real_if_real`, `hank0`, `hank1`, `cbesh`) is
  **unchanged**: `fields.py` and `green.py` use it.
- **Added:** `_kress_log_weights(nn)`, `_kress_weights(nn)` (§3.2), `_j0_h0`,
  `_j1_h1` (§3.6).
- **Rewritten:** `assemble_matrix_reference`, `assemble_matrix`,
  `assemble_matrix_dwn` with signature
  `(pol, nn, f, g, df, dg, ddf, ddg, wnum_bg, ri, kd)` — `delt` gone.
- **Removed:** the "verbatim translation of subroutine matm / fixed by the
  original Fortran" banner, and the `2e` self-patch comments. They describe a
  scheme that no longer exists, and a banner saying "do not alter" over new code
  would be false.

The reference loop's block placement is the one thing that went wrong during
verification. Row offset `0` carries M1 (φ columns `j`) and M2 (χ columns
`j + nn`); row offset `nn` carries M3 and M4. The **χ column offset is always
`nn`**, for both wavenumbers.

### 4.2 `src/pysie2d/parametrisation.py`

- Module docstring: no longer "additive, not wired".
- `nn_from_band: int | None`, documented `None` for `uniform_theta`.
- **Added** `Parametrisation.uniform_theta()`:
  `a0 = 1.0`, `coef = np.zeros(0, complex)`, `theta_fine = linspace(0, 2π, 1024,
  endpoint=False)`, `t_fine = append(theta_fine, 2π)`, `nn_from_band = None`,
  `contrast_realised = 1.0`, `alpha_eff = 0.0`, `n_fine = 1024`, `n_terms = 0`.
  It goes through the ordinary `nodes()`. Do **not** add an `if identity:`
  shortcut there: parametrisation-spec §6, contract 2, forbids special cases, and
  the bit-identity test (§5) is what proves none is needed.

### 4.3 `src/pysie2d/geometry.py`

- **Removed:** `_etoil`, `_validated_theta`, `_uniform_arc_theta`, `_etoil_arc`,
  `boundary_setup`. Nothing in `src/`, `tests/` or `examples/` uses them after
  the patch.
- **Kept unchanged:** `gielis`, `_rderiv`, `_rderiv2`, `perimeter`,
  `Geometry.is_circle`.
- **Added:** `_closes`, `_checked_parametrisation`, `_boundary_arrays`.
- **`Geometry.__init__(f, g, df, dg, ddf, ddg, *, rad, x0=0.0, z0=0.0,
  parametrisation=None)`** stores `parametrisation` and
  `nodes = parametrisation.nodes(len(f))` (or `None`). It calls
  `_checked_parametrisation` first.
- **Properties:** `delt → 2π/n_pts`; `theta → nodes.theta` or `None`.
- **`Geometry.gielis(rad, n_pts=200, *, m, n1, n2, n3, a, b, x0, z0,
  parametrisation=None)`**: type check, closure check, default to
  `uniform_theta()`, `_boundary_arrays(parametrisation.nodes(n_pts), …)`, construct.
  `a` and `b` are honoured on every map (the v0.5 "ignored unless arc_length"
  note is gone with its cause).
- `nodes()` runs twice per factory call (once in `gielis`, once in `__init__`).
  That is accepted: it is pure, and trivial on the default map.

### 4.4 `src/pysie2d/solver.py`, `fields.py`, `green.py`

- `solver.py`: `g.delt` removed from the two assembly calls. `far_field` and
  `eval_field` still receive `g.delt`, now the property.
- `fields.py`: `delt: float`; the per-point-array branch in `far_field` becomes
  `* delt`; docstrings say derivatives are in `t`.
- `green.py`: the `dl` array is deleted and `* dl[None, :]` becomes `* delt`.

### 4.5 `src/pysie2d/qnm.py`

- `_perturbed_solver`: `None` check on `nodes`, with the message still containing
  `"carries no node set"`; then exact equality of `n_pts`, `theta`, `dw` and
  `ddw`, with the message still containing `"base node set"` and naming
  `parametrisation=result.geometry.parametrisation`. Tests match on those phrases.
- Docstrings rewritten because their measured claims are false under Kress:
  `refine` (it now removes the contour error), `modes(n_quad_per_side)` (the
  contour is the floor), `sensitivity` (frozen *map*), `richardson_limit` (no
  pysie2d quantity is first order any more).
- **Behaviour otherwise unchanged.** `richardson_limit` stays exported. Whether to
  deprecate it is F7.

### 4.6 `src/pysie2d/__init__.py`, `tests/api_baseline.txt`

Export `Parametrisation`, document it in the module docstring (a test enforces
that), and reword the `richardson_limit` line. Regenerate the baseline with the
recipe in `tests/test_public_api.py`:

    uv run python -c "from tests.test_public_api import _surface; print('\n'.join(_surface()))" > tests/api_baseline.txt

The diff is exactly: `Geometry.__init__` and `Geometry.gielis` signatures, the new
`Geometry.delt` and `Geometry.theta` properties, the five `Parametrisation`
lines, `delt` gone from `assemble_matrix`/`assemble_matrix_reference`, and
`delt: float` in `eval_field`/`far_field`. Anything else in that diff means the
patch was not applied as written.

---

## 5. Tests

Test count goes from 217 to **258**. "Kept" means untouched and passing. Every
changed constant carries its measurement and reason in a comment in the patch;
the table gives the headline.

### 5.1 `tests/test_kernels.py` — rewritten

| Test | Status | Measured → bound | Why it cannot pass by accident |
|---|---|---|---|
| `test_kress_log_weights_integrate_trig_polynomials_exactly` [nn 8, 9, 64, 65] | added | ≤ 1.3e-14 → `4·nn·eps·max(abs(R))` | Closed-form Fourier integral; independent of the package |
| `test_the_even_only_weight_formula_is_wrong_at_odd_nn` | added | handoff formula at nn 65 misses by > 1e-3 | Control for the parity branch (D6) |
| `test_kress_weights_are_shared_and_read_only` | added | `is` same object; write raises | Shared across contour threads |
| `test_fast_assembly_matches_reference` [circle-64, skew-graded-33 × pol × k] | rewritten | 2.6e-16 → 1e-12 | Skewed shape, graded map, odd nn break every symmetry a circle hides |
| `test_matrix_derivative_matches_assembly` [both shapes] | rewritten | bit-identical → `np.array_equal` | — |
| `test_matrix_derivative_matches_central_difference` [circle-64, skew-graded-64] | rewritten | order 2.000; finest 6.8e-8 → 1e-7 (circle), 2.0e-7 → 5e-7 (skew) | Order and magnitude both pinned |

### 5.2 `tests/test_geometry.py`

| Test | Status | Notes |
|---|---|---|
| `test_ddf_ddg_match_fine_grid_finite_difference` | rewritten | `arc_length` column removed. On uniform θ, `t = θ`, so the closed form is compared directly; the ellipse row keeps `a ≠ b`. Tolerances unchanged. |
| `test_t_derivatives_match_spectral_differentiation` [uniform θ, arc length] | added | Aspect 1.3/0.8 ellipse, `nn = 256`: 1.8e-12 / 1.2e-11 → 1e-9. A missing `w″` term misses by ~0.3. |
| `test_non_closing_boundary_is_rejected` | added | Replaces `test_coincident_arc_length_nodes_…` (D8) |
| `test_a_v05_theta_array_is_refused_with_the_migration` | added | `TypeError` naming `parametrisation=other.parametrisation` |

### 5.3 `tests/test_parametrisation.py`

`test_uniform_theta_is_the_identity_map_bit_for_bit` added: `theta == t`,
`dw == 1`, `ddw == 0` exactly at `nn = 37` and `200`, and `nn_from_band is None`.
Everything else kept.

### 5.4 `tests/test_efficiencies.py`, `test_convergence.py`, `test_green.py`

| Constant / test | Was | Now | Measured |
|---|---|---|---|
| `RTOL_MIE` (4e-3) → `RTOL_QSCA` | 4e-3 | 1e-12 | ≤ 2.7e-15 over 500/600/800 nm, both pol, both circle branches |
| → `RTOL_QEXT` | 4e-3 | 1e-5 | ≤ 4.5e-6. **All of it is F1**, the forward-index off-by-one, not the solver |
| → `RTOL_QABS` | 4e-3 | 3e-5 | ≤ 1.0e-5 (inherits F1) |
| `RTOL_ENERGY` | 1e-3 | 1e-5 | ≤ 2.8e-6 (inherits F1) |
| `test_qsca_is_independent_of_incidence_angle_on_the_circle` | kept | 1e-10 | 8.8e-16 |
| `test_convergence` | first order ×7 over 40→320 | spectral: nn 12, 16, 20, 24, each step ×100, last < 1e-11 | 2.9e-4 → 1.0e-6 → 1.2e-9 → 6.3e-13 |
| `ANCHOR_NN` | 1000 | 240 | Set by the 5-spacing dipole guard at d = 1.2a (needs nn ≥ 158), not accuracy |
| `REFERENCE_N_MAX` | — | 80 | F2: the default Wiscombe order truncates the Graf sum at 3.1e-6 (TE) / 2.4e-4 (TM) in Re S at d = 1.2a |
| self-Green rel | 1e-2 | 1e-12 | ≤ 3.4e-14 over 4 distances × 2 pol |
| `test_reciprocity` rel | 1e-6 | 1e-12 | 1.6e-15 |
| `test_free_space_limit` | kept | 1e-2 | 2.2e-3, 5.8e-3: physics (finite distance), not discretisation |

### 5.5 `tests/test_conventions.py`

| Test | Status | Notes |
|---|---|---|
| `test_efficiencies_match_mie_in_cladding` | re-anchored | 1e-12 (qsca, measured 2.7e-15), 1e-5 (qext, 2.8e-6) |
| `test_absorbing_particle_in_cladding` | re-anchored | 3e-5 (measured 1.0e-5) |
| `test_self_green_in_cladding` | re-anchored | `n_pts` 1000 → 240; rel 1e-12 (Re 8.1e-14, Im 2.4e-14; default `n_max` is enough at d = 2a) |
| `test_frozen_nodes_restore_second_order_convergence` | **replaced** by `test_shape_derivative_of_m_depends_on_the_frozen_map` | Frozen and rebuilt maps both rate > 50 (measured 100.0, 100.0); relative ∂M/∂b difference > 0.05 (measured 0.200). Ellipse `b = 1.2`, `n_pts = 60`, λ = 700 + 8i. |
| `test_wavelength_over_ds_uses_the_real_part_and_the_worst_node` | rewritten | Frozen arc-length map at b = 1.6: 24.8 < 28.2 |
| `test_wavelength_over_ds_flags_an_elongated_shape_as_under_resolved` | rewritten | Explicit arc-length map: 37.14 / 17.46 / recovered 37.19 — same assertions |
| `test_sensitivity_step_is_in_the_parameters_own_units` | rewritten | `n_pts` 40, rel 1e-9 → 1e-8 (measured 3.8e-10; floor ε·rad/h ≈ 4e-9, F6) |
| `test_sensitivity_degenerate_dispatch_agrees_with_multiplicity` | call sites | `n_pts` 40, default contour |
| `test_sensitivity_left_vector_is_not_conj_of_the_right_one` | call sites | `n_pts` 40 (1.16, 4.5e-16) |
| `test_geometry_without_a_node_set_is_allowed_but_cannot_be_differentiated` | call sites | 6-array constructor; `parametrisation`, `nodes`, `theta` all `None` |
| all others | kept | |

### 5.6 `tests/test_qnm.py`

| Item | Now | Measured at `n_pts = 40` |
|---|---|---|
| Anchors | 16 digits from Newton on `qnm_denominator` | 5-digit anchors carry a fixed 1.3e-6 nm error |
| `N_PTS`, `N_SIDE` | 40, 12; `N_SIDE_CONTOUR_LIMITED = 6` | — |
| `ATOL_RE_NM`, `ATOL_IM_NM` | 0.5, 0.32 → 1e-9, 1e-9 | TE0 1.1e-11, TE3 5.7e-13, TM0 7.1e-15 |
| `test_pole_error_is_first_order_in_resolution` | **replaced** by `test_pole_error_is_spectral_in_resolution`: nn 20, 24, 28 at 24 nodes/side, each step ×50, last < 1e-7 | 1.20e-3, 8.66e-6, 3.30e-8 |
| `test_quality_factor_matches_the_analytic_mode` | 0.02 → 1e-10 (derived from the λ budget) | 3.8e-13 |
| `test_refine_does_not_beat_discretisation` | **replaced** by `test_refine_removes_the_contour_error`: before > 1e-6, after < 1e-10 | 7.0e-6 → 1.3e-13 |
| `te_simple_refined` | now `te_simple_contour_limited.refine()` | — |
| `test_refine_polishes_the_mode_vector` | uses the contour-limited fixture | 4.8e-8 → 4.4e-16 |
| `test_left_null_vector_is_not_the_right_one` | evaluated at `ANCHOR_TE_SIMPLE + 0.5` nm | At the pole σ_min/σ_max = 1.3e-16, which makes a 1e-12 residual check pure round-off. At +0.5 nm: 4.9e-4, agreement 7.8e-15, ‖⟨u, conj v⟩‖ 0.32 |
| every other test | assertions kept; docstring numbers updated | sigma_ratio 1.1e-14; generic 2.0e-3 / 5.2e-3; conjugate 3.5e-2; rank 3, sv_ratio[1] 1.1e-6; cancellation 5.7e-11 vs 0.42; seed 1.4e-12; degenerate gap 6.8e-13, cond 1.5e15; bad box edge 0.033, σ 2.5e-7 → 7.3e-17, move 2.6e-4 |

### 5.7 `tests/test_scale_covariance.py`

| Item | Now | Measured |
|---|---|---|
| Shape rows | circle-θ, circle-arc, star-θ, star-arc, cusped-star-θ, one list for both matrix tests | — |
| `geometry(rad, shape, arc_length)` | rebuilds the arc-length map **at each `rad`** (tests the construction) | — |
| `ATOL_UNIFORM_THETA` / `ATOL_ARC_LENGTH` | 2e-14 / 5e-13 → 1e-14 / 2e-14 | ≤ 1.6e-15 / 4.3e-15 |
| exact-ratio test | asserts `theta`, `dw`, `ddw` bit-identical instead of `delt` | bit-identical |
| `test_a_cusped_boundary_can_lose_the_covariance_at_a_generic_ratio` | **deleted** | Cannot occur: uniform-θ nodes are rad-independent (measured 1.4e-15); conventions §9 records why |
| `test_reordered_theta_is_rejected…`, `test_coincident_arc_length_nodes…` | **deleted** | No θ arrays are accepted; closure moved to `test_geometry` |
| `test_frozen_theta_*` (2) | → `test_frozen_parametrisation_*` | same assertions on a frozen map |
| QNM covariance | `N_PTS_QNM = 40`, `N_SIDE = 12`; `RTOL_LAM` 5e-15 kept | ≤ 6.0e-16 |

### 5.8 `tests/test_sensitivity.py`

| Test | Now | Measured at `n_pts = 40` |
|---|---|---|
| `test_gate1_dilation_derivative_is_exact` | rel 1e-9 → 1e-8, floor re-derived as ε·rad/h ≈ 4e-9 (F6) | 5.8e-10 |
| `test_gate1_dilation_is_gauge_free` | kept (< 1e-8, rate 80–120) | 3.4e-9 |
| `test_sensitivity_rejects_a_re_inverted_node_set` | → `test_sensitivity_rejects_a_rebuilt_map`: an ellipse on its own arc-length map, rebuilt per δ | raises `"base node set"` |
| `test_gate1_degenerate_pair_does_not_split_under_dilation` | kept (1e-8) | split 3.1e-10; vs λ/rad 7.8e-10 |
| `test_gate2_*` (2) | kept | ratio 1.3e-12; vs λ 3.4e-9 |
| `test_gate3_adjoint_matches_re_extracted_poles` | kept | 4.495e-4 → 4.489e-6 → 4.493e-8 |
| `test_gate10_jacobian_is_first_order_and_richardson_is_consistent` | → `test_gate10_jacobian_converges_spectrally`: relative difference J(40) vs J(80) < 1e-8, J(60) vs J(80) < 1e-9 | 3.0e-10, 4e-11 |
| — | added `test_richardson_limit_is_exact_on_first_order_data_and_refuses_swapped_rungs` | keeps the exported estimator covered |

### 5.9 Untouched and passing

`test_beyn.py`, `test_mie_qnm.py`, `test_field.py`, `test_version.py`,
`test_public_api.py` (with the regenerated baseline), `conftest.py`.

---

## 6. `docs/conventions.md`

In the patch; summary so reviewers know what moved:

- **§5** Arrays are in `t`; `delt` is a derived property; `parametrisation`,
  `nodes`, `theta`; χ carries the Jacobian of `t`.
- **§9** The four degree-zero combinations restated for Kress, and the
  relative-truncation reason the arc-length map is covariant. The cusped-star
  knife edge is recorded as gone, with numbers. The "wrong in the first decimal"
  sentence becomes v0.5 history.
- **§10** Frozen *map*. The v0.6 rationale (gauge, 20 %, 1.2e-10) plus the v0.5
  `np.interp` mechanism as history. Cancellation floor `ε·p/h` for a length
  parameter.
- **§11** Guard wording, Gate 1–3 numbers at `n_pts = 40`, `‖M − Mᵀ‖/‖M‖ = 1.16`.
- **§12** `J` converges spectrally; `R` rungs rule kept, with the uniform-θ
  reading (9.2); v0.5 extrapolation as history.
- **§13** No longer tentative except for the adaptive density: 13.1 adds
  derivatives in `t` and nn parity; 13.2 adds Kress's λ-independence; 13.3
  restated for a smooth map; the default-map result from Appendix A.

---

## 7. Commits

Two commits, in this order, each green on its own. Subjects are short, bodies
say what changed. **No signature or co-author trailer** (owner's standing
rule).

**Commit 1** — additive:

    git add src/pysie2d/parametrisation.py tests/test_parametrisation.py
    git commit -m "feat(parametrisation): add the uniform-theta identity map" \
      -m "Parametrisation.uniform_theta() is w(t) = t through the ordinary nodes() path, bit-for-bit. nn_from_band is None for it."

This commit is green on its own *(verified: `e74c10f` plus only these two files,
218 passed, ruff clean)*, so `git bisect` stays usable. You do not need to
re-check it: pytest runs the working tree, not the index.

**Commit 2** — the break, everything else in the patch:

    git add -A src tests docs/conventions.md
    git commit -F- <<'MSG'
    feat!: Kress-Martensen quadrature on a frozen node map

    Replace the Maradudin diagonal self-patch with Kress-Martensen product
    quadrature on all four blocks, for both parities of nn, with the analytic
    wavenumber derivative and an independent loop reference. Geometry is sampled
    on a Parametrisation, equispaced in t, uniform theta by default.

    BREAKING CHANGE: Geometry.gielis(theta=...) is replaced by
    parametrisation=..., and arc_length= is removed. A shape derivative now
    freezes the node map, not the angles: pass
    parametrisation=result.geometry.parametrisation where v0.5 passed
    theta=result.geometry.theta. The angles alone cannot carry the map's
    derivatives, which the new quadrature needs, so accepting them would return
    a plausible wrong answer. Geometry(...) no longer takes delt, and its
    derivative arrays are with respect to the quadrature parameter t;
    assemble_matrix, assemble_matrix_dwn and assemble_matrix_reference no longer
    take delt. The default node map changes from uniform arc length to uniform
    theta, which converges faster under the new quadrature; for uniform arc
    length pass parametrisation=Parametrisation.gielis(...). Odd m now requires
    a == b and n2 == n3, since the boundary otherwise does not close. Results
    change everywhere: convergence is spectral instead of first order.
    MSG

Then `git push`. PR #16 is already open as a draft. Do not change its title, and
do not mark it ready.

---

## 8. Acceptance checklist

Run from the repo root after `git apply`, before committing. Every item must
pass exactly as stated. Commands are in code blocks so they can be copied as
they are.

**A1 — formatting.** Expected: "N files already formatted", no file listed.

    uv run ruff format --check .

**A2 — lint.** Expected: `All checks passed!`

    uv run ruff check .

**A3 — suite.** Expected: **258 passed**, 0 failed, 0 errors, about 20 s.

    uv run pytest -q

**A4 — the CI example scripts.** Expected: five lines `ok …`, no `FAIL`. The
scripts rewrite the tracked `figures/*.png`; the final `git checkout` discards
that, because regenerating figures is §11's job.

    for s in examples/*.py; do uv run python "$s" > /dev/null && echo "ok $s" || echo "FAIL $s"; done
    git checkout -- figures

**A5 — API baseline.** Expected: exactly the lines listed in §4.6, nothing else.

    git diff tests/api_baseline.txt

**A6 — removed API is gone.** Expected: no output.

    grep -rnE 'boundary_setup|_etoil|_uniform_arc_theta|_validated_theta|arc_length=(True|False)' src tests

**A7 — the circle anchor through the public API.** Expected: a number below
1e-14 (measured 1.2e-15).

    uv run python -c "from pysie2d import Geometry, Material, BIESolver; from pysie2d.reference import mie; import numpy as np; g = Geometry.gielis(200.0, 30, m=0); r = BIESolver(g, Material(1.5)).scatter(600.0); a, _ = r.far_field(4000); q = a[3999].imag / (r.wnum_bg * 400); ref = mie.efficiencies(2 * np.pi / 3, 1.5)['Q_ext_TE']; print(abs(q / ref - 1))"

**A8 — the cached weights are read-only.** Expected: the last line is
`ValueError: assignment destination is read-only`.

    uv run python -c "from pysie2d.kernels import _kress_weights as w; a = w(64); a[0] = 0"

**A9 — after both commits.** Expected: no output.

    git status --short

---

## 9. Traps

Each of these produces a plausible wrong answer rather than an error.

1. **Even-only weights at odd `nn`** (D6). 1.1 nm on a QNM, nothing raised.
2. **θ-derivatives on a graded map.** Kress needs `ẋ`, `ẍ` in `t`. Forgetting
   `f′·w″` is invisible on every circle test (`w″ = 0` there) and on uniform θ.
   Only `test_t_derivatives_match_spectral_differentiation` sees it.
3. **Using `theta` differences as quadrature weights** anywhere. The step is
   `2π/nn` in `t`, always. There is no per-node weight any more.
4. **Mutating the cached weights.** Shared by every assembly and every contour
   thread. They are read-only; keep them so.
5. **"Simplifying" `_j0_h0` to real-only**, or calling `j0` on a complex array.
   The QNM path dies silently or raises deep in Beyn (non-negotiable 1).
6. **Reordering expressions in one of `assemble_matrix` / `assemble_matrix_dwn`.**
   Rounding changes, and `np.array_equal` fails. That failure is the guard
   working, not a flaky test.
7. **Reference loop placing χ columns at `j + row`.** The χ offset is always
   `nn` (§4.1). The parity test catches it — it did, at 0.13 relative, while
   this spec was verified — so a parity failure of that size means block
   placement, not rounding.
8. **An `if identity:` shortcut in `Parametrisation.nodes`.** It breaks the
   no-special-case contract, and the identity already comes out bit-exact.
9. **Comparing only `theta` in the sensitivity guard.** Two maps can share
   angles and differ in `w′`, `w″`.
10. **Evaluating the left-null residual identity at the pole.** σ_min is at
    round-off there under Kress; test 0.5 nm off (§5.6).
11. **Trusting `efficiencies()["qext"]` beyond 1e-5.** F1 is a pre-existing index
    bug, not the solver. Probe accuracy with `far_field` at the exact forward
    index, as A7 does.
12. **Self-Green reference at default `n_max` close to the particle** (F2).

---

## 10. Found while measuring — out of scope, for the owner

| # | Finding | Evidence | Suggested owner |
|---|---|---|---|
| F1 | `ScatterResult.efficiencies` computes `nforw = int((2π − angle)/delthe)`, which at the default `n_angles = 3000` evaluates to 2998, one grid step (0.12°) off forward. It is exact at 1000, 2000, 4000. | Measured indices; `qext` error 4.5e-6 at every `nn` | far-field spec (G5): `round`, plus a test at 3000 |
| F2 | `reference.mie.self_green_cylinder` default `n_max` (Wiscombe) truncates the Graf sum near the particle | Re S floor 3.1e-6 TE, 2.4e-4 TM at d = 1.2a; closes to 5.5e-15 / 2.4e-14 at `n_max = 80` | owner: a `d`-aware default, or document |
| F3 | Under Kress, `qsca` from the existing uniform angular grid is already at round-off (≤ 2.7e-15 at `n_angles = 3000`) | §5.4 | G5 scope may shrink to F1 |
| F4 | `Parametrisation.gielis` on the exponent-1 star (`m = 6`, `n = 1/1/1`) raises an uncaught `ZeroDivisionError` in `_layer_bandwidth` (`kappa.size / inside` with `inside = 0`), not the documented `ValueError` | reproduced | parametrisation spec |
| F5 | `Parametrisation.gielis` on the arm-ratio-64 spike (`m = 6`, `n = 0.5/8/8`) fails Newton in 50 iterations; on `n = 3/3/3` it spends 0.2 s doubling to `N_f = 2²⁰` before raising | reproduced | parametrisation spec |
| F6 | The v0.5 Gate 1 docstrings quote the cancellation floor as ε/h = 1e-11; for `rad` in nm it is ε·rad/h ≈ 4e-9 | derived; measured 5.8e-10 | fixed in the patch (docstrings only) |
| F7 | `richardson_limit` is exported but no pysie2d quantity is first order any more | §5.8 | owner: keep or deprecate |
| F8 | Related to handoff §9, not a resolution of it: under Kress with an exact smooth `w`, uniform arc length is still slower than uniform θ on the `m = 6` 6/12/12 star, so `np.interp` is not needed for that ordering | Appendix A | recorded in doc B; §9 itself (shipped scheme) stays open by decision |

---

## 11. Merge blockers owned by later work

This spec leaves the tree correct and the prose stale. **Do not merge v0.6 until
all of these are done**; none is done here:

- `README.md`: the old call → new call migration (`theta=` → `parametrisation=`,
  `arc_length`, direct `Geometry(...)` construction), the first-order claims, and
  the frozen-node example around its §"sensitivity". Doc A §5.2 item 3 requires
  it in the release.
- `CLAUDE.md`: "Near-field quantities converge at first order in nn (hence
  nn = 1000 …)", the `Geometry`/`delt` conventions line, the performance-shape
  paragraph (Kress adds `jv` at complex λ: 1.45×), and the roadmap line.
- `docs/design/performance.md`: the "hankel1 is 98 %" split under Kress.
- `docs/qnm-guide.md`: the `n_pts`, contour and refine guidance.
- `examples/*.py` prose (e.g. "0.38 nm discretisation error at n_pts = 200") and
  the three README figures, regenerated.
- Default `n_pts` decision (doc A §7).
- `uv lock` after the releasing merge (CLAUDE.md, Dependencies).

---

## Appendix A — Default-map study (13 Sep 2026)

Script: [studies/kress_default_map.py](studies/kress_default_map.py). Probe:
`qext`, TE, `n_core = 1.5`, λ = 600 nm, relative error against uniform θ at high
`nn`. Bands: arc length 20–20, adaptive 20–100 (`Parametrisation.gielis`).
Columns are `nn`.

**Rounded squares and the §9 star** (`rad = 200`; reference `nn = 1280`; spread
between the uniform-θ and arc-length references in brackets):

| shape | map | 40 | 60 | 80 | 120 | 160 | 240 | 320 | 480 | 640 |
|---|---|---|---|---|---|---|---|---|---|---|
| m4 2/4/4 [9.6e-16] | θ | 1.5e-6 | 1.8e-9 | 9.5e-14 | 6.0e-16 | | | | | |
| | arc | 2.8e-5 | 3.1e-7 | 5.8e-9 | 2.0e-12 | 9.6e-16 | | | | |
| | adapt | 2.2e-6 | 1.4e-8 | 4.5e-10 | 1.7e-14 | 3.6e-16 | | | | |
| m4 6/12/12 [1.1e-15] | θ | 3.7e-3 | 5.4e-4 | 2.1e-5 | 8.9e-7 | 2.4e-8 | 1.5e-11 | 8.1e-15 | | |
| | arc | 8.6e-4 | 1.3e-3 | 4.1e-4 | 8.1e-5 | 1.5e-5 | 5.7e-7 | 2.2e-8 | 3.6e-11 | 6.4e-14 |
| | adapt | 5.2e-3 | 2.3e-3 | 8.8e-6 | 6.7e-6 | 5.2e-8 | 2.3e-10 | 3.2e-12 | 1.3e-15 | |
| m4 12/24/24 [4.2e-13] | θ | 1.5e-2 | 9.0e-3 | 1.8e-3 | 3.2e-4 | 5.7e-5 | 1.8e-6 | 5.6e-8 | 6.0e-11 | 7.1e-14 |
| | arc | 2.5e-3 | 1.3e-2 | 2.5e-3 | 1.3e-3 | 6.0e-4 | 1.2e-4 | 2.5e-5 | 1.2e-6 | 5.6e-8 |
| | adapt | 1.1e-2 | 8.4e-3 | 1.9e-3 | 3.1e-4 | 4.6e-5 | 1.4e-6 | 1.1e-7 | 9.6e-10 | 6.1e-12 |
| m4 20/50/50 [1.9e-6] | θ | 2.8e-2 | 1.5e-1 | 2.6e-2 | 1.4e-2 | 7.4e-3 | 1.9e-3 | 4.8e-4 | 3.2e-5 | 2.2e-6 |
| | arc | 3.8e-3 | 1.7e-1 | 6.6e-3 | 6.5e-3 | 5.4e-3 | 3.3e-3 | 1.9e-3 | 5.8e-4 | 1.8e-4 |
| | adapt | 8.6e-2 | 2.5e-2 | 2.5e-2 | 6.5e-3 | 7.9e-3 | 1.2e-3 | 9.0e-4 | 3.0e-5 | 6.8e-6 |
| m6 6/12/12 [5.3e-11] | θ | 4.9e-2 | 1.3e-2 | 1.9e-3 | 5.6e-4 | 1.3e-4 | 2.5e-6 | 6.0e-8 | 7.4e-11 | 1.2e-13 |
| | arc | 2.6e-2 | 6.8e-3 | 9.3e-3 | 3.0e-3 | 9.7e-4 | 4.1e-4 | 1.7e-4 | 8.0e-6 | 5.2e-7 |
| | adapt | 2.1e-2 | 8.6e-3 | 3.9e-3 | 8.8e-4 | 1.4e-4 | 3.2e-5 | 9.9e-6 | 5.4e-8 | 5.9e-10 |

Blank cells are at the round-off floor. On the 20/50/50 shape the two references
disagree at 1.9e-6, so only the uniform-θ column is meaningful below that.

**Spiky stars** (`m = 6`, n2 = n3 = 8, tip radius 400 nm; reference `nn = 1600`):

| n1 (arm ratio) | map | 40 | 60 | 80 | 120 | 160 | 240 | 320 | 480 | 640 | 960 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 (1.7) | θ | 8.9e-3 | 2.9e-4 | 2.9e-4 | 3.6e-6 | 2.5e-6 | 6.3e-10 | 9.3e-12 | 2e-16 | 2e-16 | 0 |
| | arc | 5.4e-3 | 9.9e-4 | 8.5e-3 | 1.6e-4 | 2.3e-4 | 5.2e-6 | 9.6e-6 | 8.3e-9 | 7.2e-10 | 3.4e-14 |
| | adapt | 1.0e-2 | 6.9e-4 | 2.9e-3 | 2.1e-5 | 2.6e-5 | 1.3e-9 | 2.1e-7 | 3.5e-11 | 1.5e-13 | 2e-16 |
| 2 (2.8) | θ | 2.3e-2 | 1.5e-2 | 6.5e-3 | 1.1e-3 | 5.8e-4 | 9.8e-6 | 6.7e-7 | 1.2e-9 | 8.3e-12 | 1.4e-16 |
| | arc | 6.3e-2 | 1.8e-2 | 1.2e-2 | 6.1e-3 | 1.0e-3 | 1.0e-3 | 8.7e-4 | 5.3e-5 | 1.0e-5 | 2.5e-7 |
| | adapt | 3.5e-2 | 9.0e-3 | 1.0e-2 | 2.9e-3 | 1.2e-3 | 5.1e-4 | 3.8e-4 | 1.9e-5 | 3.2e-6 | 3.3e-8 |
| 1 (8) | θ | 4.0 | 1.2 | 1.0e-1 | 1.5e-1 | 2.1e-2 | 9.1e-3 | 9.4e-4 | 6.4e-5 | 2.2e-6 | 5.8e-9 |
| | arc | 2.0e-1 | 8.8e-2 | 1.4e-2 | 5.0e-3 | 2.0e-2 | 9.2e-4 | 6.9e-3 | 2.6e-4 | 5.4e-4 | 3.1e-4 |
| | adapt | 1.4e-1 | 7.0e-2 | 3.5e-2 | 2.4e-3 | 9.3e-3 | 5.4e-3 | 7.8e-3 | 2.3e-3 | 4.7e-4 | 2.4e-4 |
| 0.5 (64) | θ vs θ(1920) | 80: 1.6 · 120: 3.9 · 160: 7.4e-2 · 240: 9.6e-1 · 320: 4.8e-2 · 480: 5.9e-2 · 640: 1.7e-3 · 960: 4.2e-4 · 1280: 4.3e-6 | | | | | | | | | |
| | arc, adapt | `Parametrisation.gielis` fails (F5) | | | | | | | | | |

For n1 = 1 the uniform-θ reference is self-converged to 2.8e-14 (1600 against
1920). The arc-length column therefore genuinely stalls, and it is non-monotone,
which conventions §8 reads as a defect signal. The map is not the defect: it
matches an independent quadrature inversion to 2.7e-15, with density residual
6.6e-14. Node spacing at `nn = 160`: uniform θ has max/min 38, largest gap
74.6 nm on the arm flanks; arc length is uniform at 27.2 nm.

**Reading.** The v0.5 argument for arc length — uniform θ undersamples arms —
is true and measurable, but only while uniform θ's worst node is badly
under-resolved: `R ≲ 8` on the arm-ratio-8 spike, where arc length sits near
`R ≈ 22`. Past that, the analyticity strip decides, and it favours the identity.
If adaptive grading earns a place anywhere, it is that pre-asymptotic spiky
regime, and even there it is not better than plain arc length. This is G3's
question, not this spec's.

**Correction (14 Sep 2026).** The pre-asymptotic arc-length win read above
does not survive more wavelengths: below ~1e-2 every map's error swings 3–10×
between neighbouring `nn`, so the winner at one rung is noise. Uniform θ is the
recommendation at every resolution; see `studies/spiky_low_nn.py` and the study
plan. G3 found grading's shape class elsewhere: flat-sided near-corners.

## Appendix B — Where each number came from

All scratch scripts ran against `e74c10f` with Kress monkeypatched in or on the
patched tree. The committed, reproducible ones:

| Numbers | Source |
|---|---|
| Appendix A | `studies/kress_default_map.py` (reproduces circle 6.7e-16 at nn 30/31 and m6 θ 6.0e-8 at 320) |
| §5 constants | the patched tree's own test suite; every constant's comment quotes its measurement |
| §3.6 cost | patched `assemble_matrix` vs `git show e74c10f:src/pysie2d/kernels.py`, best of 7, `OMP_NUM_THREADS=1`, complex λ = 530.8 + 26.4i |
| D6, D9, F1–F5 | reproduced in scratch during this spec. The commands are one-liners over the patched API: `Geometry.gielis`, `Parametrisation.gielis`, `QNMResult.sensitivity` with the guard bypassed for D9 |
