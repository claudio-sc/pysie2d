# Cluster reduction — code-spec (draft 1)

**Status: implemented 2026-11-06, with the two changes in §9. Merge is the owner's decision.** Study: [studies/cluster-reduction/](studies/cluster-reduction/README.md).
Every number below is from there.

**Goal.** Two additions on one operator, both at a fixed wavelength:

1. **Driven (Part A).** Solve a cluster for many material sets, at a small fraction
   of today's cost, with the same answer as the dense solve.
2. **Modes (Part B).** Eigenpermittivities of a cluster: the complex ε at which it
   supports a field with no source, in free space and over any background.

**Additive.** `ClusterBIESolver.scatter*`, `_assemble` and every existing result
stay bit-identical. Nothing here replaces the dense path.

## 1. Scope

**In.** Parts A and B; `pol` 1 and 2; free space, `HalfSpace` (PEC included) and
`Stack`; mixed shapes, `nn` and materials in Part A; a new conventions §18; notes
on the two existing behaviours in §6.

**Out, not to be worked around:**

- **Routes 2 and 3 and the mode expansion** (the prototype's eqs. 21′, 26′). On clusters
  they miss eigenvalues by |Δε| up to 0.5 and 3.5, and route 2's contour fails over the
  slab.
- **Changing `scatter`, or using reduction by default.** A one-off solve only wins
  above `P` ≈ 20–30.
- **Slab assembly cost** (6–8× per doubling of `P`). A real problem, and a separate one.
- **Deduplicating identical particles.** See D5.
- **A cluster `QNMSolver` in λ.** D12 and spec v0.8 §1.3 stand: this feature varies ε
  at fixed λ and needs no half-plane argument.
- **Fixing the irregular frequency or the H_n zeros.** Documented only.

## 2. Decisions

| | Decision | Reasoning |
|---|---|---|
| D1 | **Entry point `ClusterBIESolver.at_wavelength(wavelength)`** returning a `ClusterEigenEps`. | The solver already holds the cluster, `pol` and background, and validates `n_clad`. Names per §7. |
| D2 | **`ClusterEigenEps.scatter(materials, angle=0.0)`** and `scatter_dipole`, `scatter_mode`, mirroring the solver with `materials` first; all return an ordinary `ClusterScatterResult`. | Every observable then works with no new code. `materials` is required: a default would hide which set a sweep solved. |
| D3 | **Exterior rows are built one particle at a time** and discarded after compression. Never call `_assemble` from the sweep. | Otherwise the memory saving (380 MB vs 2.6 GB at `P` = 64) does not exist. |
| D4 | **SVD compression at `1e-13`,** relative to the block's largest singular value. | Tol 1e-4 buys 1.3× per ε and costs nine digits (study `s1` C). |
| D5 | **No identical-particle deduplication.** One factorisation per particle per call. | Worth 2.5–5× per ε, and needs a rule for "identical" that holds over a substrate. Add it if a measured use needs it. |
| D6 | **Always refine once.** Superseded the a-priori guard; see §9. | The step size measures the error directly, for any right-hand side, at the cost of one more apply. |
| D7 | **Modes run Beyn on `G`, not `M`.** | `G⁻¹ = I − Y·M⁻¹·U` is meromorphic with poles only at cluster modes (§3.3), and far cheaper per node. Gate B5 checks it against `M`. |
| D8 | **Upper branch for modes:** `√ε → i·√(−ε)`, cut on `[0, ∞)`. A contour meeting that ray raises. | The principal branch has exact singularities at `H_n^{(1)}(√ε·x) = 0` with no physical mode. The two branches coincide for Im ε > 0, so driven solves are unchanged. |
| D9 | **Public ε is absolute,** like `Material.from_eps`; internals are background-relative. Converted once, in `eigenpermittivities`. | Conventions §2's rule, applied to ε. Owner, §7. |
| D10 | **Modes share one ε across the cluster.** | What the anchors can check. Per-particle ε is a different eigenproblem. Owner, §7. |
| D11 | **Modes require a real wavelength;** driven accepts complex. | Non-negotiable 1 covers every assembly path, so `G` and the sweep take complex `k`. A modal ε at a complex λ has no meaning here, so that entry point refuses it. |

## 3. The mathematics

### 3.1 Splitting

For particles `p = 1…P` in the conventions §14 layout, each with `nn_p`:

    M(ε) = D(ε) + U·Y

- `D = diag(M_p(ε))`, `M_p` a `2nn_p × 2nn_p` block: exterior rows from free space
  plus particle `p`'s own substrate reflection; interior rows at `k_core = ri·k_bg`.
- For each `p`, the neighbour coupling `C_p` is the `nn_p × N` exterior rows of
  particle `p` with its own columns zeroed. Its SVD gives `U_p` (`nn_p × r_p`) and
  `Y_p = U_pᴴ·C_p` (`r_p × N`), with `r_p` = count of `σ > 1e-13·σ_0`. If `C_p` is all
  zero (one particle), `r_p = 0`.
- `Û_p = [U_p; 0]`, `2nn_p × r_p`. The interior rows of a coupling block are exactly
  zero (conventions §14), which is what makes this valid.

Only the interior rows of `D` depend on ε. `U`, `Y` and the exterior rows of `M_p`
are built once per wavelength.

### 3.2 Solve

For `b` split as `b_p` (both exterior and interior rows, so refinement works):

    a_p = M_p⁻¹·b_p
    R_p = M_p⁻¹·Û_p
    G[p,q] = δ_pq·I + Y_p[:, slice(q)]·R_q      (each block r_p × r_q)
    g_p = Σ_q Y_p[:, slice(q)]·a_q
    y = G⁻¹·g
    x_p = a_p − R_p·y_p

**Refinement** (D6): with `α = max_p ‖a_p‖/‖x‖`, while `α > 1e3` and steps < 3:
`r = b − (D + U·Y)·x`, solve for `Δx` with the same factors, `x += Δx`. Stop early
when `‖Δx‖/‖x‖ < 1e-13`. If the last `‖Δx‖/‖x‖ > 1e-10`, warn
`ClusterCancellationWarning` naming `ClusterBIESolver.scatter` as the alternative.
Do not fall back automatically: the dense matrix may not fit (D3).

### 3.3 Modes

`G(ε)` is square of size `R₀`, and Woodbury gives

    G⁻¹ = I − Y·M⁻¹·U

So `G⁻¹` is holomorphic at isolated-particle poles and has poles only at cluster
modes. A mode with `M v = 0` has residue `∝ Y v`, which vanishes only if its field
is zero on every neighbour's boundary, hence throughout the exterior. Truncation
below `1e-13` is the one remaining way to lose it, which is why gate B6 exists.

From a null vector `y` of `G(ε_c)`, the mode is `x_p = −R_p·y_p`, normalised to unit
2-norm. With one particle, `R₀ = 0` and `G` is empty: use `M_1` directly.

## 4. The implementation

### 4.1 `kernels.py`

    def _interior_rows(pol, nn, f, g, df, dg, ddf, ddg, wnum_bg, ri, kd) -> np.ndarray:
        """complex (nn, 2nn): assemble_matrix's lower half, by the same expressions."""

A deliberate duplicate of the core half of `assemble_matrix`, expression for
expression and in the same order, with the same `WARNING` comment
`assemble_matrix_dwn` carries. `assemble_matrix` is not refactored. Keep
`_real_if_real`: real ε > 0 must take the Cephes branch.

### 4.2 `cluster.py`

Beside the existing classes:

    class ClusterCancellationWarning(UserWarning): ...

    class ClusterBIESolver:
        def at_wavelength(self, wavelength: float | complex) -> "ClusterEigenEps": ...

    class ClusterEigenEps:
        def scatter(self, materials, angle=0.0) -> ClusterScatterResult: ...
        def scatter_dipole(self, materials, x_s, z_s, *, moment=None) -> ClusterScatterResult: ...
        def scatter_mode(self, materials, mode, direction, *, x_ref) -> ClusterScatterResult: ...
        def eigenpermittivities(self, eps_lo, eps_hi, *, n_quad_per_side=32, n_probe=16,
                                rank_tol=1e-8, tol=1e-12, max_iter=30) -> EigenpermittivityResult: ...

    @dataclass(frozen=True)
    class EigenpermittivityResult:
        eps: np.ndarray          # complex (K,) absolute, sorted as BeynModes
        vectors: np.ndarray      # complex (n_dof, K), conventions §14 layout, unit columns
        converged: np.ndarray    # bool (K,) from newton_refine
        rank: int                # BeynModes.rank, before the in-contour filter
        sv_ratio: np.ndarray     # BeynModes.sv_ratio: the rank gap, visible

**Refactor without behaviour change:** move the right-hand-side code out of
`scatter`, `scatter_dipole` and `scatter_mode` into private
`_rhs_plane_wave(wavelength, angle)`, `_rhs_dipole(wavelength, x_s, z_s, moment)`
and `_rhs_mode(wavelength, mode, direction, x_ref)`, so both paths share it.

**Exterior rows** go in a private `ClusterBIESolver._exterior_rows(q, wavelength, refl)`
returning `nn_q × N`, assembled from the same calls as `_assemble`. The self block's
exterior half is `assemble_matrix(pol, nn, …, k_bg, 1.0, 1.0)[:nn]`: those rows do not
depend on `ri` or `kd`.

**Materials on each call:** validate exactly as `__init__` does (factor the check
out), and additionally that `n_clad` equals the solver's. Run `_check_resolution` per
call and `_check_gap` once at construction.

**`eigenpermittivities`:**

1. Raise `ValueError` for a complex wavelength (D11).
2. Divide `eps_lo`, `eps_hi` by `n_clad²`. Raise `ValueError` if the rectangle meets
   `[0, ∞)`, that is `hi.imag ≥ 0`, `lo.imag ≤ 0` and `hi.real ≥ 0` (D8). `ε = 0` is an
   exact TM singularity and is on that ray, so this covers it.
3. `beyn_modes` on `G`, passing `max_workers=None`, `rank_tol`, `n_quad_per_side`,
   `n_probe`.
4. Polish each with `newton_refine` on `G`, central difference `h = 1e-6` for `dG/dε`;
   this is the prototype's step and measured sufficient.
5. Reconstruct `x` by §3.3 and return.

Interior rows use `ri = 1j*np.sqrt(-complex(eps))` and `kd = eps`, both
background-relative.

### 4.3 Exports and docs

- `__init__.py`: `ClusterEigenEps`, `ClusterCancellationWarning`, `EigenpermittivityResult`.
- `docs/conventions.md` **§18**: the §3.1 split, D8's branch, D9's absolute/relative
  rule, why `G` is a legal Beyn operator, the refinement rule.
- `docs/conventions.md` §1 (or the nearest fitting place): a note on the exterior
  irregular frequency, with the measured window. `docs/qnm-guide.md`: a note on the
  near-singular points at `H_n^{(1)}` zeros, stating that `QNMSolver` was not shown to
  report them. Run it first and state which.
- `performance.md`: a §8 with the table from the study, including the slab-assembly
  finding.
- `README.md`: a short section under *Clusters of particles*. `CLAUDE.md`: roadmap.

## 5. Gates

Tolerances rule: **measure, then set**, per non-negotiable 4. The "measured" column is
the study's figure; the builder re-measures on the final code, and the test's tolerance
cites that number and its reason. Never widen one to pass.

### Part A

| Gate | Check | Measured |
|---|---|---|
| A1 | `_interior_rows` `array_equal` to `assemble_matrix(...)[nn:]`: `pol` 1, 2; real ε > 0 (Cephes), complex ε, complex `k_bg` | new |
| A2 | Self-exterior rows do not depend on material: `array_equal` over three materials | new |
| A3 | `_exterior_rows` `array_equal` to `_assemble`'s exterior rows, all four backgrounds, ragged `nn` | new |
| A4 | `ClusterEigenEps.scatter` vs `ClusterBIESolver.scatter`: `x`, far field, on mixed particles; free, glass, PEC, n=2; both pols | `x` ≤ 1.4e-13, far field ≤ 1.6e-15 |
| A5 | Gap ladder to gap/a = 0.04 with `nn` from the envelope | ≤ 4.7e-13 |
| A6 | Independent anchor: `ClusterEigenEps.scatter` far field vs `reference.two_cylinder` | as `test_two_cylinder.py` |
| A7 | One-particle cluster equals `BIESolver.scatter` | new |
| A8 | Complex wavelength: sweep vs `_assemble` + solve | new |
| A9 | Guard: at `|ε−ε_iso|` = 1e-6, 1e-10, 1e-12 the result matches dense to refinement's floor, and refinement ran. **Must fail with refinement disabled** | 1.2e-13 to 1.4e-13; unrefined 3.3e-11 to 4.1e-5 |
| A10 | `ClusterCancellationWarning` fires at ε exactly on an isolated pole | new |
| A11 | Cost: `P` = 16, distinct particles, per ε, against cached-dense; assert a ratio with stated margin | 8× |
| A12 | Existing suite passes unchanged; `scatter*` bit-identical after the RHS refactor | existing tests |

### Part B

| Gate | Check | Measured |
|---|---|---|
| B1 | One circle vs analytic Mie denominators, TM and TE; count checked by winding number | 1e-15 at `nn` = 32 (prototype; re-run) |
| B2 | Two circles vs a Graf determinant, TM and TE, polished; d = 400, 1000. Graf matrix is built in the test, from `reference.mie` | TE ≤ 7.9e-14, TM ≤ 2.9e-14 |
| B3 | PEC mirror: every PEC mode in the free-space circle + image pair; both pols, h = 200, 500 | ≤ 1.9e-15 |
| B4 | **Slab:** pole fitted to passive `BIESolver` solves against `eigenpermittivities`. Uses `Material.from_eps`, no contour, no branch | 5.9e-7 to 1.3e-5 |
| B5 | `G` vs `M` on the same box: same count, polished eigenvalues agree. `M` built in the test | ≤ 8e-16 TM circles; ≤ 4.6e-15 triangles, free and slab |
| B6 | High-Q cannot hide: whispering-gallery ladder at d = 6000 and the Graf roots at d = 2500; counts equal, `M` and `G` singular at each root | 4 = 4 to Im ε = 5.6e-13; 2 = 2 to Q = 2.7e6 |
| B7 | Box on `[0, ∞)` raises; box containing `ε = 0` raises; complex wavelength raises | new |
| B8 | Absolute/relative: at `n_clad` = 1.444 the same physical mode results from either scaling by hand; `Material.from_eps(result.eps)` makes the interior rows singular with `G` | new |
| B9 | A box with no rank gap raises as `beyn_modes` does, loudly | observed in the study |

**A transparent stack must not be used as a gate.** `_effective` turns it into no
background, so it cannot fail (study `s8` K).

## 6. Existing behaviour to document, not fix

- **Irregular frequency.** At k_bg·a = j₀₁, exterior rows lose rank for every ε;
  `Q_ext` silently wrong by 17 % (TE), 33 % (TM) at exactly that λ, 5e-11 at δλ/λ = 1e-6.
  Whether the left null vector survives in a cluster is argued, not measured.
- **`H_n^{(1)}` zeros in the λ-QNM operator.** σmin/σmax 2.8e-8 at Q ≈ 0.17.

If either turns out to matter, it is a new decision for the owner, not this milestone.

## 7. Owner decisions

Taken 2026-11-06.

1. **Names.** `ClusterEigenEps` (owner's choice, replacing the proposed `ClusterSweep`),
   `at_wavelength`, `eigenpermittivities`, `EigenpermittivityResult`. The owner named only
   the class; the other three stay as proposed. The class docstring and the README
   section must state the measured speedup and why it suits repeated solves at one
   wavelength, since the name no longer says so.
2. **Absolute ε in the public API (D9).**
3. **One shared ε (D10).**
4. **Modes over a `Stack` ship with a clear caveat.** B4 is the gate. The
   `eigenpermittivities` docstring, the README section and conventions §18 all state that
   over a `Stack` the result has been checked only against this package's own driven
   path, with no external reference. No MEEP comparison in this milestone.

## 8. Builder contract

**Order**, one commit per step, Part A complete and green before Part B starts:

1. `_interior_rows` + A1, A2.
2. `_exterior_rows` + A3.
3. RHS refactor + A12.
4. `ClusterEigenEps.scatter*` without refinement + A4–A8.
5. Guard and refinement + A9, A10.
6. Cost gate A11, `performance.md`.
7. `eigenpermittivities` + B1, B7, B8, B9, conventions §18.
8. B2, B3, B5, B6.
9. B4.
10. Docs, exports, README, `CLAUDE.md`.

**Stop rule.** On a failed gate, a measurement contradicting this spec, or any
ambiguity: stop and report. Do not widen a tolerance, change a gate or improvise a
design. Do not touch `assemble_matrix`, `assemble_matrix_dwn`, `_assemble`, `beyn.py` or
`qnm.py`.

**Commits.** `feat:` for the public additions, `test:` and `docs:` as appropriate.
Merging to `main` releases, so that is the owner's decision.

## 9. Changes made during the build

The stop rule says stop and report. These are reported here and in the hand-over; the
spec's author made them, on measurements:

1. **The `‖a‖/‖x‖` guard of D6 and §3.2 was wrong, and is gone.** The study's "error tracks
   `1.5e-14·‖a‖/‖x‖`" held only for its random right-hand side. A plane wave that does not
   drive the resonance keeps the ratio at 0.26 while the error reaches 5e-5, so the guard
   never fired. Gate A9, written with a physical plane wave, caught it. Now: refine once
   always, up to `MAX_REFINE_STEPS = 3`, stop below `1e-12`, warn above `1e-10`. Cost is
   within measurement noise of the unrefined probe, because the small matrix is factorised
   once.
2. **Polish uses the chain rule, not a difference of `G` (§4.2 step 4).** `G` has poles
   closer than `h = 1e-6` at high Q. `dG/dε = −Y·D⁻¹·(dD/dε)·D⁻¹·U`, differencing only the
   interior rows.

Also: `_rhs_plane_wave` takes an optional reflection context, because rebuilding one per
material tripled the slab cost; gate A5's bound is `5·cond(M)·eps`, because the measured
difference scales with the conditioning (2.3–2.8 × cond·eps over all rungs) where the study's
flat figure came from a random right-hand side; and B4 gained a dielectric half-space case.
