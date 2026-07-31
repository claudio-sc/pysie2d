# Beyn QNM port — status, gotchas, and remaining work

> **Superseded in part, 2026-07-31 (later the same day).** §3.1 (commit the
> tree) is done, and §3.3 — the D1 vacuum-wavelength fix, together with D5
> (`Material.epsi` absolute) — has since **shipped**. Gotcha 1 below is
> therefore resolved: the caveat Note it refers to has been removed from
> `docs/conventions.md` §8, `tests/test_conventions.py` now exists, and the
> `test_efficiencies.py` double-count is fixed. Public wavelengths are vacuum
> everywhere and the primitives take `wnum_bg`. Everything else below —
> Gotchas 2–8 and §§3.4–3.8 — still stands.

Handoff note, 2026-07-31. Written to be read **on its own**, then alongside the
two design documents it tracks.

## Context documents

Read in this order:

| Document | What it is | How to treat it |
|---|---|---|
| `docs/design/beyn-port-spec.md` | The implementation spec, draft 2. §§1–5 are decided and now largely *built*; §§6–13 are still forward-looking. | **The authority on intent.** Section numbers referenced throughout below. |
| `docs/design/beyn-port-strategy.md` | The earlier high-level strategy. | **Superseded.** Kept for provenance only; the spec corrects several of its numbers (notably §5's "no additional special-function evaluations" and the 3× saving claim). |
| `docs/design/qnm-methods.pdf` | Background on QNM extraction methods. | Reference. |
| `docs/conventions.md` §8 | The shipped convention text for the QNM half-plane. | **Normative.** Contains a live caveat — see Gotcha 1. |
| `CLAUDE.md` | Repo working agreement. | Non-negotiables 1–4 apply to everything below, especially #3 (independent validation anchor) and #4 (justify every tolerance). |

---

## 1. Status: the port is built and green, and entirely uncommitted

`uv run pytest` → **73 passed in 86 s**. Of those, 43 are new QNM/Beyn tests.
Baseline before this work was 55 s, so the feature costs ~31 s of CI — the
optimistic end of the spec §13 Q5 estimate (+60–90 s), and no `slow` marker was
needed.

**Nothing is committed.** Working tree state:

```
untracked:  src/pysie2d/beyn.py            536 lines
            src/pysie2d/qnm.py             255
            tests/test_beyn.py             378
            tests/test_qnm.py              293
            tests/test_mie_qnm.py          195
            docs/design/                   (this file + spec + strategy + pdf)

modified:   src/pysie2d/reference/mie.py   +265   (qnm_denominator, qnm_size_parameters,
                                                   qnm_wavelengths, complex annotations)
            docs/conventions.md            +45    (§8, the QNM half-plane)
            src/pysie2d/__init__.py        +6     (QNMSolver, QNMResult exported)
            src/pysie2d/solver.py          +7     (assemble accepts complex λ; docstring)
            tests/conftest.py              +6     (QNM_N_CORE = 3.0)
```

First action for whoever picks this up: **commit it.** A day of work sits
untracked, and the repo convention is conventional commits parsed by
semantic-release.

### Phase status against the spec

| Spec phase | State | Evidence |
|---|---|---|
| §2 Phase 0 — D1 vacuum-wavelength fix | **Deliberately deferred to v0.4.0** | `conventions.md` §8 carries an explicit Note; see Gotcha 1 |
| §3 Phase 1 — analytic anchor | **Done** | `tests/test_mie_qnm.py`, 9 tests |
| §4 Phase 2 — Beyn on synthetic pencils | **Done** | `tests/test_beyn.py`, 18 tests |
| §5 Phase 3 — extractor + circle anchor | **Done** | `tests/test_qnm.py`, 16 tests |
| §6.1 Phase 4 — analytic `dM/dλ` | **Not started** | no `assemble_matrix_dwn`, no `assemble_derivative` |
| §6.2 Phase 5 — `QNMResult.refine()` | **Half** | `beyn.newton_refine` exists and is tested on synthetic pencils; not reachable from the façade |
| §6.3 — `σ_min` by inverse iteration | **Not started, knowingly** | `qnm._sigma_ratio` does a full SVD and says so in its docstring |
| §7 Public API | **Shipped, with drift** | see Gotcha 3 |
| §11 `examples/qnm_spectrum.py` | **Missing** | `examples/` has only the three v0.2 scripts |
| §12 Repo drift | **Unfixed** | see Gotcha 5 |

### What is actually implemented

`src/pysie2d/beyn.py` — EM-free, callable-driven:
`rect_contour_quad`, `probe_matrix`, `contour_moments`, `_detect_rank`,
`beyn_poles`, `beyn_modes`, `newton_refine`, and the `BeynModes` / `RefinedMode`
result types.

`src/pysie2d/qnm.py` — the façade: `QNMSolver(geometry, material).modes(z_lo,
z_hi, *, n_quad_per_side=12, n_probe=12, rank_tol=1e-8, rng_seed=0)` returning a
frozen `QNMResult`. It composes a `BIESolver` internally and calls
`BIESolver.assemble` — deliberately the driven solver's own matrix, because the
claim "a QNM is a singularity of the scattering operator" only holds if the two
are literally the same assembly.

`src/pysie2d/reference/mie.py` — the analytic anchor: `qnm_denominator`,
`qnm_size_parameters`, `qnm_wavelengths`, plus complex-argument annotations.

---

## 2. Gotchas

Ordered by how much damage they do if missed.

### Gotcha 1 — D1 is deferred, and the deferral has two loose ends

`docs/conventions.md` §8 ends with an explicit Note: the public API documents
**vacuum** wavelengths, the implementation still forms `k = 2π/λ` (the **medium**
reading), and reconciling them is a breaking change shipping separately in
v0.4.0. That is a defensible call — it keeps the QNM release's validation story
clean, matching the spec's own §13 Q2 recommendation (a).

It is safe *today* only because every test in the suite, including the QNM
anchor, runs at `n_clad = 1.0`, where the two readings coincide. Two things were
left behind:

1. **`tests/test_efficiencies.py:29` and `:50` still read
   `m = complex(mat.nc) / N_CLAD`.** `mat.nc` is *already* `n_core/n_clad`, so
   this is a double-count under either convention — spec §2.2 calls it a genuine
   bug independent of D1. It passes only because `N_CLAD = 1.0`.
2. **`tests/test_conventions.py` does not exist.** Spec §2.5's
   `test_vacuum_wavelength_scaling` and the `n_clad ∈ {1.0, 1.33}`
   parametrisation are both absent, so nothing guards the convention in either
   direction.

### Gotcha 2 — the analytic `dM/dλ` was never written, so refinement is unreachable

`beyn.newton_refine` takes `dm_builder: Callable[[complex], np.ndarray]` as an
argument. The algorithm side is complete and tested; the EM side that would
supply that callable does not exist. Consequently `QNMResult` has no `refine()`
and no `converged` field, and spec §6.2's three façade tests
(`test_refine_is_idempotent_at_convergence`,
`test_refine_recovers_from_coarse_contour`, `test_refine_flags_degenerate_pole`)
are unwritten.

This also blocks §6.3: the cheap `σ_min` by inverse iteration wants the LU that
refinement would already have formed.

Per spec §6.2 this is *insurance, not accuracy* — on a well-resolved simple pole
the Beyn estimate is already converged to 13 digits and Newton makes a no-op
step. So the feature is honest without it; it is the coarse-contour recovery
path that justifies shipping it.

### Gotcha 3 — `QNMResult` drifted from spec §7, mostly for the better

Gained (not in the spec, both good): `rank`, and `cancellation`, a diagnostic for
how completely the contour integral cancelled.

Lost: `converged` (follows from Gotcha 2).

**`rank` can legitimately exceed `n_modes`.** A pole just *outside* the contour
leaks a rank direction in through imperfect quadrature cancellation; its
eigenvalue is then discarded by the in-contour filter. This is tested
(`test_rank_may_exceed_mode_count_from_outside_leakage`) and is **not a bug** —
do not "fix" it by forcing the two to agree.

### Gotcha 4 — the Phase-1 root finder's `Im x` floor is a Q ceiling

`qnm_size_parameters` seeds Newton from a grid whose imaginary axis bottoms out
at `Im x ≈ 1e−4`, which cannot see a mode with `Q ≳ 1.5e4`. It does not bite at
the current fixture — the winding-number check confirms nothing was missed — but
it is closer than it looks: the TE ladder runs `Q = 164, 598, 2289` for
`n = 4, 5, 6`, i.e. `Im x` falling ~3.4× per order. Two more orders, or a higher
`m`, crosses the floor **silently**.

The mitigation is already the design: `test_root_count_matches_winding_number`
asserts completeness independently of the seeder. Keep it that way — a
completeness claim the seeder makes about itself is not a check.

### Gotcha 5 — three different version stories in one repo

- `pyproject.toml` → `0.3.0`
- `src/pysie2d/__init__.py` → `__version__ = "0.2.0"`, and
  `tests/test_placeholder.py` asserts the stale value, so CI is green on the drift
- `docs/conventions.md` §8 heading says **(v0.5)**
- `CLAUDE.md` roadmap says **v0.3 = QNM extraction**, but v0.3.0 shipped as a
  performance release

Pick one story before releasing. Spec §12 also notes `CHANGELOG.md` has a
hand-written `## Unreleased / Performance` block for work already shipped in
v0.3.0, and `README.md` figure URLs are pinned to `/v0.2.0/` so a new figure
would 404.

### Gotcha 6 — a rectangle in `x` is not a rectangle in `λ`

The Phase-1 completeness statement is made over a box in the **size parameter**
`x`. `λ = 2π·rad/x` is a Möbius map, so it does not carry over to a `λ`-rectangle
without checking the corner mapping. The public API searches `λ`-rectangles. Any
new test box must be argued for completeness in its own coordinates.

### Gotcha 7 — the TE `n=0` anchor box is crowded in `Re λ`, isolated in `Im λ`

`530.83 + 26.38j` has TE `n=5` at `505.68 + 0.42j` and TE `n=2` at
`550.47 + 20.24j`, only ~20 nm away in `Re λ`. What separates them is `Im λ`.
The working box is `Re λ ∈ [520, 545]`, `Im λ ∈ [15, 40]`. Drawn generously in
`Im`, it swallows the `Q = 598` mode and the rank-1 assertion fails for a reason
that has nothing to do with Beyn. TE `n=3` at `760.69 + 7.95j` is far more
forgiving.

### Gotcha 8 — conventions that will bite anyone carrying real-eigenvalue intuition

- **`Im λ > 0`** for decaying modes; `Q = +Re λ / (2 Im λ)`. The research code's
  sign is wrong relative to the operator it calls — spec §10 lists "the sign
  convention" under *stays behind*.
- **Poles do not come in conjugate pairs.** The reality condition is `λ → −λ̄`,
  putting mirror partners at negative `Re λ`.
- **`Re λ > 0` is load-bearing, not cosmetic** — it keeps Hankel arguments off
  the `H^{(1)}` branch cut, which is what makes `M(λ)` holomorphic and the
  contour argument valid at all. Both bounds are asserted.
- Degeneracy is structural: every `n ≥ 1` circle mode is doubly degenerate via
  `exp(±inθ)`; only `n = 0` is simple.
- `D^{TM}_0 ≡ D^{TE}_1` identically, so those families land on the same
  wavelengths. Harmless, because `M(λ)` is assembled per polarisation — but it
  means a spectrum plot overlaying both polarisations shows coincident lines.

---

## 3. Work to be done

Ordered. Each item states its acceptance criterion, per CLAUDE.md's "define the
check before the work".

### 3.1 Commit the working tree — *immediately*

Conventional commits; semantic-release parses them. Suggested split: one commit
for the `reference/mie.py` anchor + `test_mie_qnm.py`, one for
`beyn.py` + `test_beyn.py`, one for `qnm.py` + `test_qnm.py` + exports +
`conventions.md` §8.
**Check:** `git status` clean, suite still 73 green.

### 3.2 Answer the six open questions — *blocks 3.3*

Spec §13, still unanswered. Q1 (`Material.epsi` absolute or relative) is the only
hard blocker; the spec recommends **absolute**, shipped in the same commit as D1,
with an `n_clad ≠ 1` lossy test added. Q6 asks permission to edit `CLAUDE.md`.
**Check:** each of the six has a recorded decision in the spec.

### 3.3 Ship D1 as v0.4.0, standalone

Convert `λ_vac → λ_med` at each public entry point; leave internals
non-dimensionalised (spec §2.2 — do **not** thread `n_clad` through all seven
internal sites). Fix the `test_efficiencies.py:29,50` double-count. Add
`tests/test_conventions.py`. Remove the caveat Note from `conventions.md` §8.
**Check:** `test_vacuum_wavelength_scaling` passes at `1e-12`;
`test_efficiencies_match_mie` parametrised over `n_clad ∈ {1.0, 1.33}` passes at
the **unchanged** `RTOL_MIE = 5e-3`; rest of the suite numerically unchanged.

### 3.4 Analytic `dM/dλ` (spec §6.1)

`assemble_matrix_dwn` in `kernels.py`, `assemble_derivative` in `solver.py`. The
identities are written out in spec §6.1 and were verified against
`kernels.py:285-338`. Two traps recorded there: the diagonal terms `d_m2`, `d_m4`
need `H₁` at the log-singularity arguments, which assembly never computes; and
the saving is 3 assemblies → 2 (~1.5×), not → 1. Exact only for **non-dispersive**
materials — say so in the docstring, it is a real precondition.
**Check:** `test_matrix_derivative_matches_assembly` bit-identical via
`np.array_equal`; `test_matrix_derivative_matches_central_difference` at real
*and* complex `wn`, both polarisations, observed order 2.

### 3.5 Wire `QNMResult.refine()` (spec §6.2, D3)

Add `refine(*, tol=1e-9, max_iter=30) -> QNMResult` returning a *new* frozen
result, plus the `converged` field. Opt-in; not on the accuracy path.
**Check:** the three §6.2 tests. Note the spec's warning — `research-port`'s
proposed criterion "error drops ≥10× versus the Beyn estimate" **cannot be met
and must not be written**; the 0.37 nm residual on the `n=0` pole is 100 %
discretization, 0 % extraction. Degenerate poles give `cond(J) ≈ 2e15`; Newton
must no-op gracefully and be flagged, never raise.

### 3.6 `σ_min` by inverse iteration (spec §6.3)

Only after 3.5 supplies the LU. 3 iterations of `y = M⁻¹v; z = M⁻ᴴy` seeded with
the mode vector reproduced the full-SVD value to `2.6e−7` relative in 2–6 ms
against 497 ms. Over-engineering to avoid, per the spec: Lanczos/`svds`, adaptive
iteration counts, chasing accuracy at generic points.
**Check:** matches the current full-SVD `sigma_ratio` to ~`1e−6` relative at the
anchors; measurable speedup.

### 3.7 `examples/qnm_spectrum.py` + README figure

**Load the `dataviz` skill first** — CLAUDE.md requires it for any plotting code,
and these scripts are the package's public face. `MPLBACKEND=Agg` in CI. Axes
carry units (nm); both polarisations distinguishable without relying on colour
alone. A log `Im λ` axis with iso-`Q` guide lines reads well for this data, since
`Q = Re λ / (2 Im λ)` becomes a family of straight lines.
**Check:** renders headless; README figure URL not pinned to a stale tag.

### 3.8 Repo drift (spec §12)

`__init__.__version__` and `test_placeholder.py`; the stale `CHANGELOG.md`
`## Unreleased` block; README figure URLs; the `CLAUDE.md` roadmap line
promising v0.3 = QNM. Add `per-file-ignores` for `beyn.py` `N806` and
`version_variables` to `pyproject.toml` if not already present.
**Check:** one consistent version story; `uv run ruff check .` clean.

---

## 4. Deliberately out of scope

From spec §§9–10, so nobody re-litigates it:

- **Mode fields and normalisation** (D4) — `vectors` are exposed raw; a QNM norm
  is v0.5+.
- **Degenerate poles are found and counted but not refined** — bordered Newton
  assumes a simple eigenvalue. Flagged via `cond(J)`, never silent.
- **Homogeneous background only.** Slab/image backgrounds would break the
  holomorphy argument in spec §5.2 — which is precisely why they are out.
- **`dM/dλ` is exact only for non-dispersive materials.**
- Everything in spec §10's *stays behind* list: Dask, `sweep.py`, `config.py`,
  progress bars, `print` tracing, the surrogate-error paths, `n_expected`,
  `sv_gap_factor`, and the research sign convention.
