# QNM / Beyn port — status and merge gate

Status as of **2026-07-31, end of the convention-fix session**. Branch
`beyn-port`, not merged. Written to be read on its own.

This replaces the earlier handoff note of the same name (the one carrying the
"superseded in part" banner) and folds in `convention-fix-status.md`, which was
written outside the repo and is now history. Where that note is still the
authority — the migration table, the injection-test evidence — it says so.

## Context documents

| Document | What it is | How to treat it |
|---|---|---|
| `docs/design/beyn-port-spec.md` | The implementation spec, draft 2. §§1–5 are built; §§6–13 are forward-looking. | **The authority on intent.** Section numbers below refer to it. |
| `docs/design/beyn-port-strategy.md` | The earlier high-level strategy. | **Superseded.** Provenance only. |
| `docs/design/qnm-methods.pdf` | Background on QNM extraction methods. | Reference. |
| `docs/design/performance.md` | Where the runtime actually goes, measured; the accepted threading plan; the rejected JAX migration. | **Read before optimising anything.** It corrects two of the spec's performance numbers. |
| `docs/conventions.md` §2, §8 | Shipped normative text: vacuum wavelengths, the QNM half-plane. | **Normative.** No live caveats remain. |
| `CLAUDE.md` | Repo working agreement. | Non-negotiables 1–4 apply, especially #3 (independent anchor) and #4 (justify every tolerance). |

---

## 1. Where the repo actually is

`uv run pytest` → **90 passed in 27 s**, against a **5-minute wall-clock budget**
for the whole suite (raised 2026-07-31; see §2, Q5). `uv run ruff check .` and
`ruff format --check .` clean, 26 files. All three `examples/` scripts render
headless.

Committed on `beyn-port`, ahead of `main` by three commits:

- `b9e4817 fix: ongoing Beyn integration` — the port itself: `beyn.py`, `qnm.py`,
  the `reference/mie.py` analytic anchor, the three test files, exports,
  `conventions.md` §8.
- `43f7c53 feat!: expose vacuum wavelengths and cladding size parameter
  uniformly` — the D1 + D5 convention fix, `tests/test_conventions.py`.
- `000a9ca docs: add the Beyn port design record` — spec, strategy, this file,
  the PDF.

**Working tree is not clean.** `CLAUDE.md` is modified and `examples/CLAUDE.md`
is untracked — the split of the figure conventions into a directory-scoped
instructions file, plus the removal of the commands block and the duplicated
one-line description. That is the tidy-up work; it is written but **not
committed**. Commit it before anything else, or it will be read as scratch.

### Phase status against the spec

| Spec phase | State | Evidence |
|---|---|---|
| §2 Phase 0 — D1 vacuum wavelengths (+ D5 `epsi` absolute) | **Shipped** | `43f7c53`; `tests/test_conventions.py`, 11 tests |
| §3 Phase 1 — analytic anchor | **Done** | `tests/test_mie_qnm.py`, 7 tests |
| §4 Phase 2 — Beyn on synthetic pencils | **Done** | `tests/test_beyn.py`, 17 tests |
| §5 Phase 3 — extractor + circle anchor | **Done** | `tests/test_qnm.py`, 16 tests |
| §6.1 Phase 4 — analytic `dM/dλ` | **Not started — merge blocker** | no `assemble_matrix_dwn`, no `assemble_derivative` |
| §6.2 Phase 5 — `QNMResult.refine()` | **Half — merge blocker** | `beyn.newton_refine` exists and is tested on synthetic pencils; unreachable from the façade |
| §6.3 — `σ_min` by inverse iteration | **Not started, knowingly** | `qnm._sigma_ratio` does a full SVD and says so |
| §7 Public API | **Shipped, with drift** | see Gotcha 3 |
| §11 `examples/qnm_spectrum.py` | **Missing** | `examples/` still has only the three v0.2 scripts |
| §12 Repo drift | **Unfixed** | see Gotcha 5 — this is now the main merge blocker |
| §13 Open questions | **All six resolved** | Q1, Q2 answered in the spec; Q3–Q6 answered by what shipped, see §2 below |

### What exists

`src/pysie2d/beyn.py` — EM-free, callable-driven: `rect_contour_quad`,
`probe_matrix`, `contour_moments`, `_detect_rank`, `beyn_poles`, `beyn_modes`,
`newton_refine`, and the `BeynModes` / `RefinedMode` result types.

`src/pysie2d/qnm.py` — the façade: `QNMSolver(geometry, material).modes(z_lo,
z_hi, *, n_quad_per_side=12, n_probe=12, rank_tol=1e-8, rng_seed=0)` returning a
frozen `QNMResult`. It composes a `BIESolver` and calls `BIESolver.assemble` —
deliberately the driven solver's own matrix, because "a QNM is a singularity of
the scattering operator" only holds if the two are literally the same assembly.

`src/pysie2d/reference/mie.py` — the analytic anchor: `qnm_denominator`,
`qnm_size_parameters`, `qnm_wavelengths` (vacuum, takes `n_clad`).

---

## 2. Decisions now closed

Recorded so nobody re-opens them.

- **D1 vacuum wavelengths** — shipped. Public methods take λ_vac in nm;
  primitives take `wnum_bg = 2π·n_clad/λ_vac` and **no wavelength at all**, so
  `Material.wnum_bg` is the single conversion point and no call path can apply
  `n_clad` twice.
- **D5 `Material.epsi` absolute** — shipped. `eps_rel = (n_core² + i·epsi)/n_clad²`;
  `Material.epsi_rel` exposes the divided value. It touches **both**
  `Material.eps` and `Material.nc`; spec §2.3 named only `eps`.
- **Size parameter is derived-only and circle-only**, gated on the new
  `Geometry.is_circle` (a numerical test, not a check on `m`). Same argument that
  rejected a complex-frequency entry point: a second entry point lets the two
  disagree.
- **No QNM round-trip conversion.** The contour is drawn on `assemble`, which is
  itself the conversion point, so boxes and eigenvalues are both vacuum λ. This
  was an open worry; it dissolved.
- **Q3 degenerate poles** — reported twice, with a `multiplicity` array. Shipped.
- **Q4 naming** — `QNMSolver` / `.modes()` / `wavelengths`. Shipped.
- **Q5 CI budget** — accepted unmarked, and since **raised to a 5-minute
  wall-clock budget for the whole suite** (2026-07-31). The suite is 27 s today.
  Spec §13 Q5's "+60–90 s on a 55 s baseline" is superseded; tests should be
  resolved well enough to prove the physics rather than trimmed to save seconds.
- **Q6 `CLAUDE.md`** — edited with permission; the `Im λ > 0` / `Q` convention
  and the vacuum-wavelength rule are both in non-negotiable #2.

Taken 2026-07-31, after the convention release:

- **`assemble_matrix_dwn` returns the fused `(M, dM)`**, sharing
  `h0w`/`h1w`/`h0w1`/`h1w1` — spec §6.1's recommended form, ~1.3× cheaper than
  two passes. This follows from the next decision rather than from the speed:
  a bit-identity test needs an `M` to compare against.
- **The derivative parity test stays bit-identical** (`np.array_equal`, not
  `np.allclose`). It is the right guard for ~50 lines of deliberate duplication
  of the hot path, and the only freedom it costs — computing the derivative by
  some other route — is one we have now explicitly declined (see below).
- **JAX is rejected**, on measurement. `docs/design/performance.md` §4 has the
  numbers; the short version is that `scipy.special.hankel1` on complex argument
  is 95.5 % of a `modes()` call, JAX has no complex Hankel, and the migration
  ceiling is ~1.02× against 5.02× for stdlib threading. Do not re-open without
  new information.
- **The contour loop will be threaded, inside `contour_moments`**, as the first
  post-merge change: 5.02× measured, stdlib only, and it changes no number the
  solver produces. Not before the merge — see §4.6.

Evidence for the convention fix that is not repeated here: the pre/post
worktree comparison over 31 quantities at `n_clad = 1` (**max abs diff 0.0** on
every one), and the five-injection verification of `test_conventions.py`. Both
are in the `43f7c53` commit message, which is the authoritative copy.

---

## 3. Gotchas

Ordered by how much damage they do if missed. Gotcha 1 of the previous version
(the D1 deferral) is resolved and gone; the numbering below is fresh.

### Gotcha 1 — refinement is unreachable today, because the analytic `dM/dλ` was never written *(merge blocker)*

`beyn.newton_refine` takes `dm_builder: Callable[[complex], np.ndarray]`. The
algorithm side is complete and tested; the EM side that would supply that
callable does not exist. So `QNMResult` has no `refine()` and no `converged`
field, and spec §6.2's three façade tests are unwritten. This also blocks §6.3,
which wants the LU that refinement would already have formed.

**This ships before the merge** — §4.2. The one thing to keep straight while
building it, because it will otherwise be mis-sold: per spec §6.2 refinement is
**insurance, not accuracy**. On a well-resolved simple pole the Beyn estimate is
already converged to ~1e−8 nm while the *discretisation* error is 0.38 nm — 100 %
of the error at the anchor is `nn`, 0 % is extraction. What `refine()` buys is
the coarse-contour recovery path and a `converged` flag, not a better number at
a good contour.

### Gotcha 2 — `rank` can legitimately exceed `n_modes`

A pole just *outside* the contour leaks a rank direction in through imperfect
quadrature cancellation; its eigenvalue is then discarded by the in-contour
filter. Tested (`test_rank_may_exceed_mode_count_from_outside_leakage`). **Not a
bug** — do not "fix" it by forcing the two to agree.

### Gotcha 3 — `QNMResult` drifted from spec §7, mostly for the better

Gained: `rank`, `cancellation` (how completely the contour integral cancelled),
`edge_margin` (distance to the nearest contour edge as a fraction of the shorter
side), `size_parameters`. Lost: `converged` — which §4.2 puts back, since it is
the natural output of the refinement step and the field a user checks before
trusting a mode. The three diagnostics `sigma_ratio`, `cancellation`,
`edge_margin` remain the *pre*-refinement quality signals: they say whether the
box was drawn well, which `converged` does not.

### Gotcha 4 — the Phase-1 root finder's `Im x` floor is a Q ceiling

`qnm_size_parameters` seeds Newton from a grid whose imaginary axis bottoms out
at `Im x ≈ 1e−4`, which cannot see a mode with `Q ≳ 1.5e4`. It does not bite at
the current fixture, but it is closer than it looks: the TE ladder runs
`Q = 164, 598, 2289` for `n = 4, 5, 6`, `Im x` falling ~3.4× per order. Two more
orders, or a higher `m`, crosses the floor **silently**. The mitigation is the
design: `test_root_count_matches_winding_number` asserts completeness
independently of the seeder. Keep it that way.

### Gotcha 5 — three version stories in one repo *(merge blocker)*

- `pyproject.toml` → `0.3.0`; semantic-release will cut `0.4.0` from the
  breaking commit on merge.
- `src/pysie2d/__init__.py` → `__version__ = "0.2.0"`, and
  `tests/test_placeholder.py` asserts that stale value, so CI is green on the
  drift.
- `docs/conventions.md` §8 heading says **(v0.5)** — right for the QNM release,
  wrong for a file merging before it.
- `CLAUDE.md` roadmap and `README.md` §Roadmap both say **v0.3.0 = QNM
  extraction**, but v0.3.0 shipped as a performance release.

One story: **v0.4.0 = conventions, v0.5.0 = QNM.** See §4.1.

### Gotcha 6 — a rectangle in `x` is not a rectangle in `λ`

The Phase-1 completeness statement is made over a box in the **size parameter**.
`λ = 2π·n_clad·rad/x` is a Möbius map and does not carry corners to corners. The
public API searches `λ`-rectangles. Any new test box must be argued for
completeness in its own coordinates.

### Gotcha 7 — the TE `n=0` anchor box is crowded in `Re λ`, isolated in `Im λ`

`530.83 + 26.38j` has TE `n=5` at `505.68 + 0.42j` and TE `n=2` at
`550.47 + 20.24j`, ~20 nm away in `Re λ`. What separates them is `Im λ`. The
working box is `Re λ ∈ [520, 545]`, `Im λ ∈ [15, 40]`; drawn generously in `Im`
it swallows the `Q = 598` mode and the rank-1 assertion fails for a reason that
has nothing to do with Beyn. TE `n=3` at `760.69 + 7.95j` is far more forgiving.

### Gotcha 8 — conventions that bite anyone carrying real-eigenvalue intuition

- **`Im λ > 0`** for decaying modes; `Q = +Re λ / (2 Im λ)`. The research code's
  sign is wrong relative to the operator it calls; spec §10 leaves it behind.
- **Poles do not come in conjugate pairs.** The reality condition is `λ → −λ̄`,
  putting mirror partners at negative `Re λ`.
- **`Re λ > 0` is load-bearing** — it keeps Hankel arguments off the `H^{(1)}`
  branch cut, which is what makes `M(λ)` holomorphic and the contour argument
  valid. Both bounds are asserted, not documented.
- Degeneracy is structural: every `n ≥ 1` circle mode is doubly degenerate via
  `exp(±inθ)`; only `n = 0` is simple.
- `D^{TM}_0 ≡ D^{TE}_1` identically, so a spectrum plot overlaying both
  polarisations shows coincident lines. Harmless — `M(λ)` is assembled per
  polarisation.

### Gotcha 9 — carried over from the convention fix, none blocking

1. `CHANGELOG.md` holds a stale `## Unreleased / Performance` block describing
   work already in v0.3.0, with the hand-written breaking-change section above
   it. Semantic-release inserts rather than folds. Reconcile after 0.4.0 cuts.
2. `figures/purcell_map.png` is stale in git — pre- and post-change code
   regenerate an identical md5 that differs from the committed file, so it
   predates this work. Regeneration is deterministic.
3. `Material.nc` is wrong for gain (`epsi < 0`): the principal-root closed form
   reconstructs `Im nc` from `|eps|`, returning a lossy index and `nc² ≠ eps`.
   Pre-existing; docstring and test scope the identity to passive materials.
4. Weak-loss cancellation in `Material.nc`: `aeps - epsr` underflows
   (`Im nc == 0.0` at `epsi_rel ≈ 5e-15`). Pre-existing, but `/n_clad²` shifts
   where the threshold sits.
5. `material.py`'s doctest never runs — pytest has `testpaths = ["tests"]` and no
   `--doctest-modules`. Value verified by hand.
6. `Geometry.is_circle` is sensitive to absolute position: a 200 nm circle at
   `x0 = 1e7` returns `False` from coordinate cancellation, so `size_parameter`
   would spuriously raise. Fine at realistic centres (9.4e-15 at `x0 = 1e4`).
7. `pysie2d.reference` is not in `__all__`, yet the size-parameter docs lean on
   it. Pre-existing.

---

## 4. The merge gate

The agreed bar for merging `beyn-port` into `main`: **a first working version of
Beyn's method within the known limitations, plus documentation that walks the
user through the gotchas.** Extraction is done and green; **refinement ships with
it**, so users can refine rather than being told in prose why they do not need
to. The rest of the gate is documentation and version hygiene.

Each item states its check, per CLAUDE.md's "define the check before the work".
4.2 is the only one with real numerical risk; do it first, since 4.3 and 4.4
both describe its output.

### 4.1 Commit the tidy-up, then fix the version story — *blocker*

Commit the `CLAUDE.md` / `examples/CLAUDE.md` split. Then pick one story:
**v0.4.0 = conventions, v0.5.0 = QNM.** Add
`version_variables = ["src/pysie2d/__init__.py:__version__"]` to
`[tool.semantic_release]`, and make `test_placeholder.py` assert against
`importlib.metadata.version("pysie2d")` so it cannot drift again. Update the
`CLAUDE.md` and `README.md` roadmap lines. Leave `conventions.md` §8's `(v0.5)`
heading — it becomes correct the moment the QNM release cuts, and this branch is
that release.
**Check:** `git status` clean; one version visible from `pyproject.toml`,
`__init__`, and the roadmap; suite still 90 green; `ruff check .` clean.

### 4.2 Analytic `dM/dλ`, then `QNMResult.refine()` — *blocker, and the only numerical work left*

Two steps, in order; the first exists only to serve the second.

**(a) `dM/dλ` (spec §6.1).** `assemble_matrix_dwn` in `kernels.py` returning the
fused **`(M, dM)`**, `assemble_derivative` in `solver.py`. The identities are
written out in spec §6.1 and were verified against `kernels.py:285-338`. Two
traps recorded there: the diagonal terms `d_m2`, `d_m4` need `H₁` at the
log-singularity arguments, which assembly never computes; and the saving is 3
assemblies → 2 (~1.5×), not → 1. Exact only for **non-dispersive** materials — a
real precondition, say so in the docstring, and note that the derivative is with
respect to the wavenumber, so the façade owes it the vacuum-λ chain factor
(conventions §2 — the conversion still happens exactly once).
*Check:* `test_matrix_derivative_matches_assembly` bit-identical via
`np.array_equal`; `test_matrix_derivative_matches_central_difference` at real
*and* complex `wn`, both polarisations, observed order 2.

**Correction to spec §6.1, found while scoping this.** The fused form's ~1.3×
saving is **unreachable through `newton_refine` as written**: `beyn.py:483-484`
calls `m_builder(lam)` and `dm_builder(lam)` as two independent callables, so the
Hankel work happens twice per Newton iteration however well `assemble_matrix_dwn`
shares internally. **Accept the two assemblies** (~190 ms per mode; Newton
converges in one step on a simple pole per spec §6.2) and say so in the
docstring. Do **not** change `newton_refine` to take a single `(M, dM)` callable
— that trades `beyn.py`'s EM-free purity, the most portable and most reusable
property in the repo, for 1.3× on 4 % of the runtime.

**(b) `QNMResult.refine(*, tol=1e-9, max_iter=30) -> QNMResult` (spec §6.2, D3).**
Returns a *new* frozen result plus the `converged` field. Opt-in — the docs must
not imply it improves a well-drawn contour (Gotcha 1).
*Check:* the three §6.2 façade tests —
`test_refine_is_idempotent_at_convergence`,
`test_refine_recovers_from_coarse_contour`,
`test_refine_flags_degenerate_pole`.

Two things that will go wrong if they are not read first:

- The spec's warning stands: `research-port`'s acceptance criterion "error drops
  ≥10× versus the Beyn estimate" **cannot be met and must not be written**. The
  0.38 nm residual at the anchor is 100 % discretisation. The test that *can*
  pass is `test_refine_recovers_from_coarse_contour` — deliberately under-resolve
  the contour, then show refinement returns to the well-resolved value.
- Degenerate poles give `cond(J) ≈ 2e15`. Newton must **no-op gracefully and be
  flagged, never raise** — a circle's `n ≥ 1` modes are all degenerate, so this is
  the common case, not the corner case.

CI cost: refinement adds assemblies per mode. **The suite's budget is now 5
minutes wall-clock** — raised deliberately from spec §13 Q5's 55 s baseline +
60–90 s estimate, which is superseded. At 27 s today that is ~10× headroom, so
resolution is no longer the scarce resource: prefer a well-resolved test that
proves the physics over a cheap one that needs a tolerance argument. Still
report the suite time in the commit message, and if a single test dominates say
why. Past 5 minutes, cut mode count or `nn` before reaching for a `slow`
marker — a second CI step is still not wanted.

### 4.3 A user-facing QNM section — *blocker, the real one*

`README.md` does not mention QNMs outside a roadmap line, and `conventions.md`
§8 is written for a contributor, not a caller. Someone who runs
`QNMSolver.modes` today has no document telling them how to draw a box or how to
tell a good answer from a bad one. Write it — `docs/qnm-guide.md` linked from the
README, or a README section if it stays short. It must cover, in this order:

1. **How to draw a box.** `Im λ > 0`, `Re λ > 0`, both asserted. A box in `λ`, not
   in `x` (Gotcha 6). Isolation in `Im λ` matters more than width in `Re λ`
   (Gotcha 7).
2. **How to tell the answer is trustworthy.** `edge_margin` near zero means the
   box is clipping a pole; `sigma_ratio` is the singularity measure and is
   dimensionless on purpose; `cancellation` says whether the contour integral
   closed; `sv_ratio` / `max_gap` is where the rank gap lives. These are
   *contour-quality* signals and stay the first thing to check even with
   refinement available.
3. **What `refine()` is and is not for.** It recovers a mode from a coarse or
   badly placed contour and sets `converged`. It does **not** improve a
   well-resolved mode, and on a degenerate pole it no-ops and says so. Show the
   coarse-contour case, since that is the one where it earns its place.
4. **What the accuracy actually is.** At the anchor, extraction is converged to
   ~1e−8 nm and the answer is off by 0.38 nm — *all* of it discretisation in
   `nn`. Refining the contour buys nothing there; raising `nn` does. Say this
   explicitly next to `refine()` or users will tune the wrong knob.
5. **`rank` may exceed the mode count** (Gotcha 2), and degenerate partners come
   back as two entries with `multiplicity = 2`.
6. **The limitations, named** (§5): homogeneous background only, no mode
   normalisation, degenerate poles found and refined-but-flagged rather than
   resolved, non-dispersive materials only.

**Check:** a reader who has not seen this design record can, from that page
alone, draw a box around the TE `n=3` mode at `760.69 + 7.95j`, run it, and say
why the result is or is not trustworthy. Every code snippet in it executes.

### 4.4 `examples/qnm_spectrum.py` + README figure — *blocker*

The package's public face; also the fastest way to make 4.3 concrete.
**Load the `dataviz` skill first** — CLAUDE.md requires it for plotting code, and
`examples/CLAUDE.md` carries the figure conventions. `MPLBACKEND=Agg` in CI. Axes
in nm; both polarisations distinguishable without relying on colour alone. A log
`Im λ` axis with iso-`Q` guide lines reads well here, since `Q = Re λ/(2 Im λ)`
becomes a family of straight lines.
**Check:** renders headless; README figure URLs no longer pinned to `/v0.2.0/`
(Gotcha 5's fourth face — a new figure at the old pin 404s).

### 4.5 `CHANGELOG.md` reconciliation — *blocker, small*

Fold the stale `## Unreleased / Performance` block into v0.3.0 and let the
breaking-change section be what semantic-release turns into v0.4.0. The
`BREAKING CHANGE:` commit footer of `43f7c53` is the authoritative copy of the
migration table.
**Check:** no `## Unreleased` block describing shipped work; the migration table
survives into the released changelog.

### 4.6 Not blockers — the v0.5.x line

Deliberately after the merge, in this order. Each is an improvement to a working
feature, not a repair of a broken one.

- **Thread the contour loop — the highest value per line in the repo.**
  `ThreadPoolExecutor` over the contour nodes in `contour_moments`: **5.02×
  measured** at 8 workers on 93 % of `modes()`, stdlib only, and it changes no
  number the solver produces. `scipy.special.hankel1` releases the GIL, which is
  why this works and why `multiprocessing` would be strictly worse. Details,
  measurements, and the two traps (per-node LU-failure status must be returned
  rather than accumulated in a shared counter; `warnings.warn` from a worker does
  not reliably reach the caller) are in `docs/design/performance.md` §3.1.
  Post-merge only — it touches a green module for a speedup, which is exactly
  what the merge gate exists to defer.
- **`σ_min` by inverse iteration (spec §6.3).** Now unblocked by 4.2, which forms
  the LU — but a performance nicety, and a smaller one than the spec suggests.
  Measured at the fixture: `_sigma_ratio` is 2.7 % of a `modes()` call and its own
  dominant term is *another assembly* (94 ms), not the SVD (38 ms). The spec's
  "497 ms vs 2–6 ms" was measured at a larger `nn`. Avoid, per the spec:
  Lanczos/`svds`, adaptive iteration counts, chasing accuracy at generic points.
- **Structured Hankel evaluation — parked, not rejected.** 13× measured at 7e-15
  relative accuracy, and it multiplies with the threading. But it is a
  from-scratch special-function implementation needing its own anchor under
  non-negotiable #3; `performance.md` §3.2 states the validation burden and what
  the probe did *not* cover. A feature with a validation plan, not an
  optimisation.
- The seven carried-over items in Gotcha 9.

---

## 5. Known limitations — the list the release ships with

From spec §§9–10, and the thing §4.3 must state plainly:

- **Homogeneous background only.** Slab or image backgrounds break the
  holomorphy argument of spec §5.2 — which is precisely why they are out.
- **Mode fields are not normalised** (D4). `vectors` are raw `φ`/`χ` columns; a
  QNM norm is v0.5+.
- **Refinement recovers a contour, it does not beat discretisation.** At a
  well-drawn contour, accuracy is set by `nn`. `refine()` is for the coarse or
  badly placed box.
- **Degenerate poles are found and counted but not refined** — bordered Newton
  assumes a simple eigenvalue, and every `n ≥ 1` circle mode is degenerate.
  `refine()` no-ops on them and flags it via `cond(J)`, never silently.
- **`dM/dλ` is exact only for non-dispersive materials**, which makes `refine()`
  non-dispersive-only too.
- **A high-Q ceiling in the *anchor*, not the solver** (Gotcha 4): the analytic
  seeder cannot see `Q ≳ 1.5e4`, so validation coverage stops there.
- Everything on spec §10's *stays behind* list: Dask, `sweep.py`, `config.py`,
  progress bars, `print` tracing, the surrogate-error paths, `n_expected`,
  `sv_gap_factor`, and the research sign convention.
