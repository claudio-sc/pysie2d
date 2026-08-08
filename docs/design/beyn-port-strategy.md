# Beyn QNM extraction → pysie2d: port strategy (high level)

Target: pysie2d v0.4.0. Source: `sie/qnm/` (private research code).
Status: strategy only — no implementation detail, no code.

---

## 1. What this is, and what it is not

**Is:** bring contour-integral QNM extraction — find every complex λ where
`M(λ)` is singular inside a chosen region — into pysie2d as a first-class,
validated, documented feature of the homogeneous-background single-particle
solver.

**Is not:** a port of the research QNM *pipeline*. The research code's value is
concentrated in ~27 KB of algorithm (`contour.py` plus the bordered-Newton core);
the surrounding ~400 KB is Dask orchestration, surrogate dispatch, sweep
management, and slab/waveguide physics. None of that belongs in pysie2d.

The scope boundary is already written down: pysie2d covers "the part that can be
validated end-to-end against a closed-form reference." QNM extraction for a
circular cylinder passes that test. Slab-background QNMs do not, and stay out.

---

## 2. The validation anchor comes first

This is the gate, not a later phase. `CLAUDE.md` non-negotiable #3 requires an
independent anchor, and agreement between two pysie2d code paths does not count.

**The anchor exists and is cheap.** For a circular cylinder the QNMs are exactly
the complex-λ zeros of the Mie coefficient denominators, which
`reference/mie.py` already builds explicitly (`an`, `bn`). Finding those zeros is
scalar 1-D complex root-finding per azimuthal order `n` — milliseconds, no BIE
involved, fully independent of the machinery under test.

So the acceptance criterion for the whole feature is available up front: Beyn on
a circle must recover the analytic denominator roots, per polarisation, to a
stated tolerance with a stated convergence order in `nn`.

Two consequences:

- Build the analytic QNM reference **before** the extractor. It is the cheapest
  artifact in this project and it de-risks everything after it.
- `reference/mie.py`'s coefficient helpers are currently typed for real size
  parameter. Complex-argument evaluation is a precondition for the anchor —
  a signature/typing question to settle early, not a rewrite.

Secondary checks worth having, none of which substitute for the anchor: poles
must sit in the lower half-plane (decaying modes under `exp(-iωt)`); `Q` values
must match the linewidths of the `qsca(λ)` peaks the driven solver already
produces; a pole must be stable under contour reshaping and under `nn`.

---

## 3. What ports, what gets rebuilt, what stays behind

**Ports nearly as-is.** `contour.py` — Gauss-Legendre rectangular contour,
moment-matrix accumulation, SVD rank detection with gap truncation, in-contour
filtering, null-space vectors returned for refinement. It is clean, depends only
on numpy/scipy, and already takes a `m_builder: λ → matrix` callable, which is
exactly the shape of `BIESolver.assemble`. Drop the Dask path; keep the
sequential one.

**Rebuilt, not ported.** The bordered-system Newton refinement. The algorithm is
right and worth keeping; its research *implementation* carries a flat-scalar
signature that exists only to cross a Dask boundary, plus print-based logging,
surrogate-domain guards, and `_malloc_trim`. Rebuild it against pysie2d's object
API — and change two things while doing so (§5).

**Stays behind.** `runner.py`, `runner_legacy.py`, `_workers.py`, `sweep.py`,
`config.py`, and everything reaching into `sommerfeld.py` / `slab_image.py`.
Note that the research QNM path assembles the *image-corrected* matrix, not the
free-space one — so "port the QNM code" and "port the slab physics" are
entangled in the source and must be deliberately separated here. pysie2d's
extractor targets `assemble_matrix` only.

---

## 4. Conventions to pin before any code

These are the likeliest sources of a silently wrong pole.

1. **Wavelength units.** The research code works in "code" wavelengths and
   converts at the boundary (`z_lo / n_clad` in, `poles × n_clad` out).
   pysie2d's public API has no such conversion, while `Material.epsr` is already
   `(n_core/n_clad)²` — so the cladding scaling is implicit somewhere. Decide
   what `wavelength` means in pysie2d, state it in `docs/conventions.md`, and
   test on a non-unit `n_clad`. Everything downstream depends on this.
2. **Search-region parameterisation.** Rectangle in complex λ, or in complex
   frequency/wavenumber? The research code uses a λ-rectangle; the physics
   literature usually quotes complex ω. Pick one for the public API, convert
   internally, document the mapping.
3. **Sign of `Im λ`.** Fixed by the `exp(-iωt)` convention already recorded in
   `docs/conventions.md`. Assert it rather than assume it — a sign slip here
   produces plausible-looking growing modes.
4. **`Q` definition.** The research code uses `Q = −Re λ / (2 Im λ)`. State it;
   there are competing conventions.

---

## 5. Two deliberate improvements over the research implementation

Both are measured, not speculative, and both belong in the port rather than in a
later optimisation pass — retrofitting them means rewriting the Newton loop
twice.

**Analytic `dM/dλ` instead of a finite-difference Jacobian.** λ enters `M` only
through the wavenumber, and the Hankel derivative identities collapse onto the
`H₀`/`H₁` arrays the assembly already computes — so the derivative costs no
additional special-function evaluations. This replaces three assemblies per
Newton iteration with one (measured: 1.08× one assembly vs 3×), and it is also
an *accuracy* fix: the research default `fd_step=1e-4` yields a Jacobian with
~3e-7 relative error while the convergence test is `1e-8` on the step. It also
deletes a tuning knob from the public API.

**σ_min by inverse iteration, not a full SVD.** The research code takes a
complete SVD of the system matrix to obtain one number, twice per pole
(measured: 449 ms at `2nn=600`, ≈4.3 assemblies, growing cubically). σ_min here
is a diagnostic and an acceptance threshold, not the Newton driver, so a few
inverse-iteration steps on the LU factor that already exists are sufficient.

A third, larger optimisation — tabulating the Hankel functions along the single
complex ray every argument lies on — is deliberately **out of scope for the
port**. It trades exactness for ~1e-11 and should be evaluated against a working,
exact extractor, not baked into one.

---

## 6. Public API shape

Follow the existing façade pattern (`BIESolver` → `ScatterResult`), not a
function-with-many-keywords. The research `QNMExtractor` / `QNMResult` pair is
already the right shape and maps cleanly:

- an extractor composed from a `Geometry` and a `Material`, mirroring
  `BIESolver`, with one method that takes a search region;
- a result object carrying poles, mode vectors, rank diagnostics, and
  convergence flags, exposing derived physics (`Q`) as properties and offering
  refinement as a method returning a new result.

Reuse `BIESolver.assemble` as the matrix builder rather than introducing a
parallel assembly path — the fact that `assemble` already accepts complex
wavenumbers is the reason this port is possible at all, and exercising it here is
what keeps non-negotiable #1 honest.

Open API question to settle: are mode *fields* (evaluating a QNM on a spatial
grid, via the existing representation formula) in v0.4.0 or deferred? They are
the natural next ask and the main reason a caller wants the mode vectors.

---

## 7. Phasing

Each phase ends in something verifiable; none is worth starting before the one
above it lands.

| # | Phase | Ends when |
|---|---|---|
| 0 | Pin conventions (§4) into `docs/conventions.md` | Units test on non-unit `n_clad` passes |
| 1 | Analytic QNM reference for the circle in `reference/` | Roots reproduce known circular-cylinder resonances |
| 2 | Contour extractor (`contour.py` port, sequential) | Beyn on a circle matches Phase 1 for both polarisations |
| 3 | Analytic `dM/dλ` | Validated against central differences at the h² rate |
| 4 | Bordered-Newton refinement + inverse-iteration σ_min | Refined poles hit the anchor at the tight tolerance |
| 5 | Public façade, docs, `examples/` figure, roadmap update | CI green, tolerances justified per non-negotiable #4 |

Phase 2 is the honest go/no-go: if Beyn on a circle does not reproduce the
analytic denominator roots, nothing later is worth building.

---

## 8. Risks and open questions

- **Contour placement is the user-facing difficulty.** Beyn finds what is inside
  the box; a badly chosen box silently finds nothing or finds spurious
  eigenvalues. The research code mitigates with `n_expected` and SVD-gap
  truncation. The port needs a documented recipe for choosing a region from a
  driven `qsca(λ)` spectrum, plus diagnostics that distinguish "no modes here"
  from "contour too tight" — otherwise the feature is unusable by anyone who did
  not write it.
- **Cost per extraction is real.** Contour assemblies are fixed and cheap
  (≈64); refinement dominates. All of it runs at complex λ, where the v0.3.0
  real-argument fast path does nothing. Set expectations in the docs, and note
  that the contour loop is embarrassingly parallel if it ever needs to be —
  without adding a parallelism dependency to satisfy #2 of the repo's dependency
  rule.
- **Rank detection is heuristic.** `tol_rank`, `n_expected`, `sv_gap_factor` are
  three coupled knobs in the research code. Decide how many of them are public.
  Fewer is better; each one is a way for a user to get a wrong pole count.
- **Degenerate and near-degenerate modes.** A symmetric Gielis particle has
  symmetry-degenerate modes; Beyn's rank detection and the bordered-Newton
  anchor both assume simple eigenvalues. Worth a known-limitation note even if
  not handled in v0.4.0.
- **Convergence order in `nn` is unknown for poles.** Near-field quantities in
  this repo converge at first order, which is why the self-Green anchor runs at
  `nn=1000`. Pole convergence order must be *measured*, then used to justify the
  test tolerance — not assumed.
