# v0.6 quadrature — preliminary study plan

**Status:** plan only; no gate has been run. This is doc B of two. The
decisions this study works within are
[../v0.6-architecture.md](../v0.6-architecture.md); the measurements it starts
from are [../pysie2d-quadrature-handoff.md](../pysie2d-quadrature-handoff.md).

This document is **superseded by its own results** — each gate's findings are
written back into it, and it is retired once the code-specs exist.

---

## 1. What the study must output

Three things, and it is not finished until all three exist:

1. **Every convergence gain quantified.** Not "it improves" — the rate, the
   constant, the resolution at which each shape class saturates, and the
   resolution at which each *stops* improving.
2. **The tooling demonstrably works.** Smooth inversion, adaptive density,
   smooth clamp, far-field quadrature — each exercised on a real shape, not
   only on the circle.
3. **Code-specs**, written from the refined scripts, detailed enough that a
   worker who cannot derive the physics can execute the migration. Coarse
   granularity — by capability, not per commit.

The study runs on the `v0.6-quadrature` branch and touches **no package code**.
Scripts live here, in `docs/design/studies/`, alongside the existing gate
scripts.

---

## 2. Provenance: the scripts inherited from the handoff

Seven scripts from the September 2026 investigation, moved into this directory
and committed as-is. They import shipped 0.5.0 unmodified except where noted.
They are **reused for the study and then superseded** — production code is
written from the code-specs, never from these.

| Script | Produces |
|---|---|
| `exp_kappa.py` | handoff §4.1, 4.2, 4.4 — κ, equilibrated κ, block-scaled κ |
| `exp_decisive.py` | §4.3 — perturbation sweep and Powell optimisation vs Mie |
| `exp_diag.py` | §5.2 — self-patch constant comparison (patches a copy of `kernels.py`) |
| `exp_kress.py` | §5.3 — Kress on M2/M4 only |
| `exp_kress_full.py` | §5.3 — Kress on all four blocks |
| `exp_weights2.py`, `exp_inversion.py` | §9 — the star anomaly and the hypotheses ruled out |

They carry absolute `/home/claude/...` paths from the machine they were written
on; fixing those is the first thing G0 does.

---

## 3. Reference and error probe

**Probe: the forward-direction amplitude `qext`, throughout.** Not `qsca`.
`ScatterResult.efficiencies` integrates the far field over a fixed `n_angles`
grid and carries a ~1e-4 angular floor that masks convergence — and that floor
is not fixed until G5, the last gate. `qext` is a single amplitude and has no
such floor (handoff §3).

**Reference:**

- **Circle** — analytic Mie, `pysie2d.reference.mie`. Exact ground truth. This
  is the only anchor that is genuinely independent, and it does the load-bearing
  work.
- **Non-circular** — a high-`nn` **uniform-θ Kress** solution. Deliberately
  *not* the arc-length path: handoff §9 shows self-convergence misleads when the
  reference itself sits on the suspect path, and §9 is unresolved by decision.
  Routing the reference around it keeps the anomaly out of the study's own
  measurements without pretending it is solved.

This is a self-convergence reference and is therefore **not** an independent
validation anchor in the sense of CLAUDE.md non-negotiable 3. It is adequate to
*measure rates* and inadequate to *validate physics*; a proper non-circular
anchor is a known open item (§7).

Baseline case, unchanged from the handoff so numbers are comparable:

    geometry   : Geometry.gielis(rad=200, n_pts=nn, m=0)      # circle
    material   : Material(n_core=1.5, n_clad=1.0, pol=2)      # TE
    wavelength : 600.0 nm (vacuum)
    star       : m=6, n1=6, n2=12, n3=12, rad=200

---

## 4. Gates

Sequential; each depends on the one before. **G1 is the riskiest and everything
after it depends on the answer.**

### G0 — Reproduce

Re-run the handoff's decisive numbers against shipped 0.5.0 on this machine:
first-order baseline (§5.1), Kress on M2/M4 → rate 3 (§5.3), Kress on all four
blocks + analytic `ddf`/`ddg` → 3.4e-15 at `nn = 30`.

*Passes when* the table reproduces to the digits quoted. *Fails* if it does
not, and then nothing downstream is trustworthy — the handoff was measured
elsewhere.

### G1 — A smooth `w`  ← the risk

Replace the `np.interp` arc-length inversion with a smooth one and demonstrate
that Kress-in-`t` recovers high order on a shape where θ and arc length differ.

Candidates: Fourier/spectral representation of `w(t) − t`; cubic spline; Newton
on the analytic arc-length integral of `γ`. Choose on measured smoothness and
simplicity, not elegance.

*Must establish:* `w` is smooth enough that the rate does not degrade; `w'` and
`w''` are available and correct; and the choice does not introduce an absolute
length (conventions §9 — `n_fine` is the thing to watch).

*Passes when* a non-circular shape shows a rate materially above 1 against the
§3 reference. *Fails* if smoothing the inversion does not recover order — see
§6.

### G2 — Adaptive density and the smooth clamp

Build `ρ` from curvature, bounded by `[R_min, R_max]` in the `R =
wavelength_over_ds` units of conventions §12, at the `Parametrisation`'s fixed
reference wavelength (default 1550 nm).

The whole gate is the **clamp**. `np.clip` is C⁰ and reintroduces the G1 failure
with two kinks per lobe. Measure a hard clamp against a smooth saturation
(softmin/softmax, or Fourier low-pass with bandwidth tied to `nn`) and show the
difference in rate — that measurement is what justifies the extra machinery in
the spec.

*Passes when* adaptive beats const-density at equal `nn` on a high-curvature
shape, **and** the smooth clamp measurably beats the hard one.

### G3 — Near-corner validity envelope

Error versus exponent at fixed `nn`, on the superellipse path (`m = 4, a = b =
1, n1 = n2 = n3 = n`), which reaches a square as `n → ∞` (architecture §6).
Coarse and decisive: a handful of exponents, a couple of `nn`.

*Output:* the exponent at which graded Kress stops delivering, stated as a
number. This is the release's honest validity envelope and gives `gielis-oed`'s
D6 curvature rejection a threshold to cite.

*This gate cannot fail* — it measures a boundary rather than testing a
hypothesis. It can only be uninformative, which happens if the sweep is too
coarse to locate the knee.

### G4 — Holomorphy and the QNM path

Confirm the contour method still works: `R`, `L` and `h` are λ-independent by
construction and every Bessel factor is holomorphic, so Beyn's premise should
survive — but it must be *shown*, because degradation is silent. Mode count and
`edge_margin` are the tells.

Also confirm `w` does not move with λ (invariant 3.3.2) or with a differentiated
parameter (3.3.3), and check whether `dλ/dp` still needs `richardson_limit`.

*Passes when* QNM positions against analytic Mie poles improve in rate, and the
mode count is stable across the contour.

### G5 — Far-field angular quadrature

Last, deliberately: until here, `qext` is the probe precisely to avoid this.
Fix the ~1e-4 floor in `ScatterResult.efficiencies` so `Q_sca` — and hence the
downstream `σ_sca = Q_sca · 2·rad` observable — is spectral too.

Note `dc9da93` already fixed a closing-angle double-count on this grid; check
the fix interacts correctly with whatever replaces the grid.

*Passes when* `Q_sca` against Mie tracks `qext` instead of stalling at 1e-4.

---

## 5. Reporting

**A single running artifact**, republished to the same URL as each gate closes.
One link, always current, history in versions:

    https://claude.ai/code/artifact/e9ceb9ef-69d0-40bf-9e9e-1b1329a1fa34

Findings are also written back into this file as each gate lands — the artifact
is the readable surface, this file is the record on the branch.

**Nothing goes into `README.md` or `docs/` proper until the migration.** They
can only be truthful about a tree that implements this.

---

## 6. If G1 fails

If the smooth inversion does not restore high order on stars, **stop and
re-evaluate — do not decide the fallback in advance and do not push on under
sunk cost.**

Context for that conversation, recorded now while it is cheap: a v0.6
restricted to circles and ellipses would still be a large gain and would still
unblock Levels 1–2 of the downstream ladder, which are ellipse-only. So a
partial result is not a dead end. But it is a different release, with a
different README claim, and it is decided when there is a measurement to decide
on.

---

## 7. Open items

Carried deliberately, not oversights.

- **No independent non-circular anchor.** The circle has Mie; the ellipse and
  the star have only self-convergence (§3). An ellipse against Mathieu functions
  is the realistic candidate and is real work. Deferred by decision; the
  validation scheme is to be designed when the work reaches it.
- **The uniform-arc-length star anomaly**, handoff §9. Untouched by decision.
  May be made moot by the `Parametrisation` rewrite — the `np.interp` inversion
  it implicates disappears — but that is a hypothesis the study does not test
  and must not claim.
- **`gielis-oed` gates 9 and 10 are invalidated by this release.** Both are
  blocking gates over there, and both were measured against first-order
  assembly: G9's `h` window may move, and G10's premise — "the whole compute
  budget is conditional on `n_pts = 200` passing; assembly is 4.3× dearer at
  400" — is transformed if 200 becomes overkill. **They are re-run on the
  `gielis-oed` side, in a future session, not here.** Recorded so whoever
  arrives at that work months from now knows the gates are stale and why.
- **Default `n_pts`.** Currently 200 in both repos. Deferred to the migration,
  when the numbers exist (architecture §7).
- **True geometric corners.** Non-goal, architecture §6.
