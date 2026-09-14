# v0.6 quadrature — preliminary study plan

**Status:** G1 passed, G2 passed (13 Sep 2026, with a caveat — see findings
below), G3 closed, G4 passed (14 Sep 2026, on uniform θ and adaptive maps;
`sigma_ratio`, not mode count, is the holomorphy tell), G0 retired, G5 next. Kress–Martensen is now **landed in
production** (`feat!: Kress-Martensen quadrature on a frozen node map`,
commit `771bc9b`), not just the `kress_t.py` prototype this status line used to
point to. Three pieces are now in production from this study's findings:
`Parametrisation` (`src/pysie2d/parametrisation.py`, including the
curvature-adaptive density of §5), analytic `ddf`/`ddg` (`geometry._rderiv2`),
and Kress quadrature on all four blocks — see architecture §1, items 1–3. Its
code-spec, with a verified reference patch, is
[../kress-spec.md](../kress-spec.md) (13 Sep 2026); that spec's own status line
is stale in the same way and should be corrected when next touched. Findings
are in the `####` sections under each gate. This is doc B of two. The
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

**Every non-circular convergence measurement starts at the circle and reaches
its shape by continuation** (conventions §8). The circle is the only labelled
spectrum available, so a mode on a deformed shape is identified by carrying a
labelled circle mode along a smooth perturbation — elliptical or star-like —
and the analytic Mie comparison is the first step of the measurement rather
than a separate circular exercise.

Two things follow, and they apply to every gate below that touches a
non-circular shape:

- **The continuation is itself a convergence check.** A smooth perturbation of
  a smooth boundary traces a smooth pole trajectory, so a kink, a jump or a
  non-monotone excursion is a defect signal — discretisation, identification or
  physics — and is read that way before it is read as a result. This is a check
  the study gets for free and should not waste.
- **The pole landscape is a standing point of care.** It gets richer as the
  deformation proceeds: circle degeneracies split, modes enter and leave a
  fixed box, trajectories approach one another. Identification is expected to
  be **refined as the study reaches richer families**, not settled once — a
  change of shape family, path or box is a reason to re-examine it. The
  measured failure mode is silent (see the orientation study below).

Baseline case, unchanged from the handoff so numbers are comparable:

    geometry   : Geometry.gielis(rad=200, n_pts=nn, m=0)      # circle
    material   : Material(n_core=1.5, n_clad=1.0, pol=2)      # TE
    wavelength : 600.0 nm (vacuum)
    star       : m=6, n1=6, n2=12, n3=12, rad=200

---

## 4. Gates

Sequential; each depends on the one before. **G1 is the riskiest and everything
after it depends on the answer.**

### G0 — Reproduce  ← RETIRED, not run

Re-run the handoff's decisive numbers against shipped 0.5.0 on this machine:
first-order baseline (§5.1), Kress on M2/M4 → rate 3 (§5.3), Kress on all four
blocks + analytic `ddf`/`ddg` → 3.4e-15 at `nn = 30`.

*Passes when* the table reproduces to the digits quoted. *Fails* if it does
not, and then nothing downstream is trustworthy — the handoff was measured
elsewhere.

> **RETIRED by decision, 12 Sep 2026 — superseded, not skipped.** G0 existed to
> confirm that the handoff's numbers transfer to this machine before anything
> downstream was trusted. The work below instead jumped to a Kress prototype
> and anchored it against the **full-precision analytic Mie pole** — recomputed
> by Newton on `qnm_denominator`, accurate to machine precision, against the
> handoff's five-decimal `qext` table. The *phenomenon* G0 was to verify is
> reproduced: first order on the shipped scheme (rate 1.00–1.10 on three
> ladders), spectral under Kress (1.2e-3 → 1.65e-9 → 1.1e-11 on the circle).
>
> What is therefore **not** established, and is the price of retiring it: the
> handoff's specific digits (`6.253e-5` at `nn = 30` for Kress on four blocks,
> `3.36e-15` with analytic `ddf`/`ddg`) have never been reproduced here. Should
> a later discrepancy make those digits matter, this gate is where to start.

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

> **PASSED (2026-09-12) — the release's largest risk is retired.** The smooth
> `w` is Newton inversion of an exact Fourier antiderivative, with `np.interp`
> demoted to the initial guess (`adaptive_density.py`); `w'` and `w''` come in
> closed form from `w' = 1/T'(w)` and `w'' = −T''(w)·(w')³`. Three pieces of
> evidence, in ascending strength:
>
> 1. **Density residual 4e-16** on four stars — the map delivers the density it
>    was asked for.
> 2. **Parametrisation invariance at 1.6e-11.** A circle's poles cannot depend
>    on how the circle is parametrised; a deliberately graded
>    `θ = t + 0.3 sin 2t` agrees with uniform θ to 1.6e-11 once resolved. A C⁰
>    map or a misplaced Jacobian fails this.
> 3. **Spectral rates on a non-circular shape**, which is the stated criterion:
>    at aspect 2 the error runs 3.5e-1 → 1.4e-3 → 7.7e-7 → 6.9e-10 → 2.2e-12
>    over `nn = 20…80`. "Materially above 1" is an understatement.
>
> `n_fine` stayed a function of `nn` alone, so no absolute length entered and
> conventions §9 is intact. The §6 fallback conversation is not needed.

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

> **Two clauses above are superseded by the findings that follow.** "Fourier
> low-pass with bandwidth tied to `nn`" is refuted — a bandwidth ∝ `nn` makes
> `w` a different map at every resolution.
>
> And the first pass criterion, "adaptive beats const-density at equal `nn`",
> **has now failed twice, for two different reasons**. On the shipped scheme it
> is unachievable in principle: a first-order error is set by the
> worst-resolved node, so grading at fixed `nn` necessarily loses. Under the
> Kress prototype it fails again on ellipses, because grading narrows the
> analyticity strip of `f∘w` and delays the spectral onset. **The criterion is
> therefore not a gate on the density's construction at all** — it is a
> question about which shapes grading is for, and only **G3**'s near-corner
> regime can answer it. Until then the meaningful half of this gate is the map,
> not the solution error.

#### G2 findings — the density is designed, node placement only (2026-09-12)

Script: `adaptive_density.py`; figure `star_adaptive_sampling.png`. Four 4-peak
Gielis stars (`m = 4`, exponents `n1/n2 = 2/4`, `6/12`, `12/24`, `20/50`) at an
`R` band of **20–100**, λ_ref = 1550 nm, `n_core = 1.5`. **Placement and map
smoothness only — no convergence rate is claimed here**, since that needs G0
and the Kress prototype. What is established is the construction, the cost
curve into the near-corner regime, and six corrections to architecture §4.

**The construction.** One density `σ(θ) ∝ |κ|^α_eff`, dimensionless in
`u = |κ|·L/2π` (`u ≡ 1` on a circle); `T(θ) = 2π/Z ∫₀^θ σγ` built from an exact
Fourier antiderivative and inverted by **Newton**, with `np.interp` demoted to
the initial guess; `w' = 1/T'(w)` and `w'' = −T''(w)·(w')³` in closed form.
The delivered density residual `max |σ·|dx/dt| / mean − 1|` is **4e-16** on all
four stars — the inversion machinery is exact, and what error remains is
resolution, not the map.

**The cost curve, and it is the useful output.** The density's smoothing
bandwidth `M` is set by the **angular width of the curvature layer** — the
fraction of the boundary on which `ln|κ|` sits within 1 of its maximum, which
is the arm tip and which narrows as the exponent rises. That is the geometric
signature of the thinning analyticity strip §4 invokes, and it is measurable
from the shape alone, before any solve:

| star | layer width | `M` | `nn` from the band | `nn ≳ 4M` needed | `∫w' dt − 2π` |
|---|---|---|---|---|---|
| mild `n1=2` | 1.700 rad | 4 | 38 | 16 | 1.7e-8 |
| baseline `n1=6` | 0.362 rad | 18 | 80 | 72 | 2.2e-3 |
| sharp `n1=12` | 0.166 rad | 38 | 82 | **152** | 3.9e-4 |
| near-corner `n1=20` | 0.067 rad | 94 | 94 | **376** | 1.7e-2 |

**Cheap on the smooth shapes, expensive at the corner, and the two failure
modes are different.** On the mild star the band's `nn = 38` already exceeds
what the density needs and `∫w' dt = 2π` holds to 1e-8 — a graded map is
essentially free. From the sharp star on, the `R` band stops being the binding
constraint: `nn` is set by the *density*, not by the wavelength, and it grows
like the reciprocal layer width. **This is the regime future convergence
studies should live in**, and the table is the prediction they test — G3's
envelope should locate its knee where `4M` overtakes what the band asks for,
which on this ladder is between `n1 = 6` and `n1 = 12`.

Note also that α_eff falls along the ladder (0.50, 0.37, 0.24, 0.19): the
sharper the shape, the *weaker* the grading exponent needed to span the same
band, because the curvature contrast it is spanning is larger.

1. **Anchor the band at the curvature *maximum*, not the minimum.** The stars
   carry a curvature contrast of ~8.8e3 against a band of 5. Scaling `σ` up
   from `min κ` makes the *upper* bound bind over essentially the whole
   boundary: grading degenerates into uniform arc length with a few
   artificially coarsened points, which is the opposite of the intent.
   Anchored at `max κ`, σ = C sits at the sharpest point and σ = 1 on the
   smoothest, as §4 specifies.

2. **Derive α from the band; do not pin it.** With `α = 1/2` fixed, the natural
   spread is ~93× against a band of 5×: the clamp saturates and the density is
   two-level rather than graded. `α_eff = min(α, ln C / range(ln|κ|))` spans
   the band smoothly. On the mild star the cap does *not* bind, so `α = 1/2`
   stands and the band's top is simply unreachable — R spans 20–29 of a
   requested 20–100, because a nearly-round shape has no curvature contrast to
   grade. The band is a bound, not a promise.

3. **The smooth clamp is then never active, which is the result.** With α
   derived, hard and smooth clamps are indistinguishable. Pin α so the band
   binds and the difference is enormous: the `w(t) − t` tail at mode 500 is
   **2× / 4.6e6× / 1.4e7× / 1.9e3×** worse under `np.clip` across the ladder
   (the mild star escapes because at `M = 4` the map is almost trivially
   smooth either way). So §4's trap is real but reachable only by pinning α —
   the smooth saturation stays in as a guard, not as the normal path.

4. **Bandwidth must not be tied to `nn`,** which §4 suggests. A bandwidth ∝ nn
   makes `w` a different map at every resolution: refinement chases a moving
   target and `∫w' dt = 2π` stalls near 1e-3 instead of converging. It is also
   invariant 13.3 in another guise — the frozen object has to be one map. The
   `nn` coupling survives only as the **check** `nn ≥ 4M` in the table above,
   whose failure says the band is too coarse to carry a graded map on that
   shape rather than that the map is wrong.

   A first attempt tied `M` to a detected lobe count and got the ordering
   **backwards** — the sharpest star drew the *lowest* bandwidth (`M = 8`),
   because the log-curvature's dominant harmonic reads 8, 8, 12 and 4 across
   four shapes of identical symmetry. The layer-width rule replaces it and is
   monotone in sharpness by construction.

5. **Smooth with a positive kernel, not a sharp Fourier cutoff.** Truncation
   Gibbs-undershoots the peaked log-curvature and digs a narrow artificial
   minimum; since `min σ = 1` is what pins `nn`, the band then gets set by the
   undershoot and every real node is over-resolved — measured on the mild star,
   where every node landed at `R ≈ 28.7` for a requested 15–30. A Gaussian
   (heat-kernel) taper cannot leave the range of its input.

6. **Grading at a fixed `R_min` adds nodes; it does not save them.** 38 vs 32
   on the mild star and 94 vs 56 at the corner, because the band asks for
   *more* resolution at the peaks while holding the floor. The gain is accuracy
   at equal `nn`, so the fair const-density baseline is uniform arc length at
   the adaptive `nn` — that is what the figure overlays, and what the G2 pass
   criterion has to mean.

**`nn` is derived from the band, not supplied** (D17: rungs in `R`, never in
raw `n_pts`). `min σ = 1` gives the first estimate
`nn = ⌈R_min·Z·n_core/λ_ref⌉`, which over-resolves by up to ~2× because nodes
are sparsest exactly where σ is smallest, so a narrow σ minimum goes unsampled.
Descending from it while **verifying** each candidate — not extrapolating,
which lands below the band — gives the `nn` column above.

**Scale covariance (§9) holds by construction.** The density reads only
`|κ|·L`, so it carries no absolute length; the `R` band's absolute level enters
`nn` alone, and `nn` is unchanged when `rad` and λ scale together. A density
that compared λ_ref against an absolute `Δs` pointwise would move the nodes
under `rad` scaling alone — a §9 violation the specification invites and this
construction avoids.

**Not established here:** any convergence rate; the clamp's effect on solution
error as opposed to map smoothness; whether `α_eff` is right for *accuracy*
rather than for filling the band; and the constant in `nn ≳ 4M`, which is a
Nyquist-style rule of thumb and wants pinning against a measured rate once the
Kress prototype exists.

#### Orientation study — the ladder that says grading cannot pay yet (2026-09-12)

Script: `g2_pole_ladder.py`; figure `pole_ladder.png`. **Preliminary,
orientation only**, and run on **shipped 0.5.0** — Kress is not implemented and
G0 has not run, so the quadrature is Maradudin and the rate is 1 by
construction. What is being read is the error *constant*.

Equal-area ellipses, `m = 4`, `n1 = n2 = n3 = 2`, `a = 1/√A`, `b = √A`, so
`a·b = 1` and the area is the circle's at every aspect — scale covariance (§9)
makes a pure size change exactly `λ → s·λ`, so holding the area removes that
trivial motion and leaves the shape effect. Tracked mode: **TE n = 0 at
530.83214 + 26.37850j**, the simple anchor of `tests/test_qnm.py`
(rad 200, n_core 3.0, TE). Angular order 0, so it stays simple under elongation.
R band 20–80, adaptive map rebuilt per shape, `λ_ref` fixed per shape at the
predicted Re λ (invariant 13.2).

**The result: at equal `nn`, adaptive grading is 1.4–1.6× *worse* than constant
density, and the mechanism is arithmetic rather than a defect.**

| aspect | `nn` | adaptive \|err\| | const-density \|err\| | ratio | R_worst ratio |
|---|---|---|---|---|---|
| 1 | 72–286 | 1.27 / 0.62 / 0.31 | identical | 1.00 | 1.00 |
| 2 | 144 | 3.459e-1 | 2.367e-1 | 1.46 | 1.41 |
| 3 | 208 | 2.615e-1 | 1.651e-1 | 1.58 | 1.51 |
| 4 | 246 | 2.203e-1 | 1.529e-1 | 1.44 | 1.39 |

Observed order 1.00–1.10 on every rung and both schemes, as expected. On the
circle the two schemes are **bit-identical**: κ is constant, so σ ≡ 1 and the
adaptive map collapses to uniform arc length — a free correctness check on the
whole construction, and the reason the aspect-1 row has no ratio.

**Why grading loses, and the condition under which it wins.** A first-order
scheme's error is set by the **worst-resolved node**, so at fixed `nn` any
grading strictly increases `max Δs` — it buys resolution at the tips by
coarsening the flanks, and the flanks are what the error is measuring. The
error ratios (1.46, 1.58, 1.44) match the worst-node resolution ratios (1.41,
1.51, 1.39) **to within 4 %**, which is the mechanism stated quantitatively.
Grading can therefore only pay when the error is controlled by the
*analyticity* of the integrand rather than by `max Δs` — that is, once the
quadrature is spectral. It predicts nothing about Kress, and it does say
**adaptive density must not ship on the Maradudin scheme.**

This is the same shape of result as handoff §9, where uniform arc length was
worse than uniform θ on a star, and it is consistent with it: both are cases of
redistributing nodes under a scheme whose error the redistribution cannot help.

**Consequence for the gate order.** G2's stated criterion — "adaptive beats
const-density at equal `nn`" — is **unachievable before Kress lands**, and
would have failed here for a reason that has nothing to do with the density.
G2 should be re-sequenced after the Kress prototype, or its criterion restated
as a property of the map (which the 12 Sep findings above do measure) rather
than of the solution error.

**Mode identification was the fragile part, not the quadrature.** The first
attempt — aspect steps of 0.25, nearest-neighbour tracking, a fixed ±14 nm box
— hopped silently between branches: Q swinging 10 → 48 → 25 → 40 along what
should be a smooth trajectory, and two walks with different step sizes
reporting *different modes at the same aspect* (550.5+10.9j against
529.6+29.9j). Nothing was raised and `edge_margin` stayed healthy throughout.
Fixed with 0.05 steps and a **secant predictor**, which shrinks the box to the
trajectory's curvature rather than its whole step; the repaired trajectory is
monotone in Re λ with Q pinned at 10.1–11.4, one mode per box, no widening.
Whatever continuation the downstream ladder uses needs a predictor and a
per-step ambiguity tell, not proximity to the previous solve. This measurement
is what §3's standing rule and conventions §8 are written from.

**An identification check beyond proximity.** The n = 0 mode is radial, so it
should track the **minor** semi-axis `rad/√A`; measured `Re λ(A)/Re λ(1)`
follows `A^{-1/2}` to ~8 % across the whole ladder (0.552 against 0.500 at
aspect 4). Not an independent anchor — the ellipse reference here is Richardson
self-convergence (§12, exponent pinned at 1), and CLAUDE.md non-negotiable 3
still wants Mathieu — but it is a physical argument that the tracked branch is
the intended one, which proximity alone cannot give.

**Also worth carrying forward:** holding `R ≥ 20` costs `nn` 144 → 490 from
circle to aspect 4 under grading, against 144 → 352 at constant density, and
`λ_ref` shrinking along the ladder (530 → 293 nm) is most of that growth — `R`
is points per *interior* wavelength, and the wavelength is getting shorter.

#### G2 findings, part 2 — the solution-error rate, now that Kress is landed (2026-09-13)

Script: `g2_solution_error.py`; full findings in
[g2-solution-error-findings.md](g2-solution-error-findings.md). Closes the
three items the 12 Sep findings above left open, running entirely through
**production** code on the landed Kress path (`771bc9b`): `Parametrisation`,
`Geometry.gielis(parametrisation=...)`, `BIESolver.scatter`. Shape: the G2
ladder's baseline star (`m=4, n1=6, n2=n3=12`), TE, `qext` (not `qsca` — the
angular-quadrature trap), reference = uniform-θ Kress at `nn=800` (its own
floor measured at 1.3e-15, five decades below every reported error). Rate is
the least-squares slope of `ln err` vs `nn` over rungs `nn ≥ 4M = 72`
(rounded to the next ladder step, 80) — below that the graded map is
under-sampled by its own construction, per the 12 Sep table, and a rate fit
there is not measuring accuracy.

**Both pass clauses are now met, reproducibly across three wavelengths (600,
450, 900 nm, `b` stable to <8%):**

1. **Adaptive beats uniform arc length, decisively, at equal `nn`**: rate `b`
   0.0657 vs 0.0416 (1.58×), error 46–104× smaller across `nn = 80–200`. This
   reverses the 12 Sep orientation-ladder finding, and the reversal has a
   cause: pre-Kress the error is set by the worst-resolved node, so grading at
   fixed `nn` cannot win by construction; under Kress the error is set by the
   analyticity strip of `f∘w`, where grading toward curvature helps. **The
   "superseded" note above, which restated this clause as unachievable, is
   itself now superseded** — it was correct for the shipped (pre-Kress) scheme
   and for the ellipse it was measured on, not in general.
2. **Smooth clamp beats hard `np.clip`, decisively, on solution error** (not
   just map smoothness): at matched realised contrast, 1.53× in rate and 10.7×
   in error at `nn=200`. Stronger and solve-free: the hard-clipped `σγ` is C⁰,
   so its spectral tail (1.3e-5 at `N_f=4096`) never reaches the production
   constructor's round-off acceptance threshold (`4·eps = 8.9e-16`, itself
   justified as the doubling loop's own termination criterion) — the smooth
   clamp is not a refinement of `np.clip`, it is what makes `Parametrisation`'s
   Newton/series construction terminate at all.

**The caveat that keeps the gate from being a green light for the density in
general:** on this same shape, at the same `nn`, the *production default*
(uniform θ, no grading) still beats adaptive by 1.37× in rate and 46× in error
at `nn=200`. The density is well constructed and the gate's stated criterion
is met against the baseline it names — but nobody should prefer it to uniform
θ on this shape. Whether it acquires a shape where it should is **G3's
question** (the cost-curve table above predicts the knee between `n1=6`, used
here, and `n1=12`); this study does not move conventions §13's "still
tentative" note to settled.

Also measured: `α_eff` derived from the `R` band (D17) sits inside a genuine
interior accuracy optimum (`α_eff ≈ 0.20–0.25`) on this shape — it is not
merely band-filling — but the optimum tracks the band (`C=2→0.123` up to
`C=20→0.293`), so the un-derived, un-validated knob is the band's own
calibration, not `α_eff` itself. Not run here: a near-corner shape (G3's, not
this gate's), a complex-λ/QNM probe (G4's), and a second observable — all
listed as what would change the verdict, in the findings doc.

#### The same ladder under Kress — spectral, and grading still does not pay (2026-09-12)

Scripts: `kress_t.py` (assembly), `kress_pole_ladder.py` (study); figure
`kress_pole_ladder.png`. Same shapes, same mode, same continuation as the
orientation ladder above — only the quadrature changes.

**The assembly works in `t`, not θ.** The inherited `exp_kress_full.py` uses
`geom.theta` both in the singular factor `4 sin²((θ−θ')/2)` and as the step
`2π/N`, which is correct only for nodes equispaced in θ; under grading it is
**silently wrong** (conventions §13.1). Rather than carry the `w'` Jacobian
through every term, `kress_t.py` samples at `θ_j = w(t_j)` and takes **every**
derivative with respect to `t` spectrally by FFT. The Jacobian then never
appears, and the analytic-quality `ddf`/`ddg` that the handoff's
machine-precision column needed come for free (`_der_real_3` caps the rate at
3, §6.4).

**Two anchors had to be fixed before anything could be measured.**

- *The test-suite constant is rounded.* `tests/test_qnm.py`'s
  `530.83214 + 26.37850j` is quoted to five decimals, which reads as a fixed
  **1.259e-6 nm** error — invisible against the shipped 0.38 nm and dominant
  under Kress. Newton on `qnm_denominator` gives
  `530.8321407288238 + 26.37849897377869j`, and the offset is exactly the
  1.2587e-6 seen. **The suite's anchors need more digits when it is
  re-anchored during the migration.**
- *The contour was the floor at `n_quad_per_side = 6`.* It pins at 5e-6 there
  and drops to the discretisation once raised; 12 is used throughout. The
  justification in `test_qnm.py` ("identical modes to 1e-8 against a 0.38 nm
  discretisation error") is a statement about the *shipped* error and does not
  survive the migration.

**Parametrisation invariance is the correctness test, and it passes.** A
circle's poles cannot depend on how the circle is parametrised, so a
deliberately graded map `θ = t + 0.3 sin 2t` must give the same pole as uniform
θ. It does, at **1.6e-11** once the map is resolved (`nn ≥ 60`); a misplaced
Jacobian fails this. Against the full-precision Mie pole on the circle:

| nn | Kress uniform | Kress graded | shipped |
|---|---|---|---|
| 20 | 1.198e-3 | 5.348e-1 | 4.92 |
| 30 | 1.652e-9 | 4.259e-3 | 3.20 |
| 60 | 1.146e-11 | 1.600e-11 | 1.53 |
| 120 | 1.066e-11 | 1.083e-11 | 0.75 |

**Spectral convergence survives the deformation** — the result the migration
rests on, now shown on a non-circular shape and on a QNM rather than on `qext`.
At aspect 2 the constant-density error runs 3.5e-1 → 1.4e-3 → 7.7e-7 → 6.9e-10
→ 2.2e-12 across `nn = 20 … 80`. The shipped scheme needed `nn = 572` to reach
8.2e-2 on the same shape.

**Grading still does not pay, for a different reason than before.** At equal
`nn`, adaptive against constant density (error ratio, >1 means grading loses):

| aspect | nn 20 | 30 | 40 | 60 | 80 | 120 |
|---|---|---|---|---|---|---|
| 1 | 1.00 | 1.00 | at floor | at floor | at floor | at floor |
| 2 | 15.0 | 119 | 5842 | 107 | 0.25 | 0.93 |
| 3 | 1.44 | 63.4 | 1364 | 82.5 | 0.48 | **0.01** |
| 4 | 2.08 | — | — | — | 0.61 | 0.88 |

Grading **delays the onset of spectral convergence**, heavily — up to 5800×
pre-asymptotically — and then buys a modest-to-large constant once both schemes
are converging (100× at aspect 3, `nn = 120`; a wash at aspect 4). On the
circle the two are identical at low `nn` because κ is constant, so σ ≡ 1 and
the maps coincide.

**The `nn ≳ 4M` rule does not explain this, and is therefore necessary rather
than sufficient** — correcting the framing of finding 4 above. Aspect 2 has
`M = 5`, so `4M = 20`, yet grading is still 5842× worse at `nn = 40`. The
mechanism is not resolving the density: composing with `w` narrows the
**analyticity strip of `f∘w` in the `t`-plane**, and the spectral rate is set
by that strip. Resolving the density is a floor, not a guarantee.

**What this says about the release, stated carefully.** On ellipses to aspect
4, the curvature-adaptive density (scope item 4) **is not earning its place**:
the smooth parametrisation (item 3) delivers the spectral convergence, and
grading mostly costs. That is not a verdict on grading in general — these are
ellipses, whose curvature contrast is ~3 against the ~8.8e3 of the sharp stars,
and the near-corner regime grading exists for is exactly what **G3** measures
and what this ladder does not touch. The honest reading is that **item 4 should
be justified by G3 or deferred**, and that the case for it cannot be made on
smooth shapes.

**Also measured:** at aspect 4 and `nn ≤ 60` the mode is not identifiable by
either scheme — the probe saturates on spurious spectrum. That is a property of
the resolution, not an error, and the sweep records it rather than aborting.

#### The default node map under Kress — uniform θ, and where arc length still wins (2026-09-13)

Script: `kress_default_map.py`; tables in
[../kress-spec.md](../kress-spec.md) Appendix A. Asked because uniform θ
undersamples the arms of sharp stars, which is why v0.5 defaulted to uniform
arc length. Three maps — uniform θ, uniform arc length, adaptive band 20–100 —
on the rounded-square ladder (`m = 4`, 2/4 … 20/50, plus the `m = 6` §9 star)
and on a spiky ladder (`m = 6`, n2 = n3 = 8, n1 = 4, 2, 1, 0.5: arm ratio 1.7 …
64), probe `qext`.

- **Uniform θ is the most accurate map once resolved, on every shape
  measured.** At `nn = 640` on the near-corner 20/50: θ 2.2e-6, adaptive 6.8e-6,
  arc length 1.8e-4. On the arm-ratio-8 spike at `nn = 960`: θ 5.8e-9, arc
  length 3.1e-4.
- **The v0.5 argument is true, but only pre-asymptotically.** Arc length is 2–7×
  better at `nn = 40` on the rounded squares, and up to 30× better at `nn = 120`
  on the arm-ratio-8 spike, where uniform θ's widest gap (the arm flanks) leaves
  its worst node near `R ≈ 4`. θ overtakes by `nn = 320` there.
- **The arc-length map is not the defect.** It matches an independent
  `scipy.quad` inversion to 2.7e-15, with density residual 6.6e-14; its
  convergence still stalls non-monotonically near 3e-4 on the spike. Composing
  with `w` narrows the analyticity strip, the mechanism the ellipse ladder above
  already found. A related observation, **not** a resolution of handoff §9:
  under Kress with an exact smooth map, uniform arc length is still slower than
  uniform θ on the §9 star, so `np.interp` is not needed for that ordering. §9
  itself was measured on the shipped scheme and stays open by decision.
- **Adaptive grading does not beat plain arc length** in the one regime where
  grading helps at all. That regime — spiky, under-resolved — is G3's question.
- `Parametrisation.gielis` fails on the arm-ratio-64 spike (Newton) and on the
  exponent-1 star (`ZeroDivisionError`); uniform θ runs on both.

**Decision (owner's question, answered by this measurement):** the v0.6 default
is uniform θ; arc length stays available as `parametrisation=
Parametrisation.gielis(...)`. Recorded as kress-spec D1 and conventions §13.

#### Spiky stars at campaign resolution — no map rescues low `nn` (2026-09-14)

Script: `spiky_low_nn.py`, nine jobs on four cores, 22 s. Asked for
exploratory campaigns: at `nn` 50–300 and a useful accuracy of **1e-3
relative**, is uniform arc length the most accurate map on a spiky star?
Stars `m = 6`, n2 = n3 = 8, arm ratio 4 / 8 / 16 (n1 = 1.5 / 1 / 0.75), tip
400 nm; `qext`, TE, n_core 1.5, at λ = 450, 600, 900 nm; reference uniform θ
at `nn = 1600`, self-converged to ≤ 7e-11 against 2000.

Smallest `nn` on the ladder that reaches 1e-3 (at that rung and above), per λ:

| arm ratio | uniform θ | arc length | adaptive |
|---|---|---|---|
| 4 | 200 / 200 / 300 | 300 / — / — | 300 / 300 / — |
| 8 | — / — / — (best 1.3e-3 at 300) | — / 300 / map fails to build | — / — / — |
| 16 | — (1e-2 – 1e-1 at 300) | fails to build | — |

**The answer is no.** Arc length is not the most accurate map in this regime
at any arm ratio, for three reasons:

1. **Below ~1e-2, which map wins is noise.** Errors swing 3–10× between
   adjacent rungs for every map (arm ratio 8, λ 600: uniform θ 2.6e-2 at
   `nn = 100`, 7.3e-2 at 150). Appendix A's "arc length 7–60× better at `nn`
   80–120" was read off that noise at one λ; here, at `nn = 100`, the two
   split one wavelength each (arc length fails to build at the third).
2. **Where 1e-3 is reachable (arm ratio 4), uniform θ reaches it first**, at
   `nn = 200` against 300, and is the only map below it at 300 on all three
   wavelengths.
3. **Arc length is not usable in production on these shapes.** Newton in
   `Parametrisation.nodes` fails at `nn` = 50, 75, 150 on arm ratio 8, and
   the constructor fails outright at arm ratio 8 / λ 900 and on every arm
   ratio 16 job (adaptive on one of three). Recorded as a defect alongside the
   G3 flat-point one; not fixed here.

**For campaigns.** Use uniform θ. At arm ratio ≤ 4, `nn = 200` gives 1e-3 and
300 gives ~1e-4. At arm ratio ≥ 8, no map reaches 1e-3 by `nn = 300`, so the
remedy is resolution, not node placement: Appendix A has uniform θ at 6.4e-5
by `nn = 480` on arm ratio 8.

**Not established:** a second observable; TM; arm ratios between 4 and 8,
where the 1e-3 crossover for `nn ≤ 300` lies.

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

#### G3 findings — the envelope, and the first shape where grading pays (2026-09-13)

Script: `g3_corner_envelope.py`, one exponent per process. Superellipse
`n = 2, 4, … 256` (even only — odd `n` is finitely smooth at the axes and would
measure regularity, not sharpness), `rad` 200, TE, n_core 1.5, λ 600 nm, `qext`
against uniform-θ Kress at `nn = 1280`. Production code throughout, except the
density floor below. `κ_max·rad ≈ 0.70·n` along the whole ladder (2.5 at
`n = 4`, 90 at 128, 181 at 256), which is the conversion D6 needs.

**Envelope.** Relative `qext` error at `nn = 320` (a little above the current
default `n_pts = 200`), reference floor in the last column:

| n | κ_max·rad | 4M | uniform θ | arc length | adaptive 20–100 | ref floor |
|---|---|---|---|---|---|---|
| 4 | 2.5 | 10 | 1.0e-15 | 2.9e-16 | 2.9e-16 | 7e-16 |
| 8 | 5.4 | 27 | 1.2e-15 | 1.2e-15 | 1.4e-16 | 2e-15 |
| 16 | 11 | 61 | 2.3e-15 | 1.2e-10 | 4.1e-16 | 2e-15 |
| 32 | 22 | 129 | 1.1e-9 | 2.5e-7 | 2.2e-12 | 5e-16 |
| 64 | 45 | 266 | 7.2e-7 | 6.3e-6 | 2.9e-9 | 2e-11 |
| 128 | 90 | 538 | 2.7e-5 | 2.1e-5 | 3.9e-7 | 1e-8 |
| 256 | 181 | 1084 | 8.4e-5 | 3.3e-5 | 2.8e-6 | 3e-7 |

**The number:** at a 1e-6 relative `qext` target and `nn ≤ 320`, **graded Kress
delivers to `n = 128` (`κ_max·rad ≈ 90`) and stops at `n = 256`**; uniform θ
stops one rung earlier, at `n = 128`. Doubling to `nn = 640` buys one more rung
for uniform θ (1.3e-7 at 128) and leaves graded Kress at 5e-8 on `n = 256`,
which is already at that shape's reference floor — beyond `n ≈ 256` this sweep
cannot resolve anything, so that is where the envelope is honestly stated to
end. Every entry sits below the shipped first-order scheme's ~1.5e-3 at
`nn = 300` *on the circle*, its easiest shape; the envelope is where Kress stops
being spectral at practical `nn`, not where it stops beating v0.5.

**G2's knee prediction holds.** The band-derived `nn` stays at 80–110 along the
ladder, and `4M` overtakes it between `n = 16` (61) and `n = 32` (129). That is
exactly where the ordering flips: uniform θ is the most accurate map through
`n = 16`, and **adaptive beats it at equal `nn` from `n = 32` on** — 500× at
`n = 32`, 250× at 64, 70× at 128 (`nn = 320`). This is the regime the
2026-09-13 G2 caveat and the default-map study both deferred to G3, and it
answers scope item 4: **grading has a shape class where it pays** — flat-sided,
corner-dominated shapes with `κ_max·rad ≳ 20`. Plain arc length never beats
adaptive, and only overtakes uniform θ once both are out of the envelope
(`n ≥ 128`).

**Defect found: `Parametrisation.gielis` raises on every `n ≥ 4`
superellipse**, including the `C = 1` arc-length map that needs no curvature.
The sides are flat points (`κ = 0` on the axes), `ln|κ|` in `_density` is −∞,
`0·NaN` survives even at `α_eff = 0`, and the doubling loop exits with
`ValueError` at `N_f = 2^20`. The study wraps `_speed_curvature` with a smooth
relative floor `hypot(κ, ε·max|κ|)` (no absolute length, §9). The arc-length
map is floor-independent to all digits, as it must be. **The adaptive map is
not:** `ε = 1e-3` gives `α_eff ≈ 0.18`, `ε = 1e-6` gives `≈ 0.10` — the floor
enters `range(s̃)` and so `α_eff` — and the error moves by up to 200× (`n = 8`,
`nn = 80`: 1.7e-7 vs 3.4e-5). The table uses `ε = 1e-3`. **So a flat point is
not just a crash to guard; the density needs a stated floor, and that floor is
a design parameter production does not have yet.** Code-spec item, not fixed
here.

**Not established:** the floor's accuracy-optimal value; any shape with true
inflection points (a superellipse's curvature has zeros but no sign change);
anything past `n = 256`; QNM or complex λ (G4).

### G4 — Holomorphy and the QNM path

Confirm the contour method still works: `R`, `L` and `h` are λ-independent by
construction and every Bessel factor is holomorphic, so Beyn's premise should
survive — but it must be *shown*, because degradation is silent. Mode count and
`edge_margin` are the tells.

Also confirm `w` does not move with λ (invariant 3.3.2) or with a differentiated
parameter (3.3.3), and check whether `dλ/dp` still needs `richardson_limit`.

*Passes when* QNM positions against analytic Mie poles improve in rate, and the
mode count is stable across the contour.

> **PASSED (2026-09-14), on uniform θ and adaptive maps.** Uniform arc length
> was not measured: it is disqualified as a default (§13) and kept as legacy.
> One of this gate's own premises is corrected below — mode count and
> `edge_margin` are **not** the tells for lost holomorphy; `sigma_ratio` is.

#### G4 findings — holomorphy holds, the tells are not the ones named (2026-09-14)

Script: `g4_holomorphy_qnm.py`, four jobs (`holo`, `circle`, `shape NAME
MODE`), ~8 min wall on two processes. Production code throughout (`QNMSolver`,
`refine`, `sensitivity`), except the superellipse adaptive map, which takes G3's
curvature floor `ε = 1e-3` because production cannot build it. Material is the
QNM fixture: rad 200, n_core 3.0, TE. Contour 12 nodes per side, every pole
polished by `refine`.

**1. Invariant 13.2 holds by construction, and more strongly than stated.**
The adaptive map's nodes are **bit-identical** at `wavelength_ref` = 450, 600
and 1550 nm on the ellipse and the star; `wavelength_ref` reaches only
`nn_from_band` (234/176/68 on the ellipse). `_density` has no λ input at all,
so no choice of reference wavelength can move a node.

**2. `M(λ)` is holomorphic, measured directly.** Central differences of `M`
along Re λ and along Im λ (Cauchy–Riemann) agree to 2.9e-6 / 2.9e-8 / 3.1e-10
at h = 0.1 / 0.01 / 0.001 nm — exactly h², i.e. zero in the limit — and match
`assemble_derivative` at the same rate. Identical on the aspect-2 ellipse, the
6/12/12 star and the n = 32 superellipse, on both maps, at `nn = 120`.

**3. Negative control — and the correction to this gate's premise.** A circle
on the graded map `θ = t + ε sin 2t` with `ε = 0.3 + slope·(Re λ − 530)/25`. At
each fixed λ the map is valid and gives the Mie pole (parametrisation
invariance, 1e-11), so everything Beyn gets wrong is holomorphy loss alone:

| slope per 25 nm | modes | `edge_margin` | `max_gap` | `sigma_ratio` | \|λ − Mie\| nm |
|---|---|---|---|---|---|
| 0 | 1 | 0.433 | 1.8e7 | **8.1e-15** | 1.1e-11 |
| 1e-4 | 1 | 0.433 | 2.4e4 | **3.0e-7** | 3.9e-4 |
| 1e-3 | 1 | 0.433 | 2.4e3 | **3.0e-6** | 3.9e-3 |
| 1e-2, 1e-1 | raises "probe saturated" | | | | |

Mode count and `edge_margin` are **blind** to it — same count, same margin to
three digits, while the pole moves linearly with the violation. The rank gap
`max_gap` shrinks but stays far above any threshold. `sigma_ratio` — `σ_min/σ_max`
of `M` at the returned λ — rises from round-off to 3e-7 at the mildest
violation: a pole that is not a singular point of `M` is the direct signature,
since a holomorphic `M` puts Beyn's eigenvalues exactly on its singularities.
Beyond that the failure is loud (the probe saturates). **So the tell is
`sigma_ratio`, and any holomorphy guard written into the migration should read
it, not the mode count.**

**4. Circle against Mie: spectral on both maps, count stable throughout.** Box
`515+5j … 565+35j` (TE n = 0 simple, TE n = 2 pair), `|λ − Mie|` in nm, worst of
the three modes:

| nn | 20 | 30 | 40 | 60 | 120 | 160 | 240 | 320 | 480 |
|---|---|---|---|---|---|---|---|---|---|
| uniform θ | 1.9e-2 | 1.1e-7 | **2.6e-13** | 1.0e-12 | 5.8e-13 | 8.2e-13 | — | — | — |
| star-6 adaptive map | 2.1 | 6.5e-2 | 7.1e-1 | 3.8e-2 | 5.2e-4 | 8.0e-5 | 2.5e-7 | 1.1e-9 | 1.0e-12 |

The graded map is the production adaptive map of the star, applied to the
circle — a genuinely graded, C₄-symmetric node set. Mode count is 3 at **every**
rung on both maps, including `nn = 20` with a 2 nm error: count stability is
necessary and says nothing about accuracy. Contour vs Newton agree to ≤ 1.7e-12
everywhere. On the graded map the n = 2 pair **splits by the error's own size**
(2.3 nm at 20, 1.1e-9 at 320, 1.2e-12 at 480) and `multiplicity` reads
`[1, 1, 1]` until 160, `[1, 2, 2]` from 240: C₄ nodes cannot represent the
`cos 2θ`/`sin 2θ` degeneracy exactly, so the splitting is a free, reference-free
discretisation estimate. Grading costs the circle 10⁸× at `nn = 160`, the
analyticity-strip mechanism of the ellipse ladder.

**5. Continuation from the circle: clean on all four paths.** Equal-area paths
(`rad` rescaled so the area stays `π·200²` — a pure size change is exactly
`λ → s·λ` by §9, and the first unnormalised step of the star swelled the area
8 % and moved Re λ ~20 nm out of the box), 40 steps, `nn = 120`, secant
predictor, ±3 nm box. Every path starts on the Mie pole to ≤ 2.3e-13, **never
widens, never loses the mode, one mode (the pair: two) per box**, and the
second difference of the trajectory is largest at the first step and decays
monotonically — no kink. The n = 3 pair stays degenerate to 2e-13 the whole way
(C₄v is preserved), while its Q falls 48 → 11. Endpoints: ellipse
404.741+18.109j, star n = 0 498.467+22.255j, star n = 3 820.988+36.612j,
superellipse 516.731+24.277j.

**6. Non-circular QNM ladders.** Reference uniform θ at `nn = 480`; its floor,
from the adaptive map at the same `nn` (a different node set), is 1.4e-14 /
3.9e-14 / 2.6e-13 / 6.9e-11 nm on the four endpoints. `|λ − ref|` in nm:

| endpoint | map | 40 | 80 | 160 | 240 | 320 |
|---|---|---|---|---|---|---|
| ellipse A=2 | uniform θ | 8.0e-5 | 1.0e-10 | floor | floor | floor |
| | adaptive | 6.2e-4 | **floor** (60: 1.7e-10) | floor | floor | floor |
| star 6/12/12, n=0 | uniform θ | 1.5 | 3.1e-2 | 3.7e-6 | 3.0e-9 | **2.1e-12** |
| | adaptive | 1.3 | 4.0e-2 | 4.3e-5 | 2.1e-7 | 1.2e-9 |
| star, n=3 pair | uniform θ | 2.9 | 3.9e-2 | 1.7e-5 | 1.1e-8 | **8.2e-12** |
| | adaptive | 1.1 | 4.5e-2 | 1.5e-4 | 7.5e-7 | 4.1e-9 |
| superellipse n=32 | uniform θ | 5.7e-1 | 3.4e-2 | 5.9e-4 | 1.2e-5 | 2.2e-7 |
| | adaptive | 2.2e-1 | 6.2e-3 | **3.4e-5** | **2.5e-7** | **2.3e-9** |

**The `qext` map ordering carries over to poles unchanged.** On the star,
uniform θ wins by 10–570× once converging (G2 part 2's caveat, now on a
complex-λ observable and a degenerate pair); on the superellipse — G3's
grading-pays class, `κ_max·rad ≈ 22` — adaptive wins by 17× / 48× / 96× at
`nn` = 160 / 240 / 320. On the aspect-2 ellipse adaptive reaches the floor one
rung earlier (`nn = 60`–80), a different verdict from the `kress_pole_ladder`
table, which compared adaptive against uniform *arc length*, at band 20–80 and
`n_core = 3`; not investigated further. **Grading neither breaks nor delays the
QNM path in any way the scatter path does not already show** — its effect is
the same analyticity-strip trade.

Tells at the endpoints: mode count is stable across `nn` and across
contours — a shifted, larger box and 16 nodes per side return the same count
and the same poles to ≤ 1.7e-13 nm on every endpoint and both maps. The ladder
boxes (±5 nm) on the star and superellipse n = 0 hold a second, neighbouring
mode at `edge_margin` 0.05–0.12; the tracked pole is unaffected, and this is the
§8 landscape note in action — the continuation's ±3 nm box never saw it.

**7. `dλ/dp` on the frozen map: spectral, gauge-free, second order in the step,
and `richardson_limit` makes it worse.** `p = b` on the ellipse, `n1` on the
star and superellipse, map frozen at `p₀` (conventions §10).

- *Ladder:* relative error vs the reference at `nn = 320` is 1.2e-11 / 2.3e-11
  / 1.8e-11 / 1.9e-8 on uniform θ, 2.3e-12 / 3.6e-11 / 8.9e-12 / 7.2e-10 on
  adaptive (ellipse / star n=0 / star n=3 / superellipse), tracking the λ ladder
  above — including the superellipse, where adaptive is 26× better.
- *Gauge-free:* `J` on the adaptive map at `nn = 480` agrees with uniform θ to
  3.7e-11 / 1.7e-11 / 1.4e-11 / 6.8e-11. Two frozen maps are two
  discretisations of one derivative, as §10 argues.
- *Adjoint vs re-extracted poles* (§11 Gate 3, `nn = 160`): relative
  disagreement falls 98.4× / 99.9× / 99.9× / 95.1× (uniform θ) and 98.4× /
  99.9× / 99.9× / 99.0× (adaptive) per decade of step, 1e-2 → 1e-3 — second
  order, on both maps and on the degenerate pair's secular branch.
- *Degenerate pair:* the two `dλ/dn1` of the star n = 3 pair agree to 8.7e-10
  — `n1` preserves C₄v, so they must not split.
- *Richardson* on `(J₁₆₀, J₃₂₀)`, exponent pinned at 1: uniform θ 2.3e-11 →
  4.4e-7 (star), 1.9e-8 → 8.8e-5 (superellipse); adaptive 3.6e-11 → 1.6e-6,
  7.2e-10 → 8.3e-7. **Four to five orders worse.** The F7 deprecation stands,
  now measured on non-circular shapes and both maps.

**Not established:** TM; a pole close enough to another to challenge
`DEGENERACY_RTOL` under grading (the circle's n = 2 split crosses it cleanly
between `nn` 160 and 240, unexamined in between); a derivative that *breaks*
the pair's symmetry on a graded map; the superellipse floor's effect on the QNM
ladder (G3 found up to 200× on `qext`, only `ε = 1e-3` run here).

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

  Continuation from the analytic circle (§3, conventions §8) does **not** close
  this item and must not be mistaken for it: it identifies *which* mode is being
  measured and flags defects through the smoothness of the trajectory, but the
  quantity it is compared against on the deformed shape is still a
  self-convergence extrapolant. It is a check, not an anchor.
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
