# Multiple particles — high-level handoff

**Status: guide, 20 Sep 2026.** This is **not** a code-spec and must not be
executed as one. It fixes the goal, states what the legacy implementation
actually provides and what it silently assumes, lists the decisions a code-spec
has to take before any of it can be written, and sets the validation ladder. The
code-spec comes later and separately.

**Where the legacy code is:** `~/Documents/sie-legacy-0726/sie/bie.py`, lines
540–1000. It is the pre-v0.6 research solver, so none of it can be copied: it
predates Kress–Martensen quadrature, the `Parametrisation` node map, and the
vacuum-wavelength convention. Read it for the physics, not the code.

**The target this document commits to:** *arbitrary* particles — different
shapes, different sizes, different materials, and **different `n_pts` per
particle**. The legacy code supports none of those four, and the gap between
what it does and what we want is the substance of the work.

---

## 1. Why this is the last big thing, and why it gates v1.0

The package's stated scope is "the homogeneous-background, **single-particle**
core" ([README.md](../../README.md) §Scope). Multiple particles is the one
roadmap item that changes the shape of the public API rather than adding to it:
`BIESolver(geometry, material)` is singular in both arguments, and a cluster is
a list of geometries *and* a list of materials.

That makes it the remaining API-breaking item, and therefore — together with the
v0.8 external validation — the gate on **v1.0**. A 1.0 is a promise that the
surface is stable. Until the multiparticle API shape is decided, that promise
cannot honestly be made. §4.1 is where that decision is taken, and the
recommendation there is the one that lets 1.0 ship without waiting for the
feature.

## 2. The physics, and the one fact that makes it cheap

For `Np` particles the coupled system is `2·Σ_p nn_p` square. Each particle
contributes its **own single-particle matrix on the diagonal** — bit-identical
to what `assemble_matrix` already produces — plus off-diagonal blocks coupling
it to every other particle.

The fact worth building on:

> The interior Green's function of particle `p` is confined to particle `p`'s
> own volume. Inter-particle coupling therefore enters **only** through the
> exterior background Green's function, so the M3 and M4 cross-blocks are
> exactly zero and the M1/M2 cross-blocks are the ordinary exterior kernels
> evaluated between two **non-overlapping** boundaries.

Two consequences, both good:

1. **No new singular quadrature.** The cross-block integrand has no singularity
   — the two boundaries never touch — so the plain periodic trapezoid rule
   applies, and on an analytic boundary it is spectrally accurate, exactly like
   the Kress scheme it sits beside. No Kress splitting is needed off-diagonal.
2. **The diagonal is free.** Every self-block is the existing
   `assemble_matrix`, at that particle's own `nn` and its own material. A
   correct implementation reuses it verbatim, which means the whole v0.6
   quadrature validation carries over to the diagonal unchanged.

**The catch, and it is the main numerical risk of the whole feature:** point 1
holds *uniformly* only while the particles are well separated. As a gap closes,
the exterior kernel becomes near-singular across it, the integrand's
analyticity strip narrows, and the trapezoid rule's spectral rate collapses —
silently, in exactly the way the near-boundary degradation of `eval_field` does.
There is a minimum gap below which a given `nn` is not enough, and **nobody
knows what it is**. Measuring it is a gate, not an afterthought (§5, G3).

## 3. What the legacy code gives, and what it assumes

| Legacy function | What it does | Reusable? |
|---|---|---|
| `assemble_matrix_multi` (779) | Full coupled matrix; self-blocks from `assemble_matrix_fast`, cross-blocks from the exterior kernels | **Structure yes, code no.** The block layout and the cross-kernel algebra are right. The self-blocks are pre-Kress and the weights are the old `delt`. |
| `make_particle_array` (565) | `Np` identical particles at given centres | **Drop it.** `Geometry.gielis` already takes `x0`/`z0`, so a cluster is a list comprehension. A constructor that only makes *identical* particles is the assumption we are removing. |
| `make_dimer_chain` (628) | SSH chain, `axis="x"/"z"`, intra/inter spacing | **Drop from the core.** It is an application, not a solver feature, and it is the only part of the legacy multiparticle code with tests. |
| `plane_wave_rhs_multi` (855) | Stacked plane-wave RHS | Trivial; one line per particle over the existing `plane_wave_rhs`. |
| `far_field_multi` (884) | Coherent sum of per-particle far fields | Correct as physics: the far field is additive over boundaries. Reuses `_far_field_at` per particle. |
| `eval_field_multi` (918) | Exterior = sum over all boundaries; interior = that particle's boundary only | Correct as physics. Its implementation is an `O(M · Np)` Python double loop. |

### 3.1 The four assumptions to remove

Each is load-bearing in the legacy code and each must be named in the code-spec
as removed, because each one hides in the *indexing* rather than in a check:

1. **Equal `nn`.** `assemble_matrix_multi` raises if they differ, and the block
   indices `p·nn` and `Np·nn + p·nn` presume it everywhere. `plane_wave_rhs_multi`
   and `far_field_multi` go further: they read `nn = boundaries[0][0]` and then
   index with `nn_p`, which is a latent wrong-answer bug the moment `nn` varies.
2. **Identical shape.** Only the constructors impose this;
   `assemble_matrix_multi` itself never uses it. Heterogeneous shapes are
   **closer than the legacy API suggests** — this is the cheapest of the four.
3. **One material for all particles.** `assemble_matrix_multi(pol, boundaries,
   wn, ri, kd)` takes a single `ri` and `kd`; so does `eval_field_multi`. Per-
   particle materials mean the interior wavenumber is per-block, which the
   self-block assembly already supports — it is a plumbing change, not a physics
   one.
4. **One polarisation.** Correctly so — `pol` is a property of the *problem*,
   not of a particle (conventions §1). But `Material` currently bundles `pol`
   with the optical constants, so "a list of Materials" would carry `Np` copies
   of a quantity that must be identical. §4.2.

### 3.2 What is *not* there

**No solve-level test exists.** `sie-legacy-0726/tests/test_dimer_chain.py`
covers geometry construction only — ordering, axis swap, spacings, error paths.
Nothing in the legacy tree ever solves a coupled system and checks the answer
against anything. The two `Np = 1` and reciprocity checks of §5 do not exist and
have never been run. **Treat the legacy physics as unverified**: it is a
credible derivation with a clean argument behind it, not evidence.

## 4. Decisions a code-spec must take

Recommendations given, with the reasoning. None of these is settled here.

### 4.1 The API shape — the one that decides v1.0

**Recommendation: additive.** A new `Cluster` (or similarly named) object and
its own solver and result, alongside `BIESolver`/`ScatterResult`, which keep
their current signatures untouched.

- It is the only option under which multiparticle does **not** gate v1.0: if the
  existing surface provably cannot change, 1.0 can ship after v0.8 and the
  cluster arrives as 1.1.
- The single-particle path stays the simple thing. Most users have one particle,
  and making them construct a one-element cluster is a tax on the common case.
- The cost is real and must be stated in the spec: the result object duplicates
  `far_field`, `eval_field`, and the cross-section methods. Mitigation — they
  are all thin façades over primitives that are *already* per-boundary and
  additive, so the duplication is in the façade, not the physics.

The alternative — making `BIESolver` polymorphic over one-or-many — collapses
the duplication but breaks or muddies a published signature, and buys a
generalisation for a case that is not the common one.

### 4.2 Where `pol` lives

`Material` currently carries `pol` alongside `n_core`/`n_clad`/`epsi`. A cluster
needs `Np` materials and **one** polarisation. Three ways out, in order of
preference:

1. Polarisation is a parameter of the cluster solver; per-particle `Material`
   instances have their `pol` **validated equal** and otherwise ignored, with a
   clear error naming the offending particle. No break, slightly redundant.
2. Split `pol` out of `Material` into the solver. Cleanest, and arguably where
   it always belonged — but it breaks `Material`, which is a published
   constructor, and so it is a v1.0-or-never change.
3. Let it differ per particle. **Wrong** — it is not a per-particle quantity;
   conventions §1 says so.

### 4.3 The degree-of-freedom ordering

The legacy layout is `[φ_0 … φ_{Np−1}, χ_0 … χ_{Np−1}]`, chosen to extend the
single-particle `ei[:nn]`/`ei[nn:]` convention (conventions §4). With ragged
`nn_p` that layout needs two offset tables and makes no block contiguous.

**Recommendation: per-particle contiguous**, `[φ_0, χ_0, φ_1, χ_1, …]`, so each
particle's `2·nn_p` sub-vector is a contiguous slice that can be handed straight
to the existing single-particle primitives. Whichever is chosen, it is a
**convention** and belongs in `docs/conventions.md` in the same change — with
the explicit note that it does *not* extend the single-particle layout, so that
nobody later "fixes" it back.

### 4.4 Per-particle resolution

The point of ragged `nn` is that each particle gets the resolution *it* needs.
`wavelength_over_ds` already computes exactly that diagnostic per geometry. The
spec should decide whether `nn_p` is purely the user's business, or whether the
cluster warns when one particle is far coarser than its neighbours — the coupled
system is only as accurate as its worst block, and that is not locally visible.

### 4.5 Overlap and gap validation

Two particles that overlap make the formulation meaningless and the matrix
plausible. A decisive test for arbitrary shapes is hard; a **conservative** one
is easy and cheap — circumscribing circles disjoint ⇒ certainly fine;
inscribed circles intersecting ⇒ certainly overlapping; between the two, a
sampled boundary test. The spec must decide what happens in the undecidable
band: raise, or warn and continue. Given §2's near-gap risk, the same machinery
should also report the minimum gap, which is what G3 turns into a usable rule.

### 4.6 The observable

`efficiencies()` normalises by `2·rad`, which is already documented as only
approximate for a non-circular shape and is outright meaningless for a cluster —
whose `rad` is not defined at all. **Absolute cross-sections `C_sca`, `C_ext`,
`C_abs` in nm** are the right multiparticle observable, and the v0.8 comparison
contract has already fixed exactly that choice for exactly that reason
([v0.8-external-validation.md](v0.8-external-validation.md) §2). Adopt it;
do not re-derive it.

### 4.7 Cost, and what not to optimise

The system is dense, `N = 2·Σ_p nn_p`, and the solve is `O(N³)`. But assembly is
`Np²` cross-blocks of `nn_p × nn_q` Hankel evaluations, and **Hankel functions
are 99 % of this package's runtime** while dense linear algebra is ~2 %
([performance.md](performance.md)). So assembly, not the solve, dominates until
`Np` is large, and the first optimisation to reach for is the one already proven
here: `hankel1` releases the GIL, so threading the cross-block loop is a real
win where threading a Python loop normally is not.

The spec should **measure** the `Np` ceiling and document it rather than
engineer past it. Dense is the right first implementation; iterative solvers,
fast multipole, or block preconditioning are all speculative until there is a
measurement saying dense is not enough.

## 5. The validation ladder

Sequential; each rung is a stop. This is where the feature earns the right to
exist, and it is the part the legacy code has none of.

**G1 — `Np = 1` reduces exactly.** A one-element cluster reproduces
`BIESolver.scatter` to round-off, in both polarisations, on a circle and on a
star. Cheap, and it pins the block layout, the RHS stacking and the field
summation in one test. Anything that fails here fails everything below it,
unattributably.

**G2 — the far-separated limit.** As the gap grows, the cluster's total far
field approaches the coherent sum of the independent single-particle far fields
with their geometric phases. This is a *limit*, so the test is a measured
convergence rate toward it, not a tolerance at one gap. It validates the cross-
blocks in the regime where they are weakest — which is the only regime where an
independent answer is available for free.

**G3 — the near-gap envelope.** §2's open question, and the one measurement this
feature cannot ship without. Sweep the gap down at fixed `nn` and find where the
spectral rate breaks; repeat at several `nn` to get a rule of the form
"gap ≳ f(nn, λ)". The deliverable is a documented validity envelope with the
numbers behind it, in the style of the `eval_field` near-boundary rule — and,
unlike that rule, actually measured rather than inherited.

**G4 — the analytic two-cylinder anchor.** Two circular cylinders have a closed-
form multiple-scattering solution via the cylindrical addition theorem, built
from single-cylinder Mie coefficients. This is a genuine independent anchor —
not two paths in this repo agreeing (non-negotiable 3) — and it is the only one
available for the coupled problem without leaving the package.

> **This is where the v0.7 multipole work pays for itself a second time.**
> [multipole-spec.md](multipole-spec.md) delivers exactly the cylindrical-
> harmonic machinery the addition theorem consumes, already validated against
> Mie to 2.8e-15. G4 should be scoped *after* v0.7 lands and should reuse it
> rather than re-deriving harmonics. If G4 is attempted first, it will rebuild
> that module badly.

**G5 — reciprocity.** The far-field amplitude from direction `a` observed at `b`
equals that from `b` observed at `a`. It is a property of the operator, holds for
any number of particles of any shape, and is sensitive to precisely the kind of
block-transposition and index error that G1 on a symmetric configuration will
not catch. Cheap; run it on a deliberately asymmetric cluster.

**G6 — external.** Two and three particles against the v0.8 anchors.

## 6. How this absorbs v0.8

The external validation milestone was scoped for the *single-particle* non-
circular problem, but almost all of its apparatus is indifferent to particle
count, and the multiparticle work should inherit rather than rebuild it:

- **The comparison contract** (§2 of that doc) — absolute cross-sections,
  spectra compared pointwise rather than fitted, both polarisations, vacuum
  wavelengths in, lossless and lossy — transfers verbatim. §4.6 above is that
  contract already deciding a multiparticle API question.
- **The two anchors** handle multiple particles natively. A Yee grid does not
  care how many scatterers are in it; gmsh meshes two contours as readily as
  one. The per-tool polarisation-mapping gate and the convergence study are
  done once, in v0.8, and are not repeated.
- **The three-layer packaging** — frozen self-describing reference data,
  committed-but-not-installed regeneration scripts with an honest conda recipe,
  and a numpy-only repo test asserted against the frozen data — is the part
  worth most here, and it is designed once. A multiparticle comparison is a new
  data file in an existing format, not new infrastructure.
- **The validation guide** gains a multiparticle section rather than a second
  document.

The practical consequence for sequencing: **v0.8 before multiparticle**, not
because the physics depends on it, but because building the external-validation
machinery twice would be the expensive mistake, and G6 is the rung that makes
the whole ladder worth climbing.

## 7. What the code-spec should explicitly exclude

- **Periodic or infinite arrays.** The legacy tree has a `periodic.py` and an
  `ssh.py`; both are Bloch-mode machinery for a different question. A finite
  cluster is this feature.
- **Particles in contact, or overlapping.** Excluded by the formulation, not by
  an implementation limit. §4.5 decides how loudly.
- **A substrate or a layered background.** Separate roadmap item, and the one
  the legacy `slab_image.py` and `sommerfeld.py` address.
- **QNMs of a cluster.** They come nearly free — Beyn's contour method does not
  care about the internal structure of `M(λ)`, so a multiparticle assembly makes
  `QNMSolver` work on clusters with no new eigenvalue code. That is a real
  payoff and it should be *named* as future work, not attempted in the same
  milestone: the half-plane argument (conventions §8) and the degeneracy handling
  would both need re-examining for a cluster, and neither is a small question.
- **LDOS inside a cluster.** Likewise close to free — `line_dipole_rhs` stacks
  like the plane wave, and `relative_ldos` needs only the source to lie outside
  every particle. It is the observable most likely to be wanted first. A separate
  milestone, after the driven observables are validated.

## 8. Open questions this document does not answer

- **[?]** Does `Cluster` own the materials, or does the solver take two parallel
  lists? §4.1's recommendation is silent on it, and it determines whether a
  particle can be moved without rebuilding its material.
- **[?]** Is ragged `nn` chosen by the user, or derived from a target
  points-per-wavelength across the cluster? §4.4.
- **[?]** What is the `Np` ceiling of the dense solve on this machine, in both
  memory and time? Unmeasured. §4.7.
- **[?]** Does G3's envelope depend on the *shapes* of the facing boundaries, or
  only on the gap and `nn`? Two flat facing sides and two facing tips are
  plausibly different problems, and the answer changes whether the envelope is
  one number or a family.
