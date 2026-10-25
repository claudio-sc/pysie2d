# Waveguide-coupled clusters — roadmap (v1.2 → v1.4)

**Status: roadmap, 2026-10-23; owner decisions R1–R10 complete. v1.1 shipped
2026-10-24 as v1.1.0, so v1.2 is unblocked.** This is
the planning document above the code-specs. It fixes the goal, the physics shape, the milestone split, the
validation anchors and the owner decisions (§1, R1–R10). Each milestone gets its own
builder-ready code-spec later, in the form of
[layered-spec.md](layered-spec.md) §8.

## 1. Goal

S-parameters of **arbitrary clusters of scatterers evanescently coupled to a
slab waveguide**, computed by the boundary-integral solver. Then a 2.5D
front end, so that a planar photonic-chip layout can be simulated the way
Lumerical's varFDTD does: in the chip plane, with effective indices.

Owner decisions taken 2026-10-23:

| # | Decision |
|---|---|
| R1 | **Scatterers stay outside the core.** No particle intersects or sits inside a layer of the stack. Etched gratings, holes in the core and photonic-crystal waveguides are out of scope. |
| R2 | **S-parameters first.** QNMs with a slab background come later and are low priority: isolated-cluster QNMs (shipped) guide the design, and the slab-coupled response is read from the driven S-parameters. |
| R3 | **v1.1 (transmitted fields) stays first** (done: v1.1.0). Its generic τ = T/α₁ and its `P_down` by interface flux are prerequisites. The G14 → GT7 power equality is also the slab's best anchor. |
| R4 | **Particles on both sides of the stack**, staged as v1.2 item 6. In the 2.5D top view the strip has two claddings, and side-coupled structures commonly sit on both. |
| R5 | **General N-layer `Stack` in v1.2**, not a three-layer slab first. The gates run on the three-layer case first. |
| R6 | **v1.3 is a MEEP milestone**: eigenmode S-parameters, plus one MEEP LDOS run that closes a v1.0 gap. |
| R7 | **The 2.5D front end lives in the library**, as a thin layer on the v1.2 mode solver. |
| R8 | **Out-of-plane loss in 2.5D is a documented limitation**, not estimated. An estimate would be new physics with no anchor. |
| R9 | **The S-parameter API shape** (layout over mode and direction, phase reference plane) **is settled in the v1.2 code-spec**, with the options laid out there. |
| R10 | **The 2.5D method is variational EIM** (owner, 2026-10-23), with plain EIM as an option for regions whose stack guides a mode. See v1.4 for the reasons. |

## 2. Why this is one reframing, not a new solver

In varFDTD the vertical direction is folded into an effective index, so the 2D
computational plane is the **top view of the chip**. A straight waveguide in
that plane is a strip of `n_eff,core` between claddings of `n_eff,clad`, which
is a **three-layer background**: the same object as a slab waveguide seen in
cross-section. Only the coordinate labels change: layers are stacked in `z`,
and guided modes propagate along `x`.

Every piece above the background already exists:

| needed | status |
|---|---|
| arbitrary smooth shapes, spectral quadrature | v0.6 |
| clusters of different shapes, materials, `nn` | v0.8 |
| metals (`Material.from_eps`) | v0.9 |
| reflected Green function on a deformed Sommerfeld path, complex λ | v0.9 |
| field below the stack, downward far field, `P_down` by flux, `incident_field` | v1.1 |
| factorise once, solve many | v0.3 |

What is missing is the background itself, its guided modes, a guided-mode
source and guided-mode outputs.

### Why a stack only adds poles

With N layers the reflection coefficient is an Airy/recursion expression in the
layer wavenumbers α_j = √(k_j² − q²). It is **even in every interior α_j**, so
interior layers create no branch points; the only branch points are those of
the two outer half-spaces, ±k_cover and ±k_bottom, which v0.9 already handles.
The interior layers contribute only **poles**, the guided modes at q = ±β_m
with `max(k_cover, k_bottom) < β_m < max_j k_j`. At real λ they lie on the real
q axis. This is the reason the multilayer generalisation is "only R(q)"
([sommerfeld-holomorphy.md](sommerfeld-holomorphy.md) §5).

## 3. The physics of the observable

The quantities are written down here so that the code-spec starts from
statements to verify, not from words. Every formula below is **to be derived
and verified numerically in the v1.2 spec** before it is pinned.

- **Guided modes.** For the stack at real λ: propagation constants β_m (roots
  of the dispersion relation, bracketed on `(max k_outer, max k_j)`) and
  closed-form profiles u_m(z), TE and TM, with power normalisation (TM carries
  the `1/ε(z)` weight, as v1.1's `a_down` does).
- **Mode source.** Excitation by a guided mode, `u_inc = u_m(z) e^{±iβ_m x}`.
  It is bounded on every particle boundary, so the RHS is a closed form, just
  as for the plane wave. `incident_field` (v1.1 D7, D13) gains this excitation.
- **Mode outputs.** As `x → ±∞` the scattered field splits into guided terms
  `Σ_m c_m^± u_m(z) e^{±iβ_m x}` plus radiation and lateral waves that decay.
  `c_m^±` is the **residue of the Sommerfeld integral at q = ±β_m**, applied to
  the boundary densities. By reciprocity it equals an overlap of the
  backward-travelling mode field with φ, χ on the boundaries: one boundary
  integral per mode and direction, with no far-field limit taken numerically.
- **S-parameters.** Power-normalised, with a phase reference plane `x_ref`
  (a convention to pin): `S_{n←m}` = `c_n^+` (plus δ_{nm} for transmission) and
  `c_n^-` for reflection. Radiation loss up and down, and absorption, complete
  the account.
- **The balance, an equality.**
  `Σ_n |S_{n←m}|² (both directions) + P_up + P_down + P_abs = P_in`. The pieces
  come from different integrals: residues, the upward far field, the downward
  interface flux and the boundary absorption flux. v1.1's finding 2 (the
  surface plasmon leak over a lossless metal) is exactly the half-space case
  of the missing `P_guided` term, and this closes it.

**Polarisation in 2.5D** (a convention to pin before code). A slab TE₀ mode
(E in the chip plane, the dominant SOI mode) has H normal to the chip, so its
2D top-view problem is pysie2d **`pol = 1`** (labelled TM here). A slab TM₀
mode maps to `pol = 2`. The names clash, so conventions.md records this in the
v1.4 commit.

## 4. Milestones

### v1.1 — transmitted fields (shipped, v1.1.0)

As [transmitted-spec.md](transmitted-spec.md), shipped 2026-10-24. Its
`Transmission` takes a generic τ, the bottom-layer wavenumber and a feature
list (D3), so the v1.2 `Stack` plugs in its own. The dipole power balance is an
equality over every passive substrate except the one D12 excludes: TM over a
lossless metal, where the missing term is v1.2's `P_guided`.

### v1.2 — slab-waveguide background and S-parameters

Branch `v1.2-slab-waveguide`. Additive.

1. **`Stack` background.** N layers between two half-spaces, complex ε per
   layer, absolute, principal root, as v0.9's layered-spec D3. R(q) and
   τ(q) by a numerically stable recursion. At complex q on the hump, never form
   `e^{+|Im α| d}`. `HalfSpace` becomes the N = 0 case, bit-identical to v0.9
   (G7-style).
2. **Guided-pole-aware path.** `SommerfeldPath` takes a `features` list
   (deferred from v1.1 §1). Size the hump so it clears every guided pole
   (holomorphy §5: a fixed hump is enough; assert `dD/dq ≠ 0` at the roots).
   The path must pass on the radiation-condition side of each pole: below
   +β, above −β.
3. **Mode solver.** β_m and u_m(z) for the stack at real λ, TE and TM, all
   guided orders. The v1.4 effective-index front end reuses it, and so would
   slab QNMs (complex root-finding, later).
4. **Mode source and outputs.** `scatter_mode(...)` and the mode amplitudes by
   residue/overlap, on single particles and clusters. **One factorisation
   serves every incident mode and direction**, so a full S-matrix at one λ costs
   one LU plus 2·N_modes solves. A wavelength sweep has nothing to reuse; the
   win there is threading (performance §3.1).
5. **Power account.** `P_up` (v0.9, generalised to the stack's R), `P_down`
   (v1.1, the stack's τ), `P_abs`, plus the guided powers.
6. **Particles on both sides of the stack** (R4). Coupling across the stack is
   v1.1's transmitted G, generalised to the stack's τ; bottom-to-bottom needs
   the stack's reflection from below (the reversed stack's R). Staged last, so
   items 1–5 are gated on cover-only particles first.

### v1.3 — external validation of waveguide coupling (MEEP)

The S-parameters of a **non-circular cluster** beside a slab, against MEEP 2D
with an eigenmode source and `get_eigenmode_coefficients`. This is a regime
MEEP does well: real λ, broadband, lossless, low-to-moderate Q. It reuses
v1.0's frozen-data contract (§6 there): conda-first drivers in `validation/`,
committed JSON with provenance, a numpy-only test in CI. It is a separate
milestone, under the rule that a finished additive feature is not held behind a
conda-first tool.

It also carries **one MEEP LDOS run** (R6) on a non-circular particle, the
v1.0 §4 milestone that never ran.

One study closes three gaps from §6 at once: a non-circular **cluster**, a
**layered background** with driven observables, and the **new mode channel**.

### v1.4 — 2.5D effective-index front end

- `effective_index(stack, wavelength, pol, order)` from the v1.2 mode solver,
  and a helper that turns a chip layout (vertical stack per region) into the 2D
  problem: background `Stack` of n_eff values plus particles of n_eff.
- In the library (R7). **The method is variational EIM (R10)**, the
  approximation varFDTD is built on. One reference vertical profile Ψ_ref(z)
  (the slab mode of a reference stack at a reference wavelength) is assumed
  everywhere, and each region's ε₂D is the overlap of its real stack with
  |Ψ_ref|², weighted by 1/ε in the TM-like case. The exact form is derived and
  verified in the v1.4 spec. Reasons:
  - **Plain EIM is undefined exactly where R1's scatterers sit.** A thin metal
    antenna, a short pillar or any region with no core under it guides no slab
    mode, so it has no local n_eff. The overlap is always defined.
  - It is the same approximation as the stated comparison target.
  - At complex λ only Ψ_ref needs continuing; the overlaps are plain integrals.

  Plain EIM (a local slab mode per region, ε₂D = n_eff²) stays available as an
  option for regions whose stack guides a mode. The reference stack and the
  reference wavelength are explicit, required inputs, and each frozen comparison
  records them.
- **What it cannot do, stated in the docstrings and README:** out-of-plane
  radiation (R8: documented, not estimated). A 2.5D model loses power scattered into the substrate and cover
  in 3D, and for a scatterer beside a waveguide that loss is often the quantity
  of interest. It also cannot see vertical-mode conversion.
- **Validation.** The 2D solve is anchored by v1.2/v1.3. The **approximation**
  itself has no closed form, and only a 3D solver can measure it. The deliverable
  is a characterised error on one canonical case (e.g. an SOI strip with a
  side-coupled pillar) against MEEP 3D, frozen like v1.0, with the error
  reported as an error of the model, never as a tolerance of the code.
- **Conventions it breaks, by construction:** n_eff(λ) is dispersive even for
  non-dispersive materials, so scale covariance (§9) does not hold for the 2.5D
  problem as a function of λ. It still holds for the 2D solve at fixed indices.

### Later, unscheduled

- **Slab QNMs** (R2, low priority). Needs n_eff or β_m continued to complex λ
  by complex root-finding, the holomorphy argument of §5 re-asserted with poles
  on the path's side, and an external anchor (the v1.0 MEEP harminv driver over
  a slab). [qnm-guide.md](../qnm-guide.md) §6 states the current limit: one
  half-space, single particles, non-dispersive ε.
- **QNM-to-waveguide coupling rates.** The isolated-cluster QNM projected on a
  guided mode gives the coupled-mode-theory decay rate into the waveguide, which
  is the bridge R2 implies. It is new physics; its anchor would be the v1.2
  S-parameters near resonance.
- **Multilayer backgrounds beyond the slab** fall out of v1.2's `Stack` for
  free. Only the gates are extra.

## 5. Validation anchors for v1.2 (non-negotiable 3)

Agreement between two pysie2d paths does not count. The anchors that do:

| # | Anchor | Independent because |
|---|---|---|
| A1 | `Stack` with all layers equal to the cover → v0.8 bit-identity; with one layer → v0.9 `HalfSpace` bit-identity | regression |
| A2 | R(q) poles = β_m of the closed-form slab dispersion relation; profiles u_m closed form | closed form |
| A3 | Reflected G against the real-axis QUADPACK reference with explicit pole indentation (extends v0.9 G2 and v1.1 GT2) | second quadrature, different path, residues added by hand |
| A4 | **Line-dipole emission into mode m**: the residue amplitude equals the closed form `∝ u_m(z₀)/P_m`, both polarisations | closed form; pins residue, normalisation and phase reference |
| A5 | **Small-cylinder limit**: as rad → 0 at a fixed gap, `S` of a circle → the Born closed form from the analytic Mie polarisability and the mode field, at the predicted rate (rad² in 2D, with dressing corrections falling with the gap) | analytic Mie plus analytic mode; the scaling makes an accidental pass impossible |
| A6 | Power balance as an **equality** (lossless, lossy, metal), every term from a different integral | energy conservation |
| A7 | **Reciprocity** on an asymmetric cluster: `S_{n←m}` from the left equals `S_{m←n}` from the right | Lorentz reciprocity; an asymmetric shape cannot satisfy it by symmetry |
| A8 | Scale covariance with every layer thickness scaled | conventions §9 |
| A9 | v1.3: MEEP eigenmode coefficients on a non-circular cluster | external solver |

A4 and A5 are the closed-form anchors. A6 and A7 are physical laws that a
wiring error breaks. A9 is the external one.

## 6. External validation still missing (state after v1.0.1)

v1.1.0 is the latest release (tag and PyPI, 2026-10-24); v1.0.1 shipped 2026-10-14. What v1.0 anchored
externally: cross-sections of one Gielis star and the circle (dolfinx and
MEEP), and dielectric-substrate QNMs of a circle (MEEP, high-Q, glass). What
has **no external anchor** yet, from [validation.md](../validation.md) §5 and
the specs:

| gap | today's anchor | closed by |
|---|---|---|
| Non-circular **clusters** | two-cylinder addition theorem (circles only) | v1.3 |
| Driven observables **over a substrate** (`c_sca_up`, `c_sca_down`, `c_abs`, near field, LDOS) | PEC mirror, a second quadrature, the v1.1 power-balance equality | partly v1.3 (driven, layered) |
| **Near field, LDOS, Purcell** on a non-circular shape | analytic circle only; v1.0 §4 planned MEEP LDOS and never ran it | v1.3 (one MEEP LDOS run, R6) |
| **Absorption** cross-check between two externals | dolfinx alone (MEEP has no frequency-independent Im ε) | open |
| **Metals** (`from_eps`, silver/gold particles and substrates) | Mie on a silver circle; metal-substrate QNMs are consistency only (G9) | open |
| QNMs and `dλ/dp` on a **non-circular** shape | circle only | open, low priority |
| Only **one** non-circular shape | — | v1.3 adds a second |
| MEEP TE sheet-source systematic | attributed, bounded, not removed | accepted |

The two stale documents noted here (validation.md §5 on `Re ε < 0`, and the
qnm-guide limitation in §4) were fixed on 2026-10-24.

## 7. Risks

- **Near-cutoff modes.** As β_m → k_outer the pole approaches the branch point
  and the hump clearance shrinks. The path cost then grows, and the residue and
  branch-cut contributions are hard to separate. This needs a measured floor in
  the v1.2 study, before any gate.
- **Small gaps.** The evanescent coupling is strongest at small gaps, which is
  where the Sommerfeld integrand decays slowest. v0.9 handles 5 nm over silver;
  the cost should be measured with a guided pole present.
- **Thick or high-index layers.** Exponential growth in the recursion at complex
  q. A stable form is required, with a test that fails on the naive one.
- **Mode counting.** A thick core has many guided orders. The S-matrix size and
  the residue sum must find *all* of them. A missed mode shows up only as a
  power-balance shortfall, so A6 is also the completeness check.
- **2.5D overclaiming.** The error of the EIM approximation is a property of
  the model, not of the code. The documentation must keep the two apart.

## 8. Still open

The owner decisions are R1–R10 (§1). What is left is measured, not chosen:

- **Negative ε₂D.** For strong local contrast the variational overlap can fall
  below the cladding value or go negative. `Material.from_eps` can express that.
  But the slab TE₀ case maps to `pol = 1`, where a lossless negative-ε region
  supports surface-plasmon-like modes (compare v1.1's finding 2), and those may
  be artefacts of the 2D model rather than physics. The v1.4 study checks this
  against MEEP 3D before any such result is trusted, and the spec states what
  happens when a region comes out negative.

## 9. Next step

v1.1 has shipped, so v1.2 starts, in this order:

1. **v1.2 study plan** in `docs/design/studies/slab/`: the `Stack` R(q)
   prototype with a stable recursion (and a test that fails on the naive
   form), and the guided-pole path. *S1–S3 done (2026-10-25).* The recursion is
   bitwise v0.9 at N = 0. The naive form fails by overflow, not by digit loss,
   at SOI with a 5 nm gap. The v0.9 path needs only a feature list, and it
   matches a pole-indented real-axis reference to the reference's floor
   (≤ 4e-10). *S4 done:* near cutoff is benign for G down to
   (β − k_out)/k = 1e-8. But a q-grid finder misses such modes, and no grid
   resolves a near-degenerate supermode pair, so the mode solver brackets in γ and
   checks against an argument-principle count. S5–S10 are planned there.
   *S5 done:* gaps down to 2 nm cost v0.9's M plus 1–3 %.
2. **Pin the §3 formulas**: mode normalisation, the identity between residue
   and overlap, and the A4 dipole closed form, each verified numerically.
   *S8, S9 done (2026-10-25):* Res_β R = γ₁u_m(z_top)²/(β_m N_m) with the
   bilinear N_m = ∫p u_m² (to 4e-14), and the dipole power balance
   Im G = P_up + P_down + Σ u_m(z₀)²/(2β_m N_m) as an equality (to 4e-15). The
   wiring of the overlap form is left to the code-spec gates (A6, A7).
3. **Measure the floors** for A2–A5 and the §7 risks (near-cutoff, small gaps).
4. **v1.2 code-spec** with the R9 S-parameter API and a §8 builder contract and
   stop rule.
5. **Build v1.2**: items 1–5 on cover-only particles, item 6 last.
