# v0.9 half-space background — code-spec

Status: **owner decisions of 2026-09-27 folded in** (§7). No package code yet.
The half-space ships as **v0.9**; the MEEP comparison for substrates joins the
v1.0 external-validation milestone. Multilayers follow after v1.0.

Inputs:
- [sommerfeld-holomorphy.md](sommerfeld-holomorphy.md) is the theory (step 1).
- [studies/halfspace/](studies/halfspace/README.md) is the measured prototype
  (step 2a).
- The legacy derivations are in `sie-legacy-0726/docs/sommerfeld_greens_function.tex`.
  They are sound for real k in the top-cladding cell, and **wrong for complex k**
  (holomorphy note §1).

## D. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | A `Background` enters as an optional keyword, `background=None`, on `BIESolver`, `QNMSolver`, `ClusterBIESolver` and `relative_ldos_map`. `None` is the homogeneous cladding and is bit-identical to v0.8. | Additive, with no signature change. One solver family serves both integrated-photonics and scattering users. |
| D2 | v0.9 ships `HalfSpace(eps_sub, z_int)` only. The multilayer later adds a `Multilayer` that supplies a different R(q) and nothing else. | The Sommerfeld functionals depend on the stack only through R(q). |
| D3 | Layer permittivities are **absolute complex ε**, not `Material`. Layers use `k = k₀·√ε` (principal complex root, signed zero normalised). `Im ε ≥ 0` is asserted on every layer. | `Material` builds Re ε from a real `n_core`, so it cannot express Re ε < 0; gain breaks the continuation argument (holomorphy §2). |
| D4 | Everything below the facade takes `wnum_bg` and the **background-relative** `eps_sub/n_clad²`, never a wavelength. `Background` exposes `eps_rel(n_clad)`. | Conventions §2.3: one conversion point. It also preserves scale covariance (§9), since the path is built in units of k. |
| D5 | The particle (and every source and observation point) lies strictly in the cover, `min(g) > z_int`. Fields in the substrate are out of scope. | That is the only Green-function cell the milestone needs. The others need transmission functionals. |
| D6 | The reflected blocks are assembled as a **banded separable GEMM** on a deformed path, with no surrogate and no tabulation. | Measured at 4–14 ms per block at nn ≤ 256, the same at real and complex k, and about 10× cheaper than the free-space assembly at complex k. |
| D7 | The path is **fixed per call context**: per wavelength for driven solves, and **once per search box for QNMs**, sized from the shadow box B̄. The clearance condition C1 is asserted, not inferred. | Beyn needs one holomorphic M(λ) on the box. A crossing shows up as a cut, which rank detection does not catch. |
| D8 | No TM `R_∞` image subtraction in the matrix path. | Measured net loss (study trap 5). |
| D9 | With a background, v0.9 ships the **upward far field**, the **power scattered into the cover** `σ_sca,up`, and the **absorption** `σ_abs`. Transmitted far field, extinction, multipoles and the shape sensitivity raise `NotImplementedError`. The analytic dM/dλ for `QNMResult.refine` ships. | Owner's scope. `σ_sca,up` is incomplete by construction, because it omits the power sent into the substrate, so it carries a checked upper bound (§2a). |
| D10 | LDOS stays normalised to the **unbounded cover**: `relative_ldos = 1 + 4·Im S`, where S now includes the substrate's reflected self-field. It therefore equals the substrate-only enhancement when no particle is present. | Keeps §7 unchanged. The alternative normalisation, to the bare interface, is a one-line ratio the user can take. |
| D11 | The owner excluded plane-wave illumination from the substrate side, including TIR, from this milestone. Plane waves from the cover (incident plus Fresnel-reflected) are in. | Owner's scope. Cover illumination needs only `r(q_inc)`. |

## 1. Module layout

New module `layered.py` (primitives, no wavelength anywhere):

    alpha(q, k)                          vertical-cut sheet, α(0)=+k       (holomorphy §3)
    fresnel_r(q, k1, eps_rel, pol)       TE (α1−α2)/(α1+α2); TM admittance form; PEC sentinel
    SommerfeldPath                       nodes q, weights w, band edges; built by
        .for_wavenumber(k, eps_rel, pol, D, z_min)     driven solves
        .for_box(k_box_corners, eps_rel, pol, D, z_min) QNM: sized on ∂B̄, asserts C1
    reflected_blocks(path, pol, k, eps_rel, f, g, df, dg, z_int, x_c)
        → (m1_ind, m2_ind), (nn_q, nn_p)          exterior rows; cross-particle ready
    reflected_blocks_dk(...)             d/dk of the above, for Newton refinement
    reflected_green(path, pol, k, eps_rel, x, z, xs, zs, z_int)   pointwise, for RHS/field/LDOS

`background.py`: `HalfSpace` (frozen dataclass: `eps_sub: complex`, `z_int: float`)
with `eps_rel(n_clad)`, validation (D3) and `r(q, k, pol)`. It is exported from
`__init__.py`.

The prototype's `hs.py` and `gemm.py` port almost unchanged. The node rule is the
study's rule; its constants (13·T/δ, 6/D, 38/Z_min, 24-point panels) move into
named module constants, each with a comment giving the measurement behind it.

## 2. Wiring

- **Matrix:** `BIESolver.assemble(λ)` returns M + [[m1_ind, m2_ind], [0, 0]]. The
  reflected term goes on the exterior rows only, with the same sign as
  `assemble_cross_block`. For a cluster, every (q, p) pair gets reflected blocks,
  including q = p, so D is the **cluster** extent.
- **Right-hand sides:**
  - `line_dipole_rhs` adds `G_ind(r_i, r_s)`.
  - `plane_wave_rhs` adds the reflected plane wave `r(q_inc)·e^{i q_inc x + i α₁ (z − 2 z_int)}`. Only downward incidence from the cover is accepted.
  - A custom `incident_rhs` must supply the full background incident field; its docstring says so.
- **Near field:** `eval_field` returns the scattered field with `G_free + G_ind` in
  the exterior representation. Total field = scattered + incident (+ reflected
  incident). Points at or below `z_int` raise.
- **LDOS:** S = scattered(particle) + G_ind(r_s, r_s), the latter from
  `reflected_green`. `relative_ldos_map` keeps its single LU factorisation.
- **QNM:** `QNMSolver(geometry, material, background=None)`. `modes(box)` builds
  one `SommerfeldPath.for_box` and reuses it for every contour point and every
  Newton step. `assemble_derivative` adds `reflected_blocks_dk` with the §2 chain
  factor.

## 2a. Far field with a substrate (D9)

- **Upward amplitude.** For an upward direction θ, the far field is the particle's
  direct radiation plus its specular image, obtained by stationary phase on the
  reflected spectrum:

      a_up(θ) = a_dir(θ) + r(k₁ sin θ) · a_dir(θ') · e^{−2 i k₁ cos θ (z_c − z_int)},

  where θ' is the mirror direction. Both terms reuse the existing `_far_field_at`
  about the particle centre. The exact phase reference is fixed in the
  implementing commit and pinned by G12.
- **σ_sca,up.** This is the integral of |a_up|² over the upper half circle, with
  the v0.8 normalisation. It excludes the specularly reflected incident beam,
  which is background, not scattering.
- **σ_abs.** This is computed from the boundary flux, `−Im ∮ φ*·(interior-side
  derivative)` in the §4 layout. It needs no far field, so it is exact with a
  substrate. In homogeneous background it must equal the v0.8 `qext − qsca`
  (G13).
- **The upper bound.** Energy conservation reads
  `P_up + P_down + P_guided + P_abs = P_in`, and the omitted terms are all ≥ 0.
  The check therefore uses a source whose input power is known exactly: a line
  dipole, whose total emitted power is `P_in ∝ relative_ldos` (D10). The assertion is

      P_up + P_abs ≤ P_in,

  with **equality over a PEC substrate**, where P_down = 0.

  For a plane wave the input power needs extinction, which is out of scope, so the
  plane-wave bound is checked only in the PEC case, where `P_up` must equal the
  mirror-cluster total σ_sca. The substrate-only case (no particle, dipole over a
  lossless dielectric) has a closed form, `P_up/P_in` from the Fresnel integral
  over the propagating q, which anchors the normalisation independently.

## 3. Assertions and warnings

| Condition | Action | Source |
|---|---|---|
| `Im ε_sub < 0` | raise | holomorphy §2 |
| any node, source or observation point with `z ≤ z_int` | raise | D5 |
| clearance C1 fails on ∂B̄ (branch points ±k₁, ±k₂; TM plasmon q_sp classified with the code's own α) | raise, naming the singularity and the offending corner | holomorphy §3–5 |
| `λi_max/λr_min > tan 45°` | raise | holomorphy §8 |
| `δ·D > ln(10)·3` after the depth floor | warn: expected digits lost | holomorphy §7, study "open" |
| `spacing/(2·gap) > 0.25` (TE) or `> 0.2` (TM) | warn, like `ClusterGapWarning` | study trap 4 |
| x not centred | never exposed: `reflected_blocks` always centres on the particle or cluster centroid | study trap 3 |

## 4. Validation gates (each a test; tolerances quote the study's measured floors)

| Gate | Check | Independent because |
|---|---|---|
| G1 | Pointwise PEC closed form ∓(i/4)H₀^{(1)}(k₁ρ_img) and its derivatives, at real and complex k | closed form |
| G2 | Pointwise against the real-axis QUADPACK reference (legacy substitution, separate branch rule) for glass, lossy and Ag/Au, both polarisations, at real k | a second quadrature on a different path with a different branch implementation |
| G3 | Complex k against Chebyshev continuation of the G2 reference, at Q = 10 | uses no complex path at all |
| G4 | The legacy regression: the real-axis `Im α ≥ 0` integral at `Im k < 0` equals `−(i/4)H₀^{(2)}` | pins the failure mode so it cannot come back |
| G5 | The plasmon pole is a zero of the TM denominator on the sheet, and `for_box` raises when a box forces a crossing | closed form, plus a guard test that must fail |
| G6 | End-to-end: circle over PEC equals `ClusterBIESolver` particle + mirror, for φ, χ, near field and LDOS; TE/TM; real/complex λ; n_clad ∈ {1, 1.33} | v0.8 reference; pins the sign wiring |
| G7 | `ε_sub = n_clad²` gives v0.8 bit-identity (G_ind ≡ 0 path short-circuited) and `background=None` gives v0.8 bit-identity | regression |
| G8 | QNM over PEC: poles of the half-space solver equal the poles of the mirror-cluster matrix (Beyn on both). The gap → ∞ limit recovers the Mie roots. | v0.8 reference plus analytic Mie |
| G9 | QNM over a dielectric or Ag substrate: continuation from a large gap (Mie-labelled) to a small gap is smooth (conventions §8), and the poles are unchanged when `for_box` depth is varied | consistency only. **The independent anchor for dielectric-substrate QNMs is external (MEEP) and is listed as open** |
| G10 | Scale covariance: M(s·rad, s·λ) = M(rad, λ) with a background, and z_int scaled too | conventions §9 |
| G12 | Upward far field over PEC equals the mirror-cluster far field on the upper half circle | v0.8 reference |
| G13 | `σ_abs` from boundary flux equals `qext − qsca` (v0.8) with no background and equals Mie `Q_abs` on a lossy circle | analytic Mie |
| G14 | Dipole bound: `P_up + P_abs ≤ P_in`, with equality to round-off over PEC; with no particle, `P_up/P_in` equals the closed-form Fresnel integral over a lossless dielectric | energy conservation + closed form |
| G11 | Timing: reflected-block assembly ≤ the free-space assembly at complex k, nn = 256 | performance claim D6 |

## 5. Conventions entries (new §15, written in the implementing commit)

The substrate is below, the cover above, and the interface is horizontal at `z = z_int`.
Also recorded there:
- the absolute complex ε with the principal root and signed-zero normalisation;
- `Im ε ≥ 0` asserted;
- the vertical-cut α sheet;
- R sign per polarisation (TE −1 / TM +1 in the PEC limit);
- the shadow box;
- the fixed path per box;
- the LDOS normalisation (D10);
- the 2-D caveat: these are line-source fields, so absolute LDOS near the
  interface is not a 3-D number, while Q, detuning and ratios carry over;
- non-dispersive ε only, with interpolated tabulated data forbidden in QNM work
  (holomorphy C0).

## 6. Order of work (one commit each)

1. `layered.py` primitives plus `HalfSpace`, with G1–G5.
2. Driven wiring (matrix, RHS, near field, LDOS) plus G6, G7 and G10 for driven
   quantities.
3. Cluster wiring (in scope per the owner; G6 on a two-particle cluster over PEC,
   which becomes a four-cylinder mirror reference).
3a. Far field, `σ_sca,up`, `σ_abs` (§2a) and G12–G14.
4. QNM: `for_box`, `reflected_blocks_dk`, `assemble_derivative`, G8, G9.
5. G11 plus the `performance.md` update, and conventions §15 plus a README section.

## 7. Owner decisions (2026-09-27)

1. The MEEP comparison for dielectric-substrate QNMs (G9's missing independent
   anchor) belongs to v1.0 external validation. The half-space ships as v0.9.
2. Several particles on the substrate are in scope (step 3).
3. The reflected (upward) far field, `σ_sca,up` with its checked upper bound, and
   `σ_abs` are in scope. Transmission and extinction are not.
4. LDOS stays normalised to the unbounded cover (D10).
5. `HalfSpace.pec(z_int)` is public API: R ≡ ∓1 evaluated as the exact image
   Hankel term, with no Sommerfeld integral.
6. Dispersion: `eps_sub` may be a callable `ε(λ_vac)` for **driven** solves
   (real λ, evaluated once per call at the facade). `QNMSolver` rejects a
   callable, because tabulated data is not holomorphic (holomorphy C0).
   Analytic Drude/Lorentz models are deferred.
7. The `Material.nc` gain fix ships separately to main, before v0.9. **Shipped in
   v0.8.2** (PR #24): `nc` is now the principal root of the complex ε.
8. `Material.from_eps(eps, n_clad, pol)`: additive classmethod taking an absolute
   complex ε (Re ε < 0 allowed, `nc = √ε` principal root), anchored on Mie for a
   metal cylinder. It was needed for the tungsten heater below. With that
   figure parked, it is optional for v0.9 (metal particles over a substrate).
9. **Parked 2026-10-06; not a v0.9 deliverable.** The v0.9 README section
   needs a figure, still to be chosen. The heater design below is kept for a
   later release.

   The parked design (LDOS only) is a heater cross-section at 1550 nm:
   - Si half-space substrate under an SiO₂ cover (n = 1.444);
   - a rounded-square (Gielis) SiN core 200 nm above the substrate;
   - a horizontally elongated tungsten heater 1 µm above the SiN;
   - relative-LDOS maps in two panels, TE and TM.

   Particle sizes are still to be fixed with the owner. The figure also needs
   a 2-D caveat: these are line-source LDOS values, not the guided mode along
   the heater axis.
