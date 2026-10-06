# v0.9 half-space background — code-spec

Status: **ready for implementation (2026-10-06).** Owner decisions are in §7, and
the builder contract that closes every open interface is §8. No package code yet.
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
| D1 | A `Background` enters as an optional keyword, `background=None`, on the constructors of `BIESolver`, `QNMSolver` and `ClusterBIESolver`. `self_green`, `relative_ldos` and `relative_ldos_map` keep their signatures and read `solver.background`. `None` is the homogeneous cladding and is bit-identical to v0.8. | Additive, with no signature change. One solver family serves both integrated-photonics and scattering users. |
| D2 | v0.9 ships `HalfSpace(eps_sub, z_int)` only. The multilayer later adds a `Multilayer` that supplies a different R(q) and nothing else. | The Sommerfeld functionals depend on the stack only through R(q). |
| D3 | Layer permittivities are **absolute complex ε**, not `Material`. Layers use `k = k₀·√ε` (principal complex root, signed zero normalised). `Im ε ≥ 0` is asserted on every layer. | `Material` builds Re ε from a real `n_core`, so it cannot express Re ε < 0; gain breaks the continuation argument (holomorphy §2). |
| D4 | Everything below the facade takes `wnum_bg` and the **background-relative** `eps_sub/n_clad²`, never a wavelength. `Background` exposes `eps_rel(n_clad)`. | Conventions §2.3: one conversion point. It also preserves scale covariance (§9), since the path is built in units of k. |
| D5 | The particle (and every source and observation point) lies strictly in the cover, `min(g) > z_int`. Fields in the substrate are out of scope. | That is the only Green-function cell the milestone needs. The others need transmission functionals. |
| D6 | The reflected blocks are assembled as a **banded separable GEMM** on a deformed path, with no surrogate and no tabulation. | Measured at 4–14 ms per block at nn ≤ 256, the same at real and complex k, and about 10× cheaper than the free-space assembly at complex k. |
| D7 | The path is **fixed per call context**: per wavelength for driven solves, and **once per search box for QNMs**, sized from the shadow box B̄. The clearance condition C1 is asserted, not inferred. | Beyn needs one holomorphic M(λ) on the box. A crossing shows up as a cut, which rank detection does not catch. |
| D8 | No TM `R_∞` image subtraction in the matrix path. | Measured net loss (study trap 5). |
| D9 | With a background, v0.9 ships the **upward far field**, the **power scattered into the cover** `σ_sca,up`, and the **absorption** `σ_abs`. Transmitted far field, extinction, multipoles and the shape sensitivity are not provided; §8.4 lists exactly which methods raise `NotImplementedError` and which change their returned keys. The analytic dM/dλ for `QNMResult.refine` ships. | Owner's scope. `σ_sca,up` is incomplete by construction, because it omits the power sent into the substrate, so it carries a checked upper bound (§2a). |
| D10 | LDOS stays normalised to the **unbounded cover**: `relative_ldos = 1 + 4·Im S`, where S now includes the substrate's reflected self-field. It therefore equals the substrate-only enhancement when no particle is present. | Keeps §7 unchanged. The alternative normalisation, to the bare interface, is a one-line ratio the user can take. |
| D11 | The owner excluded plane-wave illumination from the substrate side, including TIR, from this milestone. Plane waves from the cover (incident plus Fresnel-reflected) are in. | Owner's scope. Cover illumination needs only `r(q_inc)`. |

## 1. Module layout

New module `layered.py` (primitives, no wavelength anywhere). The exact
signatures of `reflected_blocks`, `reflected_blocks_dk` and `reflected_green` are
in §8.2; the lines below are the overview:

    alpha(q, k)                          vertical-cut sheet, α(0)=+k       (holomorphy §3)
    fresnel_r(q, k1, eps_rel, pol)       TE (α1−α2)/(α1+α2); TM admittance form; PEC sentinel
    SommerfeldPath                       nodes q, weights w, band edges; built by
        .for_wavenumber(k, eps_rel, pol, D, z_min)     driven solves
        .for_box(k_box_corners, eps_rel, pol, D, z_min) QNM: sized on ∂B̄, asserts C1
    reflected_blocks(path, pol, k, eps_rel, f, g, df, dg, z_int, x_c)
        → (m1_ind, m2_ind), (nn_q, nn_p)          exterior rows; cross-particle ready
    reflected_blocks_dk(...)             d/dk of the above, for Newton refinement
    reflected_green(path, pol, k, eps_rel, x, z, xs, zs, z_int)   pointwise, for RHS/field/LDOS

`background.py`: `HalfSpace`, exactly as in §8.1. It is exported from `__init__.py`.

The prototype's `hs.py` and `gemm.py` port almost unchanged. The node rule is the
study's rule; its constants (13·T/δ, 6/D, 38/Z_min, 24-point panels) move into
named module constants, each with a comment giving the measurement behind it.

## 2. Wiring

- **Matrix:** `BIESolver.assemble(λ)` returns M + [[m1_ind, m2_ind], [0, 0]]. The
  reflected term goes on the exterior rows only, with the same sign as
  `assemble_cross_block`. For a cluster, every (q, p) pair gets reflected blocks,
  including q = p, so D is the **cluster** extent.
- **Right-hand sides** (φ half only, like v0.8; χ half stays zero):
  - `line_dipole_rhs` adds `G_ind(r_i, r_s)` from `reflected_green`.
  - The plane wave adds the reflected wave `r(q)·e^{i q f + i α₁ (g − 2 z_int)}`, with `q = k sin θ`, `α₁ = k cos θ` and θ = `angle` in radians. This matches `plane_wave_rhs`'s incident `e^{i k (f sin θ − g cos θ)}`, which travels downward. The facade raises `ValueError` unless |angle| < 90°. Verified against the mirror cluster to 6e-14.
  - A custom `incident_rhs` must supply the full background incident field; its docstring says so.
- **Near field:** at exterior points, `eval_field` returns the scattered field, using
  `G_free + G_ind` in the representation; the reflected-image term is
  `−Σ delt·((∂x′G_ind·dg − ∂z′G_ind·df)·φ + G_ind·χ)`, with the same sign as the
  free term. Interior points are unchanged from v0.8.1, because the interior
  representation uses the core Green function only. The total exterior field is
  scattered + incident + reflected incident; the user adds the last two.
  Points at or below `z_int` raise `ValueError`.
- **LDOS:** S = scattered(particle) + G_ind(r_s, r_s). `self_green` and
  `relative_ldos_map` add the second term from `reflected_green` at each source. `relative_ldos_map` keeps its single LU factorisation.
- **QNM:** `QNMSolver(geometry, material, background=None)`. `modes(box)` builds
  one `SommerfeldPath.for_box` and reuses it for every contour point and every
  Newton step. `assemble_derivative` adds `reflected_blocks_dk` with the §2 chain
  factor.

## 2a. Far field with a substrate (D9)

- **Upward amplitude** (verified: 1.6e-15 against the mirror cluster, TE and TM,
  `z_int = 30`, 25° incidence). θ is measured from +z, as in `far_field`, and is
  upward for |θ| < π/2. With `ffa = fields._far_field_at` about the **origin**
  (no centre shift):

      a_up(θ) = ffa(θ; f, g, df, dg, ei) + r(k sin θ) · ffa(π − θ; f, g − 2·z_int, df, dg, ei)

  Here r is the real-axis `fresnel_r` at q = k sin θ, and `g − 2·z_int` shifts
  only the z coordinates, not `dg`. The image term needs no extra phase factor.
- **σ_sca,up** = (1/(8π k)) ∫_{−π/2}^{π/2} |a_up|² dθ, by Gauss–Legendre on
  `n_angles` nodes (default 500; measured: round-off already at 64 nodes for kR = 10). It
  excludes the specularly reflected incident beam, which is background, not
  scattering.
- **σ_abs** = −(1/k) · Im Σ_j delt · conj(φ_j) · χ_j, where φ = `ei[:nn]` and
  χ = `ei[nn:]` are exterior-side values (conventions §4). There is no ε factor, because
  the exterior medium is the background. It needs no far field, so it is exact with a
  substrate. Verified against the v0.8 `c_ext − c_sca`: it agrees to 12 digits on a lossy circle,
  and to 1e-11 / 1e-13 (TM / TE) on the README star at nn = 512 with n_clad = 1.33.
- **Dipole power balance** (verified to 12 digits over PEC, TE and TM). Unit line
  dipole at r_s, powers in units where the unbounded-cover dipole emits 1/4:

      P_in  = relative_ldos(r_s) / 4
      P_abs = −Im Σ_j delt · conj(φ_j) · χ_j
      P_up  = (1/(8π)) ∫_{−π/2}^{π/2} |s(θ) + a_up(θ)|² dθ,
      s(θ)  = e^{−ik(x_s sin θ + z_s cos θ)} + r(k sin θ)·e^{−ik(x_s sin θ + (2 z_int − z_s) cos θ)}

  Energy conservation gives `P_up + P_abs ≤ P_in`, with **equality over PEC**, where
  nothing enters the substrate. A lossless or lossy dielectric substrate gives strict
  inequality.

## 3. Assertions and warnings

| Condition | Action | Source |
|---|---|---|
| `Im ε_sub < 0` | raise | holomorphy §2 |
| any node, source or observation point with `z ≤ z_int` | raise | D5 |
| clearance C1 fails on ∂B̄ (branch points ±k₁, ±k₂; TM plasmon q_sp classified with the code's own α) | raise, naming the singularity and the offending corner | holomorphy §3–5 |
| `λi_max/λr_min > tan 45°` | raise | holomorphy §8 |
| `δ·D > ln(10)·3` after the depth floor | warn: expected digits lost | holomorphy §7, study "open" |
| `abs(ε_cover + ε_sub)` at round-off (lossless surface-plasmon resonance: R_∞ and q_sp infinite) | raise | study t13 |
| `spacing/(2·gap) > 0.25` (TE) or `> 0.2` (TM), with spacing the **local** node spacing on the nodes nearest the interface (not the mean), so a flat facet near the interface is caught | warn, like `ClusterGapWarning` | study trap 4, t14 |
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
| G14 | Dipole balance (§2a): `P_up + P_abs = P_in` to round-off over PEC (lossy particle, TE/TM); `P_up + P_abs < P_in` over glass and Ag | energy conservation |
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

## 6. Order of work (one commit each, conventional-commit subjects)

0. `feat: Material.from_eps` (§8.6), with its Mie test. It is independent of the
   rest, so it goes first.
1. `layered.py` primitives, `HalfSpace` and `reference/sommerfeld.py`, with G1–G5.
2. Driven wiring (matrix, RHS, near field, LDOS), plus G6, G7 and G10 for driven
   quantities.
3. Cluster wiring (owner decision 2). G6 on a two-particle cluster over PEC, which
   becomes a four-cylinder mirror reference. LDOS stays single-particle (§8.4).
3a. Far field, `σ_sca,up`, `σ_abs` (§2a), the result-key changes (§8.4), and G12–G14.
4. QNM: `for_box`, `reflected_blocks_dk`, `assemble_derivative`, G8, G9.
5. G11 plus the `performance.md` update, conventions §15, the README section and the
   figure (§8.7). Update the CLAUDE.md roadmap: v0.9 = half-space, then v1.0
   external validation, then v1.1 multilayer.

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
8. `Material.from_eps(eps, n_clad, pol)` **ships in v0.9** (owner, 2026-10-06).
   The design is in §8.6.
9. **The heater figure is parked (2026-10-06) and is not a v0.9 deliverable.** The
   v0.9 figure is the QNM-vs-gap trajectory over silver (owner, 2026-10-06), as
   specified in §8.7. The heater design below is kept for a later release.

   The parked design (LDOS only) is a heater cross-section at 1550 nm:
   - Si half-space substrate under an SiO₂ cover (n = 1.444);
   - a rounded-square (Gielis) SiN core 200 nm above the substrate;
   - a horizontally elongated tungsten heater 1 µm above the SiN;
   - relative-LDOS maps in two panels, TE and TM.

   Particle sizes are still to be fixed with the owner. The figure also needs
   a 2-D caveat: these are line-source LDOS values, not the guided mode along
   the heater axis.
10. Result keys with a background (owner, 2026-10-06): `cross_sections()` returns
    `{'c_sca_up', 'c_abs'}` and `efficiencies()` returns `{'qsca_up', 'qabs'}`.
    The v0.8 keys are absent rather than raising, and the docstrings say why.

## 8. Builder contract (2026-10-06)

Everything a builder needs that §1–§7 leave open. **Stop rule:** if a gate fails,
a measurement contradicts this spec, or something here is ambiguous, stop and
report. Do not widen a tolerance, change a formula, or pick an alternative design.

### 8.1 `HalfSpace` (`background.py`)

    @dataclass(frozen=True)
    class HalfSpace:
        eps_sub: complex | Callable[[float], complex] | None   # absolute; None = PEC
        z_int: float = 0.0
        @classmethod
        def pec(cls, z_int: float = 0.0) -> "HalfSpace"        # eps_sub=None
        @property
        def is_pec(self) -> bool
        def eps_rel(self, n_clad: float, wavelength: float | None = None) -> complex | None

- `eps_rel` returns `eps_sub/n_clad²`, or `None` for PEC. A callable is evaluated
  at the **real** vacuum `wavelength`. A complex wavelength with a callable
  raises `ValueError`. A missing wavelength with a callable also raises.
- Validation, raising `ValueError`:
  - `Im ε_sub < 0`: checked in `__post_init__` for a constant, and in `eps_rel`
    for each value a callable returns.
  - `|n_clad² + ε_sub| ≤ 1e-12·|ε_sub|` (the lossless plasmon resonance):
    checked in `eps_rel`, because `n_clad` is only known there.
  - `eps_rel` then applies the signed-zero normalisation of holomorphy §2 before
    returning.
- G7 short-circuit: when `eps_rel == 1+0j` exactly, no reflected term is built at all.

### 8.2 `layered.py` signatures (these replace §1's sketches)

    reflected_blocks(path, pol, k, eps_rel,
                     tgt: Geometry, src: Geometry, z_int, x_c)
        -> (m1_ind, m2_ind)          # (tgt.n_pts, src.n_pts); tgt = src for q = p
    reflected_blocks_dk(...same...) -> (dm1_dk, dm2_dk)
    reflected_green(path, pol, k, eps_rel, x, z, xs, zs, z_int)
        -> (G, dG/dx', dG/dz')       # broadcasting arrays, source-side derivatives

Here `x_c` is the centroid of the whole particle set (cluster), and D is that set's
horizontal extent. PEC (`eps_rel is None`) bypasses the path and uses the closed
form ∓(i/4)H₀^{(1)}(k ρ_img) (owner decision 5). The block convention is
`assemble_cross_block`'s, ported from `gemm.py`, so M1 = h·(∂x′G·dg − ∂z′G·df)
and M2 = h·G.

### 8.3 Warnings and errors

- `InterfaceGapWarning(UserWarning)`: the local-spacing rule in §3.
- `SommerfeldPrecisionWarning(UserWarning)`: `δ·D > 3·ln 10`.
- Every "raise" in §3 is a `ValueError` whose message names the quantity and its
  value. All of these are exported from `__init__.py`, together with `HalfSpace`.

### 8.4 What changes on the results when `background` is set

| API | Behaviour with a background |
|---|---|
| `ScatterResult.far_field(n_angles=3000)` | `a_up` on a uniform grid over [−π/2, π/2] inclusive |
| `ScatterResult.cross_sections(n_angles=500)` | `{'c_sca_up', 'c_abs'}` (§2a, Gauss–Legendre) |
| `ScatterResult.efficiencies(n_angles=500)` | `{'qsca_up', 'qabs'}` = the above / `2·rad` |
| `ScatterResult.multipoles` | `NotImplementedError` |
| `ScatterResult.eval_field` | §2 near field |
| `ClusterScatterResult.far_field / cross_sections / eval_field` | same as the single-particle rows |
| `self_green`, `relative_ldos`, `relative_ldos_map` | add `G_ind(r_s, r_s)` (§2). `BIESolver` only, as in v0.8 |
| `QNMSolver(geometry, material, background=None)` | single particle only; a callable `eps_sub` raises `ValueError` |
| `QNMResult.refine` | works (analytic dM/dλ) |
| `QNMResult.sensitivity` | `NotImplementedError` |

### 8.5 Test-side reference and tolerances

- Port `studies/halfspace/refquad.py` to `src/pysie2d/reference/sommerfeld.py`.
  It is a second implementation kept as a validation anchor, like `mie.py`.
  G2 and G3 import it. Its pole breakpoints must include `q_sp ± {1, 10, 100, 1000}`
  pole widths, because without them it misses a near-real pole (study t13).
- Tolerances for G1–G6, G8, G11 and G12 quote the study's measured numbers. G7 is
  bit-identity.
- G9, G10, G13 and G14 have no study number yet. Measure each once in its
  implementing commit, and set the tolerance at 10× the measured value with the
  measurement quoted in the comment (non-negotiable 4). A measurement worse than
  1e-10 for G10, G13 or G14 is a stop (§8 stop rule).

### 8.6 `Material.from_eps`

- Add the field `epsr_abs: float | None = None` (absolute Re ε) to `Material`.
  When it is set, `epsr = epsr_abs/n_clad²`; otherwise `epsr = (n_core/n_clad)²` as in
  v0.8. Nothing else reads it.
- `from_eps(eps, n_clad=1.0, pol=2)` returns
  `Material(n_core=abs(√eps), n_clad, pol, epsi=eps.imag, epsr_abs=eps.real)`.
  `n_core = |√ε|` is only the length scale that `wavelength_over_ds` reads, and
  the docstring says so.
- Tests:
  - `from_eps(n²)` agrees with `Material(n_core=n)` to round-off on `nc` and `eps`.
  - A silver cylinder (`ε = −18.3 + 0.48i`, rad = 50 nm, λ = 633 nm, TE and TM)
    matches `reference/mie.py` efficiencies. Pick the nn at which the
    efficiencies have converged, and set the tolerance per §8.5's measure-then-10× rule.

### 8.7 README figure: QNM trajectories over silver

- **Setup.**
  - Particle: circle, rad = 100 nm, `Material(n_core=3.0, n_clad=1.0)`, nn = 320.
    nn = 320 keeps spacing/(2·gap) ≤ 0.2 down to a 5 nm gap.
  - Substrate: `HalfSpace(-18.3+0.48j, z_int=0)`, with ε held at its 633 nm value
    because QNM requires non-dispersive ε (holomorphy C0). The README caption
    says so, and adds the 2-D caveat.
- **Modes.** One non-degenerate (m = 0) pole per polarisation, from the isolated
  circle (measured with `QNMSolver`, nn = 64):
  - TE: 473 + 43.2i nm (Q = 5.5);
  - TM: 819 + 92.4i nm (Q = 4.4).
- **Gaps.** `g_j = 400·(5/400)^(j/15)`, j = 0…15, in nm.
- **Continuation**, one gap at a time:
  - Search box: Re ∈ Re λ_prev·[0.9, 1.1], Im ∈ Im λ_prev·[0.5, 2.0]. Then call
    `refine()`.
  - The box must hold **exactly one** pole. Otherwise insert the midpoint gap and
    retry, at most 4 halvings. After that, or if `for_box` raises, stop and report.
- **Plot.** `examples/qnm_halfspace.py` (follow `examples/CLAUDE.md`, load
  `dataviz`):
  - Two panels against log gap: Re λ (nm) and Q.
  - TE and TM distinguishable without colour.
  - Isolated-circle values drawn as dashed horizontal lines.
- **G9 reuses this trajectory.** Every step must succeed under the halving rule.
  Recomputing every pole with the `for_box` depth × 1.5 must agree to ≤ 10× the
  `refine` tolerance.

