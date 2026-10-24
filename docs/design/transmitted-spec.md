# v1.1 transmitted fields — code-spec

Status: **implemented, shipped as v1.1.0 (2026-10-24).** The owner decisions
are §D, the verified formulas §2–§3, and the builder contract §8. Measured gate
values are in [docs/validation.md](../validation.md). One quote here was low:
GT2's worst is 4.9e-11 (glass TM, reproduced by `t1_green`), not 2.9e-11.
Prototypes: [studies/transmitted/](studies/transmitted/README.md).

Branch: `v1.1-transmitted-fields`. Multilayers move to v1.2.

Scope in one line: the field **below** the interface of a v0.9 `HalfSpace`, for
sources and particles that stay in the cover. That covers the near field, the
downward far field, the power entering the substrate and a background-field
helper, all written against a generic transmission coefficient so that a v1.2
multilayer plugs in its own.

## D. Owner decisions (2026-10-13)

| # | Decision |
|---|---|
| D1 | Deliver all four: the near field below `z_int` (`eval_field`, single particles and clusters), the downward far field, the power entering the substrate, and a background-field helper. |
| D2 | **Ill-defined far fields raise a custom error** whose message names the cause: `NoFarFieldError(ValueError)`. The far field exists only for `Im ε_sub = 0` **and** `Re ε_sub > 0`. Absorbing substrates, `Re ε ≤ 0` and PEC all raise. |
| D3 | Generality: **generic T, bottom layer only.** The new code takes a transmission coefficient, the bottom-layer wavenumber and a singularity list, and `HalfSpace` supplies them. Fields inside intermediate layers are out of scope. v0.9's reflected code is untouched. |
| D4 | Power into the substrate comes from the **interface flux, for every passive substrate**, lossy and metallic included, not from a far-field integral. |
| D5 | Builder compute budget: **< 3 min per measurement run, < 40 min in total.** New CI tests stay around 30 s in total. |
| D6 | API: a new method `far_field_down()`. `far_field()` keeps its v0.9 meaning (upward). |
| D7 | Helper: `incident_field(x, z)` gives the background field only: incident + reflected above `z_int`, transmitted below. `eval_field` stays the scattered field. |
| D8 | Error name `NoFarFieldError`, a subclass of `ValueError`. |
| D9 | The old `v1.1-layered` branch is deleted (it was identical to main). Multilayers become v1.2, on a new branch when they start. |
| D10 | No MEEP check. The anchors are closed forms and energy conservation, all in CI. |

### Owner decisions (2026-10-14)

| # | Decision |
|---|---|
| D11 | **Finding 1 ships first as a patch on `main`** (v1.0.1): a short fix branch and PR, merged before v1.1, then `main` is merged into `v1.1-transmitted-fields`. GT7's low-index gate builds on it. |
| D12 | **Finding 2: documentation and gate exclusion only.** The `c_sca_down` docstring and conventions §16 state the plasmon leak, and GT7 excludes TM over a lossless `Re ε < −n_clad²`. There is no runtime warning. `P_guided` by residues belongs to v1.2. |
| D13 | **`incident_field` contract:** `eval_field + incident_field` is the total field at **every** point. `incident_field` therefore returns 0 inside particles, classified exactly as that result class's `eval_field` classifies them (`_is_outside` for `ScatterResult`, ray casting for clusters). It works without a background too: a plane wave, or `0.25j·H₀` for a dipole. A custom `incident_rhs` makes it raise `ValueError`. The result therefore records its excitation: `ScatterResult.__init__` gains keyword-only `excitation: str = "plane_wave"` and `source: tuple[float, float] \| None = None`, and `ClusterScatterResult.__init__` gains `source`. `scatter` with a custom `incident_rhs` sets `"custom"`, and `scatter_dipole` re-wraps the result with `"dipole"` and the source position. `tests/api_baseline.txt` is regenerated. |
| D14 | Below `z_int`, `eval_field` returns the **scattered** (transmitted) field, matching v0.9's convention above it. Over PEC it is exactly 0. |
| D15 | Extinction is still not provided. |
| D16 | `far_field_down` at complex λ over a real `ε_sub > 0` returns the analytic continuation, as `far_field` already does. |
| D17 | **Points exactly at `z = z_int` keep raising `ValueError`**, as in v0.9. The field is continuous there (§3), but the user steps off the interface explicitly. GT4 checks continuity at `z_int ± δ`. |
| D18 | The near field accepts complex λ (non-negotiable 1). At complex λ the `c_sca_down` key is **absent**, because the real-axis flux is undefined when `Im k < 0` (the vertical cuts cross the real axis). This follows v0.9's "absent rather than raising". |
| D19 | `eps_rel == 1` exactly: substrate points go through the free-space representation (G7-style bit-identity), and `a_down` returns the free-space `_far_field_at`. The formula's `2α₂/(α₁+α₂)` is 0/0 at grazing there. |
| D20 | `far_field_down` without a background raises `ValueError` ("`far_field()` already covers the full circle"). |
| D21 | Lossless `Re ε_sub ≤ 0` and PEC give `c_sca_down = 0.0` exactly, with no integral. |

## 1. Generic interface (D3)

The quantity a stack supplies is **τ(q) = T(q)/α₁(q)**, not T. T carries a factor
α₁ for any stack (t₁₂ = 2α₁/(α₁+α₂)), so τ is finite at `q = ±k₁`, where T/α₁
evaluated literally is 0/0. Proposed, in `layered.py`:

    @dataclass(frozen=True, eq=False)
    class Transmission:
        pol: int
        k: complex            # cover wavenumber k₁
        k_bot: complex        # bottom-layer wavenumber, k₁·√ε_bot (principal root, signed zero normalised)
        eps_bot: complex      # bottom-layer ε relative to the cover
        z_top: float          # top interface: phase reference on the cover side
        z_bot: float          # bottom interface: phase reference in the bottom layer
        tau: Callable[[np.ndarray], np.ndarray]     # T/α₁ on the code's sheet
        features: tuple[tuple[str, complex], ...]   # singularities, for the flux breakpoints
        @classmethod
        def half_space(cls, pol, k, eps_rel, z_int) -> "Transmission"   # z_top = z_bot = z_int; features = _singularities(k, eps_rel, pol)

Half-space τ: **TE `2/(α₁+α₂)`, TM `2/(α₁+α₂/ε)`**, with the TM admittance
form of `fresnel_r` (T = 1 + R). The denominator is R's, so the singularity list
(+k₁, +k₂ and the TM plasmon q_sp) is v0.9's `_singularities` unchanged.

The Sommerfeld path is still built by `SommerfeldPath.for_wavenumber(k, eps_rel,
pol, D, z_min)`, which is half-space-specific. Generalising it to take `features`
belongs to v1.2, which needs the guided poles anyway (holomorphy §5).

## 2. Formulas (all verified, §3)

Notation: `h' = z' − z_top > 0` (source height), `|h| = z_bot − z ≥ 0`
(observation depth), `X = x − x'`. Every source is on a particle boundary or is
the dipole, and lies in the cover.

**Transmitted Green function.**

    G_t = (i/4π) ∫_Γ τ(q) · exp(i q X + i α₁ h' + i α_bot |h|) dq

Same path Γ as v0.9, folded onto `q ≥ 0` exactly as `reflected_green` is
(`pre = i/2π`, `cos(qX)`). Source derivatives: `∂x' → −iq` (folded: `q·sin(qX)`)
and `∂z' → +iα₁`, as in holomorphy §6. The large-q decay is `exp(−q(h' + |h|))`,
so the path's `z_min` is `min h'_nodes + min |h|_obs`. D is the horizontal extent
of nodes plus observation points.

**Representation below the interface.** The scattered field at a substrate point
is the transmitted term alone. There is no direct free-space term:

    ψ(r) = −Σ_j delt · ((∂x'G_t · dg_j − ∂z'G_t · df_j)·φ_j + G_t · χ_j)

This is the sign and convention of v0.9's reflected-image term. It holds for
both polarisations, by reciprocity of the 1/ε-weighted operator: the cover is
the reference medium, ε = 1. The GEMM form is `A·diag(w·τ)·Bᵀ`, with
`A = e^{iq(x−x_c) + iα_bot|h|}` and `B = e^{−iq(x'−x_c) + iα₁h'}`, banded like
`_reflected_blocks`. A pair (i, j) is reached by a tail band from `a0` iff
`|h_i| + h'_j < 38/a0`. The GEMM is **not prototyped yet** (§7).

**Plane-wave spectrum of the boundary** (generalises `_far_field_at`, which is
the case `q = k sin θ`, `α₁ = −k cos θ`):

    A(q, α₁) = Σ_j delt · e^{−i q f_j + i α₁ (g_j − z_top)} · [ i(q·dg_j + α₁·df_j)·φ_j − χ_j ]

The `g − z_top` shift keeps the exponent bounded for evanescent α₁, which would
overflow otherwise. The dipole at `(x_s, z_s)` has `A_dip = e^{−iq x_s + iα₁(z_s − z_top)}`
in the same normalisation, because `(i/4)H₀` has unit far-field amplitude in this
code's convention.

**Downward far field.** θ is measured from +z, and the substrate side is
`θ ∈ [π/2, 3π/2]`. The definition is

    ψ ~ (i/4)·√(2/(π k_bot ρ))·e^{i(k_bot ρ − π/4)} · a_down(θ)

which is `far_field`'s convention with `k_bot`, about the origin. Stationary
phase at `q_s = k_bot sin θ` gives

    a_down(θ) = α_bot(q_s) · τ(q_s) · e^{i α_bot(q_s) z_bot} · A(q_s, α₁(q_s))

with `α₁(q_s)` evaluated on the code's sheet, so it is evanescent (`+i|…|`) beyond
the critical angle when `n_sub > n_clad`. At grazing (`θ = π/2, 3π/2`)
`a_down = 0` for `ε ≠ 1`.

**Power into the substrate (D4).** In G14's units, where the unbounded-cover
dipole emits 1/4:

    P_down = (1/8π) ∫_{−∞}^{∞} Re w(q) · |τ(q)|² · |A_tot(q, α₁(q))|² dq,     w = α_bot (TE), α_bot/ε_bot (TM)

`c_sca_down = P_down / Re k₁` uses the scattered spectrum only, which parallels
`c_sca_up`. A dielectric with `Im ε = 0` has `Re w ≡ 0` for `|q| > Re k_bot`.
Over a lossless substrate this equals the far-field integral
`(1/8π)·w_far·∫|a_down|² dθ`, with `w_far = 1` (TE) and `1/ε_bot` (TM).

**Flux quadrature rule (prototype, `tproto.flux_rule`).** Real λ only.
- Breakpoints at `|Re s|` for every feature s, and at `|Re s| ± 10^m·|Im s|`,
  m = 0…6. Grading only to 10³ widths is **not** enough (§3).
- The panel width is capped at `min(4·2π/D, 8/Z_min)`, where `Z_min = 2·h_min`
  and `h_min` is the lowest node or source above the interface.
- Each panel gets 24-point Gauss–Legendre under the sine map
  `q = c + h·sin(πu/2)`, which removes √ endpoint behaviour at the branch points.
- The integration runs to `q_max = 1.5·max|Re s| + 38/Z_min`, or to `Re k_bot`
  for a lossless dielectric.
- The rule is mirrored to `q < 0`: the integrand is not even in q.

**Transmitted plane wave** (for `incident_field`). The incident wave is
`e^{ik(x sin θ − z cos θ)}`, with `q = k sin θ` and `α₁ = k cos θ`. Below the
interface it becomes `α₁·τ(q) · e^{iqx − iα₁ z_top − iα_bot(z − z_bot)}`. Under
TIR, α_bot is `+i|…|` on the sheet, so the wave decays. This formula is **not
yet prototyped**.

## 3. Measurements (prototype, 2026-10-13; light compute, seconds per run)

| Check | Result | Script |
|---|---|---|
| `ε_rel = 1`: G_t and both source derivatives vs `(i/4)H₀(kρ)`, TE/TM | 7.2e-14 at real k, 5.3e-14 at complex k (Q = 10) | t1_green |
| G_t, path vs path with the hump depth × 1.5 | ≤ 6.7e-15 (glass, silver, low-n; TE/TM) | t1b |
| G_t vs the real-axis QUADPACK reference (radiation-condition branch) | ≤ 2e-15 at most points. Up to 5.8e-12 (glass TM, deep points) and 2.9e-11 (silver TE, X = −310, h'+\|h\| = 16), where QUADPACK reports round-off: the **reference** is the limit | t1_green, t1b |
| Interface continuity `G_free + G_ind` at `z_int + 1e-12` vs `G_t` at `z_int` | ≤ 1.1e-13 (glass, silver; TE/TM; real and complex k) | t1b |
| Dipole balance `P_up + P_down + P_abs = P_in`, G14 setup (circle rad 100, nn 256, z0 115, n_core 2, epsi 0.5, n_clad 1.33, λ 633, dipole at (60, 330)) | glass 4.4e-15 / 9.6e-15 (TE/TM); lossy 2.25+0.1i 1.2e-14 / 9.5e-15; silver 1.9e-14 / 8.7e-14; low-n (ε_sub = 1) 5.2e-15 / 5.5e-15 **with the fixed P_up** (finding 1) | t2_power, t2b |
| Weak loss, flux grading 10^0–10^3 vs 10^0–10^6 widths | 2.25+1e-6i: 1.6e-11 → 4.6e-15. silver+0.01i TM: 8e-12 → 7e-14. silver+1e-4i TM: **1.0e-6** → 1.6e-12. 10^0–10^9 is the same as 10^0–10^6 | t2c |
| Flux vs far-field P_down (lossless; far field split at the critical angle, sine map) | ≤ 2.6e-14 (glass, low-n; TE/TM) | t2_power |
| Flux, 24 vs 48 nodes per panel | ~5e-15; silver TM 5.6e-13; silver+1e-4i TM 2.3e-12 | t2_power, t2c |
| Lossless metal, `P_down = 0` | ε = −1 (no plasmon on the sheet): 7.1e-15 / 5.9e-15. ε = −4: TE 1.8e-14, **TM −2.5e-2** (finding 2) | t2b, t2c |
| `ε = 1`: `a_down` vs free-space `_far_field_at` on the lower half | 5.1e-14 / 3.7e-14 | t3_far |
| Near field at ρ = 20, 40, 80 µm vs `a_down` (normalisation and phase), offsets 0, ±0.3, ±0.6 rad from −z (inside glass θ_c) | glass: 9.2e-4 / 1.0e-3 at 80 µm, Richardson (20/40 → 40/80) 6.7e-6 → 1.7e-6 (TE) and 1.3e-5 → 3.2e-6 (TM). low-n: 5.7e-4 / 6.3e-4, Richardson 6.8e-6 → 1.7e-6 and 7.5e-6 → 1.9e-6. The residual falls 4× per doubling, so it is O(1/ρ²). 15 s | t3_far |
| Per-angle near-to-far error, glass, ρ = 20 / 80 / 320 µm (2026-10-14) | Inside θ_c = 1.09 rad (offsets 0–0.9): exactly 4.0× per 4× depth in both polarisations, θ = π included, so pure O(1/ρ). Beyond θ_c (offsets 1.3, 1.45): TE 8.4e-2 → 2.5e-2 → 3.1e-3 and 1.1e-1 → 3.2e-2 → 3.7e-3; TM 6.2e-2 → 2.0e-2 → 7.1e-3 and 5.5e-2 → 1.7e-2 → 8.9e-3. Falling, but irregularly: the lateral wave from the k₁ branch point carries its own ρ-dependent phase, so `a + c/ρ` Richardson does not apply there. That is what the earlier "4e-2 stall" was. **6 min**, almost all of it at 320 µm (15,689 path nodes) | t3b |
| Banded transmitted GEMM vs the pointwise G_t sum on the same path, 200 random points of a 100 × 100 substrate map (x ∈ ±400, depth 5–600 nm), G14 particle, plane wave 20° (2026-10-24) | ≤ 2.5e-16 (glass, silver, low-n; TE/TM). The banding drops nothing. Timing: 0.19–0.23 s for the substrate map against 0.32–0.34 s for a 100 × 100 cover map through the shipped `eval_field`; 1537–1543 path nodes | t4_gemm |
| Transmitted plane wave: ψ and ∂zψ (TM: (1/ε)∂zψ) continuous at `z_int = −37`, θ ∈ {0, 20, 45, 60, 80, −35}°, glass, low-n, lossy, silver, ε_abs = −4; real and complex k (Q = 10) (2026-10-24) | value ≤ 1.8e-15, derivative ≤ 4.7e-16 (relative to k). Lossless energy `|r|² + Re(w_bot)/α₁·|T|² = 1`: ≤ 1.1e-15. TIR (low-n, 60° > θ_c = 48.8°): decay rate = Im α_bot to 1.5e-16 | t5_planewave |

## 4. Findings

1. **v0.9 bug: `c_sca_up` is inaccurate over a substrate with `Re √ε_rel < 1`.**
   `_upward_cross_sections` uses plain Gauss–Legendre over `[−π/2, π/2]`. When
   `n_sub < n_clad`, r(k sin θ) has a √ kink at the cover's critical angle
   `±θ_c = ±asin(Re √ε_rel)`, and convergence is algebraic. Plane wave at 20°, the
   G14 particle over ε_sub = 1.0 (n_clad 1.33):
   - TE: `c_sca_up` = 78.80363 / 78.78960 / 78.78624 at n_angles = 500 / 2000 / 8000,
     so the default is off by 2e-4.
   - TM: 30.09722 / 30.07444 / 30.07263.
   - Glass is unaffected: equal to 1e-13 at every node count.

   The dipole `P_up` of G14's helper has the same error (3e-5 / 4.5e-5 of
   P_in). CI never saw it, because G14 covers only PEC, glass and silver.
   **Fix (prototyped, `t2b.p_up_split`):** split at ±θ_c and use the sine map on
   each panel. The balance then closes to 5e-15, and 200 vs 400 nodes per panel
   agree to 1.2e-14. Without a critical angle the v1.0 path stays bit-identical.
   **Shipped in v1.0.1 (D11, PR #26).** The default error was worse than above:
   up to 1.3e-3 (TM, ε_sub = 1.5), and weak loss does not cure it (ε_sub =
   1 + 1e-3i: 6.5e-4), so the split applies whenever `0 < Re ε_rel < 1`, at
   `asin √(Re ε_rel)`. 500 vs 2000 nodes: ≤ 2.5e-14. G14's test helper
   `_powers` still uses plain Gauss–Legendre; GT7 must use the split rule.
2. **A lossless metal in TM (`Im ε = 0`, `Re ε < −n_clad²`) leaks power into a
   surface plasmon** that runs along the interface to infinity. That power is in
   none of `P_up`, `P_down` or `P_abs`: the balance is short by 2.5% at ε = −4.
   With any loss the plasmon is absorbed, and the flux captures it (silver
   closes to 1e-13). The docstrings must say this, and the balance gate excludes
   the case (D12). It is the half-space instance of v1.2's guided-mode channel
   (`P_guided`, by residues).
3. **Trap: size the path for the actual smallest Z.** A reflected path sized at
   `2·h'` but used at an observation point on the interface (Z = h') truncates
   the TM tail (`R_∞ ≠ 0`). The result was a 5e-12–3e-11 continuity error, which
   vanished once the path was sized correctly.
4. QUADPACK's limit at oscillatory points (2.9e-11) caps any gate against the
   reference. Tolerances there must quote it, not the path's 7e-15.

## 5. Gates (sketch; each is a test, tolerance = 10× measured with the number quoted)

| Gate | Check | Independent because |
|---|---|---|
| GT1 | `ε_rel = 1`: G_t and its derivatives equal `(i/4)H₀`, at real and complex k | closed form |
| GT2 | G_t vs the real-axis reference (extended to τ in `reference/sommerfeld.py`), glass, lossy, silver and low-n, TE/TM, real k | second quadrature, different branch rule |
| GT3 | Complex k: G_t unchanged when the hump depth is × 1.5 | consistency (GT1 is the complex-k anchor) |
| GT4 | Continuity `G_free + G_ind = G_t` across the interface, at `z_int ± δ` (D17; finding 3 for the sizing) | pins the z_top/z_bot phase references and the sign wiring |
| GT5 | `ε = 1`: `a_down` = free-space far field on the lower half. Near-to-far: Richardson of the near field at ρ = 40 and 80 µm equals `a_down` at offsets 0, ±0.3, ±0.6 rad from −z, glass and low-n, TE/TM (measured ≤ 3.2e-6; never beyond θ_c, §3 `t3b`) | free space is Mie-anchored; the near field and `a_down` are different integrals |
| GT6 | Lossless: flux `P_down` = far-field integral | two quadratures of different form |
| GT7 | **G14 becomes an equality:** `P_up + P_down + P_abs = P_in` over glass, low-n, lossy, silver and a lossless metal with −n_clad² < ε < 0; TE/TM. It replaces the shortfall test, and TM over ε < −n_clad² lossless is excluded (finding 2) | energy conservation |
| GT8 | PEC: field below = 0, `c_sca_down = 0`; `NoFarFieldError` for PEC, lossy and `Re ε ≤ 0`, each with its message | guard tests |
| GT9 | `incident_field`: equals the solver's RHS φ-half on the boundary (an unmasked internal), and is continuous across the interface (plane wave, dipole) | pins the excitation wiring |
| GT10 | Scale covariance of the transmitted field, `a_down` and `c_sca_down` | conventions §9 |
| GT11 | Regression: `background=None` and every v1.0 output above the interface bit-identical; the existing `cross_sections` keys unchanged | regression |

## 6. Conventions entries (new §16, written in the implementing commit)

τ = T/α₁ as the generic coefficient; the `z_top`/`z_bot` phase references; the
`a_down` normalisation with `k_bot` and the TM power weight `1/ε_bot`; `P_down`
by interface flux with the scattered spectrum only; the `NoFarFieldError`
conditions; the plasmon channel (finding 2); and the `incident_field`
contract.

## 7. Left to do before this is builder-ready

1. ~~Glass near-to-far stall~~: resolved 2026-10-14 (§3, `t3b`). There is no stall
   and `a_down` is right: inside θ_c the error is pure O(1/ρ) and Richardson
   reaches 1.7e-6 / 3.2e-6. The angle π+1.2 lay beyond θ_c, where the lateral
   wave breaks Richardson's model. GT5's near-to-far half uses offsets within
   ±0.6 rad of −z, at ρ = 40 and 80 µm.
2. ~~Banded GEMM and transmitted plane wave~~: prototyped 2026-10-24 (§3,
   `t4_gemm`, `t5_planewave`). The GEMM equals the pointwise sum to 2.5e-16 and
   the substrate map is cheaper than the cover map. The plane wave meets both
   interface conditions to round-off, conserves energy and decays at Im α_bot
   under TIR.
3. ~~Owner calls~~: done, D11–D21 (2026-10-14). The D11 patch shipped as
   v1.0.1 and is merged into this branch.
4. ~~Builder contract~~: written 2026-10-24, §8.
5. Update the CLAUDE.md roadmap status when the spec is ready: done in the
   commit that lands §8.

## 8. Builder contract (2026-10-24)

Everything a builder needs that §1–§7 leave open. **Stop rule:** if a gate
fails, a measurement contradicts this spec, a run exceeds the D5 budget, or
something here is ambiguous, stop and report. Do not widen a tolerance, change a
formula, or pick an alternative design. The detours in §8.8 are the only
pre-approved deviations.

### 8.1 `layered.py`

    class NoFarFieldError(ValueError)    # D2, D8

    @dataclass(frozen=True, eq=False)
    class Transmission:                  # §1, fields exactly as listed there
        pol: int; k: complex; k_bot: complex; eps_bot: complex
        z_top: float; z_bot: float
        tau: Callable[[np.ndarray], np.ndarray]
        features: tuple[tuple[str, complex], ...]
        path: SommerfeldPath | None      # None until built for a point set
        x_c: float
        @classmethod
        def half_space(cls, pol, k, eps_rel, z_int) -> "Transmission"
        @classmethod
        def build(cls, pol, k, eps_rel, z_int, f_src, g_src, f_obs, g_obs) -> "Transmission | None"
        def blocks(self, tgt: Points, src: Geometry) -> tuple[np.ndarray, np.ndarray]
        def green(self, x, z, xs, zs) -> tuple[np.ndarray, np.ndarray, np.ndarray]
        def plane_wave(self, angle: float, f: np.ndarray, g: np.ndarray) -> np.ndarray

    transmitted_green(path, tr, x, z, xs, zs) -> (G_t, ∂x'G_t, ∂z'G_t)
    transmitted_blocks(path, tr, tgt: Points, src: Geometry, x_c) -> (m1_t, m2_t)

- `half_space` sets `k_bot = k·√eps_rel` (after `_normalised`), `eps_bot =
  eps_rel`, `z_top = z_bot = z_int`, `tau` as §1, `features =
  _singularities(k, eps_rel, pol)`, `path = None`, `x_c = 0.0`. It is never called
  with PEC (`eps_rel is None`).
- `build` returns `None` for PEC **and** for `eps_rel == 1` exactly (D14, D19:
  the caller then uses 0 or the free-space representation). Otherwise it checks
  `g_src > z_int` and `g_obs < z_int` (`ValueError` naming the offending set), and
  sizes the path as §2: `z_min = min(g_src − z_int) + min(z_int − g_obs)`, `D =
  max(ptp(concat(f_src, f_obs)), z_min)` (the floor keeps scale covariance, as
  `Reflection.build`), `path = SommerfeldPath.for_wavenumber(k, eps_rel, pol, D,
  z_min)`, `x_c` = midpoint of the same concatenated x range.
- `transmitted_green` and `transmitted_blocks` are `t_green` and
  `t4_gemm.t_blocks` from the prototypes, written against `tr` (`tr.tau`,
  `tr.k`, `tr.k_bot`, `tr.z_top`, `tr.z_bot`), never against `eps_rel` directly:
  that is D3. The banded rule is the one in `t4_gemm`. The block convention is
  `reflected_blocks`' (`M1 = h·(∂x'G·dg − ∂z'G·df)`, `M2 = h·G`), so the
  field is `−(m1 @ φ + m2 @ χ)`.
- `plane_wave` is §2's transmitted plane wave, `|angle| ≥ 90°` raising
  `ValueError` with `Reflection.plane_wave`'s message.
- Complex `k` is accepted everywhere (non-negotiable 1). No real-only branch.

### 8.2 `fields.py`

    _spectrum(q, a1, boundaries, z_top) -> np.ndarray              # A(q, α₁), §2
    _downward_amplitude(angles, boundaries, wnum_bg, eps_rel, pol, z_int) -> np.ndarray
    _downward_far_field(n_angles, boundaries, wnum_bg, eps_rel, pol, z_int)
        -> (amp, angles)                                            # uniform on [π/2, 3π/2] inclusive
    _flux_rule(wnum_bg, eps_rel, pol, D, z_min) -> (q, w)            # §2, tproto.flux_rule, widths m = 0…6
    _downward_power(q, w, A, wnum_bg, eps_rel, pol) -> float          # (1/8π) Σ w·Re w_bot·|τ A|²
    _downward_cross_section(boundaries, wnum_bg, eps_rel, pol, z_int) -> float

- `_downward_amplitude` is `tproto.a_down`. For `eps_rel == 1` exactly it returns
  `_far_field_at` summed over the boundaries (D19).
- `_downward_cross_section` sizes the rule with `h_min` = lowest node above
  `z_int`, `z_min = 2·h_min`, `D = max(ptp(all f), z_min)`, and returns
  `_downward_power / Re k`. It returns `0.0` with no integral for PEC and for
  `Im ε = 0, Re ε ≤ 0` (D21). For a lossless dielectric the rule stops at
  `Re k_bot` (§2).
- `_downward_power` takes the spectrum as an argument so that GT7 can add the
  dipole's own `A_dip` (§2). It is the only entry point GT7 uses.
- The flux is defined at real `k` only. The callers never reach it at complex
  `k` (D18).

### 8.3 `solver.py` and `cluster.py`

    ScatterResult.__init__(self, ei, geometry, material, wavelength, angle=0.0,
                           background=None, *, excitation: str = "plane_wave",
                           source: tuple[float, float] | None = None)
    ScatterResult.far_field_down(self, n_angles: int = 3000) -> tuple[np.ndarray, np.ndarray]
    ScatterResult.incident_field(self, x, z) -> np.ndarray
    ClusterScatterResult.__init__(..., excitation, background=None, *,
                                  source: tuple[float, float] | None = None)
    ClusterScatterResult.far_field_down(self, n_angles: int = 3000) -> tuple[np.ndarray, np.ndarray]
    ClusterScatterResult.incident_field(self, x, z) -> np.ndarray

- `excitation` is `"plane_wave"`, `"dipole"` or `"custom"` (D13).
  `BIESolver.scatter` with `incident_rhs` sets `"custom"`; `scatter_dipole`
  re-wraps its result as `ScatterResult(res.ei, ..., excitation="dipole",
  source=(x_s, z_s))`. `ClusterBIESolver.scatter_dipole` passes `source`.
- Both result classes gain an internal `_background_field(x, z)`: the
  unmasked background field (§8.4 row `incident_field`), which GT9 calls on
  boundary nodes. `incident_field` is `_background_field` with 0 at interior
  points, classified as that class's `eval_field` classifies them (D13).
- `eval_field` splits its points by `z`: above `z_int` the v0.9 path, unchanged
  and bit-identical; below it, `Transmission.build(...)` once for all the
  substrate points and the result's boundaries, then `−(m1 @ φ + m2 @ χ)` summed
  over particles. `z == z_int` raises `ValueError` (D17). No substrate point can
  be inside a particle (particles are in the cover), so there is no
  classification below `z_int`.
- `far_field_down` raises `ValueError` without a background (D20), and
  `NoFarFieldError` for PEC, `Im ε_rel ≠ 0` or `Re ε_rel ≤ 0`, with a message that
  names the condition and the value (D2). A callable `eps_sub` is evaluated at
  the real wavelength, as `eps_rel` already does.

### 8.4 Behaviour with a background (additions to v0.9 §8.4)

| API | Behaviour |
|---|---|
| `eval_field(x, z)`, `z < z_int` | scattered (transmitted) field, §2; 0 over PEC (D14); free-space representation at `ε_rel = 1` (D19); complex λ accepted (D18) |
| `eval_field(x, z)`, `z == z_int` | `ValueError` (D17) |
| `far_field_down(n_angles=3000)` | `(a_down, angles)` on `[π/2, 3π/2]` inclusive; `a_down = 0` at both ends for `ε ≠ 1`; analytic continuation at complex λ over a real `ε > 0` (D16) |
| `cross_sections()` | real λ: `{'c_sca_up', 'c_sca_down', 'c_abs'}`; complex λ: `'c_sca_down'` absent (D18). `c_sca_up` and `c_abs` unchanged bit for bit |
| `efficiencies()` | adds `'qsca_down' = c_sca_down / (2·rad)`, under the same rule |
| `incident_field(x, z)` | plane wave: incident + `Reflection.plane_wave` above, `Transmission.plane_wave` below. Dipole: `0.25j·H₀(kρ)` + `Reflection.green(...)[0]` above, `Transmission.green(...)[0]` below. 0 below PEC. 0 inside particles. `ValueError` at `z == z_int`, at the dipole's own position, and for `"custom"` |
| `incident_field(x, z)`, no background | plane wave `exp(ik(x sinθ − z cosθ))` or `0.25j·H₀`; 0 inside particles |

The existing `test_result_keys_change_with_a_background` asserts the key set
`{'c_sca_up', 'c_abs'}`. It is updated to the three-key set, in the commit
that adds `c_sca_down`. That is the one existing assertion this milestone
changes.

### 8.5 Exports and the API baseline

`NoFarFieldError` is exported from `__init__.py`. `Transmission` is internal,
like `Reflection`. `tests/api_baseline.txt` is regenerated in the
`incident_field` commit (§8.7), by `test_baseline_regeneration_recipe`. Every
change is an addition (new keyword-only parameters, new methods), so no
`BREAKING CHANGE` footer.

### 8.6 Test-side reference and tolerances

- `reference/sommerfeld.py` gains `gt_ref(X, hs, hb, k1, eps_rel, pol, which=0)`,
  ported from `tproto.t_ref`, with the module's `POLE_WIDTHS` breakpoints and
  its radiation-condition branch rule. Real k only, as `g_ref`.
- Tolerances, 10× measured, each with its measurement in the comment:

| Gate | Setup | Bar |
|---|---|---|
| GT1 | `t1_green` points and sources, real and complex k, TE/TM, G and both derivatives | 7.5e-13 (7.2e-14) |
| GT2 | `t1_green` points, glass, lossy, silver, low-n, TE/TM, `err / max|ref|` | 3e-10 (2.9e-11, the reference's limit, finding 4) |
| GT3 | `t1b`, depth × 1.5 at complex k (Q = 10) | measure-then-10× (6.7e-15 was measured at real k only); worse than 1e-12 is a stop |
| GT4 | `t1b` continuity at `z_int ± 1e-12`, glass and silver, real and complex k | 1.2e-12 (1.1e-13); path sized per finding 3 |
| GT5 | `ε = 1`: `far_field_down` vs `_far_field_at`, 1e-12 (5.1e-14). Near-to-far: Richardson at ρ = 40, 80 µm, offsets 0, ±0.3, ±0.6 rad, glass and low-n, TE/TM | 3.5e-5 (3.2e-6) |
| GT6 | lossless glass and low-n, flux vs far-field `P_down` | 3e-13 (2.6e-14) |
| GT7 | G14 setup, glass, low-n, lossy `2.25+0.1i`, silver, `ε_abs = −1`; TE/TM. TM over a lossless `ε_rel < −1` excluded (D12). `P_up` by the v1.0.1 split rule | 1e-12 (8.7e-14, silver TM) |
| GT8 | guards | exact |
| GT9 | boundary nodes: `_background_field = rhs[:nn]` for the plane wave and the dipole, over glass and silver; continuity of `incident_field` at `z_int ± 1e-9` | 1e-13 vs RHS (same formula, round-off); continuity measure-then-10× |
| GT10 | scale covariance, `s = 1.7`, of `eval_field` below, `far_field_down`, `c_sca_down` | measure-then-10×; worse than 1e-12 is a stop |
| GT11 | `background=None` and every output above the interface | bit-identical (`np.array_equal`) |
| GT12 | GEMM vs pointwise `transmitted_green` sum on a 20 × 20 substrate map | 3e-15 (2.5e-16, `t4_gemm`) |

GT7 replaces `test_dipole_power_falls_short_of_the_input_over_a_lossy_or_open_substrate`
and its `MIN_SHORTFALL`. The PEC balance test stays as it is. GT7's helper
moves `_powers` to the v1.0.1 split rule (finding 1).

### 8.7 Order of work (one commit each, conventional form)

1. `feat: transmitted Green function for a half-space`: §8.1, `gt_ref`; GT1–GT4,
   GT12.
2. `feat: near field below the interface`: `eval_field` on both result
   classes; GT11, plus GT4 at the `eval_field` level (a scattered field
   continuous with the total above, PEC giving 0).
3. `feat: downward far field`: `far_field_down`, `NoFarFieldError`; GT5, GT8.
4. `feat: power into the substrate`: §8.2 flux, `c_sca_down`, `qsca_down`; GT6,
   GT7, the key-set test update.
5. `feat: incident_field on scatter results`: §8.3 excitation recording, the
   API baseline; GT9.
6. `test: scale covariance of the transmitted field`: GT10.
7. `docs: transmitted fields`: conventions §16 (§6 list), the `c_sca_down`
   plasmon note (D12), README feature line, docs/validation.md rows for
   GT1–GT12, CLAUDE.md roadmap status.

Each commit passes the full suite. Run each new gate's measurement once before
setting its bar (§8.6), and quote the number.

### 8.8 Pre-approved detours

- A flux gate stalls above its bar: 48 nodes per panel instead of 24 (§3:
  5e-15 apart), then decades m = 0…9 instead of 0…6 (§3: identical). Neither
  helps: stop.
- GT7 fails on one substrate only: compare flux and far-field `P_down` there
  (GT6 machinery) to split the blame between the flux and `P_up`, then stop and
  report.
- GT5's near-to-far half exceeds the D5 budget: drop the low-n case, keep glass.
- Nothing else.

### 8.9 Budget

D5: each measurement run under 3 min, under 40 min in total. The new CI tests
stay around 30 s in total. Measure their time in commit 7 and report it.
