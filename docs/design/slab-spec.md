# v1.2 slab waveguide — code-spec

Status: **builder-ready, 2026-10-25.** The owner decisions are R1–R18 in
[waveguide-roadmap.md](waveguide-roadmap.md) §1, with R16 revised below. The
formulas in §2 are each verified in the study,
[studies/slab/](studies/slab/README.md) S1–S12, and the measured numbers are
in §3. The builder contract is §8.

Branch: `v1.2-slab-waveguide`. **Additive**: no existing signature changes
meaning, and `HalfSpace` results are bit-identical.

Scope in one line: an N-layer dielectric `Stack` background under particles
that stay in the cover. It provides the guided modes, the near field
everywhere (cover, layers, bottom), mode and in-plane-dipole sources, complex
power-normalised mode amplitudes with a full power account, and a
finite-difference sensitivity helper. The R12 shape optimisation is an
`examples/` script.

## D. Decisions

### Owner decisions this spec implements

R1 (particles outside the core), R2 (no slab QNMs), R5 (N layers), R11
(cross-section first), R12 (first objective: the dipole's +x guided
efficiency), R13 (complex amplitudes + powers), R14 (`x_ref` required), R15
(near-degenerate pairs refused), R17 (Gielis now, optimiser in `examples/`),
R18 (in-plane electric dipoles). R4 is dropped.

| # | Decision (owner, 2026-10-25) |
|---|---|
| R16′ | **R16 revised:** a finite-difference sensitivity helper, not an adjoint. An adjoint whose ∂M/∂p is a central difference costs the same 2 assemblies per parameter as differencing the objective. A real saving needs analytic kernel shape derivatives, which move to the Fourier-shape milestone. |
| R19 | **The near field inside the layers is in scope.** `eval_field` works at every z outside the particles and off the interfaces. |
| R20 | **Names accepted:** `Stack`, `Stack.guided_modes`, `GuidedMode`, `BIESolver.scatter_mode`, `ScatterResult.mode_amplitudes`, `ModeAmplitudes`, `scatter_dipole(..., moment=)`, `NearDegenerateModesError`. |

### Technical decisions taken by this spec (flagged for owner review)

| # | Decision | Why |
|---|---|---|
| T1 | `Stack.guided_modes(wavelength, pol, n_clad)` takes the cover index as a third argument. | The cover is `Material.n_clad`; a stack does not know it. |
| T2 | **Guided outputs need a lossless stack at real λ.** `guided_modes`, `mode_amplitudes`, `scatter_mode` and the `c_guided` key raise or are absent (T8) otherwise. The field, LDOS and the other cross-sections accept loss and complex λ. | A lossy or complex-λ guided mode is not an output channel at x → ±∞. |
| T3 | **Layers and the bottom must have Re ε > 0** when given as constants. A callable is checked at evaluation. `ValueError` otherwise. A bare metal substrate stays `HalfSpace`. | The mode solver brackets on the dielectric window; plasmonic layers have poles outside it and are a later item. |
| T4 | **The R12 example uses pol = 2 (slab TE₀ in cross-section) with the scalar line source**, with directionality from the antenna's asymmetry. In-plane electric dipoles couple to pol = 1 (slab TM₀) in cross-section; spin-locked TM₀ is a second objective in the same script. | E_y is driven only by a y-directed current. The in-plane dipole reaches slab TE₀ only in the v1.4 top view. |
| T5 | `QNMSolver(..., background=Stack)` raises `ValueError` (R2). | Slab QNMs are out of scope. |
| T6 | The stack is **normalised at evaluation**: adjacent equal layers merge, trailing layers equal to the bottom merge into it, leading layers equal to the cover (`ε_rel == 1`) are dropped and `z_top` moves down. With no layer left, the call routes to the `HalfSpace` code (or to no background), bit-identically. | It removes r = 0 interfaces, where the interior and outer sheets can disagree (0/0), and it makes A1 exact. |
| T7 | `moment=` is accepted for pol = 1 only. pol = 2 with `moment` raises `ValueError`. | R18: TM. The TE form would be a magnetic dipole, with another constant to pin. |
| T8 | At complex λ, or over a lossy stack, `cross_sections()` omits `c_guided` (as v1.1 omits `c_sca_down` at complex λ), and `mode_amplitudes` raises. | "Absent rather than raising" for dictionaries, as in D18; an explicit call raises. |
| T9 | The sensitivity helper is `pysie2d.mode_sensitivity(at, run, *, step=SHAPE_STEP)`. | A name the owner has not seen yet. |

## 1. Interface

### 1.1 `background.py`

    @dataclass(frozen=True)
    class Stack:
        layers: tuple[tuple[complex | Callable[[float], complex], float], ...]   # (absolute ε, thickness nm), top → bottom
        eps_bot: complex | Callable[[float], complex]                           # absolute
        z_top: float = 0.0
        @property
        def z_bot(self) -> float                                                # z_top − Σ thickness
        def eps_rel(self, n_clad, wavelength=None) -> tuple[tuple[complex, ...], tuple[float, ...], complex, float]
            # (layer ε_rel, thicknesses, ε_bot,rel, z_top) after normalisation T6
        def guided_modes(self, wavelength: float, pol: int, n_clad: float) -> tuple[GuidedMode, ...]

    class NearDegenerateModesError(ValueError)

`__post_init__`: at least one layer; every thickness > 0; constants passive
(`Im ε ≥ 0`, `_check_passive`) and `Re ε > 0` (T3). `eps_rel` applies
`HalfSpace.eps_rel`'s signed-zero normalisation to every value and evaluates
callables at a real λ, raising as `HalfSpace.eps_rel` does.

### 1.2 `GuidedMode` (in `layered.py`, exported)

    @dataclass(frozen=True, eq=False)
    class GuidedMode:
        order: int            # 0 = highest β
        pol: int
        beta: float           # rad/nm, background-independent
        n_eff: float          # beta / (2π/λ_vac)
        norm: float           # N = ∫ p u² dz (nm), u(z_top) = 1, p = 1 (TE), 1/ε_rel (TM)
        gamma: float          # cover decay γ₁ = √(β² − k₁²)
        wavelength: float
        def profile(self, z) -> np.ndarray     # u(z), u(z_top) = 1, every z (real)

### 1.3 Solvers and results

    BIESolver(geometry, material, background: HalfSpace | Stack | None = None)
    BIESolver.scatter_mode(self, wavelength, mode: GuidedMode | int, direction: int, *, x_ref: float) -> ScatterResult
    BIESolver.scatter_dipole(self, wavelength, x_s, z_s, *, moment: tuple[complex, complex] | None = None) -> ScatterResult
    ScatterResult.mode_amplitudes(self, x_ref: float) -> ModeAmplitudes
    ClusterBIESolver: the same three, with the same signatures; ClusterScatterResult.mode_amplitudes likewise

    @dataclass(frozen=True)
    class ModeAmplitudes:
        modes: tuple[GuidedMode, ...]
        x_ref: float
        excitation: str                 # "dipole", "mode", "plane_wave"
        s_plus: np.ndarray              # complex (n_modes,): amplitude of ũ_n(z)e^{+iβ_n(x − x_ref)}
        s_minus: np.ndarray             # complex: of ũ_n(z)e^{−iβ_n(x − x_ref)}
        p_plus: np.ndarray              # |s_plus|²
        p_minus: np.ndarray
        p_up: float
        p_down: float
        p_abs: float
        p_source: float | None          # dipole: emitted power; mode: 1.0; plane wave: None
        def efficiency(self, mode: int, direction: int) -> float     # p_±[mode]/p_source; ValueError for a plane wave
        def balance(self) -> float                                   # |p_source − Σ|/p_source; ValueError for a plane wave

    def mode_sensitivity(at: Callable[[float], tuple[Geometry, Material]],
                         run: Callable[[Geometry, Material], ModeAmplitudes], *,
                         step: float = SHAPE_STEP) -> dict[str, np.ndarray | float]

`mode_sensitivity` returns the central differences, `(run(at(+step)) −
run(at(−step)))/(2·step)`, of `s_plus`, `s_minus` (complex arrays), `p_plus`,
`p_minus` (arrays), `p_up`, `p_down`, `p_abs` and `p_source` (floats; `p_source`
only when it is not `None`), under those keys. Before differencing it checks the
frozen map as `QNMResult.sensitivity` does: `at(±step)` must carry
`at(0).geometry.parametrisation`'s θ, w′, w″ exactly, and a geometry with no
map raises by name. The check is factored out of `qnm.py` into one shared
helper, not copied. It costs 2 solves per parameter; the caller loops over
parameters.

`ũ_n = u_n/√(β_n N_n)` is the **power-normalised** mode. Powers are in G14's
units, where the free line source emits 1/4. Under mode incidence the incident
power is 1, so `s_plus`/`s_minus` are the S-matrix column directly:
`S_{n←m}^{fwd} = s_plus[n]` (the δ_nm included) for `direction = +1`. For a
plane wave, `p/Re k` is a coupling cross-section in nm, like `c_sca_up`.

## 2. Formulas (each verified; § and study item given)

Frame as v0.9: cover above `z_top`, layers downward, bottom below `z_bot`,
background-relative ε (cover = 1), `k = k_bg`. Admittance Y = α (TE) or α/ε
(TM). Outer α on the vertical-cut sheet (`layered.alpha`); **interior α_j with
Im α_j ≥ 0** (R and τ are even in every interior α_j).

**2.1 Recursion (S1).** Γ at the top of each medium, bottom up, Γ_{n+1} = 0.
Across the interface a|b, in the division-free form (T6 removes r = 0
interfaces; this form also never divides by Y_a + Y_b alone):

    G_a = ((Y_a − Y_b) + (Y_a + Y_b)Γ_b) / ((Y_a + Y_b) + (Y_a − Y_b)Γ_b),     Γ_j = G_j e^{2iα_j d_j}

`R = G_0`. **τ = T/α₁** = [2/(Y₀+Y₁)]·Π_{i≥1}[2Y_i/(Y_i+Y_{i+1})]·e^{iα_i d_i} ·
Π_{i=0}^{n} 1/(1 + r_iΓ_{i+1}), with r_i = (Y_i − Y_{i+1})/(Y_i + Y_{i+1}).
Never form e^{+|Im α|d}.

**2.2 In-layer amplitudes (`nearfield.py`).** Per unit *downward* amplitude at
`z_top`: in layer j, with s = z_j − z ∈ [0, d_j], W = A_j[e^{iα_j s} +
G_j^{bot}e^{iα_j(2d_j − s)}], where A₁ = t₀/(1 + r₀Γ₁) and A_{j+1} =
A_j e^{iα_j d_j}t_j/(1 + r_jΓ_{j+1}). In the bottom, W = α₁τ·e^{−iα_b(z − z_bot)}.
`W/α₁` is finite at q = ±k₁, because t₀/α₁ = 2/(Y₀+Y₁).

**2.3 Dispersion and modes (S2, S4).** Real roots of the **scaled** D (the
transfer-matrix denominator of R, each layer's matrix × e^{−|Im α_j|d_j},
same roots, real on the window), bracketed on a uniform grid in
γ = √(q² − k_out²) **from γ = 0**:

    n_grid = max(4000, ⌈400·(1 + k_max Σd/π)⌉)

Window `(k_out, k_max)`, `k_out = max(Re k₁, Re k_bot)`, `k_max = max Re k_j`.
Refine with `brentq` (`rtol = 8.9e-16`). **Check** the count against the
argument principle on D(q(γ)) along the γ-rectangle [0, 1.05γ_hi] × [−h, h],
`h = 0.01·γ_hi`, trapezoid with `n = max(8192, ⌈64·k_max Σd⌉)` points. **h must
stay thin**: Re γ > 0 is the Im α ≥ 0 sheet, which differs from the code's sheet
left of k_out (S4 counted a leaky pole at h = 0.05). On a shortfall, refine the
grid ×10, twice. If it is still short, raise `NearDegenerateModesError` in
`guided_modes`, naming the γ-cell and the lower bound π/Δβ_cell on the beat
length.

Profile (S8): (u, v = p·u′) transfer from u(z_top) = 1, v = −γ₁, with
`N = 1/(2γ₁) + Σ_j ∫ p u² dz + p_b u_b²/(2γ_b)`, 200-point Gauss–Legendre per
layer. The bottom mismatch |v_b − p_bγ_bu_b|/|p_bγ_bu_b| is ≤ 1e-12 at a root.
**Identity: Res_β R = γ₁u(z_top)²/(βN)**, the bilinear norm, the 1/ε TM weight
load-bearing (S8: 74 % off without it).

**2.4 Path (S3–S6).** `SommerfeldPath._build` takes the feature list
`[("+k₁", k), ("+k_bot", k_bot)] + [("β_m", β_m)]`. Every v0.9 rule applies
unchanged: T = 1.5·max Re, δ, clearance C1, node counts, banding. For complex k
or a lossy stack, the β_m are the real roots at `Re k` with lossless `Re ε`,
**continued** (S6): a homotopy in 16 steps, ε(s) and k(s) linear, Newton on D,
D′ from a 64-point Cauchy circle of radius 1e-4·|q|, |Δq| < 1e-15|q|, at most 30
steps per stage. Non-convergence raises `ValueError`. The path for fields needs
only clearance, so if the count check is still short after refinement (§2.3),
the unresolved γ-cell's centre is added as a feature, and only the guided
outputs raise.

**2.5 Reflected and transmitted Green functions.** The v0.9 `G_ind` with `R` of
§2.1 and `Z = z + z′ − 2z_top`. v1.1's `G_t` with τ of §2.1 and
`Transmission(z_top, z_bot, k_bot, eps_bot, tau, features)` — already generic
(D3). In-layer: `G_l = (i/4π)∫_Γ (W/α₁)(q, z)·e^{iqX + iα₁h′}dq`.

**2.6 Mode amplitudes (S12).** With u_s = −(M1φ + M2χ) (conventions §15), as
the amplitude of u_m(z)e^{±iβx}:

    c_m^± = −(i/(2β_mN_m)) Σ_j h u_m(z_j) e^{∓iβ_m x_j} [(∓iβ_m dg_j + γ₁ df_j)φ_j + χ_j]

summed over every particle. The source's own term is added: scalar dipole
`(i/(2βN))u(z_s)e^{∓iβx_s}` (S9); in-plane `(i/(2βN))u(z_s)(±iβb_x + γ₁b_z)e^{∓iβx_s}`
(S11). Then `s_± = √(βN)·c_±·e^{±iβx_ref}`. Mode incidence adds the incident
mode itself to `s_plus[m]` (direction +1) or `s_minus[m]` (−1).

**2.7 Power account (S9, S11, S12).** With the spectra of the total field,
referred to `z_top` (`fields._spectrum`'s normalisation, so the scalar dipole's
downward spectrum is `e^{−iqx_s + iα₁(z_s − z_top)}`, v1.1 §16):

    u↑(q) = −Σ_j h e^{−iqx_j − iα₁h_j}[(−iq dg_j + iα₁ df_j)φ_j + χ_j]  + source↑
    d(q)  = −Σ_j h e^{−iqx_j + iα₁h_j}[(−iq dg_j − iα₁ df_j)φ_j + χ_j]  + source↓
    scalar source:   source↑ = e^{−iqx_s − iα₁h_s},  source↓ = e^{−iqx_s + iα₁h_s}
    in-plane source: × i(qb_x + α₁b_z) (↑) and × i(qb_x − α₁b_z) (↓)
    P_up   = (1/2π) ∫_{−π/2}^{π/2} |u↑ + R d|²/4 dθ        (q = k sin θ; split at the critical angle of k_bot if Re k_bot < k)
    P_down = (1/2π) ∫ Re(Y_b)|τ d|²/4 dq                  (v1.1 `_flux_rule` with the stack's features)

**Amended at build (owner, 2026-10-26).** Over a lossless bottom `P_down` uses
S12's rule — `q = k_bot sin s`, 16 Gauss–Legendre panels of 200 nodes, split at
`±asin(k₁/k_bot)`, sine-mapped per panel — not `_flux_rule`. The flux rule
gets no breakpoints from real poles beyond `Re k_bot` and puts one 24-point
panel on `[0, k_bot]`: the TM dipole balance stalled at 9.3e-7 (SOI) against
3.2e-15. A lossy bottom keeps `_flux_rule`.
    P_abs  = −Im Σ h conj(φ)χ                              (v0.9)
    P_guided = Σ_m (p_plus[m] + p_minus[m])

Source power: scalar `1/4 + Im G_refl(r_s, r_s) + Im u_s(r_s)`. In-plane
`k²|b|²/8 + Im[(b*·∇)(b·∇′)G_refl] + Im[−(b*·∇)u_s](r_s)`, where the reflected
self term is the path sum with weight `q²|b_x|² − α₁²|b_z|²` (the odd cross
terms vanish on the symmetric path) and ∇u_s is the **analytic** observer
gradient of the representation (§8.2).

**2.8 In-plane dipole (S11).** `moment = (p_x, p_z)` gives `b = (p_z, −p_x)` and
the source field `u = −(b·∇′)G(r, r_s)`. The returned field is `H_y/(iω)` per
unit moment in the scalar source's units (from ∇·(p∇H) + k²H = −[∇×J]_y,
J = −iωp, `exp(−iωt)`). One-way coupling into mode m when `b_z = iβ_m b_x/γ₁`.
Free-space far field `∝ i k(b_x sin θ + b_z cos θ)`, θ from +z, so
`|a|² ∝ |p_x cos θ − p_z sin θ|²`.

**2.9 Translation (sensitivity anchor).** Moving every particle by Δ along x at
fixed `x_ref` multiplies the scattered part of `S_{n←m}^{fwd}` by
`e^{i(β_m − β_n)Δ}` and `S_{n←m}^{bwd}` by `e^{i(β_m + β_n)Δ}`. So the derivatives
are `i(β_m ∓ β_n)·S_scat` exactly, and `d|S|²/dΔ = 0`.

## 3. Measurements (study, 2026-10-25; seconds per run)

| Item | Result | Script |
|---|---|---|
| N = 0 recursion vs `fresnel_r`, `_tau_half_space` | 0 (bitwise) | s1 |
| lossless energy \|R\|² + (Re Y_b/Y₁)\|α₁τ\|² = 1 | 2.0e-15 | s1 |
| naive transfer matrix on a real path | NaN from q·Σd ≈ 709 (SOI at a 5 nm gap; SiN, double core at 10–20 nm) | s1, s5 |
| roots vs closed form | ≤ 4.8e-16 relative, up to 9 modes; three-way counts agree | s2 |
| two residue routes | 4e-15 | s2 |
| path vs reference, 5 stacks, TE/TM, G and ∂z′G | ≤ 4e-10 (reference floor, spread 2.5e-10) | s3 |
| near cutoff t = 1e-6 / 1e-8 vs γ-reference | ≤ 8e-10 / ≤ 3e-8 (reference spread) | s4 |
| q-grid finder misses at t ≤ 1e-4; γ-grid finds all | — | s4 |
| gaps 10/5/2 nm, path vs reference | ≤ 5e-11; M = v0.9's + 1–3 % | s5 |
| lossy core, real k, path vs reference | ≤ 5e-14 | s6 |
| complex k Q = 10: s = 2, δ × 1.5 vs default | ≤ 4e-13 | s6 |
| transmitted G vs shooting reference | ≤ 4e-11 | s7 |
| Res_β R = γ₁u²/(βN) | ≤ 4e-14 (16 modes per pol) | s8 |
| dipole power balance (no particle) | ≤ 3.5e-15 | s9 |
| in-plane dipole power balance (no particle) | ≤ 4.8e-15; one-way P⁺/(P⁺+P⁻) = 1.0000 single-mode | s11 |
| dipole + particle power balance | ≤ 1.0e-15 | s12 |
| mode incidence Σ\|S\|² + P_rad + P_abs = 1 | ≤ 4.4e-16 | s12 |
| reciprocity t_{n←m}(L→R) = t_{m←n}(R→L) | ≤ 1.1e-16 | s12 |
| in-plane + particle balance (∇u_s by FD, h = 0.25 nm) | ≤ 2e-11, FD-limited (h⁴) | s12b |
| Born limit, a = 2 nm | 5.0e-4 (SOI), 1.2e-3 (SiN); ratio per halving 3.36–3.66 | s10 |
| interface continuity of the in-layer field | FD-limited 6e-7 (e = 1e-6 nm); z_top layer vs cover 3e-15 | nearfield |

## 4. Findings that shape the code

1. **The naive transfer matrix overflows; it does not lose digits.** The test
   that fails on it is R on the SOI path at a 5 nm gap (finite, matching the
   reference, by the recursion; NaN by the transfer matrix).
2. **Near-cutoff modes need the γ variable**, both for finding them and in the
   test reference (`g_ref_gamma`): in the cosh variable the reference swings by
   O(1).
3. **Close pairs vanish in pairs from any bracketing grid.** Only the count
   catches them, hence §2.3's check and R15.
4. **The forward T = aR + b cancels catastrophically on the tail.** The test
   reference for G_t shoots upward from the bottom (`T_ref` in s7).
5. **At Q = 3 with D = 1500 nm the hump floor overrides the 6/D cap** and the path
   loses digits. That is v0.9's open item; `SommerfeldPrecisionWarning` already
   reports it. No new handling.

## 5. Gates (each a test; bar = 10× measured, the number in the comment)

| Gate | Check | Independent because |
|---|---|---|
| GW1 | Every `HalfSpace` and no-background output bit-identical to v1.1 | regression |
| GW2 | T6 reductions: a stack of layers equal to the bottom ≡ `HalfSpace(ε_bot, z_top)`; all equal to the cover ≡ `HalfSpace(ε_bot, z_bot)`; everything equal to the cover ≡ no background. Bitwise, at the solve level | regression (A1) |
| GW3 | R by recursion vs the scaled transfer matrix at real q; energy at propagating q; large-q asymptote; R finite on the SOI gap-5 path where the naive form is NaN | second implementation; energy |
| GW4 | Reflected G vs `reference.sommerfeld.g_ref_stack` (PV fold + iπ·Res), 5 stacks, TE/TM; near cutoff t = 1e-6 vs `g_ref_stack_gamma`; lossy core vs plain QUADPACK | A3: second quadrature, different R and branch rule |
| GW5 | Transmitted G below the stack vs `gt_ref_stack` (shooting T) | second quadrature, different T |
| GW6 | Modes: roots vs the three-layer closed form; counts (q-grid, γ-grid, winding) on the double core; t = 1e-8 found; `NearDegenerateModesError` for Si/3 µm SiO₂/Si | A2: closed form |
| GW7 | Res_β R = γ₁u(z_top)²/(βN) | A2: independent integral |
| GW8 | Complex k / lossy: continued poles satisfy D = 0; path unchanged under δ × 1.5 at Q = 10 | consistency (no external anchor; v0.9 G3 style) |
| GW9 | Dipole power balance, no particle, scalar and in-plane | A6 at Green level |
| GW10 | Far-x field (path) → closed-form guided sum, both directions, scalar and in-plane | A4 |
| GW11 | In-layer field: continuity of u and p∂z u across each interface, cover match at z_top, core field at 30 µm vs the guided sum | interface conditions; closed form |
| GW12 | Dipole + particle power balance (`ModeAmplitudes.balance()`), scalar and in-plane, circle and lossy rotated triangle | A6 |
| GW13 | Mode incidence: Σ\|S\|² + P_rad + P_abs = 1 | A6 |
| GW14 | Reciprocity on the lossy rotated triangle and on the 2-mode slab | A7 |
| GW15 | Born limit, TE: error at a = 2 nm and ratio per halving | A5 |
| GW16 | In-plane free-space far field `\|a\|² ∝ \|p_x cosθ − p_z sinθ\|²`; analytic ∇u_s vs a 5-point FD | closed form; FD |
| GW17 | `mode_sensitivity`: translation derivatives `i(β_m ∓ β_n)S_scat`, `d\|S\|²/dΔ = 0`; frozen-map refusal | exact (§2.9) |
| GW18 | Scale covariance with every thickness, `z_top`, position and λ scaled: S, powers invariant, fields invariant | conventions §9 (A8) |

## 6. Conventions entries (new §17, in the docs commit)

The `Stack` frame and the T6 normalisation; the interior Im α ≥ 0 sheet and the
division-free recursion; the mode normalisation (bilinear N, u(z_top) = 1,
power-normalised ũ); the meaning and sign of `s_plus`/`s_minus` and `x_ref`;
the powers and their units; the in-plane `b = (p_z, −p_x)` map and
`H_y = iω·u`; the guided-output conditions (T2, T8); `NearDegenerateModesError`;
and the polarisation naming in cross-section (pol = 2 is slab TE).

## 7. Left to do before this is builder-ready

Nothing technical. The owner reviews T1–T9; any change to them comes before
commit 1.

## 8. Builder contract (2026-10-25)

**Stop rule:** if a gate fails, a measurement contradicts this spec, a run
exceeds §8.9, or anything here is ambiguous, stop and report. Do not widen a
tolerance, change a formula, or pick an alternative design. §8.8 lists the only
pre-approved detours.

### 8.1 `layered.py`

    SommerfeldPath._build(cls, ks, eps_rel, pol, D, z_min, depth_scale, features=None)
    SommerfeldPath.for_features(cls, k, features, D, z_min, depth_scale=1.0) -> SommerfeldPath
    stack_r(q, k, eps_layers, d, eps_bot, pol) -> np.ndarray                       # §2.1
    stack_tau(q, k, eps_layers, d, eps_bot, pol) -> np.ndarray                     # §2.1
    stack_layer_factor(q, k, eps_layers, d, eps_bot, pol, s, j) -> np.ndarray      # W/α₁ of §2.2, layer j, depth s
    stack_dispersion(q, k, eps_layers, d, eps_bot, pol) -> np.ndarray              # scaled D, §2.3
    guided_poles(k, eps_layers, d, eps_bot, pol) -> tuple[np.ndarray, int]         # roots, count check
    continue_poles(k_from, k_to, eps_from, eps_to, d, pol, poles) -> np.ndarray     # §2.4
    stack_features(k, k_bot, poles) -> tuple[tuple[str, complex], ...]
    class StackReflection: pol, k, r: Callable, features, z_top, path, x_c
        build(cls, pol, k, stack_rel, f, g) -> StackReflection | None
        blocks(tgt, src); green(x, z, xs, zs); green_grad_src(x, z, xs, zs); plane_wave(angle, f, g)
    class GuidedMode: §1.2

- `_build` with `features=None` is bit-identical to today: `_singularities` as
  now. With features, it uses them as given and skips `_singularities`.
  `for_wavenumber` and `for_box` do not change.
- `stack_*` take background-relative values and `k` only. No wavelength, no
  `Material` (conventions §2). `stack_r` with an empty layer tuple is never
  called (T6 routes it to `fresnel_r`).
- `StackReflection.build` sizes the path as `Reflection.build` does:
  `z_min = 2·min(g − z_top)` over every node, source and observation point
  served, and `D = max(ptp f, z_min)`. `x_c` is the midpoint of the f range. The
  features are `stack_features(k, k_bot, poles)`, with `poles` from
  `guided_poles` at `Re k` with `Re ε`, continued by `continue_poles` when
  `Im k ≠ 0` or any `Im ε ≠ 0`. It returns `None` only via T6 (no layer left).
  Paths for observation points **inside or below** the stack use
  `z_min = min h′` over the sources, plus the smallest depth below `z_bot` for
  bottom points (v1.1 finding 3). A layer point just under `z_top` adds no decay.
- `StackReflection.blocks` is `_reflected_blocks`' non-derivative GEMM with
  `fresnel_r` replaced by `self.r` (banding as is). There is **no**
  `blocks_dk` (T5).
- `StackReflection.plane_wave` uses `stack_r` at the one real q.
- `green_grad_src` returns ∂x′, ∂z′ of `G_ind`, for the in-plane RHS: path
  factors `−iq`, `+iα₁`.

### 8.2 `fields.py`

    _upward_amplitude(..., *, r: Callable | None = None, split: tuple[float, ...] = ())   # stack R; None = today's
    _stack_spectra(q, a1, boundaries, z_top, source=None) -> (u_up, d)                  # §2.7
    _stack_powers(boundaries, ctx, source=None) -> (p_up, p_down, p_abs)                # §2.7
    _mode_overlap(boundaries, modes, k) -> (c_plus, c_minus)                            # §2.6
    _representation_gradient_at(ei_p, nn, f, df, g, dg, delt, wnum, x, z) -> (ux, uz)   # analytic ∇ of _representation_at
    _layer_field(points, boundaries, ctx) -> np.ndarray                                 # §2.2 in-layer, banded GEMM

- `_upward_amplitude` with `r=None` is today's function, bit for bit.
- `_representation_gradient_at` differentiates `_representation_at`'s kernel in
  the observer coordinates (H₀′ = −H₁, (H₁(u)/u)′ = −H₂(u)/u). The reflected part
  of ∇u_s uses the path with target factors `iq`, `iα₁`.
- `_layer_field` is `nearfield.field_map`'s layer branch, as blocks against
  the particles' boundary data. For a point in layer j:
  `u(x, z) = (1/4π) ∫_Γ i·(W/α₁)(q, z)·d(q)·e^{iqx} dq` over the **full** path
  (q and −q: `d(q)` is not even). `d(q)` is §2.7's downward boundary sum with
  α₁ on the path's sheet. As a GEMM: target factor `(W/α₁)(q, z_i)e^{iq(x_i − x_c)}`,
  source factor `e^{−iq(x_j − x_c) + iα₁h_j}[(−iq dg_j − iα₁df_j)φ_j + χ_j]`, with
  `−h` and the path weights in the middle.

### 8.3 `solver.py`, `cluster.py`, `green.py`, `qnm.py`

- `BIESolver`/`ClusterBIESolver` accept `Stack`. They dispatch on
  `isinstance(bg, Stack)` in one helper per class (`_reflection` returns a
  `StackReflection`). `check_interface_gap` runs against `z_top`.
- `scatter_mode`: lossless and real λ (T2), else `ValueError`. `mode` as an
  int indexes `guided_modes`. `direction ∈ {+1, −1}`. RHS φ-half
  `ũ_m(z_j)e^{±iβ(x_j − x_ref)}`, χ-half 0, and no reflected term (the mode
  solves the background). The result records `excitation="mode"`, the mode,
  the direction and `x_ref`.
- `scatter_dipole(..., moment=)`: for pol = 1 the RHS φ-half is
  `−(b·∇′)(G₀ + G_ind)` at the nodes (∂′G₀ analytic as in s11; `G_ind` from
  `green_grad_src`). `moment=None` is today's code path, bit for bit. pol = 2
  with a moment raises (T7). The result records `moment`.
- `mode_amplitudes(x_ref)`: §2.6–§2.7. For `excitation="mode"`, an `x_ref`
  different from the incidence's raises `ValueError`. For `"custom"` it raises.
- `cross_sections()` with a `Stack`: `{'c_sca_up', 'c_sca_down', 'c_abs',
  'c_guided'}` (the last omitted per T8). `efficiencies()` adds `qguided`.
- `incident_field`: plane wave (stack R above, W in the layers, τ below), dipole
  (G₀ + G_ind, G_l, G_t), mode (ũ e^{±iβ(x − x_ref)} everywhere). It raises on any
  interface plane.
- `eval_field`: cover (v0.9 path with `StackReflection`), layers
  (`_layer_field`), bottom (`Transmission` with the stack's τ). It raises on
  any interface plane.
- `QNMSolver` with a `Stack` raises `ValueError` (T5).
- `relative_ldos`, `relative_ldos_map`, `self_green` work unchanged through
  `_reflection`.

### 8.4 Behaviour table (with a `Stack`)

| API | Behaviour |
|---|---|
| `Stack(...)` | `ValueError`: no layers, thickness ≤ 0, Im ε < 0, Re ε ≤ 0 (constants) |
| `guided_modes(λ, pol, n_clad)` | tuple ordered by β; `ValueError` at complex λ or a lossy stack; `NearDegenerateModesError` (§2.3) |
| `scatter`, `scatter_dipole` | any λ (complex accepted, poles continued); fields and LDOS everywhere off the interfaces |
| `scatter_mode` | real λ, lossless stack; else `ValueError` |
| `mode_amplitudes(x_ref)` | real λ, lossless stack, excitation ≠ custom; else `ValueError` |
| `cross_sections()` | as §8.3; `c_guided` absent per T8 |
| `far_field()` | upward amplitude with stack R |
| `far_field_down()` | `a_down` with the stack's τ and k_bot; `NoFarFieldError` rules on ε_bot |
| `QNMSolver(background=Stack)` | `ValueError` |

### 8.5 Exports and the API baseline

Exported: `Stack`, `GuidedMode`, `ModeAmplitudes`, `NearDegenerateModesError`,
`mode_sensitivity`. `StackReflection` stays internal. `tests/api_baseline.txt`
is regenerated in commit 6. Every change is an addition, so there is no
`BREAKING CHANGE` footer.

### 8.6 Test-side references and tolerances

`reference/sommerfeld.py` gains `g_ref_stack` (scaled-transfer-matrix R,
radiation-condition branch, θ/cosh, PV fold), `g_ref_stack_gamma` (γ variable
near k_out), `gt_ref_stack` (shooting T), and `slab3_roots` (closed form),
ported from `sproto.R_ref`, `g_ref`, `g_ref_gamma`, `s7.T_ref`, `gt_ref` and
`slab3_closed_form`. Real k only.

| Gate | Setup | Bar |
|---|---|---|
| GW1 | existing suite with `HalfSpace` / no background | bit-identical |
| GW2 | the three reductions, plane wave and dipole | bit-identical |
| GW3 | SOI, double core, air-bottom; q grid of s1 | recursion vs ref 1e-13 (1e-14); energy 2e-14 (2.0e-15); finite on the gap-5 path |
| GW4 | s3 stacks and points | 4e-9 (4e-10, reference floor); near cutoff t = 1e-6: 8e-9 (8e-10); lossy: 5e-13 (5e-14) |
| GW5 | s7 stacks and points | 4e-10 (4e-11) |
| GW6 | s2 slabs; double core; s4 t = 1e-8; Si/3 µm/Si | roots 5e-15 (4.8e-16); counts exact; error raised |
| GW7 | s8 stacks | 4e-13 (4e-14) |
| GW8 | s6 cases | D residual 1e-14 (3e-16); δ × 1.5 at Q = 10: 4e-12 (4e-13) |
| GW9 | s9 / s11 cases | 5e-14 (4.8e-15) |
| GW10 | s9a / s11a at 40 µm | remainder < 0.1·Σ\|c u\| and smaller than at 10 µm (measured ≤ 5e-2); a wrong sign gives ≥ 1 |
| GW11 | three stacks; points at z_i ± 1e-9 nm, analytic | continuity measure-then-10×, stop above 1e-10; z_top 3e-14 (3e-15); core at 30 µm < 0.05 (1.6e-2) |
| GW12 | s12 / s12b cases | scalar 1e-14 (1.0e-15); in-plane measure-then-10×, stop above 1e-11 |
| GW13 | s12 | 5e-15 (4.4e-16) |
| GW14 | s12 | 2e-15 (1.1e-16) |
| GW15 | s10, a = 4, 2 nm | error at 2 nm ≤ 1e-3 (SOI 5.0e-4) and ≤ 2.4e-3 (SiN 1.2e-3); ratio per halving in [3.2, 4.2] (3.36–3.66) |
| GW16 | free circle grid of 64 angles; ∇u_s vs FD h = 0.25 nm | pattern measure-then-10×, stop above 1e-12; gradient ≤ 10× the FD floor measured in the commit |
| GW17 | SOI, circle, Δ along x | measure-then-10×, stop above 1e-8; d\|S\|²/dΔ likewise |
| GW18 | s = 1.7, SOI + circle, mode incidence and dipole | measure-then-10×, stop above 1e-12 |

### 8.7 Order of work (one commit each, conventional form)

1. `refactor: feature-list Sommerfeld path`: `_build(features=)`,
   `for_features`; GW1.
2. `feat: Stack background with a stable recursion`: `Stack`, `stack_r`,
   `stack_tau`, `StackReflection`, the T6 normalisation, the solver dispatch for
   cover-side fields, LDOS and the plane wave; GW2–GW4, GW8 (the continuation
   lands here, because complex k is accepted from the first commit:
   non-negotiable 1).
3. `feat: guided modes of a stack`: `guided_poles`, the count check,
   `GuidedMode`, `guided_modes`, `NearDegenerateModesError`; GW6, GW7.
4. `feat: field below and inside the stack`: `Transmission` with the stack,
   `_layer_field`, `incident_field`, `far_field_down`, `c_sca_down`; GW5, GW11.
5. `feat: mode source and mode amplitudes`: `scatter_mode`, `mode_amplitudes`,
   `ModeAmplitudes`, `c_guided`; GW9 (scalar), GW10, GW12 (scalar), GW13–GW15.
6. `feat: in-plane dipole sources`: `moment=`,
   `_representation_gradient_at`, the in-plane source power; GW9, GW12 and
   GW16 (in-plane); API baseline.
7. `feat: mode sensitivity helper`: `mode_sensitivity`, the frozen-map check
   shared with `QNMResult.sensitivity` (factored, not copied); GW17.
8. `test: scale covariance over a stack`: GW18.
9. `docs: slab waveguide`: conventions §17 (§6 list),
   `examples/waveguide_antenna.py` (R12 per T4: Gielis antenna, scalar source,
   pol = 2, maximise `efficiency(0, +1)` with `scipy.optimize.minimize`
   (L-BFGS-B) on `mode_sensitivity` gradients, bounds: gap ≥ 10 nm, Gielis
   closure caught as a penalty; `--tm-inplane` for the spin-locked variant),
   README feature line, docs/validation.md rows GW1–GW18, CLAUDE.md roadmap
   status.

Each commit passes the full suite. Run each new gate's measurement once before
setting its bar, and quote the number.

### 8.8 Pre-approved detours

- GW4 stalls above its bar on one stack only: rerun the reference with
  `hw_frac = 0.2`. If the two references disagree by more than the bar, the
  reference is the floor: report it and stop.
- GW6's count disagrees on a case S2 passed: raise the winding `n` ×4 once,
  then stop.
- GW12 in-plane closes only to the FD floor: the analytic gradient is wrong.
  Compare it with FD component by component, then stop.
- Nothing else.

### 8.9 Budget

Each measurement run is under 3 min, and under 60 min in total. The new CI tests
stay around 60 s in total; measure them in commit 9 and report the time. The
example is not in CI.
