# Beyn QNM extraction → pysie2d: implementation spec (draft 2)

Supersedes `beyn-port-strategy.md` (high level). Consolidates the `research-port`
and `em-bie-theorist` proposals, with the strategy doc's errors corrected against
measurement. Every number below marked *(measured)* was computed against the real
`BIESolver.assemble`, not recalled.

Status: **round-2 answered (§13); Phases 1–3 landed and the go/no-go is GO.**
Suite green at **73 passed, 36.1 s** (baseline was 39/22.5 s, so the QNM feature
costs ~14 s — well inside the +60–90 s §13 Q5 accepted).

- **Phase 1 (§3) is in `reference/mie.py` and `tests/test_mie_qnm.py`** — 9
  tests, 8.6 s. Both independent checks hold in CI: the winding number agrees
  with the root count for every `n = 0…14` in both polarisations, and
  `D^TM_0 ≡ D^TE_1` to `1.4e-15`.
- **Phase 2 (§4) is in `beyn.py` and `tests/test_beyn.py`** — 18 tests, 0.61 s.
  Four corrections to §4 came out of it, all measured: the `Σ₁⁻¹` side, the
  rank rule (last gap, not largest), an unreachable saturation check replaced
  by a cancellation diagnostic, and the Cauchy tolerance. See §4.1.
- **Phase 3 (§5) is in `qnm.py` and `tests/test_qnm.py`** — 16 tests, 24.9 s,
  `QNMSolver`/`QNMResult` exported. **GO**: both anchors reproduce at first order
  in `n_pts`, degeneracy counted correctly, in both polarisations. Three more
  corrections came out of it (§5.1a–c), and §8's anisotropy claim does not
  survive at m=3.0 (§5.0).
- **Phase 0 is unblocked** (Q1 answered: `epsi` is absolute) and not started. It
  did not block Phase 3: the D2 anchor runs at `n_clad = 1`, where the vacuum and
  medium readings coincide. `docs/conventions.md` §8 has landed with the QNM half
  of its text and a note flagging the wavelength half as v0.4.0 work.
- **Phases 4–5 (§6)** are next: analytic `dM/dλ`, then `QNMResult.refine()` and
  the `converged` field, which are deliberately absent today (§5.3).

---

## 1. Decisions locked

| # | Decision | Consequence |
|---|---|---|
| D1 | Public `wavelength` means **vacuum** nm, with `n_core` / `n_clad` independent | §2 — a bug fix, not a redefinition |
| D2 | QNM anchor fixture: new circle at `n_core=3.0`, `rad=200` | §3 — separate from the driven-solver fixture |
| D3 | Refinement ships **opt-in** as `QNMResult.refine()` | §6 — not on the accuracy path |
| D4 | Mode fields **deferred to v0.5** | `vectors` exposed raw; no normalisation shipped |
| D5 | `Material.epsi` is **absolute**: `eps_rel = (n_core² + i·epsi)/n_clad²` | §2.3 — ships with D1, plus a lossy `n_clad≠1` test |
| D6 | **v0.4.0 = convention fix alone; v0.5.0 = QNM** | §13 Q2 — the breaking change is not buried in a feature release |

Plus, from the two agents' independent agreement (§2.1, §5.1 for evidence):

- **`Im λ > 0`** for decaying modes, and `Q = + Re λ / (2 Im λ)`. The research
  code's sign is wrong relative to the operator it calls. Do not port it.
- Search region is a **rectangle in complex λ (vacuum nm)**; complex ω is a
  documented conversion, not a second entry point.
- Rank detection exposes **one** knob. `n_expected` and `sv_gap_factor` do not
  ship.

---

## 2. Phase 0 — conventions

### 2.1 The wavelength fix is a bug fix

The package already *documents* vacuum everywhere and *implements* medium
everywhere. Only the implementation is out of step:

| Says vacuum | Implements medium |
|---|---|
| `solver.py:25,46,161,194,229` "Free-space wavelength (nm)" | `solver.py:47,113,168` `2π/wavelength` |
| `kernels.py:138,267` "Free-space wavenumber 2π/λ" | `green.py:135`, `fields.py:43` |
| `docs/conventions.md:25` "The free-space wavenumber is…" | `sources.py:44,130` |
| `tests/conftest.py:20-21` `x = 2π·n_clad·a/λ` — **already correct for vacuum** | |

`Material` already carries `n_core` and `n_clad` as independent fields
(`material.py:26-27`); only `epsr` collapses them to a ratio (`:34`). So D1 needs
no new state — it needs `n_clad` to reach the wavenumber.

This reframes the change: pysie2d is not adopting a new convention, it is being
made to honour its own documented contract. `conftest.size_parameter` is evidence
the vacuum reading was always the intent.

**Why CI never caught it:** every test runs at `N_CLAD = 1.0`, where the two
readings coincide. Nothing currently guards this. *(measured: at `λ=600`,
`nn=300`, `n_clad=1.3`, `n_core=1.5`, the medium reading matches Mie to `3.08e-3`
and the vacuum reading to `3.79e-1`.)*

### 2.2 Where the conversion goes

Seven sites compute `k` from a wavelength: `solver.py:47,113,168`, `green.py:135`,
`fields.py:43`, `sources.py:44,130`. Threading `n_clad` into all seven spreads the
background index through modules that have no business knowing it.

**Convert once at each public boundary; leave internals non-dimensionalised.**
The internal design — background index 1, only the *relative* index `m` and
relative permittivity entering the operator — is correct and worth keeping. Public
entry points take `λ_vac` and immediately form `λ_med = λ_vac / n_clad`.
Module-level helpers keep their present meaning and gain a docstring saying so.

Deliverables:
- `λ_vac → λ_med` at each public entry point taking a wavelength.
- Docstring sweep: `solver.py:25,46,161,194,229`, `green.py:37,67,110`,
  `kernels.py:138,267` — distinguish public (vacuum) from internal (medium).
- `tests/test_efficiencies.py:29`: `m = complex(mat.nc) / N_CLAD` is a genuine
  double-count under either convention (`mat.nc` is *already* `n_core/n_clad`).
  → `m = complex(mat.nc)`.
- `tests/conftest.py:19-21`: unchanged — already right.
- `docs/conventions.md` §8 (new), text in §2.4.

### 2.3 `Material.epsi` — open, see §13 Q1

Documented as "Imaginary part of the particle permittivity" (`material.py:16`),
but `eps = complex(self.epsr, self.epsi)` (`:48`) places it alongside a *relative*
real part, so today it is relative to the background. Under D1 the physical
reading is absolute: `eps_rel = (n_core² + i·epsi) / n_clad²`, i.e. the imaginary
part divides by `n_clad²`. Invisible at `n_clad=1`; the only lossy test runs there.

### 2.4 `docs/conventions.md` §8 (draft text)

> **8. Wavelength, background index, and the QNM half-plane.**
>
> Public API wavelengths are **vacuum** wavelengths in nm. `Material.n_core` and
> `Material.n_clad` are independent physical indices. Internally the background is
> non-dimensionalised to index 1: the operator sees `k = 2π·n_clad/λ_vac` and only
> the relative index `m = n_core/n_clad` (`Material.nc`) and relative permittivity
> (`Material.eps`).
>
> Search regions are rectangles in **complex λ (vacuum nm)**, matching the driven
> API's argument. Complex frequency is a documented conversion, `ω = 2πc/λ`, not a
> second entry point.
>
> Under `exp(-iωt)` (§3) a decaying mode has `Im ω < 0`, hence `Im k < 0`, hence
> **`Im λ > 0`**. A QNM search box must lie in `Im λ > 0` and strictly in
> `Re λ > 0` — the latter keeps every Hankel argument off the `H^{(1)}` branch cut
> on the negative real axis. Assert both.
>
> `Q = Re λ / (2 Im λ)`, exactly equal to `−Re ω / (2 Im ω)`.
>
> Poles do **not** occur in conjugate pairs. The reality condition is `λ → −λ̄`,
> placing mirror partners at negative `Re λ`, outside the physical region.

### 2.5 Exit criteria

- New `tests/test_conventions.py::test_vacuum_wavelength_scaling`: `qsca` at
  `(n_clad=1.3, n_core=1.95)` equals `(n_clad=1.0, n_core=1.5)` to `1e-12`
  *(measured: identical to every printed digit — pure scale invariance in `m`)*.
- `test_efficiencies_match_mie` parametrised over `n_clad ∈ {1.0, 1.33}`, passing
  at the **unchanged** `RTOL_MIE = 5e-3`.
- Existing suite green and numerically unchanged (all of it runs at `n_clad=1.0`).

---

## 3. Phase 1 — the analytic anchor

Fixture (D2): circle, `rad=200`, `n_core=3.0`, `n_clad=1.0`, in `conftest.py` as
`QNM_N_CORE`, `qnm_circle`. Kept separate from the `N_CORE=1.5` driven fixture,
whose modes are `Q ≈ 2.4` and overlapping.

Analytic roots *(measured — the **complete** set in the size-parameter box
`Re x ∈ [1, 3]`, `Im x ∈ [−1.5, −1e−5]`, root count confirmed against the winding
number for every `n = 0…14`; residuals `|D| = 8e−17 … 6e−15`)*:

| pol | n | λ (nm) | Q | mult |
|---|---|---|---|---|
| 1 (TM) | 3 | `399.73006 + 141.62663j` | 1.41 | 2 |
| 2 (TE) | 6 | `435.68857 + 0.09518j` | 2288.87 | 2 |
| 2 (TE) | 1 | `437.72991 + 17.03303j` | 12.85 | 2 |
| 1 (TM) | 0 | `437.72991 + 17.03303j` | 12.85 | **1** |
| 1 (TM) | 5 | `442.59141 + 0.38838j` | 569.79 | 2 |
| 1 (TM) | 2 | `454.40719 + 22.89506j` | 9.92 | 2 |
| 2 (TE) | 3 | `459.76308 + 9.92137j` | 23.17 | 2 |
| 2 (TE) | 5 | `505.67640 + 0.42300j` | 597.73 | 2 |
| 1 (TM) | 4 | `516.80877 + 1.84124j` | 140.34 | 2 |
| 2 (TE) | 0 | `530.83214 + 26.37850j` | 10.06 | **1** |
| 1 (TM) | 1 | `542.68825 + 27.64808j` | 9.81 | 2 |
| 2 (TE) | 2 | `550.46889 + 20.23579j` | 13.60 | 2 |
| 1 (TM) | 2 | `568.69137 + 296.39277j` | 0.96 | 2 |
| 2 (TE) | 4 | `605.52336 + 1.84489j` | 164.11 | 2 |
| 1 (TM) | 3 | `624.78707 + 8.67542j` | 36.01 | 2 |
| 2 (TE) | 1 | `690.51371 + 40.64175j` | 8.50 | 2 |
| 1 (TM) | 0 | `690.51371 + 40.64175j` | 8.50 | **1** |
| 2 (TE) | 3 | `760.68665 + 7.94771j` | 47.86 | 2 |
| 1 (TM) | 2 | `787.37060 + 37.22944j` | 10.57 | 2 |
| 2 (TE) | 0 | `946.15199 + 86.35263j` | 5.48 | **1** |
| 1 (TM) | 1 | `1029.92140 + 89.89149j` | 5.73 | 2 |
| 2 (TE) | 2 | `1035.09289 + 35.26143j` | 14.68 | 2 |

Anchors for the Phase-3 go/no-go: TE `n=0` at `530.83214 + 26.37850j` (simple)
and TE `n=3` at `760.68665 + 7.94771j` (degenerate). Both were in draft 1 and
both reproduce to every printed digit.

**Each order carries two radial branches, and TM `n=2` carries three.** Draft 1
listed one root per `(pol, n)` and that is wrong: within this box most orders
have two, and TM `n=2` has a third, very broad mode at `568.69 + 296.39j`
(`Q = 0.96`). The winding number independently returns 3 there, so it is real,
not a Newton artefact. Any test asserting a root *count* per order must use the
table above, not a rule of thumb.

**The box is a rectangle in `x`, not in `λ`.** `λ = 2π·rad/x` is a Möbius map, so
the completeness statement above transfers to a `λ`-rectangle only after the
corner mapping is checked. The public API searches `λ`-rectangles (§2.4); a test
box must be checked for completeness in its own coordinates.

**Degeneracy is structural, not a Gielis footnote:** every `n ≥ 1` mode of a
circle is doubly degenerate (`e^{±inθ}`). Only `n = 0` is simple, so `n=0` is the
Newton anchor and `n≥1` is the rank-2 test. The `D^{TM}_0 ≡ D^{TE}_1` identity
below makes TM `n=0` and TE `n=1` land on the *same two* wavelengths; this is not
a rank hazard, because `M(λ)` is assembled per polarisation and a search sees
only one of them.

`reference/mie.py` needs **no functional change** — `jv`, `jvp`, `hankel1`, `h1vp`
and `_nmax` (already `np.abs`) all accept complex argument. Annotation widening
only, at `:35,64,106,133,182`.

**As shipped** the signature is
`qnm_wavelengths(rad, m, pol, x_range, n_max=None, im_x_floor=1e-4, im_x_max=1.5)`
— a rectangle in **`x`**, not the `lam_re_range` drafted below. The paragraph
after this list is the reason: the completeness claim is established in `x`, and
a `λ`-rectangle is a different region under the Möbius map, so accepting `λ`
bounds here would have quietly returned a set whose completeness nothing had
checked. Two helpers are public alongside it: `qnm_denominator(n, x, m, pol)`
(the physics, reused by the winding-number check) and
`qnm_size_parameters(...)` (one order, roots in `x`).

Add `qnm_wavelengths(rad, m, pol, lam_re_range, n_max=None)`:
- Denominators verbatim from `mie.py:60,87` —
  `D^{TM}_n(x) = m J_n(mx) H_n'(x) − J_n'(mx) H_n(x)`;
  `D^{TE}_n(x) = J_n(mx) H_n'(x) − m J_n'(mx) H_n(x)`.
  `pol=1 → a_n → TM`, `pol=2 → b_n → TE` (`conventions.md:20-21`).
- Coarse complex grid seeds + `scipy.optimize.newton` (secant) on the **scale-free
  ratio form** — raw denominators have no detectable grid minima. Dedup at `1e-6`,
  keep `|D| < 1e-9`. Return `λ = 2π·rad/x`, multiplicity 1 for `n=0` else 2.
- **The seeding grid's `Im x` floor is a Q ceiling, and must be an argument, not a
  constant.** A grid bottoming out at `Im x = 1e−4` cannot see a mode with
  `Q ≳ 1.5e4`. It did not bite here — the winding number confirms nothing was
  missed — but it is close: the TE ladder runs `Q = 164, 598, 2289` for
  `n = 4, 5, 6`, i.e. `Im x` falling by ~3.4× per order, so two more orders (or a
  higher `m`) would cross the floor silently. Expose the floor and **assert the
  winding number rather than trusting the grid**; a completeness claim made by the
  seeder about itself is not a check.
- `n_max = ceil(m·x_max) + 5`; assert `D_{n_max}` has no root in the box
  *(measured: at `x_max = 3`, `n_max = 14`; roots die out at `n = 6` (TE) and
  `n = 5` (TM), so `n_max` clears the last root by eight orders)*.

Two independent correctness checks — neither reuses the root-finder:
- **Winding number** of `D_n` along the box boundary equals the root count found.
  `D_n` is holomorphic on the box (`Re x > 0` keeps every Hankel argument off the
  branch cut, §5.2), so it counts zeros with multiplicity. *(measured: agrees for
  all `n = 0…14`, both polarisations — the check that found the third TM `n=2`
  root.)*
- **`D^{TM}_0(x,m) ≡ D^{TE}_1(x,m)`** identically *(measured: max relative
  difference `1.43e-15` over 200 random complex points)* — the 2-D TE₀₁/TM₁₁
  degeneracy. One assert, catches a swapped-polarisation port.

Exit *(all met)*: roots satisfy `|D| < 1e-12` *(measured `≤ 6e−15`)*; all have
`Im λ > 0`; winding numbers agree; the identity holds; TE `n=3` reproduces
`760.68665 + 7.94771j` to `1e-6` nm.

---

## 4. Phase 2 — Beyn on synthetic pencils — **EXECUTED**

**New phase, absent from the strategy doc**, which jumps straight to the circle
and so conflates an algorithm bug with a convention bug.

Shipped as `src/pysie2d/beyn.py` + `tests/test_beyn.py`: **18 tests, 0.61 s**
(the ~10 s estimate was pessimistic — the pencils are 40×40). Not yet exported
from `__init__.py`; the public surface is `QNMSolver`, which Phase 3 defines.

`beyn.py` is EM-free and callable-driven, so it is testable against pencils with
known spectra. Rectangular Gauss–Legendre contour;
`A₀ = Σ wⱼ M(λⱼ)⁻¹V`, `A₁ = Σ wⱼ λⱼ M(λⱼ)⁻¹V`; SVD of `A₀`; reduced
`B = U₁ᴴ A₁ V₁ Σ₁⁻¹`; `eig(B)`; in-contour filter.

Pencils: `M(λ) = X·diag(fᵢ(λ))·Y` with `X, Y` unitary and `fᵢ` of known roots;
`fᵢ = λ − rᵢ` (linear) or `exp(λ − rᵢ) − 1` (nonlinear). Residue strength is set
by scaling `fᵢ`, which is how the weak-pole cases are built.

### 4.1 Four corrections to this section, all measured

**(a) `Σ₁⁻¹` post-multiplies.** `B = U₁ᴴ A₁ V₁ Σ₁⁻¹` (Beyn Alg. 3.1 step 5) —
column `j` divided by `s₁[j]`. The draft's left-multiplication gives a matrix
similar through `Σ₁`, so the *eigenvalues* survive the error and the
*eigenvectors* do not. A test on λ alone would never have caught it.

**(b) Rank rule: the LAST qualifying gap, not the largest.** Take the last `j`
with `s_j/s_{j+1} ≥ 1e3` and `s_j/s_0 > rank_tol`; rank `= j+1`. Everything past
the final cliff is the noise floor — that is what a floor *is* — whereas a drop
between two genuine poles is not. *(measured: two poles with a 10⁶ residue ratio
give gaps `1.1e6` then `2.1e5`, so `argmax` returns rank 1 and silently loses
the weak pole. The draft's rule fails its own degenerate row too: gaps
`1.9e7 > 9.6e4` selects rank 4 where the answer is 2 — which is why that row's
"max gap" column was left blank.)*

**(c) A flat spectrum is ambiguous, and `rank ≥ n_probe` can never fire.** The
rank is at most `p−1` by construction (there are only `p−1` gaps), so the
drafted saturation check is unreachable code. Worse, a probe saturated by more
poles than columns produces *the same flat spectrum as an empty box*, so the
natural fallback — report no modes — is a confident wrong answer.

The discriminator is absolute, not relative: over an empty contour the
integrand is analytic and `A₀` cancels to nothing, while a saturated probe
returns a perfectly large `A₀`. `contour_moments` therefore also returns

    cancellation = ‖A₀‖ / Σⱼ |wⱼ| ‖xⱼ‖   (≤ 1 by the triangle inequality)

*(measured: `2.9e-16` empty, versus `0.58–0.63` for every case containing poles,
saturated or not — fifteen orders of separation, so `EMPTY_CANCELLATION = 1e-10`
is not a tuned threshold.)* Flat spectrum + large `A₀` ⟹ `ValueError`.
`cancellation` is a fourth diagnostic alongside §5.1's three.

**(d) The Cauchy test holds at `1e-13` for 24 nodes/side, not for the default
12.** Gauss-Legendre on `1/(λ−z)` converges as `ρ^(−2n)` with `ρ` set by the
Bernstein ellipse through `z`, so the tolerance is a function of how far the
pole sits from the contour — the same effect as `edge_margin` (§5.1). *(measured
for points ≥ 0.6 from every edge: `4.1e-7` at 12 nodes/side, `1.3e-13` at 24;
for a point 0.1 from a corner, `2.6e-3` at 12 and still `5.7e-5` at 24.)*
The suite asserts geometric convergence (ratio > `1e4` from 12 to 24) rather
than resting on one number.

**Consequence for Phase 3:** the default `n_quad_per_side = 12` is worth ~`4e-7`
in the *contour integral*, which is far below the first-order discretisation
error in `nn` (§8) — so it stays the default, but a box drawn tight against a
pole loses that margin fast.

### 4.2 Rank signatures as measured on the pencils

At `n_quad_per_side ∈ {12, 24}`, `N = 40`, `p = 12`, `rank_tol = 1e-8`; the rule
in (b) returns the right rank in all eight cases at both resolutions.

| case | singular-value ratios | rank |
|---|---|---|
| empty box | `1.0, 0.59, 0.47, 0.44, …` | 0 |
| one simple pole | `1.0, 2.0e-16, …` | 1 |
| two distinct poles | `1.0, 0.75, 2.3e-16, …` | 2 |
| degenerate pair | `1.0, 0.75, 2.8e-16, …` | 2 |
| near-degenerate (`1e-6` apart) | `1.0, 0.75, 2.4e-16, …` | 2 |
| residues `1e4` apart | `1.0, 9.1e-5, 5.3e-14, …` | 2 |
| residues `1e6` apart | `1.0, 9.1e-7, 4.3e-12, …` | 2 |
| nonlinear, two poles | `1.0, 0.75, 2.0e-13, …` | 2 |

### 4.3 A structural fact that shapes the refinement test

**A linear pencil is quadrature-exact at any node count.** For a simple pole
with rank-1 residue, `A₁`'s integrand is `λ/(λ−λ₀) = 1 + λ₀/(λ−λ₀)` and
Gauss-Legendre integrates the constant exactly, so `A₁ = λ₀·A₀` *identically*
and the quadrature error cancels out of the ratio *(measured: `2.3e-15` at two
nodes per side)*. A linear pencil therefore cannot produce a coarse estimate to
refine, and `test_newton_refine_converges_on_synthetic` must use the nonlinear
pencil, where 3 nodes/side leaves `1.4e-7` of error for Newton to close.

### 4.4 Tests as shipped (18)

Quadrature: `test_contour_weights_reproduce_cauchy`,
`test_contour_quadrature_converges_geometrically`,
`test_contour_is_counter_clockwise`, `test_degenerate_rectangle_is_rejected`.

Extraction: `test_beyn_recovers_synthetic_eigenvalues[linear, nonlinear]`
(`1e-10`), `test_recovered_vectors_are_null_vectors`,
`test_eigenvalues_outside_the_contour_are_filtered`,
`test_beyn_empty_contour_returns_no_modes`, `test_beyn_is_seed_independent`,
`test_linear_pencil_is_quadrature_exact`.

Rank: `test_degenerate_eigenvalue_has_rank_two`,
`test_rank_survives_disparate_residues`,
`test_beyn_raises_when_rank_saturates_probe`,
`test_beyn_poles_requires_the_cancellation_diagnostic`.

Refinement: `test_newton_refine_converges_on_synthetic`,
`test_newton_refine_is_idempotent_at_convergence`,
`test_newton_refine_flags_degenerate_eigenvalue` (`cond(J) > 1e12`).

---

## 5. Phase 3 — extractor + circle anchor (go/no-go) — **GO**

Façade lands **here**, not in a final "public API" phase — otherwise later phases
are tested against private functions and rewritten.

Shipped as `src/pysie2d/qnm.py` + `tests/test_qnm.py`: **16 tests, 24.9 s**,
exported from `__init__.py`. `BIESolver.assemble` widened to `float | complex`;
`QNMSolver` calls **that very method** on its contour rather than a parallel
assembly path, so "a QNM is a singularity of the scattering operator" is true by
construction rather than by inspection.

### 5.0 The verdict, with numbers

Both anchors reproduce, at first order in `n_pts` — the same rate the near-field
quantities obey:

| anchor | analytic λ (nm) | found at `n_pts=200` | ΔRe | ΔIm | K | rank |
|---|---|---|---|---|---|---|
| TE `n=0` (simple) | `530.83214 + 26.37850j` | `530.4555 + 26.1384j` | −0.377 | −0.240 | 1 | 3 |
| TE `n=3` (degenerate) | `760.68665 + 7.94771j` | `760.3261 + 7.7702j` | −0.361 | −0.178 | 2 | 2 |
| TM `n=0` (simple) | `690.51371 + 40.64175j` | `690.1046 + 40.6822j` | −0.409 | +0.040 | 1 | 1 |

Convergence *(TE simple, `|Δλ|`)*: `0.903 → 0.598 → 0.447 → 0.297 → 0.223` at
`n_pts = 100, 150, 200, 300, 400`, i.e. **order 1.00–1.03 in both Re and Im**.

**§8's anisotropy claim does not survive at this contrast.** At m=1.5 the spec
measured `Re` off by 0.24 nm against `Im` by 0.017 nm and concluded "Q converges
far better than Re λ". At m=3.0 both are first order with `ΔIm/ΔRe ≈ 0.64`
*constant* across the whole sweep, and because `Im λ` is small (26 nm) its
*relative* error is the worse of the two: `7.1e-4` for `Re` against `9.1e-3` for
`Im`, so **Q converges worse than `Re λ`, not better** (0.87 % at `n_pts=200`).
Assert them separately, and give Q the looser bound.

`n_quad_per_side = 6` is used throughout the tests rather than the default 12:
*(measured: identical modes to 1e-8 nm from 6 nodes upward, against a 0.38 nm
discretisation error — the contour integral is nowhere near the bottleneck.)*
The default stays 12 because leakage (§5.1a) does depend on it.

### 5.1 Diagnostics (the doc's §8 usability risk)

Four fields on the result, each distinguishing a named failure mode. Measured
values below are at the D2 fixture (m=3.0, `n_pts=200`), replacing the m=1.5
numbers the draft carried:

### 5.1a Correction: rank exceeds the mode count, and the gap is not clean

§7 predicted a clean gap because "pysie2d assembles exactly, so the gap is
clean". Exact assembly is not the issue — **contour geometry is**. The degenerate
TE `n=2` pair at `550.47+20.24j` sits 5 nm *outside* the TE `n=0` box and leaks
two rank directions in through imperfect quadrature cancellation; their
eigenvalues land near `550.40` and are discarded by the in-contour filter. Result:
rank 3, one mode. The in-contour filter is doing real work, not tidying up.

Leakage is quadrature error, so it decays geometrically with nodes
*(measured `sv_ratio[1]` at `n_quad_per_side = 6, 8, 12, 16, 24`: `6.2e-4`,
`8.9e-5`, `1.8e-6`, `3.8e-8`, `1.6e-11` — and by 24 it falls below `rank_tol` and
the rank becomes 1, the correct value)*. So the clean-gap claim is true
*asymptotically*, just not at the default resolution. Consequences: `rank` is
documented as possibly exceeding `n_modes`, and `n_probe` must have headroom for
leaked directions, not just for real modes.

### 5.1b Correction: `EMPTY_CANCELLATION` was calibrated on pencils that are too clean

Phase 2 set `EMPTY_CANCELLATION = 1e-10` from synthetic pencils, which cancel to
`2.9e-16`. Physical contours do not: an empty box near poles at a coarse contour
reaches `5.2e-7`, and at `1e-10` that **raises a spurious "saturated probe"
error** on a genuinely empty box (reproduced at `Re λ ∈ [300,330]`,
`n_quad_per_side=6`).

The two sides scale differently, which is what makes any threshold possible:
empty-box cancellation is quadrature error and falls with the node count, while
a residue inside the contour does not move.

| box | cancellation at 24 nodes | at 48 nodes |
|---|---|---|
| empty, far from poles | `1.5e-13` | `6.0e-15` |
| empty, `Re λ ∈ [300,330]` | `8.4e-10` | `8.7e-14` |
| empty, poles 5–25 nm outside | `5.2e-7` | `1.1e-11` |
| **one mode inside** | **`3.09e-2`** | **`3.09e-2`** |
| **two modes inside** | **`4.90e-2`** | **`4.94e-2`** |

`EMPTY_CANCELLATION = 1e-4` sits ~200× above the worst empty case and ~300×
below the weakest populated one. Note the populated rows are resolution-
independent, as the argument predicts.

### 5.1c Correction: exception messages must be ASCII

`CLAUDE.md` welcomes Unicode in docstrings and comments, and that stands — but a
`ValueError` message containing `A₀` or `λ` raises `UnicodeEncodeError` when
printed to a cp1252 console, burying the real error. Found by hitting it. Error
strings use `Re(lambda)`, `A0`; docstrings keep the notation.

### 5.1d The diagnostics as shipped

1. `sv_ratio` + `max_gap` + `rank` — no gap ≥ `1e3` ⟹ **no meromorphic content
   in the box**. In a homogeneous background the warning can state outright that
   analyticity is not the explanation (§5.2). Read `rank` together with
   `n_modes`: they legitimately differ (§5.1a).
2. `edge_margin`, as a fraction of the shorter side — any λ within ~2 % of an
   edge ⟹ **contour too tight**, pole being clipped. *(The spurious lower-half-box
   result sat at `Im = −60.047` against an edge at `−60`. The shipped TE `n=0`
   box measures 42 %, i.e. comfortable on all four sides.)*
3. `sigma_ratio = σ_min/σ_max` per pole, threshold **`1e-6`** (not `1e-8`).
   **Never absolute `σ_min`** — the research `sigma_threshold=1e-12` is
   dimensionally meaningless: *(measured at m=3.0)* absolute `σ_min` at the
   *exact analytic* pole is `8.2e-4`, so that gate could never fire. Ratios at
   `n_pts=200`: extracted mode `1.5e-14` at 48 contour nodes and `8.2e-9` at 24,
   **analytic pole `4.3e-4`**, conjugate of the pole `7.2e-3`, generic points
   `2.4e-3 … 4.6e-3`.

   Two things to read off that list. First, the analytic pole scores no better
   than a generic point — it is not a pole *of the discrete operator*, and the
   gap between `4.3e-4` and `1.5e-14` is exactly the discretisation error made
   visible. Second, `sigma_ratio` is a ferociously sensitive proximity detector:
   the 24- and 48-node estimates differ only in the **eighth** decimal of λ
   (`530.4555010` vs `530.4555080`) yet their ratios differ by five decades.
   That sensitivity is why the threshold needs room: the draft's `1e-8` would
   have failed outright at 24 nodes. `1e-6` clears the worst extracted value by
   ~120× and sits ~400× below the nearest generic point.
4. `cancellation` — see §5.1b. Distinguishes an empty contour from a saturated
   probe, which produce the *same* flat singular spectrum. Without it the
   natural fallback is to report "no modes found", which is a confident wrong
   answer.

### 5.2 Analytic structure

`k = 2π·n_clad/λ`, every Hankel argument `k·r` or `nc·k·r` with `r > 0` real.
`H^{(1)}` has its only branch point at 0 and its cut on the negative real axis, so
arguments stay off the cut iff `arg λ ∈ (−π/2, π/2)`. Hence the `Re λ > 0`
assertion, and hence `M(λ)` is holomorphic on any admissible rectangle — a flat
SVD can never be blamed on analyticity here. (It can the moment a slab background
is added; that is one reason slab QNMs stay out of scope.)

### 5.3 Exit — all met

- `test_beyn_matches_analytic_pole_te` / `_tm` — tolerances measured first, then
  written: `ATOL_RE_NM = 0.5`, `ATOL_IM_NM = 0.32`, ~30 % above what `n_pts=200`
  delivers.
- `test_pole_error_is_first_order_in_resolution` — **this is what makes the two
  tolerances above falsifiable.** A pole that stagnated, or converged to
  something other than the anchor, would still pass a fixed tolerance if the
  tolerance were loose enough; asserting the *order* (error ratio in `1.74–2.46`
  over a doubling of `n_pts`) excludes both.
- `test_degenerate_pair_has_rank_two` — `n_modes == 2` and
  `multiplicity == [2,2]` for the `n=3` box, partners `2.4e-13` apart;
  `test_simple_pole_has_multiplicity_one` for the `n=0` box.
- `test_poles_are_upper_half_plane`; `test_search_region_rejects_lower_half_plane`,
  `..._rejects_negative_real_part`, `..._rejects_degenerate_rectangle`.
- `test_no_conjugate_pair_symmetry` — negative test; the mirror partner is at
  `−λ̄`, not `λ̄` *(Phase 1 verified `|D_0^{TE}(x̄₀)| = 3.03e-1`; here the BIE
  `sigma_ratio` at the conjugate is `7.2e-3`, no more singular than a generic
  point)*.
- Plus, beyond the drafted list: `test_quality_factor_matches_the_analytic_mode`
  (the 0.87 % that §5.0 corrects §8 with), `test_modes_are_seed_independent`,
  `test_modes_are_singular_and_generic_points_are_not`,
  `test_rank_may_exceed_mode_count_from_outside_leakage` (§5.1a),
  `test_empty_box_finds_nothing_and_says_why` (§5.1b),
  `test_edge_margin_reports_a_comfortable_box`.

**Deferred to Phase 4, deliberately:** `QNMResult.refine()` and the `converged`
field of §7. The bordered-Newton core already exists and is tested
(`beyn.newton_refine`), but it needs `dM/dλ`, which is §6.1's analytic
derivative. Shipping a finite-difference `refine()` now would be Phase 4 done
badly, and a `converged` field nothing can set is dead weight on a frozen
dataclass.

**Box placement, from the §3 table.** The two anchors are not equally forgiving.
TE `n=3` at `760.69 + 7.95j` is isolated — nearest TE neighbours are `690.51` and
`1035.09` — so almost any sane box works. TE `n=0` at `530.83 + 26.38j` is
crowded in `Re λ`: TE `n=5` sits at `505.68 + 0.42j` and TE `n=2` at
`550.47 + 20.24j`, only ~20 nm away. What separates them is `Im λ`, not `Re λ`,
so the box must be tight in `Im`: `Re λ ∈ [520, 545]`, `Im λ ∈ [15, 40]` isolates
it with `edge_margin` well clear on all four sides. A box drawn generously in
`Im` swallows the `Q = 598` mode and the rank-1 assertion fails for a reason that
has nothing to do with Beyn.

*(The shipped boxes are exactly these: `520+15j → 545+40j` for TE `n=0`,
`745+2j → 775+15j` for TE `n=3`, `675+30j → 705+50j` for TM `n=0`. The tight-in-Im
advice held — and §5.1a is the residue of the crowding it warned about, showing up
as leaked rank rather than as a failed assertion.)*

**This was the honest go/no-go, and it passed.** Beyn reproduces §3's roots in
both polarisations, at first order in `n_pts`, with the degeneracy counted
correctly. Phases 4–5 are now worth doing.

---

## 6. Phases 4–5 — analytic `dM/dλ`, refinement, `σ_min`

### 6.1 Derivative identities (verified against `kernels.py:285-338`)

With `z = k r` and `d/dk[k² H₁(z)/z] = k H₀(z)` exactly, `dk/dλ = −k/λ`:

```
dM1[i,j]/dk = ¼i·δⱼ·cᵢⱼ·k·H₀(kr)          (reuses h0w)
dM2[i,j]/dk = −¼i·δⱼ·k·r²·(H₁(kr)/(kr))   (reuses h1w)
dM3[i,j]/dk = nc·¼i·δⱼ·cᵢⱼ·k₁·H₀(k₁r)     (reuses h0w1)
dM4[i,j]/dk = −η·nc·¼i·δⱼ·k₁·r²·h1w1      (reuses h1w1)
d(d_m1)/dk  = d(d_m3)/dk = 0               (λ-independent, kernels.py:317,319)
d(d_m2)/dk  = −¼i·δ·ρ·H₁(kρ),  ρ = δ·γ/(2e)
d(d_m4)/dk  = −η·nc·¼i·δ·ρ·H₁(k₁ρ)
```

**Two corrections to strategy doc §5.** (a) "No additional special-function
evaluations" is true for the `O(nn²)` blocks but false on the diagonal: `d_m2`,
`d_m4` need `H₁` at the log-singularity arguments, which assembly never computes
(it takes only `H₀` there, `kernels.py:318,320`). That is `nn` extra scalar calls
out of `nn²` — hence the measured `dM/dλ` cost of **0.92–1.00× one assembly**.
(b) The saving is **3 assemblies → 2**, not → 1: `M` itself is still needed. So
~1.5×, not 3×. A fused `assemble_matrix_dwn` sharing `h0w/h1w/h0w1/h1w1` gives
~1.3× and is the recommended form.

Exactness depends on `Material` being **non-dispersive** (`material.py` takes no
λ). State that in the docstring; it is a real precondition, not a nicety.

Validation: `test_matrix_derivative_matches_assembly` — `np.array_equal` against
`assemble_matrix`, bit-identical, same expression order (guards the ~50 lines of
deliberate duplication of the v0.3 hot path). `test_matrix_derivative_matches_
central_difference` at real *and* complex `wn`, both polarisations — *(measured:
order exactly 2.00 from `h=1e-1` to `1e-3`, rel err `6.1e-11`, then round-off. At
the research default `h = 1e-4·|λ|`, FD rel err is `1.3e-7` — the doc's "~3e-7"
is the right order, and it sits above the `1e-8` convergence test it feeds.)*

### 6.2 Refinement is insurance, not accuracy (D3)

*(measured)* On the simple `n=0` pole at `nn=200`, bordered Newton converged in
**one step with `|Δλ| = 6.8e-14`**, landing on the Beyn value to 13 digits;
`cond(J) = 1.5e3`. The 0.37 nm residual against the analytic root is **100 %
discretization, 0 % extraction**. On a degenerate pole, `cond(J) = 2.1e15` — the
bordered Jacobian is numerically singular (2-D kernel); Newton no-ops gracefully
and buys nothing.

Consequences: `research-port`'s proposed Phase-5 exit criterion ("error drops ≥10×
versus the Beyn estimate") **cannot be met and must not be written**. Replace with:

- `test_refine_is_idempotent_at_convergence` — `|Δλ| < 1e-12` and
  `converged.all()` on the `n=0` anchor.
- `test_refine_recovers_from_coarse_contour` — with `n_quad_per_side=4`,
  refinement closes the gap to the Phase-3 value. *This* is the behaviour that
  justifies shipping it.
- `test_refine_flags_degenerate_pole` — `cond(J) > 1e12` detected, flagged,
  skipped, no exception.

### 6.3 `σ_min` by inverse iteration

*(measured)* 3 iterations of `y = M⁻¹v; z = M⁻ᴴy` seeded with the mode vector, on
the LU the Newton step already formed: `1.537212e-11` vs `1.537212e-11` from full
SVD (rel err `2.6e-7`) in **2–6 ms against 497 ms**.

Over-engineering to avoid: Lanczos/`svds`, adaptive iteration counts, or chasing
accuracy at generic points (there it converges slowly — 1.2 % after 10 iterations
— and it does not matter, given the `4.5e-12` vs `3.0e-3` contrast).

---

## 7. Public API

```python
class QNMSolver:
    def __init__(self, geometry: Geometry, material: Material) -> None: ...
    def modes(self, z_lo: complex, z_hi: complex, *,
              n_quad_per_side: int = 12, n_probe: int = 12,
              rank_tol: float = 1e-8, rng_seed: int = 0) -> QNMResult: ...

@dataclass(frozen=True)
class QNMResult:
    wavelengths: np.ndarray       # complex (K,), vacuum nm, sorted by Re λ
    vectors: np.ndarray           # complex (2*n_pts, K), unit columns, φ/χ layout
    multiplicity: np.ndarray      # int (K,), degenerate partners grouped
    sigma_ratio: np.ndarray       # float (K,)  σ_min/σ_max
    sv_ratio: np.ndarray          # float (n_probe,)
    max_gap: float
    edge_margin: np.ndarray       # float (K,)
    z_lo: complex
    z_hi: complex
    geometry: Geometry
    material: Material
    converged: np.ndarray | None = None   # None until refine()

    @property
    def quality_factors(self) -> np.ndarray:   # Re λ / (2 Im λ)
    @property
    def n_modes(self) -> int:
    def refine(self, *, tol: float = 1e-9, max_iter: int = 30) -> "QNMResult":
```

Mirrors `BIESolver` → `ScatterResult`: same `(geometry, material)` construction,
one method returning a result object, derived physics as properties, refinement
returning a *new* result. `frozen=True` follows from that.

Knobs: `rank_tol`, `n_probe`, `n_quad_per_side`, `rng_seed` public;
`n_expected` and `sv_gap_factor` **absent** — they exist in the research code to
suppress a tail produced by *approximate* surrogate assembly; pysie2d assembles
exactly, so the gap is clean. `rng_seed` is public so seed-independence is
checkable, not as a tuning knob. Rank saturating `n_probe` raises `ValueError`.

---

## 8. Tolerances — what must be measured before it is written

Strategy doc §8 said pole convergence order was unknown. It was measured **at
m=1.5**: exactly **first order** (1.00, 1.00, 1.00, 1.01 across `nn=100→450`;
rel err `1.58e-3 → 3.49e-4`), and **anisotropic** — at `nn=300`, `Re λ` off by
0.24 nm but `Im λ` by 0.017 nm, so `Q` converges far better than `Re λ`. Assert
them separately.

**Re-measure at D2's `n_core=3.0` fixture before writing any tolerance.** Every
number in §4's rank table and §5.1's diagnostic thresholds was also taken at
m=1.5. First order is the expectation, not a guarantee, at higher contrast.

Also to measure: the FD-vs-analytic derivative floor at the new fixture; the CI
cost. *(measured)* complex assembly is 37/67/182 ms at `nn=150/200/300`, so a
48-node contour is 1.8/3.2/8.7 s. Existing suite is 55 s.

---

## 9. Known limitations to document

- Degenerate poles are found and counted but **not refined** (bordered Newton
  assumes a simple eigenvalue). Flagged via `cond(J)`, never silent.
- No mode normalisation, hence no mode fields (D4).
- Homogeneous background only; slab/image backgrounds are out of scope and would
  break the holomorphy argument in §5.2.
- `dM/dλ` is exact only for non-dispersive materials.

---

## 10. Stays behind (confirmed against `sie/qnm/`)

`runner.py`, `runner_legacy.py`, `_workers.py`, `sweep.py`, `config.py`,
`legacy.py`; `_accumulate_parallel` and all Dask; `tqdm`/progress; `print`-based
tracing (the package has none today); `_malloc_trim`; every
`SurrogateComplexDomainError` path and the `discarded`/`discard_reason` fields;
`x_interface`/`r_eff` and `assemble_matrix_image`; `traj_re`/`traj_im`/`steps`/
`alphas`; the flat-scalar worker signature; `fd_step`; `n_expected`;
`sv_gap_factor`; **and the sign convention**.

Ported guards that retain a live failure mode: backtracking line search on `‖F‖`;
the escape check (`0.1·(L_re + L_im)` outside the rectangle); the LU-failure skip
in the contour loop; residual/σ acceptance.

---

## 11. File layout

**New:** `src/pysie2d/beyn.py` (~230, algorithm only, no EM),
`src/pysie2d/qnm.py` (~180, façade), `tests/test_beyn.py`, `tests/test_qnm.py`,
`tests/test_conventions.py`, `examples/qnm_spectrum.py` + `figures/`.
**Landed:** `tests/test_mie_qnm.py` (Phase 1's anchor tests; the layout above
had no home for them — they test `reference/mie.py`, not the extractor).

**Modified:** `kernels.py` (+~90, `assemble_matrix_dwn`), `solver.py` (+~25,
`assemble_derivative`, plus the D1 conversion), `reference/mie.py` (+~85,
annotations + `qnm_wavelengths`), `green.py`/`fields.py`/`sources.py` (D1
docstrings), `__init__.py`, `tests/conftest.py`, `tests/test_efficiencies.py`,
`docs/conventions.md`, `README.md`, `CLAUDE.md`, `pyproject.toml`
(`per-file-ignores` for `beyn.py` `N806`; `version_variables`).

---

## 12. Incidental repo drift found (independent of this port)

- `pyproject.toml` says `0.3.0`, `src/pysie2d/__init__.py` says `"0.2.0"`, and
  `tests/test_placeholder.py` asserts the stale `"0.2.0"` — semantic-release only
  bumps `version_toml`.
- `CHANGELOG.md` has a hand-written `## Unreleased / Performance` block for
  changes that shipped in v0.3.0.
- `README.md` figure URLs are pinned to `/v0.2.0/`; a new figure would 404.
- `CLAUDE.md` and `README.md` both promise "v0.3.0 = QNM"; v0.3.0 shipped as a
  perf release.

---

## 13. Open questions — round 2

1. ~~**`Material.epsi`: absolute or relative?**~~ **Answered: absolute** → D5.
   `eps_rel = (n_core² + i·epsi)/n_clad²`. A silent behaviour change at
   `n_clad≠1` that CI cannot currently catch, so it ships in the same commit as
   D1 *with* a lossy `n_clad≠1` test added. Phase 0 is unblocked.

2. ~~**Does the convention fix ship separately?**~~ **Answered: yes** → D6.
   v0.4.0 = convention fix alone, v0.5.0 = QNM.

3. **Degenerate poles: reported once or twice?** Beyn returns two numerically
   identical eigenvalues. Recommend reporting **both** columns with a
   `multiplicity` array grouping them (as specced in §7) — the ±n degeneracy is
   real physics, and collapsing it discards an independent mode vector and makes
   counts disagree with the anchor.

4. **Naming:** `QNMSolver` / `.modes()` / `wavelengths` (symmetric with
   `BIESolver` / `.scatter()` / `wavelength`) vs the research `QNMExtractor` /
   `.beyn()` / `poles`. Recommend the former; `poles` is more standard in the QNM
   literature if matching papers matters more than matching the repo.

5. **CI budget:** a full QNM suite lands around +60–90 s on a 55 s baseline.
   Recommend accepting it unmarked at `nn=200`, with the `nn`-convergence test
   capped at 300, rather than introducing a `slow` marker and a second CI step.

6. **May I edit `CLAUDE.md`?** The plan touches its roadmap line and adds the
   `Im λ > 0` / `Q` convention under non-negotiable #2. That is your instructions
   file, so I will propose a diff rather than edit it unasked.
