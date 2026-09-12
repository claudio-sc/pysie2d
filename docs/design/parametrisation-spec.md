# `Parametrisation` — code-spec, near-uniform arc length (draft 1)

**Status:** draft 1, 12 Sep 2026. Nothing implemented. Written from the G1/G2
prototypes (`studies/adaptive_density.py`, `studies/kress_t.py`) and five
rounds of questions answered on 12 Sep 2026 (§11). Decisions it works within:
[v0.6-architecture.md](v0.6-architecture.md) §3–4; contract:
`docs/conventions.md` §13 (tentative).

**Who it is for:** the worker who writes `geometry.py`'s v0.6 node placement.
Every formula here is meant to be executable without re-deriving it; every
decision carries the alternatives it beat and why.

---

## 1. Scope

**Goal.** Replace the v0.5 `np.interp` arc-length inversion with a map
`θ = w(t)` that is C^∞ in `t`, supplying `w`, `w'`, `w''` in closed form, and
place nodes **near** uniform arc length: `|dx/dt| = ρ(t)` with `ρ` uniform up
to a small curvature-driven perturbation whose size is set by the `R` band.

**Uniform arc length is the reference design.** The perturbation exists to
look for regimes where a *slightly* graded map converges faster than uniform
arc length — something the circle, and so Mie, is blind to, because on a
circle `κ` is constant and the perturbation vanishes identically. **If no
regime improves, the shipped density is `ρ = const` and the perturbation is
removed** (§9, exit criterion E-8).

**In scope.**

1. The smooth inversion (§4).
2. Closed-form θ-derivatives of the Gielis boundary to third order, and their
   `t` and arc-length forms documented (§3).
3. The near-uniform density (§5).
4. The object and its contract (§6).
5. The measurement that answers "up to which point is `np.interp` a problem"
   (§8, M-1).

**Non-goals — documented, not fixed.** Each is a real discontinuity, and each
is excluded deliberately:

| Non-goal | What is discontinuous | Consequence | Where users are told |
|---|---|---|---|
| **Shape kinks** | `|cos u|^n2`, `|sin u|^n3` with an exponent that is not an even integer: `r(θ)` is only `C^⌈n⌉−1` where `cos u` or `sin u` vanishes | No parametrisation restores spectral order on a non-smooth curve; convergence falls to algebraic at a rate set by the exponent | §10, safe shapes |
| **Branches in parameter space** | `is_circle`, `arc_length=True/False`, `_etoil` forcing `a = b = 1` | Code path switches as a parameter crosses a value; breaks continuation and `∂M/∂p`, not the rate | doc A §5 (API break removes `arc_length`) |
| **Integer-valued shape functionals** | `nn` from the band (`ceil`), the density bandwidth `M` if kept integer | Steps in parameter space; `nn` must be an integer, `M` need not be (§5.3) | this spec §5.3 |
| True corners, handoff §9 star anomaly | — | doc A §6; untouched by decision | doc A |

---

## 2. Conventions pinned here

- Boundary: `f = r(θ) sin θ + x0`, `g = r(θ) cos θ + z0` (x from `sin`, z from
  `cos`, as `geometry.gielis`). Primes are `d/dθ`; dots are `d/dt`.
- Quadrature nodes: `t_j = 2π(j + ½)/nn`, `j = 0…nn−1` — the offset matches
  `_etoil` so a circle's nodes are unchanged from v0.5. The Kress weights are
  circulant in `t_i − t_j`, so the offset has no effect on them.
- Anchoring: `T(0) = 0`, hence `w(0) = 0`.
- `u = m θ / 4`, `c = cos u`, `s = sin u`.
- `σ` is a relative density; **its overall scale cancels in `w`** (§4.1). Only
  the `nn` derivation reads its absolute level.

---

## 3. Closed-form boundary derivatives

Replaces `_der_real_3` (first-order finite differences, one-sided and
non-periodic at `j = 0, nn−1`) and the spectral `_dtheta` of the prototypes.
Needed to third order: `w''` needs `ρ'`, and a curvature-driven `ρ` needs `κ'`,
which needs `r'''`.

### 3.1 The superformula

    C = a^{−n2} |c|^{n2},   S = b^{−n3} |s|^{n3},   P = C + S,   r = rad · P^{−1/n1}

Derivatives in `u` (valid where `c ≠ 0`, resp. `s ≠ 0`; **everywhere** when the
exponent is an even integer, in which case `|c|^{n−k}·sgn` collapses to the
plain power `c^{n−k}` and must be implemented that way — see the note below):

    C_u   = −n2 a^{−n2} |c|^{n2−2} c s
    C_uu  =    a^{−n2} [ n2(n2−1) |c|^{n2−2} s² − n2 |c|^{n2} ]
    C_uuu =    a^{−n2} [ −n2(n2−1)(n2−2) |c|^{n2−4} c s³ + n2(3n2−2) |c|^{n2−2} c s ]

    S_u   =  n3 b^{−n3} |s|^{n3−2} s c
    S_uu  =    b^{−n3} [ n3(n3−1) |s|^{n3−2} c² − n3 |s|^{n3} ]
    S_uuu =    b^{−n3} [  n3(n3−1)(n3−2) |s|^{n3−4} s c³ − n3(3n3−2) |s|^{n3−2} s c ]

(`S` follows from `C` by `u → π/2 − u`, which flips odd orders.) Then
`P^{(k)} = (m/4)^k (C^{(k)}_u + S^{(k)}_u)` in θ.

**Implementation note — `0·∞`.** At `n = 2`, `C_uuu`'s first term is
`0 · |c|^{−2}`, which is `nan` at `c = 0` if evaluated as written. Build each
term as *coefficient-then-power* and skip terms whose integer coefficient is
zero; for even-integer exponents use integer powers of `c`, never `|c|` raised
to a negative float. The v0.5 `_rderiv` masks the same `0/0` with `np.where`;
under the even-integer rule the mask is unnecessary, and a mask is itself a
kink when the exponent is not even (non-goal).

### 3.2 From `P` to `r`

Through `ℓ = ln r = ln rad − q/n1`, `q = ln P`:

    q'   = P'/P
    q''  = P''/P − (P'/P)²
    q''' = P'''/P − 3 P'P''/P² + 2 (P'/P)³

    r'   = r ℓ'
    r''  = r (ℓ'' + ℓ'²)
    r''' = r (ℓ''' + 3 ℓ'ℓ'' + ℓ'³),       ℓ^{(k)} = −q^{(k)}/n1

The log form is chosen because `P` spans many decades on a sharp star and the
ratios `P^{(k)}/P` stay O(1) where `P` itself underflows the product form.

### 3.3 Boundary, speed, curvature — in θ

    f'   = r' sin θ + r cos θ                    g'   = r' cos θ − r sin θ
    f''  = r'' sin θ + 2r' cos θ − r sin θ       g''  = r'' cos θ − 2r' sin θ − r cos θ
    f''' = r''' sin θ + 3r'' cos θ − 3r' sin θ − r cos θ
    g''' = r''' cos θ − 3r'' sin θ − 3r' cos θ + r sin θ

    γ  = |x'| = √(r² + r'²),          γ' = r'(r + r'')/γ
    f'g'' − g'f'' = −(r² + 2r'² − r r'')          (negative: the curve runs clockwise in (x, z))
    |κ| = |r² + 2r'² − r r''| / γ³
    N  = r² + 2r'² − r r'',      N' = 2 r r' + 3 r' r'' − r r'''
    κ  = N/γ³,                   κ' = (N' γ − 3 N γ')/γ⁴

All of §3.1–3.3 verified numerically on 12 Sep 2026 against spectral
differentiation of `r` (`N_f = 4096`, `a = 1.3, b = 0.8, m = 4, 6/12/4`):
relative residuals 8e-13, 2e-10, 3e-8 for `r', r'', r'''`; 5e-13 for `N'` and
`γ'`; 6e-8 for `f''', g'''` — each at the spectral round-off floor for its
order.

### 3.4 In `t` (what Kress consumes)

With `θ = w(t)`:

    ẋ = x'(w) ẇ,        ẍ = x''(w) ẇ² + x'(w) ẅ,        |ẋ| = γ(w) ẇ

Kress needs `f, g, ḟ, ġ, f̈, g̈` at the nodes. Third `t`-derivatives are not
consumed by assembly. **This replaces `kress_t.py`'s spectral `_dt` at `nn`
nodes**, which aliases at low `nn` and is exactly the pre-asymptotic regime
where the ladder spends its points.

### 3.5 In arc length (documented, not coded)

`d/ds = γ⁻¹ d/dθ`:

    x_s  = x'/γ,          x_ss = (x'' − (γ'/γ) x')/γ²,          κ = f_s g_ss − g_s f_ss

**Decision deferred** (answer 12 Sep 2026): whether any consumer takes
arc-length derivatives. Nothing in this spec does. Do not add a code path for
them until one does.

---

## 4. The inversion

### 4.1 The map

    T(θ) = (2π/Z) ∫₀^θ σ(θ̃) γ(θ̃) dθ̃,       Z = ∫₀^{2π} σγ dθ,       w = T⁻¹

`T` is strictly increasing (σγ > 0), `T(2π) = 2π`, and `T(θ + 2π) = T(θ) + 2π`,
so `w` is a C^∞ monotone circle map whenever `σγ` is analytic. `σ → λσ` leaves
`T` unchanged: the density's scale is not part of the map.

Closed forms at a node `θ_j = w(t_j)`, **taken from the same series `T` is
built from** (§4.2), never from the analytic `γ` — `w'` must be the exact
derivative of the map whose root was found, or the Jacobian and the node
disagree at the truncation level:

    w'  = 1 / T'(θ_j) = Z / (2π σγ(θ_j))
    w'' = −T''(θ_j) · w'³

### 4.2 Representation of `σγ` — shape-intrinsic truncation

`σγ` is sampled on a uniform θ grid of `N_f` points, FFT'd, and kept as a
real trig series `a_0 + 2 Re Σ_{k=1}^{K} c_k e^{ikθ}`. `T` is its exact
antiderivative (secular term `a_0 θ`), so there is no quadrature error in `T`
beyond truncation.

**`N_f` and `K` are functions of the shape only** (answer 12 Sep 2026):

1. Start at `N_f = 1024`.
2. Compute `|c_k| / a_0` for `k ≤ N_f/2`.
3. If `max_{k > 3N_f/8} |c_k|/a_0 > 4 eps`, double `N_f` and repeat. Cap at
   `2^20`; raise `ValueError` beyond — a curve that needs more is outside the
   safe-shape envelope (§10), and a silent truncation is a plausible wrong map.
4. `K` = the largest `k` with `|c_k|/a_0 > eps`.

Why each piece:
- **Relative** to `a_0`: `γ ∝ rad`, so a relative threshold makes `N_f` and `K`
  identical under `rad → s·rad` — conventions §9 holds structurally rather than
  numerically.
- **Not `nn`:** a representation that changes with `nn` makes `w` a different
  map at every resolution (G2 correction, doc A §4.1; invariant 13.3).
- **Replaces** the prototype's fixed `N_FINE = 4096` and v0.5's
  `max(10·nn, 4096)`. The first under-resolves the sharp star silently; the
  second ties the map to `nn`.
- `4 eps` over the top quarter guards against round-off plateau being read as
  unconverged; the plateau of an analytic function's FFT sits at a few `eps`.

**Cost note.** `K` grows with the curvature layer's reciprocal width — expect
hundreds of terms on the sharp rung. Construction is once per shape. Measured
in M-3, not assumed.

### 4.3 Root-finding — Newton to round-off, then two steps

For each node, solve `T(θ) = t_j`:

1. **Seed:** `θ⁰ = ` linear interpolation of `(T(θ_k), θ_k)` on the `N_f` grid.
   The seed is only a starting point: Newton on a strictly monotone smooth `T`
   converges to the unique root from any seed in the bracket, so the seed's
   C⁰ character does not reach `w`.
2. **Iterate** `θ ← θ − (T(θ) − t_j)/T'(θ)`, safeguarded: if the step leaves the
   bracket `[θ_k, θ_{k+1}]` that contains the root, bisect instead.
3. **Stop:** once `max |step| ≤ 2π · 8 eps`, perform **exactly two more** full
   steps, then stop. Cap at 50 iterations and raise.

**Why two more steps rather than a tolerance** (answer 12 Sep 2026, option a):
a tolerance-based stop leaves a residual anywhere below the tolerance, and
which iteration crosses it jumps as a shape parameter moves — a small step
function inside a continuation trajectory, which is the very defect signal
continuation reads. Quadratic convergence puts every node at the round-off
floor after the extra steps, so `w` is a smooth function of parameters to
round-off.

### 4.4 Alternatives considered for the inversion

Ranked on cost and accuracy, after a **hard gate** on independence from `nn`,
λ and absolute length (answer 12 Sep 2026).

| Route | Gate | Accuracy | Cost | Verdict |
|---|---|---|---|---|
| **A. Newton on the exact Fourier antiderivative** (§4.3) | passes | round-off in `w`; `w', w''` exact for the truncated series | `O(nn·K)` per iteration, ~5 iterations | **Chosen.** Measured at 4e-16 density residual in G2. |
| B. Fourier series of `w(t) − t` itself, coefficients from A on a fine `t` grid | passes | same as A at the nodes; spectral elsewhere | A plus an FFT; evaluation at any `nn` is `O(nn·K_w)` with no iteration | **Rejected as primary, kept as an option.** `w − t`'s analyticity strip is narrower than `σγ`'s — composition narrows it (doc A §4.1) — so `K_w > K`. Worth it only if many `nn` are sampled per shape; a convergence ladder does, a solve does not. Revisit at migration if M-3 shows construction dominating a ladder. |
| C. ODE `dθ/dt = Z/(2π σγ(θ))` with a high-order integrator | passes | set by the integrator's step count — a **new discretisation parameter**; closure `θ(2π) = 2π` only to integrator error; dense output adds interpolation error at nodes | comparable to A | **Rejected.** Introduces an error that is not at round-off and a knob nobody asked for. |
| D. Cubic spline inversion of `s(θ)` (handoff `exp_inversion.py`) | passes if the grid is shape-intrinsic | C² only: algebraic floor ~`h_f⁴` in nodes and ~`h_f²` in `w''` | cheapest | **Rejected.** Finite smoothness is the failure being removed, one notch higher. |

Seed alternatives: `θ⁰ = t_j` (no interpolation at all) is safe under the
safeguard but costs extra iterations wherever `w` departs from identity, which
on a sharp star is everywhere. The interpolated seed is kept; it is a
convergence aid, not part of the map.

---

## 5. The near-uniform density

### 5.1 Form

    u(θ) = |κ(θ)| · L / 2π                   (≡ 1 on a circle; dimensionless)
    s̃(θ) = G_M ∗ ln u                        (Gaussian smoothing, §5.3)
    σ(θ) = exp(α_eff · (s̃ − ⟨s̃⟩))

`⟨·⟩` is the θ-mean. Subtracting the mean rather than the prototype's `max`
removes a `max` over a grid from the map; since `σ`'s scale cancels in `T`
(§4.1), the anchor is free and only the `nn` derivation (§5.4) reads it.

### 5.2 The exponent — one smooth formula, no branch at `C = 1` and no clamp

`C = R_max/R_min ≥ 1` is the requested contrast. Let `Δ = max s̃ − min s̃`.

    α_eff = α · ln C / (α Δ + ln C)

Properties, each of which the prototype got from a branch or a clamp:

- `C = 1` ⇒ `α_eff = 0` ⇒ `σ ≡ 1` ⇒ **exactly uniform arc length, by the same
  code path.** The prototype's `if contrast <= 1.0: return ones` is a branch at
  the reference design; forbidden (answer 12 Sep 2026, Q5).
- Circle (`Δ = 0`) ⇒ `α_eff = α` ⇒ `σ ≡ 1` since `s̃` is constant. The
  prototype's `if spread <= 0.0` branch is unnecessary.
- The realised log-contrast is `α_eff Δ = ln C · αΔ/(αΔ + ln C) < ln C` —
  **the band's upper bound can never be exceeded, so no clamp is needed**, hard
  or soft. It approaches `ln C` when the shape's curvature contrast is large.
- `α_eff ≤ α` always.

Alternatives: the prototype's `min(α, ln C/Δ)` (C⁰ in parameters where the two
arguments cross, and still needs the clamp as a guard); `softmin` with a
temperature (a knob). Rejected for a knob-free rational form.

**The band is a bound, not a promise** (skill, G2): the realised contrast is
reported on the object (§6) so a caller never reads the request as achieved.

`Δ` still contains `max`/`min` over the `N_f` grid. They are values, not
arguments, so they are continuous in shape parameters; not smooth where the
extremum migrates between lobes on an asymmetric shape. Parameter-space
non-goal.

### 5.3 Smoothing width

Kept from G2: Gaussian taper `exp(−½ (k/M)²)`, **positive kernel, never a sharp
cutoff**, `M = 2π / w_layer` with `w_layer` the angular width on which
`ln |κ|` is within 1 of its maximum.

Change from the prototype: **`M` is real, not `ceil`'d**, and `w_layer` is
measured as `(2π/N_f) · Σ_k H(ln|κ_k| − max + 1)` — still a count, so still a
step function in parameters; documented as a non-goal (§1). A smooth
replacement (e.g. a logistic in place of `H`) is a one-liner if continuation
shows a step; do not add it pre-emptively.

In the near-uniform regime the smoothing hardly matters: `α_eff Δ ≲ ln C` is
small, so `σ = 1 + O(ln C)` and its high harmonics are `O(ln C)` too.

### 5.4 `nn` from the band

Unchanged from G2 in logic, with the prototype's descent loop kept:
`nn = ⌈R_min · Z_min · n_core / λ_ref⌉` rounded up to even, then descend while
the verified worst node still satisfies `R_min`. `nn` is an input to
`nodes()` (§6), **never** to the map. λ_ref defaults to 1550 nm and is stored.

### 5.5 How small is "near-uniform"

In practice `C ∈ [1, 1.5]`. With `σ = exp(α_eff(s̃ − ⟨s̃⟩))`, node spacing in
arc length varies by a factor of at most `C` around the boundary, i.e. `±½ ln C`
≈ `±20 %` at the top of that range. No absolute length appears: `C` is a ratio
of `R` values, and `R` enters only `nn`.

---

## 6. The object

```python
@dataclass(frozen=True)
class Parametrisation:
    """A smooth monotone circle map θ = w(t), frozen at construction."""

    # the map: the truncated series of σγ and its normalisation
    ...

    @classmethod
    def gielis(cls, *, rad, a, b, m, n1, n2, n3,
               r_band: tuple[float, float] = ..., n_core: float,
               wavelength_ref: float = 1550.0, alpha: float = 0.5
               ) -> "Parametrisation": ...

    def nodes(self, nn: int) -> Nodes: ...
        # t, θ = w(t), w'(t), w''(t) at t_j = 2π(j + ½)/nn

    nn_from_band: int          # §5.4
    contrast_realised: float   # exp(α_eff Δ), ≤ C — the bound, not the promise
    alpha_eff: float
    n_fine: int                # N_f, §4.2 — shape-intrinsic
    n_terms: int               # K
```

Contract:

1. **`w` depends on the shape, `r_band`, `n_core`, `wavelength_ref`, `alpha`
   — and nothing else.** Not on `nn` (13.3), not on the solve wavelength
   (13.2), not on `rad` (§9).
2. **No special case for the circle or for `C = 1`.** Every shape goes through
   §4–§5 (answer 12 Sep 2026, Q5). The Mie anchor then exercises the code the
   stars run.
3. `nodes(nn)` is pure: same `nn`, same arrays, bit-for-bit.
4. The object is what is frozen for shape derivatives (13.3, §10): `M(p ± h)`
   is assembled from one `Parametrisation` evaluated on two shapes.

API placement, `Geometry` integration and the `theta=` break belong to the
migration spec (doc A §5), not here.

---

## 7. Answer: up to which point is `np.interp` a problem?

**Prediction, to be measured by M-1.** Three separate effects, which v0.5
conflates:

1. **On the circle, never.** `θ(s)` is linear, so piecewise-linear
   interpolation is exact. This is why no circle anchor ever saw it.
2. **In node positions: a floor, not a rate change.** `np.interp` returns
   `w(t_j) + e_j` with `|e_j| ≲ (h_f²/8) · max |d²θ/ds²|`, `h_f = L/n_fine`, and
   `e_j` is not a smooth function of `j`. Kress with exact `w'`, `w''` then
   converges spectrally until its error reaches `O(e)`, and stalls there.
   `max |d²θ/ds²|` grows with curvature contrast, so the stall comes earliest
   on the sharp rung, late on the mild one. Because v0.5 sets
   `n_fine = max(10·nn, 4096)`, the floor is constant up to `nn ≈ 410` and
   falls as `nn⁻²` beyond — **interp caps an otherwise spectral scheme at
   second order**.
3. **In the Jacobian: first order, always.** `np.interp` cannot supply `w'`.
   v0.5 uses `delt = np.diff(θ)` — a forward difference, the node at the left
   end of its interval — and `_der_real_3` for `ddf`, `ddg`, which is also
   non-periodic at the two wrap nodes. Both are O(h) whatever `n_fine` is. This
   is what makes the shipped arc-length path first order; interp is the
   second-order ceiling hiding behind it.

So: interp is harmless on circles, a second-order ceiling on smooth
non-circular shapes once the Jacobian is fixed, and an early one on sharp
shapes. M-1 measures where the ceiling sits on each rung.

---

## 8. Measurements (study scripts, before migration)

In `docs/design/studies/`, touching no package code. Probe: pole error for the
mild-star anchor (§8.1), `qext` elsewhere until G5.

### 8.1 The mild-star anchor — a smooth homotopy, not continuation in `n2`

**Critical correction to the Q1 answer "continue from the circle in `n2`".**
Continuation in `n2 = n3` from 2 to 4 passes through every non-even exponent,
where `|cos u|^{n2}` has a kink — the shape-kink non-goal, on every
intermediate step, so the trajectory would carry a rate change that is
geometry, not defect. Use instead the homotopy **in `P`**:

    P_ε(θ) = (1 − ε) + ε (cos⁴θ + sin⁴θ),     n1 = 2,  m = 4,  a = b = 1

`P_0 = 1` is the circle, `P_1` is the mild star (`n1/n2/n3 = 2/4/4`), and
every intermediate shape is analytic in θ and in ε. §3's derivatives apply with
`P^{(k)}` replaced by `ε (cos⁴ + sin⁴)^{(k)}`.

**The closed-form slope.** `cos⁴θ + sin⁴θ − 1 = −½ sin² 2θ`, so

    r_ε = rad (1 − ε/2 · sin² 2θ)^{−1/2} ≈ rad (1 + ε/8 − (ε/8) cos 4θ)

A uniform dilation `rad → rad(1 + ε/8)` plus a pure `cos 4θ` harmonic. At first
order:

- the dilation shifts every pole by `δλ = λ₀ ε/8` — **exactly**, by scale
  covariance (conventions §9);
- a `cos 4θ` boundary perturbation couples angular orders `n` and `n ± 4`
  only, so its diagonal element vanishes and a degenerate `±n` pair couples
  only if `2n = 4`.

Hence **`dλ/dε |₀ = λ₀/8` for every mode with `|n| ≠ 2`**, independent of
polarisation. `n = 2` splits at first order and needs the matrix element; out
of scope for the anchor, excluded from it.

This is independent of the solver (non-negotiable 3): selection rule plus
scale covariance, no BIE. It is checked by a central difference in ε at the
circle, with the Mie pole as `λ₀`.

### 8.2 Ladder

| Rung | Shape | Role |
|---|---|---|
| mild | homotopy `ε ∈ [0, 1]`, endpoint `2/4/4` | Mie anchor at `ε = 0`, slope anchor §8.1, continuation to `ε = 1` |
| medium | `m = 4`, `n1/n2/n3 = 6/12/12` | relative convergence only |
| sharp | `m = 4`, `n1/n2/n3 = 12/24/24` | relative convergence only |

G2's near-corner `20/50` rung belongs to G3 and is excluded. All exponents are
even integers (§10).

### 8.3 Checks

| # | Check | Pass | Tolerance reason |
|---|---|---|---|
| M-0 | §3 derivatives vs spectral derivative of analytic `r` on `N_f`, and vs complex-step `Im r(θ + ih)/h` for `r'` | ≤ `10·eps·K^k` relative for order `k` | spectral differentiation of order `k` amplifies round-off by `~K^k`; measured 8e-13 / 2e-10 / 3e-8 for `k = 1/2/3` at `K ≈ 300` |
| M-1 | `np.interp` ceiling: `|θ_interp − w(t_j)|` vs `nn` on each rung; Kress error with Newton nodes vs interp nodes (both with exact `w', w''`) | report the stall level and the `nn` where interp and Newton separate | measurement, not a gate |
| M-2 | Circle through the full path at `C ∈ {1, 1.5}`: `max|w(t_j) − t_j|` | ≤ `16·eps·2π` | round-off at `θ ~ 2π` is `eps·2π = 1.4e-15`; Newton plus FFT of a constant accumulate a small multiple. (The `1e-15` proposed in the questions is below the floor.) |
| M-3 | Construction cost per rung, `N_f`, `K` | report; flag if construction exceeds one `nn`-point assembly | cost is a ranking criterion (§4.4) |
| M-4 | Invariant 13.3: `nodes(nn)` vs `nodes(3nn)` at shared `t` (`t_j(nn) = t_{3j+1}(3nn)`) | bit-for-bit | same series, same root, same iteration path |
| M-5 | §9: `rad → 7·rad`, same `N_f`, `K`, `nn`; `θ` nodes agree | `N_f`, `K`, `nn` identical; `θ` ≤ `16·eps·2π` | scaled FFT coefficients are not bit-identical |
| M-6 | Density residual `max |σγẇ / mean − 1|` | ≤ `1e-13` | G2 measured 4e-16; margin for the sharp rung's larger `K` |
| M-7 | Continuity in `C` at `C → 1`: `w(t_j; C)` for `C = 1 + 10^{−k}`, `k = 1…8` | `‖w(C) − w(1)‖ ∝ ln C` to within 10 % over the range | first-order dependence through `α_eff ∝ ln C`; a branch shows as a jump |
| M-8 | Mie anchor: mild homotopy at `ε = 0` vs analytic pole; slope vs `λ₀/8` by central difference at `ε = ±10^{−3}` | pole at Kress floor; slope to `O(ε²) ≈ 1e-6` relative | central difference error `O(ε²)` |
| M-9 | Continuation `ε: 0 → 1` at `C ∈ {1, 1.25}`: secant predictor, ambiguity tell | smooth trajectory; any kink read as defect first | pole-continuation method |
| M-10 | Relative convergence on all rungs, `C ∈ {1, 1.1, 1.25, 1.5}`, reference high-`nn` uniform-θ Kress | report error vs `nn` per `C` | **the success criterion** (§9) |

---

## 9. Exit criteria

- **E-1…E-7** M-0, M-2, M-4, M-5, M-6, M-7, M-8 pass.
- **E-8, the decision.** From M-10: if some `C > 1` improves error at equal
  `nn` on some rung by a margin larger than the spread of the reference, record
  the regime and keep §5. **Otherwise ship `ρ = const`**: `§5` collapses to
  `σ ≡ 1`, `r_band` fixes only `nn`, `alpha` and `contrast_realised` leave the
  object, and the `r'''`/`κ'` derivatives are kept documented but unused.
- M-1, M-3, M-9 are reported whatever they show.

---

## 10. Safe shapes — for the user documentation at migration

Draft text; lands in `README.md`/docs only with the migration.

> Spectral convergence requires a boundary that is analytic in θ. For the
> Gielis superformula that means **`n2` and `n3` even integers** (2, 4, 6, …);
> `n1 > 0` and `a, b > 0` are unrestricted. Other exponents are accepted and
> produce correct results at a lower, algebraic rate set by the exponent, with
> no warning.
>
> As exponents grow, curvature concentrates at the lobe tips and the number of
> Fourier terms needed to represent the boundary grows with it. Beyond
> `N_f = 2^20` construction raises rather than silently truncating.
>
> Odd `m` with `a ≠ b` does not close (D5) and is rejected.
>
> The `R` band is a bound: a nearly round shape has little curvature to grade,
> and the achieved contrast (`contrast_realised`) can be well below the one
> requested.

---

## 11. Decisions taken in the question rounds (12 Sep 2026)

| Topic | Decision |
|---|---|
| Fix scope | `np.interp` inversion only. Shape kinks and parameter-space branches are documented non-goals, with safe-shape guidance |
| Derivatives | Closed-form θ-derivatives of `r` to third order; `t` and arc-length forms documented; which consumer uses which decided later |
| Test shapes | Ladder of three stars, mild → sharp; only mild is Mie-anchored (via homotopy, §8.1 — corrects the `n2`-continuation answer), the rest relative |
| Goal | Smooth in the perturbation, tending to uniform arc length with no switch; success = a near-uniform regime that beats uniform, else revert |
| `h` | Curvature-driven, small because the band is narrow |
| `δ` | Expressed through the `R` band, `C = R_max/R_min`, narrow in practice |
| Inversion | Newton on Fourier antiderivative, challenged against three alternatives (§4.4); ranked by cost and accuracy after a hard gate on `nn`/λ/length independence |
| Truncation | Shape-intrinsic, relative-coefficient decay (§4.2) |
| Newton stop | Converge to round-off, then exactly two more steps |
| Circle | No special branch; full path, tolerance at the round-off floor (M-2) |
