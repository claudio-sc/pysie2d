# Complex-frequency Sommerfeld integrals: holomorphy of the deformed path

Status: step 1 of the v1.1 layered-background milestone. This is an argument, not
a measurement. Each claim is marked **[proved]** (argued here), **[standard]**
(cited) or **[plausible]** (not settled; the settling check is named).

Setting: source and observer both in the cover (ε₁), above a stack whose
reflection coefficient is R(q; λ). Interface normal is z; `Z = z + z' − 2·z_int > 0`.

    G_ind = (i/4π) ∫_Γ R(q)/α₁ · exp(i q (x − x') + i α₁ Z) dq,   α_j = √(k₀²ε_j − q²)

QNM box B: `Im λ > 0`, `Re λ > 0` (conventions §8), so `arg k₀ ∈ (−π/2, 0)` and
`Im k₀ / Re k₀ = −1/(2Q)`.

## 1. The legacy real-axis argument fails [proved]

`sie-legacy-0726/docs/sommerfeld_greens_function.tex` (eq. `bp_shift`,
l. 979–1012) claims both branch points ±k₀n drop below the real axis for
`Im k₀ < 0`. Only `+k₁` does; `−k₁` rises. Both cross the path the moment
`Im k₀ < 0` — the "5 % bound" buys nothing.

What the real-axis integral computes: with any branch rule depending only on α²
(e.g. `Im α ≥ 0`), R depends on k only through `k_j² = k₀²ε_j`, so
`F_real(k₀) = F_real(−k₀) = G_phys(−k₀)`: the continuation from the wrong
side. For R ≡ 1 it is `−(i/4)·H₀^{(2)}(k₁ρ_img)` (DLMF 10.11.5), the
**incoming** wave. Its poles are not QNMs, and it jumps across real k₀.

No real-axis branch rescues it. On real q, `Im α₁² = Im k₁²` is a constant, so
the only continuous branch with decaying tails gives `α₁(0) = −k₁` (the advanced
sheet). The physical continuation needs `α₁(0) = +k₁` **and** decaying tails,
which only a path that winds between the branch points delivers.

The legacy text also says NumPy's complex sqrt picks `Im ≥ 0`. It picks
`Re ≥ 0`, which makes the tails `e^{+|q|Z}` divergent. `_kx_branch` enforces
`Im ≥ 0`, so the legacy code computes either the incoming Green function or a
divergent integral.

## 2. Permittivity, branch labels, and gain

- `k_j := k₀·√ε_j` with the **principal root of the complex ε_j**, never of
  |ε_j|. `Material.nc` rebuilds Im n from |ε| and cannot express `Re ε < 0`, so
  it must not be used for layers.
- For passive media (`Im ε_j ≥ 0`), `+k_j` stays in the open right half-plane
  over the whole box, metals included. The labelling is continuous in λ.
  [proved]
- **Signed zero trap:** `np.sqrt(complex(-4, -0.0)) == -2j`. After asserting
  `Im ε ≥ 0`, normalise with `complex(eps.real, eps.imag + 0.0)`.
- **Gain is excluded by assertion** (`Im ε_j ≥ 0` on every layer, cover
  included). With gain, `+k_j` can reach the 3rd quadrant, no path separates the
  pair, and the equality in §4 fails, because it needs `Im k_j² ≥ 0` at real ω.
  A non-dispersive gain half-space is not causal in any case (Nistad & Skaar, PRE
  78, 036603, 2008). [standard]

## 3. Construction [proved]

**Path.** `Γ: q = t − iγ(t)`, with γ odd, `γ > 0` on (0, T) and `γ ≡ 0` for
`|t| ≥ T` (e.g. `γ = h·sin(πt/T)`). It dips into the 4th quadrant on the right
and rises into the 2nd on the left. R depends on q only through q², so checking
the right half suffices. **The tails must return to the real axis:** a tail at
angle −θ converges only if `|Δx|·tanθ < Z`, which fails for laterally separated
points low over the interface.

**Branch.** Use the vertical-cut sheet in closed form, which needs no tracking:

    s_up(w) = e^{−iπ/4} √(i w)        s_dn(w) = e^{+iπ/4} √(−i w)
    α_j(q)  = i · s_up(q − k_j) · s_dn(q + k_j)

This gives `α_j(0) = +k_j` and `α_j → i|q|` on both tails. The cuts run up from
+k_j and down from −k_j, and Γ meets neither. For a multilayer, only the outer
half-spaces contribute branch points; interior α_j enter R evenly (Chew ch. 2;
Michalski & Mosig 1997). [standard]

**Theorem.** `G_Γ(λ)` is holomorphic on a neighbourhood of a compact set K if:

- **C0.** Each ε_j(λ) is holomorphic on K. Constants and Drude qualify.
  **Interpolated tabulated data (e.g. Johnson–Christy splines) does not, and
  Beyn then breaks silently.**
- **C1 (clearance).** For every λ ∈ K and every right-hand singularity
  s ∈ {+k_j(λ)} ∪ {+q_p(λ)}: `Im s + γ(Re s) ≥ δ > 0` and `Re s < T`. The poles
  q_p must be computed with the **same α_j function** the quadrature uses.
- **C2 (tails).** `T² > max_K max_j Re k_j²`. Then `|e^{iα₁Z}| ≤ e^{−c|q|Z}`,
  R is bounded (TE O(1/q²), TM → `R_∞ = (ε₂−ε₁)/(ε₂+ε₁)`), and for
  `Z ≥ Z_min > 0` the M-test and Morera give holomorphy.

**Assertable form of C1.** Im k_j(λ) is harmonic, and so is Im q_p(λ) at a
simple root, so the worst case is on ∂K. Require
`min_{[t_a, t_b]} γ ≥ max_{∂K} (−Im s)⁺ + δ`, where `[t_a, t_b]` covers
Re s over ∂K. δ also sets the Gauss–Legendre rate through the Bernstein ellipse,
so take δ as a fraction of T.

## 4. G_Γ is the physical continuation [proved, modulo one standard lemma]

Take **K = B̄ = [λr_min, λr_max] × [0, λi_max]**. This "shadow box" reaches the
real axis. The clearance C1 must hold on B̄, not only on B.

1. At real λ with passive media, the proper sheet has no Sommerfeld cuts in the
   open 2nd and 4th quadrants (`Im α_j = 0 ⇒ q_r q_i ≥ 0`).
2. It has no proper poles in the open 4th quadrant either, because a mode that
   decays in z but grows along +x at real ω would violate passivity (Chew ch. 2;
   Michalski & Mosig 1997). [standard; not re-proved for arbitrary stacks]
3. So Γ is homotopic to the physical indented path, and `G_Γ = G_phys` on the
   real segment. By holomorphy on B̄ and the identity theorem, G_Γ is the
   continuation throughout B, and its QNMs are the physical ones.

**Leaky poles** are never needed. At real λ they are on the `Im α < 0` sheet,
which is not adjacent to Γ. One can reach Γ's sheet only by crossing a vertical
cut ray inside the hump region. Computing the singularity set with the
quadrature's own α_j catches that case. [plausible: rare]

**What a violation looks like.** If a singularity crosses Γ inside B, G_Γ jumps
along a curve in λ. Beyn sees a cut, not a pole: leaked rank and spurious
eigenvalues strung along the curve. Rank detection does not flag this, so C1
must be asserted.

## 5. Poles

- **Guided modes (multilayer).** `q_p(k₀ + iκ) ≈ q_p − i·n_g·|κ|` to first order
  [proved at a simple root], with n_g the group index, which can exceed the core
  index. Backward modes rise, which is harmless. Near cutoff a pole leaves
  through the branch point, which is above Γ. **A fixed hump is enough.** Size h
  by root-finding the poles on ∂B̄; no residue tracking is needed. Double roots
  (coalescing guided poles) break the argument, so check `dD/dq ≠ 0` at the
  roots.
- **TM plasmon, metal half-space.** `q_sp = k₀·ν`, `ν = √(ε₁ε₂/(ε₁+ε₂))`,
  moving rigidly with k₀. It drops below the axis iff `λi/λr > tan(arg ν)`,
  i.e. at moderate Q for low-loss Ag or Au, so the hump must be sized for it.
  For `Re ε₂ < −ε₁` it is on the proper sheet [proved]. For
  `−ε₁ < Re ε₂ < 0` it sits near the imaginary axis with marginal sheet
  membership [not settled]; classify it by evaluating the code's α_j at the
  root. As ε₂ → −ε₁, |q_sp| → ∞ and `R_∞` diverges.

## 6. Derivative kernels [proved]

Source side: ∂/∂x' → `−iq`, ∂/∂z' → `+iα₁`. Observer side: `+iq`, `+iα₁`.
These are polynomial factors with no new singularities and no extra exponential
growth, only an extra `max_Γ|q|`. In TM with `R_∞ ≠ 0` they scale as 1/Z and
1/Z² as Z → 0.

## 7. Separable (GEMM) form and conditioning

The integrand factorises, so a block is `A · diag(w R/α₁) · Bᵀ` with
`A_iq = e^{i q x_i + i α₁ (z_i − z_int)}`. The GEMM has the same cancellation as
the direct sum, about `e^{hD + |Im k₁| Z_max}`, so it costs no extra accuracy.
[proved for the algebra]

- **Centre x.** Centre x on the particle, or on the cluster centroid. Otherwise
  the factors carry `e^{h|x_c|}` and overflow near `h|x| ≈ 700`.
- **D is the full horizontal extent of the cluster**, not one particle's
  width.
- **Digits lost.** About `(hD + |Im k₁| Z_max − ln|G_ind|)/ln 10`. Assert
  `h·D ≤ ln(10)·d_loss` with d_loss ≈ 3–4. With `|Im k₁| ≈ Re k₁/2Q`, this is
  benign for a single particle at Q ≳ 1 and prohibitive for wide clusters with
  low-Q modes. [plausible figures]

## 8. Edge cases

- **Z → 0.**
  - TE: `R = O(q⁻²)`, so the integral converges even at Z = 0.
  - TM: `R_∞ ≠ 0` gives a near-singular image log on the particle's own boundary,
    and Kress quadrature presumes a smooth reflected kernel. Subtract
    `R_∞·(i/4)H₀^{(1)}(k₁ρ_img)` analytically; it is holomorphic, and `R_∞`
    is λ-independent for non-dispersive ε.
  - Exact contact is out of scope.
- **Re λ → 0.** The branch points rotate towards the imaginary axis and h → ∞.
  Assert `λi_max/λr_min ≤ tan θ_max`, default θ_max = 45° (Q ≥ 0.5). Together
  with the `hD` bound, this is stricter than conventions §8.
- **k₂ → k₁.** R → 0, but not uniformly: at q = k₁, R = −1 whenever k₂ ≠ k₁. It is
  uniform once `|k₂ − k₁| ≪ δ`. Both branch-point pairs must satisfy C1.
- **PEC.** R ≡ −1 (TE), R ≡ +1 (TM), so `G_ind = ∓(i/4)H₀^{(1)}(k₁ρ_img)`
  exactly. Approaching PEC as a metal, q_sp crowds +k₁ from the right, so δ must be
  measured to both.

## 9. Independent anchors

1. R ≡ 1 at complex k₁: `(i/4π)∫_Γ e^{iqΔx + iα₁Z}/α₁ dq = (i/4)H₀^{(1)}(k₁ρ_img)`.
2. A regression test for the legacy failure: the real-axis `Im ≥ 0` integral
   returns `−(i/4)H₀^{(2)}(k₁ρ_img)`.
3. PEC: ∓ the image Hankel function; in the full solver, the particle plus its
   mirror through the v0.8 cluster reference.
4. Plasmon pole at `k₀·√(ε₁ε₂/(ε₁+ε₂))`, as a check on the pole finder.

## References

- Michalski & Mosig, IEEE TAP 45(3), 508 (1997); JEWA 30(3), 281 (2016).
- Chew, *Waves and Fields in Inhomogeneous Media*, ch. 2.
- Paulus, Gay-Balmaz & Martin, PRE 62, 5797 (2000); Paulus & Martin, PRE 63,
  066615 (2001) (in `sie-legacy-0726/biblio`, details not re-checked).
- Nistad & Skaar, PRE 78, 036603 (2008).
- DLMF 10.11.5.
