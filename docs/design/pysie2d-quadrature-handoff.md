# pysie2d — handoff: quadrature accuracy, node placement, and the path to spectral convergence

**Subject package:** `pysie2d` 0.5.0 (PyPI; GitHub `claudio-sc/pysie2d`)
**Date of investigation:** September 2026
**Status of findings:** all numerical claims below were measured against the shipped 0.5.0 source, on a circular cylinder where analytic Mie theory provides exact ground truth.

---

## 1. Executive summary

Two independent lines of work are covered here.

**The question asked.** Can the condition number of the BIE system matrix be used as an objective to choose boundary node angles, in place of the current uniform-arc-length placement?

**Answer: no, and the reason is structural rather than a matter of tuning.** All recoverable conditioning in this matrix is a single free diagonal scaling worth about 2.6×. On a circle, uniform nodes are already the exact minimiser of the condition number, and an optimiser started there does not move. The remaining growth of the condition number is linear in `nn` and is a property of the first-kind single-layer operator, which no node placement removes.

**What the investigation found instead.** The solver converges at exactly first order in `nn` — on a perfect circle, with perfectly uniform nodes, where node placement cannot be a factor. The cause is the quadrature treatment of the logarithmic Green-function singularity, which currently patches only the diagonal entry. Replacing it with Kress–Martensen product quadrature, plus supplying analytic second derivatives of the boundary parametrisation, takes the solver from first-order convergence to machine precision at `nn = 30`.

**Recommended priority order:**

1. Kress–Martensen product quadrature on all four blocks (Section 5–6). First order → third order.
2. Analytic `ddf` / `ddg` instead of the finite-difference `_der_real_3` (Section 6.4). Third order → machine precision.
3. Optionally, rescale the `chi` block of the unknown vector (Section 4.2). Free 2.6× on the condition number; cosmetic next to items 1 and 2.
4. Do **not** pursue condition-number-driven node placement (Section 4).

---

## 2. The code as it stands

### 2.1 Formulation

`pysie2d` solves 2-D time-harmonic electromagnetic scattering from a single smooth cylinder in a homogeneous background, reduced to a scalar Helmholtz transmission problem. Conventions: lengths in nm, time convention `exp(-iωt)`, outgoing waves `H_n^{(1)}`, wavelengths are vacuum wavelengths.

The unknown vector is

```
ei[:nn]  = phi    boundary field values
ei[nn:]  = chi    normal derivative, with the Jacobian folded in
```

giving a dense `2nn × 2nn` complex system `M(λ)·ei = rhs`, solved directly by LU.

### 2.2 Block structure

From `src/pysie2d/kernels.py`, `assemble_matrix`:

| Block | Rows | Cols | Kernel | Wavenumber | Character |
|---|---|---|---|---|---|
| M1 | `0:nn` | `0:nn` | double layer, `+1/2` identity on diagonal | `wnum_bg` | second kind |
| M2 | `0:nn` | `nn:2nn` | single layer `H₀` | `wnum_bg` | **first kind** |
| M3 | `nn:2nn` | `0:nn` | double layer, `−1/2` identity on diagonal | `wnum_core` | second kind |
| M4 | `nn:2nn` | `nn:2nn` | single layer `H₀`, times `eta` | `wnum_core` | **first kind** |

with `eta = kd` for TM (`pol=1`) and `1` for TE (`pol=2`).

This split matters. The `phi` columns carry a second-kind operator whose spectrum clusters and whose conditioning is bounded. The `chi` columns carry the single-layer operator, which is of order −1 and whose discrete conditioning grows linearly in `nn`. The system as a whole is therefore not uniformly well conditioned, and cannot be made so by moving nodes.

Note the absence of an explicit Jacobian factor in the M2/M4 off-diagonals — this is consistent, because `chi` carries the Jacobian.

### 2.3 Current singularity treatment

```python
arg_d      = wnum_bg   * delt / (2.0 * e) * gamma
arg_d_core = wnum_core * delt / (2.0 * e) * gamma
d_m2 = c2 * hank0(arg_d)
d_m4 = c2 * eta * hank0(arg_d_core)
```

with `e = np.e`, `c2 = 0.25j * delt`, `gamma = sqrt(df² + dg²)`.

This is a Maradudin-style diagonal self-patch. It comes from the identity

```
∫ over one element of ln|s| ds  =  Δ · ln( Δ / (2e) )
```

so the diagonal is evaluated at the effective radius `Δ/(2e)`. It handles the diagonal entry and nothing else. Every off-diagonal entry, including immediate neighbours where the kernel is still steeply varying, is a plain point evaluation carrying the trapezoid weight.

The M1/M3 diagonals carry the curvature limit of the double layer,
`0.5 − deriv·delt/(4π·gamma²)` and `−(0.5 + deriv·delt/(4π·gamma²))`, with
`deriv = df·ddg − ddf·dg`. These limits are correct; the issue there is that the *log part* of the double-layer kernel (which does not vanish for Helmholtz, unlike Laplace) is left to the trapezoid rule.

### 2.4 Node placement and weights

`geometry.boundary_setup` offers two paths:

- **Uniform theta** (`arc_length=False`): `theta = (arange(1,nn+1) − 0.5)·2π/nn`, `delt` a scalar. This is a midpoint rule on an equispaced grid.
- **Uniform arc length** (`arc_length=True`, the default): `_uniform_arc_theta` builds a fine uniform-theta grid of size `n_fine = max(10·nn, 4096)`, accumulates chord lengths, and inverts with `np.interp` (piecewise linear). Then `delt = np.diff(theta, append=theta[0] + 2π)` — a forward difference, so each node sits at the left end of its own weight interval.

Second derivatives `ddf`, `ddg` come from `_der_real_3`, a three-point finite difference on the theta grid, even though the Gielis superformula is differentiable in closed form.

---

## 3. Reproduction environment

All measurements below used:

```
geometry : Geometry.gielis(rad=200, n_pts=nn, m=0)      # circle
material : Material(n_core=1.5, n_clad=1.0, pol=2)      # TE
wavelength : 600.0 nm  (vacuum)
reference  : pysie2d.reference.mie.efficiencies(x, 1.5)["Q_ext_TE"]
             with x = 2π·200/600
```

`qext` was used as the error probe rather than `qsca`, because `ScatterResult.efficiencies` integrates the far field over a fixed `n_angles` grid and therefore has an angular-quadrature floor around 1e-4 that masks convergence. `qext` is a single forward-direction amplitude and has no such floor:

```python
amp, _ = result.far_field(4000)
d = 2*np.pi/3999.0
qext = amp[int((2*np.pi - np.deg2rad(result.angle))/d)].imag / (result.wnum_bg * 2 * rad)
```

The Gielis star used for the non-circular tests was `m=6, n1=6, n2=12, n3=12, rad=200`.

---

## 4. Investigation 1: condition number as a node-placement objective

### 4.1 The condition number grows linearly in nn

Circle, as above:

| nn | κ(M) | after row+col equilibration | after one scalar on the chi block |
|---|---|---|---|
| 50 | 4.08e1 | 1.60e1 | 1.59e1 (α = 10) |
| 100 | 8.20e1 | 3.15e1 | 3.15e1 (α = 20) |
| 200 | 1.64e2 | 6.27e1 | 6.28e1 (α = 39.8) |
| 400 | 3.28e2 | 1.25e2 | 1.25e2 (α = 79.4) |

κ doubles whenever `nn` doubles. This is the single-layer block and it is not removable by node placement. Note that even after optimal scaling the growth remains linear.

### 4.2 All recoverable conditioning is one number

Full Sinkhorn-style row and column equilibration and a *single scalar* rescaling of the `chi` block agree to three digits at every resolution. The optimal scalar tracks `nn`, i.e. it is proportional to `1/delt` — the `chi` unknown is simply carried in the wrong units relative to `phi`.

**Practical consequence.** If conditioning is ever a concern, nondimensionalise the unknowns: scale the `chi` block by something proportional to `nn` (equivalently `1/Δs`). This is free, exact, and captures everything that node motion could have offered. It buys a factor of about 2.6 and does not change the growth rate.

**Why node motion cannot do better.** In `assemble_matrix` the weights `delt` enter through `c1[j]`, `c2[j]`, `c3[j]`, `depi4[j]`, all indexed by the *source column* `j`. Moving nodes therefore acts on the matrix largely as a diagonal column scaling — precisely the thing equilibration already does for free.

### 4.3 On a circle, uniform nodes are already the optimum

Perturbation test, `nn = 120`, nodes `θ → θ + ε·sin 2θ`:

| ε | κ | κ after block scaling | rel. error of qext vs Mie |
|---|---|---|---|
| 0.00 | 9.84e1 | 3.77e1 | 3.642e-3 |
| 0.02 | 1.02e2 | 3.91e1 | 3.654e-3 |
| 0.05 | 1.09e2 | 4.17e1 | 3.681e-3 |
| 0.10 | 1.22e2 | 4.73e1 | 3.752e-3 |
| 0.20 | 1.64e2 | 6.57e1 | 3.980e-3 |
| 0.30 | 2.53e2 | 1.06e2 | 4.318e-3 |

Both κ and the error rise monotonically from ε = 0, in both directions.

Direct optimisation, `nn = 60`, Powell over 59 free spacing parameters (log-spacing parametrisation, normalised to 2π, rotational freedom pinned), objective `log κ`, started from uniform: **the optimiser does not move.** Minimum and maximum node spacing both return 0.1047 against a uniform value of 0.1047; κ unchanged to six digits.

This is a genuine minimum, not a flat landscape — which is the good news about the premise, and also the end of it for the canonical validation case.

### 4.4 Non-circular shapes: κ moves, but only by a constant factor

Gielis `m=6` star, TE, 600 nm:

| nn | uniform arc length: κ | block-scaled | uniform theta: κ | block-scaled |
|---|---|---|---|---|
| 100 | 1.64e2 | 4.93e1 | 1.15e2 | 3.25e1 |
| 200 | 3.36e2 | 9.62e1 | 2.06e2 | 5.28e1 |
| 400 | 6.88e2 | 1.97e2 | 4.11e2 | 1.03e2 |

Node choice moves κ by a factor of roughly 1.7–1.9. Against a quantity growing linearly in `nn`, this is a constant-factor adjustment, and it is dwarfed by the accuracy considerations in Section 5.

### 4.5 Hard constraints from the QNM and sensitivity features

Two further reasons not to make the node set depend on λ or on a shape parameter.

**Holomorphy (Beyn's method).** `QNMSolver` finds modes by contour integration of `M(λ)`, which requires `M(λ)` to be holomorphic inside the contour — an assumption the code already protects with its `Re λ > 0` assertion. The smallest singular value of a matrix is not an analytic function of λ. If node angles were chosen by minimising κ(M(λ)), then `θ*(λ)` would be non-analytic and `M(λ; θ*(λ))` would lose holomorphy, invalidating the contour integral *silently*. Separately, a quasi-normal mode is by definition a λ where `M` is singular, so κ → ∞ at every target; minimising κ inside a search box is minimising the quantity whose blowup defines the answer.

**Frozen node set (conventions §10).** `QNMResult.sensitivity` requires the node set to be identical across the finite difference `M(p₀−h)` / `M(p₀+h)`. A κ-optimal `θ*` that depends on the shape parameter would re-place nodes between the two evaluations, reintroducing exactly the O(h) term that freezing eliminated (measured convergence rate on `∂M/∂b`: 2.7 without freezing, 100.1 with).

If any κ-informed placement is ever adopted, it must be computed once at a reference λ and frozen. The v0.5 `Geometry.gielis(theta=...)` API is already the right shape for that.

---

## 5. Investigation 2: the actual accuracy ceiling

### 5.1 First-order convergence, on a circle, with uniform nodes

| nn | rel. error of qext vs Mie | rate |
|---|---|---|
| 30 | 1.444e-2 | — |
| 60 | 7.288e-3 | 0.99 |
| 120 | 3.642e-3 | 1.00 |
| 240 | 1.819e-3 | 1.00 |
| 480 | 9.092e-4 | 1.00 |

Textbook first order. On a circle the nodes are exactly uniform, so node placement is definitionally not a variable. This matches the first-order rates the README already reports for the self-Green anchor, the QNM positions, and `dλ/dp`. One cause, and it is the kernel quadrature.

### 5.2 Correcting the self-patch constant is not enough

The exact log weight on an equispaced grid can be derived in closed form and compared with the `2e` constant. The correct constant is `π·e^γ ≈ 5.5951` against the current `2e ≈ 5.4366`. Measured:

| nn | constant `2e` | constant `π·e^γ` |
|---|---|---|
| 60 | 7.288e-3 | 5.830e-3 |
| 120 | 3.642e-3 | 2.916e-3 |
| 240 | 1.819e-3 | 1.457e-3 |
| 480 | 9.092e-4 | 7.282e-4 |
| rate | 1.00 | 1.00 |

A 20 % constant-factor improvement at every resolution, and no change in order. The O(h) error is distributed across the near-diagonal entries, not localised at the diagonal. Only a scheme that changes *every* entry of the log part removes it.

This is worth knowing as a one-line interim change, but it is not the fix.

### 5.3 The fix, measured

| nn | as shipped | Kress on all four blocks | Kress + analytic `ddf`/`ddg` |
|---|---|---|---|
| 30 | 1.444e-2 | 6.253e-5 | 3.36e-15 |
| 60 | 7.288e-3 | 7.829e-6 | 2.52e-15 |
| 120 | 3.642e-3 | 9.790e-7 | 1.68e-15 |
| 240 | 1.819e-3 | 1.224e-7 | 1.85e-15 |
| rate | 1.00 | 3.00 | machine precision |

Intermediate data point: splitting only the single-layer blocks M2/M4 and leaving M1/M3 alone already gives rate 3.00 (1.896e-5 at nn=60). Splitting the double-layer blocks as well improves the constant by about 2.4× but not the rate — the residual third-order term is the finite-difference second derivative, which is why the third column exists.

---

## 6. Kress–Martensen product quadrature

### 6.1 The idea

The periodic trapezoid rule is spectrally accurate on smooth periodic integrands and loses that property entirely in the presence of a logarithmic singularity. Patching one entry does not restore it, because the error is an area error spread over the neighbourhood of the singularity.

Kress's scheme instead splits the kernel so that everything sampled is analytic, and the singular factor — which is universal, the same for every geometry, wavelength and material — is integrated exactly against a trigonometric interpolant.

Write

```
K(t, τ) = K₁(t, τ) · ln( 4 sin²((t − τ)/2) ) + K₂(t, τ)
```

with `K₁` and `K₂` both analytic including on the diagonal. The factor `4 sin²((t−τ)/2)` is the 2π-periodic stand-in for `(t − τ)²`.

Then:

- `∫ K₂ φ dτ` — smooth periodic, use the trapezoid rule with weight `h = 2π/N`. Spectral.
- `∫ K₁ ln(...) φ dτ` — interpolate `K₁φ` by a trigonometric polynomial through the same N nodes, and integrate that polynomial against the log analytically, using the closed-form Fourier integral

```
∫₀^{2π} ln( 4 sin²((t − τ)/2) ) · e^{imτ} dτ  =  −(2π/|m|) e^{imt}   for m ≠ 0,   and 0 for m = 0.
```

### 6.2 The weights

With `N = 2n` equispaced nodes `t_j`, the resulting weights are

```
R_j(t) = −(2π/n) [ Σ_{m=1}^{n−1} (1/m) cos( m (t − t_j) ) ]  −  (π/n²) cos( n (t − t_j) )
```

and

```
∫ K₁(t,τ) ln(4 sin²((t−τ)/2)) φ(τ) dτ  ≈  Σ_j R_j(t) K₁(t, t_j) φ(t_j)
```

Reference implementation:

```python
def kress_R(N):
    """R_m for m = 0..N-1. N must be even, N = 2n. Circulant: index by (i-j) % N."""
    n = N // 2
    d = 2*np.pi*np.arange(N)/N
    p = np.arange(1, n)
    return -(2*np.pi/n) * ((np.cos(np.outer(d, p))/p).sum(1) + np.cos(n*d)/(2*n))
```

Three properties worth noting:

- `R` depends only on `t − t_j`, so it is **circulant**: N distinct values, not N². Index with `(i − j) % N`.
- It is independent of geometry, wavelength, material and refractive index. Compute once per `nn` and cache for the life of the process. This matters enormously for QNM work, where `M(λ)` is reassembled `4 · n_quad_per_side` times per contour.
- It requires **equispaced nodes in the parameter**. This is the one real constraint the scheme imposes; see Section 7.

### 6.3 The splittings for this code's conventions

Derived for `pysie2d`'s specific normalisation, in which the `chi` unknown carries the Jacobian so no explicit `|x'(τ)|` appears in the kernel. Let `r = |x(t) − x(τ)|`, `γ = |x'(t)| = sqrt(df² + dg²)`, `L = ln(4 sin²((t−τ)/2))`, and `γ_E = 0.5772156649…` (Euler's constant).

**Single-layer blocks (M2, M4).** The θ-kernel is `0.25i · H₀^{(1)}(k r)`.

```
A₁(t,τ) = − J₀(k r) / (4π)
A₂(t,τ) = 0.25i · H₀^{(1)}(k r)  −  A₁(t,τ) · L

A₁(t,t) = − 1/(4π)
A₂(t,t) = 0.25i  −  ( γ_E + ln( k·γ / 2 ) ) / (2π)

entry[i,j] = A₁[i,j] · R[(i−j) % N]  +  A₂[i,j] · h
```

M4 additionally carries the `eta` factor and uses `wnum_core`.

**Double-layer blocks (M1, M3).** The θ-kernel is `0.25i · k² · arg2 · H₁^{(1)}(k r)/(k r)`, where
`arg2[i,j] = (f_i − f_j)·dg_j − (g_i − g_j)·df_j`.

```
B₁(t,τ) = − (k / (4π)) · arg2 · J₁(k r) / r
B₂(t,τ) = 0.25i · k² · arg2 · H₁^{(1)}(k r)/(k r)  −  B₁(t,τ) · L

B₁(t,t) = 0                                  (arg2 vanishes quadratically)
B₂(t,t) = − deriv / (4π · γ²)                (the existing curvature limit)

M1[i,j] = B₁[i,j]·R[(i−j) % N] + B₂[i,j]·h  +  0.5·δ_ij
M3[i,j] = B₁[i,j]·R[(i−j) % N] + B₂[i,j]·h  −  0.5·δ_ij
```

with `deriv = df·ddg − ddf·dg`, matching the existing code.

Sanity check on the existing diagonals: at `i = j`, `B₁ = 0`, so `M1[i,i] = 0.5 − deriv·h/(4πγ²)`, which is exactly the shipped `d_m1`. The double-layer diagonals are therefore already correct; only their off-diagonal log content changes.

**Note on the Helmholtz double layer.** Unlike Laplace, the 2-D Helmholtz double-layer kernel is *not* free of a logarithmic part on a smooth curve. `B₁` vanishes quadratically at the diagonal, so `B₁·L` is C¹ but not C², which caps trapezoid convergence at a finite algebraic order. This is a smaller effect than the single layer (rate 3 either way in the measurements above) but should be done for completeness.

### 6.4 Analytic second derivatives

Once the quadrature is fixed, the next term is `_der_real_3`. It computes `ddf` and `ddg` by three-point finite differences on the theta grid, giving O(h²) relative error in the curvature, which enters the M1/M3 diagonals and caps the overall rate at three.

The Gielis superformula is differentiable in closed form. `_rderiv` already computes `dr/dθ` analytically; the second derivative follows the same way. Supplying analytic `ddf`, `ddg` alongside the Kress quadrature takes the circle to machine precision at `nn = 30`.

Verification used for the circle case (`f = r sinθ`, `g = r cosθ` with `r` constant):

```python
geom.ddf = -RAD * np.sin(geom.theta)
geom.ddg = -RAD * np.cos(geom.theta)
```

For the general Gielis case, `ddf` and `ddg` require `d²r/dθ²`, obtainable by differentiating the existing `_rderiv` expression. This is a self-contained piece of work with an obvious test: compare against `_der_real_3` and confirm the difference decays as O(h²).

---

## 7. Implementation notes and constraints

### 7.1 Equispaced nodes are required

The Kress weights come from trigonometric interpolation, which presumes an equispaced grid in the quadrature parameter. This is incompatible with the current uniform-arc-length node placement *as implemented*, where nodes are non-uniform in θ.

There are two clean resolutions:

**(a) Use the uniform-theta path.** `boundary_setup(..., arc_length=False)` already places nodes at `(j + 0.5)·2π/nn`. Kress applies directly. The half-step offset is harmless because `R` depends only on differences. This is the minimal-change route and is what the prototype used.

**(b) Grade through an analytic substitution.** If grading is wanted for star shapes, introduce a smooth, analytic, 2π-periodic reparametrisation `w(t)` with `w'(t) ≥ 0`, place nodes equispaced in `t`, and evaluate the geometry at `w(t_j)`. The grading lives in `w`, the quadrature stays equispaced, and spectral accuracy is preserved. Kress's sigmoid substitution is the canonical example (it is normally used to resolve corner singularities, but the same machinery grades toward high-curvature lobes).

This is worth emphasising as the principled answer to the original node-placement question: grading *is* legitimate and useful, but the mechanism is a smooth analytic change of variables, not a discrete optimisation over node positions, and the nodes must remain equispaced in the new parameter.

### 7.2 Cost

Assembly cost is unchanged in order. It gains:

- One extra Bessel evaluation array per block (`J₀` or `J₁` alongside the existing `H₀`/`H₁`). For real arguments this is cheap — `J₀`/`J₁` are exactly the Cephes kernels the existing fast path already uses, and `H₀ = J₀ + iY₀`, so `J₀` is available for free if the fast path is restructured to return both parts.
- One `log(4 sin²(Δt/2))` array, `O(nn²)`, computed once per geometry and reusable across all λ.
- One circulant `R` lookup, `O(nn)` to build, cached.

Against this, `nn` can drop by an order of magnitude for the same accuracy, so the solve cost (`O(nn³)`) falls by three orders. Net effect is strongly positive.

### 7.3 What must not change

- **Holomorphy in λ.** Every quantity above depends on λ analytically: `J₀`, `J₁`, `H₀^{(1)}`, `H₁^{(1)}` are entire or holomorphic in the relevant domain, and `R`, `L`, `h` are λ-independent. Beyn's contour method remains valid. Confirm this explicitly in the tests.
- **Complex-wavenumber support.** The prototype used `scipy.special.jv(0, z)` and `jv(1, z)` which accept complex arguments; the existing `_real_if_real` demotion trick should be extended to the new Bessel calls so the real-argument fast path is preserved.
- **Scale covariance (conventions §9).** `R` is dimensionless and depends only on `nn`, so §9 is untouched.
- **Frozen node set (conventions §10).** Unaffected; if anything the equispaced requirement makes the frozen node set easier to guarantee.
- **`assemble_matrix_dwn` parity.** The fused matrix-plus-derivative routine duplicates `assemble_matrix` expression for expression and is guarded by `test_matrix_derivative_matches_assembly` with `np.array_equal`. Both must change together. The new `dM/dk` terms follow from the same Bessel identities already documented there, plus `dJ₀/dk = −r J₁(kr)` and `dJ₁(z)/dz = J₀(z) − J₁(z)/z`.

### 7.4 Suggested sequencing

1. Add `kress_R(N)` with a unit test against the analytic value of `∫ ln(4sin²((t−τ)/2)) cos(mτ) dτ`.
2. Implement the single-layer splitting for M2/M4 behind a flag, uniform-theta nodes only. Expect rate 3 on the circle.
3. Extend to M1/M3.
4. Replace `_der_real_3` with analytic second derivatives. Expect machine precision on the circle.
5. Re-anchor every existing tolerance in `tests/` — most will be enormously slack, and the QNM and sensitivity tests may no longer need `richardson_limit` at all.
6. Then, and only then, revisit graded parametrisations for star shapes (Section 7.1b).

---

## 8. Validation plan

The circle case with analytic Mie is the primary anchor and is already decisive. Beyond it:

- **Optical theorem** on a lossy particle and **energy conservation** on a lossless one — existing tests, should tighten by orders of magnitude.
- **Self-Green function** against the closed-form Graf-addition-theorem sum. Currently converges at first order and is run at `nn = 1000` to reach 1 %. Expect this to become trivially cheap; it is the strongest near-field anchor in the suite and the best evidence the splitting is right near the boundary.
- **QNM positions** against analytic Mie poles. Currently first order in `nn`, in both `Re λ` and `Im λ`. This is also the test that will confirm holomorphy is preserved — if the contour integral silently degrades, the mode count or `edge_margin` will show it.
- **`dλ/dp` sensitivity.** Currently first order, needing `richardson_limit`. Check whether extrapolation is still required.
- **Parity between `assemble_matrix` and `assemble_matrix_dwn`** must continue to hold bit-for-bit.
- **A Gielis star self-convergence study** on the uniform-theta path, to confirm the improvement survives away from the circle before any graded parametrisation work begins.

---

## 9. Open item: the uniform-arc-length anomaly on star shapes

Unresolved, deferred by decision. Recorded here so the trail is not lost.

On the Gielis `m=6` star (`n1=6, n2=12, n3=12`), uniform arc-length nodes converge markedly worse than uniform-theta nodes:

| nn | uniform arc length | uniform theta |
|---|---|---|
| 100 | 1.29e-4 | 7.55e-3 |
| 200 | 7.75e-4 | 9.18e-4 |
| 400 | 5.13e-4 | 2.03e-4 |
| 800 | 2.70e-4 | 6.14e-5 |
| rates | −2.59, 0.59, 0.93 | 3.04, 2.18, 1.72 |

(relative error of `qext` against a uniform-theta `nn = 2400` reference; the arc-length column also stagnates against its own `nn = 2400` reference, so this is not an artefact of the reference choice)

The arc-length column is non-monotone and stagnates near 3e-4. Since uniform arc length exists precisely to beat uniform theta on star shapes, this is backwards.

**Ruled out:**

- *Inversion grid resolution.* Raising `n_fine` from `10·nn` to `160·nn` changes the result by less than one part in 10⁴ at every resolution tested.
- *The weight formula.* Replacing the forward difference `delt_j = θ_{j+1} − θ_j` with the centred `(θ_{j+1} − θ_{j−1})/2` gives no improvement, despite the centred version producing visibly more uniform arc elements (relative spread of `γ·delt`: 0.027 vs 0.073 at `nn = 200`).

**Observed but unexplained:** `γ·delt` should be exactly constant for uniform-arc-length nodes. Its relative spread is 7.3e-2, 3.8e-2, 1.9e-2 at `nn = 200, 400, 800` — halving with each doubling, i.e. a first-order relative error in the effective element lengths. This is a real defect in the arc-length path; whether it is the *cause* of the stagnation is not established, since the centred-weight fix reduces the spread without improving the error.

**Suggested next probes:** the piecewise-linear `np.interp` inversion produces a node map `θ(s)` that is only C⁰, which would break the smooth-change-of-variables argument that the trapezoid rule's accuracy relies on — a C² or spectral inversion (cubic spline, or Newton on the exact arc-length integral of the analytic `γ`) would test this. Note that Section 7.1 may make the question moot: if grading moves to an analytic substitution with equispaced nodes, the current arc-length inversion path disappears.

---

## 10. Reference implementation

Complete working prototype for the circle, verified to reach 3.4e-15 at `nn = 30`.

```python
import numpy as np
from scipy.special import jv
from pysie2d.kernels import hank0, hank1

TWOPI, GE = 2*np.pi, np.euler_gamma

def kress_R(N):
    n = N // 2
    d = TWOPI*np.arange(N)/N
    p = np.arange(1, n)
    return -(TWOPI/n)*((np.cos(np.outer(d, p))/p).sum(1) + np.cos(n*d)/(2*n))

def demote(k):
    kc = complex(k)
    return kc.real if kc.imag == 0.0 else kc

def kress_matrix(geom, k_bg, k_co, N, eta=1.0):
    f, g, th = geom.f, geom.g, geom.theta
    df, dg = geom.df, geom.dg
    gam = np.sqrt(df**2 + dg**2)
    deriv = df*geom.ddg - geom.ddf*dg

    dx, dz = f[:, None] - f[None, :], g[:, None] - g[None, :]
    r = np.sqrt(dx**2 + dz**2)
    np.fill_diagonal(r, 1.0)                       # placeholder; diagonals set below
    arg2 = dx*dg[None, :] - dz*df[None, :]

    s2 = 4.0*np.sin((th[:, None] - th[None, :])/2.0)**2
    np.fill_diagonal(s2, 1.0)
    L = np.log(s2)

    R = kress_R(N)[(np.arange(N)[:, None] - np.arange(N)[None, :]) % N]
    h = TWOPI/N

    def single(k):
        k = demote(k)
        A1 = -jv(0, k*r)/(4.0*np.pi)
        A2 = 0.25j*hank0(k*r) - A1*L
        np.fill_diagonal(A1, -1.0/(4.0*np.pi))
        np.fill_diagonal(A2, 0.25j - (GE + np.log(k*gam/2.0))/(2.0*np.pi))
        return (A1*R + A2*h).astype(complex)

    def double(k):
        k = demote(k)
        B1 = -(k/(4.0*np.pi))*arg2*jv(1, k*r)/r
        D  = 0.25j*k**2*arg2*hank1(k*r)/(k*r)
        B2 = D - B1*L
        np.fill_diagonal(B1, 0.0)
        np.fill_diagonal(B2, -deriv/(4.0*np.pi*gam**2))
        return (B1*R + B2*h).astype(complex)

    M = np.zeros((2*N, 2*N), dtype=complex)
    M[:N, :N] = double(k_bg) + 0.5*np.eye(N)
    M[:N, N:] = single(k_bg)
    M[N:, :N] = double(k_co) - 0.5*np.eye(N)
    M[N:, N:] = eta*single(k_co)
    return M
```

Caveats on this snippet: it assumes `eta = 1` handling as shown is correct for TM as well (verify against `assemble_matrix`, where `eta` multiplies only the M4 block); it assumes `geom.theta` is equispaced; and `np.fill_diagonal(r, 1.0)` is a placeholder that is safe only because every diagonal is overwritten afterwards.

---

## 11. References

- R. Kress, *Linear Integral Equations*, 3rd ed., Springer 2014 — Chapter 12 for the trigonometric-interpolation quadrature and the `R_j` weights.
- D. Colton and R. Kress, *Inverse Acoustic and Electromagnetic Scattering Theory*, 4th ed., Springer 2019 — §3.5 for the Helmholtz single- and double-layer splittings and the diagonal limits, and for the sigmoid substitution used for graded parametrisations.
- E. Martensen, *Über eine Methode zum räumlichen Neumannschen Problem mit einer Anwendung für torusartige Berandungen*, Acta Math. 109 (1963) 75–135 — the original product-quadrature idea.
- A. A. Maradudin, T. Michel, A. R. McGurn, E. R. Méndez, *Enhanced backscattering of light from a random grating*, Ann. Phys. 203 (1990) 255–307 — the scheme `pysie2d` currently implements.
- L. N. Trefethen and J. A. C. Weideman, *The exponentially convergent trapezoidal rule*, SIAM Review 56 (2014) 385–458 — the background on why the periodic trapezoid rule is spectral and exactly what breaks it.
- C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles*, ch. 8 — the analytic Mie reference already used by `pysie2d.reference.mie`.

---

## Appendix: measurement scripts

The experiments in Sections 4 and 5 were produced by six standalone scripts, all of which import the shipped package unmodified except where noted:

| Script | Produces |
|---|---|
| `exp_kappa.py` | Section 4.1, 4.2, 4.4 — κ, equilibrated κ, block-scaled κ |
| `exp_decisive.py` | Section 4.3 — perturbation sweep and Powell optimisation vs Mie |
| `exp_diag.py` | Section 5.2 — self-patch constant comparison (patches a copy of `kernels.py`) |
| `exp_kress.py` | Section 5.3 — Kress on M2/M4 only |
| `exp_kress_full.py` | Section 5.3 — Kress on all four blocks |
| `exp_weights2.py`, `exp_inversion.py` | Section 9 — the star anomaly and the hypotheses ruled out |

Key plumbing detail for anyone reproducing the matrix-override tests: to solve with a replacement matrix while reusing the package's right-hand side and far-field machinery, recover the RHS by multiplication rather than solving —

```python
base = BIESolver(geom, mat).scatter(wavelength=LAM)
rhs  = BIESolver(geom, mat).assemble(LAM) @ base.ei
ei   = np.linalg.solve(M_new, rhs)
result = pysie2d.solver.ScatterResult(ei, geom, mat, LAM, base.angle)
```
