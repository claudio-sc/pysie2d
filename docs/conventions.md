# Conventions

Every gotcha in this solver traces back to one of the conventions below. Read
them before reading the code.

## 1. 2-D scalar problem

The geometry is invariant along the cylinder axis (`y`), so Maxwell's equations
reduce to a scalar Helmholtz equation for one field component. The polarisation
is selected by the integer `Material.pol`:

- `pol = 2` → **TE**: the scalar field is `E_y` (E parallel to the axis).
- `pol = 1` → **TM**: the scalar field is `H_y`.

These integer codes are kept for fidelity with the original formulation. Never
change their internal meaning.

Mapping to analytic Mie coefficients (Bohren & Huffman ch. 8):

- `pol = 2` (TE, `E_y`) ↔ `b_n` ↔ efficiency keys `Q_*_TE`.
- `pol = 1` (TM, `H_y`) ↔ `a_n` ↔ efficiency keys `Q_*_TM`.

## 2. Units, wavelength, and the background index

Lengths are in **nanometres**, everywhere.

### 2.1 Public wavelengths are vacuum wavelengths

Every wavelength on the public façade — `BIESolver.scatter`, `.scatter_dipole`,
`.assemble`, `ScatterResult.wavelength`, `self_green`, `relative_ldos`,
`relative_ldos_map`, and the `QNMSolver.modes` search rectangle — is a **vacuum**
wavelength `λ_vac` in nm. There is no second reading anywhere on that surface, so
the parameter is named plainly `wavelength`.

`Material.n_core` and `Material.n_clad` are independent **absolute** refractive
indices. `Material.epsi` is likewise an **absolute** imaginary permittivity,
referred to vacuum exactly like `n_core`.

### 2.2 Internals are background-relative

The operator is non-dimensionalised to a background of index 1. It sees only

- the **background wavenumber** `k_bg = 2π·n_clad/λ_vac` (rad/nm), and
- background-relative material quantities: `Material.nc = n_core/n_clad` (the
  `m` of Mie theory) and `Material.eps = (n_core² + i·epsi)/n_clad²`.

Because `epsi` is absolute, making it relative divides it by `n_clad²` — the
same factor that turns `n_core²` into `Material.epsr`. `Material.epsi_rel`
exposes that value.

### 2.3 One conversion point, and why the primitives take no wavelength

`Material.wnum_bg(λ_vac)` is the **only** place the background index enters a
wavenumber. Every low-level primitive — `assemble_matrix`,
`assemble_matrix_reference`, `eval_field`, `far_field`, `plane_wave_rhs`,
`line_dipole_rhs`, and `reference.mie.self_green_cylinder` — takes `wnum_bg`
and **no wavelength at all**.

That is deliberate, and it is the invariant to protect. The façade methods call
each other (`scatter` → `assemble`, `relative_ldos_map` → `assemble`,
`self_green` → `scatter_dipole`), so if two of them each converted a wavelength
the factor `n_clad` would be applied twice. With no wavelength below the façade
there is nothing to convert twice. A custom `incident_rhs` callable therefore has
signature `(nn, wnum_bg, f, g) → complex (2·nn,)`.

At `n_clad = 1` the vacuum and background readings coincide, which is where the
entire fixture suite runs — `tests/test_conventions.py` is what distinguishes
them, in both directions.

### 2.4 Size parameter

    x = k_bg·a = 2π·n_clad·rad/λ_vac

referred to the cladding, matching Mie theory and every function in
`pysie2d.reference.mie`. It is **derived only** — `pysie2d.size_parameter`,
`ScatterResult.size_parameter`, `QNMResult.size_parameters` — and never an
input, for the same reason complex frequency is not an input (§8): a second
entry point lets the two disagree, and `x` additionally depends on the geometry.

It is defined only for a **circular** boundary: on a non-circular Gielis shape
`Geometry.rad` is a scale parameter with no single physical radius behind it, so
`size_parameter` raises rather than returning a meaningless number.
`Geometry.is_circle` is the test, and it is numerical — several Gielis parameter
sets (`m = 0`, or `n1 = n2 = n3 = 2` at any `m`) produce a genuine circle.

## 3. Time convention

The time convention is `exp(-iωt)`. Outgoing waves are therefore Hankel
functions of the **first** kind, `H_n^{(1)}`. If a validation matches only
after complex conjugation, that is a convention clash in the *reference*, not
a bug in the solver.

## 4. Solution-vector layout

A BIE solve returns `ei` of shape `(2·nn,)`, where `nn` is the number of
boundary quadrature points:

- `ei[:nn]` — `φ`: boundary field values.
- `ei[nn:]` — `χ`: boundary normal-derivative values.

Excitation right-hand sides follow the same layout. The plane-wave and (later)
line-dipole sources populate only the `φ` half; the `χ` half stays zero.

## 5. Geometry arrays

`Geometry` holds the boundary coordinates and their derivatives with respect to
the boundary parameter `θ`:

- `f` — x-coordinates; `g` — z-coordinates. **`g` is a coordinate array, not a
  Green function** (an unfortunate historical name).
- `df`, `dg` — first derivatives of `f`, `g` w.r.t. `θ`.
- `ddf`, `ddg` — second derivatives.
- `delt` — the quadrature `θ`-step: a scalar for uniform-`θ` sampling, or a
  per-point array for uniform arc-length sampling.

## 6. Complex wavenumbers are supported deliberately

Every matrix-assembly and field-evaluation path accepts a complex wavenumber.
This is intentional: it is what makes quasi-normal-mode extraction (a planned
extension) possible. Do not "simplify" any code path to real-only arithmetic,
even where it looks like you could.

Real-argument *fast paths* are permitted, provided the complex fallback stays
intact. `kernels.hank0` / `hank1` / `cbesh` branch at runtime on the argument
dtype: real arguments use the Cephes `j0/y0/j1/y1` identity
`H_n^{(1)}(x) = J_n(x) + i·Y_n(x)` (11–13× faster, agreeing to 4e-15), complex
arguments go to `scipy.special.hankel1` exactly as before. `_real_if_real`
demotes an exactly-real complex scalar (e.g. `Material.nc = 2+0j`) to a float so
the branch can trigger; a genuinely complex `wnum_bg`, `ri`, or `wnum_core`
passes through untouched. Any change that removes the ability to pass a complex wavenumber is
wrong, fast path or not.

## 7. Self-Green function sign convention (v0.2)

The self-Green function `S(r_s, r_s, ω)` is the *scattered* field from a
line-dipole source evaluated back at the source point. The literature genuinely
differs on the sign convention for the scattered field, so the analytic
reference `reference.mie.self_green_cylinder` carries an explicit global sign:

    c_n = SIGN · b_n  (TE, pol=2)   /   c_n = SIGN · a_n  (TM, pol=1)

`SIGN = -1` was pinned by matching the BIE solver against the analytic Graf
addition-theorem sum (test `test_self_green_vs_analytic_cylinder`): with this
sign the two agree on both `Re S` and `Im S` across `d ∈ {1.2a … 3a}` and both
polarisations. The reciprocity, free-space-limit, and LDOS-positivity tests
triangulate the rest, so the solver computes the physical Green function up to
this one documented convention.

The LDOS is normalised to the **homogeneous background**, not to vacuum. This
distinction was vacuous while every case ran at `n_clad = 1` and is now
load-bearing: at `n_clad ≠ 1` a Purcell factor of 1 means "as in the unbounded
cladding", not "as in vacuum". The normalisation uses `Im[g₀(r→r)] = 1/4` (the log
divergence of `H₀^{(1)}` lives in its imaginary part; with the `i/4` prefactor
the imaginary part of `g₀` tends to `J₀(0)/4 = 1/4`), giving
`relative_ldos = 1 + 4·Im(S)`.

## 8. The quasi-normal-mode half-plane (v0.4)

A quasi-normal mode is a source-free solution: a complex wavelength where
`M(λ)` is singular. `QNMSolver.modes` searches a **rectangle in complex λ**,
matching the driven API's argument. Complex frequency is a documented
conversion, `ω = 2πc/λ`, not a second entry point.

Under `exp(-iωt)` (§3) a decaying mode has `Im ω < 0`, hence `Im k < 0`, hence

    Im λ > 0.

A search box must therefore lie strictly in `Im λ > 0`, and strictly in
`Re λ > 0` — the latter keeps every Hankel argument off the `H^{(1)}` branch cut
on the negative real axis, which is what makes `M(λ)` holomorphic on the
rectangle. Holomorphy is the premise of the contour argument, not a detail, so
both bounds are asserted rather than documented.

The quality factor is

    Q = Re λ / (2 Im λ),   exactly equal to −Re ω / (2 Im ω).

**Poles do not occur in conjugate pairs.** The reality condition is `λ → −λ̄`,
which places mirror partners at negative `Re λ`, outside the physical region.
Carrying real-eigenvalue intuition into this non-Hermitian problem and expecting
`λ̄` to be a mode is the natural mistake; `test_no_conjugate_pair_symmetry` is
the guard.

**Degeneracy is structural for a circle.** Every `n ≥ 1` mode is doubly
degenerate through `exp(±inθ)`; only `n = 0` is simple. Degenerate partners are
reported as separate entries with `multiplicity = 2`, never collapsed — the pair
carries two independent mode vectors, and merging them would make the count
disagree with the analytic table. Note also the identity
`D^{TM}_0 ≡ D^{TE}_1`, so those two families land on the same wavelengths; this
is not a hazard, because `M(λ)` is assembled per polarisation.

Mode **vectors** are exposed raw, in the `φ`/`χ` layout of §4. They are not
normalised as mode fields — that needs a QNM norm, which is out of scope.

Search rectangles and mode wavelengths are both **vacuum** wavelengths (§2),
with no conversion on the return leg: the contour is drawn directly on
`BIESolver.assemble`, which is itself the single vacuum-to-background conversion
point, so the eigenvalues come back in the coordinate the box was given in.

`QNMResult.size_parameters` exposes the modes in the analytic anchor's
coordinate. Note that a rectangle in `x` is **not** a rectangle in `λ`:
`λ = 2π·n_clad·rad/x` is a Möbius map and does not carry corners to corners. A
completeness argument must be made in the coordinates the box is drawn in.

## 9. Scale covariance (v0.4.2)

**`M` depends on `rad` and `λ` only through the dimensionless ratio
`k_bg·rad`** — the size parameter of §2.4 when the boundary is a circle, and on
any other Gielis shape only a ratio, since §2.4's `x` needs a single physical
radius that a star does not have. Every length and every wavenumber in
`assemble_matrix` appears in one of exactly four combinations, each of total
degree zero under `rad → s·rad`, `λ → s·λ`:

    k·r                 all off-diagonal Hankel arguments
    k·delt/(2e)·gamma   both singular diagonals
    k²·cij              c1 and c3 against the boundary cross products
    deriv/gamma²        the M1 and M3 diagonals

`delt` and the θ-nodes are degree 0. That is true on the arc-length path too,
and for a better reason than "the inversion is accurate": the chord-length arc
estimate in `_uniform_arc_theta` is inexact *as an arc length* but exactly
homogeneous of degree 1 in `rad`, and `np.interp` is homogeneous of degree 0 in
its query and table jointly. Covariance needs the homogeneity, not the
accuracy. `n_fine` depends on `nn` alone, and must keep doing so — choosing it
from an absolute chord length in nm would break this silently.

Hence, entrywise and at any `n_pts`:

    M(s·rad, s·λ) = M(rad, λ),      ∂M/∂rad = −(λ/rad)·∂M/∂λ

and therefore `λ(s·rad) = s·λ(rad)`, `dλ/drad = λ/rad`, `dQ/drad = 0`. The
adjoint form `dλ/dp = −uᴴ(∂M/∂p)v / uᴴ(∂M/∂λ)v` returns `λ/rad` for **any**
`u, v`, so the result is gauge-free; on the semisimple `±n` pair the 2×2
secular problem is a multiple of the identity, so both partners share it in any
null-space basis and **a dilation can never split a degeneracy**.

Three things this does *not* say. It is not accuracy: the discrete pole sits at
a fixed `x_disc(n_pts) ≠ x_Mie`, so covariance is exact while the wavelength is
still wrong in the first decimal. It is largely not a convention check: signs
and the `H^{(1)}` choice are scale-free and wholly invisible to it, and a `pol`
swap is caught only indirectly, by moving the poles out of the search boxes and
tripping the mode counts. And it holds only for a **non-dispersive** material —
`ri` and `kd` are degree 0 only because `Material` holds constant indices, and
the day dispersion is added
`dQ/drad = 0` stops being true as physics at the same moment it stops being
true here.

`tests/test_scale_covariance.py` is the guard, at two scale ratios: a power of
two, where binary floating point makes bit-identity a theorem and the assertion
carries no tolerance at all, and a generic ratio, which is the only variant
that can fail from conditioning.

**The algebra holds for any boundary; the conditioning can fail on one that is
not C¹.** At a ratio with no exact binary representation the θ-nodes move by an
ulp. On a boundary with a corner — the superformula at exponent 1, say — a node
that sits numerically *on* the kink then jumps to the other one-sided tangent,
and `df, dg` and the cross product `cij` change by a finite amount, so the
matrix differs by O(1).

This is a knife edge and not a property of inexact ratios: *(measured on that
shape at `n_pts = 200`: 0.264 relative at s = 1.7 and at s = 0.61, but 3.2e-13
at s = 0.37 and 1.7e-13 at s = 3.0 — against 1.5e-13 for a smooth star at all
four, and bit-identity for the cusped shape itself at s = 2)*. It is a
statement about discretising a corner rather than about covariance, and it is
the reason a rough boundary handed to this solver is better C¹: not that it
will lose the covariance, but that it may.

**One exception, in `QNMResult.refine` / `newton_refine`.** `tol` is a Newton
step size in *absolute* nm, so it is the one scale-dependent quantity in the
QNM path: `step` scales with the radius and `tol` does not, and a step landing
between `tol` and `s·tol` stops the iteration at different points at the two
radii. It does bite, narrowly *(measured on the simple TE anchor at `s = 2`:
refined wavelengths bit-identical for `tol` = 1e-9, 1e-7, 1e-6, 1e-4, 1e-3,
1e-2, and differing by 7.1e-15 relative at `tol` = 1e-5, both radii reporting
`converged`)* — narrowly because a quadratically convergent step passes through
the marginal band only for a thin set of `tol`. The exact statement above is
therefore made on the unrefined `modes()` output.

## 10. Shape derivatives use a frozen node set (v0.5)

**A finite difference in a shape parameter holds the boundary node set fixed.**
`Geometry` stores `theta`, and `Geometry.gielis(..., theta=...)` builds a shape
on angles supplied from elsewhere instead of re-inverting arc length. Every
`∂M/∂p` takes `M(p₀±h)` on the θ of `p₀`, and so do `∂M/∂λ` and the left and
right null vectors that enter the adjoint quotient — all four on one node set,
or the quotient mixes two discretisations.

**Why it is not merely convenient.** Node placement is a **parametrisation
gauge**. The BIE discretises a boundary integral, and λ — the thing the adjoint
differentiates — does not depend on how the boundary was sampled. Re-inverting
arc length between the two evaluations differentiates the gauge along with the
physics, and the gauge is not differentiable: the inversion goes through
`np.interp`, which is continuous but only *piecewise* linear in the shape
parameter, so a node whose bracketing cell differs between `p₀−h` and `p₀+h`
contributes an O(1) error to the quotient.

The resulting term is O(h), not O(h²); it is not monotone in `h`; and it **grows
with `n_pts`**, because a finer boundary has more cells to cross. That last
property is why it cannot be refined away and had to be removed structurally.
*(measured, `docs/design/studies/shape-derivative-smoothness.md`: unfrozen, the
`h`-ladder on `∂M/∂b` falls 8.29e-5 → 3.24e-5 → 5.60e-9 — a stall then a cliff,
with the fraction of matrix entries carrying the deviation going 0.49 → 0.05 →
0.00, a decaying count rather than a decaying magnitude. Frozen, the same ladder
is 1.01e-5 → 1.01e-7 → 1.67e-9: exactly ×100 per decade in every parameter until
the cancellation floor.)*

Uniformity in arc length still drifts by O(h) across the difference. That is
accepted and is the point of `h` being small — the alternative is to
re-equidistribute, which is the error being removed. Over the larger parameter
steps of a continuation path the nodes **are** re-equidistributed per step, and
the resulting jitter is measured under Gate 7 rather than assumed away.

**Step size.** `h = 1e-5` in the parameter's own units, with the cancellation
floor at ~1e-8 and truncation at ~1e-7 a decade above, i.e. about a decade of
margin on each side. The margin, not the best value at one design point, is the
reason for the choice: the truncation coefficient scales with the parameter's
geometric leverage, which moves across the shape catalogue. **No second
derivatives** — the source of the O(h) term above is a kinked first derivative,
and freezing the nodes removes it from the difference quotient without making
the underlying inversion C².

**Two traps this pins.** A prescribed θ is validated for strict ordering and a
sub-2π span, because `delt` is a bare `np.diff`: a reordered set gives negative
quadrature weights and a boundary integral that counts part of the curve
backwards, with nothing raised. And `theta` is a **required** field rather than
an optional one, because an optional θ means a silent fallback to re-inversion
— which is exactly the failure being removed.

Frozen nodes preserve §9 exactly: a supplied θ carries no length, so
`M(s·rad, s·λ) = M(rad, λ)` entrywise still holds, and it is asserted on the
frozen path in `tests/test_scale_covariance.py`.

## 11. Adjoint eigenvalue sensitivity (v0.5)

`QNMResult.sensitivity(at, step=SHAPE_STEP)` returns `dλ/dp` for every mode in
the result, one parameter at a time, from

    dλ/dp = − uᴴ (∂M/∂p) v / [ uᴴ (∂M/∂λ) v ]

No eigenvalue is re-extracted: that is the whole point of the adjoint, and it
is what makes a Jacobian over seven parameters affordable.

**`at` is a callable, not a perturbed geometry.** Its signature is
`δ → (geometry, material)` at offset `δ` from the base point, `δ = 0` being the
base point itself. Both halves are returned because `n_core` and `n_clad` are
parameters of the same Jacobian as the shape ones, and one signature covering
all of them is what keeps the caller from having two code paths that can drift
apart. The **offset is in the parameter's own units**, so `step` and `dλ/dp`
are both in those units and the caller owns any reparametrisation — a `log`
gauge is a two-line lambda, and §9 says the answer must be gauge-free.

**The frozen node set is enforced, not documented.** The geometry returned by
`at` must carry `result.geometry.theta` **exactly**, and `sensitivity` raises
otherwise. Exact equality is the right test because there is no threshold at
which node motion becomes acceptable: the term it introduces is O(h) and grows
with `n_pts` (§10). On a circle re-inversion moves nodes by only 1.5e-13 rad,
which is precisely why a tolerance-based check would be the wrong instrument.

**`u` is a genuine left null vector**, obtained from the same SVD as `v` — the
smallest singular triplet of `M(λ)`, `U[:, -1]` and `V[:, -1]`. It is **not**
`conj(v)`: M is not complex-symmetric here *(measured: `‖M − Mᵀ‖/‖M‖ = 1.06`,
and `|⟨u, conj(v)⟩| = 0.32` at the TE n=0 pole of the reference circle)*, so
substituting `conj(v)` gives a quotient wrong by an O(1) factor with every
residual still looking right. Cost is one assembly plus one full SVD per mode,
0.080 s at `n_pts = 200`.

**Degenerate poles dispatch to a secular problem, they do not raise.** A k-fold
pole has a k-dimensional null space, and the scalar quotient would pick an
arbitrary vector out of it. The k derivatives are the eigenvalues of

    − (Uᴴ ∂_p M V) (Uᴴ ∂_λ M V)⁻¹

with `U`, `V` the smallest k singular triplets; this reduces to the quotient at
k = 1. Multiplicity is read from the **same** `DEGENERACY_RTOL` criterion that
`QNMResult.multiplicity` reports, so the branch taken can never contradict the
multiplicity printed beside it. Within a degenerate group the returned values
are sorted by `(Re, Im)`: which partner receives which derivative is not
defined, because the null basis is fixed only up to a k×k rotation.

**Anchors.** Gate 1 — `dλ/drad = λ/rad`, and in the linear gauge this is
*machine-exact*, since §9 makes λ exactly linear in `rad`, leaving only the
cancellation floor ~ε/h *(measured 5.1e-11)*. Gate 1 degenerate half — a
dilation cannot lift the ±n degeneracy of a circle, so the 2×2 secular matrix
is a multiple of the identity *(measured: off-diagonal/‖S‖ = 2.7e-11, splitting
1.3e-10)*. Gate 2 — at `n2 = n3`, `(log a + log b)` and `log rad` move λ
identically, so their difference is an exact null direction of `J` *(measured
ratio − 1 = 5.1e-14)*. Gate 3 — against central differences of independently
re-extracted Beyn poles, second order in the step *(3.816e-4 → 3.809e-6 →
3.816e-8, ratios 100.2 and 99.8)*. All four in
`tests/test_sensitivity.py`.

## 12. Jacobian accuracy is bought by extrapolation, not by resolution (v0.5)

`J = dλ/dp` converges at **first order in `n_pts`**, exactly like λ itself:
observed order 0.98–1.02 on every component, on three independent ladders
(`docs/design/studies/jacobian-convergence.md`). §9's argument that the fixed
`x_disc(n_pts) ≠ x_Mie` error is smooth in the shape parameter and cancels in
the ratio `∂λ/∂p` is **true of the constant and false of the order** — `dλ/drad`
is three decades better resolved than `dλ/db` at the same `R`, and neither
converges faster than `1/n_pts`. Do not read §9 as promising more than that.

Two consequences, both measured rather than argued:

**Differencing does not help.** `ΔJ` between two designs `δb = 0.02` apart
converges at order 0.94 and lands *further* from its limit than `J` does — 3×
further on `dλ/db`, 116× on `dλ/dn_core`. Subtracting two quantities whose
errors are the same size and only partly common-mode keeps the error and loses
the signal.

**Two-rung Richardson does.** `richardson_limit(coarse, fine, n_coarse, n_fine)`
implements `q* = q_f + (q_f − q_c)/(n_f/n_c − 1)`, exponent **pinned at 1, not
fitted**. From `R = 15 + 30` it puts every J component inside **6.4e-4** of the
limit at **0.46×** the cost of one `R = 50` rung — which is itself 1.5 % out and
misses the gate's 1 % bar. `R = 30 + 50` is the fallback, 4× more margin at 3×
the cost.

`n_fine` must exceed `n_coarse` and the function raises otherwise, because the
formula is antisymmetric in its two rungs: swapping them extrapolates the wrong
way, silently, with every residual still plausible. That inversion is what
`test_gate10_jacobian_is_first_order_and_richardson_is_consistent` exists to
catch — it pins the first-order premise and requires two extrapolants built
from different rung pairs to agree to 2e-3, an order of magnitude tighter than
the raw rungs they came from.

**Rungs are placed in `R = wavelength_over_ds`, never in raw `n_pts`** (D17):
200 points read as `R = 37.1` on a circle and 17.5 on an aspect-3 ellipse, so a
ladder in `n_pts` measures different resolutions at different points of a
catalogue.

## Formulation and validation references

- Bohren & Huffman, *Absorption and Scattering of Light by Small Particles*,
  ch. 8 — the analytic Mie solution used as the validation reference.
- Valencia et al, *Second-harmonic generation in the scattering of light by   two-dimensional particles*, JOSA B, 2003 (10.1364/JOSAB.20.002150) — The surface integral formulation.
