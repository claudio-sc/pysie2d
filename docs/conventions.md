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

`Geometry` holds the boundary sampled at nodes **equispaced in the quadrature
parameter `t`** (§13.1), `t_j = 2π(j + ½)/nn`, and derivatives with respect to
that same `t`:

- `f` — x-coordinates; `g` — z-coordinates. **`g` is a coordinate array, not a
  Green function** (an unfortunate historical name).
- `df`, `dg` — first derivatives of `f`, `g` w.r.t. `t`.
- `ddf`, `ddg` — second derivatives w.r.t. `t`.
- `delt` — the trapezoid step `2π/nn`, a read-only property. It is not an input
  anywhere: Kress's weights presume exactly this step (v0.6).
- `parametrisation` — the map `θ = w(t)` the arrays were sampled on, and
  `nodes` its `t`, `θ`, `w'`, `w''` at this `nn`. `theta` is `nodes.theta`.

On a map other than the identity, `x_t = x_θ·w'` and `x_tt = x_θθ·w'² + x_θ·w''`.
The χ half of the solution vector (§4) carries the Jacobian of `t`, so it is not
comparable across two maps; φ is.

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

**Identifying modes on a non-circular shape is done by continuation from the
circle (v0.6).** The circle is the only shape in this package with a *labelled*
spectrum — the analytic Mie roots of `reference/mie.py`. A deformed shape has
none, so a mode there is identified by carrying a labelled circle mode along a
smooth shape perturbation, and **the analytic comparison is therefore the
starting point of any non-circular convergence study**, not an optional extra
for the circular case.

That makes continuation a **convergence check in its own right**: a smooth
perturbation of a smooth boundary moves a pole smoothly across the complex
plane, so a kink, a jump, or a non-monotone excursion in the trajectory is a
defect signal — in the discretisation, in the identification, or in the
physics — and is read as such before it is read as a result.

**The failure is silent, which is why this is recorded here.** Nearest-neighbour
tracking with steps that are too long hops onto a neighbouring branch while
every diagnostic stays healthy: measured on an equal-area ellipse ladder at
aspect steps of 0.25, `Q` swung 10 → 48 → 25 → 40 along what should have been a
smooth trajectory, two walks with different step sizes reported *different modes
at the same aspect*, and `edge_margin` looked fine throughout. What fixed it was
a **predictor** — secant extrapolation of the last two steps — so the search box
has to contain only the trajectory's curvature rather than its whole step, plus
a per-step ambiguity tell (the distance to the nearest other mode in the box).
Where a physical argument is available it is worth more than proximity: the
`n = 0` mode is radial and tracks the minor semi-axis, `Re λ(A)/Re λ(1) ≈
A^{-1/2}` to ~8 % out to aspect 4.

**The pole landscape gets richer as the shape deforms, so identification is not
a one-time setup.** The circle's `n ≥ 1` degeneracies split under a deformation
that breaks the symmetry, modes enter and leave a fixed box, and trajectories
approach each other. Any change to the shape family, the deformation path, or
the box is a reason to re-examine the identification rather than to assume the
previous scheme still holds; expect it to be refined as the study reaches
richer families.

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

    k·r                 all off-diagonal Bessel and Hankel arguments
    k·gamma             inside ln(k·γ/2) on the M2 and M4 diagonals
    k²·cij              the double-layer kernels against the cross products
    deriv/gamma²        the M1 and M3 diagonals

The Kress weights `R`, `W` and the step `2π/nn` depend on `nn` alone. The nodes
are degree 0: trivially on the default uniform-θ map, and on an arc-length
`Parametrisation` because its series is truncated **relative to its own mean
coefficient**, so `N_f`, `K` and every node are the same at every `rad`.
Covariance needs that homogeneity, not the accuracy of the map. A truncation
threshold in absolute nm would break this silently.

Hence, entrywise and at any `n_pts`:

    M(s·rad, s·λ) = M(rad, λ),      ∂M/∂rad = −(λ/rad)·∂M/∂λ

and therefore `λ(s·rad) = s·λ(rad)`, `dλ/drad = λ/rad`, `dQ/drad = 0`. The
adjoint form `dλ/dp = −uᴴ(∂M/∂p)v / uᴴ(∂M/∂λ)v` returns `λ/rad` for **any**
`u, v`, so the result is gauge-free; on the semisimple `±n` pair the 2×2
secular problem is a multiple of the identity, so both partners share it in any
null-space basis and **a dilation can never split a degeneracy**.

Three things this does *not* say. It is not accuracy: the discrete pole sits at
a fixed `x_disc(n_pts) ≠ x_Mie`, so covariance is exact whatever the
discretisation error — v0.5 held it while the wavelength was still wrong in the
first decimal. It is largely not a convention check: signs
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

**The v0.5 knife edge on non-C¹ boundaries is gone.** v0.5's arc-length nodes
moved by an ulp at an inexact ratio, and a node sitting numerically on a kink of
the exponent-1 superformula jumped to the other one-sided tangent, changing the
matrix by O(1) *(measured then: 0.264 at s = 1.7)*. The uniform-θ map's nodes
are bit-identical at every `rad`, so no node can cross a kink under rescaling
*(measured on the same cusped star: 1.4e-15 at s = 1.7)*, and an arc-length
`Parametrisation` refuses a cusped shape rather than approximating it. A rough
boundary still converges only algebraically; it no longer loses covariance.

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

## 10. Shape derivatives use a frozen map (v0.5, amended v0.6)

**A finite difference in a shape parameter holds the node map fixed.** Every
`∂M/∂p` takes `M(p₀±h)` on the `Parametrisation` of `p₀`
(`Geometry.gielis(..., parametrisation=base.parametrisation)`), and so do
`∂M/∂λ` and the left and right null vectors that enter the adjoint quotient —
all four on one discretisation. `QNMResult.sensitivity` enforces it: the
perturbed geometry must carry the base geometry's `θ`, `w'` and `w''` exactly,
and a geometry with no map is refused by name. On the default uniform-θ map this
holds automatically, since the identity depends on no shape parameter.

**Why, as of v0.6.** Node placement is a parametrisation gauge: λ does not depend
on how the boundary was sampled, but `M` does. Rebuilding the map at `p₀ ± h`
therefore differentiates the gauge along with the shape. Under the smooth
`Parametrisation` that term is **not** an error in the rate — both derivatives
are second order in `h` — but it changes `∂M/∂p` by an O(1) fraction *(measured
on an ellipse, `b = 1.2`, `n_pts = 60`: rate 100.0 frozen and 100.0 rebuilt,
‖∂M/∂b(frozen) − ∂M/∂b(rebuilt)‖/‖∂M/∂b‖ = 0.200)*, while `dλ/db` from the two
agrees only to discretisation accuracy *(1.2e-10)*. Freezing is what makes the
quotient the exact derivative of the discrete eigenvalue rather than an
approximation to it, and what keeps integer-valued properties of a rebuilt map
(`N_f`, `K`) from stepping inside a difference.

**Why, in v0.5 — history.** The v0.5 arc-length inversion went through
`np.interp`, piecewise linear in the shape parameter, so a node whose bracketing
cell differed between `p₀ − h` and `p₀ + h` put an O(h) term into the quotient
that grew with `n_pts` *(measured then: the h-ladder on `∂M/∂b` fell 8.29e-5 →
3.24e-5 → 5.60e-9 unfrozen, against 1.01e-5 → 1.01e-7 → 1.67e-9 frozen;
`docs/design/studies/shape-derivative-smoothness.md`)*. That mechanism no longer
exists.

**Step size.** `h = 1e-5` in the parameter's own units, with the cancellation
floor at ~1e-8 and truncation at ~1e-7 a decade above, i.e. about a decade of
margin on each side. The margin, not the best value at one design point, is the
reason for the choice: the truncation coefficient scales with the parameter's
geometric leverage, which moves across the shape catalogue. For a length
parameter in nm the cancellation floor on `dλ/dp` is `ε·p/h`, not `ε/h` — about
4e-9 for `rad = 200` — which is the bound the Gate 1 tests are derived from.

**`parametrisation` is optional to store and mandatory to differentiate.** On
`Geometry.__init__` it defaults to `None`: a boundary assembled from arrays that
came from elsewhere has no map to report, and the solver — assembly, fields,
LDOS, mode extraction — never reads it, so refusing to construct such a geometry
would break scattering-only users for a reason unrelated to scattering. The
requirement belongs at the point of use, and `QNMResult.sensitivity` raises on a
missing map naming *which* of the two geometries lacks it. Both branches are
needed: without the base-side check, `None == None` compares equal and two
unrelated discretisations are accepted.

Frozen maps preserve §9 exactly: a map carries no length, so
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

**The frozen map is enforced, not documented.** The geometry returned by `at`
must carry the node set of `result.geometry.parametrisation` **exactly** — `θ`,
`w'` and `w''` — and `sensitivity` raises otherwise. Exact equality is the right
test because there is no threshold at which a different discretisation becomes
the same one (§10).

**`u` is a genuine left null vector**, obtained from the same SVD as `v` — the
smallest singular triplet of `M(λ)`, `U[:, -1]` and `V[:, -1]`. It is **not**
`conj(v)`: M is not complex-symmetric here *(measured: `‖M − Mᵀ‖/‖M‖ = 1.16`,
and `|⟨u, conj(v)⟩| = 0.32` at the TE n=0 pole of the reference circle)*, so
substituting `conj(v)` gives a quotient wrong by an O(1) factor with every
residual still looking right. Cost is one assembly plus one full SVD per mode,
0.080 s at `n_pts = 200` (v0.5 measurement).

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
cancellation floor ε·rad/h ≈ 4e-9 *(measured 5.8e-10 at `n_pts = 40`)*. Gate 1
degenerate half — a dilation cannot lift the ±n degeneracy of a circle, so the
2×2 secular matrix is a multiple of the identity *(measured: splitting
3.1e-10)*. Gate 2 — at `n2 = n3`, `(log a + log b)` and `log rad` move λ
identically, so their difference is an exact null direction of `J` *(measured
ratio − 1 = 1.3e-12)*. Gate 3 — against central differences of independently
re-extracted Beyn poles, second order in the step *(4.495e-4 → 4.489e-6 →
4.493e-8, ratios 100.1 and 99.9)*. All four in `tests/test_sensitivity.py`.

## 12. Jacobian accuracy is bought by resolution (v0.6; extrapolation in v0.5)

**Since v0.6, `J = dλ/dp` converges spectrally in `n_pts`, like λ itself**, and
no extrapolation is needed *(measured on the Gate-10 ellipse, `m = 4`, `b = 1.2`,
TE, `n_core = 3`: `|J(40) − J(80)|/|J(80)| = 3.0e-10`, `|J(60) − J(80)|/|J(80)|
= 4e-11`, the latter at the contour and central-difference floor)*.
`richardson_limit` is kept for genuinely first-order quantities; applying it,
with its exponent pinned at 1, to a spectrally converged pair adds error.

**Rungs are placed in `R = wavelength_over_ds`, never in raw `n_pts`** (D17):
200 points read as `R = 37.1` on a circle and 17.5 on an aspect-3 ellipse at
uniform arc length, so a ladder in `n_pts` measures different resolutions at
different points of a catalogue. On the default uniform-θ map the same ellipse
reads 9.2: `R` reports the worst-resolved node, and uniform θ stretches the
flanks.

**v0.5 history.** Under the Maradudin self-patch `J` converged at first order,
observed order 0.98–1.02 on every component on three independent ladders
(`docs/design/studies/jacobian-convergence.md`); differencing two designs did
not help, and two-rung Richardson from `R = 15 + 30` put every component inside
6.4e-4 of the limit at 0.46× the cost of one `R = 50` rung. That is the problem
v0.6 removed rather than solved.

## 13. Node placement and the parametrisation (v0.6)

v0.6 replaces the Maradudin diagonal self-patch with **Kress–Martensen product
quadrature** (`docs/design/kress-spec.md`). Under the self-patch the solver
converged at exactly first order in `nn` even on a perfect circle with perfectly
uniform nodes; under Kress, with analytic `ddf`/`ddg`, the circle is at
round-off by `nn = 30` against analytic Mie *(measured: ≤ 1.1e-15 on `qext`, TE
and TM, both circle branches)*.

Kress's weights come from trigonometric interpolation, and that imposes one
constraint with three consequences. All three are of the kind this file exists
for: **breaking them produces a plausible wrong answer, not an error.**

**13.1 Nodes are equispaced in the quadrature parameter `t`, and every
derivative is taken in `t`.** Not in θ, not in arc length. The boundary is
`θ = w(t)` with `t` equispaced and `w` a smooth 2π-periodic monotone map;
`4 sin²((t − t')/2)` is then the correct periodic stand-in for the singular
factor, and `w'` is absorbed into `|dx/dt|` exactly the way `|dx/dθ|` was.
Grading lives in `w`, not in the node positions. Violating this drops the
quadrature from spectral to first order silently.

`w` must be **smooth**. The v0.5 `np.interp` inversion was C⁰, and a C⁰ change of
variables destroys the smoothness the trapezoid rule's accuracy rests on. Uniform
θ, uniform arc length and curvature-adaptive grading are one map with three
densities `|dx/dt| = ρ(t)`, not three code paths: a `Parametrisation` carries
`w`, `w'`, `w''`, and `arc_length: bool` is gone.

**The weights depend on the parity of `nn`.** For odd `nn` there is no Nyquist
mode, and the textbook even-`nn` formula applied there returns a plausible wrong
matrix *(measured: a QNM displaced by 1.1 nm at `nn = 115`)*.
`kernels._kress_log_weights` handles both.

**13.2 `w` does not depend on λ.** If the parametrisation varies with the
wavelength, `M(λ)` loses holomorphy and Beyn's contour integral silently returns
wrong modes — holomorphy is the premise of the contour argument (§8). This is
the same reasoning that rules out condition-number-optimal node placement: the
smallest singular value of a matrix is not analytic in λ, and a quasi-normal mode
is by definition a λ where `M` is singular. A density specified in
`R = wavelength_over_ds` (§12) therefore carries its **own fixed reference
wavelength**, set once when the `Parametrisation` is built and never taken from
the solve. Kress itself adds nothing λ-dependent: `R`, `W` and `h` are
geometry- and wavelength-free, and `J₀`, `J₁`, `H₀^{(1)}`, `H₁^{(1)}` and
`ln k` are holomorphic on the search half-plane of §8.

**The tell for lost holomorphy is `sigma_ratio`, not the mode count.** A
holomorphic `M` puts Beyn's eigenvalues exactly on its singularities, so a
returned λ at which `σ_min/σ_max` of `M` is not at round-off is not a pole of
the operator that was integrated. Mode count, `edge_margin` and the rank gap do
not see a mild violation at all *(measured, TE n = 0 circle, `nn = 80`, on the
graded map `θ = t + ε sin 2t` with `ε` made to depend on Re λ — valid, and
Mie-exact to 1e-11, at every fixed λ: at a slope of 1e-4 and 1e-3 per 25 nm the
pole moved 3.9e-4 and 3.9e-3 nm with the count at 1 and `edge_margin` at 0.433
unchanged to three digits, while `sigma_ratio` rose from 8.1e-15 to 3.0e-7 and
3.0e-6; from 1e-2 the probe saturates and Beyn raises;
`docs/design/studies/g4_holomorphy_qnm.py`)*. Any check that a map, a material
model or an assembly path has kept `M(λ)` holomorphic reads `sigma_ratio`. A
stable mode count is necessary and says nothing about accuracy: on a circle at
`nn = 20` the count is right with the poles 2 nm out.

**13.3 `w` does not depend on any parameter being differentiated.** The frozen
object of §10 is the map, not the angle array: under Kress the θ array still says
where the nodes are, but assembly also needs `w'` and `w''`, and those cannot be
recovered from it. That is why `Geometry.gielis` takes `parametrisation=` and not
`theta=`. Under a smooth map a rebuilt `w` no longer degrades the rate of `∂M/∂p`
— it changes what is differentiated (§10).

**The default map is uniform θ** (`Parametrisation.uniform_theta()`). Under
Kress it converges fastest once resolved on every shape measured, from mild
superellipses to near-corner ones and spiky stars *(measured, `qext` at
`nn = 640` on the `m = 4`, `n = 20/50/50` near-corner shape: uniform θ 2.2e-6,
adaptive 6.8e-6, uniform arc length 1.8e-4)*: composing the boundary with a
graded `w` narrows the analyticity strip the trapezoid rule's rate is set by.
**Uniform arc length is not the better choice at low resolution either.** On
spiky stars at campaign resolutions, below ~1e-2 relative error no map is
consistently ahead: every map's error swings 3–10× between neighbouring `nn`,
so a single-rung comparison picks a winner by noise. Where 1e-3 is reachable
by `nn = 300`, uniform θ reaches it first; where it is not, the remedy is more
nodes, not another map *(measured, `qext`, TE, λ = 450/600/900 nm, `m = 6`,
n2 = n3 = 8, `nn` 50–300: arm ratio 4, uniform θ at 1e-3 by `nn` = 200/200/300
against 300/—/— for arc length; arm ratio 8 and 16, no map at 1e-3 by 300;
uniform θ 6.4e-5 at `nn = 480` on arm ratio 8)*. `Parametrisation.gielis` is
also not robust there: Newton fails at some `nn` from arm ratio 8, and the map
cannot be built at arm ratio 16. Pass a non-default map only to freeze one
across a shape derivative (§10) or for a study.

**Scale covariance (§9) is preserved.** `R`, `W` and `h` depend only on `nn`;
nothing in the map construction carries an absolute length.

**The curvature-adaptive density ships as a non-default map** (architecture
item 4), `Parametrisation.gielis` with a non-degenerate `r_band`. G3 found the
one shape class where it pays — flat-sided, near-corner shapes with
`κ_max·rad ≳ 20` — but it cannot be built on them (`ln|κ|` is infinite at flat
points), and very spiky stars can fail its Newton inversion. Both limits are
stated in its docstring; nothing in this section depends on it.

## 14. The cluster degree-of-freedom layout (v0.8)

A cluster of `Np` particles carries `N = 2·Σ_p nn_p` unknowns. Particle `p`
occupies the **contiguous** slice `[o_p : o_p + 2·nn_p]`, with the offsets
cumulative over `2·nn_q`:

    o_p = Σ_{q<p} 2·nn_q                 (Cluster.offsets, Cluster.slice)
    ei  = [φ_0, χ_0, φ_1, χ_1, …, φ_{Np−1}, χ_{Np−1}]

Within one particle's slice the first `nn_p` entries are φ_p and the next
`nn_p` are χ_p — that is §4 unchanged, applied to the sub-vector.

**Contiguity is what makes every primitive reusable.** Each particle's
`2·nn_p` sub-vector is *exactly* the argument `assemble_matrix`,
`plane_wave_rhs`, `line_dipole_rhs`, `_far_field_at` and the representation
integral already take, so the cluster path calls them verbatim rather than
reimplementing them at an offset. The measurable consequence is that a
one-particle cluster reproduces `BIESolver` **bit-identically** — matrix,
solution vector, far field and near field — which is what
`tests/test_cluster.py` asserts with `np.array_equal` rather than `allclose`.
If that assertion ever degrades to "close", this layout has been changed and
this section is no longer true; that, not the number, is what the test guards.

**This section does not extend §4, and must not be read as reordering it.**
The alternative grouping `[φ_0 … φ_{Np−1}, χ_0 … χ_{Np−1}]` — the legacy
research code's layout, which §4 can look like it implies — needs two offset
tables under ragged `nn_p` and leaves no block of the system matrix
contiguous. §4 governs the layout *within* one particle; §14 governs only how
particles concatenate. Do not "fix" one into the other.

**The blocks that layout addresses.** The block at (field particle `q`, source
particle `p`) is `2·nn_q × 2·nn_p`. The diagonal `q = p` is the single-particle
matrix at that particle's own `nn_p`, `nc_p`, `eps_p` and the **common** `k_bg`
— a cluster sits in one background, so `n_clad` is validated equal across the
materials while the background-relative `nc` and `eps` (§2) may differ freely.
The off-diagonal has M1/M2 in its upper half and **exactly zero** below:
particle `p`'s interior Green function is confined to `p`'s own volume, so
coupling enters only through the exterior background kernel. That argument
needs the interior domains disjoint, which is why overlapping boundaries are
rejected by the formulation and not merely by an implementation limit.

## Formulation and validation references

- Bohren & Huffman, *Absorption and Scattering of Light by Small Particles*,
  ch. 8 — the analytic Mie solution used as the validation reference.
- Valencia et al, *Second-harmonic generation in the scattering of light by   two-dimensional particles*, JOSA B, 2003 (10.1364/JOSAB.20.002150) — The surface integral formulation.
