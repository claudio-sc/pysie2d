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

## Formulation and validation references

- Bohren & Huffman, *Absorption and Scattering of Light by Small Particles*,
  ch. 8 — the analytic Mie solution used as the validation reference.
- Valencia et al, *Second-harmonic generation in the scattering of light by   two-dimensional particles*, JOSA B, 2003 (10.1364/JOSAB.20.002150) — The surface integral formulation.
