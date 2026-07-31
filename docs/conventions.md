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

## 2. Units

Lengths are in **nanometres**, everywhere. The free-space wavenumber is
`k = 2π/λ` with `λ` in nm, so `k` is in rad/nm.

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
the branch can trigger; a genuinely complex `wn`, `ri`, or `wnum` passes through
untouched. Any change that removes the ability to pass a complex wavenumber is
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

The free-space normalisation of the LDOS uses `Im[g₀(r→r)] = 1/4` (the log
divergence of `H₀^{(1)}` lives in its imaginary part; with the `i/4` prefactor
the imaginary part of `g₀` tends to `J₀(0)/4 = 1/4`), giving
`relative_ldos = 1 + 4·Im(S)`.

## 8. The quasi-normal-mode half-plane (v0.5)

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

> **Note.** Public wavelengths are documented as vacuum wavelengths (§2) but the
> implementation currently forms `k = 2π/λ`, which is the *medium* reading. The
> two coincide only at `n_clad = 1`, which is where every test in the suite —
> including the QNM anchor — runs. Reconciling this is a breaking change to the
> driven solver, shipping separately in v0.4.0; see
> `docs/design/beyn-port-spec.md` §2.

## Formulation and validation references

- Bohren & Huffman, *Absorption and Scattering of Light by Small Particles*,
  ch. 8 — the analytic Mie solution used as the validation reference.
- Valencia et al, *Second-harmonic generation in the scattering of light by   two-dimensional particles*, JOSA B, 2003 (10.1364/JOSAB.20.002150) — The surface integral formulation.
