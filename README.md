# pysie2d

[![CI](https://github.com/claudio-sc/pysie2d/actions/workflows/ci.yml/badge.svg)](https://github.com/claudio-sc/pysie2d/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pysie2d.svg)](https://pypi.org/project/pysie2d/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A 2-D surface-integral-equation solver for time-harmonic
electromagnetic scattering from a single smooth cylinder — circular or
Gielis-superformula cross-section, embedded in a homogeneous background. Validated
against analytic Mie theory, with a typed public API and CI.

## Scope and non-goals

This package is distilled from a larger private research code; it deliberately
covers only the **homogeneous-background core** — the part that can be
validated end-to-end against a closed-form reference. That core includes
quasi-normal-mode extraction from the surface-integral operator (see
[Quasi-normal modes](#quasi-normal-modes)) and finite clusters of particles
(see [Clusters of particles](#clusters-of-particles)), validated against an
independent two-cylinder addition-theorem anchor. Potential extensions in the
mid/long-term include slab waveguide backgrounds and non-circular external
validation against independent solvers (MEEP, dolfinx).

## Figures

Relative error of the scattering efficiency `Q_sca` versus the number of
boundary points, converging toward analytic Mie theory (both polarisations):

![Convergence to Mie theory](https://raw.githubusercontent.com/claudio-sc/pysie2d/main/figures/convergence_study.png)

The same error over size parameter `x = 0.1 … 30` and `nn`, log colour scale.
The error falls from `10⁻³` to `10⁻¹²` across a narrow front (black contours,
`10⁻⁶` solid); past it the solver is at round-off. The white line is 6 points
per interior wavelength `λ/n_core`, which tracks the front's round-off edge
from `x ≈ 2` up — `10⁻⁶` itself is reached at a median of 4.7:

![Convergence map](https://raw.githubusercontent.com/claudio-sc/pysie2d/main/figures/convergence_map.png)

Near field of a Gielis `m = 6` star under plane-wave illumination (scattered
field outside the boundary, internal field inside):

![Near-field map](https://raw.githubusercontent.com/claudio-sc/pysie2d/main/figures/nearfield_map.png)

Relative local density of states (Purcell map) around the same Gielis `m = 6`
star, at one of its `qsca` resonances: a line-dipole emitter placed in a red
lobe decays faster than in free space (`1 + 4·Im S > 1`), while blue regions
suppress it. The six-fold pattern mirrors the particle's symmetry. The drive
*and* the decay rate of an embedded emitter both come from this map — it is the
entry point of quantum-dynamics calculations downstream:

![Purcell map](https://raw.githubusercontent.com/claudio-sc/pysie2d/main/figures/purcell_map.png)

Quasi-normal modes of a circular cylinder in the complex wavelength plane,
extracted from seven search boxes and plotted over the analytic Mie poles of the
same cylinder (open marks analytic, filled extracted). The axes are log-log so
that each iso-`Q` contour is a straight line, since `Q = Re λ / (2 Im λ)`:

![QNM spectrum](https://raw.githubusercontent.com/claudio-sc/pysie2d/main/figures/qnm_spectrum.png)

The same physics, one contour. A single rectangle spanning 600 nm of `Re λ`
returns **all eleven** TE modes inside it — counting the doubly degenerate pairs
— from one call and 128 matrix assemblies. Beyn's method costs
`4·n_quad_per_side` assemblies *whatever* is inside the contour, so eleven modes
cost no more than one; only the probe count has to exceed the mode count. Grey
marks are poles outside the box (three of them leak a rank direction in, hence
`rank = 14` against 11 modes):

![Wide-window QNM extraction](https://raw.githubusercontent.com/claudio-sc/pysie2d/main/figures/qnm_wide_window.png)

Regenerate them with:

```bash
uv run python examples/convergence_study.py
uv run python examples/convergence_map.py
uv run python examples/nearfield_map.py
uv run python examples/purcell_map.py
uv run python examples/qnm_spectrum.py
uv run python examples/qnm_wide_window.py
```

## Formulation (summary)

- The cylinder is invariant along its axis, so Maxwell reduces to a scalar
  Helmholtz problem for one field component (`E_y` for TE, `H_y` for TM).
- The self-consistent field solution is given **everywhere** in terms of the surface field and its normal derivative;
  matching across the interface gives a Fredholm integral equation of the second kind.
- Discretising the boundary with `nn` quadrature points yields a dense
  `2nn × 2nn` complex system `M(λ)·ei = rhs`, solved directly.
- The logarithmic Green-function singularity is handled analytically in the
  diagonal terms; complex wavenumbers are supported throughout.
- Lengths are in nm, the time convention is `exp(-iωt)`, and outgoing waves are
  `H_n^{(1)}`.
- **Wavelengths are vacuum wavelengths.** `Material.n_core`, `Material.n_clad`
  and `Material.epsi` are absolute; the background index enters through the
  single conversion `Material.wnum_bg(λ_vac) = 2π·n_clad/λ_vac`, and the
  operator sees only background-relative quantities (`Material.nc`,
  `Material.eps`). The Mie size parameter `x = 2π·n_clad·rad/λ_vac` is exposed
  as the derived `pysie2d.size_parameter` (circular geometry only).

Full details and every sign/layout convention are in
[docs/conventions.md](https://github.com/claudio-sc/pysie2d/blob/main/docs/conventions.md). The analytic reference is Bohren &
Huffman, *Absorption and Scattering of Light by Small Particles*, ch. 8; the
boundary-integral approach follows Maradudin, Michel, McGurn & Méndez,
*Enhanced backscattering of light from a random grating*, Ann. Phys. **203**
(1990) 255–307, developed there for randomly rough surfaces; this
implementation uses the closed-surface (particle) form given in
[Valencia *et al.*](https://doi.org/10.1364/JOSAB.20.002150).

## Validation

The physics test suite compares the solver against analytic Mie theory for a
circular cylinder: scattering / extinction / absorption efficiencies, the
optical theorem on a lossy particle, energy conservation on a lossless one, the
convergence rate, and the 2-D `1/√(kr)` far-field decay. Boundary quadrature is
Kress–Martensen product quadrature, so convergence is **spectral**: on the circle
the efficiencies reach double-precision round-off against Mie by `nn ≈ 30`, and
the tests hold them to `10⁻¹²`. See `tests/` for the exact tolerances and the
reasoning behind them.

The line-dipole / self-Green machinery (v0.2) is validated the same way:
reciprocity of the scattered field (to `10⁻⁶`), the free-space limit
(`LDOS → 1` far from the particle), LDOS positivity, and — the strong anchor —
the self-Green function of a circular cylinder against its closed-form
Graf-addition-theorem sum on both `Re S` and `Im S`, which agree to `10⁻¹³`
at `nn = 240`; the resolved scattered-field sign convention is recorded in
[docs/conventions.md](https://github.com/claudio-sc/pysie2d/blob/main/docs/conventions.md).

Quasi-normal-mode extraction (v0.4) is anchored the same way, in three
independent layers: the analytic Mie poles are located first and their
completeness checked against a winding-number count; the contour algorithm is
checked on synthetic matrix pencils with known spectra; and only then is the
composition tested — that the BIE operator's singularities *are* the Mie poles,
to within its discretisation error and nothing more. That error is spectral too:
`10⁻¹³` nm by `nn = 40` on the circle, where the contour integral rather than the
boundary becomes the accuracy floor. On non-circular shapes, modes are
identified by continuation from the circle
([conventions](https://github.com/claudio-sc/pysie2d/blob/main/docs/conventions.md) §8).

## Install / run / test

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # create the environment
uv run pytest                 # run the validation suite
uv run ruff format --check .  # formatting
uv run ruff check .           # lint
```

Minimal use:

```python
from pysie2d import BIESolver, Geometry, Material

geom = Geometry.gielis(rad=200, n_pts=300, m=0)   # circular cylinder, nm
mat = Material(n_core=1.5, n_clad=1.0, pol=2)      # TE
result = BIESolver(geom, mat).scatter(wavelength=600.0)  # vacuum nm

print(result.efficiencies())                       # {'qsca', 'qext', 'qabs'}
```

Line-dipole emitter and Purcell effect:

```python
from pysie2d import BIESolver, Geometry, Material, relative_ldos

geom = Geometry.gielis(rad=200, n_pts=300, m=6, n1=6, n2=12, n3=12)  # Gielis star
solver = BIESolver(geom, Material(n_core=2.0))
# wavelength is vacuum nm; the LDOS is relative to the unbounded background
print(relative_ldos(solver, wavelength=540.0, x_s=430.0, z_s=0.0))
```

## Quasi-normal modes

A quasi-normal mode is a source-free solution: a complex wavelength where the
boundary-integral operator `M(λ)` is singular. `QNMSolver` finds **every** mode
inside a rectangle of the complex λ-plane by contour integration (Beyn's
method) — no initial guess and no scan, at a cost independent of how many modes
are inside.

```python
from pysie2d import Geometry, Material, QNMSolver

geom = Geometry.gielis(rad=200, n_pts=200, m=0)
mat = Material(n_core=3.0, n_clad=1.0, pol=2)             # TE
res = QNMSolver(geom, mat).modes(745 + 2j, 775 + 15j)     # box corners, vacuum nm

print(res.wavelengths)      # 760.687 + 7.948j, twice — a degenerate pair
print(res.quality_factors)  # Q = Re λ / (2 Im λ)
print(res.edge_margin)      # contour-quality diagnostic; near zero is a warning
```

Search boxes must lie in `Im λ > 0` (the decaying half-plane under `exp(-iωt)`)
and `Re λ > 0` (which keeps `M(λ)` holomorphic); both are asserted. Poles do
**not** come in conjugate pairs here, and every `n ≥ 1` mode of a circle is
doubly degenerate.

**Read [docs/qnm-guide.md](https://github.com/claudio-sc/pysie2d/blob/main/docs/qnm-guide.md)
before using this** — it covers how to place a box, how to read the diagnostics,
what `refine()` does and does not buy you, and the limitations the feature ships
with.

### Mode sensitivity

`QNMResult.sensitivity(at)` returns `dλ/dp` for every mode from the adjoint
quotient `− uᴴ(∂M/∂p)v / uᴴ(∂M/∂λ)v`, re-extracting no eigenvalue. `at` is a
callable `δ → (geometry, material)`, so a shape parameter and a refractive index
go through one signature and one code path.

```python
res = QNMSolver(geom, mat).modes(745 + 2j, 775 + 15j).refine()
frozen = res.geometry.parametrisation          # the node map must be frozen

def wider(delta):                              # dλ/db, b in its own units
    return Geometry.gielis(rad=200, n_pts=200, m=4, b=1.0 + delta,
                           parametrisation=frozen), mat

print(res.sensitivity(wider))                  # ≈ λ/2 for both partners
```

On the circle, stretching one axis is half of a uniform dilation, so each
partner moves at `λ/2`. The geometry `at` returns must be built on
`res.geometry.parametrisation` — a shape derivative holds the node map fixed,
and `sensitivity` checks the nodes and both map derivatives exactly and raises
otherwise. Degenerate poles dispatch to a secular problem rather than raising.
`dλ/dp` converges spectrally in `n_pts`, like λ itself, so no extrapolation is
needed. Conventions
[§10](https://github.com/claudio-sc/pysie2d/blob/main/docs/conventions.md),
§11 and §12 carry the details and the measured anchors.

## Multipole decomposition

Outside the circumscribing circle of the particle the scattered field is a sum
of outgoing cylindrical harmonics about the particle centre,

```text
ψ_sc(r, θ) = Σ_m c_m · i^m · H_m^(1)(k_bg·r) · e^{imθ}
```

with `θ` measured from `+z` toward `+x`, as everywhere else in this package.
`ScatterResult.multipoles()` returns those coefficients:

```python
mp = BIESolver(geom, mat).scatter(wavelength=600.0, angle=30.0).multipoles(mmax=8)

mp.c          # signed orders m = −mmax … mmax
mp.c_plus     # symmetric   c_m + c_{−m}  (c_0⁺ = c_0)
mp.c_minus    # antisymmetric c_m − c_{−m}
mp.reconstruct(r, theta)     # the field back, on any larger circle
```

For `pol = 1` (TM) the symmetric coefficients read as magnetic dipole (`m = 0`),
electric dipole (`m = 1`) and electric quadrupole (`m = 2`). On a circle the
coefficients are the analytic Mie cylinder coefficients exactly —
`c_m = −(−1)^m·χ_m·e^{imα}`, with `χ = b_n` for TE and `a_n` for TM — and the
tests hold both polarisations to `3·10⁻¹⁵` at normal and oblique incidence and
at complex wavelength.

The defaults are the substance: the evaluation circle sits at **3 × the
circumscribing radius** (which is *not* `Geometry.rad` — on a rounded square the
circumscribing radius is 2.29 × `rad`), and the angular grid is sized from both
`mmax` and `|k_bg·r0|`. Evaluating inside the circumscribing circle is rejected,
an angular grid too coarse to separate the retained orders is rejected, and an
under-resolved angular spectrum raises `RuntimeWarning` through the
`spectrum_tail` diagnostic on the result. Each guard stands in front of a
measured silent failure — the first is 58 % error at 0.9 × the circumscribing
radius.

What it does not give you: the expansion says nothing about the field *between*
the circumscribing circle and the boundary, the coefficients are referred to the
particle centre and no translation theorem is implemented, and the only
closed-form anchor is Mie on a circle — on a non-circular shape the available
check is self-consistency between two circles, which validates the expansion
against the solver's own field and not the field itself.

## Clusters of particles

A `Cluster` is a finite arrangement of non-overlapping particles sharing one
background. The feature is purely additive: nothing in the single-particle
`BIESolver` / `ScatterResult` API changes. Each particle keeps its own
`Geometry`, its own `n_pts` and its own `Material`; the polarisation and the
background index `n_clad` belong to the *problem* and must agree across the
cluster.

```python
from pysie2d import Cluster, ClusterBIESolver, Geometry, Material

left = Geometry.gielis(rad=200, n_pts=200, m=0, x0=-350.0)    # circle, nm
right = Geometry.gielis(rad=150, n_pts=200, m=4, b=1.0, x0=350.0)
cluster = Cluster([left, right])

mats = [Material(n_core=1.5, n_clad=1.0, pol=2),              # TE, one background
        Material(n_core=2.0, n_clad=1.0, pol=2)]
res = ClusterBIESolver(cluster, mats, pol=2).scatter(wavelength=633.0, angle=30.0)

amp, angles = res.far_field()                # total amplitude of the cluster
print(res.cross_sections())                  # {'c_sca', 'c_ext', 'c_abs'}, nm
print(res.resolution())                      # points per interior wavelength
```

`scatter_dipole(wavelength, x_s, z_s)` drives the same cluster with a line
dipole, `eval_field(x, z)` gives the field at arbitrary points inside or
outside, and `ei_particle(p)` hands back one particle's `(φ_p, χ_p)` sub-vector.

What you need to know before using it:

- **Overlapping boundaries raise `ClusterOverlapError`.** The coupled
  formulation represents a region exterior to every particle and interior to
  exactly one; where two boundaries cross, that partition does not exist and no
  resolution repairs it.
- **A too-small gap warns, it does not raise.** Two nearly touching boundaries
  put a near-singularity of the cross-block integrand a conformal half-width
  `~√(2·gap/a)` off the real axis, so the quadrature needs about
  `GAP_ENVELOPE_C·√(a/gap)` nodes per particle; below that the solution degrades
  *silently*, with a well-conditioned matrix and clean single-particle
  diagnostics. `ClusterGapWarning` says so. The constant is a **measured**
  envelope, not a guess — BIE self-convergence over `gap/a ∈ [0.032, 5.06]`, two
  polarisations and three size parameters, written up in
  [docs/design/studies/cluster-gap-envelope.md](https://github.com/claudio-sc/pysie2d/blob/main/docs/design/studies/cluster-gap-envelope.md).
  It was measured on circles; for non-circular facing boundaries the warning is
  diagnostic only, and says so in its text.
- **Ragged resolution warns too.** The coupled system is only as accurate as its
  worst block, and an under-resolved particle is invisible from any other one,
  so a spread wider than 4× in points per interior wavelength raises
  `ClusterResolutionWarning`.
- **`cross_sections()` (absolute, in nm) exists on both** the single-particle
  `ScatterResult` and the cluster result, which is what makes the two paths
  comparable. `efficiencies()` and `multipoles()` are single-particle only: a
  cluster has no `rad` to normalise by and no single centre to expand about.

The validation anchor is the analytic addition-theorem solution for two parallel
circular cylinders, `pysie2d.reference.two_cylinder` — the same role
`reference.mie` plays for the single-particle solver. Like `reference.mie` it is
not exported at top level; import it explicitly to reproduce the validation.

On scale: the dense solve, not assembly, is the cost here — it reaches parity
with assembly at `Np = 2` and dominates from `Np = 3`, with `Np = 16` still a
usable 2.7 s and 625 MiB ([performance](https://github.com/claudio-sc/pysie2d/blob/main/docs/design/performance.md) §6).

## Performance

The system is a dense `2nn × 2nn` complex matrix; at `nn = 300` (a `600 × 600`
solve) a single wavelength takes below one second in a modern computer, so wavelength
sweeps are cheap serial `for` loops — no parallelism required.

Matrix assembly dominates a single solve, and almost all of that cost is Hankel
evaluation. For real arguments `H_n^{(1)} = J_n + i·Y_n` exactly, and the Cephes
`J_n`/`Y_n` kernels are an order of magnitude faster than the general
complex-argument algorithm — so `hank0`/`hank1`/`cbesh` dispatch on the argument
at runtime, making `assemble_matrix` about 5× faster for a non-absorbing
particle and 1.8× for an absorbing one. Complex wavenumbers take the original
path and are bit-identical, which is what keeps quasi-normal-mode work possible.

For a Purcell map, every grid point is a different source position, hence a
different right-hand side — but the matrix `M(λ)` is the same for all of them.
`relative_ldos_map` therefore factorises `M` **once** with
`scipy.linalg.lu_factor` and reuses it across all sources, and it batches the
reuse: one multi-RHS BLAS-3 `lu_solve` and one vectorised representation-formula
evaluation per chunk rather than a per-point loop (7.5× per source point). What
would be an hour-long sweep takes seconds.

## Roadmap

- **v0.1.0** — core scattering: plane-wave excitation, near/far fields,
  cross-section efficiencies, Mie validation, convergence study, CI.
- **v0.2.0** — line-dipole (point-source) excitation and the self-Green
  function → relative LDOS / Purcell maps.
- **v0.3.0** — performance: Cephes fast path for real-argument Hankel
  functions and a batched, factorise-once `relative_ldos_map`.
- **v0.4.0** — vacuum-wavelength and background-index conventions (breaking),
  and quasi-normal-mode extraction via Beyn's contour method, validated
  against analytic Mie resonances.
- **v0.4.2** — exact scale covariance of the discrete BIE system, recorded as
  [conventions](https://github.com/claudio-sc/pysie2d/blob/main/docs/conventions.md) §9.
- **v0.5.0** — threaded contour integration in `contour_moments`, and
  an adjoint eigenvalue-sensitivity API (`dλ/dp` per mode) on top of the
  identity already proved in conventions §9.
- **v0.6.0** — spectral convergence: Kress–Martensen product quadrature
  replaces the first-order diagonal self-patch, and nodes are placed by a smooth
  `Parametrisation` map. **Breaking** — see below.
- **v0.7.0** — multipole decomposition of the scattered field into cylindrical
  harmonics about the particle centre, anchored on analytic Mie in both
  polarisations. Purely additive.
- **v0.8.0** — finite clusters of arbitrary particles: coupled assembly, dense
  solve, and absolute cross-sections, anchored on an independent two-cylinder
  addition-theorem reference. Purely additive.

### Migrating from v0.5

v0.5 converged at first order in `nn`; v0.6 converges spectrally, so **every
number changes** — for the better, typically by many orders of magnitude at the
same `nn`. Three calls change:

```python
# v0.5
geom = Geometry.gielis(rad=200, n_pts=200, m=4, b=1.2)   # uniform arc length
wider = Geometry.gielis(rad=200, n_pts=200, m=4, b=1.3,
                        theta=geom.theta)                # frozen nodes

# v0.6
geom = Geometry.gielis(rad=200, n_pts=200, m=4, b=1.2)   # uniform θ (new default)
wider = Geometry.gielis(rad=200, n_pts=200, m=4, b=1.3,
                        parametrisation=geom.parametrisation)
```

- **`theta=` is now `parametrisation=`.** The new quadrature needs the node
  map's first and second derivatives at every node, and those cannot be
  recovered from an array of angles. Accepting a bare array would return a
  plausible wrong answer, so `theta=` is gone, and an angle array passed as
  `parametrisation=` raises `TypeError` with the migration in the message.
- **Nodes are equispaced in θ by default**, not in arc length: under the new
  quadrature uniform θ converges faster on every star and superellipse measured
  within the solver's validity envelope, at low resolution as well as high. For uniform arc length pass
  `parametrisation=Parametrisation.gielis(...)`
  ([conventions](https://github.com/claudio-sc/pysie2d/blob/main/docs/conventions.md) §13).
- **`Geometry(...)` built from your own arrays** takes derivatives with respect
  to the quadrature parameter `t` and no longer takes `delt`; `assemble_matrix`,
  `assemble_matrix_dwn` and `assemble_matrix_reference` no longer take `delt`
  either. Odd `m` now requires `a == b` and `n2 == n3`, since the boundary
  otherwise does not close.

- **`n_pts` defaults to 100**, was 200. The circle anchors reach round-off
  by `nn ≈ 30–40` under the new quadrature, so the old default was sized
  for a first-order scheme that no longer exists. Pass `n_pts` explicitly
  to keep a v0.5 resolution.

`richardson_limit` still exists but is deprecated: nothing in the package
converges at first order any more, and extrapolating a spectrally converged
pair makes it worse.

## License

MIT — see [LICENSE](LICENSE).
