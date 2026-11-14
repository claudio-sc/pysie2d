# Clusters of particles

A `Cluster` is a finite arrangement of non-overlapping particles sharing one
background. Nothing in the single-particle `BIESolver` / `ScatterResult` API
changes. Each particle keeps its own `Geometry`, its own `n_pts` and its own
`Material`; the polarisation and the background index `n_clad` belong to the
*problem* and must agree across the cluster. Clusters work in free space and
over the backgrounds of [layered-backgrounds.md](layered-backgrounds.md).

From [`examples/cluster_scatter.py`](../examples/cluster_scatter.py):

```python
from pysie2d import Cluster, ClusterBIESolver, Geometry, Material

left = Geometry.gielis(rad=200, n_pts=200, m=0, x0=-350.0)  # circle, nm
right = Geometry.gielis(rad=150, n_pts=200, m=4, b=1.0, x0=350.0)
cluster = Cluster([left, right])

mats = [
    Material(n_core=1.5, n_clad=1.0, pol=2),  # TE, one background
    Material(n_core=2.0, n_clad=1.0, pol=2),
]
solver = ClusterBIESolver(cluster, mats, pol=2)
res = solver.scatter(wavelength=633.0, angle=30.0)

amp, angles = res.far_field()  # total amplitude of the cluster
print(res.cross_sections())  # {'c_sca', 'c_ext', 'c_abs'}, nm
print(res.resolution())  # points per interior wavelength
```

`scatter_dipole(wavelength, x_s, z_s)` drives the same cluster with a line
dipole, `eval_field(x, z)` gives the field at arbitrary points inside or
outside, and `ei_particle(p)` hands back one particle's `(φ_p, χ_p)` sub-vector.

## What you need to know before using it

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
  envelope: BIE self-convergence over `gap/a ∈ [0.032, 5.06]`, two
  polarisations and three size parameters, written up in
  [design/studies/cluster-gap-envelope.md](design/studies/cluster-gap-envelope.md).
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
circular cylinders, `pysie2d.reference.two_cylinder`, in the role
`reference.mie` plays for the single-particle solver. Like `reference.mie` it is
not exported at top level; import it explicitly to reproduce the validation.

On scale: the dense solve, not assembly, is the cost. It reaches parity with
assembly at `Np = 2` and dominates from `Np = 3`, with `Np = 16` still a usable
2.7 s and 625 MiB ([design/performance.md](design/performance.md) §6).

## Many materials at one wavelength

`scatter` rebuilds and factorises everything on each call. `at_wavelength`
builds the part the materials cannot change once, and returns a
`ClusterEigenEps`, whose solves are far faster with the same answer to
round-off. Continuing the example above:

```python
fast = solver.at_wavelength(633.0)  # about one scatter() call
for eps in np.linspace(2.0, 4.0, 5) + 0.1j:
    new = [Material.from_eps(eps, pol=2), mats[1]]
    print(eps, fast.scatter(new, angle=30.0).cross_sections()["c_abs"])
```

| Particles | `scatter`, free space | `scatter`, over a slab | `fast.scatter` |
|---|---|---|---|
| 4 | 0.03 s | 0.18 s | 11 ms |
| 16 | 0.46 s | 8.5 s | 50 ms |
| 32 | 2.8 s | 66 s | 0.12 s |
| 64 | 27 s | not run | 0.33 s |

(100 nodes per particle, distinct particles, an absorbing ε.) It pays for
itself within the first few materials whenever the wavelength holds still. A
sweep in *wavelength* rebuilds everything at each step, so keep using `scatter`
there.

## Eigenpermittivities

`fast.eigenpermittivities(eps_lo, eps_hi)` finds the absolute permittivities,
shared by all particles, at which the cluster supports a field with no source,
in free space and over any background. Results go straight into
`Material.from_eps`. From
[`examples/cluster_eigenpermittivities.py`](../examples/cluster_eigenpermittivities.py):

```python
circle = Geometry.gielis(rad=100, n_pts=120, m=0)
solver = ClusterBIESolver(Cluster([circle]), [Material(n_core=3.0, pol=2)], pol=2)
fast = solver.at_wavelength(633.0)

modes = fast.eigenpermittivities(12.0 - 1.4j, 15.0 - 0.1j)  # box corners in ε
print(modes.eps, modes.converged)  # also .vectors, .rank, .sv_ratio
```

It takes the same `contour=`, `adaptive=` and `interpolate=` options as
`QNMSolver.modes` ([qnm-guide.md](qnm-guide.md)); `interpolate=True` works for one
particle only, since with more the isolated-particle poles sit inside the box.

Keep the rectangle off the real ray `ε ≥ 0`, which raises. A box that ends with
no rank gap also raises, usually because a mode sits near its edge. See
[conventions](conventions.md) §18.

In free space the results are checked against the analytic circle and a Graf
addition-theorem determinant, and over a perfect conductor against the mirror
image. **Over a dielectric half-space or a `Stack` the only check is against
this package's own driven solves; there is no external reference.**
