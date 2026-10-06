# Spike: finite waveguide with tapered, rounded ends

**Status: parked, 6 Oct 2026.** This is a study script and not part of the
package. It was built on v0.6.0 (`80af82e`). It needs `matplotlib`, which is
not a dependency:
`uv run --with matplotlib python docs/design/studies/waveguide_taper_geometry.py`.

## Question

Can a finite waveguide segment (straight guide → adiabatic taper → rounded
tip) be written as one entire-analytic closed boundary, so that Kress
quadrature keeps its spectral convergence (conventions §13)? A stadium cannot
do this, because its curvature jumps at the joins.

## Answer

Geometrically, yes. Take g = x_end·cos θ and f = Y(g)·sin θ, where Y is a
tanh-windowed envelope. This has no joins, and the analytic derivatives match a
complex-step check to < 1e-9. It feeds straight into `Geometry`. The cost is
that the flat section and the tip width are only reached up to an
exponentially small tanh residual.

**For the solver, no.** The default shape is a 33 µm long guide, 400 nm wide,
with a 120 nm neck. It needs **K ≈ 4400 Fourier harmonics** to reach round-off
(K ≈ 3000–5100 across taper sharpness 2–8). That means nn ≳ 9000 nodes, well
above the dense-solve memory ceiling (nn ≈ 2500, see `beyn.py` and
performance.md). The aspect ratio (~80:1) is the cause, not the construction.

## Picking it up

Possible ways forward: a shorter structure; a curvature-adaptive
`Parametrisation` (v0.6 ships one, but with no flat points, so it would need
work); or a fast/iterative solver. Each of these is a scope decision. The
script's module docstring records the full derivation.
