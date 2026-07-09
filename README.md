# pysie2d

A 2-D surface-integral-equation solver for time-harmonic
electromagnetic scattering from a single smooth cylinder — circular or
Gielis-superformula cross-section, embedded in a homogeneous background. Validated
against analytic Mie theory, with a typed public API and CI.

## Scope and non-goals

This package is distilled from a larger private research code; it deliberately
covers only the **homogeneous-background, single-particle core** — the part
that can be validated end-to-end against a closed-form reference. Potential extensions
in the mid/long-term include slab waveguide backgrounds, multiple-particle simulations,
and quasinormal-mode searches based on the surface-integral matrix operator.

## Figures

Relative error of the scattering efficiency `Q_sca` versus the number of
boundary points, converging toward analytic Mie theory (both polarisations):

![Convergence to Mie theory](https://raw.githubusercontent.com/claudio-sc/pysie2d/v0.2.0/figures/convergence_study.png)

Near field of a Gielis `m = 6` star under plane-wave illumination (scattered
field outside the boundary, internal field inside):

![Near-field map](https://raw.githubusercontent.com/claudio-sc/pysie2d/v0.2.0/figures/nearfield_map.png)

Relative local density of states (Purcell map) around the same Gielis `m = 6`
star, at one of its `qsca` resonances: a line-dipole emitter placed in a red
lobe decays faster than in free space (`1 + 4·Im S > 1`), while blue regions
suppress it. The six-fold pattern mirrors the particle's symmetry. The drive
*and* the decay rate of an embedded emitter both come from this map — it is the
entry point of quantum-dynamics calculations downstream:

![Purcell map](https://raw.githubusercontent.com/claudio-sc/pysie2d/v0.2.0/figures/purcell_map.png)

Regenerate them with:

```bash
uv run python examples/convergence_study.py
uv run python examples/nearfield_map.py
uv run python examples/purcell_map.py
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

Full details and every sign/layout convention are in
[docs/conventions.md](https://github.com/claudio-sc/pysie2d/blob/main/docs/conventions.md). The analytic reference is Bohren &
Huffman, *Absorption and Scattering of Light by Small Particles*, ch. 8; the
surface-integral formulation follows [Valencia et al's formulation](https://doi.org/10.1364/JOSAB.20.002150).

## Validation

The physics test suite compares the solver against analytic Mie theory for a
circular cylinder: scattering / extinction / absorption efficiencies, the
optical theorem on a lossy particle, energy conservation on a lossless one, the
convergence rate, and the 2-D `1/√(kr)` far-field decay. At `nn = 300` the
efficiencies agree with Mie to a few parts in `10³`; the error decreases with
`nn` until it reaches the fixed angular-quadrature floor of the far-field
integrator. See `tests/` for the exact tolerances and the reasoning behind them.

The line-dipole / self-Green machinery (v0.2) is validated the same way:
reciprocity of the scattered field (to `10⁻⁶`), the free-space limit
(`LDOS → 1` far from the particle), LDOS positivity, and — the strong anchor —
the self-Green function of a circular cylinder against its closed-form
Graf-addition-theorem sum on both `Re S` and `Im S`. That near-field anchor
converges at first order in `nn`, so it is run at `nn = 1000` to reach `1 %`;
the resolved scattered-field sign convention is recorded in
[docs/conventions.md](https://github.com/claudio-sc/pysie2d/blob/main/docs/conventions.md).

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
result = BIESolver(geom, mat).scatter(wavelength=600.0)

print(result.efficiencies())                       # {'qsca', 'qext', 'qabs'}
```

Line-dipole emitter and Purcell effect:

```python
from pysie2d import BIESolver, Geometry, Material, relative_ldos

geom = Geometry.gielis(rad=200, n_pts=300, m=6, n1=6, n2=12, n3=12)  # Gielis star
solver = BIESolver(geom, Material(n_core=2.0))
print(relative_ldos(solver, wavelength=540.0, x_s=430.0, z_s=0.0))  # LDOS vs free space
```

## Performance

The system is a dense `2nn × 2nn` complex matrix; at `nn = 300` (a `600 × 600`
solve) a single wavelength takes below one second in a modern computer, so wavelength
sweeps are cheap serial `for` loops — no parallelism required.

For a Purcell map, every grid point is a different source position, hence a
different right-hand side — but the matrix `M(λ)` is the same for all of them.
`relative_ldos_map` therefore factorises `M` **once** with
`scipy.linalg.lu_factor` and reuses it across all sources (`lu_solve`), turning
what would be an hour-long sweep into a few seconds.

## Roadmap

- **v0.1.0** — core scattering: plane-wave excitation, near/far fields,
  cross-section efficiencies, Mie validation, convergence study, CI.
- **v0.2.0** — line-dipole (point-source) excitation and the self-Green
  function → relative LDOS / Purcell maps. _(this release)_
- **v0.3.0** — quasi-normal-mode extraction via Beyn's contour method,
  validated against analytic Mie resonances.

## License

MIT — see [LICENSE](LICENSE).
