# Layered backgrounds

Two backgrounds go under the particles: a half-space (`HalfSpace`) and N
dielectric layers over a bottom half-space (`Stack`). Both are passed as
`background=` and both are additive: without the argument nothing changes, bit
for bit. Particles and sources always sit in the cover, above the top
interface; fields are evaluated everywhere.

These are 2-D line-source fields: absolute LDOS next to an interface is not a
3-D number, while `Q`, detunings and ratios carry over.

## A half-space

`background=HalfSpace(eps_sub, z_int)` on `BIESolver`, `ClusterBIESolver` and
`QNMSolver` puts a half-space below the plane `z = z_int`. From
[`examples/halfspace.py`](../examples/halfspace.py):

```python
from pysie2d import BIESolver, Geometry, HalfSpace, Material, relative_ldos

geom = Geometry.gielis(rad=100, n_pts=256, m=0, z0=115.0)  # 15 nm above z = 0
mat = Material(n_core=2.0, n_clad=1.0, pol=1)
silver = HalfSpace(-18.3 + 0.48j, z_int=0.0)  # absolute complex ε; Re ε < 0 is fine
solver = BIESolver(geom, mat, background=silver)

res = solver.scatter(wavelength=633.0, angle=20.0)  # |angle| < 90°, from the cover
print(res.cross_sections())  # {'c_sca_up', 'c_sca_down', 'c_abs'}, nm
print(relative_ldos(solver, 633.0, 60.0, 330.0))  # LDOS above the interface
```

`HalfSpace.pec(z_int)` is a perfect conductor, evaluated as the exact image term
with no integral. `Material.from_eps(eps, n_clad, pol)` builds a particle from an
absolute complex permittivity, which a real `n_core` cannot express for a metal.
`eps_sub` may be a callable `ε(λ_vac)` for driven solves; `QNMSolver` rejects it,
because tabulated data is not holomorphic and the contour argument would break
silently.

The reflected field is a Sommerfeld integral on a **deformed** path, so it is
valid at complex wavelength; the real-axis integral with the usual `Im α ≥ 0`
rule returns the *incoming* wave there ([conventions](conventions.md) §15). The
reflected blocks cost about a tenth of the free-space assembly at `nn = 256`.

What changes on the results:

- `far_field()` is the **upward** amplitude on `[−π/2, π/2]`;
  `far_field_down()` is the transmitted one on `[π/2, 3π/2]`, normalised with
  the substrate wavenumber. It exists only below a lossless dielectric and
  raises `NoFarFieldError` (a `ValueError`) over PEC, a lossy substrate or
  `Re ε ≤ 0` ([conventions](conventions.md) §16).
- `cross_sections()` returns `{'c_sca_up', 'c_sca_down', 'c_abs'}` and
  `efficiencies()` `{'qsca_up', 'qsca_down', 'qabs'}`. The free-space keys are
  absent, not raising, because `c_ext` and the total `c_sca` are undefined when
  the incident wave is reflected. `c_sca_down` is the scattered power into the
  substrate from the interface flux, so it covers lossy and metallic substrates
  too (absent at complex λ); with it a dipole's power balances,
  `P_up + P_down + P_abs = P_in`, to 1e-13. The one leak is a lossless TM metal
  with `Re ε < −n_clad²`, whose surface plasmon carries power along the
  interface. `c_abs` comes from the boundary flux and is exact.
- `eval_field` returns the scattered field, above the interface and below it
  (the transmitted part; 0 below PEC); `incident_field` returns the background
  field (incident plus reflected above, transmitted below), so that their sum
  is the total field everywhere. `relative_ldos` is still normalised to the
  unbounded cover, so it equals the substrate-only enhancement with no particle.
- `multipoles()` and `QNMResult.sensitivity` raise `NotImplementedError`;
  `QNMResult.refine` works.

Guards: a boundary too coarse for its distance to the interface warns
(`InterfaceGapWarning`: local node spacing over twice the gap above 0.25 for TE,
0.2 for TM, so a flat facet is caught), a search box whose hump cannot clear a
singularity raises, and so does a lossless `ε_sub = −n_clad²` (a
surface-plasmon resonance with an infinite pole).

Validation: the PEC image problem (a particle plus its mirror, solved by
`ClusterBIESolver`, to ~1e-13 in φ, χ and the near field), the PEC closed form,
a real-axis QUADPACK reference, and Mie. Fields below the interface are checked
against continuity, the free-space limit and energy conservation. For QNMs over
a dielectric substrate, MEEP harminv poles of a split whispering-gallery pair
over glass are frozen in `tests/data/qnm/meep-substrate-qnm.json`; the
substrate-induced shift agrees within 1e-2 nm against shifts of 0.4–1.4 nm
(`tests/test_external_substrate_qnm.py`). Metal substrates are checked by
closed forms and energy conservation only. Details in
[validation.md](validation.md).

## A layered stack and its guided modes

`background=Stack(layers, eps_bot, z_top)` puts N dielectric layers over a
bottom half-space. The stack carries guided modes, and the solver projects
scattered fields onto them. From
[`examples/stack_modes.py`](../examples/stack_modes.py):

```python
from pysie2d import BIESolver, Geometry, Material, Stack

# One 220 nm layer of n = 3.476 on an n = 1.444 half-space, under n = 1.444.
slab = Stack(layers=((3.476**2, 220.0),), eps_bot=1.444**2)
modes = slab.guided_modes(1550.0, pol=2, n_clad=1.444)  # β, n_eff, N, u(z)
geom = Geometry.gielis(rad=80, n_pts=160, m=4, z0=110.0)  # 30 nm above the slab
solver = BIESolver(geom, Material(n_core=3.476, n_clad=1.444, pol=2), slab)

amps = solver.scatter_dipole(1550.0, 150.0, 60.0).mode_amplitudes(x_ref=0.0)
print(amps.efficiency(0, +1))  # into the fundamental TE mode, towards +x
print(amps.balance())  # power account closes to round-off
s_col = solver.scatter_mode(1550.0, 0, +1, x_ref=0.0).mode_amplitudes(0.0)  # S column
```

- `mode_amplitudes` gives complex amplitudes of the power-normalised modes,
  phase-referred to `x_ref`, and the full power account (up, down, absorbed,
  guided ±x). Under mode incidence they are a column of the scattering matrix.
- `scatter_dipole(..., moment=(p_x, p_z))` is an in-plane electric dipole (TM);
  spin-locked moments couple one way, the near-field interference effect of
  Rodríguez-Fortuño et al., [Science 340, 328 (2013)](https://doi.org/10.1126/science.1233739),
  and Petersen, Volz and Rauschenbeutel, [Science 346, 67 (2014)](https://doi.org/10.1126/science.1257671).
- `pysie2d.mode_sensitivity(at, run)` differentiates any of these outputs by a
  central difference on the frozen node set.
- Guided outputs need a lossless stack at real λ; fields and LDOS accept loss
  and complex λ. `QNMSolver` does not take a stack.

Validation: the three-layer closed form for the modes, independent real-axis
references for the reflected and transmitted Green functions, power balance and
reciprocity as equalities, the Born limit, and the exact translation law
([validation.md](validation.md), rows GW1–GW18; [conventions](conventions.md)
§17). Externally, the mode-incidence reflection and transmission of a
mirror-asymmetric Gielis pair 200 nm above the slab are compared with MEEP
eigenmode coefficients extrapolated to zero grid spacing, frozen in
`tests/data/` with tolerances from MEEP's convergence alone. One known gap: TM
complex transmission misses its tolerance by 1.3× (under 1 % of `|1 − t|`), and
is a strict expected failure rather than a wider tolerance.
