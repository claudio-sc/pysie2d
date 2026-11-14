"""A particle 15 nm above a silver half-space.

Run: uv run python examples/halfspace.py
"""

from pysie2d import BIESolver, Geometry, HalfSpace, Material, relative_ldos

geom = Geometry.gielis(rad=100, n_pts=256, m=0, z0=115.0)  # 15 nm above z = 0
mat = Material(n_core=2.0, n_clad=1.0, pol=1)
silver = HalfSpace(-18.3 + 0.48j, z_int=0.0)  # absolute complex ε; Re ε < 0 is fine
solver = BIESolver(geom, mat, background=silver)

res = solver.scatter(wavelength=633.0, angle=20.0)  # |angle| < 90°, from the cover
cs = res.cross_sections()  # {'c_sca_up', 'c_sca_down', 'c_abs'}, nm
print(cs)
print(relative_ldos(solver, 633.0, 60.0, 330.0))  # LDOS above the interface

assert set(cs) == {"c_sca_up", "c_sca_down", "c_abs"}
# The particle is lossless, so it absorbs nothing; c_abs is a boundary flux and
# lands at round-off.
assert abs(cs["c_abs"]) < 1e-9 * cs["c_sca_up"]
