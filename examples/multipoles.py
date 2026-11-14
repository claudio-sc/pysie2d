"""Cylindrical-harmonic coefficients of the scattered field.

Run: uv run python examples/multipoles.py
"""

import numpy as np

from pysie2d import BIESolver, Geometry, Material

geom = Geometry.gielis(rad=200, n_pts=200, m=4, b=1.0)  # rounded square, nm
mat = Material(n_core=1.5, n_clad=1.0, pol=1)  # TM
res = BIESolver(geom, mat).scatter(wavelength=600.0, angle=30.0)
mp = res.multipoles(mmax=8)

print(mp.c)  # signed orders m = −mmax … mmax
print(mp.c_plus)  # symmetric c_m + c_{−m}
print(mp.c_minus)  # antisymmetric c_m − c_{−m}

# The expansion reproduces the solver's own field on any larger circle.
r0 = 4.0 * geom.rad * 2.29  # outside the circumscribing circle
theta = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
field = res.eval_field(r0 * np.sin(theta), r0 * np.cos(theta))
assert np.allclose(mp.reconstruct(r0, theta), field, rtol=1e-6, atol=1e-9)
