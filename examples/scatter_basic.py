"""Plane-wave scattering and a line-dipole LDOS, the two entry points.

Run: uv run python examples/scatter_basic.py
"""

import numpy as np

from pysie2d import BIESolver, Geometry, Material, relative_ldos
from pysie2d.reference.mie import efficiencies as mie_efficiencies

# A circular cylinder, radius 200 nm, n = 1.5 in vacuum, TE.
geom = Geometry.gielis(rad=200, n_pts=300, m=0)
mat = Material(n_core=1.5, n_clad=1.0, pol=2)
result = BIESolver(geom, mat).scatter(wavelength=600.0)  # vacuum nm

q = result.efficiencies()  # {'qsca', 'qext', 'qabs'}
print(q)

# The circle has a closed form: Mie theory. Spectral convergence puts the
# solver at round-off long before n_pts = 300.
x = 2 * np.pi * 1.0 * 200 / 600.0
assert abs(q["qsca"] - mie_efficiencies(x, 1.5)["Q_sca_TE"]) < 1e-10

# A Gielis star and a line dipole 30 nm outside its tips.
star = Geometry.gielis(rad=200, n_pts=300, m=6, n1=6, n2=12, n3=12)
solver = BIESolver(star, Material(n_core=2.0))
ldos = relative_ldos(solver, wavelength=540.0, x_s=430.0, z_s=0.0)
print(ldos)  # relative to the unbounded background
