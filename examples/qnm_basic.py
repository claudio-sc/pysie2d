"""Every quasi-normal mode inside one box of the complex wavelength plane.

Run: uv run python examples/qnm_basic.py
"""

import numpy as np

from pysie2d import Geometry, Material, QNMSolver

geom = Geometry.gielis(rad=200, n_pts=200, m=0)  # circle, nm
mat = Material(n_core=3.0, n_clad=1.0, pol=2)  # TE
res = QNMSolver(geom, mat).modes(745 + 2j, 775 + 15j)  # box corners, vacuum nm

print(res.wavelengths)  # 760.687 + 7.948j, twice: a degenerate pair
print(res.quality_factors)  # Q = Re λ / (2 Im λ)
print(res.multiplicity)  # [2 2]
print(res.edge_margin)  # contour-quality diagnostic; near zero is a warning

# Every n >= 1 mode of a circle is doubly degenerate.
assert len(res.wavelengths) == 2
assert np.allclose(res.wavelengths, 760.687 + 7.948j, atol=1e-3)
