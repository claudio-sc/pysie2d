"""The opt-in contours of QNMSolver.modes, against the default rectangle.

Run: uv run python examples/qnm_contours.py
"""

import numpy as np

from pysie2d import Geometry, Material, QNMSolver

geom = Geometry.gielis(rad=200, n_pts=100, m=0)  # circle, nm
mat = Material(n_core=3.0, n_clad=1.0, pol=2)  # TE
qnm = QNMSolver(geom, mat)
z_lo, z_hi = 745 + 2j, 775 + 15j  # the mode sits near the centre, not in a corner

rect = qnm.modes(z_lo, z_hi)  # the default, unchanged
res = qnm.modes(z_lo, z_hi, contour="ellipse")  # same nodes, ellipse
ada = qnm.modes(z_lo, z_hi, contour="ellipse", adaptive=True)  # doubles to tol
fit = qnm.modes(z_lo, z_hi, contour="ellipse", adaptive=True, interpolate=True)

for name, r in [("rectangle", rect), ("ellipse", res), ("adaptive", ada), ("fit", fit)]:
    print(f"{name:9s} {r.contour:9s} {r.wavelengths}")

# Every route returns the same degenerate pair, 760.687 + 7.948j.
for r in (res, ada, fit):
    assert len(r.wavelengths) == 2
    assert np.allclose(np.sort_complex(r.wavelengths), rect.wavelengths, atol=1e-6)
