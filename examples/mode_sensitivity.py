"""Adjoint dλ/dp for every mode, here for stretching one axis of a circle.

Run: uv run python examples/mode_sensitivity.py
"""

import numpy as np

from pysie2d import Geometry, Material, QNMSolver

geom = Geometry.gielis(rad=200, n_pts=200, m=0)
mat = Material(n_core=3.0, n_clad=1.0, pol=2)
res = QNMSolver(geom, mat).modes(745 + 2j, 775 + 15j).refine()
frozen = res.geometry.parametrisation  # the node map must be frozen


def wider(delta):
    """The same circle with one semi-axis b = 1 + delta, on the frozen nodes."""
    return Geometry.gielis(
        rad=200, n_pts=200, m=4, b=1.0 + delta, parametrisation=frozen
    ), mat


dlam = res.sensitivity(wider)
print(dlam)  # ≈ λ/2 for both partners

# Stretching one axis is half of a uniform dilation, and λ scales with size.
assert np.allclose(dlam, res.wavelengths / 2, rtol=1e-2)
