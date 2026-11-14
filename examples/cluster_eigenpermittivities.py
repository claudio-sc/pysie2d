"""Permittivities at which a particle supports a field with no source.

Run: uv run python examples/cluster_eigenpermittivities.py
"""

import numpy as np

from pysie2d import Cluster, ClusterBIESolver, Geometry, Material

circle = Geometry.gielis(rad=100, n_pts=120, m=0)
solver = ClusterBIESolver(Cluster([circle]), [Material(n_core=3.0, pol=2)], pol=2)
fast = solver.at_wavelength(633.0)

modes = fast.eigenpermittivities(12.0 - 1.4j, 15.0 - 0.1j)  # box corners in ε
print(modes.eps, modes.converged)  # also .vectors, .rank, .sv_ratio

# One degenerate pair, as every n >= 1 resonance of a circle is.
assert len(modes.eps) == 2 and np.all(modes.converged)
assert np.allclose(modes.eps, 13.868 - 0.527j, atol=1e-3)
