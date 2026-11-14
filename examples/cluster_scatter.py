"""A two-particle cluster, then the same cluster over many materials.

Run: uv run python examples/cluster_scatter.py
"""

import numpy as np

from pysie2d import Cluster, ClusterBIESolver, Geometry, Material

left = Geometry.gielis(rad=200, n_pts=200, m=0, x0=-350.0)  # circle, nm
right = Geometry.gielis(rad=150, n_pts=200, m=4, b=1.0, x0=350.0)
cluster = Cluster([left, right])

mats = [
    Material(n_core=1.5, n_clad=1.0, pol=2),  # TE, one background
    Material(n_core=2.0, n_clad=1.0, pol=2),
]
solver = ClusterBIESolver(cluster, mats, pol=2)
res = solver.scatter(wavelength=633.0, angle=30.0)

amp, angles = res.far_field()  # total amplitude of the cluster
cs = res.cross_sections()  # {'c_sca', 'c_ext', 'c_abs'}, nm
print(cs)
print(res.resolution())  # points per interior wavelength

# Many materials at one wavelength: build the material-independent part once.
fast = solver.at_wavelength(633.0)  # about one scatter() call
for eps in np.linspace(2.0, 4.0, 5) + 0.1j:
    new = [Material.from_eps(eps, pol=2), mats[1]]
    print(eps, fast.scatter(new, angle=30.0).cross_sections()["c_abs"])

# The fast path gives the same answer as a full solve, to round-off.
same = fast.scatter(mats, angle=30.0).cross_sections()
assert np.isclose(same["c_ext"], cs["c_ext"], rtol=1e-10)
