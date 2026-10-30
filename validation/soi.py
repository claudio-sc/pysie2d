"""v1.3's waveguide case, defined once for MEEP and for pysie2d.

Like ``gielis.py`` for shapes: numpy only, so the MEEP driver (conda) and the
reference (uv) read the same slab, materials and wavelength grid without either
importing the other's tool.
"""

import numpy as np

NM_PER_A = 100.0
N_SI = 3.476
N_OX = 1.444
T_SI_NM = 220.0
"""Slab thickness. Single-mode in both polarisations only for λ ≳ 1390 nm."""

WAVELENGTHS = np.linspace(1450.0, 1750.0, 31)
"""The frozen grid, vacuum nm. Starts 60 nm above the TE₁/TM₁ cutoff (≈ 1390 nm
for a symmetric 220 nm slab, 2d·√(n_Si² − n_ox²)); at 1300 nm the slab guides
two modes and a band-1 S-parameter would no longer be the whole guided account."""
