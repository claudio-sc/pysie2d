"""Guided modes of a dielectric slab, and a particle coupling into them.

Run: uv run python examples/stack_modes.py
"""

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

assert abs(amps.balance()) < 1e-10
