import numpy as np, sys
sys.path.insert(0, ".")
from anchor import root, D
from epsbeyn import EpsOperator, solve_box
from pysie2d import Geometry
x = 2*np.pi*100/600
# poles just above the top edge (Im between -0.3 and 0): higher angular orders are the high-Q ones
for n in (2, 3, 4):
    for g in (12-0.05j, 20-0.05j, 28-0.02j, 35-0.01j):
        try:
            r = root(g, n, x, 2)
            if -0.3 < r.imag < 0 and 0 < r.real < 40: print(f"  TE n={n}: eps_m={r:.6f}  (above the box top edge)")
        except Exception: pass
op = EpsOperator(Geometry.gielis(rad=100.0, n_pts=64, m=0), pol=2, wavelength=600.0, branch="upper")
eigs, bm, t = solve_box(op, 0.5-4.0j, 30.0-0.3j, nq=160, n_probe=24, refine=False)
print("long box found:", np.round(np.sort_complex(eigs), 5))
