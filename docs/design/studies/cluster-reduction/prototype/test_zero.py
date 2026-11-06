import numpy as np, sys
sys.path.insert(0, ".")
from epsbeyn import EpsOperator
from pysie2d import Geometry
g = Geometry.gielis(rad=100.0, n_pts=64, m=0)
for pol in (1, 2):
    op = EpsOperator(g, pol, 600.0, branch="upper")
    for e in (1e-3*(-1-1j), 1e-6*(-1-1j), 1e-9*(-1-1j), 0.05-0.05j):
        s = np.linalg.svd(op.M(e), compute_uv=False)
        print(f"pol={pol} eps={e:.1e}: smin/smax={s[-1]/s[0]:.1e}")
