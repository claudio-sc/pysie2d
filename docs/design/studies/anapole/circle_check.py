import os
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("EIGEN_EPS_SCRIPTS", str(HERE.parents[4] / "eigen-eps/manuscripts/anapole-design-rules/scripts")))
from anapole_core import anapole_eps
from pysie2d import BIESolver, Geometry, Material
X = 0.5; LAM = 2*np.pi/X
geo = Geometry.gielis(1.0, 128, m=0)
for ch, pol, m in [("wire TM0", 2, 0), ("wire TE1", 1, 1)]:
    e0 = anapole_eps(ch, X)
    for e in e0*np.array([0.99, 1.0, 1.01]):
        r = BIESolver(geo, Material(n_core=np.sqrt(e), pol=pol)).scatter(LAM, angle=90)
        c = r.multipoles(mmax=6).c
        print(ch, f"eps {e:.5f}", " ".join(f"|c{k}|={abs(c[k+6]):.2e}" for k in range(0, 4)))
