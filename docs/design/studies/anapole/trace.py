import os
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq, minimize_scalar
HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("EIGEN_EPS_SCRIPTS", str(HERE.parents[4] / "eigen-eps/manuscripts/anapole-design-rules/scripts")))
from pysie2d_crosscheck import dtn, class_basis, CLASSES, POL
from pysie2d import BIESolver, Geometry, Material
from pysie2d.sources import plane_wave_rhs as pw
X = 0.5; LAM = 2*np.pi/X
cls = {1: lambda nn,k,f,g: pw(nn,90,k,f,g)-pw(nn,270,k,f,g),
       2: lambda nn,k,f,g: sum(pw(nn,a,k,f,g) for a in (0,90,180,270))}
prev = {"TM": (59.76, 59.80), "TE": (60.82, 60.82)}
for p in (2.0, 2.25, 2.5, 3.0, 3.5, 4.0):
    geo = Geometry.gielis(1.0, 256, m=4, n1=p, n2=p, n3=p)
    row = f"p={p:4.2f}"
    for ch, m in [("TM", 0), ("TE", 1)]:
        pol = POL[ch]
        B = class_basis(geo, CLASSES[ch]); Lb = B.T @ dtn(geo, pol, X, 1.0) @ B
        eig = lambda e: np.linalg.eigvals(B.T @ dtn(geo, pol, X, e) @ B - Lb)
        es = np.linspace(42, 64, 111)
        neg = np.array([np.sum(eig(e).real < 0) for e in es])
        ites = []
        for k in np.nonzero(np.diff(neg) > 0)[0]:
            f = lambda e: (lambda l: l[np.argmin(abs(l))].real)(eig(e))
            ites.append(brentq(f, es[k], es[k+1], xtol=1e-9))
        def cm(e):
            r = BIESolver(geo, Material(n_core=np.sqrt(e), pol=pol)).scatter(LAM, incident_rhs=cls[pol])
            return abs(r.multipoles(mmax=12).c[12+m])
        g = prev[ch][1]; grid = np.linspace(g-6, g+2, 81); vals = [cm(e) for e in grid]
        k = int(np.argmin(vals))
        o = minimize_scalar(cm, bounds=(grid[max(k-1,0)], grid[min(k+1,80)]), method="bounded", options={"xatol":1e-9})
        prev[ch] = (None, o.x)
        row += f" | {ch}/pol{pol}: ITE {', '.join(f'{x:.4f}' for x in ites)}  driven|c{m}|=0 @ {o.x:.4f} ({o.fun:.0e})"
    print(row, flush=True)
