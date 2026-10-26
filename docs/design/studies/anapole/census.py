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
geo = Geometry.gielis(1.0, 256, m=4, n1=4, n2=4, n3=4)
for ch, m in [("TM", 0), ("TE", 1)]:
    pol = POL[ch]
    B = class_basis(geo, CLASSES[ch]); Lb = B.T @ dtn(geo, pol, X, 1.0) @ B
    eig = lambda e: np.linalg.eigvals(B.T @ dtn(geo, pol, X, e) @ B - Lb)
    es = np.linspace(40, 60, 401)
    neg = np.array([np.sum(eig(e).real < 0) for e in es])
    ev = []
    for k in np.nonzero(np.diff(neg))[0]:
        kind = "ITE (count up)" if neg[k+1] > neg[k] else "pole (count down)"
        if neg[k+1] > neg[k]:
            f = lambda e: (lambda l: l[np.argmin(abs(l))].real)(eig(e))
            r = brentq(f, es[k], es[k+1], xtol=1e-10)
        else:
            r = 0.5*(es[k]+es[k+1])
        print(f"{ch} (pol={pol}): {kind} at eps = {r:.6f}")
    def cm(e):
        r = BIESolver(geo, Material(n_core=np.sqrt(e), pol=pol)).scatter(LAM, incident_rhs=cls[pol])
        return abs(r.multipoles(mmax=12).c[12+m])
    lo = {"TM": (49, 51), "TE": (52.5, 53.5)}[ch]
    o = minimize_scalar(cm, bounds=lo, method="bounded", options={"xatol": 1e-9})
    print(f"   driven |c{m}| min at eps = {o.x:.6f}, value {o.fun:.2e}")
