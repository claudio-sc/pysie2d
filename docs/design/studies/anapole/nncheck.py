import numpy as np
from scipy.optimize import minimize_scalar
from pysie2d import BIESolver, Geometry, Material
from pysie2d.sources import plane_wave_rhs as pw
X = 0.5; LAM = 2*np.pi/X
cls = {1: lambda nn,k,f,g: pw(nn,90,k,f,g)-pw(nn,270,k,f,g),
       2: lambda nn,k,f,g: sum(pw(nn,a,k,f,g) for a in (0,90,180,270))}
for pol, m, b in [(2, 0, (49.5, 50.5)), (1, 1, (52.8, 53.2))]:
    for nn in (128, 256, 512):
        geo = Geometry.gielis(1.0, nn, m=4, n1=4, n2=4, n3=4)
        def cm(e, mm=m):
            r = BIESolver(geo, Material(n_core=np.sqrt(e), pol=pol)).scatter(LAM, incident_rhs=cls[pol])
            return r.multipoles(mmax=12).c
        o = minimize_scalar(lambda e: abs(cm(e)[12+m]), bounds=b, method="bounded", options={"xatol": 1e-11})
        c = cm(o.x)
        print(f"pol={pol} nn={nn}: driven zero eps = {o.x:.8f}, |c{m}| = {abs(c[12+m]):.1e}, other orders max {np.delete(abs(c), 12+m).max():.1e}")
