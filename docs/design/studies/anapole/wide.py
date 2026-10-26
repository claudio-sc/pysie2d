import numpy as np
from pysie2d import BIESolver, Geometry, Material
from pysie2d.sources import plane_wave_rhs as pw
X = 0.5; LAM = 2*np.pi/X
cls = {1: lambda nn,k,f,g: pw(nn,90,k,f,g)-pw(nn,270,k,f,g),
       2: lambda nn,k,f,g: sum(pw(nn,a,k,f,g) for a in (0,90,180,270))}
for p in (2, 4):
    geo = Geometry.gielis(1.0, 256, m=4, n1=p, n2=p, n3=p)
    for pol, m in [(2, 0), (1, 1)]:
        es = np.linspace(35, 70, 351)
        c = []
        for e in es:
            r = BIESolver(geo, Material(n_core=np.sqrt(e), pol=pol)).scatter(LAM, incident_rhs=cls[pol])
            c.append(abs(r.multipoles(mmax=12).c[12+m]))
        c = np.array(c)
        loc = [i for i in range(1, len(c)-1) if c[i] < c[i-1] and c[i] < c[i+1]]
        mx = [i for i in range(1, len(c)-1) if c[i] > c[i-1] and c[i] > c[i+1]]
        print(f"p={p} pol={pol} |c{m}|: minima " + ", ".join(f"{es[i]:.2f}:{c[i]:.1e}" for i in loc)
              + " | maxima " + ", ".join(f"{es[i]:.2f}:{c[i]:.2f}" for i in mx))
