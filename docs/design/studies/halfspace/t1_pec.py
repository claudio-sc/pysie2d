import numpy as np
from hs import *
# Anchor (a): PEC vs Hankel, real and complex k. Error = max rel over 5 (X,Z) pairs and G, dG/dx', dG/dz'.
print("lam  pol R h | (M_full : maxrel err) for osc/panel = 2,1,0.5,0.25 ; p_tail=16, n_hump=32")
for lam in [500., 1600.]:
  for cplx in [0.0, 0.05]:
    lv = lam * (1 + 1j * cplx); k0 = 2 * PI / lv
    for pol in (2,):
      for (Rp, h) in [(100, 5), (100, 20), (500, 5), (500, 20)]:
        zmin, xmax = 2 * h, 2 * Rp
        pts = [(0, 2*h), (2*Rp, 2*h), (1.4*Rp, 2*h+Rp), (0, 2*h+4*Rp), (-2*Rp, 2*h+2*Rp)]
        X = np.array([p[0] for p in pts],float); Z = np.array([p[1] for p in pts],float)
        ex = pec_exact(X, Z, k0, pol)
        row = []
        for osc in [2, 1, 0.5, 0.25]:
          q, wq, info = make_path(k0, 1.0, 'pec', pol, zmin, xmax, 32, osc=osc)
          G = g_ind(X, Z, k0, 1.0, 'pec', pol, q, wq)
          err = max(np.max(np.abs(G[i]-ex[i])/np.max(np.abs(ex[i]))) for i in range(3))
          row.append(f"{2*len(q):5d}:{err:.1e}")
        print(f"{lv:.0f} {pol} {Rp} {h} kZmin={abs(k0)*zmin:.3f} kD={abs(k0)*xmax:.1f} |", " ".join(row))
