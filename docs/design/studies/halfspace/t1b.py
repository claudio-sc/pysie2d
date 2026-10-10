import numpy as np
from hs import *
lam=1600.; k0=2*PI/lam; pol=2; Rp=100; h=5
pts = [(0, 2*h), (2*Rp, 2*h), (1.4*Rp, 2*h+Rp), (0, 2*h+4*Rp), (2*Rp, 2*h+2*Rp)]
X = np.array([p[0] for p in pts],float); Z = np.array([p[1] for p in pts],float)
ex = pec_exact(X, Z, k0, pol)
for nh in [16,32,48,64,96]:
  for npan in [80,160,320]:
    for L in [38., 45.]:
      q, wq, info = make_path(k0, 1.0, 'pec', pol, 2*h, 2*Rp, nh, npan, L_decay=L)
      G = g_ind(X, Z, k0, 1.0, 'pec', pol, q, wq)
      print(nh, npan, L, np.abs(G[0]-ex[0])/np.abs(ex[0]))
