import numpy as np
from hs import *
def run(lv, Rp, h, pol=2, nh=32, osc=2, p=16, dfrac=0.4, eps2='pec', ref=None):
    k0 = 2*PI/lv; zmin, xmax = 2*h, 2*Rp
    pts = [(0, 2*h), (2*Rp, 2*h), (1.4*Rp, 2*h+Rp), (0, 2*h+4*Rp), (-2*Rp, 2*h+2*Rp)]
    X = np.array([p_[0] for p_ in pts],float); Z = np.array([p_[1] for p_ in pts],float)
    ex = pec_exact(X, Z, k0, pol) if ref is None else ref
    q, wq, info = make_path(k0, 1.0, eps2, pol, zmin, xmax, nh, osc=osc, p_tail=p, delta_frac=dfrac)
    G = g_ind(X, Z, k0, 1.0, eps2, pol, q, wq)
    return 2*len(q), max(np.max(np.abs(G[i]-ex[i])/np.max(np.abs(ex[i]))) for i in range(3))
print("hump nodes sweep (osc=2), lam=500(1+0.05i), R=500,h=20, delta 0.4k")
for nh in [8,12,16,24,32,48,64]:
    print(nh, *run(500*(1+.05j),500,20,nh=nh), *run(500,500,20,nh=nh))
print("delta sweep nh=24")
for d in [0.1,0.2,0.3,0.4,0.6,0.8]:
    print(d, *run(500*(1+.05j),500,20,nh=24,dfrac=d), *run(500*(1+.1j),500,20,nh=24,dfrac=d))
print("osc per panel sweep p=16 and p=24,nh=32")
for osc in [2,4,8,16]:
  for (R_,h_) in [(100,5),(500,5),(500,20)]:
    print(osc, R_, h_, *run(500*(1+.05j),R_,h_,osc=osc), *run(500*(1+.05j),R_,h_,osc=osc,p=24), *run(1600,R_,h_,osc=osc))
