import numpy as np, warnings
from hs import *
from refquad import g_ref
warnings.simplefilter("ignore")
for name, lam, e2, pol, Rp, h in [("Si",1600.,12.25,2,100,5), ("glass",1600.,2.25,2,500,20), ("glass",500.,2.25,2,100,5)]:
    k0=2*PI/lam
    pts = [(0, 2*h), (2*Rp, 2*h), (1.4*Rp, 2*h+Rp), (0, 2*h+4*Rp)]
    ref = np.array([g_ref(X, Z, k0, 1.0, e2, pol) for X, Z in pts])
    ref2 = np.array([g_ref(X, Z, k0, 1.0, e2, pol, s_max=np.arcsinh(60/(k0*Z))) for X, Z in pts])
    print(name, "quad vs quad(longer s)", np.max(np.abs(ref-ref2))/np.max(np.abs(ref)), "|G|=",np.abs(ref))
    for nh in [48, 96, 192]:
      for d in [0.4, 1.0]:
        q,wq,_ = make_path(k0,1.0,e2,pol,2*h,2*Rp,nh,osc=2,p_tail=24,delta_frac=d)
        num = np.array([g_ind(X, Z, k0, 1.0, e2, pol, q, wq)[0] for X, Z in pts])
        print("  nh",nh,"d",d, np.abs(num-ref)/np.max(np.abs(ref)))
