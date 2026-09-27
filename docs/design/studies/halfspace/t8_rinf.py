"""(4) Gain from subtracting R_inf (i/4) H0(k1 rho_img) in TM.  Error vs M for
tail length L (in units 1/Z_min) and density; reference = unsubtracted s=3 path."""
import numpy as np, warnings
from hs import *
warnings.simplefilter("ignore")
def P(Rp,h): return np.array([[0,2*h],[2*Rp,2*h],[1.4*Rp,2*h+Rp],[0,2*h+4*Rp],[-2*Rp,2*h+2*Rp]],float)
for name, lam, e2 in [("glass",500.,2.25),("Si",1600.,12.25),("Ag",633.,-18.3+.48j),("Au",800*(1+.05j),-24.1+1.5j)]:
    k0=2*PI/lam
    for Rp,h in [(100,5),(500,5)]:
        Pt=P(Rp,h)
        qr,wr,_=make_path(k0,1.0,e2,1,2*h,2*Rp,n_hump=400,osc=0.7,L_decay=45.)
        ref=np.array(g_ind(Pt[:,0],Pt[:,1],k0,1.0,e2,1,qr,wr))
        sc=np.max(np.abs(ref),axis=1,keepdims=True)
        for L in [10,15,20,25,38]:
            q,wq,_=make_path(k0,1.0,e2,1,2*h,2*Rp,L_decay=L)
            a=np.max(np.abs(np.array(g_ind(Pt[:,0],Pt[:,1],k0,1.0,e2,1,q,wq))-ref)/sc)
            b=np.max(np.abs(np.array(g_ind_sub(Pt[:,0],Pt[:,1],k0,1.0,e2,1,q,wq))-ref)/sc)
            print(f"{name:5s} R{Rp} h{h} Rinf={abs(r_inf(1.0,e2,1)):.2f} L={L:2d}/Zmin M={2*len(q):5d}  plain {a:.0e}  subtracted {b:.0e}")
