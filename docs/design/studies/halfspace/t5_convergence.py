"""Pointwise convergence vs path node count. Refinement knob s scales hump nodes
(n_hump = s*n_rule) and tail density (osc per 24-node panel = 4/s).  Reference:
PEC closed form where R = -+1, otherwise the s = 3 path (consistency ref; its
correctness is anchored separately by t2/t3)."""
import numpy as np, warnings
from hs import *
warnings.simplefilter("ignore")
def pts_for(Rp, h):
    return np.array([[0, 2*h], [2*Rp, 2*h], [1.4*Rp, 2*h+Rp], [0, 2*h+4*Rp], [-2*Rp, 2*h+2*Rp]], float)
def ev(k0, e2, pol, Rp, h, s):
    P = pts_for(Rp, h)
    q, wq, info = make_path(k0, 1.0, e2, pol, 2*h, 2*Rp, osc=4.0/s, delta_frac=0.4)
    nh = int(round(s*max(48, np.ceil(13*info['T']/info['delta']))))
    q, wq, info = make_path(k0, 1.0, e2, pol, 2*h, 2*Rp, n_hump=nh, osc=4.0/s)
    return 2*len(q), np.array(g_ind(P[:,0], P[:,1], k0, 1.0, e2, pol, q, wq))
print("case | M:err (max rel over 5 pts x {G, dG/dx', dG/dz'}) for s = 0.25 0.5 0.75 1 1.5 2")
for name, lam, e2, pol in [("PEC",500.,'pec',2),("PEC",1600.*(1+.05j),'pec',1),("glass",500.,2.25,2),
                           ("glass",800*(1+.05j),2.25,1),("Si",1600.,12.25,2),("Ag",633.,-18.3+.48j,1),
                           ("Ag",633*(1+1/6j*-1),-18.3+.48j,1),("Au",800*(1+.05j),-24.1+1.5j,1)]:
    k0 = 2*PI/lam
    for Rp, h in [(100,5),(500,5),(500,20)]:
        if e2 == 'pec':
            P = pts_for(Rp,h); ref = np.array(pec_exact(P[:,0],P[:,1],k0,pol))
        else:
            ref = ev(k0,e2,pol,Rp,h,3)[1]
        row=[]
        for s in [0.25,0.5,0.75,1,1.5,2]:
            M, v = ev(k0,e2,pol,Rp,h,s)
            err = np.max(np.abs(v-ref)/np.max(np.abs(ref),axis=1,keepdims=True))
            row.append(f"{M:5d}:{err:.0e}")
        print(f"{name:5s} {np.real(lam):5.0f}{np.imag(lam):+4.0f}i pol{pol} R{Rp} h{h:2d} kZ={abs(k0)*2*h:.3f} kD={abs(k0)*2*Rp:5.2f} | "+" ".join(row))
