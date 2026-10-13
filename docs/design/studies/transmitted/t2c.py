import time, numpy as np
from pysie2d import HalfSpace, relative_ldos
from pysie2d.layered import alpha, _normalised
from t2_power import LAM, XS, ZS, solver, powers
from t2b import p_up_split
import tproto
from tproto import flux_rule, flux_weight, spectrum, tau, PI
t0=time.perf_counter()
def p_down_custom(s, widths, n_pan=24):
    res = s.scatter_dipole(LAM, XS, ZS); k=res.wnum_bg; pol=s.material.pol
    eps = s.background.eps_rel(1.33, LAM); bnds,_,_,_,z_int = res._substrate()
    g=s.geometry; h_min=min(g.g.min(), ZS)-z_int; D=max(np.ptp(np.r_[g.f,XS]), 2*h_min)
    q,w = flux_rule(k, eps, pol, D, 2*h_min, n_pan=n_pan, widths=widths)
    a1=alpha(q,k); A=spectrum(q,a1,bnds,z_int)+np.exp(-1j*q*XS+1j*a1*(ZS-z_int))
    return np.sum(w*flux_weight(q,k,eps,pol)*np.abs(tau(q,k,eps,pol)*A)**2)/(8*PI), q.size
for name, e in (("metal -1.0 lossless", -1.0+0j),):
    for pol in (2,1):
        s=solver(pol, HalfSpace(e)); res=s.scatter_dipole(LAM,XS,ZS)
        up=p_up_split(s); g=s.geometry; nn=g.n_pts
        ab=-np.imag(g.delt*np.sum(np.conj(res.ei[:nn])*res.ei[nn:])); pin=relative_ldos(s,LAM,XS,ZS)/4
        print(f"{name} pol={pol}: balance with P_down=0: {(up+ab-pin)/pin:+.1e}")
W3=(1,10,100,1000); W6=tuple(10.0**m for m in range(0,7)); W9=tuple(10.0**m for m in range(0,10))
for name, e in (("glass+1e-6j",2.25+1e-6j),("silver+0.01j",-18.3+0.01j),("silver+1e-4j",-18.3+1e-4j),("silver",-18.3+0.48j)):
    for pol in (2,1):
        s=solver(pol, HalfSpace(e)); res=s.scatter_dipole(LAM,XS,ZS)
        up=p_up_split(s); g=s.geometry; nn=g.n_pts
        ab=-np.imag(g.delt*np.sum(np.conj(res.ei[:nn])*res.ei[nn:])); pin=relative_ldos(s,LAM,XS,ZS)/4
        out=[]
        for W in (W3,W6,W9):
            pd,nq=p_down_custom(s,W); pd48,_=p_down_custom(s,W,48)
            out.append(f"bal={(up+pd+ab-pin)/pin:+.1e} 24v48={(pd-pd48)/pin:+.1e} nq={nq}")
        print(f"{name:13s} pol={pol}: W3 {out[0]} | W6 {out[1]} | W9 {out[2]}", flush=True)
print(f"elapsed {time.perf_counter()-t0:.1f}s")
