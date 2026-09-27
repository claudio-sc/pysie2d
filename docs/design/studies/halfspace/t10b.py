import numpy as np, warnings
from hs import *
from gemm import reflected_blocks
import sys; sys.path.insert(0, "/Users/claudiosc/Documents/sie/src")
from pysie2d import Geometry
warnings.simplefilter("ignore")
def pecb(k1,pol,f,g,df,dg):
    nn=len(f); hs_=2*PI/nn; X=f[:,None]-f[None,:]; Z=g[:,None]+g[None,:]
    G,Gx,Gz=pec_exact(X,Z,k1,pol); return hs_*(Gx*dg[None,:]-Gz*df[None,:]), hs_*G
for lam in [500*(1+.05j), 500.]:
  k0=2*PI/lam
  for Rp in [1000,2000,4000]:
    nn=max(128,int(8*2*PI*Rp/500*1.2)); geo=Geometry.gielis(Rp,nn,m=0,z0=Rp+20)
    e1,e2=pecb(k0,2,geo.f,geo.g,geo.df,geo.dg)
    row=[]
    for cap in [None, 12., 6.]:
      for oh in [None, 6., 10.]:
        m1,m2,M,Ga=reflected_blocks(k0,1.0,'pec',2,geo.f,geo.g,geo.df,geo.dg,0.0,dD_cap=cap,osc_hump=oh)
        err=max(np.max(abs(m1-e1))/np.max(abs(e1)), np.max(abs(m2-e2))/np.max(abs(e2)))
        row.append(f"cap{cap}/n{oh}:M{M} {err:.0e}")
    print(f"lam={lam:.0f} R={Rp} kD={abs(k0)*2*Rp:.0f} | "+" ".join(row))
