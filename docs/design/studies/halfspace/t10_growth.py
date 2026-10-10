"""Digits lost in the GEMM vs delta*D (delta = hump depth, D = x-span), and centring."""
import numpy as np, warnings
from hs import *
import gemm
from gemm import reflected_blocks
import sys; sys.path.insert(0, "/Users/claudiosc/Documents/sie/src")
from pysie2d import Geometry
warnings.simplefilter("ignore")
def pecb(k1,pol,f,g,df,dg):
    nn=len(f); hs_=2*PI/nn; X=f[:,None]-f[None,:]; Z=g[:,None]+g[None,:]
    G,Gx,Gz=pec_exact(X,Z,k1,pol); return hs_*(Gx*dg[None,:]-Gz*df[None,:]), hs_*G
for lam in [500*(1+.05j), 500*(1+1/6j*-1)]:
  k0=2*PI/lam
  for Rp in [250,1000,2000,4000]:
    nn=max(128,int(8*2*PI*Rp/500*1.2)); geo=Geometry.gielis(Rp,nn,m=0,x0=0.0,z0=Rp+20)
    e1,e2=pecb(k0,2,geo.f,geo.g,geo.df,geo.dg)
    row=[]
    for xc_off in [0.0, 5e4, 2e5]:
        m1,m2,M,Ga=reflected_blocks(k0,1.0,'pec',2,geo.f,geo.g,geo.df,geo.dg,0.0,x_c=xc_off)
        err=max(np.max(abs(m1-e1))/np.max(abs(e1)), np.max(abs(m2-e2))/np.max(abs(e2)))
        gr=np.max(Ga/np.maximum(np.abs(e2/(2*PI/nn)),1e-300))
        row.append(f"origin{-xc_off:+.0e}: err {err:.0e} growth(max entrywise sum|AcB|/|G|) {gr:.0e}")
    d=max(0.4*k0.real,3*max(-k0.imag,0)); print(f"lam={lam:.0f} R={Rp} nn={nn} M={M} delta*D={d*2*Rp:.1f} | "+" | ".join(row))
