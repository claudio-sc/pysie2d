import os, time, numpy as np, warnings
from hs import *
from gemm import reflected_blocks
import sys; sys.path.insert(0, "/Users/claudiosc/Documents/sie/src")
from pysie2d import Geometry
from pysie2d.kernels import assemble_matrix
from pysie2d.material import Material
warnings.simplefilter("ignore")
def tmin(fn, rep=5):
    ts=[]
    for _ in range(rep):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return 1e3*min(ts)
# accuracy of GEMM (dielectric/metal) vs direct pairwise g_ind on an s=3 path
print("GEMM vs direct pairwise (nn=64): max|dM|/max|M|")
for name, lam, e2, pol in [("glass",500.,2.25,1),("Ag",633*(1+.05j),-18.3+.48j,1),("Si",1600.,12.25,2)]:
    k0=2*PI/lam
    for Rp,gap in [(100,5),(500,20)]:
        geo=Geometry.gielis(Rp,64,m=0,z0=Rp+gap)
        f,g,df,dg=geo.f,geo.g,geo.df,geo.dg
        X=f[:,None]-f[None,:]; Z=g[:,None]+g[None,:]
        q,wq,_=make_path(k0,1.0,e2,pol,2*gap,2*Rp,n_hump=400,osc=0.7,L_decay=45.)
        G,Gx,Gz=g_ind(X,Z,k0,1.0,e2,pol,q,wq)
        hst=2*PI/64; r1=hst*(Gx*dg[None,:]-Gz*df[None,:]); r2=hst*G
        for sub,L in [(False,38.),(True,20.)]:
            m1,m2,M,_=reflected_blocks(k0,1.0,e2,pol,f,g,df,dg,0.0,L_decay=L,subtract=sub)
            print(f"  {name:5s} pol{pol} R{Rp} gap{gap} sub={sub!s:5s} L={L:.0f} M={M:5d}: M1 {np.max(abs(m1-r1))/np.max(abs(r1)):.0e}  M2 {np.max(abs(m2-r2))/np.max(abs(r2)):.0e}")
print("\nTiming (ms, best of 5, VECLIB_MAXIMUM_THREADS=%s): assemble_matrix vs reflected blocks (banded GEMM)" % os.environ.get("VECLIB_MAXIMUM_THREADS"))
mat=Material(n_core=2.0, n_clad=1.0, epsi=0.0, pol=1)
for lam in [633., 633*(1+.05j)]:
    k0=2*PI/lam
    for Rp,gap in [(100,5),(500,5),(500,20)]:
        for nn in [64,128,256]:
            geo=Geometry.gielis(Rp,nn,m=0,z0=Rp+gap)
            a=tmin(lambda: assemble_matrix(1,nn,geo.f,geo.g,geo.df,geo.dg,geo.ddf,geo.ddg,mat.wnum_bg(lam),mat.nc,mat.eps))
            res=[]
            for band,sub,L in [(False,False,38.),(True,False,38.),(True,True,20.)]:
                out={}
                def run(): out['M']=reflected_blocks(k0,1.0,-18.3+.48j,1,geo.f,geo.g,geo.df,geo.dg,0.0,banded=band,L_decay=L,subtract=sub)[2]
                t=tmin(run,3); res.append(f"{'band' if band else 'full'}{'+sub' if sub else ''} M={out['M']:5d} {t:7.1f}")
            print(f"lam={lam:.0f} R{Rp} gap{gap} nn={nn:3d} | assemble {a:6.1f} | "+" | ".join(res))
