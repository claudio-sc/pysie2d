import warnings, time, numpy as np
from pysie2d import *
warnings.simplefilter("ignore")
NC=1.444; LAM=1550.0; SIN=Stack(((4.0,400.0),),NC**2)
rel=lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
print("gap ladder, plane wave")
for pitch,nn,bg in ((300.,100,None),(250.,200,SIN),(240.,400,None),(234.,800,None)):
    gs=[Geometry.gielis(rad=90.0,n_pts=nn,m=3,n1=4.0,n2=6.0,n3=6.0,x0=pitch*(p-1.5),z0=140.0) for p in range(4)]
    mats=[Material.from_eps((-2.0+0.1j)*NC**2,n_clad=NC,pol=1)]*4; s=ClusterBIESolver(Cluster(gs),mats,1,bg)
    M=s._assemble(LAM); c=np.linalg.cond(M); r=rel(s.at_wavelength(LAM).scatter(mats).ei,s.scatter(LAM).ei)
    print(f" pitch {pitch} nn {nn}: err {r:.1e}, cond {c:.1e}, cond*u {c*1.1e-16:.1e}")
print("cost per material set, shipped code, nn=100, pitch 600, distinct particles, best of 3")
for name,bg,Ps in (("free",None,(4,8,16,32,64)),("slab",SIN,(4,8,16,32))):
  for P in Ps:
    gs=[Geometry.gielis(rad=90.0+(p%5),n_pts=100,m=3,n1=4.0,n2=6.0,n3=6.0,x0=600.0*(p-(P-1)/2),z0=140.0) for p in range(P)]
    cl=Cluster(gs); mats=[Material.from_eps((-2.0+0.1j)*NC**2,n_clad=NC,pol=1)]*P; s=ClusterBIESolver(cl,mats,1,bg)
    t0=time.perf_counter(); pr=s.at_wavelength(LAM); ts=time.perf_counter()-t0
    def best(f,n=3):
        o=[]
        for _ in range(n):
            t=time.perf_counter(); f(); o.append(time.perf_counter()-t)
        return min(o)
    tf=best(lambda: pr.scatter(mats))
    td=best(lambda: s.scatter(LAM),1 if P>=32 else 2) if not (P>=64) else float('nan')
    mb=(sum(y.size+u.size for y,u in zip(pr._y,pr._u))+sum(t.size for t in pr._top))*16/1e6
    print(f" {name} P={P:3d}: scatter {td:7.2f} s | setup {ts:6.2f} s | per material {tf*1e3:7.1f} ms | ranks R0={sum(pr.coupling_ranks)} | stored {mb:.0f} MB vs dense {16*cl.n_dof**2/1e6:.0f} MB",flush=True)
