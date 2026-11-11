import warnings, numpy as np
from pysie2d import *
from pysie2d.beyn import beyn_modes
import pysie2d.cluster as C
warnings.simplefilter("ignore")
NC=1.444; LAM=1550.0; SLAB=Stack(((4.0,400.0),),NC**2)
gs=[Geometry.gielis(rad=90.0,n_pts=100,m=3,n1=4.0,n2=6.0,n3=6.0,x0=600.0*(p-1.5),z0=140.0) for p in range(4)]
cl=Cluster(gs); s0=ClusterBIESolver(cl,[Material(n_core=1.5,n_clad=NC,pol=1)]*4,1,SLAB); pr=s0.at_wavelength(LAM)
iso=lambda e: np.vstack([pr._top[0],pr._interior(0,np.sqrt(e),e)])
bm=beyn_modes(iso,-1.6-0.2j,-1.1-0.002j,n_quad_per_side=32,n_probe=12); pole=bm.eigenvalues[np.argmin(np.abs(bm.eigenvalues.imag))]
print("pole",pole, "sigma",np.linalg.svd(iso(pole),compute_uv=False)[-1])
rng=np.random.default_rng(0); brand=np.zeros(cl.n_dof,complex)
for p,g in enumerate(gs):
    sl=cl.slice(p); brand[sl.start:sl.start+g.n_pts]=rng.standard_normal(g.n_pts)+1j*rng.standard_normal(g.n_pts)
for name,getb in (("plane",lambda: s0._rhs_plane_wave(LAM,0.0)),("random",lambda: brand)):
  for d in (0.0,1e-12,1e-10,1e-6):
    e=pole+d*np.exp(0.7j); mats=[Material.from_eps(e*NC**2,n_clad=NC,pol=1)]*4; b=getb()
    print(f"{name} d={d:.0e}: mat.eps-e={abs(mats[0].eps-e):.1e} nc-principal={abs(mats[0].nc-np.sqrt(e)):.1e} sigmamin(M_p)/max={(lambda s:s[-1]/s[0])(np.linalg.svd(iso(mats[0].eps),compute_uv=False)):.1e}")
    M=ClusterBIESolver(cl,mats,1,SLAB)._assemble(LAM); xd=np.linalg.solve(M,b)
    fac=pr._factor([(m.nc,m.eps) for m in mats]); x,a=pr._apply(fac,b)
    be=lambda x:np.linalg.norm(M@x-b)/(np.linalg.norm(M)*np.linalg.norm(x))
    print(f"   alpha={a:.1e} err={np.linalg.norm(x-xd)/np.linalg.norm(xd):.1e} be={be(x):.1e} residual_diff={np.linalg.norm((b-M@x)-pr._residual(fac,b,x))/np.linalg.norm(b):.1e} cond(G)={np.linalg.cond(fac.reduced):.1e}")
    for it in range(4):
        dx,_=pr._apply(fac,pr._residual(fac,b,x)); x=x+dx
        print(f"   step {it+1}: |dx|/|x|={np.linalg.norm(dx)/np.linalg.norm(x):.1e} err={np.linalg.norm(x-xd)/np.linalg.norm(xd):.1e} be={be(x):.1e} |r|/|b|={np.linalg.norm(b-M@x)/np.linalg.norm(b):.1e}")
print("=== gap floor (free, plane wave)")
for pitch,nn in ((300.,100),(250.,200),(240.,400),(234.,800)):
    gs=[Geometry.gielis(rad=90.0,n_pts=nn,m=3,n1=4.0,n2=6.0,n3=6.0,x0=pitch*(p-1.5),z0=140.0) for p in range(4)]
    mats=[Material.from_eps((-2.0+0.1j)*NC**2,n_clad=NC,pol=1)]*4; s=ClusterBIESolver(Cluster(gs),mats,1); M=s._assemble(LAM); b=s._rhs_plane_wave(LAM,0.0)
    xd=np.linalg.solve(M,b); pr=s.at_wavelength(LAM); fac=pr._factor([(m.nc,m.eps) for m in mats]); x,a=pr._apply(fac,b)
    be=lambda x:np.linalg.norm(M@x-b)/(np.linalg.norm(M)*np.linalg.norm(x)); c=np.linalg.cond(M)
    rngb=np.random.default_rng(0).standard_normal(M.shape[0])+0j; xrd=np.linalg.solve(M,rngb); xr,_=pr._apply(fac,rngb)
    print(f"pitch {pitch} nn {nn}: cond(M)={c:.1e} u*cond={c*1.1e-16:.1e} plane err={np.linalg.norm(x-xd)/np.linalg.norm(xd):.1e} be fast={be(x):.1e} be dense={be(xd):.1e} alpha={a:.1f}; random-rhs err={np.linalg.norm(xr-xrd)/np.linalg.norm(xrd):.1e}")
