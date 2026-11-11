import sys, warnings, numpy as np
sys.path.insert(0,"tests")
import test_cluster_eigen_eps as T
from pysie2d.beyn import beyn_modes
from pysie2d import *
from pysie2d.reference import two_cylinder
from scipy.optimize import newton
from scipy.special import hankel1, jn_zeros
warnings.simplefilter("ignore")
d=T._distance; R=T._rel
print("-- high Q")
x=2*np.pi*1000.0/T.LAM_M; n=24
iso=complex(newton(lambda e:T._mie_denominator(e,n,x,2)/hankel1(n,x),complex((jn_zeros(n,1)[0]/x)**2),tol=1e-15)); w=8*abs(iso.imag); lo,hi=iso-w-1j*w,complex(iso.real+w,iso.imag/2)
nn=int(np.ceil(12*np.sqrt(iso.real)*x)); p=T._prepared([Geometry.gielis(rad=1000.0,n_pts=nn,m=0,x0=s*3000.0) for s in (-1,1)],2)
a=beyn_modes(p._reduced_at,lo,hi,n_quad_per_side=24,n_probe=10); b=beyn_modes(T._dense_operator(p),lo,hi,n_quad_per_side=24,n_probe=10)
print(f" iso {iso:.6e} Q {iso.real/(2*abs(iso.imag)):.1e} counts {a.eigenvalues.size},{b.eigenvalues.size} dist {d(a.eigenvalues,b.eigenvalues):.1e}")
print("-- driven figures")
cl=Cluster(T._mixed())
for pol in (1,2):
  for nm,bg in T.BACKGROUNDS.items():
    m=T._mixed_materials(pol); s=ClusterBIESolver(cl,m,pol,bg); dn=s.scatter(T.LAM,13.0); f=s.at_wavelength(T.LAM).scatter(m,13.0); A,_=f.far_field(181); B,_=dn.far_field(181)
    print(f" pol {pol} {nm}: x {R(f.ei,dn.ei):.1e} ff {np.max(np.abs(A-B))/np.max(np.abs(B)):.1e}")
m=T._mixed_materials(1); s=ClusterBIESolver(cl,m,1,T.BACKGROUNDS["slab"]); pr=s.at_wavelength(T.LAM)
print(f" dipole {R(pr.scatter_dipole(m,100.,500.).ei,s.scatter_dipole(T.LAM,100.,500.).ei):.1e} in-plane {R(pr.scatter_dipole(m,100.,500.,moment=(1.,.5j)).ei,s.scatter_dipole(T.LAM,100.,500.,moment=(1.,.5j)).ei):.1e} mode {R(pr.scatter_mode(m,0,1,x_ref=0.).ei,s.scatter_mode(T.LAM,0,1,x_ref=0.).ei):.1e}")
m2=T._mixed_materials(2); s2=ClusterBIESolver(cl,m2,2); w=1550.0+40.0j; print(f" complex wavelength {R(s2.at_wavelength(w).scatter(m2).ei,s2.scatter(w).ei):.1e}")
for pol in (1,2):
    g=Geometry.gielis(rad=90.0,n_pts=64,m=3,n1=4.0,n2=6.0,n3=6.0); mat=Material(n_core=2.0,pol=pol)
    print(f" one particle pol {pol}: {R(ClusterBIESolver(Cluster([g]),[mat],pol).at_wavelength(600.0).scatter([mat]).ei,BIESolver(g,mat).scatter(600.0).ei):.1e}")
radii,core=(180.,110.),(2.,1.6); ctr=np.array([[-450.,0.],[520.,0.]])
for pol in (1,2):
    gs=[Geometry.gielis(r,60,m=0,x0=c[0]) for r,c in zip(radii,ctr)]; mm=[Material(n,1.0,pol=pol) for n in core]
    res=ClusterBIESolver(Cluster(gs),mm,pol).at_wavelength(633.0).scatter(mm,37.0); A,th=res.far_field(361); r=two_cylinder.scattering_amplitude(pol,radii,ctr,[q.nc for q in mm],res.wnum_bg,37.0,th)
    print(f" Graf far field pol {pol}: {np.max(np.abs(A-r))/np.max(np.abs(r)):.1e}")
print("-- glass half-space: find a mode near the real axis, then the pole fit")
geom=T._rotated_triangle(); bg=T.BACKGROUNDS["glass"]; p=T._prepared([geom],1,T.LAM,T.NC,bg)
for box in ((-0.99-0.1j,-0.8-0.002j),(-6.0-1.0j,-1.15-0.004j)):
    try:
        r=p.eigenpermittivities(box[0]*T.NC**2,box[1]*T.NC**2,n_probe=16); print(" box",box,"modes rel",np.round(r.eps/T.NC**2,6),"rank",r.rank,"conv",r.converged)
    except ValueError as ex: print(" box",box,str(ex)[:70]); continue
    if not r.eps.size: continue
    tg=r.eps[np.argmin(np.abs(r.eps.imag))]/T.NC**2
    def f(e):
        return BIESolver(geom,Material.from_eps(e*T.NC**2,n_clad=T.NC,pol=1),background=bg).scatter(T.LAM,13.0).eval_field(np.array([400.0]),np.array([500.0]))[0]
    for im,half,npts,deg in ((1e-4,0.03,21,4),(1e-4,0.06,31,5),(1e-3,0.03,15,3),(1e-4,0.015,21,4)):
        es=tg.real+np.linspace(-half,half,npts)+1j*im; fs=np.array([f(e) for e in es]); t=(es-tg.real)/half
        pole=np.linalg.lstsq(np.column_stack([fs]+[t**k for k in range(deg+1)]),es*fs,rcond=None)[0][0]
        print(f"   target {tg:.6f} Im {im:.0e} ±{half} n{npts} deg{deg}: |p-t|={abs(pole-tg):.1e}",flush=True)
print("-- does QNMSolver report the H_n^(1) zero as a mode? n_core=1.5, circle a=100, zero of H_2 at z=0.429485-1.281374j")
geom=Geometry.gielis(rad=100.0,n_pts=64,m=0); n_core=1.5
zero=complex(newton(lambda z:hankel1(2,z),0.429485-1.281374j,tol=1e-14))
lam_s=2*np.pi*n_core*100.0/zero
print(" lam_s",lam_s,"Q",lam_s.real/(2*lam_s.imag))
q=QNMSolver(geom,Material(n_core=n_core,pol=2))
for w in (20.0,60.0):
    lo,hi=lam_s-w-1j*w,lam_s+w+1j*w
    try:
        r=q.modes(z_lo=lo,z_hi=hi,n_quad_per_side=32,n_probe=16)
        print(f" box ±{w}: found {np.round(r.wavelengths,4)}; nearest to lam_s {np.min(np.abs(r.wavelengths-lam_s)) if r.wavelengths.size else np.nan:.1e}")
    except Exception as ex: print(f" box ±{w}: {type(ex).__name__}: {str(ex)[:90]}")
from pysie2d.reference import mie
print(" Mie root check: is there a true resonance near lam_s? |D|, for TE n=2, relative scale")
for n_ in (2,):
    z=2*np.pi*100.0/lam_s; m=n_core
    from scipy.special import jv,jvp,h1vp
    Dv=m*jvp(n_,m*z)*hankel1(n_,z)-jv(n_,m*z)*h1vp(n_,z); print(f"  D_2(lam_s)={abs(Dv):.2e}, scale {abs(m*jvp(n_,m*z)*hankel1(n_,z)):.2e}")
