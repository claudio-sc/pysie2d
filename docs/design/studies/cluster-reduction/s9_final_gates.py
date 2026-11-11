import sys, warnings, numpy as np
sys.path.insert(0,"tests")
import test_cluster_eigen_eps as T
from pysie2d.beyn import beyn_modes
warnings.simplefilter("ignore")
from pysie2d import *
d=T._distance
print("-- B2 Graf")
for pol,dd in ((2,1000.),(1,1000.),(2,400.),(1,400.)):
    lo,hi=T.BOX[pol]; r=T._prepared(T._circles(dd),pol).eigenpermittivities(lo,hi,n_quad_per_side=48)
    out={}
    for n in (10,12):
        g=beyn_modes(lambda e:T._graf(e,pol,dd,n),lo,hi,n_quad_per_side=48,n_probe=24,max_workers=1); out[n]=T._polish(lambda e:T._graf(e,pol,dd,n),g.eigenvalues,lo,hi)
    print(f" pol {pol} d {dd}: N={r.eps.size} |BIE-Graf12| {max(d(r.eps,out[12]),d(out[12],r.eps)):.1e}  Graf 10v12 {d(out[10],out[12]):.1e} conv {r.converged.all()}")
print("-- B1 circle")
from scipy.optimize import newton
for pol in (1,2):
    lo,hi=T.BOX[pol]; r=T._prepared(T._circles(nn=64),pol).eigenpermittivities(lo,hi,n_quad_per_side=48)
    roots=[]
    for e in r.eps:
        for n in range(13):
            pass
    # residual: nearest root over orders by polishing each returned value on the denominator
    err=[]
    for e in r.eps:
        best=np.inf
        for n in range(13):
            try:
                q=complex(newton(lambda z,n=n:T._mie_denominator(z,n,T.X,pol),complex(e),tol=1e-14,maxiter=100)); best=min(best,abs(q-e))
            except RuntimeError: pass
        err.append(best)
    print(f" pol {pol}: N={r.eps.size} max err {max(err):.1e}")
print("-- B3 PEC")
for pol,h in ((2,200.),(2,500.),(1,200.),(1,500.)):
    lo,hi=T.BOX[pol]; c=Geometry.gielis(rad=T.RAD,n_pts=128,m=0,z0=h); im=Geometry.gielis(rad=T.RAD,n_pts=128,m=0,z0=-h)
    a=T._prepared([c],pol,background=HalfSpace.pec()).eigenpermittivities(lo,hi,n_quad_per_side=40,n_probe=20); b=T._prepared([c,im],pol).eigenpermittivities(lo,hi,n_quad_per_side=40,n_probe=20)
    rest=[x for x in b.eps if np.min(np.abs(a.eps-x))>1e-8]; miss=min((np.min(np.abs(a.eps-x)) for x in rest),default=np.nan)
    print(f" pol {pol} h {h}: {a.eps.size} of {b.eps.size}, dist {d(a.eps,b.eps):.1e}; other-parity modes are {miss:.1e} away")
print("-- B5 G vs M, null vector")
for pol in (1,2):
    lo,hi=T.BOX[pol]; p=T._prepared(T._circles(400.),pol); r=p.eigenpermittivities(lo,hi,n_quad_per_side=48); op=T._dense_operator(p)
    dn=beyn_modes(op,lo,hi,n_quad_per_side=48,n_probe=24,max_workers=1); ref=T._polish(op,dn.eigenvalues,lo,hi)
    res=max(np.linalg.norm(op(e)@r.vectors[:,k])/np.linalg.norm(op(e)) for k,e in enumerate(r.eps))
    print(f" pol {pol}: N {r.eps.size}/{ref.size} dist {d(r.eps,ref):.1e} max|Mx|/|M| {res:.1e} conv {r.converged.all()}")
print("-- B8 absolute")
lo,hi=T.BOX[2]; v=T._prepared(T._circles(nn=64),2).eigenpermittivities(lo,hi,n_quad_per_side=48); c=T._prepared(T._circles(nn=64),2,T.LAM_M*T.NC,T.NC).eigenpermittivities(lo*T.NC**2,hi*T.NC**2,n_quad_per_side=48)
print(f" dist {d(c.eps/T.NC**2,v.eps):.1e}")
print("-- B4 slab and glass pole fits")
geom=T._rotated_triangle()
for name,bg,obs in (("n=2 slab",T.BACKGROUNDS["slab"],"s0"),("n=2 slab",T.BACKGROUNDS["slab"],"field")):
    p=T._prepared([geom],1,T.LAM,T.NC,bg); box=((-1.6-0.2j)*T.NC**2,(-1.1-0.002j)*T.NC**2)
    r=p.eigenpermittivities(*box,n_probe=12); tg=r.eps[np.argmin(np.abs(r.eps.imag))]/T.NC**2
    def f(e):
        res=BIESolver(geom,Material.from_eps(e*T.NC**2,n_clad=T.NC,pol=1),background=bg).scatter(T.LAM,13.0 if obs=="field" else 0.0)
        return res.mode_amplitudes(x_ref=0.0).s_plus[0] if obs=="s0" else res.eval_field(np.array([400.0]),np.array([500.0]))[0]
    for im,half,npts,deg in ((1e-4,0.03,21,4),(1e-4,0.06,31,5),(1e-3,0.03,15,3)):
        es=tg.real+np.linspace(-half,half,npts)+1j*im; fs=np.array([f(e) for e in es]); t=(es-tg.real)/half
        pole=np.linalg.lstsq(np.column_stack([fs]+[t**k for k in range(deg+1)]),es*fs,rcond=None)[0][0]
        print(f" {name} {obs} target {tg:.6f} Im {im:.0e} ±{half} n{npts} deg{deg}: |p-t|={abs(pole-tg):.1e}",flush=True)
print("-- high Q")
from scipy.special import hankel1, jn_zeros
x=2*np.pi*1000.0/T.LAM_M; n=24
iso=complex(newton(lambda e:T._mie_denominator(e,n,x,2)/hankel1(n,x),complex((jn_zeros(n,1)[0]/x)**2),tol=1e-15)); w=8*abs(iso.imag); lo,hi=iso-w-1j*w,complex(iso.real+w,iso.imag/2)
nn=int(np.ceil(12*np.sqrt(iso.real)*x)); p=T._prepared([Geometry.gielis(rad=1000.0,n_pts=nn,m=0,x0=s*3000.0) for s in (-1,1)],2)
a=beyn_modes(p._reduced_at,lo,hi,n_quad_per_side=24,n_probe=10); b=beyn_modes(T._dense_operator(p),lo,hi,n_quad_per_side=24,n_probe=10)
print(f" iso {iso:.6e} Q {iso.real/(2*abs(iso.imag)):.1e} counts {a.eigenvalues.size},{b.eigenvalues.size} dist {d(a.eigenvalues,b.eigenvalues):.1e}")
