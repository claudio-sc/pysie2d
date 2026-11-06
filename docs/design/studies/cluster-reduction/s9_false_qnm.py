import numpy as np, warnings
from scipy.special import jv,jvp,hankel1,h1vp
from scipy.optimize import newton
from pysie2d import QNMSolver,Geometry,Material,BIESolver
from pysie2d.fields import eval_field
warnings.simplefilter("ignore")
a=100.0; m=1.5; g=Geometry.gielis(rad=a,n_pts=64,m=0); zero=complex(newton(lambda z:hankel1(2,z),0.429485-1.281374j,tol=1e-14)); ls=2*np.pi*m*a/zero
print("lam_s",ls,"Q",ls.real/(2*ls.imag))
z=2*np.pi*a/ls
rows=[]
for n in range(0,13):
    A=m*jvp(n,m*z)*hankel1(n,z); B=jv(n,m*z)*h1vp(n,z); rows.append((abs(A-B)/(abs(A)+abs(B)),n))
print("TE Mie denominator, relative |A-B|/(|A|+|B|), smallest over n=0..12:",sorted(rows)[:3])
q=QNMSolver(g,Material(n_core=m,pol=2)); r=q.modes(z_lo=ls-20-20j,z_hi=ls+20+20j,n_quad_per_side=32,n_probe=16)
print("reported",r.wavelengths)
M=BIESolver(g,Material(n_core=m,pol=2)).assemble(r.wavelengths[0]); u,s,vh=np.linalg.svd(M); v=vh[-1].conj()
print("sv ratio",s[-1]/s[0],"left null interior frac",np.linalg.norm(u[64:,-1]))
k=2*np.pi/r.wavelengths[0]
X=np.array([300.,0.,-300.,212.]);Z=np.array([0.,300.,0.,212.])
print("radiated exterior field of null vector |F|",np.abs(eval_field(v,64,g.f,g.df,g.g,g.dg,g.delt,k,X,Z)), "|phi|,|chi| max",np.abs(v[:64]).max(),np.abs(v[64:]).max())
# a true mode for contrast: nearby physical pole, by Mie root search at same n
for n in (2,):
    D=lambda l:(lambda z_: m*jvp(n,m*z_)*hankel1(n,z_)-jv(n,m*z_)*h1vp(n,z_))(2*np.pi*a/l)
    try:
        t=complex(newton(D,400+100j,tol=1e-12,maxiter=200)); print("true TE n=2 Mie pole near",t)
    except Exception as e: print("no root",e)
