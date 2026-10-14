import warnings
import numpy as np
from scipy.special import hankel1
from pysie2d.layered import reflected_green
from tproto import PI, t_green, t_path, t_ref
warnings.simplefilter("ignore")
n_clad, lam, z_int = 1.33, 633.0, 0.0
k = 2*PI*n_clad/lam
obs = np.array([[0.0,-5.0],[150.0,-40.0],[-300.0,-1.0],[20.0,-800.0],[900.0,-200.0],[0.0,0.0]])
xs, zs = 10.0, 15.0
x, z = obs[:,0], obs[:,1]
D = np.ptp(np.r_[x, xs]); zmin = zs - z_int + (z_int - z).min()
for eps_abs, name in ((2.25,"glass"),(-18.3+0.48j,"silver"),(1.0,"low-n")):
    eps = eps_abs/n_clad**2
    for pol in (2,1):
        g1 = t_green(t_path(k,eps,pol,D,zmin),pol,k,eps,x,z,xs,zs,z_int)[0]
        g2 = t_green(t_path(k,eps,pol,D,zmin,1.5),pol,k,eps,x,z,xs,zs,z_int)[0]
        ref = np.array([t_ref(xi-xs, zs-z_int, z_int-zi, k, eps, pol) for xi,zi in obs])
        sc = np.max(np.abs(g1))
        print(f"{name:7s} pol={pol} path-vs-path(x1.5) {np.max(np.abs(g1-g2))/sc:.1e}  per-point path-vs-ref", np.array2string(np.abs(g1-ref)/sc, precision=1))
# continuity with eta = 1e-12
for eps_abs in (2.25, -18.3+0.48j):
    eps = eps_abs/n_clad**2
    for kk in (k, 2*PI*n_clad/(lam*(1+0.05j))):
        for pol in (2,1):
            xx = np.array([-200.0,0.0,30.0,400.0]); zz = np.zeros_like(xx)
            Dd = np.ptp(np.r_[xx, xs])
            gt = t_green(t_path(kk,eps,pol,Dd,zs),pol,kk,eps,xx,zz,xs,zs,z_int)[0]
            gi = reflected_green(t_path(kk,eps,pol,Dd,zs),pol,kk,eps,xx,zz+1e-12,xs,zs,z_int)[0]
            g0 = 0.25j*hankel1(0,kk*np.hypot(xx-xs, 1e-12-zs))
            print("continuity", eps_abs, pol, f"{np.max(np.abs(g0+gi-gt)/np.abs(gt)):.1e}")
