import numpy as np, sys
sys.path.insert(0, ".")
from scipy.special import hankel1
from scipy.optimize import newton
from epsbeyn import EpsOperator
from pysie2d import Geometry
rad, lam = 100.0, 600.0; x = 2*np.pi*rad/lam
# sound-soft (H^(1)) resonances of the interior wavenumber, principal sqrt: H_n^(1)(sqrt(eps) x) = 0
for n, z0 in ((2, np.sqrt(-1.33-1.0j)*x), (3, 2.0-1.0j), (4, 2.8-1.4j)):
    z = newton(lambda zz: hankel1(n, zz), z0, tol=1e-14, maxiter=200)
    eps_s = (z/x)**2
    print(f"n={n}: H_n zero z={z:.6f}  -> eps_s={eps_s:.6f}")
    for branch in ("principal", "upper"):
        for nn in (64, 128):
            op = EpsOperator(Geometry.gielis(rad=rad, n_pts=nn, m=0), pol=1, wavelength=lam, branch=branch)
            s_at = np.linalg.svd(op.M(eps_s), compute_uv=False)
            s_off = np.linalg.svd(op.M(eps_s + 0.05), compute_uv=False)
            # left null vector of the spurious pole should live in the interior rows only
            u, s, vh = np.linalg.svd(op.M(eps_s)); ul = u[:, -1]
            frac_int = np.linalg.norm(ul[nn:])/np.linalg.norm(ul)
            print(f"   {branch:9s} nn={nn:3d}: smin/smax at eps_s = {s_at[-1]/s_at[0]:.1e}, "
                  f"0.05 away = {s_off[-1]/s_off[0]:.1e}; |u_int|/|u| = {frac_int:.3f}")
