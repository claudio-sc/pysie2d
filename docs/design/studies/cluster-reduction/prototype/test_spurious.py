import numpy as np, sys
sys.path.insert(0, ".")
from scipy.special import hankel1
from scipy.optimize import newton
from epsbeyn import EpsOperator, solve_box
from pysie2d import Geometry
rad, lam = 100.0, 600.0
x = 2*np.pi*rad/lam
lo, hi = -2.0 - 1.5j, -0.2 - 0.15j
true = np.array([-1.4795414544-0.3030041479j, -0.4641441151-0.9859744318j])
for nn in (64, 96):
    op = EpsOperator(Geometry.gielis(rad=rad, n_pts=nn, m=0), pol=1, wavelength=lam, branch="principal")
    eigs, bm, t = solve_box(op, lo, hi, nq=48, n_probe=80, refine=False)
    print(f"[principal nn={nn}] rank={bm.rank} found={eigs.size}")
    for e in np.sort_complex(eigs):
        if np.min(np.abs(true - e)) < 1e-8:
            print(f"   {e:.6f}  TRUE mode"); continue
        # test the sound-soft hypothesis: H_n^(1)(sqrt(eps) x) = 0 for some n, principal sqrt
        z = np.sqrt(e)*x
        best = min(range(0, nn//2), key=lambda n: abs(hankel1(n, z))/abs(hankel1(n, abs(z)+0j)))
        zr = newton(lambda zz: hankel1(best, zz), z, tol=1e-13, maxiter=100)
        print(f"   {e:.6f}  spurious? nearest H_{best}^(1) zero at eps={(zr/x)**2:.6f}  "
              f"|diff|={abs((zr/x)**2 - e):.1e}")
