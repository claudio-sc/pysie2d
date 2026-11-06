import numpy as np, sys
sys.path.insert(0, ".")
from anchor import root, winding
from epsbeyn import EpsOperator, solve_box
from pysie2d import Geometry

rad, lam = 100.0, 600.0
x = 2*np.pi*rad/lam
lo, hi = -2.0 - 1.5j, -0.2 - 0.15j
# analytic set in the box: count every family n=0..12 by winding number
ref = []
for n in range(0, 13):
    c = winding(n, x, 1, lo, hi)
    if c:
        guess = -1.0 - 0.3j if n else 0.5 - 0.5j
        r = root(guess, n, x, 1)
        ref += [r] * (2 if n else 1)
        print(f"analytic n={n}: count {c}, eps_m={r:.10f}")
ref = np.sort_complex(np.array(ref))
for branch in ("upper", "principal"):
    for nn in (32, 64):
        op = EpsOperator(Geometry.gielis(rad=rad, n_pts=nn, m=0), pol=1, wavelength=lam, branch=branch)
        eigs, bm, t = solve_box(op, lo, hi)
        eigs = np.sort_complex(eigs)
        print(f"\n[{branch:9s} nn={nn}] rank={bm.rank} found={eigs.size} cancel={bm.cancellation:.1e} t={t:.2f}s")
        for e in eigs:
            d = np.min(np.abs(ref - e))
            print(f"   {e.real:+.12f} {e.imag:+.12f}i   |err vs analytic| = {d:.1e}")
