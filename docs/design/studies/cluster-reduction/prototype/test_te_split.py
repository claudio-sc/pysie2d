import numpy as np, sys
sys.path.insert(0, ".")
from epsbeyn import EpsOperator, solve_box
from pysie2d import Geometry
rad, lam = 100.0, 600.0
ref = np.array([14.284784-2.058696j, 1.179109-1.535828j] + [4.865354-1.702231j]*2 + [27.189192-1.575727j]*2 + [12.377726-0.561886j]*2)
op = EpsOperator(Geometry.gielis(rad=rad, n_pts=64, m=0), pol=2, wavelength=lam, branch="upper")
print("single long box, more nodes:")
for nq in (96, 160):
    eigs, bm, t = solve_box(op, 0.5-4.0j, 30.0-0.3j, nq=nq, n_probe=24)
    err = max(np.min(np.abs(ref - e)) for e in eigs)
    print(f"  nq={nq}: rank={bm.rank} found={eigs.size}/8 max|err|={err:.1e}  t={t:.1f}s")
print("three 10-wide boxes, nq=32 each:")
tot, errs, tt = 0, [], 0.0
for a in (0.5, 10.0, 20.0):
    eigs, bm, t = solve_box(op, a-4.0j, a+10.0-0.3j, nq=32, n_probe=12)
    tot += eigs.size; tt += t; errs += [np.min(np.abs(ref - e)) for e in eigs]
print(f"  found={tot}/8 max|err|={max(errs):.1e}  t={tt:.1f}s")
