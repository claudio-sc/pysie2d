import numpy as np, sys
sys.path.insert(0, ".")
from epsbeyn import solve_box
import test_expand as T          # reuses op, b, s_tm0, exact sweep (re-run is fine)
import os
HERE = os.path.dirname(os.path.abspath(__file__))
op, b, s_tm0, eps_sweep, exact = T.op, T.b, T.s_tm0, T.eps_sweep, T.exact
boxes = [(-12.0-3.0j, -1.15-0.004j), (-0.95-3.0j, -0.40-0.0005j), (-0.40-3.0j, -0.02-0.004j)]
modes = []
for lo, hi in boxes:
    eigs, bm, _ = solve_box(op, lo, hi, nq=48, n_probe=24)
    print(f"box {lo}..{hi}: rank={bm.rank} ->", np.round(np.sort_complex(eigs), 5))
    modes += list(eigs)
modes = np.array(modes)
amps = []
for e in modes:
    u, s, vh = np.linalg.svd(op.M(e)); uu, vv = u[:, -1], vh[-1].conj()
    amps.append(s_tm0(vv, e) * (uu.conj() @ b) / (uu.conj() @ op.dM(e) @ vv))
amps = np.array(amps)
poles = np.array([np.sum(amps/(e - modes)) for e in eps_sweep])
re = eps_sweep.real
regions = {"far side  [-3.0,-1.3]": (re <= -1.3), "near -1   (-1.3,-0.7)": (re > -1.3) & (re < -0.7),
           "near side [-0.7,-0.2]": (re >= -0.7)}
scale = np.max(np.abs(exact))
for name, msk in regions.items():
    e_pole = np.max(np.abs(poles[msk]-exact[msk]))/scale
    # remainder = smooth on the region: quadratic through 3 exact solves inside it
    idx = np.flatnonzero(msk)[[0, msk.sum()//2, -1]]
    c = np.linalg.solve(np.vander(eps_sweep[idx], 3, increasing=True), exact[idx]-poles[idx])
    fit = poles[msk] + np.vander(eps_sweep[msk], 3, increasing=True) @ c
    print(f"{name}: pole sum ({modes.size} modes) err={e_pole:.1e};  + quadratic remainder (3 solves) err="
          f"{np.max(np.abs(fit-exact[msk]))/scale:.1e}")
np.savez(os.path.join(HERE, "expand2.npz"), eps=eps_sweep, exact=exact, poles=poles, modes=modes, amps=amps)
