import numpy as np, sys, warnings
sys.path.insert(0, ".")
import test_expand as T
from pysie2d import Material
from pysie2d.solver import ScatterResult
import os
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "expand2.npz")); modes = d["modes"]
print("mode eps_m                   P_TM0(+x)  P_TM0(-x)  P_up     P_down   -P_abs(gain)  balance   beta_wg")
for e in modes:
    u, s, vh = np.linalg.svd(T.op.M(e)); v = vh[-1].conj()
    r = ScatterResult(v, T.geom, Material.from_eps(e*T.N_CLAD**2, n_clad=T.N_CLAD, pol=T.pol), T.lam, 0.0,
                      T.SIN, excitation="plane_wave")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = r.mode_amplitudes(x_ref=0.0)
    pp, pm = float(np.sum(a.p_plus)), float(np.sum(a.p_minus))
    out = pp + pm + a.p_up + a.p_down
    print(f"{e.real:+.5f}{e.imag:+.5f}j   {pp:9.2e}  {pm:9.2e}  {a.p_up:7.2e}  {a.p_down:7.2e}  {-a.p_abs:10.3e}"
          f"  {abs(out + a.p_abs)/abs(a.p_abs):8.1e}  {(pp+pm)/out:6.3f}")
