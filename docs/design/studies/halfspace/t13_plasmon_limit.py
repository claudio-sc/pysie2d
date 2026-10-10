"""Open item 1: eps2 -> -eps1 (q_sp -> inf, R_inf diverges), TM, real k.
Deformed path vs the independent real-axis QUADPACK reference (refquad), at the
production node rule and at s = 2 (twice the nodes), over Re eps2 from a Ag-like
-18 through the resonance -1 into the -eps1 < Re eps2 < 0 band."""
import numpy as np, warnings
from hs import *
from refquad import g_ref
warnings.simplefilter("ignore")
lam, e1, pol = 633.0, 1.0, 1
k0 = 2 * PI / lam
Rp, h = 100, 5
pts = [(0, 2*h), (2*Rp, 2*h), (1.4*Rp, 2*h+Rp), (0, 2*h+4*Rp)]


def err(e2, s):
    q, wq, info = make_path(k0, e1, e2, pol, 2*h, 2*Rp, osc=4.0/s)
    nh = int(round(s * max(48, np.ceil(13*info['T']/info['delta']))))
    q, wq, info = make_path(k0, e1, e2, pol, 2*h, 2*Rp, n_hump=nh, osc=4.0/s)
    e = []
    for w in range(3):
        num = np.array([g_ind(X, Z, k0, e1, e2, pol, q, wq)[w] for X, Z in pts])
        ref = np.array([g_ref(X, Z, k0, e1, e2, pol, which=w) for X, Z in pts])
        e.append(np.max(np.abs(num - ref)) / np.max(np.abs(ref)))
    return 2*len(q), max(e), info


import sys
G_LIST = tuple(float(a) for a in sys.argv[1:]) or (0.5, 0.1)
print("eps2 | q_sp/k1 | |R_inf| | T/k1 | M:err (rule) | M:err (s=2)")
for g in G_LIST:
    for x in (18.3, 2.0, 1.2, 1.1, 1.05, 1.02, 1.01, 1.0, 0.99, 0.98, 0.9, 0.5):
        e2 = -x + 1j*g
        qsp = np.sqrt(e2/(e1+e2))
        M1, E1, info = err(e2, 1.0)
        M2, E2, _ = err(e2, 2.0)
        print(f"{e2.real:+6.2f}{e2.imag:+.3f}i | {qsp.real:6.2f}{qsp.imag:+6.2f}i | {abs(r_inf(e1, e2, pol)):6.1f} | "
              f"{info['T']/k0:5.2f} | {M1:5d}:{E1:.0e} | {M2:5d}:{E2:.0e}")
