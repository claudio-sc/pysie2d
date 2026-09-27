import numpy as np, warnings
from hs import *
from refquad import g_ref
warnings.simplefilter("ignore")
NH, OSC, P = 48, 2, 24

def path(k0, eps1, eps2, pol, zmin, xmax, **kw):
    return make_path(k0, eps1, eps2, pol, zmin, xmax, kw.pop('nh', NH), osc=kw.pop('osc', OSC), p_tail=P, **kw)

cases = [  # (name, lam, eps1, eps2)
    ("glass", 500., 1.0, 2.25), ("glass", 1600., 1.0, 2.25),
    ("Si(lossless)", 1600., 1.0, 12.25), ("water/glass", 800., 1.33**2, 2.25),
    ("lossy", 800., 1.0, 2.25 + 1.0j),
    ("Ag", 633., 1.0, -18.3 + 0.48j), ("Au", 800., 1.0, -24.1 + 1.5j),
    ("Ag", 900., 1.0, -38.0 + 0.5j),   # illustrative, small loss
]
geoms = [(100, 5), (500, 20)]
print("(b) eps2 = eps1 ->", np.max(np.abs(g_ind(np.array([0., 300.]), np.array([10., 50.]), 2*PI/500*(1+.03j), 1.0, 1.0+0j, 1,
                                   *path(2*PI/500, 1.0, 1.0+0j, 1, 10, 300)[:2]))))
print("\n(c) deformed path vs real-axis quad (independent), real k; max rel err over pts, per quantity G, dx', dz'")
for name, lam, e1, e2 in cases:
    k0 = 2 * PI / lam
    for pol in (2, 1):
        for Rp, h in geoms:
            pts = [(0, 2*h), (2*Rp, 2*h), (1.4*Rp, 2*h+Rp), (0, 2*h+4*Rp)]
            q, wq, info = path(k0, e1, e2, pol, 2*h, 2*Rp)
            errs = []
            for w in range(3):
                num = np.array([g_ind(X, Z, k0, e1, e2, pol, q, wq)[w] for X, Z in pts])
                ref = np.array([g_ref(X, Z, k0, e1, e2, pol, which=w) for X, Z in pts])
                errs.append(np.max(np.abs(num - ref)) / np.max(np.abs(ref)))
            print(f"{name:13s} {lam:5.0f} pol={pol} R={Rp:3d} h={h:2d} M={2*len(q):5d}  " + " ".join(f"{e:.1e}" for e in errs))
