"""S3 (anchor A3): reflected G on the feature-aware deformed path vs the real-axis reference.

The reference differs from the path in every ingredient that could hide a
shared error: the transfer-matrix R (not the recursion), the radiation-condition
branch rule (not the vertical-cut sheet), QUADPACK on the real axis with θ/cosh
substitutions (not Gauss–Legendre on a hump), and the guided poles handled by a
Cauchy-weight principal value plus iπ·Res for the path passing below +β.
"""
import numpy as np

import sproto as S


def case(lam, n_clad, layers, n_bot):
    k = 2 * np.pi * n_clad / lam
    st = S.Stack([n**2 / n_clad**2 for n, _ in layers], [d for _, d in layers], n_bot**2 / n_clad**2)
    return k, st


cases = {
    "SOI 220 @1550": (1550, 1.444, [(3.476, 220)], 1.444),
    "air|Si 220|SiO2": (1550, 1.0, [(3.476, 220)], 1.444),
    "SiN 400 @800 (2 modes)": (800, 1.444, [(2.0, 400)], 1.444),
    "Si 2000 (9 modes)": (1550, 1.444, [(3.476, 2000)], 1.444),
    "Si/SiO2/Si coupled": (1550, 1.444, [(3.476, 220), (1.444, 300), (3.476, 220)], 1.444),
}
pts = [(0.0, 20.0), (300.0, 20.0), (1500.0, 100.0), (40.0, 400.0)]  # (X, Z) nm
D = max(x for x, _ in pts)
z_min = min(z for _, z in pts)

for name, (lam, nc, layers, nb) in cases.items():
    k, st = case(lam, nc, layers, nb)
    for pol in (2, 1):
        poles = S.guided_poles(st, k, pol)
        res = [S.residue_R(st, k, pol, b) for b in poles]
        feats = S.stack_features(st, k, pol, poles)
        paths = {s: S.build_path(k, feats, D, z_min, nodes_scale=s) for s in (0.5, 1.0, 2.0)}
        deep = S.build_path(k, feats, D, z_min, depth_scale=0.5)
        p1 = paths[1.0]
        worst = min(p1["clear"].values())
        print(f"{name:24s} {'TE' if pol == 2 else 'TM'}: {len(poles)} poles, M = {p1['q'].size}, "
              f"T/k = {p1['T'] / k:.2f}, δ/k = {p1['delta'] / k:.3f}, min clearance/δ = {worst:.2f}")
        for X, Z in pts:
            for which in (0, 2):
                ref = S.g_ref(st, k, pol, X, Z, poles, res, which=which)
                g = {s: S.g_path(p, st, k, pol, X, Z, which) for s, p in paths.items()}
                gd = S.g_path(deep, st, k, pol, X, Z, which)
                sc = abs(ref)
                print(f"    X={X:6.0f} Z={Z:5.0f} {'G ' if which == 0 else '∂z'}: |ref|={sc:.2e}  "
                      f"path−ref: s=.5 {abs(g[0.5] - ref) / sc:.0e}  s=1 {abs(g[1.0] - ref) / sc:.0e}  "
                      f"s=2 {abs(g[2.0] - ref) / sc:.0e}   δ/2 vs δ: {abs(gd - g[1.0]) / sc:.0e}")
