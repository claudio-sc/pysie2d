"""S5: small gaps with guided poles present (roadmap risk §7.2).

The evanescent coupling is strongest at small gaps, where the Sommerfeld tail is
longest (38/z_min). Checks: (a) G on the path vs the real-axis reference at
Z = 2·gap down to 4 nm, TE and TM; (b) M against the v0.9 half-space path at the
same D and z_min, which is the cost the stack adds to the GEMM; (c) the one
per-path cost that is new, evaluating R(q) on every node, against N_layers.
"""
import time

import numpy as np

import sproto as S
from pysie2d import layered as L


def case(lam, n_clad, layers, n_bot):
    k = 2 * np.pi * n_clad / lam
    st = S.Stack([n**2 / n_clad**2 for n, _ in layers], [d for _, d in layers], n_bot**2 / n_clad**2)
    return k, st


cases = {
    "SOI 220": (1550, 1.444, [(3.476, 220)], 1.444),
    "SiN 400 @800": (800, 1.444, [(2.0, 400)], 1.444),
    "Si/SiO2/Si": (1550, 1.444, [(3.476, 220), (1.444, 300), (3.476, 220)], 1.444),
}
D = 300.0  # a 100 nm-radius particle plus margin: the box the path serves

print("(a) path vs reference at small Z; (b) M vs the v0.9 half-space path (glass, same D, z_min)")
for gap in (10.0, 5.0, 2.0):
    z_min = 2 * gap
    for name, (lam, nc, layers, nb) in cases.items():
        k, st = case(lam, nc, layers, nb)
        for pol in (2, 1):
            poles = S.guided_poles_gamma(st, k, pol)
            res = [S.residue_R(st, k, pol, b) for b in poles]
            feats = S.stack_features(st, k, pol, poles)
            p1 = S.build_path(k, feats, D, z_min)
            p2 = S.build_path(k, feats, D, z_min, nodes_scale=2.0)
            ph = S.build_path(k, feats, D, z_min, depth_scale=0.5)
            m_hs = L.SommerfeldPath.for_wavenumber(k, 2.25 / nc**2, pol, D, z_min).q.size
            with np.errstate(all="ignore"):
                naive_ok = bool(np.all(np.isfinite(st.R_naive(p1["q"], k, pol))))
            e_ref = e_s = e_d = 0.0
            for X in (0.0, 50.0, 300.0):
                for Z in (z_min, 4 * z_min):
                    for which in (0, 2):
                        ref = S.g_ref(st, k, pol, X, Z, poles, res, which=which)
                        g1 = S.g_path(p1, st, k, pol, X, Z, which)
                        sc = abs(ref)
                        e_ref = max(e_ref, abs(g1 - ref) / sc)
                        e_s = max(e_s, abs(S.g_path(p2, st, k, pol, X, Z, which) - g1) / sc)
                        e_d = max(e_d, abs(S.g_path(ph, st, k, pol, X, Z, which) - g1) / sc)
            print(f"  gap={gap:4.1f} {name:13s} {'TE' if pol == 2 else 'TM'}: M={p1['q'].size:5d} "
                  f"(v0.9 glass {m_hs:5d})  naive R finite on path: {str(naive_ok):5s}  "
                  f"path−ref={e_ref:.0e}  s2−s1={e_s:.0e}  δ/2−δ={e_d:.0e}")

print("\n(c) cost of R(q) on every path node vs N_layers (gap 2 nm, SOI-like layers)")
k = 2 * np.pi * 1.444 / 1550
for n_lay in (0, 1, 3, 9, 21):
    eps = [(3.476 / 1.444) ** 2 if j % 2 == 0 else 1.0 for j in range(n_lay)]
    st = S.Stack(eps, [100.0] * n_lay, 1.0 if n_lay else 2.25 / 1.444**2)
    poles = S.guided_poles_gamma(st, k, 2) if n_lay else []
    p = S.build_path(k, S.stack_features(st, k, 2, poles), D, 4.0)
    t0 = time.perf_counter()
    for _ in range(20):
        st.R(p["q"], k, 2)
    t_r = (time.perf_counter() - t0) / 20
    # one reflected block pair as a dense separable product, nn = 256 per side (no banding)
    nn = 256
    A = np.exp(1j * np.outer(np.linspace(-150, 150, nn), p["q"]))
    t0 = time.perf_counter()
    for _ in range(5):
        A @ (p["w"][:, None] * A.T)
    t_g = (time.perf_counter() - t0) / 5
    print(f"  N={n_lay:2d}: {len(poles)} TE poles, M={p['q'].size}, R on path {t_r * 1e3:6.2f} ms, "
          f"unbanded GEMM nn=256 {t_g * 1e3:6.1f} ms  (R/GEMM = {t_r / t_g:.1e})")
