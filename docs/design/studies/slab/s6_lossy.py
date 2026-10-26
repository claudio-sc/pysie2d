"""S6: lossy layers and complex k — poles off the axis, continued from the lossless roots.

(a) Pole continuation (homotopy + Newton on D): Im β > 0 at real k with loss (decaying
    modes), Im β < 0 at Im k < 0 (the QNM side, holomorphy §5: q ≈ β − i n_g|κ|).
(b) Lossy, real k: path vs the real-axis reference, poles now off-axis (plain QUADPACK
    with breakpoints at Re β; no PV, no residue).
(c) Complex k (Q = 10, 3): no external anchor here; path independence (δ × 0.5, × 1.5)
    and node convergence (s = 2), as v0.9 G3/G9.
"""
import numpy as np

import sproto as S

nc = 1.444
k0 = 2 * np.pi * nc / 1550
esi = (3.476 / nc) ** 2
base = S.Stack([esi], [220.0], 1.0)
pts = [(0.0, 20.0), (300.0, 20.0), (1500.0, 100.0)]

print("(a) pole continuation")
for pol in (2, 1):
    b0 = S.guided_poles_gamma(base, k0, pol)
    for lab, eps_core, kk in (("Im n = 1e-3", (3.476 + 1e-3j) ** 2 / nc**2, k0),
                              ("Im n = 5e-2", (3.476 + 5e-2j) ** 2 / nc**2, k0),
                              ("k(1 − 0.05i), Q=10", esi, k0 * (1 - 0.05j)),
                              ("k(1 − i/6), Q=3", esi, k0 * (1 - 1j / 6))):
        st = S.Stack([eps_core], [220.0], 1.0)
        poles, worst = S.continue_poles(base, st, k0, kk, pol, b0)
        print(f"  {'TE' if pol == 2 else 'TM'} {lab:20s}: β/k₀ = {[f'{p / k0:.6f}' for p in poles]}  "
              f"Newton last step {worst:.0e}  |D(β)|/|D'β| = "
              f"{max(abs(st.D(np.array([p]), kk, pol)[0] / (S.dD(st, kk, pol, p) * p)) for p in poles):.0e}")

print("\n(b) lossy core at real k: path vs real-axis reference (D = 1500, z_min = 20)")
for pol in (2, 1):
    b0 = S.guided_poles_gamma(base, k0, pol)
    for im in (1e-3, 5e-2):
        st = S.Stack([(3.476 + im * 1j) ** 2 / nc**2], [220.0], 1.0)
        poles, _ = S.continue_poles(base, st, k0, k0, pol, b0)
        feats = S.stack_features(st, k0, pol, poles)
        p1 = S.build_path(k0, feats, 1500.0, 20.0)
        ph = S.build_path(k0, feats, 1500.0, 20.0, depth_scale=0.5)
        e_ref = e_d = 0.0
        for X, Z in pts:
            ref = S.g_ref(st, k0, pol, X, Z, [], [], extra_pts=[p.real for p in poles])
            g = S.g_path(p1, st, k0, pol, X, Z)
            e_ref = max(e_ref, abs(g - ref) / abs(ref))
            e_d = max(e_d, abs(S.g_path(ph, st, k0, pol, X, Z) - g) / abs(g))
        print(f"  {'TE' if pol == 2 else 'TM'} Im n = {im:.0e}: Im β/k₀ = {poles[0].imag / k0:.2e}  "
              f"path−ref = {e_ref:.0e}  δ/2−δ = {e_d:.0e}")

print("\n(c) complex k, lossless SOI: path independence and convergence")
for pol in (2, 1):
    b0 = S.guided_poles_gamma(base, k0, pol)
    for lab, kk in (("Q = 10", k0 * (1 - 0.05j)), ("Q = 3", k0 * (1 - 1j / 6))):
        poles, _ = S.continue_poles(base, base, k0, kk, pol, b0)
        feats = S.stack_features(base, kk, pol, poles)
        p1 = S.build_path(kk, feats, 1500.0, 20.0)
        p2 = S.build_path(kk, feats, 1500.0, 20.0, nodes_scale=2.0)
        pa = S.build_path(kk, feats, 1500.0, 20.0, depth_scale=1.5)
        pb = S.build_path(kk, feats, 1500.0, 20.0, depth_scale=0.7)
        e = [0.0, 0.0, 0.0]
        for X, Z in pts:
            g = S.g_path(p1, base, kk, pol, X, Z)
            for i, p in enumerate((p2, pa, pb)):
                e[i] = max(e[i], abs(S.g_path(p, base, kk, pol, X, Z) - g) / abs(g))
        print(f"  {'TE' if pol == 2 else 'TM'} {lab}: Im β/k = {poles[0].imag / abs(kk):+.3f}, δ/k = {p1['delta'] / abs(kk):.3f}, "
              f"min clearance/δ = {min(p1['clear'].values()):.2f}  s2−s1 = {e[0]:.0e}  δ×1.5 = {e[1]:.0e}  δ×0.7 = {e[2]:.0e}")
