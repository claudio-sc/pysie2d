"""S4: near-cutoff poles (roadmap risk §7.1) and near-degenerate poles (mode counting).

(a) A mode is placed at (β − k_out)/k = t exactly, t = 1e-1 … 1e-8: pick β, then the
    core thickness from the closed form d = (mπ + atan(p_c/κ) + atan(p_s/κ))/κ. So the
    pole is known by construction, not found. Checks: do the finders see it, is the
    residue well defined, and does G on the path stay at the reference floor (g_ref_gamma: the
    cosh-variable g_ref breaks down as t → 0, see README)?
(b) Two cores behind an ever thicker oxide: supermodes split by ~e^{−γ d_ox}. Checks:
    do the finders resolve both, and when does the double root of D (holomorphy §5:
    dD/dq ≠ 0) become a numerical fact?
"""
import numpy as np

import sproto as S

PTS = [(0.0, 20.0), (1500.0, 100.0), (40.0, 400.0)]
D_BOX, Z_MIN = 1500.0, 20.0


def d_for(k, eps_c, eps_f, eps_s, beta, m, pol):
    kc, kf, ks_ = (k * np.sqrt(e) for e in (eps_c, eps_f, eps_s))
    kap = np.sqrt(kf**2 - beta**2)
    gc, gs = np.sqrt(beta**2 - kc**2), np.sqrt(beta**2 - ks_**2)
    pc, ps = (gc, gs) if pol == 2 else (gc * eps_f / eps_c, gs * eps_f / eps_s)
    return (m * np.pi + np.arctan(pc / kap) + np.arctan(ps / kap)) / kap


def near_cutoff(label, lam, n_clad, n_core, n_bot, m):
    k = 2 * np.pi * n_clad / lam
    ef, es = (n_core / n_clad) ** 2, (n_bot / n_clad) ** 2
    k_out = max(k, k * np.sqrt(es))
    print(f"\n{label}: mode m = {m} near cutoff, k_out = {'k_bot' if es > 1 else 'k1'}")
    for pol in (2, 1):
        print(f"  {'TE' if pol == 2 else 'TM'}")
        for t in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8):
            beta = k_out * (1 + t)
            d = d_for(k, 1.0, ef, es, beta, m, pol)
            st = S.Stack([ef], [d], es)
            nq = len(S.guided_poles(st, k, pol))
            ng_roots = S.guided_poles_gamma(st, k, pol)
            ng = len(ng_roots)
            wn = S.winding_count(st, k, pol)
            wg = S.winding_count_gamma(st, k, pol)
            # β by construction; the γ-finder's root must be it
            bf = min(ng_roots, key=lambda b: abs(b - beta)) if ng else np.nan
            r = 0.3 * (beta - k_out)  # the circle must not enclose the branch point
            res = S.residue_R(st, k, pol, beta, r=r)
            num = st._tm(np.array([beta + 0j]), k, pol)[0][0]
            res2 = num / S.dD(st, k, pol, beta, r=r)
            gam = np.sqrt(beta**2 - k**2)  # cover decay constant
            poles = sorted(set(ng_roots) - {bf} | {beta}, reverse=True)
            resid = [S.residue_R(st, k, pol, b, r=min(0.3 * (b - k_out), 1e-3 * b)) for b in poles]
            feats = S.stack_features(st, k, pol, poles)
            p1 = S.build_path(k, feats, D_BOX, Z_MIN)
            p2 = S.build_path(k, feats, D_BOX, Z_MIN, nodes_scale=2.0)
            ph = S.build_path(k, feats, D_BOX, Z_MIN, depth_scale=0.5)
            e_ref = e_s = e_d = 0.0
            share = 0.0
            for X, Z in PTS:
                ref = S.g_ref_gamma(st, k, pol, X, Z, poles, resid)
                g1 = S.g_path(p1, st, k, pol, X, Z)
                sc = abs(ref)
                e_ref = max(e_ref, abs(g1 - ref) / sc)
                e_s = max(e_s, abs(S.g_path(p2, st, k, pol, X, Z) - g1) / sc)
                e_d = max(e_d, abs(S.g_path(ph, st, k, pol, X, Z) - g1) / sc)
                a1 = 1j * gam
                pole = abs(np.pi / (2 * np.pi) * res / a1 * np.cos(beta * X) * np.exp(1j * a1 * Z))
                share = max(share, pole / sc)
            print(f"    t={t:.0e} d={d:8.2f}nm  found q/γ/wind_q/wind_γ={nq}/{ng}/{wn}/{wg}  "
                  f"|β_γ−β|/β={abs(bf - beta) / beta:.0e}  |Res|/k={abs(res) / k:.2e}  "
                  f"Res circle vs N/D'={abs(res - res2) / abs(res):.0e}  pole/|G|≤{share:.1e}  "
                  f"M={p1['q'].size} clr={min(p1['clear'].values()):.2f}  "
                  f"path−ref={e_ref:.0e} s2−s1={e_s:.0e} δ/2−δ={e_d:.0e}")


near_cutoff("SOI symmetric, SiO2|Si|SiO2 @1550", 1550, 1.444, 3.476, 1.444, 1)
near_cutoff("air|SiN|SiO2 @1550 (true fundamental cutoff)", 1550, 1.0, 2.0, 1.444, 0)

print("\n(b) Si 220 | SiO2 d_ox | Si 220 in SiO2 @1550: supermode splitting")
nc = 1.444
k = 2 * np.pi * nc / 1550
esi = (3.476 / nc) ** 2
for pol in (2, 1):
    for d_ox in (300, 1000, 2000, 3000, 4000, 6000):
        st = S.Stack([esi, 1.0, esi], [220, d_ox, 220], 1.0)
        rq = S.guided_poles(st, k, pol)
        rg = S.guided_poles_gamma(st, k, pol)
        rf = S.guided_poles_gamma(st, k, pol, n_grid=400000)
        wq, wg = S.winding_count(st, k, pol), S.winding_count_gamma(st, k, pol)
        line = (f"  {'TE' if pol == 2 else 'TM'} d_ox={d_ox:5d}: found q/γ/γ×100/wind_q/wind_γ = "
                f"{len(rq)}/{len(rg)}/{len(rf)}/{wq}/{wg}")
        if len(rf) == 2:
            b0, b1 = rf
            line += f"  Δβ/β = {(b0 - b1) / b0:.1e}"
            feats = S.stack_features(st, k, pol, rf)
            p1 = S.build_path(k, feats, D_BOX, Z_MIN)
            ph = S.build_path(k, feats, D_BOX, Z_MIN, depth_scale=0.5)
            e = max(abs(S.g_path(ph, st, k, pol, X, Z) - S.g_path(p1, st, k, pol, X, Z))
                    / abs(S.g_path(p1, st, k, pol, X, Z)) for X, Z in PTS)
            res = [S.residue_R(st, k, pol, b, r=0.3 * (b0 - b1)) for b in rf]
            line += f"  |Res|/k = {abs(res[0]) / k:.1e}, {abs(res[1]) / k:.1e}  path δ/2−δ = {e:.0e}"
        print(line)

print("\n(c) whose floor is it? Both references at t = 1e-6 and 1e-8, fold half-width 0.45 vs 0.2 (path s = 2)")
for label, n_clad, n_core, n_bot, m in (("SOI sym TE1", 1.444, 3.476, 1.444, 1), ("air|SiN|SiO2 TE0", 1.0, 2.0, 1.444, 0)):
    k = 2 * np.pi * n_clad / 1550
    ef, es = (n_core / n_clad) ** 2, (n_bot / n_clad) ** 2
    k_out = max(k, k * np.sqrt(es))
    for t in (1e-6, 1e-8):
        beta = k_out * (1 + t)
        st = S.Stack([ef], [d_for(k, 1.0, ef, es, beta, m, 2)], es)
        poles = [beta if abs(b - beta) < 1e-12 * beta else b for b in S.guided_poles_gamma(st, k, 2)]
        resid = [S.residue_R(st, k, 2, b, r=min(0.3 * (b - k_out), 1e-3 * b)) for b in poles]
        p2 = S.build_path(k, S.stack_features(st, k, 2, poles), D_BOX, Z_MIN, nodes_scale=2.0)
        for X, Z in PTS[:2]:
            g = S.g_path(p2, st, k, 2, X, Z)
            r = [f(st, k, 2, X, Z, poles, resid, hw_frac=h) for f in (S.g_ref, S.g_ref_gamma) for h in (0.45, 0.2)]
            e = [f"{abs(x - g) / abs(g):.0e}" for x in r]
            print(f"  {label:17s} t={t:.0e} X={X:5.0f} Z={Z:4.0f}: path − g_ref (.45/.2) = {e[0]}/{e[1]}   "
                  f"path − g_ref_gamma (.45/.2) = {e[2]}/{e[3]}")
