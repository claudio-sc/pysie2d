"""Plan step 2: circle radius × gap × Im ε over the SiN slab (E1), TE0 (pol 2), 1550 nm.

Records T, R, P_scatt = P_up + P_down (E5) and P_abs; checks balance on every
solve and compares the best P_abs with the 50 % single-resonance ceiling.

Run from the repo root:  uv run python docs/design/studies/anapole/wg_baseline.py
"""
import sys, json, time
sys.path.insert(0, "docs/design/studies/anapole")
import numpy as np
from shapes import geometry, N_SI
from pysie2d import BIESolver, Material, Stack, Geometry

LAM, NN = 1550.0, 192
stack = Stack(layers=((2.0**2, 400.0),), eps_bot=1.444**2, z_top=0.0)
print("TE modes n_eff:", [round(m.beta / (2*np.pi/LAM), 4) for m in stack.guided_modes(LAM, 2, 1.0)])

radii = np.arange(150.0, 651.0, 25.0)
gaps = (50.0, 100.0, 200.0)
epsis = (0.01, 0.03, 0.1, 0.3, 1.0)
rows, worst = [], 0.0
t0 = time.perf_counter()
for rad in radii:
    g0 = geometry(np.zeros(10), np.pi * rad**2, NN)
    zmin = g0.g.min()
    for gap in gaps:
        g = Geometry(g0.f, g0.g - zmin + gap, g0.df, g0.dg, g0.ddf, g0.ddg, rad=g0.rad, z0=-zmin + gap)
        for epsi in epsis:
            s = BIESolver(g, Material(n_core=N_SI, n_clad=1.0, pol=2, epsi=epsi), background=stack)
            ma = s.scatter_mode(LAM, 0, +1, x_ref=0.0).mode_amplitudes(x_ref=0.0)
            worst = max(worst, abs(ma.balance()))
            rows.append(dict(rad=rad, gap=gap, epsi=epsi, T=float(np.sum(ma.p_plus)),
                             R=float(np.sum(ma.p_minus)), scatt=float(ma.p_up + ma.p_down),
                             abs=float(ma.p_abs)))
print(f"{len(rows)} solves in {time.perf_counter()-t0:.0f}s, worst |balance| {worst:.1e}")
json.dump(rows, open("docs/design/studies/anapole/results/wg_baseline.json", "w"), indent=1)

best = max(rows, key=lambda r: r["abs"])
print("best P_abs:", best, f"(single-resonance ceiling 0.5)")
for gap in gaps:
    for epsi in epsis:
        b = max((r for r in rows if r["gap"] == gap and r["epsi"] == epsi), key=lambda r: r["abs"])
        print(f"gap {gap:3.0f} Im eps {epsi:4.2f}: max abs {b['abs']:.4f} at R {b['rad']:.0f}"
              f"  T {b['T']:.3f} R {b['R']:.3f} scatt {b['scatt']:.3f}")
