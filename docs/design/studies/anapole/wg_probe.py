"""Probe: the R_eq 500 nm designs above a SiN slab under TE0 incidence (pol 2): cost and power account.

Run from the repo root:  uv run python docs/design/studies/anapole/wg_probe.py
"""
import sys, json, time
sys.path.insert(0, "docs/design/studies/anapole")
import numpy as np
from shapes import geometry, N_SI
from pysie2d import BIESolver, Material, Stack, Geometry

LAM = 1550.0
# SiN slab 400 nm on SiO2, air cover; particle above the slab with a 100 nm gap
stack = Stack(layers=((2.0**2, 400.0),), eps_bot=1.444**2, z_top=0.0)
modes = stack.guided_modes(LAM, 2, 1.0)
print("TE modes n_eff:", [round(m.beta / (2*np.pi/LAM), 4) for m in modes])
c5 = json.load(open("docs/design/studies/anapole/results/campaign_500.json"))
a_nom = min((o for o in c5["phase_a"] if o["feasible"]), key=lambda o: o["f"])["a"]
A = np.pi * 500**2
for name, a in (("circle R_eq 500", np.zeros(10)), ("dark design (free-space opt.)", np.array(a_nom))):
    g0 = geometry(np.asarray(a), A, 192)
    zmin = g0.g.min()
    gap = 100.0
    g = Geometry(g0.f, g0.g - zmin + gap, g0.df, g0.dg, g0.ddf, g0.ddg, rad=g0.rad, z0=-zmin + gap)
    for epsi in (0.0, 0.05):
        s = BIESolver(g, Material(n_core=N_SI, n_clad=1.0, pol=2, epsi=epsi), background=stack)
        t = time.perf_counter(); r = s.scatter_mode(LAM, 0, +1, x_ref=0.0); t1 = time.perf_counter()
        ma = r.mode_amplitudes(x_ref=0.0); t2 = time.perf_counter()
        T = float(np.sum(ma.p_plus)); R = float(np.sum(ma.p_minus))
        print(f"{name:30s} Im eps {epsi:4.2f}: T {T:.4f} R {R:.4f} up {ma.p_up:.4f} down {ma.p_down:.4f} abs {ma.p_abs:.4f}"
              f"  balance {ma.balance():.0e}  solve {t1-t:.2f}s amps {t2-t1:.2f}s")
