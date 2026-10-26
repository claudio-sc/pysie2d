"""Is the guided mode's failure to see a dark particle a drive effect or a slab-feedback effect?

In the cover the TE0 field is e^{iβx − γz}, a plane wave at complex angle with cos α = n_eff.
Expanded about the particle centre, order m carries |e^{−imα}| ~ (n_eff + √(n_eff² − 1))^{|m|}
(≈ 3.1^{|m|} here) relative to a real plane wave, so a free-space dark design whose residual
sits in its highest orders is lit up. Test: solve in free space (no slab, so no feedback)
under that evanescent drive and under the plane wave, for the circle and the free-space dark
design (R_eq 500, CD rules, campaign_500), and compare ratios with the stack's P_scatt.

Run:  uv run python docs/design/studies/anapole/wg_evanescent.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shapes import N_SI, geometry  # noqa: E402
from pysie2d import BIESolver, Geometry, Material, Stack  # noqa: E402

LAM, GAP, NN, MMAX = 1550.0, 100.0, 192, 14
AREA = np.pi * 500.0**2
STACK = Stack(layers=((2.0**2, 400.0),), eps_bot=1.444**2, z_top=0.0)
k0 = 2 * np.pi / LAM
n_eff = STACK.guided_modes(LAM, 2, 1.0)[0].beta / k0
beta, gamma = n_eff * k0, k0 * np.sqrt(n_eff**2 - 1)
print(f"n_eff {n_eff:.4f}: predicted per-order weight (n_eff + √(n_eff²−1)) = {n_eff + np.sqrt(n_eff**2 - 1):.3f}")


def evanescent(nn, k, f, g):
    ei = np.zeros(2 * nn, dtype=complex)
    ei[:nn] = np.exp(1j * beta * f - gamma * g)
    return ei


c5 = json.loads((HERE / "results/campaign_500.json").read_text())
dark = min((o for o in c5["phase_a"] if o["feasible"]), key=lambda o: o["f"])["a"]
out = {}
for name, a in (("circle", np.zeros(10)), ("free-space dark", np.array(dark))):
    g0 = geometry(np.asarray(a, float), AREA, NN)
    dz = GAP - g0.g.min()
    g = Geometry(g0.f, g0.g + dz, g0.df, g0.dg, g0.ddf, g0.ddg, rad=g0.rad, z0=dz)
    mat = Material(n_core=N_SI, n_clad=1.0, pol=2)
    # the plane wave along x has unit amplitude everywhere; scale the tail to unit amplitude at
    # the particle centre so the two drives compare at equal local strength
    pw = BIESolver(g, mat).scatter(LAM, angle=90.0).multipoles(mmax=MMAX).c
    ev = BIESolver(g, mat).scatter(LAM, incident_rhs=evanescent).multipoles(mmax=MMAX).c * np.exp(gamma * dz)
    stack = BIESolver(g, mat, background=STACK).scatter_mode(LAM, 0, +1, x_ref=0.0).mode_amplitudes(x_ref=0.0)
    out[name] = dict(pw=float(np.sum(abs(pw) ** 2)), ev=float(np.sum(abs(ev) ** 2)), stack=float(stack.p_up + stack.p_down))
    m = np.arange(-MMAX, MMAX + 1)
    print(f"\n{name}: Σ|c_m|² plane wave {out[name]['pw']:.3e}, evanescent {out[name]['ev']:.3e};  stack P_scatt {out[name]['stack']:.4f}")
    for mm in range(0, 7):
        sel = abs(m) == mm
        print(f"  |m|={mm}: plane wave {np.sum(abs(pw[sel])**2):.2e}  evanescent {np.sum(abs(ev[sel])**2):.2e}")
r = {q: out["free-space dark"][q] / out["circle"][q] for q in ("pw", "ev", "stack")}
print(f"\ndark/circle: plane wave {r['pw']:.2e}, evanescent (free space) {r['ev']:.2f}, stack P_scatt {r['stack']:.2f}")
