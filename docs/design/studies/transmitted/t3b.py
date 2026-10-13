"""Pending (spec §7.1): per-angle near-to-far error for glass, incl. far from θ_c.

Not run yet. Expectation: small away from the substrate critical angle, slow
near it (lateral wave / non-uniform asymptotics). If θ = π also fails, suspect
the phase factor e^{iα_bot z_bot} in a_down.
"""

import numpy as np

from pysie2d.layered import _normalised
from t3_far import solve
from tproto import PI, a_down, t_green, t_path

for pol in (2, 1):
    s, res = solve(pol, 2.25)
    bnds, k, eps, _, z_int = res._substrate()
    k2 = k * np.sqrt(_normalised(eps))
    tc = np.arcsin(k / k2.real)
    print(f"pol={pol}: substrate critical angle from -z: {tc:.3f} rad")
    offs = np.array([0.0, 0.3, 0.6, 0.9, 1.3, 1.45])
    thetas = PI + offs
    ad = a_down(thetas, bnds, k, eps, pol, z_int)
    g = s.geometry
    nn = g.n_pts
    for R in (2e4, 8e4, 3.2e5):
        x, z = R * np.sin(thetas), R * np.cos(thetas)
        p = t_path(k, eps, pol, np.ptp(np.r_[g.f, x]), (g.g - z_int).min() + (z_int - z).min())
        psi = np.zeros(len(x), complex)
        for j in range(nn):
            G, Gx, Gz = t_green(p, pol, k, eps, x, z, g.f[j], g.g[j], z_int)
            psi -= g.delt * ((Gx * g.dg[j] - Gz * g.df[j]) * res.ei[j] + G * res.ei[nn + j])
        fac = 0.25j * np.sqrt(2 / (PI * k2 * R)) * np.exp(1j * (k2 * R - PI / 4))
        err = np.abs(psi / fac - ad) / np.abs(ad)
        print(f"  R={R:.0e} offsets {offs} -> rel err {np.array2string(err, precision=1)} (path nodes {p.q.size})")
