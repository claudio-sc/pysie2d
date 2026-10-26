"""P_abs two ways on the circle over the slab (handoff §3.3): boundary flux vs volume integral.

mode_amplitudes().p_abs comes from the boundary densities. The volume path fits the
interior field on a circle r0 well inside the particle — eval_field needs ~5 node
spacings of clearance, which a general shape does not leave — to u = Σ c_m J_m(k_in r)e^{imθ},
then integrates ∫|u|² over the disk analytically in θ and by Gauss–Legendre in r. P_abs ∝
k0²·Im ε·∫|u|², with a constant fixed by the mode normalisation only, so the ratio must
be the same for every radius and Im ε. Two fit radii guard against truncation of the series.

Run:  uv run python docs/design/studies/anapole/wg_abs_check.py
"""
import numpy as np
from scipy.special import jv

from pysie2d import BIESolver, Geometry, Material, Stack

LAM, GAP, NN, N_SI, MMAX = 1550.0, 100.0, 384, 3.476, 30
STACK = Stack(layers=((2.0**2, 400.0),), eps_bot=1.444**2, z_top=0.0)
k0 = 2 * np.pi / LAM
th = 2 * np.pi * (np.arange(NN) + 0.5) / NN
ratios = []
for rad in (300.0, 450.0, 500.0):
    zc = GAP + rad
    g = Geometry(rad * np.sin(th), zc + rad * np.cos(th), rad * np.cos(th), -rad * np.sin(th),
                 -rad * np.sin(th), -rad * np.cos(th), rad=rad, z0=zc)
    for epsi in (0.03, 0.3, 1.0):
        s = BIESolver(g, Material(n_core=N_SI, n_clad=1.0, pol=2, epsi=epsi), background=STACK)
        r = s.scatter_mode(LAM, 0, +1, x_ref=0.0)
        p_abs = r.mode_amplitudes(x_ref=0.0).p_abs
        k_in = k0 * np.sqrt(N_SI**2 + 1j * epsi)
        vol = []
        for r0 in (0.5 * rad, 0.65 * rad):
            phi = 2 * np.pi * np.arange(256) / 256
            u = r.eval_field(r0 * np.sin(phi), zc + r0 * np.cos(phi))
            cm = np.fft.fft(u) / 256  # u(φ) = Σ c'_m e^{imφ}
            m = np.r_[0:MMAX + 1, -MMAX:0]
            cm = cm[m % 256] / jv(m, k_in * r0)
            x, w = np.polynomial.legendre.leggauss(80)
            rr = 0.5 * rad * (x + 1)
            radial = (np.abs(jv(m[:, None], k_in * rr[None, :])) ** 2 * rr) @ w * 0.5 * rad
            vol.append(2 * np.pi * np.sum(np.abs(cm) ** 2 * radial))
        ratio = p_abs / (k0**2 * epsi * vol[1])
        ratios.append(ratio)
        print(f"R {rad:.0f} Im eps {epsi:4.2f}: p_abs {p_abs:.5f}  fit-radius spread {abs(vol[0]/vol[1]-1):.1e}"
              f"  p_abs/(k0² Im ε ∫|u|²) {ratio:.10f}")
ratios = np.array(ratios)
print(f"ratio spread (max/min − 1): {ratios.max() / ratios.min() - 1:.1e}")
