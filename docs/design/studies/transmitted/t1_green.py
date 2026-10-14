"""T1: G_t vs (i/4)H0 at eps=1 (real/complex k); vs real-axis reference; continuity."""

import time

import numpy as np
from scipy.special import hankel1

from pysie2d.layered import reflected_green
from tproto import PI, t_green, t_path, t_ref

n_clad = 1.33
lam = 633.0
k_r = 2 * PI * n_clad / lam
k_c = 2 * PI * n_clad / (lam * (1 + 0.05j))  # Q = 10
z_int = 0.0
# observation (x, z) below, source (xs, zs) above
obs = np.array([[0.0, -5.0], [150.0, -40.0], [-300.0, -1.0], [20.0, -800.0], [900.0, -200.0], [0.0, 0.0]])
src = np.array([[10.0, 15.0], [-40.0, 120.0]])

t0 = time.perf_counter()
print("== eps=1 vs (i/4)H0 and source derivatives ==")
for k in (k_r, k_c):
    for pol in (2, 1):
        worst = 0.0
        for xs, zs in src:
            x, z = obs[:, 0], obs[:, 1]
            D = max(np.ptp(np.r_[x, xs]), 1.0)
            zmin = (zs - z_int) + (z_int - z).min()
            p = t_path(k, 1.0 + 0j, pol, D, zmin)
            g, gx, gz = t_green(p, pol, k, 1.0, x, z, xs, zs, z_int)
            rho = np.hypot(x - xs, z - zs)
            g0 = 0.25j * hankel1(0, k * rho)
            h1 = hankel1(1, k * rho)
            gx0 = 0.25j * k * h1 * (x - xs) / rho
            gz0 = 0.25j * k * h1 * (z - zs) / rho
            for a, b in ((g, g0), (gx, gx0), (gz, gz0)):
                worst = max(worst, float(np.max(np.abs(a - b) / np.abs(b))))
        print(f"k={'real' if k == k_r else 'cplx'} pol={pol}: max rel err {worst:.2e}")

print("\n== vs real-axis reference (real k) ==")
for eps_abs, name in ((2.25, "glass"), (2.25 + 0.1j, "lossy"), (-18.3 + 0.48j, "silver"), (1.0, "low-n")):
    eps = eps_abs / n_clad**2
    for pol in (2, 1):
        worst = 0.0
        for xs, zs in src:
            x, z = obs[:, 0], obs[:, 1]
            D = max(np.ptp(np.r_[x, xs]), 1.0)
            zmin = (zs - z_int) + (z_int - z).min()
            p = t_path(k_r, eps, pol, D, zmin)
            got = t_green(p, pol, k_r, eps, x, z, xs, zs, z_int)
            for which in range(3):
                ref = np.array([t_ref(xi - xs, zs - z_int, z_int - zi, k_r, eps, pol, which) for xi, zi in obs])
                err = np.abs(got[which] - ref) / np.max(np.abs(ref))
                worst = max(worst, float(err.max()))
        print(f"{name:7s} pol={pol}: max err / max|ref| {worst:.2e}")

print("\n== continuity G_free + G_ind (z_int+1e-9) vs G_t (z_int) ==")
for eps_abs, name in ((2.25, "glass"), (-18.3 + 0.48j, "silver")):
    eps = eps_abs / n_clad**2
    for k in (k_r, k_c):
        for pol in (2, 1):
            xs, zs = 10.0, 15.0
            x = np.array([-200.0, 0.0, 30.0, 400.0])
            z = np.zeros_like(x)
            D = max(np.ptp(np.r_[x, xs]), 1.0)
            p = t_path(k, eps, pol, D, zs - z_int)
            gt = t_green(p, pol, k, eps, x, z, xs, zs, z_int)[0]
            pr = SP = t_path(k, eps, pol, D, 2 * (zs - z_int))
            gi = reflected_green(pr, pol, k, eps, x, z + 1e-9, xs, zs, z_int)[0]
            g0 = 0.25j * hankel1(0, k * np.hypot(x - xs, z + 1e-9 - zs))
            err = np.max(np.abs(g0 + gi - gt) / np.abs(gt))
            print(f"{name:7s} k={'real' if k == k_r else 'cplx'} pol={pol}: {err:.2e}")
print(f"\nelapsed {time.perf_counter() - t0:.1f}s")
