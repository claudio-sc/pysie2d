"""Near field of a line dipole over a Stack, everywhere: cover, layers, bottom.

Spectral field per unit source, (∇·p∇ + k²w)G = −δ, source at height h₀ above z_top:
  cover   g = (i/2α₁)[e^{iα₁|z−z₀|} + R e^{iα₁(h+h₀)}]   (direct part done as (i/4)H₀ in space)
  layer j g = A_j [e^{iα_j s} + G_j e^{iα_j(2d_j − s)}],  s = z_j − z ∈ [0, d_j]
  bottom  g = A_b e^{−iα_b(z − z_bot)}
with G_j the generalised reflection at the *bottom* of layer j, A₁ = a₀ t₀/(1 + r₀Γ₁),
a₀ t₀ = (i/2)e^{iα₁h₀}·2/(Y₀+Y₁) (no 0/0 at q = k₁), and
A_{j+1} = A_j e^{iα_j d_j} t_j/(1 + r_jΓ_{j+1}). Every exponential is bounded (Im α_j ≥ 0).
Space: G = (1/π)∫_Γ g cos(qX) dq on the folded deformed path, as g_path.
"""
import numpy as np
from scipy.special import hankel1

import sproto as S


def spectra(st, k, pol, q, z, h0):
    """g(z; q) for every z below z_top = 0 (layers and bottom), shape (nz, nq)."""
    Gam, Y, al = st._gammas(q, k, pol)
    _, kb = st.ks(k)
    ab = S.alpha(q, kb)
    a1 = S.alpha(q, k)
    n = len(st.d)
    z_tops = np.concatenate([[0.0], -np.cumsum(st.d)])
    out = np.zeros((z.size, q.size), dtype=complex)
    r0 = (Y[0] - Y[1]) / (Y[0] + Y[1])
    A = 0.5j * np.exp(1j * a1 * h0) * 2.0 / (Y[0] + Y[1]) / (1.0 + r0 * Gam[1])
    for j in range(1, n + 1):
        d = st.d[j - 1]
        r = (Y[j] - Y[j + 1]) / (Y[j] + Y[j + 1])
        Gb = (r + Gam[j + 1]) / (1.0 + r * Gam[j + 1])  # reflection at the bottom of layer j
        m = (z <= z_tops[j - 1]) & (z >= z_tops[j])
        s = (z_tops[j - 1] - z[m])[:, None]
        a = al[j - 1][None, :]
        out[m] = A * (np.exp(1j * a * s) + Gb * np.exp(1j * a * (2 * d - s)))
        A = A * np.exp(1j * al[j - 1] * d) * (2.0 * Y[j] / (Y[j] + Y[j + 1])) / (1.0 + r * Gam[j + 1])
    m = z < z_tops[-1]
    out[m] = A * np.exp(-1j * ab[None, :] * (z[m] - z_tops[-1])[:, None])
    return out


def field_map(st, k, pol, h0, x, z, path):
    """Total G on the grid x (≥ 0 or any), z; z_top = 0, source at (0, h0)."""
    q, w = path["q"], path["w"]
    C = np.cos(np.outer(q, x))
    U = np.empty((z.size, x.size), dtype=complex)
    up = z > 0
    if up.any():
        a1 = S.alpha(q, k)
        R = st.R(q, k, pol)
        gr = 0.5j * R / a1 * np.exp(1j * np.outer(z[up] + h0, a1))
        X, Z = np.meshgrid(x, z[up])
        rho = np.hypot(X, Z - h0)
        U[up] = 0.25j * hankel1(0, k * np.maximum(rho, 1e-9)) + (gr * w) @ C / np.pi
    dn = ~up
    if dn.any():
        U[dn] = (spectra(st, k, pol, q, z[dn], h0) * w) @ C / np.pi
    return U


if __name__ == "__main__":
    # checks before any plot: continuity, cover match, far-x guided sum inside the core
    nc = 1.444
    k = 2 * np.pi * nc / 1550
    esi = (3.476 / nc) ** 2
    for name, st in {"SOI 220": S.Stack([esi], [220], 1.0),
                     "Si/SiO2/Si": S.Stack([esi, 1.0, esi], [220, 300, 220], 1.0),
                     "air|Si|SiO2": S.Stack([(3.476) ** 2], [220], 1.444**2)}.items():
        kk = 2 * np.pi / 1550 if name.startswith("air") else k
        for pol in (2, 1):
            h0 = 50.0
            poles = S.guided_poles_gamma(st, kk, pol)
            q = np.linspace(0.01, 6, 7) * kk + 0j
            q = np.concatenate([q, q - 0.3j * kk])
            # (i) continuity of u and p u' across each interior interface
            zt = np.concatenate([[0.0], -np.cumsum(st.d)])
            e = 1e-6
            worst = 0.0
            ps = [1.0] + ([1.0] * len(st.d) if pol == 2 else [1 / x for x in st.eps]) + \
                 [1.0 if pol == 2 else 1 / st.eps_bot]
            for i, zi in enumerate(zt[1:], start=1):
                ga = spectra(st, kk, pol, q, np.array([zi + e, zi + 2 * e]), h0)
                gb = spectra(st, kk, pol, q, np.array([zi - e, zi - 2 * e]), h0)
                ua, ub = 2 * ga[0] - ga[1], 2 * gb[0] - gb[1]
                da = ps[i] * (ga[1] - ga[0]) / e
                db = ps[i + 1] * (gb[0] - gb[1]) / e
                worst = max(worst, np.max(np.abs(ua - ub) / np.abs(ua)), np.max(np.abs(da - db) / np.abs(da)))
            # (ii) at z_top: layer formula vs cover formula (direct + reflected)
            a1 = S.alpha(q, kk)
            cov = 0.5j / a1 * (np.exp(1j * a1 * h0) + st.R(q, kk, pol) * np.exp(1j * a1 * h0))
            lay = spectra(st, kk, pol, q, np.array([0.0]), h0)[0]
            top = np.max(np.abs(cov - lay) / np.abs(cov))
            # (iii) far x inside the core: the guided sum of S9
            zc = np.array([-60.0, -110.0])
            X = 3e4
            p = S.build_path(kk, S.stack_features(st, kk, pol, poles), X, h0)
            U = field_map(st, kk, pol, h0, np.array([X]), zc, p)[:, 0]
            gm = np.zeros(2, dtype=complex)
            scale = np.zeros(2)
            for b in poles:
                u, N, _ = S.mode_profile(st, kk, pol, b)
                t = 1j / (2 * b * N) * u(zc) * u(np.array([h0]))[0] * np.exp(1j * b * X)
                gm += t
                scale += np.abs(t)
            print(f"{name:12s} {'TE' if pol == 2 else 'TM'}: continuity (FD, e=1e-6 nm) ≤ {worst:.0e}  "
                  f"z_top layer vs cover = {top:.0e}  core field at 30 µm vs guided sum = "
                  f"{np.max(np.abs(U - gm) / scale):.1e}")
