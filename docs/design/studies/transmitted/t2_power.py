"""T2: dipole power balance P_up + P_down + P_abs = P_in (G14 setup), flux vs far field."""

import time

import numpy as np

from pysie2d import BIESolver, Geometry, HalfSpace, Material, relative_ldos
from pysie2d.fields import _upward_amplitude
from pysie2d.layered import _normalised, alpha, fresnel_r
from tproto import PI, a_down, flux_rule, flux_weight, spectrum, tau

LAM, XS, ZS = 633.0, 60.0, 330.0


def solver(pol, bg):
    geo = Geometry.gielis(rad=100.0, n_pts=256, m=0, x0=0.0, z0=115.0)
    return BIESolver(geo, Material(n_core=2.0, n_clad=1.33, pol=pol, epsi=0.5), bg)


def powers(s, n_pan=24, sine_map=True, n_theta=400):
    res = s.scatter_dipole(LAM, XS, ZS)
    k, pol, bg = res.wnum_bg, s.material.pol, s.background
    eps = bg.eps_rel(1.33, LAM)
    bnds, _, _, _, z_int = res._substrate()
    # P_up exactly as G14
    x, w = np.polynomial.legendre.leggauss(500)
    th = 0.5 * PI * x
    a_up = _upward_amplitude(th, *res._substrate())
    z_img = 2 * z_int - ZS
    r = fresnel_r(k * np.sin(th), k, eps, pol)
    sdir = np.exp(-1j * k * (XS * np.sin(th) + ZS * np.cos(th))) + r * np.exp(
        -1j * k * (XS * np.sin(th) + z_img * np.cos(th)))
    p_up = np.sum(0.5 * PI * w * np.abs(sdir + a_up) ** 2) / (8 * PI)
    g, nn = s.geometry, s.geometry.n_pts
    p_abs = -np.imag(g.delt * np.sum(np.conj(res.ei[:nn]) * res.ei[nn:]))
    p_in = relative_ldos(s, LAM, XS, ZS) / 4
    if bg.is_pec:
        return p_in, p_up, p_abs, 0.0, None
    # P_down by interface flux
    h_min = min(g.g.min(), ZS) - z_int
    D = max(np.ptp(np.r_[g.f, XS]), 2 * h_min)
    q, wq = flux_rule(k, eps, pol, D, 2 * h_min, n_pan=n_pan, sine_map=sine_map)
    a1 = alpha(q, k)
    A = spectrum(q, a1, bnds, z_int) + np.exp(-1j * q * XS + 1j * a1 * (ZS - z_int))
    p_flux = np.sum(wq * flux_weight(q, k, eps, pol) * np.abs(tau(q, k, eps, pol) * A) ** 2) / (8 * PI)
    # P_down by far field (lossless, Re eps > 0 only), split at the critical angle
    p_far = None
    epsn = _normalised(eps)
    if epsn.imag == 0 and epsn.real > 0:
        k2 = k * np.sqrt(epsn)
        brk = [0.5 * PI, 1.5 * PI]
        if k2.real > k:  # critical angles in the substrate
            tc = np.arcsin(k / k2.real)
            brk += [PI - tc, PI + tc]
        brk = sorted(brk)
        xg, wg = np.polynomial.legendre.leggauss(n_theta)
        tot = 0.0
        for lo, hi in zip(brk[:-1], brk[1:]):
            c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
            tt = c + h * np.sin(0.5 * PI * xg)
            ww = h * 0.5 * PI * np.cos(0.5 * PI * xg) * wg
            qq = k2 * np.sin(tt)
            aa1, aa2 = alpha(qq, k), alpha(qq, k2)
            src = aa2 * tau(qq, k, epsn, pol) * np.exp(1j * aa2 * z_int) * np.exp(
                -1j * qq * XS + 1j * aa1 * (ZS - z_int))
            amp = a_down(tt, bnds, k, epsn, pol, z_int) + src
            tot += np.sum(ww * np.abs(amp) ** 2)
        wfar = 1.0 if pol == 2 else 1.0 / epsn.real
        p_far = wfar * tot / (8 * PI)
    return p_in, p_up, p_abs, p_flux, p_far


if __name__ == '__main__':
    t0 = time.perf_counter()
    cases = [("PEC", HalfSpace.pec()), ("glass", HalfSpace(2.25)), ("low-n", HalfSpace(1.0)),
             ("lossy", HalfSpace(2.25 + 0.1j)), ("silver", HalfSpace(-18.3 + 0.48j)),
             ("weakloss", HalfSpace(2.25 + 1e-6j)), ("realmetal", HalfSpace(-4.0 + 0j))]
    for name, bg in cases:
        for pol in (2, 1):
            s = solver(pol, bg)
            p_in, p_up, p_abs, p_flux, p_far = powers(s)
            bal = (p_up + p_flux + p_abs - p_in) / p_in
            line = f"{name:9s} pol={pol} P_in={p_in:.6f} up={p_up:.4f} down={p_flux:.4f} abs={p_abs:.4f} balance={bal:+.2e}"
            if p_far is not None:
                line += f" flux-vs-far={(p_flux - p_far) / p_flux:+.2e}"
            if not bg.is_pec:
                p48 = powers(s, n_pan=48)[3]
                line += f" n24-vs-48={(p_flux - p48) / max(p48, 1e-300):+.1e}"
            print(line, flush=True)
    print(f"elapsed {time.perf_counter() - t0:.1f}s")
