"""T4: banded transmitted GEMM vs pointwise G_t; 100 × 100 substrate map timing (spec §7.2)."""

import time

import numpy as np

from pysie2d import BIESolver, Geometry, HalfSpace, Material
from pysie2d.layered import TAIL_DECAY, SommerfeldPath, _normalised, alpha
from tproto import PI, t_green, tau

LAM = 633.0


def t_blocks(path, pol, k, eps, x_o, z_o, geo, z_int, x_c):
    """Banded A·diag(w·τ)·Bᵀ: (M1, M2) with M1 = h(∂x'G_t·dg − ∂z'G_t·df), M2 = h·G_t."""
    eps = _normalised(eps)
    k2 = k * np.sqrt(eps)
    hb, hs = z_int - z_o, geo.g - z_int  # observation depth, source height
    hb_min, hs_min = hb.min(), hs.min()
    xt, xs = x_o - x_c, geo.f - x_c
    m1 = np.zeros((x_o.size, geo.n_pts), complex)
    m2 = np.zeros_like(m1)
    for a0, i0, i1 in path.bands:
        if a0 == 0.0:
            rows, cols = np.arange(x_o.size), np.arange(geo.n_pts)
        else:
            lim = TAIL_DECAY / a0
            rows = np.flatnonzero(hb < lim - hs_min)
            cols = np.flatnonzero(hs < lim - hb_min)
            if rows.size == 0 or cols.size == 0:
                continue
        q = np.concatenate([path.q[i0:i1], -path.q[i0:i1]])
        w = np.concatenate([path.w[i0:i1], path.w[i0:i1]])
        a1, a2 = alpha(q, k), alpha(q, k2)
        coef = (1j / (4 * PI)) * w * tau(q, k, eps, pol)
        e_t = np.exp(1j * (np.outer(xt[rows], q) + np.outer(hb[rows], a2)))
        e_s = np.exp(1j * (-np.outer(xs[cols], q) + np.outer(hs[cols], a1)))
        poly = -1j * (np.outer(geo.dg[cols], q) + np.outer(geo.df[cols], a1))
        left = e_t * coef
        ix = np.ix_(rows, cols)
        m2[ix] += left @ e_s.T
        m1[ix] += left @ (e_s * poly).T
    return geo.delt * m1, geo.delt * m2


def main():
    geo = Geometry.gielis(rad=100.0, n_pts=256, m=0, x0=0.0, z0=115.0)
    z_int = 0.0
    for name, eps_sub in (("glass", 2.25), ("silver", -18.3 + 0.48j), ("low-n", 1.0)):
        for pol in (2, 1):
            s = BIESolver(geo, Material(n_core=2.0, n_clad=1.33, pol=pol, epsi=0.5),
                          HalfSpace(eps_sub, z_int=z_int))
            res = s.scatter(LAM, 20.0)
            k = res.wnum_bg
            eps = s.background.eps_rel(1.33, LAM)
            n = geo.n_pts
            # 100 × 100 substrate map, depth 5 … 600 nm
            xg, zg = np.meshgrid(np.linspace(-400, 400, 100), -np.linspace(5, 600, 100))
            xo, zo = xg.ravel(), zg.ravel()
            hs_min, hb_min = (geo.g - z_int).min(), (z_int - zo).min()
            zmin = hs_min + hb_min
            fx = np.r_[geo.f, xo]
            D = max(np.ptp(fx), zmin)
            t0 = time.perf_counter()
            path = SommerfeldPath.for_wavenumber(k, eps, pol, D, zmin)
            m1, m2 = t_blocks(path, pol, k, eps, xo, zo, geo, z_int, 0.5 * (fx.max() + fx.min()))
            psi = -(m1 @ res.ei[:n] + m2 @ res.ei[n:])
            t_sub = time.perf_counter() - t0
            # pointwise reference on 200 random map points (same path, no banding, no GEMM)
            idx = np.random.default_rng(0).choice(xo.size, 200, replace=False)
            ref = np.empty(idx.size, complex)
            for r, i in enumerate(idx):
                g_, gx, gz = t_green(path, pol, k, eps, xo[i], zo[i], geo.f, geo.g, z_int)
                ref[r] = -np.sum(geo.delt * ((gx * geo.dg - gz * geo.df) * res.ei[:n] + g_ * res.ei[n:]))
            err = np.max(np.abs(psi[idx] - ref)) / np.max(np.abs(ref))
            # cover map of the same size, via the shipped eval_field
            xc, zc = np.meshgrid(np.linspace(-400, 400, 100), np.linspace(235, 830, 100))
            t0 = time.perf_counter()
            res.eval_field(xc.ravel(), zc.ravel())
            t_cov = time.perf_counter() - t0
            print(f"{name:6s} pol={pol}  GEMM vs pointwise {err:.1e}  "
                  f"nodes {path.q.size}  substrate {t_sub:.2f} s  cover {t_cov:.2f} s")


if __name__ == "__main__":
    main()
