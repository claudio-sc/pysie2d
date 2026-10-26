"""S12b: in-plane dipole (derivative source, S11) + particle over the stack: power balance.

RHS on the nodes: −(b·∇')(G₀ + G_refl)(r_j, r₀).  Source power
  P_src = k²|b|²/8 + Im[(b*·∇)(b·∇')G_refl](r₀, r₀) + Im[−(b*·∇)u_s](r₀),
the last term by a 5-point central difference of the scattered field (h = 0.5 nm; the
code-spec makes it analytic and gates it against this). Outputs: P_up, P_down with the
source amplitudes A↑ = i(qb_x + α₁b_z), A↓ = i(qb_x − α₁b_z); P_abs; guided
c^± = overlap + (i/2βN)u(z₀)(±iβb_x + γ₁b_z)e^{∓iβx₀}.
"""
import numpy as np
from scipy.special import hankel1

import sproto as S
from pysie2d import Geometry, Material
from s12_particle import Ctx, p_abs, shifted


def grad_fd(f, x0, z0, h=0.5):
    c = np.array([1, -8, 8, -1]) / (12 * h)
    off = np.array([-2, -1, 1, 2]) * h
    gx = sum(ci * f(x0 + o, z0) for ci, o in zip(c, off))
    gz = sum(ci * f(x0, z0 + o) for ci, o in zip(c, off))
    return gx, gz


def run(label, st, lam, nc, mat, geom, pol, src, b, h=0.5):
    ctx = Ctx(st, lam, nc, mat, geom, pol, extra_pts=[src])
    g, nn, k = geom, geom.n_pts, ctx.k
    x0, z0 = src
    M = ctx.matrix()
    X, dz = g.f - x0, g.g - z0
    rho = np.hypot(X, dz)
    h1 = hankel1(1, k * rho)
    gx0, gz0 = 0.25j * k * h1 * X / rho, 0.25j * k * h1 * dz / rho
    e = np.exp(1j * (np.outer(g.f - x0, ctx.q) + np.outer(g.g + z0, ctx.a1)))
    gxr = (e * (-1j * ctx.q)) @ ctx.coef
    gzr = (e * (1j * ctx.a1)) @ ctx.coef
    rhs = np.zeros(2 * nn, dtype=complex)
    rhs[:nn] = -(b[0] * (gx0 + gxr) + b[1] * (gz0 + gzr))
    ei = np.linalg.solve(M, rhs)
    # source power
    self_r = (np.exp(2j * ctx.a1 * z0) * (ctx.q**2 * abs(b[0]) ** 2 - ctx.a1**2 * abs(b[1]) ** 2)) @ ctx.coef
    us = lambda x, z: ctx.scattered(ei, np.array([x]), np.array([z]))[0]
    dx, dzs = grad_fd(us, x0, z0, h)
    P_src = k * k * (abs(b[0]) ** 2 + abs(b[1]) ** 2) / 8 + self_r.imag + (-(np.conj(b[0]) * dx + np.conj(b[1]) * dzs)).imag
    # outputs
    old = ctx.spectra

    def spectra(q, ei_, src_=None):
        up, dn = old(q, ei_)
        a = S.a_phys_vec(k, q)
        up = up + 1j * (q * b[0] + a * b[1]) * np.exp(-1j * q * x0 - 1j * a * z0)
        dn = dn + 1j * (q * b[0] - a * b[1]) * np.exp(-1j * q * x0 + 1j * a * z0)
        return up, dn

    ctx.spectra = spectra
    P_up, P_dn = ctx.radiated(ei)
    P_g = 0.0
    for (bb, u, N), (cp, cm) in zip(ctx.modes, ctx.overlap(ei)):
        g1 = np.sqrt(bb * bb - k * k)
        u0 = u(np.array([z0]))[0]
        tp = cp + 1j / (2 * bb * N) * u0 * (1j * bb * b[0] + g1 * b[1]) * np.exp(-1j * bb * x0)
        tm = cm + 1j / (2 * bb * N) * u0 * (-1j * bb * b[0] + g1 * b[1]) * np.exp(1j * bb * x0)
        P_g += (bb * N).real * (abs(tp) ** 2 + abs(tm) ** 2)
    Pa = p_abs(ctx, ei)
    return P_src, P_up, P_dn, Pa, P_g


nc = 1.444
SOI = S.Stack([(3.476 / nc) ** 2], [220.0], 1.0)
circ = Geometry.gielis(rad=80.0, n_pts=160, m=4)
tri = Geometry.gielis(rad=90.0, n_pts=200, m=3, n1=4.0, n2=6.0, n3=6.0)
k = 2 * np.pi * nc / 1550
for pol in (1, 2):
    b0 = S.guided_poles_gamma(SOI, k, pol)[0]
    g1 = np.sqrt(b0 * b0 - k * k)
    for lab, geom, epsi, src in (("circle", shifted(circ, 110.0), 0.0, (150.0, 60.0)),
                                 ("lossy rotated triangle", shifted(tri, 140.0, 17.0), 0.3, (-170.0, 50.0))):
        mat = Material(n_core=3.476, n_clad=nc, pol=pol, epsi=epsi)
        for blab, b in (("(1,0)", (1.0, 0.0)), ("(0,1)", (0.0, 1.0)), ("spin-locked", (1.0, 1j * b0 / g1))):
            r = [run("", SOI, 1550, nc, mat, geom, pol, src, b, h) for h in (0.5, 0.25)]
            P, up, dn, ab, gd = r[1]
            e = [abs(x[0] - sum(x[1:])) / x[0] for x in r]
            print(f"  SOI {'TM' if pol == 1 else 'TE'} {lab:22s} b={blab:11s}: up {100 * up / P:5.1f} %  down {100 * dn / P:5.1f} %  "
                  f"abs {100 * ab / P:4.1f} %  guided {100 * gd / P:5.1f} %   |Δ|/P_src (FD h = 0.5 / 0.25 nm) = {e[0]:.0e} / {e[1]:.0e}")
