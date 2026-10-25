"""S12: a particle over the stack, wired as v1.2 will wire it. Pins the overlap formula.

The package's free-space assembly (BIESolver.assemble with no background) plus the
stack's reflected blocks on the exterior rows, in the M1/M2 convention of
conventions §15 (M1 = h(∂x'G dg − ∂z'G df), M2 = hG, u_s = −(M1φ + M2χ)). Applying
S9's guided pole to u_s gives the mode amplitudes, referred to u_m(z)e^{±iβ_m x}:

  c_m^± = −(i/(2β_m N_m)) Σ_j h u_m(z_j) e^{∓iβ_m x_j} [(∓iβ_m dg_j + γ₁ df_j) φ_j + χ_j]

Checks (each independent of the formula's derivation):
  (a) far-x scattered field (representation formula on the path) → Σ c_m^± u_m e^{±iβx}
  (b) dipole + particle: P_in = 1/4 + Im G_refl + Im u_s at the source equals
      P_up + P_down + P_abs + Σ βN(|c⁺|² + |c⁻|²), each from its own integral
  (c) mode incidence: Σ|S|² + P_up + P_down + P_abs = 1 (power-normalised modes)
  (d) reciprocity on an x-asymmetric particle: t_{n←m}(L→R) = t_{m←n}(R→L)
"""
import numpy as np
from scipy.integrate import quad

import sproto as S
from pysie2d import BIESolver, Geometry, Material
from pysie2d.fields import eval_field
from pysie2d.sources import line_dipole_rhs


def shifted(geom, dz, rot_deg=0.0):
    """Rotate about the origin, then lift by dz: an x-asymmetric shape for (d)."""
    c, s = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
    rot = lambda x, z: (c * x - s * z, s * x + c * z)
    f, g = rot(geom.f, geom.g)
    df, dg = rot(geom.df, geom.dg)
    ddf, ddg = rot(geom.ddf, geom.ddg)
    return Geometry(f=f, g=g + dz, df=df, dg=dg, ddf=ddf, ddg=ddg, rad=geom.rad, x0=0.0, z0=dz)


class Ctx:
    """Stack + one particle at one wavelength: path, reflected blocks, modes."""

    def __init__(self, st, lam, n_clad, mat, geom, pol, extra_pts=()):
        self.st, self.lam, self.mat, self.geom, self.pol = st, lam, mat, geom, pol
        self.k = 2 * np.pi * n_clad / lam
        self.poles = S.guided_poles_gamma(st, self.k, pol)
        self.modes = [(b,) + S.mode_profile(st, self.k, pol, b)[:2] for b in self.poles]
        xs = np.concatenate([geom.f] + [[p[0]] for p in extra_pts])
        zs = np.concatenate([geom.g] + [[p[1]] for p in extra_pts])
        self.z_min = 2 * zs.min()
        self.D = max(np.ptp(xs), self.z_min)
        self.x_c = 0.5 * (xs.max() + xs.min())
        p = S.build_path(self.k, S.stack_features(st, self.k, pol, self.poles), self.D, self.z_min)
        q = np.concatenate([p["q"], -p["q"]])
        w = np.concatenate([p["w"], p["w"]])
        self.q, self.w = q, w
        self.a1 = S.alpha(q, self.k)
        self.coef = (1j / (4 * np.pi)) * w * st.R(q, self.k, pol) / self.a1
        self.h = 2 * np.pi / geom.n_pts

    def blocks(self, xt, zt):
        """Reflected M1, M2 from the particle's nodes to points (xt, zt)."""
        g, q, a1 = self.geom, self.q, self.a1
        e_t = np.exp(1j * (np.outer(xt - self.x_c, q) + np.outer(zt, a1)))
        e_s = np.exp(1j * (-np.outer(g.f - self.x_c, q) + np.outer(g.g, a1)))
        poly = -1j * (np.outer(g.dg, q) + np.outer(g.df, a1))
        left = e_t * self.coef
        return self.h * (left @ (e_s * poly).T), self.h * (left @ e_s.T)

    def g_refl(self, xt, zt, x0, z0):
        """Reflected G at points from a point source (same path, same sum)."""
        e = np.exp(1j * (np.outer(xt - x0, self.q) + np.outer(zt + z0, self.a1)))
        return e @ self.coef

    def matrix(self):
        g = self.geom
        M = BIESolver(g, self.mat).assemble(self.lam)
        m1, m2 = self.blocks(g.f, g.g)
        M[: g.n_pts, : g.n_pts] += m1
        M[: g.n_pts, g.n_pts:] += m2
        return M

    def scattered(self, ei, x, z):
        g = self.geom
        u = eval_field(ei, g.n_pts, g.f, g.df, g.g, g.dg, g.delt, self.k, x, z)
        m1, m2 = self.blocks(x, z)
        return u - (m1 @ ei[: g.n_pts] + m2 @ ei[g.n_pts:])

    def overlap(self, ei):
        """c_m^± from the boundary unknowns (the formula under test), per mode."""
        g, nn = self.geom, self.geom.n_pts
        phi, chi = ei[:nn], ei[nn:]
        out = []
        for b, u, N in self.modes:
            g1 = np.sqrt(b * b - self.k**2)
            uz = u(g.g)
            cp = -1j / (2 * b * N) * self.h * np.sum(uz * np.exp(-1j * b * g.f) * ((-1j * b * g.dg + g1 * g.df) * phi + chi))
            cm = -1j / (2 * b * N) * self.h * np.sum(uz * np.exp(1j * b * g.f) * ((1j * b * g.dg + g1 * g.df) * phi + chi))
            out.append((cp, cm))
        return out

    def spectra(self, q, ei, src=None):
        """(u_up, d) of the total field: U = (i/2α)(u_up + R d), down at z_top = (i/2α)d."""
        g, nn = self.geom, self.geom.n_pts
        a = S.a_phys_vec(self.k, q)
        phi, chi = ei[:nn], ei[nn:]
        eu = np.exp(-1j * np.outer(q, g.f) - 1j * np.outer(a, g.g))
        ed = np.exp(-1j * np.outer(q, g.f) + 1j * np.outer(a, g.g))
        up = -self.h * np.sum(eu * ((-1j * np.outer(q, g.dg) + 1j * np.outer(a, g.df)) * phi + chi), axis=1)
        dn = -self.h * np.sum(ed * ((-1j * np.outer(q, g.dg) - 1j * np.outer(a, g.df)) * phi + chi), axis=1)
        if src is not None:
            x0, z0 = src
            up = up + np.exp(-1j * q * x0 - 1j * a * z0)
            dn = dn + np.exp(-1j * q * x0 + 1j * a * z0)
        return up, dn

    def radiated(self, ei, src=None):
        """P_up, P_down by real-axis quadrature of the spectra (lossless stack)."""
        k, st, pol = self.k, self.st, self.pol
        _, kb = st.ks(k)
        kb = float(kb.real)
        wb = 1.0 if pol == 2 else st.eps_bot.real
        th = np.linspace(-np.pi / 2, np.pi / 2, 4001)[1:-1]
        x, wx = np.polynomial.legendre.leggauss(400)

        def up_int(t):
            q = k * np.sin(t)
            u, d = self.spectra(q, ei, src)
            R = st.R(q + 0j, k, pol)
            return np.abs(u + R * d) ** 2 / 4

        def dn_int(q):
            u, d = self.spectra(q, ei, src)
            a1 = S.a_phys_vec(k, q)
            ab = np.sqrt(np.maximum(kb * kb - q * q, 0.0))
            return (ab / wb) * np.abs(st.tau(q + 0j, k, pol) * d) ** 2 / 4

        def gl(fn, a, b, n=8):
            edges = np.linspace(a, b, n + 1)
            return sum(np.sum(wx * fn(0.5 * (e1 - e0) * x + 0.5 * (e1 + e0))) * 0.5 * (e1 - e0)
                       for e0, e1 in zip(edges[:-1], edges[1:]))

        # sine map on each panel absorbs the √ endpoints
        P_up = gl(lambda s: up_int(s), -np.pi / 2, np.pi / 2, 16) / (2 * np.pi)
        sm = lambda s: kb * np.sin(s)
        P_dn = gl(lambda s: dn_int(sm(s)) * kb * np.cos(s), -np.pi / 2, np.pi / 2, 16) / (2 * np.pi)
        return P_up, P_dn


S.a_phys_vec = lambda kk, q: np.where(np.abs(q) <= kk, np.sqrt(np.maximum(kk * kk - q * q, 0) + 0j),
                                      1j * np.sqrt(np.maximum(q * q - kk * kk, 0) + 0j))


def p_abs(ctx, ei):
    nn = ctx.geom.n_pts
    return float(-np.imag(ctx.h * np.sum(np.conj(ei[:nn]) * ei[nn:])))


def run(label, st, lam, nc, mat, geom, pol, src):
    ctx = Ctx(st, lam, nc, mat, geom, pol, extra_pts=[src])
    g, nn = geom, geom.n_pts
    M = ctx.matrix()
    x0, z0 = src
    rhs = line_dipole_rhs(nn, ctx.k, g.f, g.g, x0, z0)
    rhs[:nn] += ctx.g_refl(g.f, g.g, x0, z0)
    ei = np.linalg.solve(M, rhs)
    # (b) dipole power balance
    P_in = 0.25 + ctx.g_refl(np.array([x0]), np.array([z0]), x0, z0)[0].imag + ctx.scattered(ei, np.array([x0]), np.array([z0]))[0].imag
    P_up, P_dn = ctx.radiated(ei, src)
    c_sc = ctx.overlap(ei)
    P_g = 0.0
    c_tot = []
    for (b, u, N), (cp, cm) in zip(ctx.modes, c_sc):
        g1 = np.sqrt(b * b - ctx.k**2)
        cd = 1j / (2 * b * N) * u(np.array([z0]))[0]  # dipole's own: (i/2βN) u(z0) e^{∓iβx0}
        tp, tm = cp + cd * np.exp(-1j * b * x0), cm + cd * np.exp(1j * b * x0)
        c_tot.append((tp, tm))
        P_g += (b * N).real * (abs(tp) ** 2 + abs(tm) ** 2)
    Pa = p_abs(ctx, ei)
    bal = abs(P_in - (P_up + P_dn + Pa + P_g)) / P_in
    # (a) far-x scattered field vs overlap amplitudes
    rows = []
    for X in (2e4, 4e4):
        for sgn in (1, -1):
            xx = np.array([sgn * X + ctx.x_c])
            zz = np.array([100.0])
            # a path sized for the far points too
            far = Ctx(st, lam, nc, mat, geom, pol, extra_pts=[src, (xx[0], zz[0])])
            us = far.scattered(ei, xx, zz)[0]
            pred = sum((cp if sgn > 0 else cm) * u(zz)[0] * np.exp(1j * sgn * b * xx[0])
                       for (b, u, N), (cp, cm) in zip(ctx.modes, c_sc))
            scale = sum(abs(cp * u(zz)[0]) + abs(cm * u(zz)[0]) for (b, u, N), (cp, cm) in zip(ctx.modes, c_sc))
            rows.append(f"{'+' if sgn > 0 else '−'}{X / 1e3:.0f}µm {abs(us - pred) / scale:.1e}")
    print(f"  {label}: P_in={P_in:.6f}  up {P_up:.4f} down {P_dn:.4f} abs {Pa:.4f} guided {P_g:.4f}  "
          f"|Δ|/P_in = {bal:.1e}   far-x |u_s − Σc u e^(iβx)|/Σ|c u|: " + "  ".join(rows))
    return ctx, M


def smatrix(ctx, M, x_ref=0.0):
    """S for each incident mode/direction, power-normalised, phase referred to x_ref."""
    g, nn = ctx.geom, ctx.geom.n_pts
    lu = np.linalg.solve
    out = {}
    for mi, (b, u, N) in enumerate(ctx.modes):
        a = 1.0 / np.sqrt((b * N).real)
        for d in (+1, -1):
            rhs = np.zeros(2 * nn, dtype=complex)
            rhs[:nn] = a * u(g.g) * np.exp(1j * d * b * (g.f - x_ref))
            ei = lu(M, rhs)
            c = ctx.overlap(ei)
            for ni, ((bn, un, Nn), (cp, cm)) in enumerate(zip(ctx.modes, c)):
                s = np.sqrt((bn * Nn).real)
                fwd = (cp if d > 0 else cm) * s * np.exp(1j * d * bn * x_ref)
                bwd = (cm if d > 0 else cp) * s * np.exp(-1j * d * bn * x_ref)
                out[(ni, mi, d, "t")] = fwd + (1.0 if ni == mi else 0.0)
                out[(ni, mi, d, "r")] = bwd
            P_up, P_dn = ctx.radiated(ei)
            out[(mi, d, "rad")] = (P_up, P_dn, p_abs(ctx, ei))
    return out


if __name__ == "__main__":
    nc = 1.444
    SOI = S.Stack([(3.476 / nc) ** 2], [220.0], 1.0)
    SIN = S.Stack([(2.0 / nc) ** 2], [400.0], 1.0)
    circ = Geometry.gielis(rad=80.0, n_pts=160, m=4, n1=2.0, n2=2.0, n3=2.0)
    tri = Geometry.gielis(rad=90.0, n_pts=200, m=3, n1=4.0, n2=6.0, n3=6.0)
    cases = [
        ("SOI, Si circle r80 gap30", SOI, 1550, shifted(circ, 110.0), 3.476, 0.0, (150.0, 60.0)),
        ("SOI, lossy circle (epsi 0.8)", SOI, 1550, shifted(circ, 110.0), 3.476, 0.8, (150.0, 60.0)),
        ("SOI, rotated triangle", SOI, 1550, shifted(tri, 140.0, 17.0), 2.0, 0.3, (-170.0, 50.0)),
        ("SiN@800 2 modes, triangle", SIN, 800, shifted(tri, 140.0, 17.0), 3.476, 0.0, (-170.0, 50.0)),
    ]
    print("(a), (b) dipole + particle")
    keep = []
    for label, st, lam, geom, n, epsi, src in cases:
        for pol in (2, 1):
            mat = Material(n_core=n, n_clad=nc, pol=pol, epsi=epsi)
            ctx, M = run(f"{label} {'TE' if pol == 2 else 'TM'}", st, lam, nc, mat, geom, pol, src)
            keep.append((label, pol, ctx, M))
    print("\n(c), (d) mode incidence: energy and reciprocity")
    for label, pol, ctx, M in keep:
        S_ = smatrix(ctx, M, x_ref=37.0)
        nm = len(ctx.modes)
        worst_e = 0.0
        for mi in range(nm):
            for d in (1, -1):
                tot = sum(abs(S_[(ni, mi, d, "t")]) ** 2 + abs(S_[(ni, mi, d, "r")]) ** 2 for ni in range(nm))
                tot += sum(S_[(mi, d, "rad")])
                worst_e = max(worst_e, abs(tot - 1))
        worst_r = max(abs(S_[(ni, mi, 1, "t")] - S_[(mi, ni, -1, "t")]) for ni in range(nm) for mi in range(nm))
        refl_lr = max(abs(S_[(ni, mi, 1, "r")] - S_[(mi, ni, 1, "r")]) for ni in range(nm) for mi in range(nm))
        t00 = S_[(0, 0, 1, "t")]
        print(f"  {label} {'TE' if pol == 2 else 'TM'}: |Σ|S|²+P_rad+P_abs − 1| ≤ {worst_e:.1e}   "
              f"|t_nm(L→R) − t_mn(R→L)| ≤ {worst_r:.1e}   |r_nm − r_mn| (left) ≤ {refl_lr:.1e}   "
              f"|t₀₀|² = {abs(t00) ** 2:.5f}")
