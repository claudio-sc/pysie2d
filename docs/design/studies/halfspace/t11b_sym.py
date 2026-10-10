"""End-to-end: circle over PEC half-space via the Sommerfeld path (R == -+1),
vs the exact two-cylinder image problem solved with the cluster primitives."""
import numpy as np, warnings
from hs import *
from gemm import reflected_blocks
import sys; sys.path.insert(0, "/Users/claudiosc/Documents/sie/src")
from pysie2d import Geometry, BIESolver, Cluster, ClusterBIESolver
from pysie2d.material import Material
from pysie2d.sources import line_dipole_rhs
from pysie2d.fields import _representation_at
warnings.simplefilter("ignore")

def run(pol, lam, nn, s, Rp=100., gap=10., n_clad=1.0, z_int=0.0):
    mat = Material(n_core=2.0, n_clad=n_clad, pol=pol, epsi=0.0)  # epsi >= 0 (Material.nc |eps| bug avoided)
    k_bg = mat.wnum_bg(lam); k0 = 2 * PI / lam; eps1 = n_clad ** 2
    assert abs(k0 * np.sqrt(eps1) - k_bg) < 1e-15 * abs(k_bg)   # single conversion point
    zc = z_int + Rp + gap
    geo = Geometry.gielis(Rp, nn, m=0, x0=0.0, z0=zc)
    xs, zs = 60.0, zc + Rp + 80.0
    sign = -1.0 if pol == 2 else 1.0
    # --- half-space solve (Sommerfeld reflected blocks, exterior rows only)
    M = BIESolver(geo, mat).assemble(lam)
    m1, m2, Mq, _ = reflected_blocks(k0, eps1, 'pec', pol, geo.f, geo.g, geo.df, geo.dg, z_int, s=s)
    M[:nn, :nn] += m1; M[:nn, nn:] += m2
    q, wq, _ = make_path(k0, eps1, 'pec', pol, zs + geo.g.min() - 2 * z_int, np.ptp(np.r_[geo.f, xs]))
    Gi = g_ind(geo.f - xs, geo.g + zs - 2 * z_int, k0, eps1, 'pec', pol, q, wq)[0]
    rhs = line_dipole_rhs(nn, k_bg, geo.f, geo.g, xs, zs); rhs[:nn] += Gi
    ei = np.linalg.solve(M, rhs)
    # --- image cluster
    img = Geometry.gielis(Rp, nn, m=0, x0=0.0, z0=2 * z_int - zc)
    cs = ClusterBIESolver(Cluster([geo, img]), [mat, mat], pol=pol)
    A = cs._assemble(lam)
    r = np.zeros(4 * nn, complex)
    for p, gp in enumerate((geo, img)):
        r[2*nn*p:2*nn*(p+1)] = line_dipole_rhs(nn, k_bg, gp.f, gp.g, xs, zs) + sign * line_dipole_rhs(nn, k_bg, gp.f, gp.g, xs, 2*z_int - zs)
    ec = np.linalg.solve(A, r)
    # --- fields at observation points in the cover
    xo = np.array([-400., 0., 250., 500.]); zo = np.array([60., zc + 2.5 * Rp, 40., 700.]) + z_int
    fc = sum(_representation_at(ec[2*nn*p:2*nn*(p+1)], nn, gp.f, gp.df, gp.g, gp.dg, gp.delt, k_bg, xo, zo) for p, gp in enumerate((geo, img)))
    fh = _representation_at(ei, nn, geo.f, geo.df, geo.g, geo.dg, geo.delt, k_bg, xo, zo)
    qo, wqo, _ = make_path(k0, eps1, 'pec', pol, zo.min() + geo.g.min() - 2*z_int, 1500.)
    for i in range(len(xo)):
        G, Gx, Gz = g_ind(xo[i] - geo.f, zo[i] + geo.g - 2*z_int, k0, eps1, 'pec', pol, qo, wqo)
        fh[i] += -np.sum(geo.delt * ((Gx * geo.dg - Gz * geo.df) * ei[:nn] + G * ei[nn:]))
    e_phi = np.max(abs(ei[:nn] - ec[:nn])) / np.max(abs(ec[:nn]))
    e_chi = np.max(abs(ei[nn:2*nn] - ec[nn:2*nn])) / np.max(abs(ec[nn:2*nn]))
    perm = np.argmin(np.hypot(img.f[None,:]-geo.f[:,None], img.g[None,:]-(2*z_int-geo.g)[:,None]), axis=1)
    e_img = np.max(abs(ec[2*nn:3*nn][perm] - sign*ec[:nn])) / np.max(abs(ec[:nn]))  # mirror symmetry of cluster sol
    return Mq, e_phi, e_chi, np.max(abs(fh - fc)) / np.max(abs(fc)), e_img

for pol in (2, 1):
    for lam, nc in [(633., 1.0), (633.*(1+.05j), 1.0), (800., 1.33)][:2]:
        for gap in (10., 40.):
            for nn in (64,):
                row = []
                for s in (1.0,):
                    Mq, a, b, c, d = run(pol, lam, nn, s, gap=gap, n_clad=nc)
                    row.append(f"s={s}: M={Mq} phi {a:.0e} chi {b:.0e} field {c:.0e}")
                print(f"pol={pol} lam={lam:.0f} n_clad={nc} gap={gap:.0f} nn={nn} (mirror-sym of cluster {d:.0e}) | " + " | ".join(row))
