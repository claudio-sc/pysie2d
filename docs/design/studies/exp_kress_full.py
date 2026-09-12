import sys
sys.path.insert(0, "/home/claude/pysie2d-0.5.0/src")
import numpy as np
from scipy.special import jv
from pysie2d import BIESolver, Geometry, Material, solver as smod
from pysie2d.reference import mie
from pysie2d.kernels import hank0, hank1

RAD, NCORE, LAM = 200.0, 1.5, 600.0
TWOPI, GE = 2 * np.pi, np.euler_gamma
mat = Material(n_core=NCORE, n_clad=1.0, pol=2)
QM = mie.efficiencies(TWOPI * RAD / LAM, complex(NCORE))["Q_ext_TE"]


def kress_R(N):
    n = N // 2
    d = TWOPI * np.arange(N) / N
    p = np.arange(1, n)
    return -(TWOPI / n) * ((np.cos(np.outer(d, p)) / p).sum(1) + np.cos(n * d) / (2 * n))


def demote(k):
    kc = complex(k)
    return kc.real if kc.imag == 0.0 else kc


def kress_matrix(geom, k_bg, k_co, N):
    f, g, th = geom.f, geom.g, geom.theta
    df, dg = geom.df, geom.dg
    gam = np.sqrt(df**2 + dg**2)
    deriv = df * geom.ddg - geom.ddf * dg
    dx, dz = f[:, None] - f[None, :], g[:, None] - g[None, :]
    r = np.sqrt(dx**2 + dz**2)
    np.fill_diagonal(r, 1.0)
    arg2 = dx * dg[None, :] - dz * df[None, :]

    s2 = 4.0 * np.sin((th[:, None] - th[None, :]) / 2.0) ** 2
    np.fill_diagonal(s2, 1.0)
    L = np.log(s2)
    R = kress_R(N)[(np.arange(N)[:, None] - np.arange(N)[None, :]) % N]
    h = TWOPI / N

    def single(k):
        k = demote(k)
        A1 = -jv(0, k * r) / (4.0 * np.pi)
        A2 = 0.25j * hank0(k * r) - A1 * L
        np.fill_diagonal(A1, -1.0 / (4.0 * np.pi))
        np.fill_diagonal(A2, 0.25j - (GE + np.log(k * gam / 2.0)) / (2.0 * np.pi))
        return (A1 * R + A2 * h).astype(complex)

    def double(k):
        k = demote(k)
        B1 = -(k / (4.0 * np.pi)) * arg2 * jv(1, k * r) / r
        D = 0.25j * k**2 * arg2 * hank1(k * r) / (k * r)
        B2 = D - B1 * L
        np.fill_diagonal(B1, 0.0)
        np.fill_diagonal(B2, -deriv / (4.0 * np.pi * gam**2))
        return (B1 * R + B2 * h).astype(complex)

    M = np.zeros((2 * N, 2 * N), dtype=complex)
    M[:N, :N] = double(k_bg) + 0.5 * np.eye(N)
    M[:N, N:] = single(k_bg)
    M[N:, :N] = double(k_co) - 0.5 * np.eye(N)
    M[N:, N:] = single(k_co)
    return M


def qext(geom, M):
    s = BIESolver(geom, mat)
    base = s.scatter(wavelength=LAM)
    rhs = s.assemble(LAM) @ base.ei
    r = smod.ScatterResult(np.linalg.solve(M, rhs), geom, mat, LAM, base.angle)
    amp, _ = r.far_field(4000)
    d = TWOPI / 3999.0
    return amp[int((TWOPI - np.deg2rad(r.angle)) / d)].imag / (r.wnum_bg * 2 * RAD)


print("Circle, rad=200nm, n_core=1.5, TE, 600nm.  qext relative error vs Mie\n")
print(f"{'nn':>6} {'as shipped':>13} {'rate':>6} {'Kress, all 4 blocks':>21} {'rate':>6}")
pa = pb = None
for nn in (30, 60, 120, 240):
    geom = Geometry.gielis(rad=RAD, n_pts=nn, m=0)
    s = BIESolver(geom, mat)
    k_bg = mat.wnum_bg(LAM)
    ea = abs(qext(geom, s.assemble(LAM)) - QM) / QM
    eb = abs(qext(geom, kress_matrix(geom, k_bg, mat.nc * k_bg, nn)) - QM) / QM
    print(f"{nn:6d} {ea:13.3e} {np.log2(pa/ea) if pa else float('nan'):6.2f}"
          f" {eb:21.3e} {np.log2(pb/eb) if pb else float('nan'):6.2f}")
    pa, pb = ea, eb
