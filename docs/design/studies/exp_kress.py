import sys
sys.path.insert(0, "/home/claude/pysie2d-0.5.0/src")
import numpy as np
from pysie2d import BIESolver, Geometry, Material
from pysie2d.reference import mie
from pysie2d.kernels import hank0

RAD, NCORE, LAM = 200.0, 1.5, 600.0
TWOPI = 2 * np.pi
GAMMA_E = np.euler_gamma
mat = Material(n_core=NCORE, n_clad=1.0, pol=2)
x_size = TWOPI * RAD / LAM
QM = mie.efficiencies(x_size, complex(NCORE))["Q_ext_TE"]


def kress_R(N):
    """R_m for m = 0..N-1, the log-weight circulant. N must be even, N = 2n."""
    n = N // 2
    m = np.arange(N)
    d = TWOPI * m / N                      # t_i - t_j
    p = np.arange(1, n)
    S = (np.cos(np.outer(d, p)) / p).sum(axis=1)
    return -(TWOPI / n) * (S + np.cos(n * d) / (2 * n))


def single_layer_block(geom, k, N):
    """Kress-split single-layer block: 0.25i * H0(k r) integrated in theta."""
    f, g, th = geom.f, geom.g, geom.theta
    gam = np.sqrt(geom.df**2 + geom.dg**2)
    dx = f[:, None] - f[None, :]
    dz = g[:, None] - g[None, :]
    r = np.sqrt(dx**2 + dz**2)
    np.fill_diagonal(r, 1.0)               # placeholder, diagonal handled below

    dt = th[:, None] - th[None, :]
    s2 = 4.0 * np.sin(dt / 2.0) ** 2
    np.fill_diagonal(s2, 1.0)
    L = np.log(s2)

    from scipy.special import jv
    kc = complex(k)
    kk = kc.real if kc.imag == 0.0 else kc
    A1 = -jv(0, kk * r) / (4.0 * np.pi)
    A2 = 0.25j * hank0(kk * r) - A1 * L

    np.fill_diagonal(A1, -1.0 / (4.0 * np.pi))
    np.fill_diagonal(
        A2, 0.25j - (GAMMA_E + np.log(kk * gam / 2.0)) / (2.0 * np.pi)
    )

    R = kress_R(N)
    idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
    h = TWOPI / N
    return (A1 * R[idx] + A2 * h).astype(complex)


def qext_from_matrix(geom, M):
    solver = BIESolver(geom, mat)
    base = solver.scatter(wavelength=LAM)          # for the rhs and plumbing
    from pysie2d import solver as smod
    rhs = solver.assemble(LAM) @ base.ei   # recover rhs
    ei = np.linalg.solve(M, rhs)
    r = smod.ScatterResult(ei, geom, mat, LAM, base.angle)
    amp, _ = r.far_field(4000)
    d = TWOPI / 3999.0
    return amp[int((TWOPI - np.deg2rad(r.angle)) / d)].imag / (r.wnum_bg * 2 * RAD)


print("Circle, rad=200nm, n_core=1.5, TE, 600nm.  qext relative error vs Mie\n")
print(f"{'nn':>6} {'as shipped':>13} {'rate':>6} {'Kress on M2/M4':>16} {'rate':>6}")
prev_a = prev_b = None
for nn in (60, 120, 240, 480):
    geom = Geometry.gielis(rad=RAD, n_pts=nn, m=0)
    solver = BIESolver(geom, mat)
    M0 = solver.assemble(LAM)
    k_bg = mat.wnum_bg(LAM)
    k_co = mat.nc * k_bg

    M1 = M0.copy()
    M1[:nn, nn:] = single_layer_block(geom, k_bg, nn)
    M1[nn:, nn:] = single_layer_block(geom, k_co, nn)      # pol=2 -> eta = 1

    ea = abs(qext_from_matrix(geom, M0) - QM) / QM
    eb = abs(qext_from_matrix(geom, M1) - QM) / QM
    ra = np.log2(prev_a / ea) if prev_a else np.nan
    rb = np.log2(prev_b / eb) if prev_b else np.nan
    print(f"{nn:6d} {ea:13.3e} {ra:6.2f} {eb:16.3e} {rb:6.2f}")
    prev_a, prev_b = ea, eb
