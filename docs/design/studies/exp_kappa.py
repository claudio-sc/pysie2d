import sys
sys.path.insert(0, "/home/claude/pysie2d-0.5.0/src")
import numpy as np
from pysie2d import BIESolver, Geometry, Material

np.set_printoptions(precision=3, suppress=False)


def equilibrate(M, iters=30):
    """Sinkhorn-Knopp style row/col equilibration in inf-norm (van der Sluis)."""
    A = M.copy()
    r = np.ones(A.shape[0])
    c = np.ones(A.shape[1])
    for _ in range(iters):
        rn = np.abs(A).max(axis=1)
        A = A / rn[:, None]
        r = r / rn
        cn = np.abs(A).max(axis=0)
        A = A / cn[None, :]
        c = c / cn
    return A, r, c


def block_scaled_kappa(M, nn):
    """Best kappa over a single scalar scaling of the chi (second) block."""
    best = np.inf
    besta = None
    for a in np.logspace(-4, 4, 81):
        S = np.ones(2 * nn)
        S[nn:] = a
        k = np.linalg.cond(M * S[None, :])
        if k < best:
            best, besta = k, a
    return best, besta


def report(tag, geom, mat, lam):
    nn = len(geom.f)
    M = BIESolver(geom, mat).assemble(lam)
    k_raw = np.linalg.cond(M)
    Me, _, _ = equilibrate(M)
    k_eq = np.linalg.cond(Me)
    k_blk, a_blk = block_scaled_kappa(M, nn)
    print(f"{tag:34s} nn={nn:4d}  kappa={k_raw:10.3e}  "
          f"equil={k_eq:10.3e}  blockscaled={k_blk:10.3e} (a={a_blk:.3g})")
    return k_raw, k_eq, k_blk


mat = Material(n_core=1.5, n_clad=1.0, pol=2)
lam = 600.0

print("=== CIRCLE  rad=200nm, n_core=1.5, TE, lambda=600nm ===")
for nn in (50, 100, 200, 400):
    geom = Geometry.gielis(rad=200, n_pts=nn, m=0)
    report("circle", geom, mat, lam)

print()
print("=== GIELIS m=6 STAR (n1=6,n2=12,n3=12) ===")
for nn in (100, 200, 400):
    geom = Geometry.gielis(rad=200, n_pts=nn, m=6, n1=6, n2=12, n3=12)
    report("star, uniform arc length", geom, mat, lam)

print()
print("=== SAME STAR, uniform-theta nodes instead of uniform arc length ===")
from pysie2d.geometry import boundary_setup
for nn in (100, 200, 400):
    th = np.linspace(0, 2 * np.pi, nn, endpoint=False) + np.pi / nn
    geom = Geometry.gielis(rad=200, n_pts=nn, m=6, n1=6, n2=12, n3=12, theta=th)
    report("star, uniform theta", geom, mat, lam)
