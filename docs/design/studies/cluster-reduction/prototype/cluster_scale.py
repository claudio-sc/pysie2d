import numpy as np, sys, time, warnings
sys.path.insert(0, ".")
from scipy.linalg import lu_factor, lu_solve
from perf import interior_rows, SLAB, nclad, lam, pol, k_bg
from pysie2d import Geometry, Material, BIESolver, Cluster, ClusterBIESolver
warnings.simplefilter("ignore")
ri = lambda e: 1j*np.sqrt(-complex(e))
nn, z0, pitch, e = 100, 140.0, 600.0, -2.0-0.3j
def tmin(fn, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return min(ts)
print(" P  | full size | full point | reduced size | reduced point | one-off compression | memory full / reduced (MB)")
for P in (4, 8, 16):
    geoms = [Geometry.gielis(rad=90.0, n_pts=nn, m=3, n1=4.0, n2=6.0, n3=6.0, x0=pitch*(p-(P-1)/2), z0=z0) for p in range(P)]
    mat = Material.from_eps(-2.0*nclad**2, n_clad=nclad, pol=pol)
    full0 = ClusterBIESolver(Cluster(geoms), [mat]*P, pol=pol, background=SLAB)._assemble(lam)
    sl = [slice(2*nn*p, 2*nn*(p+1)) for p in range(P)]
    def Mfull(ev):
        m = full0.copy()
        for p in range(P):
            m[sl[p].start+nn:sl[p].stop, sl[p]] = interior_rows(pol, geoms[p], k_bg, ri(ev), ev)
        return m
    top1 = full0[:nn, :2*nn]
    t0 = time.perf_counter(); U, Y, rk = [], [], []
    for p in range(P):
        D = full0[sl[p].start:sl[p].start+nn, :].copy(); D[:, sl[p]] = 0
        u, s, vh = np.linalg.svd(D, full_matrices=False); r = int(np.sum(s > 1e-13*s[0]))
        U.append(u[:, :r]); Y.append(u[:, :r].conj().T @ D); rk.append(r)
    t_comp = time.perf_counter()-t0
    offs = np.cumsum([0]+rk); R0 = offs[-1]
    def Gred(ev):
        lu = lu_factor(np.vstack([top1, interior_rows(pol, geoms[0], k_bg, ri(ev), ev)]))
        out = np.eye(R0, dtype=complex)
        for q in range(P):
            Rq = lu_solve(lu, np.vstack([U[q], np.zeros((nn, rk[q]))]))
            for p in range(P):
                out[offs[p]:offs[p+1], offs[q]:offs[q+1]] += Y[p][:, sl[q]] @ Rq
        return lu_factor(out)
    tf = tmin(lambda: lu_factor(Mfull(e)), 2); tr = tmin(lambda: Gred(e))
    print(f"{P:3d} | {2*nn*P:9d} | {tf*1e3:8.0f} ms | {R0:12d} | {tr*1e3:11.0f} ms | {t_comp:15.2f} s   | {16*(2*nn*P)**2/1e6:7.1f} / {16*R0**2/1e6:.1f}")
