"""Route 1 for ordinary driven cluster solves: same answer as the full solve, cheaper per permittivity."""
import numpy as np, sys, time, warnings
from scipy.linalg import lu_factor, lu_solve
from perf import interior_rows, SLAB, nclad, lam, pol, k_bg
from pysie2d import Geometry, Material, Cluster, ClusterBIESolver
warnings.simplefilter("ignore")
P, nn, pitch = int(sys.argv[1]) if len(sys.argv) > 1 else 4, 100, 600.0
geoms = [Geometry.gielis(rad=90.0, n_pts=nn, m=3, n1=4.0, n2=6.0, n3=6.0, x0=pitch*(p-(P-1)/2), z0=140.0) for p in range(P)]
cs = ClusterBIESolver(Cluster(geoms), [Material.from_eps(-2.0*nclad**2, n_clad=nclad, pol=pol)]*P, pol=pol, background=SLAB)
full0 = cs._assemble(lam)
sl = [slice(2*nn*p, 2*nn*(p+1)) for p in range(P)]
rng = np.random.default_rng(0)
b = np.zeros(2*nn*P, complex)                       # a right side lives in the outside rows only
for p in range(P): b[sl[p].start:sl[p].start+nn] = rng.standard_normal(nn) + 1j*rng.standard_normal(nn)
# once per wavelength: compress what each particle receives from its neighbours
U, Y, rk = [], [], []
for p in range(P):
    D = full0[sl[p].start:sl[p].start+nn].copy(); D[:, sl[p]] = 0
    u, s, vh = np.linalg.svd(D, full_matrices=False); r = int(np.sum(s > 1e-13*s[0]))
    U.append(u[:, :r]); Y.append(u[:, :r].conj().T @ D); rk.append(r)
offs = np.cumsum([0] + rk)
top1 = full0[:nn, :2*nn]
def full_solve(e):
    m = full0.copy()
    for p in range(P): m[sl[p].start+nn:sl[p].stop, sl[p]] = interior_rows(pol, geoms[p], k_bg, np.sqrt(e), e)
    return np.linalg.solve(m, b)
def reduced_solve(e):
    lu = lu_factor(np.vstack([top1, interior_rows(pol, geoms[0], k_bg, np.sqrt(e), e)]))   # one small particle matrix
    a = [lu_solve(lu, np.concatenate([b[sl[p].start:sl[p].start+nn], np.zeros(nn)])) for p in range(P)]   # alone
    R = [lu_solve(lu, np.vstack([U[q], np.zeros((nn, rk[q]))])) for q in range(P)]           # reply to neighbours
    G, rhs = np.eye(offs[-1], dtype=complex), np.zeros(offs[-1], complex)
    for p in range(P):
        for q in range(P):
            G[offs[p]:offs[p+1], offs[q]:offs[q+1]] += Y[p][:, sl[q]] @ R[q]
            rhs[offs[p]:offs[p+1]] += Y[p][:, sl[q]] @ a[q]
    y = np.linalg.solve(G, rhs)                                                             # small coupled problem
    return np.concatenate([a[p] - R[p] @ y[offs[p]:offs[p+1]] for p in range(P)])
eps_list = np.linspace(-4.0, 6.0, 40) + 0.05j                                                # metals to dielectrics
t0 = time.perf_counter(); xf = [full_solve(e) for e in eps_list]; tf = time.perf_counter() - t0
t0 = time.perf_counter(); xr = [reduced_solve(e) for e in eps_list]; tr = time.perf_counter() - t0
err = max(np.linalg.norm(a-c)/np.linalg.norm(a) for a, c in zip(xf, xr))
print(f"{P} particles: {len(eps_list)} permittivities, worst relative difference {err:.1e}; "
      f"full {tf/len(eps_list)*1e3:.0f} ms each, reduced {tr/len(eps_list)*1e3:.0f} ms each "
      f"(small problem {offs[-1]} vs {2*nn*P})")
