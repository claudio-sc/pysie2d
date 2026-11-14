"""Cluster eigenpermittivities: full matrix vs re-expansion routes built on single-particle quantities.

Cluster matrix at fixed wavelength, particles p = 1..P, unknowns x_p = [phi_p; chi_p]:
    M_p(eps) x_p + [ sum_{q != p} Delta_pq x_q ; 0 ] = 0
M_p = isolated particle over the slab (its own slab reflection included); Delta_pq = what particle q's
boundary values send to particle p's outside rows (free space + slab). Delta does not depend on eps.
"""
import numpy as np, sys, time, warnings
sys.path.insert(0, ".")
from scipy.linalg import lu_factor, lu_solve, eig
from perf import interior_rows, SLAB, nclad, lam, pol, k_bg
from pysie2d import Geometry, Material, BIESolver, Cluster, ClusterBIESolver
from pysie2d.beyn import beyn_modes, newton_refine
warnings.simplefilter("ignore")
ri = lambda e: 1j*np.sqrt(-complex(e))
P, nn, z0 = 4, 100, 140.0
pitch = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
BG = None if (len(sys.argv) > 2 and sys.argv[2] == "free") else SLAB
print("background:", "free space" if BG is None else "n=2 slab")
print(f"pitch {pitch:.0f} nm (gap ~{pitch-190:.0f} nm)")
geoms = [Geometry.gielis(rad=90.0, n_pts=nn, m=3, n1=4.0, n2=6.0, n3=6.0, x0=pitch*(p-(P-1)/2), z0=z0) for p in range(P)]
mat = Material.from_eps(-2.0*nclad**2, n_clad=nclad, pol=pol)
cs = ClusterBIESolver(Cluster(geoms), [mat]*P, pol=pol, background=BG)
full0 = cs._assemble(lam)
sl = [cs.cluster.slice(p) for p in range(P)]
ext = [np.arange(s.start, s.start+nn) for s in sl]          # outside rows of particle p
inn = [np.arange(s.start+nn, s.stop) for s in sl]           # inside rows of particle p
# isolated particle over the slab: identical particles at one height -> one M_p serves all
top1 = BIESolver(geoms[0], mat, background=BG).assemble(lam)[:nn]
print("isolated outside rows == cluster self block:",
      np.allclose(top1, full0[np.ix_(ext[0], np.arange(sl[0].start, sl[0].stop))], atol=1e-13))
def M1(e):
    return np.vstack([top1, interior_rows(pol, geoms[0], k_bg, ri(e), e)])
def Mfull(e):
    m = full0.copy()
    for p in range(P):
        m[inn[p], sl[p]] = interior_rows(pol, geoms[p], k_bg, ri(e), e)
    return m
# --- route 1: compress the neighbour coupling seen by each particle -------------------------------
t0 = time.perf_counter()
U, Y, rk = [], [], []
for p in range(P):
    D = full0[ext[p], :].copy(); D[:, sl[p]] = 0.0                 # neighbours only
    u, s, vh = np.linalg.svd(D, full_matrices=False)
    r = int(np.sum(s > 1e-13*s[0])); rk.append(r)
    U.append(u[:, :r]); Y.append(u[:, :r].conj().T @ D)          # D = U Y
t_comp = time.perf_counter() - t0
R0 = sum(rk); print(f"neighbour coupling rank per particle (1e-13): {rk}  -> reduced size {R0} vs full {2*nn*P}")
offs = np.cumsum([0]+rk)
def Gred(e):
    lu = lu_factor(M1(e))                                           # ONE small factorisation for all P
    out = np.eye(R0, dtype=complex)
    for q in range(P):
        Rq = lu_solve(lu, np.vstack([U[q], np.zeros((nn, rk[q]))]))    # 2nn x r_q
        for p in range(P):
            out[offs[p]:offs[p+1], offs[q]:offs[q+1]] += Y[p][:, sl[q]] @ Rq
    return out
lo, hi = -6.0-1.2j, -1.2-0.02j
def run(Mf, nq, npb, label):
    t0 = time.perf_counter()
    bm = beyn_modes(Mf, lo, hi, n_quad_per_side=nq, n_probe=npb, max_workers=1)
    t = time.perf_counter()-t0
    return np.sort_complex(bm.eigenvalues), t, bm.rank
ref, t_full, rf = run(Mfull, 32, 40, "full")
print(f"full cluster matrix ({2*nn*P}):  {ref.size} modes, rank {rf}, {t_full:.1f} s for {4*4*32} contour points")
red, t_red, rr = run(Gred, 32, 40, "reduced")
def polish(Mf, eigs):
    out = []
    for e in eigs:
        u, sv, vh = np.linalg.svd(Mf(e)); v = vh[-1].conj()
        dMf = lambda z, h=1e-6: (Mf(z+h) - Mf(z-h))/(2*h)
        out.append(newton_refine(Mf, dMf, e, v, lo, hi, tol=1e-13).eigenvalue)
    return np.sort_complex(np.array(out))
ref, red = polish(Mfull, ref), polish(Gred, red)
d = max(np.min(np.abs(red - e)) for e in ref) if red.size == ref.size else np.nan
print(f"route 1, exact reduction ({R0}): {red.size} modes, max |diff| vs full {d:.1e}, {t_red:.1f} s (+{t_comp:.2f} s compression)")
# per-point costs
def tmin(fn, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return min(ts)
tf = tmin(lambda: lu_factor(Mfull(-2-0.3j))); tr = tmin(lambda: lu_factor(Gred(-2-0.3j)))
print(f"cost per contour point: full {tf*1e3:.0f} ms, reduced {tr*1e3:.0f} ms")
# --- routes 2 and 3: isolated modes (+ smooth leftover) -> small algebraic problem -----------------
iso = beyn_modes(M1, -6.0-1.0j, -1.15-0.004j, n_quad_per_side=32, n_probe=16, max_workers=1)
iso2 = beyn_modes(M1, -0.95-1.0j, -0.40-0.002j, n_quad_per_side=32, n_probe=16, max_workers=1)
em = np.concatenate([iso.eigenvalues, iso2.eigenvalues])
print("isolated particle modes:", np.round(em, 5))
def dM1(e, h=1e-6):
    out = np.zeros((2*nn, 2*nn), complex); out[nn:] = (M1(e+h)[nn:] - M1(e-h)[nn:])/(2*h); return out
V, W, N = [], [], []
for e in em:
    u, s, vh = np.linalg.svd(M1(e)); v, w = vh[-1].conj(), u[:, -1].conj()
    V.append(v); W.append(w); N.append(w @ dM1(e) @ v)
V, W, N = np.array(V).T, np.array(W).T, np.array(N)
K = em.size
def Rq_exact(e, q):
    return np.linalg.solve(M1(e), np.vstack([U[q], np.zeros((nn, rk[q]))]))
def Rq_poles(e, q):
    B = np.vstack([U[q], np.zeros((nn, rk[q]))])
    return V @ ((W.T @ B) / ((e - em)*N)[:, None])
# smooth leftover per particle: quadratic in eps through 3 exact responses (same for all particles up to U_q)
samp = np.array([-4.5-0.3j, -2.6-0.3j, -1.3-0.3j])
S = []
for q in range(P):
    vals = np.array([Rq_exact(e, q) - Rq_poles(e, q) for e in samp])
    A = np.vander(samp, 3, increasing=True); S.append(np.einsum("ij,jkl->ikl", np.linalg.inv(A), vals))
def Gfit(e):
    out = np.eye(R0, dtype=complex)
    for q in range(P):
        Rq = Rq_poles(e, q) + S[q][0] + S[q][1]*e + S[q][2]*e**2
        for p in range(P):
            out[offs[p]:offs[p+1], offs[q]:offs[q+1]] += Y[p][:, sl[q]] @ Rq
    return out
for e in (-4.0-0.6j, -2.2-0.5j, -1.6-0.1j, -1.3-0.05j):
    a, b = Gfit(e), Gred(e)
    print(f"  leftover fit check at {e}: |Gfit-Gexact|/|Gexact| = {np.linalg.norm(a-b)/np.linalg.norm(b):.1e}")
try:
    fit, t_fit, rfit = run(Gfit, 32, 40, "fit")
except ValueError as ex:
    print("  route 2 contour:", str(ex)[:90]); fit, t_fit = np.array([]), np.nan
d2 = [np.min(np.abs(fit - e)) if fit.size else np.nan for e in ref]
print(f"route 2, modes + leftover: {fit.size} modes, |diff| vs full: median {np.median(d2):.1e}, max {np.max(d2):.1e}, {t_fit*1e3:.0f} ms contour")
# route 3, pure hybridization: eps a = (diag(em) - diag(N)^-1 C) a, one dense eigenproblem of size P*K
C = np.zeros((P*K, P*K), complex)
# build exactly: coupling of mode m of particle q into mode n of particle p = w_n^T [Delta_pq v_m ; 0]
for p in range(P):
    for q in range(P):
        if p == q: continue
        Dpq = full0[np.ix_(ext[p], np.arange(sl[q].start, sl[q].stop))]
        C[p*K:(p+1)*K, q*K:(q+1)*K] = W[:nn].T @ (Dpq @ V)
t0 = time.perf_counter()
hyb = np.linalg.eigvals(np.diag(np.tile(em, P)) - np.diag(1/np.tile(N, P)) @ C)
t_h = time.perf_counter()-t0
hyb = hyb[(hyb.real > lo.real) & (hyb.real < hi.real) & (hyb.imag > lo.imag) & (hyb.imag < hi.imag)]
d3 = [np.min(np.abs(hyb - e)) if hyb.size else np.nan for e in ref]
print(f"route 3, pure hybridization ({P*K}x{P*K}): {hyb.size} modes in box, |diff| vs full: median {np.median(d3):.1e}, max {np.max(d3):.1e}, {t_h*1e3:.1f} ms")
print("\nfull       ", np.round(ref, 5)); print("route 2    ", np.round(np.sort_complex(fit), 5)); print("route 3    ", np.round(np.sort_complex(hyb), 5))
