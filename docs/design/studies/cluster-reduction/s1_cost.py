"""Route 1 outside the prototype's comfort zone: tight gaps, non-identical particles, one-off solves, large P."""
import os, sys, time, warnings
import numpy as np
from scipy.linalg import lu_factor, lu_solve
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "prototype"))
from perf import interior_rows, SIN, nclad, lam, pol, k_bg
from pysie2d import Geometry, Material, Cluster, ClusterBIESolver
warnings.simplefilter("ignore")
E = -2.0 + 0.1j            # background-relative, passive, so k_core is complex: the Amos path
now = time.perf_counter


def geoms(P, nn, pitch, jitter=0.0, z0=140.0):
    return [Geometry.gielis(rad=90.0 + jitter * (p % 5), n_pts=nn, m=3, n1=4.0, n2=6.0, n3=6.0,
                            x0=pitch * (p - (P - 1) / 2), z0=z0) for p in range(P)]


def compress(full0, cl, tol=1e-13):
    U, Y, rk = [], [], []
    for p, g in enumerate(cl.geometries):
        s_ = cl.slice(p)
        D = full0[s_.start:s_.start + g.n_pts].copy(); D[:, s_] = 0
        u, s, vh = np.linalg.svd(D, full_matrices=False)
        r = int(np.sum(s > tol * s[0]))
        U.append(u[:, :r]); Y.append(u[:, :r].conj().T @ D); rk.append(r)
    return U, Y, rk


def self_rows(full0, cl, p):
    s_ = cl.slice(p)
    return full0[s_.start:s_.start + cl.geometries[p].n_pts, s_]


def reduced_solve(full0, cl, U, Y, rk, b, e, dedup):
    """Woodbury on the compressed coupling. dedup: every particle shares one small factorisation."""
    P = len(cl); offs = np.cumsum([0] + rk)
    lus = []
    for p, g in enumerate(cl.geometries):
        if dedup and p:
            lus.append(lus[0]); continue
        lus.append(lu_factor(np.vstack([self_rows(full0, cl, p), interior_rows(pol, g, k_bg, np.sqrt(e), e)])))
    a, R = [], []
    for p, g in enumerate(cl.geometries):
        s_ = cl.slice(p); nn = g.n_pts
        a.append(lu_solve(lus[p], np.concatenate([b[s_.start:s_.start + nn], np.zeros(nn)])))
        R.append(lu_solve(lus[p], np.vstack([U[p], np.zeros((nn, rk[p]))])))
    G, rhs = np.eye(offs[-1], dtype=complex), np.zeros(offs[-1], complex)
    for p in range(P):
        for q in range(P):
            sq = cl.slice(q)
            G[offs[p]:offs[p + 1], offs[q]:offs[q + 1]] += Y[p][:, sq] @ R[q]
            rhs[offs[p]:offs[p + 1]] += Y[p][:, sq] @ a[q]
    y = np.linalg.solve(G, rhs)
    return np.concatenate([a[p] - R[p] @ y[offs[p]:offs[p + 1]] for p in range(P)])


def full_solve(full0, cl, b, e):
    m = full0.copy()
    for p, g in enumerate(cl.geometries):
        s_ = cl.slice(p)
        m[s_.start + g.n_pts:s_.stop, s_] = interior_rows(pol, g, k_bg, np.sqrt(e), e)
    return np.linalg.solve(m, b), m


def rhs(cl):
    rng = np.random.default_rng(0); b = np.zeros(cl.n_dof, complex)
    for p, g in enumerate(cl.geometries):
        s_ = cl.slice(p); b[s_.start:s_.start + g.n_pts] = rng.standard_normal(g.n_pts) + 1j * rng.standard_normal(g.n_pts)
    return b


def mat_of(P, cl):
    return [Material.from_eps(E * nclad**2, n_clad=nclad, pol=pol)] * P


print("=== A. rank vs gap (P=4, identical; nn from the gap envelope; worst rank is a middle particle)")
print(" bg   pitch  gap/a   nn   ranks          R0 / N      full/ε   red/ε  ratio  err(x)")
a = 90.0
for name, bg in (("free", None), ("slab", SIN)):
    for pitch, nn in ((1200, 100), (600, 100), (400, 100), (300, 100), (250, 200), (240, 400), (234, 800)):
        gs = geoms(4, nn, pitch)
        cl = Cluster(gs); a_ = max(np.hypot(g.f - g.x0, g.g - g.z0).max() for g in gs)
        cs = ClusterBIESolver(cl, mat_of(4, cl), pol=pol, background=bg)
        full0 = cs._assemble(lam); b = rhs(cl)
        U, Y, rk = compress(full0, cl)
        t0 = now(); xf, _ = full_solve(full0, cl, b, E); tf = now() - t0
        t0 = now(); xr = reduced_solve(full0, cl, U, Y, rk, b, E, True); tr = now() - t0
        print(f"{name:5s} {pitch:5d} {cl.min_gap/a_:6.3f} {nn:4d}  {str(rk):22s} {sum(rk):4d}/{cl.n_dof:5d} {tf*1e3:7.0f}ms {tr*1e3:6.0f}ms {tf/tr:5.1f}x  {np.linalg.norm(xf-xr)/np.linalg.norm(xf):.1e}", flush=True)

print("\n=== B. cost vs P (nn=100, pitch 600). Non-identical: radii differ, so no shared factorisation")
print(" bg     P      N   R0 | today: assemble+solve | setup: compress | per ε: exterior cached full | route1 distinct | route1 shared | one-off route1 | matrix MB vs Y+U MB")
for name, bg, Ps in (("free", None, (4, 8, 16, 32, 64)), ("slab", SIN, (4, 8, 16, 32))):
    for P in Ps:
        gs = geoms(P, 100, 600.0, jitter=1.0)
        cl = Cluster(gs); cs = ClusterBIESolver(cl, mat_of(P, cl), pol=pol, background=bg); b = rhs(cl)
        t0 = now(); full0 = cs._assemble(lam); t_asm = now() - t0
        t0 = now(); x = np.linalg.solve(full0, b); t_sol = now() - t0
        t0 = now(); U, Y, rk = compress(full0, cl); t_c = now() - t0
        t0 = now(); xf, _ = full_solve(full0, cl, b, E); t_f = now() - t0
        t0 = now(); xd = reduced_solve(full0, cl, U, Y, rk, b, E, False); t_d = now() - t0
        err = np.linalg.norm(xf - xd) / np.linalg.norm(xf)
        # shared-factorisation timing is on an identical array of the same size
        gi = geoms(P, 100, 600.0); cli = Cluster(gi)
        full_i = ClusterBIESolver(cli, mat_of(P, cli), pol=pol, background=bg)._assemble(lam)
        Ui, Yi, rki = compress(full_i, cli)
        t0 = now(); reduced_solve(full_i, cli, Ui, Yi, rki, rhs(cli), E, True); t_s = now() - t0
        mem_full = 16 * cl.n_dof**2 / 1e6; mem_red = 16 * (sum(rk) * cl.n_dof + sum(r * 100 for r in rk)) / 1e6
        print(f"{name:5s} {P:3d} {cl.n_dof:6d} {sum(rk):4d} | {t_asm:6.2f}+{t_sol:6.2f} s | {t_c:6.2f} s | {t_f*1e3:8.0f} ms | {t_d*1e3:7.0f} ms | {t_s*1e3:6.0f} ms | {t_c+t_d:6.2f} s | {mem_full:7.0f} vs {mem_red:6.0f} | err {err:.1e}", flush=True)
        del full0, full_i

print("\n=== C. what truncation buys and costs (P=8, nn=100, pitch 600, slab)")
gs = geoms(8, 100, 600.0); cl = Cluster(gs); cs = ClusterBIESolver(cl, mat_of(8, cl), pol=pol, background=SIN)
full0 = cs._assemble(lam); b = rhs(cl); xf, _ = full_solve(full0, cl, b, E)
for tol in (1e-13, 1e-10, 1e-8, 1e-6, 1e-4):
    U, Y, rk = compress(full0, cl, tol)
    t0 = now(); xr = reduced_solve(full0, cl, U, Y, rk, b, E, True); tr = now() - t0
    print(f"tol {tol:.0e}: R0={sum(rk):4d}  per ε {tr*1e3:5.1f} ms  err {np.linalg.norm(xf-xr)/np.linalg.norm(xf):.1e}", flush=True)
