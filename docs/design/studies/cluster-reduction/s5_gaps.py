"""The cases s1-s4 leave open: both polarisations on every background with mixed particles, rank vs particle size,
and the worst case for a hidden mode (a high-Q pole, where the reduced G nearly cancels pole against zero)."""
import sys, time, warnings
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import hankel1, h1vp, jn_zeros, jv, jvp
from pysie2d import Cluster, ClusterBIESolver, Geometry, HalfSpace, Material, Stack
from pysie2d.beyn import beyn_modes
from pysie2d.kernels import assemble_matrix
warnings.simplefilter("ignore")
now = time.perf_counter
ri_upper = lambda e: 1j * np.sqrt(-complex(e))


def compress(full0, cl, tol=1e-13):
    U, Y, rk = [], [], []
    for p, g in enumerate(cl.geometries):
        s_ = cl.slice(p)
        D = full0[s_.start:s_.start + g.n_pts].copy(); D[:, s_] = 0
        u, s, _ = np.linalg.svd(D, full_matrices=False)
        r = int(np.sum(s > tol * s[0]))
        U.append(u[:, :r]); Y.append(u[:, :r].conj().T @ D); rk.append(r)
    return U, Y, rk, np.cumsum([0] + rk)


def solve_reduced(blocks, cl, U, Y, rk, offs, b):
    """blocks[p] is particle p's whole 2nn x 2nn matrix, its own substrate reflection included."""
    a, R = [], []
    for p, g in enumerate(cl.geometries):
        lu = lu_factor(blocks[p])
        a.append(lu_solve(lu, b[cl.slice(p)]))
        R.append(lu_solve(lu, np.vstack([U[p], np.zeros((g.n_pts, rk[p]))])))
    G, rhs = np.eye(offs[-1], dtype=complex), np.zeros(offs[-1], complex)
    for p in range(len(cl)):
        for q in range(len(cl)):
            G[offs[p]:offs[p + 1], offs[q]:offs[q + 1]] += Y[p][:, cl.slice(q)] @ R[q]
            rhs[offs[p]:offs[p + 1]] += Y[p][:, cl.slice(q)] @ a[q]
    y = np.linalg.solve(G, rhs)
    return np.concatenate([a[p] - R[p] @ y[offs[p]:offs[p + 1]] for p in range(len(cl))]), a


print("=== H. driven, physical plane wave: mixed shapes, nn and materials; both polarisations on every background")
lam, nc = 1550.0, 1.444
SHAPES = lambda: [
    Geometry.gielis(rad=70.0, n_pts=64, m=0, x0=-420.0, z0=170.0),
    Geometry.gielis(rad=90.0, n_pts=120, m=3, n1=4.0, n2=6.0, n3=6.0, x0=-80.0, z0=200.0),
    Geometry.gielis(rad=60.0, n_pts=90, m=4, n1=8.0, n2=8.0, n3=8.0, x0=240.0, z0=140.0),
    Geometry.gielis(rad=110.0, n_pts=140, m=0, x0=640.0, z0=230.0, a=1.0, b=0.8),
]
BGS = (("free", None), ("glass half-space", HalfSpace(2.25, z_int=0.0)), ("PEC", HalfSpace.pec()),
       ("n=2 slab", Stack(((4.0, 400.0),), nc**2)))
print(" background        pol  ranks              R0/N       |x_r-x_d|/|x_d|  |Δamp|/|amp|  ‖a‖/‖x‖")
for name, bg in BGS:
    for pol in (1, 2):
        gs = SHAPES(); cl = Cluster(gs)
        mats = [Material.from_eps((-2.0 + 0.2j) * nc**2, n_clad=nc, pol=pol), Material(n_core=2.4, n_clad=nc, pol=pol),
                Material.from_eps((4.0 + 0.3j) * nc**2, n_clad=nc, pol=pol), Material(n_core=1.8, n_clad=nc, pol=pol)]
        cs = ClusterBIESolver(cl, mats, pol=pol, background=bg)
        M = cs._assemble(lam); res = cs.scatter(lam, 13.0); b = M @ res.ei
        leak = max(np.linalg.norm(b[cl.slice(p).start + g.n_pts:cl.slice(p).stop]) for p, g in enumerate(gs)) / np.linalg.norm(b)
        U, Y, rk, offs = compress(M, cl)
        xr, a = solve_reduced([M[np.ix_(range(cl.slice(p).start, cl.slice(p).stop), range(cl.slice(p).start, cl.slice(p).stop))]
                               for p in range(len(gs))], cl, U, Y, rk, offs, b)
        amp_d, _ = res.far_field(361)
        res.ei = xr; amp_r, _ = res.far_field(361)
        guard = max(np.linalg.norm(a[p]) for p in range(len(gs))) / np.linalg.norm(xr)
        res.ei = None
        xd = np.linalg.solve(M, b)
        print(f" {name:17s} {pol}   {str(rk):19s} {sum(rk):4d}/{cl.n_dof}  {np.linalg.norm(xr-xd)/np.linalg.norm(xd):13.1e}  "
              f"{np.max(np.abs(amp_r-amp_d))/np.max(np.abs(amp_d)):11.1e}  {guard:7.1e}  (rhs interior leak {leak:.0e})", flush=True)

print("\n=== I. rank vs particle size: 4 circles at pitch 2.5a, TE, n=2, λ=600, nn = 12 points per interior wavelength")
print("     a      x    nn  ranks                 r_max/2nn   R0/N        dense/ε   route1 shared/ε  ratio  err")
E = 4.0 + 0.01j
for a_ in (250.0, 500.0, 1000.0, 2000.0, 4000.0):
    x = 2 * np.pi * a_ / 600.0; nn = max(64, int(np.ceil(12 * 2 * x)))
    gs = [Geometry.gielis(rad=a_, n_pts=nn, m=0, x0=2.5 * a_ * (p - 1.5), z0=0.0) for p in range(4)]
    cl = Cluster(gs); k = 2 * np.pi / 600.0; mat = Material.from_eps(E, pol=2)
    M = ClusterBIESolver(cl, [mat] * 4, pol=2)._assemble(600.0)
    rng = np.random.default_rng(0); b = np.zeros(cl.n_dof, complex)
    for p in range(4):
        s_ = cl.slice(p); b[s_.start:s_.start + nn] = rng.standard_normal(nn) + 1j * rng.standard_normal(nn)
    U, Y, rk, offs = compress(M, cl)
    t0 = now(); xd = np.linalg.solve(M, b); td = now() - t0
    block = M[:2 * nn, :2 * nn]
    t0 = now(); lu = lu_factor(block); A = [lu_solve(lu, b[cl.slice(p)]) for p in range(4)]
    R = [lu_solve(lu, np.vstack([U[p], np.zeros((nn, rk[p]))])) for p in range(4)]
    G, rhs = np.eye(offs[-1], dtype=complex), np.zeros(offs[-1], complex)
    for p in range(4):
        for q in range(4):
            G[offs[p]:offs[p + 1], offs[q]:offs[q + 1]] += Y[p][:, cl.slice(q)] @ R[q]
            rhs[offs[p]:offs[p + 1]] += Y[p][:, cl.slice(q)] @ A[q]
    y = np.linalg.solve(G, rhs); xr = np.concatenate([A[p] - R[p] @ y[offs[p]:offs[p + 1]] for p in range(4)])
    tr = now() - t0
    print(f"{a_:6.0f} {x:6.1f} {nn:5d}  {str(rk):22s} {max(rk)/(2*nn):8.3f}  {sum(rk):5d}/{cl.n_dof:5d}  {td*1e3:7.0f} ms  {tr*1e3:7.0f} ms  {td/tr:6.1f}x  {np.linalg.norm(xr-xd)/np.linalg.norm(xd):.1e}", flush=True)


def te_root(n, x):
    """Newton on the TE Mie denominator, divided by H_n(x) so a high order does not overflow the iteration."""
    def f(e):
        m = np.sqrt(complex(e))
        return m * jvp(n, m * x) - jv(n, m * x) * h1vp(n, x) / hankel1(n, x)
    e = complex((jn_zeros(n, 1)[0] / x) ** 2)
    for _ in range(60):
        de = 1e-7 * (1 + abs(e))
        step = f(e) / ((f(e + de) - f(e - de)) / (2 * de))
        e -= step
        if abs(step) < 1e-14 * abs(e):
            break
    return e


class Two:
    def __init__(self, gs, pol, lam):
        self.gs, self.pol, self.k = gs, pol, 2 * np.pi / lam
        self.cl = Cluster(gs); self.nn = gs[0].n_pts
        self.full0 = ClusterBIESolver(self.cl, [Material(n_core=1.5, pol=pol)] * len(gs), pol=pol)._assemble(lam)
        self.U, self.Y, self.rk, self.offs = compress(self.full0, self.cl)
        self.cache = {}

    def inner(self, e):
        # identical circles in free space: their interior rows are translation invariant, so one assembly serves all
        if e not in self.cache:
            g = self.gs[0]
            self.cache[e] = assemble_matrix(self.pol, g.n_pts, g.f, g.g, g.df, g.dg, g.ddf, g.ddg, self.k, ri_upper(e), complex(e))[self.nn:]
        return self.cache[e]

    def M(self, e):
        m = self.full0.copy()
        for p in range(len(self.gs)):
            s_ = self.cl.slice(p); m[s_.start + self.nn:s_.stop, s_] = self.inner(e)
        return m

    def G(self, e):
        lu = lu_factor(np.vstack([self.full0[:self.nn, :2 * self.nn], self.inner(e)]))
        R = [lu_solve(lu, np.vstack([self.U[q], np.zeros((self.nn, self.rk[q]))])) for q in range(len(self.gs))]
        out = np.eye(self.offs[-1], dtype=complex)
        for p in range(len(self.gs)):
            for q in range(len(self.gs)):
                out[self.offs[p]:self.offs[p + 1], self.offs[q]:self.offs[q + 1]] += self.Y[p][:, self.cl.slice(q)] @ R[q]
        return out


def count(f, lo, hi, nq, npb):
    try:
        bm = beyn_modes(f, lo, hi, n_quad_per_side=nq, n_probe=npb, max_workers=4)
        return np.sort_complex(bm.eigenvalues), bm.rank
    except ValueError as ex:
        return np.array([]), str(ex)[:60]


print("\n=== J. hidden-mode worst case: TE whispering-gallery ladder of two circles (a=1000, λ=600, x=10.47), rising Q")
x = 2 * np.pi * 1000.0 / 600.0
roots = {n: te_root(n, x) for n in range(10, 46)}
for t in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    cand = [n for n, e in roots.items() if e.imag < 0 and e.real > 1]
    n = min(cand, key=lambda q: abs(np.log10(-roots[q].imag) - np.log10(t)))
    em = roots[n]; w = 8 * abs(em.imag)
    lo, hi = em - w - 1j * w, complex(em.real + w, em.imag / 2)
    nn = max(128, int(np.ceil(12 * np.sqrt(em.real) * x)))
    for d in (2500.0, 6000.0):
        gs = [Geometry.gielis(rad=1000.0, n_pts=nn, m=0, x0=s * d / 2, z0=0.0) for s in (-1, 1)]
        tw = Two(gs, 2, 600.0)
        t0 = now()
        em_M, rm = count(tw.M, lo, hi, 24, 10); em_G, rg = count(tw.G, lo, hi, 24, 10)
        dd = max((np.min(np.abs(em_G - z)) for z in em_M), default=np.nan) if em_M.size and em_G.size else np.nan
        print(f"n={n:2d} ε_iso={em.real:.6f}{em.imag:+.2e}j d={d:5.0f} nn={nn} rk={tw.rk}: M {em_M.size} (rank {rm}), G {em_G.size} (rank {rg}); max|M-G| {dd:.1e}, "
              f"split/|Im ε_iso| {np.ptp(em_M.imag)/abs(em.imag) if em_M.size else np.nan:.2f}  [{now()-t0:.0f}s]", flush=True)
