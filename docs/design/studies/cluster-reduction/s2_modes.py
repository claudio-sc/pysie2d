"""Route 1's failure modes and the anchor: pole cancellation, hidden weakly split modes, Graf eigenpermittivities."""
import os, sys, warnings
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import hankel1, jn_zeros
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "prototype"))
from perf import interior_rows, SIN, nclad
from pysie2d import BIESolver, Geometry, Material, Cluster, ClusterBIESolver
from pysie2d.beyn import beyn_modes
from pysie2d.reference.mie import an, bn, efficiencies
warnings.simplefilter("ignore")
ri = lambda e: 1j * np.sqrt(-complex(e))


class Pair:
    """Full M(ε) and the reduced G(ε) for a cluster, with the same exterior rows."""
    def __init__(self, gs, pol, lam, n_clad=1.0, bg=None, tol=1e-13):
        self.gs, self.pol, self.k = gs, pol, 2 * np.pi * n_clad / lam
        self.cl = Cluster(gs)
        m0 = Material(n_core=1.5, n_clad=n_clad, pol=pol)
        self.full0 = ClusterBIESolver(self.cl, [m0] * len(gs), pol=pol, background=bg)._assemble(lam)
        self.U, self.Y, self.rk = [], [], []
        for p, g in enumerate(gs):
            s_ = self.cl.slice(p)
            D = self.full0[s_.start:s_.start + g.n_pts].copy(); D[:, s_] = 0
            u, s, _ = np.linalg.svd(D, full_matrices=False); r = int(np.sum(s > tol * s[0]))
            self.U.append(u[:, :r]); self.Y.append(u[:, :r].conj().T @ D); self.rk.append(r)
        self.offs = np.cumsum([0] + self.rk)

    def inner(self, p, e):
        return interior_rows(self.pol, self.gs[p], self.k, ri(e), e)

    def M(self, e):
        m = self.full0.copy()
        for p, g in enumerate(self.gs):
            s_ = self.cl.slice(p); m[s_.start + g.n_pts:s_.stop, s_] = self.inner(p, e)
        return m

    def small(self, p, e):
        s_ = self.cl.slice(p); nn = self.gs[p].n_pts
        return np.vstack([self.full0[s_.start:s_.start + nn, s_], self.inner(p, e)])

    def G(self, e):
        out = np.eye(self.offs[-1], dtype=complex); R = []
        for q, g in enumerate(self.gs):
            R.append(lu_solve(lu_factor(self.small(q, e)), np.vstack([self.U[q], np.zeros((g.n_pts, self.rk[q]))])))
        for p in range(len(self.gs)):
            for q in range(len(self.gs)):
                out[self.offs[p]:self.offs[p + 1], self.offs[q]:self.offs[q + 1]] += self.Y[p][:, self.cl.slice(q)] @ R[q]
        return out

    def solve_reduced(self, b, e):
        a, R = [], []
        for p, g in enumerate(self.gs):
            s_ = self.cl.slice(p); lu = lu_factor(self.small(p, e))
            a.append(lu_solve(lu, np.concatenate([b[s_.start:s_.start + g.n_pts], np.zeros(g.n_pts)])))
            R.append(lu_solve(lu, np.vstack([self.U[p], np.zeros((g.n_pts, self.rk[p]))])))
        rhs = np.zeros(self.offs[-1], complex)
        for p in range(len(self.gs)):
            for q in range(len(self.gs)):
                rhs[self.offs[p]:self.offs[p + 1]] += self.Y[p][:, self.cl.slice(q)] @ a[q]
        y = np.linalg.solve(self.G(e), rhs)
        return np.concatenate([a[p] - R[p] @ y[self.offs[p]:self.offs[p + 1]] for p in range(len(self.gs))])


def modes(f, lo, hi, nq=48, npb=24):
    try:
        bm = beyn_modes(f, lo, hi, n_quad_per_side=nq, n_probe=npb, max_workers=1)
        return np.sort_complex(bm.eigenvalues), bm.rank
    except ValueError as ex:
        return np.array([]), str(ex)[:70]


def graf(eps, pol, x_radius, k, d, n_max):
    """Two equal circles, the two_cylinder.py matrix I − s·T as a function of ε (independent of the BIE)."""
    m = ri(eps); n = np.arange(-n_max, n_max + 1); nd = n.size
    s = -(an if pol == 1 else bn)(np.abs(n), k * x_radius, m)
    mat = np.eye(2 * nd, dtype=complex)
    for el, j, phi in ((0, 1, np.pi), (1, 0, 0.0)):      # polar angle of the vector from centre j to centre el
        nm = n[None, :] - n[:, None]
        t = hankel1(nm, k * d) * np.exp(1j * nm * phi)
        mat[el * nd:(el + 1) * nd, j * nd:(j + 1) * nd] -= s[:, None] * t
    return mat


rad, lam = 100.0, 600.0; k = 2 * np.pi / lam
def two(d, nn=128):
    return [Geometry.gielis(rad=rad, n_pts=nn, m=0, x0=-d / 2, z0=0.0), Geometry.gielis(rad=rad, n_pts=nn, m=0, x0=d / 2, z0=0.0)]

print("=== D. independent anchor: cluster eigenpermittivities, BIE full M vs Graf addition theorem vs reduced G")
cases = ((1, -2.0 - 1.5j, -0.2 - 0.15j, "TM plasmonic"), (2, 11.0 - 1.4j, 13.8 - 0.1j, "TE dielectric"))
for pol, lo, hi, label in cases:
    print(f"-- {label}, box {lo}..{hi}")
    for d in (300.0, 400.0, 1000.0):
        pr = Pair(two(d), pol, lam)
        em, rm = modes(pr.M, lo, hi); eg, rg = modes(pr.G, lo, hi)
        gr = {}
        for nm_ in (8, 10):
            gr[nm_], _ = modes(lambda e: graf(e, pol, rad, k, d, nm_), lo, hi, npb=min(24, 2 * (2 * nm_ + 1)))
        ea = gr[10]
        dist = lambda a, b: max((np.min(np.abs(b - x)) for x in a), default=np.nan) if a.size and b.size else np.nan
        print(f"d={d:6.0f} ranks {pr.rk}: M {em.size} (rank {rm}), G {eg.size} (rank {rg}), Graf {ea.size}; "
              f"|M-Graf| {dist(em, ea):.1e}, |M-G| {dist(em, eg):.1e}, Graf n8 vs n10 {dist(gr[8], gr[10]):.1e}")
        print("      M:", np.round(em, 6), flush=True)

print("\n=== E. can G hide a weakly split pair? TE n-pair of two far circles, separation sweep; M vs G count in the same box")
lo, hi = 11.0 - 1.4j, 13.8 - 0.1j
for d in (1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0):
    pr = Pair(two(d, 64), 2, lam)
    em, rm = modes(pr.M, lo, hi); eg, rg = modes(pr.G, lo, hi)
    print(f"d={d:8.0f} rank {pr.rk}: M finds {em.size} (rank {rm}); G finds {eg.size} (rank {rg}); "
          f"spread(M)={np.ptp(np.abs(em)) if em.size else np.nan:.1e}", flush=True)
print("-- and a higher-Q family (n=4+ TE at larger ε) at moderate pitch")
lo2, hi2 = 25.0 - 2.2j, 40.0 - 0.02j
for d in (400.0, 1000.0, 3000.0):
    pr = Pair(two(d, 96), 2, lam)
    em, rm = modes(pr.M, lo2, hi2, nq=96); eg, rg = modes(pr.G, lo2, hi2, nq=96)
    print(f"d={d:6.0f}: M finds {em.size} (rank {rm}); G finds {eg.size} (rank {rg})")
    print("   M", np.round(em, 5)); print("   G", np.round(eg, 5), flush=True)

print("\n=== F. cancellation: x = a − R y when ε nears an isolated-particle pole (M_p singular, cluster M regular)")
pol = 1; lam1, n1 = 1550.0, nclad
gs = [Geometry.gielis(rad=90.0, n_pts=100, m=3, n1=4.0, n2=6.0, n3=6.0, x0=600.0 * (p - 1.5), z0=140.0) for p in range(4)]
pr = Pair(gs, pol, lam1, n_clad=n1, bg=SIN)
iso, _ = modes(lambda e: pr.small(0, e), -1.6 - 0.2j, -1.1 - 0.002j, nq=32, npb=12)
em0 = iso[np.argmin(np.abs(iso.imag))]
print(f"isolated pole used: {em0:.8f}")
rng = np.random.default_rng(0); b = np.zeros(pr.cl.n_dof, complex)
for p, g in enumerate(gs):
    s_ = pr.cl.slice(p); b[s_.start:s_.start + g.n_pts] = rng.standard_normal(g.n_pts) + 1j * rng.standard_normal(g.n_pts)
for dlt in (1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    e = em0 + dlt * np.exp(0.7j)
    m = pr.M(e); xf = np.linalg.solve(m, b); xr = pr.solve_reduced(b, e)
    be = lambda x: np.linalg.norm(m @ x - b) / (np.linalg.norm(m) * np.linalg.norm(x) + np.linalg.norm(b))
    print(f"|ε−ε_iso|={dlt:.0e}: |x_r−x_f|/|x_f|={np.linalg.norm(xr-xf)/np.linalg.norm(xf):.1e}  backward err full {be(xf):.1e}, reduced {be(xr):.1e}  cond(M_full)~{np.linalg.cond(m):.1e}", flush=True)

print("\n=== G. the exterior irregular frequency: k_bg·a = j01, driven observable vs Mie (single circle)")
j01 = jn_zeros(0, 1)[0]; lam_d = 2 * np.pi * rad / j01
for p_, name in ((2, "TE"), (1, "TM")):
    for dl in (1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 0.0):
        lm = lam_d * (1 + dl)
        s = BIESolver(Geometry.gielis(rad=rad, n_pts=64, m=0), Material(n_core=1.5, pol=p_))
        m = s.assemble(lm); sv = np.linalg.svd(m, compute_uv=False)
        r = s.scatter(lm); q = r.efficiencies(); x = 2 * np.pi * rad / lm
        ref = efficiencies(x, 1.5)
        if dl == 1e-1 and p_ == 2: print("   keys", q if not isinstance(q, dict) else list(q), type(ref))
        qe = q["Q_ext"] if isinstance(q, dict) else q
        rv = ref[f"Q_ext_{name}"] if isinstance(ref, dict) else ref
        print(f"  {name} δλ/λ={dl:.0e}: σmin/σmax={sv[-1]/sv[0]:.1e}  Q_ext BIE={qe:.12f} Mie={rv:.12f} rel err {abs(qe-rv)/abs(rv):.1e}", flush=True)
