"""Cost of one particle-permittivity value: today vs cached outside rows vs mode expansion."""
import numpy as np, sys, time, os
sys.path.insert(0, ".")
from scipy.linalg import lu_factor, lu_solve
from pysie2d import BIESolver, Geometry, Material, Stack, Cluster, ClusterBIESolver
from pysie2d.kernels import _kress_weights, _j0_h0, _j1_h1, _real_if_real, PI

def interior_rows(pol, g, k_bg, ri, eps):
    """Inside rows only: the core half of kernels.assemble_matrix (same expressions)."""
    nn = g.n_pts; kc = _real_if_real(ri*k_bg); eta = np.complex128(eps if pol == 1 else 1.0)
    h = 2*PI/nn; w = _kress_weights(nn)
    gamma = np.sqrt(g.df**2 + g.dg**2); deriv = g.df*g.ddg - g.ddf*g.dg
    ui, uj = np.triu_indices(nn, k=1); wt = w[uj-ui]
    dx, dz = g.f[ui]-g.f[uj], g.g[ui]-g.g[uj]
    cij = dx*g.dg[uj] - dz*g.df[uj]; cji = -dx*g.dg[ui] + dz*g.df[ui]
    z = kc*np.sqrt(dx**2+dz**2)
    j0c, h0c = _j0_h0(z); j1c, h1c = _j1_h1(z)
    single = 0.25j*h*h0c - wt*j0c/(4*PI)
    double = kc**2*(0.25j*h*(h1c/z) - wt*(j1c/z)/(4*PI))
    out = np.zeros((nn, 2*nn), complex)
    out[ui, uj] = double*cij; out[uj, ui] = double*cji
    out[ui, uj+nn] = eta*single; out[uj, ui+nn] = eta*single
    d = np.arange(nn)
    out[d, d] = -(0.5 + deriv*h/(4*PI)/gamma**2)
    out[d, d+nn] = eta*(-w[0]/(4*PI) + h*(0.25j - (np.euler_gamma + np.log(kc*gamma/2))/(2*PI)))
    return out

def best(fn, n=3):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter()-t0)
    return min(t)

lam, pol, nclad = 1550.0, 1, 1.444
k_bg = 2*np.pi*nclad/lam
SLAB = Stack(((2.0**2, 400.0),), nclad**2)
eps = -2.0 + 0.1j; ri = 1j*np.sqrt(-eps)
if __name__ == "__main__":
    print(f"cores: {os.cpu_count()}  (single-threaded timings; ratios are the point)")
    print(" nn  | full, free | full, slab | inside rows | LU (2nn) | solve | matrix MB")
    for nn in (100, 200, 400, 800, 1600):
        geom = Geometry.gielis(rad=90.0, n_pts=nn, m=3, n1=4.0, n2=6.0, n3=6.0, z0=250.0)
        mat = Material.from_eps(eps*nclad**2, n_clad=nclad, pol=pol)
        s_free, s_slab = BIESolver(geom, mat), BIESolver(geom, mat, background=SLAB)
        # sanity: trimmed inside rows == bottom half of the package's own matrix (same branch on passive side)
        m_pkg = BIESolver(geom, Material.from_eps((-2.0+0.1j)*nclad**2, n_clad=nclad, pol=pol)).assemble(lam)
        assert np.allclose(interior_rows(pol, geom, k_bg, np.sqrt(-2.0+0.1j), -2.0+0.1j), m_pkg[nn:], rtol=0, atol=1e-13)
        t_free = best(lambda: s_free.assemble(lam)); t_slab = best(lambda: s_slab.assemble(lam), 2 if nn > 400 else 3)
        t_int = best(lambda: interior_rows(pol, geom, k_bg, ri, eps))
        m = s_slab.assemble(lam); t_lu = best(lambda: lu_factor(m)); lu = lu_factor(m)
        b = np.random.default_rng(0).standard_normal(2*nn) + 0j
        t_sv = best(lambda: lu_solve(lu, b))
        print(f"{nn:5d}| {t_free*1e3:8.1f}ms | {t_slab*1e3:8.1f}ms | {t_int*1e3:9.1f}ms | {t_lu*1e3:6.1f}ms | {t_sv*1e3:4.2f}ms | {16*(2*nn)**2/1e6:7.1f}")
    # mode expansion evaluation: K modes, many permittivity values, one observable
    K, N = 6, 100_000
    em = np.random.default_rng(1).standard_normal(K) - 1j; am = np.ones(K) + 0j
    ev = np.linspace(-3, -0.2, N) + 0.02j; coef = np.ones(3) + 0j
    def expand(): return (am[None, :]/(ev[:, None]-em[None, :])).sum(1) + coef[0] + coef[1]*ev + coef[2]*ev**2
    t = best(expand)
    print(f"mode expansion: {N} permittivity values, {K} modes + quadratic leftover: {t*1e3:.1f} ms total, {t/N*1e9:.0f} ns each")
