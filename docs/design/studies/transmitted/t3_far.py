"""a_down: eps=1 equals free-space far field; near-to-far consistency at large depth."""
import time, warnings, numpy as np
from pysie2d import BIESolver, Geometry, HalfSpace, Material
from pysie2d.fields import _far_field_at
from pysie2d.layered import alpha, _normalised
from tproto import PI, a_down, t_green, t_path
t0 = time.perf_counter()
LAM = 633.0
def solve(pol, eps_abs, angle=20.0):
    geo = Geometry.gielis(rad=100.0, n_pts=256, m=0, x0=0.0, z0=115.0)
    s = BIESolver(geo, Material(n_core=2.0, n_clad=1.33, pol=pol, epsi=0.5), HalfSpace(eps_abs))
    return s, s.scatter(LAM, angle)
if __name__ == '__main__':
    # eps = 1 (eps_abs = n_clad^2): a_down == free-space far field on the lower half
    th = np.linspace(0.5*PI + 1e-3, 1.5*PI - 1e-3, 301)
    for pol in (2, 1):
        s, res = solve(pol, 1.33**2)
        bnds, k, eps, _, z_int = res._substrate()
        ad = a_down(th, bnds, k, eps, pol, z_int)
        g = s.geometry
        ff = _far_field_at(th, g.n_pts, k, g.f, g.g, g.df, g.dg, g.delt, res.ei)
        print(f"eps=1 pol={pol}: max|a_down - ff|/max|ff| = {np.max(np.abs(ad-ff))/np.max(np.abs(ff)):.1e}")
    # near-to-far: psi(rho, theta) * [ (i/4) sqrt(2/(pi k2 rho)) e^{i(k2 rho - pi/4)} ]^{-1} -> a_down
    for eps_abs, name in ((2.25, "glass"), (1.0, "low-n")):
        for pol in (2, 1):
            s, res = solve(pol, eps_abs)
            bnds, k, eps, _, z_int = res._substrate()
            epsn = _normalised(eps); k2 = k*np.sqrt(epsn)
            thetas = np.array([PI - 0.3, PI, PI + 0.7, PI + 1.2])
            ad = a_down(thetas, bnds, k, eps, pol, z_int)
            g = s.geometry; nn = g.n_pts
            ests = []
            for R in (2e4, 4e4, 8e4):
                x, z = R*np.sin(thetas), R*np.cos(thetas)
                D = np.ptp(np.r_[g.f, x]); zmin = (g.g - z_int).min() + (z_int - z).min()
                p = t_path(k, eps, pol, D, zmin)
                psi = np.zeros(len(x), complex)
                for j in range(nn):  # -sum delt [ (dGx dg - dGz df) phi + G chi ]
                    G, Gx, Gz = t_green(p, pol, k, eps, x, z, g.f[j], g.g[j], z_int)
                    psi -= g.delt*((Gx*g.dg[j] - Gz*g.df[j])*res.ei[j] + G*res.ei[nn+j])
                fac = 0.25j*np.sqrt(2/(PI*k2*R))*np.exp(1j*(k2*R - PI/4))
                ests.append(psi/fac)
            e1, e2, e3 = ests
            rich = (4*e3 - 6*e2 + 2*e1)/ (4 - 6 + 2) if False else None
            # Richardson for a/(rho) correction: e(R) = a + c/R -> a ≈ 2 e(2R) - e(R)
            r1 = 2*e2 - e1; r2 = 2*e3 - e2
            print(f"{name} pol={pol}: |e(80um)-a|/|a| = {np.max(np.abs(e3-ad)/np.abs(ad)):.1e}; Richardson {np.max(np.abs(r2-ad)/np.abs(ad)):.1e} (prev {np.max(np.abs(r1-ad)/np.abs(ad)):.1e}); nodes {p.q.size}")
    print(f"elapsed {time.perf_counter()-t0:.1f}s")
