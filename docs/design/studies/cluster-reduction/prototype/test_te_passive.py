import numpy as np, sys
sys.path.insert(0, ".")
from anchor import root, winding
from epsbeyn import EpsOperator, solve_box, BRANCHES
from pysie2d import BIESolver, Geometry, Material
from pysie2d.sources import plane_wave_rhs
rad, lam = 100.0, 600.0; x = 2*np.pi*rad/lam
# (a) passive half-plane: upper branch == principal branch == what BIESolver uses
for e in (-2+0.02j, -40+3j, 4+0.1j, 11+0.5j):
    print(f"passive eps={e}: principal={BRANCHES['principal'](e):.6f} upper={BRANCHES['upper'](e):.6f}")
geom = Geometry.gielis(rad=rad, n_pts=48, m=0)
for pol in (1, 2):
    op = EpsOperator(geom, pol, lam, branch="upper")
    e = -2.3 + 0.4j
    m_pkg = BIESolver(geom, Material.from_eps(e, pol=pol)).assemble(lam)
    print(f"pol={pol}: ||M_upper - M_pysie2d||/||M|| at passive eps = {np.linalg.norm(op.M(e)-m_pkg)/np.linalg.norm(m_pkg):.1e}")
# (b) dielectric modes, TE, upper branch, box strictly in Im eps < 0
lo, hi = 0.5 - 4.0j, 30.0 - 0.3j
ref = []
for n in range(0, 12):
    c = winding(n, x, 2, lo, hi, m=6000)
    if not c: continue
    # seed each zero from a coarse grid scan of |D|
    from anchor import D
    ers = np.linspace(lo.real, hi.real, 300); eis = np.linspace(lo.imag, hi.imag, 60)
    grid = np.array([[abs(D(a+1j*b, n, x, 2)) for a in ers] for b in eis])
    found = []
    for idx in np.argsort(grid, axis=None)[:200]:
        bi, ai = np.unravel_index(idx, grid.shape)
        r = root(ers[ai]+1j*eis[bi], n, x, 2)
        if lo.real < r.real < hi.real and lo.imag < r.imag < hi.imag and all(abs(r-f) > 1e-6 for f in found):
            found.append(r)
        if len(found) == c: break
    for r in found: ref += [r]*(2 if n else 1)
    print(f"analytic TE n={n}: {c} zero(s) {['%.6f%+.6fj' % (r.real, r.imag) for r in found]}")
ref = np.array(ref)
for nn in (48, 96):
    op = EpsOperator(Geometry.gielis(rad=rad, n_pts=nn, m=0), pol=2, wavelength=lam, branch="upper")
    eigs, bm, t = solve_box(op, lo, hi, nq=48, n_probe=24)
    err = max(np.min(np.abs(ref - e)) for e in eigs) if eigs.size else np.nan
    print(f"[upper TE nn={nn}] rank={bm.rank} found={eigs.size} (analytic {ref.size}) max|err|={err:.1e} t={t:.2f}s")
