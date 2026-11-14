import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_modes.py")).read()
exec(src[:src.index('print("=== D.')])
def solve_full_rhs(pr, r, e):
    """Woodbury with a right side in both exterior and interior rows; the coupling U·Y still acts on exterior rows only."""
    a, R = [], []
    for p, g in enumerate(pr.gs):
        lu = lu_factor(pr.small(p, e))
        a.append(lu_solve(lu, r[pr.cl.slice(p)]))
        R.append(lu_solve(lu, np.vstack([pr.U[p], np.zeros((g.n_pts, pr.rk[p]))])))
    rhs = np.zeros(pr.offs[-1], complex)
    for p in range(len(pr.gs)):
        for q in range(len(pr.gs)):
            rhs[pr.offs[p]:pr.offs[p + 1]] += pr.Y[p][:, pr.cl.slice(q)] @ a[q]
    y = np.linalg.solve(pr.G(e), rhs)
    return np.concatenate([a[p] - R[p] @ y[pr.offs[p]:pr.offs[p + 1]] for p in range(len(pr.gs))])
gs = [Geometry.gielis(rad=90.0, n_pts=100, m=3, n1=4.0, n2=6.0, n3=6.0, x0=600.0 * (p - 1.5), z0=140.0) for p in range(4)]
pr = Pair(gs, 1, 1550.0, n_clad=nclad, bg=SLAB)
iso, _ = modes(lambda e: pr.small(0, e), -1.6 - 0.2j, -1.1 - 0.002j, nq=32, npb=12); em0 = iso[np.argmin(np.abs(iso.imag))]
rng = np.random.default_rng(0); b = np.zeros(pr.cl.n_dof, complex)
for p, g in enumerate(gs):
    s_ = pr.cl.slice(p); b[s_.start:s_.start + g.n_pts] = rng.standard_normal(g.n_pts) + 1j * rng.standard_normal(g.n_pts)
for dlt in (1e-2, 1e-6, 1e-8, 1e-10, 1e-12):
    e = em0 + dlt * np.exp(0.7j); m = pr.M(e); xf = np.linalg.solve(m, b)
    x0 = solve_full_rhs(pr, b, e); x1 = x0 + solve_full_rhs(pr, b - m @ x0, e); x2 = x1 + solve_full_rhs(pr, b - m @ x1, e)
    r = lambda x: np.linalg.norm(x - xf) / np.linalg.norm(xf)
    print(f"|ε−ε_iso|={dlt:.0e}: err {r(x0):.1e} → 1 step {r(x1):.1e} → 2 steps {r(x2):.1e}", flush=True)
