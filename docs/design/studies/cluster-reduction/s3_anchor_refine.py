import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_modes.py")).read()
exec(src[:src.index('print("=== D.')])
from pysie2d.beyn import newton_refine
def polish(f, eigs, lo, hi, tol=1e-13):
    out = []
    for e in eigs:
        _, _, vh = np.linalg.svd(f(e)); v = vh[-1].conj()
        df = lambda z, h=1e-6: (f(z + h) - f(z - h)) / (2 * h)
        out.append(newton_refine(f, df, e, v, lo, hi, tol=tol).eigenvalue)
    return np.sort_complex(np.array(out))
dist = lambda a, b: max((np.min(np.abs(b - x)) for x in a), default=np.nan) if a.size and b.size else np.nan
print("=== D'. TM anchor with Newton polish on every operator; top edge pulled down from the accumulation at ε=-1")
for lo, hi in ((-2.0 - 1.5j, -0.2 - 0.15j), (-2.0 - 1.5j, -0.2 - 0.25j)):
    print(f"-- box {lo}..{hi}")
    for d in (300.0, 400.0, 1000.0):
        pr = Pair(two(d), 1, lam)
        em, rm = modes(pr.M, lo, hi); eg, rg = modes(pr.G, lo, hi)
        fg = {n: (lambda e, n=n: graf(e, 1, rad, k, d, n)) for n in (8, 10, 12)}
        ea = {n: modes(fg[n], lo, hi, npb=24)[0] for n in fg}
        if not em.size:
            print(f"d={d:6.0f}: M {rm}; G {rg}; Graf n10 finds {ea[10].size}: {np.round(ea[10], 6)}", flush=True); continue
        pm, pg = polish(pr.M, em, lo, hi), polish(pr.G, eg, lo, hi)
        pa = {n: polish(fg[n], ea[n], lo, hi) for n in fg}
        print(f"d={d:6.0f}: counts M {pm.size} G {pg.size} Graf {pa[12].size}; polished |M-G| {dist(pm, pg):.1e}, |M-Graf(12)| {dist(pm, pa[12]):.1e}; Graf 8v12 {dist(pa[8], pa[12]):.1e}, 10v12 {dist(pa[10], pa[12]):.1e}", flush=True)

print("\n=== G. the exterior irregular frequency: k_bg·a = j01, driven observable vs Mie (single circle)")
j01 = jn_zeros(0, 1)[0]; lam_d = 2 * np.pi * rad / j01
for p_, name in ((2, "TE"), (1, "TM")):
    for dl in (1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 0.0):
        lm = lam_d * (1 + dl)
        s = BIESolver(Geometry.gielis(rad=rad, n_pts=64, m=0), Material(n_core=1.5, pol=p_))
        sv = np.linalg.svd(s.assemble(lm), compute_uv=False)
        qe = s.scatter(lm).efficiencies()["qext"]; rv = efficiencies(2 * np.pi * rad / lm, 1.5)[f"Q_ext_{name}"]
        print(f"  {name} δλ/λ={dl:.0e}: σmin/σmax={sv[-1]/sv[0]:.1e}  Q_ext BIE={qe:.12f} Mie={rv:.12f} rel err {abs(qe-rv)/abs(rv):.1e}", flush=True)
