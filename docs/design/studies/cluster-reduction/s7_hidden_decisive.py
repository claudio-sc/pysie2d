"""Decisive form of s6: find the true cluster modes from Graf (independent of the BIE), then ask M and G about exactly
those points. A hidden mode would show as: M singular at the Graf root, G not; or a tight box where M counts more than G."""
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s5_gaps.py")).read()
a = src.index('print("=== H.'); b = src.index("def te_root"); c = src.index('print("\\n=== J.')
exec(src[:a] + src[b:c])
from scipy.signal import argrelmin
from pysie2d.reference.mie import bn
x = 2 * np.pi * 1000.0 / 600.0; k = 2 * np.pi / 600.0; d = 2500.0
def graf(e, n_max=34):
    n = np.arange(-n_max, n_max + 1); nd = n.size; s = -bn(np.abs(n), x, ri_upper(e))
    mat = np.eye(2 * nd, dtype=complex)
    for el, j, phi in ((0, 1, np.pi), (1, 0, 0.0)):
        nm = n[None, :] - n[:, None]
        mat[el * nd:(el + 1) * nd, j * nd:(j + 1) * nd] -= s[:, None] * hankel1(nm, k * d) * np.exp(1j * nm * phi)
    return mat
def newton(f, e, its=12, rel=1e-4):
    """Singular-vector Newton; FD step tied to the current update, since the nearby Mie pole sets the local scale."""
    h = 1e-7
    for _ in range(its):
        u, s, vh = np.linalg.svd(f(e)); w, v = u[:, -1].conj(), vh[-1].conj()
        dF = (f(e + h) - f(e - h)) / (2 * h)
        step = (w @ f(e) @ v) / (w @ dF @ v); e = e - step
        h = max(abs(step) * rel, 1e-12 * abs(e))
        if abs(step) < 1e-14 * abs(e): break
    return e
def rs(f, e):
    s = np.linalg.svd(f(e), compute_uv=False); return s[-1] / s[0]
for n in (18, 20, 22, 24):
    em = te_root(n, x); nn = max(128, int(np.ceil(12 * np.sqrt(em.real) * x)))
    tw = Two([Geometry.gielis(rad=1000.0, n_pts=nn, m=0, x0=s * d / 2, z0=0.0) for s in (-1, 1)], 2, 600.0)
    print(f"-- n={n} ε_iso={em.real:.6f}{em.imag:+.2e}j nn={nn} rk={tw.rk}", flush=True)
    grid = em.real + np.linspace(-0.05, 0.05, 2001)
    sig = np.array([np.linalg.svd(graf(complex(r, em.imag)), compute_uv=False)[-1] for r in grid])
    idx = argrelmin(sig, order=3)[0]; idx = idx[np.argsort(sig[idx])][:6]
    roots = []
    for i in idx:
        r = newton(graf, complex(grid[i], em.imag))
        if r.imag < 0 and abs(r - em) < 0.1 and all(abs(r - q) > 1e-9 for q in roots): roots.append(r)
    print(f"   Graf found {len(roots)} roots from {idx.size} σmin minima")
    for r in sorted(roots, key=lambda z: z.real):
        rm, rg = newton(tw.M, r), newton(tw.G, r)
        w = max(6 * abs(r.imag), 1e-12); lo, hi = r - w - 1j * w, complex(r.real + w, r.imag / 2)
        cm, km = count(tw.M, lo, hi, 32, 10); cg, kg = count(tw.G, lo, hi, 32, 10)
        print(f"   ε_c−ε_iso={r-em:.3e}  Q≈{r.real/(2*abs(r.imag)):.1e} | |M root−Graf| {abs(rm-r):.1e} |G root−Graf| {abs(rg-r):.1e} | "
              f"σmin/σmax M {rs(tw.M, rm):.1e} G {rs(tw.G, rg):.1e} | tight box w={w:.0e}: M {cm.size} (rank {km}) G {cg.size} (rank {kg})", flush=True)
