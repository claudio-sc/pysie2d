"""s5 section J left a discrepancy: at d=2500, n>=20, M reports rank 4 and G rank 0 in a box 8|Im ε_iso| wide.
Either the evanescent split pushed the modes out of that box, or G hides modes. Widen the box and ask Graf too."""
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s5_gaps.py")).read()
a = src.index('print("=== H.'); b = src.index("def te_root"); c = src.index('print("\\n=== J.')
exec(src[:a] + src[b:c])
from scipy.special import hankel1
from pysie2d.reference.mie import bn
x = 2 * np.pi * 1000.0 / 600.0; k = 2 * np.pi / 600.0; d = 2500.0
def graf(e, n_max):
    n = np.arange(-n_max, n_max + 1); nd = n.size; s = -bn(np.abs(n), x, ri_upper(e))
    mat = np.eye(2 * nd, dtype=complex)
    for el, j, phi in ((0, 1, np.pi), (1, 0, 0.0)):
        nm = n[None, :] - n[:, None]
        mat[el * nd:(el + 1) * nd, j * nd:(j + 1) * nd] -= s[:, None] * hankel1(nm, k * d) * np.exp(1j * nm * phi)
    return mat
for n in (18, 20, 22, 24):
    em = te_root(n, x); nn = max(128, int(np.ceil(12 * np.sqrt(em.real) * x)))
    tw = Two([Geometry.gielis(rad=1000.0, n_pts=nn, m=0, x0=s * d / 2, z0=0.0) for s in (-1, 1)], 2, 600.0)
    print(f"-- n={n} ε_iso={em.real:.6f}{em.imag:+.2e}j nn={nn}", flush=True)
    # w0: the box s5 used, whose top edge sits at Im ε_iso/2
    for w in (8 * abs(em.imag), 1e-5, 1e-4, 1e-3, 1e-2):
        lo = em - w - 1j * w; hi = complex(em.real + w, -max(abs(em.imag) / 2, w * 1e-2))
        M_, rm = count(tw.M, lo, hi, 48, 10); G_, rg = count(tw.G, lo, hi, 48, 10)
        A_, ra = count(lambda e: graf(e, 34), lo, hi, 48, 10)
        dmg = max((np.min(np.abs(G_ - z)) for z in M_), default=np.nan) if M_.size and G_.size else np.nan
        dma = max((np.min(np.abs(A_ - z)) for z in M_), default=np.nan) if M_.size and A_.size else np.nan
        print(f"  w={w:.1e}: M {M_.size} (rank {rm}) | G {G_.size} (rank {rg}) | Graf {A_.size} (rank {ra}) | |M-G| {dmg:.1e} |M-Graf| {dma:.1e}", flush=True)
        if M_.size: print("     M:", np.array2string(M_ - em, precision=3))
