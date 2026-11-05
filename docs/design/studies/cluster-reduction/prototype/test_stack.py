import numpy as np, sys, time, warnings
sys.path.insert(0, ".")
from epsbeyn import EpsOperator, solve_box
from pysie2d import Geometry, Stack, BIESolver, Material
from pysie2d.kernels import assemble_matrix
N_CLAD = float(sys.argv[1]) if len(sys.argv) > 1 else 1.444
lam, pol = 1550.0, 1
def shifted(geom, dz, rot_deg=17.0):
    c, s = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
    rot = lambda x, z: (c*x - s*z, s*x + c*z)
    f, g = rot(geom.f, geom.g); df, dg = rot(geom.df, geom.dg); ddf, ddg = rot(geom.ddf, geom.ddg)
    return Geometry(f=f, g=g+dz, df=df, dg=dg, ddf=ddf, ddg=ddg, rad=geom.rad, z0=dz)
SIN = Stack(((2.0**2, 400.0),), N_CLAD**2)
if __name__ == "__main__":
    boxes = [(-6.0-1.0j, -1.15-0.004j), (-0.87-1.0j, -0.05-0.004j)]
    res = {}
    for label, bg in (("free space", None), ("SiN slab", SIN)):
        for nn in (160, 240):
            geom = shifted(Geometry.gielis(rad=90.0, n_pts=nn, m=3, n1=4.0, n2=6.0, n3=6.0), 140.0)
            op = EpsOperator(geom, pol, lam, n_clad=N_CLAD, background=bg, branch="upper")
            t0 = time.perf_counter(); op.bottom(-2.0-0.1j); t_bot = time.perf_counter() - t0
            found, t_all = [], 0.0
            for lo, hi in boxes:
                eigs, bm, t = solve_box(op, lo, hi, nq=32, n_probe=16)
                found += list(eigs); t_all += t
                sig = [np.linalg.svd(op.M(e), compute_uv=False) for e in eigs]
            found = np.sort_complex(np.array(found))
            res[(label, nn)] = found
            smax = max((s[-1]/s[0] for s in sig), default=np.nan)
            print(f"[{label} nn={nn}] t_exterior_rows_once={op.t_top:.2f}s  t_interior_rows/node={t_bot:.3f}s  "
                  f"Beyn total={t_all:.1f}s for {2*4*32} nodes;  modes:")
            for e in found: print(f"    {e.real:+.8f} {e.imag:+.3e}i")
    for label in ("free space", "SiN slab"):
        a, b = res[(label, 160)], res[(label, 240)]
        if a.size == b.size:
            print(f"{label}: max |eps(nn=240)-eps(nn=160)| = {np.max(np.abs(a-b)):.1e}")
