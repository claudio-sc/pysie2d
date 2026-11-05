import numpy as np, sys, time, warnings
sys.path.insert(0, ".")
from scipy.linalg import lu_factor
from perf import interior_rows, best, SIN, nclad, lam, pol, k_bg
from epsbeyn import EpsOperator, solve_box
def shifted(geom, dz, rot_deg=17.0):
    c, s = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
    rot = lambda x, z: (c*x - s*z, s*x + c*z)
    f, g = rot(geom.f, geom.g); df, dg = rot(geom.df, geom.dg); ddf, ddg = rot(geom.ddf, geom.ddg)
    return Geometry(f=f, g=g+dz, df=df, dg=dg, ddf=ddf, ddg=ddg, rad=geom.rad, z0=dz)
from pysie2d import Geometry, Material, Cluster, ClusterBIESolver
warnings.simplefilter("ignore")
# (1) smallest contour that still finds the slab modes
nn = 160
geom = shifted(Geometry.gielis(rad=90.0, n_pts=nn, m=3, n1=4.0, n2=6.0, n3=6.0), 140.0)
op = EpsOperator(geom, pol, lam, n_clad=nclad, background=SIN, branch="upper")
boxes = [(-6.0-1.0j, -1.15-0.004j), (-0.87-1.0j, -0.05-0.004j)]
ref = np.array([e for lo, hi in boxes for e in solve_box(op, lo, hi, nq=48, n_probe=16)[0]])
print(f'reference (48 per side): {ref.size} modes')
for nq in (12, 16, 24, 32):
    found = []
    try:
        for lo, hi in boxes:
            found += list(solve_box(op, lo, hi, nq=nq, n_probe=16)[0])
        found = np.array(found)
        err = max(np.min(np.abs(ref - e)) for e in found) if found.size else np.nan
        print(f"nq={nq:2d}: {8*nq:3d} contour points, found {found.size}/{ref.size}, max error after polishing {err:.1e}")
    except ValueError as e:
        print(f"nq={nq:2d}: failed ({str(e)[:60]})")
# (2) clusters: per-permittivity cost = P small inside blocks, not one big matrix
print("\n P  | dof  | full cluster+slab | inside blocks | LU       | memory MB")
for P in (1, 2, 4, 8):
    geoms = [Geometry.gielis(rad=90.0, n_pts=100, m=3, n1=4.0, n2=6.0, n3=6.0, x0=400.0*p, z0=250.0) for p in range(P)]
    mats = [Material.from_eps((-2.0+0.1j)*nclad**2, n_clad=nclad, pol=pol)]*P
    cs = ClusterBIESolver(Cluster(geoms), mats, pol=pol, background=SIN)
    t_full = best(lambda: cs._assemble(lam), 2)
    t_int = best(lambda: [interior_rows(pol, g, k_bg, 1j*np.sqrt(2.0-0.1j), -2.0+0.1j) for g in geoms], 2)
    m = cs._assemble(lam); t_lu = best(lambda: lu_factor(m), 2)
    print(f"{P:3d} | {m.shape[0]:4d} | {t_full*1e3:13.1f} ms | {t_int*1e3:9.1f} ms | {t_lu*1e3:6.1f} ms | {16*m.shape[0]**2/1e6:6.1f}")
