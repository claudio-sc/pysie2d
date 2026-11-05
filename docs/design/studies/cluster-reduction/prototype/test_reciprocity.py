"""Boundary version of the paper's eq. (34): is the source-side factor w_m^T b(r') proportional to F_m(r')?"""
import numpy as np, sys
sys.path.insert(0, ".")
from epsbeyn import EpsOperator, solve_box
from pysie2d import Geometry
from pysie2d.fields import eval_field
from pysie2d.sources import line_dipole_rhs
lam = 1000.0; k = 2*np.pi/lam
geom = Geometry.gielis(rad=90.0, n_pts=160, m=4, n1=4.0, n2=6.0, n3=6.0, a=1.0, b=0.8)   # no symmetry -> simple modes
op = EpsOperator(geom, 1, lam, branch="upper")
modes = solve_box(op, -6.0-1.0j, -1.15-0.004j, nq=32, n_probe=16)[0]
pts = [(300.0, 40.0), (-250.0, 180.0), (60.0, -320.0), (400.0, -400.0), (-150.0, -260.0)]
nn = geom.n_pts
for e in modes:
    u, s, vh = np.linalg.svd(op.M(e)); v = vh[-1].conj(); w = u[:, -1].conj()     # w^T M = 0
    F = np.array([eval_field(v, nn, geom.f, geom.df, geom.g, geom.dg, geom.delt, k,
                             np.array([x]), np.array([z]))[0] for x, z in pts])
    Ft = np.array([w @ line_dipole_rhs(nn, k, geom.f, geom.g, x, z) for x, z in pts])
    r = Ft / F
    print(f"eps_m={e:.5f}: ratio F~/F at 5 points spread = {np.max(np.abs(r - r.mean()))/abs(r.mean()):.1e}")
