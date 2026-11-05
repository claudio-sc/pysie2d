import numpy as np, sys
sys.path.insert(0, ".")
from epsbeyn import EpsOperator
from pysie2d import Geometry
from pysie2d.beyn import rect_contour_quad, probe_matrix, contour_moments
op = EpsOperator(Geometry.gielis(rad=100.0, n_pts=64, m=0), pol=2, wavelength=600.0, branch="upper")
for lo, hi in ((0.5-4.0j, 30.0-0.3j), (0.5-4.0j, 10.5-0.3j), (20.0-4.0j, 30.0-0.3j)):
    pts, wts = rect_contour_quad(lo, hi, 96)
    v = probe_matrix(128, 24, 0)
    a0, a1, c = contour_moments(op.M, pts, wts, v, max_workers=1)
    s = np.linalg.svd(a0, compute_uv=False)
    print(f"box {lo}..{hi}: sv/s0 =", " ".join(f"{q:.0e}" for q in s[:11]/s[0]))
