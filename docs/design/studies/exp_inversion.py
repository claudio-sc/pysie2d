import sys
sys.path.insert(0, "/home/claude/pysie2d-0.5.0/src")
import numpy as np
from pysie2d import BIESolver, Geometry, Material

TWOPI = 2 * np.pi
SHAPE = dict(rad=200, m=6, n1=6, n2=12, n3=12)
mat = Material(n_core=1.5, n_clad=1.0, pol=2)
lam = 600.0


def qext(geom):
    r = BIESolver(geom, mat).scatter(wavelength=lam)
    amp, _ = r.far_field(3000)
    d = TWOPI / 2999.0
    return amp[int((TWOPI - np.deg2rad(r.angle)) / d)].imag / (r.wnum_bg * 400.0)


def unif(nn):
    th = np.linspace(0, TWOPI, nn, endpoint=False) + np.pi / nn
    return Geometry.gielis(n_pts=nn, theta=th, **SHAPE)


ref = qext(unif(2400))
print(f"reference (uniform theta, nn=2400): qext = {ref:.10f}\n")

print("Effect of the arc-length inversion grid n_fine:")
print(f"{'nn':>6} " + "".join(f"{'n_fine='+str(m)+'x':>14}" for m in (10, 40, 160)))
for nn in (100, 200, 400, 800):
    row = []
    for mult in (10, 40, 160):
        from pysie2d.geometry import boundary_setup
        arrs = boundary_setup(nn, 200, 1.0, 1.0, SHAPE["m"], SHAPE["n1"],
                              SHAPE["n2"], SHAPE["n3"],
                              n_fine=max(mult * nn, 4096))
        f_, g_, df_, dg_, ddf_, ddg_, delt_, th_ = arrs
        g = Geometry(f_, g_, df_, dg_, ddf_, ddg_, delt_, theta=th_, rad=200)
        row.append(abs(qext(g) - ref) / abs(ref))
    print(f"{nn:6d} " + "".join(f"{v:14.3e}" for v in row))

print("\nSame, but with a spectrally-inverted arc-length node set")
print("(exact gamma integrated on a fine uniform grid, cubic inversion):")
from scipy.interpolate import CubicSpline
from pysie2d.geometry import gielis as gielis_raw, _rderiv


def smooth_arc_theta(nn, n_fine=200_000):
    th = np.linspace(0, TWOPI, n_fine + 1)
    f, g, r, co, se, arg = gielis_raw(th, 200, 1, 1, SHAPE["m"],
                                      SHAPE["n2"], SHAPE["n1"], SHAPE["n3"], 0, 0)
    rd = _rderiv(th, r, co, se, arg, 200, 1, 1, SHAPE["m"],
                 SHAPE["n2"], SHAPE["n1"], SHAPE["n3"])
    gam = np.sqrt(r**2 + rd**2)                      # |dx/dtheta| exactly
    s = np.concatenate([[0.0], np.cumsum((gam[1:] + gam[:-1]) / 2 * np.diff(th))])
    L = s[-1]
    inv = CubicSpline(s, th)
    return inv(np.linspace(0, L, nn, endpoint=False))


for nn in (100, 200, 400, 800):
    g = Geometry.gielis(n_pts=nn, theta=smooth_arc_theta(nn), **SHAPE)
    print(f"{nn:6d} {abs(qext(g)-ref)/abs(ref):14.3e}")
