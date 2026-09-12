import sys
sys.path.insert(0, "/home/claude/pysie2d-0.5.0/src")
import numpy as np
from scipy.optimize import minimize
from pysie2d import BIESolver, Geometry, Material
from pysie2d.reference import mie

TWOPI = 2 * np.pi
RAD, NCORE, LAM = 200.0, 1.5, 600.0
mat = Material(n_core=NCORE, n_clad=1.0, pol=2)

x = TWOPI * 1.0 * RAD / LAM
eff = mie.efficiencies(x, complex(NCORE))
QEXT_MIE = eff["Q_ext_TE"]
print(f"Mie reference: x = {x:.6f}, qext = {QEXT_MIE:.10f}\n")


def build(theta):
    return Geometry.gielis(rad=RAD, n_pts=len(theta), m=0, theta=np.sort(theta))


def qext(geom):
    r = BIESolver(geom, mat).scatter(wavelength=LAM)
    amp, _ = r.far_field(4000)
    d = TWOPI / 3999.0
    return amp[int((TWOPI - np.deg2rad(r.angle)) / d)].imag / (r.wnum_bg * 2 * RAD)


def kappa(geom):
    return np.linalg.cond(BIESolver(geom, mat).assemble(LAM))


def kappa_scaled(geom):
    """kappa after the best single scalar rescaling of the chi block."""
    M = BIESolver(geom, mat).assemble(LAM)
    nn = len(geom.f)
    return min(np.linalg.cond(M * np.r_[np.ones(nn), a * np.ones(nn)][None, :])
               for a in np.logspace(0, 3, 40))


NN = 120
base = np.linspace(0, TWOPI, NN, endpoint=False) + np.pi / NN

print("PART A -- smooth one-parameter perturbation theta -> theta + eps*sin(2 theta)")
print(f"{'eps':>8} {'kappa':>12} {'kappa_scaled':>14} {'rel err qext':>14}")
for eps in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, -0.10, -0.20):
    th = base + eps * np.sin(2 * base)
    g = build(th)
    print(f"{eps:8.2f} {kappa(g):12.4e} {kappa_scaled(g):14.4e}"
          f" {abs(qext(g)-QEXT_MIE)/QEXT_MIE:14.4e}")

print("\nPART B -- actually minimise kappa over the node angles")
NOPT = 60
b0 = np.linspace(0, TWOPI, NOPT, endpoint=False) + np.pi / NOPT


def unpack(z):
    """Map free params to a strictly increasing, 2pi-periodic node set."""
    d = np.exp(np.r_[0.0, z])
    d = TWOPI * d / d.sum()
    return np.cumsum(d) - d[0]


def obj(z):
    try:
        return np.log(kappa(build(unpack(z))))
    except Exception:
        return 1e3


z0 = np.zeros(NOPT - 1)
g0 = build(unpack(z0))
print(f"start:  kappa = {kappa(g0):.5e}   rel err = {abs(qext(g0)-QEXT_MIE)/QEXT_MIE:.4e}")
res = minimize(obj, z0, method="Powell",
               options=dict(maxfev=8000, xtol=1e-3, ftol=1e-4))
g1 = build(unpack(res.x))
print(f"optim:  kappa = {kappa(g1):.5e}   rel err = {abs(qext(g1)-QEXT_MIE)/QEXT_MIE:.4e}")
sp = np.diff(np.r_[unpack(res.x), unpack(res.x)[0] + TWOPI])
print(f"        node spacing: min {sp.min():.4f}  max {sp.max():.4f}  "
      f"uniform would be {TWOPI/NOPT:.4f}")
print(f"        kappa reduction factor {kappa(g0)/kappa(g1):.3f}, "
      f"error changed by factor "
      f"{(abs(qext(g1)-QEXT_MIE))/(abs(qext(g0)-QEXT_MIE)):.3f}")
