import sys
sys.path.insert(0, "/home/claude/pysie2d-0.5.0/src")
import numpy as np
from pysie2d import BIESolver, Geometry, Material

TWOPI = 2 * np.pi
SHAPE = dict(rad=200, m=6, n1=6, n2=12, n3=12)
mat = Material(n_core=1.5, n_clad=1.0, pol=2)
lam = 600.0


def arc(nn):
    return Geometry.gielis(n_pts=nn, **SHAPE)


def unif(nn):
    th = np.linspace(0, TWOPI, nn, endpoint=False) + np.pi / nn
    return Geometry.gielis(n_pts=nn, theta=th, **SHAPE)


def w_exact(g):
    gam = np.sqrt(g.df**2 + g.dg**2)
    g.delt = TWOPI / (gam * np.sum(1.0 / gam))
    return g


def w_centred(g):
    th = g.theta
    nxt = np.append(th[1:], th[0] + TWOPI)
    prv = np.append(th[-1] - TWOPI, th[:-1])
    g.delt = 0.5 * (nxt - prv)
    return g


# ---------------------------------------------------------------- part 1
# Pure quadrature: integrate a smooth periodic F over theta with the weights
# the assembly actually uses.  Exact value from a very fine uniform grid.
print("PART 1 -- pure quadrature error, integrand F(theta)=exp(sin 3th)*cos(2th)")


def F(th):
    return np.exp(np.sin(3 * th)) * np.cos(2 * th)


exact = np.trapezoid(F(np.linspace(0, TWOPI, 2_000_001)),
                     dx=TWOPI / 2_000_000)
print(f"exact integral = {exact:.14f}")
print(f"{'nn':>6} {'current delt':>13} {'rate':>6} {'centred':>13} {'rate':>6}"
      f" {'exact 1/gamma':>14} {'rate':>6}")
prev = {}
for nn in (100, 200, 400, 800, 1600):
    row = {}
    row["cur"] = abs(np.sum(F(arc(nn).theta) * arc(nn).delt) - exact)
    gc = w_centred(arc(nn))
    row["ctr"] = abs(np.sum(F(gc.theta) * gc.delt) - exact)
    ge = w_exact(arc(nn))
    row["exa"] = abs(np.sum(F(ge.theta) * ge.delt) - exact)
    r = {k: (np.log2(prev[k] / v) if k in prev and v > 0 else np.nan)
         for k, v in row.items()}
    print(f"{nn:6d} {row['cur']:13.3e} {r['cur']:6.2f} {row['ctr']:13.3e}"
          f" {r['ctr']:6.2f} {row['exa']:14.3e} {r['exa']:6.2f}")
    prev = row

# ---------------------------------------------------------------- part 2
print("\nPART 2 -- solver convergence measured on qext (no angular quadrature)")


def qext(geom):
    r = BIESolver(geom, mat).scatter(wavelength=lam)
    amp, _ = r.far_field(3000)
    delthe = TWOPI / (3000 - 1.0)
    nforw = int((TWOPI - np.deg2rad(r.angle)) / delthe)
    return amp[nforw].imag / (r.wnum_bg * 2.0 * geom.rad)


ref = qext(unif(2400))
print(f"reference: uniform theta nn=2400 -> qext = {ref:.10f}")
schemes = {
    "uniform theta (midpoint)": unif,
    "arc length, current delt": arc,
    "arc length, centred delt": lambda n: w_centred(arc(n)),
    "arc length, exact 1/gamma": lambda n: w_exact(arc(n)),
}
NS = (100, 200, 400, 800)
print(f"{'scheme':28s}" + "".join(f"{n:>12d}" for n in NS) + "   rates")
for k, f in schemes.items():
    e = [abs(qext(f(n)) - ref) / abs(ref) for n in NS]
    rates = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
    print(f"{k:28s}" + "".join(f"{v:12.2e}" for v in e)
          + "   " + " ".join(f"{r:5.2f}" for r in rates))
