"""Does a driven scattering dip sit on the interior transmission eigenvalue (ITE)?

Superellipse |x|^4 + |z|^4 = 1 (Gielis m=4, n=4), X = k_bg*rad = 0.5, both pols.
ITE eps from the eigen-eps cross-check (validated vs MPS to ~2e-9).
Driven side: pysie2d BIE, scattered power ~ sum |c_m|^2 from multipoles(), for
  - a plane wave along x (what an experiment does),
  - the symmetry-class projection the ITE lives in:
      pol=1 (note TE): odd in x, even in z  -> pw(+x) - pw(-x)
      pol=2 (note TM): A1, D4-invariant      -> pw(+x)+pw(-x)+pw(+z)+pw(-z)
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("EIGEN_EPS_SCRIPTS", str(HERE.parents[4] / "eigen-eps/manuscripts/anapole-design-rules/scripts")))
from pysie2d_crosscheck import anapole  # noqa: E402

from pysie2d import BIESolver, Geometry, Material  # noqa: E402
from pysie2d.sources import plane_wave_rhs  # noqa: E402

P, X, NN = 4, 0.5, 256
RAD = 1.0
LAM = 2 * np.pi * RAD / X  # n_clad = 1, so k_bg*rad = X
geo = Geometry.gielis(RAD, NN, m=4, n1=P, n2=P, n3=P)

ite = {1: anapole(geo, "TE", X, 52.0, 54.5, ngrid=60), 2: anapole(geo, "TM", X, 53.0, 54.5, ngrid=60)}


def rhs_class(pol):
    if pol == 1:
        return lambda nn, k, f, g: plane_wave_rhs(nn, 90, k, f, g) - plane_wave_rhs(nn, 270, k, f, g)
    return lambda nn, k, f, g: sum(plane_wave_rhs(nn, a, k, f, g) for a in (0, 90, 180, 270))


def power(res):
    mp = res.multipoles(mmax=16)
    return float(np.sum(abs(mp.c) ** 2)), mp


for pol in (1, 2):
    e0 = ite[pol]
    es = e0 * (1 + np.linspace(-0.02, 0.02, 81))
    es = np.sort(np.r_[es, e0 * (1 + np.linspace(-1e-3, 1e-3, 41))])
    pw, cl, orders = [], [], []
    for e in es:
        s = BIESolver(geo, Material(n_core=np.sqrt(e), n_clad=1.0, pol=pol))
        pw.append(power(s.scatter(LAM, angle=90))[0])
        p_c, mp = power(s.scatter(LAM, incident_rhs=rhs_class(pol)))
        cl.append(p_c)
        orders.append(abs(mp.c))
    pw, cl = np.array(pw), np.array(cl)
    i_pw, i_cl = np.argmin(pw), np.argmin(cl)
    print(f"\npol={pol}  ITE eps = {e0:.8f}")
    print(f"  plane wave (+x): min at eps = {es[i_pw]:.6f} (rel {es[i_pw]/e0-1:+.2e}),"
          f" depth min/max = {pw[i_pw]/pw.max():.2e}")
    print(f"  class drive    : min at eps = {es[i_cl]:.6f} (rel {es[i_cl]/e0-1:+.2e}),"
          f" depth min/max = {cl[i_cl]/cl.max():.2e}")
    c = orders[i_cl]
    mmax = (len(c) - 1) // 2
    top = np.argsort(c)[::-1][:6]
    print("  class drive at its min, |c_m| largest:", ", ".join(f"m={k-mmax}:{c[k]:.1e}" for k in top))
    c = orders[i_pw]
    top = np.argsort(c)[::-1][:6]
    print("  plane wave  at its min, |c_m| largest:", ", ".join(f"m={k-mmax}:{c[k]:.1e}" for k in top))
    np.savez(Path(__file__).resolve().parent / f"results/ite_pol{pol}.npz",
             es=es, pw=pw, cl=cl, e0=e0, orders=np.array(orders))
