"""Open item 2: a flat facet lying on the interface (Z_min -> 0 over a span).
Superellipse |x/R|^8 + |z/R|^8 = 1 (R = 100, D = 200) whose flat bottom sits at
`gap` above a PEC half-space. Half-space solve (Sommerfeld blocks, gemm.py) vs
the independent mirror-cluster reference (v0.8 ClusterBIESolver, particle +
image), converged at NN_REF. Reports path size M, reflected-block time, and the
field error in the cover for each (gap, nn). Usage: t14_flat_facet.py [pol ...]"""
import sys, time, warnings
import numpy as np
from hs import *
from gemm import reflected_blocks
sys.path.insert(0, "/Users/claudiosc/Documents/sie/src")
from pysie2d import Geometry, BIESolver, Cluster, ClusterBIESolver
from pysie2d.material import Material
from pysie2d.sources import line_dipole_rhs
from pysie2d.fields import _representation_at
warnings.simplefilter("ignore")

RP = 100.0
NN_REF = 768


def shape(nn, z0):
    return Geometry.gielis(RP, nn, m=4, n1=8, n2=8, n3=8, z0=z0)


def obs(zc):
    return np.array([-400., 0., 250., 500.]), np.array([60., zc + 2.5*RP, 40., 700.])


def halfspace(pol, lam, nn, gap):
    mat = Material(n_core=2.0, pol=pol)
    k0 = 2*PI/lam; zc = RP + gap
    geo = shape(nn, zc)
    xs, zs = 60.0, zc + RP + 80.0
    M = BIESolver(geo, mat).assemble(lam)
    t0 = time.perf_counter()
    m1, m2, Mq, _ = reflected_blocks(k0, 1.0, 'pec', pol, geo.f, geo.g, geo.df, geo.dg, 0.0)
    tb = time.perf_counter() - t0
    M[:nn, :nn] += m1; M[:nn, nn:] += m2
    rhs = line_dipole_rhs(nn, k0, geo.f, geo.g, xs, zs)
    rhs[:nn] += pec_exact(geo.f - xs, geo.g + zs, k0, pol)[0]
    ei = np.linalg.solve(M, rhs)
    xo, zo = obs(zc)
    fh = _representation_at(ei, nn, geo.f, geo.df, geo.g, geo.dg, geo.delt, k0, xo, zo)
    for i in range(len(xo)):  # reflected part of the representation, closed form over PEC
        G, Gx, Gz = pec_exact(xo[i] - geo.f, zo[i] + geo.g, k0, pol)
        fh[i] += -np.sum(geo.delt*((Gx*geo.dg - Gz*geo.df)*ei[:nn] + G*ei[nn:]))
    return fh, Mq, tb


def mirror(pol, lam, nn, gap):
    mat = Material(n_core=2.0, pol=pol)
    k0 = 2*PI/lam; zc = RP + gap; sign = -1.0 if pol == 2 else 1.0
    geo, img = shape(nn, zc), shape(nn, -zc)
    xs, zs = 60.0, zc + RP + 80.0
    A = ClusterBIESolver(Cluster([geo, img]), [mat, mat], pol=pol)._assemble(lam)
    r = np.concatenate([line_dipole_rhs(nn, k0, g.f, g.g, xs, zs) + sign*line_dipole_rhs(nn, k0, g.f, g.g, xs, -zs)
                        for g in (geo, img)])
    ec = np.linalg.solve(A, r)
    xo, zo = obs(zc)
    return sum(_representation_at(ec[2*nn*p:2*nn*(p+1)], nn, g.f, g.df, g.g, g.dg, g.delt, k0, xo, zo)
               for p, g in enumerate((geo, img)))


pols = [int(a) for a in sys.argv[1:]] or [2, 1]
for pol in pols:
    for lam in (633.0, 633.0*(1 + 0.05j)):
        for gap in (50.0, 20.0, 10.0, 5.0, 2.0):
            ref = mirror(pol, lam, NN_REF, gap)
            row = []
            for nn in (64, 128, 256, 512):
                fh, Mq, tb = halfspace(pol, lam, nn, gap)
                fm = mirror(pol, lam, nn, gap)
                e = np.max(abs(fh - ref))/np.max(abs(ref))
                e_same = np.max(abs(fh - fm))/np.max(abs(fm))
                row.append(f"nn={nn} M={Mq} {tb*1e3:.0f}ms err={e:.0e} (vs mirror@nn {e_same:.0e})")
            print(f"pol={pol} lam={lam:.0f} gap={gap:4.0f} D/gap={2*RP/gap:5.0f} | " + " | ".join(row), flush=True)
