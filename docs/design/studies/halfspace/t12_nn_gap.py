"""Discretisation (nn) convergence of the half-space solve vs gap: the image kernel is
smooth but near-singular at distance 2*gap, so the plain trapezoid needs spacing << 2*gap.
Quantity: scattered field at 4 cover points; reference nn = 768. PEC, R=100, lam=633."""
import numpy as np, warnings
warnings.simplefilter("ignore")
import sys
from hs import *
from gemm import reflected_blocks
sys.path.insert(0, "/Users/claudiosc/Documents/sie/src")
from pysie2d import Geometry, BIESolver
from pysie2d.material import Material
from pysie2d.sources import line_dipole_rhs
from pysie2d.fields import _representation_at
def field(pol, nn, gap, lam=633., Rp=100.):
    mat = Material(n_core=2.0, pol=pol); k=mat.wnum_bg(lam); zc=Rp+gap
    geo = Geometry.gielis(Rp, nn, m=0, z0=zc); xs, zs = 60., zc+Rp+80.
    M = BIESolver(geo, mat).assemble(lam)
    m1, m2, _, _ = reflected_blocks(k, 1.0, 'pec', pol, geo.f, geo.g, geo.df, geo.dg, 0.0)
    M[:nn,:nn]+=m1; M[:nn,nn:]+=m2
    X=geo.f-xs; Z=geo.g+zs; rhs=line_dipole_rhs(nn,k,geo.f,geo.g,xs,zs); rhs[:nn]+=pec_exact(X,Z,k,pol)[0]
    ei=np.linalg.solve(M,rhs)
    xo=np.array([-400.,0.,250.,500.]); zo=np.array([60.,zc+2.5*Rp,40.,700.])
    fh=_representation_at(ei,nn,geo.f,geo.df,geo.g,geo.dg,geo.delt,k,xo,zo)
    for i in range(4):
        G,Gx,Gz=pec_exact(xo[i]-geo.f, zo[i]+geo.g, k, pol)
        fh[i]+=-np.sum(geo.delt*((Gx*geo.dg-Gz*geo.df)*ei[:nn]+G*ei[nn:]))
    return fh
for pol in (2,1):
    for gap in (5.,10.,40.):
        ref=field(pol,768,gap)
        row=[f"nn={nn}(h/2gap={2*PI*100/nn/(2*gap):.2f}):{np.max(abs(field(pol,nn,gap)-ref))/np.max(abs(ref)):.0e}" for nn in (48,64,96,128,192,256,384)]
        print(f"pol={pol} gap={gap:.0f} | "+" ".join(row))
