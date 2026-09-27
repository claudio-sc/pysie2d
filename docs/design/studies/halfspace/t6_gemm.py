import os, time, numpy as np, warnings
from scipy.special import hankel1
from hs import *
from gemm import reflected_blocks
import sys; sys.path.insert(0, "/Users/claudiosc/Documents/sie/src")
from pysie2d import Geometry
from pysie2d.kernels import assemble_matrix
warnings.simplefilter("ignore")

def pec_blocks(k1, pol, f, g, df, dg, z_int):
    nn = len(f); hs_ = 2*PI/nn
    X = f[:, None] - f[None, :]; Z = g[:, None] + g[None, :] - 2*z_int
    G, Gx, Gz = pec_exact(X, Z, k1, pol)
    return hs_*(Gx*dg[None, :] - Gz*df[None, :]), hs_*G

print("Accuracy of GEMM blocks vs PEC closed form (Hankel image), circle; err = max|dM|/max|M| for M1, M2; growth = max(sum|A c B|)/max|G|")
for lam in [500., 500*(1+.05j)]:
  k0 = 2*PI/lam
  for pol in (2, 1):
    for Rp, gap in [(100, 5), (500, 5), (500, 20)]:
      for x_c in [0.0, 3000.0]:
        nn = 128
        geo = Geometry.gielis(Rp, nn, m=0, x0=x_c, z0=Rp+gap)
        ex1, ex2 = pec_blocks(k0, pol, geo.f, geo.g, geo.df, geo.dg, 0.0)
        out=[]
        for centre in (True, False):
          for banded in (False, True):
            m1, m2, M, Gabs = reflected_blocks(k0, 1.0, 'pec', pol, geo.f, geo.g, geo.df, geo.dg, 0.0,
                                               x_c=None if centre else 0.0, banded=banded)
            e1 = np.max(np.abs(m1-ex1))/np.max(np.abs(ex1)); e2 = np.max(np.abs(m2-ex2))/np.max(np.abs(ex2))
            out.append(f"{'c' if centre else 'raw'}{'/band' if banded else '/full'} M={M:5d} {e1:.0e} {e2:.0e} gr={np.max(Gabs)/np.max(np.abs(ex2/(2*PI/nn))):.0e}")
        print(f"lam={lam:.0f} pol={pol} R={Rp} gap={gap} x_c={x_c:.0f} | " + " | ".join(out))
