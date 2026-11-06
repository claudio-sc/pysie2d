"""Eigenpermittivity extraction on pysie2d's BIE operator by Beyn's method in eps.

At fixed real vacuum wavelength, M(eps) = [[A, B], [C(k_c), eta(eps) D(k_c)]] with
k_c = ri(eps)*k_bg.  The exterior rows [A B] -- including any HalfSpace/Stack
reflected blocks -- do not depend on eps, so they are built ONCE per wavelength.
"""
import time
import numpy as np
from pysie2d import BIESolver, Geometry, Material
from pysie2d.kernels import assemble_matrix
from pysie2d.beyn import beyn_modes, newton_refine

BRANCHES = {
    "principal": lambda e: np.sqrt(complex(e)),          # cut (-inf,0]; Im ri<0 for Im eps<0
    "upper":     lambda e: 1j*np.sqrt(-complex(e)),      # cut [0,inf);  Im ri>0 off the cut
}

class EpsOperator:
    def __init__(self, geom, pol, wavelength, n_clad=1.0, background=None, branch="upper"):
        self.g, self.pol, self.lam = geom, pol, wavelength
        self.kbg = 2*np.pi*n_clad/wavelength
        self.nn = geom.n_pts
        self.ri = BRANCHES[branch]
        t0 = time.perf_counter()
        # any material: the exterior rows never see it
        solver = BIESolver(geom, Material(n_core=1.5, n_clad=n_clad, pol=pol), background=background)
        self.top = solver.assemble(wavelength)[: self.nn].copy()
        self.t_top = time.perf_counter() - t0
        self.solver = solver

    def bottom(self, eps):
        g = self.g
        m = assemble_matrix(self.pol, g.n_pts, g.f, g.g, g.df, g.dg, g.ddf, g.ddg,
                            self.kbg, self.ri(eps), complex(eps))
        return m[self.nn:]

    def M(self, eps):
        return np.vstack([self.top, self.bottom(eps)])

    def dM(self, eps, h=1e-6):
        # only the interior rows move with eps; central difference is enough for polishing
        out = np.zeros((2*self.nn, 2*self.nn), complex)
        out[self.nn:] = (self.bottom(eps + h) - self.bottom(eps - h)) / (2*h)
        return out

def solve_box(op, lo, hi, nq=32, n_probe=16, refine=True):
    t0 = time.perf_counter()
    bm = beyn_modes(op.M, lo, hi, n_quad_per_side=nq, n_probe=n_probe, max_workers=1)
    t_beyn = time.perf_counter() - t0
    eigs = bm.eigenvalues
    if refine:
        eigs = np.array([newton_refine(op.M, op.dM, e, bm.vectors[:, k], lo, hi).eigenvalue
                         for k, e in enumerate(eigs)])
    return eigs, bm, t_beyn
