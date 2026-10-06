"""1550 nm port of the quasi-BIC cell: LPCVD SiN teeth embedded in SiO₂.

Same topology as qbic.py (two superellipse teeth per cell, widths w and w − δw,
Γ point, TE), rescaled: height 350 nm (the film), fill 50 %, Λ tuned so the
folded band-edge mode sits at 1550 nm. Background is homogeneous SiO₂ — no
substrate, pysie2d has no layered background.

ε″ = 1.1e-6 from 0.1 dB/cm bulk: α = 2.3 m⁻¹, n″ = αλ/4π = 2.8e-7,
ε″ = 2n′n″. Im λ is linear in ε″ (sweep.txt: 0.31736 → 0.63471 for
ε″ 0.01 → 0.02), so δw_crit ∝ √ε″ and other losses rescale by hand.
"""

import sys
import time

import numpy as np

from pysie2d import Cluster, ClusterBIESolver, Geometry, Material
from pysie2d.beyn import beyn_modes

from periodic_kernel import assemble_periodic

N_SIN, N_SIO2 = 1.996, 1.444
EPS_I = 1.1e-6
HEIGHT = 350.0
NN = 144


def tooth(x0, width, nn=NN):
    # superellipse |x/(w/2)|^8 + |z/(H/2)|^8 = 1; Gielis x half-width = b·rad
    return Geometry.gielis(
        rad=HEIGHT / 2, n_pts=nn, m=4, n1=8, n2=8, n3=8, a=1.0,
        b=width / HEIGHT, x0=x0,
    )


def cell(L, dw, epsi, nn=NN):
    w = L / 4
    geoms = [tooth(-L / 4, w, nn), tooth(L / 4, w - dw, nn)]
    mats = [Material(n_core=N_SIN, n_clad=N_SIO2, pol=2, epsi=epsi) for _ in geoms]
    return ClusterBIESolver(Cluster(geoms), mats)


def modes(L, dw, epsi, z_lo, z_hi, nn=NN, n_quad=12):
    s = cell(L, dw, epsi, nn)
    t0 = time.perf_counter()
    bm = beyn_modes(
        lambda lam: assemble_periodic(s, lam, 0.0, L),
        z_lo, z_hi, n_quad_per_side=n_quad, max_workers=2,
    )
    return bm, time.perf_counter() - t0


if __name__ == "__main__":
    L, dw, epsi = (float(a) for a in sys.argv[1:4])
    lo, hi = complex(sys.argv[4]), complex(sys.argv[5])
    nn = int(sys.argv[6]) if len(sys.argv) > 6 else NN
    bm, dt = modes(L, dw, epsi, lo, hi, nn)
    print(f"L={L} δw={dw} ε''={epsi} nn={nn} rank={bm.rank} gap={bm.max_gap:.1e} ({dt:.0f} s)")
    for lam in bm.eigenvalues:
        print(f"  λ = {lam.real:.6f} + {lam.imag:.4e}i   Q = {lam.real / (2 * lam.imag):.4g}")
