"""Step 4: quasi-BIC of a two-tooth cell (L = 340 nm, widths 85 and 85 − δw), Γ point.

At δw = 0 the lattice is the 170 nm grating, whose X-point band-edge modes fold
to Γ below the light line of the 170 nm lattice: bound, Q_rad = ∞. δw ≠ 0 opens
the zeroth order with Q_rad ∝ δw⁻². Loss adds a δw-independent width, so the
design knob is δw at which Q_rad matches Q_abs.
"""

import sys
import time

import numpy as np

from pysie2d.beyn import beyn_modes

from check_kernel import solver, tooth
from periodic_kernel import assemble_periodic

L = 340.0
NN = 144  # R+T−1 = 8e-10 at 144 vs 1.6e-6 at 96 (check_kernel, λ = 611 nm)


def cell(dw, epsi):
    return solver([tooth(-85.0, nn=NN), tooth(85.0, width=85.0 - dw, nn=NN)], epsi=epsi)


def modes(dw, epsi, z_lo, z_hi, n_quad=12, kb=0.0):
    s = cell(dw, epsi)
    t0 = time.perf_counter()
    bm = beyn_modes(
        lambda lam: assemble_periodic(s, lam, kb, L),
        z_lo,
        z_hi,
        n_quad_per_side=n_quad,
        max_workers=2,  # the (144,144,400) u-arrays are ~130 MB each
    )
    return bm, time.perf_counter() - t0


if __name__ == "__main__":
    dw, epsi = float(sys.argv[1]), float(sys.argv[2])
    lo, hi = complex(sys.argv[3]), complex(sys.argv[4])
    kb = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    bm, dt = modes(dw, epsi, lo, hi, kb=kb)
    print(f"δw={dw} ε''={epsi} kb={kb} rank={bm.rank} gap={bm.max_gap:.1e} ({dt:.0f} s)")
    for lam in bm.eigenvalues:
        print(f"  λ = {lam.real:.4f} + {lam.imag:.5f}i   Q = {lam.real / (2 * lam.imag):.1f}")
