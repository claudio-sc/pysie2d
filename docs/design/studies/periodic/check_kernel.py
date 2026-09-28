"""Steps 1–3 of the spike: kernel vs direct sum, Bloch folding, R + T = 1."""

import time

import numpy as np

from pysie2d import Cluster, ClusterBIESolver, Geometry, Material
from pysie2d.sources import plane_wave_rhs

from periodic_kernel import (
    assemble_periodic,
    orders,
    periodic_blocks,
    periodic_blocks_direct,
)

NN = 96  # artifact study resolution for n = 8 teeth


def tooth(x0, width=85.0, nn=NN):
    # Gielis b sets the x half-width, a the z one (measured)
    # superellipse |x/42.5|^8 + |z/100|^8 = 1 as Gielis m=4, n=8
    return Geometry.gielis(
        rad=100.0, n_pts=nn, m=4, n1=8, n2=8, n3=8, a=1.0, b=width / 200.0, x0=x0
    )


def solver(geoms, epsi=0.01):
    mats = [Material(n_core=3.0, n_clad=1.0, pol=2, epsi=epsi) for _ in geoms]
    return ClusterBIESolver(Cluster(geoms), mats)


def step1():
    # Direct sum converges only for Im k > 0, i.e. Im λ < 0 (the non-QNM side);
    # the Veysoglu integral is analytic in k, so agreement there carries over.
    L = 170.0
    t = tooth(0.0)
    for lam, kb, n_max in [(600 - 60j, 0.0, 400), (600 - 60j, 0.004, 400), (600 - 6j, 0.0, 4000)]:
        k = 2 * np.pi / lam
        t0 = time.perf_counter()
        a1, a2 = periodic_blocks(k, kb, L, t, t)
        dt = time.perf_counter() - t0
        b1, b2 = periodic_blocks_direct(k, kb, L, t, t, n_max)
        e1 = np.abs(a1 - b1).max() / np.abs(b1).max()
        e2 = np.abs(a2 - b2).max() / np.abs(b2).max()
        print(f"step1 λ={lam} kb={kb}: M1 rel err {e1:.1e}, M2 rel err {e2:.1e}, {dt*1e3:.0f} ms")


def step2(lam=612.0 + 0.5j, kb=0.003):
    # One tooth at period L == two identical teeth at period 2L, at the same k_B:
    # the eigenvalues of the one-tooth problem must appear in the folded one.
    L = 170.0
    s1 = solver([tooth(0.0)])
    s2 = solver([tooth(-L / 2), tooth(L / 2)])
    # Bloch solution on the 2L cell is ψ on tooth 0 and e^{ik_B L}ψ on tooth 1
    m1 = assemble_periodic(s1, lam, kb, L)
    m2 = assemble_periodic(s2, lam, kb, 2 * L)
    rhs1 = plane_wave_rhs(NN, np.rad2deg(np.arcsin(kb / (2 * np.pi / lam.real))), 2 * np.pi / lam, s1.cluster.geometries[0].f, s1.cluster.geometries[0].g)
    x1 = np.linalg.solve(m1, rhs1)
    x0 = np.concatenate([x1 * np.exp(-1j * kb * L / 2), x1 * np.exp(1j * kb * L / 2)])
    # tooth(-L/2) is tooth(0) shifted: its nodes are f − L/2, so ψ picks e^{−ik_B L/2}
    res = np.abs(m2 @ x0 - np.concatenate([rhs1 * np.exp(-1j * kb * L / 2), rhs1 * np.exp(1j * kb * L / 2)])).max()
    print(f"step2 folding residual {res:.1e} (|rhs| = 1)")


def step3():
    L = 340.0
    s = solver([tooth(-85.0), tooth(85.0, width=70.0)], epsi=0.0)
    for lam in (520.0, 611.0, 700.0):
        k = 2 * np.pi / lam
        m = assemble_periodic(s, lam, 0.0, L)
        rhs = np.concatenate([plane_wave_rhs(g.n_pts, 0.0, k, g.f, g.g) for g in s.cluster.geometries])
        # cluster layout is [φ0 χ0 φ1 χ1]; plane_wave_rhs already zeroes χ
        ei = np.linalg.solve(m, rhs)
        R, T = orders(s, ei, lam, 0.0, L, 0.0)
        print(f"step3 λ={lam}: R={R:.6f} T={T:.6f} R+T-1={R+T-1:.1e}")


if __name__ == "__main__":
    step1()
    step2()
    step3()
