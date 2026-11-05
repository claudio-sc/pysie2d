"""Candidates for an anchor on eigenpermittivities over a substrate, where no closed form exists."""
import os, sys, warnings
import numpy as np
from pysie2d import BIESolver, Cluster, ClusterBIESolver, Geometry, HalfSpace, Material, Stack
from pysie2d.beyn import beyn_modes
from pysie2d.kernels import assemble_matrix
warnings.simplefilter("ignore")
ri = lambda e: 1j * np.sqrt(-complex(e))
class Op:
    def __init__(self, gs, pol, lam, n_clad=1.0, bg=None):
        self.gs, self.pol, self.k = gs, pol, 2 * np.pi * n_clad / lam; self.cl = Cluster(gs)
        self.full0 = ClusterBIESolver(self.cl, [Material(n_core=1.5, n_clad=n_clad, pol=pol)] * len(gs), pol=pol, background=bg)._assemble(lam)
    def __call__(self, e):
        m = self.full0.copy()
        for p, g in enumerate(self.gs):
            s_ = self.cl.slice(p); m[s_.start + g.n_pts:s_.stop, s_] = assemble_matrix(self.pol, g.n_pts, g.f, g.g, g.df, g.dg, g.ddf, g.ddg, self.k, ri(e), complex(e))[g.n_pts:]
        return m
def newton(f, e):
    h = 1e-6
    for _ in range(10):
        u, s, vh = np.linalg.svd(f(e)); w, v = u[:, -1].conj(), vh[-1].conj()
        st = (w @ f(e) @ v) / (w @ ((f(e + h) - f(e - h)) / (2 * h)) @ v); e -= st
        if abs(st) < 2e-15 * abs(e): break
    return e
def modes(f, lo, hi, nq=40, npb=20):
    bm = beyn_modes(f, lo, hi, n_quad_per_side=nq, n_probe=npb, max_workers=4)
    return np.sort_complex(np.array([newton(f, e) for e in bm.eigenvalues]))
nearest = lambda a, b: max(np.min(np.abs(b - x)) for x in a)
lam, rad = 600.0, 100.0
circle = lambda nn, x0=0.0, z0=0.0: Geometry.gielis(rad=rad, n_pts=nn, m=0, x0=x0, z0=z0)

print("=== K. transparent stack: every layer index equal to the cladding, so R(q) = 0 and M must be the free-space M")
for pol, lo, hi in ((2, 11.0 - 1.4j, 13.8 - 0.1j), (1, -2.0 - 1.5j, -0.2 - 0.25j)):
    free = Op([circle(96, z0=300.0)], pol, lam); clear = Op([circle(96, z0=300.0)], pol, lam, bg=Stack(((1.0, 150.0),), 1.0))
    a, b = modes(free, lo, hi), modes(clear, lo, hi)
    print(f"pol={pol}: free {a.size} modes, transparent stack {b.size}; max diff {nearest(a, b):.1e}; ‖M_stack−M_free‖/‖M‖ {np.linalg.norm(clear.full0[:96]-free.full0[:96])/np.linalg.norm(free.full0[:96]):.1e}", flush=True)

print("\n=== L. PEC mirror: a circle at height h over PEC vs the circle + image in free space, by parity")
for pol, lo, hi, parity in ((2, 11.0 - 1.4j, 13.8 - 0.1j, "odd (E_y=0 on the mirror)"), (1, -2.0 - 1.5j, -0.2 - 0.25j, "even (∂H_y/∂z=0)")):
    for h in (200.0, 500.0):
        pec = Op([circle(128, z0=h)], pol, lam, bg=HalfSpace.pec())
        mir = Op([circle(128, z0=h), circle(128, z0=-h)], pol, lam)
        a, b = modes(pec, lo, hi), modes(mir, lo, hi)
        print(f"pol={pol} h={h:.0f} [{parity}]: PEC {a.size} modes, mirror pair {b.size}; every PEC mode in the pair to {nearest(a, b):.1e}", flush=True)

print("\n=== M. pole from the passive driven response: Material.from_eps, ScatterResult, no contour, no branch choice")
nc = 1.444; L = 1550.0; SIN = Stack(((4.0, 400.0),), nc**2)
def shifted(g, dz, rot=17.0):
    c, s = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot)); r = lambda x, z: (c * x - s * z, s * x + c * z)
    f, gg = r(g.f, g.g); df, dg = r(g.df, g.dg); ddf, ddg = r(g.ddf, g.ddg)
    return Geometry(f=f, g=gg + dz, df=df, dg=dg, ddf=ddf, ddg=ddg, rad=g.rad, z0=dz)
geom = shifted(Geometry.gielis(rad=90.0, n_pts=160, m=3, n1=4.0, n2=6.0, n3=6.0), 140.0)
op = Op([geom], 1, L, n_clad=nc, bg=SIN)
em = modes(op, -1.6 - 0.2j, -1.1 - 0.002j, nq=32, npb=12)
print("ε-Beyn over the slab (polished):", np.round(em, 8))
target = em[np.argmin(np.abs(em.imag))]
def s0(e):
    r = BIESolver(geom, Material.from_eps(e * nc**2, n_clad=nc, pol=1), background=SIN).scatter(L, 0.0)
    return r.mode_amplitudes(x_ref=0.0).s_plus[0]
for im, half, npts, deg in ((1e-3, 0.03, 15, 3), (1e-3, 0.06, 21, 4), (1e-4, 0.03, 21, 4), (1e-4, 0.06, 31, 5)):
    es = target.real + np.linspace(-half, half, npts) + 1j * im
    fs = np.array([s0(e) for e in es])
    # ε f = p f + Σ b_k ε^k is linear in the pole p and the b_k (a pole plus a smooth remainder)
    t = (es - target.real) / half
    A = np.column_stack([fs] + [t**k for k in range(deg + 1)])
    sol, *_ = np.linalg.lstsq(A, es * fs, rcond=None)
    print(f"Im ε={im:.0e}, ±{half}, {npts} solves, degree {deg}: pole {sol[0]:.8f}; |p−ε_Beyn|={abs(sol[0]-target):.1e}", flush=True)
