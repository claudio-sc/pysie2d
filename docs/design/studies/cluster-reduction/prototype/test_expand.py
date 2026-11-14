import numpy as np, sys, time
sys.path.insert(0, ".")
from epsbeyn import EpsOperator, solve_box
from test_stack import shifted, SLAB, N_CLAD, lam, pol
from pysie2d import Geometry, Material
from pysie2d.solver import ScatterResult
from pysie2d.sources import plane_wave_rhs
import os
HERE = os.path.dirname(os.path.abspath(__file__))
nn = 160
geom = shifted(Geometry.gielis(rad=90.0, n_pts=nn, m=3, n1=4.0, n2=6.0, n3=6.0), 140.0)
op = EpsOperator(geom, pol, lam, n_clad=N_CLAD, background=SLAB, branch="upper")
# where did the second member of the lower pair go?
for lo, hi in ((-0.95-0.5j, -0.40-0.0005j), (-1.15-0.05j, -0.87-0.0005j)):
    try:
        eigs, bm, _ = solve_box(op, lo, hi, nq=40, n_probe=24)
        print(f"box {lo}..{hi}: rank={bm.rank}", np.round(np.sort_complex(eigs), 6))
    except ValueError as e: print(f"box {lo}..{hi}: {str(e)[:80]}")
modes = []
for lo, hi in ((-6.0-1.0j, -1.15-0.004j), (-0.87-1.0j, -0.05-0.004j)):
    modes += list(solve_box(op, lo, hi, nq=32, n_probe=16)[0])
modes = np.array(modes)
# physical excitation: plane wave + its reflection by the stack (exterior rows only)
kbg = 2*np.pi*N_CLAD/lam
b = plane_wave_rhs(nn, 0.0, kbg, geom.f, geom.g)
b[:nn] += op.solver._reflection(lam).plane_wave(0.0, geom.f, geom.g)
def s_tm0(x, eps_rel):
    r = ScatterResult(x, geom, Material.from_eps(eps_rel*N_CLAD**2, n_clad=N_CLAD, pol=pol), lam, 0.0, SLAB,
                      excitation="plane_wave")
    return r.mode_amplitudes(x_ref=0.0).s_plus[0]
# residues of the driven observable at each mode: L(v) (u^H b) / (u^H M'(eps_m) v)
amps = []
for e in modes:
    u, s, vh = np.linalg.svd(op.M(e)); uu, vv = u[:, -1], vh[-1].conj()
    amps.append(s_tm0(vv, e) * (uu.conj() @ b) / (uu.conj() @ op.dM(e) @ vv))
    print(f"mode {e:.6f}: |u_int|/|u|={np.linalg.norm(uu[nn:]):.2f}  weight |a_m|={abs(amps[-1]):.2e}")
amps = np.array(amps)
eps_sweep = np.linspace(-3.0, -0.2, 141) + 0.02j       # passive: a weakly lossy metal
t0 = time.perf_counter()
exact = np.array([s_tm0(np.linalg.solve(op.M(e), b), e) for e in eps_sweep])
t_exact = time.perf_counter() - t0
poles = np.array([np.sum(amps/(e - modes)) for e in eps_sweep])
# smooth remainder: R(eps) ~ c0 + c1 eps + c2 eps^2 from 3 exact solves
ks = [0, 70, 140]
A = np.vander(eps_sweep[ks], 3, increasing=True)
c = np.linalg.solve(A, exact[ks] - poles[ks])
fitted = poles + np.vander(eps_sweep, 3, increasing=True) @ c
rel = lambda y: np.max(np.abs(y - exact))/np.max(np.abs(exact))
print(f"exact sweep: {eps_sweep.size} solves, {t_exact:.1f}s")
print(f"pole sum only ({modes.size} modes): max rel err = {rel(poles):.1e}")
print(f"pole sum + quadratic remainder (3 exact solves): max rel err = {rel(fitted):.1e}")
np.savez(os.path.join(HERE, "expand.npz"), eps=eps_sweep, exact=exact, poles=poles, fitted=fitted, modes=modes)
