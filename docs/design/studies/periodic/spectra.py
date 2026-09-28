"""R/T/A spectra across the quasi-BIC and the Cramér–Rao bound on ε″ against δw.

Observables: zeroth-order R and T at normal incidence, 1 % relative noise (the
artifact's convention). Parameters (ε″, ε′): ε′ is the nuisance that shifts
the resonance and could masquerade as linewidth. Jacobian by forward
difference; R, T are analytic in ε, step 1e-5 keeps truncation ~1e-5 relative.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pysie2d import Cluster, ClusterBIESolver, Material
from pysie2d.sources import plane_wave_rhs

from check_kernel import tooth
from periodic_kernel import assemble_periodic, orders

L, NN = 340.0, 144
EPS_R, EPS_I = 9.0, 0.01
H = 1e-5
N_LAM = 31
DWS = [2.0, 4.0, 7.0, 10.0, 13.0, 16.0]
# Re λ(δw) and Im λ(δw) interpolated from the sweep (ε″ = 0.01)
SWEEP_DW = [0.0, 4.0, 8.0, 16.0]
SWEEP_RE = [607.648, 604.443, 600.813, 592.080]
SWEEP_IM = [0.31736, 0.35988, 0.50791, 1.23616]


def rta(dw, lam, eps_r, eps_i):
    geoms = [tooth(-85.0, nn=NN), tooth(85.0, width=85.0 - dw, nn=NN)]
    mats = [Material(n_core=np.sqrt(eps_r), n_clad=1.0, pol=2, epsi=eps_i)] * 2
    s = ClusterBIESolver(Cluster(geoms), mats)
    k = 2 * np.pi / lam
    rhs = np.concatenate([plane_wave_rhs(g.n_pts, 0.0, k, g.f, g.g) for g in geoms])
    ei = np.linalg.solve(assemble_periodic(s, lam, 0.0, L), rhs)
    R, T = orders(s, ei, lam, 0.0, L, 0.0)
    return R, T


def run(dw):
    re = np.interp(dw, SWEEP_DW, SWEEP_RE)
    im = np.interp(dw, SWEEP_DW, SWEEP_IM)
    lams = re + im * np.linspace(-5, 5, N_LAM)
    jobs = [(lam, er, ei) for lam in lams for er, ei in
            [(EPS_R, EPS_I), (EPS_R, EPS_I + H), (EPS_R + H, EPS_I)]]
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(lambda j: rta(dw, *j), jobs))
    out = np.array(out).reshape(N_LAM, 3, 2)
    y = out[:, 0, :].ravel()
    jac = np.stack([(out[:, 1, :] - out[:, 0, :]).ravel() / H,
                    (out[:, 2, :] - out[:, 0, :]).ravel() / H], axis=1)
    sig = 0.01 * np.abs(y)
    fim = (jac / sig[:, None]).T @ (jac / sig[:, None])
    crb = np.sqrt(np.diag(np.linalg.inv(fim)))
    crb_known = 1 / np.sqrt(fim[0, 0])  # ε′ known exactly
    return dict(dw=dw, lam=lams.tolist(), R=out[:, 0, 0].tolist(),
                T=out[:, 0, 1].tolist(), crb=crb[0], crb_known=crb_known,
                pole=[re, im])


if __name__ == "__main__":
    res = []
    for dw in DWS:
        r = run(dw)
        print(f"δw={dw}: σ(ε″)={r['crb']:.2e}  (ε′ known {r['crb_known']:.2e})  "
              f"Amax={max(1 - a - b for a, b in zip(r['R'], r['T'])):.3f}", flush=True)
        res.append(r)
    json.dump(res, open("spectra.json", "w"))
