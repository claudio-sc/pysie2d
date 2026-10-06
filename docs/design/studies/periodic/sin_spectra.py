"""R/T/A spectra and the Cramér–Rao bound on ε″ for the 1550 nm SiN cell.

Same observables and noise model as spectra.py (zeroth-order R and T, 1 %
relative noise, ε′ fitted as nuisance). Each pole is located by Beyn rather
than interpolated: the linewidth is ~1e-4 nm, below any interpolation error.

Finite-difference steps: ε′ shifts Re λ by ~λΓΔε′/2ε′ ≈ 0.1·Δε′·λ, so a
1e-5 step (spectra.py) would move the resonance ~1e-3 nm, ten linewidths —
not a derivative. 1e-9 keeps the shift at 1e-7 nm (1e-3 linewidth) while
ΔR ≈ 1e-4 stays nine digits above the R+T−1 floor (8e-10 at nn = 144).
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pysie2d import Cluster, ClusterBIESolver, Material
from pysie2d.sources import plane_wave_rhs

from periodic_kernel import assemble_periodic, orders
from sin import EPS_I, N_SIN, N_SIO2, NN, modes, tooth

L = 942.3
H = 1e-9
N_LAM = 31
DWS = [0.25, 0.5, 0.75, 1.07, 1.5, 2.0, 3.0, 4.0]


def rta(dw, lam, eps_r, eps_i):
    w = L / 4
    geoms = [tooth(-L / 4, w), tooth(L / 4, w - dw)]
    mats = [Material(n_core=np.sqrt(eps_r), n_clad=N_SIO2, pol=2, epsi=eps_i)] * 2
    s = ClusterBIESolver(Cluster(geoms), mats)
    k = mats[0].wnum_bg(lam)
    rhs = np.concatenate([plane_wave_rhs(g.n_pts, 0.0, k, g.f, g.g) for g in geoms])
    ei = np.linalg.solve(assemble_periodic(s, lam, 0.0, L), rhs)
    return orders(s, ei, lam, 0.0, L, 0.0)


def run(dw):
    bm, _ = modes(L, dw, EPS_I, 1544 - 0.01j, 1551 + 0.05j)
    assert bm.eigenvalues.size == 1, bm.eigenvalues
    pole = complex(bm.eigenvalues[0])
    lams = pole.real + pole.imag * np.linspace(-5, 5, N_LAM)
    er = N_SIN**2
    jobs = [(lam, a, b) for lam in lams for a, b in
            [(er, EPS_I), (er, EPS_I + H), (er + H, EPS_I)]]
    with ThreadPoolExecutor(2) as ex:  # (144,144,400) u-arrays, ~130 MB each
        out = np.array(list(ex.map(lambda j: rta(dw, *j), jobs))).reshape(N_LAM, 3, 2)
    y = out[:, 0, :].ravel()
    jac = np.stack([(out[:, 1, :] - out[:, 0, :]).ravel() / H,
                    (out[:, 2, :] - out[:, 0, :]).ravel() / H], axis=1)
    sig = 0.01 * np.abs(y)
    fim = (jac / sig[:, None]).T @ (jac / sig[:, None])
    crb = np.sqrt(np.diag(np.linalg.inv(fim)))
    return dict(dw=dw, re=pole.real, im=pole.imag, lam=lams.tolist(),
                R=out[:, 0, 0].tolist(), T=out[:, 0, 1].tolist(),
                crb=crb[0], crb_known=1 / np.sqrt(fim[0, 0]))


if __name__ == "__main__":
    res = []
    for dw in DWS:
        r = run(dw)
        A = 1 - np.array(r["R"]) - np.array(r["T"])
        print(f"δw={dw}: λ={r['re']:.5f}+{r['im']:.4e}i σ(ε″)={r['crb']:.2e} "
              f"(ε′ known {r['crb_known']:.2e}) Amax={A.max():.3f}", flush=True)
        res.append(r)
        json.dump(res, open("sin_spectra.json", "w"))
