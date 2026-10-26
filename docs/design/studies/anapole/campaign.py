"""Dark-state campaign: minimise c_sca / area of a Si wire in air at 1550 nm, pol = 2.

c_sca/area alone is minimised by shrinking the particle (Rayleigh: c_sca ∝ A²), so
the area is FIXED per run and swept over a ladder of equivalent radii. Each run
minimises log10(c_sca/A) over M cosine orders (shapes.py) under the perimeter cap
L ≤ 10 interior wavelengths, from the circle plus random starts. The circle of the
same area is the reference every design is compared against.

Run:  uv run python docs/design/studies/anapole/campaign.py   (→ results/campaign.json)
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from pysie2d.reference.mie import efficiencies  # noqa: E402
from shapes import L_MAX, LAM0, N_SI, c_sca, perimeter  # noqa: E402

M = 10  # cosine orders
NN = 192  # ≥ 19 nodes per interior wavelength at the cap; checked against 384 at the end
N_STARTS = 8
R_EQ = np.arange(120.0, 701.0, 20.0)  # equivalent radii (nm); 700 nm circle is at L = 0.99 L_MAX
WORKERS = 4  # performance cores on this machine
FD_H = 1e-6  # central difference on log10 c_sca: O(h²) ≈ 1e-12 truncation vs ~1e-14 noise / h


def objective(a, area):
    return np.log10(c_sca(a, area, NN) / area)


def grad(a, area):
    g = np.empty_like(a)
    for k in range(len(a)):
        e = np.zeros_like(a)
        e[k] = FD_H
        g[k] = (objective(a + e, area) - objective(a - e, area)) / (2 * FD_H)
    return g


def run(job):
    r_eq, start = job
    area = np.pi * r_eq**2
    rng = np.random.default_rng(int(r_eq) * 100 + start)
    a0 = np.zeros(M) if start == 0 else rng.normal(0, 0.12, M) / np.arange(1, M + 1)
    # Pull a random start inside the cap: scale its deformation down until it fits.
    while perimeter(a0, area) > L_MAX:
        a0 *= 0.7
    t = time.perf_counter()
    res = minimize(
        objective, a0, args=(area,), jac=grad, method="SLSQP",
        bounds=[(-0.6, 0.6)] * M,
        constraints=[{"type": "ineq", "fun": lambda a: (L_MAX - perimeter(a, area)) / L_MAX}],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    return {
        "r_eq": float(r_eq), "start": start, "a": res.x.tolist(), "log10_csca_per_area": float(res.fun),
        "c_sca": float(c_sca(res.x, area, NN)), "perimeter": perimeter(res.x, area),
        "nit": int(res.nit), "success": bool(res.success), "message": res.message,
        "seconds": time.perf_counter() - t,
    }


def main():
    # Anchor: the circle at nn = NN against analytic Mie (Q = C/(2R) in 2-D).
    x = 2 * np.pi * 300.0 / LAM0
    q_mie = efficiencies(x, N_SI)["Q_sca_TE"]
    c_bie = c_sca(np.zeros(M), np.pi * 300.0**2, NN)
    print(f"anchor: circle R=300 c_sca BIE {c_bie:.12f} Mie {q_mie * 600:.12f} rel {abs(c_bie / (q_mie * 600) - 1):.1e}")

    jobs = [(r, s) for r in R_EQ for s in range(N_STARTS)]
    out = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as ex:
        for k, r in enumerate(ex.map(run, jobs)):
            out.append(r)
            print(f"[{k + 1}/{len(jobs)} {time.perf_counter() - t0:6.0f}s] R_eq {r['r_eq']:.0f} start {r['start']}: "
                  f"log10(c/A) {r['log10_csca_per_area']:+.3f}  L/Lmax {r['perimeter'] / L_MAX:.3f}  nit {r['nit']}", flush=True)

    summary = []
    for r_eq in R_EQ:
        area = np.pi * r_eq**2
        runs = [o for o in out if o["r_eq"] == r_eq]
        best = min(runs, key=lambda o: o["log10_csca_per_area"])
        c_circ = c_sca(np.zeros(M), area, NN)
        a = np.array(best["a"])
        c384 = c_sca(a, area, 384)
        summary.append({**best, "c_sca_circle": c_circ, "gain_vs_circle": c_circ / best["c_sca"],
                        "c_sca_nn384": c384, "nn_rel_diff": abs(c384 / best["c_sca"] - 1),
                        "n_starts_within_0p1dec": sum(o["log10_csca_per_area"] < best["log10_csca_per_area"] + 0.1 for o in runs)})
        s = summary[-1]
        print(f"R_eq {r_eq:4.0f}: best c/A {s['c_sca'] / area:.3e}  circle {c_circ / area:.3e}  "
              f"×{s['gain_vs_circle']:.1f} darker  L/Lmax {s['perimeter'] / L_MAX:.3f}  nn384 rel {s['nn_rel_diff']:.1e}")

    meta = {"lam0": LAM0, "n_core": N_SI, "pol": 2, "L_max": L_MAX, "M": M, "nn": NN, "n_starts": N_STARTS}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/campaign.json").write_text(json.dumps({"meta": meta, "best": summary, "all": out}, indent=1))


if __name__ == "__main__":
    main()
