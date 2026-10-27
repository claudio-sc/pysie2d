"""All-direction darkness: minimise ‖T‖²_F = Σ_j sin²δ_j at fixed λ and ε.

T is the free-space T-matrix of wg_reactance.py (|n| ≤ 9). With a_n = e^{−inα} for a plane wave
at angle α, the mean over α of ‖T a‖² is ‖T‖²_F: the objective is the scattering averaged over
every incidence direction, and needs no eigenvectors (no channel tracking). Si (ε = 12.08),
1550 nm, pol 2, R_eq 500 nm at fixed area, CD 200 nm rules in the loop (campaign_500.CONS).
It cannot reach zero: only ε = 1 is invisible from every direction.

Run:  uv run python docs/design/studies/anapole/wg_alldir.py   (→ results/wg_alldir.json)
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from campaign_500 import AREA, CONS, M, feasible  # noqa: E402
from campaign_cd import width_gap_ok  # noqa: E402
from shapes import N_SI, geometry  # noqa: E402
import wg_reactance as wr  # noqa: E402

EPS = N_SI**2
N_RANDOM = 13
FD_H = 1e-6  # log10 ‖T‖² is O(1) and smooth; round-off ~1e-13/h


def tnorm(a, nn=wr.NN):
    S = wr.smatrix(geometry(np.asarray(a, float), AREA, nn), EPS)
    return float(np.sum(np.abs((S - np.eye(len(S))) / 2) ** 2))


def f(a):
    return np.log10(tnorm(a))


def grad(a):
    g = np.empty(M)
    for k in range(M):
        e = np.zeros(M)
        e[k] = FD_H
        g[k] = (f(a + e) - f(a - e)) / (2 * FD_H)
    return g


def run(job):
    name, a0 = job
    a0 = np.array(a0, float)
    while not feasible(a0):
        a0 *= 0.85
    t = time.time()
    res = minimize(f, a0, jac=grad, method="SLSQP", bounds=[(-0.6, 0.6)] * M, constraints=CONS,
                   options={"maxiter": 80, "ftol": 1e-10})
    a = res.x
    return {"start": name, "a": a.tolist(), "T2": tnorm(a), "T2_384": tnorm(a, 384), "feasible": feasible(a),
            "width_gap_ok": width_gap_ok(a, AREA), "nit": res.nit, "seconds": time.time() - t}


def main():
    c5 = json.loads((HERE / "results/campaign_500.json").read_text())
    dk = json.loads((HERE / "results/wg_dark.json").read_text())
    fs = min((o for o in c5["phase_a"] if o["feasible"]), key=lambda o: o["f"])["a"]
    an = min((o for o in dk["phase1"] if o["feasible"] and o["width_gap_ok"]), key=lambda o: o["scatt"])["a"]
    rng = np.random.default_rng(1208)
    starts = [("circle", np.zeros(M)), ("fs-dark", np.array(fs)), ("guided-dark", np.array(an))]
    starts += [(f"rnd{k}", rng.normal(0, rng.uniform(0.08, 0.25), M) / np.arange(1, M + 1) ** 1.5) for k in range(N_RANDOM)]
    t2c = tnorm(np.zeros(M))
    print(f"circle ‖T‖² {t2c:.4f}; fs-dark {tnorm(fs):.4f}; guided-dark {tnorm(an):.4f}", flush=True)
    out = []
    with ProcessPoolExecutor(4) as ex:  # owner: 4 cores
        for o in ex.map(run, [(n, a.tolist()) for n, a in starts]):
            out.append(o)
            print(f"{o['start']:12s}: ‖T‖² {o['T2']:.4f} (nn384 {o['T2_384']:.4f}) ×{t2c / o['T2']:.2f} below circle"
                  f"  feas {o['feasible']} w/g {o['width_gap_ok']} nit {o['nit']} {o['seconds']:.0f}s", flush=True)
            (HERE / "results/wg_alldir.json").write_text(json.dumps({"T2_circle": t2c, "runs": out}, indent=1))


if __name__ == "__main__":
    main()
