"""Data for the all-direction darkness page (wg_alldir.py results).

Per design: outline; scattering vs incidence angle ‖T a(α)‖² (its mean is ‖T‖²_F, checked);
eigenchannel sin²δ_j at Si; ‖T‖² vs λ and vs ε; free-space near fields for two incidences;
interior eigenvalues (wg_interior_check.roots) for the best design; guided P_scatt over the slab.

Run:  uv run --with mpmath python docs/design/studies/anapole/wg_alldir_page_data.py   (→ results/wg_alldir_page.json)
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import wg_reactance as wr  # noqa: E402
from shapes import N_SI, geometry  # noqa: E402
from wg_dark import AREA, LAM, solve as stack_solve  # noqa: E402
from pysie2d import BIESolver, Material  # noqa: E402

EPS = N_SI**2
K0 = 2 * np.pi / LAM


def t2(geo, eps, lam=LAM):
    if lam == LAM:
        S = wr.smatrix(geo, eps)
    else:  # wr.smatrix is at LAM; rebuild at another wavelength
        s = BIESolver(geo, Material(n_core=float(np.sqrt(eps)), n_clad=1.0, pol=2))
        T = np.empty((len(wr.N), len(wr.N)), complex)
        for j, n in enumerate(wr.N):
            T[:, j] = s.scatter(lam, incident_rhs=wr.regular_rhs(n, geo.x0, geo.z0)).multipoles(mmax=wr.M).c
        return float(np.sum(abs(T) ** 2)), T
    T = (S - np.eye(len(S))) / 2
    return float(np.sum(abs(T) ** 2)), T


def design(job):
    key, label, a = job
    geo = geometry(np.asarray(a, float), AREA, wr.NN)
    T2, T = t2(geo, EPS)
    alphas = np.linspace(0, 360, 181)
    ang = []
    for al in alphas:
        th = np.deg2rad(al)
        a_in = wr.drive_coeffs(lambda x, z: np.exp(1j * K0 * (x * np.sin(th) - z * np.cos(th))), geo.x0, geo.z0)
        ang.append(float(np.sum(abs(T @ a_in) ** 2)))
    S = np.eye(len(T)) + 2 * T
    chans = []
    for b, Q in wr.B.items():
        d, V = wr.phases(S, Q)
        for dj, vj in zip(d, V.T):
            if np.sin(dj) ** 2 > 1e-6:
                chans.append({"block": b, "delta": float(dj), "s2": float(np.sin(dj) ** 2),
                              "n": int(np.abs(wr.N)[np.argmax(abs(vj))])})
    chans.sort(key=lambda c: -c["s2"])
    lams = np.linspace(1300, 1800, 51)
    epss = np.linspace(4.0, 20.0, 81)
    X, Z = np.meshgrid(np.linspace(-1200, 1200, 121), np.linspace(-1200, 1200, 121))
    fields = {}
    for al in (0.0, 90.0):
        r = BIESolver(geo, Material(n_core=N_SI, n_clad=1.0, pol=2)).scatter(LAM, angle=al)
        u = (r.eval_field(X.ravel(), Z.ravel()) + r.incident_field(X.ravel(), Z.ravel())).reshape(X.shape)
        fields[str(int(al))] = np.round(np.abs(u), 3).tolist()
    ps = stack_solve(a, 0.0)
    return {"key": key, "label": label, "a": list(map(float, a)), "T2": T2,
            "outline": [np.r_[geo.f, geo.f[0]].round(1).tolist(), np.r_[geo.g, geo.g[0]].round(1).tolist()],
            "alpha": alphas.tolist(), "ang": ang, "ang_mean": float(np.mean(ang[:-1])), "chans": chans[:10],
            "lam": lams.tolist(), "t2_lam": [t2(geo, EPS, l)[0] for l in lams],
            "eps": epss.tolist(), "t2_eps": [t2(geo, e)[0] for e in epss],
            "fields": fields, "guided_scatt": ps["scatt"], "guided_T": ps["T"]}


def interior(job):
    from wg_interior_check import roots, unit_geo
    a, cls = job
    return roots(unit_geo(a), cls, lo=4.0, hi=20.0, step=0.01)


def main():
    run = json.loads((HERE / "results/wg_alldir.json").read_text())
    ok = [o for o in run["runs"] if o["feasible"] and o["width_gap_ok"]]
    best = min(ok, key=lambda o: o["T2"])
    c5 = json.loads((HERE / "results/campaign_500.json").read_text())
    dk = json.loads((HERE / "results/wg_dark.json").read_text())
    fs = min((o for o in c5["phase_a"] if o["feasible"]), key=lambda o: o["f"])["a"]
    an = min((o for o in dk["phase1"] if o["feasible"] and o["width_gap_ok"]), key=lambda o: o["scatt"])["a"]
    jobs = [("circle", "Circle", [0.0] * 10), ("alldir", "All-direction optimum", best["a"]),
            ("fsdark", "Dark for z incidence", fs), ("guided", "Dark over the slab", an)]
    with ProcessPoolExecutor(4) as ex:  # owner: 4 cores
        D = list(ex.map(design, jobs))
        ints = list(ex.map(interior, [(np.array(best["a"]), "even"), (np.array(best["a"]), "odd")]))
    for d in D:
        print(f"{d['key']:8s} ‖T‖² {d['T2']:.4f}  angle-mean {d['ang_mean']:.4f}  guided P_scatt {d['guided_scatt']:.4f}", flush=True)
    out = {"designs": D, "x": np.linspace(-1200, 1200, 121).tolist(), "runs": run["runs"],
           "interior_best": {"even": ints[0], "odd": ints[1]}, "eps_si": EPS}
    (HERE / "results/wg_alldir_page.json").write_text(json.dumps(out, separators=(",", ":")))
    print("interior eigenvalues of the optimum:", {k: [round(x, 3) for x in v] for k, v in out["interior_best"].items()})


if __name__ == "__main__":
    main()
