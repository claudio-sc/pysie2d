"""Data for the R_eq = 500 nm manufacturable section of the showcase page.

Two designs from campaign_500.py — the best nominal one (phase A) and the best worst-case
one (phase B) — each with: shape and its ±10 nm offsets, c_sca/A at 1550 nm against a
uniform boundary bias δ, spectra at δ = −10, 0, +10 nm, per-order multipole spectra and
1550 nm field maps (multipole_data.py conventions). The circle of the same area is analytic
Mie throughout (a circle biased by δ is the circle of radius R + δ).

Run after campaign_500.py:  uv run python docs/design/studies/anapole/showcase_500_data.py
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

from multipole_data import MMAX, field_map, per_order, sig  # noqa: E402
from pysie2d import BIESolver, Material  # noqa: E402
from pysie2d.reference.mie import efficiencies, mie_coefficients  # noqa: E402
from shapes import N_SI, POL, c_sca, geometry  # noqa: E402

R_EQ = 500.0
AREA = np.pi * R_EQ**2
NN = 384
BIAS = np.arange(-15.0, 15.01, 1.0)
LAMS = np.arange(1300.0, 1800.01, 1.0)
LAMS_MP = np.arange(1300.0, 1800.01, 2.0)


def circle(lam, delta=0.0):
    r = R_EQ + delta
    return efficiencies(2 * np.pi * r / lam, N_SI)["Q_sca_TE"] * 2 * r / AREA


def job(task):
    kind, a = task
    a = np.asarray(a)
    if kind == "bias":
        return [c_sca(a, AREA, NN, delta=d) / AREA for d in BIAS]
    if kind.startswith("spec"):
        d = float(kind[4:])
        return [c_sca(a, AREA, NN, wavelength=lam, delta=d) / AREA for lam in LAMS]
    if kind == "mp":
        s = BIESolver(geometry(a, AREA, NN), Material(n_core=N_SI, pol=POL))
        rows = [per_order(s.scatter(lam).multipoles(mmax=MMAX).c, 2 * np.pi / lam, AREA) for lam in LAMS_MP]
        return np.array(rows).T.tolist()
    if kind == "field":
        return field_map((R_EQ, a, map_ext(a)))
    raise ValueError(kind)


def map_ext(a):
    """Field-map half-width; the page reuses it to subtract the incident wave, so one definition."""
    g = geometry(np.asarray(a), AREA, 1024)
    return 1.25 * max(np.hypot(g.f, g.g).max(), R_EQ)


def outline(a, delta):
    g = geometry(np.asarray(a), AREA, 256, delta)
    return [[round(float(x), 1), round(float(z), 1)] for x, z in zip(g.f, g.g)]


def main():
    c5 = json.loads((HERE / "results/campaign_500.json").read_text())
    nominal = min((o for o in c5["phase_a"] if o["feasible"] and o["width_gap_ok"]), key=lambda o: o["f"])
    robust = min((o for o in c5["phase_b"] if o["feasible"] and o["width_gap_ok"]), key=lambda o: o["worst"])
    designs = {"nominal": nominal["a"], "tolerant": robust["a"]}
    kinds = ["bias", "spec-10", "spec0", "spec10", "mp", "field"]
    tasks = [(k, a) for a in designs.values() for k in kinds]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(job, tasks))
    out = {"lams": LAMS.tolist(), "lams_mp": LAMS_MP.tolist(), "bias": BIAS.tolist(), "designs": {}}
    for j, (name, a) in enumerate(designs.items()):
        r = dict(zip(kinds, res[j * len(kinds):(j + 1) * len(kinds)]))
        out["designs"][name] = {
            "a": a, "shape": outline(a, 0.0), "shape_m10": outline(a, -10.0), "shape_p10": outline(a, 10.0),
            "bias": sig(r["bias"], 4), "spec": {d: sig(r["spec" + d], 4) for d in ("-10", "0", "10")},
            "mp": [sig(row) for row in r["mp"]], "field": r["field"], "ext": map_ext(a),
        }
        print(f"{name}: c/A at δ=0 {r['bias'][15]:.3e}, worst over ±10 nm {max(r['bias'][5:26]):.3e}", flush=True)
    out["circle"] = {"bias": sig([circle(1550.0, d) for d in BIAS], 4),
                     "spec": {d: sig([circle(lam, float(d)) for lam in LAMS], 4) for d in ("-10", "0", "10")},
                     "mp": [sig(row) for row in np.array([per_order(np.r_[b[::-1], b[1:]], 2 * np.pi / lam, AREA)
                                                         for lam in LAMS_MP for b in [mie_coefficients(2 * np.pi * R_EQ / lam, N_SI, n_max=MMAX)[1]]]).T]}
    (HERE / "results/showcase_500.json").write_text(json.dumps(out, separators=(",", ":")))
    print("bytes", (HERE / "results/showcase_500.json").stat().st_size)


if __name__ == "__main__":
    main()
