"""Pack campaign + sweep results for the showcase page, adding a 0.2 nm zoom around 1550 nm.

Run after sweep.py and dip_widths.py:  uv run python docs/design/studies/anapole/showcase_data.py
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

from shapes import c_sca, geometry  # noqa: E402

ZOOM = np.round(np.arange(1510.0, 1590.001, 0.2), 1)


def zoom(job):
    r_eq, a, nn = job
    area = np.pi * r_eq**2
    return [c_sca(np.asarray(a), area, nn, wavelength=lam) / area for lam in ZOOM]


def sig(x, n=4):
    return [float(f"{v:.{n}g}") for v in x]


def main():
    camp = json.loads((HERE / "results/campaign.json").read_text())
    sw = json.loads((HERE / "results/sweep.json").read_text())
    widths = {w["r_eq"]: w for w in json.loads((HERE / "results/dip_widths.json").read_text())}
    best = {b["r_eq"]: b for b in camp["best"]}
    designs = sw["designs"]
    with ProcessPoolExecutor(4) as ex:
        zooms = list(ex.map(zoom, [(d["r_eq"], best[d["r_eq"]]["a"], d["nn"]) for d in designs]))

    out = []
    for d, z in zip(designs, zooms):
        r_eq = d["r_eq"]
        area = np.pi * r_eq**2
        b = best[r_eq]
        g = geometry(np.array(b["a"]), area, 2048)
        sp = np.hypot(g.df, g.dg)
        kap = (g.df * g.ddg - g.dg * g.ddf) / sp**3
        idx = np.arange(0, 2048, 8)
        w = widths.get(r_eq, {})
        out.append({
            "r_eq": r_eq,
            "c_per_area": b["c_sca"] / area,
            "circle_per_area": b["c_sca_circle"] / area,
            "gain": b["gain_vs_circle"],
            "perim_frac": b["perimeter"] / camp["meta"]["L_max"],
            "min_curv_radius": float(1 / abs(kap).max()),
            "r_ratio": float(np.hypot(g.f, g.g).max() / np.hypot(g.f, g.g).min()),
            "width_x10": w.get("width_x10"),
            "width_x100": w.get("width_x100"),
            "starts_at_best": b["n_starts_within_0p1dec"],
            "shape": [[round(float(g.f[i]), 1), round(float(g.g[i]), 1)] for i in idx],
            "spec": sig(np.array(d["c"]) / area),
            "spec_circle": sig(np.array(d["c_circle"]) / area),
            "zoom": sig(z),
        })
    data = {
        "lams": sw["lams"],
        "zoom_lams": ZOOM.tolist(),
        "meta": camp["meta"],
        "designs": out,
        # §1 of the README: ITE vs driven anapole on superellipses, X = 0.5.
        "ite_trace": {
            "p": [2.0, 2.25, 2.5, 3.0, 3.5, 4.0],
            "pol2_ite": [59.7601, 57.5353, 56.1648, 54.7271, 54.0899, 53.7838],
            "pol2_driven": [59.7601, 57.3148, 55.4655, 52.8780, 51.1785, 49.9948],
            "pol1_ite": [60.8227, 58.4922, 56.8897, 54.9097, 53.8001, 53.1315],
            "pol1_driven": [60.8227, 58.4872, 56.8717, 54.8570, 53.7116, 53.0108],
        },
    }
    (HERE / "results/showcase.json").write_text(json.dumps(data, separators=(",", ":")))
    print("bytes", (HERE / "results/showcase.json").stat().st_size)


if __name__ == "__main__":
    main()
