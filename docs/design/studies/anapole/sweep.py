"""Wavelength sweep of the campaign's best designs: where are the dips, how deep, how sharp.

Index held fixed (n = 3.476, lossless) so only geometric dispersion shows. Each design
and the circle of the same area are swept on a 1 nm grid, then every local minimum is
refined by a bounded minimiser (a dip narrower than the grid is otherwise missed or
under-reported). Per dip: λ_min, depth = c_min / (lower adjacent maximum), the full width
at half depth (linear c_sca, halfway between c_min and that maximum) and Q_dip = λ/FWHM.

Run after campaign.py:  uv run python docs/design/studies/anapole/sweep.py
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
from scipy.optimize import brentq, minimize_scalar  # noqa: E402

from pysie2d.reference.mie import efficiencies  # noqa: E402
from shapes import N_SI, c_sca  # noqa: E402

LAMS = np.arange(1000.0, 2400.01, 1.0)
WORKERS = 4


def dips(a, area, lams, c, nn):
    out = []
    i_min = [i for i in range(1, len(c) - 1) if c[i] <= c[i - 1] and c[i] <= c[i + 1]]
    i_max = [0] + [i for i in range(1, len(c) - 1) if c[i] >= c[i - 1] and c[i] >= c[i + 1]] + [len(c) - 1]
    f = lambda lam: c_sca(a, area, nn, wavelength=lam)  # noqa: E731
    for i in i_min:
        r = minimize_scalar(lambda lam: np.log(f(lam)), bounds=(lams[i - 1], lams[i + 1]),
                            method="bounded", options={"xatol": 1e-7})
        lam0, cmin = r.x, float(np.exp(r.fun))
        left = max(j for j in i_max if j < i)
        right = min(j for j in i_max if j > i)
        shoulder = min(c[left], c[right])
        half = 0.5 * (cmin + shoulder)
        try:
            l1 = brentq(lambda lam: f(lam) - half, lams[left], lam0, xtol=1e-6)
            l2 = brentq(lambda lam: f(lam) - half, lam0, lams[right], xtol=1e-6)
            fwhm = l2 - l1
        except ValueError:  # an edge shoulder that never reaches half depth
            fwhm = float("nan")
        out.append({"lam": float(lam0), "c_min": cmin, "c_min_2nn": c_sca(a, area, 2 * nn, wavelength=lam0),
                    "depth": cmin / shoulder, "c_shoulder": float(shoulder),
                    "fwhm": float(fwhm), "q_dip": float(lam0 / fwhm) if fwhm == fwhm else float("nan")})
    return out


def pick_nn(a, area):
    """384, or 768 where 384 is not converged against 1536 at 1550 nm (needle-like optima).

    Checked at the design wavelength only; the 1000 nm end has k 1.55× larger, so each
    dip is re-evaluated at 2·nn (c_min_2nn) rather than assuming the choice carries over.
    """
    ref = c_sca(a, area, 1536)
    return 384 if abs(c_sca(a, area, 384) / ref - 1) < 1e-10 else 768


def sweep(job):
    r_eq, a = job
    area = np.pi * r_eq**2
    a = np.asarray(a)
    nn = pick_nn(a, area)
    c = np.array([c_sca(a, area, nn, wavelength=lam) for lam in LAMS])
    z = np.zeros_like(a)
    cc = np.array([c_sca(z, area, 192, wavelength=lam) for lam in LAMS])  # circle: round-off by nn ≈ 40
    return {"r_eq": r_eq, "nn": nn, "c": c.tolist(), "c_circle": cc.tolist(),
            "dips": dips(a, area, LAMS, c, nn), "dips_circle": dips(z, area, LAMS, cc, 192)}


def main():
    camp = json.loads((HERE / "results/campaign.json").read_text())
    jobs = [(b["r_eq"], b["a"]) for b in camp["best"]]
    # Anchor: circle sweep at the first ladder rung against Mie at three wavelengths.
    r0 = jobs[0][0]
    for lam in (1000.0, 1550.0, 2400.0):
        q = efficiencies(2 * np.pi * r0 / lam, N_SI)["Q_sca_TE"] * 2 * r0
        cb = c_sca(np.zeros(len(jobs[0][1])), np.pi * r0**2, 192, wavelength=lam)
        print(f"anchor R={r0:.0f} λ={lam:.0f}: rel {abs(cb / q - 1):.1e}")
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(sweep, jobs))
    for r in res:
        area = np.pi * r["r_eq"] ** 2
        deep = sorted(r["dips"], key=lambda d: d["depth"])[:3]
        print(f"R_eq {r['r_eq']:4.0f} (nn {r['nn']}): " + "; ".join(
            f"λ {d['lam']:.2f} c/A {d['c_min'] / area:.1e} depth {d['depth']:.1e} FWHM {d['fwhm']:.2f} nm Q {d['q_dip']:.0f} (2nn {abs(d['c_min_2nn'] / d['c_min'] - 1):.0e})"
            for d in deep))
    (HERE / "results/sweep.json").write_text(json.dumps({"lams": LAMS.tolist(), "designs": res}))


if __name__ == "__main__":
    main()
