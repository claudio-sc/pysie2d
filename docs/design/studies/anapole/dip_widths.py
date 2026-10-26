"""Decade widths of the 1550 nm dips: detuning range over which c_sca stays below 10× and 100× its minimum.

For dips many decades deep the linear FWHM in sweep.py sits at the shoulder and says nothing
about sharpness; these widths do. Run after sweep.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shapes import c_sca  # noqa: E402

camp = {b["r_eq"]: b for b in json.loads((HERE / "results/campaign.json").read_text())["best"]}
sw = {d["r_eq"]: d for d in json.loads((HERE / "results/sweep.json").read_text())["designs"]}
rows = []
for r_eq, d in sw.items():
    area = np.pi * r_eq**2
    a, nn = np.array(camp[r_eq]["a"]), d["nn"]
    dip = min((p for p in d["dips"] if abs(p["lam"] - 1550) < 5), key=lambda p: p["c_min"], default=None)
    if dip is None:
        continue
    lam0, cmin = dip["lam"], dip["c_min"]
    f = lambda lam: np.log10(c_sca(a, area, nn, wavelength=lam) / cmin)  # noqa: E731
    w = {}
    for dec in (1, 2):
        try:
            lo = brentq(lambda x: f(x) - dec, lam0 - 300, lam0, xtol=1e-6) if f(lam0 - 300) > dec else np.nan
            hi = brentq(lambda x: f(x) - dec, lam0, lam0 + 300, xtol=1e-6) if f(lam0 + 300) > dec else np.nan
        except ValueError:
            lo = hi = np.nan
        w[dec] = hi - lo
    rows.append((r_eq, lam0, cmin / area, camp[r_eq]["gain_vs_circle"], w[1], w[2]))
    print(f"R_eq {r_eq:4.0f}: λ_min {lam0:8.2f}  c/A {cmin / area:.1e}  ×{camp[r_eq]['gain_vs_circle']:8.0f} vs circle"
          f"  width(×10) {w[1]:7.2f} nm  width(×100) {w[2]:7.2f} nm", flush=True)
(HERE / "results/dip_widths.json").write_text(json.dumps(
    [dict(zip(("r_eq", "lam_min", "c_per_area", "gain_vs_circle", "width_x10", "width_x100"), r)) for r in rows]))
