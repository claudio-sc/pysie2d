"""MPB n_eff of the 500 × 220 nm strip's TE₀ over the 3D reference band.

Input to the inverse-EIM side of the 3D comparison (README, rule step 3):
res 64 and 128, second-order extrapolated as R2, at the 31 λ of
``run_strip3d.py``. MEEP conda environment::

    ~/miniforge3/envs/pysie2d-meep/bin/python band_neff.py
"""

import json
from pathlib import Path

import meep as mp
import numpy as np

import mpb_neff as m

mp.verbosity(0)
LAM = 1000 / np.linspace(1 / 1.65, 1 / 1.50, 31)  # nm; the 3D run's frequency grid
# TE₀ is the y-parity whose band 1 is E_y-dominant; find it once at band centre.
par = next(p for p in ("even", "odd") if "te" in m.solve(500, 220, 32, 1550.0, m.CELL_Y, (p,)))
rows = []
for lam in LAM:
    n64 = m.solve(500, 220, 64, lam, m.CELL_Y, (par,))["te"]
    n128 = m.solve(500, 220, 128, lam, m.CELL_Y, (par,))["te"]
    rows.append({"lam_nm": lam, "r64": n64, "r128": n128, "n_eff": n128 + (n128 - n64) / 3})
    print(f"{lam:8.2f} nm  n_eff {rows[-1]['n_eff']:.5f}", flush=True)
(Path(__file__).resolve().parent / "band_neff.json").write_text(json.dumps(rows, indent=1))
