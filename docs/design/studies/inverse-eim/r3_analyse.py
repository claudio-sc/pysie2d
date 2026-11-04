"""R3: β error of the three 2D models against MPB over the single-mode domain.

Reads ``r3.jsonl`` (``r3_sweep.py``) and, for each single-mode strip and each
lateral convention of the top-view slab, computes the relative error of the
2D waveguide's β against the 3D β for: no approximation (core 3.47), plain
EIM (core = the side-view slab's n_eff) and inverse EIM (core = the root that
reproduces β). Writes ``r3_errors.json`` and prints the summary::

    uv run python docs/design/studies/inverse-eim/r3_analyse.py

3D n_eff: second-order extrapolation of res 64 and 128 (R2 measured second
order from res 64). Points whose cell hit ``CELL_MAX`` are excluded.

Polarisation: the 3D TE-like mode (E_y, in the chip plane) is a side-view slab
TE mode (pysie2d ``pol = 2``) and a top-view ``pol = 1`` mode; TM-like the
reverse (paper §II-B).
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pysie2d import Stack
from pysie2d.background import NearDegenerateModesError

HERE = Path(__file__).resolve().parent
LAM, N_SI, N_OX, N_AIR = 1550.0, 3.47, 1.44, 1.0
SIDE = {"te": 2, "tm": 1}
TOP = {"te": 1, "tm": 2}
CONVENTIONS = {"air|air": (N_AIR, N_AIR), "ox|ox": (N_OX, N_OX), "air|ox": (N_AIR, N_OX)}


def _neff_2d(core: float, width: float, pol: int, top: float, bot: float) -> float:
    try:
        modes = Stack(((core**2, width),), bot**2).guided_modes(LAM, pol, top)
    except NearDegenerateModesError:
        # A mode at the light line, i.e. at cutoff: no reliable β there.
        return np.nan
    return modes[0].n_eff if modes else np.nan


def _side(h: float, pol: int) -> float:
    return _neff_2d(N_SI, h, pol, N_AIR, N_OX)


def _inverse(target: float, w: float, pol: int, top: float, bot: float) -> float:
    lo = max(target, top, bot) + 1e-9
    f = lambda n: np.nan_to_num(_neff_2d(n, w, pol, top, bot), nan=0.0) - target  # noqa: E731
    return brentq(f, lo, N_SI, xtol=1e-14) if f(N_SI) > 0 else np.nan


def main() -> None:
    recs = [json.loads(s) for s in (HERE / "r3.jsonl").read_text().splitlines()]
    pts = [r for r in recs if r["pass"] == "B" and "r128" in r]
    rows = []
    for r in pts:
        w, h = float(r["w"]), float(r["h"])
        for pol, n128 in r["r128"].items():
            if r["cell_capped"][pol]:
                continue
            n3d = n128 + (n128 - r["r64"][pol]) / 3
            side = _side(h, SIDE[pol])
            row = {"w": w, "h": h, "pol": pol, "n3d": n3d, "side": side}
            for name, (top, bot) in CONVENTIONS.items():
                p = TOP[pol]
                none = _neff_2d(N_SI, w, p, top, bot)
                eim = _neff_2d(side, w, p, top, bot) if np.isfinite(side) else np.nan
                n2d = _inverse(n3d, w, p, top, bot)
                inv = _neff_2d(n2d, w, p, top, bot) if np.isfinite(n2d) else np.nan
                row[name] = {"none": none / n3d - 1, "eim": eim / n3d - 1,
                             "inverse": inv / n3d - 1, "n2d": n2d}
            rows.append(row)
    (HERE / "r3_errors.json").write_text(json.dumps(rows))
    print(f"{len(pts)} strips, {len(rows)} (strip, pol) pairs, capped excluded")
    for pol in ("te", "tm"):
        sub = [x for x in rows if x["pol"] == pol]
        print(f"\n{pol.upper()}: {len(sub)} strips")
        for name in CONVENTIONS:
            line = f"  {name:8s}"
            for m in ("none", "eim", "inverse"):
                e = np.abs([x[name][m] for x in sub])
                ok = np.isfinite(e)
                line += f"  {m} max {100 * e[ok].max():8.3g} % med {100 * np.median(e[ok]):7.3g} % ({(~ok).sum()} undef)"
            print(line)


if __name__ == "__main__":
    main()
