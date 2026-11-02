"""R3: MPB n_eff over the SOI strip's single-mode domain (paper Figs. 4–5).

Single core, unattended, resumable, with a wall-clock cap. Run detached in
the MEEP conda environment::

    nohup caffeinate -is ~/miniforge3/envs/pysie2d-meep/bin/python r3_sweep.py \
        > r3.log 2>&1 &

Two passes, both appending to ``r3.jsonl`` (one record per line, skipped on
restart):

A. **Domain.** Guided bands per y-parity at res 32 (``mpb_neff.guided_count``)
   for every w × h on a 10 nm grid. Single-mode = at most one guided band in
   each parity and at least one in total. TE₀ and TM₀ sit in opposite
   parities, as do TE₀ and TE₁, so a second band in either parity is a
   higher-order mode. Res 32 can misplace the boundary by about a grid step.
B. **n_eff**, single-mode points only, coarse 20 nm grid first so a capped run
   still covers the domain: res 64 and 128 (second order from res 64, R2), in
   a cell grown until the slowest mode's oxide-side tail fits.

MPB aborts the process on some inputs ("crazy number detected"), which no
``try`` catches, so each pass runs in a worker subprocess. Before each point the
worker writes it to ``r3.current``; if the worker dies, the launcher records
that point as failed and starts a new worker, which resumes after it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT, CUR = HERE / "r3.jsonl", HERE / "r3.current"
CAP_S = 20 * 3600.0
"""Owner's cap (2026-11-02): 20 h wall clock for the whole sweep."""
W_NM = range(100, 1001, 10)
H_NM = range(100, 501, 10)
LAM_NM = 1550.0
N_OX = 1.44
CELL_MIN, CELL_MAX = 4.0, 12.0
"""µm. 12 µm costs 9× the area of 4 µm, ~100 s per res-128 solve; a mode
needing more is flagged ``cell_capped`` rather than run without bound."""
TAIL = 7.0
"""Clearance per side in decay lengths: field e⁻⁷ ≈ 1e-3 at the cell edge, so
the n_eff error from truncation, ∝ |field|², is ~1e-6, below the res-128
discretisation error (~1e-3, R2)."""


def _done() -> set:
    if not OUT.exists():
        return set()
    return {(r["pass"], r["w"], r["h"]) for r in map(json.loads, OUT.read_text().splitlines())}


def _write(rec: dict) -> None:
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _cell(w_nm: float, h_nm: float, neff_min: float) -> tuple[float, bool]:
    """Square cell whose oxide-side clearance holds TAIL decay lengths."""
    gamma = 2 * math.pi / (LAM_NM / 1000) * math.sqrt(max(neff_min**2 - N_OX**2, 1e-12))
    need = 2 * TAIL / gamma + max(w_nm, h_nm) / 1000
    cell = max(CELL_MIN, math.ceil(2 * need) / 2)
    return min(cell, CELL_MAX), cell > CELL_MAX


def _single_mode(rec: dict) -> bool:
    return rec["even"] <= 1 and rec["odd"] <= 1 and rec["even"] + rec["odd"] >= 1


def worker(t_end: float) -> None:
    import meep as mp
    import mpb_neff as m

    mp.verbosity(0)
    done = _done()
    for w in W_NM:
        for h in H_NM:
            if ("A", w, h) in done:
                continue
            if time.time() > t_end:
                return
            CUR.write_text(json.dumps({"pass": "A", "w": w, "h": h}))
            t0 = time.time()
            g = m.guided_count(w, h, 32, LAM_NM)
            _write({"pass": "A", "w": w, "h": h, **g, "s": round(time.time() - t0, 2)})
    recs = [json.loads(s) for s in OUT.read_text().splitlines()]
    sm = [(r["w"], r["h"], r) for r in recs if r["pass"] == "A" and "even" in r and _single_mode(r)]
    sm.sort(key=lambda p: (p[0] % 20 != 0 or p[1] % 20 != 0, p[0], p[1]))  # 20 nm grid first
    done = _done()
    for w, h, a in sm:
        if ("B", w, h) in done:
            continue
        if time.time() > t_end:
            return
        CUR.write_text(json.dumps({"pass": "B", "w": w, "h": h}))
        t0 = time.time()
        rec = {"pass": "B", "w": w, "h": h, "r64": {}, "r128": {}, "cell": {}, "cell_capped": {}}
        # One cell per parity: the weakly guided TM mode needs ~3× the cell of
        # TE, and res-128 cost grows with its area.
        for par in (p for p in ("even", "odd") if a[p] == 1):
            r64 = m.solve(w, h, 64, LAM_NM, m.CELL_Y, (par,))
            ((pol, n64),) = r64.items()
            cell, capped = _cell(w, h, n64)
            if cell > m.CELL_Y:
                r64 = m.solve(w, h, 64, LAM_NM, cell, (par,))
            r128 = m.solve(w, h, 128, LAM_NM, cell, (par,))
            for k, v in (("r64", r64), ("r128", r128)):
                rec[k].update(v)
            ((pol, _),) = r128.items()
            rec["cell"][pol], rec["cell_capped"][pol] = cell, capped
        _write({**rec, "s": round(time.time() - t0, 2)})
    CUR.unlink(missing_ok=True)


def launcher() -> None:
    t_end = time.time() + CAP_S
    print(f"=== start {time.strftime('%F %T')}, cap {CAP_S / 3600:.0f} h ===", flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}  # single core, by the owner's rule
    while time.time() < t_end:
        p = subprocess.run([sys.executable, __file__, "--worker", str(t_end)], cwd=HERE, env=env)
        if p.returncode == 0:
            break
        if CUR.exists():  # the worker died on this point: record it and move past it
            bad = json.loads(CUR.read_text())
            _write({**bad, "failed": f"worker exit {p.returncode}"})
            print(f"[fail] {bad} exit {p.returncode}", flush=True)
            CUR.unlink()
    n = {"A": 0, "B": 0, "fail": 0}
    for r in map(json.loads, OUT.read_text().splitlines()):
        n["fail" if "failed" in r else r["pass"]] += 1
    left = "capped" if time.time() >= t_end else "complete"
    print(f"=== {left} {time.strftime('%F %T')}: {n} ===", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(float(sys.argv[2]))
    else:
        launcher()
