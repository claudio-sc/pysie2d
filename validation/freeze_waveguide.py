"""Freeze v1.3's MEEP S-parameters into ``tests/data/waveguide/``.

Reads the 200 nm-gap ladder (``_study/v13``: ``ladder-sb4`` norms, ``pair200``
cluster runs, res 20/25/30, four sub-band sources) and writes one JSON with the
geometry, MEEP's extrapolated observables, the raw ladder, and a tolerance per
observable that is computed from MEEP's own convergence evidence alone::

    uv run python validation/freeze_waveguide.py

Extrapolation
-------------
First order, from each consecutive pair of rungs (``compare_waveguide.richardson``;
the order is measured on MPB's n_eff against the closed-form slab). The frozen
value is the (25, 30) extrapolation.

Tolerance
---------
Per observable, ``3 × (spread + floor)``, maxed over λ:

* ``spread`` = |extrapolation(20, 25) − extrapolation(25, 30)|: how far the
  extrapolated value still moves when the ladder is shifted by one rung, i.e.
  the extrapolation's own uncertainty;
* ``floor`` = MEEP's resolution-independent noise, measured in the same runs:
  the decay-snapshot change of t (truncation) plus 2·|r|·|ρ_PML| (the far
  x-PML's guided-mode reflection, |α⁻/α⁺| in the norm run).

Neither term uses pysie2d. An observable whose tolerance exceeds half its own
signal (max over λ) is recorded but not asserted: at this gap, TE reflection
(|r|² ≤ 2e-5 against a floor of a few e-4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

V = Path(__file__).resolve().parent
ROOT = V.parent
sys.path.insert(0, str(V))
import compare_waveguide as cw  # noqa: E402
import gielis  # noqa: E402
import soi  # noqa: E402

S = V / "_study" / "v13"
OUT = ROOT / "tests" / "data" / "waveguide" / "meep-soi-pair200.json"
RES = (20, 25, 30)
PLANE = 1500.0
CLUSTER = "pair200"


def _runs(case: str, res: int) -> dict:
    tag = "-subbands4"
    load = lambda d, drive: dict(np.load(d / f"{case}-{drive}-r{res}{tag}.npz"))  # noqa: E731
    return {
        "norm": load(S / "ladder-sb4", "norm"),
        "norm_right": load(S / "ladder-sb4", "norm_right"),
        "left": load(S / CLUSTER, "left"),
        "right": load(S / CLUSTER, "right"),
    }


def _derived(q: dict) -> dict[str, np.ndarray]:
    """Power observables from complex t, r."""
    out = {}
    for side in ("left", "right"):
        t2, r2 = np.abs(q[f"t_{side}"]) ** 2, np.abs(q[f"r_{side}"]) ** 2
        out[f"t2_{side}"], out[f"r2_{side}"] = t2, r2
        out[f"p_rad_{side}"] = 1 - t2 - r2
    return out


def freeze() -> dict:
    """Build the frozen record for both polarisations."""
    record = {
        "schema": "pysie2d-external-validation/waveguide-1",
        "tool": "meep",
        "geometry": {
            "stack": {"core_n": soi.N_SI, "core_nm": soi.T_SI_NM, "clad_n": soi.N_OX},
            "particles_n": soi.N_SI,
            "cluster": [
                {"shape": s, **gielis.SHAPES[s], "x0_nm": x0, "z0_nm": z0}
                for s, x0, z0 in gielis.CLUSTERS[CLUSTER]
            ],
            "gap_nm": 200.0,
            "plane_nm": PLANE,
        },
        "wavelength_nm": soi.WAVELENGTHS.tolist(),
        "pol_mapping": "te = pysie2d pol 2 (E_y) = MEEP Ez/ODD_Z; tm = pol 1 (H_y) = "
        "MEEP Hz/EVEN_Z; TM reflection sign r_pysie2d = −r_MEEP (compare_waveguide)",
        "cases": {},
    }
    for case in ("te", "tm"):
        runs = {r: _runs(case, r) for r in RES}
        q = {r: cw.observables(runs[r], PLANE, case) for r in RES}
        e1 = cw.richardson(q[20], 20, q[25], 25)
        e2 = cw.richardson(q[25], 25, q[30], 30)
        d1, d2 = _derived(e1), _derived(e2)
        fine = runs[30]
        snap = np.max(
            np.abs(
                cw.s_params(fine, PLANE, snapshot=True)["t_left"]
                - cw.s_params(fine, PLANE)["t_left"]
            )
        )
        rho = np.max(
            np.abs(fine["norm"]["alpha"][..., 1] / fine["norm"]["alpha"][..., 0])
        )
        values, tol, asserted = {}, {}, {}
        for k in d2:
            r2 = d2[f"r2_{k.split('_')[-1]}"]
            floor = snap + 2 * np.sqrt(np.abs(r2)) * rho
            t = float(3 * np.max(np.abs(d1[k] - d2[k]) + floor))
            values[k], tol[k] = d2[k].tolist(), t
            asserted[k] = bool(t < 0.5 * np.max(np.abs(d2[k])))
        for k in ("t_left", "t_right"):
            values[k] = [[z.real, z.imag] for z in e2[k]]
            t = float(3 * np.max(np.abs(e1[k] - e2[k]) + snap))
            tol[k], asserted[k] = t, True
        record["cases"][case] = {
            "values": values,
            "tolerance_abs": tol,
            "asserted": asserted,
            "n_eff_extrapolated": e2["n_eff"].tolist(),
            "raw_ladder": {
                str(r): {k: v.tolist() for k, v in _derived(q[r]).items()} for r in RES
            },
            "noise_floor": {"decay_snapshot_dt": float(snap), "x_pml_rho": float(rho)},
            "knobs": json.loads(str(fine["left"]["knobs"])),
        }
    return record


def main() -> None:
    """Write the frozen file and print what it asserts."""
    rec = freeze()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    for case, c in rec["cases"].items():
        for k, t in c["tolerance_abs"].items():
            how = "asserted" if c["asserted"][k] else "recorded only"
            print(f"{case} {k:12s} tol {t:.2e}  {how}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
