"""Difference v1.3's MEEP runs against pysie2d's reference, per polarisation.

    uv run python validation/compare_waveguide.py --ref ref.npz \
        --runs DIR --case te --resolution 25

Reads ``{case}-{drive}-r{res}.npz`` from ``DIR`` for ``drive`` in ``norm``,
``left``, ``right`` and ``norm_right``.

How MEEP's coefficients become S-parameters
-------------------------------------------
Each ratio divides a cluster run by the ``norm`` run *at the same plane and
direction*, so the source spectrum and MPB's arbitrary mode phase cancel. Each
drive has its own ``norm``: the mirror image of the left launch is *not* the
right launch. At resolution 8 the −x launch carried 1.2 % (TE) and 0.7 % (TM)
more power than the +x one, with the phase mirrored to 1e-6 rad, so a mirrored
normalisation would put a percent into |t_R|² and |r_R|².

**Reflection phase in TM.** MEEP's coefficient overlaps the field with the
mode, and the backward mode is the forward one with ``H_t`` negated (``E_t``
kept). Its α⁻/α⁺ is therefore a ratio of *E* fields. pysie2d's ``pol = 1``
scalar is ``H_y`` (MEEP's ``Hz``, transverse to x), whose ratio is the negative
of the E ratio: ``r_pysie2d = −r_MEEP`` in TM and ``+r_MEEP`` in TE. This sign
is derived, not fitted; a wrong derivation shows up as a π offset in arg r.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load(d: Path, case: str, drive: str, res: int) -> dict | None:
    p = d / f"{case}-{drive}-r{res}.npz"
    return dict(np.load(p)) if p.exists() else None


def s_params(
    runs: dict, plane_nm: float, snapshot: bool = False
) -> dict[str, np.ndarray]:
    """MEEP's t, r at the planes ±plane_nm, for each drive whose runs exist."""
    key = "alpha_snapshot" if snapshot else "alpha"
    planes = runs["norm"]["planes_nm"]
    i_m = int(np.argmin(np.abs(planes + plane_nm)))
    i_p = int(np.argmin(np.abs(planes - plane_nm)))
    nl, a = runs["norm"]["alpha"], runs["left"][key]
    out = {
        "t_left": a[i_p, :, 0] / nl[i_p, :, 0],
        "r_left": a[i_m, :, 1] / nl[i_m, :, 0],
    }
    if runs.get("right") is not None and runs.get("norm_right") is not None:
        b, nr = runs["right"][key], runs["norm_right"]["alpha"]
        out["t_right"] = b[i_m, :, 1] / nr[i_m, :, 1]
        out["r_right"] = b[i_p, :, 0] / nr[i_p, :, 1]
    return out


def observables(runs: dict, plane_nm: float, case: str) -> dict[str, np.ndarray]:
    """MEEP's n_eff, t and r in pysie2d's convention (TM reflection sign applied)."""
    sign = -1.0 if case == "tm" else 1.0
    m = s_params(runs, plane_nm)
    return {"n_eff": runs["norm"]["n_eff"]} | {
        k: v * (sign if k.startswith("r") else 1.0) for k, v in m.items()
    }


def errors(ref: dict, q: dict[str, np.ndarray], case: str) -> dict[str, float]:
    """Max-over-λ differences of MEEP observables ``q`` from pysie2d."""
    out = {"n_eff": np.max(np.abs(q["n_eff"] - ref[f"{case}_n_eff"]))}
    for k in (k for k in q if k != "n_eff"):
        rv = ref[f"{case}_{k}"]
        out[f"|{k}|²"] = np.max(np.abs(np.abs(q[k]) ** 2 - np.abs(rv) ** 2))
        out[f"{k} complex"] = np.max(np.abs(q[k] - rv))
    for side in ("left", "right"):
        if f"t_{side}" not in q:
            continue
        p_meep = 1 - np.abs(q[f"t_{side}"]) ** 2 - np.abs(q[f"r_{side}"]) ** 2
        p_ref = ref[f"{case}_up_{side}"] + ref[f"{case}_down_{side}"]
        out[f"P_rad {side}"] = np.max(np.abs(p_meep - p_ref))
    return out


def richardson(coarse: dict, res_c: int, fine: dict, res_f: int) -> dict:
    """First-order Richardson extrapolation of every observable to res → ∞.

    First order is measured, not assumed: on the grid-aligned cell MPB's n_eff
    error × res is constant to three digits in TM (0.124 at res 10–40) and
    drifts 0.083 → 0.071 in TE, against the closed-form slab, an anchor
    independent of pysie2d. Complex t and r are extrapolated as complex numbers,
    so a phase error converging at first order is removed with the magnitude's.
    """
    return {
        k: (res_f * fine[k] - res_c * coarse[k]) / (res_f - res_c)
        for k in fine
        if k in coarse
    }


def compare(ref: dict, runs: dict, case: str) -> dict[str, float]:
    """Max-over-λ differences, MEEP minus pysie2d, plus MEEP's own diagnostics."""
    plane = float(ref["plane_nm"])
    m = s_params(runs, plane)
    out = errors(ref, observables(runs, plane, case), case)
    snap = s_params(runs, plane, snapshot=True)
    out["MEEP decay snapshot |Δt_L|"] = np.max(np.abs(snap["t_left"] - m["t_left"]))
    nl = runs["norm"]["alpha"]
    # A bare slab has nothing to reflect but the far x-PML: |α⁻/α⁺| in the norm
    # run is that reflection, and it enters |r|² at ~2|r|·|ρ|.
    out["x-PML reflection |α⁻/α⁺|_norm"] = np.max(np.abs(nl[..., 1] / nl[..., 0]))
    if "t_right" in m:
        out["MEEP reciprocity |t_L − t_R|"] = np.max(np.abs(m["t_left"] - m["t_right"]))
        nr = runs["norm_right"]["alpha"]
        q = nr[np.arange(len(nr)) ^ 1][..., ::-1][..., 0] / nl[..., 0]  # ±X paired
        out["launch asymmetry ||q| − 1|"] = np.max(np.abs(np.abs(q) - 1))
        out["launch mirror phase |arg q|"] = np.max(np.abs(np.angle(q)))
    return out


def main() -> None:
    """Print the comparison for one polarisation and resolution."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--case", choices=("te", "tm"), required=True)
    p.add_argument("--resolution", type=int, required=True)
    a = p.parse_args()
    ref = dict(np.load(a.ref))
    runs = {
        d: _load(a.runs, a.case, d, a.resolution)
        for d in ("norm", "left", "right", "norm_right")
    }
    for k, v in compare(ref, runs, a.case).items():
        print(f"{a.case} r{a.resolution}  {k:32s} {v:.3e}")


if __name__ == "__main__":
    main()
