"""Difference an external result against pysie2d, on that file's own grid.

Gate 4's number. The external drivers never import pysie2d and pysie2d never
reads their meshes, so the only thing the two share is the boundary — which
``validation/gielis.py`` owns and ``tests/test_gielis_contour.py`` pins bit for
bit. What is left when they are differenced is physics, not geometry.

The material is read from the result file's own ``knobs`` record rather than
passed in. A comparison that is *told* what was simulated can be told wrong,
and a star run at ``n_core = 1.5`` differenced against pysie2d at 2.0 produces
a plausible, meaningless number. Files written before ``n_core`` joined the
record are refused unless ``--n-core`` is given explicitly; guessing it from
the shape would reinstate exactly the ambiguity the record exists to close.

Run it from the repo root under uv::

    uv run python validation/compare.py --tool dolfinx --name final-star-L4 \
        --case lossless-te --nn 800
    uv run python validation/compare.py --nn-check          # what nn is enough
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

import gielis  # noqa: E402

from pysie2d import BIESolver, Geometry, Material  # noqa: E402

OUT = ROOT / "validation" / "_study"
OBSERVABLES = ("c_ext", "c_sca", "c_abs")

NM_PER_A = 100.0
"""MEEP's length unit, in nm. Mirrors ``NM_PER_A`` in its driver."""

POL = {"te": 2, "tm": 1}
"""conventions §2, non-negotiable 2: pol 2 is TE (E_y, Mie b_n), pol 1 is TM."""


def _n_core(knobs: dict, n_core: float | None) -> float:
    """The core index this result was produced at.

    Raises:
        SystemExit: if the file predates the ``n_core`` record and none was
            given. Better to stop than to difference against a guess.
    """
    if n_core is not None:
        return n_core
    if "n_core" not in knobs:
        raise SystemExit(
            "this file has no n_core in its knobs record (written before that "
            "field existed); pass --n-core explicitly"
        )
    return float(knobs["n_core"])


def pysie2d_spectrum(
    wavelengths: np.ndarray,
    shape: str,
    n_core: float,
    n_clad: float,
    epsi: float,
    pol: int,
    nn: int,
    angle: float = 0.0,
) -> dict[str, np.ndarray]:
    """pysie2d's cross-sections on the same shape, grid and material."""
    # The circle is the superformula at m = 0, which is how both drivers'
    # geometry blocks already describe it — not a separate constructor.
    params = (
        {"rad": 200.0, "m": 0, "n1": 2.0, "n2": 2.0, "n3": 2.0}
        if shape == "circle"
        else gielis.SHAPES[shape]
    )
    geom = Geometry.gielis(
        rad=params["rad"],
        n_pts=nn,
        m=int(params["m"]),
        n1=params["n1"],
        n2=params["n2"],
        n3=params["n3"],
    )
    solver = BIESolver(geom, Material(n_core=n_core, n_clad=n_clad, pol=pol, epsi=epsi))
    out = {k: np.empty(wavelengths.size) for k in OBSERVABLES}
    for i, lam in enumerate(wavelengths):
        cs = solver.scatter(wavelength=float(lam), angle=angle).cross_sections()
        for k in out:
            out[k][i] = cs[k]
    return out


def _drift(a: dict, b: dict, key: str, scale: float) -> float:
    """Worst pointwise |delta|, on study.py's normalisation.

    Normalised by peak ``C_ext`` rather than per observable, so the number is
    the same quantity the ladder's floors are quoted in and the two can be put
    in one table. A per-observable relative figure is arbitrarily large
    wherever a spectrum passes near zero, which ``C_sca`` does at its minima.
    """
    return float(np.max(np.abs(a[key] - b[key])) / scale)


def compare(tool: str, name: str, case: str, nn: int, n_core: float | None) -> None:
    """Report one external result against pysie2d, pointwise."""
    stem = name if tool == "meep" else f"{name}-{case}"
    path = OUT / tool / f"{stem}.npz"
    with np.load(path) as d:
        ext = {k: d[k] for k in d.files if k in OBSERVABLES}
        wl = d["wavelength_nm"]
        knobs = json.loads(str(d["knobs"]))

    # MEEP works in units of a = 100 nm and its .npz holds the raw arrays:
    # only freeze() and the progress print multiply by NM_PER_A. Differencing
    # them against pysie2d's nm without this reads as a factor-100 disagreement.
    if tool == "meep":
        ext = {k: v * NM_PER_A for k, v in ext.items()}

    if "lossy" in case:
        # epsi is not in the knobs record, and Im(eps) is absolute
        # (conventions §2) — there is no safe default to fall back on.
        raise SystemExit("lossy cases are not supported here: epsi is unrecorded")
    pol_key = case.split("-")[-1]
    n_core_used = _n_core(knobs, n_core)
    epsi = 0.0
    ours = pysie2d_spectrum(
        wl,
        # Unlike n_core, a wrong shape cannot produce a plausible number — it
        # moves every point by orders — so the circle-only runs that predate
        # this field are defaulted rather than refused.
        knobs.get("shape", "circle"),
        n_core_used,
        1.0,
        epsi,
        POL[pol_key],
        nn,
        float(knobs.get("angle_deg", 0.0)),
    )
    scale = float(np.max(np.abs(ours["c_ext"])))
    drifts = "  ".join(
        f"{k}={_drift(ext, ours, k, scale):.2e}" for k in OBSERVABLES if k in ext
    )
    print(f"{tool:8s} {stem:28s} n_core={n_core_used:.2f} nn={nn:4d}  {drifts}")


def nn_check(shape: str, n_core: float, nn_list: tuple[int, ...]) -> None:
    """pysie2d against itself, to justify the nn the comparisons run at.

    The reference has to be converged far below the error it is measuring, or
    the comparison reports pysie2d's discretisation and calls it the external
    tool's. Under Kress quadrature this reaches round-off quickly, so what is
    printed is a measurement and not a convergence order.
    """
    wl = np.linspace(400.0, 900.0, 51)
    for pol_key, pol in POL.items():
        ref = pysie2d_spectrum(wl, shape, n_core, 1.0, 0.0, pol, max(nn_list))
        scale = float(np.max(np.abs(ref["c_ext"])))
        for nn in nn_list[:-1]:
            got = pysie2d_spectrum(wl, shape, n_core, 1.0, 0.0, pol, nn)
            drifts = "  ".join(
                f"{k}={_drift(got, ref, k, scale):.2e}" for k in OBSERVABLES
            )
            print(f"pysie2d  {shape} {pol_key}  nn={nn:4d} vs {max(nn_list)}  {drifts}")


def main() -> None:
    """Parse the command line and run one comparison, or the nn check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("dolfinx", "meep"), default="dolfinx")
    parser.add_argument("--name", default="final-star-L4")
    parser.add_argument("--case", default="lossless-te")
    parser.add_argument("--nn", type=int, default=800)
    parser.add_argument("--n-core", type=float, default=None)
    parser.add_argument("--nn-check", action="store_true")
    parser.add_argument("--shape", default="star")
    args = parser.parse_args()

    if args.nn_check:
        nn_check(args.shape, args.n_core or 2.0, (100, 200, 400, 800))
    else:
        compare(args.tool, args.name, args.case, args.nn, args.n_core)


if __name__ == "__main__":
    main()
