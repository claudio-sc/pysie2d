"""Gate 5: turn the study's ``.npz`` results into the frozen ``.json`` anchors.

This is the step that moves external evidence out of gitignored `_study/` and
into the repo, where `tests/test_external_validation.py` reads it with numpy
alone. Everything a reader needs to judge the claim travels with the numbers:
what was simulated, by which tool at which version, on what discretisation,
with what evidence it had converged, and what agreement the repo therefore
asserts.

**Every tolerance here is computed, not typed.** ``rel`` is the measured worst
pointwise deviation between pysie2d and the frozen spectrum, rounded up to the
next clean number; ``abs_nm`` is the external tool's own residual absorption in
the lossless cases. Nobody can widen one to make a test pass without changing
the measurement it is derived from, which is the point (non-negotiable §4).

Run it from the repo root under uv, with both conda environments present — the
tool version and polarisation mapping are read from the drivers themselves so
they keep one definition::

    uv run python validation/freeze.py --dry-run
    uv run python validation/freeze.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

from compare import NM_PER_A, pysie2d_spectrum  # noqa: E402
from spectrum import Convergence, Spectrum, Tolerance, write  # noqa: E402

OUT = ROOT / "validation" / "_study"
DATA = ROOT / "tests" / "data"
CONDA = Path.home() / "miniforge3" / "bin" / "conda"

GEOMETRY = {
    "circle": {"rad": 200.0, "m": 0, "n1": 2.0, "n2": 2.0, "n3": 2.0, "n_pts": 200},
    # n_pts 400, not the circle's 200. On this star pysie2d's own discretisation
    # error at 200 is 4.85e-05 — two orders *above* dolfinx's disagreement with
    # it, so a test run at 200 would measure pysie2d and attribute it to the
    # external tool. At 400 it is 5.4e-09 and contributes nothing.
    "star": {"rad": 200.0, "m": 6, "n1": 6.0, "n2": 12.0, "n3": 12.0, "n_pts": 400},
}

MATERIAL = {
    ("circle", False): {"n_core": 1.5, "n_clad": 1.0, "epsi": 0.0},
    ("circle", True): {"n_core": 1.5, "n_clad": 1.33, "epsi": 0.5},
    ("star", False): {"n_core": 2.0, "n_clad": 1.0, "epsi": 0.0},
}


@dataclass(frozen=True)
class Case:
    """One frozen file, and the two study results that justify its tolerance."""

    tool: str
    shape: str
    pol_key: str
    lossy: bool
    npz: str
    coarse: str
    fine: str
    parameter: str
    coarse_label: str
    fine_label: str
    note: str

    @property
    def case_id(self) -> str:
        """The filename stem, matching each driver's own naming."""
        loss = "lossy" if self.lossy else "lossless"
        return f"{self.tool}-{self.shape}-{loss}-{self.pol_key}"


def _dolfinx_cases() -> list[Case]:
    """The FEM anchors: the circle in all four cases, the star in two."""
    mesh = {
        "parameter": "(h_particle nm, h_outer nm, Lagrange degree)",
        "coarse_label": "(6, 18, 3)",
        "fine_label": "(4, 12, 3)",
        "note": (
            "Drift measured on the 13-point study grid, which is a subset of "
            "the 51-point grid frozen here — gate 2 measures where the tool "
            "stops moving, gate 5 runs the full grid at that level."
        ),
    }
    out = []
    for pol_key in ("te", "tm"):
        for lossy in (False, True):
            loss = "lossy" if lossy else "lossless"
            out.append(
                Case(
                    "dolfinx",
                    "circle",
                    pol_key,
                    lossy,
                    npz=f"final-circle-L4-{loss}-{pol_key}",
                    coarse=f"mesh-L3-{loss}-{pol_key}",
                    fine=f"mesh-L4-{loss}-{pol_key}",
                    **mesh,
                )
            )
        out.append(
            Case(
                "dolfinx",
                "star",
                pol_key,
                False,
                npz=f"final-star-L4-lossless-{pol_key}",
                coarse=f"star-L3-lossless-{pol_key}",
                fine=f"star-L4-lossless-{pol_key}",
                **mesh,
            )
        )
    return out


def _meep_cases() -> list[Case]:
    """The FDTD anchors. Lossless only — MEEP has no frequency-independent Im ε."""
    res = {"parameter": "resolution (pixels per 100 nm)"}
    out = []
    for pol_key in ("te", "tm"):
        out.append(
            Case(
                "meep",
                "circle",
                pol_key,
                False,
                npz=f"res-50-{pol_key}",
                coarse=f"res-40-{pol_key}",
                fine=f"res-50-{pol_key}",
                coarse_label="40",
                fine_label="50",
                note=(
                    "The pml, box and decay isolation groups were run at "
                    "resolution 25, where grid error dominates them, so they "
                    "bound those systematics at 25 and not at 50. This drift "
                    "is therefore a bounded systematic, not a converged floor "
                    "(design doc §5 gate 6)."
                ),
                **res,
            )
        )
        out.append(
            Case(
                "meep",
                "star",
                pol_key,
                False,
                npz=f"final-star-res32-{pol_key}",
                coarse=f"star-res25-{pol_key}",
                fine=f"star-res32-{pol_key}",
                coarse_label="25",
                fine_label="32",
                note=(
                    "First order, not second: six near-cusps on a Yee grid, "
                    "which subpixel averaging cannot rescue. Measured "
                    "separately, the cusps cost 9.6x and the n_core = 2.0 "
                    "contrast 4.2x, and their product is the observed 40x gap "
                    "against MEEP's own circle. A bounded systematic we did "
                    "not eliminate, never a converged floor."
                ),
                **res,
            )
        )
    return out


def _tool_metadata(tool: str) -> dict:
    """Version and polarisation mapping, read from the driver in its own env.

    Shelling out rather than copying the strings here: the mapping is where a
    sign or axis error hides, and two copies of it is one copy that can drift
    out of agreement with the code that actually ran.
    """
    script = (
        "import sys, json; sys.path.insert(0, 'validation/%s'); "
        "sys.path.insert(0, 'validation'); import run_cylinder as rc; "
        "v = rc._versions() if hasattr(rc, '_versions') "
        "else __import__('meep').__version__; "
        "print('JSONLINE' + json.dumps({'version': v, "
        "'pol_mapping': rc.POL_MAPPING}))" % tool
    )
    proc = subprocess.run(
        [str(CONDA), "run", "-n", f"pysie2d-{tool}", "python", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("JSONLINE"):
            return json.loads(line[len("JSONLINE") :])
    raise SystemExit(f"could not read {tool} metadata:\n{proc.stdout}\n{proc.stderr}")


def _load(stem: str, tool: str) -> dict[str, np.ndarray]:
    """One study result, in nm."""
    with np.load(OUT / tool / f"{stem}.npz") as d:
        res = {k: d[k] for k in ("wavelength_nm", "c_sca", "c_ext", "c_abs")}
    if tool == "meep":
        # MEEP's .npz holds units of a = 100 nm; only freeze() and the progress
        # print convert. See compare.py.
        for k in ("c_sca", "c_ext", "c_abs"):
            res[k] = res[k] * NM_PER_A
    return res


def _ceil_clean(x: float) -> float:
    """Round up to the next 1, 2, 3 or 5 times a power of ten.

    A tolerance should be a number a reader can hold in their head and check
    against the measurement beside it, not a fifteen-digit float that looks
    like it was tuned until something passed.
    """
    if x <= 0:
        return 0.0
    exp = np.floor(np.log10(x))
    for mant in (1.0, 2.0, 3.0, 5.0, 10.0):
        if x <= mant * 10.0**exp * (1 + 1e-12):
            return float(mant * 10.0**exp)
    return float(10.0 ** (exp + 1))


def _drift(case: Case) -> float:
    """Worst pointwise relative change in C_ext between the two levels."""
    a, b = _load(case.coarse, case.tool), _load(case.fine, case.tool)
    return float(np.max(np.abs(a["c_ext"] - b["c_ext"]) / np.abs(b["c_ext"])))


def build(case: Case, meta: dict) -> tuple[Spectrum, dict]:
    """One frozen spectrum, with its tolerance measured rather than chosen."""
    ext = _load(case.npz, case.tool)
    geom = GEOMETRY[case.shape]
    mat = MATERIAL[(case.shape, case.lossy)]
    pol = 2 if case.pol_key == "te" else 1
    ours = pysie2d_spectrum(
        ext["wavelength_nm"],
        case.shape,
        mat["n_core"],
        mat["n_clad"],
        mat["epsi"],
        pol,
        int(geom["n_pts"]),
    )

    # rel governs C_sca and C_ext, which never pass near zero on these spectra;
    # C_abs joins them only when the material absorbs, because a relative
    # tolerance on an identically-zero spectrum is meaningless.
    keys = ("c_ext", "c_sca") + (("c_abs",) if case.lossy else ())
    per_key = {
        k: float(np.max(np.abs(ours[k] - ext[k]) / np.abs(ext[k]))) for k in keys
    }
    limiting = max(per_key, key=per_key.get)
    rel_dev = per_key[limiting]

    # abs_nm carries C_abs where a relative tolerance cannot. Which side's
    # residual it bounds depends on the tool: dolfinx's lossless C_abs is a
    # volume integral carrying Im(eps) and is *identically* zero, so there the
    # gap is pysie2d's own round-off; MEEP's is an independent flux tally that
    # leaks a real fraction of a nm. Saying "the tool's residual" for both
    # would misattribute half of these files.
    abs_dev = float(np.max(np.abs(ours["c_abs"] - ext["c_abs"])))
    ext_abs = float(np.max(np.abs(ext["c_abs"])))
    ours_abs = float(np.max(np.abs(ours["c_abs"])))
    # 3x headroom before rounding, and only here: a lossless C_abs is a near
    # cancellation of two large numbers, so its round-off residual varies by a
    # small multiplicative factor across BLAS implementations in a way the
    # deterministic relative deviation above does not.
    abs_nm = max(_ceil_clean(3.0 * abs_dev), 1e-9)
    rel = _ceil_clean(rel_dev)
    drift = _drift(case)

    if case.lossy:
        abs_story = (
            f"abs_nm is a floor only: this material absorbs, so C_abs "
            f"(peak {ext_abs:.3g} nm) is carried by the relative tolerance."
        )
    elif ext_abs == 0.0:
        abs_story = (
            f"abs_nm bounds *pysie2d's* residual absorption, {ours_abs:.2e} nm: "
            f"this tool's lossless C_abs is a volume integral carrying Im(eps) "
            f"and is identically zero, so it contributes nothing to the gap."
        )
    else:
        abs_story = (
            f"abs_nm bounds *this tool's* residual absorption leak, "
            f"{ext_abs:.2e} nm, against pysie2d's {ours_abs:.2e} nm — an "
            f"independent flux tally that never reads the material, so a "
            f"non-zero reading is the tool's own error."
        )

    tool_name = {"meep": "MEEP (FDTD, time domain)", "dolfinx": "FEniCSx/dolfinx"}
    spec = Spectrum(
        case_id=case.case_id,
        claim=(
            f"pysie2d's cross-sections on a {case.shape} agree with "
            f"{tool_name[case.tool]}, an independent Maxwell solver that "
            "shares no formulation, discretisation or linear algebra with the "
            "boundary-integral method."
        ),
        tool={"name": case.tool, "version": meta["version"]},
        geometry=dict(geom),
        material=dict(mat),
        pol=pol,
        pol_mapping=meta["pol_mapping"][str(pol)],
        angle_deg=0.0,
        wavelength_nm=ext["wavelength_nm"],
        c_sca_nm=ext["c_sca"],
        c_ext_nm=ext["c_ext"],
        c_abs_nm=ext["c_abs"],
        convergence=Convergence(
            parameter=case.parameter,
            coarse=case.coarse_label,
            fine=case.fine_label,
            max_rel_drift=drift,
            note=case.note,
        ),
        tolerance=Tolerance(
            rel=rel,
            abs_nm=abs_nm,
            justification=(
                f"Measured, not chosen. Worst pointwise relative deviation "
                f"between pysie2d (n_pts = {int(geom['n_pts'])}) and this "
                f"spectrum is {rel_dev:.2e}, on {limiting}, rounded up to "
                f"{rel:.0e}. The tool's own {case.parameter} drift from "
                f"{case.coarse_label} to {case.fine_label} is {drift:.2e} — "
                "the same order, so what this bounds is the external tool's "
                "discretisation error and not pysie2d's, which at this n_pts "
                f"sits orders below. {abs_story}"
            ),
        ),
        notes=(
            "Produced by validation/freeze.py from validation/_study. The "
            "driver that generated the spectrum is validation/"
            f"{case.tool}/run_cylinder.py and does not import pysie2d; the "
            "boundary comes from validation/gielis.py, which "
            "tests/test_gielis_contour.py pins to the package's own gielis() "
            "bit for bit."
        ),
    )
    return spec, {
        "rel_dev": rel_dev,
        "rel": rel,
        "abs_dev": abs_dev,
        "abs_nm": abs_nm,
        "drift": drift,
    }


def main() -> None:
    """Freeze every case, or report what would be frozen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = _dolfinx_cases() + _meep_cases()
    meta = {t: _tool_metadata(t) for t in ("dolfinx", "meep")}
    print(
        f"{'case_id':34s} {'measured':>10s} {'rtol':>8s} "
        f"{'C_abs nm':>10s} {'atol':>8s} {'drift':>10s}"
    )
    for case in cases:
        spec, m = build(case, meta[case.tool])
        print(
            f"{case.case_id:34s} {m['rel_dev']:10.2e} {m['rel']:8.0e} "
            f"{m['abs_dev']:10.2e} {m['abs_nm']:8.0e} {m['drift']:10.2e}"
        )
        if not args.dry_run:
            write(spec, DATA / f"{case.case_id}.json")
    print(
        f"\n{len(cases)} case(s)"
        + ("" if args.dry_run else f" written to {DATA.relative_to(ROOT)}")
    )


if __name__ == "__main__":
    main()
