r"""Pin the incident-direction convention against MEEP's far-field pattern.

Both drivers pin the incident wave to pysie2d's ``angle = 0``, travelling
along −z. Nothing in gate 1–3 can check that. On a circle the choice is
invisible by symmetry, and on the star it is invisible for a deeper reason:
**reciprocity**. The forward scattering amplitude for ``k̂`` equals the one for
``−k̂`` for any reciprocal scatterer, so ``C_ext`` is direction-invariant
whatever the shape, and losslessness carries ``C_sca`` with it. Measured, on a
deliberately asymmetric shape: reversing the incidence moves ``C_ext`` by
3e-13 in dolfinx and 2.5e-05 in MEEP.

The angular *pattern* is not invariant, and it is the only observable in this
milestone that can fail. On the skew shape at λ = 600 nm, pysie2d's own two
patterns differ by 5.1e-02 of the peak once the trivial π rotation is taken
out, and the forward/backward ratio differs by a factor of ~100.

Why the comparison allows a transform
-------------------------------------
pysie2d and MEEP do not share an observation-angle origin, and pinning that
mapping by derivation is the same kind of unchecked convention this script
exists to remove. So the pattern is compared under all four maps that relate
two orthonormal angular frames — identity, a π rotation, a reflection, and
both — and the *best* is reported for each incidence. The check is not that
some transform matches: it is that the transform which matches ``a0`` does
**not** rescue ``a180``, by a margin far larger than MEEP's own error. A
result where both match equally well means the pattern, too, failed to
discriminate, and it is reported as such rather than read as a pass.

Usage (both under uv; the MEEP runs come from the driver)::

    conda run -n pysie2d-meep python validation/meep/run_cylinder.py \\
        --shape skew --case te --resolution 25 --grid study \\
        --angle-deg 0 --out validation/_study/meep/skew-dir-a0-te.npz
    uv run python validation/direction_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

import gielis  # noqa: E402

from pysie2d import BIESolver, Geometry, Material  # noqa: E402

STUDY = ROOT / "validation" / "_study" / "meep"
SHAPE = "skew"
N_CORE = 2.0
N_CLAD = 1.0
POL = 2


def _transforms(pattern: np.ndarray) -> dict[str, np.ndarray]:
    """The four maps relating two angular frames with no shared origin.

    A rotation by π and a reflection of the angle axis; the angular grid is
    uniform and closed, so both are exact index operations and introduce no
    interpolation error of their own.
    """
    half = pattern.shape[0] // 2
    return {
        "identity": pattern,
        "rot-pi": np.roll(pattern, half, axis=0),
        "mirror": pattern[::-1],
        "mirror+rot-pi": np.roll(pattern[::-1], half, axis=0),
    }


def _normalise(pattern: np.ndarray) -> np.ndarray:
    """Scale each wavelength's pattern to unit peak.

    The comparison is about *shape*: the overall level is the integrated
    cross-section, which gates 1–3 already check and which reciprocity has
    just shown cannot see the direction anyway.
    """
    return pattern / np.max(np.abs(pattern), axis=0, keepdims=True)


def pysie2d_pattern(wavelengths: np.ndarray, angles: np.ndarray, angle_deg: float):
    """dC_sca/dφ from pysie2d on the same angular grid, unit peak per λ."""
    shape = gielis.SHAPES[SHAPE]
    geom = Geometry.gielis(
        rad=shape["rad"],
        n_pts=400,
        m=shape["m"],
        n1=shape["n1"],
        n2=shape["n2"],
        n3=shape["n3"],
    )
    solver = BIESolver(geom, Material(n_core=N_CORE, n_clad=N_CLAD, pol=POL, epsi=0.0))
    out = np.empty((angles.size, wavelengths.size))
    for j, lam in enumerate(wavelengths):
        amp, grid = solver.scatter(wavelength=float(lam), angle=angle_deg).far_field(
            n_angles=angles.size + 1
        )
        # far_field spans [-π, π] inclusive, so the last point repeats the
        # first; dropping it leaves a uniform closed grid matching MEEP's.
        out[:, j] = np.abs(amp[:-1]) ** 2
        del grid
    return _normalise(out)


def main() -> None:
    """Compare both MEEP incidences against pysie2d's angle = 0 pattern."""
    runs = {}
    for angle in (0, 180):
        path = STUDY / f"skew-dir-a{angle}-te.npz"
        if not path.exists():
            raise SystemExit(f"missing {path} — see this module's docstring")
        with np.load(path) as d:
            if "far_pattern" not in d.files:
                raise SystemExit(
                    f"{path.name} predates the pattern output; re-run the driver"
                )
            runs[angle] = {k: d[k] for k in d.files}

    wavelengths = runs[0]["wavelength_nm"]
    angles = runs[0]["far_angles"]
    ours = pysie2d_pattern(wavelengths, angles, angle_deg=0.0)

    print(f"skew shape, TE, {wavelengths.size} wavelengths, {angles.size} angles")
    print("worst pointwise |Δ| of the unit-peak pattern, best transform first\n")
    best = {}
    for angle, run in runs.items():
        theirs = _normalise(run["far_pattern"])
        scores = {
            name: float(np.max(np.abs(ours - candidate)))
            for name, candidate in _transforms(theirs).items()
        }
        best[angle] = min(scores.values())
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        print(f"  MEEP angle_deg={angle:3d} vs pysie2d angle=0")
        for name, score in ranked:
            print(f"      {name:14s} {score:.3e}")

    margin = best[180] / best[0] if best[0] else float("inf")
    print(f"\n  best a0   {best[0]:.3e}\n  best a180 {best[180]:.3e}")
    print(f"  discrimination margin {margin:.1f}x")
    print(
        "\nThe pinned -z convention is confirmed only if a0 matches and a180 "
        "does not,\nby a margin larger than MEEP's own pattern error. A margin "
        "near 1 means the\npattern did not discriminate either, and the "
        "convention stays unchecked."
    )


if __name__ == "__main__":
    main()
