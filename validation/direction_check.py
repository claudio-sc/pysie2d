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
pysie2d and MEEP do not share an observation-angle origin: pysie2d measures θ
from **+z** (``x = r sin θ``, ``z = r cos θ``) and MEEP's far-field circle is
parametrised from **+x**, so the two frames differ by a rotation *and*
possibly a reflection. Pinning that mapping by derivation would be the same
kind of unchecked convention this script exists to remove — and guessing it
wrong is indistinguishable from a failed check, which is how the first version
of this script reported 0.99 for both incidences.

So the pattern is compared under **every** map relating two uniform angular
frames: all ``N_FAR`` rotations, with and without a reflection, and the best
is reported for each incidence. The check is not that some map matches — one
always will, to within how well the patterns agree at all. It is that the map
which matches ``a0`` does **not** rescue ``a180``, by a margin far larger than
MEEP's own pattern error. A margin near 1 means the pattern failed to
discriminate too, and it is reported as such rather than read as a pass.

Usage (both under uv; the MEEP runs come from the driver)::

    conda run -n pysie2d-meep python validation/meep/run_cylinder.py \\
        --shape skew --case te --resolution 25 --grid study \\
        --angle-deg 0 --out validation/_study/meep/skew-dir-a0-te.npz
    uv run python validation/direction_check.py
"""

from __future__ import annotations

import argparse
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


OVERSAMPLE = 10
"""How much finer pysie2d's angular grid is than MEEP's.

The frames' relative rotation is a real number, not a multiple of MEEP's 1°
sample spacing, and rounding it to the nearest sample is itself a mismatch —
on a pattern with lobes a few degrees wide, half a sample of misalignment is
worth more than the difference the check is trying to see. pysie2d's far field
is cheap, so its grid is sampled ``OVERSAMPLE`` times finer and the offset is
scanned on that grid; MEEP's samples are never interpolated.
"""


def _best_match(fine: np.ndarray, theirs: np.ndarray) -> tuple[float, str]:
    """Best agreement over every rotation and reflection of the angle axis.

    Both grids are uniform and closed, so a rotation is an exact index roll
    and introduces no interpolation error of its own. The search covers all
    ``OVERSAMPLE · N_FAR`` offsets, with and without a reflection, rather than
    a guessed subgroup — guessing it wrong is indistinguishable from a failed
    check.

    Args:
        fine: pysie2d's pattern on the oversampled grid.
        theirs: MEEP's pattern on its own grid.

    Returns:
        The worst pointwise ``|Δ|`` under the best map, and that map's angle.
    """
    best, name = np.inf, ""
    for mirrored, candidate in (("", theirs), ("mirror+", theirs[::-1])):
        for shift in range(fine.shape[0]):
            ours = np.roll(fine, shift, axis=0)[::OVERSAMPLE]
            score = float(np.max(np.abs(ours - candidate)))
            if score < best:
                best = score
                name = f"{mirrored}rot-{360.0 * shift / fine.shape[0]:.1f}deg"
    return best, name


def _normalise(pattern: np.ndarray) -> np.ndarray:
    """Scale each wavelength's pattern to unit peak.

    The comparison is about *shape*: the overall level is the integrated
    cross-section, which gates 1–3 already check and which reciprocity has
    just shown cannot see the direction anyway.
    """
    return pattern / np.max(np.abs(pattern), axis=0, keepdims=True)


def pysie2d_pattern(wavelengths: np.ndarray, n_angles: int, angle_deg: float):
    """dC_sca/dφ from pysie2d on a uniform grid of ``n_angles``, unit peak."""
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
    out = np.empty((n_angles, wavelengths.size))
    for j, lam in enumerate(wavelengths):
        amp, _ = solver.scatter(wavelength=float(lam), angle=angle_deg).far_field(
            n_angles=n_angles + 1
        )
        # far_field spans [-π, π] inclusive, so the last point repeats the
        # first; dropping it leaves a uniform closed grid, as MEEP's is.
        out[:, j] = np.abs(amp[:-1]) ** 2
    return _normalise(out)


def main() -> None:
    """Compare both MEEP incidences against pysie2d's angle = 0 pattern."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="",
        help='suffix of the MEEP run pair to read, e.g. "32" for the res-32 runs',
    )
    tag = parser.parse_args().tag
    runs = {}
    for angle in (0, 180):
        path = STUDY / f"skew-dir{tag}-a{angle}-te.npz"
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
    ours = pysie2d_pattern(wavelengths, angles.size * OVERSAMPLE, angle_deg=0.0)

    print(f"skew shape, TE, {wavelengths.size} wavelengths, {angles.size} angles")
    print("worst pointwise |Δ| of the unit-peak pattern, best transform first\n")
    best = {}
    for angle, run in runs.items():
        score, name = _best_match(ours, _normalise(run["far_pattern"]))
        best[angle] = score
        print(f"  MEEP angle_deg={angle:3d} vs pysie2d angle=0: {score:.3e} [{name}]")

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
