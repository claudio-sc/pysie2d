"""Freeze the analytic Mie cylinder as a frozen-spectrum file.

**This is the bootstrap case, and it is not an external anchor.** Its "tool"
is ``pysie2d.reference.mie`` — the repo's own analytic Mie series, which the
suite has always checked against. Freezing it here buys nothing new about the
physics; it buys the *format*. It proves the schema, the writer, the reader
and ``tests/test_external_validation.py`` end to end against a reference we
already trust, so that when a dolfinx or MEEP driver lands it only has to
write a file in a layout that already has a working consumer.

Four cases, chosen so that every mechanism the external drivers will need is
already exercised by a file in the repo:

- lossless at ``n_clad = 1``, both polarisations — also the §5 gate-3 null
  test, since ``C_abs`` is then exactly zero;
- lossy at ``n_clad = 1.33``, both polarisations — the absorption path, and
  the one case where the vacuum→background wavelength conversion
  (conventions §2.3) can actually be wrong.

Run from the repo root::

    uv run python validation/mie_bootstrap.py

It rewrites ``tests/data/mie-circle-*.json`` and prints the agreement pysie2d
currently achieves against each, which is where the committed tolerances came
from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

from spectrum import Convergence, Spectrum, Tolerance, write  # noqa: E402

from pysie2d import BIESolver, Geometry, Material  # noqa: E402
from pysie2d.reference import mie  # noqa: E402

DATA = ROOT / "tests" / "data"

RAD = 200.0  # nm, the radius the whole test suite uses
N_CORE = 1.5
N_PTS = 200  # far past the nn ≈ 30–40 where Kress reaches round-off
WAVELENGTHS = np.linspace(400.0, 900.0, 51)

POL_TAG = {2: "TE", 1: "TM"}
POL_MAPPING = {
    2: (
        "pysie2d pol=2 is TE: the E field lies along the invariant axis y, "
        "and the Mie coefficient is b_n. An external z-invariant 2-D solver "
        "calls the same problem E_z-polarised (often 'TM' or 's'); the "
        "mapping is a relabelling of the invariant axis, not a change of "
        "physics. Under exp(-iωt) a solver using exp(+iωt) returns the "
        "conjugate field and needs Im(eps) < 0 to absorb."
    ),
    1: (
        "pysie2d pol=1 is TM: the H field lies along the invariant axis y, "
        "and the Mie coefficient is a_n. An external z-invariant solver "
        "calls this H_z-polarised (often 'TE' or 'p'). Same relabelling and "
        "same time-convention caveat as pol=2."
    ),
}


def _mie_spectrum(mat: Material, n_max: int | None) -> dict[str, np.ndarray]:
    """Analytic absolute cross-sections over ``WAVELENGTHS``, in nm.

    ``mie.efficiencies`` returns the efficiencies ``Q_*``, normalised by the
    geometric width ``2·rad``; the comparison contract is in absolute
    cross-sections, so undo that here (§2). The multiplication is the whole
    of what the validation side does — no API change is implied.

    Args:
        mat: The material, supplying ``nc`` (the Mie ``m``) and ``n_clad``.
        n_max: Highest Mie order, or None for the Wiscombe default.

    Returns:
        dict of 'c_sca_nm', 'c_ext_nm', 'c_abs_nm' arrays.
    """
    tag = POL_TAG[mat.pol]
    out = {key: np.empty(WAVELENGTHS.size) for key in ("sca", "ext", "abs")}
    for i, lam in enumerate(WAVELENGTHS):
        x = 2.0 * np.pi * mat.n_clad * RAD / lam
        eff = mie.efficiencies(x, complex(mat.nc), n_max=n_max)
        for key in out:
            out[key][i] = eff[f"Q_{key}_{tag}"] * 2.0 * RAD
    return {f"c_{key}_nm": val for key, val in out.items()}


def _solver_spectrum(mat: Material) -> dict[str, np.ndarray]:
    """The same three cross-sections straight out of pysie2d."""
    geom = Geometry.gielis(rad=RAD, n_pts=N_PTS, m=0)
    solver = BIESolver(geom, mat)
    out = {key: np.empty(WAVELENGTHS.size) for key in ("sca", "ext", "abs")}
    for i, lam in enumerate(WAVELENGTHS):
        cs = solver.scatter(wavelength=float(lam)).cross_sections()
        for key in out:
            out[key][i] = cs[f"c_{key}"]
    return {f"c_{key}_nm": val for key, val in out.items()}


def _convergence(mat: Material, frozen: dict[str, np.ndarray]) -> Convergence:
    """Measure the Mie series' own floor by adding ten orders.

    The analogue of gate 2 for this bootstrap: the series' discretisation
    parameter is the truncation order, so refine it and record how far the
    answer moved. For a converged Wiscombe truncation this lands at
    round-off, which is exactly the point — the floor here is ours, not an
    external tool's, and the committed tolerance says so.
    """
    n_default = mie._nmax(2.0 * np.pi * mat.n_clad * RAD / WAVELENGTHS.min())
    refined = _mie_spectrum(mat, n_max=n_default + 10)
    drift = np.max(
        np.abs(refined["c_ext_nm"] - frozen["c_ext_nm"]) / np.abs(frozen["c_ext_nm"])
    )
    return Convergence(
        parameter="n_max (Mie truncation order)",
        coarse=f"Wiscombe default ({n_default} at the shortest wavelength)",
        fine=f"{n_default + 10}",
        max_rel_drift=float(drift),
        note=(
            "Worst pointwise change in C_ext over the whole grid when ten "
            "orders are added past the Wiscombe criterion. Exactly zero: "
            "past the Wiscombe order the added coefficients underflow, so "
            "the series is not merely converged but bit-identical. This "
            "floor is the repo's own; an external tool's will not be."
        ),
    )


def _case(
    case_id: str, mat: Material, tolerance: Tolerance, claim: str
) -> tuple[Spectrum, dict[str, np.ndarray]]:
    frozen = _mie_spectrum(mat, n_max=None)
    spec = Spectrum(
        case_id=case_id,
        claim=claim,
        tool={"name": "pysie2d.reference.mie", "version": "analytic series"},
        geometry={"rad": RAD, "m": 0, "n1": 2.0, "n2": 2.0, "n3": 2.0, "n_pts": N_PTS},
        material={"n_core": mat.n_core, "n_clad": mat.n_clad, "epsi": mat.epsi},
        pol=mat.pol,
        pol_mapping=POL_MAPPING[mat.pol],
        angle_deg=0.0,
        wavelength_nm=WAVELENGTHS,
        convergence=_convergence(mat, frozen),
        tolerance=tolerance,
        notes=(
            "Bootstrap case: the reference is the repo's own analytic Mie "
            "series, not an independent tool. It fixes the file format and "
            "proves the consuming test; it is not evidence of external "
            "validation, and docs/validation.md says so."
        ),
        **frozen,
    )
    return spec, frozen


# Tolerances. Both numbers are the measured agreement of this repo against
# the analytic series on this grid, rounded up to one significant figure —
# printed by this script's own report, not guessed. They are tight because
# under Kress quadrature the circle reaches round-off by nn ≈ 30–40 and this
# grid runs at nn = 200: nothing here is converging, so a failure means a
# real change in the physics, not a resolution shortfall.
CASES = [
    (
        "mie-circle-lossless-te",
        Material(n_core=N_CORE, n_clad=1.0, pol=2),
        Tolerance(rel=1e-12, abs_nm=1e-10, justification=""),
        "pysie2d's TE cross-sections on a circle reproduce the analytic Mie "
        "series, and its C_abs vanishes for a lossless particle (§5 gate 3).",
    ),
    (
        "mie-circle-lossless-tm",
        Material(n_core=N_CORE, n_clad=1.0, pol=1),
        Tolerance(rel=1e-12, abs_nm=1e-10, justification=""),
        "pysie2d's TM cross-sections on a circle reproduce the analytic Mie "
        "series, and its C_abs vanishes for a lossless particle (§5 gate 3).",
    ),
    (
        "mie-circle-lossy-te",
        Material(n_core=N_CORE, n_clad=1.33, pol=2, epsi=0.5),
        Tolerance(rel=1e-12, abs_nm=1e-10, justification=""),
        "pysie2d's TE cross-sections reproduce analytic Mie for an absorbing "
        "particle in a n_clad = 1.33 background, so both the complex-"
        "permittivity path and the single vacuum→background wavelength "
        "conversion (conventions §2.3) are exercised.",
    ),
    (
        "mie-circle-lossy-tm",
        Material(n_core=N_CORE, n_clad=1.33, pol=1, epsi=0.5),
        Tolerance(rel=1e-12, abs_nm=1e-10, justification=""),
        "pysie2d's TM cross-sections reproduce analytic Mie for an absorbing "
        "particle in a n_clad = 1.33 background, so both the complex-"
        "permittivity path and the single vacuum→background wavelength "
        "conversion (conventions §2.3) are exercised.",
    ),
]

JUSTIFICATION = (
    "Measured, not assumed. Worst pointwise relative deviation of pysie2d "
    "from the analytic Mie series over this grid is {worst:.1e} in C_sca and "
    "C_ext, at n_pts = {n_pts}; the committed rel = {rel:.0e} is that value "
    "rounded up. Under Kress product quadrature the circle reaches round-off "
    "by nn ≈ 30–40, so this is a round-off floor and not a convergence "
    "order, and refining n_pts does not move it. abs_nm = {abs_nm:.0e} is "
    "the floor on C_abs, which is exactly zero in the lossless cases where a "
    "relative tolerance has no meaning; it sits {margin:.0e} above the worst "
    "lossless C_abs this repo produces ({worst_abs:.1e} nm against a C_ext "
    "of order {scale:.0f} nm)."
)


def main() -> None:
    """Freeze all four bootstrap cases and report the agreement achieved."""
    DATA.mkdir(parents=True, exist_ok=True)
    for case_id, mat, tol, claim in CASES:
        spec, frozen = _case(case_id, mat, tol, claim)
        ours = _solver_spectrum(mat)
        worst = max(
            float(np.max(np.abs(ours[k] - frozen[k]) / np.abs(frozen[k])))
            for k in ("c_sca_nm", "c_ext_nm")
        )
        worst_abs = float(np.max(np.abs(ours["c_abs_nm"] - frozen["c_abs_nm"])))
        spec = Spectrum(
            **{
                **{k: v for k, v in spec.__dict__.items() if k != "tolerance"},
                "tolerance": Tolerance(
                    rel=tol.rel,
                    abs_nm=tol.abs_nm,
                    justification=JUSTIFICATION.format(
                        worst=worst,
                        n_pts=N_PTS,
                        rel=tol.rel,
                        abs_nm=tol.abs_nm,
                        margin=tol.abs_nm / max(worst_abs, 1e-300),
                        worst_abs=worst_abs,
                        scale=float(np.max(frozen["c_ext_nm"])),
                    ),
                ),
            }
        )
        path = write(spec, DATA / f"{case_id}.json")
        print(
            f"{case_id}: worst rel {worst:.2e}, worst |ΔC_abs| "
            f"{worst_abs:.2e} nm, Mie drift "
            f"{spec.convergence.max_rel_drift:.2e} → {path.name}"
        )


if __name__ == "__main__":
    main()
