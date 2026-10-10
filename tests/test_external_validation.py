"""Assert pysie2d against the frozen validation spectra.

This is layer 3 of docs/design/v1.0-external-validation.md §6: the external
tools' job is to *produce* a reference once, not to be present when it is
checked. Every file in ``tests/data`` is read with the standard library and
compared with numpy, so this runs for every user of the repo with no conda
environment, no MEEP and no dolfinx — and it fails if pysie2d's answers move.

These tests cannot pass by accident. Each file pins a full cross-section
spectrum, not a scalar: a sign error, a lost factor of ``n_clad``, a
polarisation swap or a wrong incidence convention moves every point of it at
once, far outside a tolerance that sits at round-off.

The Mie cases are **bootstrap** files whose reference is the repo's own
analytic series (see ``validation/mie_bootstrap.py``). They prove the format
and this consumer; they are not independent anchors, and no claim of external
validation rests on them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from pysie2d import BIESolver, Geometry, Material

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

import spectrum as spectrum_io  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
CASES = sorted(DATA.glob("*.json"))
IDS = [p.stem for p in CASES]


@pytest.fixture(scope="module", params=CASES, ids=IDS)
def frozen(request):
    """One frozen spectrum, and the pysie2d spectrum to compare it against.

    Computed once per file: a 51-wavelength sweep at nn = 200 is a second of
    work, and every test in this module wants the same arrays.
    """
    spec = spectrum_io.read(request.param)
    geom = Geometry.gielis(
        rad=spec.geometry["rad"],
        n_pts=int(spec.geometry["n_pts"]),
        m=int(spec.geometry["m"]),
        n1=spec.geometry["n1"],
        n2=spec.geometry["n2"],
        n3=spec.geometry["n3"],
    )
    mat = Material(
        n_core=spec.material["n_core"],
        n_clad=spec.material["n_clad"],
        pol=spec.pol,
        epsi=spec.material["epsi"],
    )
    solver = BIESolver(geom, mat)
    ours = {
        key: np.empty(spec.wavelength_nm.size) for key in ("c_sca", "c_ext", "c_abs")
    }
    for i, lam in enumerate(spec.wavelength_nm):
        cs = solver.scatter(
            wavelength=float(lam), angle=spec.angle_deg
        ).cross_sections()
        for key in ours:
            ours[key][i] = cs[key]
    return spec, ours


@pytest.mark.parametrize("observable", ["c_sca", "c_ext"])
def test_cross_section_spectrum_matches_frozen(frozen, observable):
    # The core assertion: the whole spectrum, pointwise, against a reference
    # produced outside this solver. The tolerance is not chosen here — it
    # travels in the file with the measurement that set it, so widening it
    # means editing the recorded justification, which is reviewable.
    spec, ours = frozen
    reference = getattr(spec, f"{observable}_nm")
    np.testing.assert_allclose(
        ours[observable],
        reference,
        rtol=spec.tolerance.rel,
        atol=spec.tolerance.abs_nm,
        err_msg=f"{spec.case_id}: {observable} disagrees with the frozen "
        f"spectrum. Tolerance justification: {spec.tolerance.justification}",
    )


def test_absorption_matches_frozen(frozen):
    # C_abs is split out because in the lossless cases it is *exactly* zero,
    # where a relative tolerance is meaningless — this is §5's null test, and
    # the only tolerance that can carry it is the absolute floor.
    spec, ours = frozen
    np.testing.assert_allclose(
        ours["c_abs"],
        spec.c_abs_nm,
        rtol=spec.tolerance.rel,
        atol=spec.tolerance.abs_nm,
        err_msg=f"{spec.case_id}: C_abs disagrees with the frozen spectrum.",
    )


def test_lossless_case_absorbs_nothing(frozen):
    # The null test stated as physics rather than as a comparison: a real
    # permittivity cannot absorb, so C_abs must sit at the numerical floor and
    # not merely agree with a reference that might share the same error.
    spec, ours = frozen
    if spec.material["epsi"] != 0.0:
        pytest.skip("lossy case")
    assert np.max(np.abs(ours["c_abs"])) < spec.tolerance.abs_nm


def test_frozen_file_is_self_describing(frozen):
    # Provenance is the deliverable, not decoration (§6): a file that has lost
    # its convergence evidence or its tolerance justification can no longer
    # support the claim it was frozen to support, whatever its numbers say.
    spec, _ = frozen
    assert spec.claim.strip()
    assert spec.tool["name"] and spec.tool["version"]
    assert spec.pol in (1, 2)
    assert spec.pol_mapping.strip()
    assert spec.convergence.parameter.strip()
    assert spec.convergence.max_rel_drift >= 0.0
    assert spec.tolerance.justification.strip()
    assert np.all(np.diff(spec.wavelength_nm) > 0.0)
