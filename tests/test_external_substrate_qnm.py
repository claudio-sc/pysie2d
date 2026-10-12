"""Substrate QNMs against MEEP — the independent anchor for layered-spec G9.

The frozen poles are MEEP ringdowns fitted by harminv, one parity per run
(tests/data/qnm/meep-substrate-qnm.json carries the provenance). What is
compared is the shift a glass half-space imposes on each pole, each code
against itself: that cancels MEEP's own grid error, which at resolution 100 is
still 0.02–0.09 nm in absolute λ and would otherwise swamp the substrate
effect being tested.

It cannot pass by accident: the shifts are 0.4–1.4 nm against a 1e-2 nm
tolerance, and the two members of each split pair move differently (the TM
pair even moves Re λ in opposite directions), so matching the wrong pole, or a
substrate with the wrong sign of R(q), fails.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pysie2d import Geometry, HalfSpace, Material, QNMSolver

DATA = json.loads(
    (Path(__file__).parent / "data" / "qnm" / "meep-substrate-qnm.json").read_text()
)
GEO = DATA["geometry"]
TOL = DATA["tolerance"]["shift_abs_nm"]
POL = {"te": 2, "tm": 1}


def _meep(case: str, parity: int, eps: str) -> complex:
    return complex(*DATA["poles_nm"][f"{case},{parity:+d},{eps}"])


def _pysie2d_poles(case: str, background) -> np.ndarray:
    # The box is the high-Q study's final continuation box around the pair
    # (validation/_study/substrate-qnm-highq-2026-10-11/homotopy.py): it holds
    # exactly the split pair, which the continuation from the isolated circle
    # labelled. nn = 200 agrees with 400 to < 1e-12 nm there.
    centre = 0.5 * (_meep(case, 1, "2.25") + _meep(case, -1, "2.25"))
    geo = Geometry.gielis(
        GEO["rad_nm"], 200, m=0, x0=0.0, z0=GEO["rad_nm"] + GEO["gap_nm"]
    )
    mat = Material(n_core=GEO["n_core"], n_clad=GEO["n_clad"], pol=POL[case])
    lo = centre.real * 0.97 + 1j * centre.imag * 0.3
    hi = centre.real * 1.03 + 1j * centre.imag * 3.0
    res = QNMSolver(geo, mat, background).modes(lo, hi, n_quad_per_side=24, n_probe=24)
    return res.wavelengths


@pytest.mark.parametrize("case", ["te", "tm"])
def test_substrate_shift_of_split_wgm_pair_matches_meep(case):
    iso = _pysie2d_poles(case, HalfSpace(1.0, 0.0))
    glass = _pysie2d_poles(case, HalfSpace(complex(GEO["eps_sub"]), 0.0))
    assert iso.size == 2 and glass.size == 2
    lam_iso = iso.mean()  # degenerate on the isolated circle
    for parity in (1, -1):
        meep_shift = _meep(case, parity, "2.25") - _meep(case, parity, "1.0")
        # Each parity run rings one member of the pair; pair it with the
        # pysie2d pole whose shift it matches. Nearest is unambiguous: the two
        # members' shifts differ by 0.37–0.46 nm, 40× the tolerance.
        ours = glass - lam_iso
        assert np.abs(ours - meep_shift).min() < TOL
