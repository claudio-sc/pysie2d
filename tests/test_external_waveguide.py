"""Guided-mode S-parameters of a non-circular cluster against MEEP (roadmap A9).

The frozen values are MEEP eigenmode coefficients for two Si Gielis particles
(a rounded triangle and a rounded square) 200 nm above an SOI slab, extrapolated
to zero grid spacing (tests/data/waveguide/meep-soi-pair200.json carries the
geometry, the raw ladder and the provenance). MEEP shares none of pysie2d's
machinery: no Green function, no Sommerfeld path, no residue. Its guided mode
comes from MPB, not from ``Stack.guided_modes``.

Each tolerance comes from MEEP's convergence evidence alone: three times the
spread between two independent extrapolations plus MEEP's measured noise floor
(validation/freeze_waveguide.py). Observables whose tolerance exceeds half
their signal are recorded but not asserted (TE reflection at this gap).

It cannot pass by accident. The cluster removes 0.5–0.9 % (TE) and 3–8 % (TM)
of the incident power, and the tolerances are a small fraction of that. The
pair is mirror-asymmetric, so left and right drives give different |r|² (TM,
by up to 2×), and transmission phase is asserted as a complex number.

Why 200 nm and not the 50 nm of the design studies: at 50 nm MEEP's TM error
converges slowly in the narrow gap (circle, 6 % of the scattered power at
res 20–25, 2.4 % at res 40), while at 200 nm it falls to 0.5 %, like TE.

One known gap: TM complex t misses its tolerance by 1.3× (|Δt| ≤ 1.1e-3 against
7.8e-4 at 6–8 of 31 λ), under 1 % of |1 − t|. It is a strict xfail, not a wider
tolerance; see ``test_tm_transmission_phase_known_gap``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pysie2d import Cluster, ClusterBIESolver, Geometry, Material, Stack

DATA = json.loads(
    (Path(__file__).parent / "data" / "waveguide" / "meep-soi-pair200.json").read_text()
)
GEO = DATA["geometry"]
LAM = np.array(DATA["wavelength_nm"])
POL = {"te": 2, "tm": 1}
# Every sixth wavelength: the ends and four interior points keep the band's
# shape (the error grows towards both edges) at a fifth of the cost.
IDX = np.arange(0, LAM.size, 6)
# 200 nodes per particle: the cluster's |S| changes by < 1e-11 from 200 to 400.
N_PTS = 200
SHAPE_KEYS = ("rad", "m", "n1", "n2", "n3")
# TM complex t at 1.3× its MEEP-only tolerance. Every pysie2d check is clean
# (Sommerfeld knobs move t ≤ 7e-16, residue vs near-field projection ≤ 7e-12,
# R_TM vs the Airy closed form 1e-15), while MEEP's TM error falls with
# resolution and with the gap (circle: 6.1 % → 0.5 % of the scattered power
# from 50 to 200 nm). The TM ladder is non-monotone at 1450 nm, so its
# first-order spread likely understates MEEP's uncertainty. A res-40 rung
# would decide; until it runs the gap is recorded, not tolerated.
KNOWN_GAP = {("tm", "t_left"), ("tm", "t_right")}


def _solver(case):
    geos = [
        Geometry.gielis(
            n_pts=N_PTS,
            x0=p["x0_nm"],
            z0=p["z0_nm"],
            **{k: p[k] for k in SHAPE_KEYS},
        )
        for p in GEO["cluster"]
    ]
    s = GEO["stack"]
    stack = Stack(((s["core_n"] ** 2, s["core_nm"]),), s["clad_n"] ** 2)
    mat = Material(n_core=GEO["particles_n"], n_clad=s["clad_n"], pol=POL[case])
    return ClusterBIESolver(
        Cluster(geos), [mat] * len(geos), pol=POL[case], background=stack
    )


@pytest.fixture(scope="module", params=("te", "tm"))
def computed(request):
    case = request.param
    sol, x = _solver(case), DATA["geometry"]["plane_nm"]
    out = {k: [] for k in ("t_left", "r_left", "t_right", "r_right")}
    for lam in LAM[IDX]:
        a = sol.scatter_mode(lam, 0, +1, x_ref=-x).mode_amplitudes(-x)
        b = sol.scatter_mode(lam, 0, -1, x_ref=x).mode_amplitudes(x)
        out["t_left"].append(a.s_plus[0])
        out["r_left"].append(a.s_minus[0])
        out["t_right"].append(b.s_minus[0])
        out["r_right"].append(b.s_plus[0])
    out = {k: np.array(v) for k, v in out.items()}
    for side in ("left", "right"):
        out[f"t2_{side}"] = np.abs(out[f"t_{side}"]) ** 2
        out[f"r2_{side}"] = np.abs(out[f"r_{side}"]) ** 2
        out[f"p_rad_{side}"] = 1 - out[f"t2_{side}"] - out[f"r2_{side}"]
    return case, out


def _assert_matches(case, ours, k):
    frozen = DATA["cases"][case]
    meep = np.array(frozen["values"][k])[IDX]
    if meep.ndim == 2:  # complex, stored as [re, im]
        meep = meep[:, 0] + 1j * meep[:, 1]
    np.testing.assert_allclose(
        ours[k], meep, rtol=0, atol=frozen["tolerance_abs"][k], err_msg=f"{case} {k}"
    )


def test_guided_mode_s_parameters_match_meep(computed):
    case, ours = computed
    frozen = DATA["cases"][case]
    checked = 0
    for k in frozen["tolerance_abs"]:
        if not frozen["asserted"][k] or (case, k) in KNOWN_GAP:
            continue
        _assert_matches(case, ours, k)
        checked += 1
    # Transmission must be asserted in both directions: phase included in TE,
    # |t|² in TM while its phase is the known gap.
    assert checked >= 4


@pytest.mark.xfail(strict=True, reason="TM complex t at 1.3× its MEEP tolerance")
def test_tm_transmission_phase_known_gap(computed):
    # strict: a re-freeze that closes the gap turns this into a failure, which
    # is the prompt to move TM t back into the main assertion.
    case, ours = computed
    if case != "tm":
        pytest.skip("the known gap is TM only")
    for side in ("left", "right"):
        _assert_matches(case, ours, f"t_{side}")
