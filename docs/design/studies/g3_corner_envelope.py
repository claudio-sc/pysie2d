"""G3: near-corner validity envelope of Kress on the superellipse path.

`m = 4, a = b = 1, n1 = n2 = n3 = n` gives `|cos θ|^n + |sin θ|^n = r^-n`, a
circle at `n = 2` and a square as `n → ∞`. Even `n` only: an odd exponent makes
`|cos θ|^n` finitely smooth at the axes, and the rate would then measure that
regularity rather than corner sharpness.

Three maps through **production** code (`Parametrisation`, `Geometry.gielis`,
`BIESolver.scatter`): uniform θ (the v0.6 default), uniform arc length (band
20–20) and adaptive (band 20–100). Probe `qext`, TE, n_core 1.5, λ 600 nm,
against uniform-θ Kress at `NN_REF`. Its floor is measured twice: against
itself at `NN_REF_CHECK`, and against the arc-length map at `NN_REF` — a
different node set, so agreement cannot be a shared-node accident.

Usage: `uv run python docs/design/studies/g3_corner_envelope.py N`, one exponent
per process. Prints one JSON line.

**Production `Parametrisation.gielis` raises on every `n ≥ 4` superellipse**:
the sides are flat points, `κ = 0` on the axes, so `ln|κ|` in `_density` is
−∞ and the NaN density never resolves. The graded maps below therefore wrap
`_speed_curvature` to return `hypot(κ, ε·max|κ|)` — a smooth relative floor,
so no absolute length enters (§9) — and run at two `ε` to show the floor does
not set the answer. Uniform θ needs no density and is untouched.
"""

import json
import sys

import numpy as np

from pysie2d.geometry import Geometry
from pysie2d.material import Material
import pysie2d.parametrisation as pmod
from pysie2d.parametrisation import Parametrisation
from pysie2d.solver import BIESolver

RAD, N_CORE, POL, LAM = 200.0, 1.5, 2, 600.0
NNS = (40, 80, 160, 320, 640)
NN_REF = 1280
NN_REF_CHECK = 960
FLOORS = (1e-3, 1e-6)
_speed_curvature = pmod._speed_curvature


def _floored(eps: float):
    """`_speed_curvature` with |κ| smoothly floored at `eps·max|κ|`."""

    def wrapped(*rr):
        gam, kappa = _speed_curvature(*rr)
        return gam, np.hypot(kappa, eps * np.abs(kappa).max())

    return wrapped


def qext(par: Parametrisation, nn: int, n: float) -> float:
    """TE `qext` of the `n` superellipse at `nn` nodes on map `par`."""
    geom = Geometry.gielis(
        RAD, nn, m=4, n1=n, n2=n, n3=n, a=1.0, b=1.0, parametrisation=par
    )
    mat = Material(n_core=N_CORE, n_clad=1.0, pol=POL)
    return BIESolver(geom, mat).scatter(LAM).efficiencies()["qext"]


def run(n: float) -> None:
    """One exponent: map diagnostics, reference floor, error ladder per map."""
    shape = {"rad": RAD, "a": 1.0, "b": 1.0, "m": 4, "n1": n, "n2": n, "n3": n}
    out = {"n": n}
    maps = {"theta": Parametrisation.uniform_theta()}
    try:
        Parametrisation.gielis(**shape, n_core=N_CORE, wavelength_ref=LAM)
        out["production_gielis"] = "ok"
    except (ValueError, ZeroDivisionError) as exc:
        out["production_gielis"] = type(exc).__name__
    for eps in FLOORS:
        pmod._speed_curvature = _floored(eps)
        for label, band in (("arc", (20.0, 20.0)), ("adapt", (20.0, 100.0))):
            key = f"{label}_{eps:g}"
            try:
                maps[key] = Parametrisation.gielis(
                    **shape, r_band=band, n_core=N_CORE, wavelength_ref=LAM
                )
                out[f"{key}_K"] = maps[key].n_terms
                out[f"{key}_alpha_eff"] = maps[key].alpha_eff
                out[f"{key}_nn_band"] = maps[key].nn_from_band
            except (ValueError, ZeroDivisionError) as exc:
                out[f"{key}_error"] = type(exc).__name__
    pmod._speed_curvature = _speed_curvature
    ref = qext(maps["theta"], NN_REF, n)
    # Floor of the reference: the theta self-difference at a lower rung, and
    # the arc-length map at the same rung (a different node set).
    out["ref_floor_theta"] = abs(qext(maps["theta"], NN_REF_CHECK, n) - ref) / ref
    if f"arc_{FLOORS[0]:g}" in maps:
        out["ref_floor_arc"] = abs(qext(maps[f"arc_{FLOORS[0]:g}"], NN_REF, n) - ref) / ref
    out["qext_ref"] = ref
    for label, par in maps.items():
        out[f"{label}_err"] = [abs(qext(par, nn, n) - ref) / abs(ref) for nn in NNS]
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    run(float(sys.argv[1]))
