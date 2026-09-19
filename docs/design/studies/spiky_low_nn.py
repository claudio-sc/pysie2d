"""Is uniform arc length the most accurate map for a spiky star at low `nn`?

Campaign question (14 Sep 2026): exploratory sweeps over star-shaped particles
run at `nn` 50–300 and want the most accurate map for a given `nn`, at a
useful accuracy of 1e-3 relative. Appendix A of kress-spec.md measured arm
ratios 1.7, 2.8, 8 and 64 at one wavelength; this fills 4 and 16 and adds two
wavelengths so a `qext` zero-crossing cannot pick the winner.

Stars `m = 6`, n2 = n3 = 8, arm ratio `2^(3/n1)` = 4, 8, 16, tip radius 400 nm.
Maps through production code: uniform θ, uniform arc length (band 20–20),
adaptive (band 20–100). Probe `qext`, TE, n_core 1.5. Reference: uniform θ at
`NN_REF`, floor = its difference from `NN_REF_CHECK` — an error below that
floor is not a measurement.

Usage: `uv run python docs/design/studies/spiky_low_nn.py RATIO LAMBDA`, one
job per process. Prints one JSON line.
"""

import json
import sys

import numpy as np

from pysie2d.geometry import Geometry
from pysie2d.material import Material
from pysie2d.parametrisation import Parametrisation
from pysie2d.solver import BIESolver

M, N23, TIP, N_CORE, POL = 6, 8.0, 400.0, 1.5, 2
NNS = (50, 75, 100, 150, 200, 300)
NN_REF, NN_REF_CHECK = 1600, 2000


def run(ratio: float, lam: float) -> None:
    """One (arm ratio, λ): reference floor and error per map per `nn`."""
    n1 = 3.0 / np.log2(ratio)
    rad = TIP / ratio  # r(tip) = rad·2^(3/n1), so the tip sits at TIP
    shape = {"m": M, "n1": n1, "n2": N23, "n3": N23, "a": 1.0, "b": 1.0}
    mat = Material(n_core=N_CORE, n_clad=1.0, pol=POL)

    def qext(par, nn):
        geom = Geometry.gielis(rad, nn, **shape, parametrisation=par)
        return BIESolver(geom, mat).scatter(lam).efficiencies()["qext"]

    out = {"ratio": ratio, "lam": lam, "n1": n1}
    maps = {"theta": Parametrisation.uniform_theta()}
    for label, band in (("arc", (20.0, 20.0)), ("adapt", (20.0, 100.0))):
        try:
            maps[label] = Parametrisation.gielis(
                rad=rad, **shape, r_band=band, n_core=N_CORE, wavelength_ref=lam
            )
        except (ValueError, ZeroDivisionError) as exc:
            out[f"{label}_error"] = type(exc).__name__
    ref = qext(maps["theta"], NN_REF)
    out["ref_floor"] = abs(qext(maps["theta"], NN_REF_CHECK) - ref) / abs(ref)
    # Newton in `nodes(nn)` can fail at one nn and not the next, so a failure is
    # recorded per rung (None) — it is part of what a campaign would hit.
    for label, par in maps.items():
        row = []
        for nn in NNS:
            try:
                row.append(abs(qext(par, nn) - ref) / abs(ref))
            except ValueError:
                row.append(None)
        out[label] = row
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]))
