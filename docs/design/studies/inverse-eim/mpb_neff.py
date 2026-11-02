"""n_eff of an SOI strip's fundamental TE-like and TM-like modes, by MPB.

The 3D side of the inverse-EIM reproduction (Høvik & Aksnes, IEEE Photon. J.
16, 6600106, 2024): a rectangular Si core (n = 3.47) on SiO₂ (1.44) under air
(1.0), the paper's materials. Run in the MEEP conda environment, never under
uv; it does not import pysie2d::

    ~/miniforge3/envs/pysie2d-meep/bin/python mpb_neff.py W_NM H_NM RES [LAM_NM]

Prints one JSON line: ``{"w":…, "h":…, "res":…, "te":…, "tm":…, "s":…}``.

Units a = 1 µm. The cross-section is the MPB cell (y lateral, z vertical),
propagation along x, frequency fixed and k found (``find_k``). The two
y-parities are solved separately and each band 1 is classified by its
dominant transverse E component, so neither label depends on MPB's parity
sign convention: TE-like = E_y dominant, TM-like = E_z dominant.
"""

import json
import sys
import time

import meep as mp
import numpy as np
from meep import mpb

N_SI, N_OX, N_AIR = 3.47, 1.44, 1.0
CELL_Y, CELL_Z = 4.0, 4.0
"""Periodic cell, µm. The fundamental modes decay as exp(−γ·d) with
γ ≥ k₀·√(n_eff² − 1.44²) ≈ 4.5 µm⁻¹ at n_eff ≥ 2.0, so ≥ 1.8 µm of cladding
leaves < 3e-4 of the field amplitude at the cell edge."""


def solve(w_nm: float, h_nm: float, res: int, lam_nm: float = 1550.0) -> dict:
    w, h, f = w_nm / 1000, h_nm / 1000, 1000 / lam_nm
    geometry = [
        # Substrate fills the lower half; the core sits on it.
        mp.Block(mp.Vector3(mp.inf, mp.inf, CELL_Z / 2), center=mp.Vector3(0, 0, -CELL_Z / 4),
                 material=mp.Medium(index=N_OX)),
        mp.Block(mp.Vector3(mp.inf, w, h), center=mp.Vector3(0, 0, h / 2), material=mp.Medium(index=N_SI)),
    ]
    ms = mpb.ModeSolver(
        geometry_lattice=mp.Lattice(size=mp.Vector3(0, CELL_Y, CELL_Z)),
        geometry=geometry,
        default_material=mp.Medium(index=N_AIR),
        resolution=res,
        num_bands=1,
    )
    out = {}
    for parity in (mp.EVEN_Y, mp.ODD_Y):
        k = ms.find_k(parity, f, 1, 1, mp.Vector3(1), 1e-10, f * 2.5, f * N_OX, f * N_SI)[0]
        e = ms.get_efield(1, bloch_phase=False)
        ey, ez = np.sum(np.abs(e[..., 1]) ** 2), np.sum(np.abs(e[..., 2]) ** 2)
        out["te" if ey > ez else "tm"] = k / f
    return out


if __name__ == "__main__":
    w_nm, h_nm, res = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else 1550.0
    mp.verbosity(0)
    t0 = time.time()
    r = solve(w_nm, h_nm, res, lam)
    print(json.dumps({"w": w_nm, "h": h_nm, "res": res, "lam": lam, **r, "s": round(time.time() - t0, 2)}))
