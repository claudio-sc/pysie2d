"""pysie2d's side of v1.3: the cluster's S-parameters on MEEP's grid.

Runs in the uv environment. It reads the cluster from ``gielis.CLUSTER`` and the
contour *function* (not the committed polygon MEEP reads), so pysie2d sees the
analytic curve; the two agree to the 1440-vertex chord error, ~0.01 nm on these
shapes, far below a grid cell.

The phase reference ``x_ref`` is the MEEP plane at which each amplitude is
measured, so ``r`` from either code is the field ratio at the same plane:
``x_ref = −X`` for the left drive's reflection, ``+X`` for the right drive's.
``t`` does not depend on ``x_ref`` (incident and transmitted waves carry the
same ``e^{iβx}``).

    uv run python validation/waveguide_reference.py --out ref.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pysie2d import Cluster, ClusterBIESolver, Geometry, Material, Stack

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gielis  # noqa: E402
from soi import N_OX, N_SI, T_SI_NM, WAVELENGTHS  # noqa: E402

N_PTS = 400
"""Per particle. The cluster's |S| converges spectrally: 100 → 200 → 400 → 800
nodes change it by 2e-10, 7e-12, 2e-13 (TM, 1550 nm), so 400 sits at the
1e-12 floor — eight orders below MEEP's grid error."""


def cluster() -> Cluster:
    """The v1.3 cluster at the reference's node count."""
    return Cluster(
        [
            Geometry.gielis(n_pts=N_PTS, x0=x0, z0=z0, **gielis.SHAPES[shape])
            for shape, x0, z0 in gielis.CLUSTER
        ]
    )


def reference(plane_nm: float) -> dict[str, np.ndarray]:
    """t, r for both drives and both polarisations, r referred to ``±plane_nm``."""
    stack = Stack(((N_SI**2, T_SI_NM),), N_OX**2)
    out: dict[str, np.ndarray] = {"wavelengths": WAVELENGTHS}
    for case, pol in (("te", 2), ("tm", 1)):
        mats = [Material(n_core=N_SI, n_clad=N_OX, pol=pol)] * len(gielis.CLUSTER)
        sol = ClusterBIESolver(cluster(), mats, pol=pol, background=stack)
        rows = []
        for lam in WAVELENGTHS:
            (mode,) = stack.guided_modes(lam, pol, N_OX)  # single-mode by the grid
            a = sol.scatter_mode(lam, 0, +1, x_ref=-plane_nm).mode_amplitudes(-plane_nm)
            b = sol.scatter_mode(lam, 0, -1, x_ref=plane_nm).mode_amplitudes(plane_nm)
            rows.append(
                (
                    mode.n_eff,
                    a.s_plus[0],
                    a.s_minus[0],
                    b.s_minus[0],
                    b.s_plus[0],
                    a.p_up,
                    a.p_down,
                    b.p_up,
                    b.p_down,
                    a.balance(),
                    b.balance(),
                )
            )
        r = np.array(rows)
        names = (
            "n_eff",
            "t_left",
            "r_left",
            "t_right",
            "r_right",
            "up_left",
            "down_left",
            "up_right",
            "down_right",
            "balance_left",
            "balance_right",
        )
        for i, name in enumerate(names):
            col = r[:, i]
            out[f"{case}_{name}"] = (
                col if np.iscomplexobj(col) and i in (1, 2, 3, 4) else col.real
            )
    return out


def main() -> None:
    """Write the reference ``.npz``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plane-nm", type=float, default=1500.0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    np.savez(a.out, plane_nm=a.plane_nm, **reference(a.plane_nm))


if __name__ == "__main__":
    main()
