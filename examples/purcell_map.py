"""Purcell map: relative LDOS around a Gielis star near a resonance.

For every source position on a 2-D grid, the relative local density of states
``1 + 4·Im[S(r_s, r_s)]`` is computed, where ``S`` is the self-Green function
(the environment's scattered field back at the emitter). It reports, in one
picture, both how strongly an emitter placed there couples to the structure and
how its spontaneous-decay rate is modified — the entry point of the downstream
quantum-dynamics story: the drive *and* the decay rate of the emitter both come
from this map.

The particle is the same Gielis ``m = 6`` star used by the plane-wave near-field
example (``n_core = 2.0``). The wavelength sits on a Mie-like ``qsca(λ)``
resonance found by scanning the efficiency spectrum; the peak was checked to
persist unchanged from ``nn = 300`` to ``nn = 600`` boundary points, so it is a
real resonance and not a low-resolution artefact. On resonance the map shows
enhancement (relative LDOS > 1) in lobes around the star and suppression (< 1)
elsewhere.

Performance: every grid point needs its own BIE solve because the right-hand
side depends on the source position, but the system matrix depends only on λ.
:func:`pysie2d.relative_ldos_map` therefore factorises the matrix once
(``scipy.linalg.lu_factor``) and reuses it across all sources — turning a sweep
that would take an hour into one that takes seconds. A deliberately coarse grid
keeps this example quick to run.

Run:
    uv run python examples/purcell_map.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from pysie2d import BIESolver, Geometry, Material, relative_ldos_map

RAD = 200.0
N_CORE = 2.0  # same particle as the plane-wave near-field example
N_CLAD = 1.0
M_SYMMETRY = 6
N1, N2, N3 = 6.0, 12.0, 12.0
WAVELENGTH = 540.0  # on a qsca resonance of this star (see module docstring)
N_PTS = 300
GRID_HALF_WIDTH = 3.0 * RAD
GRID_N = 120  # coarse for a quick run


def main() -> None:
    """Compute the relative-LDOS map on a grid and save the figure."""
    geom = Geometry.gielis(rad=RAD, n_pts=N_PTS, m=M_SYMMETRY, n1=N1, n2=N2, n3=N3)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=2)
    solver = BIESolver(geom, mat)

    axis = np.linspace(-GRID_HALF_WIDTH, GRID_HALF_WIDTH, GRID_N)
    xx, zz = np.meshgrid(axis, axis)
    ldos = relative_ldos_map(solver, WAVELENGTH, xx, zz)  # NaN inside / too close

    finite = ldos[np.isfinite(ldos)]
    vmin = float(np.nanmin(finite))
    vmax = float(np.percentile(finite, 99))
    print(f"relative LDOS ∈ [{vmin:.3g}, {np.nanmax(finite):.3g}]")
    print(f"colour scale clipped to [{vmin:.3g}, {vmax:.3g}] (min .. 99th pct)")

    # Diverging scale centred on 1 (free-space value): red = enhancement
    # (Purcell hot spots), blue = suppression. vmin=0 is the physical LDOS
    # floor, vmax the 99th percentile.
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.6")  # interior / near-boundary points shown grey
    im = ax.imshow(
        ldos,
        extent=(-GRID_HALF_WIDTH, GRID_HALF_WIDTH, -GRID_HALF_WIDTH, GRID_HALF_WIDTH),
        origin="lower",
        cmap=cmap,
        norm=norm,
    )
    fx = np.append(geom.f, geom.f[0])
    gz = np.append(geom.g, geom.g[0])
    ax.plot(fx, gz, color="black", lw=1.2)

    ax.set_xlabel("$x_s$ (nm)")
    ax.set_ylabel("$z_s$ (nm)")
    ax.set_title(
        f"Relative LDOS (Purcell map), Gielis $m$={M_SYMMETRY} star, "
        f"$n_c$={N_CORE}\n$\\lambda$={WAVELENGTH:.0f} nm (qsca resonance), TE"
    )
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="relative LDOS  $1 + 4\\,\\mathrm{Im}\\,S$")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "purcell_map.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
