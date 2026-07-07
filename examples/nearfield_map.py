"""Near-field map of a Gielis star under plane-wave illumination.

Evaluates the field on a 2-D grid around an ``m = 6`` Gielis superformula
particle. The interior is filled by passing the particle index ``ri`` to the
representation formula, so the map shows the scattered field outside the
boundary and the internal field inside it. The particle boundary is overlaid.

Colour scale: ``|field|`` from 0 to the 99th percentile of the grid values
(printed at run time). Values within a few boundary-point spacings of the
surface are near-singular in the representation formula and are shown as-is,
not masked — hence the bright rim right at the boundary.

Run:
    uv run python examples/nearfield_map.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pysie2d import BIESolver, Geometry, Material

RAD = 200.0
N_CORE = 2.0
N_CLAD = 1.0
WAVELENGTH = 600.0
M_SYMMETRY = 6
# Gielis exponents giving a clear 6-lobed star (r varies ~1.8×) that is still
# only mildly concave, so the _is_outside nearest-point test stays reliable.
N1, N2, N3 = 6.0, 12.0, 12.0
N_PTS = 400
GRID_HALF_WIDTH = 3.0 * RAD
GRID_N = 300


def main() -> None:
    """Compute the near field on a grid and save the figure."""
    geom = Geometry.gielis(rad=RAD, n_pts=N_PTS, m=M_SYMMETRY, n1=N1, n2=N2, n3=N3)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=2)
    result = BIESolver(geom, mat).scatter(wavelength=WAVELENGTH)

    axis = np.linspace(-GRID_HALF_WIDTH, GRID_HALF_WIDTH, GRID_N)
    xx, zz = np.meshgrid(axis, axis)
    field = result.eval_field(xx.ravel(), zz.ravel()).reshape(xx.shape)
    magnitude = np.abs(field)

    vmax = float(np.percentile(magnitude, 99))
    print(f"colour scale |field| ∈ [0, {vmax:.3g}] (99th percentile)")

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(
        magnitude,
        extent=(-GRID_HALF_WIDTH, GRID_HALF_WIDTH, -GRID_HALF_WIDTH, GRID_HALF_WIDTH),
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
    )
    # Overlay the closed boundary.
    fx = np.append(geom.f, geom.f[0])
    gz = np.append(geom.g, geom.g[0])
    ax.plot(fx, gz, color="white", lw=1.2, alpha=0.9)

    ax.set_xlabel("$x$ (nm)")
    ax.set_ylabel("$z$ (nm)")
    ax.set_title(
        f"Near field, Gielis $m$={M_SYMMETRY} star\n"
        f"$n_c$={N_CORE}, $\\lambda$={WAVELENGTH:.0f} nm, plane wave (TE)"
    )
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="|field|")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "nearfield_map.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
