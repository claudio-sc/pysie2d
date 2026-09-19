"""Convergence map: Q_sca error against analytic Mie over nn and size parameter.

A heatmap of the relative error in ``qsca`` for a circular cylinder
(``n_core = 1.5``, vacuum background, λ = 600 nm, uniform-θ nodes) over
``nn = 8 … 300`` and size parameter ``x = 0.1 … 30``, one panel per
polarisation. ``x`` is swept through the radius at fixed λ; scale covariance
(docs/conventions.md §9) makes that identical to sweeping λ.

One node-spacing line is traced in white, 6 points per *interior* wavelength
``λ/n_core`` — ``nn = 6·n_core·x``, in the ``R = wavelength_over_ds`` the
library reports (§12). Black contours
bracket the steep front where the error falls from 1e-3 to 1e-12, with 1e-6
drawn heavier; the script prints the ``R`` at which 1e-6 is actually reached.

Run:
    uv run python examples/convergence_map.py
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from pysie2d import BIESolver, Geometry, Material
from pysie2d.reference import mie

N_CORE = 1.5
N_CLAD = 1.0
WAVELENGTH = 600.0
# Node-spacing rule, in points per interior wavelength λ/n_core (§12's R).
R_RULE = 6.0
STEEP_LEVEL = 1e-6
# Up to 300: everything right of the convergence front is round-off, which a
# wider axis only repeats.
NN = np.arange(8, 301, 8)
X = np.geomspace(0.1, 30.0, 50)
# Double-precision floor for display: an exact zero is round-off too.
FLOOR = 1e-16
# hankel1/jv release the GIL, so threads parallelise the assembly; sized to the
# machine's performance cores.
WORKERS = 4

CASES = {2: ("Q_sca_TE", "TE ($E_y$)"), 1: ("Q_sca_TM", "TM ($H_y$)")}


def relative_error(nn: int, x: float, pol: int) -> float:
    """Relative qsca error of the BIE against Mie for one cell."""
    rad = x * WAVELENGTH / (2.0 * np.pi * N_CLAD)
    ref = mie.efficiencies(x, N_CORE / N_CLAD)[CASES[pol][0]]
    geom = Geometry.gielis(rad=rad, n_pts=int(nn), m=0)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol)
    eff = BIESolver(geom, mat).scatter(wavelength=WAVELENGTH).efficiencies()
    return abs(eff["qsca"] - ref) / ref


def error_grid(pol: int) -> np.ndarray:
    """(len(X), len(NN)) relative errors for one polarisation."""
    cells = [(nn, x) for x in X for nn in NN]
    with ThreadPoolExecutor(WORKERS) as pool:
        errs = list(pool.map(lambda c: relative_error(*c, pol), cells))
    return np.maximum(np.array(errs).reshape(X.size, NN.size), FLOOR)


def main() -> None:
    """Compute both grids and save the heatmap."""
    # The reference must not be the floor: Wiscombe truncation against 120
    # orders at the largest x.
    ref_check = max(
        abs(
            mie.efficiencies(X[-1], N_CORE)[k]
            / mie.efficiencies(X[-1], N_CORE, n_max=120)[k]
            - 1
        )
        for k, _ in CASES.values()
    )
    print(f"Mie truncation check at x = {X[-1]:.0f}: {ref_check:.1e}")

    grids = {pol: error_grid(pol) for pol in CASES}
    norm = LogNorm(vmin=FLOOR, vmax=1.0)

    # Where the 1e-6 level is crossed, in points per interior wavelength
    # R = nn / (n_core·x): the number a resolution rule would be stated in.
    for pol, (_, label) in CASES.items():
        r_cross = []
        for row, x in zip(grids[pol], X, strict=True):
            above = np.flatnonzero(row > STEEP_LEVEL)
            if above.size and above[-1] + 1 < NN.size and x >= 1.0:
                r_cross.append(NN[above[-1] + 1] / (N_CORE * x))
        print(
            f"{label}: 1e-6 first held at R = {np.median(r_cross):.1f} points per "
            f"interior wavelength (median over x ≥ 1; range "
            f"{min(r_cross):.1f}–{max(r_cross):.1f})"
        )

    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.8), sharey=True, layout="constrained"
    )
    x_line = np.geomspace(X[0], X[-1], 200)
    for ax, (pol, (_, label)) in zip(axes, CASES.items(), strict=True):
        err = grids[pol]
        # Smooth rendering, as in the Purcell map: colours interpolated between
        # the computed cells rather than drawn as blocks.
        im = ax.pcolormesh(NN, X, err, norm=norm, cmap="jet", shading="gouraud")
        # The steep front: the error falls from 1e-3 to 1e-12 across a narrow
        # band of nn, bracketed by thin lines, with 1e-6 marked in the middle.
        ax.contour(
            NN,
            X,
            err,
            levels=[1e-12, 1e-3],
            colors="black",
            linewidths=0.6,
            linestyles=":",
        )
        ax.contour(NN, X, err, levels=[STEEP_LEVEL], colors="black", linewidths=1.0)

        nn_rule = R_RULE * N_CORE * x_line
        ax.plot(nn_rule, x_line, color="white", lw=1.4)
        x_tag = 12.0
        ax.annotate(
            r"6 points per $\lambda/n_\mathrm{core}$",
            xy=(R_RULE * N_CORE * x_tag, x_tag),
            xytext=(8, -4),
            textcoords="offset points",
            color="white",
            fontsize=10,
            ha="left",
            va="top",
        )

        ax.set_yscale("log")
        ax.set_xlim(NN[0], NN[-1])
        ax.set_ylim(X[0], X[-1])
        ax.set_xlabel("boundary points $nn$")
        ax.set_title(label)
    axes[0].set_ylabel(r"size parameter $x = 2\pi a/\lambda$")
    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.92,
        pad=0.02,
        label=r"relative error in $Q_\mathrm{sca}$ vs Mie",
    )
    cbar.set_ticks([10.0**k for k in range(-16, 1, 2)])
    for lv, lw, ls in ((1e-3, 0.6, ":"), (STEEP_LEVEL, 1.0, "-"), (1e-12, 0.6, ":")):
        cbar.ax.axhline(lv, color="black", lw=lw, ls=ls)
    fig.suptitle(
        f"Convergence to Mie theory over size and resolution, circle, $n_c$={N_CORE}"
        "\nblack: error contours 1e-3 · 1e-6 · 1e-12 — white: 6 points per "
        "interior wavelength",
        fontsize=11,
    )

    out = Path(__file__).resolve().parent.parent / "figures" / "convergence_map.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
