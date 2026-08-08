"""Quasi-normal-mode spectrum of a circular cylinder, against analytic Mie poles.

Extracts QNMs with :class:`pysie2d.QNMSolver` from seven search boxes — four TE,
three TM — and plots them in the complex λ-plane over the analytic Mie poles of
the same cylinder (``n_core = 3.0``, radius 200 nm, vacuum background).

The axes are log-log because ``Q = Re λ / (2 Im λ)`` then makes each iso-Q
contour a straight line of unit slope, so the quality factor can be read off the
figure directly. Every mode plotted lies in ``Im λ > 0``, the decaying
half-plane under ``exp(-iωt)`` (docs/conventions.md §8).

The search boxes are drawn as well, because where a box is placed is the whole
skill: what matters is isolation in ``Im λ``, not width in ``Re λ``. See
docs/qnm-guide.md.

Run:
    uv run python examples/qnm_spectrum.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pysie2d import Geometry, Material, QNMSolver
from pysie2d.reference import mie

RAD = 200.0
N_CORE = 3.0
N_CLAD = 1.0
N_PTS = 200
# 6 nodes per side rather than the default 12: measured to give identical modes
# to 1e-8 nm, against a 0.38 nm discretisation error at n_pts = 200. The contour
# integral is nowhere near the accuracy bottleneck, so the extra nodes would buy
# only runtime.
N_SIDE = 6

# Size-parameter window for the analytic reference, chosen to cover the plotted
# λ range with margin: λ = 2π·n_clad·rad/x, so x ∈ [0.5, 3.2] ↔ λ ∈ [393, 2513] nm.
X_RANGE = (0.5, 3.2)

# Plot window, vacuum nm. Im λ is bounded below at 0.3 nm because the log axis
# cannot show zero and no mode in this window is narrower than that.
RE_LIM = (400.0, 1150.0)
IM_LIM = (0.3, 120.0)

# One box per mode family, each argued for isolation in Im λ. The boxes are
# deliberately *not* uniform: the low-Im (high-Q) modes need a box that is thin
# in Im λ, while the lossy ones need a tall one. pol code → 2 = TE, 1 = TM.
BOXES = [
    (2, 590.0 + 0.5j, 620.0 + 4.0j),
    (2, 520.0 + 15.0j, 545.0 + 40.0j),
    (2, 745.0 + 2.0j, 775.0 + 15.0j),
    (2, 1010.0 + 20.0j, 1060.0 + 50.0j),
    (1, 500.0 + 0.5j, 535.0 + 4.0j),
    (1, 610.0 + 4.0j, 640.0 + 14.0j),
    (1, 675.0 + 30.0j, 705.0 + 50.0j),
]

# Categorical slots 1 and 2 of the data-viz palette, in fixed order. Validated
# all-pairs on a white surface: CVD ΔE 24.7, normal-vision ΔE 33.6, contrast
# 4.42:1 and 3.20:1. Polarisation is *also* carried by marker shape, so the
# figure never encodes identity by colour alone.
CASES = {
    2: {"label": "TE ($E_y$)", "color": "#2a78d6", "marker": "o"},
    1: {"label": "TM ($H_y$)", "color": "#eb6834", "marker": "s"},
}
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#8a8a85"

ISO_Q = (10.0, 100.0, 1000.0)


def extracted_modes() -> dict[int, list[tuple[complex, int]]]:
    """Run Beyn extraction on every box, grouped by polarisation.

    Returns:
        pol code → list of ``(wavelength, multiplicity)`` for the modes found.
    """
    found: dict[int, list[tuple[complex, int]]] = {2: [], 1: []}
    geom = Geometry.gielis(rad=RAD, n_pts=N_PTS, m=0)
    for pol, z_lo, z_hi in BOXES:
        solver = QNMSolver(geom, Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol))
        res = solver.modes(z_lo, z_hi, n_quad_per_side=N_SIDE)
        # Degenerate partners come back as separate, numerically equal entries.
        # Plotting both would stack two markers on one point, so keep the first
        # of each group and carry the multiplicity into the label instead.
        seen: list[complex] = []
        for lam, mult in zip(res.wavelengths, res.multiplicity, strict=True):
            if not any(abs(lam - s) <= 1e-9 * abs(lam) for s in seen):
                seen.append(lam)
                found[pol].append((complex(lam), int(mult)))
    return found


def analytic_poles(pol: int) -> tuple[np.ndarray, np.ndarray]:
    """Analytic Mie QNM orders and vacuum wavelengths inside the plot window."""
    orders, lams, _ = mie.qnm_wavelengths(
        rad=RAD, m=N_CORE / N_CLAD, pol=pol, x_range=X_RANGE, n_clad=N_CLAD
    )
    inside = (
        (lams.real >= RE_LIM[0])
        & (lams.real <= RE_LIM[1])
        & (lams.imag >= IM_LIM[0])
        & (lams.imag <= IM_LIM[1])
    )
    return orders[inside], lams[inside]


def main() -> None:
    """Extract the modes and save the spectrum figure."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Iso-Q guide lines first, so every data mark sits above them.
    re_line = np.array(RE_LIM)
    for q in ISO_Q:
        ax.plot(re_line, re_line / (2.0 * q), ls=":", lw=1.0, color=INK_MUTED, zorder=1)
        ax.annotate(
            f"$Q={q:.0f}$",
            xy=(RE_LIM[1], RE_LIM[1] / (2.0 * q)),
            xytext=(-4, 3),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=INK_MUTED,
        )

    for pol, z_lo, z_hi in BOXES:
        ax.add_patch(
            Rectangle(
                (z_lo.real, z_lo.imag),
                z_hi.real - z_lo.real,
                z_hi.imag - z_lo.imag,
                fill=False,
                ls="-",
                lw=1.0,
                edgecolor=CASES[pol]["color"],
                alpha=0.35,
                zorder=2,
            )
        )

    found = extracted_modes()
    for pol, case in CASES.items():
        orders, lams = analytic_poles(pol)
        ax.scatter(
            lams.real,
            lams.imag,
            s=110,
            facecolors="none",
            edgecolors=case["color"],
            linewidths=2.0,
            marker=case["marker"],
            alpha=0.75,
            zorder=3,
            label=f"{case['label']} — analytic Mie pole",
        )

        pts = found[pol]
        ax.scatter(
            [lam.real for lam, _ in pts],
            [lam.imag for lam, _ in pts],
            s=34,
            color=case["color"],
            marker=case["marker"],
            zorder=4,
            label=f"{case['label']} — Beyn extraction",
        )

        # Direct-label the extracted modes with their azimuthal order, taken
        # from the nearest analytic pole. Selective by construction: only the
        # seven boxed modes are labelled, not all twenty-odd poles on the plot.
        for lam, mult in pts:
            n = int(orders[np.argmin(np.abs(lams - lam))])
            tag = f"$n={n}$" + (r"$\,(\times 2)$" if mult > 1 else "")
            ax.annotate(
                tag,
                xy=(lam.real, lam.imag),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK_PRIMARY,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*RE_LIM)
    ax.set_ylim(*IM_LIM)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ (nm)")
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ (nm)")
    ax.set_title(
        "Quasi-normal modes of a circular cylinder\n"
        f"$n_c$={N_CORE}, $a$={RAD:.0f} nm, $n_\\mathrm{{clad}}$={N_CLAD:.0f}, "
        f"$nn$={N_PTS} — open marks analytic, filled extracted"
    )
    # Ticks placed explicitly and labelled as plain nm: the reader is looking up
    # wavelengths, not decades, and the default log locator puts a single "1000"
    # on an x-axis that spans well under one decade.
    ax.set_xticks([400, 500, 600, 700, 800, 900, 1000, 1150])
    ax.set_yticks([0.3, 0.5, 1, 2, 5, 10, 20, 50, 100])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which="major", ls=":", alpha=0.4)
    # Upper left: the only corner with neither data nor an iso-Q line in it.
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "qnm_spectrum.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
