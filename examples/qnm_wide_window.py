"""Eleven quasi-normal modes from a single wide contour.

The companion example, ``qnm_spectrum.py``, places one small box per mode — the
careful way, and the way to learn where the modes are. This one does the
opposite: **one** rectangle spanning 600 nm of ``Re λ`` and the whole low-loss
band in ``Im λ``, and a single call that returns every TE mode inside it.

That is the property worth showing. Beyn's method costs
``4·n_quad_per_side`` assemblies *whatever is inside the contour*, so finding
eleven modes costs no more than finding one; only the probe count has to exceed
the mode count. The seven-box example does 7 × 4 × 6 = 168 assemblies to find
nine modes, and this one does 4 × 32 = 128 to find eleven.

The two agree. Every mode here reproduces the corresponding narrow-box value to
~1e-12 nm, except the one the wide contour under-resolves — and ``refine()``
recovers that one to ~1e-13 nm. See the printed table.

Run:
    uv run python examples/qnm_wide_window.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pysie2d import Geometry, Material, QNMSolver
from pysie2d.qnm import DEGENERATE_COND
from pysie2d.reference import mie

RAD = 200.0
N_CORE = 3.0
N_CLAD = 1.0
N_PTS = 200
POL = 2  # TE (E_y); one polarisation keeps the box a single spectrum

# The whole point of the example: one rectangle, not seven. The Im floor is 1 nm
# rather than 0 both because Im λ > 0 is asserted and because it excludes the
# Q = 598 mode at 505.68+0.42j, whose leaked rank is visible in `rank` below.
Z_LO, Z_HI = 500.0 + 1.0j, 1100.0 + 50.0j

# 32 nodes per side against the default 12. A wide contour is the one regime
# where the quadrature, not n_pts, sets the accuracy: the contour is long, and
# the poles just outside it are close relative to its size. Measured on the TE
# n=0 mode, the worst-resolved of the eleven, error against the narrow-box
# value falls 6.86 → 0.93 → 0.12 nm at 16 → 24 → 32 nodes per side, while the
# other ten sit at ~1e-12 nm from 16 upward. refine() removes even that
# residual, which is why a coarser contour plus refinement is also a valid
# strategy here.
N_SIDE = 32
# Must exceed the mode count *with multiplicity* (11), counting rank leaked from
# poles just outside the box (3 more). 20 clears both with margin.
N_PROBE = 20

# Size-parameter window for the analytic reference, wide enough to include the
# poles just outside the box — they are what `rank` sees.
X_RANGE = (0.4, 3.5)

RE_LIM = (420.0, 1150.0)
IM_LIM = (0.3, 130.0)

TE_COLOR = "#2a78d6"  # categorical slot 1; single series, so no second hue
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#8a8a85"

ISO_Q = (10.0, 100.0, 1000.0)


def analytic_poles() -> tuple[np.ndarray, np.ndarray]:
    """Analytic Mie TE orders and vacuum wavelengths across the plot window."""
    orders, lams, _ = mie.qnm_wavelengths(
        rad=RAD, m=N_CORE / N_CLAD, pol=POL, x_range=X_RANGE, n_clad=N_CLAD
    )
    keep = (
        (lams.real >= RE_LIM[0])
        & (lams.real <= RE_LIM[1])
        & (lams.imag >= IM_LIM[0])
        & (lams.imag <= IM_LIM[1])
    )
    return orders[keep], lams[keep]


def inside_box(lams: np.ndarray) -> np.ndarray:
    """Boolean mask of which wavelengths lie inside the search rectangle."""
    return (
        (lams.real > Z_LO.real)
        & (lams.real < Z_HI.real)
        & (lams.imag > Z_LO.imag)
        & (lams.imag < Z_HI.imag)
    )


def report(res, refined, orders: np.ndarray, lams: np.ndarray) -> None:
    """Print the mode table with its diagnostics and the analytic comparison."""
    print(
        f"one contour, {4 * N_SIDE} assemblies → {res.n_modes} modes (rank {res.rank})"
    )
    print(f"cancellation {res.cancellation:.1e}   max_gap {res.max_gap:.1e}\n")
    print(
        f"{'n':>3} {'lambda (vacuum nm)':>26} {'Q':>8} {'edge':>7} "
        f"{'sigma':>9} {'err_Mie':>9} {'moved':>9}"
    )
    for k in np.argsort(res.wavelengths.real):
        lam = res.wavelengths[k]
        j = int(np.argmin(np.abs(lams - lam)))
        tag = "deg" if refined.cond_jacobian[k] > DEGENERATE_COND else "   "
        print(
            f"{orders[j]:>3} {lam:>26.6f} {res.quality_factors[k]:>8.1f} "
            f"{res.edge_margin[k]:>7.3f} {res.sigma_ratio[k]:>9.1e} "
            f"{abs(lam - lams[j]):>9.4f} "
            f"{abs(refined.wavelengths[k] - lam):>9.2e} {tag}"
        )
    print(
        "\nerr_Mie  distance to the analytic Mie pole, nm. At n_pts = "
        f"{N_PTS} the discretisation\n"
        "         error alone is ~0.4 nm, so that is the floor, not the "
        "target — it is a\n"
        "         statement about n_pts, not about the contour.\n"
        "sigma    the universal resolution flag, and the one to read first. Ten "
        "of the\n"
        "         eleven sit at 1e-13 or below — on the pole to machine "
        "precision. The\n"
        "         n=0 mode at 1.1e-04 is nine orders worse, and it is the one "
        "the wide\n"
        "         contour under-resolves.\n"
        "moved    how far refine() displaced the mode, nm. Exactly 0 on the "
        "degenerate\n"
        "         rows *by construction* — refine() skips them — so it measures "
        "the\n"
        "         contour only on the simple n=0 mode, where it recovers "
        "0.118 nm.\n"
        "'deg'    the doubly degenerate n >= 1 pairs, which refine() leaves "
        "alone by design.\n\n"
        "Two readings worth having:\n"
        "  - n=0's err_Mie is *smaller* before refinement (0.365 vs 0.447) "
        "and that is not\n"
        "    a win: the raw value's quadrature error happens to cancel part of "
        "the\n"
        "    discretisation error. The refined value is the true singularity "
        "of the\n"
        "    discretised operator. Judge the contour by 'moved', the physics "
        "by n_pts.\n"
        "  - n=4 shows edge = 0.014, which normally warns of a clipped pole — "
        "it sits\n"
        "    0.67 nm above the box floor. Here sigma = 3e-15, and the mode "
        "agrees with\n"
        "    its narrow-box value to 1e-12 nm, so it is genuinely resolved. A "
        "low\n"
        "    edge_margin is a reason to check, not a verdict."
    )


def main() -> None:
    """Extract every TE mode in one box and save the figure."""
    geom = Geometry.gielis(rad=RAD, n_pts=N_PTS, m=0)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=POL)
    res = QNMSolver(geom, mat).modes(
        Z_LO, Z_HI, n_quad_per_side=N_SIDE, n_probe=N_PROBE
    )
    refined = res.refine()

    orders, lams = analytic_poles()
    report(res, refined, orders, lams)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

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

    ax.add_patch(
        Rectangle(
            (Z_LO.real, Z_LO.imag),
            Z_HI.real - Z_LO.real,
            Z_HI.imag - Z_LO.imag,
            facecolor=TE_COLOR,
            alpha=0.07,
            edgecolor=TE_COLOR,
            lw=1.6,
            zorder=2,
            label="search contour (one box)",
        )
    )

    within = inside_box(lams)
    ax.scatter(
        lams[~within].real,
        lams[~within].imag,
        s=110,
        facecolors="none",
        edgecolors=INK_MUTED,
        linewidths=2.0,
        marker="o",
        zorder=3,
        label="analytic Mie pole, outside",
    )
    ax.scatter(
        lams[within].real,
        lams[within].imag,
        s=110,
        facecolors="none",
        edgecolors=TE_COLOR,
        linewidths=2.0,
        marker="o",
        zorder=3,
        label="analytic Mie pole, inside",
    )

    # Degenerate partners are numerically equal, so plotting both would stack
    # two markers on one point; keep one per group and show the pair in the
    # label instead.
    seen: list[complex] = []
    for lam, mult in zip(res.wavelengths, res.multiplicity, strict=True):
        if any(abs(lam - s) <= 1e-9 * abs(lam) for s in seen):
            continue
        seen.append(complex(lam))
        n = int(orders[np.argmin(np.abs(lams - lam))])
        ax.annotate(
            f"$n={n}$" + (r"$\,(\times 2)$" if mult > 1 else ""),
            xy=(lam.real, lam.imag),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_PRIMARY,
        )
    ax.scatter(
        [lam.real for lam in seen],
        [lam.imag for lam in seen],
        s=34,
        color=TE_COLOR,
        marker="o",
        zorder=4,
        label="extracted in one call",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*RE_LIM)
    ax.set_ylim(*IM_LIM)
    ax.set_xticks([450, 500, 600, 700, 800, 900, 1000, 1150])
    ax.set_yticks([0.3, 0.5, 1, 2, 5, 10, 20, 50, 100])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ (nm)")
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ (nm)")
    ax.set_title(
        f"One contour, {res.n_modes} TE modes\n"
        f"$n_c$={N_CORE}, $a$={RAD:.0f} nm, $nn$={N_PTS} — "
        f"{4 * N_SIDE} assemblies, rank {res.rank}"
    )
    ax.grid(True, which="major", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "qnm_wide_window.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
