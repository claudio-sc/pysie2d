"""Convergence of the scattering efficiency toward analytic Mie theory.

Produces a semi-log plot of the relative error in ``qsca`` versus the number of
boundary quadrature points ``nn``, for both TE and TM polarisations, for a
circular cylinder (n_core = 1.5, radius 200 nm) at λ = 600 nm. Convergence is
spectral (Kress–Martensen quadrature), so the error falls as a straight line on
a linear ``nn`` axis until it reaches double-precision round-off.

Run:
    uv run python examples/convergence_study.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pysie2d import BIESolver, Geometry, Material
from pysie2d.reference import mie

RAD = 200.0
N_CORE = 1.5
N_CLAD = 1.0
WAVELENGTH = 600.0
# Spectral convergence reaches round-off by nn ≈ 30 on this circle, so the
# range is dense and short: a wider one would be a flat line at 1e-15.
N_VALUES = list(range(6, 53, 2))

# pol code → (Mie key, label, marker); marker as well as colour, so the two
# polarisations stay distinguishable without relying on colour alone.
CASES = {2: ("Q_sca_TE", "TE (E_y)", "o"), 1: ("Q_sca_TM", "TM (H_y)", "s")}


def relative_qsca_error(nn: int, pol: int) -> float:
    """Relative error of the BIE qsca versus Mie for a circle at fixed λ."""
    x = 2.0 * np.pi * N_CLAD * RAD / WAVELENGTH
    ref = mie.efficiencies(x, N_CORE / N_CLAD)[CASES[pol][0]]
    geom = Geometry.gielis(rad=RAD, n_pts=nn, m=0)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol)
    eff = BIESolver(geom, mat).scatter(wavelength=WAVELENGTH).efficiencies()
    return abs(eff["qsca"] - ref) / ref


def main() -> None:
    """Compute the errors and save the semi-log convergence figure."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for pol, (_, label, marker) in CASES.items():
        errors = [relative_qsca_error(nn, pol) for nn in N_VALUES]
        # An error of exactly zero is round-off too; clip so the log axis shows it.
        ax.semilogy(
            N_VALUES,
            np.maximum(errors, 1e-17),
            marker=marker,
            ms=5,
            lw=1.5,
            label=label,
        )
    ax.axhline(np.finfo(float).eps, color="0.45", ls="--", lw=1.0)
    ax.annotate(
        "machine ε",
        (N_VALUES[0], np.finfo(float).eps),
        textcoords="offset points",
        xytext=(0, 4),
        ha="left",
        va="bottom",
        color="0.35",
        fontsize=9,
    )

    ax.set_xlabel("boundary points $nn$")
    ax.set_ylabel(r"relative error in $Q_\mathrm{sca}$ vs Mie")
    ax.set_title(
        f"BIE convergence to Mie theory\ncircle, $n_c$={N_CORE}, "
        f"$a$={RAD:.0f} nm, $\\lambda$={WAVELENGTH:.0f} nm"
    )
    ax.grid(True, which="major", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "convergence_study.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
