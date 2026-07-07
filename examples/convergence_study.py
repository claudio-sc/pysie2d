"""Convergence of the scattering efficiency toward analytic Mie theory.

Produces a log-log plot of the relative error in ``qsca`` versus the number of
boundary quadrature points ``nn``, for both TE and TM polarisations, for a
circular cylinder (n_core = 1.5, radius 200 nm) at λ = 600 nm.

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
# Stop at 320: beyond this the BIE spatial error drops below the fixed
# angular-quadrature floor of the (verbatim) far-field integrator, so the
# qsca error stops improving. This range isolates the BIE spatial convergence.
N_VALUES = [20, 40, 80, 160, 320]

# pol code → (Mie key, label)
CASES = {2: ("Q_sca_TE", "TE (E_y)"), 1: ("Q_sca_TM", "TM (H_y)")}


def relative_qsca_error(nn: int, pol: int) -> float:
    """Relative error of the BIE qsca versus Mie for a circle at fixed λ."""
    x = 2.0 * np.pi * N_CLAD * RAD / WAVELENGTH
    ref = mie.efficiencies(x, N_CORE / N_CLAD)[CASES[pol][0]]
    geom = Geometry.gielis(rad=RAD, n_pts=nn, m=0)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol)
    eff = BIESolver(geom, mat).scatter(wavelength=WAVELENGTH).efficiencies()
    return abs(eff["qsca"] - ref) / ref


def main() -> None:
    """Compute the errors and save the log-log convergence figure."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for pol, (_, label) in CASES.items():
        errors = [relative_qsca_error(nn, pol) for nn in N_VALUES]
        ax.loglog(N_VALUES, errors, "o-", label=label)

    ax.set_xlabel("boundary points $nn$")
    ax.set_ylabel(r"relative error in $Q_\mathrm{sca}$ vs Mie")
    ax.set_title(
        f"BIE convergence to Mie theory\ncircle, $n_c$={N_CORE}, "
        f"$a$={RAD:.0f} nm, $\\lambda$={WAVELENGTH:.0f} nm"
    )
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "convergence_study.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
