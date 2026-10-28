"""S0 — is the Q budget feasible? The circle's WGM ladder as a loss tracker.

Study script — touches no package code. Plan: ``README.md`` §S0.

Setup (README, fixed): core n = 2.0 in SiO₂ n_clad = 1.45, ``pol = 1``
(``H_y`` out of plane, E in-plane), λ_vac = 1550 nm, baseline loss
0.1–10 dB/cm, tracking resolution 0.1 dB/m on one device. Stop rule: the D
range must hold a mode resolving 0.1 dB/m at σ_γ/γ ≥ 1e-3 with ≥ 1 K thermal
tolerance, within Q_L < 1e6.

What is measured
----------------
- **Q_rad** of the fundamental radial family (p = 1, the highest-Q mode near
  any given Re x) from the analytic Mie poles, ``reference.mie``. Poles depend
  only on ``x``, so each order n resonates at 1550 nm for exactly one diameter,
  ``D = Re x · λ₀/(π·n_clad)`` — the ladder is read as Q_rad(D).
- **Q_abs per mode**, not from a fill-factor guess: the core is given the bulk
  extinction κ = α·λ₀/(4π) of a quoted dB/cm, the pole is re-polished from the
  lossless one, and ``1/Q_abs = 1/Q_lossy − 1/Q_rad``. That carries the mode's
  own energy fraction in the core, Γ = Q_mat/Q_abs with Q_mat = Re ε/Im ε.
  First order in κ: checked by computing at 10 dB/cm and 100 dB/cm and comparing
  the scaled values.
- **Tracking limit**, at a 0.1 dB/cm baseline and critical coupling
  (Q_L = Q_i/2): the linewidth repeatability σ_γ/γ = Q_L·δ(1/Q_i) that a
  0.1 dB/m step needs, and the temperature excursion whose thermo-optic shift
  of Q_i (core only, dn/dT = 2.5e-5 /K, an assumed literature value) equals
  that step. Both from re-polished analytic poles.
- **Check (README S0):** ``QNMSolver`` on circles sized so a p = 1 mode sits
  at 1550 nm, against the same analytic poles. The n ≥ 1 circle modes are
  doubly degenerate and are not Newton-refined (``QNMResult.refine``), so the
  agreement is the contour floor.

Run: ``uv run python docs/design/studies/interior-enhancement/s0_qrad_budget.py``
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import newton

from pysie2d import Geometry, Material, QNMSolver
from pysie2d.reference.mie import qnm_denominator, qnm_size_parameters

N_CORE, N_CLAD, POL, LAM0 = 2.0, 1.45, 1, 1550.0
M_REL = N_CORE / N_CLAD
DB_PER_CM_TO_NP_PER_NM = 1.0 / (10.0 * np.log10(np.e)) * 1.0e-7
DN_DT = 2.5e-5  # SiN thermo-optic coefficient per K (assumed literature value)


def kappa(db_per_cm: float) -> float:
    """Bulk extinction coefficient of the core for a quoted intensity loss."""
    return db_per_cm * DB_PER_CM_TO_NP_PER_NM * LAM0 / (4.0 * np.pi)


def q_of(x: complex) -> float:
    """Q = Re x / (2|Im x|); Im x < 0 is the decaying side (conventions §8)."""
    return x.real / (2.0 * -x.imag)


def fundamental_pole(n: int) -> complex:
    """Lowest-Re x root of D_n: the p = 1 WGM of order n.

    A p = 1 mode needs m·x > n to oscillate inside, so the box starts just
    below n/m; 6 in Re x is wider than the p = 1 → p = 2 spacing at every
    order used here, so the lowest root in it is p = 1, not a skipped one.
    """
    # 1e-14 floor: the seeder cannot see a pole below its lowest row, and the
    # largest Q reached here (≈1e9 at x ≈ 70) needs |Im x| ≈ 4e-8.
    roots = qnm_size_parameters(
        n, M_REL, POL, (0.95 * n / M_REL, n / M_REL + 6.0), im_x_floor=1e-14
    )
    return roots[0]


def lossy_pole(n: int, x0: complex, db_per_cm: float, dn: float = 0.0) -> complex:
    """Polish the order-n pole with an absorbing core, from a nearby pole."""
    m = (N_CORE + dn + 1j * kappa(db_per_cm)) / N_CLAD
    with np.errstate(all="ignore"):
        return complex(
            newton(lambda z: qnm_denominator(n, z, m, POL), x0, tol=1e-14, maxiter=80)
        )


def ladder(orders: range) -> list[dict]:
    """Q_rad and Q_abs at 0.1 dB/cm for the p = 1 mode of each order."""
    q_mat_01 = N_CORE / (2.0 * kappa(0.1))  # Re ε / Im ε of the bulk core
    rows = []
    for n in orders:
        x = fundamental_pole(n)
        q_rad = q_of(x)
        inv_abs = {}
        for db in (10.0, 100.0):
            inv_abs[db] = 1.0 / q_of(lossy_pole(n, x, db)) - 1.0 / q_rad
        # First-order check: 1/Q_abs ∝ κ, so the 100 dB/cm value scaled down
        # tenfold must reproduce the 10 dB/cm one.
        lin = abs(inv_abs[100.0] / 10.0 - inv_abs[10.0]) / inv_abs[10.0]
        q_abs_01 = 1.0 / (inv_abs[10.0] / 100.0)
        rows.append(
            {
                "n": n,
                "x": x,
                "D_um": x.real * LAM0 / (np.pi * N_CLAD) / 1e3,
                "q_rad": q_rad,
                "q_abs_01": q_abs_01,
                "gamma": q_mat_01 / q_abs_01,
                "lin": lin,
            }
        )
    return rows


def tracking(n: int, base_db: float = 0.1, step_db: float = 1.0e-3) -> dict:
    """Linewidth repeatability and ΔT tolerated for a `step_db` dB/cm change."""
    x0 = lossy_pole(n, fundamental_pole(n), base_db)
    inv_i = 1.0 / q_of(x0)
    d_sig = 1.0 / q_of(lossy_pole(n, x0, base_db + step_db)) - inv_i
    d_one_k = abs(1.0 / q_of(lossy_pole(n, x0, base_db, DN_DT)) - inv_i)
    q_l = 0.5 / inv_i  # critical coupling, Q_c = Q_i
    return {"q_l": q_l, "sigma": d_sig * q_l, "dT_mK": 1e3 * d_sig / d_one_k}


def check_solver(n: int, n_pts_list: tuple[int, ...]) -> None:
    """QNMSolver on a circle placing the order-n p = 1 mode at 1550 nm."""
    x = fundamental_pole(n)
    rad = x.real * LAM0 / (2.0 * np.pi * N_CLAD)
    lam = 2.0 * np.pi * N_CLAD * rad / x  # vacuum λ of the analytic pole
    # Box roughly square on the scale Im λ: Beyn's 12 Gauss nodes per side
    # resolve a pole only at a distance comparable to the side length, and a
    # sliver 1 nm wide around a 70 nm-tall pole finds nothing (tried). At
    # these Q, ±0.6·Im λ stays well inside one FSR ≈ λ/n, so the box holds the
    # ±n pair alone.
    h = 0.6 * lam.imag
    z_lo = complex(lam.real - h, 0.4 * lam.imag)
    z_hi = complex(lam.real + h, 1.6 * lam.imag)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=POL)
    for n_pts in n_pts_list:
        t0 = time.perf_counter()
        res = QNMSolver(Geometry.gielis(rad=rad, n_pts=n_pts, m=0), mat).modes(
            z_lo, z_hi
        )
        dt = time.perf_counter() - t0
        errs = [abs(w - lam) for w in res.wavelengths] or [np.nan]
        print(
            f"  n={n:3d} rad={rad:8.2f} nm  n_pts={n_pts:4d}  modes={res.n_modes}  "
            f"λ_Mie={lam.real:.6f}+{lam.imag:.3e}j  "
            f"max|Δλ|={max(errs):.2e} nm  "
            f"Q_BIE={min(res.quality_factors, default=np.nan):.6g}  "
            f"Q_Mie={q_of(x):.6g}  "
            f"({dt:.1f} s)"
        )


def main() -> None:
    """Print the ladder, the D needed per loss level, and the solver check."""
    t0 = time.perf_counter()
    rows = ladder(range(4, 101, 2))
    print(f"pol={POL}, n={N_CORE}/{N_CLAD}, λ0={LAM0} nm; p = 1 WGM ladder")
    print(
        f"Q_mat(0.1 dB/cm) = Re ε/Im ε = {N_CORE / (2 * kappa(0.1)):.3e}; "
        f"10 dB/cm: {N_CORE / (2 * kappa(10.0)):.3e}"
    )
    print(
        f"{'n':>4} {'Re x':>8} {'D[µm]':>7} {'Q_rad':>10} {'Q_abs(0.1)':>11} "
        f"{'Γ':>6} {'lin':>8}"
    )
    for r in rows:
        print(
            f"{r['n']:4d} {r['x'].real:8.3f} {r['D_um']:7.2f} {r['q_rad']:10.3e} "
            f"{r['q_abs_01']:11.3e} {r['gamma']:6.3f} {r['lin']:8.1e}"
        )
    print(f"ladder: {time.perf_counter() - t0:.1f} s")

    print("\nTracking 0.1 dB/m on a 0.1 dB/cm baseline, critical coupling:")
    print(f"{'n':>4} {'D[µm]':>7} {'Q_L':>10} {'σγ/γ needed':>12} {'ΔT [mK]':>10}")
    for r in rows:
        if 30 <= r["n"] <= 70:
            t = tracking(r["n"])
            print(
                f"{r['n']:4d} {r['D_um']:7.2f} {t['q_l']:10.3e} "
                f"{t['sigma']:12.2e} {t['dT_mK']:10.3g}"
            )

    print("\nCheck: QNMSolver vs analytic Mie pole (circle, contour estimate)")
    check_solver(10, (40, 60, 80))
    check_solver(36, (160, 200, 240))


if __name__ == "__main__":
    main()
