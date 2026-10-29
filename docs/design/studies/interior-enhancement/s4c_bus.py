"""S4c — the bus: Q_c(g), the Q_i/Q_c split of the dip, and cladding aging.

Study script — touches no package code. Plan: ``README.md`` §S4.

Bus: a 500 nm SiN strip as an in-plane ``Stack`` (ε = n_core², 500 nm, over a
cladding half-space), ring above it at gap g. ``scatter_mode`` drives the one
guided mode forward; ``mode_amplitudes(...).s_plus[0]`` is the through
amplitude t(λ), direct term included. ``scatter_mode`` refuses complex λ, so the
loaded pole cannot be contour-located. Near resonance t is a ratio of linear
functions of λ,

    t(λ) = t_bg (λ − λ_z)/(λ − λ_p),

so a three-parameter linear least-squares fit to sampled t gives the pole and
the zero. Under the S0 convention (Im λ > 0 decays):

    1/Q_L = 2 Im λ_p / Re λ,   1/Q_i = (Im λ_p + Im λ_z)/Re λ,
    1/Q_c = (Im λ_p − Im λ_z)/Re λ,

with the zero below the axis (Im λ_z < 0) meaning overcoupled. The fit is
iterated onto the pole, and its residual over the sample is reported. Spare
points that the fit was not given test it.

- **Anchor (rule 3):** as g grows, every effect of the bus on the ring
  vanishes and the extracted Q_i must tend to the isolated ring's analytic Mie
  Q_i (S1). A wrong split cannot converge there by accident: it would put Q_c's
  g-dependence into Q_i. Measured, the approach is oscillatory and slow:
  the bus slab reflects the ring's radiation (76 % of its loss at n = 58)
  back onto it, and in 2D that return decays only slowly with g. The ladder
  runs until the dip is too shallow to fit (Q_c ≳ 1e16).
- **Critical gap** g_c where Q_c = Q_i, for the main (n = 58) and sister
  (n = 62) rings.
- **Coupling slope:** coupled-mode theory has κ ∝ the bus mode's evanescent
  tail at the ring, so d ln Q_c/dg = 2γ_bus = 2k₀√(n_eff² − n_cl²), a closed
  form from the stack's own guided mode.
- **Aging at g_c:** central differences in n_clad (cover *and* bottom
  half-space) of the extracted 1/Q_i and 1/Q_c, as gains T ≡ Q·∂(1/Q)/∂n_cl.
  T_i must match S4a's isolated T_cl if the split is clean and parasitic
  radiation is negligible. T_c is what the dip fit has to absorb. With S_i,
  the in-situ ∂(1/Q_i)/∂ε″ by central difference, S4b's two-ring σ(δε″) is
  recomputed from in-situ numbers.

Run: ``uv run python docs/design/studies/interior-enhancement/s4c_bus.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from pysie2d import BIESolver, Geometry, Material, Stack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_qrad_budget import N_CLAD, N_CORE, POL  # noqa: E402
from s1_circle_sensitivity import EPSI0, inv_q  # noqa: E402
from s4_cladding_drift import ring, sigma_eps, to_db_per_m  # noqa: E402

BUS_W = 500.0  # nm, single-mode below the 563 nm cutoff of this 2D model
N_PTS = {58: 340, 62: 360}  # S1: ≈ 5.8n reaches the λ floor
H_CL = 1.0e-4  # n_clad step: shifts λ by ≈ 6 nm·1e-3, ≈ 6 half-widths
# ε″ step ≈ 0.9 dB/m: moves 1/Q_i by S·h ≈ 2 %, far above the fit's 1e-5
# held-out residual, and 1/Q_i is linear in ε″ to O(h²).
H_EPS = 1.0e-7


def through(
    rad: float, n_pts: int, gap: float, lam: float, n_cl: float, epsi: float
) -> complex:
    """Through amplitude t of the guided mode past the ring at `gap`."""
    st = Stack(layers=((N_CORE**2, BUS_W),), eps_bot=n_cl**2, z_top=0.0)
    mat = Material(n_core=N_CORE, n_clad=n_cl, pol=POL, epsi=epsi)
    geom = Geometry.gielis(rad=rad, n_pts=n_pts, m=0, z0=rad + gap)
    res = BIESolver(geom, mat, background=st).scatter_mode(lam, 0, +1, x_ref=0.0)
    return complex(res.mode_amplitudes(0.0).s_plus[0])


def fit(lams: np.ndarray, ts: np.ndarray) -> tuple[complex, complex, complex]:
    """Least-squares t = (aλ + b)/(λ + c); returns (pole, zero, t_bg)."""
    # t·c − a·λ − b = −t·λ is linear in (c, a, b). λ is centred and scaled so
    # the normal matrix is O(1) — raw λ ≈ 1550 against widths of 1e-3 nm loses
    # every digit the pole needs.
    l0 = lams.mean()
    sc = np.ptp(lams) or 1.0
    u = (lams - l0) / sc
    a_mat = np.column_stack([ts, -u, -np.ones_like(u)])
    c, a, b = np.linalg.lstsq(a_mat, -ts * u, rcond=None)[0]
    pole = l0 - c * sc
    zero = l0 - (b / a) * sc
    return complex(pole), complex(zero), complex(a)


def locate(n: int, gap: float, n_cl: float = N_CLAD, epsi: float = EPSI0) -> dict:
    """Iterate the rational fit onto the loaded resonance of ring n at `gap`."""
    r = ring(n)
    rad, n_pts = r["rad"], N_PTS[n]
    lam_iso = r["lam"]
    # Start at the isolated pole, with a window wide enough for a strongly
    # overcoupled dip; then shrink onto the fitted pole until it stops moving.
    centre, half = lam_iso.real, 400.0 * lam_iso.imag
    pole = None
    for _ in range(8):
        lams = centre + half * np.linspace(-1.0, 1.0, 7)
        ts = np.array([through(rad, n_pts, gap, lam, n_cl, epsi) for lam in lams])
        new_pole, zero, t_bg = fit(lams, ts)
        done = pole is not None and abs(new_pole - pole) < 1e-7 * new_pole.imag
        pole = new_pole
        centre, half = pole.real, 3.0 * pole.imag
        if done:
            break
    # Held-out check: two points between the fitted ones, inside the dip.
    held = [pole.real - 0.5 * pole.imag, pole.real + 1.5 * pole.imag]
    with np.errstate(all="ignore"):
        resid = max(
            abs(
                t_bg * (lam - zero) / (lam - pole)
                - through(rad, n_pts, gap, lam, n_cl, epsi)
            )
            for lam in held
        )
    lr = pole.real
    return {
        "pole": pole,
        "zero": zero,
        "inv_l": 2.0 * pole.imag / lr,
        "inv_i": (pole.imag + zero.imag) / lr,
        "inv_c": (pole.imag - zero.imag) / lr,
        "resid": resid,
        "inv_i_iso": inv_q(lam_iso),
    }


def anchor() -> None:
    """Extracted Q_i → isolated Mie Q_i as g grows, and an n_pts check."""
    iso = ring(58)["y"]
    print("anchor, n=58: Q_i/Q_i,iso − 1 and Q_L/Q_i,iso − 1 vs gap")
    for gap in (1600.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0):
        r = locate(58, gap)
        if 1.0 / r["inv_c"] > 1e16 or not np.isfinite(r["resid"]):
            print(f"  g={gap:5.0f}: dip too shallow to fit (Q_c ≳ 1e16), ladder ends")
            break
        print(
            f"  g={gap:5.0f}: Q_i {iso / r['inv_i'] - 1:+.2e}  "
            f"Q_L {iso / r['inv_l'] - 1:+.2e}  Q_c={1 / r['inv_c']:.2e}  "
            f"held-out {r['resid']:.1e}"
        )
    base = locate(58, 1600.0)
    N_PTS[58] = 400
    fine = locate(58, 1600.0)
    N_PTS[58] = 340
    print(
        f"  n_pts 340 → 400 at g=1600: Δ(1/Q_i)/(1/Q_i) = "
        f"{abs(fine['inv_i'] - base['inv_i']) / base['inv_i']:.1e}, "
        f"Δ(1/Q_c)/(1/Q_c) = {abs(fine['inv_c'] - base['inv_c']) / base['inv_c']:.1e}"
    )


def main() -> None:
    """Anchor, then per ring the gap scan, critical gap and aging gains at g_c."""
    t0 = time.perf_counter()
    anchor()
    n_eff = (
        Stack(layers=((N_CORE**2, BUS_W),), eps_bot=N_CLAD**2)
        .guided_modes(1550.0, POL, N_CLAD)[0]
        .n_eff
    )
    two_g = 2.0 * (2.0 * np.pi / 1550.0) * np.sqrt(n_eff**2 - N_CLAD**2)
    print(f"bus n_eff = {n_eff:.6f}: 2γ_bus = {two_g * 100:.4f} per 100 nm")
    in_situ = []
    for n in (58, 62):
        print(f"\nring n={n}: isolated Q_i (Mie) = {1 / ring(n)['y']:.6g}")
        print(
            f"{'gap':>6} {'Q_L':>10} {'Q_i':>11} {'Q_c':>10} {'Q_i/Q_i,iso−1':>14} "
            f"{'regime':>6} {'held-out':>9}"
        )
        rows = []
        for gap in (800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0):
            r = locate(n, gap)
            q_i, q_c = 1 / r["inv_i"], 1 / r["inv_c"]
            rows.append((gap, q_i, q_c))
            print(
                f"{gap:6.0f} {1 / r['inv_l']:10.4g} {q_i:11.6g} {q_c:10.4g} "
                f"{r['inv_i_iso'] / r['inv_i'] - 1:14.2e} "
                f"{'over' if r['zero'].imag < 0 else 'under':>6} {r['resid']:9.1e}"
            )
        g_arr = np.array([x[0] for x in rows])
        lr = np.log(np.array([x[2] / x[1] for x in rows]))  # ln(Q_c/Q_i)
        g_c = float(np.interp(0.0, lr, g_arr))
        slope = np.polyfit(g_arr, np.log([x[2] for x in rows]), 1)[0]
        print(
            f"  d ln Q_c/dg = {slope * 100:.3f} per 100 nm; critical gap ≈ {g_c:.0f} nm"
        )

        rp, rm = locate(n, g_c, N_CLAD + H_CL), locate(n, g_c, N_CLAD - H_CL)
        r0 = locate(n, g_c)
        t_i = (rp["inv_i"] - rm["inv_i"]) / (2 * H_CL) / r0["inv_i"]
        t_c = (rp["inv_c"] - rm["inv_c"]) / (2 * H_CL) / r0["inv_c"]
        t_iso = ring(n)["T_cl"]
        ep, em = locate(n, g_c, epsi=EPSI0 + H_EPS), locate(n, g_c, epsi=EPSI0 - H_EPS)
        d_eps = (ep["inv_i"] - em["inv_i"]) / (2 * H_EPS)
        d_cl = (rp["inv_i"] - rm["inv_i"]) / (2 * H_CL)
        in_situ.append({"y": r0["inv_i"], "J": np.array([d_eps, d_cl])})
        print(
            f"  at g_c: Q_i={1 / r0['inv_i']:.6g} Q_c={1 / r0['inv_c']:.6g}  "
            f"T_i={t_i:.4g} (isolated T_cl={t_iso:.4g})  T_c={t_c:.4g}  "
            f"S_i={d_eps / r0['inv_i']:.5g} (isolated S={ring(n)['S']:.5g})"
        )
    free, known = (to_db_per_m(v) for v in sigma_eps(in_situ))
    iso_free, iso_known = (to_db_per_m(v) for v in sigma_eps([ring(58), ring(62)]))
    print(
        f"\npair 58 + 62 at their critical gaps: σ(δε″) = {free:.4f} dB/m with δn_cl "
        f"free, {known:.4f} known "
        f"(isolated rings, S4b: {iso_free:.4f}, {iso_known:.4f})"
    )
    print(f"\n{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
