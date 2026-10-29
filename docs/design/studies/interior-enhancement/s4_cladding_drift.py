"""S4a–b — cladding index aging: main-ring tolerance and sister-ring choice.

Study script — touches no package code. Plan: ``README.md`` §S4.

Readout: the through-port dip gives Q_i of each ring. Nuisance: a common drift
δn_cl of the SiO₂ cladding. Signal: a common change δε″ of the SiN film. Every
ring is a circle on the p = 1 ladder at the 0.1 dB/cm baseline (S1).

- **S4a.** T_cl = Q_i·∂(1/Q_i)/∂n_clad of the main ring (n = 58, D ≈ 16 µm):
  BIE (``QNMResult.sensitivity`` in ``n_clad``) against the analytic
  derivative. n_clad enters twice, through m = n_c/n_cl and through
  λ = 2π·n_cl·rad/x, so dλ/dn_cl = λ/n_cl + (λ/x)·(dx/dm)·(m/n_cl). Output: the
  δn_cl whose leak equals a 0.1 dB/m step, δn_cl* = δε″_step·S/|T_cl|.
- **S4b.** Main + one sister of order n_ref. Channels y_k = 1/Q_i,k with
  σ(y_k) = 2(σ_γ/γ)/Q_i,k, the single-ring rule δε″_min = 2σ/S of README
  §Goal, at σ_γ/γ = 1e-3. Fisher F = Σ_k J_kᵀJ_k/σ_k² over (δε″, δn_cl);
  σ(δε″) = √[F⁻¹]₁₁ with δn_cl free, against the same pair with δn_cl known
  (1/√F₁₁). The ratio is the price of the aging alone; comparing with one
  ring would credit the sister for being a second absorption sensor. Sisters
  are kept under the Q_L < 1e6 ceiling, Q_i ≤ 2e6 at critical coupling.

Run: ``uv run python docs/design/studies/interior-enhancement/s4_cladding_drift.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from pysie2d import Geometry, Material, QNMSolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_qrad_budget import (  # noqa: E402
    LAM0,
    N_CLAD,
    N_CORE,
    POL,
    fundamental_pole,
    kappa,
)
from s1_circle_sensitivity import (  # noqa: E402
    EPSI0,
    dx_dm,
    gain,
    inv_q,
    m_of,
    mie_pole,
)

N_MAIN, N_PTS_MAIN = 58, 340  # S1: n_pts = 340 is at the floor for n = 58
SIGMA = 1.0e-3  # linewidth repeatability σ_γ/γ (README S0 working point)
EPS_STEP = 2.0 * N_CORE * kappa(1.0e-3)  # δε″ of 0.1 dB/m (1e-3 dB/cm)


def d_inv_q(lam: complex, dlam: complex) -> float:
    """∂(1/Q)/∂p from λ and dλ/dp, with 1/Q = 2 Im λ / Re λ."""
    return 2.0 * (dlam.imag * lam.real - lam.imag * dlam.real) / lam.real**2


def ring(n: int) -> dict:
    """Analytic λ, dλ/dε″, dλ/dn_cl, and 1/Q_i of the order-n ring at 1550 nm."""
    m0 = m_of(N_CORE, EPSI0)
    x = mie_pole(n, m0, fundamental_pole(n))
    rad = LAM0 / (2.0 * np.pi * N_CLAD * (1.0 / x).real)
    lam = 2.0 * np.pi * N_CLAD * rad / x
    dxm = dx_dm(n, x, m0)
    nc = m0 * N_CLAD
    dl_eps = -(lam / x) * dxm * 1j / (2.0 * nc * N_CLAD)
    dl_cl = lam / N_CLAD + (lam / x) * dxm * m0 / N_CLAD
    return {
        "n": n,
        "rad": rad,
        "lam": lam,
        "dl_eps": dl_eps,
        "dl_cl": dl_cl,
        "y": inv_q(lam),
        "J": np.array([d_inv_q(lam, dl_eps), d_inv_q(lam, dl_cl)]),
        "S": gain(lam, dl_eps),
        "T_cl": gain(lam, dl_cl),
    }


def bie_check(r: dict, n_pts: int) -> tuple[float, float]:
    """Relative error of the BIE dλ/dn_cl and of T_cl, worst partner."""
    lam, rad = r["lam"], r["rad"]
    geom = Geometry.gielis(rad=rad, n_pts=n_pts, m=0)

    def mat(d: float = 0.0) -> Material:
        return Material(n_core=N_CORE, n_clad=N_CLAD + d, pol=POL, epsi=EPSI0)

    h = 0.6 * lam.imag
    res = QNMSolver(geom, mat()).modes(
        complex(lam.real - h, 0.4 * lam.imag), complex(lam.real + h, 1.6 * lam.imag)
    )
    # Step 1e-5 in n_clad: S1 measured the O(h²) truncation of this step at
    # 1e-8 on dλ/dn_core, the same kind of parameter.
    d_cl = res.sensitivity(lambda d: (geom, mat(d)), step=1.0e-5)
    t_bie = np.array([gain(w, d) for w, d in zip(res.wavelengths, d_cl, strict=True)])
    e_d = float(np.max(np.abs(d_cl - r["dl_cl"])) / abs(r["dl_cl"]))
    e_t = float(np.max(np.abs(t_bie - r["T_cl"])) / abs(r["T_cl"]))
    return e_d, e_t


def sigma_eps(rings: list[dict]) -> tuple[float, float]:
    """σ(δε″) from linewidth channels: δn_cl free (√[F⁻¹]₁₁), and known."""
    f = np.zeros((2, 2))
    for r in rings:
        s_y = 2.0 * SIGMA * r["y"]  # σ(1/Q_i) = 2σ/Q_i
        f += np.outer(r["J"], r["J"]) / s_y**2
    return float(np.sqrt(np.linalg.inv(f)[0, 0])), float(1.0 / np.sqrt(f[0, 0]))


def to_db_per_m(eps: float) -> float:
    """Express a δε″ as the dB/m change it corresponds to (linear)."""
    return 0.1 * eps / EPS_STEP


def main() -> None:
    """S4a tolerance and BIE check, then the S4b sister-ring scan."""
    t0 = time.perf_counter()
    main_r = ring(N_MAIN)
    tol = EPS_STEP * main_r["S"] / abs(main_r["T_cl"])
    print(
        f"S4a  main n={N_MAIN} D={2 * main_r['rad'] / 1e3:.2f} µm: "
        f"S={main_r['S']:.5g}  T_cl={main_r['T_cl']:.5g}  "
        f"dλ/dn_cl={main_r['dl_cl']:.6g} nm"
    )
    print(f"S4a  δn_cl mimicking 0.1 dB/m: {tol:.2e}")
    for n_pts in (N_PTS_MAIN, N_PTS_MAIN + 40):
        e_d, e_t = bie_check(main_r, n_pts)
        print(f"S4a  BIE n_pts={n_pts}: dλ/dn_cl rel err {e_d:.1e}, T_cl {e_t:.1e}")

    single = 2.0 * SIGMA / main_r["S"]
    print(
        f"\nS4b  single ring, δn_cl known: σ(δε″) = {single:.3e} "
        f"= {to_db_per_m(single):.3f} dB/m (σ_γ/γ = {SIGMA:g})"
    )
    print(
        f"{'n_ref':>6} {'D_ref[µm]':>10} {'Q_i,ref':>10} {'T_cl/S':>10} "
        f"{'σ free':>8} {'σ known':>8} {'price':>6}   [dB/m]"
    )
    best = None
    for n_ref in range(20, 64, 2):
        r = ring(n_ref)
        if n_ref == N_MAIN or 1.0 / r["y"] > 2.0e6:
            continue
        free, known = (to_db_per_m(v) for v in sigma_eps([main_r, r]))
        print(
            f"{n_ref:6d} {2 * r['rad'] / 1e3:10.2f} {1 / r['y']:10.3e} "
            f"{r['T_cl'] / r['S']:10.3e} {free:8.4f} {known:8.4f} {free / known:6.2f}"
        )
        if best is None or free < best[1]:
            best = (n_ref, free, known, 2 * r["rad"] / 1e3)
    print(
        f"S4b  main T_cl/S = {main_r['T_cl'] / main_r['S']:.3e}. Best sister "
        f"n_ref={best[0]} (D={best[3]:.2f} µm): σ = {best[1]:.4f} dB/m with δn_cl "
        f"free, {best[2]:.4f} known"
    )
    print(f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
