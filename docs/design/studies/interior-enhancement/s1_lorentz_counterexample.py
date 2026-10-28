"""Why the driven interior intensity at complex λ is not the objective.

Study script — touches no package code. Plan: ``README.md`` §"Why not the
driven interior intensity".

The first plan took J = (1/A)∫_Ω |E|²/|E_inc|² under plane-wave incidence at
λ₀ + iΔλ as the Lorentzian average of J over real λ. Tested here on the
closed-form interior field of a circle, at its n = 34 p = 1 WGM:

- the true average ⟨J⟩ = ∫ L_Δλ(λ − λ₀) J(λ) dλ by quadrature over real λ;
- J(λ₀ − iΔλ), the pole-free side (QNM poles sit at Im λ > 0, §8);
- J(λ₀ + iΔλ), the side the first plan named.

Interior field, ``pol = 1``: H = Σ iⁿ dₙ Jₙ(m k r) e^{inθ} with
dₙ = (2i m/(π x)) / (m Jₙ(mx) Hₙ'(x) − Jₙ'(mx) Hₙ(x)), the TM denominator of
``reference.mie.qnm_denominator``, from continuity of H and of (1/ε)∂H/∂r and
the Wronskian JₙHₙ' − Jₙ'Hₙ = 2i/(πx). In-plane E = (i/ωε)∇×H, so
|E|²/|E_inc|² = |∇H|²/(k² m⁴) for a unit incident H.

Run: ``uv run python docs/design/studies/interior-enhancement/s1_lorentz_counterexample.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_qrad_budget import LAM0, M_REL, N_CLAD, fundamental_pole, q_of  # noqa: E402

# Orders |n| ≤ 60 and 80 radial Gauss nodes: the n = 34 WGM has m·x ≈ 39, so
# Jₙ(m k r) is negligible beyond n ≈ 60 and the radial integrand is smooth.
ORDERS = np.arange(-60, 61)
T_GL, W_GL = np.polynomial.legendre.leggauss(80)


def interior_intensity(lam: complex, rad: float) -> float:
    """Disk-averaged |E|²/|E_inc|² at (possibly complex) vacuum λ."""
    k = 2.0 * np.pi * N_CLAD / lam
    x, m, n = k * rad, M_REL, ORDERS
    d = (2j * m / (np.pi * x)) / (
        m * jv(n, m * x) * h1vp(n, x) - jvp(n, m * x) * hankel1(n, x)
    )
    r, w = 0.5 * rad * (T_GL + 1.0), 0.5 * rad * W_GL
    q = m * k
    grad2 = (
        np.abs(q * jvp(n[:, None], q * r)) ** 2
        + (n[:, None] ** 2 / r**2) * np.abs(jv(n[:, None], q * r)) ** 2
    )
    total = 2.0 * np.pi * np.sum(np.abs(d) ** 2 * (grad2 @ (w * r)))
    return total / (abs(k) ** 2 * m**4) / (np.pi * rad**2)


def main() -> None:
    """Compare the Lorentzian average with both complex-λ evaluations."""
    t0 = time.perf_counter()
    x = fundamental_pole(34)
    rad = x.real * LAM0 / (2.0 * np.pi * N_CLAD)
    lam_p = 2.0 * np.pi * N_CLAD * rad / x
    g, l0 = lam_p.imag, lam_p.real
    print(f"n=34 pole λ = {lam_p:.5f} nm, γ = Im λ = {g:.4f} nm, Q = {q_of(x):.0f}")
    for dl in (0.3 * g, g, 3.0 * g, 10.0 * g):
        # λ = λ₀ + γ tan φ puts nodes on the narrowest feature, the pole; the
        # Lorentzian mass beyond 200(γ + Δλ) is ≈ 0.3 %, far below the factors
        # being compared.
        p_max = np.arctan(200.0 * (g + dl) / g)
        p, wp = np.polynomial.legendre.leggauss(1500)
        p, wp = p * p_max, wp * p_max
        lam = l0 + g * np.tan(p)
        lor = (dl / np.pi) / ((lam - l0) ** 2 + dl**2)
        jac = g / np.cos(p) ** 2
        avg = sum(
            wp[i] * jac[i] * lor[i] * interior_intensity(lam[i], rad)
            for i in range(lam.size)
        )
        below = interior_intensity(l0 - 1j * dl, rad)
        above = interior_intensity(l0 + 1j * dl, rad)
        print(
            f"Δλ/γ={dl / g:5.1f}  ⟨J⟩={avg:9.4g}  J(λ₀−iΔλ)={below:9.4g}  "
            f"ratio={avg / below:5.2f}  J(λ₀+iΔλ)={above:10.4g}"
        )
    print(f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
