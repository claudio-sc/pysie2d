"""S1 — the tracking objective on the circle: S and T, BIE vs analytic Mie.

Study script — touches no package code. Plan: ``README.md`` §S1.

Objective (README §Goal), at the 0.1 dB/cm baseline ε″₀ = ``Material.epsi``:

    S = Q_i · ∂(1/Q_i)/∂ε″      (absorption signal per unit Im ε)
    T = Q_i · ∂(1/Q_i)/∂n_core  (drift gain of the real index)

with 1/Q = 2 Im λ / Re λ (conventions §8), so each needs λ and one dλ/dp.

- **BIE:** ``QNMSolver.modes`` for λ, ``QNMResult.sensitivity`` for dλ/dε″ and
  dλ/dn_core — the adjoint quotient with a central difference of M in the
  parameter (conventions §11). The circle's ±n pair is degenerate and goes
  through the secular branch; a material change keeps the circle a circle, so
  both partners must get the same derivative.
- **Analytic (rule 3):** the pole x of D_n(x, m) = m Jₙ(mx)Hₙ'(x) − Jₙ'(mx)Hₙ(x)
  (``reference.mie.qnm_denominator``, pol = 1) and its implicit derivative
  dx/dm = −(∂D/∂m)/(∂D/∂x), with
  m = √(n_core² + iε″)/n_clad, matching ``Material.eps``. λ = 2π·n_clad·rad/x,
  so dλ/dp = −(λ/x)·(dx/dm)·(dm/dp). The implicit derivative is checked
  against a central difference of the polished root, so a slip in ∂D is caught
  before it is used as the reference.

The radius puts the baseline pole at Re λ = 1550 nm. S and T do not depend on
it (§9 — M sees only k·rad), only on the mode.

Run: ``uv run python docs/design/studies/interior-enhancement/s1_circle_sensitivity.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import newton
from scipy.special import h1vp, hankel1, jv, jvp

from pysie2d import Geometry, Material, QNMSolver
from pysie2d.reference.mie import qnm_denominator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_qrad_budget import (  # noqa: E402
    LAM0,
    N_CLAD,
    N_CORE,
    POL,
    fundamental_pole,
    kappa,
)

EPSI0 = 2.0 * N_CORE * kappa(0.1)  # Im ε of the 0.1 dB/cm baseline


def m_of(n_core: float, epsi: float) -> complex:
    """Relative index √(n_core² + iε″)/n_clad, as ``Material`` builds ε."""
    return np.sqrt(n_core**2 + 1j * epsi) / N_CLAD


def mie_pole(n: int, m: complex, x0: complex) -> complex:
    """Polish the order-n pole of D_n at relative index m, from x0."""
    with np.errstate(all="ignore"):
        return complex(
            newton(lambda z: qnm_denominator(n, z, m, POL), x0, tol=1e-15, maxiter=80)
        )


def dx_dm(n: int, x: complex, m: complex) -> complex:
    """Implicit derivative of the pole, −(∂D/∂m)/(∂D/∂x), TM denominator."""
    j, jp, jpp = jv(n, m * x), jvp(n, m * x), jvp(n, m * x, 2)
    h, hp, hpp = hankel1(n, x), h1vp(n, x), h1vp(n, x, 2)
    d_x = m**2 * jp * hp + m * j * hpp - m * jpp * h - jp * hp
    d_m = j * hp + m * x * jp * hp - x * jpp * h
    return -d_m / d_x


def inv_q(lam: complex) -> float:
    """1/Q = 2 Im λ / Re λ."""
    return 2.0 * lam.imag / lam.real


def gain(lam: complex, dlam: complex) -> float:
    """Q · ∂(1/Q)/∂p from λ and dλ/dp."""
    d_inv = 2.0 * (dlam.imag * lam.real - lam.imag * dlam.real) / lam.real**2
    return d_inv / inv_q(lam)


def analytic(n: int) -> dict:
    """Baseline pole, radius, dλ/dε″, dλ/dn_core, S and T from Mie."""
    m0 = m_of(N_CORE, EPSI0)
    x = mie_pole(n, m0, fundamental_pole(n))
    rad = LAM0 / (2.0 * np.pi * N_CLAD * (1.0 / x).real)
    lam = 2.0 * np.pi * N_CLAD * rad / x
    dxm = dx_dm(n, x, m0)
    nc = m0 * N_CLAD
    dm = {"eps": 1j / (2.0 * nc * N_CLAD), "n": N_CORE / (nc * N_CLAD)}
    dlam = {k: -(lam / x) * dxm * v for k, v in dm.items()}
    # Reference self-check: central difference of the polished root in ε″.
    # h = 1e-6 sits between truncation O(h²·x''') and the root tolerance
    # 1e-15/h, both below 1e-8 relative.
    h = 1.0e-6
    fd = (
        mie_pole(n, m_of(N_CORE, EPSI0 + h), x)
        - mie_pole(n, m_of(N_CORE, EPSI0 - h), x)
    ) / (2 * h)
    fd_lam = -(lam / x) * fd
    return {
        "x": x,
        "rad": rad,
        "lam": lam,
        "dlam": dlam,
        "ref_check": abs(fd_lam - dlam["eps"]) / abs(dlam["eps"]),
        "S": gain(lam, dlam["eps"]),
        "T": gain(lam, dlam["n"]),
    }


def bie(ref: dict, n_pts: int, step: float) -> dict:
    """QNMSolver + sensitivity on the circle of `ref`, worst partner reported."""
    rad, lam = ref["rad"], ref["lam"]
    geom = Geometry.gielis(rad=rad, n_pts=n_pts, m=0)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=POL, epsi=EPSI0)
    # Box square on the scale Im λ, as in S0 (a sliver finds nothing).
    h = 0.6 * lam.imag
    res = QNMSolver(geom, mat).modes(
        complex(lam.real - h, 0.4 * lam.imag), complex(lam.real + h, 1.6 * lam.imag)
    )
    d_eps = res.sensitivity(
        lambda d: (
            geom,
            Material(n_core=N_CORE, n_clad=N_CLAD, pol=POL, epsi=EPSI0 + d),
        ),
        step=step,
    )
    d_n = res.sensitivity(
        lambda d: (
            geom,
            Material(n_core=N_CORE + d, n_clad=N_CLAD, pol=POL, epsi=EPSI0),
        ),
        step=step,
    )

    def rel(a: np.ndarray, b: complex) -> float:
        return float(np.max(np.abs(a - b)) / abs(b))

    s_bie = np.array([gain(w, d) for w, d in zip(res.wavelengths, d_eps, strict=True)])
    t_bie = np.array([gain(w, d) for w, d in zip(res.wavelengths, d_n, strict=True)])
    return {
        "modes": res.n_modes,
        "dlam": float(np.max(np.abs(res.wavelengths - lam))),
        "e_eps": rel(d_eps, ref["dlam"]["eps"]),
        "e_n": rel(d_n, ref["dlam"]["n"]),
        "e_S": rel(s_bie, ref["S"]),
        "e_T": rel(t_bie, ref["T"]),
    }


def main() -> None:
    """Analytic S and T per mode, then the BIE agreement on an n_pts ladder."""
    print(f"pol={POL}, n={N_CORE}/{N_CLAD}, baseline 0.1 dB/cm (ε″₀={EPSI0:.4e})")
    cases = {46: (260, 300), 58: (340, 380), 66: (380, 420)}
    for n, ladder in cases.items():
        ref = analytic(n)
        lam, dn = ref["lam"], ref["dlam"]["n"]
        print(
            f"\nn={n}  D={2 * ref['rad'] / 1e3:.2f} µm  "
            f"λ={lam.real:.6f}+{lam.imag:.4e}j  "
            f"Q_i={1 / inv_q(lam):.4g}  S={ref['S']:.5g}  T={ref['T']:.5g}  "
            f"(ref self-check {ref['ref_check']:.1e})"
        )
        print(
            f"  dλ/dε″={ref['dlam']['eps']:.6g}  dλ/dn={ref['dlam']['n']:.6g}  "
            f"Im/|·| of dλ/dn = {abs(dn.imag) / abs(dn):.1e}"
        )
        runs = [(p, 1.0e-5) for p in ladder] + [(ladder[-1], 1.0e-4)]
        for n_pts, step in runs:
            t0 = time.perf_counter()
            b = bie(ref, n_pts, step)
            print(
                f"  n_pts={n_pts:4d} h={step:.0e} modes={b['modes']}  "
                f"|Δλ|={b['dlam']:.1e} nm  dλ/dε″ {b['e_eps']:.1e}  "
                f"dλ/dn {b['e_n']:.1e}  S {b['e_S']:.1e}  T {b['e_T']:.1e}  "
                f"({time.perf_counter() - t0:.0f} s)"
            )


if __name__ == "__main__":
    main()
