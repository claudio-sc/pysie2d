"""Independent real-axis reference for the half-space reflected Green function.

A validation anchor, kept as a second implementation like ``mie.py``: it shares
nothing with :mod:`pysie2d.layered` but the formula. It integrates on the **real
axis** with ``scipy.integrate.quad`` after the legacy trigonometric and
hyperbolic substitutions, which cancel the ``1/α₁`` endpoint singularity, and
chooses the branch of α by a different rule from ``layered.alpha`` — the
radiation condition ``Im α ≥ 0`` — instead of the vertical-cut sheet. It is
valid for **real** ``k`` only; complex ``k`` is anchored by continuation of
these values (G3), never by this integral (see :func:`legacy_pec_green`).

Frame and prefactor as in :mod:`pysie2d.layered`:
``G_ind = (i/4π) ∫ R(q)/α₁ · exp(i q X + i α₁ Z) dq``, ``Z = z + z' − 2·z_int``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

PI = np.pi

# Breakpoints at q_sp ± {1, 10, 100, 1000} pole widths. Without them quad misses
# a near-real plasmon pole: a pole 3e-6·k₁ above the axis gave 1e-1 error (t13).
POLE_WIDTHS = (1.0, 10.0, 100.0, 1000.0)
# Integrate the quasi-static tail to exp(−40): below the 1e-14 relative target.
TAIL_DECAY = 40.0


def _alpha_radiating(k: complex, q: float) -> complex:
    """``√(k² − q²)`` on the radiation-condition sheet, ``Im α ≥ 0``."""
    a = np.sqrt(complex(k * k - q * q))
    if a.imag < 0 or (a.imag == 0 and a.real < 0):
        a = -a
    return a


def _reflection(
    q: float, k1: float, k2: complex, eps_rel: complex, pol: int
) -> complex:
    a1 = _alpha_radiating(k1, q)
    a2 = _alpha_radiating(k2, q)
    if pol == 2:
        return (a1 - a2) / (a1 + a2)
    return (a1 - a2 / eps_rel) / (a1 + a2 / eps_rel)


def _breakpoints(k1: float, k2: complex, eps_rel: complex, pol: int) -> list[float]:
    """Real-axis ``q`` values the integrand varies fast around."""
    pts = [k2.real]
    if pol == 1 and eps_rel.real < 0:
        q_sp = k1 * np.sqrt(eps_rel / (1.0 + eps_rel))
        if q_sp.real < 0:
            q_sp = -q_sp
        width = abs(q_sp.imag)
        pts.append(q_sp.real)
        if width > 0:
            for m in POLE_WIDTHS:
                pts += [q_sp.real - m * width, q_sp.real + m * width]
    return [p for p in pts if p > 0]


def g_ref(
    X: float,
    Z: float,
    k1: float,
    eps_rel: complex,
    pol: int,
    which: int = 0,
) -> complex:
    """Reflected Green function (or a source derivative) by real-axis quadrature.

    Args:
        X: Horizontal separation ``x − x'`` (nm).
        Z: ``z + z' − 2·z_int`` (nm), positive.
        k1: **Real** cover wavenumber (rad/nm).
        eps_rel: Substrate permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.
        which: 0 for ``G``, 1 for ``∂G/∂x'``, 2 for ``∂G/∂z'``.

    Returns:
        The requested quantity.
    """
    eps_rel = complex(eps_rel)
    k2 = k1 * np.sqrt(complex(eps_rel.real, eps_rel.imag + 0.0))

    def factor(q: float, a: complex) -> complex:
        return (np.cos(q * X), q * np.sin(q * X), 1j * a * np.cos(q * X))[which]

    def propagating(th: float) -> complex:  # q = k₁ sin θ, α₁ = k₁ cos θ
        q, a = k1 * np.sin(th), k1 * np.cos(th)
        r = _reflection(q, k1, k2, eps_rel, pol)
        return 1j / (2 * PI) * r * factor(q, a) * np.exp(1j * a * Z)

    def evanescent(s: float) -> complex:  # q = k₁ cosh s, α₁ = i k₁ sinh s
        q, a = k1 * np.cosh(s), 1j * k1 * np.sinh(s)
        r = _reflection(q, k1, k2, eps_rel, pol)
        return 1.0 / (2 * PI) * r * factor(q, a) * np.exp(-k1 * np.sinh(s) * Z)

    inside, outside = [], []
    for f in _breakpoints(k1, k2, eps_rel, pol):
        if f < k1:
            inside.append(np.arcsin(f / k1))
        elif f > k1:
            outside.append(np.arccosh(f / k1))
    s_max = np.arcsinh(TAIL_DECAY / (k1 * Z))
    kw = {"epsabs": 1e-17, "epsrel": 1e-14, "limit": 4000}
    total = 0j
    for fn, a, b, pts in (
        (propagating, 0.0, PI / 2, inside),
        (evanescent, 0.0, s_max, outside),
    ):
        pts = sorted(p for p in pts if a < p < b)
        re = quad(lambda t, fn=fn: fn(t).real, a, b, points=pts or None, **kw)[0]
        im = quad(lambda t, fn=fn: fn(t).imag, a, b, points=pts or None, **kw)[0]
        total += re + 1j * im
    return total


def legacy_pec_green(X: float, Z: float, k: complex) -> complex:
    """The legacy real-axis integral with ``R ≡ 1``, kept to pin its failure.

    ``(i/4π) ∫ exp(i q X + i α Z)/α dq`` with the per-point rule ``Im α ≥ 0``,
    evaluated at complex ``k`` with ``Im k < 0``. It does **not** return the
    outgoing ``(i/4)·H₀^{(1)}(kρ)``: R depends on ``k`` only through ``k²``, so
    the integral equals its value at ``−k``, which is the *incoming* wave
    ``−(i/4)·H₀^{(2)}(kρ)`` (DLMF 10.11.5; holomorphy note §1). It is here so
    that the defect cannot come back unnoticed, not to be used.
    """

    def integrand(q: float) -> complex:
        a = np.sqrt(k * k - q * q + 0j)
        if a.imag < 0:
            a = -a
        return 1j / (2 * PI) * np.cos(q * X) * np.exp(1j * a * Z) / a

    kw = {"limit": 4000, "epsabs": 1e-16, "epsrel": 1e-13, "points": [k.real]}
    q_max = k.real + 45.0 / Z
    re = quad(lambda q: integrand(q).real, 0, q_max, **kw)[0]
    im = quad(lambda q: integrand(q).imag, 0, q_max, **kw)[0]
    return re + 1j * im
