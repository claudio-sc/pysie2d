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


def gt_ref(
    X: float,
    hs: float,
    hb: float,
    k1: float,
    eps_rel: complex,
    pol: int,
    which: int = 0,
) -> complex:
    """Transmitted Green function (or a source derivative) by real-axis quadrature.

    ``G_t = (i/4π) ∫ τ(q)·exp(i q X + i α₁ h' + i α₂ |h|) dq`` with
    ``τ = 2/(α₁+α₂)`` (TE) or ``2/(α₁+α₂/ε)`` (TM), folded onto ``q ≥ 0`` and
    integrated directly in ``q``: τ is finite at ``q = k₁``, so no substitution
    is needed. Branch by the radiation condition, breakpoints at ``k₁``,
    ``Re k₂`` and the plasmon's ``POLE_WIDTHS``, as :func:`g_ref`.

    Args:
        X: Horizontal separation ``x − x'`` (nm).
        hs: Source height above the interface (nm), positive.
        hb: Observation depth below it (nm), non-negative.
        k1: **Real** cover wavenumber (rad/nm).
        eps_rel: Substrate permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.
        which: 0 for ``G_t``, 1 for ``∂G_t/∂x'``, 2 for ``∂G_t/∂z'``.

    Returns:
        The requested quantity.
    """
    eps_rel = complex(eps_rel)
    k2 = k1 * np.sqrt(complex(eps_rel.real, eps_rel.imag + 0.0))

    def integrand(q: float) -> complex:
        a1, a2 = _alpha_radiating(k1, q), _alpha_radiating(k2, q)
        tau = 2 / (a1 + a2) if pol == 2 else 2 / (a1 + a2 / eps_rel)
        fac = (np.cos(q * X), q * np.sin(q * X), 1j * a1 * np.cos(q * X))[which]
        return 1j / (2 * PI) * tau * fac * np.exp(1j * (a1 * hs + a2 * hb))

    q_max = max(k1, abs(k2)) * 1.5 + TAIL_DECAY / (hs + hb)
    pts = sorted(p for p in [k1, *_breakpoints(k1, k2, eps_rel, pol)] if 0 < p < q_max)
    kw = {"epsabs": 1e-17, "epsrel": 1e-13, "limit": 4000, "points": pts}
    re = quad(lambda q: integrand(q).real, 0, q_max, **kw)[0]
    im = quad(lambda q: integrand(q).imag, 0, q_max, **kw)[0]
    return re + 1j * im


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


# --- N-layer stack (v1.2): ported from studies/slab/sproto.py ------------------
#
# Independent of pysie2d.layered's stack code in every ingredient that could
# hide a shared error: R by the transfer matrix (not the recursion), each
# layer scaled by e^{−|Im α|d} (R is homogeneous of degree 0 in the matrix, so
# the scaling is exact and borrows nothing), the radiation-condition branch,
# QUADPACK on the real axis, and the guided poles by a symmetric-fold principal
# value plus iπ·Res, the residue from a Cauchy circle on this module's own R.


def _stack_r_ref(
    q: float,
    k1: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
) -> complex:
    """Transfer-matrix R with the radiation-condition branch (real ``k``)."""
    y1 = _alpha_radiating(k1, q)
    a_b = _alpha_radiating(k1 * np.sqrt(complex(eps_bot)), q)
    y_b = a_b if pol == 2 else a_b / eps_bot
    m = np.eye(2, dtype=complex)
    for e, dj in zip(eps_layers, d, strict=True):
        kj = k1 * np.sqrt(complex(e))
        a = np.sqrt(complex(kj * kj - q * q))
        w = 1.0 if pol == 2 else e
        # Formed from bounded exponentials, since cos itself overflows first.
        ep = np.exp(1j * a * dj - abs(a.imag) * dj)
        em = np.exp(-1j * a * dj - abs(a.imag) * dj)
        c = 0.5 * (ep + em)
        if abs(a * dj) > 1e-3:
            sinc = (ep - em) / (2j * a)
        else:
            sinc = np.sinc(a * dj / PI) * dj * np.exp(-abs(a.imag) * dj)
        m = np.array([[c, -w * sinc], [(a * a / w) * sinc, c]]) @ m
    a_ = m[0, 0] + 1j * y1 * m[0, 1]
    b_ = m[0, 0] - 1j * y1 * m[0, 1]
    c_ = m[1, 0] + 1j * y1 * m[1, 1]
    d_ = m[1, 0] - 1j * y1 * m[1, 1]
    return (d_ + 1j * y_b * b_) / (-1j * y_b * a_ - c_)


def stack_r_ref(
    q: float,
    k1: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
) -> complex:
    """Reference R of a stack at one ``q``; see :func:`g_ref_stack`.

    Args:
        q: Transverse wavenumber (real, or complex off the branch cuts).
        k1: **Real** cover wavenumber.
        eps_layers: Layer permittivities relative to the cover, top first.
        d: Layer thicknesses (nm).
        eps_bot: Bottom permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.

    Returns:
        ``R(q)``.
    """
    return _stack_r_ref(q, k1, eps_layers, d, eps_bot, pol)


def _residue_ref(
    beta: float,
    k_out: float,
    fn,
) -> complex:
    """Residue of ``fn`` at the real pole β by a 64-point Cauchy circle.

    The radius ``min(0.3(β − k_out), 1e-3·β)`` keeps the branch point outside.
    """
    r = min(0.3 * (beta - k_out), 1e-3 * beta)
    t = 2 * PI * np.arange(64) / 64
    return complex(
        np.mean([fn(beta + r * np.exp(1j * ti)) * r * np.exp(1j * ti) for ti in t])
    )


def g_ref_stack(
    X: float,
    Z: float,
    k1: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
    poles: tuple[float, ...],
    which: int = 0,
    extra_pts: tuple[float, ...] = (),
    hw_frac: float = 0.45,
) -> complex:
    """Reflected G of a stack (or a source derivative) by real-axis quadrature.

    θ-substitution on ``[0, k₁]``, cosh on ``[k₁, ∞)``. Real guided poles:
    principal value by the symmetric fold ``∫₀ʰ [f(s_p+u) + f(s_p−u)] du`` (the
    ±Res/u parts cancel, so a wrong residue cannot hide in it), plus ``iπ·Res``
    for the path passing below +β. For a lossy stack pass no poles and their
    real parts as ``extra_pts`` (plain QUADPACK with breakpoints).

    Args:
        X: Horizontal separation ``x − x'`` (nm).
        Z: ``z + z' − 2·z_top`` (nm), positive.
        k1: **Real** cover wavenumber.
        eps_layers: Layer permittivities relative to the cover, top first.
        d: Layer thicknesses (nm).
        eps_bot: Bottom permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.
        poles: Real guided poles β, all beyond ``k₁``.
        which: 0 for ``G``, 1 for ``∂G/∂x'``, 2 for ``∂G/∂z'``.
        extra_pts: Further breakpoints in q.
        hw_frac: Fold half-width as a fraction of the nearest feature spacing.

    Returns:
        The requested quantity.
    """
    k_bot = float(np.real(k1 * np.sqrt(complex(eps_bot))))
    k_out = max(k1, k_bot)

    def rr(q):
        return _stack_r_ref(q, k1, eps_layers, d, eps_bot, pol)

    def fac(q, a):
        return (np.cos(q * X), q * np.sin(q * X), 1j * a * np.cos(q * X))[which]

    def f1(th):
        q, a = k1 * np.sin(th), k1 * np.cos(th)
        return 1j / (2 * PI) * rr(q) * fac(q, a) * np.exp(1j * a * Z)

    def f2(s):
        q, a = k1 * np.cosh(s), 1j * k1 * np.sinh(s)
        return 1 / (2 * PI) * rr(q) * fac(q, a) * np.exp(-k1 * np.sinh(s) * Z)

    kw = {"epsabs": 1e-18, "epsrel": 1e-14, "limit": 4000}

    def qd(fn, a, b, pts=None):
        re = quad(lambda t: fn(t).real, a, b, points=pts or None, **kw)[0]
        im = quad(lambda t: fn(t).imag, a, b, points=pts or None, **kw)[0]
        return re + 1j * im

    brk = [k_bot, *extra_pts]
    pts1 = [np.arcsin(f / k1) for f in brk if 0 < f < k1]
    s_pole = [np.arccosh(b / k1) for b in poles]
    pts2 = sorted({np.arccosh(f / k1) for f in brk if f > k1})
    s_max = np.arccosh(40.0 / (k1 * Z) + max([*poles, *brk, k1]) / k1)
    out = qd(f1, 0, PI / 2, pts1)
    edges = sorted({0.0, s_max, *pts2})
    gaps = [abs(a - b) for a in s_pole for b in s_pole + edges if a != b]
    hw = hw_frac * min([*gaps, 0.1])
    cuts = sorted(edges + [s - hw for s in s_pole] + [s + hw for s in s_pole])
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        c = [s for s in s_pole if a < s < b]
        if c:
            sp = c[0]
            out += qd(lambda u, sp=sp: f2(sp + u) + f2(sp - u), 0.0, b - sp)
        else:
            out += qd(f2, a, b)
    pre = 1j / (2 * PI)
    for b in poles:
        res = _residue_ref(b, k_out, rr)
        a1 = 1j * np.sqrt(b * b - k1 * k1)
        out += 1j * PI * pre * res / a1 * fac(b, a1) * np.exp(1j * a1 * Z)
    return out


def g_ref_stack_gamma(
    X: float,
    Z: float,
    k1: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
    poles: tuple[float, ...],
    which: int = 0,
    hw_frac: float = 0.45,
) -> complex:
    """:func:`g_ref_stack` for near-cutoff poles: ``q ≥ k_out`` integrated in γ.

    ``q = √(k_out² + γ²)`` makes the outer branch point a smooth endpoint, so
    a pole at ``γ_p = √(β² − k_out²)`` sits ``√(2t)·k_out`` from it instead of
    ``t·k_out``; in the cosh variable the reference swings by O(1) as t → 0
    (finding 2). ``[k₁, k_out]`` (when ``k₁ < k_out``) by cosh near ``k₁`` and by
    ``q = √(k_out² − y²)`` near ``k_out``. Same fold, same residue rule.

    Args:
        X: Horizontal separation (nm).
        Z: ``z + z' − 2·z_top`` (nm), positive.
        k1: **Real** cover wavenumber.
        eps_layers: Layer permittivities relative to the cover, top first.
        d: Layer thicknesses (nm).
        eps_bot: Bottom permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.
        poles: Real guided poles β.
        which: 0 for ``G``, 1 for ``∂G/∂x'``, 2 for ``∂G/∂z'``.
        hw_frac: Fold half-width fraction.

    Returns:
        The requested quantity.
    """
    ko = max(k1, float(np.real(k1 * np.sqrt(complex(eps_bot)))))
    pre = 1j / (2 * PI)

    def rr(q):
        return _stack_r_ref(q, k1, eps_layers, d, eps_bot, pol)

    def fac(q, a):
        return (np.cos(q * X), q * np.sin(q * X), 1j * a * np.cos(q * X))[which]

    def f1(th):
        q, a = k1 * np.sin(th), k1 * np.cos(th)
        return pre * rr(q) * fac(q, a) * np.exp(1j * a * Z)

    def f2(s):
        q, a = k1 * np.cosh(s), 1j * k1 * np.sinh(s)
        return 1 / (2 * PI) * rr(q) * fac(q, a) * np.exp(-k1 * np.sinh(s) * Z)

    def f3(g):
        q = np.sqrt(ko * ko + g * g)
        a = 1j * np.sqrt(q * q - k1 * k1) if ko > k1 else 1j * g
        # R/α₁·dq/dγ: for k₁ = k_out, α₁ = iγ cancels γ in the Jacobian.
        w = (1.0 / (1j * q)) if ko == k1 else (g / q) / a
        return pre * rr(q) * w * fac(q, a) * np.exp(1j * a * Z)

    def f4(y):  # q = √(k_out² − y²): the pole's left flank, uniformised
        q = np.sqrt(ko * ko - y * y)
        a = 1j * np.sqrt(q * q - k1 * k1)
        return pre * rr(q) / a * (y / q) * fac(q, a) * np.exp(1j * a * Z)

    kw = {"epsabs": 1e-18, "epsrel": 1e-14, "limit": 4000}

    def qd(fn, a, b):
        re = quad(lambda t: fn(t).real, a, b, **kw)[0]
        im = quad(lambda t: fn(t).imag, a, b, **kw)[0]
        return re + 1j * im

    out = qd(f1, 0, PI / 2)
    if ko > k1:
        mid = 0.5 * (k1 + ko)
        out += qd(f2, 0, np.arccosh(mid / k1))
        out += qd(f4, 0, np.sqrt(ko * ko - mid * mid))
    g_pole = sorted(np.sqrt(b * b - ko * ko) for b in poles)
    g_max = np.sqrt((40.0 / Z + max([*poles, ko])) ** 2 - ko * ko)
    edges = [0.0, g_max]
    gaps = [abs(a - b) for a in g_pole for b in g_pole + edges if a != b]
    hw = hw_frac * min([*gaps, g_max])
    cuts = sorted(edges + [g - hw for g in g_pole] + [g + hw for g in g_pole])
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        c = [g for g in g_pole if a < g < b]
        if c:
            gp = c[0]
            out += qd(lambda u, gp=gp: f3(gp + u) + f3(gp - u), 0.0, b - gp)
        else:
            out += qd(f3, a, b)
    for b in poles:
        res = _residue_ref(b, ko, rr)
        a1 = 1j * np.sqrt(b * b - k1 * k1)
        out += 1j * PI * pre * res / a1 * fac(b, a1) * np.exp(1j * a1 * Z)
    return out


def slab3_roots(
    k: float, eps_c: float, eps_f: float, eps_s: float, d: float, pol: int
) -> list[float]:
    """Guided β of a three-layer slab by the textbook closed form (Marcuse, Kogelnik).

    ``κd = mπ + atan(p_c/κ) + atan(p_s/κ)`` with ``κ = √(k_f² − β²)`` and
    ``p = γ`` (TE) or ``γ·ε_f/ε`` (TM): independent of any transfer matrix.

    Args:
        k: Vacuum-referred wavenumber the permittivities multiply (rad/nm).
        eps_c: Cover permittivity (same reference as the others).
        eps_f: Film permittivity.
        eps_s: Substrate permittivity.
        d: Film thickness (nm).
        pol: 2 = TE, 1 = TM.

    Returns:
        β of every guided mode, descending.
    """
    from scipy.optimize import brentq

    kc, kf, ks = (k * np.sqrt(e) for e in (eps_c, eps_f, eps_s))
    lo, hi = max(kc, ks), kf
    roots: list[float] = []
    m = 0
    while True:

        def g(b, m=m):
            kap = np.sqrt(kf**2 - b**2)
            gc, gs = np.sqrt(b**2 - kc**2), np.sqrt(b**2 - ks**2)
            pc, ps = (gc, gs) if pol == 2 else (gc * eps_f / eps_c, gs * eps_f / eps_s)
            return kap * d - m * PI - np.arctan(pc / kap) - np.arctan(ps / kap)

        a, b = lo * (1 + 1e-15), hi * (1 - 1e-15)
        if g(a) <= 0:
            return roots
        roots.append(brentq(g, a, b, xtol=1e-16 * hi, rtol=8.9e-16, maxiter=200))
        m += 1


def _stack_t_ref(
    q: complex,
    k1: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
) -> complex:
    """T by shooting **upward** from the bottom (s7's ``T_ref``).

    The forward form ``T = aR + b`` cancels catastrophically on the tail (true
    T ~ e^{−qd} from O(1) terms, measured 1e37, finding 4); shooting from the
    evanescent side makes the growth the dominant direction, so ``T = 1/A_down``
    is relatively accurate. Independent of the recursion.
    """
    y1 = _alpha_radiating(k1, q)
    a_b = _alpha_radiating(k1 * np.sqrt(complex(eps_bot)), q)
    y_b = a_b if pol == 2 else a_b / eps_bot
    u, v = 1.0 + 0j, -1j * y_b  # bottom field e^{−iα_b(z − z_bot)}: u = 1, v = p·u′
    for e, dj in zip(eps_layers[::-1], d[::-1], strict=True):
        a = np.sqrt(complex(k1 * k1 * e - q * q))
        w = 1.0 if pol == 2 else e
        c = np.cos(a * dj)
        sinc = np.sinc(a * dj / PI) * dj
        u, v = c * u + w * sinc * v, -(a * a / w) * sinc * u + c * v
        if not (np.isfinite(u) and np.isfinite(v)) or abs(u) > 1e290:
            return 0.0
    return 1.0 / (0.5 * (u - v / (1j * y1)))  # 1/(downward amplitude at z_top)


def gt_ref_stack(
    X: float,
    hs: float,
    hb: float,
    k1: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
    poles: tuple[float, ...],
    hw_frac: float = 0.45,
) -> complex:
    """Transmitted G below a stack by real-axis quadrature (s7's ``gt_ref``).

    ``G_t = (i/4π) ∫ τ·e^{iqX + iα₁h' + iα_b|h|} dq`` with ``τ = T/α₁``, T by
    upward shooting, the radiation-condition branch, θ/cosh substitutions, and
    the guided poles by the symmetric-fold PV plus ``iπ·Res τ``, the residue a
    Cauchy circle on this module's own ``T/α₁``.

    Args:
        X: Horizontal separation (nm).
        hs: Source height above ``z_top`` (nm), positive.
        hb: Observation depth below ``z_bot`` (nm), non-negative.
        k1: **Real** cover wavenumber.
        eps_layers: Layer permittivities relative to the cover, top first.
        d: Layer thicknesses (nm).
        eps_bot: Bottom permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.
        poles: Real guided poles β.
        hw_frac: Fold half-width fraction.

    Returns:
        ``G_t``.
    """
    k_b = k1 * np.sqrt(complex(eps_bot))
    kbr = float(np.real(k_b))

    def tt(q):
        return _stack_t_ref(q, k1, eps_layers, d, eps_bot, pol)

    def f1(th):
        q, a = k1 * np.sin(th), k1 * np.cos(th)
        ab = _alpha_radiating(k_b, q)
        # τ dq = (T/α₁)·α₁ dθ
        return (
            1j / (2 * PI) * tt(q) * np.cos(q * X) * np.exp(1j * a * hs + 1j * ab * hb)
        )

    def f2(s):
        q, a = k1 * np.cosh(s), 1j * k1 * np.sinh(s)
        ab = _alpha_radiating(k_b, q)
        # τ dq = (T/α₁)·k₁ sinh s ds = −iT ds, so (i/2π)·τ dq = T ds/2π
        return 1 / (2 * PI) * tt(q) * np.cos(q * X) * np.exp(1j * a * hs + 1j * ab * hb)

    kw = {"epsabs": 1e-18, "epsrel": 1e-13, "limit": 4000}

    def qd(fn, a, b, pts=None):
        re = quad(lambda t: fn(t).real, a, b, points=pts, **kw)[0]
        im = quad(lambda t: fn(t).imag, a, b, points=pts, **kw)[0]
        return re + 1j * im

    out = qd(f1, 0, PI / 2, [np.arcsin(kbr / k1)] if kbr < k1 else None)
    sp = [np.arccosh(b / k1) for b in poles]
    edges = sorted({0.0, *([np.arccosh(kbr / k1)] if kbr > k1 else [])})
    edges.append(np.arccosh(40.0 / (k1 * (hs + hb)) + max([*poles, kbr, k1]) / k1))
    gaps = [abs(a - b) for a in sp for b in sp + edges if a != b]
    hw = hw_frac * min([*gaps, 0.1])
    cuts = sorted(edges + [s - hw for s in sp] + [s + hw for s in sp])
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        c = [s for s in sp if a < s < b]
        if c:
            s0 = c[0]
            out += qd(lambda u, s0=s0: f2(s0 + u) + f2(s0 - u), 0.0, b - s0)
        else:
            out += qd(f2, a, b)
    k_out = max(k1, kbr)
    for b in poles:
        res = _residue_ref(b, k_out, lambda q: tt(q) / _alpha_radiating(k1, q))
        a1 = 1j * np.sqrt(b * b - k1 * k1)
        ab = 1j * np.sqrt(b * b - kbr * kbr)
        pre = 1j * PI * (1j / (2 * PI))
        out += pre * res * np.cos(b * X) * np.exp(1j * a1 * hs + 1j * ab * hb)
    return out
