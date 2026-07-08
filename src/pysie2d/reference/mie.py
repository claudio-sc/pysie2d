"""Analytic Mie scattering coefficients for an infinite circular cylinder.

Illumination is a plane wave with propagation direction perpendicular to the
cylinder axis (normal incidence, zeta = 90 deg).

Reference: Bohren & Huffman, "Absorption and Scattering of Light by Small
Particles", Chapter 8, eqs. (8.30) and (8.32).

Notation:
    x: size parameter x = k_med * a = 2*pi * n_med * a / lambda_0.
    m: relative refractive index m = n_cyl / n_med (may be complex).

Polarisations:
    TM (Case II, E perp. cylinder axis) → coefficients a_n (eq. 8.32).
    TE (Case I, E par. cylinder axis) → coefficients b_n (eq. 8.30).

This is the mapping the BIE solver's polarisation codes must match:
    pol = 2 (TE, field E_y) ↔ b_n ↔ dict keys Q_*_TE.
    pol = 1 (TM, field H_y) ↔ a_n ↔ dict keys Q_*_TM.

At normal incidence the cross-coupling coefficients vanish:
    b_{n,II} = 0 and a_{n,I} = 0 (text after eq. 8.31).
"""

from __future__ import annotations

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp

# ---------------------------------------------------------------------------
# Low-level coefficient formulas
# ---------------------------------------------------------------------------


def an(n: int | np.ndarray, x: float, m: complex) -> complex | np.ndarray:
    """TM coefficient a_n (B&H eq. 8.32, normal incidence).

    a_n = [ m J_n(mx) J_n'(x) - J_n'(mx) J_n(x) ] /
        [ m J_n(mx) H_n'(x) - J_n'(mx) H_n(x) ]

    where H_n = H_n^(1) and primes denote differentiation w.r.t. the argument.

    Args:
        n: Order(s) of the coefficient.
        x: Size parameter k_med * a.
        m: Relative refractive index n_cyl / n_med.

    Returns:
        The TM coefficient a_n (scalar or array, matching n).
    """
    mx = m * x
    Jn_mx = jv(n, mx)
    Jnp_mx = jvp(n, mx)
    Jn_x = jv(n, x)
    Jnp_x = jvp(n, x)
    Hn_x = hankel1(n, x)
    Hnp_x = h1vp(n, x)

    num = m * Jn_mx * Jnp_x - Jnp_mx * Jn_x
    den = m * Jn_mx * Hnp_x - Jnp_mx * Hn_x
    return num / den


def bn(n: int | np.ndarray, x: float, m: complex) -> complex | np.ndarray:
    """TE coefficient b_n (B&H eq. 8.30, normal incidence).

    b_n = [ J_n(mx) J_n'(x) - m J_n'(mx) J_n(x) ] /
        [ J_n(mx) H_n'(x) - m J_n'(mx) H_n(x) ]

    Args:
        n: Order(s) of the coefficient.
        x: Size parameter k_med * a.
        m: Relative refractive index n_cyl / n_med.

    Returns:
        The TE coefficient b_n (scalar or array, matching n).
    """
    mx = m * x
    Jn_mx = jv(n, mx)
    Jnp_mx = jvp(n, mx)
    Jn_x = jv(n, x)
    Jnp_x = jvp(n, x)
    Hn_x = hankel1(n, x)
    Hnp_x = h1vp(n, x)

    num = Jn_mx * Jnp_x - m * Jnp_mx * Jn_x
    den = Jn_mx * Hnp_x - m * Jnp_mx * Hn_x
    return num / den


# ---------------------------------------------------------------------------
# Truncation order (Wiscombe criterion)
# ---------------------------------------------------------------------------


def _nmax(x: float) -> int:
    return int(np.ceil(np.abs(x) + 4.0 * np.abs(x) ** (1.0 / 3.0) + 2.0)) + 10


# ---------------------------------------------------------------------------
# Vectorised coefficient arrays
# ---------------------------------------------------------------------------


def mie_coefficients(
    x: float,
    m: complex,
    n_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (a, b) of Mie coefficients for orders n = 0 … n_max.

    Args:
        x: Size parameter k_med * a.
        m: Relative refractive index n_cyl / n_med.
        n_max: Highest order included (default: Wiscombe criterion).

    Returns:
        a: complex (n_max+1,) TM coefficients a_0 … a_{n_max}.
        b: complex (n_max+1,) TE coefficients b_0 … b_{n_max}.
    """
    if n_max is None:
        n_max = _nmax(x)
    orders = np.arange(n_max + 1)
    return an(orders, x, m), bn(orders, x, m)


# ---------------------------------------------------------------------------
# Efficiency factors
# ---------------------------------------------------------------------------


def efficiencies(
    x: float,
    m: complex,
    n_max: int | None = None,
) -> dict[str, float]:
    """Extinction, scattering, and absorption efficiency factors.

    Uses the optical theorem for a cylinder (2-D cross-section normalised by
    the geometric width 2a):

        Q_ext = (2/x) Re[ c_0 + 2 * sum_{n>=1} c_n ]
        Q_sca = (2/x)   [ |c_0|^2 + 2 * sum_{n>=1} |c_n|^2 ]
        Q_abs = Q_ext - Q_sca

    where c_n = a_n (TM) or b_n (TE).

    Args:
        x: Size parameter k_med * a.
        m: Relative refractive index n_cyl / n_med.
        n_max: Highest order included (default: Wiscombe criterion).

    Returns:
        dict with keys 'Q_ext_TM', 'Q_sca_TM', 'Q_abs_TM', 'Q_ext_TE',
        'Q_sca_TE', 'Q_abs_TE'.
    """
    a_arr, b_arr = mie_coefficients(x, m, n_max)

    def _qext(c: np.ndarray) -> float:
        return (2.0 / x) * float(np.real(c[0] + 2.0 * np.sum(c[1:])))

    def _qsca(c: np.ndarray) -> float:
        return (2.0 / x) * float(np.abs(c[0]) ** 2 + 2.0 * np.sum(np.abs(c[1:]) ** 2))

    result: dict[str, float] = {}
    for label, c in (("TM", a_arr), ("TE", b_arr)):
        qext = _qext(c)
        qsca = _qsca(c)
        result[f"Q_ext_{label}"] = qext
        result[f"Q_sca_{label}"] = qsca
        result[f"Q_abs_{label}"] = qext - qsca
    return result


# ---------------------------------------------------------------------------
# Self-Green function of a circular cylinder (analytic anchor for v0.2)
# ---------------------------------------------------------------------------


def self_green_cylinder(
    x: float,
    m: complex,
    k: float,
    d: float,
    pol: int,
    n_max: int | None = None,
) -> complex:
    """Analytic self-Green function S(r_s, r_s) for a circular cylinder.

    A line-dipole source sits at distance ``d`` from the cylinder centre
    (``d > a``); ``S`` is the *scattered* field evaluated back at the source.
    Via Graf's addition theorem the modal sum is

        S(r_s, r_s) = (i/4) · Σ_{n=-∞}^{∞} c_n · [H_n^{(1)}(k·d)]²
                    = (i/4) · ( c_0 H_0² + 2 Σ_{n≥1} c_n H_n² ),

    folding to n ≥ 0 using c_{-n} = c_n and H_{-n}² = H_n².

    Convention: ``c_n = SIGN · b_n`` (TE, ``pol=2``) or ``SIGN · a_n``
    (TM, ``pol=1``), where the global ``SIGN`` fixes the scattered-field sign
    convention. The literature genuinely differs here; ``SIGN`` was pinned to
    the BIE solver's convention by the ``test_self_green_vs_analytic_cylinder``
    validation and is documented in ``docs/conventions.md``.

    Args:
        x: Size parameter k·a.
        m: Relative refractive index n_cyl / n_med.
        k: Background wavenumber 2π/λ (rad/nm).
        d: Source distance from the cylinder centre (nm); must exceed a.
        pol: Polarisation code: 2 = TE (b_n), 1 = TM (a_n).
        n_max: Highest order included (default: Wiscombe criterion).

    Returns:
        The complex self-Green function S(r_s, r_s).
    """
    # Sign convention pinned against the BIE solver (see module/conventions).
    SIGN = -1.0
    if n_max is None:
        n_max = _nmax(x)
    orders = np.arange(n_max + 1)
    a_arr, b_arr = mie_coefficients(x, m, n_max)
    c = SIGN * (b_arr if pol == 2 else a_arr)
    h = hankel1(orders, k * d)
    terms = c * h**2
    total = terms[0] + 2.0 * np.sum(terms[1:])
    return complex(0.25j * total)
