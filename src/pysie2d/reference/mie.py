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
from scipy.optimize import newton
from scipy.special import h1vp, hankel1, jv, jvp

# Every routine here accepts a complex size parameter: the QNM poles of the
# coefficients live at complex x, and that is what makes this module usable as
# the analytic anchor for contour-integral mode extraction.
Complexable = float | complex | np.ndarray

# ---------------------------------------------------------------------------
# Low-level coefficient formulas
# ---------------------------------------------------------------------------


def an(n: int | np.ndarray, x: Complexable, m: complex) -> complex | np.ndarray:
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


def bn(n: int | np.ndarray, x: Complexable, m: complex) -> complex | np.ndarray:
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


def _nmax(x: Complexable) -> int:
    return int(np.ceil(np.abs(x) + 4.0 * np.abs(x) ** (1.0 / 3.0) + 2.0)) + 10


# ---------------------------------------------------------------------------
# Vectorised coefficient arrays
# ---------------------------------------------------------------------------


def mie_coefficients(
    x: Complexable,
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
    x: Complexable,
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
    x: Complexable,
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


# ---------------------------------------------------------------------------
# Quasi-normal modes: complex-x poles of the Mie coefficients (anchor for v0.5)
# ---------------------------------------------------------------------------


def qnm_denominator(
    n: int | np.ndarray,
    x: Complexable,
    m: complex,
    pol: int,
) -> complex | np.ndarray:
    """Denominator of the Mie coefficient, whose zeros are the QNMs.

    Verbatim the denominators of ``an`` (eq. 8.32) and ``bn`` (eq. 8.30):

        D^TM_n(x) = m J_n(mx) H_n'(x) - J_n'(mx) H_n(x)     (pol = 1, a_n)
        D^TE_n(x) = J_n(mx) H_n'(x) - m J_n'(mx) H_n(x)     (pol = 2, b_n)

    A zero of ``D`` is a pole of the coefficient: a source-free solution, i.e.
    a quasi-normal mode. Under ``exp(-iωt)`` a decaying mode has ``Im x < 0``,
    which maps to ``Im λ > 0`` (see docs/conventions.md).

    ``D`` depends only on ``(x, m)``, so the modes are radius-independent; the
    radius only maps them onto a wavelength.

    Args:
        n: Order(s) of the coefficient.
        x: Size parameter, generally complex.
        m: Relative refractive index n_cyl / n_med.
        pol: Polarisation code: 2 = TE (b_n), 1 = TM (a_n).

    Returns:
        The complex denominator (scalar or array, broadcasting n against x).
    """
    mx = m * x
    Jn_mx = jv(n, mx)
    Jnp_mx = jvp(n, mx)
    Hn_x = hankel1(n, x)
    Hnp_x = h1vp(n, x)

    if pol == 2:
        return Jn_mx * Hnp_x - m * Jnp_mx * Hn_x
    return m * Jn_mx * Hnp_x - Jnp_mx * Hn_x


def _qnm_residual(
    n: int | np.ndarray,
    x: Complexable,
    m: complex,
    pol: int,
) -> complex | np.ndarray:
    """Scale-free form of ``qnm_denominator``: D divided by its term scale.

    ``|D|`` spans many orders of magnitude across a search box (it inherits the
    exponential growth of ``H_n`` at large order and small argument), so it has
    no usable grid minima to seed from. Dividing by the magnitude of the two
    terms that cancel at a root turns a root into a clean local minimum of a
    quantity of order 1, and makes a single residual tolerance meaningful at
    every order.
    """
    mx = m * x
    Jn_mx = jv(n, mx)
    Jnp_mx = jvp(n, mx)
    Hn_x = hankel1(n, x)
    Hnp_x = h1vp(n, x)

    if pol == 2:
        t1, t2 = Jn_mx * Hnp_x, m * Jnp_mx * Hn_x
    else:
        t1, t2 = m * Jn_mx * Hnp_x, Jnp_mx * Hn_x
    # The +tiny guards 0/0 where both terms underflow; it never perturbs a
    # residual that is being compared against a tolerance of 1e-9.
    return (t1 - t2) / (np.abs(t1) + np.abs(t2) + 1e-300)


def _interior_local_minima(a: np.ndarray) -> np.ndarray:
    """Boolean mask of grid points strictly below all eight neighbours."""
    ok = np.ones_like(a, dtype=bool)
    ok[0, :] = ok[-1, :] = ok[:, 0] = ok[:, -1] = False
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(a, di, axis=0), dj, axis=1)
            ok[1:-1, 1:-1] &= a[1:-1, 1:-1] < shifted[1:-1, 1:-1]
    return ok


def qnm_size_parameters(
    n: int,
    m: complex,
    pol: int,
    x_range: tuple[float, float],
    im_x_floor: float = 1.0e-4,
    im_x_max: float = 1.5,
) -> np.ndarray:
    """Complex size parameters x where ``D_n`` vanishes, inside a rectangle.

    Seeds from local minima of the scale-free residual on a coarse grid, then
    polishes each seed with a secant Newton iteration on the raw denominator.

    ``im_x_floor`` is a Q ceiling, not a numerical detail: a mode with
    ``Q = Re x / (2|Im x|)`` sits below the grid's lowest row once
    ``Q > Re x / (2 · im_x_floor)``, and the seeder cannot see what it never
    samples. It is an argument precisely so that a caller who suspects a
    high-Q mode can lower it — and so that the completeness of a result is
    always checked against the winding number, never against the seeder's own
    say-so.

    Args:
        n: Azimuthal order.
        m: Relative refractive index n_cyl / n_med.
        pol: Polarisation code: 2 = TE (b_n), 1 = TM (a_n).
        x_range: ``(re_lo, re_hi)`` bounds on ``Re x``; ``re_lo`` must be
            positive, which keeps every Hankel argument off the ``H^(1)``
            branch cut on the negative real axis.
        im_x_floor: Smallest ``|Im x|`` the seeding grid samples.
        im_x_max: Largest ``|Im x|`` searched.

    Returns:
        complex (K,) roots inside the rectangle, sorted by ``Re x``. Each is a
        simple zero of ``D_n``; the ±n degeneracy of the 2-D problem is not a
        repeated zero here.

    Raises:
        ValueError: If ``re_lo`` is not positive, or the imaginary bounds are
            not ``0 < im_x_floor < im_x_max``.
    """
    re_lo, re_hi = x_range
    if re_lo <= 0.0:
        raise ValueError(
            f"x_range must lie in Re x > 0 to stay off the H^(1) branch cut; "
            f"got re_lo={re_lo}"
        )
    if not 0.0 < im_x_floor < im_x_max:
        raise ValueError(
            f"need 0 < im_x_floor < im_x_max; got {im_x_floor}, {im_x_max}"
        )

    # 0.02 in Re x resolves the closest pair in the m=3 anchor table (TE n=2
    # and TM n=2 are 0.03 apart); the Im axis is geometric because Q spans
    # four decades across a single ladder of orders.
    re = np.arange(re_lo, re_hi, 0.02)
    im = -np.geomspace(im_x_floor, im_x_max, 60)
    grid = re[None, :] + 1j * im[:, None]
    with np.errstate(all="ignore"):
        resid = np.abs(_qnm_residual(n, grid, m, pol))
    resid = np.nan_to_num(resid, nan=1.0e9, posinf=1.0e9)
    # 0.5 keeps the seed list short without discarding basins: at a true root
    # the residual is 0, and the shallowest genuine minimum measured on the
    # m=3 table is well under 0.1.
    seeds = grid[_interior_local_minima(resid) & (resid < 0.5)]

    found: list[complex] = []
    for z0 in seeds:
        try:
            with np.errstate(all="ignore"):
                root = newton(
                    lambda z: qnm_denominator(n, z, m, pol),
                    z0,
                    tol=1.0e-13,
                    maxiter=80,
                )
        except (RuntimeError, ValueError, OverflowError):
            continue  # a seed that fails to converge is not evidence of a root
        if not np.isfinite(root) or root.real <= 0.0 or root.imag >= 0.0:
            continue
        # Allow a small overshoot past the seeding box, then reject: a root
        # polished outside the requested rectangle belongs to a neighbouring
        # box and would double-count if both were searched.
        if not re_lo <= root.real <= re_hi or not -im_x_max <= root.imag < 0.0:
            continue
        with np.errstate(all="ignore"):
            if abs(_qnm_residual(n, root, m, pol)) > 1.0e-9:
                continue
        if all(abs(root - f) > 1.0e-6 for f in found):
            found.append(complex(root))
    return np.array(sorted(found, key=lambda z: z.real), dtype=complex)


def qnm_wavelengths(
    rad: float,
    m: complex,
    pol: int,
    x_range: tuple[float, float],
    n_max: int | None = None,
    im_x_floor: float = 1.0e-4,
    im_x_max: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic QNM wavelengths of a circular cylinder, over all orders.

    The search rectangle is specified in the **size parameter** ``x``, not in
    ``λ``. ``λ = 2π·rad/x`` is a Möbius map, so a rectangle in one is not a
    rectangle in the other: a completeness claim only transfers between the two
    after the corner mapping is checked. Tests that need a λ-rectangle must
    verify completeness in their own coordinates.

    Args:
        rad: Cylinder radius (nm).
        m: Relative refractive index n_cyl / n_med.
        pol: Polarisation code: 2 = TE (b_n), 1 = TM (a_n).
        x_range: ``(re_lo, re_hi)`` bounds on ``Re x``, with ``re_lo > 0``.
        n_max: Highest order searched. Default ``ceil(|m|·re_hi) + 5`` — the
            order above which a mode cannot be confined, plus margin.
        im_x_floor: Smallest ``|Im x|`` sampled; a Q ceiling, see
            ``qnm_size_parameters``.
        im_x_max: Largest ``|Im x|`` searched.

    Returns:
        orders: int (K,) azimuthal order of each mode.
        wavelengths: complex (K,) wavelengths ``2π·rad/x`` **in the background
            medium**, sorted by ``Re λ``. Multiply by ``n_clad`` for vacuum
            wavelengths; the two coincide for the ``n_clad = 1`` anchor.
        multiplicity: int (K,) 1 for ``n = 0``, else 2 — every ``n ≥ 1`` mode
            of a circle is doubly degenerate through ``exp(±inθ)``.

    Raises:
        ValueError: If a root survives at ``n_max``, which means the truncation
            is too low and modes above it are being silently discarded.
    """
    re_hi = x_range[1]
    if n_max is None:
        n_max = int(np.ceil(abs(m) * re_hi)) + 5

    orders: list[int] = []
    lams: list[complex] = []
    mults: list[int] = []
    for n in range(n_max + 1):
        roots = qnm_size_parameters(n, m, pol, x_range, im_x_floor, im_x_max)
        if n == n_max and roots.size:
            raise ValueError(
                f"D_{n_max} still has {roots.size} root(s) in the box: n_max is "
                f"too low and higher-order modes are being discarded"
            )
        for x in roots:
            orders.append(n)
            lams.append(2.0 * np.pi * rad / x)
            mults.append(1 if n == 0 else 2)

    order_by_re = np.argsort([lam.real for lam in lams])
    return (
        np.array(orders, dtype=int)[order_by_re],
        np.array(lams, dtype=complex)[order_by_re],
        np.array(mults, dtype=int)[order_by_re],
    )
