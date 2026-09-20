"""BIE matrix assembly and Hankel-function helpers — the numerical core.

Implements the boundary integral equation (BIE) for 2-D electromagnetic
scattering from a cylinder in a homogeneous background, discretised with
Kress–Martensen product quadrature on nodes equispaced in the quadrature
parameter t. All functions accept complex wavenumbers, making this module
usable for both driven (real λ) and quasi-normal-mode (complex λ) problems.

Convention:
    The BIE solution vector ``ei`` has shape (2*nn,):
        ei[:nn]  — φ  : electric-field values on the boundary
        ei[nn:]  — χ  : normal-derivative values on the boundary, carrying the
                        Jacobian |dx/dt| of the quadrature parameter

Public API:
    hank0, hank1, cbesh: Hankel-function wrappers.
    assemble_matrix: fast vectorised 2nn × 2nn BIE system matrix M(λ).
    assemble_matrix_dwn: the same matrix and its analytic derivative with
        respect to the background wavenumber, sharing the Bessel evaluations.
    assemble_matrix_reference: slow loop-based assembly, kept only as the
        rounding-accurate truth anchor for the parity test.
"""

import functools

import numpy as np
from scipy.special import hankel1, j0, j1, jv, y0, y1

PI = np.pi


# ---------------------------------------------------------------------------
# Hankel function helpers
#
# For real x, H_n^(1)(x) = J_n(x) + i·Y_n(x) exactly, and the Cephes j0/y0/j1/y1
# kernels are 11-13x faster than the Amos algorithm behind scipy's hankel1
# (~45 ns vs ~515 ns per evaluation), agreeing to 4e-15 relative. Complex
# arguments fall through to hankel1 unchanged — complex-wavenumber (QNM)
# support is untouched. See docs/conventions.md section 6.
# ---------------------------------------------------------------------------


def _real_if_real(w: complex) -> complex | float:
    """Demote an exactly-real complex scalar to a float.

    ``scipy.special.hankel1`` dispatches on argument *dtype*, not value, so a
    wavenumber carrying a zero imaginary part (e.g. ``Material.nc`` returns
    ``2+0j`` for a non-absorbing particle) would force every downstream Hankel
    evaluation onto the slow complex Amos path. Demoting such values to float
    keeps the derived argument arrays in real dtype so :func:`hank0` and
    :func:`hank1` can take their fast branch.

    Args:
        w: Scalar wavenumber, real or complex.

    Returns:
        ``float(w.real)`` when ``w.imag`` is exactly zero, otherwise ``w``
        unchanged.
    """
    wc = complex(w)
    return wc.real if wc.imag == 0.0 else w


def hank0(x: complex | np.ndarray) -> complex | np.ndarray:
    """H_0^(1)(x), first-kind Hankel order 0. Real or complex x.

    Real arguments take a Cephes fast path; complex arguments use the Amos
    algorithm as before.
    """
    if np.iscomplexobj(x):
        return hankel1(0, x)
    return j0(x) + 1j * y0(x)


def hank1(x: complex | np.ndarray) -> complex | np.ndarray:
    """H_1^(1)(x), first-kind Hankel order 1. Real or complex x.

    Real arguments take a Cephes fast path; complex arguments use the Amos
    algorithm as before.
    """
    if np.iscomplexobj(x):
        return hankel1(1, x)
    return j1(x) + 1j * y1(x)


def cbesh(z: complex | np.ndarray, order: int) -> complex | np.ndarray:
    """H_order^(1)(z) for complex z. Replaces Fortran cbesh.

    Orders 0 and 1 with real argument take a Cephes fast path; everything else
    uses the Amos algorithm as before.
    """
    if not np.iscomplexobj(z):
        if order == 0:
            return hank0(z)
        if order == 1:
            return hank1(z)
    return hankel1(order, z)


# ---------------------------------------------------------------------------
# Kress–Martensen product quadrature
#
# The log singularity of H₀ at r → 0 is split off as J₀(kr)·ln(4 sin²((t−τ)/2))
# and integrated exactly against the trigonometric interpolant through the
# nodes (Kress, Linear Integral Equations, ch. 12); everything left over is
# smooth and periodic, so the trapezoid rule is spectral on it. See
# docs/design/kress-spec.md §3 for the derivation of every expression below.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _kress_log_weights(nn: int) -> np.ndarray:
    """Kress's weights ``R_d`` for the singular factor ``ln(4 sin²((t−τ)/2))``.

    With nodes equispaced in the quadrature parameter,

        ∫₀^{2π} ln(4 sin²((t_i − τ)/2)) φ(τ) dτ ≈ Σ_j R_{(j−i) mod nn} φ(t_j)

    exactly whenever φ is a trigonometric polynomial the nodes interpolate.
    ``R`` is circulant — it depends only on the node offset ``d`` — and
    independent of geometry, wavelength and material, which is why it is
    cached per ``nn`` for the life of the process.

    The formula holds for **both parities** of ``nn``. The textbook form
    (``nn = 2n``) carries a half-weighted Nyquist term that does not exist for
    odd ``nn``; using it there returns a plausible, wrong matrix *(measured: a
    QNM displaced by 1.1 nm at ``nn = 115`` with nothing raised)*.

    Args:
        nn: Number of boundary nodes.

    Returns:
        Read-only (nn,) array ``R_d``, ``d = 0 … nn−1``.
    """
    d = 2.0 * PI * np.arange(nn) / nn
    orders = np.arange(1, (nn - 1) // 2 + 1)
    weights = -(4.0 * PI / nn) * (np.cos(np.outer(d, orders)) / orders).sum(axis=1)
    if nn % 2 == 0:
        weights = weights - (4.0 * PI / nn**2) * np.cos(0.5 * nn * d)
    weights.flags.writeable = False
    return weights


@functools.lru_cache(maxsize=64)
def _kress_weights(nn: int) -> np.ndarray:
    """Circulant correction ``W_d = R_d − h·ln(4 sin²(πd/nn))``, ``W_0 = R_0``.

    Kress writes each off-diagonal entry as ``K₁·R + (K − K₁·L)·h`` with
    ``L = ln(4 sin²(πd/nn))``. Collecting the ``K₁`` terms gives
    ``h·K + K₁·W``: the plain trapezoid entry plus a correction proportional
    to ``W``. That is the form assembled here, so ``L`` never needs its own
    O(nn²) array. On the diagonal ``L`` is undefined and only ``R_0`` enters.

    The array is shared between every assembly at this ``nn`` and between
    threads (``beyn.contour_moments``), so it is returned read-only: an
    in-place edit would corrupt every later matrix silently.

    Args:
        nn: Number of boundary nodes.

    Returns:
        Read-only (nn,) array ``W_d``, ``d = 0 … nn−1``.
    """
    w = np.array(_kress_log_weights(nn))
    d = np.arange(1, nn)
    w[1:] -= (2.0 * PI / nn) * np.log(4.0 * np.sin(PI * d / nn) ** 2)
    w.flags.writeable = False
    return w


def _j0_h0(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``J₀(z)`` and ``H₀^{(1)}(z)`` from one pass, real or complex ``z``.

    Real arguments reuse the Cephes ``J₀`` inside ``H₀ = J₀ + i·Y₀``, so the
    extra Bessel array Kress needs costs nothing there. Complex arguments —
    every QNM assembly — need a separate ``jv`` call; that is the real cost of
    the scheme *(measured 1.45× one shipped assembly at nn = 200, complex λ)*.
    The complex branch must stay: it is what makes ``M(λ)`` holomorphic.
    """
    if np.iscomplexobj(z):
        return jv(0, z), hankel1(0, z)
    j = j0(z)
    return j, j + 1j * y0(z)


def _j1_h1(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``J₁(z)`` and ``H₁^{(1)}(z)``; see :func:`_j0_h0`."""
    if np.iscomplexobj(z):
        return jv(1, z), hankel1(1, z)
    j = j1(z)
    return j, j + 1j * y1(z)


def assemble_matrix_reference(
    pol: int,
    nn: int,
    f: np.ndarray,
    g: np.ndarray,
    df: np.ndarray,
    dg: np.ndarray,
    ddf: np.ndarray,
    ddg: np.ndarray,
    wnum_bg: complex,
    ri: complex,
    kd: complex,
) -> np.ndarray:
    """Assemble the 2nn × 2nn BIE system matrix, one entry at a time.

    Kept **only** as the truth anchor for the parity test against
    :func:`assemble_matrix`. It is independent of the fast path in the three
    places a vectorisation bug hides: it indexes node pairs in an explicit
    double loop rather than through ``triu_indices``, it builds ``W`` from
    ``R`` and the logarithm inline rather than from the cached
    :func:`_kress_weights`, and it always evaluates Bessel functions at a
    **complex** argument, so the real Cephes branch is checked too.

    Args:
        pol: Polarisation: 1 = p (TM), 2 = s (TE).
        nn: Number of boundary nodes, equispaced in the quadrature parameter t.
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).
        df: (nn,) df/dt.
        dg: (nn,) dg/dt.
        ddf: (nn,) d²f/dt².
        ddg: (nn,) d²g/dt².
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm), where
            λ_vac is the public **vacuum** wavelength. Build it with
            :meth:`pysie2d.material.Material.wnum_bg`; this function takes no
            wavelength, so the vacuum conversion cannot be applied twice.
            Pass a complex value for quasi-normal-mode searches.
        ri: Refractive index of the particle: nc = √(εᵣ + iεᵢ).
        kd: Dielectric constant of the particle: ε = εᵣ + iεᵢ.

    Returns:
        me: complex (2nn, 2nn) BIE system matrix.
    """
    nt = 2 * nn
    me = np.zeros((nt, nt), dtype=complex)
    wnum_core = complex(ri * wnum_bg)
    wnum_bg = complex(wnum_bg)
    eta = complex(kd if pol == 1 else 1.0)
    h = 2.0 * PI / nn
    r_log = _kress_log_weights(nn)

    for i in range(nn):
        for j in range(i + 1, nn):
            d = j - i
            w_d = r_log[d] - h * np.log(4.0 * np.sin(PI * d / nn) ** 2)
            dx = f[i] - f[j]
            dz = g[i] - g[j]
            r = np.sqrt(dx**2 + dz**2)
            c_ij = dx * dg[j] - dz * df[j]
            c_ji = -dx * dg[i] + dz * df[i]
            # Row offset 0: M1 (φ columns) and M2 (χ columns) at k_bg.
            # Row offset nn: M3 and M4 at k_core, M4 carrying eta.
            for k, row, scale in ((wnum_bg, 0, 1.0), (wnum_core, nn, eta)):
                z = k * r
                single = 0.25j * h * hankel1(0, z) - w_d * jv(0, z) / (4.0 * PI)
                ratio = 0.25j * h * hankel1(1, z) / z - w_d * jv(1, z) / z / (4.0 * PI)
                me[i + row, j] = k**2 * ratio * c_ij
                me[j + row, i] = k**2 * ratio * c_ji
                me[i + row, j + nn] = scale * single
                me[j + row, i + nn] = scale * single

    gamma = np.sqrt(df**2 + dg**2)
    curv = (df * ddg - ddf * dg) * h / (4.0 * PI) / gamma**2
    for i in range(nn):
        me[i, i] = 0.5 - curv[i]
        me[i + nn, i] = -(0.5 + curv[i])
        for k, row, scale in ((wnum_bg, 0, 1.0), (wnum_core, nn, eta)):
            log_term = np.euler_gamma + np.log(k * gamma[i] / 2.0)
            me[i + row, i + nn] = scale * (
                -r_log[0] / (4.0 * PI) + h * (0.25j - log_term / (2.0 * PI))
            )
    return me


def assemble_matrix(
    pol: int,
    nn: int,
    f: np.ndarray,
    g: np.ndarray,
    df: np.ndarray,
    dg: np.ndarray,
    ddf: np.ndarray,
    ddg: np.ndarray,
    wnum_bg: complex,
    ri: complex,
    kd: complex,
) -> np.ndarray:
    """Vectorised Kress–Martensen BIE system matrix M(λ).

    Nodes must be **equispaced in the quadrature parameter t** and ``df`` …
    ``ddg`` must be derivatives with respect to that same ``t``
    (``docs/conventions.md`` §13.1); the step is ``h = 2π/nn`` and is not an
    argument. Anything else — nodes graded in t, derivatives in θ on a graded
    map — returns a plausible matrix with first-order error and raises nothing.
    :meth:`pysie2d.geometry.Geometry.gielis` builds conforming arrays.

    Each off-diagonal entry is the trapezoid term ``h·K`` plus the Kress
    correction ``K₁·W_d`` (:func:`_kress_weights`); Hankel and Bessel functions
    are evaluated on the ``nn(nn−1)/2`` unique distances and fill both ``(i, j)``
    and ``(j, i)``. Parameters and return value are identical to
    :func:`assemble_matrix_reference`, against which it is validated to
    rounding by the parity test.

    Args:
        pol: Polarisation: 1 = p (TM), 2 = s (TE).
        nn: Number of boundary nodes, equispaced in the quadrature parameter t.
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).
        df: (nn,) df/dt.
        dg: (nn,) dg/dt.
        ddf: (nn,) d²f/dt².
        ddg: (nn,) d²g/dt².
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm), where
            λ_vac is the public **vacuum** wavelength. Build it with
            :meth:`pysie2d.material.Material.wnum_bg`; this function takes no
            wavelength, so the vacuum conversion cannot be applied twice.
            Pass a complex value for quasi-normal-mode searches.
        ri: Refractive index of the particle: nc = √(εᵣ + iεᵢ).
        kd: Dielectric constant of the particle: ε = εᵣ + iεᵢ.

    Returns:
        me: complex (2nn, 2nn) BIE system matrix.
    """
    # Demote exactly-real wavenumbers so the Bessel arguments stay real dtype
    # and _j0_h0/_j1_h1 take their Cephes branch. Complex wnum_bg / complex ri
    # (QNM searches, absorbing particles) pass through untouched -- see
    # docs/conventions.md section 6.
    wnum_bg = _real_if_real(wnum_bg)
    wnum_core = _real_if_real(ri * wnum_bg)
    eta = np.complex128(kd if pol == 1 else 1.0)
    h = 2.0 * PI / nn
    w = _kress_weights(nn)

    gamma = np.sqrt(df**2 + dg**2)
    deriv = df * ddg - ddf * dg

    # ── Upper-triangle index pairs (i < j), offset d = j − i ─────────────────
    ui, uj = np.triu_indices(nn, k=1)
    w_tri = w[uj - ui]
    fi_fj = f[ui] - f[uj]
    gi_gj = g[ui] - g[uj]

    # cross products: cij[k] = (f[i]-f[j])*dg[j] - (g[i]-g[j])*df[j]
    #                cji[k] = (f[j]-f[i])*dg[i] - (g[j]-g[i])*df[i]
    cij = fi_fj * dg[uj] - gi_gj * df[uj]
    cji = -fi_fj * dg[ui] + gi_gj * df[ui]

    # ── Bessel/Hankel pairs on the unique distances ──────────────────────────
    r_tri = np.sqrt(fi_fj**2 + gi_gj**2)
    z_bg = wnum_bg * r_tri
    z_core = wnum_core * r_tri
    j0_bg, h0_bg = _j0_h0(z_bg)
    j1_bg, h1_bg = _j1_h1(z_bg)
    j0_core, h0_core = _j0_h0(z_core)
    j1_core, h1_core = _j1_h1(z_core)

    # ── Kress kernels: trapezoid term h·K plus correction K₁·W ───────────────
    single_bg = 0.25j * h * h0_bg - w_tri * j0_bg / (4.0 * PI)
    single_core = 0.25j * h * h0_core - w_tri * j0_core / (4.0 * PI)
    ratio_bg = 0.25j * h * (h1_bg / z_bg) - w_tri * (j1_bg / z_bg) / (4.0 * PI)
    ratio_core = 0.25j * h * (h1_core / z_core) - w_tri * (j1_core / z_core) / (
        4.0 * PI
    )
    double_bg = wnum_bg**2 * ratio_bg
    double_core = wnum_core**2 * ratio_core

    # ── Diagonals ────────────────────────────────────────────────────────────
    # M1/M3: the double layer's K₁ vanishes on the diagonal, so only the
    # curvature limit of K₂ survives — unchanged from the Maradudin scheme.
    # M2/M4: K₁(t,t) = −1/4π against R_0, plus the regular part of H₀ at r → 0
    # with r ≈ γ·|t − τ| (handoff §6.3). γ_E is Euler's constant, not γ.
    diag_idx = np.arange(nn)
    d_m1 = (0.5 - deriv * h / (4.0 * PI) / gamma**2).astype(complex)
    d_m3 = -(0.5 + deriv * h / (4.0 * PI) / gamma**2).astype(complex)
    d_m2 = -w[0] / (4.0 * PI) + h * (
        0.25j - (np.euler_gamma + np.log(wnum_bg * gamma / 2.0)) / (2.0 * PI)
    )
    d_m4 = eta * (
        -w[0] / (4.0 * PI)
        + h * (0.25j - (np.euler_gamma + np.log(wnum_core * gamma / 2.0)) / (2.0 * PI))
    )

    # ── Assemble ─────────────────────────────────────────────────────────────
    nt = 2 * nn
    me = np.zeros((nt, nt), dtype=complex)
    me[ui, uj] = double_bg * cij
    me[uj, ui] = double_bg * cji
    me[ui, uj + nn] = single_bg
    me[uj, ui + nn] = single_bg
    me[ui + nn, uj] = double_core * cij
    me[uj + nn, ui] = double_core * cji
    me[ui + nn, uj + nn] = eta * single_core
    me[uj + nn, ui + nn] = eta * single_core
    me[diag_idx, diag_idx] = d_m1
    me[diag_idx, diag_idx + nn] = d_m2
    me[diag_idx + nn, diag_idx] = d_m3
    me[diag_idx + nn, diag_idx + nn] = d_m4
    return me


# ---------------------------------------------------------------------------
# Fused matrix + analytic wavenumber derivative
#
# WARNING: the M half below is a deliberate duplication of assemble_matrix,
# expression for expression and in the same order, so that the two agree
# bit-for-bit. test_matrix_derivative_matches_assembly asserts that with
# np.array_equal and is the only thing guarding the copy. If you change one,
# change both.
# ---------------------------------------------------------------------------


def assemble_matrix_dwn(
    pol: int,
    nn: int,
    f: np.ndarray,
    g: np.ndarray,
    df: np.ndarray,
    dg: np.ndarray,
    ddf: np.ndarray,
    ddg: np.ndarray,
    wnum_bg: complex,
    ri: complex,
    kd: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """BIE system matrix and its analytic derivative dM/dk_bg.

    Same arithmetic as :func:`assemble_matrix` for the matrix itself, plus the
    exact term-by-term derivative with respect to the **background wavenumber**
    ``wnum_bg``. Like every primitive in this module it takes no wavelength, so
    a caller wanting dM/dλ_vac supplies the chain factor ``dk/dλ = −k/λ``
    itself — see :meth:`pysie2d.solver.BIESolver.assemble_derivative`, which is
    the one place that does.

    The derivative is what bordered Newton refinement of a quasi-normal mode
    needs (:func:`pysie2d.beyn.newton_refine`).

    Derivative identities, with ``z = k·r`` and each holding for ``J`` and
    ``H^{(1)}`` alike::

        d/dk[k²·C₁(z)/z] = k·C₀(z)          → double layer: k · (single kernel)
        d/dk[C₀(z)]      = −k·r²·(C₁(z)/z)  → single layer: −k·r² · (ratio kernel)

    so both derivative blocks are the matrix's own kernel arrays times a
    scalar, and no O(nn²) special-function evaluation is repeated. ``W`` and
    ``h`` carry no wavenumber. The M3/M4 blocks pick up a factor ``nc`` from
    ``d k_core/d k_bg``. On the diagonal, M1 and M3 carry no wavenumber (their
    derivative is exactly zero) and the M2/M4 diagonals depend on ``k`` only
    through ``ln k``, giving ``−h/(2π k)``.

    Exact only for a **non-dispersive** material: ``ri`` and ``kd`` are held
    constant through the differentiation, which is true of
    :class:`pysie2d.material.Material` — it takes no wavelength — and would be
    false for any dispersive model added later. This is a real precondition,
    not a nicety: it propagates to everything built on the derivative.

    Args:
        pol: Polarisation: 1 = p (TM), 2 = s (TE).
        nn: Number of boundary nodes, equispaced in the quadrature parameter t.
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).
        df: (nn,) df/dt.
        dg: (nn,) dg/dt.
        ddf: (nn,) d²f/dt².
        ddg: (nn,) d²g/dt².
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm), where
            λ_vac is the public **vacuum** wavelength. Build it with
            :meth:`pysie2d.material.Material.wnum_bg`; this function takes no
            wavelength, so the vacuum conversion cannot be applied twice.
            Pass a complex value for quasi-normal-mode searches.
        ri: Refractive index of the particle: nc = √(εᵣ + iεᵢ).
        kd: Dielectric constant of the particle: ε = εᵣ + iεᵢ.

    Returns:
        me: complex (2nn, 2nn) BIE system matrix, bit-identical to
            :func:`assemble_matrix` on the same arguments.
        dme: complex (2nn, 2nn) derivative dM/dk_bg.
    """
    # Demote exactly-real wavenumbers so the Bessel arguments stay real dtype
    # and _j0_h0/_j1_h1 take their Cephes branch. Complex wnum_bg / complex ri
    # (QNM searches, absorbing particles) pass through untouched -- see
    # docs/conventions.md section 6.
    wnum_bg = _real_if_real(wnum_bg)
    wnum_core = _real_if_real(ri * wnum_bg)
    eta = np.complex128(kd if pol == 1 else 1.0)
    h = 2.0 * PI / nn
    w = _kress_weights(nn)

    gamma = np.sqrt(df**2 + dg**2)
    deriv = df * ddg - ddf * dg

    # ── Upper-triangle index pairs (i < j), offset d = j − i ─────────────────
    ui, uj = np.triu_indices(nn, k=1)
    w_tri = w[uj - ui]
    fi_fj = f[ui] - f[uj]
    gi_gj = g[ui] - g[uj]

    # cross products: cij[k] = (f[i]-f[j])*dg[j] - (g[i]-g[j])*df[j]
    #                cji[k] = (f[j]-f[i])*dg[i] - (g[j]-g[i])*df[i]
    cij = fi_fj * dg[uj] - gi_gj * df[uj]
    cji = -fi_fj * dg[ui] + gi_gj * df[ui]

    # ── Bessel/Hankel pairs on the unique distances ──────────────────────────
    r_tri = np.sqrt(fi_fj**2 + gi_gj**2)
    z_bg = wnum_bg * r_tri
    z_core = wnum_core * r_tri
    j0_bg, h0_bg = _j0_h0(z_bg)
    j1_bg, h1_bg = _j1_h1(z_bg)
    j0_core, h0_core = _j0_h0(z_core)
    j1_core, h1_core = _j1_h1(z_core)

    # ── Kress kernels: trapezoid term h·K plus correction K₁·W ───────────────
    single_bg = 0.25j * h * h0_bg - w_tri * j0_bg / (4.0 * PI)
    single_core = 0.25j * h * h0_core - w_tri * j0_core / (4.0 * PI)
    ratio_bg = 0.25j * h * (h1_bg / z_bg) - w_tri * (j1_bg / z_bg) / (4.0 * PI)
    ratio_core = 0.25j * h * (h1_core / z_core) - w_tri * (j1_core / z_core) / (
        4.0 * PI
    )
    double_bg = wnum_bg**2 * ratio_bg
    double_core = wnum_core**2 * ratio_core

    # ── Diagonals ────────────────────────────────────────────────────────────
    diag_idx = np.arange(nn)
    d_m1 = (0.5 - deriv * h / (4.0 * PI) / gamma**2).astype(complex)
    d_m3 = -(0.5 + deriv * h / (4.0 * PI) / gamma**2).astype(complex)
    d_m2 = -w[0] / (4.0 * PI) + h * (
        0.25j - (np.euler_gamma + np.log(wnum_bg * gamma / 2.0)) / (2.0 * PI)
    )
    d_m4 = eta * (
        -w[0] / (4.0 * PI)
        + h * (0.25j - (np.euler_gamma + np.log(wnum_core * gamma / 2.0)) / (2.0 * PI))
    )

    # ── Derivative kernels: scalars times the arrays above ───────────────────
    r2_tri = r_tri**2
    d_single_bg = -wnum_bg * r2_tri * ratio_bg
    d_single_core = -wnum_core * r2_tri * ratio_core
    d_double_bg = wnum_bg * single_bg
    d_double_core = wnum_core * single_core

    # ── Assemble ─────────────────────────────────────────────────────────────
    nt = 2 * nn
    me = np.zeros((nt, nt), dtype=complex)
    dme = np.zeros((nt, nt), dtype=complex)
    me[ui, uj] = double_bg * cij
    me[uj, ui] = double_bg * cji
    me[ui, uj + nn] = single_bg
    me[uj, ui + nn] = single_bg
    me[ui + nn, uj] = double_core * cij
    me[uj + nn, ui] = double_core * cji
    me[ui + nn, uj + nn] = eta * single_core
    me[uj + nn, ui + nn] = eta * single_core
    me[diag_idx, diag_idx] = d_m1
    me[diag_idx, diag_idx + nn] = d_m2
    me[diag_idx + nn, diag_idx] = d_m3
    me[diag_idx + nn, diag_idx + nn] = d_m4
    dme[ui, uj] = d_double_bg * cij
    dme[uj, ui] = d_double_bg * cji
    dme[ui, uj + nn] = d_single_bg
    dme[uj, ui + nn] = d_single_bg
    dme[ui + nn, uj] = ri * d_double_core * cij
    dme[uj + nn, ui] = ri * d_double_core * cji
    dme[ui + nn, uj + nn] = eta * ri * d_single_core
    dme[uj + nn, ui + nn] = eta * ri * d_single_core
    # d_m1 and d_m3 are wavenumber-free, so dme keeps its zeros there.
    dme[diag_idx, diag_idx + nn] = -h / (2.0 * PI * wnum_bg)
    dme[diag_idx + nn, diag_idx + nn] = -eta * ri * h / (2.0 * PI * wnum_core)
    return me, dme


def assemble_cross_block(
    wnum_bg: complex,
    nn_p: int,
    f_p: np.ndarray,
    g_p: np.ndarray,
    df_p: np.ndarray,
    dg_p: np.ndarray,
    f_q: np.ndarray,
    g_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """M1 and M2 coupling blocks between two non-overlapping boundaries.

    The exterior-equation coupling of the multiparticle system
    (docs/design/multiparticle-spec.md §3.2): field node ``i`` on particle
    ``q``, source node ``j`` on particle ``p``. These are
    :func:`assemble_matrix`'s ``double_bg`` and ``single_bg`` with the
    circulant Kress weight ``W`` set to zero — **no Kress correction appears
    here**, because the two boundaries never touch, so the integrand has no
    singularity and the plain periodic trapezoid rule is already spectrally
    accurate on an analytic boundary.

    There is no M3/M4 counterpart: the interior Green function of particle
    ``p`` is confined to ``p``'s own volume, so the lower half of a cross
    block is exactly zero (§3.1). That is also why no ``eta`` is needed —
    ``eta = eps`` at ``pol = 1`` multiplies M4 alone.

    ``nn_p``, ``df_p`` and ``dg_p`` belong to the **source** particle: the
    integral is over ``p``'s boundary, so the quadrature step is
    ``h_p = 2π/nn_p`` and the normal is ``p``'s. Using the field particle's
    step instead is invisible at equal ``nn`` and a wrong answer that raises
    nothing the moment the resolutions differ.

    Args:
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm), common
            to the whole cluster. Like every primitive here this takes no
            wavelength. May be complex.
        nn_p: Number of boundary nodes of the **source** particle.
        f_p: (nn_p,) source boundary x coordinates (nm).
        g_p: (nn_p,) source boundary z coordinates (nm).
        df_p: (nn_p,) source df/dt.
        dg_p: (nn_p,) source dg/dt.
        f_q: (nn_q,) field boundary x coordinates (nm).
        g_q: (nn_q,) field boundary z coordinates (nm).

    Returns:
        ``(m1, m2)``, each complex ``(nn_q, nn_p)``: the double-layer and
        single-layer background blocks acting on φ_p and χ_p respectively.
    """
    h_p = 2.0 * PI / nn_p
    dx = f_q[:, None] - f_p[None, :]
    dz = g_q[:, None] - g_p[None, :]
    z = wnum_bg * np.sqrt(dx**2 + dz**2)
    c = dx * dg_p[None, :] - dz * df_p[None, :]
    m1 = wnum_bg**2 * (0.25j * h_p * hank1(z) / z) * c
    m2 = 0.25j * h_p * hank0(z)
    return m1, m2
