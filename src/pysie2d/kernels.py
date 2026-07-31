"""BIE matrix assembly and Hankel-function helpers — the numerical core.

Implements the boundary integral equation (BIE) for 2-D electromagnetic
scattering from a cylinder in a homogeneous background. All functions accept
complex wavenumbers, making this module usable for both driven (real λ) and
quasi-normal-mode (complex λ) problems.

Convention:
    The BIE solution vector ``ei`` has shape (2*nn,):
        ei[:nn]  — φ  : electric-field values on the boundary
        ei[nn:]  — χ  : normal-derivative values on the boundary

Public API:
    hank0, hank1, cbesh: Hankel-function wrappers.
    assemble_matrix: fast vectorised 2nn × 2nn BIE system matrix M(λ).
    assemble_matrix_reference: slow loop-based assembly, kept only as the
        rounding-accurate truth anchor for the parity test.
"""

import numpy as np
from scipy.special import hankel1, j0, j1, y0, y1

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
# BIE matrix  (verbatim translation of subroutine matm)
#
# WARNING: every coefficient, sign, and diagonal term below is fixed by the
# original Fortran source.  Do not alter any expression without verifying
# against the Fortran or the published derivation.
# ---------------------------------------------------------------------------


def assemble_matrix_reference(
    pol: int,
    nn: int,
    f: np.ndarray,
    g: np.ndarray,
    df: np.ndarray,
    dg: np.ndarray,
    ddf: np.ndarray,
    ddg: np.ndarray,
    delt: float | np.ndarray,
    wnum_bg: complex,
    ri: complex,
    kd: complex,
) -> np.ndarray:
    """Assemble the 2nn × 2nn BIE system matrix (slow reference loop).

    Verbatim translation of subroutine matm from the original Fortran source.
    Accepts complex *wnum_bg* for quasi-normal-mode (complex-λ) computations.

    This slow, loop-based routine is kept **only** as the truth anchor for the
    parity test against :func:`assemble_matrix`: the two implement identical
    arithmetic in a different loop order, so they must agree to rounding. It is
    not used on any hot path.

    Args:
        pol: Polarisation: 1 = p (TM), 2 = s (TE).
        nn: Number of boundary quadrature points.
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).
        df: (nn,) first derivative of f w.r.t. the parameterisation variable θ.
        dg: (nn,) first derivative of g w.r.t. θ.
        ddf: (nn,) second derivative of f w.r.t. θ.
        ddg: (nn,) second derivative of g w.r.t. θ.
        delt: Quadrature θ-step. Scalar for uniform-theta; per-point array for
            arc-length sampling.
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm), where
            λ_vac is the public **vacuum** wavelength. Build it with
            :meth:`pysie2d.material.Material.wnum_bg`; this function takes no
            wavelength, so the vacuum conversion cannot be applied twice.
            Pass a complex value for quasi-normal-mode searches.
        ri: Refractive index of the particle: nc = √(εᵣ + iεᵢ).
        kd: Dielectric constant of the particle: ε = εᵣ + iεᵢ.

    Returns:
        me: complex (2nn, 2nn) BIE system matrix.

    Matrix block structure:
        The 2nn × 2nn matrix is partitioned into four nn × nn blocks:
            M1  (rows 0:nn,    cols 0:nn)    φ–φ coupling,  background wnum_bg
            M2  (rows 0:nn,    cols nn:2nn)  φ–χ coupling,  background wnum_bg
            M3  (rows nn:2nn,  cols 0:nn)    χ–φ coupling,  particle wnum_core
            M4  (rows nn:2nn,  cols nn:2nn)  χ–χ coupling,  particle wnum_core
    """
    nt = 2 * nn
    me = np.zeros((nt, nt), dtype=complex)
    e = np.e
    wnum_core = ri * wnum_bg
    eta = kd if pol == 1 else complex(1.0)

    # Allow delt to be a scalar (uniform theta) or a per-point array (arc-length).
    # All quadrature weights are indexed by the source column j.
    delt = np.full(nn, delt) if np.isscalar(delt) else np.asarray(delt)
    c1 = 0.25j * wnum_bg**2 * delt  # (nn,)
    c2 = 0.25j * delt  # (nn,)
    c3 = 0.25j * wnum_core**2 * delt  # (nn,)
    depi4 = delt / (4.0 * PI)  # (nn,)

    gamma = np.sqrt(df**2 + dg**2)
    deriv = df * ddg - ddf * dg

    # ---- Block M1 (φ–φ) -----------------------------------------------
    for i in range(nn):
        for j in range(i + 1, nn):
            r2 = (f[i] - f[j]) ** 2 + (g[i] - g[j]) ** 2
            arg1 = wnum_bg * np.sqrt(r2)
            arg2 = (f[i] - f[j]) * dg[j] - (g[i] - g[j]) * df[j]
            arg2p = (f[j] - f[i]) * dg[i] - (g[j] - g[i]) * df[i]
            h1a1 = hank1(arg1) / arg1
            me[i, j] = c1[j] * arg2 * h1a1
            me[j, i] = c1[i] * arg2p * h1a1
    for i in range(nn):
        me[i, i] = 0.5 - deriv[i] * depi4[i] / gamma[i] ** 2

    # ---- Block M2 (φ–χ) -----------------------------------------------
    for i in range(nn):
        for j in range(i + 1, nn):
            r2 = (f[i] - f[j]) ** 2 + (g[i] - g[j]) ** 2
            h0 = hank0(wnum_bg * np.sqrt(r2))
            me[i, j + nn] = c2[j] * h0
            me[j, i + nn] = c2[i] * h0
    for i in range(nn):
        me[i, i + nn] = c2[i] * hank0(wnum_bg * delt[i] / (2.0 * e) * gamma[i])

    # ---- Block M3 (χ–φ) -----------------------------------------------
    for i in range(nn):
        for j in range(i + 1, nn):
            r2 = (f[i] - f[j]) ** 2 + (g[i] - g[j]) ** 2
            arg1c = wnum_core * np.sqrt(r2)
            arg2 = (f[i] - f[j]) * dg[j] - (g[i] - g[j]) * df[j]
            arg2p = (f[j] - f[i]) * dg[i] - (g[j] - g[i]) * df[i]
            h1c = cbesh(arg1c, 1) / arg1c
            me[i + nn, j] = c3[j] * arg2 * h1c
            me[j + nn, i] = c3[i] * arg2p * h1c
    for i in range(nn):
        me[i + nn, i] = -(0.5 + deriv[i] * depi4[i] / gamma[i] ** 2)

    # ---- Block M4 (χ–χ) -----------------------------------------------
    for i in range(nn):
        for j in range(i + 1, nn):
            r2 = (f[i] - f[j]) ** 2 + (g[i] - g[j]) ** 2
            h0c = cbesh(wnum_core * np.sqrt(r2), 0)
            me[i + nn, j + nn] = eta * c2[j] * h0c
            me[j + nn, i + nn] = eta * c2[i] * h0c
    for i in range(nn):
        me[i + nn, i + nn] = (
            c2[i] * eta * cbesh(wnum_core * delt[i] / (2.0 * e) * gamma[i], 0)
        )

    return me


# ---------------------------------------------------------------------------
# Fast vectorised BIE matrix assembly
# ---------------------------------------------------------------------------


def assemble_matrix(
    pol: int,
    nn: int,
    f: np.ndarray,
    g: np.ndarray,
    df: np.ndarray,
    dg: np.ndarray,
    ddf: np.ndarray,
    ddg: np.ndarray,
    delt: float | np.ndarray,
    wnum_bg: complex,
    ri: complex,
    kd: complex,
) -> np.ndarray:
    """Vectorised BIE system matrix assembly (drop-in for the reference loop).

    Key optimisations vs the original Python double-loop:

    1. Upper-triangle Hankel evaluation: r[i,j] = r[j,i], so Hankel
       functions are evaluated on only nn*(nn-1)/2 unique distances instead
       of nn², halving the dominant cost for complex wnum_bg (QNM mode).
    2. Vectorised scipy.special.hankel1: a single C-level call per Hankel
       order replaces nn*(nn-1)/2 scalar Python calls per block.
    3. Both (i,j) and (j,i) matrix entries are filled from the same Hankel
       value, mirroring the original Fortran loop exactly.

    At nn ≤ 400 this pure-NumPy path solves in milliseconds. Parameters and
    return value are identical to :func:`assemble_matrix_reference`, against
    which it is validated to rounding by the parity test.

    Args:
        pol: Polarisation: 1 = p (TM), 2 = s (TE).
        nn: Number of boundary quadrature points.
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).
        df: (nn,) first derivative of f w.r.t. θ.
        dg: (nn,) first derivative of g w.r.t. θ.
        ddf: (nn,) second derivative of f w.r.t. θ.
        ddg: (nn,) second derivative of g w.r.t. θ.
        delt: Quadrature θ-step. Scalar for uniform-theta; per-point array for
            arc-length sampling.
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
    e = np.e
    # Demote exactly-real wavenumbers so the Hankel arguments stay real dtype
    # and hank0/hank1 can take their Cephes branch. Complex wnum_bg / complex ri
    # (QNM searches, absorbing particles) pass through untouched -- see
    # docs/conventions.md section 6.
    wnum_bg = _real_if_real(wnum_bg)
    wnum_core = _real_if_real(ri * wnum_bg)
    eta = np.complex128(kd if pol == 1 else 1.0)

    delt = np.full(nn, delt) if np.isscalar(delt) else np.asarray(delt)
    c1 = 0.25j * wnum_bg**2 * delt  # (nn,)
    c2 = 0.25j * delt  # (nn,)
    c3 = 0.25j * wnum_core**2 * delt  # (nn,)
    depi4 = delt / (4.0 * PI)  # (nn,)

    gamma = np.sqrt(df**2 + dg**2)
    deriv = df * ddg - ddf * dg

    # ── Upper-triangle index pairs (i < j) ───────────────────────────────────
    ui, uj = np.triu_indices(nn, k=1)  # each shape (nn*(nn-1)//2,)
    fi_fj = f[ui] - f[uj]  # f[i] - f[j]  for each pair
    gi_gj = g[ui] - g[uj]  # g[i] - g[j]

    # cross products: cij[k] = (f[i]-f[j])*dg[j] - (g[i]-g[j])*df[j]
    #                cji[k] = (f[j]-f[i])*dg[i] - (g[j]-g[i])*df[i]
    cij = fi_fj * dg[uj] - gi_gj * df[uj]
    cji = -fi_fj * dg[ui] + gi_gj * df[ui]

    # ── Vectorised Hankel on the unique distances (half the nn² evaluations) ─
    r_tri = np.sqrt(fi_fj**2 + gi_gj**2)  # (n_pairs,)
    arg_wnum = wnum_bg * r_tri
    arg_wnum_core = wnum_core * r_tri
    h0w = hank0(arg_wnum)
    h1w = hank1(arg_wnum) / arg_wnum  # H_1(z)/z
    h0w1 = hank0(arg_wnum_core)
    h1w1 = hank1(arg_wnum_core) / arg_wnum_core

    # ── Exact diagonal entries ────────────────────────────────────────────────
    # The hankel1(0, wnum_bg*delt/(2e)*gamma) terms (e = np.e, Euler's number) are
    # the
    # analytic handling of the logarithmic Green-function singularity — not a
    # typo. Do not "clean up".
    diag_idx = np.arange(nn)
    d_m1 = (0.5 - deriv * depi4 / gamma**2).astype(complex)
    d_m2 = c2 * hank0(wnum_bg * delt / (2.0 * e) * gamma)
    d_m3 = -(0.5 + deriv * depi4 / gamma**2).astype(complex)
    d_m4 = c2 * eta * hank0(wnum_core * delt / (2.0 * e) * gamma)

    # ── Assemble ──────────────────────────────────────────────────────────────
    nt = 2 * nn
    me = np.zeros((nt, nt), dtype=complex)
    # Off-diagonal: fill both (i,j) and (j,i) from the same Hankel value
    me[ui, uj] = c1[uj] * cij * h1w
    me[uj, ui] = c1[ui] * cji * h1w
    me[ui, uj + nn] = c2[uj] * h0w
    me[uj, ui + nn] = c2[ui] * h0w
    me[ui + nn, uj] = c3[uj] * cij * h1w1
    me[uj + nn, ui] = c3[ui] * cji * h1w1
    me[ui + nn, uj + nn] = eta * c2[uj] * h0w1
    me[uj + nn, ui + nn] = eta * c2[ui] * h0w1
    # Diagonal
    me[diag_idx, diag_idx] = d_m1
    me[diag_idx, diag_idx + nn] = d_m2
    me[diag_idx + nn, diag_idx] = d_m3
    me[diag_idx + nn, diag_idx + nn] = d_m4
    return me
