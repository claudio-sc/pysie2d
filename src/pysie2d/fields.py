"""Field evaluation: far-field amplitude and arbitrary-point near fields."""

import numpy as np

from .kernels import _real_if_real, hank0, hank1

PI = np.pi


# ---------------------------------------------------------------------------
# Far-field amplitude  (subroutine efi)
# ---------------------------------------------------------------------------


def far_field(
    nn: int,
    nff: int,
    wnum_bg: complex,
    f: np.ndarray,
    g: np.ndarray,
    df: np.ndarray,
    dg: np.ndarray,
    delt: float | np.ndarray,
    ei: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 2-D far-field scattering amplitude.

    Args:
        nn: Number of boundary points.
        nff: Number of far-field angles.
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm), where
            λ_vac is the public **vacuum** wavelength. Build it with
            :meth:`pysie2d.material.Material.wnum_bg`; this function takes no
            wavelength, so the vacuum conversion cannot be applied twice.
            Pass a complex value for quasi-normal-mode searches.
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).
        df: (nn,) first derivative of f w.r.t. θ.
        dg: (nn,) first derivative of g w.r.t. θ.
        delt: Quadrature θ-step (scalar or per-point array).
        ei: complex (2nn,) BIE solution vector.

    Returns:
        amp: complex (nff,) far-field amplitude.
        angles: float (nff,) observation angles (rad), from −π to π.
    """
    angles = -PI + np.arange(nff) * 2.0 * PI / (nff - 1.0)
    se = np.sin(angles)
    co = np.cos(angles)

    # shape (nn, nff)
    arg = -1j * wnum_bg * (f[:, None] * se + g[:, None] * co)
    phi_j = ei[:nn, None]
    chi_j = ei[nn:, None]
    puto = 1j * wnum_bg * (dg[:, None] * se - df[:, None] * co) * phi_j - chi_j
    # delt may be a scalar or a per-point (nn,) array
    delt_col = np.reshape(delt, (-1, 1)) if np.ndim(delt) > 0 else delt
    amp = np.sum(np.exp(arg) * puto * delt_col, axis=0)

    return amp, angles


# ---------------------------------------------------------------------------
# Inside/outside test  (subroutine eicero)
# ---------------------------------------------------------------------------


def _is_outside(
    x1: float,
    x3: float,
    f: np.ndarray,
    g: np.ndarray,
    df: np.ndarray,
    dg: np.ndarray,
) -> bool:
    """Return True if (x1, x3) is outside the particle.

    Uses the sign of the dot product between the displacement vector and the
    outward normal at the nearest boundary point. This nearest-neighbour
    heuristic is reliable for convex and mildly star-shaped boundaries, but
    unreliable for extreme concave superformula shapes.

    Args:
        x1: Observation x-coordinate (nm).
        x3: Observation z-coordinate (nm).
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).
        df: (nn,) first derivative of f w.r.t. θ.
        dg: (nn,) first derivative of g w.r.t. θ.

    Returns:
        True if the point lies outside the particle.
    """
    imin = np.argmin((x1 - f) ** 2 + (x3 - g) ** 2)
    norm = -(x1 - f[imin]) * dg[imin] + (x3 - g[imin]) * df[imin]
    return norm > 0.0


# ---------------------------------------------------------------------------
# Scattered field at arbitrary points
# ---------------------------------------------------------------------------


def eval_field(
    ei: np.ndarray,
    nn: int,
    f: np.ndarray,
    df: np.ndarray,
    g: np.ndarray,
    dg: np.ndarray,
    delt: float | np.ndarray,
    wnum_bg: complex,
    x_pts: np.ndarray,
    z_pts: np.ndarray,
    ri: complex | None = None,
) -> np.ndarray:
    """Evaluate the field at arbitrary (x, z) points.

    Uses the BIE representation formula (Huygens principle) for exterior points.
    For interior points, the same formula is used with the particle wavenumber
    wnum_core = ri * wnum_bg when ri is provided; otherwise interior points
    return 0+0j.

    Never evaluate on or very near the boundary: the representation-formula
    integrand is near-singular there and degrades within roughly 2–3 boundary-
    point spacings (~2π·rad/nn) of the surface. Keep observation points at
    least ~5 spacings away.

    Args:
        ei: complex (2nn,) BIE solution vector (φ = ei[:nn], χ = ei[nn:]).
        nn: Number of boundary points.
        f: (nn,) boundary x coordinates (nm).
        df: (nn,) first derivative of f w.r.t. θ.
        g: (nn,) boundary z coordinates (nm).
        dg: (nn,) first derivative of g w.r.t. θ.
        delt: Quadrature θ-step.
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm), where
            λ_vac is the public **vacuum** wavelength. Build it with
            :meth:`pysie2d.material.Material.wnum_bg`; this function takes no
            wavelength, so the vacuum conversion cannot be applied twice.
            Pass a complex value for quasi-normal-mode searches.
        x_pts: (M,) observation x-coordinates (nm).
        z_pts: (M,) observation z-coordinates (nm).
        ri: Particle refractive index (relative to background). When provided,
            interior points are evaluated using wnum_core = ri * wnum_bg. When None
            (default), interior points return 0+0j.

    Returns:
        field: complex (M,) field E_y at each observation point (exterior
            scattered field outside, interior field inside when ri is given).
    """
    x_pts = np.asarray(x_pts)
    z_pts = np.asarray(z_pts)
    n_pts = len(x_pts)
    field = np.zeros(n_pts, dtype=complex)

    # Demote exactly-real wavenumbers so exterior points (and interior points of
    # a non-absorbing particle) reach the Cephes fast path in hank0/hank1.
    wnum_bg = _real_if_real(wnum_bg)
    wnum_core = _real_if_real(ri * wnum_bg) if ri is not None else None

    for j in range(n_pts):
        outside = _is_outside(x_pts[j], z_pts[j], f, g, df, dg)
        if not outside and wnum_core is None:
            continue
        xmf = x_pts[j] - f
        zmg = z_pts[j] - g
        dist = np.sqrt(xmf**2 + zmg**2)
        arg2 = -dg * xmf + df * zmg
        k = wnum_bg if outside else wnum_core
        arg1 = k * dist
        sum_h = np.sum(
            (k**2 * arg2 * hank1(arg1) / arg1 * ei[:nn] - hank0(arg1) * ei[nn:]) * delt
        )
        field[j] = (1j / 4.0) * sum_h

    return field
