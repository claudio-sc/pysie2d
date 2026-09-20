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
    delt: float,
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
        df: (nn,) df/dt, t the quadrature parameter the nodes are equispaced in.
        dg: (nn,) dg/dt.
        delt: Trapezoid step 2π/nn in t (:attr:`pysie2d.geometry.Geometry.delt`).
        ei: complex (2nn,) BIE solution vector.

    Returns:
        amp: complex (nff,) far-field amplitude.
        angles: float (nff,) observation angles (rad), from −π to π.
    """
    angles = -PI + np.arange(nff) * 2.0 * PI / (nff - 1.0)
    amp = _far_field_at(angles, nn, wnum_bg, f, g, df, dg, delt, ei)
    return amp, angles


def _far_field_at(
    angles: np.ndarray,
    nn: int,
    wnum_bg: complex,
    f: np.ndarray,
    g: np.ndarray,
    df: np.ndarray,
    dg: np.ndarray,
    delt: float,
    ei: np.ndarray,
) -> np.ndarray:
    """Far-field amplitude at arbitrary observation angles; see :func:`far_field`."""
    se = np.sin(angles)
    co = np.cos(angles)

    # shape (nn, nff)
    arg = -1j * wnum_bg * (f[:, None] * se + g[:, None] * co)
    phi_j = ei[:nn, None]
    chi_j = ei[nn:, None]
    puto = 1j * wnum_bg * (dg[:, None] * se - df[:, None] * co) * phi_j - chi_j
    return np.sum(np.exp(arg) * puto * delt, axis=0)


def _cross_sections(
    amp: np.ndarray, amp_fwd: complex, wnum_bg: complex
) -> dict[str, float]:
    """Absolute cross-sections (nm) from a far-field amplitude.

    Implements dσ/dθ = |amp(θ)|²/(8π·k_bg), C_ext = Im[amp(π−α)]/k_bg and
    C_abs = C_ext − C_sca. Shared by the single- and multi-particle result
    classes so the two paths cannot disagree. C_ext presumes a unit-amplitude
    incident **plane wave**.

    Args:
        amp: complex (nff,) far-field amplitude on the ``[−π, π]`` inclusive
            grid of :func:`far_field`.
        amp_fwd: Amplitude at the exact forward direction π − α, evaluated off
            the grid with :func:`_far_field_at`.
        wnum_bg: Background wavenumber k_bg (rad/nm).

    Returns:
        dict with keys 'c_sca', 'c_ext', 'c_abs', in nm.
    """
    nff = len(amp)
    delthe = 2.0 * PI / (nff - 1.0)
    # amp spans [-pi, pi] inclusive, so index 0 and nff-1 are the same physical
    # direction; summing both double-counts it. Dropping the duplicate (not
    # halving both) is what makes C_sca independent of where that one grid
    # angle falls relative to the forward peak.
    c_sca = float(np.sum(np.abs(amp[:-1]) ** 2) / (8.0 * PI * wnum_bg) * delthe)
    c_ext = float(amp_fwd.imag / wnum_bg)
    return {"c_sca": c_sca, "c_ext": c_ext, "c_abs": c_ext - c_sca}


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
        df: (nn,) df/dt.
        dg: (nn,) dg/dt.

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
    delt: float,
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
        df: (nn,) df/dt, t the quadrature parameter the nodes are equispaced in.
        g: (nn,) boundary z coordinates (nm).
        dg: (nn,) dg/dt.
        delt: Trapezoid step 2π/nn in t (:attr:`pysie2d.geometry.Geometry.delt`).
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


def _representation_at(
    ei_p: np.ndarray,
    nn: int,
    f: np.ndarray,
    df: np.ndarray,
    g: np.ndarray,
    dg: np.ndarray,
    delt: float,
    wnum: complex,
    x_pts: np.ndarray,
    z_pts: np.ndarray,
) -> np.ndarray:
    """BIE representation integral over one boundary, at every point.

    The vectorised twin of the body of :func:`eval_field`'s point loop, with
    the inside/outside test removed so a caller can sum it over several
    boundaries at one wavenumber (docs/design/multiparticle-spec.md §3.5) —
    which is what a cluster's exterior field is. Classification is the
    caller's business here, and :class:`pysie2d.cluster.ClusterScatterResult`
    does it with ray casting rather than the nearest-point normal.

    :func:`eval_field` is deliberately **not** refactored onto this helper: its
    point loop keeps its memory at O(nn) where this one is O(M·nn). The
    duplication is a single four-line formula and the two are measured
    bit-identical, which the Np = 1 cluster test asserts with
    ``np.array_equal`` — so if they ever drift, a test fails.

    Args:
        ei_p: complex (2nn,) BIE solution vector for this boundary
            (φ = ``ei_p[:nn]``, χ = ``ei_p[nn:]``).
        nn: Number of boundary points.
        f: (nn,) boundary x coordinates (nm).
        df: (nn,) df/dt.
        g: (nn,) boundary z coordinates (nm).
        dg: (nn,) dg/dt.
        delt: Trapezoid step 2π/nn in t.
        wnum: Wavenumber to evaluate at (rad/nm) — k_bg outside the particle,
            k_core = nc·k_bg inside it. May be complex.
        x_pts: (M,) observation x-coordinates (nm).
        z_pts: (M,) observation z-coordinates (nm).

    Returns:
        complex (M,) contribution of this boundary at each observation point.
    """
    xmf = x_pts[:, None] - f[None, :]
    zmg = z_pts[:, None] - g[None, :]
    arg1 = wnum * np.sqrt(xmf**2 + zmg**2)
    arg2 = -dg[None, :] * xmf + df[None, :] * zmg
    integrand = (
        wnum**2 * arg2 * hank1(arg1) / arg1 * ei_p[None, :nn]
        - hank0(arg1) * ei_p[None, nn:]
    )
    return (1j / 4.0) * np.sum(integrand * delt, axis=1)
