"""Excitation right-hand sides for the BIE.

Every source fills the BIE right-hand-side vector ``ei`` of shape (2*nn,).
Following the plane-wave convention, only the φ half ``ei[:nn]`` is populated;
the χ half ``ei[nn:]`` stays zero. Both the plane-wave and line-dipole sources
share this layout.
"""

import numpy as np

from .kernels import hank0

# ---------------------------------------------------------------------------
# Incident plane-wave RHS  (subroutine ein)
# ---------------------------------------------------------------------------


def plane_wave_rhs(
    nn: int,
    titad: float,
    wnum_bg: complex,
    f: np.ndarray,
    g: np.ndarray,
) -> np.ndarray:
    """Right-hand-side vector for an s-polarised plane wave.

    Fills only the φ half of the solution vector; the χ half stays zero.

    Args:
        nn: Number of boundary points.
        titad: Incidence angle (degrees).
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm), where
            λ_vac is the public **vacuum** wavelength. Build it with
            :meth:`pysie2d.material.Material.wnum_bg`; this function takes no
            wavelength, so the vacuum conversion cannot be applied twice.
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).

    Returns:
        ei: complex (2*nn,) with ei[:nn] = exp(i k_bg (f sinθ − g cosθ)) and
            ei[nn:] = 0.
    """
    theta = np.deg2rad(titad)
    fb = f * np.sin(theta) - g * np.cos(theta)
    ei = np.zeros(2 * nn, dtype=complex)
    ei[:nn] = np.exp(1j * wnum_bg * fb)
    return ei


# ---------------------------------------------------------------------------
# Incident line-dipole (2-D point source) RHS
# ---------------------------------------------------------------------------


def _point_inside(x_s: float, z_s: float, f: np.ndarray, g: np.ndarray) -> bool:
    """Return True if (x_s, z_s) lies inside the closed boundary polygon.

    Ray-casting (even-odd) test on the ordered boundary vertices (f, g). This
    uses only the boundary coordinates — unlike :func:`pysie2d.fields._is_outside`
    it needs no tangent vectors, so it fits the line-dipole RHS signature, and
    it stays correct for concave superformula shapes where the nearest-point
    normal heuristic can misfire.

    Args:
        x_s: Source x-coordinate (nm).
        z_s: Source z-coordinate (nm).
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).

    Returns:
        True if the source lies inside the particle.
    """
    fj = np.roll(f, 1)
    gj = np.roll(g, 1)
    # Edges straddling the horizontal ray z = z_s cast to the right of x_s.
    straddles = (g > z_s) != (gj > z_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = (fj - f) * (z_s - g) / (gj - g) + f
    crossings = straddles & (x_s < x_cross)
    return bool(np.count_nonzero(crossings) % 2 == 1)


def line_dipole_rhs(
    nn: int,
    wnum_bg: complex,
    f: np.ndarray,
    g: np.ndarray,
    x_s: float,
    z_s: float,
) -> np.ndarray:
    """Right-hand side for a line-dipole (2-D point) source at (x_s, z_s).

    The source radiates the background field
    ``ψ_inc(r) = (i/4)·H₀^{(1)}(k_bg·|r − r_s|)``, i.e. the 2-D homogeneous-
    background Green function ``g₀``. Mirrors :func:`plane_wave_rhs`: only the
    φ half ``ei[:nn]`` is populated, the χ half ``ei[nn:]`` stays zero.

    Args:
        nn: Number of boundary points.
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm), where
            λ_vac is the public **vacuum** wavelength. Build it with
            :meth:`pysie2d.material.Material.wnum_bg`; this function takes no
            wavelength, so the vacuum conversion cannot be applied twice.
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).
        x_s: Source x-coordinate (nm).
        z_s: Source z-coordinate (nm).

    Returns:
        ei: complex (2*nn,) with ei[:nn] = 0.25j·H₀^{(1)}(k_bg·dist) and
            ei[nn:] = 0, where ``dist`` is the source-to-boundary distance.

    Raises:
        ValueError: If the source is inside the particle, or within five
            boundary-point spacings of the surface (where the incident field
            and the representation-formula integrand are near-singular).
    """
    dist = np.sqrt((f - x_s) ** 2 + (g - z_s) ** 2)

    # Mean spacing between consecutive boundary points (≈ 2π·rad/nn for a
    # circle); the near-boundary exclusion zone is five such spacings.
    seg = np.sqrt(np.diff(f, append=f[0]) ** 2 + np.diff(g, append=g[0]) ** 2)
    spacing = float(np.mean(seg))
    if _point_inside(x_s, z_s, f, g):
        raise ValueError(f"line dipole source ({x_s}, {z_s}) is inside the particle")
    if float(dist.min()) < 5.0 * spacing:
        raise ValueError(
            f"line dipole source ({x_s}, {z_s}) is within 5 boundary spacings "
            f"({5.0 * spacing:.3g} nm) of the surface; move it farther out"
        )

    ei = np.zeros(2 * nn, dtype=complex)
    ei[:nn] = 0.25j * hank0(wnum_bg * dist)
    return ei
