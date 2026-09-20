"""Analytic multiple scattering from parallel circular cylinders.

The closed-form anchor for the coupled BIE: each cylinder's single-cylinder Mie
coefficients (:mod:`pysie2d.reference.mie`) are coupled by Graf's addition
theorem, giving an independent answer for a cluster of circles — not a second
path through this repository (CLAUDE.md non-negotiable 3).

**Conventions.** The algebra here runs in the *standard* polar convention
``x = ρ cos φ``, ``z = ρ sin φ``, converting to the solver's θ-from-+z
convention only at the two boundaries (``angle`` in and ``theta`` in). Forcing
the solver's convention through the translation algebra buys nothing and risks
a silent rotation by ``e^{imπ/2}``. The bridge is
docs/design/multiparticle-spec.md §3.7, verified to 1.8e-15:

    φ   = π/2 − θ                 (observation)
    α_i = deg2rad(angle) − π/2    (incidence)
    s_n = −c_n                    (c_n = mie.an for TM, mie.bn for TE)
    amp_solver(θ) = −4i · S(π/2 − θ)
"""

import numpy as np
from scipy.special import hankel1

from .mie import _nmax, an, bn

PI = np.pi

# Above this the Graf translation matrix is ill-conditioned: H_M(k·d) grows
# superexponentially past M ~ k·d while s_n decays superexponentially past
# n ~ x, so a generously truncated system returns garbage rather than a more
# accurate answer. Measured (spec §3.8): the error is at round-off through
# |H_M(kd)| = 4.9e4 (M = 22) and has degraded to 2.3e-12 by 4.6e6 (M = 25).
CONDITION_LIMIT = 1.0e6


def scattering_amplitude(
    pol: int,
    radii,
    centres,
    m_rel,
    wnum_bg: complex,
    angle: float,
    theta: np.ndarray,
    n_max: int | None = None,
) -> np.ndarray:
    """Far-field amplitude of a cluster of circles, in the solver's units.

    Args:
        pol: 1 = TM (a_n), 2 = TE (b_n), matching the solver's codes.
        radii: (J,) cylinder radii (nm).
        centres: (J, 2) centres as (x, z) in nm — the solver's coordinates.
        m_rel: (J,) relative refractive indices (``Material.nc``).
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm). This is
            a primitive and takes no wavelength (conventions §2). May be
            complex.
        angle: Plane-wave incidence angle (degrees), the solver's ``angle``.
        theta: (nff,) observation angles (rad), the solver's far-field angles.
        n_max: Truncation order M. Default: the Wiscombe criterion on the
            largest size parameter.

    Returns:
        complex (nff,) amplitude, directly comparable to
        ``ClusterScatterResult.far_field``.

    Raises:
        ValueError: If the truncation order is too high for the closest centre
            separation; see ``CONDITION_LIMIT``.
    """
    k = wnum_bg
    R = np.asarray(centres, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n_cyl = len(radii)
    alpha_i = np.deg2rad(angle) - PI / 2.0
    khat = np.array([np.cos(alpha_i), np.sin(alpha_i)])

    x = k * radii
    if n_max is None:
        n_max = _nmax(np.max(np.abs(x)))
    n = np.arange(-n_max, n_max + 1)
    nd = len(n)

    d_min = min(
        float(np.hypot(*(R[j] - R[i])))
        for i in range(n_cyl)
        for j in range(i + 1, n_cyl)
    )
    h_max = float(np.abs(hankel1(n_max, np.abs(k) * d_min)))
    if h_max > CONDITION_LIMIT:
        raise ValueError(
            f"truncation order {n_max} gives |H_M(k*d_min)| = {h_max:.3g}, "
            f"above the conditioning limit {CONDITION_LIMIT:.0e}: these "
            f"cylinders are too close for the addition theorem at the order "
            f"their size parameters demand"
        )

    # s_n = -c_n, forced by unitarity against mie.efficiencies (§3.7).
    coeff = an if pol == 1 else bn
    s = [-coeff(np.abs(n), x[j], m_rel[j]) for j in range(n_cyl)]
    inc = [
        np.exp(1j * k * (R[j] @ khat)) * (1j**n) * np.exp(-1j * n * alpha_i)
        for j in range(n_cyl)
    ]

    mat = np.zeros((n_cyl * nd, n_cyl * nd), dtype=complex)
    rhs = np.zeros(n_cyl * nd, dtype=complex)
    for el in range(n_cyl):
        blk = slice(el * nd, (el + 1) * nd)
        mat[blk, blk] += np.eye(nd)
        rhs[blk] = s[el] * inc[el]
        for j in range(n_cyl):
            if j == el:
                continue
            # (d, φ) are the polar coordinates of the vector FROM centre j TO
            # centre l. Check it at ρ_l → 0: only m = 0 survives, and
            # H_n(kd)·e^{inφ} must reproduce the left-hand side (§3.8).
            dvec = R[el] - R[j]
            d = float(np.hypot(*dvec))
            phi_jl = float(np.arctan2(dvec[1], dvec[0]))
            nm = n[None, :] - n[:, None]  # nm[m, n] = n − m
            t = hankel1(nm, k * d) * np.exp(1j * nm * phi_jl)
            mat[blk, j * nd : (j + 1) * nd] -= s[el][:, None] * t
    amp_coef = np.linalg.solve(mat, rhs)

    phi = PI / 2.0 - np.asarray(theta, dtype=float)
    rhat = np.stack([np.cos(phi), np.sin(phi)])
    total = np.zeros(len(phi), dtype=complex)
    for j in range(n_cyl):
        a_j = amp_coef[j * nd : (j + 1) * nd]
        phase = np.exp(-1j * k * (R[j][:, None] * rhat).sum(0))
        total += phase * (
            a_j[:, None]
            * ((-1j) ** n)[:, None]
            * np.exp(1j * n[:, None] * phi[None, :])
        ).sum(0)
    return -4j * total
