"""Field evaluation: far-field amplitude and arbitrary-point near fields."""

import math
from collections.abc import Callable

import numpy as np

from .kernels import _real_if_real, hank0, hank1
from .layered import (
    TAIL_DECAY,
    TAIL_EFOLDS_PER_PANEL,
    TAIL_PANEL_POINTS,
    TAIL_PERIODS_PER_PANEL,
    NoFarFieldError,
    Reflection,
    SommerfeldPath,
    StackReflection,
    Transmission,
    _stack_poles,
    alpha,
    fresnel_r,
    stack_layer_factor,
    stack_r,
)

PI = np.pi

# Flux breakpoints at |Re s| ± 10^m·|Im s|, m = 0…6, for every singularity s.
# Grading only to 10³ widths is not enough near a weakly damped pole: silver +
# 1e-4i TM was off by 1.0e-6 there and 1.6e-12 with six decades; nine decades
# changed nothing (studies/transmitted t2c).
FLUX_WIDTH_DECADES = 6


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
# Far field over a half-space substrate
# ---------------------------------------------------------------------------

# One boundary's data, in the argument order of ``_far_field_at``:
# (f, g, df, dg, delt, ei) with ``ei`` that boundary's (2nn,) sub-vector.
_Boundary = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]


def _upward_amplitude(
    angles: np.ndarray,
    boundaries: list[_Boundary],
    wnum_bg: complex,
    eps_rel: complex | None,
    pol: int,
    z_int: float,
    *,
    r: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Upward far-field amplitude ``a_up(θ)`` of particles over a half-space.

    ``a_up(θ) = A(θ) + r(k sin θ)·A(π − θ; g → g − 2·z_int)``, about the
    **origin**, with ``A`` the free-space amplitude of the same boundary data and
    ``r`` the real-axis Fresnel coefficient. The image term is the reflected
    wave's source seen through the interface: it shifts only the z coordinates,
    not ``dg``, and needs no extra phase factor (spec §2a). θ is from +z, as in
    :func:`far_field`, and is upward for ``|θ| < π/2``.

    Args:
        angles: (M,) upward angles ``|θ| ≤ π/2`` (rad).
        boundaries: ``(f, g, df, dg, delt, ei)`` per particle.
        wnum_bg: Background wavenumber.
        eps_rel: Substrate permittivity relative to the cover, or ``None``.
        pol: 2 = TE, 1 = TM.
        z_int: Interface height (nm).
        r: ``q → R(q)`` replacing :func:`fresnel_r` (a stack, referred to
            ``z_int = z_top``); ``eps_rel`` is then unused here.

    Returns:
        complex (M,) amplitude.
    """
    direct = np.zeros(len(angles), dtype=complex)
    image = np.zeros(len(angles), dtype=complex)
    for f, g, df, dg, delt, ei in boundaries:
        nn = f.size
        direct += _far_field_at(angles, nn, wnum_bg, f, g, df, dg, delt, ei)
        image += _far_field_at(
            PI - angles, nn, wnum_bg, f, g - 2.0 * z_int, df, dg, delt, ei
        )
    if r is not None:
        return direct + r(wnum_bg * np.sin(angles)) * image
    if eps_rel is not None and eps_rel == 1.0:
        return direct  # no interface: r ≡ 0, and 0/0 at grazing is avoided
    rr = fresnel_r(wnum_bg * np.sin(angles), wnum_bg, eps_rel, pol)
    return direct + rr * image


def _upward_far_field(
    n_angles: int,
    boundaries: list[_Boundary],
    wnum_bg: complex,
    eps_rel: complex | None,
    pol: int,
    z_int: float,
    *,
    r: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``a_up`` on a uniform grid over ``[−π/2, π/2]`` inclusive."""
    angles = np.linspace(-0.5 * PI, 0.5 * PI, n_angles)
    amp = _upward_amplitude(angles, boundaries, wnum_bg, eps_rel, pol, z_int, r=r)
    return amp, angles


def _upward_cross_sections(
    n_angles: int,
    boundaries: list[_Boundary],
    wnum_bg: complex,
    eps_rel: complex | None,
    pol: int,
    z_int: float,
    *,
    r: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    """``σ_sca,up`` and ``σ_abs`` (nm) of particles over a half-space.

    ``σ_sca,up = (1/(8πk))·∫|a_up|² dθ`` over ``[−π/2, π/2]`` by Gauss–Legendre.
    It excludes the specularly reflected incident beam (background, not
    scattering) and the power sent into the substrate, so it is incomplete by
    construction. ``σ_abs`` comes from the boundary flux and needs no far field,
    so it is exact with a substrate.

    When ``0 < Re ε_rel < 1`` the substrate is optically thinner than the cover,
    and ``r(k sin θ)`` has a √ kink at the cover's critical angle
    ``±θ_c = ±asin √(Re ε_rel)``. Plain Gauss–Legendre then converges only
    algebraically: at 500 nodes it is off by up to 1.3e-3 (TM, ε_sub = 1.5,
    n_clad = 1.33). The integral is therefore split at ±θ_c, and each panel uses
    the sine map ``θ = c + h·sin(πu/2)``, which removes the √ endpoint
    behaviour, so 500 nodes reach 1.2e-13. Weak loss moves the branch point
    just off the real axis but leaves the near-kink, so the split also applies
    to a lossy substrate (ε_sub = 1 + 1e-3i: 6.5e-4 unsplit, 1.2e-13 split).
    Every other substrate keeps the v1.0 rule bit for bit.

    Args:
        n_angles: Gauss–Legendre nodes for the angular integral, shared between
            the panels in proportion to their width when the integral is split.
        boundaries: ``(f, g, df, dg, delt, ei)`` per particle.
        wnum_bg: Background wavenumber.
        eps_rel: Substrate permittivity relative to the cover, or ``None``.
        pol: 2 = TE, 1 = TM.
        z_int: Interface height (nm).
        r: ``q → R(q)`` of a stack; ``eps_rel`` is then the **bottom**
            permittivity, whose critical angle is R's one kink (slab-spec §2.7).

    Returns:
        ``{"c_sca_up", "c_abs"}`` in nm.
    """
    theta, wt = _upward_rule(n_angles, eps_rel)
    amp = _upward_amplitude(theta, boundaries, wnum_bg, eps_rel, pol, z_int, r=r)
    c_sca_up = float(np.sum(wt * np.abs(amp) ** 2) / (8.0 * PI * np.real(wnum_bg)))
    return {"c_sca_up": c_sca_up, "c_abs": _absorbed_cross_section(boundaries, wnum_bg)}


def _upward_rule(
    n_angles: int, eps_rel: complex | None
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights on ``[−π/2, π/2]`` of :func:`_upward_cross_sections`."""
    if eps_rel is not None and 0.0 < eps_rel.real < 1.0:
        theta_c = float(np.arcsin(np.sqrt(eps_rel.real)))
        edges = [-0.5 * PI, -theta_c, theta_c, 0.5 * PI]
        angles, weights = [], []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            x, w = np.polynomial.legendre.leggauss(math.ceil(n_angles * (hi - lo) / PI))
            c, h = 0.5 * (hi + lo), 0.5 * (hi - lo)
            angles.append(c + h * np.sin(0.5 * PI * x))
            weights.append(h * 0.5 * PI * np.cos(0.5 * PI * x) * w)
        theta, wt = np.concatenate(angles), np.concatenate(weights)
    else:
        x, w = np.polynomial.legendre.leggauss(n_angles)
        theta, wt = 0.5 * PI * x, 0.5 * PI * w
    return theta, wt


def _spectrum(
    q: np.ndarray, a1: np.ndarray, boundaries: list[_Boundary], z_top: float
) -> np.ndarray:
    """Plane-wave spectrum ``A(q, α₁)`` of the boundary sources (spec v1.1 §2).

    ``A = Σ_j delt·e^{−i q f_j + i α₁ (g_j − z_top)}·[i(q·dg_j + α₁·df_j)·φ_j − χ_j]``,
    which is :func:`_far_field_at` at ``q = k sin θ``, ``α₁ = −k cos θ``. The
    ``g − z_top`` shift keeps the exponent bounded for evanescent ``α₁ = +i|…|``,
    which would overflow about the origin.

    Args:
        q: (M,) transverse wavenumbers.
        a1: (M,) cover ``α₁(q)`` on the code's sheet.
        boundaries: ``(f, g, df, dg, delt, ei)`` per particle.
        z_top: Phase reference height (nm).

    Returns:
        complex (M,) spectrum.
    """
    out = np.zeros(np.shape(q), dtype=complex)
    for f, g, df, dg, delt, ei in boundaries:
        nn = f.size
        ph = np.exp(-1j * np.outer(f, q) + 1j * np.outer(g - z_top, a1))
        poly = 1j * (np.outer(dg, q) + np.outer(df, a1)) * ei[:nn, None] - ei[nn:, None]
        out += delt * np.sum(ph * poly, axis=0)
    return out


def _amplitude_below(
    angles: np.ndarray, boundaries: list[_Boundary], tr: Transmission
) -> np.ndarray:
    """``a_down(θ) = α_bot·τ·e^{iα_bot z_bot}·A(q_s, α₁(q_s))``, ``q_s = k_bot sin θ``.

    Stationary phase of the transmitted representation; ``α₁(q_s)`` is on the
    code's sheet, so it is evanescent beyond the critical angle when the
    substrate is denser than the cover. Written against ``tr`` only (D3).
    """
    q = tr.k_bot * np.sin(angles)
    a_bot = alpha(q, tr.k_bot)
    spec = _spectrum(q, alpha(q, tr.k), boundaries, tr.z_top)
    return a_bot * tr.tau(q) * np.exp(1j * a_bot * tr.z_bot) * spec


def _downward_amplitude(
    angles: np.ndarray,
    boundaries: list[_Boundary],
    wnum_bg: complex,
    eps_rel: complex,
    pol: int,
    z_int: float,
) -> np.ndarray:
    """Downward far-field amplitude ``a_down(θ)`` of particles over a half-space.

    Defined by ``ψ ~ (i/4)·√(2/(π k_bot ρ))·e^{i(k_bot ρ − π/4)}·a_down(θ)``
    about the origin: :func:`far_field`'s convention with ``k_bot``. θ is from
    +z, downward for ``θ ∈ [π/2, 3π/2]``. At ``ε_rel = 1`` exactly this is the
    free-space amplitude (D19): the formula's ``2α₂/(α₁+α₂)`` is 0/0 at grazing.

    Args:
        angles: (M,) downward angles (rad).
        boundaries: ``(f, g, df, dg, delt, ei)`` per particle.
        wnum_bg: Background wavenumber.
        eps_rel: Substrate permittivity relative to the cover (not PEC).
        pol: 2 = TE, 1 = TM.
        z_int: Interface height (nm).

    Returns:
        complex (M,) amplitude.
    """
    if eps_rel == 1.0:
        out = np.zeros(len(angles), dtype=complex)
        for f, g, df, dg, delt, ei in boundaries:
            out += _far_field_at(angles, f.size, wnum_bg, f, g, df, dg, delt, ei)
        return out
    tr = Transmission.half_space(pol, wnum_bg, eps_rel, z_int)
    return _amplitude_below(angles, boundaries, tr)


def _downward_far_field(
    n_angles: int,
    boundaries: list[_Boundary],
    wnum_bg: complex,
    eps_rel: complex,
    pol: int,
    z_int: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``a_down`` on a uniform grid over ``[π/2, 3π/2]`` inclusive."""
    angles = np.linspace(0.5 * PI, 1.5 * PI, n_angles)
    amp = _downward_amplitude(angles, boundaries, wnum_bg, eps_rel, pol, z_int)
    return amp, angles


def _flux_rule(
    wnum_bg: complex,
    eps_rel: complex,
    pol: int,
    D: float,
    z_min: float,
    tr: Transmission | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Real-axis nodes and weights for the interface flux, on ``(−q_max, q_max)``.

    Breakpoints at ``|Re s|`` and ``|Re s| ± 10^m·|Im s|`` for every
    singularity s of τ; panels capped at ``min(4·2π/D, 8/z_min)``; 24-point
    Gauss–Legendre under the sine map ``q = c + h·sin(πu/2)``, which removes the
    √ endpoint behaviour at the branch points. ``q_max = 1.5·max|Re s| +
    38/z_min``, or ``Re k_bot`` for a lossless dielectric, where ``Re w ≡ 0``
    beyond it. Mirrored to ``q < 0``: the integrand is not even in q. Real
    ``k`` only.

    Args:
        wnum_bg: Background wavenumber (real).
        eps_rel: Substrate permittivity relative to the cover (not PEC).
        pol: 2 = TE, 1 = TM.
        D: Horizontal extent of the sources (nm).
        z_min: Twice the lowest source height above the interface (nm).
        tr: A stack's transmission (its features and bottom), replacing the
            half-space one built from ``eps_rel``.

    Returns:
        ``(q, w)``, real nodes and weights.
    """
    if tr is None:
        tr = Transmission.half_space(pol, wnum_bg, eps_rel, 0.0)
    bp = {0.0}
    for _, s in tr.features:
        bp.add(abs(s.real))
        width = abs(s.imag)
        if width > 0:
            for m in range(FLUX_WIDTH_DECADES + 1):
                bp.update(
                    (abs(s.real) - 10.0**m * width, abs(s.real) + 10.0**m * width)
                )
    if tr.eps_bot.imag == 0.0 and tr.eps_bot.real > 0.0:
        q_max = tr.k_bot.real
    else:
        q_max = 1.5 * max(abs(s.real) for _, s in tr.features) + TAIL_DECAY / z_min
    w_max = min(TAIL_PERIODS_PER_PANEL * 2.0 * PI / D, TAIL_EFOLDS_PER_PANEL / z_min)
    edges = sorted(b for b in bp if 0.0 <= b < q_max) + [q_max]
    fine = [edges[0]]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        n = max(1, math.ceil((hi - lo) / w_max))
        fine += list(np.linspace(lo, hi, n + 1)[1:])
    x, wx = np.polynomial.legendre.leggauss(TAIL_PANEL_POINTS)
    u, du = np.sin(0.5 * PI * x), 0.5 * PI * np.cos(0.5 * PI * x) * wx
    lo, hi = np.array(fine[:-1]), np.array(fine[1:])
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
    q = (c[:, None] + h[:, None] * u).ravel()
    w = (h[:, None] * du).ravel()
    return np.concatenate([-q[::-1], q]), np.concatenate([w[::-1], w])


def _downward_power(
    q: np.ndarray,
    w: np.ndarray,
    A: np.ndarray,
    wnum_bg: complex,
    eps_rel: complex,
    pol: int,
    tr: Transmission | None = None,
) -> float:
    """``P_down = (1/8π)·Σ w·Re w_bot·|τ·A|²``, the power through the interface.

    ``w_bot = α_bot`` (TE) or ``α_bot/ε_bot`` (TM), in G14's units, where the
    unbounded-cover dipole emits 1/4. ``A`` is any plane-wave spectrum, so a
    dipole's own ``A_dip`` can be added by the caller.

    Args:
        q: Flux nodes (:func:`_flux_rule`).
        w: Their weights.
        A: Spectrum ``A(q, α₁(q))`` at the nodes.
        wnum_bg: Background wavenumber (real).
        eps_rel: Substrate permittivity relative to the cover (not PEC).
        pol: 2 = TE, 1 = TM.
        tr: A stack's transmission, replacing the half-space one.

    Returns:
        The downward power.
    """
    if tr is None:
        tr = Transmission.half_space(pol, wnum_bg, eps_rel, 0.0)
    a_bot = alpha(q, tr.k_bot)
    w_bot = (a_bot if pol == 2 else a_bot / tr.eps_bot).real
    return float(np.sum(w * w_bot * np.abs(tr.tau(q) * A) ** 2) / (8.0 * PI))


def _downward_cross_section(
    boundaries: list[_Boundary],
    wnum_bg: complex,
    eps_rel: complex | None,
    pol: int,
    z_int: float,
) -> float:
    """``σ_sca,down = P_down/Re k`` (nm) from the interface flux (D4).

    The scattered spectrum only, which parallels ``σ_sca,up``. Valid for every
    passive substrate, lossy and metallic included; real ``k`` only.

    Args:
        boundaries: ``(f, g, df, dg, delt, ei)`` per particle.
        wnum_bg: Background wavenumber (real).
        eps_rel: Substrate permittivity relative to the cover, or ``None``.
        pol: 2 = TE, 1 = TM.
        z_int: Interface height (nm).

    Returns:
        The cross-section; exactly 0.0 for PEC and a lossless ``Re ε ≤ 0``,
        which take no power (D21). Over a lossless TM metal with
        ``Re ε < −1`` the surface plasmon carries power along the interface
        that this does not include (finding 2, conventions §16).
    """
    if eps_rel is None or (eps_rel.imag == 0.0 and eps_rel.real <= 0.0):
        return 0.0
    h_min = min(float(g.min()) for _, g, *_ in boundaries) - z_int
    z_min = 2.0 * h_min
    f_all = np.concatenate([b[0] for b in boundaries])
    q, w = _flux_rule(wnum_bg, eps_rel, pol, max(float(np.ptp(f_all)), z_min), z_min)
    A = _spectrum(q, alpha(q, wnum_bg), boundaries, z_int)
    return _downward_power(q, w, A, wnum_bg, eps_rel, pol) / float(np.real(wnum_bg))


def _check_far_field_below(eps_rel: complex | None) -> None:
    """Raise :class:`NoFarFieldError` unless the substrate is a lossless dielectric.

    A far field below exists only for ``Im ε = 0`` and ``Re ε > 0`` (D2): with
    loss the field decays with depth, and ``Re ε ≤ 0`` or PEC carry nothing to
    infinity.
    """
    if eps_rel is None:
        raise NoFarFieldError("no far field below a PEC substrate: nothing enters it")
    if eps_rel.imag != 0.0:
        raise NoFarFieldError(
            f"no far field below an absorbing substrate: Im eps_rel = {eps_rel.imag} "
            "is not 0, so the transmitted field decays with depth"
        )
    if eps_rel.real <= 0.0:
        raise NoFarFieldError(
            f"no far field below a substrate with Re eps_rel = {eps_rel.real} <= 0: "
            "no wave propagates in it"
        )


def _absorbed_cross_section(boundaries: list[_Boundary], wnum_bg: complex) -> float:
    """``σ_abs = −(1/k)·Im Σ_j delt·conj(φ_j)·χ_j``, from the boundary flux.

    φ and χ are the exterior-side values (conventions §4), so there is no ε
    factor: the exterior medium is the background. It needs no far field, which
    is why it stays exact over a substrate.
    """
    flux = 0.0
    for f, _, _, _, delt, ei in boundaries:
        nn = f.size
        flux += delt * np.sum(np.conj(ei[:nn]) * ei[nn:])
    return float(-np.imag(flux) / np.real(wnum_bg))


def _background_field(
    x: np.ndarray,
    z: np.ndarray,
    wnum_bg: complex,
    pol: int,
    excitation: str,
    angle: float,
    source: tuple[float, float] | None,
    eps_rel: complex | None,
    z_int: float | None,
) -> np.ndarray:
    """Background (unscattered) field at every point, with no particle mask.

    The field the particles are driven by, so that ``eval_field`` plus this is
    the total field (D13). Plane wave ``exp(ik(x sinθ − z cosθ))`` or dipole
    ``(i/4)·H₀(kρ)``; with a substrate, plus its reflection above ``z_int`` and
    replaced by the transmitted wave below it (0 below PEC). These are the very
    terms the right-hand side of the solve is built from.

    Args:
        x: (M,) x coordinates (nm).
        z: (M,) z coordinates (nm).
        wnum_bg: Background wavenumber, complex allowed.
        pol: 2 = TE, 1 = TM.
        excitation: ``"plane_wave"``, ``"dipole"`` or ``"custom"``.
        angle: Plane-wave incidence angle (degrees).
        source: Dipole position, for ``"dipole"``.
        eps_rel: Substrate permittivity relative to the cover, or ``None`` for
            PEC; ignored without a substrate.
        z_int: Interface height, or ``None`` for no substrate.

    Returns:
        complex (M,) field.

    Raises:
        ValueError: A ``"custom"`` excitation, whose field the result does not
            know; a point exactly at the interface (D17); a point at the
            dipole's own position, where ``H₀`` is singular.
    """
    if excitation == "custom":
        raise ValueError(
            "incident_field is undefined for a custom incident_rhs: the result "
            "does not know the field it was driven by"
        )
    k = wnum_bg
    dipole = excitation == "dipole"
    if dipole:
        assert source is not None
        xs, zs = source
        if np.any((x == xs) & (z == zs)):
            raise ValueError(
                f"incident_field at the dipole's own position ({xs}, {zs}): "
                "the line-source field is singular there"
            )
    if z_int is not None and np.any(z == z_int):
        raise ValueError(
            f"observation point exactly at the interface z_int = {z_int}: "
            "step off it (the field is continuous there)"
        )

    def free(xp: np.ndarray, zp: np.ndarray) -> np.ndarray:
        if dipole:
            return 0.25j * hank0(k * np.hypot(xp - xs, zp - zs))
        theta = np.deg2rad(angle)
        return np.exp(1j * k * (xp * np.sin(theta) - zp * np.cos(theta)))

    if z_int is None:
        return free(x, z)
    out = np.zeros(x.size, dtype=complex)
    above = z > z_int
    xa, za = x[above], z[above]
    xb, zb = x[~above], z[~above]
    no_interface = eps_rel is not None and eps_rel == 1.0
    if xa.size:
        out[above] = free(xa, za)
        if not no_interface:
            if dipole:
                # Z = z + z' − 2·z_int is never below h_obs + h_src; sizing at
                # 2·min(h) would explode the tail for a point next to the
                # interface (finding 3 is the converse trap).
                path = None
                if eps_rel is not None:
                    z_min = float((za - z_int).min() + (zs - z_int))
                    extent = max(float(np.ptp(np.append(xa, xs))), z_min)
                    path = SommerfeldPath.for_wavenumber(k, eps_rel, pol, extent, z_min)
                refl = Reflection(pol, k, eps_rel, z_int, path, 0.0)
                out[above] += refl.green(xa, za, xs, zs)[0]
            else:  # one real q: no path
                refl = Reflection(pol, k, eps_rel, z_int, None, 0.0)
                out[above] += refl.plane_wave(angle, xa, za)
    if xb.size and eps_rel is not None:
        if no_interface:
            out[~above] = free(xb, zb)
        elif dipole:
            tr = Transmission.build(pol, k, eps_rel, z_int, [xs], [zs], xb, zb)
            assert tr is not None
            out[~above] = tr.green(xb, zb, xs, zs)[0]
        else:
            tr = Transmission.half_space(pol, k, eps_rel, z_int)
            out[~above] = tr.plane_wave(angle, xb, zb)
    return out


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
    eta_in: complex = 1.0,
) -> np.ndarray:
    """Evaluate the field at arbitrary (x, z) points.

    Uses the BIE representation formula (Huygens principle) for exterior points.
    For interior points, when ri is provided, it uses the interior
    representation: wavenumber wnum_core = ri * wnum_bg, the opposite sign
    (the boundary normal points *out* of the interior region), and the
    interior-side derivative eta_in·χ (conventions §4). Without ri, interior
    points return 0+0j.

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
        eta_in: Ratio of the interior- to the exterior-side normal derivative,
            ``Material.eps`` for TM (pol = 1) and 1 for TE (pol = 2). Used only
            with ri.

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
        chi = ei[nn:] if outside else eta_in * ei[nn:]
        sign = 1.0 if outside else -1.0
        arg1 = k * dist
        sum_h = np.sum(
            (k**2 * arg2 * hank1(arg1) / arg1 * ei[:nn] - hank0(arg1) * chi) * delt
        )
        field[j] = sign * (1j / 4.0) * sum_h

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


# ---------------------------------------------------------------------------
# Fields over an N-layer stack (v1.2)
# ---------------------------------------------------------------------------


def _stack_planes(stack_rel: tuple) -> np.ndarray:
    """Every interface height of a normalised stack, top first."""
    _, d, _, z_top = stack_rel
    return z_top - np.concatenate([[0.0], np.cumsum(d)])


def _stack_source_spectra(
    q: np.ndarray, a1: np.ndarray, z_top: float, source: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """``(↑, ↓)`` spectra of a source ``(x_s, z_s, b)``, referred to ``z_top``.

    Scalar line source (``b is None``): ``e^{−iqx_s ∓ iα₁h_s}`` (v1.1 §16). An
    in-plane dipole multiplies them by ``i(q b_x ± α₁ b_z)`` (slab-spec §2.7).
    """
    xs, zs, b = source
    hs = zs - z_top
    up = np.exp(-1j * q * xs - 1j * a1 * hs)
    dn = np.exp(-1j * q * xs + 1j * a1 * hs)
    if b is not None:
        up = up * 1j * (q * b[0] + a1 * b[1])
        dn = dn * 1j * (q * b[0] - a1 * b[1])
    return up, dn


def _stack_spectra(
    q: np.ndarray,
    a1: np.ndarray,
    boundaries: list[_Boundary],
    z_top: float,
    source: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Upward and downward spectra ``(u↑, d)`` of the total field (slab-spec §2.7).

    ``d`` is :func:`_spectrum` itself and ``u↑`` is it at ``−α₁``; the source's
    own spectra are added when given. Above the stack the field is
    ``(i/2α₁)(u↑ + R·d)·e^{iα₁(z − z_top)}``; ``d`` drives everything below.
    """
    up = _spectrum(q, -a1, boundaries, z_top)
    dn = _spectrum(q, a1, boundaries, z_top)
    if source is not None:
        s_up, s_dn = _stack_source_spectra(q, a1, z_top, source)
        up, dn = up + s_up, dn + s_dn
    return up, dn


def _layer_field(
    x: np.ndarray,
    z: np.ndarray,
    boundaries: list[_Boundary],
    wnum_bg: complex,
    pol: int,
    stack_rel: tuple,
    source: tuple | None = None,
) -> np.ndarray:
    """Field at points inside the layers of a stack (slab-spec §8.2, R19).

    ``u(x, z) = (1/4π)∫_Γ i·(W/α₁)(q, z)·d(q)·e^{iqx} dq`` over the **full**
    path (``d`` is not even in q), ``d`` the downward spectrum of the boundary
    data and of ``source``, with α₁ on the path's sheet. The path is sized for
    ``z_min = min h'`` over the sources: a point just under ``z_top`` adds no
    decay. Both factors are referred to the centre ``x_c`` of the point set
    (study trap 3).

    Args:
        x: (P,) x of points strictly inside the layers (nm).
        z: (P,) their z.
        boundaries: ``(f, g, df, dg, delt, ei)`` per particle; may be empty.
        wnum_bg: Background wavenumber.
        pol: 2 = TE, 1 = TM.
        stack_rel: The normalised relative stack.
        source: ``(x_s, z_s, b)`` of a source above the stack, or ``None``.

    Returns:
        complex (P,) field.
    """
    lay, d, e_bot, z_top = stack_rel
    k = wnum_bg
    tops = _stack_planes(stack_rel)
    xs_all = [b[0] for b in boundaries] + ([[source[0]]] if source else [])
    hs_all = [b[1] - z_top for b in boundaries] + (
        [[source[1] - z_top]] if source else []
    )
    z_min = float(np.concatenate(hs_all).min())
    fx = np.concatenate([*xs_all, x])
    x_c = 0.5 * float(fx.max() + fx.min())
    feats = _stack_poles(complex(k), lay, d, e_bot, pol)[1]
    path = SommerfeldPath.for_features(k, feats, max(float(np.ptp(fx)), z_min), z_min)
    q = np.concatenate([path.q, -path.q])
    w = np.concatenate([path.w, path.w])
    a1 = alpha(q, k)
    shifted = [(b[0] - x_c, *b[1:]) for b in boundaries]
    spec = _spectrum(q, a1, shifted, z_top)
    if source is not None:
        spec = (
            spec
            + _stack_source_spectra(q, a1, z_top, (source[0] - x_c, *source[1:]))[1]
        )
    out = np.zeros(x.size, dtype=complex)
    for j in range(len(lay)):
        m = (z < tops[j]) & (z > tops[j + 1])
        if not m.any():
            continue
        # W/α₁ is even in q (it sees q only through the α's): the half path, mirrored.
        wa = stack_layer_factor(path.q, k, lay, d, e_bot, pol, tops[j] - z[m], j)
        e = np.exp(1j * np.outer(x[m] - x_c, q))
        out[m] = (1j / (4.0 * PI)) * (np.concatenate([wa, wa], axis=1) * e) @ (w * spec)
    return out


def _stack_downward_cross_section(
    boundaries: list[_Boundary], wnum_bg: complex, stack_rel: tuple, pol: int
) -> float:
    """``σ_sca,down = P_down/Re k`` (nm) through the stack, by the bottom flux.

    :func:`_downward_cross_section` with the stack's τ and features (slab-spec
    §2.7): the scattered spectrum at ``z_top`` carried down by τ.
    """
    _, _, e_bot, z_top = stack_rel
    tr = Transmission.stack(pol, wnum_bg, stack_rel)
    z_min = 2.0 * (min(float(g.min()) for _, g, *_ in boundaries) - z_top)
    f_all = np.concatenate([b[0] for b in boundaries])
    q, w = _flux_rule(
        wnum_bg, e_bot, pol, max(float(np.ptp(f_all)), z_min), z_min, tr=tr
    )
    A = _spectrum(q, alpha(q, wnum_bg), boundaries, z_top)
    p_down = _downward_power(q, w, A, wnum_bg, e_bot, pol, tr=tr)
    return p_down / float(np.real(wnum_bg))


def _stack_downward_far_field(
    n_angles: int,
    boundaries: list[_Boundary],
    wnum_bg: complex,
    stack_rel: tuple,
    pol: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``a_down`` below a stack on ``[π/2, 3π/2]``: the stack's τ and ``k_bot``."""
    _check_far_field_below(stack_rel[2])
    angles = np.linspace(0.5 * PI, 1.5 * PI, n_angles)
    tr = Transmission.stack(pol, wnum_bg, stack_rel)
    return _amplitude_below(angles, boundaries, tr), angles


def _stack_background_field(
    x: np.ndarray,
    z: np.ndarray,
    wnum_bg: complex,
    pol: int,
    excitation: str,
    angle: float,
    source: tuple[float, float] | None,
    stack_rel: tuple,
) -> np.ndarray:
    """Background field over a stack, everywhere off the interfaces.

    :func:`_background_field`'s stack twin (slab-spec §8.3): plane wave
    ``exp(ik(x sinθ − z cosθ))`` plus ``R`` above, ``W`` in the layers and ``τ``
    below; dipole ``(i/4)·H₀ + G_ind`` above, ``G_l`` in the layers and ``G_t``
    below.

    Raises:
        ValueError: A custom excitation, a point on an interface plane, or the
            dipole's own position.
    """
    if excitation == "custom":
        raise ValueError(
            "incident_field is undefined for a custom incident_rhs: the result "
            "does not know the field it was driven by"
        )
    planes = _stack_planes(stack_rel)
    if np.any(np.isin(z, planes)):
        raise ValueError(
            "observation point exactly on an interface plane of the stack: step "
            "off it (the field is continuous there)"
        )
    k = wnum_bg
    lay, d, e_bot, z_top = stack_rel
    z_bot = planes[-1]
    out = np.zeros(x.size, dtype=complex)
    above, below = z > z_top, z < z_bot
    inside = ~above & ~below
    if excitation == "dipole":
        assert source is not None
        xs, zs = source
        if np.any((x == xs) & (z == zs)):
            raise ValueError(
                f"incident_field at the dipole's own position ({xs}, {zs}): "
                "the line-source field is singular there"
            )
        if above.any():
            xa, za = x[above], z[above]
            # Z = z + z' − 2·z_top is never below h_obs + h_src: sizing at
            # 2·min(h) would explode the tail next to the interface.
            refl = StackReflection.build(
                pol,
                k,
                stack_rel,
                np.append(xa, xs),
                np.append(za, zs),
                z_min=float((za - z_top).min() + (zs - z_top)),
            )
            assert refl is not None
            out[above] = 0.25j * hank0(k * np.hypot(xa - xs, za - zs))
            out[above] += refl.green(xa, za, xs, zs)[0]
        if inside.any():
            out[inside] = _layer_field(
                x[inside], z[inside], [], k, pol, stack_rel, (xs, zs, None)
            )
        if below.any():
            xb, zb = x[below], z[below]
            tr = Transmission.build_stack(pol, k, stack_rel, [xs], [zs], xb, zb)
            out[below] = tr.green(xb, zb, xs, zs)[0]
        return out
    theta = np.deg2rad(angle)
    q, a1 = k * np.sin(theta), k * np.cos(theta)
    if above.any():
        xa, za = x[above], z[above]
        r = stack_r(np.asarray(q), k, lay, d, e_bot, pol)
        out[above] = np.exp(1j * (q * xa - a1 * za))
        out[above] += r * np.exp(1j * (q * xa + a1 * (za - 2.0 * z_top)))
    for j in range(len(lay)):
        m = (z < planes[j]) & (z > planes[j + 1])
        if m.any():
            w = stack_layer_factor(
                np.array([q]), k, lay, d, e_bot, pol, planes[j] - z[m], j
            )
            out[m] = a1 * w[:, 0] * np.exp(1j * (q * x[m] - a1 * z_top))
    if below.any():
        tr = Transmission.stack(pol, k, stack_rel)
        out[below] = tr.plane_wave(angle, x[below], z[below])
    return out
