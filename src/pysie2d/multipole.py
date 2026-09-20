"""Cylindrical-harmonic multipole decomposition of the BIE scattered field.

Outside the circumscribing circle of the particle the scattered field is a sum
of outgoing cylindrical harmonics about the particle centre ``(x0, z0)``:

    ψ_sc(r, θ) = Σ_m c_m · i^m · H_m^{(1)}(k_bg·r) · e^{imθ}

with `H_m^{(1)}` outgoing under the ``exp(−iωt)`` convention
(``docs/conventions.md`` §3). The ``i^m`` is part of the basis, kept from the
source formulation so that the ``c_m`` are the numbers the reference paper
tabulates. Negative orders come from ``H_{−m}^{(1)} = (−1)^m H_m^{(1)}``.

**θ is measured from +z toward +x**, so a point on the circle is
``x = x0 + r·sin θ``, ``z = z0 + r·cos θ`` — the same θ that
:func:`pysie2d.geometry.gielis` and :func:`pysie2d.fields.far_field` use, and
**not** the usual ``x = r cos θ``. A cos/sin swap rotates every coefficient by
``e^{imπ/2}`` and is silent otherwise.

Because ``i^{−m} H_{−m}^{(1)} = i^m H_m^{(1)}``, the ±m pair shares a radial
factor and the expansion reorganises into a symmetric/antisymmetric basis

    ψ_m^+ = i^m H_m^{(1)}(k r) · cos(mθ)        ψ_m^- = i^m H_m^{(1)}(k r) · i·sin(mθ)

with ``c_m^+ = c_m + c_{−m}`` and ``c_m^- = c_m − c_{−m}`` for m ≥ 1, and
``c_0^+ = c_0`` (the m = 0 term appears once in the sum, not twice). That basis
is kept because it reads physically: for ``pol = 1`` (TM, H_y) the leading
terms are m = 0 → magnetic dipole, m = 1 → electric dipole, m = 2 → electric
quadrupole.

**What this does not cover.** The anchor is analytic Mie, on a circle; for a
non-circular shape the only available check is internal self-consistency
(decompose on one circle, reconstruct on another), which shows the expansion
reproduces the solver's own field and nothing more. The coefficients describe
the field **outside the circumscribing circle only** — nothing between that
circle and the boundary — and they are referred to the particle centre:
coefficients about any other centre are a different set of numbers, related by
a translation addition theorem that is not implemented here.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import hankel1

from .fields import eval_field

PI = np.pi

# Above this, the angular grid has not exhausted the field's angular content
# and the coefficients are suspect. A screen, not an error bar: at a tail of
# 1.1e-4 the measured coefficient error was 1.1e-12, at 8.5e-1 it was 1.2.
# The default ntheta (`_default_ntheta`) at the default r0 of
# `ScatterResult.multipoles` lands at round-off on a circle and ~2e-15 on the
# worst shape measured (a spiky m=5 star) — both several decades below this
# screen, so a default call never warns.
SPECTRUM_TAIL_WARN = 1e-10


@dataclass(frozen=True)
class Multipoles:
    """Cylindrical-harmonic coefficients of a scattered field.

    Attributes:
        c: complex (2*mmax+1,) coefficients c_m for m = −mmax … mmax, with
            ``c[m + mmax]`` the coefficient of order m. The expansion is
            ψ_sc = Σ_m c_m · i^m · H_m^(1)(k_bg·r) · e^{imθ}, valid outside
            the circumscribing circle of the particle only, with θ measured
            from +z toward +x about ``(x0, z0)``.
        mmax: Highest order retained.
        r0: Radius of the evaluation circle (nm).
        wnum_bg: Background wavenumber the field was evaluated at (rad/nm).
        x0: Expansion centre x-coordinate — the particle centre (nm).
        z0: Expansion centre z-coordinate — the particle centre (nm).
        ntheta: Number of angular samples used.
        spectrum_tail: Angular-DFT content nearest Nyquist, relative to the
            largest coefficient. A resolution screen: ≲1e-12 means the grid
            exhausted the field's angular content. See ``SPECTRUM_TAIL_WARN``.
    """

    c: np.ndarray
    mmax: int
    r0: float
    wnum_bg: complex
    x0: float
    z0: float
    ntheta: int
    spectrum_tail: float

    @property
    def orders(self) -> np.ndarray:
        """(2*mmax+1,) integer orders m = −mmax … mmax, aligned with ``c``."""
        return np.arange(-self.mmax, self.mmax + 1)

    @property
    def c_plus(self) -> np.ndarray:
        """(mmax+1,) symmetric coefficients c_m^+ = c_m + c_{−m}, m = 0 … mmax.

        ``c_0^+ = c_0``, not ``2·c_0``: the m = 0 term appears once in the sum.
        """
        m = self.mmax
        out = self.c[m:] + self.c[m::-1]
        out[0] = self.c[m]
        return out

    @property
    def c_minus(self) -> np.ndarray:
        """(mmax,) antisymmetric coefficients c_m^- = c_m − c_{−m}, m = 1 … mmax.

        Identically zero at normal incidence on a boundary symmetric about the
        incidence axis — which is why a test of this half must be run oblique.
        """
        m = self.mmax
        return self.c[m + 1 :] - self.c[m - 1 :: -1]

    def reconstruct(self, r: float, theta: np.ndarray) -> np.ndarray:
        """Field on a circle of radius ``r`` about the expansion centre.

        Args:
            r: Radius (nm), measured from ``(x0, z0)``. Must exceed the
                circumscribing radius; nothing here can check that.
            theta: (M,) angles (rad), from +z toward +x.

        Returns:
            complex (M,) reconstructed scattered field.
        """
        return reconstruct(self.c, self.wnum_bg, r, theta)


def _hankel_orders(mmax: int, z: complex) -> np.ndarray:
    """H_m^(1)(z) for m = −mmax … mmax, from the orders 0 … mmax only.

    Uses H_{−m}^(1) = (−1)^m H_m^(1) rather than calling scipy at negative
    order: the special functions are 99 % of this package's runtime, so
    evaluating each order once is the whole optimisation available here.

    Args:
        mmax: Highest order.
        z: Argument, complex for a quasi-normal-mode wavenumber.

    Returns:
        complex (2*mmax+1,) Hankel values, ordered m = −mmax … mmax.
    """
    h_pos = hankel1(np.arange(mmax + 1), z)
    signs = (-1.0) ** np.arange(mmax, 0, -1)
    return np.concatenate([signs * h_pos[:0:-1], h_pos])


def _default_ntheta(mmax: int, wnum_bg: complex, r0: float) -> int:
    """Angular samples that resolve both the field and the retained orders.

    The field's angular content runs to roughly |k·r0|; the 2·mmax+1 retained
    orders must additionally be separated. Measured at this rule: agreement
    with a 4× finer grid of 2.5e-16 … 6.6e-16 over |k·r0| = 3.1 … 10.8 and
    mmax = 8 … 30 (docs/design/multipole-spec.md §2, D3).

    Args:
        mmax: Highest order retained.
        wnum_bg: Background wavenumber (rad/nm).
        r0: Radius of the evaluation circle (nm).

    Returns:
        Number of angular samples.
    """
    return int(2 * (mmax + np.ceil(abs(wnum_bg * r0))) + 16)


def decompose(
    ei: np.ndarray,
    nn: int,
    f: np.ndarray,
    df: np.ndarray,
    g: np.ndarray,
    dg: np.ndarray,
    delt: float,
    wnum_bg: complex,
    r0: float,
    *,
    mmax: int = 8,
    ntheta: int | None = None,
    x0: float = 0.0,
    z0: float = 0.0,
) -> Multipoles:
    """Project the scattered field onto cylindrical harmonics.

    Samples the scattered field at midpoint nodes θ_j = 2π(j + ½)/N on the
    circle of radius ``r0`` about ``(x0, z0)``, takes its angular DFT, and
    divides out the radial basis factor ``i^m·H_m^(1)(k_bg·r0)``. Exact for a
    field whose angular content the grid resolves — which is what the default
    ``ntheta`` and the guards below are for. The orthogonality lives in θ,
    which is always real, so the projection is exact at **complex** ``wnum_bg``
    as well (``docs/conventions.md`` §8).

    ``r0`` must exceed the circumscribing radius of the boundary; that check
    lives on :meth:`pysie2d.solver.ScatterResult.multipoles`, where the centre
    is known to be the particle's. Here the backstop is the zero tripwire
    below.

    Args:
        ei: complex (2nn,) BIE solution vector (φ = ei[:nn], χ = ei[nn:]).
        nn: Number of boundary points.
        f: (nn,) boundary x coordinates (nm).
        df: (nn,) df/dt, t the quadrature parameter the nodes are equispaced in.
        g: (nn,) boundary z coordinates (nm).
        dg: (nn,) dg/dt.
        delt: Trapezoid step 2π/nn in t (:attr:`pysie2d.geometry.Geometry.delt`).
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm). This
            function takes no wavelength, so the vacuum conversion cannot be
            applied twice. Complex for quasi-normal-mode work.
        r0: Radius of the evaluation circle (nm), about ``(x0, z0)``. Must
            exceed the circumscribing radius of the boundary.
        mmax: Highest order retained.
        ntheta: Angular samples. ``None`` (default) uses a rule that resolves
            both the field and the retained orders.
        x0: Expansion centre x-coordinate — the particle centre (nm).
        z0: Expansion centre z-coordinate — the particle centre (nm).

    Returns:
        Multipoles carrying the signed-order coefficients and the
        angular-resolution diagnostic.

    Raises:
        ValueError: If ``mmax`` is negative; if ``ntheta`` is below
            ``2·mmax + 1``; if ``H_mmax^(1)(k_bg·r0)`` overflows; or if the
            field evaluation places a sample point inside the particle.
    """
    if mmax < 0:
        raise ValueError(f"mmax must be non-negative, got {mmax}")

    if ntheta is None:
        ntheta = _default_ntheta(mmax, wnum_bg, r0)
    if ntheta < 2 * mmax + 1:
        raise ValueError(
            f"ntheta = {ntheta} is below the minimum 2*mmax + 1 = "
            f"{2 * mmax + 1} for mmax = {mmax}: below it the retained orders "
            f"are not orthogonal on the angular grid and the coefficients are "
            f"catastrophically wrong (measured 2.4e+12 at mmax = 8, "
            f"ntheta = 16) with no diagnostic that catches it"
        )

    h_all = _hankel_orders(mmax, wnum_bg * r0)
    if not np.all(np.isfinite(h_all)):
        raise ValueError(
            f"H_m^(1)(k_bg*r0) overflows at k_bg*r0 = {wnum_bg * r0:.6g} for "
            f"orders up to mmax = {mmax}; every coefficient beyond the "
            f"overflow would be silently 0 or nan. Increase r0 or reduce mmax."
        )

    theta = (np.arange(ntheta) + 0.5) * 2.0 * PI / ntheta
    x_pts = x0 + r0 * np.sin(theta)
    z_pts = z0 + r0 * np.cos(theta)
    # ri=None is load-bearing: it makes a point that eval_field's
    # nearest-neighbour inside/outside test misclassifies as interior return
    # exactly 0+0j rather than an interior field value, turning a plausible
    # wrong number into the detectable one the tripwire below reads.
    psi = eval_field(ei, nn, f, df, g, dg, delt, wnum_bg, x_pts, z_pts, ri=None)

    if np.any(psi == 0.0):
        raise ValueError(
            f"eval_field returned exactly zero at "
            f"{int(np.count_nonzero(psi == 0.0))} of {ntheta} points on the "
            f"r0 = {r0:.6g} nm circle, which means its inside/outside test "
            f"placed them inside the particle. r0 must exceed the "
            f"circumscribing radius of the boundary."
        )

    orders = np.arange(-mmax, mmax + 1)
    spectrum = (np.exp(-1j * np.outer(orders, theta)) @ psi) / ntheta
    c = spectrum / (1j**orders) / h_all

    # The diagnostic reads the *full* grid, not the retained orders: the
    # question it answers is whether the grid exhausted the field's angular
    # content, which the retained band cannot see.
    full = np.fft.fft(psi) / ntheta
    nyq = ntheta // 2
    peak = np.abs(full).max()
    tail = float(np.abs(full[nyq - 1 : nyq + 2]).max() / peak) if peak > 0.0 else 0.0
    if tail > SPECTRUM_TAIL_WARN:
        warnings.warn(
            f"angular spectrum tail {tail:.3g} exceeds {SPECTRUM_TAIL_WARN:.0e}: "
            f"ntheta = {ntheta} is too small for k_bg*r0 = "
            f"{abs(wnum_bg * r0):.6g}, so the grid has not exhausted the "
            f"field's angular content and the coefficients are under-resolved",
            RuntimeWarning,
            stacklevel=2,
        )

    return Multipoles(
        c=c,
        mmax=mmax,
        r0=r0,
        wnum_bg=wnum_bg,
        x0=x0,
        z0=z0,
        ntheta=ntheta,
        spectrum_tail=tail,
    )


def reconstruct(
    c: np.ndarray,
    wnum_bg: complex,
    r: float,
    theta: np.ndarray,
) -> np.ndarray:
    """Rebuild the scattered field from signed-order coefficients.

    Args:
        c: complex (2*mmax+1,) signed-order coefficients, m = −mmax … mmax.
        wnum_bg: Background wavenumber k_bg (rad/nm); complex is supported.
        r: Radius (nm) about the expansion centre. Must exceed the
            circumscribing radius of the particle, which is not checkable here.
        theta: (M,) angles (rad), measured from +z toward +x.

    Returns:
        complex (M,) reconstructed scattered field.

    Raises:
        ValueError: If ``len(c)`` is even — the signed-order array always has
            odd length, and an even one is a ``c_plus`` array passed by
            mistake, which would otherwise return a plausible wrong answer.
    """
    if len(c) % 2 == 0:
        raise ValueError(
            f"c must hold the signed orders m = -mmax ... mmax and therefore "
            f"have odd length, got {len(c)}; a c_plus or c_minus array is not "
            f"a valid input here"
        )
    theta = np.asarray(theta, dtype=float)
    mmax = (len(c) - 1) // 2
    orders = np.arange(-mmax, mmax + 1)
    radial = c * (1j**orders) * _hankel_orders(mmax, wnum_bg * r)
    return radial @ np.exp(1j * np.outer(orders, theta))
