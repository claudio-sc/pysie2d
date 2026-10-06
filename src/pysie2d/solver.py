"""BIESolver / ScatterResult façade for single-particle scattering.

Every wavelength on this façade is a **vacuum** wavelength in nm
(``docs/conventions.md`` §2). The low-level primitives this module calls
(:mod:`pysie2d.kernels`, :mod:`pysie2d.fields`, :mod:`pysie2d.sources`) take a
background wavenumber ``wnum_bg`` instead of a wavelength, so the
vacuum-to-background conversion — :meth:`pysie2d.material.Material.wnum_bg` —
happens exactly once per call path and cannot be applied twice.
"""

from collections.abc import Callable

import numpy as np

from . import layered
from .background import HalfSpace
from .fields import (
    _cross_sections,
    _far_field_at,
    _is_outside,
    _upward_cross_sections,
    _upward_far_field,
    eval_field,
    far_field,
)
from .geometry import Geometry
from .kernels import assemble_matrix, assemble_matrix_dwn
from .material import Material
from .multipole import Multipoles, decompose
from .sources import line_dipole_rhs, plane_wave_rhs

PI = np.pi


def size_parameter(
    geometry: Geometry,
    material: Material,
    wavelength: float | complex,
) -> float | complex:
    """Size parameter of a circular scatterer, referred to the cladding.

        ``x = k_bg·a = 2π·n_clad·rad/λ_vac``

    This is the ``x`` of Mie theory and the argument of every function in
    :mod:`pysie2d.reference.mie`. It is a **derived** quantity only: no public
    method accepts a size parameter as input, because ``x`` additionally depends
    on the geometry, and a second entry point would let the two disagree. The
    same reasoning rules out a complex-frequency entry point — see
    ``docs/conventions.md`` §8.

    Args:
        geometry: The scatterer boundary. Must be circular — ``x`` needs a
            single physical radius, which a non-circular Gielis shape does not
            have. Tested with ``Geometry.is_circle``.
        material: Supplies the background index ``n_clad``.
        wavelength: **Vacuum** wavelength (nm). Complex for the QNM case, giving
            a complex ``x``.

    Returns:
        The size parameter, complex if ``wavelength`` is.

    Raises:
        ValueError: If ``geometry`` is not circular.
    """
    if not geometry.is_circle:
        raise ValueError(
            "size_parameter is defined only for a circular boundary: it needs "
            "a single physical radius, which a non-circular Gielis shape does "
            "not have"
        )
    return material.wnum_bg(wavelength) * geometry.rad


def wavelength_over_ds(
    geometry: Geometry,
    material: Material,
    wavelength: float | complex,
) -> float:
    """Boundary points per interior wavelength, at the worst-resolved node.

        ``R = (Re λ_vac / n_core) / max(Δs)``

    The discretisation criterion for this solver: how many boundary samples
    describe one spatial oscillation of the field. Unlike
    :func:`size_parameter` this is defined for **any** shape — it is a
    resolution diagnostic and not a gauge, so there is no second entry point
    for it to disagree with.

    **The interior wavelength, not the vacuum one.** The BIE carries an
    interior and an exterior kernel, and the interior one oscillates faster by
    ``n_core``; it is the binding constraint. At ``n_core = 3`` that is a factor
    of three, so reading ``R`` against λ_vac overstates the resolution
    threefold.

    **The real part of λ.** ``R`` describes oscillation, and for a QNM
    wavelength the decay ``Im λ`` is not oscillation. On the shared fixture the
    two readings differ by 0.007 %, so this is a statement of convention rather
    than a numerical choice — but the QNM search boxes run to tens of nm in
    ``Im λ``, where it stops being negligible.

    **The worst node, not the mean.** Under uniform arc-length sampling every
    ``Δs`` is equal and the distinction is empty. It stops being empty under
    the frozen node sets of ``docs/conventions.md`` §10, where the spacing at a
    perturbed shape is only approximately uniform, and the risk is
    under-resolution — so the largest gap is the one that decides. Measured
    from the actual chords between consecutive boundary points, which is what
    ``perimeter/n_pts`` stops being once the spacing is not uniform.

    Args:
        geometry: The scatterer boundary, of any shape.
        material: Supplies ``n_core``, the index setting the interior
            wavelength.
        wavelength: **Vacuum** wavelength (nm); the real part is used.

    Returns:
        Points per interior wavelength at the coarsest point of the boundary.
        Higher is better resolved.
    """
    df = np.diff(geometry.f, append=geometry.f[0])
    dg = np.diff(geometry.g, append=geometry.g[0])
    ds_max = float(np.hypot(df, dg).max())
    return float(np.real(wavelength)) / material.n_core / ds_max


class ScatterResult:
    """Result of a single-wavelength BIE solve.

    Attributes:
        ei: complex (2*n_pts,) BIE solution vector (boundary fields φ and
            normal derivatives χ).
        geometry: The scatterer boundary.
        material: The scatterer optical properties.
        wavelength: **Vacuum** wavelength (nm) this was solved at.
        angle: Incident plane-wave angle (degrees).
        background: The substrate the particle sits on, or ``None`` for the
            homogeneous cladding. With one, ``far_field`` is the upward
            amplitude, ``cross_sections`` and ``efficiencies`` carry the
            ``*_up`` keys, and ``multipoles`` raises ``NotImplementedError``.
    """

    def __init__(
        self,
        ei: np.ndarray,
        geometry: Geometry,
        material: Material,
        wavelength: float,
        angle: float = 0.0,
        background: HalfSpace | None = None,
    ) -> None:
        """Store the BIE solution and the problem it was solved for."""
        self.ei = ei
        self.geometry = geometry
        self.material = material
        self.wavelength = wavelength
        self.angle = angle
        self.background = background

    def _upward(self, n_angles: int) -> dict[str, float]:
        """``σ_sca,up`` and ``σ_abs`` over the substrate; see ``fields``."""
        return _upward_cross_sections(n_angles, *self._substrate())

    def _substrate(self) -> tuple:
        """Arguments the half-space far-field helpers share."""
        g = self.geometry
        assert self.background is not None
        return (
            [(g.f, g.g, g.df, g.dg, g.delt, self.ei)],
            self.wnum_bg,
            self.background.eps_rel(self.material.n_clad, self.wavelength),
            self.material.pol,
            self.background.z_int,
        )

    @property
    def wnum_bg(self) -> complex:
        """Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm)."""
        return self.material.wnum_bg(self.wavelength)

    @property
    def size_parameter(self) -> float | complex:
        """Size parameter x = 2π·n_clad·rad/λ_vac; circular geometry only.

        See :func:`size_parameter`.
        """
        return size_parameter(self.geometry, self.material, self.wavelength)

    def eval_field(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Scattered field at arbitrary (x, z) points (nm).

        Keep observation points at least ~5 boundary-point spacings away from
        the surface (see :func:`pysie2d.fields.eval_field`).

        Args:
            x: Observation x-coordinates (nm).
            z: Observation z-coordinates (nm).

        Returns:
            complex ndarray, same shape as x/z.

        Raises:
            ValueError: With a background, a point at or below the interface.

        With a background this is still the **scattered** field: the free-space
        representation plus the reflected-image term
        ``−Σ delt·((∂x'G_ind·dg − ∂z'G_ind·df)·φ + G_ind·χ)`` at exterior points.
        Interior points are unchanged, because the interior representation uses
        the core Green function only. The total exterior field is scattered +
        incident + reflected incident; the last two are the caller's.
        """
        g = self.geometry
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        refl = None
        if self.background is not None:
            refl = layered.Reflection.build(
                self.material.pol,
                self.wnum_bg,
                self.background.eps_rel(self.material.n_clad, self.wavelength),
                self.background.z_int,
                np.concatenate([g.f, x.ravel()]),
                np.concatenate([g.g, z.ravel()]),
            )
        field = eval_field(
            self.ei,
            g.n_pts,
            g.f,
            g.df,
            g.g,
            g.dg,
            g.delt,
            self.wnum_bg,
            x,
            z,
            ri=self.material.nc,
            eta_in=self.material.eps if self.material.pol == 1 else 1.0,
        )
        if refl is not None:
            outside = np.array(
                [
                    _is_outside(xi, zi, g.f, g.g, g.df, g.dg)
                    for xi, zi in zip(x.ravel(), z.ravel(), strict=True)
                ],
                dtype=bool,
            )
            if outside.any():
                pts = layered.Points(x.ravel()[outside], z.ravel()[outside])
                m1, m2 = refl.blocks(pts, g)
                n = g.n_pts
                image = -(m1 @ self.ei[:n] + m2 @ self.ei[n:])
                field = field.copy()
                field[np.flatnonzero(outside)] += image
        return field

    def multipoles(
        self,
        mmax: int = 8,
        *,
        r0: float | None = None,
        ntheta: int | None = None,
    ) -> Multipoles:
        """Cylindrical-harmonic decomposition of the scattered field.

        Args:
            mmax: Highest order retained. The default of 8 resolves the
                electric quadrupole and two orders beyond it.
            r0: Radius of the evaluation circle (nm), about the particle
                centre. Default ``3.0 × the circumscribing radius`` — which is
                **not** 3.0 × ``Geometry.rad``: on a rounded square the
                circumscribing radius is 2.29 × ``rad``.
            ntheta: Angular samples. Default resolves both the field and the
                retained orders; see ``pysie2d.multipole``.

        Returns:
            Multipoles, about the particle centre ``(geometry.x0,
            geometry.z0)``.

        Raises:
            ValueError: If ``r0`` does not exceed the circumscribing radius of
                the boundary — inside it the expansion does not converge and
                the coefficients are meaningless (measured: 58 % error at
                0.9 × the circumscribing radius, with nothing else to warn you).
        """
        if self.background is not None:
            raise NotImplementedError(
                "multipoles is not available with a background: the expansion "
                "is of the free-space scattered field"
            )
        geo = self.geometry
        r_circ = float(np.hypot(geo.f - geo.x0, geo.g - geo.z0).max())
        if r0 is None:
            r0 = 3.0 * r_circ
        elif r0 <= r_circ:
            raise ValueError(
                f"r0 = {r0:.6g} nm does not exceed the circumscribing radius "
                f"{r_circ:.6g} nm of this boundary; the multipole expansion "
                f"does not converge inside it. Note that the circumscribing "
                f"radius is not Geometry.rad = {geo.rad:.6g} nm."
            )
        return decompose(
            self.ei,
            geo.n_pts,
            geo.f,
            geo.df,
            geo.g,
            geo.dg,
            geo.delt,
            self.wnum_bg,
            r0,
            mmax=mmax,
            ntheta=ntheta,
            x0=geo.x0,
            z0=geo.z0,
        )

    def far_field(self, n_angles: int = 3000) -> tuple[np.ndarray, np.ndarray]:
        """Far-field scattering amplitude.

        With a background this is the **upward** amplitude ``a_up(θ)`` — the
        particle's own plus its reflection off the interface — on a uniform grid
        over ``[−π/2, π/2]`` inclusive. There is no transmitted far field.

        Args:
            n_angles: Number of observation angles.

        Returns:
            amplitude: complex (n_angles,) far-field amplitude.
            angles: float (n_angles,) observation angles (rad), from −π to π
                (``−π/2`` to ``π/2`` with a background), measured from +z.
        """
        if self.background is not None:
            return _upward_far_field(n_angles, *self._substrate())
        g = self.geometry
        return far_field(
            g.n_pts,
            n_angles,
            self.wnum_bg,
            g.f,
            g.g,
            g.df,
            g.dg,
            g.delt,
            self.ei,
        )

    def efficiencies(self, n_angles: int = 500) -> dict[str, float]:
        """Scattering, extinction, and absorption efficiencies.

        Efficiencies are normalised by the geometric width ``2·rad``, which
        equals the true projected width only for a circle. For non-circular
        Gielis shapes this normalisation is only approximate.

        Args:
            n_angles: Number of far-field angles used in the angular integral.
                The periodic trapezoid rule is spectral in it: Q_sca reaches
                round-off by 257 at size parameter 50 (n_core = 3.5), so the
                default of 500 leaves ~2× headroom. Raise it for larger
                particles. Q_ext does not depend on it.

        Returns:
            dict with keys 'qsca', 'qext', 'qabs'. With a background, instead
            ``{'qsca_up', 'qabs'}`` — the :meth:`cross_sections` values over
            ``2·rad``. The free-space keys are absent rather than raising:
            ``qext`` and the total ``qsca`` are not defined over a substrate,
            since power also goes into it and the incident wave is reflected.
        """
        if self.background is not None:
            c = self._upward(n_angles)
            norm = 2.0 * self.geometry.rad
            return {"qsca_up": c["c_sca_up"] / norm, "qabs": c["c_abs"] / norm}
        wnum_bg = self.wnum_bg
        norfac = 8.0 * PI * wnum_bg
        delthe = 2.0 * PI / (n_angles - 1.0)

        amp, _ = self.far_field(n_angles)
        i_sc = np.abs(amp) ** 2 / norfac
        # far_field's angles span [-pi, pi] inclusive, so index 0 and index
        # n_angles-1 are the *same* physical direction. Summing all n_angles
        # samples double-counts it — a periodic integrand's uniform-grid
        # quadrature wants the n_angles-1 distinct points, each already
        # carrying weight delthe = 2*pi/(n_angles-1). Dropping the duplicate
        # (not halving both copies) is what makes qsca independent of where
        # that one grid angle happens to fall relative to the forward peak.
        qsca = np.sum(i_sc[:-1]) * delthe / (2.0 * self.geometry.rad)
        # The forward direction π − angle lies on the far-field grid only for
        # incidence angles that happen to be grid multiples; reading the nearest
        # sample instead was 1.5e-7 off at 37° with n_angles = 3000. Evaluate
        # the amplitude there exactly, so qext does not depend on n_angles.
        g = self.geometry
        forward = np.array([PI - np.deg2rad(self.angle)])
        amp_fwd = _far_field_at(
            forward, g.n_pts, wnum_bg, g.f, g.g, g.df, g.dg, g.delt, self.ei
        )[0]
        qext = amp_fwd.imag / (wnum_bg * 2.0 * g.rad)
        qabs = qext - qsca
        return {"qsca": float(qsca), "qext": float(qext), "qabs": float(qabs)}

    def cross_sections(self, n_angles: int = 500) -> dict[str, float]:
        """Absolute scattering, extinction and absorption cross-sections (nm).

        The multiparticle observable of docs/design/multiparticle-spec.md §3.6,
        provided here too so the single- and multi-particle paths report the
        same quantity. Equal to :meth:`efficiencies` scaled by the geometric
        width ``2·rad`` — which is why this is the quantity to compare across
        shapes, since that normalisation is only approximate off a circle.

        Plane-wave excitation only: ``C_ext`` is defined against a
        unit-amplitude incident plane wave.

        Args:
            n_angles: Number of far-field angles used in the angular integral;
                see :meth:`efficiencies` for how to size it. C_ext does not
                depend on it.

        Returns:
            dict with keys 'c_sca', 'c_ext', 'c_abs', in nm. With a background,
            instead ``{'c_sca_up', 'c_abs'}``: the power scattered **into the
            cover**, by Gauss–Legendre over ``[−π/2, π/2]`` (``n_angles`` nodes),
            and the absorption from the boundary flux. ``c_sca_up`` is incomplete
            by construction — it omits the power sent into the substrate and the
            specularly reflected incident beam. The free-space keys are absent
            rather than raising.
        """
        if self.background is not None:
            return self._upward(n_angles)
        g = self.geometry
        wnum_bg = self.wnum_bg
        amp, _ = self.far_field(n_angles)
        # π − angle lands on the far-field grid only by accident, so evaluate
        # the forward amplitude there exactly (same trap as in efficiencies).
        forward = np.array([PI - np.deg2rad(self.angle)])
        amp_fwd = _far_field_at(
            forward, g.n_pts, wnum_bg, g.f, g.g, g.df, g.dg, g.delt, self.ei
        )[0]
        return _cross_sections(amp, amp_fwd, wnum_bg)


class BIESolver:
    """BIE solver for 2-D scattering from a Gielis particle.

    Composes a :class:`~pysie2d.geometry.Geometry` and a
    :class:`~pysie2d.material.Material` into a reusable solver object. Each call
    to :meth:`scatter` returns a :class:`ScatterResult` that carries the
    solution and exposes analysis methods.

    Attributes:
        geometry: Discretized particle boundary.
        material: Optical properties of the scatterer.
        background: A :class:`~pysie2d.background.HalfSpace` the particle sits
            above, or ``None`` (default) for the homogeneous cladding, which is
            bit-identical to a solver without the argument.

    Examples:
        >>> geom = Geometry.gielis(rad=200, n_pts=300, m=6)
        >>> mat = Material(n_core=1.5, n_clad=1.0, pol=2)
        >>> solver = BIESolver(geom, mat)
        >>> result = solver.scatter(wavelength=600.0)
        >>> eff = result.efficiencies()
        >>> field = result.eval_field(x_grid, z_grid)
    """

    def __init__(
        self,
        geometry: Geometry,
        material: Material,
        background: HalfSpace | None = None,
    ) -> None:
        """Compose a geometry and a material into a reusable solver.

        Raises:
            ValueError: With a background, a boundary node at or below the
                interface.
        """
        self.geometry = geometry
        self.material = material
        self.background = background
        # A path fixed for a whole QNM search box; None builds one per call.
        self._path: layered.SommerfeldPath | None = None
        if background is not None:
            layered.check_interface_gap([geometry], background.z_int, material.pol)

    def _reflection(
        self,
        wavelength: float | complex,
        f: np.ndarray | None = None,
        g: np.ndarray | None = None,
    ) -> layered.Reflection | None:
        """Reflected-field context at ``wavelength``, or ``None`` without one.

        Args:
            wavelength: **Vacuum** wavelength (nm).
            f: x coordinates of every point to serve; the boundary by default.
            g: Their z coordinates.
        """
        if self.background is None:
            return None
        geo = self.geometry
        mat = self.material
        return layered.Reflection.build(
            mat.pol,
            mat.wnum_bg(wavelength),
            self.background.eps_rel(mat.n_clad, wavelength),
            self.background.z_int,
            geo.f if f is None else f,
            geo.g if g is None else g,
            path=self._path,
        )

    def assemble(self, wavelength: float | complex) -> np.ndarray:
        """Assemble the 2nn × 2nn BIE system matrix M(λ).

        The matrix depends only on the wavelength (and the fixed geometry and
        material), not on the excitation, so it can be factorised once and
        reused across many right-hand sides at the same wavelength — see
        :func:`pysie2d.green.relative_ldos_map` for the LU-reuse pattern.

        This is the **only** place the system matrix converts a wavelength into
        a wavenumber, which is why :class:`pysie2d.qnm.QNMSolver` can hand it
        vacuum wavelengths straight off its contour and read vacuum wavelengths
        back out of the eigenvalues, with no conversion on the return leg.

        Args:
            wavelength: **Vacuum** wavelength (nm). A **complex** wavelength is
                the quasi-normal-mode case: the whole assembly path is complex
                throughout, and :class:`pysie2d.qnm.QNMSolver` calls this very
                method on its contour.

        Returns:
            complex (2·n_pts, 2·n_pts) system matrix.
        """
        g = self.geometry
        mat = self.material
        m = assemble_matrix(
            mat.pol,
            g.n_pts,
            g.f,
            g.g,
            g.df,
            g.dg,
            g.ddf,
            g.ddg,
            mat.wnum_bg(wavelength),
            mat.nc,
            mat.eps,
        )
        refl = self._reflection(wavelength)
        if refl is not None:
            # The substrate's reflected field enters the exterior equation only,
            # with the sign of the cross-particle blocks.
            m1, m2 = refl.blocks(g, g)
            m[: g.n_pts, : g.n_pts] += m1
            m[: g.n_pts, g.n_pts :] += m2
        return m

    def assemble_derivative(self, wavelength: float | complex) -> np.ndarray:
        """Analytic derivative dM/dλ of the BIE system matrix, λ vacuum in nm.

        The companion to :meth:`assemble`, and the callable bordered Newton
        refinement needs: :func:`pysie2d.beyn.newton_refine` takes a
        ``dm_builder: λ → dM/dλ`` alongside its ``m_builder``.

        This is the **only** place the wavelength chain factor is applied.
        :func:`pysie2d.kernels.assemble_matrix_dwn` differentiates with respect
        to the background wavenumber, in keeping with every primitive in that
        module taking no wavelength; here ``k_bg = 2π·n_clad/λ_vac`` gives

            ``dM/dλ = (dM/dk)·(dk/dλ) = −(k/λ)·dM/dk``

        with the same ``k`` the assembly was built from, so ``n_clad`` enters
        once on both legs (``docs/conventions.md`` §2). A caller applying the
        factor itself would be a second conversion point, which is what that
        convention exists to prevent.

        Exact only for a **non-dispersive** material — see
        :func:`pysie2d.kernels.assemble_matrix_dwn`, which states the
        precondition and the identities.

        The matrix half of the fused pair is discarded here, which is
        deliberate: ``newton_refine`` asks for ``M`` and ``dM`` through two
        independent callables, and that keeps :mod:`pysie2d.beyn` free of any
        electromagnetics. It costs almost nothing — the pair is 1.05× one
        assembly, so a Newton step is 2.05 assemblies rather than the 2.0 a
        single fused callback would buy *(measured at nn = 200, complex λ)*.
        The fused return exists so the parity test has an ``M`` to compare
        against, not to save time here.

        Args:
            wavelength: **Vacuum** wavelength (nm). Complex for the
                quasi-normal-mode case, exactly as :meth:`assemble`.

        Returns:
            complex (2·n_pts, 2·n_pts) matrix dM/dλ (units of M per nm).
        """
        g = self.geometry
        mat = self.material
        wnum_bg = mat.wnum_bg(wavelength)
        _, dm_dk = assemble_matrix_dwn(
            mat.pol,
            g.n_pts,
            g.f,
            g.g,
            g.df,
            g.dg,
            g.ddf,
            g.ddg,
            wnum_bg,
            mat.nc,
            mat.eps,
        )
        refl = self._reflection(wavelength)
        if refl is not None:
            # Added before the chain factor, so it is applied exactly once. The
            # path is fixed, so d/dk at fixed nodes is the exact derivative.
            dm1, dm2 = refl.blocks_dk(g, g)
            dm_dk[: g.n_pts, : g.n_pts] += dm1
            dm_dk[: g.n_pts, g.n_pts :] += dm2
        return dm_dk * (-wnum_bg / wavelength)

    def scatter(
        self,
        wavelength: float,
        angle: float = 0.0,
        incident_rhs: Callable[[int, complex, np.ndarray, np.ndarray], np.ndarray]
        | None = None,
    ) -> ScatterResult:
        """Solve the BIE for one wavelength.

        Args:
            wavelength: **Vacuum** wavelength (nm).
            angle: Plane-wave incidence angle (degrees). Default 0.
            incident_rhs: Custom incident-field callable with signature
                ``(nn, wnum_bg, f, g) → complex (2*nn,)``, where ``wnum_bg`` is
                the background wavenumber ``2π·n_clad/λ_vac`` — **not** a
                wavelength. Replaces the default plane-wave excitation. With a
                background it must supply the **full** incident field, the
                substrate's reflection of it included.

        Returns:
            ScatterResult carrying the solution vector and analysis methods.

        Raises:
            ValueError: With a background and the default plane wave,
                ``|angle| ≥ 90°``.
        """
        g = self.geometry
        mat = self.material

        # `assemble` converts the vacuum wavelength itself; the RHS builders take
        # the wavenumber directly. Neither path can convert twice.
        m = self.assemble(wavelength)
        wnum_bg = mat.wnum_bg(wavelength)

        if incident_rhs is not None:
            rhs = incident_rhs(g.n_pts, wnum_bg, g.f, g.g)
        else:
            rhs = plane_wave_rhs(g.n_pts, angle, wnum_bg, g.f, g.g)
            refl = self._reflection(wavelength)
            if refl is not None:
                rhs[: g.n_pts] += refl.plane_wave(angle, g.f, g.g)

        ei = np.linalg.solve(m, rhs)
        return ScatterResult(ei, g, mat, wavelength, angle, self.background)

    def scatter_dipole(
        self,
        wavelength: float,
        x_s: float,
        z_s: float,
    ) -> ScatterResult:
        """Solve the BIE for a line-dipole (2-D point) source at (x_s, z_s).

        Convenience wrapper that binds the source position into
        :func:`pysie2d.sources.line_dipole_rhs` and plugs it into the
        ``incident_rhs`` hook of :meth:`scatter`.

        Args:
            wavelength: **Vacuum** wavelength (nm).
            x_s: Source x-coordinate (nm).
            z_s: Source z-coordinate (nm).

        Returns:
            ScatterResult carrying the scattered-field solution vector.

        Raises:
            ValueError: If the source is inside the particle or too close to
                its surface (see :func:`pysie2d.sources.line_dipole_rhs`).
        """
        geo = self.geometry
        refl = self._reflection(
            wavelength, np.append(geo.f, x_s), np.append(geo.g, z_s)
        )

        def rhs(nn: int, wnum_bg: complex, f: np.ndarray, g: np.ndarray) -> np.ndarray:
            ei = line_dipole_rhs(nn, wnum_bg, f, g, x_s, z_s)
            if refl is not None:
                ei[:nn] += refl.green(f, g, x_s, z_s)[0]
            return ei

        return self.scatter(wavelength, incident_rhs=rhs)
