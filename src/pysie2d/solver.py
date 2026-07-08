"""BIESolver / ScatterResult façade for single-particle scattering."""

import functools
from collections.abc import Callable

import numpy as np

from .fields import eval_field, far_field
from .geometry import Geometry
from .kernels import assemble_matrix
from .material import Material
from .sources import line_dipole_rhs, plane_wave_rhs

PI = np.pi


class ScatterResult:
    """Result of a single-wavelength BIE solve.

    Attributes:
        ei: complex (2*n_pts,) BIE solution vector (boundary fields φ and
            normal derivatives χ).
        geometry: The scatterer boundary.
        material: The scatterer optical properties.
        wavelength: Free-space wavelength (nm).
        angle: Incident plane-wave angle (degrees).
    """

    def __init__(
        self,
        ei: np.ndarray,
        geometry: Geometry,
        material: Material,
        wavelength: float,
        angle: float = 0.0,
    ) -> None:
        """Store the BIE solution and the problem it was solved for."""
        self.ei = ei
        self.geometry = geometry
        self.material = material
        self.wavelength = wavelength
        self.angle = angle

    @property
    def wnum(self) -> complex:
        """Free-space wavenumber 2π/λ."""
        return 2.0 * PI / self.wavelength

    def eval_field(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Scattered field at arbitrary (x, z) points (nm).

        Keep observation points at least ~5 boundary-point spacings away from
        the surface (see :func:`pysie2d.fields.eval_field`).

        Args:
            x: Observation x-coordinates (nm).
            z: Observation z-coordinates (nm).

        Returns:
            complex ndarray, same shape as x/z.
        """
        g = self.geometry
        return eval_field(
            self.ei,
            g.n_pts,
            g.f,
            g.df,
            g.g,
            g.dg,
            g.delt,
            self.wnum,
            np.asarray(x, dtype=float),
            np.asarray(z, dtype=float),
            ri=self.material.nc,
        )

    def far_field(self, n_angles: int = 3000) -> tuple[np.ndarray, np.ndarray]:
        """Far-field scattering amplitude.

        Args:
            n_angles: Number of observation angles.

        Returns:
            amplitude: complex (n_angles,) far-field amplitude.
            angles: float (n_angles,) observation angles (rad), from −π to π.
        """
        g = self.geometry
        return far_field(
            g.n_pts,
            n_angles,
            self.wavelength,
            g.f,
            g.g,
            g.df,
            g.dg,
            g.delt,
            self.ei,
        )

    def efficiencies(self, n_angles: int = 3000) -> dict[str, float]:
        """Scattering, extinction, and absorption efficiencies.

        Efficiencies are normalised by the geometric width ``2·rad``, which
        equals the true projected width only for a circle. For non-circular
        Gielis shapes this normalisation is only approximate.

        Args:
            n_angles: Number of far-field angles used in the angular integral.

        Returns:
            dict with keys 'qsca', 'qext', 'qabs'.
        """
        wnum = 2.0 * PI / self.wavelength
        norfac = 8.0 * PI * wnum
        delthe = 2.0 * PI / (n_angles - 1.0)
        nforw = int((2.0 * PI - np.deg2rad(self.angle)) / delthe)

        amp, _ = self.far_field(n_angles)
        i_sc = np.abs(amp) ** 2 / norfac
        qsca = np.sum(i_sc) * delthe / (2.0 * self.geometry.rad)
        qext = amp[nforw].imag / (wnum * 2.0 * self.geometry.rad)
        qabs = qext - qsca
        return {"qsca": float(qsca), "qext": float(qext), "qabs": float(qabs)}


class BIESolver:
    """BIE solver for 2-D scattering from a Gielis particle.

    Composes a :class:`~pysie2d.geometry.Geometry` and a
    :class:`~pysie2d.material.Material` into a reusable solver object. Each call
    to :meth:`scatter` returns a :class:`ScatterResult` that carries the
    solution and exposes analysis methods.

    Attributes:
        geometry: Discretized particle boundary.
        material: Optical properties of the scatterer.

    Examples:
        >>> geom = Geometry.gielis(rad=200, n_pts=300, m=6)
        >>> mat = Material(n_core=1.5, n_clad=1.0, pol=2)
        >>> solver = BIESolver(geom, mat)
        >>> result = solver.scatter(wavelength=600.0)
        >>> eff = result.efficiencies()
        >>> field = result.eval_field(x_grid, z_grid)
    """

    def __init__(self, geometry: Geometry, material: Material) -> None:
        """Compose a geometry and a material into a reusable solver."""
        self.geometry = geometry
        self.material = material

    def assemble(self, wavelength: float) -> np.ndarray:
        """Assemble the 2nn × 2nn BIE system matrix M(λ).

        The matrix depends only on the wavelength (and the fixed geometry and
        material), not on the excitation, so it can be factorised once and
        reused across many right-hand sides at the same wavelength — see
        :func:`pysie2d.green.relative_ldos_map` for the LU-reuse pattern.

        Args:
            wavelength: Free-space wavelength (nm).

        Returns:
            complex (2·n_pts, 2·n_pts) system matrix.
        """
        g = self.geometry
        mat = self.material
        wnum = 2.0 * PI / wavelength
        return assemble_matrix(
            mat.pol,
            g.n_pts,
            g.f,
            g.g,
            g.df,
            g.dg,
            g.ddf,
            g.ddg,
            g.delt,
            wnum,
            mat.nc,
            mat.eps,
        )

    def scatter(
        self,
        wavelength: float,
        angle: float = 0.0,
        incident_rhs: Callable[[int, float, np.ndarray, np.ndarray], np.ndarray]
        | None = None,
    ) -> ScatterResult:
        """Solve the BIE for one wavelength.

        Args:
            wavelength: Free-space wavelength (nm).
            angle: Plane-wave incidence angle (degrees). Default 0.
            incident_rhs: Custom incident-field callable with signature
                ``(nn, lambd, f, g) → complex (2*nn,)``. Replaces the default
                plane-wave excitation.

        Returns:
            ScatterResult carrying the solution vector and analysis methods.
        """
        g = self.geometry
        mat = self.material

        m = self.assemble(wavelength)

        if incident_rhs is not None:
            rhs = incident_rhs(g.n_pts, wavelength, g.f, g.g)
        else:
            rhs = plane_wave_rhs(g.n_pts, angle, wavelength, g.f, g.g)

        ei = np.linalg.solve(m, rhs)
        return ScatterResult(ei, g, mat, wavelength, angle)

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
            wavelength: Free-space wavelength (nm).
            x_s: Source x-coordinate (nm).
            z_s: Source z-coordinate (nm).

        Returns:
            ScatterResult carrying the scattered-field solution vector.

        Raises:
            ValueError: If the source is inside the particle or too close to
                its surface (see :func:`pysie2d.sources.line_dipole_rhs`).
        """
        rhs = functools.partial(line_dipole_rhs, x_s=x_s, z_s=z_s)
        return self.scatter(wavelength, incident_rhs=rhs)
