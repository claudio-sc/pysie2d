"""QNMSolver / QNMResult façade for quasi-normal-mode extraction.

A quasi-normal mode is a source-free solution: a complex wavelength where the
BIE operator M(λ) is singular. :class:`QNMSolver` finds every one inside a
rectangle of the complex λ-plane by contour integration (:mod:`pysie2d.beyn`),
with no initial guess and no scan.

Mirrors the driven solver: same ``(geometry, material)`` construction, one
method returning a result object, derived physics as properties.

Conventions (docs/conventions.md §8). Wavelengths are vacuum nm. Under
``exp(-iωt)`` a decaying mode has ``Im ω < 0``, hence ``Im k < 0``, hence
**``Im λ > 0``** — a search box must lie in the upper half λ-plane, and in
``Re λ > 0`` so that every Hankel argument stays off the ``H^(1)`` branch cut.
Both are asserted. Poles do *not* come in conjugate pairs: the reality
condition is ``λ → −λ̄``, which puts mirror partners at negative ``Re λ``.
"""

from dataclasses import dataclass

import numpy as np

from .beyn import beyn_modes
from .geometry import Geometry
from .material import Material
from .solver import BIESolver

# Two eigenvalues closer than this, relatively, are the same mode: the ±n
# partners of a circle. *(measured: a true degenerate pair agrees to 2.4e-13,
# while the nearest distinct mode in the same spectrum is 20 nm away, so the
# grouping is not a close call.)*
DEGENERACY_RTOL = 1.0e-9


@dataclass(frozen=True)
class QNMResult:
    """Quasi-normal modes found inside a search rectangle.

    Attributes:
        wavelengths: complex (K,) mode wavelengths in vacuum nm, sorted by
            ``Re λ``. Degenerate partners appear as separate, numerically
            equal entries — see ``multiplicity``.
        vectors: complex (2·n_pts, K) boundary-field vectors of each mode, unit
            columns in the ``φ`` / ``χ`` layout of the driven solver. Not
            normalised as mode fields; that needs a QNM norm, which is out of
            scope (v0.5).
        multiplicity: int (K,) how many entries share this wavelength. Every
            ``n ≥ 1`` mode of a circle is doubly degenerate through
            ``exp(±inθ)``; only ``n = 0`` is simple.
        sigma_ratio: float (K,) ``σ_min/σ_max`` of M at each mode. The
            singularity measure, and dimensionless — an absolute ``σ_min``
            means nothing, since it carries the scale of the operator.
        sv_ratio: float (n_probe,) singular-value spectrum of the Beyn moment
            matrix, divided by its largest. The rank gap lives here.
        max_gap: float largest ratio between consecutive singular values.
        rank: int rank detected in the moment matrix. This can exceed
            ``n_modes``: a pole just *outside* the contour leaks a rank
            direction in through imperfect quadrature cancellation, and its
            eigenvalue is then discarded by the in-contour filter.
        cancellation: float how completely the contour integral cancelled.
        edge_margin: float (K,) distance from each mode to the nearest contour
            edge, as a fraction of the shorter side. A value near zero means
            the box is clipping the pole and the result is not trustworthy.
        z_lo: complex bottom-left corner of the search rectangle.
        z_hi: complex top-right corner.
        geometry: The scatterer boundary.
        material: The scatterer optical properties.
    """

    wavelengths: np.ndarray
    vectors: np.ndarray
    multiplicity: np.ndarray
    sigma_ratio: np.ndarray
    sv_ratio: np.ndarray
    max_gap: float
    rank: int
    cancellation: float
    edge_margin: np.ndarray
    z_lo: complex
    z_hi: complex
    geometry: Geometry
    material: Material

    @property
    def quality_factors(self) -> np.ndarray:
        """Q = Re λ / (2 Im λ), exactly equal to −Re ω / (2 Im ω)."""
        return self.wavelengths.real / (2.0 * self.wavelengths.imag)

    @property
    def n_modes(self) -> int:
        """Number of modes found inside the rectangle."""
        return int(self.wavelengths.size)


class QNMSolver:
    """Quasi-normal-mode solver for a single particle in a homogeneous background.

    Attributes:
        geometry: Discretized particle boundary.
        material: Optical properties of the scatterer.

    Examples:
        >>> geom = Geometry.gielis(rad=200, n_pts=200, m=0)
        >>> mat = Material(n_core=3.0, n_clad=1.0, pol=2)
        >>> qnm = QNMSolver(geom, mat)
        >>> res = qnm.modes(z_lo=520 + 15j, z_hi=545 + 40j)
        >>> res.wavelengths
        >>> res.quality_factors
    """

    def __init__(self, geometry: Geometry, material: Material) -> None:
        """Compose a geometry and a material into a reusable mode solver."""
        self.geometry = geometry
        self.material = material
        # Deliberately the driven solver's own assembly rather than a parallel
        # path: a QNM is a singularity of the *scattering* operator, and that
        # claim only holds if the two are literally the same matrix.
        self._bie = BIESolver(geometry, material)

    def modes(
        self,
        z_lo: complex,
        z_hi: complex,
        *,
        n_quad_per_side: int = 12,
        n_probe: int = 12,
        rank_tol: float = 1.0e-8,
        rng_seed: int = 0,
    ) -> QNMResult:
        """Find every quasi-normal mode inside a rectangle of the λ-plane.

        Cost is ``4·n_quad_per_side`` complex assemblies and LU factorisations,
        independent of how many modes are inside.

        Args:
            z_lo: Bottom-left corner of the search rectangle, vacuum nm.
            z_hi: Top-right corner, vacuum nm.
            n_quad_per_side: Gauss-Legendre nodes per contour edge. The
                default resolves the contour integral far below the
                discretisation error in ``n_pts`` *(measured: identical modes
                to 1e-8 nm from 6 nodes per side upward, against a 0.38 nm
                discretisation error at n_pts = 200)*.
            n_probe: Probe columns; must exceed the number of modes inside,
                counting rank leaked from poles just outside.
            rank_tol: Relative singular-value floor for rank detection.
            rng_seed: Probe-matrix seed. Results must not depend on it.

        Returns:
            The modes inside the rectangle, with their diagnostics.

        Raises:
            ValueError: If the rectangle is not strictly inside the physical
                quadrant ``Re λ > 0``, ``Im λ > 0``, or if the modes inside
                outnumber ``n_probe``.
        """
        self._validate_region(z_lo, z_hi)

        found = beyn_modes(
            self._bie.assemble,
            z_lo,
            z_hi,
            n_quad_per_side=n_quad_per_side,
            n_probe=n_probe,
            rank_tol=rank_tol,
            rng_seed=rng_seed,
        )
        lams = found.eigenvalues

        return QNMResult(
            wavelengths=lams,
            vectors=found.vectors,
            multiplicity=_multiplicity(lams),
            sigma_ratio=np.array([self._sigma_ratio(lam) for lam in lams]),
            sv_ratio=found.sv_ratio,
            max_gap=found.max_gap,
            rank=found.rank,
            cancellation=found.cancellation,
            edge_margin=_edge_margin(lams, z_lo, z_hi),
            z_lo=z_lo,
            z_hi=z_hi,
            geometry=self.geometry,
            material=self.material,
        )

    def _sigma_ratio(self, wavelength: complex) -> float:
        """σ_min/σ_max of M at one wavelength.

        A full SVD of the 2·n_pts matrix, run once per mode. Inverse iteration
        on the factorisation would be ~100× cheaper and is worth doing once
        refinement exists to supply the LU; with a handful of modes the full
        SVD is not the cost that matters.
        """
        s = np.linalg.svd(self._bie.assemble(wavelength), compute_uv=False)
        return float(s[-1] / s[0])

    @staticmethod
    def _validate_region(z_lo: complex, z_hi: complex) -> None:
        """Reject rectangles outside the physical quadrant.

        ``Im λ > 0`` is the decaying half-plane under ``exp(-iωt)``; a box
        below it searches for growing modes. ``Re λ > 0`` keeps every Hankel
        argument off the ``H^(1)`` branch cut on the negative real axis, which
        is what makes M(λ) holomorphic on the rectangle — and holomorphy is the
        premise of the whole contour argument, not a detail.
        """
        if z_hi.real <= z_lo.real or z_hi.imag <= z_lo.imag:
            raise ValueError(
                f"z_hi must be strictly above and right of z_lo; "
                f"got z_lo={z_lo}, z_hi={z_hi}"
            )
        # Exception text is ASCII on purpose: it may be printed to a cp1252
        # console, where a bare "λ" raises UnicodeEncodeError and hides the
        # actual error. Docstrings keep the Unicode notation.
        if z_lo.real <= 0.0:
            raise ValueError(
                f"search region must lie in Re(lambda) > 0 to keep Hankel "
                f"arguments off the H^(1) branch cut; got Re z_lo = {z_lo.real}"
            )
        if z_lo.imag <= 0.0:
            raise ValueError(
                f"search region must lie in Im(lambda) > 0: under exp(-i.omega.t) "
                f"a decaying mode has Im(lambda) > 0, so a box reaching "
                f"Im(lambda) = {z_lo.imag} is searching for growing modes"
            )


def _multiplicity(lams: np.ndarray) -> np.ndarray:
    """Count how many entries share each wavelength.

    Degenerate partners are reported separately rather than collapsed: the ±n
    pair carries two independent mode vectors, and merging them would make the
    mode count disagree with the analytic one.
    """
    if lams.size == 0:
        return np.zeros(0, dtype=int)
    close = np.abs(lams[:, None] - lams[None, :]) <= DEGENERACY_RTOL * np.abs(
        lams[None, :]
    )
    return close.sum(axis=1).astype(int)


def _edge_margin(lams: np.ndarray, z_lo: complex, z_hi: complex) -> np.ndarray:
    """Distance to the nearest contour edge, as a fraction of the shorter side."""
    if lams.size == 0:
        return np.zeros(0, dtype=float)
    side = min(z_hi.real - z_lo.real, z_hi.imag - z_lo.imag)
    dist = np.minimum.reduce(
        [
            lams.real - z_lo.real,
            z_hi.real - lams.real,
            lams.imag - z_lo.imag,
            z_hi.imag - lams.imag,
        ]
    )
    return dist / side
