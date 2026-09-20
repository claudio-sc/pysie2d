"""Clusters of non-overlapping particles sharing one background.

Degree-of-freedom layout (conventions §14, D4): particle ``p`` occupies the
contiguous slice ``[o_p : o_p + 2·nn_p]`` of the coupled unknown vector, with
``o_p = Σ_{q<p} 2·nn_q``; within that slice the first ``nn_p`` entries are φ_p
and the next ``nn_p`` are χ_p. Contiguity is what lets each particle's
sub-vector go straight into the existing single-particle primitives unchanged.

The background wavenumber ``k_bg = 2π·n_clad/λ_vac`` is **common to the whole
cluster** (D3b): a cluster sits in exactly one background, so the per-particle
materials may differ in ``nc``/``eps`` (which are background-relative,
conventions §2) but not in ``n_clad``.

Overlapping boundaries are rejected rather than approximated. The coupled
boundary-integral formulation represents the field in the region exterior to
every particle and interior to exactly one; where two boundaries cross, that
partition does not exist and the unknowns φ, χ have no meaning. This is a limit
of the formulation, not of the implementation, so no amount of resolution
repairs it.
"""

import warnings
from collections.abc import Sequence

import numpy as np

from .fields import _cross_sections, _far_field_at, _representation_at
from .geometry import Geometry
from .kernels import _real_if_real, assemble_cross_block, assemble_matrix
from .material import Material
from .solver import wavelength_over_ds
from .sources import _point_inside, line_dipole_rhs, plane_wave_rhs

PI = np.pi

# A resolution spread wider than this between the best- and worst-resolved
# particle warns. The coupled system is only as accurate as its worst block,
# and that is not visible from any one particle. Four is two halvings of nn,
# which in the M2 pilot is the difference between round-off and 1e-6 at a
# fixed gap (docs/design/studies/cluster-gap-envelope.md).
RESOLUTION_SPREAD_WARN = 4.0


class ClusterOverlapError(ValueError):
    """Two particle boundaries intersect; the formulation does not apply."""


class ClusterGapWarning(UserWarning):
    """A gap is too small for the quadrature resolution in use."""


class ClusterResolutionWarning(UserWarning):
    """One particle is far more coarsely resolved than its neighbours."""


class Cluster:
    """An arrangement of non-overlapping particle boundaries.

    Geometry only: materials live on the solver (D2), so one arrangement can be
    swept over material sets without being rebuilt. Construction validates that
    no two boundaries intersect and records the smallest node-to-node gap, both
    of which are properties of the arrangement alone.

    Attributes:
        geometries: The particle boundaries, in DOF order.
        offsets: ``len(self) + 1`` cumulative DOF offsets, ``offsets[p] = o_p``.
        n_dof: Total number of unknowns ``N = 2·Σ_p nn_p``.
        min_gap: Smallest node-to-node distance between distinct particles
            (nm); ``inf`` for a single-particle cluster.
    """

    def __init__(self, geometries: Sequence[Geometry]) -> None:
        """Build a cluster from a sequence of boundaries.

        Args:
            geometries: One or more :class:`~pysie2d.geometry.Geometry`
                boundaries. Resolutions ``n_pts`` may differ between them.

        Raises:
            ValueError: If ``geometries`` is empty.
            ClusterOverlapError: If two boundaries intersect.
        """
        if len(geometries) == 0:
            raise ValueError("a Cluster needs at least one geometry")
        self.geometries = tuple(geometries)
        sizes = [2 * g.n_pts for g in self.geometries]
        self.offsets = tuple(np.cumsum([0] + sizes).tolist())
        self.n_dof = self.offsets[-1]
        self._check_overlap()
        self.min_gap = self._min_gap()

    def __len__(self) -> int:
        """Return the number of particles in the cluster."""
        return len(self.geometries)

    def slice(self, p: int) -> slice:
        """Return the ``2·nn_p`` slice of particle ``p`` (conventions §14).

        Args:
            p: Particle index.

        Returns:
            The slice ``[o_p : o_p + 2·nn_p]`` into a coupled vector of length
            ``n_dof``; its first half is φ_p and its second half χ_p.
        """
        return slice(self.offsets[p], self.offsets[p + 1])

    def _radii(self, p: int) -> tuple[float, float]:
        """Return the inscribed and circumscribed node radii of particle ``p``.

        Both are measured from the stored centre over the boundary *nodes*, not
        taken from ``Geometry.rad``: on a rounded square the superformula scale
        is 2.29× smaller than the true circumscribing radius, the same trap
        ``ScatterResult.multipoles`` guards against, and an overlap test built
        on it would accept intersecting boundaries.

        Args:
            p: Particle index.

        Returns:
            ``(r_insc, r_circ)`` in nm — the min and max node radius.
        """
        g = self.geometries[p]
        r = np.hypot(g.f - g.x0, g.g - g.z0)
        return float(r.min()), float(r.max())

    def _check_overlap(self) -> None:
        """Reject any pair of intersecting boundaries.

        Three bands per pair, cheapest first: separated circumscribing circles
        certainly do not intersect; overlapping inscribed circles certainly do;
        in between, every node of each boundary is ray-cast against the other.

        The test is a strong filter, not a decision procedure. **Two boundaries
        can cross in a lens-shaped sliver that contains no node of either**, and
        band 3 misses exactly that case. The arrangement's ``min_gap`` is then
        tiny, so the resolution-versus-gap envelope check (§4.9) warns — that
        warning is the backstop for this residual failure mode.

        Raises:
            ClusterOverlapError: If a pair is found to intersect.
        """
        for p in range(len(self)):
            gp = self.geometries[p]
            rp_in, rp_out = self._radii(p)
            for q in range(p + 1, len(self)):
                gq = self.geometries[q]
                rq_in, rq_out = self._radii(q)
                d = float(np.hypot(gq.x0 - gp.x0, gq.z0 - gp.z0))
                if rp_out + rq_out < d:
                    continue  # certainly disjoint
                if rp_in + rq_in > d:  # certainly overlapping
                    raise ClusterOverlapError(
                        f"particles {p} and {q} overlap: their inscribed "
                        f"circles ({rp_in:.4g} + {rq_in:.4g} nm) already "
                        f"exceed the centre separation {d:.4g} nm"
                    )
                hit = any(
                    _point_inside(x, z, gp.f, gp.g)
                    for x, z in zip(gq.f, gq.g, strict=True)
                ) or any(
                    _point_inside(x, z, gq.f, gq.g)
                    for x, z in zip(gp.f, gp.g, strict=True)
                )
                if hit:
                    raise ClusterOverlapError(
                        f"particles {p} and {q} overlap: a boundary node of "
                        f"one lies inside the other"
                    )

    def _min_gap(self) -> float:
        """Return the smallest node-to-node distance between particles (nm).

        This slightly **overestimates** the true surface-to-surface gap: the
        closest approach of two smooth boundaries generally falls between
        nodes, so the discrete minimum is an upper bound that tightens as
        ``n_pts`` grows. It is used as a diagnostic scale, where an error of
        order the node spacing is immaterial.

        Cost is ``O((Σ nn)²)`` — 9e6 distances at ``Np = 10``, ``nn = 300``,
        milliseconds, far below one assembly — so it is computed once in
        ``__init__``.

        Returns:
            The minimum distance in nm, or ``inf`` for a single particle.
        """
        best = np.inf
        for p in range(len(self)):
            gp = self.geometries[p]
            for q in range(p + 1, len(self)):
                gq = self.geometries[q]
                d = np.hypot(
                    gq.f[:, None] - gp.f[None, :], gq.g[:, None] - gp.g[None, :]
                )
                best = min(best, float(d.min()))
        return best


class ClusterBIESolver:
    """Coupled boundary-integral solver for a finite cluster of particles.

    The diagonal blocks are the single-particle matrix, :func:`~pysie2d.kernels
    .assemble_matrix`, called verbatim at each particle's own ``nn_p``, ``nc_p``
    and ``eps_p``; the off-diagonal blocks couple particles through the
    background Green function alone and so have a zero lower half
    (docs/design/multiparticle-spec.md §3.1). The whole v0.6 Kress–Martensen
    validation therefore carries over untouched.

    ``pol`` is an argument of the *problem*, not of a particle (conventions
    §1), and the background index ``n_clad`` is a property of the *cluster*
    (D3b): both are validated equal across the materials, because a
    disagreement in either is a meaningless coupled system that nothing else
    would notice. The background-relative ``nc`` and ``eps`` (conventions §2)
    may differ freely between particles.

    There is deliberately no public ``assemble`` (D12): the only caller for one
    would be a cluster quasi-normal-mode search, and the half-plane argument of
    conventions §8 has not been re-examined for a cluster.

    Attributes:
        cluster: The arrangement being solved.
        materials: Per-particle optical properties, in DOF order.
        pol: Polarisation: 1 = TM (H_y), 2 = TE (E_y).
    """

    def __init__(
        self,
        cluster: Cluster,
        materials: Sequence[Material],
        pol: int = 2,
    ) -> None:
        """Bind materials and a polarisation to an arrangement.

        Args:
            cluster: The particle arrangement.
            materials: One :class:`~pysie2d.material.Material` per particle, in
                the cluster's own order.
            pol: Polarisation: 1 = TM, 2 = TE.

        Raises:
            ValueError: If the number of materials does not match the number of
                particles, if any material's ``pol`` differs from ``pol``, or
                if the materials disagree on ``n_clad``.
        """
        if len(materials) != len(cluster):
            raise ValueError(f"{len(materials)} materials for {len(cluster)} particles")
        for i, mat in enumerate(materials):
            if mat.pol != pol:
                raise ValueError(
                    f"material {i} has pol = {mat.pol}, but the cluster is "
                    f"being solved at pol = {pol}. Polarisation is a property "
                    f"of the problem, not of a particle (conventions sec. 1)."
                )
            if mat.n_clad != materials[0].n_clad:
                raise ValueError(
                    f"material {i} has n_clad = {mat.n_clad}, material 0 has "
                    f"{materials[0].n_clad}. A cluster has one background, and "
                    f"k_bg = 2*pi*n_clad/lambda_vac must be unambiguous. "
                    f"Per-particle optical properties go in n_core and epsi, "
                    f"which enter as the background-relative nc and eps "
                    f"(conventions sec. 2)."
                )
        self.cluster = cluster
        self.materials = tuple(materials)
        self.pol = pol

    def _assemble(self, wavelength: float | complex) -> np.ndarray:
        """Build the coupled system matrix at one wavelength.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm. May be complex.

        Returns:
            complex ``(n_dof, n_dof)`` matrix in the layout of conventions §14.
        """
        cl = self.cluster
        k_bg = _real_if_real(self.materials[0].wnum_bg(wavelength))
        me = np.zeros((cl.n_dof, cl.n_dof), dtype=complex)

        # Each task writes a disjoint block, so this loop is embarrassingly
        # parallel, and hank0/hank1 both release the GIL — threading it is the
        # proven 5.02x pattern of contour_moments (docs/design/performance.md
        # §3.1). Serial until the Np ceiling study says otherwise (D8); the
        # task list is the seam that keeps that a one-line change.
        tasks = [("self", p, p) for p in range(len(cl))]
        tasks += [
            ("cross", q, p) for q in range(len(cl)) for p in range(len(cl)) if q != p
        ]

        for kind, q, p in tasks:
            gp = cl.geometries[p]
            if kind == "self":
                mat = self.materials[p]
                me[cl.slice(p), cl.slice(p)] = assemble_matrix(
                    self.pol,
                    gp.n_pts,
                    gp.f,
                    gp.g,
                    gp.df,
                    gp.dg,
                    gp.ddf,
                    gp.ddg,
                    k_bg,
                    mat.nc,
                    mat.eps,
                )
                continue
            gq = cl.geometries[q]
            m1, m2 = assemble_cross_block(
                k_bg, gp.n_pts, gp.f, gp.g, gp.df, gp.dg, gq.f, gq.g
            )
            r0 = cl.offsets[q]
            c0 = cl.offsets[p]
            me[r0 : r0 + gq.n_pts, c0 : c0 + gp.n_pts] = m1
            me[r0 : r0 + gq.n_pts, c0 + gp.n_pts : c0 + 2 * gp.n_pts] = m2
            # The lower half of a cross block stays at its initialised zero:
            # that is the M3/M4 result of §3.1, not an omission. Particle p's
            # interior Green function is confined to p's own volume, so
            # coupling enters only through the exterior background kernel.
        return me

    def scatter(self, wavelength: float, angle: float = 0.0) -> "ClusterScatterResult":
        """Solve the coupled system for plane-wave excitation.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm.
            angle: Incidence angle in degrees; 0 propagates along −z.

        Returns:
            The :class:`ClusterScatterResult` carrying the coupled solution.

        Warns:
            ClusterResolutionWarning: If the particles' resolutions differ by
                more than ``RESOLUTION_SPREAD_WARN``.
        """
        self._check_resolution(wavelength)
        cl = self.cluster
        k_bg = self.materials[0].wnum_bg(wavelength)
        rhs = np.zeros(cl.n_dof, dtype=complex)
        for p, gp in enumerate(cl.geometries):
            rhs[cl.slice(p)] = plane_wave_rhs(gp.n_pts, angle, k_bg, gp.f, gp.g)
        ei = np.linalg.solve(self._assemble(wavelength), rhs)
        return ClusterScatterResult(
            ei, cl, self.materials, self.pol, wavelength, angle, "plane_wave"
        )

    def scatter_dipole(
        self, wavelength: float, x_s: float, z_s: float
    ) -> "ClusterScatterResult":
        """Solve the coupled system for a line-dipole source.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm.
            x_s: Source x-coordinate (nm).
            z_s: Source z-coordinate (nm).

        Returns:
            The :class:`ClusterScatterResult` carrying the coupled solution.

        Raises:
            ValueError: If the source lies inside, or too close to, any
                particle — raised by :func:`~pysie2d.sources.line_dipole_rhs`.

        Warns:
            ClusterResolutionWarning: If the particles' resolutions differ by
                more than ``RESOLUTION_SPREAD_WARN``.
        """
        self._check_resolution(wavelength)
        cl = self.cluster
        k_bg = self.materials[0].wnum_bg(wavelength)
        rhs = np.zeros(cl.n_dof, dtype=complex)
        for p, gp in enumerate(cl.geometries):
            # line_dipole_rhs's own inside/too-close guards run once per
            # particle, which is exactly the cluster condition: the source must
            # lie outside every particle. No extra check is needed here and
            # none should be added.
            rhs[cl.slice(p)] = line_dipole_rhs(gp.n_pts, k_bg, gp.f, gp.g, x_s, z_s)
        ei = np.linalg.solve(self._assemble(wavelength), rhs)
        return ClusterScatterResult(
            ei, cl, self.materials, self.pol, wavelength, 0.0, "dipole"
        )

    def _check_resolution(self, wavelength: float | complex) -> None:
        """Warn when one particle is far coarser than its best-resolved peer.

        The user picks ``n_pts`` per particle (D7) — that freedom is the point
        of ragged resolution — but the coupled system is only as accurate as
        its worst block, and a particle that is under-resolved for its own size
        and index cannot be spotted from any other particle's numbers.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm.
        """
        res = [
            wavelength_over_ds(g, m, wavelength)
            for g, m in zip(self.cluster.geometries, self.materials, strict=True)
        ]
        if len(res) < 2:
            return
        worst = int(np.argmin(res))
        best = int(np.argmax(res))
        if res[best] <= RESOLUTION_SPREAD_WARN * res[worst]:
            return
        warnings.warn(
            f"particle {worst} is resolved at {res[worst]:.3g} points per "
            f"interior wavelength while particle {best} reaches "
            f"{res[best]:.3g}; the coupled solution is only as accurate as its "
            f"worst block",
            ClusterResolutionWarning,
            stacklevel=3,
        )


class ClusterScatterResult:
    """Result of a single-wavelength coupled cluster solve.

    Observables are additive over boundaries, which is what makes the façade
    thin: each particle's ``2·nn_p`` sub-vector (conventions §14) goes straight
    into the existing single-particle primitives and the contributions are
    summed. The coordinates ``f_p``, ``g_p`` are **absolute**, so each
    particle's geometric phase is already inside the far-field integrand and
    there is no extra phase factor to apply (§3.4) — a tempting and wrong
    addition.

    There is deliberately no ``efficiencies()``: it normalises by ``2·rad``,
    and a cluster has no ``rad`` (D5); ``cross_sections()`` is the absolute
    quantity that replaces it. There is deliberately no ``multipoles()``: the
    v0.7 expansion is about one particle centre, and a cluster needs an
    output-side translation theorem that ``multipole.py`` does not implement
    (§1.3). Both absences are decisions, not omissions.

    Attributes:
        ei: complex ``(n_dof,)`` coupled solution vector.
        cluster: The arrangement solved.
        materials: Per-particle optical properties.
        pol: Polarisation: 1 = TM, 2 = TE.
        wavelength: Vacuum wavelength λ_vac in nm.
        angle: Plane-wave incidence angle in degrees (0.0 for a dipole solve).
    """

    def __init__(
        self,
        ei: np.ndarray,
        cluster: Cluster,
        materials: Sequence[Material],
        pol: int,
        wavelength: float,
        angle: float,
        excitation: str,
    ) -> None:
        """Store a coupled solution and the problem that produced it.

        Args:
            ei: complex ``(n_dof,)`` coupled solution vector.
            cluster: The arrangement solved.
            materials: One material per particle, in the cluster's order.
            pol: Polarisation: 1 = TM, 2 = TE.
            wavelength: Vacuum wavelength λ_vac in nm.
            angle: Plane-wave incidence angle in degrees.
            excitation: ``"plane_wave"`` or ``"dipole"``.
        """
        self.ei = ei
        self.cluster = cluster
        self.materials = tuple(materials)
        self.pol = pol
        self.wavelength = wavelength
        self.angle = angle
        self._excitation = excitation

    @property
    def wnum_bg(self) -> complex:
        """Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm)."""
        return self.materials[0].wnum_bg(self.wavelength)

    def ei_particle(self, p: int) -> np.ndarray:
        """Return particle ``p``'s own sub-vector, in conventions §4 order.

        Args:
            p: Particle index.

        Returns:
            complex ``(2·nn_p,)`` slice: φ_p first, then χ_p — exactly what the
            single-particle primitives take.
        """
        return self.ei[self.cluster.slice(p)]

    def far_field(self, n_angles: int = 3000) -> tuple[np.ndarray, np.ndarray]:
        """Total far-field amplitude of the cluster.

        Args:
            n_angles: Number of observation angles over [−π, π] inclusive.

        Returns:
            ``(amp, angles)``: complex ``(n_angles,)`` amplitude and the
            observation angles in radians, measured from +z.
        """
        angles = -PI + np.arange(n_angles) * 2.0 * PI / (n_angles - 1.0)
        return self._amp_at(angles), angles

    def _amp_at(self, angles: np.ndarray) -> np.ndarray:
        """Far-field amplitude at arbitrary angles.

        Args:
            angles: (M,) observation angles in radians.

        Returns:
            complex (M,) total amplitude.
        """
        # The absolute coordinates f_p, g_p already carry each particle's
        # geometric phase inside exp(-i k (f sinθ + g cosθ)); adding one here
        # would apply it twice (§3.4).
        k = self.wnum_bg
        total = np.zeros(len(angles), dtype=complex)
        for p, gp in enumerate(self.cluster.geometries):
            total += _far_field_at(
                angles,
                gp.n_pts,
                k,
                gp.f,
                gp.g,
                gp.df,
                gp.dg,
                gp.delt,
                self.ei_particle(p),
            )
        return total

    def cross_sections(self, n_angles: int = 500) -> dict[str, float]:
        """Absolute scattering, extinction and absorption cross-sections (nm).

        The cluster observable of §3.6, and the same quantity
        :meth:`pysie2d.solver.ScatterResult.cross_sections` reports for one
        particle, so the two paths are comparable.

        Args:
            n_angles: Far-field grid size for the C_sca quadrature.

        Returns:
            ``{"c_sca", "c_ext", "c_abs"}`` in nm.

        Raises:
            ValueError: If this result did not come from plane-wave excitation.
        """
        if self._excitation != "plane_wave":
            raise ValueError(
                "cross_sections() is defined against a unit-amplitude "
                "incident plane wave; this result came from "
                f"{self._excitation} excitation"
            )
        amp, _ = self.far_field(n_angles)
        # π − α lands on the far-field grid only by accident, so the forward
        # direction is evaluated exactly rather than picked off it.
        amp_fwd = self._amp_at(np.array([PI - np.deg2rad(self.angle)]))[0]
        return _cross_sections(amp, amp_fwd, self.wnum_bg)

    def eval_field(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate the field at arbitrary points, inside or outside.

        Outside every particle the field is the sum of the representation
        integral over all boundaries at ``k_bg``; inside particle ``p`` it is
        that boundary alone at ``k_core,p = nc_p·k_bg``. Overlap being
        excluded, a point is inside at most one particle.

        Classification uses ray casting (:func:`pysie2d.sources._point_inside`)
        rather than the nearest-point normal test
        :func:`pysie2d.fields._is_outside` that
        :meth:`pysie2d.solver.ScatterResult.eval_field` uses — the latter is
        documented unreliable for concave superformula shapes, and a cluster
        runs the test once per particle, so a misclassification is that many
        times more likely (§3.5).

        Never evaluate on or very near a boundary: the representation-formula
        integrand is near-singular there. Keep points at least ~5 node spacings
        clear of every particle.

        Args:
            x: Observation x-coordinates (nm); any shape, flattened.
            z: Observation z-coordinates (nm); same length as ``x``.

        Returns:
            complex (M,) field at each point.
        """
        x = np.asarray(x, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        k = _real_if_real(self.wnum_bg)
        inside = np.full(len(x), -1, dtype=int)
        for p, gp in enumerate(self.cluster.geometries):
            for j in range(len(x)):
                if inside[j] < 0 and _point_inside(x[j], z[j], gp.f, gp.g):
                    inside[j] = p
        field = np.zeros(len(x), dtype=complex)
        out = inside < 0
        if out.any():
            for p, gp in enumerate(self.cluster.geometries):
                field[out] += _representation_at(
                    self.ei_particle(p),
                    gp.n_pts,
                    gp.f,
                    gp.df,
                    gp.g,
                    gp.dg,
                    gp.delt,
                    k,
                    x[out],
                    z[out],
                )
        for p, gp in enumerate(self.cluster.geometries):
            sel = inside == p
            if sel.any():
                k_core = _real_if_real(self.materials[p].nc * self.wnum_bg)
                field[sel] = _representation_at(
                    self.ei_particle(p),
                    gp.n_pts,
                    gp.f,
                    gp.df,
                    gp.g,
                    gp.dg,
                    gp.delt,
                    k_core,
                    x[sel],
                    z[sel],
                )
        return field

    def resolution(self) -> tuple[float, ...]:
        """Points per interior wavelength, per particle (D7).

        Returns:
            One :func:`~pysie2d.solver.wavelength_over_ds` value per particle,
            in the cluster's order.
        """
        return tuple(
            wavelength_over_ds(g, m, self.wavelength)
            for g, m in zip(self.cluster.geometries, self.materials, strict=True)
        )
