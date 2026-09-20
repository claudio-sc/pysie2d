"""The `Cluster` container, its geometric guards, and the coupled solver.

A cluster is a purely geometric object (spec D2): it fixes the coupled
degree-of-freedom layout of conventions §14 and rejects arrangements the
boundary-integral formulation cannot represent. These tests pin both — the
layout arithmetic under *ragged* `n_pts`, where an off-by-one offset silently
mixes one particle's φ with another's χ, and the three-band overlap test in
each of its bands.
"""

import warnings

import numpy as np
import pytest

from pysie2d import (
    Cluster,
    ClusterBIESolver,
    ClusterOverlapError,
    ClusterResolutionWarning,
    Geometry,
    Material,
)

RAD = 100.0


def _circle(n_pts: int, x0: float = 0.0, z0: float = 0.0) -> Geometry:
    return Geometry.gielis(RAD, n_pts, m=0, x0=x0, z0=z0)


def test_dof_layout_is_contiguous_per_particle_with_ragged_resolutions():
    """Offsets and slices follow conventions §14 for unequal `n_pts`.

    Ragged resolutions are the case the layout exists for: with equal `n_pts`
    almost any offset rule happens to work, so the test uses three different
    ones. Each slice must have length exactly 2·nn_p and the slices must tile
    range(n_dof) without gap or overlap — a property that cannot hold by
    accident if any offset is wrong, since a single misplaced boundary both
    shortens one slice and lengthens its neighbour.
    """
    n_pts = (40, 61, 33)
    cluster = Cluster([_circle(n, x0=400.0 * i) for i, n in enumerate(n_pts)])

    assert len(cluster) == 3
    assert cluster.n_dof == 2 * sum(n_pts)
    assert cluster.offsets == (0, 80, 202, 268)

    covered = np.zeros(cluster.n_dof, dtype=int)
    for p, n in enumerate(n_pts):
        sl = cluster.slice(p)
        assert sl.stop - sl.start == 2 * n
        covered[sl] += 1
    assert np.all(covered == 1)


def test_single_particle_cluster_is_allowed():
    """One particle is a legal cluster; §6.1 compares it to `BIESolver`.

    There is no pair, so no overlap test and no finite gap.
    """
    cluster = Cluster([_circle(40)])
    assert len(cluster) == 1
    assert cluster.n_dof == 80
    assert cluster.slice(0) == slice(0, 80)
    assert cluster.min_gap == np.inf


def test_empty_cluster_is_rejected():
    """An empty cluster has no unknowns and no background to scatter in."""
    with pytest.raises(ValueError, match="at least one geometry"):
        Cluster([])


def test_band_one_clearly_disjoint_particles_are_accepted():
    """Separated circumscribing circles end the test at band 1, silently.

    Two circles of radius 100 nm whose centres are 500 nm apart cannot touch;
    accepting them is the common case and must cost no warning.
    """
    cluster = Cluster([_circle(60), _circle(60, x0=500.0)])
    assert len(cluster) == 2


def test_band_two_overlapping_inscribed_circles_raise():
    """Interpenetrating circles are rejected before any node is sampled.

    Centres 150 nm apart with radii 100 nm each: the inscribed circles alone
    already overlap, so the cheap band-2 test must catch it. The error names
    both particles, because in a ten-particle cluster the index is the only
    way to find the offender.
    """
    with pytest.raises(ClusterOverlapError, match="particles 0 and 1 overlap"):
        Cluster([_circle(60), _circle(60, x0=150.0)])


def test_band_three_node_inside_the_other_boundary_raises():
    """A crossing missed by both circle tests is caught by node sampling.

    Two perpendicular ellipses (semi-axes 200 × 100 nm, the second rotated a
    quarter turn) with centres 250 nm apart. Their inscribed radii sum to
    200 < 250, so band 2 accepts; their circumscribed radii sum to 400 > 250,
    so band 1 does not accept. Only the ray-cast of band 3 sees that the tips
    interpenetrate — which is why this configuration, and not a pair of
    circles, is the one that exercises it.
    """
    wide = Geometry.gielis(RAD, 120, m=4, b=2.0)
    tall = Geometry.gielis(RAD, 120, m=4, a=2.0, x0=250.0)
    with pytest.raises(ClusterOverlapError, match="boundary node"):
        Cluster([wide, tall])


def test_min_gap_matches_the_analytic_surface_separation():
    """`min_gap` recovers the true gap from above, to the node spacing.

    Two circles of radius 100 nm with centres 300 nm apart have a surface gap
    of exactly 100 nm along the line of centres. The node-to-node minimum can
    only overestimate it: the closest approach generally falls between nodes,
    and at n_pts = 120 the two boundaries each contribute at most
    RAD·(1 − cos(π/n_pts)) ≈ 0.034 nm of sagitta, so 1e-3 relative is a bound
    on that geometric slop and not a fitted tolerance. The one-sidedness is
    checked separately — an underestimate would mean the guard is measuring
    something else entirely.
    """
    cluster = Cluster([_circle(120), _circle(120, x0=300.0)])
    assert cluster.min_gap >= 100.0
    assert cluster.min_gap == pytest.approx(100.0, rel=1e-3)


# ---------------------------------------------------------------------------
# ClusterBIESolver — construction guards, assembly, and the coupled solve
# ---------------------------------------------------------------------------


def _dimer(n_pts_a: int = 60, n_pts_b: int = 60) -> Cluster:
    """The asymmetric working pair: unequal radii, 300 nm centre separation."""
    return Cluster(
        [
            Geometry.gielis(180.0, n_pts_a, m=0),
            Geometry.gielis(110.0, n_pts_b, m=0, x0=520.0),
        ]
    )


def test_mismatched_polarisation_names_the_offending_material():
    """`pol` belongs to the problem, so the materials cannot disagree (D3).

    Polarisation selects which physical field the unknowns represent
    (conventions §1). A cluster assembled from a TE and a TM material is not a
    harder problem, it is not a problem at all — and nothing downstream would
    notice, so this must raise at construction and name the index, which in a
    ten-particle cluster is the only way to find the offender.
    """
    cluster = _dimer()
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=1)]
    with pytest.raises(ValueError, match="material 1 has pol = 1"):
        ClusterBIESolver(cluster, mats, pol=2)


def test_mismatched_background_index_is_rejected():
    """A cluster has exactly one background, so `n_clad` must agree (D3b).

    `k_bg = 2π·n_clad/λ_vac` is shared by every cross-block; with two values in
    play the coupling wavenumber is ambiguous and every off-diagonal block is
    meaningless. The error must point at `n_core`/`epsi` as the place
    per-particle optical contrast belongs, since that is the mistake being
    made.
    """
    cluster = _dimer()
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.4, pol=2)]
    with pytest.raises(ValueError, match="material 1 has n_clad = 1.4"):
        ClusterBIESolver(cluster, mats, pol=2)


def test_material_count_must_match_the_particle_count():
    """One material per particle; a short list would silently mis-pair them."""
    cluster = _dimer()
    with pytest.raises(ValueError, match="1 materials for 2 particles"):
        ClusterBIESolver(cluster, [Material(2.0, 1.0, pol=2)], pol=2)


def test_complex_wavelength_reaches_the_cluster_assembly():
    """A complex λ survives every cross-block (CLAUDE.md non-negotiable 1).

    The cross-block kernel is written once and must take a complex wavenumber
    unchanged — no cluster QNM search ships, but a path that quietly becomes
    real-only is exactly how the complex branch dies unnoticed. Assembling at
    λ = 600 − 20i must give a finite matrix of the full coupled size whose
    off-diagonal blocks are genuinely populated; a real-only fast path would
    either raise inside the Hankel calls or return NaNs here.
    """
    cluster = _dimer(40, 50)
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    me = ClusterBIESolver(cluster, mats, pol=2)._assemble(600.0 - 20.0j)

    assert me.shape == (cluster.n_dof, cluster.n_dof)
    assert me.dtype == complex
    assert np.all(np.isfinite(me))
    # The (1, 0) cross-block: upper half populated, lower half exactly zero —
    # the M3/M4 result of the spec's §3.1, which holds at complex k too.
    r0, c0 = cluster.offsets[1], cluster.offsets[0]
    nn_q, nn_p = 50, 40
    assert np.any(me[r0 : r0 + nn_q, c0 : c0 + 2 * nn_p] != 0.0)
    assert np.all(me[r0 + nn_q : r0 + 2 * nn_q, c0 : c0 + 2 * nn_p] == 0.0)


def test_resolution_imbalance_warns_and_names_the_worst_particle():
    """The coupled answer is only as accurate as its worst block (D7).

    A 180 nm particle at n_pts = 20 and a 110 nm one at n_pts = 400 differ by
    far more than `RESOLUTION_SPREAD_WARN` in points per interior wavelength.
    Nothing about particle 1's own numbers reveals that particle 0 is starving
    the solution, which is why the warning exists and why it must name the
    index and both readings.
    """
    cluster = Cluster(
        [
            Geometry.gielis(180.0, 20, m=0),
            Geometry.gielis(110.0, 400, m=0, x0=520.0),
        ]
    )
    mats = [Material(2.0, 1.0, pol=2), Material(2.0, 1.0, pol=2)]
    solver = ClusterBIESolver(cluster, mats, pol=2)
    with pytest.warns(ClusterResolutionWarning, match="particle 0 is resolved"):
        solver.scatter(wavelength=633.0)


def test_balanced_cluster_solves_without_warning():
    """Two comparably resolved particles must cost no warning at all.

    The mirror of the test above: a guard that fires on the ordinary case is
    worse than no guard, because users learn to filter it.
    """
    cluster = _dimer(80, 60)
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    solver = ClusterBIESolver(cluster, mats, pol=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver.scatter(wavelength=633.0, angle=37.0)


def test_plane_wave_solve_returns_finite_observables_of_the_right_shape():
    """Smoke test of the coupled plane-wave path end to end.

    Shapes and finiteness only — the far field is anchored against the
    addition theorem in `test_two_cylinder.py` and against `BIESolver` by the
    Np = 1 reduction; this checks that every façade method runs on a genuine
    two-particle solution and returns the advertised sizes.
    """
    cluster = _dimer(60, 50)
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    res = ClusterBIESolver(cluster, mats, pol=2).scatter(wavelength=633.0, angle=37.0)

    assert res.ei.shape == (cluster.n_dof,)
    assert np.all(np.isfinite(res.ei))
    assert res.ei_particle(0).shape == (120,)
    assert res.ei_particle(1).shape == (100,)

    amp, angles = res.far_field(201)
    assert amp.shape == angles.shape == (201,)
    assert np.all(np.isfinite(amp))

    cs = res.cross_sections(201)
    assert np.isfinite(cs["c_sca"]) and cs["c_sca"] > 0.0
    assert cs["c_abs"] == pytest.approx(cs["c_ext"] - cs["c_sca"])

    # One point outside both particles, one well inside each: three different
    # wavenumbers are used and all three branches must return finite values.
    field = res.eval_field([1500.0, 0.0, 520.0], [1500.0, 0.0, 0.0])
    assert field.shape == (3,)
    assert np.all(np.isfinite(field))
    assert np.all(field != 0.0)

    assert len(res.resolution()) == 2


def test_dipole_solve_returns_a_finite_solution():
    """Smoke test of the coupled line-dipole path.

    The single-particle right-hand side builder stacks verbatim under the §14
    layout, so the only thing new here is the stacking.
    """
    cluster = _dimer(60, 50)
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    res = ClusterBIESolver(cluster, mats, pol=2).scatter_dipole(633.0, 260.0, 700.0)

    assert res.ei.shape == (cluster.n_dof,)
    assert np.all(np.isfinite(res.ei))


def test_cross_sections_after_a_dipole_solve_is_rejected():
    """`C_ext` is defined against a unit-amplitude incident plane wave (§3.6).

    With a dipole source there is no incident plane wave to extinguish, so the
    optical theorem's forward-amplitude relation does not apply and the number
    would be meaningless rather than merely imprecise.
    """
    cluster = _dimer(40, 40)
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    res = ClusterBIESolver(cluster, mats, pol=2).scatter_dipole(633.0, 260.0, 700.0)
    with pytest.raises(ValueError, match="incident plane wave"):
        res.cross_sections()


def test_dipole_source_inside_one_particle_of_a_cluster_raises():
    """The source must lie outside *every* particle, not merely the first.

    `line_dipole_rhs`'s own guard runs once per particle, which is exactly the
    cluster condition — so a source at the centre of particle 1 must raise even
    though it is comfortably outside particle 0. This is the test that would
    fail if a future change tried to "optimise" the per-particle guard away.
    """
    cluster = _dimer(40, 40)
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    solver = ClusterBIESolver(cluster, mats, pol=2)
    with pytest.raises(ValueError, match="inside the particle"):
        solver.scatter_dipole(633.0, 520.0, 0.0)
