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
    BIESolver,
    Cluster,
    ClusterBIESolver,
    ClusterGapWarning,
    ClusterOverlapError,
    ClusterResolutionWarning,
    Geometry,
    Material,
)
from pysie2d.cluster import GAP_ENVELOPE_C

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
            # n_pts = 90, not 20: the 230 nm gap (a = 180 nm) needs n_pts >= 89
            # per GAP_ENVELOPE_C, and this test isolates the resolution guard
            # from the gap guard — both firing here would leave the assertion
            # unable to tell which one it caught.
            Geometry.gielis(180.0, 90, m=0),
            Geometry.gielis(110.0, 400, m=0, x0=520.0),
        ]
    )
    mats = [Material(2.0, 1.0, pol=2), Material(2.0, 1.0, pol=2)]
    solver = ClusterBIESolver(cluster, mats, pol=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ClusterGapWarning)
        with pytest.warns(ClusterResolutionWarning, match="particle 0 is resolved"):
            solver.scatter(wavelength=633.0)


def test_balanced_cluster_solves_without_warning():
    """Two comparably resolved particles must cost no warning at all.

    The mirror of the test above: a guard that fires on the ordinary case is
    worse than no guard, because users learn to filter it. The resolutions are
    120/90 rather than 80/60 so the pair also clears the gap envelope of
    `GAP_ENVELOPE_C` (230 nm gap, a = 180 nm, 89 nodes needed) — this test
    asserts silence from *every* guard, not just the resolution one.
    """
    cluster = _dimer(120, 90)
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
    # 96/96, not 60/50: the 230 nm gap (a = 180 nm) needs n_pts >= 89 per
    # GAP_ENVELOPE_C, and this smoke test asserts silence isn't the point here.
    cluster = _dimer(96, 96)
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    res = ClusterBIESolver(cluster, mats, pol=2).scatter(wavelength=633.0, angle=37.0)

    assert res.ei.shape == (cluster.n_dof,)
    assert np.all(np.isfinite(res.ei))
    assert res.ei_particle(0).shape == (192,)
    assert res.ei_particle(1).shape == (192,)

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
    cluster = _dimer(96, 96)
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
    cluster = _dimer(96, 96)
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
    cluster = _dimer(96, 96)
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    solver = ClusterBIESolver(cluster, mats, pol=2)
    with pytest.raises(ValueError, match="inside the particle"):
        solver.scatter_dipole(633.0, 520.0, 0.0)


# ---------------------------------------------------------------------------
# G1 — the Np = 1 reduction, and G5 — far-field reciprocity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [0, 5], ids=["circle", "star"])
@pytest.mark.parametrize("pol", [1, 2], ids=["TM", "TE"])
def test_one_particle_cluster_is_bit_identical_to_BIESolver(m: int, pol: int):
    """A one-particle cluster reproduces `BIESolver` to the last bit.

    With `Np = 1` there is no cross-block, so the coupled matrix *is* the
    single-particle matrix — but only because conventions §14 lays the
    unknowns out contiguously per particle (`[φ_0, χ_0, …]`), so each
    particle's `2·nn_p` sub-vector goes straight into the existing
    single-particle primitives with no gather or reorder. Every comparison
    here is `np.array_equal`, deliberately: **if this degrades to merely
    "close", the DOF layout has been changed and conventions §14 has silently
    broken.** That, not the number, is what the test guards.

    `eval_field` is compared too, because the cluster evaluates the
    representation through a different helper than the single-particle point
    loop; equality is what pins the two against drift.

    The particle is awkward on purpose — off-centre, `n_clad ≠ 1`, lossy,
    oblique incidence — so that a convention that happens to cancel at the
    origin, in vacuum, at normal incidence cannot hide.
    """
    geom = Geometry.gielis(rad=200.0, n_pts=180, m=m, n1=4.0, x0=13.0, z0=-7.0)
    mat = Material(n_core=2.1, n_clad=1.3, pol=pol, epsi=0.4)

    single = BIESolver(geom, mat).scatter(wavelength=700.0, angle=23.0)
    cl = ClusterBIESolver(Cluster([geom]), [mat], pol=pol)
    multi = cl.scatter(wavelength=700.0, angle=23.0)

    assert np.array_equal(cl._assemble(700.0), BIESolver(geom, mat).assemble(700.0))
    assert np.array_equal(multi.ei, single.ei)
    assert np.array_equal(multi.far_field(401)[0], single.far_field(401)[0])

    # Mixed interior/exterior points: the interior branch uses k_core and the
    # exterior one k_bg, so both sides of the representation are exercised.
    x = np.array([13.0, 100.0, 13.0, 900.0, -700.0])
    z = np.array([-7.0, -7.0, 120.0, 400.0, -250.0])
    assert np.array_equal(multi.eval_field(x, z), single.eval_field(x, z))


def test_coupling_decays_at_the_two_dimensional_Green_function_rate():
    """The coupling correction dies as `H₀⁽¹⁾(k·d) ~ d^{−1/2}` (spec §6.2).

    The first-order multiple-scattering correction at one particle is
    proportional to the *other* particle's field there, which in 2-D falls off
    as `d^{−1/2}` — not `d^{−1}`, the 3-D rate, and not a constant. The
    "uncoupled" reference is `Σ_p amp_p` from two entirely separate
    `BIESolver` runs on the same `Geometry` objects, summed with **no phase
    factor**: the geometries carry absolute coordinates, so the inter-particle
    phase is already in each far field (spec §3.4). A test that needed a
    hand-added phase would be reporting a bug in that convention.

    **This is a slope test, not a tolerance test, on purpose.** A wrong
    cross-block normalisation — `h_p` used where `h_q` belongs in
    `assemble_cross_block`, a missing factor of 2, a swapped source/observer
    particle — changes the *power* of the decay, not merely its constant. A
    comparison at a single gap can be absorbed into the constant and pass; a
    fitted exponent cannot.

    Measured here (the §6.4 dimer, nn = 200, TE, λ = 633 nm), relative
    deviation `max_θ|amp_coupled − Σ amp_p| / max_θ|amp_coupled|`:

        gap/λ    2.7     4.5     7.5    12.5    21     35     55     75
        dev    8.8e-2  6.9e-2  5.8e-2  4.6e-2  3.8e-2 2.9e-2 2.3e-2 2.0e-2

    Full-range log-log slope −0.4465, four widest gaps −0.5211. The rate
    approaches −1/2 *from above* as the gap grows, because the higher-order
    terms that survive at short range bend the slope; that is why the fit uses
    the four widest gaps only. The asserted band (−0.60, −0.40) brackets the
    measured −0.52 with headroom while still excluding both −1 and 0, which is
    the whole point of the check.
    """
    lam, nn = 633.0, 200
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    gaps = lam * np.array([2.7, 4.5, 7.5, 12.5, 21.0, 35.0, 55.0, 75.0])

    devs = []
    for gap in gaps:
        g0 = Geometry.gielis(180.0, nn, m=0)
        g1 = Geometry.gielis(110.0, nn, m=0, x0=180.0 + 110.0 + gap)
        coupled = ClusterBIESolver(Cluster([g0, g1]), mats, pol=2).scatter(
            wavelength=lam, angle=37.0
        )
        amp, _ = coupled.far_field(361)
        amp_0, _ = (
            BIESolver(g0, mats[0]).scatter(wavelength=lam, angle=37.0).far_field(361)
        )
        amp_1, _ = (
            BIESolver(g1, mats[1]).scatter(wavelength=lam, angle=37.0).far_field(361)
        )
        devs.append(np.max(np.abs(amp - (amp_0 + amp_1))) / np.max(np.abs(amp)))

    slope = np.polyfit(np.log(gaps[-4:]), np.log(np.array(devs[-4:])), 1)[0]
    assert -0.60 < slope < -0.40


def _lossy_dimer_materials() -> list[Material]:
    """The §6.4 dimer's materials with particle 1 lossy, for G5."""
    return [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2, epsi=0.3)]


def test_far_field_reciprocity_on_an_asymmetric_cluster():
    """Colton & Kress Thm 3.13: `u_∞(x̂, d̂) = u_∞(−d̂, −x̂)`.

    In the solver's angles, with incidence α and observation θ in degrees,
    that reads `amp(θ = θ₁; angle = α₁) == amp(θ = −α₁; angle = −θ₁)`: two
    entirely separate coupled solves, related only through the symmetry of the
    operator. It is a property of the operator, so it holds for any number of
    particles of any shape — including the absorbing one used here, since the
    reciprocity relation needs a symmetric, not a lossless, medium.

    **The cluster must be asymmetric or the test is vacuous**: on a mirror-
    symmetric pair at normal incidence the two amplitudes coincide for
    geometric reasons and the identity says nothing. The dimer here has
    unequal radii, unequal indices and only one lossy particle, which is what
    makes the check sensitive to exactly the block-transposition and
    cross-block index errors that the `Np = 1` reduction of §6.1 cannot see.

    Measured deviation 8.8e-16 at nn = 300; `rtol = 5e-15` is that round-off
    floor with headroom, not a convergence order.
    """
    cluster = _dimer(300, 300)
    mats = _lossy_dimer_materials()
    solver = ClusterBIESolver(cluster, mats, pol=2)

    alpha1, theta1 = 20.0, 75.0
    direct = solver.scatter(wavelength=633.0, angle=alpha1)._amp_at(
        np.array([np.deg2rad(theta1)])
    )
    reciprocal = solver.scatter(wavelength=633.0, angle=-theta1)._amp_at(
        np.array([np.deg2rad(-alpha1)])
    )

    assert direct[0] == pytest.approx(reciprocal[0], rel=5e-15)


def test_gap_comfortably_inside_the_envelope_is_silent():
    """The envelope guard must not fire on a well-separated pair (G3).

    Radii 180/110 nm with a 910 nm gap needs `n_pts = 31` by the envelope;
    at 60 nodes each the measured far field is at round-off (1.4e-15 against
    an `nn = 1024` truth, docs/design/studies/cluster-gap-envelope.md), so
    *any* warning here is a guard firing for the wrong reason.
    """
    cluster = Cluster(
        [
            Geometry.gielis(180.0, 60, m=0),
            Geometry.gielis(110.0, 60, m=0, x0=1200.0),
        ]
    )
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    solver = ClusterBIESolver(cluster, mats, pol=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver.scatter(wavelength=633.0, angle=37.0)


def test_gap_below_the_envelope_warns_with_a_sufficient_n_pts():
    """A 20 nm gap at 60 nodes is below the envelope and must say so (G3).

    This is the failure mode the guard exists for: the coupled matrix stays
    well conditioned, every single-particle diagnostic stays clean, and the
    far field is wrong at the 1e-5 level (measured at gap/a = 0.11, nn = 64).
    The message must name a *sufficient* resolution, so the assertion checks
    the number against `GAP_ENVELOPE_C·sqrt(a/gap)` recomputed here rather
    than merely checking that the warning type appeared.
    """
    cluster = Cluster(
        [
            Geometry.gielis(180.0, 60, m=0),
            Geometry.gielis(110.0, 60, m=0, x0=310.0),
        ]
    )
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    solver = ClusterBIESolver(cluster, mats, pol=2)
    expected = GAP_ENVELOPE_C * np.sqrt(180.0 / cluster.min_gap)
    assert expected > 60  # the premise: 60 nodes really is below the envelope
    with pytest.warns(ClusterGapWarning) as record:
        solver.scatter(wavelength=633.0, angle=37.0)
    message = str(record[0].message)
    assert f"n_pts = {expected:.0f}" in message
    assert "has 60" in message
    # A circular pair carries no D9 caveat; that clause belongs to case three.
    assert "not validated" not in message


def test_non_circular_pair_below_the_envelope_repeats_the_D9_caveat():
    """The envelope was measured on circles only, and the warning says so.

    A rounded square facing a circle has a flat facing boundary, whose
    curvature is nowhere the circle's, so the conformal half-width
    sqrt(2·gap/a) that fixes the envelope is not the right one. Per D9 the
    guard still fires — a small gap is still a diagnostic — but it must tell
    the reader the bound is not
    validated here, or the number reads as authoritative.
    """
    cluster = Cluster(
        [
            Geometry.gielis(180.0, 60, m=0),
            Geometry.gielis(110.0, 60, m=4, n1=8.0, n2=8.0, n3=8.0, x0=380.0),
        ]
    )
    mats = [Material(2.0, 1.0, pol=2), Material(1.6, 1.0, pol=2)]
    solver = ClusterBIESolver(cluster, mats, pol=2)
    with pytest.warns(ClusterGapWarning, match="is not validated for"):
        solver.scatter(wavelength=633.0, angle=37.0)
