"""G6 on a cluster: two particles over a PEC plane are four cylinders.

The reference is ``ClusterBIESolver`` on the two particles and their two mirror
images, with the images' sources signed by the reflection. It shares no code with
the half-space blocks, so it pins the (q, p) wiring, the cluster-extent path and
the sign of every reflected term. The particles differ in size, resolution and
material so a source-versus-field mix-up cannot hide behind symmetry.
"""

import numpy as np
import pytest

from pysie2d import (
    Cluster,
    ClusterBIESolver,
    Geometry,
    HalfSpace,
    Material,
)
from pysie2d.fields import _representation_at
from pysie2d.sources import line_dipole_rhs, plane_wave_rhs

# Measured at nn = 160/128 over TE and TM, real and complex λ, n_clad ∈ {1, 1.33},
# a dipole and plane waves at 0°, 25°, −40°: worst φ 1.8e-14, χ 2.2e-13 (the χ
# half carries the matrix's conditioning, as in the single-particle G6), near
# field 1.1e-15. Bar 1e-12 holds the study's end-to-end floor (3e-15–1e-13)
# with room and sits nine decades under the O(1e-3) of a wrong reflection sign.
RTOL_CLUSTER = 1e-12

CASES = [(633.0, 1.0), (633.0 * (1 + 0.05j), 1.0), (800.0, 1.33)]


def build(pol, n_clad):
    a = Geometry.gielis(100.0, 160, m=0, x0=-130.0, z0=130.0)
    b = Geometry.gielis(60.0, 128, m=0, x0=110.0, z0=90.0)
    mats = [
        Material(n_core=2.0, n_clad=n_clad, pol=pol),
        Material(n_core=3.0, n_clad=n_clad, pol=pol, epsi=0.4),
    ]
    mirror = [Geometry.gielis(g.rad, g.n_pts, m=0, x0=g.x0, z0=-g.z0) for g in (a, b)]
    return [a, b], mirror, mats


def solve_pair(pol, wavelength, n_clad, kind, angle=0.0, src=(0.0, 330.0)):
    geoms, mirror, mats = build(pol, n_clad)
    sign = -1.0 if pol == 2 else 1.0
    k = mats[0].wnum_bg(wavelength)
    half = ClusterBIESolver(Cluster(geoms), mats, pol=pol, background=HalfSpace.pec())
    four = ClusterBIESolver(Cluster(geoms + mirror), mats + mats, pol=pol)

    rhs = np.zeros(four.cluster.n_dof, dtype=complex)
    for p, gp in enumerate(geoms + mirror):
        if kind == "plane":
            direct = plane_wave_rhs(gp.n_pts, angle, k, gp.f, gp.g)
            image = plane_wave_rhs(gp.n_pts, angle, k, gp.f, -gp.g)
        else:
            direct = line_dipole_rhs(gp.n_pts, k, gp.f, gp.g, *src)
            image = line_dipole_rhs(gp.n_pts, k, gp.f, gp.g, src[0], -src[1])
        rhs[four.cluster.slice(p)] = direct + sign * image
    ec = np.linalg.solve(four._assemble(wavelength), rhs)

    if kind == "plane":
        res = half.scatter(wavelength, angle=angle)
    else:
        res = half.scatter_dipole(wavelength, *src)

    xo = np.array([-400.0, 0.0, 250.0, 500.0, -130.0])
    zo = np.array([60.0, 60.0, 40.0, 700.0, 330.0])
    ref = sum(
        _representation_at(
            ec[four.cluster.slice(p)],
            gp.n_pts,
            gp.f,
            gp.df,
            gp.g,
            gp.dg,
            gp.delt,
            k,
            xo,
            zo,
        )
        for p, gp in enumerate(geoms + mirror)
    )
    return res, ec, four, (xo, zo), ref


def _compare(res, ec, four):
    for p in range(2):
        got, want = res.ei_particle(p), ec[four.cluster.slice(p)]
        nn = got.size // 2
        assert (
            np.abs(got[:nn] - want[:nn]).max() / np.abs(want[:nn]).max() < RTOL_CLUSTER
        )
        assert (
            np.abs(got[nn:] - want[nn:]).max() / np.abs(want[nn:]).max() < RTOL_CLUSTER
        )


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("wavelength,n_clad", CASES)
def test_dipole_over_pec_equals_four_cylinders(pol, wavelength, n_clad):
    res, ec, four, pts, ref = solve_pair(pol, wavelength, n_clad, "dipole")
    _compare(res, ec, four)
    near = res.eval_field(*pts)
    assert np.abs(near - ref).max() / np.abs(ref).max() < RTOL_CLUSTER


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("angle", [0.0, 25.0, -40.0])
@pytest.mark.parametrize("wavelength,n_clad", [CASES[0], CASES[2]])
def test_plane_wave_over_pec_equals_four_cylinders(pol, angle, wavelength, n_clad):
    res, ec, four, pts, ref = solve_pair(pol, wavelength, n_clad, "plane", angle=angle)
    _compare(res, ec, four)
    near = res.eval_field(*pts)
    assert np.abs(near - ref).max() / np.abs(ref).max() < RTOL_CLUSTER


def test_one_particle_cluster_over_a_substrate_is_bit_identical_to_biesolver():
    from pysie2d import BIESolver

    geo = Geometry.gielis(100.0, 160, m=0, x0=0.0, z0=115.0)
    mat = Material(n_core=2.0, n_clad=1.33, pol=1)
    bg = HalfSpace(-18.3 + 0.48j)
    single = BIESolver(geo, mat, background=bg)
    cluster = ClusterBIESolver(Cluster([geo]), [mat], pol=1, background=bg)
    assert np.array_equal(single.assemble(633.0), cluster._assemble(633.0))
    a = single.scatter(633.0, angle=20.0)
    b = cluster.scatter(633.0, angle=20.0)
    assert np.array_equal(a.ei, b.ei)
    a = single.scatter_dipole(633.0, 60.0, 330.0)
    b = cluster.scatter_dipole(633.0, 60.0, 330.0)
    assert np.array_equal(a.ei, b.ei)
    x, z = np.array([-400.0, 250.0]), np.array([60.0, 40.0])
    # The cluster's vectorised helper rounds differently with batch size
    # (conventions §14); measured at round-off, so the bar is one ulp-scale.
    assert np.abs(a.eval_field(x, z) - b.eval_field(x, z)).max() < 1e-14


def test_substrate_equal_to_the_cladding_is_bit_identical_for_a_cluster():
    geoms, _, mats = build(2, 1.33)
    plain = ClusterBIESolver(Cluster(geoms), mats, pol=2)
    same = ClusterBIESolver(Cluster(geoms), mats, pol=2, background=HalfSpace(1.33**2))
    assert np.array_equal(plain._assemble(633.0), same._assemble(633.0))
    assert np.array_equal(plain.scatter(633.0).ei, same.scatter(633.0).ei)


def test_cluster_below_the_interface_is_refused():
    geo = Geometry.gielis(100.0, 64, m=0, x0=0.0, z0=50.0)
    with pytest.raises(ValueError, match="strictly above the interface"):
        ClusterBIESolver(
            Cluster([geo]),
            [Material(n_core=2.0)],
            background=HalfSpace.pec(),
        )
