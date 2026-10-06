"""Driven half-space wiring: matrix, right-hand sides, near field, LDOS.

G6 pins the sign wiring against the v0.8 cluster: a circle over a PEC plane is
the circle plus its mirror image, solved by ``ClusterBIESolver``. G7 pins the
no-op limits to bit-identity, and G10 the scale covariance of the new blocks.
"""

import numpy as np
import pytest

from pysie2d import (
    BIESolver,
    Cluster,
    ClusterBIESolver,
    Geometry,
    HalfSpace,
    Material,
    relative_ldos,
    relative_ldos_map,
    self_green,
)
from pysie2d.fields import _representation_at
from pysie2d.sources import line_dipole_rhs

PI = np.pi
RAD = 100.0


def pec_pair(pol, wavelength, n_clad=1.0, nn=160, gap=10.0, z_int=0.0):
    """The half-space solve and its two-cylinder image reference, side by side."""
    mat = Material(n_core=2.0, n_clad=n_clad, pol=pol)
    k = mat.wnum_bg(wavelength)
    zc = z_int + RAD + gap
    geo = Geometry.gielis(RAD, nn, m=0, x0=0.0, z0=zc)
    img = Geometry.gielis(RAD, nn, m=0, x0=0.0, z0=2 * z_int - zc)
    xs, zs = 60.0, zc + RAD + 80.0
    sign = -1.0 if pol == 2 else 1.0

    solver = BIESolver(geo, mat, background=HalfSpace.pec(z_int))
    res = solver.scatter_dipole(wavelength, xs, zs)

    cs = ClusterBIESolver(Cluster([geo, img]), [mat, mat], pol=pol)
    rhs = np.zeros(4 * nn, dtype=complex)
    for p, gp in enumerate((geo, img)):
        rhs[2 * nn * p : 2 * nn * (p + 1)] = line_dipole_rhs(
            nn, k, gp.f, gp.g, xs, zs
        ) + sign * line_dipole_rhs(nn, k, gp.f, gp.g, xs, 2 * z_int - zs)
    ec = np.linalg.solve(cs._assemble(wavelength), rhs)

    def cluster_field(x, z):
        return sum(
            _representation_at(
                ec[2 * nn * p : 2 * nn * (p + 1)],
                nn,
                gp.f,
                gp.df,
                gp.g,
                gp.dg,
                gp.delt,
                k,
                x,
                z,
            )
            for p, gp in enumerate((geo, img))
        )

    xo = np.array([-400.0, 0.0, 250.0, 500.0])
    zo = np.array([60.0, zc + 2.5 * RAD, 40.0, 700.0]) + z_int
    return {
        "solver": solver,
        "res": res,
        "ec": ec,
        "nn": nn,
        "obs": (xo, zo),
        "cluster_field": cluster_field(xo, zo),
        "src": (xs, zs),
        "cluster_self": cluster_field(np.array([xs]), np.array([zs]))[0],
        "sign": sign,
        "k": k,
    }


# G6 tolerances. Measured against the two-cylinder reference at nn = 160 over
# TE/TM, real and complex λ and n_clad ∈ {1, 1.33}: φ ≤ 6.6e-15, χ ≤ 2.6e-13
# (the χ half carries the Jacobian and the double-layer cancellation), near field
# ≤ 5.8e-16, S ≤ 9.8e-16; the plane wave gives φ ≤ 1e-14, χ ≤ 7e-14. 1e-12 holds
# the study's end-to-end floor (3e-15–1e-13) with room and is nine decades below
# the O(1e-3) of a flipped reflection sign.
RTOL_E2E = 1e-12

CASES = [(633.0, 1.0), (633.0 * (1 + 0.05j), 1.0), (800.0, 1.33)]
CASE_IDS = ["real-vacuum", "complex-vacuum", "real-n1.33"]


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("wavelength,n_clad", CASES, ids=CASE_IDS)
def test_dipole_over_pec_equals_particle_plus_mirror_cluster(pol, wavelength, n_clad):
    d = pec_pair(pol, wavelength, n_clad)
    nn, ei, ec = d["nn"], d["res"].ei, d["ec"]
    assert np.abs(ei[:nn] - ec[:nn]).max() / np.abs(ec[:nn]).max() < RTOL_E2E
    chi = ec[nn : 2 * nn]
    assert np.abs(ei[nn:] - chi).max() / np.abs(chi).max() < RTOL_E2E
    near = d["res"].eval_field(*d["obs"])
    ref = d["cluster_field"]
    assert np.abs(near - ref).max() / np.abs(ref).max() < RTOL_E2E


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("wavelength,n_clad", CASES, ids=CASE_IDS)
def test_self_green_and_ldos_over_pec_equal_particle_plus_mirror(
    pol, wavelength, n_clad
):
    d = pec_pair(pol, wavelength, n_clad)
    xs, zs = d["src"]
    # The reflected self-field is the closed-form image Hankel term, which shares
    # no code with the Sommerfeld machinery.
    from scipy.special import hankel1

    image = d["sign"] * 0.25j * hankel1(0, d["k"] * 2 * zs)
    ref = d["cluster_self"] + image
    s = self_green(d["solver"], wavelength, xs, zs)
    assert abs(s - ref) / abs(ref) < RTOL_E2E
    assert relative_ldos(d["solver"], wavelength, xs, zs) == pytest.approx(
        1 + 4 * ref.imag, rel=RTOL_E2E
    )


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("angle", [0.0, 25.0, -40.0])
@pytest.mark.parametrize(
    "wavelength,n_clad", [CASES[0], CASES[2]], ids=["vac", "n1.33"]
)
def test_plane_wave_over_pec_equals_incident_plus_reflected_on_the_mirror_cluster(
    pol, angle, wavelength, n_clad
):
    from pysie2d.sources import plane_wave_rhs

    d = pec_pair(pol, wavelength, n_clad)
    nn, k, sign = d["nn"], d["k"], d["sign"]
    geo = d["solver"].geometry
    mat = d["solver"].material
    img = Geometry.gielis(RAD, nn, m=0, x0=0.0, z0=-geo.z0)
    cs = ClusterBIESolver(Cluster([geo, img]), [mat, mat], pol=pol)
    rhs = np.zeros(4 * nn, dtype=complex)
    for p, gp in enumerate((geo, img)):
        # Incident plus its mirror image, the latter signed by the reflection.
        rhs[2 * nn * p : 2 * nn * (p + 1)] = plane_wave_rhs(
            nn, angle, k, gp.f, gp.g
        ) + sign * plane_wave_rhs(nn, angle, k, gp.f, -gp.g)
    ec = np.linalg.solve(cs._assemble(wavelength), rhs)
    res = d["solver"].scatter(wavelength, angle=angle)
    assert np.abs(res.ei[:nn] - ec[:nn]).max() / np.abs(ec[:nn]).max() < RTOL_E2E
    chi = ec[nn : 2 * nn]
    assert np.abs(res.ei[nn:] - chi).max() / np.abs(chi).max() < RTOL_E2E


def test_plane_wave_at_grazing_incidence_is_refused():
    solver = pec_pair(2, 633.0)["solver"]
    for angle in (90.0, -90.0, 120.0):
        with pytest.raises(ValueError, match="angle"):
            solver.scatter(633.0, angle=angle)


def test_source_at_or_below_the_interface_is_refused():
    solver = pec_pair(2, 633.0)["solver"]
    with pytest.raises(ValueError, match="strictly above the interface"):
        solver.scatter_dipole(633.0, 0.0, -50.0)
    with pytest.raises(ValueError, match="strictly above the interface"):
        self_green(solver, 633.0, 0.0, 0.0)


def test_particle_crossing_the_interface_is_refused_at_construction():
    geo = Geometry.gielis(RAD, 64, m=0, x0=0.0, z0=50.0)
    with pytest.raises(ValueError, match="strictly above the interface"):
        BIESolver(geo, Material(n_core=2.0), background=HalfSpace.pec())


def test_eval_field_below_the_interface_is_refused():
    res = pec_pair(2, 633.0)["res"]
    with pytest.raises(ValueError, match="strictly above the interface"):
        res.eval_field(np.array([0.0]), np.array([-10.0]))


def test_interior_field_is_unchanged_by_the_substrate_term():
    # The interior representation uses the core Green function only, so for a
    # given boundary solution an interior point cannot see G_ind: the result is
    # bit-identical to the bare free-space evaluation of the same solution.
    from pysie2d.fields import eval_field

    d = pec_pair(2, 633.0)
    res, nn, k = d["res"], d["nn"], d["k"]
    geo, mat = d["solver"].geometry, d["solver"].material
    inside = np.array([geo.x0 + 10.0]), np.array([geo.z0 + 5.0])
    ref = eval_field(
        res.ei, nn, geo.f, geo.df, geo.g, geo.dg, geo.delt, k, *inside, ri=mat.nc
    )
    assert ref[0] != 0.0
    assert np.array_equal(res.eval_field(*inside), ref)


def test_multipoles_are_refused_over_a_substrate():
    # The expansion is of the free-space scattered field; over a substrate it
    # would silently omit the reflected one.
    res = pec_pair(2, 633.0)["res"]
    with pytest.raises(NotImplementedError, match="multipoles"):
        res.multipoles()


# LDOS map against the single-source path: the same physics through the batched
# multi-RHS back-substitution; measured ≤ 1.7e-16 over PEC, glass and silver,
# TE and TM, with NaN at a point inside the particle and one below the interface.
RTOL_MAP = 1e-12


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize(
    "background",
    [HalfSpace.pec(), HalfSpace(2.25), HalfSpace(-18.3 + 0.48j)],
    ids=["pec", "glass", "silver"],
)
def test_ldos_map_matches_single_source_ldos_over_a_substrate(pol, background):
    geo = Geometry.gielis(RAD, 160, m=0, x0=0.0, z0=RAD + 10.0)
    solver = BIESolver(geo, Material(n_core=2.0, n_clad=1.33, pol=pol), background)
    xs = np.array([60.0, -300.0, 0.0, 150.0, 0.0])
    zs = np.array([geo.z0 + RAD + 80.0, 50.0, geo.z0, 400.0, -20.0])
    got = relative_ldos_map(solver, 633.0, xs, zs)
    assert np.isnan(got[2]) and np.isnan(got[4])  # inside the particle; below z_int
    for i in (0, 1, 3):
        assert got[i] == pytest.approx(
            relative_ldos(solver, 633.0, xs[i], zs[i]), rel=RTOL_MAP
        )


# --- G7: the no-op limits are bit-identical to v0.8 -----------------------------------


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("n_clad", [1.0, 1.33])
def test_substrate_equal_to_the_cladding_is_bit_identical_to_no_background(pol, n_clad):
    # ε_sub = n_clad² gives eps_rel == 1 exactly: the reflected term is not built.
    geo = Geometry.gielis(RAD, 160, m=0, x0=0.0, z0=RAD + 10.0)
    mat = Material(n_core=2.0, n_clad=n_clad, pol=pol)
    plain = BIESolver(geo, mat)
    same = BIESolver(geo, mat, HalfSpace(n_clad**2))
    for lam in (633.0, 633.0 * (1 + 0.05j)):
        assert np.array_equal(plain.assemble(lam), same.assemble(lam))
    a = plain.scatter(633.0, angle=20.0)
    b = same.scatter(633.0, angle=20.0)
    assert np.array_equal(a.ei, b.ei)
    xs, zs = 60.0, geo.z0 + RAD + 80.0
    a, b = plain.scatter_dipole(633.0, xs, zs), same.scatter_dipole(633.0, xs, zs)
    assert np.array_equal(a.ei, b.ei)
    x, z = np.array([-400.0, 250.0]), np.array([60.0, 40.0])
    assert np.array_equal(a.eval_field(x, z), b.eval_field(x, z))
    assert self_green(plain, 633.0, xs, zs) == self_green(same, 633.0, xs, zs)


def test_background_none_is_the_default_and_changes_nothing():
    geo = Geometry.gielis(RAD, 64, m=0)
    mat = Material(n_core=2.0)
    assert BIESolver(geo, mat).background is None
    explicit = BIESolver(geo, mat, background=None)
    assert np.array_equal(BIESolver(geo, mat).assemble(633.0), explicit.assemble(633.0))


# --- G10: scale covariance with a background (conventions §9) -------------------
#
# M sees only k·rad, so scaling rad, λ, z_int and every position by s must leave
# the matrix entrywise unchanged. Powers of 4: fl(√(4x)) = 2·fl(√x), so every
# degree-zero combination in the path construction (T, δ, the nodes, α) comes out
# bit-identical and the assertion carries no tolerance. A generic ratio is the
# variant that can fail from conditioning: measured 3.3e-15 (TE and TM, PEC,
# glass and silver, real and complex λ); the bar is 10× that.
RTOL_SCALE = 3.3e-14

BACKGROUNDS = [
    HalfSpace.pec,
    lambda z: HalfSpace(2.25, z),
    lambda z: HalfSpace(-18.3 + 0.48j, z),
]
BACKGROUND_IDS = ["pec", "glass", "silver"]


def _scaled(s, pol, make_bg, z_int=-20.0):
    geo = Geometry.gielis(RAD * s, 160, m=0, x0=30.0 * s, z0=115.0 * s)
    mat = Material(n_core=2.0, n_clad=1.33, pol=pol)
    return BIESolver(geo, mat, make_bg(z_int * s))


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("make_bg", BACKGROUNDS, ids=BACKGROUND_IDS)
@pytest.mark.parametrize("wavelength", [633.0, 633.0 * (1 + 0.05j)])
def test_matrix_is_scale_covariant_over_a_substrate(pol, make_bg, wavelength):
    base = _scaled(1.0, pol, make_bg).assemble(wavelength)
    for s in (4.0, 0.25):
        assert np.array_equal(
            _scaled(s, pol, make_bg).assemble(wavelength * s), base
        ), s
    generic = _scaled(1.7, pol, make_bg).assemble(wavelength * 1.7)
    assert np.abs(generic - base).max() / np.abs(base).max() < RTOL_SCALE


# Observables at a generic ratio, measured over PEC, glass and silver, TE and TM:
# φ ≤ 4.1e-15, near field ≤ 2.5e-15, LDOS ≤ 8.2e-16. The χ half is
# worse, ≤ 4.3e-13, because it carries the conditioning of M (the same level as its
# G6 comparison, 2.6e-13). Each bar is 10× its measurement, and far below the
# 1e-10 at which the spec says to stop.
RTOL_SCALE_OBS = 5e-14
RTOL_SCALE_CHI = 5e-12


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("make_bg", BACKGROUNDS, ids=BACKGROUND_IDS)
def test_driven_observables_are_scale_covariant_over_a_substrate(pol, make_bg):
    # The 2-D Green function is dimensionless and χ carries the Jacobian of t
    # (conventions §5), so φ, χ, the near field and the LDOS are all invariant
    # when every length and the wavelength scale.
    def observables(s):
        sv = _scaled(s, pol, make_bg)
        xs, zs = 60.0 * s, 330.0 * s
        res = sv.scatter_dipole(633.0 * s, xs, zs)
        nn = sv.geometry.n_pts
        field = res.eval_field(
            np.array([-400.0, 250.0]) * s, np.array([60.0, 40.0]) * s
        )
        return res.ei[:nn], res.ei[nn:], field, relative_ldos(sv, 633.0 * s, xs, zs)

    base = observables(1.0)
    for a, b in zip(base, observables(4.0), strict=True):
        assert np.array_equal(a, b)
    bars = (RTOL_SCALE_OBS, RTOL_SCALE_CHI, RTOL_SCALE_OBS, RTOL_SCALE_OBS)
    for a, b, bar in zip(base, observables(1.7), bars, strict=True):
        assert np.abs(b - a).max() / np.abs(a).max() < bar
