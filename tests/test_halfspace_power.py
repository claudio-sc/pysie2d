"""Far field, scattered power and absorption over a half-space (G12–G14).

Over a PEC plane nothing enters the substrate, so the upward far field must equal
the four-cylinder mirror reference's on the upper half circle, and a dipole's
power must balance exactly. Over a dielectric or a metal power also leaves
downward, so the same balance must come up short — a test that could not fail if
the upward bookkeeping were wrong.
"""

import numpy as np
import pytest

from pysie2d import (
    BIESolver,
    Cluster,
    Geometry,
    HalfSpace,
    Material,
    relative_ldos,
)
from pysie2d.cluster import ClusterScatterResult
from pysie2d.fields import _absorbed_cross_section, _upward_amplitude
from pysie2d.layered import fresnel_r
from pysie2d.reference import mie
from test_halfspace import CASES, RAD, pec_pair
from test_halfspace_cluster import build, solve_pair

PI = np.pi

# --- G12: upward far field over PEC equals the mirror cluster's ---------------------

# Measured worst 1.6e-15 (single particle, 12 cases) and 1.6e-15 (two-particle
# cluster against four cylinders, 12 cases); the spec's own figure is 1.6e-15.
# 1e-13 is two decades of room for BLAS differences and nine under a wrong sign.
RTOL_FAR = 1e-13


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("wavelength,n_clad", CASES)
def test_upward_far_field_over_pec_equals_the_mirror_cluster(pol, wavelength, n_clad):
    d = pec_pair(pol, wavelength, n_clad)
    geo, mat, nn = d["solver"].geometry, d["solver"].material, d["nn"]
    image = Geometry.gielis(RAD, nn, m=0, x0=0.0, z0=-geo.z0)
    mirror = ClusterScatterResult(
        d["ec"], Cluster([geo, image]), [mat, mat], pol, wavelength, 0.0, "dipole"
    )
    amp, angles = d["res"].far_field(401)
    assert angles[0] == -PI / 2 and angles[-1] == PI / 2
    ref = mirror._amp_at(angles)
    assert np.abs(amp - ref).max() / np.abs(ref).max() < RTOL_FAR


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("kind,angle", [("dipole", 0.0), ("plane", 25.0)])
@pytest.mark.parametrize("wavelength,n_clad", CASES)
def test_cluster_upward_far_field_over_pec_equals_four_cylinders(
    pol, kind, angle, wavelength, n_clad
):
    res, ec, four, _, _ = solve_pair(pol, wavelength, n_clad, kind, angle=angle)
    _, mirror_images, mats = build(pol, n_clad)
    ref_result = ClusterScatterResult(
        ec, four.cluster, mats + mats, pol, wavelength, 0.0, "dipole"
    )
    amp, angles = res.far_field(401)
    ref = ref_result._amp_at(angles)
    assert np.abs(amp - ref).max() / np.abs(ref).max() < RTOL_FAR


# --- G13: absorption from the boundary flux -----------------------------------------

# σ_abs from Im Σ conj(φ)χ against v0.8's c_ext − c_sca (two independent routes:
# the flux needs no far field, the difference needs two), and against Mie's
# Q_abs on a lossy circle. Measured ≤ 5.9e-16 (v0.8) and ≤ 1.2e-15 (Mie) on the
# circle, so 1.2e-14 (10× the worst). On the README star at nn = 512, n_clad =
# 1.33 the v0.8 side is the noisy one — c_ext − c_sca is a difference of
# near-equal numbers — and the measured gap is 4.3e-14 (TE) and 8.9e-13 (TM), the
# spec's "1e-13 / 1e-11"; the bars are 10× those. All are under the spec's stop
# line of 1e-10.
RTOL_ABS = 1.2e-14
RTOL_ABS_STAR = {2: 5e-13, 1: 1e-11}


def _flux_cabs(result):
    g = result.geometry
    return _absorbed_cross_section(
        [(g.f, g.g, g.df, g.dg, g.delt, result.ei)], result.wnum_bg
    )


@pytest.mark.parametrize("pol", [2, 1])
def test_flux_absorption_equals_v08_difference_and_mie_on_a_lossy_circle(pol):
    geo = Geometry.gielis(rad=200.0, n_pts=300, m=0)
    mat = Material(n_core=1.5, n_clad=1.33, pol=pol, epsi=0.5)
    res = BIESolver(geo, mat).scatter(600.0)
    flux = _flux_cabs(res)
    assert flux == pytest.approx(res.cross_sections()["c_abs"], rel=RTOL_ABS)
    x = 2 * PI * 1.33 * 200.0 / 600.0
    q_abs = mie.efficiencies(x, complex(mat.nc))[f"Q_abs_{'TE' if pol == 2 else 'TM'}"]
    assert flux == pytest.approx(q_abs * 2 * 200.0, rel=RTOL_ABS)


@pytest.mark.parametrize("pol", [2, 1])
def test_flux_absorption_equals_v08_difference_on_the_readme_star(pol):
    geo = Geometry.gielis(rad=200.0, n_pts=512, m=6, n1=6.0, n2=12.0, n3=12.0)
    mat = Material(n_core=1.5, n_clad=1.33, pol=pol, epsi=0.5)
    res = BIESolver(geo, mat).scatter(600.0)
    assert _flux_cabs(res) == pytest.approx(
        res.cross_sections()["c_abs"], rel=RTOL_ABS_STAR[pol]
    )


# --- G14: dipole power balance ------------------------------------------------------

# Over PEC nothing enters the substrate: P_up + P_abs = P_in. Measured residual
# 3.7e-14 relative (lossy particle, TE and TM, n_angles 64/200/500 all at or
# below 2e-14); bar 10× the worst. The spec's stop line is 1e-10.
RTOL_BALANCE = 4e-13
# Over glass and silver the substrate takes power: measured shortfall ≥ 3.3e-3
# of P_in (silver, TE, where the field barely penetrates) up to 0.48. 1e-3 is
# below the smallest measured value and nine decades above the PEC residual, so
# the inequality cannot hold by round-off alone.
MIN_SHORTFALL = 1e-3


def _powers(solver, wavelength, xs, zs, n_angles=500):
    res = solver.scatter_dipole(wavelength, xs, zs)
    k, pol, bg = res.wnum_bg, solver.material.pol, solver.background
    r_of = lambda th: fresnel_r(  # noqa: E731
        k * np.sin(th), k, bg.eps_rel(solver.material.n_clad, wavelength), pol
    )
    x, w = np.polynomial.legendre.leggauss(n_angles)
    th = 0.5 * PI * x
    a_up = _upward_amplitude(th, *res._substrate())
    # The dipole's own field upward, plus its direct reflection off the interface.
    z_img = 2 * bg.z_int - zs
    s = np.exp(-1j * k * (xs * np.sin(th) + zs * np.cos(th))) + r_of(th) * np.exp(
        -1j * k * (xs * np.sin(th) + z_img * np.cos(th))
    )
    g, nn = solver.geometry, solver.geometry.n_pts
    p_up = np.sum(0.5 * PI * w * np.abs(s + a_up) ** 2) / (8 * PI)
    p_abs = -np.imag(g.delt * np.sum(np.conj(res.ei[:nn]) * res.ei[nn:]))
    p_in = relative_ldos(solver, wavelength, xs, zs) / 4
    return p_up, p_abs, p_in


def _dipole_solver(pol, background):
    geo = Geometry.gielis(rad=100.0, n_pts=256, m=0, x0=0.0, z0=115.0)
    return BIESolver(
        geo, Material(n_core=2.0, n_clad=1.33, pol=pol, epsi=0.5), background
    )


@pytest.mark.parametrize("pol", [2, 1])
def test_dipole_power_balances_exactly_over_pec(pol):
    p_up, p_abs, p_in = _powers(
        _dipole_solver(pol, HalfSpace.pec()), 633.0, 60.0, 330.0
    )
    assert p_abs > 0.01 * p_in  # the particle is lossy: the balance is not trivial
    assert abs(p_up + p_abs - p_in) / p_in < RTOL_BALANCE


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize(
    "background", [HalfSpace(2.25), HalfSpace(-18.3 + 0.48j)], ids=["glass", "silver"]
)
def test_dipole_power_falls_short_of_the_input_over_a_lossy_or_open_substrate(
    pol, background
):
    p_up, p_abs, p_in = _powers(_dipole_solver(pol, background), 633.0, 60.0, 330.0)
    assert (p_in - p_up - p_abs) / p_in > MIN_SHORTFALL


# --- result keys and the cluster ----------------------------------------------------


def test_result_keys_change_with_a_background():
    solver = _dipole_solver(2, HalfSpace(2.25))
    res = solver.scatter(633.0, angle=20.0)
    assert set(res.cross_sections()) == {"c_sca_up", "c_abs"}
    eff = res.efficiencies()
    assert set(eff) == {"qsca_up", "qabs"}
    c = res.cross_sections()
    assert eff["qsca_up"] == pytest.approx(c["c_sca_up"] / (2 * 100.0), rel=1e-15)
    assert eff["qabs"] == pytest.approx(c["c_abs"] / (2 * 100.0), rel=1e-15)
    # And the cluster result reports the same pair, equal for a one-particle cluster.
    from pysie2d import ClusterBIESolver

    cl = ClusterBIESolver(
        Cluster([solver.geometry]),
        [solver.material],
        pol=2,
        background=solver.background,
    ).scatter(633.0, angle=20.0)
    assert set(cl.cross_sections()) == {"c_sca_up", "c_abs"}
    assert cl.cross_sections()["c_sca_up"] == pytest.approx(c["c_sca_up"], rel=1e-12)


def test_cross_section_quadrature_converges_in_the_number_of_angles():
    # Gauss–Legendre on a smooth upward amplitude: spec measured round-off by 64
    # nodes at kR = 10. The default of 500 must agree with 64 to the same level.
    res = _dipole_solver(1, HalfSpace(-18.3 + 0.48j)).scatter(633.0)
    a, b = res.cross_sections(64)["c_sca_up"], res.cross_sections(500)["c_sca_up"]
    assert a == pytest.approx(b, rel=1e-12)
    assert b > 0.0


def test_substrate_equal_to_the_cladding_has_no_nan_at_grazing():
    # eps_rel = 1: r ≡ 0, but α₁ = α₂ = 0 at θ = ±π/2 would give 0/0.
    res = _dipole_solver(2, HalfSpace(1.33**2)).scatter(633.0)
    amp, _ = res.far_field(101)
    assert np.all(np.isfinite(amp))
