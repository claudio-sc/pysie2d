"""Far field, scattered power and absorption over a half-space (G12–G14).

Over a PEC plane nothing enters the substrate, so the upward far field must equal
the four-cylinder mirror reference's on the upper half circle, and a dipole's
power must balance exactly. Over a dielectric or a metal power also leaves
downward, and with the interface flux P_down (v1.1, GT7) the balance becomes an
equality again: P_up + P_down + P_abs = P_in.
"""

import math

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
from pysie2d.fields import (
    _absorbed_cross_section,
    _downward_amplitude,
    _downward_power,
    _flux_rule,
    _spectrum,
    _upward_amplitude,
)
from pysie2d.layered import Transmission, alpha, fresnel_r
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


def _upward_rule(eps_rel, n_angles):
    """Gauss–Legendre on [−π/2, π/2], split at ±θ_c when 0 < Re ε_rel < 1.

    The v1.0.1 rule of ``_upward_cross_sections`` (finding 1): unsplit, the
    dipole P_up over a substrate thinner than the cover is off by 3e-5 of P_in.
    """
    if eps_rel is None or not 0.0 < eps_rel.real < 1.0:
        x, w = np.polynomial.legendre.leggauss(n_angles)
        return 0.5 * PI * x, 0.5 * PI * w
    theta_c = float(np.arcsin(np.sqrt(eps_rel.real)))
    edges = [-0.5 * PI, -theta_c, theta_c, 0.5 * PI]
    th, wt = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        x, w = np.polynomial.legendre.leggauss(math.ceil(n_angles * (hi - lo) / PI))
        c, h = 0.5 * (hi + lo), 0.5 * (hi - lo)
        th.append(c + h * np.sin(0.5 * PI * x))
        wt.append(h * 0.5 * PI * np.cos(0.5 * PI * x) * w)
    return np.concatenate(th), np.concatenate(wt)


def _powers(solver, wavelength, xs, zs, n_angles=500):
    res = solver.scatter_dipole(wavelength, xs, zs)
    k, pol, bg = res.wnum_bg, solver.material.pol, solver.background
    eps_rel = bg.eps_rel(solver.material.n_clad, wavelength)
    r_of = lambda th: fresnel_r(k * np.sin(th), k, eps_rel, pol)  # noqa: E731
    th, w = _upward_rule(eps_rel, n_angles)
    a_up = _upward_amplitude(th, *res._substrate())
    # The dipole's own field upward, plus its direct reflection off the interface.
    z_img = 2 * bg.z_int - zs
    s = np.exp(-1j * k * (xs * np.sin(th) + zs * np.cos(th))) + r_of(th) * np.exp(
        -1j * k * (xs * np.sin(th) + z_img * np.cos(th))
    )
    g, nn = solver.geometry, solver.geometry.n_pts
    p_up = np.sum(w * np.abs(s + a_up) ** 2) / (8 * PI)
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


def _p_down(solver, wavelength, xs, zs):
    """Dipole P_down by interface flux: the scattered spectrum plus A_dip."""
    res = solver.scatter_dipole(wavelength, xs, zs)
    bnds, k, eps_rel, pol, z_int = res._substrate()
    g = solver.geometry
    h_min = min(g.g.min(), zs) - z_int  # the lowest source of either kind
    extent = max(np.ptp(np.r_[g.f, xs]), 2 * h_min)
    q, w = _flux_rule(k, eps_rel, pol, extent, 2 * h_min)
    a1 = alpha(q, k)
    # (i/4)·H₀ has unit far-field amplitude, so A_dip is the bare plane wave.
    spec = _spectrum(q, a1, bnds, z_int) + np.exp(-1j * q * xs + 1j * a1 * (zs - z_int))
    return _downward_power(q, w, spec, k, eps_rel, pol)


# GT7: with the flux P_down the balance closes over every passive substrate.
# Measured worst 8.7e-14 of P_in (silver TM; glass 1.1e-14, low-n 6.8e-15,
# lossy 1.2e-14, ε = −1 2.0e-14). Bar 1e-12, the spec's. P_down is 0.12–0.48 of
# P_in except silver TE (3.3e-3) and the lossless metal (0), so a missing or
# mis-normalised P_down fails by decades. TM over a lossless ε_rel < −1 is
# excluded (D12): the surface plasmon carries 2.5 % of P_in along the interface
# to infinity, in none of the three terms (finding 2).
RTOL_BALANCE_SUBSTRATE = 1e-12


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize(
    "background",
    [
        HalfSpace(2.25),
        HalfSpace(1.0),
        HalfSpace(2.25 + 0.1j),
        HalfSpace(-18.3 + 0.48j),
        HalfSpace(-1.0),
    ],
    ids=["glass", "low-n", "lossy", "silver", "metal-1"],
)
def test_dipole_power_balances_with_the_flux_into_the_substrate(pol, background):
    solver = _dipole_solver(pol, background)
    p_up, p_abs, p_in = _powers(solver, 633.0, 60.0, 330.0)
    p_down = _p_down(solver, 633.0, 60.0, 330.0)
    assert abs(p_up + p_down + p_abs - p_in) / p_in < RTOL_BALANCE_SUBSTRATE


# GT6: over a lossless dielectric the flux equals the far-field integral
# (1/8π)·w_far·∫|a_down|² dθ, w_far = 1 (TE), 1/ε (TM), split at the substrate
# critical angles with the sine map, 400 nodes per panel. Two quadratures of
# different form (q on the real axis, θ on the lower half circle). Measured
# 2.6e-14 (glass and low-n, TE/TM). Bar 3e-13, the spec's.
RTOL_FLUX_FAR = 3e-13


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("eps_sub", [2.25, 1.0], ids=["glass", "low-n"])
def test_flux_into_a_lossless_substrate_equals_the_far_field_integral(pol, eps_sub):
    xs, zs = 60.0, 330.0
    solver = _dipole_solver(pol, HalfSpace(eps_sub))
    res = solver.scatter_dipole(633.0, xs, zs)
    bnds, k, eps_rel, _, z_int = res._substrate()
    tr = Transmission.half_space(pol, k, eps_rel, z_int)
    k_bot = tr.k_bot
    edges = [0.5 * PI, 1.5 * PI]
    if k_bot.real > k:  # α₁(k_bot sin θ) turns evanescent past θ_c
        theta_c = np.arcsin(k / k_bot.real)
        edges += [PI - theta_c, PI + theta_c]
    x, wx = np.polynomial.legendre.leggauss(400)
    total = 0.0
    for lo, hi in zip(sorted(edges)[:-1], sorted(edges)[1:], strict=True):
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        th = c + h * np.sin(0.5 * PI * x)
        wt = h * 0.5 * PI * np.cos(0.5 * PI * x) * wx
        q = k_bot * np.sin(th)
        a_bot = alpha(q, k_bot)
        a_dip = (
            a_bot
            * tr.tau(q)
            * np.exp(1j * a_bot * z_int)
            * np.exp(-1j * q * xs + 1j * alpha(q, k) * (zs - z_int))
        )
        amp = _downward_amplitude(th, bnds, k, eps_rel, pol, z_int) + a_dip
        total += np.sum(wt * np.abs(amp) ** 2)
    p_far = (1.0 if pol == 2 else 1.0 / eps_rel.real) * total / (8 * PI)
    p_flux = _p_down(solver, 633.0, xs, zs)
    assert abs(p_flux - p_far) / p_flux < RTOL_FLUX_FAR


# --- result keys and the cluster ----------------------------------------------------


def test_result_keys_change_with_a_background():
    solver = _dipole_solver(2, HalfSpace(2.25))
    res = solver.scatter(633.0, angle=20.0)
    assert set(res.cross_sections()) == {"c_sca_up", "c_sca_down", "c_abs"}
    eff = res.efficiencies()
    assert set(eff) == {"qsca_up", "qsca_down", "qabs"}
    c = res.cross_sections()
    assert eff["qsca_up"] == pytest.approx(c["c_sca_up"] / (2 * 100.0), rel=1e-15)
    assert eff["qsca_down"] == pytest.approx(c["c_sca_down"] / (2 * 100.0), rel=1e-15)
    assert eff["qabs"] == pytest.approx(c["c_abs"] / (2 * 100.0), rel=1e-15)
    # And the cluster result reports the same pair, equal for a one-particle cluster.
    from pysie2d import ClusterBIESolver

    cl = ClusterBIESolver(
        Cluster([solver.geometry]),
        [solver.material],
        pol=2,
        background=solver.background,
    ).scatter(633.0, angle=20.0)
    assert set(cl.cross_sections()) == {"c_sca_up", "c_sca_down", "c_abs"}
    assert cl.cross_sections()["c_sca_up"] == pytest.approx(c["c_sca_up"], rel=1e-12)
    assert cl.cross_sections()["c_sca_down"] == pytest.approx(
        c["c_sca_down"], rel=1e-12
    )


def test_cross_section_quadrature_converges_in_the_number_of_angles():
    # Gauss–Legendre on a smooth upward amplitude: spec measured round-off by 64
    # nodes at kR = 10. The default of 500 must agree with 64 to the same level.
    res = _dipole_solver(1, HalfSpace(-18.3 + 0.48j)).scatter(633.0)
    a, b = res.cross_sections(64)["c_sca_up"], res.cross_sections(500)["c_sca_up"]
    assert a == pytest.approx(b, rel=1e-12)
    assert b > 0.0


# A substrate optically thinner than the cover puts a √ kink in r(k sin θ) at the
# critical angle, where plain Gauss–Legendre converges only algebraically: on
# main, 500 nodes were off by 2.1e-4 to 1.3e-3 against 2000 for these cases.
# Split at ±θ_c with the sine map, 500 and 2000 agree to 2.5e-14 (worst, ε = 1,
# TE); the bar is 10× that and four decades under the unsplit error.
RTOL_CRITICAL_ANGLE = 2.5e-13


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize(
    "background",
    [HalfSpace(1.0), HalfSpace(1.5), HalfSpace(1.0 + 1e-3j)],
    ids=["eps1", "eps1.5", "eps1-weak-loss"],
)
def test_cross_section_converges_over_a_substrate_thinner_than_the_cover(
    pol, background
):
    res = _dipole_solver(pol, background).scatter(633.0, angle=20.0)
    a, b = res.cross_sections(500)["c_sca_up"], res.cross_sections(2000)["c_sca_up"]
    assert a == pytest.approx(b, rel=RTOL_CRITICAL_ANGLE)


def test_substrate_equal_to_the_cladding_has_no_nan_at_grazing():
    # eps_rel = 1: r ≡ 0, but α₁ = α₂ = 0 at θ = ±π/2 would give 0/0.
    res = _dipole_solver(2, HalfSpace(1.33**2)).scatter(633.0)
    amp, _ = res.far_field(101)
    assert np.all(np.isfinite(amp))


def test_downward_cross_section_is_absent_at_complex_wavelength():
    # D18: the real-axis flux is undefined when Im k < 0, so the key is absent
    # rather than raising, as v0.9 does for the free-space keys.
    from pysie2d.solver import ScatterResult

    solver = _dipole_solver(2, HalfSpace(2.25))
    res = solver.scatter(633.0, angle=20.0)
    cplx = ScatterResult(
        res.ei, solver.geometry, solver.material, 633.0 + 31.6j, 20.0, solver.background
    )
    assert set(cplx.cross_sections()) == {"c_sca_up", "c_abs"}
    assert set(cplx.efficiencies()) == {"qsca_up", "qabs"}


@pytest.mark.parametrize(
    "background", [HalfSpace.pec(), HalfSpace(-4.0)], ids=["pec", "metal-4"]
)
def test_downward_cross_section_is_exactly_zero_where_nothing_enters(background):
    # D21 / GT8: PEC and a lossless Re ε ≤ 0 take no power, with no integral,
    # so the plasmon leak of a lossless TM metal (finding 2) does not show up as
    # a spurious c_sca_down either.
    for pol in (2, 1):
        res = _dipole_solver(pol, background).scatter(633.0, angle=20.0)
        assert res.cross_sections()["c_sca_down"] == 0.0
