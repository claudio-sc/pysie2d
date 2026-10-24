"""Transmitted field below a half-space (v1.1, spec transmitted-spec.md §5, §8).

The transmitted Green function is anchored three independent ways: at
``ε_rel = 1`` it must be the free-space ``(i/4)·H₀`` (closed form, real and
complex k); at real k it must equal a real-axis QUADPACK integral on the
radiation-condition branch (a second quadrature with a different branch rule);
and across the interface it must continue ``G_free + G_ind`` (which pins the
``z_top``/``z_bot`` phase references and the sign wiring).
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning
from scipy.special import hankel1

from pysie2d import BIESolver, Geometry, HalfSpace, Material
from pysie2d.layered import (
    Points,
    SommerfeldPath,
    Transmission,
    reflected_green,
    transmitted_green,
)
from pysie2d.reference.sommerfeld import gt_ref

PI = np.pi
N_CLAD, LAM = 1.33, 633.0
K_REAL = 2 * PI * N_CLAD / LAM
K_CPLX = 2 * PI * N_CLAD / (LAM * (1 + 0.05j))  # Q = 10
# The t1_green point set without its on-interface point: depths 1–800 nm,
# lateral offsets to 900 nm, two source heights.
OBS = np.array(
    [[0.0, -5.0], [150.0, -40.0], [-300.0, -1.0], [20.0, -800.0], [900.0, -200.0]]
)
SRC = np.array([[10.0, 15.0], [-40.0, 120.0]])
SUBSTRATES = {
    "glass": 2.25,
    "lossy": 2.25 + 0.1j,
    "silver": -18.3 + 0.48j,
    "low-n": 1.0,
}


def _green(k, eps_rel, pol, x, z, xs, zs, depth_scale=1.0):
    """G_t on a path sized as Transmission.build sizes it (finding 3)."""
    tr = Transmission.half_space(pol, k, eps_rel, 0.0)
    extent = max(np.ptp(np.r_[x, xs]), 1.0)
    z_min = zs + (-np.asarray(z)).min()
    path = SommerfeldPath.for_wavenumber(k, eps_rel, pol, extent, z_min, depth_scale)
    return transmitted_green(path, tr, x, z, xs, zs)


# --- GT1: ε_rel = 1 is free space -----------------------------------------------

# Measured worst 7.2e-14 relative (G and both source derivatives, TE/TM, real
# and complex k); the path quadrature's round-off floor. Bar 10×.
RTOL_FREE = 7.5e-13


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("k", [K_REAL, K_CPLX], ids=["real", "complex"])
def test_transmitted_green_without_an_interface_is_the_free_space_hankel(k, pol):
    # τ = 2/(2α₁) = 1/α₁ at ε = 1, so G_t is the Sommerfeld integral of
    # (i/4)·H₀(kρ) with ρ through the interface: no shared code with hank0.
    x, z = OBS[:, 0], OBS[:, 1]
    for xs, zs in SRC:
        g, gx, gz = _green(k, 1.0 + 0j, pol, x, z, xs, zs)
        rho = np.hypot(x - xs, z - zs)
        h1 = hankel1(1, k * rho)
        for got, ref in (
            (g, 0.25j * hankel1(0, k * rho)),
            (gx, 0.25j * k * h1 * (x - xs) / rho),
            (gz, 0.25j * k * h1 * (z - zs) / rho),
        ):
            assert np.max(np.abs(got - ref) / np.abs(ref)) < RTOL_FREE


# --- GT2: against the real-axis reference ---------------------------------------

# err/max|ref| worst 4.9e-11 (glass TM; silver TE 2.9e-11), at deep or far
# points where QUADPACK itself reports round-off — the reference is the limit
# (finding 4), not the path, which agrees with itself to 7e-15 (GT3). The
# spec's bar is 3e-10; it was set from the 2.9e-11 quote and is 6× the worst.
RTOL_REF = 3e-10


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("name", list(SUBSTRATES))
def test_transmitted_green_equals_the_real_axis_reference(name, pol):
    eps = SUBSTRATES[name] / N_CLAD**2
    x, z = OBS[:, 0], OBS[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        for xs, zs in SRC:
            got = _green(K_REAL, eps, pol, x, z, xs, zs)
            for which in range(3):
                ref = np.array(
                    [
                        gt_ref(xi - xs, zs, -zi, K_REAL, eps, pol, which)
                        for xi, zi in OBS
                    ]
                )
                assert np.max(np.abs(got[which] - ref)) / np.max(np.abs(ref)) < RTOL_REF


# --- GT3: path independence at complex k ----------------------------------------

# Hump depth × 1.5 at Q = 10, glass / silver / low-n, TE/TM: measured worst
# 8.8e-15 (low-n TM) over G and both derivatives. Bar 10×. A singularity on the
# wrong side of either path would show as an O(1) jump, not a round-off one.
RTOL_PATH = 9e-14


# δ·D at depth × 1.5 trips SommerfeldPrecisionWarning for low-n; the measured
# agreement above is what that warning would be guarding.
@pytest.mark.filterwarnings("ignore::pysie2d.layered.SommerfeldPrecisionWarning")
@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("name", ["glass", "silver", "low-n"])
def test_transmitted_green_at_complex_k_does_not_depend_on_the_path(name, pol):
    eps = SUBSTRATES[name] / N_CLAD**2
    x, z = OBS[:, 0], OBS[:, 1]
    xs, zs = SRC[0]
    a = _green(K_CPLX, eps, pol, x, z, xs, zs)
    b = _green(K_CPLX, eps, pol, x, z, xs, zs, depth_scale=1.5)
    for ga, gb in zip(a, b, strict=True):
        assert np.max(np.abs(ga - gb)) / np.max(np.abs(ga)) < RTOL_PATH


# --- GT4: continuity across the interface ---------------------------------------

# G_free + G_ind at z_int + 1e-12 against G_t at z_int − 1e-12: measured worst
# 1.1e-13 (silver TE). Bar 10×. Both paths are sized for the actual Z = h' + δ
# of an observer at the interface (finding 3: a reflected path sized at 2h'
# truncates the TM tail and gave 5e-12–3e-11).
RTOL_CONTINUITY = 1.2e-12
DELTA = 1e-12


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("k", [K_REAL, K_CPLX], ids=["real", "complex"])
@pytest.mark.parametrize("name", ["glass", "silver"])
def test_transmitted_green_continues_the_cover_field_across_the_interface(name, k, pol):
    eps = SUBSTRATES[name] / N_CLAD**2
    xs, zs = SRC[0]
    x = np.array([-200.0, 0.0, 30.0, 400.0])
    extent = np.ptp(np.r_[x, xs])
    g_t = _green(k, eps, pol, x, np.full_like(x, -DELTA), xs, zs)[0]
    path_r = SommerfeldPath.for_wavenumber(k, eps, pol, extent, zs + DELTA)
    g_ind = reflected_green(path_r, pol, k, eps, x, DELTA, xs, zs, 0.0)[0]
    g_free = 0.25j * hankel1(0, k * np.hypot(x - xs, DELTA - zs))
    assert np.max(np.abs(g_free + g_ind - g_t) / np.abs(g_t)) < RTOL_CONTINUITY


# --- GT12: the banded GEMM is the pointwise sum ---------------------------------

# 20 × 20 substrate map (x ∈ ±400, depth 5–600 nm) under the G14 particle,
# plane wave at 20°: measured worst 2.7e-16 over the whole map and 2.8e-16 on
# the 40-point sample the test checks (glass, silver, low-n; TE/TM), so the
# banding drops nothing. Bar 3e-15, the spec's (11× the t4_gemm 2.5e-16).
RTOL_GEMM = 3e-15


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("name", ["glass", "silver", "low-n"])
def test_banded_transmitted_blocks_equal_the_pointwise_green_sum(name, pol):
    geo = Geometry.gielis(rad=100.0, n_pts=256, m=0, x0=0.0, z0=115.0)
    mat = Material(n_core=2.0, n_clad=N_CLAD, pol=pol, epsi=0.5)
    solver = BIESolver(geo, mat, HalfSpace(SUBSTRATES[name]))
    res = solver.scatter(LAM, 20.0)
    eps = solver.background.eps_rel(N_CLAD, LAM)
    xg, zg = np.meshgrid(np.linspace(-400, 400, 20), -np.linspace(5, 600, 20))
    xo, zo = xg.ravel(), zg.ravel()
    tr = Transmission.build(pol, res.wnum_bg, eps, 0.0, geo.f, geo.g, xo, zo)
    m1, m2 = tr.blocks(Points(xo, zo), geo)
    n = geo.n_pts
    psi = -(m1 @ res.ei[:n] + m2 @ res.ei[n:])
    # The pointwise sum is the slow side; check the full GEMM map on 40 random
    # points of it, as t4_gemm checked 200 of a 100 × 100 map.
    idx = np.random.default_rng(0).choice(xo.size, 40, replace=False)
    g, gx, gz = tr.green(xo[idx, None], zo[idx, None], geo.f[None, :], geo.g[None, :])
    ref = -np.sum(
        geo.delt * ((gx * geo.dg - gz * geo.df) * res.ei[:n] + g * res.ei[n:]), axis=1
    )
    psi = psi[idx]
    assert np.max(np.abs(psi - ref)) / np.max(np.abs(ref)) < RTOL_GEMM


def test_build_returns_none_without_an_interface_and_refuses_misplaced_points():
    # PEC transmits nothing (D14); ε_rel = 1 has no interface (D19).
    f, g = np.array([0.0]), np.array([10.0])
    fo, go = np.array([0.0]), np.array([-10.0])
    assert Transmission.build(2, K_REAL, None, 0.0, f, g, fo, go) is None
    assert Transmission.build(2, K_REAL, 1.0 + 0j, 0.0, f, g, fo, go) is None
    with pytest.raises(ValueError, match="strictly below"):
        Transmission.build(2, K_REAL, 2.0 + 0j, 0.0, f, g, fo, -go)
    with pytest.raises(ValueError, match="strictly above"):
        Transmission.build(2, K_REAL, 2.0 + 0j, 0.0, f, -g, fo, go)


# --- eval_field below the interface (commit 2) ----------------------------------


def _g14(pol, background):
    geo = Geometry.gielis(rad=100.0, n_pts=256, m=0, x0=0.0, z0=115.0)
    mat = Material(n_core=2.0, n_clad=N_CLAD, pol=pol, epsi=0.5)
    return BIESolver(geo, mat, background)


# Scattered field at z_int ± δ: ψ(+δ) − ψ(−δ) = a·δ + b·δ² + …, so the Richardson
# combination 2·J(δ/2) − J(δ) of that difference is O(δ²) if ψ is continuous and
# O(1) if any sign or phase reference in the wiring is wrong. Measured at
# δ = 0.4 nm, relative to max|ψ|: glass 1.9e-6, silver 7.9e-5 (TE/TM, real and
# complex λ), falling exactly 4× per halving of δ. Bar 10× the worst.
RTOL_EVAL_CONTINUITY = 8e-4


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("lam", [LAM, LAM + 31.6j], ids=["real", "complex"])
@pytest.mark.parametrize("name", ["glass", "silver"])
def test_scattered_field_below_continues_the_scattered_field_above(name, lam, pol):
    from pysie2d.solver import ScatterResult

    solver = _g14(pol, HalfSpace(SUBSTRATES[name]))
    real = solver.scatter(LAM, 20.0)
    # Any boundary data is a valid density; re-wrapping at complex λ exercises
    # the complex-k path (D18) without a complex solve.
    res = ScatterResult(
        real.ei, solver.geometry, solver.material, lam, 20.0, solver.background
    )
    x = np.array([-150.0, 0.0, 80.0, 300.0])

    def jump(d):
        return res.eval_field(x, np.full(4, d)) - res.eval_field(x, np.full(4, -d))

    scale = np.max(np.abs(res.eval_field(x, np.full(4, 1.0))))
    resid = np.max(np.abs(2 * jump(0.2) - jump(0.4))) / scale
    assert resid < RTOL_EVAL_CONTINUITY


@pytest.mark.parametrize(
    "background",
    [HalfSpace(2.25), HalfSpace(-18.3 + 0.48j), HalfSpace.pec(), HalfSpace(1.33**2)],
    ids=["glass", "silver", "pec", "eps1"],
)
def test_cover_points_are_bit_identical_with_or_without_substrate_points(background):
    # GT11: the cover path is the v0.9 one; adding points below must not move
    # its path, centre or GEMM, so the cover values are equal bit for bit. One
    # cover point is inside the particle. The one-particle cluster must give the
    # same substrate field as the single-particle result, also bit for bit.
    from pysie2d import Cluster, ClusterBIESolver

    solver = _g14(1, background)
    res = solver.scatter(LAM, 20.0)
    cl = ClusterBIESolver(
        Cluster([solver.geometry]), [solver.material], pol=1, background=background
    ).scatter(LAM, 20.0)
    xa, za = (
        np.array([-300.0, 0.0, 50.0, 400.0, 0.0]),
        np.array([300, 40, 500, 90, 115.0]),
    )
    xb, zb = np.array([0.0, 100.0, -200.0]), np.array([-5.0, -50.0, -300.0])
    for r in (res, cl):
        mixed = r.eval_field(np.r_[xa, xb], np.r_[za, zb])
        assert np.array_equal(mixed[:5], r.eval_field(xa, za))
    assert np.array_equal(
        res.eval_field(xb, zb), cl.eval_field(np.r_[xa, xb], np.r_[za, zb])[5:]
    )
    with pytest.raises(ValueError, match="exactly at the interface"):
        cl.eval_field(np.array([0.0]), np.array([0.0]))  # D17, as the single result


def test_eval_field_without_a_background_is_the_bare_representation():
    # GT11, background=None: no split by z at all, so z < 0 is an ordinary point.
    from pysie2d.fields import eval_field

    res = _g14(2, None).scatter(LAM, 20.0)
    geo, mat = res.geometry, res.material
    x, z = np.array([0.0, 200.0, 0.0]), np.array([-50.0, 300.0, 115.0])
    ref = eval_field(
        res.ei,
        geo.n_pts,
        geo.f,
        geo.df,
        geo.g,
        geo.dg,
        geo.delt,
        res.wnum_bg,
        x,
        z,
        ri=mat.nc,
    )
    assert np.array_equal(res.eval_field(x, z), ref)


def test_field_below_pec_is_zero_and_without_an_interface_is_free_space():
    xb, zb = np.array([0.0, 100.0]), np.array([-5.0, -300.0])
    assert np.array_equal(
        _g14(2, HalfSpace.pec()).scatter(LAM, 20.0).eval_field(xb, zb), np.zeros(2)
    )
    # ε_sub = n_clad²: ε_rel == 1 exactly, the free-space representation (D19).
    plain = _g14(2, None).scatter(LAM, 20.0)
    same = _g14(2, HalfSpace(N_CLAD**2)).scatter(LAM, 20.0)
    assert np.array_equal(plain.eval_field(xb, zb), same.eval_field(xb, zb))


# --- GT5: downward far field ----------------------------------------------------

# The a_down formula at ε_rel = 1 against the free-space far field on the lower
# half (grazing ends excluded, where the formula is 0/0): measured 5.1e-14 (TE),
# 3.7e-14 (TM). Bar 1e-12, the spec's.
RTOL_DOWN_FREE = 1e-12


@pytest.mark.parametrize("pol", [2, 1])
def test_downward_amplitude_without_an_interface_is_the_free_space_far_field(pol):
    from pysie2d.fields import _amplitude_below, _far_field_at

    res = _g14(pol, HalfSpace(N_CLAD**2)).scatter(LAM, 20.0)
    geo, (bnds, k, _, _, z_int) = res.geometry, res._substrate()
    th = np.linspace(0.5 * PI + 1e-3, 1.5 * PI - 1e-3, 301)
    ff = _far_field_at(th, geo.n_pts, k, geo.f, geo.g, geo.df, geo.dg, geo.delt, res.ei)
    # The τ formula itself, bypassing the ε_rel == 1 short-circuit...
    got = _amplitude_below(th, bnds, Transmission.half_space(pol, k, 1.0 + 0j, z_int))
    assert np.max(np.abs(got - ff)) / np.max(np.abs(ff)) < RTOL_DOWN_FREE
    # ...and the public method, which is the free-space amplitude exactly (D19).
    amp, angles = res.far_field_down(301)
    assert angles[0] == 0.5 * PI and angles[-1] == 1.5 * PI
    ref = _far_field_at(
        angles, geo.n_pts, k, geo.f, geo.g, geo.df, geo.dg, geo.delt, res.ei
    )
    assert np.array_equal(amp, ref)


# Near-to-far: ψ/[(i/4)·√(2/(π k_bot ρ))·e^{i(k_bot ρ − π/4)}] → a_down with an
# O(1/ρ) residual inside the critical angle (t3b: exactly 4× per 4× depth), so
# the Richardson value 2·e(80 µm) − e(40 µm) removes it. Measured 1.7e-6 (TE)
# and 3.2e-6 (TM) over glass at offsets 0, ±0.3, ±0.6 rad from −z; the raw
# 80 µm value is 1e-3 off. Bar 3.5e-5 (spec, 11×). Never beyond θ_c = 1.09 rad,
# where the lateral wave breaks the a + c/ρ model (t3b). The low-n case of the
# spec is dropped under the §8.8 detour: each depth-80 µm path costs ~3.6 s in
# leggauss of its 6500-node hump, and low-n measured 1.7e-6 / 1.9e-6.
RTOL_NEAR_TO_FAR = 3.5e-5


@pytest.mark.parametrize("pol", [2, 1])
def test_deep_near_field_below_converges_to_the_downward_far_field(pol):
    from pysie2d.fields import _downward_amplitude

    res = _g14(pol, HalfSpace(2.25)).scatter(LAM, 20.0)
    k_bot = res.wnum_bg * np.sqrt(2.25 / N_CLAD**2)
    th = PI + np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    rho, tt = np.repeat([4e4, 8e4], 5), np.tile(th, 2)
    psi = res.eval_field(rho * np.sin(tt), rho * np.cos(tt))
    est = psi / (
        0.25j * np.sqrt(2 / (PI * k_bot * rho)) * np.exp(1j * (k_bot * rho - PI / 4))
    )
    rich = 2 * est[5:] - est[:5]
    a_down = _downward_amplitude(th, *res._substrate())
    assert np.max(np.abs(rich - a_down) / np.abs(a_down)) < RTOL_NEAR_TO_FAR
    amp, _ = res.far_field_down(11)
    assert amp[0] == 0 and amp[-1] == 0  # grazing, ε ≠ 1: α_bot(±k_bot) = 0


# --- GT8: guards ----------------------------------------------------------------


@pytest.mark.parametrize(
    "background,match",
    [
        (HalfSpace.pec(), "PEC"),
        (HalfSpace(2.25 + 0.1j), "absorbing"),
        (HalfSpace(-18.3 + 0.48j), "absorbing"),
        (HalfSpace(-4.0), "Re eps_rel"),
        (HalfSpace(-1.0), "Re eps_rel"),
    ],
    ids=["pec", "lossy", "silver", "metal-4", "metal-1"],
)
def test_far_field_down_raises_where_no_wave_reaches_infinity(background, match):
    from pysie2d import Cluster, ClusterBIESolver, NoFarFieldError

    solver = _g14(2, background)
    res = solver.scatter(LAM, 20.0)
    cl = ClusterBIESolver(
        Cluster([solver.geometry]), [solver.material], pol=2, background=background
    ).scatter(LAM, 20.0)
    for r in (res, cl):
        with pytest.raises(NoFarFieldError, match=match):
            r.far_field_down()


def test_far_field_down_without_a_background_is_refused():
    from pysie2d import Cluster, ClusterBIESolver, NoFarFieldError

    solver = _g14(2, None)
    cl = ClusterBIESolver(Cluster([solver.geometry]), [solver.material], pol=2)
    for r in (solver.scatter(LAM), cl.scatter(LAM)):
        with pytest.raises(ValueError, match="full circle") as err:
            r.far_field_down()
        assert not isinstance(err.value, NoFarFieldError)
    assert issubclass(NoFarFieldError, ValueError)


# --- GT9: incident_field --------------------------------------------------------

XS, ZS = 60.0, 330.0
# On the boundary nodes the unmasked background field is the φ half of the
# solver's own right-hand side, built by the same formulas: measured ≤ 1.1e-15
# (plane wave bit-identical; dipole on a differently sized path). Bar 1e-13, the
# spec's round-off bound.
RTOL_RHS = 1e-13
# incident_field at z_int ± 1e-9 nm: measured worst 8.5e-11 relative (silver;
# glass 3.0e-11), the O(δ·∂zψ) step of a continuous field. Bar 10×. A wrong
# phase reference or transmission factor is an O(1) jump.
RTOL_INCIDENT_CONTINUITY = 8.5e-10


def _rewrapped(solver, lam):
    """Plane-wave and dipole results at ``lam``, from real-λ solves.

    The background field depends only on the excitation, never on ei, so the
    complex-λ path is exercised without a complex solve.
    """
    from pysie2d.solver import ScatterResult

    geo, mat, bg = solver.geometry, solver.material, solver.background
    plane = ScatterResult(solver.scatter(LAM, 20.0).ei, geo, mat, lam, 20.0, bg)
    dip = solver.scatter_dipole(LAM, XS, ZS)
    assert dip.excitation == "dipole" and dip.source == (XS, ZS)
    dip = ScatterResult(
        dip.ei, geo, mat, lam, 0.0, bg, excitation="dipole", source=(XS, ZS)
    )
    return plane, dip


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("lam", [LAM, LAM + 31.6j], ids=["real", "complex"])
@pytest.mark.parametrize("name", ["glass", "silver"])
def test_background_field_on_the_boundary_is_the_solver_rhs(name, lam, pol):
    from pysie2d.sources import line_dipole_rhs, plane_wave_rhs

    solver = _g14(pol, HalfSpace(SUBSTRATES[name]))
    geo, nn = solver.geometry, solver.geometry.n_pts
    plane, dip = _rewrapped(solver, lam)
    k = plane.wnum_bg
    rhs_p = plane_wave_rhs(nn, 20.0, k, geo.f, geo.g)[:nn] + solver._reflection(
        lam
    ).plane_wave(20.0, geo.f, geo.g)
    refl = solver._reflection(lam, np.append(geo.f, XS), np.append(geo.g, ZS))
    rhs_d = (
        line_dipole_rhs(nn, k, geo.f, geo.g, XS, ZS)[:nn]
        + refl.green(geo.f, geo.g, XS, ZS)[0]
    )
    for res, rhs in ((plane, rhs_p), (dip, rhs_d)):
        got = res._background_field(geo.f, geo.g)
        assert np.max(np.abs(got - rhs)) / np.max(np.abs(rhs)) < RTOL_RHS


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("lam", [LAM, LAM + 31.6j], ids=["real", "complex"])
@pytest.mark.parametrize("name", ["glass", "silver"])
def test_incident_field_is_continuous_across_the_interface(name, lam, pol):
    x = np.array([-300.0, 0.0, 60.0, 250.0])
    up, down = np.full(4, 1e-9), np.full(4, -1e-9)
    for res in _rewrapped(_g14(pol, HalfSpace(SUBSTRATES[name])), lam):
        a, b = res.incident_field(x, up), res.incident_field(x, down)
        assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < RTOL_INCIDENT_CONTINUITY


def test_incident_field_contract():
    # D13: 0 inside the particle; the free-space formulas without a background;
    # 0 below PEC; refusals for a custom RHS, the interface and the source.
    from pysie2d import Cluster, ClusterBIESolver
    from pysie2d.kernels import hank0

    x, z = np.array([0.0, 200.0, -100.0]), np.array([115.0, 400.0, 600.0])
    bare = _g14(2, None)
    k = bare.material.wnum_bg(LAM)
    plane = bare.scatter(LAM, 20.0).incident_field(x, z)
    th = np.deg2rad(20.0)
    assert plane[0] == 0.0  # the particle centre
    assert np.array_equal(
        plane[1:], np.exp(1j * k * (x * np.sin(th) - z * np.cos(th)))[1:]
    )
    dip = bare.scatter_dipole(LAM, XS, ZS).incident_field(x, z)
    assert dip[0] == 0.0
    assert np.array_equal(dip[1:], 0.25j * hank0(k * np.hypot(x - XS, z - ZS))[1:])

    pec = _g14(2, HalfSpace.pec())
    for res in (pec.scatter(LAM, 20.0), pec.scatter_dipole(LAM, XS, ZS)):
        assert np.array_equal(
            res.incident_field([0.0, 50.0], [-5.0, -50.0]), np.zeros(2)
        )
        with pytest.raises(ValueError, match="exactly at the interface"):
            res.incident_field([0.0], [0.0])
    with pytest.raises(ValueError, match="own position"):
        pec.scatter_dipole(LAM, XS, ZS).incident_field([XS], [ZS])
    custom = bare.scatter(
        LAM, incident_rhs=lambda nn, k, f, g: np.ones(2 * nn, complex)
    )
    assert custom.excitation == "custom"
    with pytest.raises(ValueError, match="custom"):
        custom.incident_field([0.0], [400.0])

    # The one-particle cluster records the same excitation and agrees exactly.
    glass = _g14(1, HalfSpace(2.25))
    cl = ClusterBIESolver(
        Cluster([glass.geometry]), [glass.material], pol=1, background=glass.background
    )
    xa, za = np.array([0.0, 200.0, 0.0, 30.0]), np.array([115.0, 400.0, -40.0, -300.0])
    for single, clus in (
        (glass.scatter(LAM, 20.0), cl.scatter(LAM, 20.0)),
        (glass.scatter_dipole(LAM, XS, ZS), cl.scatter_dipole(LAM, XS, ZS)),
    ):
        assert np.array_equal(
            single.incident_field(xa, za), clus.incident_field(xa, za)
        )
    assert cl.scatter_dipole(LAM, XS, ZS).source == (XS, ZS)
