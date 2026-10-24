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
