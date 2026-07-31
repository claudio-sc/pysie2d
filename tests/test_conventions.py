"""Guards for the vacuum-wavelength and background-index conventions (§2).

Every other test in the suite runs at ``n_clad = 1.0``, where the vacuum and
background readings of a wavelength coincide and every ``n_clad`` factor is
1. That is exactly why these tests exist: without them nothing in CI can tell
the two conventions apart, and the package documented one while implementing
the other for three releases.
"""

import numpy as np
import pytest

from conftest import N_CLAD, N_CORE, POL_TAG, RAD, size_parameter
from pysie2d import (
    BIESolver,
    Geometry,
    Material,
    relative_ldos,
    relative_ldos_map,
    self_green,
)
from pysie2d import size_parameter as public_size_parameter
from pysie2d.reference import mie
from pysie2d.reference.mie import self_green_cylinder

# The physics is invariant under a common rescaling of the background: what a
# scatterer does is set by the *relative* index m = n_core/n_clad and the size
# parameter x = 2π·n_clad·a/λ_vac, not by n_clad on its own. Holding both fixed
# while changing n_clad must therefore leave every efficiency bit-for-bit
# comparable. 1e-12 is a round-off budget, not a physics tolerance: the two
# runs execute the same arithmetic on inputs that agree to ~1 ulp.
RTOL_SCALE_INVARIANCE = 1.0e-12

# Reference case and its exact rescaling. n_clad 1.0 → 1.3 with n_core scaled by
# the same factor keeps m = 1.5; λ_vac must scale by 1.3 as well, or x picks up
# the factor and the comparison fails on a different size parameter rather than
# on the convention. This is the trap the spec's §2.5 wording left open.
BASE = {"n_clad": 1.0, "n_core": 1.5, "wavelength": 600.0}
SCALED = {"n_clad": 1.3, "n_core": 1.95, "wavelength": 780.0}


@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize("epsi_base", [0.0, 0.5])
def test_vacuum_wavelength_scaling(circle, pol, epsi_base):
    # If the solver read its wavelength as a medium wavelength, the two cases
    # would see different size parameters and disagree at the 1e-1 level.
    #
    # The lossy leg (epsi_base = 0.5) is simultaneously the exact test of the
    # absolute-epsi convention: epsi is scaled by n_clad² so that the *relative*
    # permittivity is unchanged. Under the relative reading the scaled case would
    # carry 1.69× the loss of the base case and qabs would differ by ~40 %.
    geom = circle(300)

    def run(case, epsi):
        mat = Material(n_core=case["n_core"], n_clad=case["n_clad"], pol=pol, epsi=epsi)
        return BIESolver(geom, mat).scatter(wavelength=case["wavelength"])

    base = run(BASE, epsi_base)
    scaled = run(SCALED, epsi_base * SCALED["n_clad"] ** 2)

    # The premise of the comparison: identical relative permittivity, identical
    # relative index m, identical size parameter x.
    assert scaled.material.eps == pytest.approx(base.material.eps)
    assert complex(scaled.material.nc) == pytest.approx(complex(base.material.nc))
    assert scaled.size_parameter == pytest.approx(base.size_parameter, rel=1e-15)

    eff_base = base.efficiencies()
    eff_scaled = scaled.efficiencies()
    for key in ("qsca", "qext", "qabs"):
        assert eff_scaled[key] == pytest.approx(
            eff_base[key], rel=RTOL_SCALE_INVARIANCE
        )


@pytest.mark.parametrize("pol", [1, 2])
def test_efficiencies_match_mie_in_cladding(circle, pol):
    # The analytic anchor at n_clad ≠ 1. Mie theory is stated in (x, m), both of
    # which are background-relative, so the same closed form must hold with no
    # extra n_clad factor anywhere. RTOL is the unchanged RTOL_MIE = 5e-3 of
    # test_efficiencies: the discretisation error is a property of the boundary
    # quadrature and does not care what n_clad is.
    geom = circle(300)
    mat = Material(n_core=SCALED["n_core"], n_clad=SCALED["n_clad"], pol=pol)
    result = BIESolver(geom, mat).scatter(wavelength=SCALED["wavelength"])

    ref = mie.efficiencies(result.size_parameter, complex(mat.nc))
    tag = POL_TAG[pol]

    assert result.efficiencies()["qsca"] == pytest.approx(
        ref[f"Q_sca_{tag}"], rel=5.0e-3
    )
    assert result.efficiencies()["qext"] == pytest.approx(
        ref[f"Q_ext_{tag}"], rel=5.0e-3
    )


@pytest.mark.parametrize("pol", [1, 2])
def test_absorbing_particle_in_cladding(circle, pol):
    # The lossy analytic anchor at n_clad ≠ 1. epsi is absolute, so reaching the
    # relative permittivity 2.25 + 0.5j — the case test_efficiencies already
    # validates at n_clad = 1 — takes epsi = 0.5·n_clad². Landing on that same
    # physical case is deliberate: it inherits the justified RTOL_MIE = 5e-3
    # rather than needing a fresh tolerance argument. Under the *relative*
    # reading of epsi this call would instead model 2.25 + 0.845j and miss the
    # reference by ~30 %, far outside the tolerance.
    geom = circle(300)
    epsi_abs = 0.5 * SCALED["n_clad"] ** 2
    mat = Material(
        n_core=SCALED["n_core"], n_clad=SCALED["n_clad"], pol=pol, epsi=epsi_abs
    )
    assert mat.eps == pytest.approx(complex(2.25, 0.5))
    result = BIESolver(geom, mat).scatter(wavelength=SCALED["wavelength"])

    ref = mie.efficiencies(result.size_parameter, complex(mat.nc))
    tag = POL_TAG[pol]

    assert result.efficiencies()["qabs"] > 0.0
    assert result.efficiencies()["qabs"] == pytest.approx(
        ref[f"Q_abs_{tag}"], rel=5.0e-3
    )


@pytest.mark.parametrize("pol", [1, 2])
def test_self_green_in_cladding(pol):
    # The near-field paths — scatter_dipole → line_dipole_rhs, eval_field — have
    # their own wavenumber plumbing, separate from the far-field path the tests
    # above exercise. Without this test a double-applied n_clad in
    # ScatterResult.wnum_bg or in green.relative_ldos_map passes the whole suite.
    #
    # Anchor and tolerance are inherited from test_green's
    # test_self_green_vs_analytic_cylinder: the Graf-addition-theorem closed form
    # at n_pts = 1000, where the near-field quantity converges at first order and
    # reaches 1 %. Neither the anchor nor the convergence rate depends on n_clad.
    geom = Geometry.gielis(rad=RAD, n_pts=1000, m=0)
    mat = Material(n_core=SCALED["n_core"], n_clad=SCALED["n_clad"], pol=pol)
    solver = BIESolver(geom, mat)
    lam = SCALED["wavelength"]
    d = 2.0 * RAD

    s_bie = self_green(solver, lam, d, 0.0)
    s_ref = self_green_cylinder(
        public_size_parameter(geom, mat, lam), complex(mat.nc), mat.wnum_bg(lam), d, pol
    )

    assert s_bie.real == pytest.approx(s_ref.real, rel=1e-2)
    assert s_bie.imag == pytest.approx(s_ref.imag, rel=1e-2)


def test_ldos_map_matches_scalar_ldos_in_cladding():
    # relative_ldos_map builds its own wnum_bg alongside solver.assemble
    # (green.py) — the one site in the package where two wavenumbers must agree
    # by construction rather than by sharing a call. Pinning the batched path to
    # the scalar one at n_clad ≠ 1 is what makes a drift between them visible.
    # Both run the same arithmetic, so they must agree to round-off, not physics.
    geom = Geometry.gielis(rad=RAD, n_pts=300, m=0)
    mat = Material(n_core=SCALED["n_core"], n_clad=SCALED["n_clad"], pol=2)
    solver = BIESolver(geom, mat)
    lam = SCALED["wavelength"]

    pts = np.array([2.0 * RAD, 3.0 * RAD, -2.5 * RAD])
    mapped = relative_ldos_map(solver, lam, pts, np.zeros_like(pts))
    scalar = np.array([relative_ldos(solver, lam, x, 0.0) for x in pts])

    assert np.allclose(mapped, scalar, rtol=1e-12, atol=0.0)


def test_epsi_is_absolute():
    # The defining statement of the convention, as arithmetic: epsi is referred
    # to vacuum exactly like n_core, so both divide by n_clad² on their way into
    # the relative permittivity the operator sees.
    mat = Material(n_core=1.95, n_clad=1.3, epsi=0.5)
    assert mat.eps.real == pytest.approx((1.95 / 1.3) ** 2)
    assert mat.eps.imag == pytest.approx(0.5 / 1.3**2)
    # nc² == eps holds for a passive material; the principal-root form does not
    # reproduce a gain medium (epsi < 0), which is outside the validated scope.
    assert complex(mat.nc) ** 2 == pytest.approx(mat.eps)

    # At n_clad = 1 the absolute and relative readings coincide — which is why
    # the rest of the suite cannot see the difference.
    vac = Material(n_core=1.5, n_clad=1.0, epsi=0.5)
    assert vac.eps == complex(2.25, 0.5)


def test_wnum_bg_carries_the_background_index():
    # The single conversion point. k_bg = 2π·n_clad/λ_vac, so a denser
    # background shortens the wavelength the operator actually sees.
    mat = Material(n_core=1.5, n_clad=1.3)
    assert mat.wnum_bg(600.0) == pytest.approx(2.0 * np.pi * 1.3 / 600.0)
    assert Material(n_core=1.5).wnum_bg(600.0) == pytest.approx(2.0 * np.pi / 600.0)

    # Complex wavelengths (the QNM path) pass through in kind.
    assert isinstance(mat.wnum_bg(600.0 + 20.0j), complex)


def test_primitives_take_a_wavenumber_not_a_wavelength():
    # A regression guard on the layering itself: the primitives must consume
    # their third argument *as a wavenumber*. Asserting the closed form is what
    # makes this discriminating — merely checking that two different arguments
    # give two different answers would also pass if the primitive re-derived a
    # wavelength from it, which is exactly the regression being guarded.
    from pysie2d import plane_wave_rhs

    geom = Geometry.gielis(rad=RAD, n_pts=64, m=0)
    mat = Material(n_core=N_CORE, n_clad=1.3)
    wnum_bg = mat.wnum_bg(600.0)

    rhs = plane_wave_rhs(geom.n_pts, 0.0, wnum_bg, geom.f, geom.g)
    expected = np.exp(1j * wnum_bg * -geom.g)  # θ = 0: fb = f·sin0 − g·cos0
    assert np.allclose(rhs[: geom.n_pts], expected, rtol=0, atol=1e-14)
    assert np.allclose(rhs[geom.n_pts :], 0.0)


def test_size_parameter_is_referred_to_the_cladding(circle):
    geom = circle(64)
    mat = Material(n_core=N_CORE, n_clad=1.3)
    assert public_size_parameter(geom, mat, 600.0) == pytest.approx(
        2.0 * np.pi * 1.3 * RAD / 600.0
    )


def test_size_parameter_matches_fixture(circle):
    # Pins the test-local helper to the shipped one, so the suite cannot drift
    # from the public definition.
    geom = circle(64)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD)
    assert public_size_parameter(geom, mat, 600.0) == pytest.approx(
        size_parameter(600.0)
    )


def test_size_parameter_rejects_non_circular_geometry():
    # x needs a physical radius. Geometry.rad is a Gielis scale parameter, so on
    # a star it is a number with no size-parameter meaning — refuse rather than
    # return it.
    star = Geometry.gielis(rad=RAD, n_pts=200, m=6, n1=1.0, n2=1.0, n3=1.0)
    assert not star.is_circle
    with pytest.raises(ValueError, match="circular"):
        public_size_parameter(star, Material(n_core=N_CORE), 600.0)
