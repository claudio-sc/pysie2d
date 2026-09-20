"""Cylindrical-harmonic decomposition of the scattered field.

The anchor is analytic Mie on a circle (§3.4 of
``docs/design/multipole-spec.md``): for a circular particle under a plane wave
at incidence angle α the signed-order coefficients are, exactly,

    c_m = −(−1)^m · χ_m · e^{i m α_r}

with χ_m = b_m for pol = 2 (TE) and a_m for pol = 1 (TM). That relation was
measured against this solver rather than copied from a textbook, so no sign or
phase in it is a guess. Everything else here is an invariance the coefficients
must have — translation, scale covariance, grid convergence, a second
projection — plus the guards, each of which stands in front of a measured
silent failure.
"""

import numpy as np
import pytest
from scipy.special import hankel1

from conftest import N_CLAD, N_CORE, RAD
from pysie2d import (
    BIESolver,
    Geometry,
    Material,
    multipole_decompose,
    multipole_reconstruct,
)
from pysie2d.reference import mie

WAVELENGTH = 600.0
N_PTS = 100
MMAX = 8

# The star of the spec's fixtures: circumscribing radius 1.587 × rad, so
# 1.5 × rad sits *inside* it — the exact mistake the legacy "1.5 × rad_ref"
# docstring invited, and what test_guards pins.
STAR = {"m": 5, "n1": 3.0, "n2": 6.0, "n3": 6.0}

# The three Mie tests evaluate at 1.5 × the circumscribing radius rather than
# the 3.0 × default. The Mie relation is independent of the evaluation radius —
# the radial factor i^m·H_m^(1)(k·r0) is divided out of the projection — but
# the residual is a handful of ulp of a coefficient of magnitude ~1.9, and it
# jitters across the whole 2.4e-15 … 3.1e-15 band with a one-ulp change of r0
# (measured: r0 = 600.0 gives 2.58e-15, the next float up gives 3.12e-15).
# The spec's atol = 3e-15 sits inside that band, so it is pinned here to the
# radius where it was measured (worst 2.781e-15, matching the spec's 2.795e-15)
# instead of being widened. The default radius is exercised by the
# reconstruction, grid-convergence and guard tests below.
MIE_R0 = 1.5


def _star(rad: float = RAD, n_pts: int = N_PTS, **kwargs) -> Geometry:
    return Geometry.gielis(rad=rad, n_pts=n_pts, **STAR, **kwargs)


def _circ_radius(geom: Geometry) -> float:
    return float(np.hypot(geom.f - geom.x0, geom.g - geom.z0).max())


def _solve(geom: Geometry, *, pol: int = 2, wavelength=WAVELENGTH, angle=0.0):
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol)
    return BIESolver(geom, mat).scatter(wavelength, angle)


def _mie_reference(size_parameter, pol: int, angle: float, mmax: int = MMAX):
    """(c_plus, c_minus) predicted by analytic Mie — spec §3.4."""
    a, b = mie.mie_coefficients(size_parameter, N_CORE / N_CLAD, n_max=mmax)
    chi = b if pol == 2 else a
    m = np.arange(mmax + 1)
    alpha_r = np.deg2rad(angle)
    c_plus = -2.0 * (-1.0) ** m * chi * np.cos(m * alpha_r)
    c_plus[0] = -chi[0]  # the m = 0 term appears once in the sum, not twice
    c_minus = (-2.0j * (-1.0) ** m * chi * np.sin(m * alpha_r))[1:]
    return c_plus, c_minus


@pytest.mark.parametrize("pol", [2, 1])
def test_mie_coefficients_TE_and_TM_normal_incidence(circle, pol):  # noqa: N802
    """The coefficients of a circle are the analytic Mie cylinder coefficients.

    This cannot pass by accident: it pins the sign, the (−1)^m alternation and
    the i^m basis factor of every retained order at once, against a closed form
    computed by an independent code path (``reference.mie``).

    The ``c_minus`` half of this test is **trivially** satisfied: at normal
    incidence sin(mα) = 0, so an implementation whose antisymmetric half is
    broken passes here regardless. ``test_mie_coefficients_TE_and_TM_oblique_
    incidence`` is what actually checks it.
    """
    geom = circle(n_pts=N_PTS)
    result = _solve(geom, pol=pol)
    mp = result.multipoles(mmax=MMAX, r0=MIE_R0 * _circ_radius(geom))
    c_plus, c_minus = _mie_reference(result.size_parameter, pol, 0.0)

    # Round-off floor, not a convergence order: under Kress quadrature the
    # circle anchors reach round-off by nn ≈ 30–40 and this runs at 100.
    # Measured worst 2.795e-15, against coefficients of magnitude 1.86.
    np.testing.assert_allclose(mp.c_plus, c_plus, rtol=0, atol=3e-15)
    np.testing.assert_allclose(mp.c_minus, c_minus, rtol=0, atol=3e-15)


@pytest.mark.parametrize("pol", [2, 1])
def test_mie_coefficients_TE_and_TM_oblique_incidence(circle, pol):  # noqa: N802
    """Oblique incidence is the only test that exercises ``c_minus``.

    At normal incidence the antisymmetric coefficients vanish by symmetry, so
    half the output of the decomposition is untested there. At α = 30° they are
    of order one, and the ``max|c_minus| > 0.7`` assertion below makes the test
    fail loudly — rather than trivially pass — if a refactor ever makes that
    half vanish.
    """
    geom = circle(n_pts=N_PTS)
    result = _solve(geom, pol=pol, angle=30.0)
    mp = result.multipoles(mmax=MMAX, r0=MIE_R0 * _circ_radius(geom))
    c_plus, c_minus = _mie_reference(result.size_parameter, pol, 30.0)

    # Measured worst 2.763e-15; same round-off floor as the normal-incidence
    # case, against coefficients of magnitude 1.86.
    np.testing.assert_allclose(mp.c_plus, c_plus, rtol=0, atol=3e-15)
    np.testing.assert_allclose(mp.c_minus, c_minus, rtol=0, atol=3e-15)
    assert np.abs(mp.c_minus).max() > 0.7


def test_mie_relation_holds_at_complex_wavelength(circle):
    """The projection is exact at complex k, which is what QNM work needs.

    Complex wavenumbers must work on every path in this package (CLAUDE.md
    non-negotiable 1) — that is what makes quasi-normal-mode extraction
    possible, and it is why no path here may be simplified to real-only
    arithmetic however dead the complex branch looks. The projection's
    orthogonality lives in θ, which is real, so the Mie relation holds
    unchanged at a complex size parameter.
    """
    wavelength = 600.0 + 30.0j
    geom = circle(n_pts=N_PTS)
    result = _solve(geom, pol=2, wavelength=wavelength, angle=30.0)
    mp = result.multipoles(mmax=MMAX, r0=MIE_R0 * _circ_radius(geom))
    c_plus, c_minus = _mie_reference(result.size_parameter, 2, 30.0)

    # Measured 8.671e-16 — the same round-off floor as the real-k cases.
    np.testing.assert_allclose(mp.c_plus, c_plus, rtol=0, atol=3e-15)
    np.testing.assert_allclose(mp.c_minus, c_minus, rtol=0, atol=3e-15)


def test_reconstruction_matches_the_solver_field_on_a_second_circle():
    """Decompose on one circle, reconstruct on another: self-consistency only.

    This is **not** an independent validation. It shows the expansion
    reproduces the solver's own field at a radius it was never fitted at, and
    nothing more — for a non-circular shape no closed form exists, and the
    external anchor is the v0.9 milestone. What it does catch is any error in
    the radial basis (the i^m and H_m^(1) factors), which a fit on a single
    circle would absorb into the coefficients.

    The residual here is set by **truncation** in ``mmax``, whereas
    ``test_default_angular_grid_is_converged`` is set by **quadrature** in
    ``ntheta``: two different knobs. Measured convergence in mmax at the
    default r0, reconstructing at 6·r_circ: 8 → 3.7e-06, 12 → 1.3e-09,
    16 → 1.3e-13, 20 → 1.6e-15, 24 → 1.1e-15 — so mmax = 20 has just reached
    the round-off floor of a field of amplitude 0.9.
    """
    geom = _star()
    result = _solve(geom, angle=30.0)
    r_circ = _circ_radius(geom)
    mp = result.multipoles(mmax=20)  # default r0 = 3·r_circ

    theta = np.linspace(0.0, 2.0 * np.pi, 97, endpoint=False)
    r_out = 6.0 * r_circ
    rebuilt = mp.reconstruct(r_out, theta)
    direct = result.eval_field(r_out * np.sin(theta), r_out * np.cos(theta))

    # Measured 1.64e-15 at mmax = 20 against a field of amplitude 0.9, ~60×
    # below this bound. Tight deliberately: dropping four retained orders
    # (mmax = 16) gives 1.3e-13 and fails here, so the bound still pins the
    # truncation behaviour rather than only the round-off floor.
    np.testing.assert_allclose(rebuilt, direct, rtol=0, atol=1e-13)


def test_expansion_centre_follows_the_particle():
    """Moving the particle only rephases the coefficients.

    ``plane_wave_rhs`` phases the incident field to the **origin** while the
    expansion is about the **particle centre**, so a translation multiplies
    every coefficient by exp(i·k·(x0 sin α − z0 cos α)) and changes nothing
    else. This is what catches an expansion accidentally built about the origin
    — a bug the Mie tests, which sit the particle at the origin, cannot see.
    """
    angle = 30.0
    x0, z0 = 500.0, -300.0
    at_origin = _solve(_star(), angle=angle).multipoles(mmax=MMAX)
    moved = _solve(_star(x0=x0, z0=z0), angle=angle).multipoles(mmax=MMAX)

    alpha_r = np.deg2rad(angle)
    k = 2.0 * np.pi * N_CLAD / WAVELENGTH
    phase = np.exp(1j * k * (x0 * np.sin(alpha_r) - z0 * np.cos(alpha_r)))

    # Measured 1.419e-15 at this displacement, against coefficients of order 1.
    np.testing.assert_allclose(moved.c, at_origin.c * phase, rtol=0, atol=3e-15)


@pytest.mark.parametrize("s", [2.0, 7.0])
def test_coefficients_are_scale_covariant(s):
    """(rad, λ) → (s·rad, s·λ) leaves every c_m unchanged, exactly.

    The whole problem depends only on k·rad (``docs/conventions.md`` §9) and
    the c_m are dimensionless, so the scaled solve must return the same
    numbers — not approximately, but to round-off, since the discrete system
    matrix is entrywise identical. A coefficient carrying a stray absolute
    length would break this and nothing else would notice.
    """
    base = _solve(_star(), angle=30.0).multipoles(mmax=MMAX)
    scaled = _solve(_star(rad=s * RAD), angle=30.0, wavelength=s * WAVELENGTH)
    scaled_mp = scaled.multipoles(mmax=MMAX)

    # Measured exactly 0.0 at s = 2 (a power of two, so the scaling is exact in
    # binary) and 3.189e-15 at s = 7, where it is not.
    np.testing.assert_allclose(scaled_mp.c, base.c, rtol=0, atol=1e-14)


@pytest.mark.parametrize("shape", ["circle", "star"])
def test_default_angular_grid_is_converged(circle, shape):
    """The default ntheta already exhausts the field's angular content.

    The default rule 2·(mmax + ⌈|k·r0|⌉) + 16 must resolve the field itself,
    not merely separate the retained orders. A 4× finer grid is the check: if
    the default were under-resolved the two would differ at the level of the
    aliased content, which is orders of magnitude above round-off. The
    ``spectrum_tail`` assertion pins the diagnostic that reports the same
    thing to a user who passes their own grid.
    """
    geom = circle(n_pts=N_PTS) if shape == "circle" else _star()
    result = _solve(geom, angle=30.0)
    coarse = result.multipoles(mmax=MMAX)
    fine = result.multipoles(mmax=MMAX, ntheta=4 * coarse.ntheta)

    # Measured 2.5e-16 … 6.6e-16 over the four cases of spec §2, D3 — a
    # round-off floor, so any real under-resolution is decades above it.
    np.testing.assert_allclose(fine.c, coarse.c, rtol=0, atol=1e-14)
    # Measured ~1e-16 at the default, six decades below the 1e-10 warn screen.
    assert coarse.spectrum_tail < 1e-12


@pytest.mark.parametrize("wavelength", [WAVELENGTH, 600.0 + 30.0j])
def test_symmetric_basis_matches_an_independent_projection(wavelength):
    """A second, independent projection reproduces ``c_plus``/``c_minus``.

    The ± coefficients are derived from the signed-order c_m by a two-line
    transform, and a transform cannot check itself. This test projects the
    field onto the explicitly constructed basis functions
    ψ_m^± = i^m H_m^(1)(k r) · {cos mθ, i sin mθ} instead, which is the
    formulation the research code used — the same argument that keeps
    ``assemble_matrix_reference`` in the tree. In particular it is what proves
    the ``c_0^+ = c_0`` special case is *right* rather than merely consistent:
    an implementation returning 2·c_0 agrees with itself everywhere.
    """
    result = _solve(_star(), angle=30.0, wavelength=wavelength)
    mp = result.multipoles(mmax=MMAX)

    theta = (np.arange(mp.ntheta) + 0.5) * 2.0 * np.pi / mp.ntheta
    field = result.eval_field(
        mp.r0 * np.sin(theta) + mp.x0, mp.r0 * np.cos(theta) + mp.z0
    )
    c_plus = np.empty(MMAX + 1, dtype=complex)
    c_minus = np.empty(MMAX, dtype=complex)
    for m in range(MMAX + 1):
        radial = 1j**m * hankel1(m, mp.wnum_bg * mp.r0)
        even = radial * np.cos(m * theta)
        c_plus[m] = np.dot(field, even.conj()) / np.sum(np.abs(even) ** 2)
        if m >= 1:
            odd = radial * 1j * np.sin(m * theta)
            c_minus[m - 1] = np.dot(field, odd.conj()) / np.sum(np.abs(odd) ** 2)

    # Measured 8.0e-16 (real k) and 6.3e-16 (complex k) on the star — a
    # round-off floor between two algebraically identical projections of the
    # same sampled field.
    np.testing.assert_allclose(mp.c_plus, c_plus, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mp.c_minus, c_minus, rtol=0, atol=1e-15)


def test_evaluation_circle_must_exceed_the_circumscribing_radius():
    """Inside the circumscribing circle the expansion does not converge.

    And it fails **silently**: measured 58 % error at 0.9 × the circumscribing
    radius, with a healthy-looking spectrum and nothing raised. The star's
    circumscribing radius is 1.587 × ``Geometry.rad``, so the legacy advice
    "1.5 × rad is typically sufficient" puts the evaluation circle at
    0.945 × r_circ — which this guard rejects by name.
    """
    geom = _star()
    result = _solve(geom)
    r_circ = _circ_radius(geom)

    for bad_r0 in (r_circ, np.nextafter(r_circ, 0.0), 1.5 * RAD):
        with pytest.raises(ValueError, match="circumscribing radius"):
            result.multipoles(r0=bad_r0)


def test_angular_grid_must_separate_the_retained_orders():
    """Below ntheta = 2·mmax + 1 the retained orders alias into each other.

    Measured at mmax = 8: ntheta = 16 gives a coefficient error of 2.4e+12 —
    twelve decades, silently — while ntheta = 17 gives 7.1e-12. The
    spectrum-tail diagnostic does **not** catch this case (its tail at
    ntheta = 16 is a healthy 1.2e-6), so this inequality is the only thing
    standing between a user and that error. It is not redundant with the tail
    screen and must not be simplified away.
    """
    result = _solve(_star())
    with pytest.raises(ValueError, match="not orthogonal"):
        result.multipoles(mmax=MMAX, ntheta=2 * MMAX)


def test_overflowing_hankel_order_raises():
    """``hankel1`` overflows to inf at high order, and every c_m beyond is junk.

    First non-finite order is m = 133 at k·r0 = 0.5 and 217 at 6.28, so it is
    not reachable at a sane mmax — but past it the division by an infinite
    radial factor returns 0 or nan for every order, with no other signal.
    """
    result = _solve(_star())
    with pytest.raises(ValueError, match="overflow"):
        result.multipoles(mmax=400)


def test_under_resolved_angular_grid_warns():
    """A legal but marginal ntheta is reported through ``spectrum_tail``.

    ntheta = 2·mmax + 1 passes the orthogonality guard, yet it does not
    resolve the field itself once |k·r0| is comparable to mmax. The diagnostic
    warns rather than raising because it is a screen, not a calibrated error
    bar: tail 1.1e-4 went with a coefficient error of 1.1e-12, tail 8.5e-1
    with 1.2.
    """
    result = _solve(_star())
    with pytest.warns(RuntimeWarning, match="spectrum tail"):
        result.multipoles(mmax=2, ntheta=5)


def test_reconstruct_rejects_an_even_length_coefficient_array():
    """``c`` holds the signed orders −mmax … mmax and is therefore odd-length.

    An even-length array is a ``c_plus`` or ``c_minus`` array passed by
    mistake; without this check it would be silently reinterpreted as a signed
    set of a different mmax and return a plausible wrong field.
    """
    with pytest.raises(ValueError, match="odd length"):
        multipole_reconstruct(np.ones(8, dtype=complex), 0.01, 500.0, np.zeros(3))


def test_negative_mmax_raises():
    """A negative mmax has no expansion to return; reject it before any work."""
    geom = _star()
    result = _solve(geom)
    with pytest.raises(ValueError, match="non-negative"):
        multipole_decompose(
            result.ei,
            geom.n_pts,
            geom.f,
            geom.df,
            geom.g,
            geom.dg,
            geom.delt,
            result.wnum_bg,
            2.0 * _circ_radius(geom),
            mmax=-1,
        )
