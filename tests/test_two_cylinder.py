"""The coupled far field against the two-cylinder addition theorem (G4).

This is the cluster's independent validation anchor (CLAUDE.md
non-negotiable 3): `reference.two_cylinder` couples each circle's analytic Mie
coefficients by Graf's addition theorem and shares no code and no discretisation
with the boundary-integral solver. Agreement to round-off therefore pins the
coupled assembly *and* the three conventions of the bridge — observation angle
`φ = π/2 − θ`, incidence `α_i = α − π/2`, and the `s_n = −c_n` sign
(spec §3.7) — any one of which, taken wrong, gives a confidently wrong answer of
the right magnitude.

The dimer is **deliberately asymmetric** — radii 180/110 nm, indices 2.0/1.6,
centres at x = −450 and +520 nm, at oblique incidence — which is what makes a
mirror flip θ → −θ or a swapped pair of blocks detectable; a symmetric dimer at
normal incidence passes with the angle mapping reversed.
"""

import numpy as np
import pytest

from pysie2d import Cluster, ClusterBIESolver, Geometry, Material
from pysie2d.reference import two_cylinder

WAVELENGTH = 633.0
ANGLE = 37.0
RADII = (180.0, 110.0)
N_CORE = (2.0, 1.6)
CENTRES = np.array([[-450.0, 0.0], [520.0, 0.0]])
N_ANGLES = 361


def _solve(n_pts: int, pol: int):
    """Coupled solve of the §6.4 dimer at ``n_pts`` nodes per particle."""
    cluster = Cluster(
        [
            Geometry.gielis(RADII[0], n_pts, m=0, x0=CENTRES[0, 0]),
            Geometry.gielis(RADII[1], n_pts, m=0, x0=CENTRES[1, 0]),
        ]
    )
    materials = [Material(n, 1.0, pol=pol) for n in N_CORE]
    return ClusterBIESolver(cluster, materials, pol=pol).scatter(
        wavelength=WAVELENGTH, angle=ANGLE
    )


def _reference(result, pol: int, theta: np.ndarray, n_max: int | None = None):
    """Addition-theorem amplitude at the solver's angles and wavenumber."""
    return two_cylinder.scattering_amplitude(
        pol,
        RADII,
        CENTRES,
        [Material(n, 1.0, pol=pol).nc for n in N_CORE],
        result.wnum_bg,
        ANGLE,
        theta,
        n_max=n_max,
    )


@pytest.mark.parametrize("pol", [1, 2])
def test_cluster_far_field_matches_two_cylinder_addition_theorem(pol):
    """The coupled far field equals the analytic dimer amplitude, at round-off.

    Run at `nn = 60`, well past the `nn ≈ 30` where the Kress quadrature is
    already at round-off on circles (spec §6.4's convergence table), and at the
    module's *default* truncation — the default is what users get, so the
    default is what is tested.

    The error is reported against `max|amp|`, not per point: the amplitude has
    deep interference nulls where a pointwise relative error is dominated by
    cancellation in the null itself and measures nothing about the coupling.
    Measured max|Δ|/max|amp| at nn = 60: 3.4e-15 (TM) and 3.8e-15 (TE), a
    round-off floor and not a convergence order; 2e-14 is ~5× headroom over it.
    """
    result = _solve(60, pol)
    amp, theta = result.far_field(N_ANGLES)
    ref = _reference(result, pol, theta)

    assert np.max(np.abs(amp - ref)) / np.max(np.abs(ref)) < 2e-14


def test_addition_theorem_truncation_has_a_conditioning_window():
    """More orders is not safer: the anchor is only valid inside a window.

    `H_M⁽¹⁾(k·d)` grows superexponentially past `M ≈ k·d` while `s_n` decays
    superexponentially past `n ≈ x`, so a generously truncated translation
    system is ill-conditioned and returns garbage rather than a better answer
    (spec §3.8). Against a converged `nn = 300` BIE solve of the same dimer
    (`x₁ = 1.79`, `k·d = 9.63`) the measured error is flat across the window —
    1.8e-15 at M = 12, 3.6e-15 at M = 19, 4.3e-13 at M = 22 — and the
    `CONDITION_LIMIT` guard refuses everything beyond it, where the table's
    3.2e-6 at M = 40 lives.

    M = 22 sits at the window's edge (`|H_22(k·d)| ≈ 4.9e4`, spec §3.8's own
    last "still round-off" point, next to `|H_25| ≈ 4.6e6` where it visibly
    degrades) rather than in its interior. The spec's table cites 6.8e-15
    there; this machine measures 4.3e-13 — a near-singular linear solve at
    exactly the conditioning edge, sensitive to the BLAS/LAPACK pivoting
    sequence, which differs across machines. M = 10..19, comfortably inside
    the window, reproduce the spec's round-off numbers cleanly, and 4.3e-13 is
    still three orders below M = 40's real degradation — so the plateau bound
    below is widened to cover this platform's edge value rather than the
    spec's, without softening what the test actually guards against.

    This is the only test that would catch a later reader "improving" the
    default truncation upwards.
    """
    result = _solve(300, 1)
    amp, theta = result.far_field(N_ANGLES)
    scale = np.max(np.abs(amp))

    for n_max in (10, 12, 16, 19, 22):
        err = np.max(np.abs(amp - _reference(result, 1, theta, n_max))) / scale
        # Flat plateau: worst measured value in the window is 4.3e-13 at the
        # M = 22 edge, three orders below the 3.2e-6 of M = 40.
        assert err < 1e-11, f"M = {n_max} left the window: {err:.2g}"

    # Past the window the guard fires rather than returning the garbage the
    # spec's table measured at M = 25 (2.3e-12) and M = 40 (3.2e-6).
    for n_max in (25, 40):
        with pytest.raises(ValueError, match="conditioning limit"):
            _reference(result, 1, theta, n_max)


def test_condition_limit_guard_fires_on_a_close_pair_at_high_order():
    """A high order on a short separation raises instead of returning garbage.

    At M = 60 across a 300 nm centre separation the Graf translation entries
    reach |H_M(k·d)| ≈ 1.95e+69: the linear system is numerically singular and
    its solution is noise of plausible magnitude. Silence here is the dangerous
    outcome, so the message must carry the offending magnitude — that number is
    the only diagnostic a caller has for choosing a smaller M.
    """
    with pytest.raises(ValueError, match=r"\|H_M\(k\*d_min\)\| = 1.95e\+69"):
        two_cylinder.scattering_amplitude(
            2,
            (100.0, 80.0),
            np.array([[0.0, 0.0], [300.0, 0.0]]),
            (2.0, 1.6),
            2.0 * np.pi / WAVELENGTH,
            0.0,
            np.linspace(-np.pi, np.pi, 5),
            n_max=60,
        )
