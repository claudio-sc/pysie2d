"""Thin, high-aspect boundaries on the eccentric-angle map (conventions §13.4).

A thin ellipse's tips have curvature radius rad/AR² and span ~1/AR of θ, so
the default uniform-θ map leaves them almost without nodes and every result
drifts by O(1) between nn and 2nn. These tests pin that
``Parametrisation.elliptic`` fixes it, against a closed form and by convergence,
and that the worst cases of the default now warn.
"""

import warnings

import numpy as np
import pytest

from pysie2d import (
    BIESolver,
    BoundaryResolutionWarning,
    Cluster,
    ClusterBIESolver,
    Geometry,
    Material,
    Parametrisation,
)

N_CLAD = 1.45
WAVELENGTH = 1100.0


def _thin_ellipse(rad, nn, aspect):
    """Long semi-axis ``rad`` along x, short ``rad/aspect`` along z."""
    a, b = 1.0 / aspect, 1.0
    return Geometry.gielis(
        rad, nn, a=a, b=b, parametrisation=Parametrisation.elliptic(a=a, b=b)
    )


def test_thin_ellipse_reaches_the_quasistatic_plasmon_condition():
    """The long-axis dipole eigenpermittivity tends to ε/ε_bg = −L/S as rad → 0.

    Independent anchor: the electrostatic depolarisation factor of an elliptic
    cylinder along its long semi-axis L is S/(L + S), so the TM dipole
    resonance sits at ε = −(L/S)·ε_bg in the quasi-static limit. Nothing in the
    solver knows this number.

    Retardation shifts it by O((kR)² ln kR), so the test asks for two things:
    the gap lies under (kR)²·|ln kR| at both sizes (measured 9.2e-6 and 3.4e-5
    against 2.7e-5 and 9.4e-5), and it shrinks by a factor in (3, 4] per halving
    of rad. A pure R² gives 4 and the log lowers it (measured 3.65). A
    discretisation floor or a wrong limit drives the ratio toward 1.

    nn = 256 is converged here (eigenpermittivity changes by 1.7e-10 to
    nn = 512). On the default uniform-θ map at the same nn the box holds no
    mode at all at aspect 20, and at aspect 30 one 5 % off the limit.
    """
    aspect = 20.0
    target = -aspect * N_CLAD**2
    gaps = []
    for rad in (0.25, 0.5):
        solver = ClusterBIESolver(
            Cluster([_thin_ellipse(rad, 256, aspect)]),
            [Material.from_eps(-1.0, N_CLAD, pol=1)],
            pol=1,
        )
        modes = solver.at_wavelength(WAVELENGTH).eigenpermittivities(
            1.3 * target - 0.5j, 0.7 * target + 0.5j
        )
        assert len(modes.eps) == 1 and modes.converged[0]
        gap = abs(modes.eps[0] - target) / abs(target)
        kr = 2.0 * np.pi * N_CLAD * rad / WAVELENGTH
        assert gap <= kr**2 * abs(np.log(kr))
        gaps.append(gap)
    assert 3.0 < gaps[1] / gaps[0] <= 4.0


def test_thin_metal_ellipse_converges_spectrally_on_the_elliptic_map():
    """Aspect 30, ε = −53.3 + 3.8i, TM: qext settled by nn = 384.

    The case the default map gets wrong silently: on uniform θ the same pair of
    rungs differs by 0.95. On the eccentric map it is 5.0e-6 at nn = 256 and
    9.8e-10 at 384 against 768, the spectral rate the analytic composite
    boundary allows. 1e-8 sits a decade above the measured 384 value and 2.7
    decades below the 256 one, so it separates a converged rung from an
    unconverged one rather than tracking noise.

    The long semi-axis is 70 nm, inside the regime where the interior Kress
    splitting is still at round-off for this permittivity.
    """
    material = Material.from_eps(complex(-53.3, 3.8), N_CLAD, pol=1)
    qext = {
        nn: BIESolver(_thin_ellipse(70.0, nn, 30.0), material)
        .scatter(WAVELENGTH)
        .efficiencies()["qext"]
        for nn in (384, 768)
    }
    assert qext[384] == pytest.approx(qext[768], rel=1e-8)


def test_default_map_on_a_thin_ellipse_warns_and_the_elliptic_map_does_not():
    """The silent failure is no longer silent; the fix does not trip the alarm.

    Aspect 20 at nn = 256: on uniform θ the tangent turns by 2.74 rad between
    two nodes straddling a tip, and qext there is off by O(1) (measured 0.99 to
    14 between nn = 256 and 512 across gold and n = 2). On the eccentric map
    the largest turn is 0.48 rad and qext is settled to 1.5e-8. The 2 rad limit
    sits between the two with a margin of 1.4 and 4 respectively, so neither
    assertion rests on the threshold's last digit.
    """
    with pytest.warns(BoundaryResolutionWarning, match="elliptic"):
        Geometry.gielis(50.0, 256, a=1.0 / 20.0, b=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryResolutionWarning)
        _thin_ellipse(50.0, 256, 20.0)
        Geometry.gielis(50.0, 256, m=0)  # a circle: 2π/256 per node
