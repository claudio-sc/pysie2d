"""Shape a Gielis antenna to launch a slab mode one way (slab-spec R12, T4).

A Si particle sits in the cover of an SOI slab (220 nm Si in SiO₂, 1550 nm) next
to a line source at (0, 60) nm. The objective is the source's efficiency into
the fundamental mode travelling towards +x, ``ModeAmplitudes.efficiency(0, +1)``;
L-BFGS-B maximises it over the antenna's gap to the slab, its x offset and two
superformula parameters, with gradients from ``pysie2d.mode_sensitivity``
(central differences on the frozen node set, two solves per parameter).

Default (T4): pol = 2, slab TE₀ in cross-section, the scalar source, all the
directionality coming from the antenna's asymmetry. ``--tm-inplane``: pol = 1,
slab TM₀, an in-plane electric dipole spin-locked to TM₀ (``b_z = iβb_x/γ₁``),
which is one-way on its own; the antenna then works with or against it.

Bounds: gap ≥ 10 nm (the slab-spec minimum the path and the interface-gap
guard are measured for). A non-closing superformula is caught as a penalty.
Not run in CI.

    uv run python examples/waveguide_antenna.py [--tm-inplane]
"""

import argparse

import numpy as np
from scipy.optimize import minimize

from pysie2d import BIESolver, Geometry, Material, Stack, mode_sensitivity
from pysie2d.geometry import NonClosingBoundaryError

LAM, N_CLAD, N_SI = 1550.0, 1.444, 3.476
SOI = Stack(((N_SI**2, 220.0),), N_CLAD**2)
SOURCE = (0.0, 60.0)
RAD, N_PTS, M = 80.0, 160, 4  # even m closes for any b; odd m only at a = b
# p = (gap above the slab, x offset of the centre, sine scale b, exponent n1).
P0 = np.array([30.0, 260.0, 1.0, 3.0])
BOUNDS = [(10.0, 200.0), (200.0, 800.0), (0.6, 1.6), (1.5, 8.0)]
PENALTY = 1.0  # efficiency is in [0, 1]: a non-closing shape scores −1


def antenna(p: np.ndarray, base: Geometry | None) -> Geometry:
    """The Gielis particle for parameters p, its lowest point `gap` above z_top."""
    gap, x0, b, n1 = p
    kw = {"m": M, "n1": n1, "n2": 2.0, "n3": 2.0, "b": b}
    if base is not None:
        kw["parametrisation"] = base.parametrisation
    probe = Geometry.gielis(rad=RAD, n_pts=N_PTS, **kw)
    return Geometry.gielis(rad=RAD, n_pts=N_PTS, x0=x0, z0=gap - probe.g.min(), **kw)


def make_run(pol: int, moment):
    """The objective's solve: ``(geometry, material) → ModeAmplitudes`` at x_ref = 0."""

    def run(geom, mat):
        res = BIESolver(geom, mat, SOI).scatter_dipole(LAM, *SOURCE, moment=moment)
        return res.mode_amplitudes(0.0)

    return run


def objective(p, pol, moment, base, steps):
    """−efficiency(0, +1) and its gradient, one mode_sensitivity per parameter."""
    mat = Material(n_core=N_SI, n_clad=N_CLAD, pol=pol)
    run = make_run(pol, moment)
    try:
        amps = run(antenna(p, base), mat)
        grad = np.empty(p.size)
        for i in range(p.size):

            def at(delta, i=i):
                q = p.copy()
                q[i] += delta
                return antenna(q, base), mat

            d = mode_sensitivity(at, run, step=steps[i])
            # d(P₀⁺/P_src) = (dP₀⁺·P_src − P₀⁺·dP_src)/P_src²
            grad[i] = (
                d["p_plus"][0] * amps.p_source - amps.p_plus[0] * d["p_source"]
            ) / amps.p_source**2
    except NonClosingBoundaryError:
        return PENALTY, np.zeros(p.size)
    eff = amps.efficiency(0, +1)
    print(
        f"  gap {p[0]:6.1f}  x0 {p[1]:6.1f}  b {p[2]:.3f}  n1 {p[3]:.3f}"
        f"  →  η₊ = {eff:.4f}  (η₋ = {amps.efficiency(0, -1):.4f})"
    )
    return -eff, -grad


def main() -> None:
    """Optimise the TE line-source antenna, or the TM in-plane one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tm-inplane", action="store_true", help="pol = 1, spin-locked dipole"
    )
    args = parser.parse_args()
    pol = 1 if args.tm_inplane else 2
    moment = None
    if args.tm_inplane:
        m0 = SOI.guided_modes(LAM, 1, N_CLAD)[0]
        b = (1.0, 1j * m0.beta / m0.gamma)  # b = (p_z, −p_x), one-way into +x
        moment = (-b[1], b[0])
    base = Geometry.gielis(rad=RAD, n_pts=N_PTS, m=M)  # uniform θ: the map is p-free
    steps = np.array([1e-3, 1e-3, 1e-5, 1e-5])  # nm for lengths, unit for b, n1
    print(f"pol = {pol}, {'spin-locked in-plane dipole' if moment else 'line source'}")
    result = minimize(
        objective,
        P0,
        args=(pol, moment, base, steps),
        jac=True,
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={"maxiter": 25},
    )
    print(
        f"optimum η₊ = {-result.fun:.4f} at p = {np.round(result.x, 3)}",
        f"({result.message})",
    )


if __name__ == "__main__":
    main()
