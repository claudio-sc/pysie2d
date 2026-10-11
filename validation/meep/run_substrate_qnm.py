r"""MEEP driver: QNMs of a circle above a dielectric half-space, by harminv.

The independent anchor for v0.9's substrate QNMs (layered-spec G9, owner
decision 1): a time-domain ringdown on a Yee grid shares no equation,
discretisation or contour with pysie2d's Beyn search over the Sommerfeld
operator. Run it under the ``pysie2d-meep`` conda environment::

    conda run -n pysie2d-meep python validation/meep/run_substrate_qnm.py \\
        --case te --eps-sub 2.25 --resolution 25 --out result.npz

Geometry. MEEP's invariant axis is z and pysie2d's is y, so this driver's
``(x, y)`` is pysie2d's ``(x, z)``: the substrate fills ``y < 0`` and runs on
through the PML, which is how MEEP terminates an infinite half-space. The
circle sits at ``y = RAD + GAP``.

Pole convention. Harminv returns a real frequency and a ``Q``; under
``exp(−iωt)`` the decaying pole is ``f = f_r·(1 − i/(2Q))``, and with ``a``
the length unit ``λ = a/f``. Then ``Re λ / (2 Im λ) = Q`` exactly, the same
``Q`` as conventions §8, and ``Im λ > 0`` as a decaying mode requires there.
"""

from __future__ import annotations

import argparse
import json

import meep as mp
import numpy as np

NM_PER_A = 100.0
"""MEEP length unit, in nm. Every length below is in units of ``a``."""

RAD = 1.0  # 100 nm: the circle of tests/test_halfspace_qnm.py
GAP = 0.5  # 50 nm between the circle and the interface
N_CORE = 3.0
N_CLAD = 1.0
PAD = 3.0  # cladding/substrate clear of the particle before the PML, in a
DPML = 4.0  # ≥ λ_max/2 in the substrate's longest wavelength studied

CASES = {"te": (2, mp.Ez), "tm": (1, mp.Hz)}

TARGET_NM = {
    # pysie2d's n = 0 poles at GAP = 50 nm (validation/_study/substrate-qnm-
    # 2026-10-11/homotopy.log). Used only to centre the source band and pick
    # the harminv mode to report; the comparison is done elsewhere.
    ("te", 1.0): 473.0760 + 43.1763j,
    ("te", 1.5): 471.5294 + 41.7215j,
    ("te", 2.25): 470.1008 + 39.8721j,
    ("te", 3.0): 469.2741 + 38.3681j,
    ("te", 4.0): 468.6614 + 36.7895j,
    ("tm", 1.0): 818.8119 + 92.4285j,
    ("tm", 2.25): 804.4281 + 96.2868j,
}


def run(case: str, eps_sub: float, resolution: int, until: float) -> dict:
    """One ringdown. Returns every harminv mode, as vacuum wavelengths in nm."""
    _pol, component = CASES[case]
    target = TARGET_NM[(case, eps_sub)]
    f0 = NM_PER_A / target.real
    # A band ±40 % about the target: wide enough that a Q ≈ 4 line (relative
    # width 1/Q ≈ 25 %) sits well inside it, narrow enough to keep the n = 2
    # neighbours out of the fit.
    df = 0.8 * f0

    yc = RAD + GAP
    half_x = RAD + PAD + DPML
    y_top = yc + RAD + PAD + DPML
    y_bot = -(PAD + DPML)
    size = mp.Vector3(2 * half_x, y_top - y_bot)
    centre = mp.Vector3(0, 0.5 * (y_top + y_bot))  # cell centre, in world coords

    def at(x: float, y: float) -> mp.Vector3:
        return mp.Vector3(x, y) - centre

    geometry = []
    if eps_sub != N_CLAD**2:
        geometry.append(
            mp.Block(
                size=mp.Vector3(mp.inf, -y_bot + 1.0, mp.inf),
                center=at(0, 0.5 * (y_bot - 1.0)),
                material=mp.Medium(epsilon=eps_sub),
            )
        )
    geometry.append(
        mp.Cylinder(
            radius=RAD,
            height=mp.inf,
            center=at(0, yc),
            material=mp.Medium(index=N_CORE),
        )
    )

    # Source at the centre excites the n = 0 family (and even n); the probe is
    # off-centre so it also sees the odd modes harminv must then separate.
    sim = mp.Simulation(
        cell_size=size,
        boundary_layers=[mp.PML(DPML)],
        geometry=geometry,
        sources=[mp.Source(mp.GaussianSource(f0, fwidth=df), component, at(0, yc))],
        resolution=resolution,
        default_material=mp.Medium(index=N_CLAD),
    )
    # Q_thresh: MEEP's default of 50 silently discards every mode here (Q ≈ 4–6)
    # and harminv then reports nothing at all. Q < 0.5 is outside the contour's
    # domain anyway (holomorphy §8), so that is the floor. It is an attribute,
    # not a constructor argument, in MEEP 1.34.
    h = mp.Harminv(component, at(0.3 * RAD, yc + 0.2 * RAD), f0, df)
    h.Q_thresh = 0.5
    # Harminv from the pulse peak, not from source cut-off: at Q ≈ 4–6 the line
    # e-folds in ~2 periods, and the Gaussian tail (5 widths) outlasts it.
    t_peak = 5.0 / df
    sim.run(mp.after_time(t_peak, h), until=t_peak + until)

    modes = []
    for m in h.modes:
        if m.Q <= 0:
            continue  # growing: not a passive pole, harminv noise
        f = m.freq * (1 - 1j / (2 * m.Q))
        lam = NM_PER_A / f
        modes.append((lam, m.Q, abs(m.amp), m.err))
    if not modes:
        raise SystemExit("harminv found no decaying mode")
    best = min(modes, key=lambda t: abs(t[0] - target))
    return {
        "lam_nm": best[0],
        "q": best[1],
        "harminv_err": best[3],
        "all_lam_nm": np.array([t[0] for t in modes]),
        "all_amp": np.array([t[2] for t in modes]),
        "all_err": np.array([t[3] for t in modes]),
        "target_nm": target,
    }


def main() -> None:
    """Parse the command line, run one ringdown, print and save the pole."""
    global DPML, PAD
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=sorted(CASES), default="te")
    p.add_argument("--eps-sub", type=float, default=2.25)
    p.add_argument("--resolution", type=int, default=25, help="pixels per 100 nm")
    p.add_argument("--until", type=float, default=150.0, help="ringdown time, in a/c")
    p.add_argument("--dpml", type=float, default=DPML, help="PML thickness, in a")
    p.add_argument(
        "--pad", type=float, default=PAD, help="clearance before the PML, in a"
    )
    p.add_argument("--out", default=None)
    a = p.parse_args()
    DPML, PAD = a.dpml, a.pad

    res = run(a.case, a.eps_sub, a.resolution, a.until)
    lam = res["lam_nm"]
    tgt = res["target_nm"]
    print(
        f"pole λ = {lam.real:.4f}{lam.imag:+.4f}i nm  "
        f"Q = {res['q']:.3f}  err = {res['harminv_err']:.1e}"
    )
    print(f"pysie2d  {tgt.real:.4f}{tgt.imag:+.4f}i nm  |Δλ| = {abs(lam - tgt):.3e} nm")
    knobs = {
        "case": a.case,
        "eps_sub": a.eps_sub,
        "resolution": a.resolution,
        "until": a.until,
        "rad": RAD,
        "gap": GAP,
        "n_core": N_CORE,
        "pad": PAD,
        "dpml": DPML,
    }
    print(
        "JSONLINE " + json.dumps({**knobs, "lam": [lam.real, lam.imag], "q": res["q"]})
    )
    if a.out:
        np.savez(a.out, knobs=json.dumps(knobs), **res)


if __name__ == "__main__":
    main()
