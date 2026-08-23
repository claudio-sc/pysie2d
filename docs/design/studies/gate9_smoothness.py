"""Gate 9: is M(λ) smooth enough in a shape parameter to differentiate it?

Every ``∂M/∂p`` in the sensitivity API is a central difference, so the whole
adjoint rests on one empirical question: over which step sizes ``h`` does

    D(h) = [M(b₀+h) − M(b₀−h)] / (2h)

sit on a plateau? Too large and truncation dominates; too small and subtractive
cancellation does — and between them there must be a window wide enough to
place a step in without tuning.

**The metric matters.** Successive-``h`` change, ‖D(h) − D(h/10)‖, is the
natural-looking diagnostic and it is the misleading one here: it conflates the
truncation term with a second, *discrete* error source and reads as a
respectable ladder while hiding a stall. This script measures instead the
deviation from the most accurate available ``D``, together with the **fraction
of matrix entries** carrying it — because the distinguishing signature of the
second source is that it lives on a shrinking minority of entries rather than
being spread over all of them.

That second source is the arc-length node placement. Inverting arc length to
θ goes through ``np.interp``, which is continuous but only **piecewise** linear
in the shape parameter, so a node whose bracketing cell differs between
``b−h`` and ``b+h`` contributes an O(1) error to the quotient. The count of such
nodes is proportional to ``h``, which makes the resulting error term O(h) — it
does *not* fall at the O(h²) rate the truncation term does, and it is not
monotone. ``node_placement_ladder`` isolates it in the geometry alone, with no
electromagnetics in the way.

Run: ``uv run python docs/design/studies/gate9_smoothness.py``
"""

import numpy as np

from pysie2d import Geometry, Material
from pysie2d.solver import BIESolver

RAD = 200.0
B0 = 1.20  # the reference design point: an m=4 shape well off the circle
LAM = 700.0 + 8.0j  # complex λ, i.e. the QNM path and not a real fast path
STEPS = [10.0**-e for e in range(2, 8)]
H_REF = 1.0e-6  # the most accurate D measured below; see the floor discussion

# An entry is "carrying the deviation" above this relative size. Chosen a
# decade above the ~1e-8 cancellation floor so the fraction counts genuine
# error rather than round-off, and far below the 1e-5…1e-3 deviations being
# resolved, so the count is not sensitive to where exactly it sits.
CARRY_FLOOR = 1.0e-6


# The design point every ladder is taken about. Geometric and material
# parameters are listed together on purpose: D12 sets one step size for all of
# them, and whether that is legitimate is exactly what parameter_sweep asks.
DESIGN_POINT = {
    "m": 4,
    "a": 1.0,
    "b": B0,
    "n1": 2.0,
    "n2": 2.0,
    "n3": 2.0,
    "n_core": 3.0,
    "n_clad": 1.0,
}
MATERIAL_KEYS = ("n_core", "n_clad")


def _matrix_at(point: dict, n_pts: int) -> np.ndarray:
    """M(λ) at a full design point, geometry and material together."""
    shape = {k: v for k, v in point.items() if k not in MATERIAL_KEYS}
    geo = Geometry.gielis(RAD, n_pts, arc_length=True, **shape)
    mat = Material(n_core=point["n_core"], n_clad=point["n_clad"], pol=2)
    return BIESolver(geo, mat).assemble(LAM)


def _matrix(b: float, n_pts: int) -> np.ndarray:
    return _matrix_at({**DESIGN_POINT, "b": b}, n_pts)


def _node_angles(b: float, n_pts: int) -> np.ndarray:
    geo = Geometry.gielis(RAD, n_pts, m=4, b=b, arc_length=True)
    return np.unwrap(np.arctan2(geo.g - geo.z0, geo.f - geo.x0))


def _central(fn, b: float, h: float, n_pts: int) -> np.ndarray:
    return (fn(b + h, n_pts) - fn(b - h, n_pts)) / (2.0 * h)


def _ladder(fn, n_pts: int, label: str, p_0: float = B0) -> None:
    """Deviation of D(h) from D(H_REF), and how many entries carry it."""
    print(f"\n--- {label}  [n_pts = {n_pts}, expansion point {p_0}]")
    ref = _central(fn, p_0, H_REF, n_pts)
    scale = float(np.abs(ref).max())
    print(f"{'h':>10} {'rel. L2 dev':>13} {'max entry dev':>15} {'frac carrying':>15}")
    for h in STEPS:
        dev = _central(fn, p_0, h, n_pts) - ref
        frac = float((np.abs(dev) / scale > CARRY_FLOOR).mean())
        rel = float(np.linalg.norm(dev) / np.linalg.norm(ref))
        print(f"{h:10.0e} {rel:13.2e} {np.abs(dev).max() / scale:15.2e} {frac:15.3f}")


def derivative_ladder(n_pts: int) -> None:
    """The gate proper: dM/db at complex λ."""
    _ladder(_matrix, n_pts, f"dM/db, lambda = {LAM}")


def node_placement_ladder(n_pts: int) -> None:
    """The same ladder on the node angles alone — geometry, no physics.

    If this shows the stall, the stall is ``np.interp`` and not the assembly.
    """
    _ladder(_node_angles, n_pts, "dtheta/db (arc-length node placement)")


def parameter_sweep(n_pts: int) -> None:
    """The same ladder in every parameter, which is what D12 quantifies over.

    A step size is only defensible per *class* of parameter if the classes
    behave alike. They do not: only the parameters that reach the arc-length
    inversion pick up the `np.interp` term, so the material parameters make the
    control — if they showed the same stall, the diagnosis above would be wrong.
    """
    for name, p_0 in DESIGN_POINT.items():
        if name == "m":  # categorical (D5), never differentiated
            continue
        _ladder(
            lambda p, n, _k=name: _matrix_at({**DESIGN_POINT, _k: p}, n),
            n_pts,
            f"dM/d{name}",
            p_0=float(p_0),
        )


if __name__ == "__main__":
    for n in (120, 200, 400):
        derivative_ladder(n)
    node_placement_ladder(200)
    parameter_sweep(200)
