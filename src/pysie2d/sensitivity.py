"""Finite-difference sensitivity of guided outputs to one parameter (slab-spec R16′).

A central difference of the objective, not an adjoint: an adjoint whose ∂M/∂p
is itself a central difference costs the same two solves per parameter, and a
real saving needs analytic kernel shape derivatives (the Fourier-shape
milestone).
"""

from collections.abc import Callable

import numpy as np

from .geometry import Geometry, _check_frozen_nodes
from .material import Material
from .qnm import SHAPE_STEP
from .waveguide import ModeAmplitudes


def mode_sensitivity(
    at: Callable[[float], tuple[Geometry, Material]],
    run: Callable[[Geometry, Material], ModeAmplitudes],
    *,
    step: float = SHAPE_STEP,
) -> dict[str, np.ndarray | float]:
    """Central differences of every :class:`ModeAmplitudes` output in one parameter.

    ``(run(at(+step)) − run(at(−step)))/(2·step)`` of ``s_plus``, ``s_minus``
    (complex arrays), ``p_plus``, ``p_minus`` (arrays), ``p_up``, ``p_down``,
    ``p_abs`` and ``p_source`` (floats; ``p_source`` only when it is not
    ``None``), under those keys. Two solves per parameter; the caller loops
    over parameters.

    **The frozen map is enforced, as in** ``QNMResult.sensitivity``: both
    displaced geometries must carry ``at(0)``'s node set — θ, w′, w″ — exactly
    (``docs/conventions.md`` §10), so the difference is of the shape and not of
    the node gauge.

    Args:
        at: ``δ → (geometry, material)`` at offset ``δ`` from the base point, in
            the parameter's own units; ``δ = 0`` is the base point.
        run: ``(geometry, material) → ModeAmplitudes``, the solve and the
            ``x_ref`` the objective is referred to.
        step: Central-difference step, in the parameter's units.

    Returns:
        The derivatives by key.

    Raises:
        ValueError: A displaced geometry that does not carry the base node
            set, or one with no node set at all.
    """
    base = at(0.0)[0]
    plus, minus = at(+step), at(-step)
    for geom, _ in (plus, minus):
        _check_frozen_nodes(base, geom)
    a, b = run(*plus), run(*minus)
    out: dict[str, np.ndarray | float] = {}
    for key in ("s_plus", "s_minus", "p_plus", "p_minus"):
        out[key] = (getattr(a, key) - getattr(b, key)) / (2.0 * step)
    for key in ("p_up", "p_down", "p_abs"):
        out[key] = (getattr(a, key) - getattr(b, key)) / (2.0 * step)
    if a.p_source is not None and b.p_source is not None:
        out["p_source"] = (a.p_source - b.p_source) / (2.0 * step)
    return out
