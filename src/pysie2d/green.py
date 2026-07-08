"""Self-Green function and local density of states (LDOS / Purcell effect).

A line-dipole source at ``r_s`` radiates the free-space field
``ψ_inc(r) = (i/4)·H₀^{(1)}(k·|r − r_s|)``. Solving the BIE with this incident
field and evaluating the *scattered* field back at the source gives the
self-Green function ``S(r_s, r_s, ω)`` — how strongly the environment scatters
the emitter's own field onto itself. ``Im S`` sets the environment-modified
decay rate (Purcell effect); ``Re S`` the frequency shift.

The scattered field is smooth at ``r_s`` (the singular part of the total field
lives entirely in ``ψ_inc``), so evaluating it at the source point is
numerically safe — provided ``r_s`` keeps the usual distance from the boundary
(see :func:`pysie2d.fields.eval_field`).
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .solver import BIESolver, ScatterResult
from .sources import line_dipole_rhs


def self_green(
    solver: BIESolver,
    wavelength: float,
    x_s: float,
    z_s: float,
) -> complex:
    """Scattered field back at the source: S(r_s, r_s, ω).

    Solves the BIE with a line-dipole source at ``(x_s, z_s)`` (unit-source
    normalisation, i.e. the ``(i/4)·H₀^{(1)}`` incident field) and returns the
    scattered field evaluated at that same point.

    Args:
        solver: Configured BIE solver (geometry + material).
        wavelength: Free-space wavelength (nm).
        x_s: Source x-coordinate (nm).
        z_s: Source z-coordinate (nm).

    Returns:
        The complex self-Green function S(r_s, r_s, ω).
    """
    result = solver.scatter_dipole(wavelength, x_s, z_s)
    field = result.eval_field(np.array([x_s]), np.array([z_s]))
    return complex(field[0])


def relative_ldos(
    solver: BIESolver,
    wavelength: float,
    x_s: float,
    z_s: float,
) -> float:
    """Local density of states relative to free space.

    Derivation of the normalisation. As ``z → 0``,
    ``H₀^{(1)}(z) → 1 + (2i/π)·ln(z)``: the log divergence sits in the
    imaginary part, while the real part tends to ``J₀(0) = 1``. With the
    ``i/4`` prefactor of the 2-D Green function, ``Im[g₀(r→r)] = 1/4``. The
    LDOS relative to free space is therefore

        relative_ldos = 1 + Im(S) / Im(g₀_self) = 1 + 4·Im(S).

    Args:
        solver: Configured BIE solver (geometry + material).
        wavelength: Free-space wavelength (nm).
        x_s: Source x-coordinate (nm).
        z_s: Source z-coordinate (nm).

    Returns:
        The relative LDOS (1.0 in free space; > 1 for enhancement).
    """
    return 1.0 + 4.0 * self_green(solver, wavelength, x_s, z_s).imag


def relative_ldos_map(
    solver: BIESolver,
    wavelength: float,
    x_pts: np.ndarray,
    z_pts: np.ndarray,
) -> np.ndarray:
    """Relative LDOS at many source positions, reusing one LU factorisation.

    The system matrix ``M`` depends only on the wavelength, not on the source
    position, so it is assembled and LU-factorised once and reused across every
    source in the grid (``scipy.linalg.lu_factor`` / ``lu_solve``). This turns
    a per-point re-solve into a single factorisation plus cheap back-substitutions
    — an hour-long sweep becomes seconds.

    Source positions that are inside the particle or within five boundary
    spacings of it (where :func:`pysie2d.sources.line_dipole_rhs` raises) are
    returned as ``NaN`` so callers can mask them.

    Args:
        solver: Configured BIE solver (geometry + material).
        wavelength: Free-space wavelength (nm).
        x_pts: Source x-coordinates (nm), any shape.
        z_pts: Source z-coordinates (nm), same shape as x_pts.

    Returns:
        Relative LDOS at every source point, same shape as x_pts, with NaN at
        invalid (interior / too-close) positions.
    """
    g = solver.geometry
    mat = solver.material
    x_pts = np.asarray(x_pts, dtype=float)
    z_pts = np.asarray(z_pts, dtype=float)

    lu = lu_factor(solver.assemble(wavelength))

    out = np.full(x_pts.shape, np.nan)
    for idx, (x_s, z_s) in enumerate(zip(x_pts.flat, z_pts.flat, strict=True)):
        try:
            rhs = line_dipole_rhs(g.n_pts, wavelength, g.f, g.g, x_s, z_s)
        except ValueError:
            continue
        ei = lu_solve(lu, rhs)
        result = ScatterResult(ei, g, mat, wavelength)
        field = result.eval_field(np.array([x_s]), np.array([z_s]))
        out.flat[idx] = 1.0 + 4.0 * complex(field[0]).imag
    return out
