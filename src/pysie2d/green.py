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

from .kernels import _real_if_real, hank0, hank1
from .solver import BIESolver


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


_LDOS_CHUNK_ELEMS = 400_000
"""Target element count for the (chunk, nn) working arrays in the LDOS map.

The batched path holds roughly ten (chunk, nn) intermediates at once, so this
bounds the peak working set at about 64 MB of complex128 regardless of grid
size or boundary resolution.
"""


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
    — an hour-long sweep becomes seconds. The grid is additionally processed in
    chunks, with every source in a chunk back-substituted as one BLAS-3
    multi-RHS ``lu_solve`` and the representation formula evaluated as one
    ``(chunk, nn)`` NumPy expression, which is another ~4× over the per-point
    loop.

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
    geom = solver.geometry
    nn = geom.n_pts
    f, g, df, dg = geom.f, geom.g, geom.df, geom.dg
    delt = geom.delt
    dl = np.full(nn, delt) if np.isscalar(delt) else np.asarray(delt)

    x_arr = np.asarray(x_pts, dtype=float)
    z_arr = np.asarray(z_pts, dtype=float)
    if x_arr.shape != z_arr.shape:
        raise ValueError(
            f"x_pts and z_pts must have the same shape, got "
            f"{x_arr.shape} and {z_arr.shape}"
        )
    xf = x_arr.ravel()
    zf = z_arr.ravel()
    out = np.full(xf.shape, np.nan)

    wnum = _real_if_real(2.0 * np.pi / wavelength)
    lu = lu_factor(solver.assemble(wavelength))

    # Geometry-only quantities, hoisted out of the per-point work.
    seg = np.sqrt(np.diff(f, append=f[0]) ** 2 + np.diff(g, append=g[0]) ** 2)
    exclusion = 5.0 * float(np.mean(seg))
    f_prev = np.roll(f, 1)
    g_prev = np.roll(g, 1)

    chunk = max(1, _LDOS_CHUNK_ELEMS // nn)
    for lo in range(0, xf.size, chunk):
        hi = min(lo + chunk, xf.size)
        xs = xf[lo:hi]
        zs = zf[lo:hi]

        xmf = xs[:, None] - f[None, :]  # (m, nn)
        zmg = zs[:, None] - g[None, :]
        dist = np.sqrt(xmf**2 + zmg**2)

        # Batched even-odd ray cast: identical rule to sources._point_inside.
        straddles = (g[None, :] > zs[:, None]) != (g_prev[None, :] > zs[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (f_prev - f)[None, :] * (zs[:, None] - g[None, :]) / (g_prev - g)[
                None, :
            ] + f[None, :]
        crossings = np.count_nonzero(straddles & (xs[:, None] < x_cross), axis=1)
        valid = (crossings % 2 == 0) & (dist.min(axis=1) >= exclusion)
        if not valid.any():
            continue

        # One BLAS-3 back-substitution for every valid source in the chunk.
        arg = wnum * dist[valid]  # (m_valid, nn), real -> Cephes path
        h0 = hank0(arg)
        h1 = hank1(arg)
        rhs = np.zeros((2 * nn, arg.shape[0]), dtype=complex)
        rhs[:nn, :] = (0.25j * h0).T
        eis = lu_solve(lu, rhs)  # (2nn, m_valid)

        # Representation formula, each source evaluated at its own position.
        arg2 = (-dg[None, :] * xmf + df[None, :] * zmg)[valid]
        integrand = (
            wnum**2 * arg2 * (h1 / arg) * eis[:nn, :].T - h0 * eis[nn:, :].T
        ) * dl[None, :]
        field = (1j / 4.0) * integrand.sum(axis=1)
        out[np.flatnonzero(valid) + lo] = 1.0 + 4.0 * field.imag

    return out.reshape(x_arr.shape)
