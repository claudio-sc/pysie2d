"""S5 — proof of concept: interpolate M(λ), then integrate the interpolant.

The node count of a contour is set by how close the poles of M⁻¹ come to it.
M itself is smooth across the box. So: assemble M at a few nodes on the
ellipse, fit a polynomial in λ through those matrices, and run a *fine* contour
on the polynomial, which costs a factorisation per node and no assembly.

  direct        the ellipse contour on M itself, n assemblies (S2's column)
  interpolated  n assemblies on the same ellipse, then a 512-node contour on
                the polynomial through them

The polynomial is expanded in Chebyshev polynomials of ``(λ − centre)/f`` with
``±f`` the foci of the ellipse, which is the basis that stays well conditioned
on an elongated region; for a circle (f = 0) it is the monomial basis.

Columns: assemblies; error of the interpolant at points on the ellipse halfway
between the nodes, relative, Frobenius; then for each route the rank kept,
worst and median eigenvalue error, -missing +spurious.
"""

import time

import numpy as np
from common import in_ellipse, problems, score

from pysie2d import Geometry, Material, QNMSolver
from pysie2d.beyn import beyn_poles, contour_moments, ellipse_contour_quad, probe_matrix

ASSEMBLIES = (6, 8, 12, 16, 24, 32, 48, 64)
N_FINE = 512


def basis(
    lam: complex | np.ndarray, z_lo: complex, z_hi: complex, count: int
) -> np.ndarray:
    """Chebyshev (or, for a circle, monomial) polynomials 0 … count−1 at ``lam``."""
    centre = 0.5 * (z_lo + z_hi)
    half = 0.5 * (z_hi - z_lo)
    focus = np.sqrt(complex(half.real**2 - half.imag**2))
    lam = np.atleast_1d(np.asarray(lam, dtype=complex))
    out = np.empty((count, lam.size), dtype=complex)
    out[0] = 1.0
    if abs(focus) < 1.0e-12 * abs(half):
        w = (lam - centre) / half.real
        for k in range(1, count):
            out[k] = out[k - 1] * w
        return out
    w = (lam - centre) / focus
    if count > 1:
        out[1] = w
    for k in range(2, count):
        out[k] = 2.0 * w * out[k - 1] - out[k - 2]
    return out


def interpolant(prob, n_nodes: int):
    """Polynomial through M at the ``n_nodes`` trapezoid nodes of the ellipse.

    Returns the builder ``λ → P(λ)`` and the condition number of the
    interpolation matrix.
    """
    pts, _ = ellipse_contour_quad(prob.z_lo, prob.z_hi, n_nodes)
    samples = np.array([prob.m_builder(z) for z in pts])
    vander = basis(pts, prob.z_lo, prob.z_hi, n_nodes).T
    coeffs = np.linalg.solve(vander, samples.reshape(n_nodes, -1)).reshape(
        samples.shape
    )

    def p_builder(lam: complex) -> np.ndarray:
        t = basis(lam, prob.z_lo, prob.z_hi, n_nodes)[:, 0]
        return np.tensordot(t, coeffs, axes=1)

    return p_builder, float(np.linalg.cond(vander))


def extract(prob, m_builder, n_nodes: int, v_probe) -> str:
    """Ellipse contour of ``n_nodes`` on ``m_builder``, scored on the ellipse."""
    pts, wts = ellipse_contour_quad(prob.z_lo, prob.z_hi, n_nodes)
    a0, a1, canc = contour_moments(m_builder, pts, wts, v_probe)
    try:
        res = beyn_poles(a0, a1, canc, prob.z_lo, prob.z_hi, contour="ellipse")
    except ValueError:
        return f"{'no rank':^29}"
    if res.rank == 0:
        return f"{'no rank':^29}"
    return f"{res.rank:2d} {score(res.eigenvalues, prob, in_ellipse)}"


def accuracy() -> None:
    """One table per problem."""
    for prob in problems():
        v_probe = probe_matrix(prob.n_dim, prob.n_probe, 0)
        n_in = int(in_ellipse(prob.poles, prob.z_lo, prob.z_hi).sum())
        print(f"\n## {prob.name}  ({n_in} inside the ellipse, {prob.n_probe} probes)")
        print(
            f"{'asm':>4} {'fit error':>9} {'cond':>8} | {'direct':^29} | "
            f"{f'interpolated, {N_FINE} nodes':^29}"
        )
        for n in ASSEMBLIES:
            p_builder, cond = interpolant(prob, n)
            # Halfway between the nodes is where an interpolant is worst.
            mid = ellipse_contour_quad(prob.z_lo, prob.z_hi, 2 * n)[0][1::2][
                :: max(n // 8, 1)
            ]
            fit = max(
                np.linalg.norm(p_builder(z) - prob.m_builder(z))
                / np.linalg.norm(prob.m_builder(z))
                for z in mid
            )
            direct = extract(prob, prob.m_builder, n, v_probe)
            fine = extract(prob, p_builder, N_FINE, v_probe)
            print(f"{n:4d} {fit:9.1e} {cond:8.1e} | {direct} | {fine}")


def cost() -> None:
    """Where the time goes at a production boundary size, one thread."""
    print("\n## Cost per contour node, circle TE at n_pts = 200 (N = 400), one thread")
    geom = Geometry.gielis(rad=200.0, n_pts=200, m=0)
    bie = QNMSolver(geom, Material(n_core=3.0, n_clad=1.0, pol=2))._bie
    z_lo, z_hi = 500 + 1j, 1100 + 50j
    pts, _ = ellipse_contour_quad(z_lo, z_hi, 16)
    v_probe = probe_matrix(400, 20, 0)

    t0 = time.perf_counter()
    samples = np.array([bie.assemble(z) for z in pts])
    t_asm = (time.perf_counter() - t0) / pts.size

    t0 = time.perf_counter()
    vander = basis(pts, z_lo, z_hi, 16).T
    coeffs = np.linalg.solve(vander, samples.reshape(16, -1)).reshape(samples.shape)
    t_fit = time.perf_counter() - t0

    lam = 0.5 * (pts[0] + pts[1])
    t0 = time.perf_counter()
    for _ in range(20):
        mat = np.tensordot(basis(lam, z_lo, z_hi, 16)[:, 0], coeffs, axes=1)
    t_eval = (time.perf_counter() - t0) / 20

    t0 = time.perf_counter()
    for _ in range(20):
        np.linalg.solve(mat, v_probe)
    t_solve = (time.perf_counter() - t0) / 20

    print(f"assemble M           {1e3 * t_asm:8.2f} ms")
    print(f"evaluate polynomial  {1e3 * t_eval:8.2f} ms   (16 terms)")
    print(f"factorise and solve  {1e3 * t_solve:8.2f} ms   (20 probes)")
    print(f"fit, once            {1e3 * t_fit:8.2f} ms")
    print(f"stored coefficients  {coeffs.nbytes / 1e6:8.1f} MB")
    # The pairing is by accuracy on the wide box above: 32 assemblies with the
    # interpolant reach 1e-10 on the worst mode; the direct ellipse needs 256
    # nodes to pass it (S2: 4.5e-10 at 192, 2.5e-11 at 256).
    n_asm, n_direct = 32, 256
    scale = n_asm / 16
    direct = n_direct * (t_asm + t_solve)
    interp = n_asm * t_asm + t_fit * scale + N_FINE * (t_eval * scale + t_solve)
    print(
        f"{n_direct} direct nodes {direct:.2f} s; {n_asm} assemblies + {N_FINE} "
        f"interpolated nodes {interp:.2f} s ({direct / interp:.1f}x)"
    )


if __name__ == "__main__":
    accuracy()
    cost()
