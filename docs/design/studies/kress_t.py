"""Kress–Martensen assembly in the quadrature parameter `t`.

Lifted from the inherited `exp_kress_full.py` (handoff §5.3, §10) with one
change: the quadrature parameter is **`t`, not θ**.

Why that change is the whole point. `exp_kress_full.py` uses `geom.theta` both
in the singular factor `4 sin²((θ−θ')/2)` and as the uniform step `2π/N`, which
is correct only when the nodes are equispaced in θ. Graded nodes are equispaced
in `t` with `θ = w(t)` (conventions §13.1), so under grading that script is
silently wrong.

Working in `t` throughout removes the Jacobian bookkeeping instead of managing
it: sample the boundary at `θ_j = w(t_j)` with `t` equispaced, take **every**
derivative with respect to `t` spectrally by FFT, and the assembly is then the
uniform-parameter one with no `w'` anywhere. It also supplies analytic-quality
second derivatives for free, which is what the handoff's machine-precision
column needed (`ddf`/`ddg` from `_der_real_3` caps the rate at 3, §6.4).

Complex `k` throughout (non-negotiable 1): `demote` keeps the real fast path
when the wavenumber is real and falls back to complex otherwise, which is what
makes this usable inside a Beyn contour.
"""

import numpy as np
from scipy.special import jv

from pysie2d.kernels import hank0, hank1

TWOPI = 2.0 * np.pi
GE = np.euler_gamma


def kress_weights(n_pts: int) -> np.ndarray:
    """Kress's circulant log weights `R_j`. Depend on `n_pts` alone."""
    n = n_pts // 2
    d = TWOPI * np.arange(n_pts) / n_pts
    p = np.arange(1, n)
    return -(TWOPI / n) * (
        (np.cos(np.outer(d, p)) / p).sum(1) + np.cos(n * d) / (2 * n)
    )


def _demote(k):
    """Real wavenumbers keep the real fast path; complex ones fall through."""
    kc = complex(k)
    return kc.real if kc.imag == 0.0 else kc


def _dt(vals: np.ndarray, order: int = 1) -> np.ndarray:
    """Spectral derivative on an equispaced periodic `t` grid."""
    n = vals.size
    j = np.fft.fftfreq(n, d=1.0 / n)
    return np.real(np.fft.ifft(np.fft.fft(vals) * (1j * j) ** order))


def kress_matrix(f: np.ndarray, g: np.ndarray, k_bg, k_co) -> np.ndarray:
    """Assemble `M` on boundary samples taken at **equispaced `t`**.

    Args:
        f: (N,) boundary x-coordinates at `θ_j = w(t_j)`, `t` equispaced.
        g: (N,) boundary z-coordinates at the same nodes.
        k_bg: Background wavenumber, real or complex.
        k_co: Core wavenumber, real or complex.

    Returns:
        The (2N, 2N) system matrix, in the `φ`/`χ` layout of conventions §4.
    """
    n_pts = f.size
    df, dg = _dt(f), _dt(g)
    ddf, ddg = _dt(f, 2), _dt(g, 2)
    speed = np.hypot(df, dg)  # |dx/dt|
    deriv = df * ddg - ddf * dg

    dx, dz = f[:, None] - f[None, :], g[:, None] - g[None, :]
    r = np.sqrt(dx**2 + dz**2)
    np.fill_diagonal(r, 1.0)
    arg2 = dx * dg[None, :] - dz * df[None, :]

    t = TWOPI * np.arange(n_pts) / n_pts
    s2 = 4.0 * np.sin((t[:, None] - t[None, :]) / 2.0) ** 2
    np.fill_diagonal(s2, 1.0)
    log_s2 = np.log(s2)
    idx = (np.arange(n_pts)[:, None] - np.arange(n_pts)[None, :]) % n_pts
    weights = kress_weights(n_pts)[idx]
    h = TWOPI / n_pts

    def single(k):
        k = _demote(k)
        a1 = -jv(0, k * r) / (4.0 * np.pi)
        a2 = 0.25j * hank0(k * r) - a1 * log_s2
        np.fill_diagonal(a1, -1.0 / (4.0 * np.pi))
        np.fill_diagonal(a2, 0.25j - (GE + np.log(k * speed / 2.0)) / (2.0 * np.pi))
        return (a1 * weights + a2 * h).astype(complex)

    def double(k):
        k = _demote(k)
        b1 = -(k / (4.0 * np.pi)) * arg2 * jv(1, k * r) / r
        b2 = 0.25j * k**2 * arg2 * hank1(k * r) / (k * r) - b1 * log_s2
        np.fill_diagonal(b1, 0.0)
        np.fill_diagonal(b2, -deriv / (4.0 * np.pi * speed**2))
        return (b1 * weights + b2 * h).astype(complex)

    m = np.zeros((2 * n_pts, 2 * n_pts), dtype=complex)
    m[:n_pts, :n_pts] = double(k_bg) + 0.5 * np.eye(n_pts)
    m[:n_pts, n_pts:] = single(k_bg)
    m[n_pts:, :n_pts] = double(k_co) - 0.5 * np.eye(n_pts)
    m[n_pts:, n_pts:] = single(k_co)
    return m
