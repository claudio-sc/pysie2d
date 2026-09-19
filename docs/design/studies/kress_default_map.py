"""Which node map should be the v0.6 default under Kress–Martensen quadrature?

Question (13 Sep 2026): uniform θ undersamples the arms of sharp stars, which is
why v0.5 preferred uniform arc length. Does that survive spectral quadrature,
and is there a regime where arc-length or curvature-adaptive grading pays?

Method. Three maps — uniform θ (identity), uniform arc length
(`Parametrisation.gielis`, band 20–20) and adaptive (band 20–100) — on two
ladders, `qext` (TE, n_core 1.5, λ 600 nm) against a uniform-θ reference at
high `nn`, cross-checked against the arc-length reference at the same `nn`:

- rounded squares, `m = 4`, n1/n2 = 2/4, 6/12, 12/24, 20/50, plus the `m = 6`
  6/12/12 star of handoff §9 (tip-to-valley radius ratio 1.4–2.3);
- spiky stars, `m = 6`, n2 = n3 = 8, n1 = 4, 2, 1, 0.5 (arm ratio
  2^(3/n1) = 1.7, 2.8, 8, 64), tip radius held at 400 nm.

Self-contained on the pre-migration tree: the assembly below is the production
form of `docs/design/kress-spec.md` §3 (matrix only), and the geometry arrays
use the landed closed-form `_rderiv`/`_rderiv2` composed with the map.

Usage: `uv run python docs/design/studies/kress_default_map.py SHAPE`, one shape
per process (four in parallel on this machine); SHAPE is a key of `SHAPES` or
`SPIKES`. Prints one JSON line. Results: kress-spec.md Appendix A.
"""

import json
import sys

import numpy as np
from scipy.special import hankel1, j0, j1, jv, y0, y1

from pysie2d import Material
from pysie2d.geometry import _rderiv, _rderiv2, gielis
from pysie2d.kernels import _real_if_real
from pysie2d.parametrisation import Parametrisation
from pysie2d.sources import plane_wave_rhs

TWOPI = 2.0 * np.pi
LAM, N_CORE, POL = 600.0, 1.5, 2

SHAPES = {
    "mild_2_4_4": ({"m": 4, "n1": 2.0, "n2": 4.0, "n3": 4.0}, 200.0),
    "base_6_12_12": ({"m": 4, "n1": 6.0, "n2": 12.0, "n3": 12.0}, 200.0),
    "sharp_12_24_24": ({"m": 4, "n1": 12.0, "n2": 24.0, "n3": 24.0}, 200.0),
    "corner_20_50_50": ({"m": 4, "n1": 20.0, "n2": 50.0, "n3": 50.0}, 200.0),
    "star6_6_12_12": ({"m": 6, "n1": 6.0, "n2": 12.0, "n3": 12.0}, 200.0),
}
SPIKES = {f"spike_n1_{n1:g}": n1 for n1 in (4.0, 2.0, 1.0, 0.5)}
NNS = [40, 60, 80, 120, 160, 240, 320, 480, 640, 960]
NN_REF = 1600


def identity_map():
    """`w(t) = t` built directly; `Parametrisation.uniform_theta()` post-migration."""
    theta_f = np.linspace(0.0, TWOPI, 1024, endpoint=False)
    return Parametrisation(
        a0=1.0,
        coef=np.zeros(0, dtype=complex),
        theta_fine=theta_f,
        t_fine=np.append(theta_f, TWOPI),
        nn_from_band=0,
        contrast_realised=1.0,
        alpha_eff=0.0,
        n_fine=1024,
        n_terms=0,
    )


def kress_weights(nn):
    """W_d = R_d − h·ln(4 sin²(πd/nn)), W_0 = R_0; both parities of nn."""
    d = TWOPI * np.arange(nn) / nn
    orders = np.arange(1, (nn - 1) // 2 + 1)
    r = -(4.0 * np.pi / nn) * (np.cos(np.outer(d, orders)) / orders).sum(axis=1)
    if nn % 2 == 0:
        r = r - (4.0 * np.pi / nn**2) * np.cos(0.5 * nn * d)
    w = r.copy()
    w[1:] -= (TWOPI / nn) * np.log(4.0 * np.sin(np.pi * np.arange(1, nn) / nn) ** 2)
    return w


def _jh(order, z):
    if np.iscomplexobj(z):
        return jv(order, z), hankel1(order, z)
    j = j0(z) if order == 0 else j1(z)
    return j, j + 1j * (y0(z) if order == 0 else y1(z))


def assemble(pol, f, g, df, dg, ddf, ddg, wnum_bg, ri, kd):
    """Kress matrix, kress-spec.md §3.3 (the patch's assemble_matrix, condensed)."""
    nn = f.size
    wnum_bg = _real_if_real(wnum_bg)
    wnum_core = _real_if_real(ri * wnum_bg)
    eta = kd if pol == 1 else 1.0
    h = TWOPI / nn
    w = kress_weights(nn)
    gamma = np.hypot(df, dg)
    deriv = df * ddg - ddf * dg
    ui, uj = np.triu_indices(nn, k=1)
    w_tri = w[uj - ui]
    dx, dz = f[ui] - f[uj], g[ui] - g[uj]
    cij, cji = dx * dg[uj] - dz * df[uj], -dx * dg[ui] + dz * df[ui]
    r = np.hypot(dx, dz)
    me = np.zeros((2 * nn, 2 * nn), dtype=complex)
    di = np.arange(nn)
    for k, row, scale in ((wnum_bg, 0, 1.0), (wnum_core, nn, eta)):
        z = k * r
        jz0, hz0 = _jh(0, z)
        jz1, hz1 = _jh(1, z)
        single = 0.25j * h * hz0 - w_tri * jz0 / (4.0 * np.pi)
        double = k**2 * (0.25j * h * hz1 / z - w_tri * jz1 / z / (4.0 * np.pi))
        me[ui + row, uj] = double * cij
        me[uj + row, ui] = double * cji
        me[ui + row, uj + nn] = scale * single
        me[uj + row, ui + nn] = scale * single
        me[di + row, di + nn] = scale * (
            -w[0] / (4.0 * np.pi)
            + h * (0.25j - (np.euler_gamma + np.log(k * gamma / 2.0)) / TWOPI)
        )
    curv = deriv * h / (4.0 * np.pi) / gamma**2
    me[di, di] = 0.5 - curv
    me[di + nn, di] = -(0.5 + curv)
    return me


def arrays(par, nn, rad, shape):
    """f, g and their t-derivatives on the map's nodes (kress-spec.md §3.6)."""
    nodes = par.nodes(nn)
    th = nodes.theta
    a = b = 1.0
    m, n1, n2, n3 = shape["m"], shape["n1"], shape["n2"], shape["n3"]
    f, g, r, co, se, arg = gielis(th, rad, a, b, m, n1, n2, n3, 0.0, 0.0)
    r1 = _rderiv(rad, n1, n2, n3, -n2 * m / 4.0, n3 * m / 4.0, co, se, arg)
    r2 = _rderiv2(rad, n1, n2, n3, -n2 * m / 4.0, n3 * m / 4.0, m, co, se, arg)
    s, c = np.sin(th), np.cos(th)
    fp, gp = r * c + s * r1, -r * s + c * r1
    fpp = r2 * s + 2.0 * r1 * c - r * s
    gpp = r2 * c - 2.0 * r1 * s - r * c
    w1, w2 = nodes.dw, nodes.ddw
    return f, g, fp * w1, gp * w1, fpp * w1**2 + fp * w2, gpp * w1**2 + gp * w2


def qext(par, nn, rad, shape):
    """Forward-direction extinction efficiency, exactly at the forward angle."""
    mat = Material(n_core=N_CORE, n_clad=1.0, pol=POL)
    k = mat.wnum_bg(LAM)
    f, g, df, dg, ddf, ddg = arrays(par, nn, rad, shape)
    ei = np.linalg.solve(
        assemble(POL, f, g, df, dg, ddf, ddg, k, mat.nc, mat.eps),
        plane_wave_rhs(nn, 0.0, k, f, g),
    )
    # Incidence at 0°: forward is far_field's observation angle π (sin 0, cos −1).
    puto = 1j * k * df * ei[:nn] - ei[nn:]
    amp = np.sum(np.exp(1j * k * g) * puto) * TWOPI / nn
    return amp.imag / (k * 2.0 * rad)


def run(name):
    """One shape: map diagnostics, both references, error ladders per map."""
    if name in SPIKES:
        n1 = SPIKES[name]
        shape = {"m": 6, "n1": n1, "n2": 8.0, "n3": 8.0}
        rad = 400.0 / 2.0 ** (3.0 / n1)  # tip radius 400 nm
    else:
        shape, rad = SHAPES[name]
    full = {"a": 1.0, "b": 1.0, **shape}
    out = {"shape": name, "rad": rad}
    maps = {"theta": identity_map()}
    for label, band in (("arc", (20.0, 20.0)), ("adapt5", (20.0, 100.0))):
        try:
            maps[label] = Parametrisation.gielis(
                rad=rad, **full, r_band=band, n_core=N_CORE, wavelength_ref=LAM
            )
            out[f"{label}_K"] = maps[label].n_terms
        except (ValueError, ZeroDivisionError) as exc:
            out[f"{label}_error"] = repr(exc)
    refs = {label: qext(maps[label], NN_REF, rad, shape) for label in maps if label != "adapt5"}
    ref = refs["theta"]
    out["ref_spread_theta_vs_arc"] = abs(refs.get("arc", ref) - ref) / abs(ref)
    for label, par in maps.items():
        out[f"{label}_err"] = [
            abs(qext(par, nn, rad, shape) - ref) / abs(ref) for nn in NNS
        ]
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
