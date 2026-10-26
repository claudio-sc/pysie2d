"""Per-order multipole spectra and 1550 nm field maps of the campaign's best designs.

Per order n ≥ 0 the scattered power is c_sca,n = (4/k)(|c_n|² + |c_−n|²) (n = 0: |c_0|² once),
which sums to cross_sections()["c_sca"] (checked to 1e-15 on R_eq 300). The circle of the
same area comes from analytic Mie (|c_n| = |b_n|, pol 2), so it is an independent reference.

Field maps: eval_field returns the total field inside and the scattered field outside;
the total outside adds the incident e^{-ik z} (plane_wave_rhs at angle 0). Points closer to
the boundary than 5 node spacings are masked (pysie2d.fields.eval_field accuracy note),
so maps are evaluated at nn = 1536 to keep that band thin.

Run after campaign.py and sweep.py:  uv run python docs/design/studies/anapole/multipole_data.py
"""

import base64
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from pysie2d import BIESolver, Material  # noqa: E402
from pysie2d.reference.mie import mie_coefficients  # noqa: E402
from shapes import LAM0, N_SI, POL, geometry  # noqa: E402

LAMS = np.round(np.arange(1000.0, 2400.01, 2.0), 1)
ZOOM = np.round(np.arange(1530.0, 1570.001, 0.2), 1)
N_ORD = 8  # orders 0..7 kept separately; the rest summed as "≥8"
MMAX = 16  # spectrum_tail ≤ 1e-14 at R_eq 300; checked per call below
G = 80  # field grid per side
NN_FIELD = 1536


def per_order(c, k, area):
    m = len(c) // 2
    p = np.abs(c) ** 2
    out = [p[m]] + [p[m + n] + p[m - n] for n in range(1, m + 1)]
    out = np.array(out) * 4 / k / area
    return np.r_[out[:N_ORD], out[N_ORD:].sum()]


def spectra(job):
    r_eq, a, nn = job
    area = np.pi * r_eq**2
    s = BIESolver(geometry(np.asarray(a), area, nn), Material(n_core=N_SI, pol=POL))
    res, tail = [], 0.0
    for grid in (LAMS, ZOOM):
        rows = []
        for lam in grid:
            mp = s.scatter(lam).multipoles(mmax=MMAX)
            tail = max(tail, mp.spectrum_tail)
            rows.append(per_order(mp.c, 2 * np.pi / lam, area))
        res.append(np.array(rows).T)
    circ = []
    for grid in (LAMS, ZOOM):
        rows = []
        for lam in grid:
            k = 2 * np.pi / lam
            _, b = mie_coefficients(k * r_eq, N_SI, n_max=MMAX)
            c = np.r_[b[::-1], b[1:]]  # |c_±n| = |b_n|
            rows.append(per_order(c, k, area))
        circ.append(np.array(rows).T)
    return {"spec": res[0], "zoom": res[1], "circ": circ[0], "circ_zoom": circ[1], "tail": tail}


def b64(a):
    return base64.b64encode(np.asarray(a, dtype=np.float32).tobytes()).decode()


def field_map(job):
    r_eq, a, ext = job
    area = np.pi * r_eq**2
    xs = np.linspace(-ext, ext, G)
    X, Z = np.meshgrid(xs, -xs)  # row 0 at the top: z = +ext
    out = {}
    for name, coeffs in (("design", np.asarray(a)), ("circle", np.zeros(len(a)))):
        geo = geometry(coeffs, area, NN_FIELD)
        r = BIESolver(geo, Material(n_core=N_SI, pol=POL)).scatter(LAM0)
        F = r.eval_field(X.ravel(), Z.ravel()).reshape(X.shape)
        # inside / outside and distance to the boundary from a fine polygon
        fb, gb = geo.f, geo.g
        d = np.min(np.hypot(X.ravel()[:, None] - fb[None, :], Z.ravel()[:, None] - gb[None, :]), axis=1).reshape(X.shape)
        th_b = np.arctan2(fb, gb)
        rb = np.hypot(fb, gb)
        th = np.arctan2(X, Z)
        inside = np.hypot(X, Z) < np.interp(th, np.r_[th_b - 2 * np.pi, th_b, th_b + 2 * np.pi], np.r_[rb, rb, rb])
        spacing = np.hypot(geo.df, geo.dg).mean() * 2 * np.pi / NN_FIELD
        band = d < 5 * spacing
        inc = np.exp(-1j * 2 * np.pi / LAM0 * Z)
        tot = np.where(inside, F, F + inc)
        tot[band] = np.nan
        # The scattered field outside is tot − e^{-ikz}; the page derives it, so only tot and the
        # inside mask are stored (keeps the page under the 16 MB artifact limit).
        out[name] = {"tot_re": b64(tot.real), "tot_im": b64(tot.imag),
                     "inside": base64.b64encode(inside.astype(np.uint8).tobytes()).decode(), "band_nm": float(5 * spacing)}
    return out


def sig(x, n=3):
    return [float(f"{v:.{n}g}") for v in x]


def main():
    camp = json.loads((HERE / "results/campaign.json").read_text())
    sw = {d["r_eq"]: d["nn"] for d in json.loads((HERE / "results/sweep.json").read_text())["designs"]}
    best = camp["best"]
    exts = []
    for b in best:
        g = geometry(np.array(b["a"]), np.pi * b["r_eq"] ** 2, 1024)
        exts.append(1.25 * max(np.hypot(g.f, g.g).max(), b["r_eq"]))
    with ProcessPoolExecutor(4) as ex:
        sp = list(ex.map(spectra, [(b["r_eq"], b["a"], sw[b["r_eq"]]) for b in best]))
        fm = list(ex.map(field_map, [(b["r_eq"], b["a"], e) for b, e in zip(best, exts)]))
    designs = []
    for b, s, f, e in zip(best, sp, fm, exts):
        designs.append({"r_eq": b["r_eq"], "ext": e, "tail": s["tail"],
                        "spec": [sig(row) for row in s["spec"]], "zoom": [sig(row) for row in s["zoom"]],
                        "circ": [sig(row) for row in s["circ"]],
                        "field": f})
        print(f"R_eq {b['r_eq']:.0f}: max spectrum_tail {s['tail']:.0e}, band {f['design']['band_nm']:.1f} nm", flush=True)
    data = {"lams": LAMS.tolist(), "zoom_lams": ZOOM.tolist(), "n_ord": N_ORD, "grid": G, "designs": designs}
    (HERE / "results/multipoles.json").write_text(json.dumps(data, separators=(",", ":")))
    print("bytes", (HERE / "results/multipoles.json").stat().st_size)


if __name__ == "__main__":
    main()
