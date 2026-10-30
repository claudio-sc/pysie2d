"""v1.3: S-parameters of a non-circular cluster beside an SOI slab, in MEEP.

The external anchor A9 of the waveguide roadmap (§5): the guided-mode
amplitudes pysie2d computes by residue of the Sommerfeld integral, measured
here by an FDTD eigenmode source and eigenmode-coefficient monitors. MEEP
shares none of the machinery: no Green function, no Sommerfeld path, no
residue, no boundary integral. Its guided mode comes from MPB, a mode solver
independent of ``Stack.guided_modes``, and its n_eff is recorded so that the two
mode solvers are compared before any amplitude is.

Run it inside the conda environment beside this file, never under uv::

    conda run -n pysie2d-meep python validation/meep/run_waveguide.py \
        --case te --drive left --resolution 25 --out run.npz

It does not import pysie2d. ``validation/waveguide_study.py`` runs the batches
and ``validation/waveguide_reference.py`` computes pysie2d's side.

Geometry
--------
SOI: 220 nm Si (n = 3.476) between SiO₂ half-spaces (n = 1.444), non-dispersive
in both codes, so one broadband pulse per run is exact rather than fitted. The
cluster is ``gielis.CLUSTER``: two Si particles in the SiO₂ cover. Units of
``a = 100 nm``; ``f = a/λ_vac``. World coordinates are pysie2d's: this cell's
``(x, y)`` is pysie2d's ``(x, z)``, the slab top at ``y = 0``, the slab in
``−2.2 < y < 0`` (``geometry_center`` places the cell, so no shift is applied
by hand).

Polarisation
------------
pysie2d ``pol = 2`` (TE, ``E_y``) is MEEP's ``Ez`` with ``eig_parity = ODD_Z``;
``pol = 1`` (TM, ``H_y``) is ``Hz`` with ``EVEN_Z`` (v1.0's mapping, the
invariant axis renamed). The two slab modes differ grossly in n_eff (2.85 vs
2.05 at 1550 nm), so a swapped parity cannot pass the n_eff comparison.

Observables
-----------
Three runs per polarisation: ``norm`` (bare slab, launched towards +x),
``left`` (cluster, launched towards +x), ``right`` (cluster, towards −x).
``norm_right`` (bare slab, towards −x) exists to measure the mirror defect
that lets ``norm`` stand in for it.
Eigenmode coefficients ``α±`` of band 1 are recorded on vertical planes at
``x = ±MONITORS``; every run records all of them, so the monitor-distance
systematic costs no extra run. In units of power, with ``α`` the incident
coefficient from the matching ``norm`` run at the same plane:

    t   = α⁺(far plane) / α⁺_norm(far plane)     (convention-free ratio)
    r   = α⁻(near plane) / α⁺_norm(near plane)   (phase: see compare)
    P_rad = 1 − |t|² − |r|²                       (lossless: P_up + P_down)

The source's own launch spectrum and MPB's mode-phase choice cancel in every
ratio, because the ``norm`` run uses the identical source.

Stop rule
---------
Not MEEP's ``stop_when_dft_decayed``: in v1.0 that rule drove TE runs 17×
longer than TM (87 838 steps against 5 227 at resolution 25). Here the run
stops when ``|field|²`` at a point in the slab past the cluster has stayed below
``DECAY`` times its running maximum for a whole window, and the coefficients
are also recorded at ``SNAPSHOT``, earlier in the same run, so the
truncation systematic is read without a second run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import meep as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gielis  # noqa: E402
from soi import N_OX, N_SI, NM_PER_A, T_SI_NM, WAVELENGTHS  # noqa: E402

T_SI = T_SI_NM / NM_PER_A

DPML_X = 20.0
"""2 µm. The guided mode enters this PML along the slab at normal incidence."""

DPML_Y = 15.0
"""1.5 µm ≥ λ_max/n_ox = 1.21 µm: the radiation leaving up and down."""

PAD = 20.0
"""Clear cladding between the structure and the y-PML, a.

Set by TM₀ at 1750 nm: n_eff = 1.856 gives a cover decay constant of
4.18 µm⁻¹, so a 1 µm pad leaves 2e-4 of the mode's power density at the PML,
and the mode MPB solves on the monitor plane would see the absorber; 2 µm
leaves 5e-8. TE₀ decays twice as fast and is not the constraint."""

MONITORS = (10.0, 15.0, 20.0)
"""Planes at ±x, a. 15 is the production plane; 10 and 20 are the
monitor-distance systematic, which is free because every run records all three.
The nearest lies 7 a beyond the cluster's x-extent (−3.24 … 2.80 a)."""

X_SOURCE = 23.0
"""Source plane, a: 3 a outside the farthest monitor."""

DECAY = 1e-10
SNAPSHOT = 1e-6
"""Field-decay levels (of |f|², relative to its maximum) for the final record
and the earlier one that measures the truncation systematic. The cluster is
low-Q, so the probe follows the pulse's Gaussian tail: at 1e-8 the snapshot
fell in the same window as the stop and recorded nothing (res 8, TE)."""

SUBBANDS = 1
"""Eigenmode sources across the band; see :func:`run`."""

WINDOW = 20.0
"""a/c over which the decay must hold before the run stops."""

CASES = {"te": (2, mp.Ez, mp.ODD_Z), "tm": (1, mp.Hz, mp.EVEN_Z)}


def _freqs() -> np.ndarray:
    return NM_PER_A / WAVELENGTHS


def _cell() -> tuple[mp.Vector3, mp.Vector3, float, float]:
    """Cell size, cell centre (world), and the y-extent of the region inside the PML."""
    top = max(
        (z0 + gielis.load(s)["z"].max()) / NM_PER_A for s, _, z0 in gielis.CLUSTER
    )
    # MEEP anchors its lattice on the cell centre, so the cell is centred on the
    # slab's top interface: y = 0 is then always a lattice point and y = −T_SI
    # is one whenever T_SI·res is an integer (:func:`_check_aligned`). With the
    # centre elsewhere the interfaces landed at a resolution-dependent pixel
    # fraction and MPB's n_eff error jumped (TE at res 10/20/40/80: −1.7e-2,
    # −5.4e-3, −5.0e-3, −2.0e-3; res 15 twice its neighbours); aligned it is
    # clean first order (−8.3e-3, −3.8e-3, −1.8e-3, −8.6e-4), which can be
    # extrapolated. The symmetric cell costs < 2 % in area.
    # An integer half-height too: MEEP rounds a cell that is not a whole number
    # of pixels, which moves its centre by a resolution-dependent sub-pixel
    # (res 30 fell off the first-order line until this was fixed).
    h = float(np.ceil(max(T_SI + PAD, top + PAD)))
    y_lo, y_hi = -h, h
    half_x = X_SOURCE + 2.0 + DPML_X
    size = mp.Vector3(2 * half_x, (y_hi - y_lo) + 2 * DPML_Y)
    return size, mp.Vector3(), y_lo, y_hi


def _geometry(with_cluster: bool) -> list:
    si = mp.Medium(index=N_SI)
    slab = mp.Block(
        center=mp.Vector3(0, -T_SI / 2),
        size=mp.Vector3(mp.inf, T_SI, mp.inf),
        material=si,
    )
    if not with_cluster:
        return [slab]
    parts = []
    for shape, x0, z0 in gielis.CLUSTER:
        c = gielis.load(shape)
        verts = [
            mp.Vector3((x0 + x) / NM_PER_A, (z0 + z) / NM_PER_A)
            for x, z in zip(c["x"], c["z"], strict=True)
        ]
        parts.append(mp.Prism(verts, height=mp.inf, material=si))
    return [slab, *parts]


def _check_aligned(resolution: int) -> None:
    """Refuse a resolution that puts a slab interface inside a pixel."""
    if abs(T_SI * resolution - round(T_SI * resolution)) > 1e-9:
        raise SystemExit(
            f"resolution {resolution}: T_SI·res = {T_SI * resolution:g} is not an "
            "integer, so the slab's bottom interface splits a pixel"
        )


def run(case: str, drive: str, resolution: int) -> dict:
    """One FDTD run; returns everything the study and the comparison need."""
    _check_aligned(resolution)
    pol, component, parity = CASES[case]
    sign = -1.0 if drive in ("right", "norm_right") else +1.0
    size, centre, y_lo, y_hi = _cell()
    span = mp.Vector3(0, y_hi - y_lo)
    yc = centre.y
    f = _freqs()
    # SUBBANDS sources side by side in frequency, each an eigenmode source whose
    # profile MPB solves at its *own* centre. One profile for the whole band
    # misfits the mode away from the centre, and that misfit was the floor of
    # the batch-1 ladder: a 40 nm narrowband run removed 72–100 % of the TM
    # floor and 50–85 % of TE's (srcdiag, 2026-10-30). Each source's Gaussian
    # spans its sub-band, so the pulse, and the run, lengthen ~SUBBANDS-fold.
    edges = np.linspace(f.min(), f.max(), SUBBANDS + 1)
    fwidth = 1.2 * (edges[1] - edges[0])
    sources = [
        mp.EigenModeSource(
            mp.GaussianSource(0.5 * (lo + hi), fwidth=fwidth),
            center=mp.Vector3(-sign * X_SOURCE, yc),
            size=span,
            direction=mp.X,
            eig_kpoint=mp.Vector3(sign),
            eig_band=1,
            eig_parity=parity,
        )
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    sim = mp.Simulation(
        cell_size=size,
        geometry_center=centre,
        geometry=_geometry(drive in ("left", "right")),
        default_material=mp.Medium(index=N_OX),
        sources=sources,
        resolution=resolution,
        boundary_layers=[
            mp.PML(DPML_X, direction=mp.X),
            mp.PML(DPML_Y, direction=mp.Y),
        ],
    )
    planes = [s * x for x in MONITORS for s in (-1.0, 1.0)]
    mons = [
        sim.add_mode_monitor(f, mp.ModeRegion(center=mp.Vector3(x, yc), size=span))
        for x in planes
    ]
    # Diagnostics only: horizontal planes just inside the y-PML. Grazing
    # radiation leaves through the x-PML first, so these are lower bounds on
    # P_up and P_down, never compared as equalities.
    width = mp.Vector3(2 * (X_SOURCE + 2.0))
    up = sim.add_flux(f, mp.FluxRegion(center=mp.Vector3(0, y_hi), size=width))
    down = sim.add_flux(f, mp.FluxRegion(center=mp.Vector3(0, y_lo), size=width))

    def coeffs() -> np.ndarray:
        """α of band 1, shape (planes, freqs, 2): [..., 0] = +x, [..., 1] = −x."""
        return np.stack(
            [
                sim.get_eigenmode_coefficients(m, [1], eig_parity=parity).alpha[0]
                for m in mons
            ]
        )

    probe = mp.Vector3(sign * MONITORS[-1], -T_SI / 2)
    t_source = 2 * 5.0 / fwidth  # GaussianSource's default cutoff is 5 widths
    state = {"peak": 0.0, "window": 0.0, "t0": 0.0, "snap": None, "t_snap": None}

    def done(s: mp.Simulation) -> bool:
        v = abs(s.get_field_point(component, probe)) ** 2
        state["peak"] = max(state["peak"], v)
        state["window"] = max(state["window"], v)
        t = s.meep_time()
        if t - state["t0"] < WINDOW:
            return False
        level, state["window"], state["t0"] = state["window"], 0.0, t
        if t < t_source or state["peak"] == 0.0:
            return False
        if state["snap"] is None and level < SNAPSHOT * state["peak"]:
            state["snap"], state["t_snap"] = coeffs(), t
        return level < DECAY * state["peak"]

    t0 = time.perf_counter()
    sim.run(until=done)
    seconds = time.perf_counter() - t0
    alpha = coeffs()
    res = sim.get_eigenmode_coefficients(mons[len(mons) // 2], [1], eig_parity=parity)
    n_eff = np.array([abs(k.x) for k in res.kpoints]) / f

    return {
        "wavelengths": WAVELENGTHS,
        "planes_nm": np.array(planes) * NM_PER_A,
        "alpha": alpha,
        "alpha_snapshot": state["snap"],
        "n_eff": n_eff,
        "flux_up": np.array(mp.get_fluxes(up)),
        "flux_down": -np.array(mp.get_fluxes(down)),
        "t_stop": sim.meep_time(),
        "t_snapshot": state["t_snap"],
        "timesteps": sim.timestep(),
        "pixels": int(round(size.x * resolution) * round(size.y * resolution)),
        "seconds": seconds,
        "pol": pol,
        "drive": drive,
        "knobs": json.dumps(
            {
                "resolution": resolution,
                "dpml_x": DPML_X,
                "dpml_y": DPML_Y,
                "pad": PAD,
                "monitors": MONITORS,
                "x_source": X_SOURCE,
                "subbands": SUBBANDS,
                "decay": DECAY,
                "snapshot": SNAPSHOT,
                "cluster": gielis.CLUSTER,
                "n_si": N_SI,
                "n_ox": N_OX,
                "t_si_nm": T_SI * NM_PER_A,
                "meep": mp.__version__,
            }
        ),
    }


def main() -> None:
    """Run one case and drive, and write its ``.npz``."""
    global DPML_X, DPML_Y, PAD, X_SOURCE, WAVELENGTHS, SUBBANDS
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", choices=sorted(CASES), required=True)
    p.add_argument(
        "--drive", choices=("norm", "left", "right", "norm_right"), required=True
    )
    p.add_argument("--resolution", type=int, default=25, help="pixels per 100 nm")
    p.add_argument("--dpml-x", type=float, default=DPML_X)
    p.add_argument("--dpml-y", type=float, default=DPML_Y)
    p.add_argument("--pad", type=float, default=PAD)
    p.add_argument("--x-source", type=float, default=X_SOURCE)
    p.add_argument("--subbands", type=int, default=SUBBANDS)
    p.add_argument(
        "--band-nm",
        type=float,
        nargs=3,
        metavar=("LO", "HI", "N"),
        help="a sub-band of the grid, for the source-bandwidth systematic",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    DPML_X, DPML_Y, PAD, X_SOURCE = a.dpml_x, a.dpml_y, a.pad, a.x_source
    SUBBANDS = a.subbands
    if a.band_nm:
        WAVELENGTHS = np.linspace(a.band_nm[0], a.band_nm[1], int(a.band_nm[2]))
    out = run(a.case, a.drive, a.resolution)
    if out["alpha_snapshot"] is None:  # stopped by the window before the snapshot level
        out["alpha_snapshot"] = out["alpha"]
        out["t_snapshot"] = out["t_stop"]
    np.savez(a.out, **out)
    print(
        f"{a.case} {a.drive} res {a.resolution}: {out['timesteps']} steps, "
        f"t = {out['t_stop']:.0f} a/c, {out['pixels']:.3g} px, {out['seconds']:.1f} s, "
        f"{out['pixels'] * out['timesteps'] / out['seconds']:.3g} px·step/s"
    )


if __name__ == "__main__":
    main()
