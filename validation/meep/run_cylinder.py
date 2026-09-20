"""Gate 1: reproduce the analytic Mie cylinder in MEEP.

Independent FDTD check of pysie2d's cross-sections, in the *time* domain. This
is the maximally independent anchor of the two: dolfinx and pysie2d are both
frequency-domain solves of the same Helmholtz operator, so they can in
principle share a misconception about it. MEEP integrates Maxwell's curl
equations forward in time on a Yee grid and recovers the spectrum by Fourier
transforming the fields as they go — a different equation, a different
discretisation, and a different definition of "the answer at λ"
(CLAUDE.md non-negotiable §3).

Run it inside the conda environment beside this file — never under uv::

    mamba create -y -n pysie2d-meep -c conda-forge python=3.12 pymeep numpy scipy
    conda run -n pysie2d-meep python validation/meep/run_cylinder.py --case te

It writes ``tests/data/meep-circle-lossless-{te,tm}.json`` in the
frozen-spectrum format of ``validation/spectrum.py``, and does not import
pysie2d.

Scope: **lossless only**
------------------------
MEEP has no frequency-independent Im ε. A ``D_conductivity`` σ gives
ε(f) = ε_r(1 + iσ/2πf), so Im ε ∝ 1/f — dispersive across a band, while the
frozen lossy cases pin ``epsi = 0.5`` constant from 400 to 900 nm. Matching
that would mean either one narrowband run per wavelength or fitting a Lorentz
pole, and a fit is a new error source with no closed form behind it. Absorption
is therefore anchored by dolfinx, which represents a constant Im ε exactly, and
MEEP anchors the two lossless cases broadband in a single pulse per
polarisation. ``C_abs`` is still *measured* here rather than assumed zero: it
is the gate-3 null test, and it is the observable that catches a polarisation
or time-convention sign error.

Units
-----
MEEP is scale-invariant and works in units of a chosen length ``a``; here
``a = 100 nm`` (``NM_PER_A``), so every length below is in units of 100 nm and
every frequency is ``f = a/λ_vac`` with λ_vac the **vacuum** wavelength
(conventions §2). Cross-sections come back in units of ``a`` and are converted
to nm exactly once, on the way into the frozen file.

Polarisation
------------
pysie2d's invariant axis is **y**; MEEP's 2-D in-plane coordinates are
``(x, y)`` with invariant axis **z**. So this mesh's ``(x, y)`` are pysie2d's
``(x, z)``, and the field along the invariant axis is ``Ez`` for pysie2d
``pol = 2`` (TE) and ``Hz`` for ``pol = 1`` (TM). MEEP names those two
polarisations the other way round (its "TM" is the Ez one), which is the
relabelling the ``pol_mapping`` field of every frozen file spells out in
words. Both codes use ``exp(-iωt)``, so Im ε > 0 absorbs in both.

The incident wave matches pysie2d's ``plane_wave_rhs`` at ``angle = 0``,
``exp(i k_bg (x sinθ − z cosθ))`` → ``exp(−i k_bg z)``: propagating along
**−z**, which here is **−y**. On a circle this leaves every cross-section
unchanged; on the gate-4 star it would be silently wrong, so it is pinned
rather than inferred.

Observables
-----------
Two runs per case, the standard MEEP scattering pattern:

1. **Normalisation** — background only. Saves the incident Fourier fields on
   the flux box and the near-to-far surface, and measures the incident
   intensity.
2. **Scattering** — with the particle, subtracting the saved incident fields.

    C_sca  = (outward flux of the scattered field through the box) / I_inc
    C_abs  = (net *inward* flux of the total field through the box) / I_inc
    C_ext  = C_sca + C_abs

``C_ext`` is a sum here, not a third measurement, so it is not evidence the way
the dolfinx optical-theorem residual is. The independent check instead comes
from computing ``C_sca`` a **second** way, from the near-to-far-field transform
on a separate surface: a surface-equivalence integral evaluated in the radiation
zone shares no machinery with a direct flux tally on the near-field box, so a
normalisation or polarisation error cannot cancel between them. The relative
disagreement between the two travels with the frozen file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import meep as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "validation"))

from spectrum import Convergence, Spectrum, Tolerance, write  # noqa: E402

DATA = ROOT / "tests" / "data"

NM_PER_A = 100.0
"""MEEP length unit, in nm. Every length below is in units of ``a``."""

# Geometry and material of the shared fixture — the same circle the frozen Mie
# files use, so gate 1 is a diff against a file already in the repo.
RAD_NM = 200.0
N_CORE = 1.5
N_CLAD = 1.0
WAVELENGTHS = np.linspace(400.0, 900.0, 51)

RAD = RAD_NM / NM_PER_A

# Domain layout, in units of a. The flux box has to clear the evanescent skirt
# of the particle, and the near-to-far surface has to sit outside the flux box
# so the two observables do not read the same fields off the same cells.
R_FLUX = 4.0 * RAD
R_N2F = 5.0 * RAD
PAD = 2.0
DPML = 5.0  # ≈ 0.55 λ_max; a PML thinner than half a wavelength reflects.

R_FAR = 1.0e4
"""Radius (units of a) at which the far fields are evaluated.

Far enough that ``k·R ≫ 1`` at the *longest* wavelength — 1e4 a is 1 mm,
against λ_max = 9 a — so the transform is evaluated where its asymptotic form
is valid rather than in the intermediate zone.
"""

N_FAR = 360
"""Sample points on the far-field circle.

The scattered pattern of a 200 nm circle at λ = 400 nm carries a handful of
cylindrical harmonics, so 360 points oversample it by an order of magnitude and
the trapezoid rule on a closed periodic contour converges spectrally.
"""

RESOLUTION = 25
"""Yee-grid pixels per a, i.e. per 100 nm.

At λ_min = 400 nm inside n_core = 1.5 this is ~27 pixels per wavelength in the
material. FDTD is second-order accurate, so this is a gate-2 knob, not a
converged value.
"""

CASES = {
    "te": (2, mp.Ez),
    "tm": (1, mp.Hz),
}

POL_MAPPING = {
    2: (
        "pysie2d pol=2 is TE: E along the invariant axis, Mie coefficient "
        "b_n. Here the invariant axis is z and the scalar unknown is E_z, so "
        "this driver's cell coordinates (x, y) correspond to pysie2d's "
        "(x, z), and MEEP's own name for this polarisation is 'TM' "
        "(Ez, Hx, Hy). Both codes use exp(-iωt), so Im(eps) > 0 absorbs in "
        "both."
    ),
    1: (
        "pysie2d pol=1 is TM: H along the invariant axis, Mie coefficient "
        "a_n. Same axis relabelling as pol=2; the scalar unknown is H_z and "
        "MEEP's own name for this polarisation is 'TE' (Hz, Ex, Ey). Both "
        "codes use exp(-iωt)."
    ),
}


def _frequencies() -> list[float]:
    """The frozen wavelength grid as MEEP frequencies, ascending in λ.

    MEEP's ``add_flux(fcen, df, nfreq)`` form lays points out uniformly in
    *frequency*; the frozen grid is uniform in wavelength. Passing the list
    explicitly hits the grid the repo already froze, so the comparison needs no
    interpolation — which would otherwise smear a resonance and be mistaken for
    a discretisation error.
    """
    return [NM_PER_A / lam for lam in WAVELENGTHS]


def _cell() -> tuple[float, mp.Vector3]:
    """Half-width of the computational cell (units of a) and the cell size."""
    half = R_N2F + PAD + DPML
    return half, mp.Vector3(2 * half, 2 * half)


def _source(component, half: float) -> list[mp.Source]:
    """A plane wave sheet propagating along −y.

    The sheet spans the full cell width, corners included, so it has to be
    ``is_integrated`` — otherwise its ends sit inside the PML as unterminated
    current filaments and radiate spuriously. Placed at the inner PML edge, so
    the half of its radiation that goes the wrong way (+y) is absorbed within a
    wavelength and never reaches the particle.
    """
    fcen = 0.5 * (NM_PER_A / WAVELENGTHS[0] + NM_PER_A / WAVELENGTHS[-1])
    df = 1.5 * (NM_PER_A / WAVELENGTHS[0] - NM_PER_A / WAVELENGTHS[-1])
    return [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
            component=component,
            center=mp.Vector3(0, half - DPML),
            size=mp.Vector3(2 * half, 0),
        )
    ]


def _flux_box(sim: mp.Simulation, freqs: list[float], r: float):
    """A box of four flux planes, signed so the total is the *outward* flux.

    MEEP's flux through an x-normal plane is positive along +x, so the −x and
    −y faces carry weight −1 to make every face count power leaving the box.
    """
    regions = [
        mp.FluxRegion(center=mp.Vector3(+r, 0), size=mp.Vector3(0, 2 * r), weight=+1),
        mp.FluxRegion(center=mp.Vector3(-r, 0), size=mp.Vector3(0, 2 * r), weight=-1),
        mp.FluxRegion(center=mp.Vector3(0, +r), size=mp.Vector3(2 * r, 0), weight=+1),
        mp.FluxRegion(center=mp.Vector3(0, -r), size=mp.Vector3(2 * r, 0), weight=-1),
    ]
    return [sim.add_flux(freqs, reg) for reg in regions]


def _n2f_box(sim: mp.Simulation, freqs: list[float], r: float):
    """The near-to-far surface: the same box shape, one surface further out."""
    return sim.add_near2far(
        freqs,
        mp.Near2FarRegion(
            center=mp.Vector3(+r, 0), size=mp.Vector3(0, 2 * r), weight=+1
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(-r, 0), size=mp.Vector3(0, 2 * r), weight=-1
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, +r), size=mp.Vector3(2 * r, 0), weight=+1
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, -r), size=mp.Vector3(2 * r, 0), weight=-1
        ),
    )


def run_normalisation(component, freqs: list[float]):
    """Background-only run: the incident fields, saved for subtraction.

    Returns:
        ``(flux_data, n2f_data, intensity)`` — the four boxes' Fourier data,
        the near-to-far data, and the incident intensity (power per unit
        length, units of a) at every frequency.
    """
    half, size = _cell()
    sim = mp.Simulation(
        cell_size=size,
        boundary_layers=[mp.PML(DPML)],
        sources=_source(component, half),
        resolution=RESOLUTION,
        default_material=mp.Medium(index=N_CLAD),
        force_complex_fields=False,
    )
    boxes = _flux_box(sim, freqs, R_FLUX)
    n2f = _n2f_box(sim, freqs, R_N2F)
    sim.run(until_after_sources=mp.stop_when_dft_decayed())

    # The wave travels −y, so it enters through the +y face. That face's flux
    # is negative (power flowing in), and dividing by its length turns the
    # incident power into the intensity every cross-section is normalised by:
    # C = P/I then carries units of length, as a 2-D cross-section must.
    incident = np.abs(np.asarray(mp.get_fluxes(boxes[2])))
    intensity = incident / (2 * R_FLUX)

    flux_data = [sim.get_flux_data(b) for b in boxes]
    n2f_data = sim.get_near2far_data(n2f)
    return flux_data, n2f_data, intensity


def run_scattering(component, freqs: list[float], flux_data, n2f_data, intensity):
    """Run with the particle, subtracting the saved incident fields.

    Returns:
        dict of arrays ``c_sca``, ``c_abs``, ``c_ext``, ``c_sca_far`` — all in
        units of a.
    """
    half, size = _cell()
    particle = [
        mp.Cylinder(radius=RAD, material=mp.Medium(index=N_CORE), height=mp.inf)
    ]
    sim = mp.Simulation(
        cell_size=size,
        boundary_layers=[mp.PML(DPML)],
        sources=_source(component, half),
        resolution=RESOLUTION,
        default_material=mp.Medium(index=N_CLAD),
        geometry=particle,
        force_complex_fields=False,
    )

    # Two coincident boxes on purpose: the first has the incident field removed
    # and so carries the *scattered* power, the second keeps the total field
    # and so carries the net absorbed power. Reading both off one box is not
    # possible — the subtraction is destructive.
    scat_boxes = _flux_box(sim, freqs, R_FLUX)
    total_boxes = _flux_box(sim, freqs, R_FLUX)
    n2f = _n2f_box(sim, freqs, R_N2F)
    for box, data in zip(scat_boxes, flux_data, strict=True):
        sim.load_minus_flux_data(box, data)
    sim.load_minus_near2far_data(n2f, n2f_data)

    sim.run(until_after_sources=mp.stop_when_dft_decayed())

    scattered = sum(np.asarray(mp.get_fluxes(b)) for b in scat_boxes)
    # Net *inward* flux of the total field is the power the particle swallowed;
    # for a real ε it is zero, and how close to zero is the gate-3 null test.
    absorbed = -sum(np.asarray(mp.get_fluxes(b)) for b in total_boxes)

    far = _far_field_power(sim, n2f, freqs)
    return {
        "c_sca": scattered / intensity,
        "c_abs": absorbed / intensity,
        "c_ext": (scattered + absorbed) / intensity,
        "c_sca_far": far / intensity,
    }


def _far_field_power(sim: mp.Simulation, n2f, freqs: list[float]) -> np.ndarray:
    """Radiated power from the near-to-far transform, integrated on a circle.

    This is the independent second path to ``C_sca``: a surface-equivalence
    integral evaluated in the radiation zone, sharing no machinery with the
    direct flux tally on the near-field box. A factor, a sign, or a swapped
    polarisation would have to be wrong the *same* way in both to survive.

    The contour is closed and periodic, so the trapezoid rule on N_FAR equally
    spaced angles converges spectrally — no quadrature error at this sample
    count.
    """
    phi = 2 * np.pi * np.arange(N_FAR) / N_FAR
    power = np.zeros(len(freqs))
    for angle in phi:
        pt = mp.Vector3(R_FAR * np.cos(angle), R_FAR * np.sin(angle))
        ff = sim.get_farfields(n2f, 1, center=pt, size=mp.Vector3())
        ex, ey, ez = ff["Ex"], ff["Ey"], ff["Ez"]
        hx, hy, hz = ff["Hx"], ff["Hy"], ff["Hz"]
        # S = Re(E × H*), with no ½. MEEP's DFT flux omits that factor, and
        # the far fields are normalised to match it, so the two C_sca paths are
        # comparable as they stand. Carrying a ½ here showed up as a dead-flat
        # far/box − 1 = −0.500 at every wavelength — which is exactly why a
        # factor error is worth having a second path to catch: it is constant,
        # so nothing about the spectrum's *shape* looks wrong.
        s_x = np.real(ey * np.conj(hz) - ez * np.conj(hy))
        s_y = np.real(ez * np.conj(hx) - ex * np.conj(hz))
        power += (s_x * np.cos(angle) + s_y * np.sin(angle)) * R_FAR
    return power * (2 * np.pi / N_FAR)


def run_case(case: str) -> dict[str, np.ndarray]:
    """Normalisation run then scattering run, in units of a."""
    _pol, component = CASES[case]
    freqs = _frequencies()
    flux_data, n2f_data, intensity = run_normalisation(component, freqs)
    return run_scattering(component, freqs, flux_data, n2f_data, intensity)


def freeze(case: str, res: dict[str, np.ndarray]) -> Path:
    """Write one frozen spectrum, converting cross-sections to nm."""
    pol, _component = CASES[case]
    case_id = f"meep-circle-lossless-{case}"
    far_dev = np.max(np.abs(res["c_sca_far"] - res["c_sca"]) / np.abs(res["c_sca"]))
    spec = Spectrum(
        case_id=case_id,
        claim=(
            "pysie2d's cross-sections on a circle agree with an independent "
            "time-domain FDTD solve (MEEP, Yee grid, broadband pulse, PML) "
            "that shares no equation, discretisation or linear algebra with "
            "the boundary-integral method."
        ),
        tool={"name": "meep", "version": mp.__version__},
        geometry={"rad": RAD_NM, "m": 0, "n1": 2.0, "n2": 2.0, "n3": 2.0, "n_pts": 200},
        material={"n_core": N_CORE, "n_clad": N_CLAD, "epsi": 0.0},
        pol=pol,
        pol_mapping=POL_MAPPING[pol],
        angle_deg=0.0,
        wavelength_nm=WAVELENGTHS,
        c_sca_nm=res["c_sca"] * NM_PER_A,
        c_ext_nm=res["c_ext"] * NM_PER_A,
        c_abs_nm=res["c_abs"] * NM_PER_A,
        convergence=Convergence(
            parameter="resolution (pixels per 100 nm)",
            coarse="placeholder — gate 2 has not been run",
            fine=RESOLUTION,
            max_rel_drift=float("nan"),
            note="PLACEHOLDER. Not to be committed until gate 2 runs.",
        ),
        tolerance=Tolerance(
            rel=float("nan"),
            abs_nm=float("nan"),
            justification="PLACEHOLDER — set from gate 2's measured floor.",
        ),
        notes=(
            "Lossless: MEEP has no frequency-independent Im(eps), so the lossy "
            "cases are anchored by dolfinx instead — see the module docstring. "
            "C_abs is measured, not assumed, and is the gate-3 null test. "
            "C_sca was computed a second, independent way from the "
            "near-to-far-field transform; worst relative disagreement between "
            f"the two paths over the grid: {far_dev:.2e}."
        ),
    )
    return write(spec, DATA / f"{case_id}.json")


def main() -> None:
    """Run one polarisation and freeze it.

    One case per process: the conda build is ``nompi``, so the only
    parallelism available is running the two polarisations as separate
    processes, and they are independent.
    """
    global RESOLUTION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), default="te")
    parser.add_argument(
        "--resolution", type=int, default=RESOLUTION, help="pixels per 100 nm"
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="write the frozen file (omit while gate 2 is still open)",
    )
    args = parser.parse_args()
    RESOLUTION = args.resolution

    res = run_case(args.case)
    for i, lam in enumerate(WAVELENGTHS):
        print(
            f"  λ={lam:6.1f}  C_sca={res['c_sca'][i] * NM_PER_A:10.3f}  "
            f"C_ext={res['c_ext'][i] * NM_PER_A:10.3f}  "
            f"C_abs={res['c_abs'][i] * NM_PER_A:10.3e}  "
            f"far/box−1={res['c_sca_far'][i] / res['c_sca'][i] - 1:+.2e}",
            flush=True,
        )
    np.savez(
        Path(__file__).parent / f"_last-{args.case}-res{RESOLUTION}.npz",
        wavelength_nm=WAVELENGTHS,
        **res,
    )
    if args.freeze:
        print(f"  → {freeze(args.case, res).name}", flush=True)


if __name__ == "__main__":
    main()
