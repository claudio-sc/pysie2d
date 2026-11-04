"""v1.4: 3D MEEP reference for a pillar side-coupled to an SOI strip.

The 3D side of the 2.5D model error (roadmap §4, v1.4): an air-clad
500 × 220 nm Si strip on SiO₂, the inverse-EIM paper's case (Høvik & Aksnes,
2024; materials 3.47 / 1.44 / 1.0), with a Si pillar beside it. MEEP measures
the TE₀ transmission and reflection by eigenmode coefficients; the power not
in the guided mode, 1 − |t|² − |r|², is radiated, out of plane included. It
does not import pysie2d. Run inside the conda environment, single core::

    ~/miniforge3/envs/pysie2d-meep/bin/python validation/meep/run_strip3d.py \
        --drive pillar --resolution 30 --out run.npz

Units a = 1 µm. The strip runs along x with its bottom face on the oxide at
z = 0, centred at y = 0; the pillar is centred at x = 0 on the +y side. The
pillar is mirror-symmetric in x, so a drive from the right equals the drive
from the left and is not run. No mirror symmetry is used: the pillar breaks
y, the air/oxide pair breaks z.

Observables: ``alpha[plane, band-1, ±, freq]`` on the planes ``MONITORS``,
from the ``norm`` run (bare strip) and the ``pillar`` run, launched toward +x
by one broadband eigenmode source. t = α⁺(far)/α⁺_norm(far),
r = α⁻(near)/α⁺_norm(near). v1.3 measured a ~1e-3 floor from the broadband
source's single mode profile; the 2.5D error this run measures is expected at
the percent level, so one source suffices and the floor is recorded, not
removed.
"""

from __future__ import annotations

import argparse
import time

import meep as mp
import numpy as np

N_SI, N_OX, N_AIR = 3.47, 1.44, 1.0
W, H = 0.5, 0.22
R_PILLAR, GAP = 0.25, 0.06
"""Pillar radius and edge-to-edge gap to the strip, µm. r = 150 nm at a 100 nm
gap scattered 0.1–0.2 % (res 20 smoke), at the ~1e-3 MEEP floor; this one
must move the guided power by percents for a model error to be measurable."""
LAM_MIN, LAM_MAX, N_FREQ = 1.50, 1.65, 31
"""Band, µm. 500 × 220 nm air-clad is single-mode across it (R3 domain at
1550 nm; the TE₁ cutoff lies well below 1.50 µm for this width)."""
DPML = 1.5
"""µm, ≥ λ_max: the guided mode enters the x-PML, radiation every face."""
HALF_X = 3.5
PAD = 1.2
"""Clear cladding beyond the structure, µm. TE₀ (n_eff ≈ 2.38) decays into
oxide at γ ≈ 7.7 µm⁻¹, so 1.2 µm leaves e⁻⁹ of the field at the PML."""
MONITORS = (2.0, 2.5)
"""±x planes, µm: 2.0 is production, 2.5 the monitor-distance systematic."""
X_SOURCE = -3.2
"""0.7 µm outside the near monitor: at 0.1 µm (res 10 smoke) its evanescent
near field put 2 % of spurious |r|² on that plane."""
DECAY = 1e-9


def _cell() -> tuple[mp.Vector3, mp.Vector3]:
    y_lo, y_hi = -W / 2 - PAD, W / 2 + GAP + 2 * R_PILLAR + PAD
    z_lo, z_hi = -PAD, H + PAD
    size = mp.Vector3(
        2 * (HALF_X + DPML), y_hi - y_lo + 2 * DPML, z_hi - z_lo + 2 * DPML
    )
    return size, mp.Vector3(0, (y_lo + y_hi) / 2, (z_lo + z_hi) / 2)


def run(drive: str, res: int) -> dict:
    """One 3D run, bare strip (``norm``) or with the pillar, in units of a.

    Returns:
        The mode coefficients ``alpha`` and ``n_eff`` on every monitor plane,
        with the frequencies, planes, stop time, wall time and voxel count.
    """
    size, centre = _cell()
    si, ox = mp.Medium(index=N_SI), mp.Medium(index=N_OX)
    big = 4 * max(size.x, size.y, size.z)
    geometry = [
        mp.Block(
            mp.Vector3(big, big, big / 2),
            center=mp.Vector3(0, 0, -big / 4),
            material=ox,
        ),
        mp.Block(mp.Vector3(big, W, H), center=mp.Vector3(0, 0, H / 2), material=si),
    ]
    if drive == "pillar":
        yc = W / 2 + GAP + R_PILLAR
        geometry.append(
            mp.Cylinder(
                R_PILLAR, height=H, center=mp.Vector3(0, yc, H / 2), material=si
            )
        )
    f_lo, f_hi = 1 / LAM_MAX, 1 / LAM_MIN
    fcen, df = (f_lo + f_hi) / 2, 1.5 * (f_hi - f_lo)
    plane = mp.Vector3(0, size.y - 2 * DPML, size.z - 2 * DPML)
    sim = mp.Simulation(
        cell_size=size,
        geometry_center=centre,
        resolution=res,
        geometry=geometry,
        default_material=mp.Medium(index=N_AIR),
        boundary_layers=[mp.PML(DPML)],
        sources=[
            mp.EigenModeSource(
                mp.GaussianSource(fcen, fwidth=df),
                center=mp.Vector3(X_SOURCE, centre.y, centre.z),
                size=plane,
                eig_band=1,
                eig_kpoint=mp.Vector3(1),
                eig_match_freq=True,
            )
        ],
    )
    freqs = np.linspace(f_lo, f_hi, N_FREQ)
    planes = [s * x for x in MONITORS for s in (-1.0, 1.0)]
    mons = [
        sim.add_mode_monitor(
            freqs, mp.ModeRegion(center=mp.Vector3(x, centre.y, centre.z), size=plane)
        )
        for x in planes
    ]
    probe = mp.Vector3(MONITORS[-1], 0, H / 2)
    t0 = time.time()
    sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ey, probe, DECAY))
    alpha, neff = [], []
    for m in mons:
        c = sim.get_eigenmode_coefficients(m, [1])
        alpha.append(c.alpha[0])
        neff.append(np.array([k.x for k in c.kpoints]) / freqs)
    return {
        "drive": drive,
        "resolution": res,
        "freqs": freqs,
        "planes": np.array(planes),
        "alpha": np.array(alpha),
        "n_eff": np.array(neff),
        "t_stop": sim.meep_time(),
        "seconds": time.time() - t0,
        "voxels": float(size.x * size.y * size.z * res**3),
    }


def main() -> None:
    """Run one drive at one resolution and save it; one case per process."""
    p = argparse.ArgumentParser()
    p.add_argument("--drive", choices=("norm", "pillar"), required=True)
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = run(a.drive, a.resolution)
    np.savez(a.out, **out)
    print(
        f"{a.drive} res {a.resolution}: t = {out['t_stop']:.0f}, "
        f"{out['voxels']:.3g} vox, {out['seconds']:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
