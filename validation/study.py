"""Gate 2: the external convergence studies, as one launchable batch.

Every tolerance in every frozen external file comes from here. The drivers can
already be pointed at any single setting; what this module adds is the *matrix*
— which settings, in which groups, and what each group isolates — plus a runner
that survives being left alone.

Run it from the repo root, under uv (it needs numpy and the standard library
only; each job is a subprocess in its own conda environment)::

    uv run python validation/study.py --dry-run          # the matrix and the cost
    uv run python validation/study.py --tool all         # the green light
    uv run python validation/study.py --analyse          # the floors

Why two phases
--------------
``--phase study`` (the default) runs the refinement ladders on the study
wavelength grid and answers one question: *where does each tool stop moving?*
``--phase final`` then runs the full 51-point grid at whichever level the
answer names, and that is what gets frozen.

They cannot be one batch. The production level is the *output* of the study, so
a batch that ran both would be choosing the converged setting before measuring
it — which is the assumption gate 2 exists to remove.

What each group isolates
------------------------
A single ladder that refines everything at once cannot say *what* was limiting
the answer, and a floor reached for the wrong reason is indistinguishable from
a converged answer (the trap named in the handoff's open question 2). So each
group varies one thing against a shared baseline:

============  =========================================================
group         the error it separates out
============  =========================================================
``mesh``      element size — the main ladder, all four cases
``degree``    the basis, at fixed mesh
``geom``      the curved-element geometry error, at fixed mesh and basis
``pml``       absorber reflection: strength, thickness and profile order
``res``       MEEP's Yee-grid spacing — its main ladder
``box``       where the flux contour sits (a placement systematic)
``decay``     how long the DFT is accumulated before time stepping stops
============  =========================================================

The ``pml``, ``box`` and ``decay`` groups matter more than their size suggests.
Refining the mesh bounds the discretisation error and *nothing else*: a PML
that reflects at 1e-4, or a transform truncated early, produces a number that
is stable under mesh refinement and still wrong. That is what a floor reached
for the wrong reason looks like.

Resumability
------------
A job is skipped when its output files already exist, so an interrupted batch
is restarted by re-running the same command — nothing is recomputed. Use
``--force`` to override. Each job's stdout goes to its own log, and one line
per job is appended to ``_study/manifest.jsonl`` with its wall time, so the
cost model below can be checked against what actually happened.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation" / "_study"
CONDA = Path.home() / "miniforge3" / "bin" / "conda"

WORKERS = 1
"""Concurrent jobs.

Both conda builds are single-process (``nompi`` for MEEP, and the dolfinx
driver is run on one rank), so concurrency here is whole jobs, not threads
inside one. Three workers on this four-performance-core machine bought far
less than 3×: every job in the first batch overran its single-process
measurement by 2–5×, because MUMPS LU and Yee stepping are both
memory-bandwidth-bound and share one bus. One worker keeps each job's wall
time equal to its measured cost, which is what makes the estimates mean
something, and leaves the machine usable.
"""

ALL_CASES = ("lossless-te", "lossless-tm", "lossy-te", "lossy-tm")

PROBE_CASES = ("lossless-te", "lossy-tm")
"""The two cases the isolation groups run.

One lossless TE and one lossy TM covers both polarisations, both materials and
the ``n_clad = 1.33`` conversion path with two runs instead of four. The
systematics these groups measure — absorber reflection, contour placement,
basis error — are properties of the discretisation, so measuring them on all
four cases would spend four times the compute to watch the same number twice.
"""


@dataclass(frozen=True)
class Job:
    """One driver invocation, and the files it is expected to leave behind."""

    tool: str
    group: str
    name: str
    args: list[str]
    cases: tuple[str, ...] = ()
    cost: float = 1.0
    """Wall-time estimate in units of a resolution-25 MEEP TE run (~140 s),
    measured not guessed — see ``--dry-run``.

    **These are single-process measurements**, taken one job at a time. Both
    workloads are memory-bandwidth-bound — MUMPS LU and Yee stepping — so
    running ``WORKERS`` of them does not divide the wall time by ``WORKERS``:
    the first batch overran its concurrent estimate by 2–5× per job. Treat the
    serial total as the honest number and the concurrent one as a floor.
    """

    @property
    def outputs(self) -> list[Path]:
        """Where this job's results land."""
        d = OUT / self.tool
        if self.tool == "meep":
            return [d / f"{self.name}.npz"]
        return [d / f"{self.name}-{c}.npz" for c in self.cases]

    @property
    def done(self) -> bool:
        """Whether every output file this job promises is already on disk."""
        return all(p.exists() for p in self.outputs)


def _dolfinx(
    group: str, name: str, cases: tuple[str, ...], cost: float, **knobs
) -> Job:
    """A dolfinx job. The mesh is built once and shared by every case in it."""
    args = ["--cases", ",".join(cases), "--grid", "study"]
    for key, value in knobs.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    args += ["--out", str(OUT / "dolfinx" / name)]
    return Job("dolfinx", group, name, args, cases, cost)


def _meep(group: str, name: str, case: str, cost: float, **knobs) -> Job:
    """A MEEP job.

    Unlike dolfinx, MEEP runs the **full** 51-point grid even in the study
    phase. Its cost is set by how long the fields are stepped, not by how many
    wavelengths are recorded, so subsampling the grid would save almost nothing
    while giving up a worst-pointwise floor measured over every point that ends
    up in the frozen file.
    """
    args = ["--case", case, "--grid", "full"]
    for key, value in knobs.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    args += ["--out", str(OUT / "meep" / f"{name}.npz")]
    return Job("meep", group, name, args, cost=cost)


# --- dolfinx ---------------------------------------------------------------
#
# Cost unit: one resolution-25 MEEP TE run. Measured per-wavelength solve times
# are 3.3 s at (12, 40) degree 3, 3.2 s at (8, 25), 6.9 s at (6, 18); a job is
# 13 wavelengths × its case count.

DOLFINX_JOBS = [
    # The main ladder. (12, 40) deg 3 and (8, 25) deg 3 are the two levels
    # already measured at λ = 600; L3 and L4 are new, and exist because the
    # optical-theorem residual was still falling at (8, 25) — 2.0e-02, 4.0e-05,
    # 7.8e-06, 1.9e-06 across the levels tried so far. A ladder that stops
    # while the error is still falling measures a slope, not a floor.
    _dolfinx("mesh", "mesh-L1", ALL_CASES, 1.2, h_particle=12, h_outer=40, degree=3),
    _dolfinx("mesh", "mesh-L2", ALL_CASES, 1.2, h_particle=8, h_outer=25, degree=3),
    _dolfinx("mesh", "mesh-L3", ALL_CASES, 2.6, h_particle=6, h_outer=18, degree=3),
    _dolfinx("mesh", "mesh-L4", ALL_CASES, 8.0, h_particle=4, h_outer=12, degree=3),
    # Basis against mesh, at the baseline mesh: if degree 4 moves the answer
    # that the mesh ladder had already stopped moving, the ladder's floor was
    # the basis and not the elements.
    _dolfinx(
        "degree", "degree-4", PROBE_CASES, 1.4, h_particle=12, h_outer=40, degree=4
    ),
    # Geometry: order 3 curved elements against the baseline's order 2. A
    # circle approximated too coarsely converges at a rate set by the boundary
    # rather than by the basis, and that shows up as a premature floor.
    _dolfinx(
        "geom",
        "geom-3",
        PROBE_CASES,
        0.7,
        h_particle=12,
        h_outer=40,
        degree=3,
        geom_order=3,
    ),
    # The PML group. Mesh refinement does not bound absorber reflection: these
    # four vary the target reflection by two orders either way, the layer
    # thickness by 4×, and the profile order — each against the same baseline.
    _dolfinx(
        "pml",
        "pml-R6",
        PROBE_CASES,
        0.6,
        h_particle=12,
        h_outer=40,
        degree=3,
        pml_reflection=1e-6,
    ),
    _dolfinx(
        "pml",
        "pml-R10",
        PROBE_CASES,
        0.6,
        h_particle=12,
        h_outer=40,
        degree=3,
        pml_reflection=1e-10,
    ),
    _dolfinx(
        "pml",
        "pml-thin",
        PROBE_CASES,
        0.5,
        h_particle=12,
        h_outer=40,
        degree=3,
        r_pml=1000.0,
    ),
    _dolfinx(
        "pml",
        "pml-thick",
        PROBE_CASES,
        0.9,
        h_particle=12,
        h_outer=40,
        degree=3,
        r_pml=1600.0,
    ),
    _dolfinx(
        "pml",
        "pml-order3",
        PROBE_CASES,
        0.6,
        h_particle=12,
        h_outer=40,
        degree=3,
        pml_order=3,
    ),
]

# --- MEEP ------------------------------------------------------------------
#
# Cost unit as above: the measured resolution-25 runs are 142 s (TE) and 31 s
# (TM). FDTD in 2-D costs O(resolution³) — two spatial dimensions and the time
# step that the Courant condition ties to them — so the ladder's top end
# dominates the batch.

_MEEP_RES = [(15, 0.22), (20, 0.52), (25, 1.00), (32, 2.10), (40, 4.10), (50, 8.00)]

MEEP_JOBS = [
    # The main ladder, and the only group that can find MEEP's floor. Res 8 →
    # 25 improved the error ~9× for a 3.1× refinement, which is the second
    # order FDTD is supposed to have and means nothing has flattened yet.
    *[
        _meep(
            "res",
            f"res-{r}-{case}",
            case,
            cost * (1.0 if case == "te" else 0.22),
            resolution=r,
        )
        for r, cost in _MEEP_RES
        for case in ("te", "tm")
    ],
    # PML thickness, against the baseline's 5 a. Thin enough and a PML
    # reflects; the reflection is invisible to the resolution ladder because
    # refining the grid does not make the layer absorb any better.
    *[
        _meep(
            "pml",
            f"pml-{d}-{case}",
            case,
            1.0 * (1.0 if case == "te" else 0.22),
            resolution=25,
            dpml=d,
        )
        for d in (3.0, 8.0)
        for case in ("te", "tm")
    ],
    # Where the flux box sits. A placement systematic, not a discretisation
    # one: both boxes share the normalisation, so this cannot catch a
    # normalisation error — it catches a contour sitting inside the particle's
    # evanescent skirt, which would bias C_sca with no other symptom.
    *[
        _meep(
            "box",
            f"box-{r}-{case}",
            case,
            1.0 * (1.0 if case == "te" else 0.22),
            resolution=25,
            r_flux=r,
        )
        for r in (3.0, 5.0)
        for case in ("te", "tm")
    ],
    # How long the DFT accumulates. This is the dominant cost — the
    # normalisation run at resolution 25 stepped for some fifty cell-crossings
    # to reach the default 1e-11 — and it is also a convergence knob, because
    # truncating the transform early biases the recorded spectrum. 1e-9 says
    # whether the batch can be made cheaper; 1e-13 says whether the default was
    # already enough. **1e-13 is the one job with no bound on its run time.**
    *[
        _meep(
            "decay",
            f"decay-{tag}-{case}",
            case,
            budget * (1.0 if case == "te" else 0.22),
            resolution=25,
            dft_decay=tol,
        )
        for tag, tol, budget in (("1e-9", 1e-9, 0.7), ("1e-13", 1e-13, 2.0))
        for case in ("te", "tm")
    ],
]

JOBS = {"dolfinx": DOLFINX_JOBS, "meep": MEEP_JOBS}

COST_UNIT_S = 142.0
"""Seconds in one cost unit — the resolution-25 MEEP TE run measured alone."""


# --- comparisons -----------------------------------------------------------
#
# What gets differenced against what, and what the difference means. Writing
# these down rather than eyeballing the numbers afterwards is what stops the
# study from being read as "the last two levels look close".

COMPARISONS = [
    ("dolfinx", "mesh-L1", "mesh-L2", "element size"),
    ("dolfinx", "mesh-L2", "mesh-L3", "element size"),
    ("dolfinx", "mesh-L3", "mesh-L4", "element size — the candidate floor"),
    ("dolfinx", "mesh-L1", "degree-4", "basis, at fixed mesh"),
    ("dolfinx", "mesh-L1", "geom-3", "boundary geometry, at fixed mesh"),
    ("dolfinx", "mesh-L1", "pml-R6", "weaker absorber"),
    ("dolfinx", "mesh-L1", "pml-R10", "stronger absorber"),
    ("dolfinx", "mesh-L1", "pml-thin", "thinner absorber"),
    ("dolfinx", "mesh-L1", "pml-thick", "thicker absorber"),
    ("dolfinx", "mesh-L1", "pml-order3", "absorber profile order"),
    ("meep", "res-20", "res-25", "grid spacing"),
    ("meep", "res-25", "res-32", "grid spacing"),
    ("meep", "res-32", "res-40", "grid spacing"),
    ("meep", "res-40", "res-50", "grid spacing — the candidate floor"),
    ("meep", "res-25", "pml-3.0", "thinner absorber"),
    ("meep", "res-25", "pml-8.0", "thicker absorber"),
    ("meep", "res-25", "box-3.0", "flux contour nearer the particle"),
    ("meep", "res-25", "box-5.0", "flux contour further out"),
    ("meep", "res-25", "decay-1e-9", "DFT truncated earlier"),
    ("meep", "res-25", "decay-1e-13", "DFT accumulated longer"),
]

OBSERVABLES = ("c_ext", "c_sca", "c_abs")


def select(tool: str, groups: list[str] | None) -> list[Job]:
    """The jobs a run will cover, in cheapest-first order.

    Cheapest first so that an interrupted batch has produced the most
    comparisons it could have, and so a mistake in the matrix surfaces in the
    first minute rather than the last hour.
    """
    jobs = [j for t in JOBS for j in JOBS[t] if tool in ("all", t)]
    if groups:
        jobs = [j for j in jobs if j.group in groups]
    return sorted(jobs, key=lambda j: j.cost)


def run_job(job: Job) -> dict:
    """Run one job in its conda environment, logging to its own file."""
    env = f"pysie2d-{job.tool}"
    script = f"validation/{job.tool}/run_cylinder.py"
    log = OUT / "logs" / f"{job.tool}-{job.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(CONDA), "run", "-n", env, "python", script, *job.args]

    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    record = {
        "tool": job.tool,
        "group": job.group,
        "name": job.name,
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 1),
        "produced": [p.name for p in job.outputs if p.exists()],
    }
    with (OUT / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    status = "ok " if proc.returncode == 0 and job.done else "FAIL"
    print(
        f"[{status}] {job.tool:8s} {job.name:18s} {record['seconds']:8.1f} s"
        + ("" if proc.returncode == 0 else f"  → see {log}"),
        flush=True,
    )
    return record


def run(jobs: list[Job], force: bool) -> None:
    """Run the batch at ``WORKERS`` concurrency, skipping what is already done."""
    pending = [j for j in jobs if force or not j.done]
    skipped = len(jobs) - len(pending)
    if skipped:
        print(f"skipping {skipped} job(s) already complete", flush=True)
    if not pending:
        print("nothing to do", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        records = list(pool.map(run_job, pending))
    failed = [r["name"] for r in records if r["returncode"] != 0]
    print(f"\n{len(records)} job(s) in {(time.time() - t0) / 60:.1f} min", flush=True)
    if failed:
        print(f"failed: {', '.join(failed)}", flush=True)
        print("re-running the same command retries only these.", flush=True)


def estimate(jobs: list[Job]) -> None:
    """Print the matrix and what it is expected to cost."""
    total = sum(j.cost for j in jobs)
    todo = sum(j.cost for j in jobs if not j.done)
    print(f"{'tool':9s} {'group':8s} {'job':18s} {'est':>8s}  args")
    for job in jobs:
        mark = "·" if job.done else " "
        print(
            f"{mark}{job.tool:8s} {job.group:8s} {job.name:18s} "
            f"{job.cost * COST_UNIT_S / 60:7.1f}m  {' '.join(job.args[:6])}"
        )
    print(
        f"\n{len(jobs)} jobs, {total * COST_UNIT_S / 3600:.1f} h serial "
        f"({todo * COST_UNIT_S / 3600:.1f} h remaining, single-process)"
    )
    if WORKERS > 1:
        print(
            f"At {WORKERS} workers expect no better than "
            f"{todo * COST_UNIT_S / 3600 / WORKERS:.1f} h and plausibly the "
            "serial figure: both workloads are memory-bandwidth-bound, and "
            "the first batch overran per job by 2–5×."
        )
    print("(· = already complete. Every 'est' above is a serial measurement.)")


def _load(tool: str, name: str, case: str) -> dict | None:
    """One result, or None if that job has not run."""
    stem = name if tool == "meep" else f"{name}-{case}"
    path = OUT / tool / f"{stem}.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _drift(coarse: dict, fine: dict, key: str) -> float:
    """Worst pointwise change in one observable between two levels.

    Pointwise and worst-case, not a mean: the tolerance has to hold at every
    wavelength in the frozen file, and an average over a 51-point grid hides
    the one resonance where the two levels disagree.

    All three observables are normalised by the same scale — the finer level's
    peak ``C_ext`` — rather than each by its own. Two reasons, and both are
    about not producing a meaningless number. A *pointwise* relative change is
    arbitrarily large wherever a spectrum passes near zero, which ``C_sca``
    does at its minima. And ``C_abs`` is identically zero in the lossless
    cases, so normalising it by its own peak is 0/0; against the common scale
    it reads as what gate 3 actually asks — absorption leaking in, as a
    fraction of the signal that should carry it all.
    """
    a, b = coarse[key], fine[key]
    if a.shape != b.shape:
        return float("nan")
    scale = np.max(np.abs(fine["c_ext"]))
    return float(np.max(np.abs(a - b)) / scale)


def _nulls(tool: str) -> None:
    """Report gate 3's null test for every result on disk, per tool.

    The two tools need *different* quantities, and using the same one for both
    is how this gate came to pass vacuously. dolfinx's ``C_abs`` is a volume
    integral carrying ``Im(ε)``: at ``epsi = 0`` it is exactly zero by
    construction, so printing it says nothing that a broken solve could
    contradict. Its null test is the optical-theorem residual — two volume
    integrals against one flux integral over a different region — which is
    reported for every case, lossy included, because it is not a lossless
    identity. MEEP's ``C_abs`` is a flux tally that never sees the material,
    so there the zero reading *is* the evidence; it is absolute (nm) and not
    relative, because the quantity under test is zero.
    """
    # MEEP works in units of a = 100 nm and its .npz holds the raw arrays —
    # only freeze() and the driver's progress print multiply by NM_PER_A. The
    # scale below is what makes the "nm" on the next line true; without it
    # this reported gate 3's null a hundred times smaller than it is.
    per_tool = (
        ("dolfinx", "residual", "rel", 1.0),
        ("meep", "c_abs", "nm", 100.0),
    )
    for which, quantity, unit, scale in per_tool:
        if tool not in ("all", which):
            continue
        for path in sorted((OUT / which).glob("*.npz")):
            with np.load(path) as d:
                if quantity not in d.files:
                    continue
                worst = scale * float(np.max(np.abs(d[quantity])))
            print(f"{which:8s} {path.stem:26s} worst |{quantity}| = {worst:.2e} {unit}")
    print(
        "\nNull test, gate 3. dolfinx: the optical-theorem residual, because a\n"
        "lossless C_abs there is exact zero by construction and cannot fail.\n"
        "MEEP: C_abs itself, in nm — an independent flux tally that never\n"
        "reads the material, so a non-zero value is a real defect.\n"
    )


def analyse(tool: str) -> None:
    """Report every comparison that has both its levels on disk."""
    for which, coarse, fine, what in COMPARISONS:
        if tool not in ("all", which):
            continue
        cases = ("te", "tm") if which == "meep" else PROBE_CASES
        for case in cases:
            a = _load(which, f"{coarse}-{case}" if which == "meep" else coarse, case)
            b = _load(which, f"{fine}-{case}" if which == "meep" else fine, case)
            if a is None or b is None:
                continue
            drifts = "  ".join(
                f"{k}={_drift(a, b, k):.2e}" for k in OBSERVABLES if k in a and k in b
            )
            print(
                f"{which:8s} {case:12s} {coarse:12s} → {fine:12s}  {drifts}   [{what}]"
            )
    print(
        "\nDrifts are worst pointwise |Δ| over the grid, normalised by the "
        "spectrum's peak.\nThe frozen tolerance is the largest of these that "
        "survives at the production level —\nnot the mesh ladder's alone: a "
        "PML or truncation drift larger than the mesh drift\nmeans the mesh "
        "ladder found a floor that was never the limiting error."
    )
    print()
    _nulls(tool)


def main() -> None:
    """Parse the command line and either estimate, run, or analyse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("all", "meep", "dolfinx"), default="all")
    parser.add_argument(
        "--groups", default=None, help="comma-separated subset, e.g. 'mesh,res'"
    )
    parser.add_argument("--phase", choices=("study", "final"), default="study")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun completed jobs")
    args = parser.parse_args()

    if args.phase == "final":
        raise SystemExit(
            "--phase final is not a batch: it runs the full 51-point grid at "
            "the level the study named, which is not known until --analyse "
            "has been read. Invoke the driver directly with --grid full and "
            "that level's knobs."
        )

    jobs = select(args.tool, args.groups.split(",") if args.groups else None)
    if args.analyse:
        analyse(args.tool)
    elif args.dry_run:
        estimate(jobs)
    else:
        estimate(jobs)
        run(jobs, args.force)


if __name__ == "__main__":
    main()
