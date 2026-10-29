"""v1.3: the MEEP waveguide study in three single-core batches.

Run from the repo root under uv; each job is a subprocess in the MEEP conda
environment, one at a time, one core::

    uv run python validation/waveguide_study.py --batch 1 --dry-run   # matrix, cost
    caffeinate -i uv run python validation/waveguide_study.py --batch 1
    uv run python validation/waveguide_study.py --analyse

Resumable as ``study.py`` is: a job whose ``.npz`` exists is skipped, so an
interrupted batch restarts with the same command. One line per job is appended
to ``_study/v13/manifest.jsonl`` with its wall time, steps and pixels, so the
cost model below is checked against what happened.

The batches
-----------
Ordered so that what is most likely wrong fails in the cheapest batch, and each
ends at a gate the owner reads before the next is launched.

1. **Smoke test, low rungs, reciprocity.** A smoke test at res 10 and 15
   (``norm`` and ``left``, both polarisations), then the gate in
   :func:`smoke_gate`, which stops the batch on its own if it fails. Then the
   ladder at res 20, 25, 30, 40 (``norm``, ``left``) and the right drive at
   res 25 (``right``, ``norm_right``) for the first reciprocity check.

Every resolution is a multiple of 5, so the 2.2 a slab is a whole number of
pixels and both interfaces sit on lattice points (``run_waveguide._cell``).
MEEP converges at **first order** here, measured on MPB's n_eff against the
closed-form slab, so :func:`analyse` reports each pair of rungs extrapolated
to res → ∞ (``compare_waveguide.richardson``) beside the raw differences.
2. **Production.** All four drives at res 50, both polarisations.
3. **Systematics at res 50.** Each group varies one knob to its *larger* value
   against batch 2's run, in the polarisation it threatens most: the cladding
   pad in TM (its cover tail is twice TE's), the x-PML and y-PML in TE (the
   y-PML was v1.0's TE floor, 3e-4). Monitor distance and DFT truncation cost
   no runs: every run records three monitor planes and a decay snapshot.

Cost model
----------
``seconds = pixels × steps / THROUGHPUT``. pixels = cell area × res²,
steps = 2·res·T (Courant 0.5), T measured by the res-8 shakedown (900 a/c TE,
950 TM: the pulse's Gaussian tail, not the low-Q cluster, sets it), and
confirmed by the first res-15 smoke runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation" / "_study" / "v13"
CONDA = Path.home() / "miniforge3" / "bin" / "conda"
DRIVER = "validation/meep/run_waveguide.py"
REF = OUT / "reference.npz"

sys.path.insert(0, str(ROOT / "validation"))
import compare_waveguide  # noqa: E402

THROUGHPUT = 3.0e8
"""px·step/s on one M5 performance core, for planning.

Measured: 2.9–4.3e8 in the res-8 shakedown (4.4e5 px), 3.3–4.3e8 in the first
res-15 smoke runs (1.6e6 px, 2026-10-29), 2.3e8 in v1.0 at 2.9e6 px. Res 50
is 1.7e7 px and further out of cache, so the planning value sits below the
small-cell measurements."""

T_SIM = {"te": 900.0, "tm": 950.0}
"""Simulated time, a/c, from the res-8 shakedown; resolution-independent."""

DEFAULTS = {"dpml_x": 20.0, "dpml_y": 15.0, "pad": 20.0}
TOP = 3.486
"""Highest point of the cluster, a (the triangle's z0 + max z)."""


@dataclass
class Job:
    """One MEEP run."""

    case: str
    drive: str
    res: int
    group: str
    knobs: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        """File stem, knobs included."""
        tag = "".join(f"-{k}{v:g}" for k, v in sorted(self.knobs.items()))
        return f"{self.case}-{self.drive}-r{self.res}{tag}"

    @property
    def out(self) -> Path:
        """Where the driver writes."""
        return OUT / self.group / f"{self.name}.npz"

    @property
    def area(self) -> float:
        """Cell area, a², with this job's knobs."""
        k = {**DEFAULTS, **self.knobs}
        width = 2 * (23.0 + 2.0 + k["dpml_x"])
        # The cell is symmetric about the slab top with an integer half-height.
        height = 2 * np.ceil(max(2.2, TOP) + k["pad"]) + 2 * k["dpml_y"]
        return width * height

    def seconds(self, throughput: float = THROUGHPUT) -> float:
        """Predicted wall time."""
        steps = 2 * self.res * T_SIM[self.case]
        return self.area * self.res**2 * steps / throughput

    def args(self) -> list[str]:
        """The driver's command line."""
        a = ["--case", self.case, "--drive", self.drive, "--resolution", str(self.res)]
        for k, v in self.knobs.items():
            a += [f"--{k.replace('_', '-')}", str(v)]
        return [*a, "--out", str(self.out)]


CASES = ("te", "tm")
FOUR = ("norm", "left", "right", "norm_right")


def batch(n: int) -> list[Job]:
    """The jobs of batch ``n``, in the order they run."""
    if n == 1:
        smoke = [
            Job(c, d, r, "smoke")
            for r in (10, 15)
            for c in CASES
            for d in ("norm", "left")
        ]
        rest = [
            Job(c, d, r, "ladder")
            for r in (20, 25, 30, 40)
            for c in CASES
            for d in (FOUR if r == 25 else ("norm", "left"))
        ]
        return smoke + rest
    if n == 2:
        return [Job(c, d, 50, "ladder") for c in CASES for d in FOUR]
    if n == 3:
        iso = [
            ("tm", {"pad": 30.0}),
            ("te", {"dpml_x": 30.0}),
            ("te", {"dpml_y": 25.0}),
        ]
        return [Job(c, d, 50, "iso", k) for c, k in iso for d in ("norm", "left")]
    raise ValueError(n)


def ensure_reference() -> dict:
    """pysie2d's side, computed once (~35 s) and reused by every gate."""
    if not REF.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "validation/waveguide_reference.py", "--out", str(REF)],
            cwd=ROOT,
            check=True,
        )
    return dict(np.load(REF))


def run_job(job: Job) -> dict:
    """Run one job in the MEEP environment, one core, logging to its own file."""
    job.out.parent.mkdir(parents=True, exist_ok=True)
    log = OUT / "logs" / f"{job.group}-{job.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(CONDA), "run", "-n", "pysie2d-meep", "python", DRIVER, *job.args()]
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
        )
    rec = {
        "name": job.name,
        "group": job.group,
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 1),
        "predicted": round(job.seconds(), 1),
    }
    if job.out.exists():
        d = np.load(job.out)
        rec |= {
            "timesteps": int(d["timesteps"]),
            "pixels": int(d["pixels"]),
            "t_stop": float(d["t_stop"]),
            "throughput": float(d["pixels"] * d["timesteps"] / d["seconds"]),
        }
    with (OUT / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    ok = proc.returncode == 0 and job.out.exists()
    print(
        f"[{'ok ' if ok else 'FAIL'}] {job.name:28s} {rec['seconds']:8.1f} s"
        f"  (predicted {rec['predicted']:.0f} s)" + ("" if ok else f"  → {log}"),
        flush=True,
    )
    return rec


def _runs(group: str, case: str, res: int) -> dict:
    return {
        d: compare_waveguide._load(OUT / group, case, d, res)
        for d in ("norm", "left", "right", "norm_right")
    }


LADDER = {
    10: "smoke",
    15: "smoke",
    20: "ladder",
    25: "ladder",
    30: "ladder",
    40: "ladder",
    50: "ladder",
}


def _observables(ref: dict, case: str, res: int) -> dict | None:
    runs = _runs(LADDER[res], case, res)
    if runs["norm"] is None or runs["left"] is None:
        return None
    return compare_waveguide.observables(runs, float(ref["plane_nm"]), case)


def smoke_gate(records: list[dict]) -> list[str]:
    """Reasons to stop after the smoke test; empty when batch 1 may continue.

    Each threshold is one that a broken setup fails grossly and a working one
    passes with margin:

    * every run succeeded;
    * throughput and simulated time within 2× of the cost model, or the batch
      estimates are wrong by more than the plan can absorb;
    * MPB's n_eff error *falls* from res 10 to 15 by a median factor in
      [1.2, 2.5]: first order gives 1.5, second 2.25, and the unaligned grid
      that failed this gate on 2026-10-29 gave 1.0;
    * the first-order extrapolation of n_eff from res 10 and 15 within 2e-3 of
      pysie2d: on the aligned cell MPB alone extrapolates to 1.1e-3 (TE) and
      < 1e-4 (TM) of the closed form, and a swapped parity misses by 0.8;
    * the |t_L|² error at res 15 below 2.5e-2, a gross bound only (res 8 gave
      1.6–2.2e-2). Whether it *falls* is not judged on two rungs: the particles'
      curved surfaces cannot be grid-aligned, and TM stepped unevenly,
      1.23e-2 → 1.20e-2 → 7.9e-3 → 7.0e-3 at res 10/15/20/30 (2026-10-29).
      The full ladder answers that, in :func:`analyse`.
    """
    bad = [r["name"] for r in records if r["returncode"] != 0 or "throughput" not in r]
    if bad:
        return [f"failed: {', '.join(bad)}"]
    why = []
    for r in records:
        if r["throughput"] < THROUGHPUT / 2:
            why.append(f"{r['name']}: throughput {r['throughput']:.2e} px·step/s")
        case = r["name"][:2]
        if not 0.5 < r["t_stop"] / T_SIM[case] < 2.0:
            why.append(f"{r['name']}: t_stop {r['t_stop']:.0f} a/c")
    ref = ensure_reference()
    for case in CASES:
        q10, q15 = _observables(ref, case, 10), _observables(ref, case, 15)
        n_ref = ref[f"{case}_n_eff"]
        rate = np.median(np.abs(q10["n_eff"] - n_ref) / np.abs(q15["n_eff"] - n_ref))
        if not 1.2 <= rate <= 2.5:
            why.append(f"{case}: n_eff error falls by {rate:.2f}× from res 10 to 15")
        q_inf = compare_waveguide.richardson(q10, 10, q15, 15)
        e_inf = np.max(np.abs(q_inf["n_eff"] - n_ref))
        if e_inf > 2e-3:
            why.append(f"{case}: extrapolated n_eff off by {e_inf:.2e}")
        e15 = compare_waveguide.errors(ref, q15, case)["|t_left|²"]
        if e15 > 2.5e-2:
            why.append(f"{case}: |t_L|² error {e15:.2e} at res 15")
    return why


def analyse() -> None:
    """Every comparison the runs on disk allow, raw and extrapolated in pairs."""
    ref = ensure_reference()
    for case in CASES:
        prev = None
        for res, group in LADDER.items():
            runs = _runs(group, case, res)
            if runs["norm"] is None or runs["left"] is None:
                continue
            for k, v in compare_waveguide.compare(ref, runs, case).items():
                print(f"{case} r{res:<3d}       {k:32s} {v:.3e}")
            q = compare_waveguide.observables(runs, float(ref["plane_nm"]), case)
            if prev is not None:
                q_inf = compare_waveguide.richardson(prev[1], prev[0], q, res)
                for k, v in compare_waveguide.errors(ref, q_inf, case).items():
                    print(f"{case} r{prev[0]}+{res:<3d} → ∞ {k:32s} {v:.3e}")
            prev = (res, q)


def main() -> None:
    """Run, dry-run or analyse."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch", type=int, choices=(1, 2, 3))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--stop-after-smoke", action="store_true")
    a = p.parse_args()
    if a.analyse:
        analyse()
        return
    jobs = batch(a.batch)
    if a.dry_run:
        for j in jobs:
            print(f"{j.group:7s} {j.name:28s} {j.seconds() / 60:7.1f} min")
        todo = sum(j.seconds() for j in jobs if not j.out.exists())
        print(f"batch {a.batch}: {len(jobs)} runs, {todo / 3600:.1f} h left, one core")
        return
    ensure_reference()
    print(f"=== batch {a.batch} start {time.strftime('%F %T')} ===", flush=True)
    smoke = [j for j in jobs if j.group == "smoke"]
    for j in smoke:
        if not j.out.exists():
            run_job(j)
    if smoke:
        # From the manifest, not this session's records: a batch resumed
        # half-way through the smoke test must still be gated on all of it.
        latest = {}
        for line in (OUT / "manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            latest[r["name"]] = r
        reasons = smoke_gate(
            [latest.get(j.name, {"name": j.name, "returncode": -1}) for j in smoke]
        )
        if reasons:
            print("SMOKE GATE FAILED — batch stopped:", *reasons, sep="\n  ")
            return
        print("smoke gate passed", flush=True)
        if a.stop_after_smoke:
            return
    for j in jobs:
        if j.group != "smoke" and not j.out.exists():
            run_job(j)
    print(f"=== batch {a.batch} done {time.strftime('%F %T')} ===", flush=True)


if __name__ == "__main__":
    main()
