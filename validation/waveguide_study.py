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

1. **Smoke test, low rungs, reciprocity.** A res-15 smoke test (``norm`` and
   ``left``, both polarisations), then the gate in :func:`smoke_gate`, which
   stops the batch on its own if it fails. Then the ladder at res 25, 32, 40
   (``norm``, ``left``) and the right drive at res 25 (``right``,
   ``norm_right``) for the first reciprocity check.
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
950 TM: the pulse's Gaussian tail, not the low-Q cluster, sets it).
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

THROUGHPUT = 2.5e8
"""px·step/s on one M5 performance core, for planning.

Measured: 2.9–4.3e8 in the res-8 shakedown (4.4e5 px), 2.3e8 in v1.0 at
2.9e6 px. Larger cells fall out of cache, so the planning value sits near the
large-cell measurement; the smoke gate replaces it with the real one."""

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
        height = 2.2 + 2 * k["pad"] + TOP + 2 * k["dpml_y"]
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
        smoke = [Job(c, d, 15, "smoke") for c in CASES for d in ("norm", "left")]
        r25 = [Job(c, d, 25, "ladder") for c in CASES for d in FOUR]
        rest = [
            Job(c, d, r, "ladder")
            for r in (32, 40)
            for c in CASES
            for d in ("norm", "left")
        ]
        return smoke + r25 + rest
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


def smoke_gate(records: list[dict]) -> list[str]:
    """Reasons to stop after the smoke test; empty when batch 1 may continue.

    Each threshold is one that a broken setup fails grossly and a working one
    passes with margin:

    * every run succeeded;
    * throughput and simulated time within 2× of the cost model, or the batch
      estimates are wrong by more than the plan can absorb;
    * MPB's n_eff within 1e-2 of pysie2d's: res 8 gave 1.4e-2 (TE) and 2.5e-2
      (TM) at second order, so res 15 should give ≲ 7e-3, while a swapped
      parity misses by 0.8 (2.85 vs 2.05);
    * |t_L|² error below res 8's (2.2e-2 TE, 1.6e-2 TM): the grid error must
      fall with refinement before larger runs spend hours on it.
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
    for case, t8 in (("te", 2.2e-2), ("tm", 1.6e-2)):
        c = compare_waveguide.compare(ref, _runs("smoke", case, 15), case)
        if c["n_eff"] > 1e-2:
            why.append(f"{case}: n_eff off by {c['n_eff']:.2e}")
        if c["|t_left|²"] > t8:
            why.append(f"{case}: |t_L|² error {c['|t_left|²']:.2e} ≥ res 8's {t8:.1e}")
    return why


def analyse() -> None:
    """Print every comparison that the runs on disk allow."""
    ref = ensure_reference()
    for group, resolutions in (("smoke", (15,)), ("ladder", (25, 32, 40, 50))):
        for res in resolutions:
            for case in CASES:
                runs = _runs(group, case, res)
                if runs["norm"] is None or runs["left"] is None:
                    continue
                for k, v in compare_waveguide.compare(ref, runs, case).items():
                    print(f"{group:6s} {case} r{res:<3d} {k:32s} {v:.3e}")


def main() -> None:
    """Run, dry-run or analyse."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch", type=int, choices=(1, 2, 3))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--analyse", action="store_true")
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
    for j in jobs:
        if j.group != "smoke" and not j.out.exists():
            run_job(j)
    print(f"=== batch {a.batch} done {time.strftime('%F %T')} ===", flush=True)


if __name__ == "__main__":
    main()
