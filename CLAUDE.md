# pysie2d

## Read first

[docs/conventions.md](docs/conventions.md). Every gotcha in this solver traces
back to a convention listed there — read it before reading code. When a change
pins a *new* convention (a sign, a normalisation, a layout), record it there in
the same change.

## Dependencies

Python 3.12+, uv, numpy + scipy only. **Adding a runtime dependency is a scope
decision** — raise it, don't just add it.

**Touching `[project].dependencies` means running `uv lock` in the same commit.**
Both workflows install with `uv sync --frozen`, which installs the locked
resolution *without* checking it against `pyproject.toml`, so a stale lockfile
stays CI-green and bites someone later.

`uv sync --locked` would assert it instead, but it cannot be switched on until
the release commit stamps `uv.lock`: semantic-release does not, so after every
bump the lockfile pins the previous version and `--locked` would fail on main.
Two fixes that do *not* work, both tried: `build_command = "uv lock"` runs
inside the semantic-release action's container, which has no `uv` (exit 127,
release aborted), and `version_variables = ["uv.lock:version"]` rewrites every
dependency's version, not just the root package's. Re-lock by running `uv lock`
after a release rather than editing that version line by hand.

## Non-negotiables

1. **Complex wavenumbers work everywhere.** Every assembly and evaluation path
   accepts a complex `k`; that is what makes QNM extraction possible. Real-input
   fast paths are fine *only* with the complex fallback intact
   (`kernels._real_if_real`, the `np.iscomplexobj` branch in `hank0`/`hank1`).
   Never simplify a path to real-only arithmetic, however dead the complex
   branch looks.
2. **Conventions are fixed.** `pol = 2` → TE (`E_y`, Mie `b_n`, `Q_*_TE`);
   `pol = 1` → TM (`H_y`, `a_n`, `Q_*_TM`). Lengths in nm. Time convention
   `exp(-iωt)`, outgoing waves `H_n^{(1)}`. `ei[:nn]` = φ, `ei[nn:]` = χ.
   `Geometry.g` is a z-coordinate array, **not** a Green function. Self-Green
   sign `SIGN = -1`; `relative_ldos = 1 + 4·Im S`.

   **Wavelengths are vacuum, wavenumbers are background** (conventions §2).
   Public methods take `wavelength` = λ_vac in nm; low-level primitives take
   `wnum_bg = 2π·n_clad/λ_vac` and **no wavelength at all**, so the conversion
   — `Material.wnum_bg` — happens exactly once per call path. Never give a
   primitive a wavelength parameter back: façade methods call each other, and
   two conversion points means `n_clad` applied twice. `Material.epsi` is
   **absolute**; `nc` and `eps` are background-relative. Size parameter
   `x = 2π·n_clad·rad/λ_vac` is derived-only and circle-only.

   **QNM half-plane** (conventions §8): `Im λ > 0` for decaying modes and
   `Re λ > 0` to keep Hankel arguments off the `H^{(1)}` branch cut — both
   asserted, because holomorphy is the premise of the contour argument.
   `Q = Re λ / (2 Im λ)`. Poles do **not** come in conjugate pairs; the reality
   condition is `λ → −λ̄`.

   **Scale covariance** (conventions §9): `M` sees only `k_bg·rad`, so
   `M(s·rad, s·λ) = M(rad, λ)` entrywise and `λ(s·rad) = s·λ(rad)`,
   `dQ/drad = 0`, exactly. It rests on the material being non-dispersive and on
   nothing in the geometry pipeline carrying an absolute length — `n_fine` in
   the arc-length inversion is the one to watch.
3. **New physics needs an independent validation anchor.** A closed form, an
   analytic limit, or a second method — not agreement between two paths in this
   repo, which only proves they share assumptions. The package's stated scope is
   "the part that can be validated end-to-end against a closed-form reference";
   respect that boundary.
4. **Justify every tolerance.** Each `rtol`/`atol` in a test carries a reason in
   a comment: a convergence order, a quadrature floor, a precision bound.
   Under Kress quadrature near- and far-field quantities converge spectrally:
   the circle anchors reach round-off by `nn ≈ 30–40`, so a tolerance cites a
   measured value against a round-off or contour floor, not a convergence order.
   Never widen a tolerance to make a test pass.

## Style

- Google-style docstrings with `Args`/`Returns`/`Raises`. Per-file ruff ignores
  exist for math notation (`reference/mie.py` keeps `J_n`/`H_n`) — extend that
  list rather than renaming physics.
- Unicode in docstrings and comments is welcome here (φ, χ, λ, `H₀^{(1)}`); it
  makes the formulation readable against the cited papers.
- Comments explain *why the physics or numerics demands this*, not what the line
  does. The existing comments are the model: they cite equations, name the trap
  being avoided, and give the number that justified a choice.
- Public API is typed and exported from `__init__.py`. Tests are the
  documentation of behaviour (`D` rules are off in `tests/`) — their names and
  comments should state what physical property is being checked and why it
  cannot pass by accident.
- Commits use conventional-commit form; semantic-release parses them to compute
  the version and build `CHANGELOG.md`.

## Figures

The `examples/` scripts generate the README figures. Their conventions
live in [examples/CLAUDE.md](examples/CLAUDE.md).

## Git workflow

The rule behind all of it: **pushing a branch publishes nothing; merging to
`main` publishes to PyPI.** `version-release.yml` fires on `push: branches:
[main]`, runs semantic-release, tags, and `release.yml` publishes on the `v*`
tag. So branch work is free and merging is a release decision.

**Push — decide and do it, often.** Push a feature branch after the final commit
of a working session, and mid-session whenever a unit stands on its own. Use
`git push -u` on the first push so the branch tracks. The one push to ask about
is a **force-push to a branch that already exists on the remote**: it is the
only one that can destroy work that is not yours.

**Open a PR early — decide.** `ci.yml` runs on `pull_request` and on pushes to
`main`, and on nothing else, so **a branch with no PR gets no CI**. Open one
(draft is fine) as soon as the branch holds a commit worth checking, not at the
end. Short body, no signature.

**Merge to `main` — ask, every time.** Any `feat:` or `fix:` on the branch bumps
the version and ships it to PyPI. Consequences that follow from that:

- Merge on **roadmap milestones, not on sessions or units**. v0.5 is threading
  *and* the adjoint sensitivity API; merging half of it ships a partial API
  under a version number that claims the whole one.
- **Merge, do not squash.** `commit_parser_options = { ignore_merge_commits =
  true }` means semantic-release reads the individual conventional commits;
  squashing collapses them into one subject and the changelog loses the rest.
- **Run `uv lock` and commit it after every merge that releases**, per
  *Dependencies* above — semantic-release does not stamp the lockfile, so `main`
  is otherwise left pinning the previous version.

**The one merge to decide alone: docs-only.** A branch whose every commit is
`docs:`, `chore:`, `ci:` or `test:` triggers no bump and no publish under the
conventional parser, so merging it is silent and reversible. If a single commit
is `feat:` or `fix:`, or carries a `BREAKING CHANGE` footer, it is a release —
ask.

**New branch — decide.** One branch per roadmap milestone, named for it
(`v0.5-sensitivity`), *not* one per unit: the units are sequential and
dependent, so a branch each buys no isolation and costs merge traffic. Open a
new one when the previous milestone branch has merged, when the work belongs to
a different milestone or gate, or for a spike that may be thrown away — name
those `spike-<topic>` and expect to read and abandon them rather than merge.
Never commit to `main` directly except a trivial docs fix.

## Review

`/code-review` for the working diff or a PR — it takes a PR number, branch or
path as its target — and the `sci-code-reviewer` agent (defined in
`.claude/agents/`) for the convention/validation/tolerance pass that generic
review does not do. `/security-review` is not useful here — this is a numerics
library with no untrusted input, network, or auth.

`/simplify` is fine on a diff but must not be pointed at the whole repo:
`assemble_matrix_reference` and `reference/mie.py` are deliberate second
implementations kept as validation anchors, and a minimalism pass will read
them as duplication.

## Performance shape

**This is a special-function-bound code.** At complex λ the `jv`+`hankel1`
pairs of the Kress splitting are 99 % of one assembly (v0.5: `hankel1` alone,
98 % of an assembly and 95 % of a whole `QNMSolver.modes()` call); dense linear
algebra is ~2 %. Optimise anything else and you are optimising 2 % of the
runtime. Both **release the GIL**, so
threading a loop of assemblies is a real 5× and `multiprocessing` is strictly
worse. Numbers, and the rejected alternatives, in
[docs/design/performance.md](docs/design/performance.md) — read it before
optimising.

The one structural win on the driven side is **factorise once, solve many** where
`M(λ)` is independent of the right-hand side: `relative_ldos_map` LU-factorises
once and reuses across all source positions. The mirror trap is "optimising" a
loop where `M` genuinely changes each iteration — a wavelength sweep has nothing
to reuse, and the win there is concurrency, not reuse. A single wavelength at
`nn = 300` is sub-second.

## Roadmap

Shipped: v0.1 core scattering → v0.2 line dipole, self-Green, LDOS/Purcell →
v0.3 performance (Cephes Hankel fast path, batched factorise-once
`relative_ldos_map`) → v0.4 vacuum wavelength conventions (breaking) + QNM
extraction via Beyn's contour method, validated against analytic Mie resonances
→ v0.4.2 scale covariance (conventions §9)
→ v0.5 threaded `contour_moments` ([performance](docs/design/performance.md)
§3.1, the 5.02× measurement and its two traps) + the adjoint
eigenvalue-sensitivity API (conventions §11).

**v0.6, on `v0.6-quadrature`: spectral convergence.** Kress–Martensen product
quadrature replacing the Maradudin diagonal self-patch, a smooth
`Parametrisation` node map with uniform θ as the default, and the breaking
`theta=` → `parametrisation=` change. First order → machine precision at
`nn ≈ 30` on the circle. Implemented, with README, QNM guide and figures
updated; the curvature-adaptive density ships as a non-default map with known
limits (no flat points). Decisions in
[docs/design/v0.6-architecture.md](docs/design/v0.6-architecture.md), the
study in
[docs/design/studies/quadrature-study-plan.md](docs/design/studies/quadrature-study-plan.md),
the invariants in `docs/conventions.md` §13.

**v0.7 — multipole decomposition of the scattered field.** Shipped; the
spec is [multipole-spec.md](docs/design/multipole-spec.md). Cylindrical-
harmonic expansion of the exterior field about the particle centre, ported from
the pre-v0.6 research code, anchored on analytic Mie in both polarisations. It
is purely additive — no signature on the existing surface changes — which is
why it ships ahead of the slower external-validation work.

**v0.8, on `v0.8-multiparticle`: finite clusters of arbitrary particles.**
Code-spec: [multiparticle-spec.md](docs/design/multiparticle-spec.md); the
earlier high-level guide it supersedes section by section is
[multiparticle-handoff.md](docs/design/multiparticle-handoff.md). Coupled
assembly and dense solve over different shapes, sizes, materials and `nn` per
particle, anchored on a two-cylinder addition-theorem reference. **Additive**
(spec D1): `Cluster`, `ClusterBIESolver` and `ClusterScatterResult` land
*beside* `BIESolver`/`ScatterResult`, whose signatures do not change — the
one API-breaking design the handoff assumed was rejected.

**v0.9 — external validation for non-circular shapes.** Scoping:
[v0.9-external-validation.md](docs/design/v0.9-external-validation.md). Every
anchor in the repo is circular; this is the first independent check of a Gielis
star, against MEEP and dolfinx. Renumbered twice, from v0.7 and then from v0.8
(spec D11), on the same rule both times — the external tools are conda-first and
CI will never run them, so a finished additive feature is not held behind them.

Multiple particles *was* the last API-breaking item and therefore a gate on
v1.0; under D1 it is additive, so **it no longer gates v1.0**. The existing
surface provably cannot change, which leaves **v0.9 external validation as the
sole v1.0 gate**: a 1.0 promises a stable surface *and* answers that have been
checked against something that is neither circular nor ours. Longer term:
slab-waveguide backgrounds.

## How to work here

- **Think before coding.** State assumptions explicitly; if a request admits
  several readings, present them rather than silently picking one. Recommend the
  simpler path when one exists. Ask when something is genuinely unclear — a
  short question beforehand beats a wrong implementation.
- **Write the minimum that solves the stated problem.** No speculative features,
  no abstraction for something used once, no configurability nobody asked for,
  no error handling for impossible states.
- **Make surgical changes.** Every changed line must trace back to the request.
  Don't improve adjacent code, reformat, or refactor what isn't broken; match
  the file's existing style. Remove orphans *your* change created; if you spot a
  pre-existing problem, mention it — don't fix it unasked.
- **Define the check before the work.** "Add X" → the test that proves X. "Fix
  the bug" → the failing test that reproduces it. For multi-step work, state the
  plan with a verification for each step, and run them at the end.
