# pysie2d

A 2-D surface-integral-equation (Müller BIE) solver for time-harmonic EM
scattering from a single smooth cylinder in a homogeneous background. Public,
MIT, CI-gated, validated against analytic Mie theory.

## Read first

[docs/conventions.md](docs/conventions.md). Every gotcha in this solver traces
back to a convention listed there — read it before reading code. When a change
pins a *new* convention (a sign, a normalisation, a layout), record it there in
the same change.

## Commands

```bash
uv sync                       # environment
uv run pytest                 # validation suite
uv run ruff format --check .  # formatting
uv run ruff check .           # lint
uv run python examples/convergence_study.py   # regenerate figures
uv run python examples/nearfield_map.py
uv run python examples/purcell_map.py
```

Python 3.12+, uv, numpy + scipy only. **Adding a runtime dependency is a scope
decision** — raise it, don't just add it.

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
3. **New physics needs an independent validation anchor.** A closed form, an
   analytic limit, or a second method — not agreement between two paths in this
   repo, which only proves they share assumptions. The package's stated scope is
   "the part that can be validated end-to-end against a closed-form reference";
   respect that boundary.
4. **Justify every tolerance.** Each `rtol`/`atol` in a test carries a reason in
   a comment: a convergence order, a quadrature floor, a precision bound.
   Near-field quantities converge at first order in `nn` (hence `nn = 1000` for
   1 % on the self-Green anchor); far-field efficiencies are fine at `nn = 300`.
   Never widen a tolerance to make a test pass.

## Style

- Google-style docstrings with `Args`/`Returns`/`Raises`; ruff `D`, `N`, `E`,
  `F`, `B`, `I`, `C4`, `A` are on, line length 88. Per-file ignores exist for
  math notation (`reference/mie.py` keeps `J_n`/`H_n`) — extend that list rather
  than renaming physics.
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

The three `examples/` scripts generate the README figures, and they are part of
the package's public face. **Load the `dataviz` skill before writing or changing
any plotting code** — take its color formula, accessibility checks, and mark
specs; ignore the dashboard/KPI material, which does not apply here.

Conventions specific to these figures:

- `purcell_map` is **diverging around a physically meaningful midpoint**
  (`relative_ldos = 1`, the free-space value). Enhancement and suppression must
  be visually symmetric about it, and the midpoint must be pinned — an
  auto-scaled diverging colormap that centres on the data mean is wrong here.
- `nearfield_map` is sequential magnitude data; use a perceptually uniform
  colormap, not `jet`.
- `convergence_study` is log-log error vs `nn` for both polarisations. If a
  convergence *order* is being claimed, show the reference slope.
- Masked regions (the NaNs `relative_ldos_map` returns inside and near the
  particle) must read as "no data", visually distinct from a low value.
- Axes carry units (nm). Both polarisations should be distinguishable without
  relying on colour alone.
- `MPLBACKEND=Agg` in CI; figures must render headless.

## Review

`/code-review` for the working diff, `/review` for a GitHub PR, and the
`sci-code-reviewer` agent for the convention/validation/tolerance pass that
generic review does not do. `/security-review` is not useful here — this is a
numerics library with no untrusted input, network, or auth.

`ponytail-review` and `/simplify` are fine on a diff but must not be pointed at
the whole repo: `assemble_matrix_reference` and `reference/mie.py` are
deliberate second implementations kept as validation anchors, and a
minimalism pass will read them as duplication.

## Performance shape

The one structural win is **factorise once, solve many** where `M(λ)` is
independent of the right-hand side: `relative_ldos_map` LU-factorises once and
reuses across all source positions. The mirror trap is "optimising" a loop where
`M` genuinely changes each iteration — a wavelength sweep has nothing to reuse.
Everything else is a serial `for` loop by design; a single wavelength at
`nn = 300` is sub-second.

## Roadmap

v0.1 core scattering → v0.2 line dipole, self-Green, LDOS/Purcell → **v0.3 QNM
extraction via Beyn's contour method, validated against analytic Mie
resonances.** Longer term: slab-waveguide backgrounds, multiple particles.

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
