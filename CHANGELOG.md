# CHANGELOG

<!-- version list -->

## v0.4.0 (2026-08-08)

### Bug Fixes

- Ongoing Beyn integration
  ([`b9e4817`](https://github.com/claudio-sc/pysie2d/commit/b9e4817b816a65da0aeb37183752e7deab0b99c7))

- Preps for Beyn refinement
  ([`8291b9b`](https://github.com/claudio-sc/pysie2d/commit/8291b9b2380583e6a5d2502cfb507d60e7c3bb3c))

- Regenerate stale purcell map figure
  ([`e33c7dd`](https://github.com/claudio-sc/pysie2d/commit/e33c7ddb47baf8d32b113ff40ed2eaa52a19567d))

- Revert the uv.lock stamping that aborted the release
  ([`803e8fe`](https://github.com/claudio-sc/pysie2d/commit/803e8fe5497cc4450eb76463a285b2dd1c5b2067))

- Set public params for convergence diagnostics
  ([`fda1ba7`](https://github.com/claudio-sc/pysie2d/commit/fda1ba746cd3a58338d631d11c6a3f19177857a3))

- Update claude mem
  ([`956e90a`](https://github.com/claudio-sc/pysie2d/commit/956e90a2a195e442a96d75d453d1f339a686b49f))

### Build System

- Stamp uv.lock in the release commit
  ([`9073928`](https://github.com/claudio-sc/pysie2d/commit/90739281d487bebd07996aaba685522894001f1c))

### Documentation

- Add a QNM section and figures to the readme
  ([`33888a9`](https://github.com/claudio-sc/pysie2d/commit/33888a920d4720c4fbbf16b05004a2334a2f3e83))

- Add a user guide for quasi-normal modes
  ([`160c7bc`](https://github.com/claudio-sc/pysie2d/commit/160c7bc6d8b46baff207ec7217e9049ec067356b))

- Add performance doc and update status
  ([`fc9aaa6`](https://github.com/claudio-sc/pysie2d/commit/fc9aaa68a31792063d5e3d8ff33dc39cf02c5f7c))

- Add QNM spectrum and wide-window examples
  ([`1398814`](https://github.com/claudio-sc/pysie2d/commit/139881431da497e72cd368775dda5fccf8aa6f53))

- Add the Beyn port design record
  ([`000a9ca`](https://github.com/claudio-sc/pysie2d/commit/000a9ca6139c0d60e12732d1e035fa2611ccd2f8))

- Close the Beyn merge gate in the status record
  ([`75d6c2e`](https://github.com/claudio-sc/pysie2d/commit/75d6c2e952c3fb92445a813976f7d836e3187b0b))

- Correct roadmap to v0.4.0 and unpin readme figures
  ([`369d7ff`](https://github.com/claudio-sc/pysie2d/commit/369d7fff4cea444909dbd85dce54e7fbcea1fc10))

- Measure the large-nn memory and time walls
  ([`cf906b6`](https://github.com/claudio-sc/pysie2d/commit/cf906b6961545abb1d2953956140e6c178acfc03))

- Minor mods to claude md and clarifications in readme
  ([`1951dd5`](https://github.com/claudio-sc/pysie2d/commit/1951dd554392f4c40df3c2f23d3ad7f832ef4e7e))

- Move shipped perf notes to v0.3.0 and label the breaking block
  ([`ffcd46a`](https://github.com/claudio-sc/pysie2d/commit/ffcd46ad2e12316393dde42a1805e054520c1f7e))

- Record the lockfile rule in CLAUDE.md
  ([`af25078`](https://github.com/claudio-sc/pysie2d/commit/af25078bad9f1be0880d61481c46b6d52a324217))

- Update status and claude mem
  ([`684ecd0`](https://github.com/claudio-sc/pysie2d/commit/684ecd0717fbfeb2bcb79665868b87c137e3aae7))

### Features

- Expose vacuum wavelengths and cladding size parameter uniformly
  ([`43f7c53`](https://github.com/claudio-sc/pysie2d/commit/43f7c536149b7ef4606a0b8f040b260687cae9e9))

- Polish mode vectors alongside their wavelengths in refine()
  ([`c7fd28d`](https://github.com/claudio-sc/pysie2d/commit/c7fd28d04c49adddd69fb5d3a3ed4d1744bd3c2d))


## BREAKING CHANGES in v0.4.0 — wavelength and background-index conventions

Detail for the v0.4.0 entry above, where semantic-release's default template
lists the conventions change as an ordinary `Features` bullet: it derives the
version bump from the `feat!` marker but does not render a breaking-change
section into `CHANGELOG.md`. This section is that section — keep it directly
below the generated v0.4.0 block.

The package documented **vacuum** wavelengths but implemented `k = 2π/λ`, the
*medium* reading. The two coincide only at `n_clad = 1`, where the entire test
suite ran, so nothing caught it. Public wavelengths are now vacuum everywhere.

**Results change for any `n_clad ≠ 1`.** At `n_clad = 1.0` everything is
bit-identical — verified across `ei`, efficiencies, far field, `eval_field`,
`self_green`, `relative_ldos_map`, complex-λ assembly and QNM wavelengths.

Migration:

- **Public wavelengths are vacuum nm.** If you were passing medium wavelengths
  to work around the old behaviour, multiply by `n_clad`.
- **`Material.epsi` is now absolute** (referred to vacuum, like `n_core`). The
  relative permittivity is `(n_core² + i·epsi)/n_clad²`. If you were
  pre-dividing `epsi` by `n_clad²`, stop — it is now double-counted.
- **The low-level primitives take a background wavenumber, not a wavelength.**
  `Material.wnum_bg(λ_vac) = 2π·n_clad/λ_vac` is the single conversion point;
  no wavelength exists below the façade, so the factor cannot be applied twice.

  These four are **silent** breakages — same argument position, same type, no
  exception, wrong answer:

  | Symbol | Old third/second argument | New |
  |---|---|---|
  | `far_field` | `lambd` (~600) | `wnum_bg` (~0.0105) |
  | `plane_wave_rhs` | `lambd` | `wnum_bg` |
  | `line_dipole_rhs` | `wavelength` | `wnum_bg` |
  | `BIESolver.scatter(incident_rhs=…)` | callable `(nn, lambd, f, g)` | `(nn, wnum_bg, f, g)` |

  Loud renames: `eval_field` `wnum`→`wnum_bg`; `assemble_matrix` and
  `assemble_matrix_reference` `wn`→`wnum_bg`; `reference.mie.self_green_cylinder`
  `k`→`wnum_bg`.

- **`ScatterResult.wnum` is removed**, replaced by `ScatterResult.wnum_bg`
  (raises `AttributeError`, so this one is loud).
- **`reference.mie.qnm_wavelengths` now returns vacuum wavelengths** and takes
  `n_clad` (default 1.0, unchanged at the anchor).

### Added

- `pysie2d.size_parameter(geometry, material, wavelength)`, plus
  `ScatterResult.size_parameter` and `QNMResult.size_parameters`: the Mie size
  parameter `x = 2π·n_clad·rad/λ_vac`, referred to the cladding. Derived only —
  never an input — and circular geometry only.
- `Material.wnum_bg`, `Material.epsi_rel`, `Geometry.is_circle`.
- `tests/test_conventions.py`: the guard that was missing in both directions.
  Every test in it was verified to fail under the old convention by injection.

### Fixed

- `tests/test_efficiencies.py` divided `Material.nc` by `n_clad`, but `nc` is
  *already* the relative index — a double-count that passed only at
  `n_clad = 1`.
- `relative_ldos` is documented as relative to the homogeneous background, not
  to vacuum. The distinction was vacuous at `n_clad = 1` and is now
  load-bearing.


## v0.3.0 (2026-07-25)

### Bug Fixes

- Dispatch faster for non-qnm simulations
  ([`898344e`](https://github.com/claudio-sc/pysie2d/commit/898344e29497eca85356ff30a521fc60c0f91adf))

### Features

- Claude memory
  ([`14828eb`](https://github.com/claudio-sc/pysie2d/commit/14828eb7fa32efecf48e64e8c04b7fa5f1acaf07))

### Performance

- Cephes fast path for real-argument Hankel functions: `hank0`/`hank1`/`cbesh`
  use the `J_n + i·Y_n` identity when the argument is real, giving 5× faster
  `assemble_matrix` for non-absorbing particles and 1.8× for absorbing ones.
  Complex wavenumbers (QNM) are bit-identical and unchanged.

- Batch `relative_ldos_map`: one multi-RHS BLAS-3 `lu_solve` and one vectorised
  representation-formula evaluation per chunk instead of a per-point loop, 3.7×
  end to end (7.5× per valid point). NaN mask unchanged.


## v0.2.2 (2026-07-15)

### Bug Fixes

- Correct bibliography
  ([`006194c`](https://github.com/claudio-sc/pysie2d/commit/006194cce2ffbd43e0131084f954b315bfcd33dc))


## v0.2.1 (2026-07-09)

### Bug Fixes

- Abs github paths in readme + toml update
  ([`39c8611`](https://github.com/claudio-sc/pysie2d/commit/39c8611df6522ce60a6611d47caa98e299f9d0e9))


## v0.2.0 (2026-07-09)

- Initial Release
