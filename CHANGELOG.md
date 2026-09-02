# CHANGELOG

<!-- version list -->

## v0.5.0 (2026-09-02)

### Bug Fixes

- **geometry**: Make theta optional so v0.5 stays non-breaking
  ([`d448921`](https://github.com/claudio-sc/pysie2d/commit/d448921118bbec7b4f672dcd013d66cd2edecac5))

- **geometry**: Raise on coincident arc-length nodes
  ([`ed6a05b`](https://github.com/claudio-sc/pysie2d/commit/ed6a05bbdf44b2d3097f77a09bb9e027ed955c8c))

- **scatter**: Stop double-counting the far-field grid's closing angle
  ([`dc9da93`](https://github.com/claudio-sc/pysie2d/commit/dc9da93853cfb41151a3923d3ea0f3880b88a350))

### Build System

- Move changelog_file to default_templates
  ([`5e4d5d3`](https://github.com/claudio-sc/pysie2d/commit/5e4d5d34e8ad56a8b7c7d31710eeccad7c5a3da7))

- Pin uv_build to the uv version actually used
  ([`b8a2e3e`](https://github.com/claudio-sc/pysie2d/commit/b8a2e3e994527187c9aeffc141362deed90e77f8))

- Re-lock after the v0.4.2 release
  ([`c3e5dc5`](https://github.com/claudio-sc/pysie2d/commit/c3e5dc5005f2907ab9547df8427e3f6055719902))

### Documentation

- A17 - the section 12 extrapolation holds at aspect 3
  ([`b6c5f05`](https://github.com/claudio-sc/pysie2d/commit/b6c5f05dd358bd25c7b3c763eaf2e45fabdef0b1))

- Add CI, PyPI, license and Python badges
  ([`4477600`](https://github.com/claudio-sc/pysie2d/commit/44776001364264b2300a9da38a50556a99439679))

- Attribute the formulation to Maradudin and Valencia
  ([`09e0a97`](https://github.com/claudio-sc/pysie2d/commit/09e0a97fa6fb1903e673a79e42a6586e69ea352f))

- Conventions section 11 — adjoint sensitivity API
  ([`26c6068`](https://github.com/claudio-sc/pysie2d/commit/26c6068d84ff98c0f29da27fc196455478e767a3))

- Correct stale claims after the v0.4.2 release
  ([`7f83b32`](https://github.com/claudio-sc/pysie2d/commit/7f83b32e3bbfed59437f2def984c4070ba44c083))

- Describe the solver as surface-integral, not Müller BIE
  ([`dee6f9e`](https://github.com/claudio-sc/pysie2d/commit/dee6f9ee4b374a553b4737f01fff0d84e0b53d69))

- Fix the conventions link that breaks on PyPI
  ([`e6fbedf`](https://github.com/claudio-sc/pysie2d/commit/e6fbedfd890f663a644bf68bcbef4dd5c2cac9e0))

- Frame docs/design as engineering history
  ([`ab73de9`](https://github.com/claudio-sc/pysie2d/commit/ab73de923d751b3afdafe0b190378e2eed50bfba))

- Measure J convergence in R for gate 10
  ([`4bf89b1`](https://github.com/claudio-sc/pysie2d/commit/4bf89b17dc31340129c5644ff2a4d8a2c5f92385))

- Measure J-difference convergence for gate 10 route 2
  ([`4b0669c`](https://github.com/claudio-sc/pysie2d/commit/4b0669c3ddbc0592c0c2a9e24d4f06ea9ccd298a))

- Measure the frozen-node h ladder
  ([`bfbc513`](https://github.com/claudio-sc/pysie2d/commit/bfbc51303a628f3a8cbdb2454d22abf651db0375))

- Measure the R band cost across the shape family
  ([`2ef4e92`](https://github.com/claudio-sc/pysie2d/commit/2ef4e92487183614fd0fd1d5a649b85b638861e0))

- Measure the shape-derivative h window at production n_pts
  ([`6435266`](https://github.com/claudio-sc/pysie2d/commit/6435266a20ae28c99cfaafa29e2f68d8cbbca124))

- Measure two-rung Richardson extrapolation of J
  ([`612f9ca`](https://github.com/claudio-sc/pysie2d/commit/612f9ca68e03f066e446517b9e86e9b12cf6ea36))

- Note the Hankel refinement question as unproven
  ([`f137f80`](https://github.com/claudio-sc/pysie2d/commit/f137f802f89057910c25b7ff40c59a1ac3a39ebc))

- Pin the frozen node set as conventions section 10
  ([`d46c382`](https://github.com/claudio-sc/pysie2d/commit/d46c3822e489f70b860c99f6e3a2741a9870009f))

- Pin the push, merge and branching policy
  ([`982e24e`](https://github.com/claudio-sc/pysie2d/commit/982e24ee23891ba1915bc4f61b5deeaf5bb906d0))

- README section for the v0.5 sensitivity API
  ([`1c32bc8`](https://github.com/claudio-sc/pysie2d/commit/1c32bc85bdfbaee7b904254977d976039acb3bbc))

- Record the measured contour threading speedup
  ([`da4469d`](https://github.com/claudio-sc/pysie2d/commit/da4469d950a7eb38c9e87327069c661a56f60428))

- Retire the unreproducible gate 9 reference ladder
  ([`e9203cb`](https://github.com/claudio-sc/pysie2d/commit/e9203cbeccad569a825dd69754ac9778d5096c63))

- Separate unreleased v0.5 work from the shipped roadmap
  ([`11e582b`](https://github.com/claudio-sc/pysie2d/commit/11e582b3ef36cf7d9d30177e155414a5f5834ef8))

- Split the h window by parameter class
  ([`bb63e7b`](https://github.com/claudio-sc/pysie2d/commit/bb63e7bd36402750542bc4da28d88700c99b70b7))

- Warn that arc_length=False ignores a and b
  ([`564558c`](https://github.com/claudio-sc/pysie2d/commit/564558c68b06fdf792ae32e5f00090f57ccee152))

### Features

- **geometry**: Accept a prescribed theta node set
  ([`247492d`](https://github.com/claudio-sc/pysie2d/commit/247492d0eaf834ed496871c1605922078b74c20a))

- **qnm**: Adjoint eigenvalue sensitivity for simple poles
  ([`267338a`](https://github.com/claudio-sc/pysie2d/commit/267338af8d7dd63890046bbcc21a8e211c5ec155))

- **qnm**: Degenerate secular branch in sensitivity
  ([`6fcb0ed`](https://github.com/claudio-sc/pysie2d/commit/6fcb0ed72bc7c892c7358c9d85d98c9a75e992ff))

- **qnm**: Left null vector for the adjoint quotient
  ([`703d9de`](https://github.com/claudio-sc/pysie2d/commit/703d9def90c07a6d87479d00471f4c829abe2f64))

- **qnm**: Richardson_limit and the gate 10 accuracy policy
  ([`c42e51b`](https://github.com/claudio-sc/pysie2d/commit/c42e51b8ce82fbfc5d7f098547bcedb727830ee1))

- **solver**: Wavelength_over_ds discretization diagnostic
  ([`967f4c3`](https://github.com/claudio-sc/pysie2d/commit/967f4c3e18f9159acc7216853f452e86f9b739aa))

### Performance Improvements

- Thread contour_moments over the contour nodes
  ([`318421c`](https://github.com/claudio-sc/pysie2d/commit/318421c02d08b65832cdad4a521a4d076e292977))


## v0.4.2 (2026-08-09)

### Bug Fixes

- Narrow the cusp claim and the size-parameter wording
  ([`a43ee6e`](https://github.com/claudio-sc/pysie2d/commit/a43ee6e92a9e582a930b78dd058102e423094fbc))

- Test a genuinely non-circular shape, and pin the c1 limit
  ([`7ca21d6`](https://github.com/claudio-sc/pysie2d/commit/7ca21d6c3fc3725a8eebf891e6d9316034ac79e5))

### Documentation

- Record scale covariance as a convention
  ([`6479956`](https://github.com/claudio-sc/pysie2d/commit/647995642d73313a3e56a2a4c83095ccafac0f59))


## v0.4.1 (2026-08-08)

### Bug Fixes

- Push release tags with a token that triggers workflows
  ([`2de8ca9`](https://github.com/claudio-sc/pysie2d/commit/2de8ca91351c1402a8c77ac4da52f1eea1768d50))


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
