# CHANGELOG

<!-- version list -->

## v0.3.0 (2026-07-25)

### Bug Fixes

- Dispatch faster for non-qnm simulations
  ([`898344e`](https://github.com/claudio-sc/pysie2d/commit/898344e29497eca85356ff30a521fc60c0f91adf))

### Features

- Claude memory
  ([`14828eb`](https://github.com/claudio-sc/pysie2d/commit/14828eb7fa32efecf48e64e8c04b7fa5f1acaf07))


## Unreleased

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
