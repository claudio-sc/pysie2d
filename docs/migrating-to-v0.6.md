# Migrating from v0.5 to v0.6

v0.5 converged at first order in `nn`; v0.6 converges spectrally, so **every
number changes**, for the better, typically by many orders of magnitude at the
same `nn`. Three calls change:

```python
# v0.5
geom = Geometry.gielis(rad=200, n_pts=200, m=4, b=1.2)  # uniform arc length
wider = Geometry.gielis(rad=200, n_pts=200, m=4, b=1.3,
                        theta=geom.theta)  # frozen nodes

# v0.6
geom = Geometry.gielis(rad=200, n_pts=200, m=4, b=1.2)  # uniform θ (new default)
wider = Geometry.gielis(rad=200, n_pts=200, m=4, b=1.3,
                        parametrisation=geom.parametrisation)
```

- **`theta=` is now `parametrisation=`.** The new quadrature needs the node
  map's first and second derivatives at every node, and those cannot be
  recovered from an array of angles. Accepting a bare array would return a
  plausible wrong answer, so `theta=` is gone, and an angle array passed as
  `parametrisation=` raises `TypeError` with the migration in the message.
- **Nodes are equispaced in θ by default**, not in arc length: under the new
  quadrature uniform θ converges faster on every star and superellipse measured
  within the solver's validity envelope, at low resolution as well as high. For
  uniform arc length pass `parametrisation=Parametrisation.gielis(...)`
  ([conventions](conventions.md) §13).
- **`Geometry(...)` built from your own arrays** takes derivatives with respect
  to the quadrature parameter `t` and no longer takes `delt`; `assemble_matrix`,
  `assemble_matrix_dwn` and `assemble_matrix_reference` no longer take `delt`
  either. Odd `m` now requires `a == b` and `n2 == n3`, since the boundary
  otherwise does not close.
- **`n_pts` defaults to 100**, was 200. The circle anchors reach round-off by
  `nn ≈ 30–40` under the new quadrature, so the old default was sized for a
  first-order scheme that no longer exists. Pass `n_pts` explicitly to keep a
  v0.5 resolution.

`richardson_limit` still exists but is deprecated: nothing in the package
converges at first order any more, and extrapolating a spectrally converged
pair makes it worse.
