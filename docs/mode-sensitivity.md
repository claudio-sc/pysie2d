# Mode sensitivity

`QNMResult.sensitivity(at)` returns `dλ/dp` for every mode from the adjoint
quotient `− uᴴ(∂M/∂p)v / uᴴ(∂M/∂λ)v`, re-extracting no eigenvalue. `at` is a
callable `δ → (geometry, material)`, so a shape parameter and a refractive index
go through one signature and one code path. Read
[qnm-guide.md](qnm-guide.md) first; this page assumes a trustworthy
`QNMResult`.

From [`examples/mode_sensitivity.py`](../examples/mode_sensitivity.py):

```python
res = QNMSolver(geom, mat).modes(745 + 2j, 775 + 15j).refine()
frozen = res.geometry.parametrisation  # the node map must be frozen


def wider(delta):
    """The same circle with one semi-axis b = 1 + delta, on the frozen nodes."""
    return Geometry.gielis(
        rad=200, n_pts=200, m=4, b=1.0 + delta, parametrisation=frozen
    ), mat


print(res.sensitivity(wider))  # ≈ λ/2 for both partners
```

On the circle, stretching one axis is half of a uniform dilation, so each
partner moves at `λ/2`.

- **The node map is frozen.** The geometry `at` returns must be built on
  `res.geometry.parametrisation`: a shape derivative holds the node map fixed,
  and `sensitivity` checks the nodes and both map derivatives exactly and raises
  otherwise ([conventions](conventions.md) §10).
- **Degenerate poles** dispatch to a secular problem rather than raising
  ([conventions](conventions.md) §11).
- **No extrapolation is needed.** `dλ/dp` converges spectrally in `n_pts`, like
  λ itself ([conventions](conventions.md) §12, with the measured anchors).
- **Free-space only.** Over a `HalfSpace` it raises `NotImplementedError`, and
  `QNMSolver` does not take a `Stack`. For guided outputs over a stack,
  `mode_sensitivity` gives a central difference instead
  ([layered-backgrounds.md](layered-backgrounds.md)).
