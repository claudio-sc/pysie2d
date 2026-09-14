# docs/design — engineering history

These are **working documents, not user documentation**. They record how a
feature was designed, what was measured, and which decisions were reversed and
why. They are kept because the *reasoning* is expensive to reconstruct — not
because they describe the current tree. Where a number or a status line here
disagrees with the repo, the repo wins.

For using the package, read [README.md](../../README.md),
[docs/conventions.md](../conventions.md) and [docs/qnm-guide.md](../qnm-guide.md).
The live list of what is shipped and what is next is the roadmap in
[CLAUDE.md](../../CLAUDE.md).

| Document | What it is | How to treat it |
|---|---|---|
| [pysie2d-quadrature-handoff.md](pysie2d-quadrature-handoff.md) | The September 2026 investigation: why convergence is first order, the Kress–Martensen fix measured to 3.4e-15 at `nn = 30`, and the negative result on condition-number node placement. Measured against shipped 0.5.0. | **Live.** The source of every number in the v0.6 work. Its §9 star anomaly is unresolved by decision. |
| [v0.6-architecture.md](v0.6-architecture.md) | v0.6 doc A — decisions, scope, the `Parametrisation` object, the API break, the non-goals. | **Live.** Decisions taken; items 2 and 3 implemented, items 1, 6, 7 specified (kress-spec), 4, 5, 8 not. |
| [kress-spec.md](kress-spec.md), [kress-spec.patch](kress-spec.patch) | v0.6 code-spec for Kress–Martensen assembly, the frozen-map `Geometry` API and the test re-anchoring, with a reference patch verified against `e74c10f` (258 passed). Appendix A is the default-node-map study. | **Live**, draft 1. Not applied. |
| [parametrisation-spec.md](parametrisation-spec.md) | v0.6 code-spec for the smooth `Parametrisation`: Newton inversion replacing `np.interp`, closed-form boundary derivatives, the near-uniform density, the mild-star homotopy anchor. | **Live**, draft 1. `Parametrisation` and the closed-form derivatives are implemented; the adaptive density is not. |
| [studies/quadrature-study-plan.md](studies/quadrature-study-plan.md) | v0.6 doc B — the preliminary study: gates, references, kill criterion, open items. | **Live**, and superseded by its own results as each gate lands. |
| [performance.md](performance.md) | Where the runtime actually goes, measured; the accepted threading plan; the rejected JAX migration. | **Live.** Read before optimising anything. |
| [beyn-port-status.md](beyn-port-status.md) | Status and merge gate of the QNM/Beyn port, which shipped in v0.4.0. | Historical. §3's gotchas and §4's corrections are still worth reading; its status lines are not. |
| [beyn-port-spec.md](beyn-port-spec.md) | The implementation spec for that port, draft 2. | Historical. The authority on *intent*; phase and suite counts are frozen mid-port. |
| [beyn-port-strategy.md](beyn-port-strategy.md) | The earlier high-level strategy for the same port. | Superseded by the spec. Provenance only. |
| [qnm-methods.pdf](qnm-methods.pdf) | Background on QNM extraction methods. | Reference. |
