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
| [performance.md](performance.md) | Where the runtime actually goes, measured; the accepted threading plan; the rejected JAX migration. | **Live.** Read before optimising anything. |
| [beyn-port-status.md](beyn-port-status.md) | Status and merge gate of the QNM/Beyn port, which shipped in v0.4.0. | Historical. §3's gotchas and §4's corrections are still worth reading; its status lines are not. |
| [beyn-port-spec.md](beyn-port-spec.md) | The implementation spec for that port, draft 2. | Historical. The authority on *intent*; phase and suite counts are frozen mid-port. |
| [beyn-port-strategy.md](beyn-port-strategy.md) | The earlier high-level strategy for the same port. | Superseded by the spec. Provenance only. |
| [qnm-methods.pdf](qnm-methods.pdf) | Background on QNM extraction methods. | Reference. |
