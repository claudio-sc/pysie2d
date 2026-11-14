"""S6 — the interpolation route as shipped, against the direct adaptive ellipse.

S5 was a proof of concept with its own fit. This runs the package entry point,
``beyn_modes_adaptive(interpolate=True)``, on every problem and compares it with
the same driver without interpolation. Both stop on the same test: two
consecutive levels agree in count and to ``tol`` in position.

Columns per route: assemblies (the node count M was built at), converged,
worst and median error on the ellipse, -missing +spurious; for the
interpolated route also the fit error of the last polynomial checked.
"""

from common import in_ellipse, problems, score

from pysie2d.beyn import beyn_modes_adaptive

TOL = 1.0e-9
N_MAX = 512


def main() -> None:
    """One row per problem."""
    print(
        f"{'problem':<28} | {'direct':^38} | {'interpolated':^48}\n"
        f"{'':<28} | {'asm':>4} {'conv':>5} {'worst':>8} {'median':>8} {'-m +s':>7}"
        f" | {'asm':>4} {'conv':>5} {'worst':>8} {'median':>8} {'-m +s':>7}"
        f" {'fit':>8}"
    )
    for prob in problems():
        cells = []
        for interpolate in (False, True):
            res = beyn_modes_adaptive(
                prob.m_builder,
                prob.z_lo,
                prob.z_hi,
                tol=TOL,
                n_probe=prob.n_probe,
                n_max=N_MAX,
                interpolate=interpolate,
            )
            s = score(res.modes.eigenvalues, prob, in_ellipse)
            cell = (
                f"{res.n_nodes:4d} {res.converged!s:>5} {s.worst:8.1e} "
                f"{s.median:8.1e} {f'-{s.missing} +{s.spurious}':>7}"
            )
            if interpolate:
                cell += f" {res.fit_error:8.1e}"
            cells.append(cell)
        print(f"{prob.name:<28} | {cells[0]} | {cells[1]}")


if __name__ == "__main__":
    main()
