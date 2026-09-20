"""Gate 1: reproduce the analytic Mie cylinder in dolfinx.

Independent FEM check of pysie2d's cross-sections, in the frequency domain.
Nothing here shares a line of algebra with the boundary-integral solver: this
is a volume formulation on a triangulated disc with a radial PML, assembled by
FFCx and solved by PETSc. The only thing the two codes agree on beforehand is
the physics, which is the point (CLAUDE.md non-negotiable §3).

Run it inside the conda environment beside this file — never under uv::

    mamba env create -f validation/dolfinx/environment.yml
    conda run -n pysie2d-dolfinx python validation/dolfinx/run_cylinder.py

It writes ``tests/data/dolfinx-circle-*.json`` in the frozen-spectrum format of
``validation/spectrum.py``, and does not import pysie2d.

Formulation
-----------
One scalar Helmholtz equation covers both polarisations, with the material
coefficients moved between the two terms. Writing ``u`` for the field along
the invariant axis and ``ε`` for the **absolute** permittivity
(conventions §2 — ``ε = n_core² + i·epsi`` inside, ``n_clad²`` outside):

    pol = 2  (TE, E_y):   ∇·(∇u)      + k₀² ε u = 0    → α = 1,    β = ε
    pol = 1  (TM, H_y):   ∇·((1/ε)∇u) + k₀²   u = 0    → α = 1/ε,  β = 1

Note the axis. pysie2d's invariant axis is **y** and its in-plane coordinates
are ``(x, z)``; this mesh calls them ``(x, y)``. The relabelling inverts the
usual TE/TM naming, which is why each frozen file carries its ``pol_mapping``
in words rather than a code.

Solved in the scattered field ``u_s = u_t − u_i``, so the PML never has to
absorb the incident wave. With ``L[u] = −∇·(α∇u) − k₀²βu`` and the background
satisfying ``L_b[u_i] = 0``:

    ∫ α∇u_s·∇v̄ − k₀²β u_s v̄  =  −∫ (α−α_b)∇u_i·∇v̄ + k₀²∫ (β−β_b) u_i v̄

Both right-hand-side integrands are supported **only inside the particle**, so
the source is compact and no boundary term survives. For ``pol = 2`` the first
term vanishes identically (α ≡ α_b ≡ 1) and for ``pol = 1`` the second does.

The incident wave matches pysie2d's ``plane_wave_rhs`` at ``angle = 0``, which
is ``exp(i k_bg (x sinθ − z cosθ))`` → ``exp(−i k_bg z)``: **propagating along
−z**, not +z. Getting this backwards leaves every cross-section unchanged on a
circle and silently wrong on a star, so it is pinned here and not inferred.

Observables
-----------
``C_abs`` and ``C_ext`` come from volume integrals, ``C_sca`` from an
independent flux integral on the physical/PML interface. Computing all three
rather than deriving one from the other two is deliberate: the residual
``C_ext − C_sca − C_abs`` is then a free consistency check that a compensating
sign error cannot survive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gmsh
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "validation"))

from spectrum import Convergence, Spectrum, Tolerance, write  # noqa: E402

DATA = ROOT / "tests" / "data"

# Geometry and material of the shared fixture, in nm — the same circle the
# frozen Mie files use, so gate 1 is a diff against a file already in the repo.
RAD = 200.0
N_CORE = 1.5
WAVELENGTHS = np.linspace(400.0, 900.0, 51)

# Domain layout, in units of RAD. The physical disc must be far enough out
# that the scattered field is locally outgoing where the PML starts; the
# measurement contour is that same interface, so it also has to be outside the
# evanescent skirt of the particle.
R_MEAS = 3.0 * RAD
R_PHYS = 4.0 * RAD
R_PML = 6.0 * RAD

# PML strength. σ₀ is quoted as a round-trip reflection target rather than as
# a bare number, because the bare number means nothing without the thickness
# and the polynomial order it goes with.
PML_ORDER = 2
PML_REFLECTION = 1e-8

# Markers written by the gmsh model and read back on the dolfinx side. The
# background is split in two at R_MEAS for one reason: the scattered-flux
# contour has to sit where the material is the *same* on both sides, so that
# ∇u_s is continuous across it and averaging the two facet restrictions is
# exact rather than a small uncontrolled error. Measuring on the PML interface
# instead puts the absorbing layer on one side of the contour.
PARTICLE, INNER, OUTER, PML = 1, 2, 3, 4
MEASURE = 5

_ScalarT = PETSc.ScalarType


def _require_complex() -> None:
    """Refuse to run under a real-scalar PETSc build.

    The default conda-forge ``fenics-dolfinx`` resolves to a real build. It
    imports and assembles perfectly happily and then discards every imaginary
    part — the PML stops absorbing, ``C_abs`` collapses to zero, and the
    result looks plausible. Fail loudly here instead.
    """
    if not np.issubdtype(_ScalarT, np.complexfloating):
        raise RuntimeError(
            "PETSc is a real-scalar build; this driver needs complex128. "
            "Recreate the environment with the `petsc=*=*complex*` pin in "
            "validation/dolfinx/environment.yml."
        )


def build_mesh(h_particle: float, h_outer: float, order: int):
    """Mesh the circle, the background disc and the PML annulus.

    Three concentric physical surfaces so that the material coefficients are
    piecewise constant per cell and the physical/PML interface exists as a
    tagged facet set — which is what makes the scattered-flux integral
    available without embedding a separate measurement curve.

    Args:
        h_particle: Target element size on the particle boundary (nm).
        h_outer: Target element size in the background and PML (nm).
        order: Geometric order of the elements. Order 2 or higher keeps the
            polygonal-boundary error below the field error, which matters
            because a circle approximated by straight edges converges at a
            rate set by the geometry, not by the basis.

    Returns:
        (domain, cell_tags, facet_tags), unpacked from the 0.11 MeshData.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cylinder")

    discs = [
        gmsh.model.occ.addDisk(0, 0, 0, r, r) for r in (RAD, R_MEAS, R_PHYS, R_PML)
    ]
    out, _ = gmsh.model.occ.fragment([(2, discs[-1])], [(2, tag) for tag in discs[:-1]])
    gmsh.model.occ.synchronize()

    # Identify the three surfaces by their area: fragmenting does not promise
    # to preserve tag order, and guessing it is how a PML ends up on the
    # particle.
    by_area = sorted(
        ((gmsh.model.occ.getMass(2, tag), tag) for _, tag in out),
        key=lambda pair: pair[0],
    )
    for marker, (_, tag) in zip((PARTICLE, INNER, OUTER, PML), by_area, strict=True):
        gmsh.model.addPhysicalGroup(2, [tag], marker)

    # The measurement contour: the curve of radius R_MEAS.
    contour = [
        tag
        for _, tag in gmsh.model.occ.getEntities(1)
        if abs(gmsh.model.occ.getMass(1, tag) - 2.0 * np.pi * R_MEAS) < 1.0
    ]
    gmsh.model.addPhysicalGroup(1, contour, MEASURE)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_particle)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_outer)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.model.mesh.generate(2)

    # dolfinx 0.11 returns a MeshData record rather than the 3-tuple older
    # releases (and most tutorials) hand back.
    data = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return data.mesh, data.cell_tags, data.facet_tags


def _pml_tensor(x, k0: float):
    """Radial complex-coordinate stretch, as a Cartesian tensor and a scalar.

    A radial PML matches the circular domain, so the absorbing layer has
    uniform thickness in every direction — a square PML around a disc wastes
    cells in the corners and absorbs unevenly at grazing angles.

    With stretch ``s(r) = 1 + iσ(r)/k₀`` and stretched radius ``r̃ = ∫s dr``,
    the transformed problem is the untransformed one with

        Λ = (r̃/(r s)) ê_r ê_r + (r s/r̃) ê_θ ê_θ ,    J = s r̃ / r

    multiplying the gradient and the mass term respectively. Outside the layer
    σ = 0, so Λ = I and J = 1 and the same expression covers the whole domain.

    Args:
        x: UFL spatial coordinate.
        k0: Vacuum wavenumber (rad/nm).

    Returns:
        (Λ, J) as UFL expressions.
    """
    r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    d = R_PML - R_PHYS
    # σ₀ from the target round-trip reflection of a polynomial profile:
    # R = exp(-2∫σ dr) = exp(-2σ₀d/(p+1)).
    sigma0 = -(PML_ORDER + 1) * np.log(PML_REFLECTION) / (2.0 * d)
    # In a complex PETSc build UFL types every spatial coordinate as complex
    # and refuses to order them, so the profile's cut-off has to be taken on
    # an explicitly real quantity. ``r`` is a radius — real by construction —
    # so ``ufl.real`` here is exact, not a truncation.
    r_re = ufl.real(r)
    t = ufl.conditional(ufl.gt(r_re, R_PHYS), (r_re - R_PHYS) / d, 0.0)
    sigma = sigma0 * t**PML_ORDER
    s = 1.0 + 1j * sigma / k0
    # r̃ = r + (i/k₀)∫₀^{r-R} σ dr' — the profile integrates in closed form.
    r_t = r + 1j * sigma0 * d * t ** (PML_ORDER + 1) / (k0 * (PML_ORDER + 1))

    e_r = ufl.as_vector([x[0] / r, x[1] / r])
    e_th = ufl.as_vector([-x[1] / r, x[0] / r])
    lam = (r_t / (r * s)) * ufl.outer(e_r, e_r) + (r * s / r_t) * ufl.outer(e_th, e_th)
    return lam, s * r_t / r


def solve_one(
    domain,
    cell_tags,
    facet_tags,
    wavelength: float,
    pol: int,
    n_clad: float,
    epsi: float,
    degree: int,
) -> dict[str, float]:
    """Solve one wavelength and return the three absolute cross-sections.

    Args:
        domain: The mesh, from :func:`build_mesh`.
        cell_tags: Subdomain markers for particle, background and PML.
        facet_tags: Facet markers carrying the measurement contour.
        wavelength: **Vacuum** wavelength in nm (conventions §2.1). The single
            conversion to a background wavenumber happens here and nowhere
            else in this driver.
        pol: pysie2d polarisation — 2 = TE (E along the invariant axis),
            1 = TM (H along it).
        n_clad: Background refractive index.
        epsi: Absolute imaginary permittivity of the particle.
        degree: Polynomial degree of the Lagrange basis.

    Returns:
        dict with 'c_sca', 'c_ext', 'c_abs' in nm, and 'residual' — the
        normalised optical-theorem defect ``(C_ext − C_sca − C_abs)/C_ext``,
        which is a free check and not an input to anything.
    """
    k0 = 2.0 * np.pi / wavelength
    k_bg = k0 * n_clad

    eps_p = complex(N_CORE**2 + 1j * epsi)
    eps_b = complex(n_clad**2)

    # Piecewise-constant material, one value per cell — a DG0 function rather
    # than a UFL conditional on position, so the particle boundary follows the
    # mesh exactly instead of being resampled at quadrature points.
    q = fem.functionspace(domain, ("DG", 0))
    eps = fem.Function(q)
    eps.x.array[:] = eps_b
    inside = cell_tags.find(PARTICLE)
    eps.x.array[inside] = eps_p
    eps.x.scatter_forward()

    alpha_b, beta_b = (1.0, eps_b) if pol == 2 else (1.0 / eps_b, 1.0)
    alpha, beta = (1.0, eps) if pol == 2 else (1.0 / eps, 1.0)

    v_space = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(v_space)
    v = ufl.TestFunction(v_space)
    x = ufl.SpatialCoordinate(domain)
    lam, jac = _pml_tensor(x, k0)

    # exp(-i k_bg z) — pysie2d's angle = 0, travelling along −z (here −y).
    u_inc = ufl.exp(-1j * k_bg * x[1])

    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    a = (
        ufl.inner(alpha * ufl.dot(lam, ufl.grad(u)), ufl.grad(v))
        - k0**2 * ufl.inner(beta * jac * u, v)
    ) * ufl.dx
    rhs = (
        -ufl.inner((alpha - alpha_b) * ufl.grad(u_inc), ufl.grad(v))
        + k0**2 * ufl.inner((beta - beta_b) * u_inc, v)
    ) * dx(PARTICLE)

    # Direct LU throughout. The system is small in 2-D and indefinite, so an
    # iterative solver would need a preconditioner tuned per wavelength —
    # a source of variation the reference must not have.
    problem = LinearProblem(
        a,
        rhs,
        petsc_options_prefix="pysie2d_val_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    u_s = problem.solve()
    u_t = u_s + u_inc

    # --- C_abs: volumetric Joule loss inside the particle, normalised by the
    # incident intensity. Exactly zero for real ε, which is the null test.
    if pol == 2:
        abs_form = (k0 / n_clad) * ufl.imag(eps) * ufl.inner(u_t, u_t)
    else:
        abs_form = (
            (n_clad / k0)
            * ufl.imag(eps)
            / (abs(eps) ** 2)
            * ufl.inner(ufl.grad(u_t), ufl.grad(u_t))
        )
    c_abs = _assemble(abs_form * dx(PARTICLE))

    # --- C_ext: the volume form of the optical theorem. The polarisability
    # contrast is again supported only inside the particle, so extinction is
    # read off the same compact region as the source — no far-field transform,
    # and no dependence on where the measurement contour sits.
    if pol == 2:
        ext_form = (k0 / n_clad) * ufl.imag((eps - eps_b) * u_t * ufl.conj(u_inc))
    else:
        # NOT ufl.inner here. In a complex build ``inner(a, b)`` conjugates
        # its *second* argument, so inner(grad(u_t), grad(conj(u_i)))
        # conjugates twice and silently pairs u_t with u_i instead. The
        # symptom is benign-looking — a C_ext that stays smooth and positive
        # and is wrong by a factor — and the only thing that catches it is
        # the optical-theorem residual, which is why C_sca is computed
        # independently rather than inferred from C_ext − C_abs.
        # Prefactor ε_b/k_bg = n_clad/k₀, not 1/k_bg. For TM the incident
        # intensity itself carries the α_b = 1/ε_b of the flux, so it does
        # not cancel out of the normalisation the way it does for TE. At
        # n_clad = 1 the two are identical, which is exactly why the whole
        # fixture suite would miss this — the lossy case runs at
        # n_clad = 1.33 for this reason.
        ext_form = (n_clad / k0) * ufl.imag(
            (1.0 / eps_b - 1.0 / eps)
            * ufl.dot(ufl.grad(u_t), ufl.conj(ufl.grad(u_inc)))
        )
    c_ext = _assemble(ext_form * dx(PARTICLE))

    # --- C_sca: outgoing flux of the scattered field through the measurement
    # circle, independent of the two above by construction.
    #
    # Two details that are easy to get wrong here. The contour is an *interior*
    # facet set, so it needs ``dS`` — with ``ds`` the integral is silently zero,
    # because there are no exterior facets carrying that tag. And the outward
    # direction is taken as the radial unit vector rather than ``FacetNormal``,
    # whose ``+``/``-`` orientation is arbitrary per facet and would sum the
    # flux to zero. The background material is identical on both sides of this
    # circle, so ``avg`` of the two restrictions is exact.
    d_interior = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags)
    r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    e_r = ufl.as_vector([x[0] / r, x[1] / r])
    flux = (1.0 / k_bg) * ufl.imag(ufl.conj(u_s) * ufl.dot(ufl.grad(u_s), e_r))
    c_sca = _assemble(ufl.avg(flux) * d_interior(MEASURE))

    residual = (c_ext - c_sca - c_abs) / c_ext if c_ext != 0.0 else 0.0
    return {
        "c_sca": c_sca,
        "c_ext": c_ext,
        "c_abs": c_abs,
        "residual": float(residual),
    }


def _assemble(form) -> float:
    """Assemble a real-valued scalar form and reduce it across ranks."""
    local = fem.assemble_scalar(fem.form(form))
    return float(np.real(MPI.COMM_WORLD.allreduce(local, op=MPI.SUM)))


# Discretisation. Gate 2 sets these by refining until the spectrum stops
# moving; the numbers below are the converged settings, and the drift measured
# on the way to them travels in every file this script writes.
H_PARTICLE = 12.0  # nm, ≈ λ_min/(22·n_core) at the shortest wavelength
H_OUTER = 40.0
DEGREE = 3
GEOM_ORDER = 2

CASES = [
    ("dolfinx-circle-lossless-te", 2, 1.0, 0.0),
    ("dolfinx-circle-lossless-tm", 1, 1.0, 0.0),
    ("dolfinx-circle-lossy-te", 2, 1.33, 0.5),
    ("dolfinx-circle-lossy-tm", 1, 1.33, 0.5),
]

POL_MAPPING = {
    2: (
        "pysie2d pol=2 is TE: E along the invariant axis, Mie coefficient "
        "b_n. Here the invariant axis is z and the scalar unknown is E_z, so "
        "this driver's mesh coordinates (x, y) correspond to pysie2d's "
        "(x, z) and the conventional FEM name for the same problem is "
        "'TM'/'E_z-polarised'. Both codes use exp(-iωt), so Im(eps) > 0 "
        "absorbs in both."
    ),
    1: (
        "pysie2d pol=1 is TM: H along the invariant axis, Mie coefficient "
        "a_n. Same axis relabelling as pol=2; the conventional FEM name is "
        "'TE'/'H_z-polarised'. Both codes use exp(-iωt)."
    ),
}


def sweep(domain, cell_tags, facet_tags, pol, n_clad, epsi, degree):
    """Run the whole wavelength grid on one mesh.

    The mesh is reused across wavelengths deliberately: it is sized for the
    *shortest* wavelength on the grid, so every other point is over-resolved
    rather than under-resolved, and the geometry error is identical at every
    point instead of drifting with a remeshing.

    Returns:
        dict of arrays 'c_sca_nm', 'c_ext_nm', 'c_abs_nm', 'residual'.
    """
    out = {
        k: np.empty(WAVELENGTHS.size) for k in ("c_sca", "c_ext", "c_abs", "residual")
    }
    for i, lam in enumerate(WAVELENGTHS):
        res = solve_one(
            domain, cell_tags, facet_tags, float(lam), pol, n_clad, epsi, degree
        )
        for k in out:
            out[k][i] = res[k]
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"  λ={lam:6.1f}  C_sca={res['c_sca']:10.3f}  "
                f"C_ext={res['c_ext']:10.3f}  C_abs={res['c_abs']:10.3f}  "
                f"resid={res['residual']:+.2e}",
                flush=True,
            )
    return out


def main() -> None:
    """Run every case at the converged discretisation and freeze the spectra."""
    _require_complex()
    DATA.mkdir(parents=True, exist_ok=True)
    domain, cell_tags, facet_tags = build_mesh(H_PARTICLE, H_OUTER, GEOM_ORDER)

    for case_id, pol, n_clad, epsi in CASES:
        if MPI.COMM_WORLD.rank == 0:
            print(f"\n{case_id}", flush=True)
        res = sweep(domain, cell_tags, facet_tags, pol, n_clad, epsi, DEGREE)
        if MPI.COMM_WORLD.rank != 0:
            continue
        spec = Spectrum(
            case_id=case_id,
            claim=(
                "pysie2d's cross-sections on a circle agree with an "
                "independent frequency-domain FEM solve (dolfinx, radial "
                "PML, curved elements) that shares no formulation, "
                "discretisation or linear algebra with the boundary-integral "
                "method."
            ),
            tool={"name": "dolfinx", "version": _versions()},
            geometry={
                "rad": RAD,
                "m": 0,
                "n1": 2.0,
                "n2": 2.0,
                "n3": 2.0,
                "n_pts": 200,
            },
            material={"n_core": N_CORE, "n_clad": n_clad, "epsi": epsi},
            pol=pol,
            pol_mapping=POL_MAPPING[pol],
            angle_deg=0.0,
            wavelength_nm=WAVELENGTHS,
            c_sca_nm=res["c_sca"],
            c_ext_nm=res["c_ext"],
            c_abs_nm=res["c_abs"],
            convergence=Convergence(
                parameter="(h_particle, Lagrange degree)",
                coarse="placeholder — gate 2 has not been run",
                fine=f"({H_PARTICLE} nm, {DEGREE})",
                max_rel_drift=float("nan"),
                note="PLACEHOLDER. Not to be committed until gate 2 runs.",
            ),
            tolerance=Tolerance(
                rel=float("nan"),
                abs_nm=float("nan"),
                justification="PLACEHOLDER — set from gate 2's measured floor.",
            ),
            notes=(
                "Worst optical-theorem residual "
                f"(C_ext − C_sca − C_abs)/C_ext over the grid: "
                f"{np.max(np.abs(res['residual'])):.2e}. The three are "
                "computed independently — two volume integrals and one flux "
                "integral — so this residual is evidence, not an identity."
            ),
        )
        path = write(spec, DATA / f"{case_id}.json")
        print(f"  → {path.name}", flush=True)


def _versions() -> str:
    import dolfinx

    return f"{dolfinx.__version__} (petsc complex128, gmsh {gmsh.GMSH_API_VERSION})"


if __name__ == "__main__":
    main()
