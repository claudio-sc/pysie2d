"""Particles over an N-layer stack: what single particles and clusters share.

:class:`_StackResult` carries the stack branch of every observable a result
offers — far field, cross-sections, the near field inside and below the stack,
the incident field and the mode amplitudes — written once against a list of
boundaries, so :class:`~pysie2d.solver.ScatterResult` and
:class:`~pysie2d.cluster.ClusterScatterResult` cannot disagree (slab-spec §1.3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import layered
from .background import HalfSpace, Stack, _effective
from .fields import (
    _layer_field,
    _mode_overlap,
    _representation_gradient_at,
    _stack_background_field,
    _stack_downward_cross_section,
    _stack_downward_far_field,
    _stack_planes,
    _stack_powers,
    _upward_cross_sections,
    _upward_far_field,
)
from .sources import in_plane_moment_b


@dataclass(frozen=True, eq=False)
class ModeAmplitudes:
    """Guided amplitudes and the full power account of one solve over a stack.

    ``s_plus[n]`` is the amplitude of the power-normalised mode
    ``ũ_n(z)·e^{+iβ_n(x − x_ref)}``, ``ũ_n = u_n/√(β_n N_n)``, and ``s_minus[n]``
    of ``ũ_n(z)·e^{−iβ_n(x − x_ref)}``: the outgoing waves at ``x → ±∞``. Powers
    are in G14's units, where the free line source emits 1/4. Under mode
    incidence the incident power is 1, so ``s_plus``/``s_minus`` are the
    S-matrix column directly (the δ_nm included); for a plane wave ``p/Re k``
    is a coupling cross-section in nm, like ``c_sca_up`` (slab-spec §1.3).

    Attributes:
        modes: The guided modes, ordered by descending β.
        x_ref: Phase reference along x (nm).
        excitation: ``"dipole"``, ``"mode"`` or ``"plane_wave"``.
        s_plus: complex ``(n_modes,)`` amplitudes towards +x.
        s_minus: complex ``(n_modes,)`` amplitudes towards −x.
        p_plus: ``|s_plus|²``.
        p_minus: ``|s_minus|²``.
        p_up: Power radiated into the cover.
        p_down: Power radiated into the bottom.
        p_abs: Power absorbed by the particles.
        p_source: Dipole: emitted power; mode: 1.0; plane wave: ``None``.
    """

    modes: tuple[layered.GuidedMode, ...]
    x_ref: float
    excitation: str
    s_plus: np.ndarray
    s_minus: np.ndarray
    p_plus: np.ndarray
    p_minus: np.ndarray
    p_up: float
    p_down: float
    p_abs: float
    p_source: float | None

    def efficiency(self, mode: int, direction: int) -> float:
        """``p_±[mode]/p_source``: the fraction of the source's power in one channel.

        Args:
            mode: Mode index.
            direction: +1 or −1.

        Returns:
            The efficiency.

        Raises:
            ValueError: For a plane wave, which has no finite source power; or
                a direction other than ±1.
        """
        if self.p_source is None:
            raise ValueError("a plane wave has no source power to normalise by")
        if direction not in (1, -1):
            raise ValueError(f"direction must be +1 or −1, got {direction}")
        p = self.p_plus if direction == 1 else self.p_minus
        return float(p[mode] / self.p_source)

    def balance(self) -> float:
        """``|p_source − Σ|/p_source``, Σ every outgoing and absorbed power.

        Returns:
            The relative power imbalance: round-off for a converged solve.

        Raises:
            ValueError: For a plane wave.
        """
        if self.p_source is None:
            raise ValueError("a plane wave has no source power to balance against")
        out = self.p_up + self.p_down + self.p_abs
        out += float(np.sum(self.p_plus) + np.sum(self.p_minus))
        return abs(self.p_source - out) / self.p_source


def _amplitudes(
    boundaries: list,
    wnum_bg: complex,
    pol: int,
    stack_rel: layered.StackRel,
    modes: tuple[layered.GuidedMode, ...],
    x_ref: float,
    excitation: str,
    source: tuple | None = None,
    p_source: float | None = None,
    incident: tuple[int, int] | None = None,
) -> ModeAmplitudes:
    """Mode amplitudes and powers from boundary data, a source and an incident mode.

    Slab-spec §2.6–§2.7: the overlap of the scattered field plus the source's
    own term — scalar ``(i/2βN)u(z_s)e^{∓iβx_s}`` (S9), in-plane
    ``(i/2βN)u(z_s)(±iβb_x + γ₁b_z)e^{∓iβx_s}`` (S11) — then
    ``s_± = √(βN)·c_±·e^{±iβx_ref}``, and an incident mode added to its own
    channel.
    """
    c_p, c_m = _mode_overlap(boundaries, modes, wnum_bg)
    if source is not None:
        xs, zs, b = source
        for i, m in enumerate(modes):
            pre = 1j / (2.0 * m.beta * m.norm) * float(m.profile(zs))
            fp, fm = (
                (1.0, 1.0)
                if b is None
                else (
                    1j * m.beta * b[0] + m.gamma * b[1],
                    -1j * m.beta * b[0] + m.gamma * b[1],
                )
            )
            c_p[i] += pre * fp * np.exp(-1j * m.beta * xs)
            c_m[i] += pre * fm * np.exp(1j * m.beta * xs)
    beta = np.array([m.beta for m in modes])
    scale = np.sqrt(beta * np.array([m.norm for m in modes]))
    s_p = scale * c_p * np.exp(1j * beta * x_ref)
    s_m = scale * c_m * np.exp(-1j * beta * x_ref)
    if incident is not None:
        order, direction = incident
        (s_p if direction == 1 else s_m)[order] += 1.0
    p_up, p_down, p_abs = _stack_powers(boundaries, wnum_bg, pol, stack_rel, source)
    return ModeAmplitudes(
        modes=modes,
        x_ref=float(x_ref),
        excitation=excitation,
        s_plus=s_p,
        s_minus=s_m,
        p_plus=np.abs(s_p) ** 2,
        p_minus=np.abs(s_m) ** 2,
        p_up=p_up,
        p_down=p_down,
        p_abs=p_abs,
        p_source=p_source,
    )


def _resolve_mode(
    stack: Stack,
    wavelength: float,
    pol: int,
    n_clad: float,
    mode: layered.GuidedMode | int,
    direction: int,
) -> layered.GuidedMode:
    """The incident mode of ``scatter_mode``, checked against the stack's own modes.

    Raises:
        ValueError: A lossy stack or complex λ (T2, from ``guided_modes``), a
            direction other than ±1, or a mode that is not one of this stack's
            at this wavelength and polarisation.
    """
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or −1, got {direction}")
    modes = stack.guided_modes(wavelength, pol, n_clad)
    if isinstance(mode, int):
        return modes[mode]
    if not (
        mode.pol == pol
        and mode.wavelength == wavelength
        and mode.order < len(modes)
        and modes[mode.order].beta == mode.beta
    ):
        raise ValueError(
            "the mode is not a guided mode of this stack at this wavelength and "
            "polarisation; pass its index or one from Stack.guided_modes"
        )
    return modes[mode.order]


def _mode_rhs(
    mode: layered.GuidedMode, direction: int, x_ref: float, f, g
) -> np.ndarray:
    """φ-half of a mode-source RHS: ``ũ_m(z_j)e^{±iβ(x_j − x_ref)}`` (§8.3).

    The χ-half is 0 and there is no reflected term: the mode solves the
    background problem on its own.
    """
    amp = 1.0 / np.sqrt(mode.beta * mode.norm)
    return amp * mode.profile(g) * np.exp(1j * direction * mode.beta * (f - x_ref))


class _Source:
    """A boundary tuple seen as the ``src`` geometry of the layered GEMMs."""

    def __init__(self, b: tuple) -> None:
        self.f, self.g, self.df, self.dg, self.delt = b[:5]
        self.n_pts = self.f.size


class _StackResult:
    """Stack behaviour shared by the single-particle and cluster results.

    The host provides ``background``, ``wavelength``, ``wnum_bg``, ``_pol``,
    ``_n_clad``, ``excitation``-style ``_excitation_name``, ``angle``,
    ``source`` and ``_boundaries()``.
    """

    background: HalfSpace | Stack | None
    wavelength: float
    angle: float
    source: tuple[float, float] | None
    mode: layered.GuidedMode | None
    direction: int | None
    x_ref: float | None
    moment: tuple[complex, complex] | None

    @property
    def _b(self) -> tuple[complex, complex] | None:
        """``b = (p_z, −p_x)`` of an in-plane dipole, ``None`` otherwise."""
        return None if self.moment is None else in_plane_moment_b(self.moment)

    @property
    def _bg(self) -> HalfSpace | Stack | None:
        """The background this result runs on after the T6 normalisation."""
        return _effective(self.background, self._n_clad, self.wavelength)

    def _stack_rel(self) -> layered.StackRel:
        assert isinstance(self.background, Stack)
        return self.background.eps_rel(self._n_clad, self.wavelength)

    def _stack_r(self) -> Callable[[np.ndarray], np.ndarray]:
        lay, d, e_bot, _ = self._stack_rel()
        k, pol = self.wnum_bg, self._pol
        return lambda q: layered.stack_r(q, k, lay, d, e_bot, pol)

    def _stack_far_field(self, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
        """Upward amplitude with the stack's R, about the origin."""
        _, _, e_bot, z_top = self._stack_rel()
        return _upward_far_field(
            n_angles,
            self._boundaries(),
            self.wnum_bg,
            e_bot,
            self._pol,
            z_top,
            r=self._stack_r(),
        )

    def _stack_cross_sections(self, n_angles: int) -> dict[str, float]:
        """``c_sca_up``, ``c_abs`` and ``c_sca_down`` (real λ only, D18)."""
        rel = self._stack_rel()
        c = _upward_cross_sections(
            n_angles,
            self._boundaries(),
            self.wnum_bg,
            rel[2],
            self._pol,
            rel[3],
            r=self._stack_r(),
        )
        if np.imag(self.wavelength) == 0.0:
            c["c_sca_down"] = _stack_downward_cross_section(
                self._boundaries(), self.wnum_bg, rel, self._pol
            )
            if all(e.imag == 0.0 for e in (*rel[0], rel[2])):  # T8: absent otherwise
                modes = self.background.guided_modes(
                    self.wavelength, self._pol, self._n_clad
                )
                c_p, c_m = _mode_overlap(self._boundaries(), modes, self.wnum_bg)
                bn = np.array([m.beta * m.norm for m in modes])
                p = float(np.sum(bn * (np.abs(c_p) ** 2 + np.abs(c_m) ** 2)))
                c["c_guided"] = p / float(np.real(self.wnum_bg))
        return c

    def _stack_source(self) -> tuple | None:
        """``(x_s, z_s, b)`` of a dipole excitation, ``b = None`` for the scalar one."""
        if self._excitation_name != "dipole":
            return None
        assert self.source is not None
        return (*self.source, self._b)

    def _source_power(self) -> float:
        """Power the dipole emits (slab-spec §2.7).

        Scalar: ``1/4 + Im G_refl(r_s, r_s) + Im u_s(r_s)``. In-plane:
        ``k²|b|²/8 + Im[(b*·∇)(b·∇′)G_refl] + Im[−(b*·∇)u_s](r_s)``, the
        reflected self term the path sum with weight ``q²|b_x|² − α₁²|b_z|²``
        (the odd cross terms vanish on the symmetric path) and ``∇u_s`` the
        analytic observer gradient of the representation.
        """
        xs, zs = self.source
        k = self.wnum_bg
        refl = layered.StackReflection.build(
            self._pol, k, self._stack_rel(), np.array([xs]), np.array([zs])
        )
        b = self._b
        if b is None:
            g_refl = refl.green(xs, zs, xs, zs)[0]
            u_s = self.eval_field(np.array([xs]), np.array([zs])).ravel()[0]
            return float(0.25 + np.imag(g_refl) + np.imag(u_s))
        p = refl.path
        a1 = layered.alpha(p.q, k)
        weight = p.q**2 * abs(b[0]) ** 2 - a1**2 * abs(b[1]) ** 2
        h2 = 2.0 * (zs - refl.z_top)
        self_r = np.sum(p.w * 0.5j * refl.r(p.q) / a1 * weight * np.exp(1j * a1 * h2))
        ux, uz = self._scattered_gradient(xs, zs)
        free = float(np.real(k)) ** 2 * (abs(b[0]) ** 2 + abs(b[1]) ** 2) / 8.0
        grad = -(np.conj(b[0]) * ux + np.conj(b[1]) * uz)
        return float(free + np.imag(self_r) / np.pi + np.imag(grad))

    def _scattered_gradient(self, x: float, z: float) -> tuple[complex, complex]:
        """Analytic ``∇u_s`` at one cover point outside every particle."""
        xp, zp = np.array([x]), np.array([z])
        refl = self._stack_cover_reflection(xp, zp)
        pts = layered.Points(xp, zp)
        ux, uz = 0j, 0j
        for bi in self._boundaries():
            f, g, df, dg, delt, ei = bi
            nn = f.size
            gx, gz = _representation_gradient_at(
                ei, nn, f, df, g, dg, delt, self.wnum_bg, xp, zp
            )
            (m1x, m2x), (m1z, m2z) = refl.blocks_grad(pts, _Source(bi))
            ux += gx[0] - (m1x @ ei[:nn] + m2x @ ei[nn:])[0]
            uz += gz[0] - (m1z @ ei[:nn] + m2z @ ei[nn:])[0]
        return complex(ux), complex(uz)

    def mode_amplitudes(self, x_ref: float) -> ModeAmplitudes:
        """Guided amplitudes ``s_±`` referred to ``x_ref``, and the power account.

        Slab-spec §2.6–§2.7. A dipole's own emission into the modes is included,
        so ``s_±`` are the total outgoing amplitudes; under mode incidence the
        incident mode is included in its own channel, so they are the S-matrix
        column. ``x_ref`` is required (R14): the phase of every amplitude is
        referred to it.

        Args:
            x_ref: Phase reference along x (nm).

        Returns:
            The :class:`ModeAmplitudes`.

        Raises:
            ValueError: Without a stack; at complex λ or over a lossy stack
                (T2, T8); for a custom excitation; for a mode excitation with a
                different ``x_ref`` than the incidence's.
            NearDegenerateModesError: Two guided modes the finder cannot
                separate (R15).
        """
        if not isinstance(self._bg, Stack):
            raise ValueError(
                "mode_amplitudes needs a Stack background with at least one layer "
                "after normalisation: nothing else guides"
            )
        exc = self._excitation_name
        if exc == "custom":
            raise ValueError(
                "mode_amplitudes is undefined for a custom incident_rhs: the "
                "result does not know the source's own emission"
            )
        if exc == "mode" and x_ref != self.x_ref:
            raise ValueError(
                f"x_ref = {x_ref} differs from the incidence's x_ref = {self.x_ref}: "
                "the S-matrix column is referred to the incident mode's phase"
            )
        modes = self.background.guided_modes(self.wavelength, self._pol, self._n_clad)
        source, p_source, incident = None, None, None
        if exc == "dipole":
            source, p_source = self._stack_source(), self._source_power()
        elif exc == "mode":
            p_source, incident = 1.0, (self.mode.order, self.direction)
        return _amplitudes(
            self._boundaries(),
            self.wnum_bg,
            self._pol,
            self._stack_rel(),
            modes,
            x_ref,
            exc,
            source,
            p_source,
            incident,
        )

    def _stack_far_field_down(self, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
        """``a_down`` below the stack; ``NoFarFieldError`` for a lossy bottom."""
        return _stack_downward_far_field(
            n_angles, self._boundaries(), self.wnum_bg, self._stack_rel(), self._pol
        )

    def _stack_eval_field(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Scattered field over a stack: cover, layers and bottom (R19).

        Cover: the host's representation with the stack's reflection. Layers:
        :func:`~pysie2d.fields._layer_field`. Bottom: the transmitted blocks
        with the stack's τ. 1-D arrays.

        Raises:
            ValueError: A point on an interface plane.
        """
        rel = self._stack_rel()
        planes = _stack_planes(rel)
        if np.any(np.isin(z, planes)):
            raise ValueError(
                "observation point exactly on an interface plane of the stack: "
                "step off it (the field is continuous there)"
            )
        above, below = z > planes[0], z < planes[-1]
        inside = ~above & ~below
        field = np.zeros(x.size, dtype=complex)
        if above.any():
            field[above] = self._cover_field(x[above], z[above])
        b = self._boundaries()
        if inside.any():
            field[inside] = _layer_field(
                x[inside], z[inside], b, self.wnum_bg, self._pol, rel
            )
        if below.any():
            tr = layered.Transmission.build_stack(
                self._pol,
                self.wnum_bg,
                rel,
                np.concatenate([bi[0] for bi in b]),
                np.concatenate([bi[1] for bi in b]),
                x[below],
                z[below],
            )
            pts = layered.Points(x[below], z[below])
            for bi in b:
                m1, m2 = tr.blocks(pts, _Source(bi))
                nn = bi[0].size
                field[below] -= m1 @ bi[5][:nn] + m2 @ bi[5][nn:]
        return field

    def _stack_cover_reflection(
        self, x: np.ndarray, z: np.ndarray
    ) -> layered.StackReflection:
        """Reflected context for the boundary nodes and cover points ``(x, z)``."""
        b = self._boundaries()
        refl = layered.StackReflection.build(
            self._pol,
            self.wnum_bg,
            self._stack_rel(),
            np.concatenate([bi[0] for bi in b] + [x]),
            np.concatenate([bi[1] for bi in b] + [z]),
        )
        assert refl is not None
        return refl

    def _stack_background(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Unmasked background field over the stack.

        A mode excitation is ``ũ(z)·e^{±iβ(x − x_ref)}`` at every height: the
        profile is defined everywhere.
        """
        if self._excitation_name == "mode":
            if np.any(np.isin(z, _stack_planes(self._stack_rel()))):
                raise ValueError(
                    "observation point exactly on an interface plane of the stack: "
                    "step off it (the field is continuous there)"
                )
            return _mode_rhs(self.mode, self.direction, self.x_ref, x, z)
        return _stack_background_field(
            x,
            z,
            self.wnum_bg,
            self._pol,
            self._excitation_name,
            self.angle,
            self.source,
            self._stack_rel(),
            self._b,
        )
